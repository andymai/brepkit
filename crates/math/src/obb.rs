//! Oriented bounding box (OBB) for tighter spatial filtering.
//!
//! OBBs fit rotated and curved geometry more tightly than axis-aligned boxes.
//! Used as a secondary filter after BVH broad-phase queries to reject false
//! positives before expensive narrow-phase intersection tests.

use crate::vec::{Point3, Vec3};

/// A 3D oriented bounding box.
///
/// Stores a center, three orthonormal axes, and half-extents along each axis.
#[derive(Debug, Clone, Copy)]
pub struct Obb3 {
    /// Center of the box.
    pub center: Point3,
    /// Three orthonormal axes (columns of the rotation matrix).
    pub axes: [Vec3; 3],
    /// Half-extents along each axis.
    pub half_extents: [f64; 3],
}

impl Obb3 {
    /// Build an OBB from a point set using PCA (principal component analysis).
    ///
    /// Computes the covariance matrix of the points, extracts eigenvectors as
    /// the OBB axes, then projects all points to find the extents.
    ///
    /// Falls back to an axis-aligned box if the point set is degenerate
    /// (collinear or coincident).
    ///
    /// # Panics
    ///
    /// Panics if the iterator yields fewer than 1 point.
    #[must_use]
    #[allow(clippy::expect_used, clippy::missing_panics_doc)]
    pub fn from_points(points: impl IntoIterator<Item = Point3>) -> Self {
        let pts: Vec<Point3> = points.into_iter().collect();
        assert!(!pts.is_empty(), "OBB requires at least one point");

        let n = pts.len() as f64;

        // Compute centroid.
        let mut cx = 0.0_f64;
        let mut cy = 0.0_f64;
        let mut cz = 0.0_f64;
        for p in &pts {
            cx += p.x();
            cy += p.y();
            cz += p.z();
        }
        cx /= n;
        cy /= n;
        cz /= n;

        // Compute covariance matrix (symmetric 3×3).
        let mut cov = [0.0_f64; 6]; // [xx, xy, xz, yy, yz, zz]
        for p in &pts {
            let dx = p.x() - cx;
            let dy = p.y() - cy;
            let dz = p.z() - cz;
            cov[0] += dx * dx;
            cov[1] += dx * dy;
            cov[2] += dx * dz;
            cov[3] += dy * dy;
            cov[4] += dy * dz;
            cov[5] += dz * dz;
        }

        // Extract eigenvectors via Jacobi iteration on the symmetric 3×3.
        let axes = eigen_axes_3x3(cov);

        Self::from_axes_and_points(Point3::new(cx, cy, cz), axes, &pts)
    }

    /// Build an OBB with a known primary axis (e.g. face normal for planar faces).
    ///
    /// Uses the given normal as one axis and PCA in the remaining plane for the
    /// other two. This gives near-zero thickness for planar faces.
    ///
    /// # Panics
    ///
    /// Panics if the iterator yields fewer than 1 point.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn from_points_with_normal(points: impl IntoIterator<Item = Point3>, normal: Vec3) -> Self {
        let pts: Vec<Point3> = points.into_iter().collect();
        assert!(!pts.is_empty(), "OBB requires at least one point");

        let n = pts.len() as f64;

        // Centroid.
        let mut cx = 0.0_f64;
        let mut cy = 0.0_f64;
        let mut cz = 0.0_f64;
        for p in &pts {
            cx += p.x();
            cy += p.y();
            cz += p.z();
        }
        cx /= n;
        cy /= n;
        cz /= n;

        // Normalize the provided normal (axis 2 = thickness direction).
        let len =
            (normal.x() * normal.x() + normal.y() * normal.y() + normal.z() * normal.z()).sqrt();
        let axis2 = if len > 1e-15 {
            Vec3::new(normal.x() / len, normal.y() / len, normal.z() / len)
        } else {
            // Degenerate normal, fall back to full PCA.
            return Self::from_points(pts);
        };

        // Find a perpendicular direction for in-plane PCA.
        // Pick the coordinate axis most perpendicular to axis2.
        let abs_x = axis2.x().abs();
        let abs_y = axis2.y().abs();
        let abs_z = axis2.z().abs();
        let seed = if abs_x <= abs_y && abs_x <= abs_z {
            Vec3::new(1.0, 0.0, 0.0)
        } else if abs_y <= abs_z {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };

        // Gram-Schmidt to get two in-plane axes.
        let u = {
            let v = Vec3::new(
                seed.x() - axis2.x() * seed.dot(axis2),
                seed.y() - axis2.y() * seed.dot(axis2),
                seed.z() - axis2.z() * seed.dot(axis2),
            );
            let l = (v.x() * v.x() + v.y() * v.y() + v.z() * v.z()).sqrt();
            Vec3::new(v.x() / l, v.y() / l, v.z() / l)
        };

        // Project points onto u to find 2D covariance for in-plane PCA.
        let v = axis2.cross(u);

        // 2D covariance in the (u, v) plane.
        let mut cov_uu = 0.0_f64;
        let mut cov_uv = 0.0_f64;
        let mut cov_vv = 0.0_f64;
        for p in &pts {
            let d = Vec3::new(p.x() - cx, p.y() - cy, p.z() - cz);
            let du = d.dot(u);
            let dv = d.dot(v);
            cov_uu += du * du;
            cov_uv += du * dv;
            cov_vv += dv * dv;
        }

        // 2×2 eigendecomposition for in-plane axes.
        let (angle, _e1, _e2) = eigen_2x2(cov_uu, cov_uv, cov_vv);
        let (sin_a, cos_a) = angle.sin_cos();

        // Rotate (u, v) by the eigenvector angle.
        let axis0 = Vec3::new(
            cos_a * u.x() + sin_a * v.x(),
            cos_a * u.y() + sin_a * v.y(),
            cos_a * u.z() + sin_a * v.z(),
        );
        let axis1 = Vec3::new(
            -sin_a * u.x() + cos_a * v.x(),
            -sin_a * u.y() + cos_a * v.y(),
            -sin_a * u.z() + cos_a * v.z(),
        );

        Self::from_axes_and_points(Point3::new(cx, cy, cz), [axis0, axis1, axis2], &pts)
    }

    /// Build OBB from pre-computed axes by projecting points to find extents.
    fn from_axes_and_points(centroid: Point3, axes: [Vec3; 3], pts: &[Point3]) -> Self {
        let mut min_ext = [f64::INFINITY; 3];
        let mut max_ext = [f64::NEG_INFINITY; 3];

        for p in pts {
            let d = Vec3::new(
                p.x() - centroid.x(),
                p.y() - centroid.y(),
                p.z() - centroid.z(),
            );
            for (i, ax) in axes.iter().enumerate() {
                let proj = d.dot(*ax);
                if proj < min_ext[i] {
                    min_ext[i] = proj;
                }
                if proj > max_ext[i] {
                    max_ext[i] = proj;
                }
            }
        }

        // Re-center: shift center to midpoint of extent range along each axis.
        let mut center = centroid;
        let mut half_extents = [0.0_f64; 3];
        for i in 0..3 {
            let mid = (min_ext[i] + max_ext[i]) * 0.5;
            half_extents[i] = (max_ext[i] - min_ext[i]) * 0.5;
            center = Point3::new(
                center.x() + axes[i].x() * mid,
                center.y() + axes[i].y() * mid,
                center.z() + axes[i].z() * mid,
            );
        }

        Self {
            center,
            axes,
            half_extents,
        }
    }

    /// Test whether two OBBs intersect using the Separating Axis Theorem.
    ///
    /// Tests 15 potential separating axes: 3 from each OBB + 9 cross products.
    /// Returns `true` if the OBBs overlap (no separating axis found).
    #[inline]
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn intersects(&self, other: &Self) -> bool {
        // Vector from self center to other center.
        let t = Vec3::new(
            other.center.x() - self.center.x(),
            other.center.y() - self.center.y(),
            other.center.z() - self.center.z(),
        );

        let a = &self.axes;
        let b = &other.axes;
        let ea = &self.half_extents;
        let eb = &other.half_extents;

        // Precompute rotation matrix R[i][j] = a[i] · b[j]
        // and absolute values with epsilon for parallel edge cases.
        #[allow(clippy::items_after_statements)]
        const EPS: f64 = 1e-12;
        let mut r = [[0.0_f64; 3]; 3];
        let mut abs_r = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = a[i].dot(b[j]);
                abs_r[i][j] = r[i][j].abs() + EPS;
            }
        }

        // Test axes a[0], a[1], a[2]
        for i in 0..3 {
            let ra = ea[i];
            let rb = eb[0] * abs_r[i][0] + eb[1] * abs_r[i][1] + eb[2] * abs_r[i][2];
            if t.dot(a[i]).abs() > ra + rb {
                return false;
            }
        }

        // Test axes b[0], b[1], b[2]
        for j in 0..3 {
            let ra = ea[0] * abs_r[0][j] + ea[1] * abs_r[1][j] + ea[2] * abs_r[2][j];
            let rb = eb[j];
            let d = t.dot(b[j]).abs();
            if d > ra + rb {
                return false;
            }
        }

        // Test 9 cross-product axes: a[i] × b[j]
        // a[0] × b[0]
        {
            let ra = ea[1] * abs_r[2][0] + ea[2] * abs_r[1][0];
            let rb = eb[1] * abs_r[0][2] + eb[2] * abs_r[0][1];
            let d = (t.dot(a[2]) * r[1][0] - t.dot(a[1]) * r[2][0]).abs();
            if d > ra + rb {
                return false;
            }
        }
        // a[0] × b[1]
        {
            let ra = ea[1] * abs_r[2][1] + ea[2] * abs_r[1][1];
            let rb = eb[0] * abs_r[0][2] + eb[2] * abs_r[0][0];
            let d = (t.dot(a[2]) * r[1][1] - t.dot(a[1]) * r[2][1]).abs();
            if d > ra + rb {
                return false;
            }
        }
        // a[0] × b[2]
        {
            let ra = ea[1] * abs_r[2][2] + ea[2] * abs_r[1][2];
            let rb = eb[0] * abs_r[0][1] + eb[1] * abs_r[0][0];
            let d = (t.dot(a[2]) * r[1][2] - t.dot(a[1]) * r[2][2]).abs();
            if d > ra + rb {
                return false;
            }
        }
        // a[1] × b[0]
        {
            let ra = ea[0] * abs_r[2][0] + ea[2] * abs_r[0][0];
            let rb = eb[1] * abs_r[1][2] + eb[2] * abs_r[1][1];
            let d = (t.dot(a[0]) * r[2][0] - t.dot(a[2]) * r[0][0]).abs();
            if d > ra + rb {
                return false;
            }
        }
        // a[1] × b[1]
        {
            let ra = ea[0] * abs_r[2][1] + ea[2] * abs_r[0][1];
            let rb = eb[0] * abs_r[1][2] + eb[2] * abs_r[1][0];
            let d = (t.dot(a[0]) * r[2][1] - t.dot(a[2]) * r[0][1]).abs();
            if d > ra + rb {
                return false;
            }
        }
        // a[1] × b[2]
        {
            let ra = ea[0] * abs_r[2][2] + ea[2] * abs_r[0][2];
            let rb = eb[0] * abs_r[1][1] + eb[1] * abs_r[1][0];
            let d = (t.dot(a[0]) * r[2][2] - t.dot(a[2]) * r[0][2]).abs();
            if d > ra + rb {
                return false;
            }
        }
        // a[2] × b[0]
        {
            let ra = ea[0] * abs_r[1][0] + ea[1] * abs_r[0][0];
            let rb = eb[1] * abs_r[2][2] + eb[2] * abs_r[2][1];
            let d = (t.dot(a[1]) * r[0][0] - t.dot(a[0]) * r[1][0]).abs();
            if d > ra + rb {
                return false;
            }
        }
        // a[2] × b[1]
        {
            let ra = ea[0] * abs_r[1][1] + ea[1] * abs_r[0][1];
            let rb = eb[0] * abs_r[2][2] + eb[2] * abs_r[2][0];
            let d = (t.dot(a[1]) * r[0][1] - t.dot(a[0]) * r[1][1]).abs();
            if d > ra + rb {
                return false;
            }
        }
        // a[2] × b[2]
        {
            let ra = ea[0] * abs_r[1][2] + ea[1] * abs_r[0][2];
            let rb = eb[0] * abs_r[2][1] + eb[1] * abs_r[2][0];
            let d = (t.dot(a[1]) * r[0][2] - t.dot(a[0]) * r[1][2]).abs();
            if d > ra + rb {
                return false;
            }
        }

        // No separating axis found — OBBs overlap.
        true
    }
}

// ---------------------------------------------------------------------------
// Eigendecomposition helpers (no external deps)
// ---------------------------------------------------------------------------

/// Eigenvalues and rotation angle for a symmetric 2×2 matrix `[[a, b], [b, c]]`.
///
/// Returns `(angle, eigenvalue_1, eigenvalue_2)` where `angle` rotates the
/// standard basis to the eigenvector basis.
fn eigen_2x2(a: f64, b: f64, c: f64) -> (f64, f64, f64) {
    if b.abs() < 1e-30 {
        return (0.0, a, c);
    }
    let theta = 0.5 * (c - a).atan2(2.0 * b);
    let trace = a + c;
    let det = a * c - b * b;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let e1 = (trace + disc) * 0.5;
    let e2 = (trace - disc) * 0.5;
    (theta, e1, e2)
}

/// Extract principal axes from a symmetric 3×3 covariance matrix using
/// Jacobi eigenvalue iteration.
///
/// Input: `cov = [xx, xy, xz, yy, yz, zz]` (upper triangle, row-major).
/// Returns three orthonormal eigenvectors (sorted by decreasing eigenvalue).
#[allow(clippy::similar_names)]
fn eigen_axes_3x3(cov: [f64; 6]) -> [Vec3; 3] {
    // Unpack into full symmetric matrix.
    let mut m = [
        [cov[0], cov[1], cov[2]],
        [cov[1], cov[3], cov[4]],
        [cov[2], cov[4], cov[5]],
    ];
    // Eigenvector matrix (starts as identity).
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // Jacobi iteration: apply Givens rotations to diagonalize m.
    for _ in 0..50 {
        // Find largest off-diagonal element.
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..3 {
            for j in (i + 1)..3 {
                if m[i][j].abs() > max_val {
                    max_val = m[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-30 {
            break; // Converged.
        }

        // Compute Givens rotation angle.
        let theta = if (m[p][p] - m[q][q]).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * m[p][q]).atan2(m[p][p] - m[q][q])
        };
        let (sin_t, cos_t) = theta.sin_cos();

        // Apply rotation to m: m' = G^T * m * G
        let mut m2 = m;
        m2[p][p] =
            cos_t * cos_t * m[p][p] + 2.0 * sin_t * cos_t * m[p][q] + sin_t * sin_t * m[q][q];
        m2[q][q] =
            sin_t * sin_t * m[p][p] - 2.0 * sin_t * cos_t * m[p][q] + cos_t * cos_t * m[q][q];
        m2[p][q] = 0.0;
        m2[q][p] = 0.0;
        for r in 0..3 {
            if r != p && r != q {
                let mp = cos_t * m[r][p] + sin_t * m[r][q];
                let mq = -sin_t * m[r][p] + cos_t * m[r][q];
                m2[r][p] = mp;
                m2[p][r] = mp;
                m2[r][q] = mq;
                m2[q][r] = mq;
            }
        }
        m = m2;

        // Accumulate eigenvectors.
        for r in 0..3 {
            let vp = cos_t * v[r][p] + sin_t * v[r][q];
            let vq = -sin_t * v[r][p] + cos_t * v[r][q];
            v[r][p] = vp;
            v[r][q] = vq;
        }
    }

    // Sort eigenvectors by decreasing eigenvalue.
    let mut order = [0, 1, 2];
    let eigenvalues = [m[0][0], m[1][1], m[2][2]];
    order.sort_by(|&a, &b| {
        eigenvalues[b]
            .partial_cmp(&eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let make_axis = |col: usize| {
        let len = (v[0][col] * v[0][col] + v[1][col] * v[1][col] + v[2][col] * v[2][col]).sqrt();
        if len > 1e-15 {
            Vec3::new(v[0][col] / len, v[1][col] / len, v[2][col] / len)
        } else {
            // Degenerate — use canonical axis.
            match col {
                0 => Vec3::new(1.0, 0.0, 0.0),
                1 => Vec3::new(0.0, 1.0, 0.0),
                _ => Vec3::new(0.0, 0.0, 1.0),
            }
        }
    };

    [
        make_axis(order[0]),
        make_axis(order[1]),
        make_axis(order[2]),
    ]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn obb_from_axis_aligned_points() {
        let pts = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let obb = Obb3::from_points(pts);
        // Center should be at (1, 0.5, 0).
        assert!((obb.center.x() - 1.0).abs() < 1e-10);
        assert!((obb.center.y() - 0.5).abs() < 1e-10);
        assert!((obb.center.z()).abs() < 1e-10);
    }

    #[test]
    fn obb_identical_boxes_intersect() {
        let pts = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
        ];
        let obb = Obb3::from_points(pts);
        assert!(obb.intersects(&obb));
    }

    #[test]
    fn obb_separated_boxes_dont_intersect() {
        let a = Obb3::from_points([
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ]);
        let b = Obb3::from_points([
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(6.0, 0.0, 0.0),
            Point3::new(6.0, 1.0, 0.0),
            Point3::new(5.0, 1.0, 0.0),
        ]);
        assert!(!a.intersects(&b));
    }

    #[test]
    fn obb_overlapping_rotated_boxes_intersect() {
        // A unit square at origin and a rotated square overlapping it.
        let a = Obb3::from_points([
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
        ]);
        // 45-degree rotated square, overlapping.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let b = Obb3::from_points([
            Point3::new(0.0, -s, 0.0),
            Point3::new(s, 0.0, 0.0),
            Point3::new(0.0, s, 0.0),
            Point3::new(-s, 0.0, 0.0),
        ]);
        assert!(a.intersects(&b));
    }

    #[test]
    fn obb_with_normal_planar_face() {
        let pts = [
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(2.0, 0.0, 5.0),
            Point3::new(2.0, 3.0, 5.0),
            Point3::new(0.0, 3.0, 5.0),
        ];
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let obb = Obb3::from_points_with_normal(pts, normal);

        // Thickness along normal should be near zero.
        assert!(obb.half_extents[2] < 1e-10);
    }

    #[test]
    fn obb_edge_touching() {
        // Two boxes sharing an edge at x=1.
        let a = Obb3::from_points([
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ]);
        let b = Obb3::from_points([
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ]);
        // Edge-touching should still be considered intersecting.
        assert!(a.intersects(&b));
    }
}
