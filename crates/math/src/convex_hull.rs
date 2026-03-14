//! 3D convex hull via incremental Quickhull algorithm.
//!
//! Builds a convex polyhedron from a point cloud. Returns indexed triangle
//! faces suitable for conversion to B-Rep topology.
//!
//! # Algorithm
//!
//! 1. Find an initial tetrahedron from 4 non-coplanar points.
//! 2. For each remaining point, find visible faces (point is above the face plane).
//! 3. Remove visible faces, leaving a horizon ridge.
//! 4. Connect the new point to each horizon edge to form new faces.
//! 5. Repeat until no points remain above any face.
//!
//! # References
//!
//! Barber, Dobkin, Huhdanpaa — "The Quickhull Algorithm for Convex Hulls" (1996)

use crate::vec::{Point3, Vec3};

/// A convex hull result: vertices and triangular faces (CCW winding).
#[derive(Debug, Clone)]
pub struct ConvexHull {
    /// Vertex positions.
    pub vertices: Vec<Point3>,
    /// Triangle faces as index triples (indices into `vertices`).
    pub faces: Vec<[usize; 3]>,
}

/// Compute the 3D convex hull of a point cloud.
///
/// # Errors
///
/// Returns `None` if fewer than 4 non-coplanar points are provided.
#[must_use]
pub fn convex_hull_3d(points: &[Point3]) -> Option<ConvexHull> {
    if points.len() < 4 {
        return None;
    }

    // Deduplicate points within tolerance.
    let tol = 1e-10;
    let mut pts: Vec<Point3> = Vec::with_capacity(points.len());
    for &p in points {
        let dominated = pts.iter().any(|q| (*q - p).length() < tol);
        if !dominated {
            pts.push(p);
        }
    }
    if pts.len() < 4 {
        return None;
    }

    // --- Step 1: Find initial tetrahedron ---
    let tet = find_initial_tetrahedron(&pts)?;

    // Build initial face list (4 triangles of the tetrahedron).
    let mut faces: Vec<HullFace> = Vec::new();
    let tet_faces = [
        [tet[0], tet[1], tet[2]],
        [tet[0], tet[2], tet[3]],
        [tet[0], tet[3], tet[1]],
        [tet[1], tet[3], tet[2]],
    ];

    for &[a, b, c] in &tet_faces {
        let normal = face_normal(&pts, a, b, c);
        let d = -(normal.x() * pts[a].x() + normal.y() * pts[a].y() + normal.z() * pts[a].z());
        faces.push(HullFace {
            verts: [a, b, c],
            normal,
            d,
            alive: true,
        });
    }

    // Ensure all faces point outward (centroid test).
    let centroid = Point3::new(
        (pts[tet[0]].x() + pts[tet[1]].x() + pts[tet[2]].x() + pts[tet[3]].x()) / 4.0,
        (pts[tet[0]].y() + pts[tet[1]].y() + pts[tet[2]].y() + pts[tet[3]].y()) / 4.0,
        (pts[tet[0]].z() + pts[tet[1]].z() + pts[tet[2]].z() + pts[tet[3]].z()) / 4.0,
    );
    for face in &mut faces {
        let signed = signed_distance(face, centroid);
        if signed > 0.0 {
            // Normal points inward — flip.
            face.normal = -face.normal;
            face.d = -face.d;
            face.verts.swap(1, 2);
        }
    }

    // --- Step 2-5: Incremental insertion ---
    let tet_set: std::collections::HashSet<usize> = tet.iter().copied().collect();

    for (pi, &point) in pts.iter().enumerate() {
        if tet_set.contains(&pi) {
            continue;
        }

        // Find visible faces.
        let mut visible: Vec<usize> = Vec::new();
        for (fi, face) in faces.iter().enumerate() {
            if face.alive && signed_distance(face, point) > tol {
                visible.push(fi);
            }
        }

        if visible.is_empty() {
            continue; // Point is inside the hull.
        }

        // Find horizon edges (edges shared by exactly one visible face).
        let mut horizon: Vec<[usize; 2]> = Vec::new();
        for &fi in &visible {
            let f = &faces[fi];
            for edge_idx in 0..3 {
                let e = [f.verts[edge_idx], f.verts[(edge_idx + 1) % 3]];
                // Check if the opposite face (sharing reversed edge) is NOT visible.
                let twin_visible = visible
                    .iter()
                    .any(|&fj| fj != fi && faces[fj].has_edge(e[1], e[0]));
                if !twin_visible {
                    horizon.push(e);
                }
            }
        }

        // Kill visible faces.
        for &fi in &visible {
            faces[fi].alive = false;
        }

        // Create new faces from horizon edges to the new point.
        for &[a, b] in &horizon {
            let normal = face_normal(&pts, a, b, pi);
            let d = -(normal.x() * pts[a].x() + normal.y() * pts[a].y() + normal.z() * pts[a].z());
            let mut new_face = HullFace {
                verts: [a, b, pi],
                normal,
                d,
                alive: true,
            };
            // Ensure outward orientation.
            if signed_distance(&new_face, centroid) > 0.0 {
                new_face.normal = Vec3::new(
                    -new_face.normal.x(),
                    -new_face.normal.y(),
                    -new_face.normal.z(),
                );
                new_face.d = -new_face.d;
                new_face.verts.swap(0, 1);
            }
            faces.push(new_face);
        }
    }

    // Collect alive faces.
    let alive_faces: Vec<[usize; 3]> = faces.iter().filter(|f| f.alive).map(|f| f.verts).collect();

    if alive_faces.is_empty() {
        return None;
    }

    Some(ConvexHull {
        vertices: pts,
        faces: alive_faces,
    })
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

struct HullFace {
    verts: [usize; 3],
    normal: Vec3,
    d: f64,
    alive: bool,
}

impl HullFace {
    fn has_edge(&self, a: usize, b: usize) -> bool {
        for i in 0..3 {
            if self.verts[i] == a && self.verts[(i + 1) % 3] == b {
                return true;
            }
        }
        false
    }
}

fn signed_distance(face: &HullFace, point: Point3) -> f64 {
    face.normal.x() * point.x() + face.normal.y() * point.y() + face.normal.z() * point.z() + face.d
}

fn face_normal(pts: &[Point3], a: usize, b: usize, c: usize) -> Vec3 {
    let ab = pts[b] - pts[a];
    let ac = pts[c] - pts[a];
    let n = ab.cross(ac);
    let len = n.length();
    if len < 1e-15 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(n.x() / len, n.y() / len, n.z() / len)
    }
}

/// Find 4 non-coplanar points for the initial tetrahedron.
fn find_initial_tetrahedron(pts: &[Point3]) -> Option<[usize; 4]> {
    let n = pts.len();

    // Find two points that are farthest apart.
    let mut i0 = 0;
    let mut i1 = 1;
    let mut max_dist = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = (pts[j] - pts[i]).length();
            if d > max_dist {
                max_dist = d;
                i0 = i;
                i1 = j;
            }
        }
    }
    if max_dist < 1e-12 {
        return None;
    }

    // Find a third point farthest from the line i0-i1.
    let dir = pts[i1] - pts[i0];
    let dir_len = dir.length();
    let dir_n = Vec3::new(dir.x() / dir_len, dir.y() / dir_len, dir.z() / dir_len);
    let mut i2 = 0;
    let mut max_dist2 = 0.0_f64;
    for i in 0..n {
        if i == i0 || i == i1 {
            continue;
        }
        let v = pts[i] - pts[i0];
        let proj = v.x() * dir_n.x() + v.y() * dir_n.y() + v.z() * dir_n.z();
        let perp = Vec3::new(
            v.x() - proj * dir_n.x(),
            v.y() - proj * dir_n.y(),
            v.z() - proj * dir_n.z(),
        );
        let d = perp.length();
        if d > max_dist2 {
            max_dist2 = d;
            i2 = i;
        }
    }
    if max_dist2 < 1e-12 {
        return None;
    }

    // Find a fourth point farthest from the plane i0-i1-i2.
    let plane_n = face_normal(pts, i0, i1, i2);
    let plane_d =
        -(plane_n.x() * pts[i0].x() + plane_n.y() * pts[i0].y() + plane_n.z() * pts[i0].z());
    let mut i3 = 0;
    let mut max_dist3 = 0.0_f64;
    for i in 0..n {
        if i == i0 || i == i1 || i == i2 {
            continue;
        }
        let d = (plane_n.x() * pts[i].x()
            + plane_n.y() * pts[i].y()
            + plane_n.z() * pts[i].z()
            + plane_d)
            .abs();
        if d > max_dist3 {
            max_dist3 = d;
            i3 = i;
        }
    }
    if max_dist3 < 1e-12 {
        return None;
    }

    Some([i0, i1, i2, i3])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn hull_of_cube_vertices() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];
        let hull = convex_hull_3d(&points).expect("hull should succeed");
        // Cube has 8 vertices and 12 triangular faces (6 quads, each split into 2 triangles).
        assert_eq!(hull.vertices.len(), 8);
        assert_eq!(hull.faces.len(), 12);
    }

    #[test]
    fn hull_of_tetrahedron() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
            Point3::new(0.5, 0.5, 1.0),
        ];
        let hull = convex_hull_3d(&points).expect("hull should succeed");
        assert_eq!(hull.vertices.len(), 4);
        assert_eq!(hull.faces.len(), 4);
    }

    #[test]
    fn hull_with_interior_points() {
        // 8 cube corners + 1 interior point
        let mut points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];
        points.push(Point3::new(0.5, 0.5, 0.5)); // interior
        let hull = convex_hull_3d(&points).expect("hull should succeed");
        // Interior point should be ignored — still a cube.
        assert_eq!(hull.faces.len(), 12);
    }

    #[test]
    fn hull_rejects_coplanar_points() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];
        assert!(convex_hull_3d(&points).is_none());
    }

    #[test]
    fn hull_rejects_too_few_points() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        assert!(convex_hull_3d(&points).is_none());
    }
}
