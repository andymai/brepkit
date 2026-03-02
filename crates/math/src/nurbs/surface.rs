//! NURBS surface evaluation via tensor-product De Boor.

use crate::MathError;
use crate::aabb::Aabb3;
use crate::nurbs::basis;
use crate::vec::{Point3, Vec3};

/// A Non-Uniform Rational B-Spline (NURBS) surface in 3D space.
///
/// The surface is defined by degrees in the u and v directions, two knot
/// vectors, a 2D grid of control points, and matching weights.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NurbsSurface {
    /// Polynomial degree in the u direction.
    degree_u: usize,
    /// Polynomial degree in the v direction.
    degree_v: usize,
    /// Knot vector in the u direction.
    knots_u: Vec<f64>,
    /// Knot vector in the v direction.
    knots_v: Vec<f64>,
    /// Control point grid indexed as `control_points[row_u][col_v]`.
    control_points: Vec<Vec<Point3>>,
    /// Weight grid matching `control_points` dimensions.
    weights: Vec<Vec<f64>>,
}

impl NurbsSurface {
    /// Construct a new NURBS surface with validation.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::InvalidControlPointGrid`] if the control point
    /// rows have inconsistent lengths.
    ///
    /// Returns [`MathError::InvalidKnotVector`] if either knot vector has the
    /// wrong length for the given degree and control point count.
    ///
    /// Returns [`MathError::InvalidWeights`] if the weights grid dimensions
    /// do not match the control point grid.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        degree_u: usize,
        degree_v: usize,
        knots_u: Vec<f64>,
        knots_v: Vec<f64>,
        control_points: Vec<Vec<Point3>>,
        weights: Vec<Vec<f64>>,
    ) -> Result<Self, MathError> {
        let n_rows = control_points.len();

        // Validate that all rows have the same length.
        let n_cols = control_points.first().map_or(0, Vec::len);
        for row in &control_points {
            if row.len() != n_cols {
                return Err(MathError::InvalidControlPointGrid {
                    expected_rows: n_rows,
                    expected_cols: n_cols,
                });
            }
        }

        // Validate knot vectors.
        let expected_knots_u = n_rows + degree_u + 1;
        if knots_u.len() != expected_knots_u {
            return Err(MathError::InvalidKnotVector {
                expected: expected_knots_u,
                got: knots_u.len(),
            });
        }

        let expected_knots_v = n_cols + degree_v + 1;
        if knots_v.len() != expected_knots_v {
            return Err(MathError::InvalidKnotVector {
                expected: expected_knots_v,
                got: knots_v.len(),
            });
        }

        // Validate weights grid dimensions.
        if weights.len() != n_rows {
            return Err(MathError::InvalidWeights {
                expected: n_rows,
                got: weights.len(),
            });
        }
        for row in &weights {
            if row.len() != n_cols {
                return Err(MathError::InvalidWeights {
                    expected: n_cols,
                    got: row.len(),
                });
            }
        }

        Ok(Self {
            degree_u,
            degree_v,
            knots_u,
            knots_v,
            control_points,
            weights,
        })
    }

    /// Polynomial degree in the u direction.
    #[must_use]
    pub const fn degree_u(&self) -> usize {
        self.degree_u
    }

    /// Polynomial degree in the v direction.
    #[must_use]
    pub const fn degree_v(&self) -> usize {
        self.degree_v
    }

    /// Knot vector in the u direction.
    #[must_use]
    pub fn knots_u(&self) -> &[f64] {
        &self.knots_u
    }

    /// Knot vector in the v direction.
    #[must_use]
    pub fn knots_v(&self) -> &[f64] {
        &self.knots_v
    }

    /// Reference to the control point grid.
    #[must_use]
    pub fn control_points(&self) -> &[Vec<Point3>] {
        &self.control_points
    }

    /// Reference to the weights grid.
    #[must_use]
    pub fn weights(&self) -> &[Vec<f64>] {
        &self.weights
    }

    /// Evaluate the surface at parameters `(u, v)`.
    ///
    /// Uses tensor-product basis function evaluation (NURBS Book A3.5).
    #[must_use]
    pub fn evaluate(&self, u: f64, v: f64) -> Point3 {
        let pu = self.degree_u;
        let pv = self.degree_v;
        let n_rows = self.control_points.len();
        let n_cols = self.control_points[0].len();

        let span_u = basis::find_span(n_rows, pu, u, &self.knots_u);
        let span_v = basis::find_span(n_cols, pv, v, &self.knots_v);
        let nu = basis::basis_funs(span_u, u, pu, &self.knots_u);
        let nv = basis::basis_funs(span_v, v, pv, &self.knots_v);

        // Contract along v first for each relevant u-row, then along u.
        let mut wx = 0.0;
        let mut wy = 0.0;
        let mut wz = 0.0;
        let mut ww = 0.0;

        for (i, &nu_i) in nu.iter().enumerate().take(pu + 1) {
            let u_idx = span_u - pu + i;
            // Evaluate the v-direction for this row.
            let mut row_x = 0.0;
            let mut row_y = 0.0;
            let mut row_z = 0.0;
            let mut row_w = 0.0;
            for (j, &nv_j) in nv.iter().enumerate().take(pv + 1) {
                let v_idx = span_v - pv + j;
                let pt = &self.control_points[u_idx][v_idx];
                let w = self.weights[u_idx][v_idx];
                let bw = nv_j * w;
                row_x += bw * pt.x();
                row_y += bw * pt.y();
                row_z += bw * pt.z();
                row_w += bw;
            }
            wx += nu_i * row_x;
            wy += nu_i * row_y;
            wz += nu_i * row_z;
            ww += nu_i * row_w;
        }

        if ww == 0.0 {
            Point3::new(wx, wy, wz)
        } else {
            Point3::new(wx / ww, wy / ww, wz / ww)
        }
    }

    /// Compute surface derivatives up to order `d` at parameters `(u, v)`.
    ///
    /// Returns a 2D vector `ders[k][l]` representing the mixed partial
    /// derivative `∂^(k+l)S / ∂u^k ∂v^l` as a `Vec3`.
    ///
    /// Uses NURBS Book A3.6 + A4.4 (rational quotient rule).
    #[must_use]
    #[allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
    pub fn derivatives(&self, u: f64, v: f64, d: usize) -> Vec<Vec<Vec3>> {
        let pu = self.degree_u;
        let pv = self.degree_v;
        let n_rows = self.control_points.len();
        let n_cols = self.control_points[0].len();

        let span_u = basis::find_span(n_rows, pu, u, &self.knots_u);
        let span_v = basis::find_span(n_cols, pv, v, &self.knots_v);
        let du = d.min(pu);
        let dv = d.min(pv);
        let ders_u = basis::ders_basis_funs(span_u, u, pu, du, &self.knots_u);
        let ders_v = basis::ders_basis_funs(span_v, v, pv, dv, &self.knots_v);

        // Compute homogeneous derivatives Aw[k][l] = (wx, wy, wz, w)
        let mut aw = vec![vec![[0.0f64; 4]; d + 1]; d + 1];
        for k in 0..=du {
            for l in 0..=dv {
                if k + l > d {
                    continue;
                }
                for (i, &du_ki) in ders_u[k].iter().enumerate().take(pu + 1) {
                    let u_idx = span_u - pu + i;
                    for (j, &dv_lj) in ders_v[l].iter().enumerate().take(pv + 1) {
                        let v_idx = span_v - pv + j;
                        let pt = &self.control_points[u_idx][v_idx];
                        let w = self.weights[u_idx][v_idx];
                        let coeff = du_ki * dv_lj;
                        aw[k][l][0] += coeff * pt.x() * w;
                        aw[k][l][1] += coeff * pt.y() * w;
                        aw[k][l][2] += coeff * pt.z() * w;
                        aw[k][l][3] += coeff * w;
                    }
                }
            }
        }

        // Apply rational quotient rule (A4.4).
        let zero = Vec3::new(0.0, 0.0, 0.0);
        let mut skl = vec![vec![zero; d + 1]; d + 1];
        let w0 = aw[0][0][3];

        for k in 0..=du {
            for l in 0..=dv {
                if k + l > d {
                    continue;
                }
                let mut v3 = [aw[k][l][0], aw[k][l][1], aw[k][l][2]];

                for j in 1..=l {
                    let bin = binomial(l, j) as f64;
                    v3[0] -= bin * aw[0][j][3] * skl[k][l - j].x();
                    v3[1] -= bin * aw[0][j][3] * skl[k][l - j].y();
                    v3[2] -= bin * aw[0][j][3] * skl[k][l - j].z();
                }

                for i in 1..=k {
                    let bin = binomial(k, i) as f64;
                    v3[0] -= bin * aw[i][0][3] * skl[k - i][l].x();
                    v3[1] -= bin * aw[i][0][3] * skl[k - i][l].y();
                    v3[2] -= bin * aw[i][0][3] * skl[k - i][l].z();

                    let mut v2 = [0.0f64; 3];
                    for j in 1..=l {
                        let bin2 = binomial(l, j) as f64;
                        v2[0] += bin2 * aw[i][j][3] * skl[k - i][l - j].x();
                        v2[1] += bin2 * aw[i][j][3] * skl[k - i][l - j].y();
                        v2[2] += bin2 * aw[i][j][3] * skl[k - i][l - j].z();
                    }
                    v3[0] -= bin * v2[0];
                    v3[1] -= bin * v2[1];
                    v3[2] -= bin * v2[2];
                }

                if w0 == 0.0 {
                    skl[k][l] = Vec3::new(v3[0], v3[1], v3[2]);
                } else {
                    skl[k][l] = Vec3::new(v3[0] / w0, v3[1] / w0, v3[2] / w0);
                }
            }
        }

        skl
    }

    /// Compute the unit normal vector at parameters `(u, v)`.
    ///
    /// The normal is the cross product of the u- and v-partial derivatives,
    /// normalized.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ZeroVector`] if the surface is degenerate at
    /// this point (both partials are parallel or zero).
    pub fn normal(&self, u: f64, v: f64) -> Result<Vec3, MathError> {
        let d = self.derivatives(u, v, 1);
        let du = d[1][0];
        let dv = d[0][1];
        du.cross(dv).normalize()
    }

    /// Compute an axis-aligned bounding box from control point extrema.
    #[must_use]
    pub fn aabb(&self) -> Aabb3 {
        Aabb3::from_points(
            self.control_points
                .iter()
                .flat_map(|row| row.iter().copied()),
        )
    }
}

use super::basis::binomial;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::cast_lossless, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    /// A bilinear surface (degree 1x1): a flat quadrilateral.
    fn bilinear_surface() -> NurbsSurface {
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .expect("valid bilinear surface")
    }

    /// A bicubic surface patch.
    fn bicubic_surface() -> NurbsSurface {
        let mut cps = Vec::new();
        let mut ws = Vec::new();
        for i in 0..4 {
            let mut row = Vec::new();
            let mut wrow = Vec::new();
            for j in 0..4 {
                row.push(Point3::new(
                    j as f64,
                    i as f64,
                    ((i + j) as f64 * 0.5).sin(),
                ));
                wrow.push(1.0);
            }
            cps.push(row);
            ws.push(wrow);
        }
        NurbsSurface::new(
            3,
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            cps,
            ws,
        )
        .expect("valid bicubic surface")
    }

    #[test]
    fn bilinear_corners() {
        let s = bilinear_surface();
        let p00 = s.evaluate(0.0, 0.0);
        let p10 = s.evaluate(1.0, 0.0);
        let p01 = s.evaluate(0.0, 1.0);
        let p11 = s.evaluate(1.0, 1.0);

        assert!((p00.x()).abs() < 1e-14);
        assert!((p00.y()).abs() < 1e-14);
        assert!((p10.x() - 0.0).abs() < 1e-14);
        assert!((p10.y() - 1.0).abs() < 1e-14);
        assert!((p01.x() - 1.0).abs() < 1e-14);
        assert!((p01.y() - 0.0).abs() < 1e-14);
        assert!((p11.x() - 1.0).abs() < 1e-14);
        assert!((p11.y() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn bilinear_midpoint() {
        let s = bilinear_surface();
        let mid = s.evaluate(0.5, 0.5);
        assert!((mid.x() - 0.5).abs() < 1e-14);
        assert!((mid.y() - 0.5).abs() < 1e-14);
        assert!((mid.z()).abs() < 1e-14);
    }

    #[test]
    fn bilinear_normal() {
        let s = bilinear_surface();
        let n = s.normal(0.5, 0.5).expect("non-degenerate");
        // Flat surface in XY plane, normal should be (0, 0, ±1).
        assert!((n.x()).abs() < 1e-12);
        assert!((n.y()).abs() < 1e-12);
        assert!((n.z().abs() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bicubic_endpoint_interpolation() {
        let s = bicubic_surface();
        let p = s.evaluate(0.0, 0.0);
        let cp = &s.control_points()[0][0];
        assert!((p.x() - cp.x()).abs() < 1e-14);
        assert!((p.y() - cp.y()).abs() < 1e-14);
        assert!((p.z() - cp.z()).abs() < 1e-14);
    }

    #[test]
    fn derivatives_zeroth_matches_evaluate() {
        let s = bicubic_surface();
        let p = s.evaluate(0.5, 0.5);
        let d = s.derivatives(0.5, 0.5, 1);
        assert!((d[0][0].x() - p.x()).abs() < 1e-12);
        assert!((d[0][0].y() - p.y()).abs() < 1e-12);
        assert!((d[0][0].z() - p.z()).abs() < 1e-12);
    }

    #[test]
    fn aabb_contains_all_control_points() {
        let s = bicubic_surface();
        let bb = s.aabb();
        for row in s.control_points() {
            for pt in row {
                assert!(bb.contains_point(*pt));
            }
        }
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_bilinear_linear_interpolation(u in 0.0f64..=1.0, v in 0.0f64..=1.0) {
            let s = bilinear_surface();
            let p = s.evaluate(u, v);
            // Bilinear: S(u,v) = (v, u, 0) for our test surface
            prop_assert!((p.x() - v).abs() < 1e-12, "x: {} vs {}", p.x(), v);
            prop_assert!((p.y() - u).abs() < 1e-12, "y: {} vs {}", p.y(), u);
            prop_assert!(p.z().abs() < 1e-12);
        }
    }
}
