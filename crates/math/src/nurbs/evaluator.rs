//! Cached evaluator for repeated NURBS surface evaluation.
//!
//! [`SurfaceEvaluator`] wraps a `&NurbsSurface` and lazily precomputes
//! [`PowerBasis1D`] coefficients for both u and v directions. All subsequent
//! evaluations use Horner's method with zero heap allocations.

use crate::nurbs::basis;
use crate::nurbs::power_basis::PowerBasis1D;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

/// Cached evaluator for a NURBS surface.
///
/// Wraps a `&NurbsSurface` and lazily precomputes `PowerBasis1D` for both
/// u and v directions on first use. All subsequent evaluations use Horner's
/// method with zero heap allocations.
pub struct SurfaceEvaluator<'a> {
    surface: &'a NurbsSurface,
    power_u: Option<PowerBasis1D>,
    power_v: Option<PowerBasis1D>,
    uniform_step_u: Option<f64>,
    uniform_step_v: Option<f64>,
}

impl<'a> SurfaceEvaluator<'a> {
    /// Create a new evaluator for the given surface.
    ///
    /// Detects uniform knot spacing for O(1) span lookup. Power-basis
    /// coefficients are computed lazily on first evaluation.
    #[must_use]
    pub fn new(surface: &'a NurbsSurface) -> Self {
        let uniform_step_u = basis::uniform_knot_step(surface.knots_u(), surface.degree_u());
        let uniform_step_v = basis::uniform_knot_step(surface.knots_v(), surface.degree_v());
        Self {
            surface,
            power_u: None,
            power_v: None,
            uniform_step_u,
            uniform_step_v,
        }
    }

    /// Ensure power-basis coefficients are computed.
    fn ensure_power_basis(&mut self) {
        if self.power_u.is_none() {
            self.power_u = Some(PowerBasis1D::from_knots(
                self.surface.knots_u(),
                self.surface.degree_u(),
            ));
        }
        if self.power_v.is_none() {
            self.power_v = Some(PowerBasis1D::from_knots(
                self.surface.knots_v(),
                self.surface.degree_v(),
            ));
        }
    }

    /// Find span in u direction, using uniform O(1) lookup if possible.
    fn find_span_u(&self, u: f64) -> usize {
        let n = self.surface.control_points().len();
        let pu = self.surface.degree_u();
        if let Some(step) = self.uniform_step_u {
            basis::find_span_uniform(n, pu, u, self.surface.knots_u(), step)
        } else {
            basis::find_span(n, pu, u, self.surface.knots_u())
        }
    }

    /// Find span in v direction, using uniform O(1) lookup if possible.
    fn find_span_v(&self, v: f64) -> usize {
        let n = self.surface.control_points()[0].len();
        let pv = self.surface.degree_v();
        if let Some(step) = self.uniform_step_v {
            basis::find_span_uniform(n, pv, v, self.surface.knots_v(), step)
        } else {
            basis::find_span(n, pv, v, self.surface.knots_v())
        }
    }

    /// Evaluate the surface position at parameters `(u, v)`.
    ///
    /// Uses precomputed power-basis coefficients with Horner evaluation,
    /// avoiding the O(p^2) Cox-de Boor recurrence on each call.
    #[allow(clippy::many_single_char_names)]
    pub fn point(&mut self, u: f64, v: f64) -> Point3 {
        self.ensure_power_basis();

        let pu = self.surface.degree_u();
        let pv = self.surface.degree_v();
        let span_u = self.find_span_u(u);
        let span_v = self.find_span_v(v);

        let mut nu = [0.0_f64; basis::MAX_STACK_OUTPUT + 1];
        let mut nv = [0.0_f64; basis::MAX_STACK_OUTPUT + 1];

        // SAFETY of indexing: power_u/power_v are guaranteed Some after ensure_power_basis.
        // Using if-let to satisfy no-panic lint.
        if let Some(ref pb_u) = self.power_u {
            pb_u.horner(span_u, u, &mut nu[..=pu]);
        }
        if let Some(ref pb_v) = self.power_v {
            pb_v.horner(span_v, v, &mut nv[..=pv]);
        }

        let cps = self.surface.control_points();
        let ws = self.surface.weights();

        // Tensor-product contraction: v first per u-row, then u.
        let mut wx = 0.0;
        let mut wy = 0.0;
        let mut wz = 0.0;
        let mut ww = 0.0;

        for (i, &nu_i) in nu.iter().enumerate().take(pu + 1) {
            let u_idx = span_u - pu + i;
            let mut row_x = 0.0;
            let mut row_y = 0.0;
            let mut row_z = 0.0;
            let mut row_w = 0.0;
            for (j, &nv_j) in nv.iter().enumerate().take(pv + 1) {
                let v_idx = span_v - pv + j;
                let pt = &cps[u_idx][v_idx];
                let w = ws[u_idx][v_idx];
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

    /// Evaluate the unit normal at parameters `(u, v)`.
    ///
    /// Uses precomputed power-basis coefficients with Horner evaluation for
    /// both basis values and first derivatives. If the cross product of the
    /// partial derivatives is degenerate, falls back to the surface's own
    /// `normal()` method, and ultimately to `(0, 0, 1)`.
    #[allow(clippy::many_single_char_names, clippy::too_many_lines)]
    pub fn normal(&mut self, u: f64, v: f64) -> Vec3 {
        self.ensure_power_basis();

        let pu = self.surface.degree_u();
        let pv = self.surface.degree_v();
        let span_u = self.find_span_u(u);
        let span_v = self.find_span_v(v);

        let mut nu = [0.0_f64; basis::MAX_STACK_OUTPUT + 1];
        let mut dnu = [0.0_f64; basis::MAX_STACK_OUTPUT + 1];
        let mut nv = [0.0_f64; basis::MAX_STACK_OUTPUT + 1];
        let mut dnv = [0.0_f64; basis::MAX_STACK_OUTPUT + 1];

        if let Some(ref pb_u) = self.power_u {
            pb_u.horner_with_derivs(span_u, u, &mut nu[..=pu], &mut dnu[..=pu]);
        }
        if let Some(ref pb_v) = self.power_v {
            pb_v.horner_with_derivs(span_v, v, &mut nv[..=pv], &mut dnv[..=pv]);
        }

        let cps = self.surface.control_points();
        let ws = self.surface.weights();

        // Compute homogeneous sums for position and partial derivatives.
        let mut s0 = [0.0_f64; 3]; // sum(nu * nv * w * P)
        let mut w0 = 0.0_f64; // sum(nu * nv * w)
        let mut su = [0.0_f64; 3]; // sum(dnu * nv * w * P)
        let mut wu = 0.0_f64; // sum(dnu * nv * w)
        let mut sv = [0.0_f64; 3]; // sum(nu * dnv * w * P)
        let mut wv = 0.0_f64; // sum(nu * dnv * w)

        for (i, (&nu_i, &dnu_i)) in nu.iter().zip(dnu.iter()).enumerate().take(pu + 1) {
            let u_idx = span_u - pu + i;
            for (j, (&nv_j, &dnv_j)) in nv.iter().zip(dnv.iter()).enumerate().take(pv + 1) {
                let v_idx = span_v - pv + j;
                let pt = &cps[u_idx][v_idx];
                let w = ws[u_idx][v_idx];
                let px = pt.x();
                let py = pt.y();
                let pz = pt.z();

                let nn_w = nu_i * nv_j * w;
                s0[0] += nn_w * px;
                s0[1] += nn_w * py;
                s0[2] += nn_w * pz;
                w0 += nn_w;

                let dn_w = dnu_i * nv_j * w;
                su[0] += dn_w * px;
                su[1] += dn_w * py;
                su[2] += dn_w * pz;
                wu += dn_w;

                let nd_w = nu_i * dnv_j * w;
                sv[0] += nd_w * px;
                sv[1] += nd_w * py;
                sv[2] += nd_w * pz;
                wv += nd_w;
            }
        }

        // Apply rational quotient rule: d/du = (S_u - W_u * P) / W_0
        if w0.abs() < f64::MIN_POSITIVE {
            return self
                .surface
                .normal(u, v)
                .unwrap_or_else(|_| Vec3::new(0.0, 0.0, 1.0));
        }

        let inv_w0 = 1.0 / w0;
        let px = s0[0] * inv_w0;
        let py = s0[1] * inv_w0;
        let pz = s0[2] * inv_w0;

        let du = Vec3::new(
            (su[0] - wu * px) * inv_w0,
            (su[1] - wu * py) * inv_w0,
            (su[2] - wu * pz) * inv_w0,
        );
        let dv = Vec3::new(
            (sv[0] - wv * px) * inv_w0,
            (sv[1] - wv * py) * inv_w0,
            (sv[2] - wv * pz) * inv_w0,
        );

        let cross = du.cross(dv);
        if cross.length_squared() > 1e-30 {
            cross
                .normalize()
                .unwrap_or_else(|_| Vec3::new(0.0, 0.0, 1.0))
        } else {
            // Fall back to the surface's own normal method.
            self.surface
                .normal(u, v)
                .unwrap_or_else(|_| Vec3::new(0.0, 0.0, 1.0))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::vec::Point3;

    fn bicubic_surface() -> NurbsSurface {
        let mut cps = Vec::new();
        let mut ws = Vec::new();
        for i in 0..4 {
            let mut row = Vec::new();
            let mut wrow = Vec::new();
            #[allow(clippy::cast_precision_loss)]
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
    fn surface_evaluator_matches_evaluate() {
        let surface = bicubic_surface();
        let mut eval = surface.evaluator();
        // Test on a grid
        for i in 0..=10 {
            for j in 0..=10 {
                #[allow(clippy::cast_precision_loss)]
                let u = i as f64 / 10.0;
                #[allow(clippy::cast_precision_loss)]
                let v = j as f64 / 10.0;
                let expected = surface.evaluate(u, v);
                let got = eval.point(u, v);
                assert!(
                    (expected.x() - got.x()).abs() < 1e-10
                        && (expected.y() - got.y()).abs() < 1e-10
                        && (expected.z() - got.z()).abs() < 1e-10,
                    "mismatch at ({u},{v}): {expected:?} vs {got:?}"
                );
            }
        }
    }

    #[test]
    fn surface_evaluator_normal_matches() {
        let surface = bicubic_surface();
        let mut eval = surface.evaluator();
        for i in 1..10 {
            for j in 1..10 {
                #[allow(clippy::cast_precision_loss)]
                let u = i as f64 / 10.0;
                #[allow(clippy::cast_precision_loss)]
                let v = j as f64 / 10.0;
                let expected = surface.normal(u, v).expect("non-degenerate");
                let got = eval.normal(u, v);
                let dot = expected.x() * got.x() + expected.y() * got.y() + expected.z() * got.z();
                assert!(
                    (dot - 1.0).abs() < 1e-8,
                    "normal mismatch at ({u},{v}): dot={dot}"
                );
            }
        }
    }
}
