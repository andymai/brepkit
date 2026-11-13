//! NURBS surface evaluation via tensor-product De Boor.

use crate::MathError;
use crate::aabb::Aabb3;
use crate::nurbs::basis;
use crate::nurbs::evaluator::SurfaceEvaluator;
use crate::vec::{Point3, Vec3};

/// A Non-Uniform Rational B-Spline (NURBS) surface in 3D space.
///
/// The surface is defined by degrees in the u and v directions, two knot
/// vectors, a 2D grid of control points, and matching weights.
#[derive(Debug, Clone)]
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
    /// What the surface keeps once worked out (see [`Caches`]).
    #[cfg_attr(feature = "serde", serde(skip))]
    caches: Box<Caches>,
}

/// What a surface keeps once worked out, each part made on first use: a
/// surface is never changed once made. Boxed, so the surface stays small
/// (enums hold surfaces by value), and cloned with it, not shared (the ray
/// patches hold a clone of their surface, which must not hold them back).
#[derive(Clone, Default)]
struct Caches {
    /// The grid every projection seeds from.
    coarse_grid: std::sync::OnceLock<std::sync::Arc<crate::nurbs::projection::CoarseGrid>>,
    /// The control points' box.
    control_box: std::sync::OnceLock<Aabb3>,
    /// The Bezier patches rays are cast against.
    ray_surface:
        std::sync::OnceLock<Option<std::sync::Arc<crate::nurbs::intersection::RaySurface>>>,
    /// Points projected onto the surface (see [`NurbsSurface::projected`]).
    projections: ProjectionMemo,
}

/// Points already projected onto a surface, by their coordinates' bits, at
/// most [`ProjectionMemo::LIMIT`] of them. A clone starts empty.
#[derive(Debug, Default)]
struct ProjectionMemo(std::sync::Mutex<std::collections::HashMap<[u64; 3], (f64, f64)>>);

impl ProjectionMemo {
    const LIMIT: usize = 1 << 14;
}

impl Clone for ProjectionMemo {
    fn clone(&self) -> Self {
        Self::default()
    }
}

/// The surfaces split into Bezier patches lately, anywhere: each one's
/// content key, its data, and its patches, the most lately used last, at most
/// [`SPLIT_PATCHES`] patches in all.
static SPLIT: std::sync::Mutex<Vec<SplitEntry>> = std::sync::Mutex::new(Vec::new());

/// A surface's content key, its data, and its Bezier patches.
type SplitEntry = (
    u64,
    NurbsSurface,
    std::sync::Arc<Vec<crate::nurbs::decompose::BezierPatch>>,
);

/// The entry of `kept` for the surface `key` and `surface` name, moved to
/// the end as the most lately used.
fn split_kept(
    kept: &mut Vec<SplitEntry>,
    key: u64,
    surface: &NurbsSurface,
) -> Option<std::sync::Arc<Vec<crate::nurbs::decompose::BezierPatch>>> {
    let at = kept
        .iter()
        .position(|(k, held, _)| *k == key && held.same_bits(surface))?;
    let entry = kept.remove(at);
    let patches = entry.2.clone();
    kept.push(entry);
    Some(patches)
}

/// Keeps `entry` in `kept` as the most lately used, within `limit` patches
/// in all, and returns the patches to use: those `kept` already holds for
/// the same surface, where another caller split it meanwhile, else the
/// entry's own. The least lately used entries go first to make room, in
/// one pass; an entry over the limit by itself is not kept.
fn keep_split(
    kept: &mut Vec<SplitEntry>,
    entry: SplitEntry,
    limit: usize,
) -> std::sync::Arc<Vec<crate::nurbs::decompose::BezierPatch>> {
    if let Some(held) = split_kept(kept, entry.0, &entry.1) {
        return held;
    }
    let patches = entry.2.clone();
    if patches.len() > limit {
        return patches;
    }
    let mut total: usize = kept.iter().map(|e| e.2.len()).sum();
    let mut gone = 0;
    while total + patches.len() > limit {
        total -= kept[gone].2.len();
        gone += 1;
    }
    kept.drain(..gone);
    kept.push(entry);
    patches
}

/// How many Bezier patches [`SPLIT`] keeps in all.
const SPLIT_PATCHES: usize = 50_000;

/// What a surface keeps is no part of its value: it prints as nothing, so
/// a surface prints the same whatever it has worked out.
impl std::fmt::Debug for Caches {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("..")
    }
}

impl PartialEq for NurbsSurface {
    fn eq(&self, other: &Self) -> bool {
        self.degree_u == other.degree_u
            && self.degree_v == other.degree_v
            && self.knots_u == other.knots_u
            && self.knots_v == other.knots_v
            && self.control_points == other.control_points
            && self.weights == other.weights
    }
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
            caches: Box::default(),
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

    /// Whether the surface is rational (any weight differs from 1.0).
    #[must_use]
    #[allow(clippy::float_cmp)]
    pub fn is_rational(&self) -> bool {
        self.weights.iter().flatten().any(|&w| w != 1.0)
    }

    /// Return the valid parameter domain in u: `[u_min, u_max]`.
    #[must_use]
    pub fn domain_u(&self) -> (f64, f64) {
        let u_min = self.knots_u[self.degree_u];
        let u_max = self.knots_u[self.knots_u.len() - self.degree_u - 1];
        (u_min, u_max)
    }

    /// Return the valid parameter domain in v: `[v_min, v_max]`.
    #[must_use]
    pub fn domain_v(&self) -> (f64, f64) {
        let v_min = self.knots_v[self.degree_v];
        let v_max = self.knots_v[self.knots_v.len() - self.degree_v - 1];
        (v_min, v_max)
    }

    /// Whether the surface is periodic (closed) in u.
    ///
    /// A NURBS surface is considered periodic in u if the first and last
    /// control point rows coincide within a tight tolerance. This is true
    /// for surfaces converted from analytic periodic types (cylinder, cone,
    /// sphere, torus).
    #[must_use]
    pub fn is_periodic_u(&self) -> bool {
        let n = self.control_points.len();
        if n < 2 {
            return false;
        }
        let first = &self.control_points[0];
        let last = &self.control_points[n - 1];
        if first.len() != last.len() {
            return false;
        }
        // (1e-7)^2 matching Tolerance::default().linear
        first.iter().zip(last.iter()).all(|(a, b)| {
            let d = *a - *b;
            d.x() * d.x() + d.y() * d.y() + d.z() * d.z() < 1e-14
        })
    }

    /// Whether the surface is periodic (closed) in v.
    ///
    /// A NURBS surface is considered periodic in v if the first and last
    /// control point columns coincide within a tight tolerance.
    #[must_use]
    pub fn is_periodic_v(&self) -> bool {
        if self.control_points.is_empty() {
            return false;
        }
        // (1e-7)^2 matching Tolerance::default().linear
        self.control_points.iter().all(|row| {
            if row.len() < 2 {
                return false;
            }
            let d = row[0] - row[row.len() - 1];
            d.x() * d.x() + d.y() * d.y() + d.z() * d.z() < 1e-14
        })
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
        let pv = self.degree_v;
        let n_cols = self.control_points[0].len();
        let span_v = basis::find_span(n_cols, pv, v, &self.knots_v);
        let mut nv_stack = [0.0f64; basis::MAX_STACK_OUTPUT + 1];
        let mut nv_heap = Vec::new();
        let nv = basis::scratch(&mut nv_stack, &mut nv_heap, pv + 1);
        basis::basis_funs_into(span_v, v, pv, &self.knots_v, nv);
        self.evaluate_on_row(u, span_v, nv)
    }

    /// [`Self::evaluate`] at `u`, given `v`'s span and basis functions.
    fn evaluate_on_row(&self, u: f64, span_v: usize, nv: &[f64]) -> Point3 {
        let pu = self.degree_u;
        let pv = self.degree_v;
        let n_rows = self.control_points.len();

        let span_u = basis::find_span(n_rows, pu, u, &self.knots_u);
        let mut nu_stack = [0.0f64; basis::MAX_STACK_OUTPUT + 1];
        let mut nu_heap = Vec::new();
        let nu = basis::scratch(&mut nu_stack, &mut nu_heap, pu + 1);
        basis::basis_funs_into(span_u, u, pu, &self.knots_u, nu);

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
    pub fn derivatives(&self, u: f64, v: f64, d: usize) -> Vec<Vec<Vec3>> {
        let n = d + 1;
        let zero = Vec3::new(0.0, 0.0, 0.0);
        let (mut aw_stack, mut skl_stack) = ([[0.0f64; 4]; 9], [zero; 9]);
        let (mut aw_heap, mut skl_heap) = (Vec::new(), Vec::new());
        let (aw, skl) = if n * n <= 9 {
            (&mut aw_stack[..n * n], &mut skl_stack[..n * n])
        } else {
            aw_heap.resize(n * n, [0.0f64; 4]);
            skl_heap.resize(n * n, zero);
            (&mut aw_heap[..], &mut skl_heap[..])
        };
        self.derivatives_into(u, v, d, aw, skl);
        skl.chunks(n).map(<[Vec3]>::to_vec).collect()
    }

    /// The point and its first partials `(S, ∂S/∂u, ∂S/∂v)` at `(u, v)`:
    /// [`Self::derivatives`] to order 1, worked out alike, without allocating.
    #[must_use]
    pub fn first_derivatives(&self, u: f64, v: f64) -> (Vec3, Vec3, Vec3) {
        let (pu, pv) = (self.degree_u, self.degree_v);
        let span_u = basis::find_span(self.control_points.len(), pu, u, &self.knots_u);
        let span_v = basis::find_span(self.control_points[0].len(), pv, v, &self.knots_v);
        let (du, dv) = (1.min(pu), 1.min(pv));
        let mut ders_u_stack = [0.0f64; 2 * (basis::MAX_STACK_OUTPUT + 1)];
        let mut ders_u_heap = Vec::new();
        let ders_u = basis::scratch(&mut ders_u_stack, &mut ders_u_heap, (du + 1) * (pu + 1));
        basis::ders_basis_funs_into(span_u, u, pu, du, &self.knots_u, ders_u);
        let mut ders_v_stack = [0.0f64; 2 * (basis::MAX_STACK_OUTPUT + 1)];
        let mut ders_v_heap = Vec::new();
        let ders_v = basis::scratch(&mut ders_v_stack, &mut ders_v_heap, (dv + 1) * (pv + 1));
        basis::ders_basis_funs_into(span_v, v, pv, dv, &self.knots_v, ders_v);
        let (_, s, su, sv) = self.first_order_from::<false>(span_u, ders_u, span_v, &[], ders_v);
        (s, su, sv)
    }

    /// The point at `(u, v)` as [`Self::evaluate`] gives it and the first
    /// partials as [`Self::first_derivatives`] gives them, from one pass
    /// over the span's control points: the basis functions' values are the
    /// first row of their derivatives, the same numbers either way.
    #[must_use]
    pub fn point_and_first_derivatives(&self, u: f64, v: f64) -> (Point3, Vec3, Vec3) {
        let (pu, pv) = (self.degree_u, self.degree_v);
        let span_u = basis::find_span(self.control_points.len(), pu, u, &self.knots_u);
        let span_v = basis::find_span(self.control_points[0].len(), pv, v, &self.knots_v);
        let (du, dv) = (1.min(pu), 1.min(pv));
        let mut ders_u_stack = [0.0f64; 2 * (basis::MAX_STACK_OUTPUT + 1)];
        let mut ders_u_heap = Vec::new();
        let ders_u = basis::scratch(&mut ders_u_stack, &mut ders_u_heap, (du + 1) * (pu + 1));
        basis::ders_basis_funs_into(span_u, u, pu, du, &self.knots_u, ders_u);
        let mut ders_v_stack = [0.0f64; 2 * (basis::MAX_STACK_OUTPUT + 1)];
        let mut ders_v_heap = Vec::new();
        let ders_v = basis::scratch(&mut ders_v_stack, &mut ders_v_heap, (dv + 1) * (pv + 1));
        basis::ders_basis_funs_into(span_v, v, pv, dv, &self.knots_v, ders_v);
        let (point, _, su, sv) =
            self.first_order_from::<true>(span_u, ders_u, span_v, &ders_v[..=pv], ders_v);
        (point, su, sv)
    }

    /// `u`'s span and basis functions with their first derivatives, for
    /// [`NurbsRow::point_and_partials_with`] to take at many `v`. `None` past
    /// [`basis::MAX_STACK_OUTPUT`] in `u`.
    #[must_use]
    pub fn u_basis(&self, u: f64) -> Option<UBasis> {
        let pu = self.degree_u;
        if pu > basis::MAX_STACK_OUTPUT {
            return None;
        }
        let span = basis::find_span(self.control_points.len(), pu, u, &self.knots_u);
        let mut ders = [0.0; 2 * (basis::MAX_STACK_OUTPUT + 1)];
        let du = 1.min(pu);
        basis::ders_basis_funs_into(
            span,
            u,
            pu,
            du,
            &self.knots_u,
            &mut ders[..(du + 1) * (pu + 1)],
        );
        Some(UBasis { u, span, ders })
    }

    /// The surface along its line of constant `v`, for evaluating many `u`
    /// there: `v`'s span and basis functions are worked out once, and each
    /// point and partial is the one [`Self::evaluate`] and
    /// [`Self::first_derivatives`] give, bit for bit.
    #[must_use]
    pub fn at_v(&self, v: f64) -> NurbsRow<'_> {
        let pv = self.degree_v;
        let span_v = basis::find_span(self.control_points[0].len(), pv, v, &self.knots_v);
        let mut basis_v = vec![0.0; pv + 1];
        basis::basis_funs_into(span_v, v, pv, &self.knots_v, &mut basis_v);
        let dv = 1.min(pv);
        let mut ders_v = vec![0.0; (dv + 1) * (pv + 1)];
        basis::ders_basis_funs_into(span_v, v, pv, dv, &self.knots_v, &mut ders_v);
        NurbsRow {
            surface: self,
            span_v,
            basis_v,
            ders_v,
        }
    }

    /// [`Self::derivatives`] into zeroed `(d + 1)^2` buffers, row `k` of
    /// `skl` holding the `k`-th derivative in `u` against each order in `v`.
    #[allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
    fn derivatives_into(&self, u: f64, v: f64, d: usize, aw: &mut [[f64; 4]], skl: &mut [Vec3]) {
        let pv = self.degree_v;
        let n_cols = self.control_points[0].len();
        let span_v = basis::find_span(n_cols, pv, v, &self.knots_v);
        let dv = d.min(pv);
        let mut ders_v_stack =
            [0.0f64; (basis::MAX_STACK_OUTPUT + 1) * (basis::MAX_STACK_OUTPUT + 1)];
        let mut ders_v_heap = Vec::new();
        let ders_v_buf = basis::scratch(&mut ders_v_stack, &mut ders_v_heap, (dv + 1) * (pv + 1));
        basis::ders_basis_funs_into(span_v, v, pv, dv, &self.knots_v, ders_v_buf);
        self.derivatives_on_row(u, d, span_v, ders_v_buf, aw, skl);
    }

    /// [`Self::derivatives_into`] at `u`, given `v`'s span and the
    /// derivatives of its basis functions to order `d.min(degree_v)`.
    #[allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
    fn derivatives_on_row(
        &self,
        u: f64,
        d: usize,
        span_v: usize,
        ders_v_buf: &[f64],
        aw: &mut [[f64; 4]],
        skl: &mut [Vec3],
    ) {
        let pu = self.degree_u;
        let pv = self.degree_v;
        let n_rows = self.control_points.len();

        let span_u = basis::find_span(n_rows, pu, u, &self.knots_u);
        let du = d.min(pu);
        let dv = d.min(pv);
        let stride_u = pu + 1;
        let mut ders_u_stack =
            [0.0f64; (basis::MAX_STACK_OUTPUT + 1) * (basis::MAX_STACK_OUTPUT + 1)];
        let mut ders_u_heap = Vec::new();
        let ders_u_buf = basis::scratch(&mut ders_u_stack, &mut ders_u_heap, (du + 1) * stride_u);
        basis::ders_basis_funs_into(span_u, u, pu, du, &self.knots_u, ders_u_buf);
        let stride_v = pv + 1;

        // Compute homogeneous derivatives Aw[k][l] = (wx, wy, wz, w). Those
        // above the degrees are zero, but a rational surface's are not (a
        // polynomial one's are, and the quotient would only add rounding).
        // Only the span's own weights say which: equal ones make the surface
        // polynomial there, and reading every weight costs the whole net.
        let (top_u, top_v) = if self.row_varies(span_u, span_v) {
            (d, d)
        } else {
            (du, dv)
        };
        let n = d + 1;
        for k in 0..=du {
            for l in 0..=dv {
                if k + l > d {
                    continue;
                }
                for i in 0..=pu {
                    let du_ki = ders_u_buf[k * stride_u + i];
                    let u_idx = span_u - pu + i;
                    for j in 0..=pv {
                        let dv_lj = ders_v_buf[l * stride_v + j];
                        let v_idx = span_v - pv + j;
                        let pt = &self.control_points[u_idx][v_idx];
                        let w = self.weights[u_idx][v_idx];
                        let coeff = du_ki * dv_lj;
                        aw[k * n + l][0] += coeff * pt.x() * w;
                        aw[k * n + l][1] += coeff * pt.y() * w;
                        aw[k * n + l][2] += coeff * pt.z() * w;
                        aw[k * n + l][3] += coeff * w;
                    }
                }
            }
        }

        rational_quotient(d, top_u, top_v, aw, skl);
    }

    /// The span's weights all alike: the surface is polynomial there, and its
    /// derivatives past the degrees are zero (see `derivatives_on_row`).
    fn row_varies(&self, span_u: usize, span_v: usize) -> bool {
        let (pu, pv) = (self.degree_u, self.degree_v);
        let w0 = self.weights[span_u - pu][span_v - pv].to_bits();
        self.weights[span_u - pu..=span_u]
            .iter()
            .any(|row| row[span_v - pv..=span_v].iter().any(|w| w.to_bits() != w0))
    }

    /// [`NurbsRow::point_and_partials`]: [`Self::evaluate_on_row`] and
    /// [`Self::derivatives_on_row`] to order 1 in one pass over the span's
    /// control points, each sum taken in the order those take it, and `u`'s
    /// basis functions read off the first row of their derivatives (which
    /// `basis_funs_into` works out by the same operations).
    fn point_and_partials_on_row(
        &self,
        u: f64,
        span_v: usize,
        nv: &[f64],
        ders_v: &[f64],
    ) -> (Point3, Vec3, Vec3) {
        let pu = self.degree_u;
        let span_u = basis::find_span(self.control_points.len(), pu, u, &self.knots_u);
        let du = 1.min(pu);
        let stride_u = pu + 1;
        let mut ders_u_stack = [0.0f64; 2 * (basis::MAX_STACK_OUTPUT + 1)];
        let mut ders_u_heap = Vec::new();
        let ders_u = basis::scratch(&mut ders_u_stack, &mut ders_u_heap, (du + 1) * stride_u);
        basis::ders_basis_funs_into(span_u, u, pu, du, &self.knots_u, ders_u);
        let (point, _, su, sv) = self.first_order_from::<true>(span_u, ders_u, span_v, nv, ders_v);
        (point, su, sv)
    }

    /// From `u`'s span and basis derivatives (to order `1.min(degree_u)`)
    /// and `v`'s: [`Self::derivatives_into`] to order 1, `(S, S_u, S_v)`,
    /// its homogeneous sums taken in one pass over the span's control points,
    /// each in the order it takes them; and with `POINT`, the point as
    /// [`Self::evaluate_on_row`] gives it from `v`'s basis functions `nv`
    /// (`u`'s read off the first row of their derivatives, which
    /// `basis_funs_into` works out by the same operations), else the origin.
    #[allow(clippy::many_single_char_names)]
    fn first_order_from<const POINT: bool>(
        &self,
        span_u: usize,
        ders_u: &[f64],
        span_v: usize,
        nv: &[f64],
        ders_v: &[f64],
    ) -> (Point3, Vec3, Vec3, Vec3) {
        let pu = self.degree_u;
        let pv = self.degree_v;
        let du = 1.min(pu);
        let dv = 1.min(pv);
        let (stride_u, stride_v) = (pu + 1, pv + 1);
        let (mut wx, mut wy, mut wz, mut ww) = (0.0, 0.0, 0.0, 0.0);
        let mut aw = [[0.0f64; 4]; 4];
        for i in 0..=pu {
            let u_idx = span_u - pu + i;
            let (du_0i, du_1i) = (ders_u[i], ders_u[(du * stride_u) + i]);
            let (mut row_x, mut row_y, mut row_z, mut row_w) = (0.0, 0.0, 0.0, 0.0);
            for j in 0..=pv {
                let v_idx = span_v - pv + j;
                let pt = &self.control_points[u_idx][v_idx];
                let w = self.weights[u_idx][v_idx];
                if POINT {
                    let bw = nv[j] * w;
                    row_x += bw * pt.x();
                    row_y += bw * pt.y();
                    row_z += bw * pt.z();
                    row_w += bw;
                }
                let dv_0j = ders_v[j];
                let accumulate = |a: &mut [f64; 4], coeff: f64| {
                    a[0] += coeff * pt.x() * w;
                    a[1] += coeff * pt.y() * w;
                    a[2] += coeff * pt.z() * w;
                    a[3] += coeff * w;
                };
                accumulate(&mut aw[0], du_0i * dv_0j);
                if dv >= 1 {
                    accumulate(&mut aw[1], du_0i * ders_v[stride_v + j]);
                }
                if du >= 1 {
                    accumulate(&mut aw[2], du_1i * dv_0j);
                }
            }
            wx += ders_u[i] * row_x;
            wy += ders_u[i] * row_y;
            wz += ders_u[i] * row_z;
            ww += ders_u[i] * row_w;
        }
        let point = if ww == 0.0 {
            Point3::new(wx, wy, wz)
        } else {
            Point3::new(wx / ww, wy / ww, wz / ww)
        };

        let (top_u, top_v) = if self.row_varies(span_u, span_v) {
            (1, 1)
        } else {
            (du, dv)
        };
        let (s, su, sv) = first_order_quotient(top_u, top_v, &aw);
        (point, s, su, sv)
    }

    /// Compute the unit normal vector at parameters `(u, v)`.
    ///
    /// The normal is the cross product of the u- and v-partial derivatives,
    /// normalized. At degenerate points (poles, collapsed edges) where
    /// `du × dv ≈ 0`, falls back to perturbing the parameter slightly
    /// in each direction and retrying — an L'Hôpital-style approach.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ZeroVector`] if the surface is degenerate at
    /// this point and all fallback perturbations also fail.
    pub fn normal(&self, u: f64, v: f64) -> Result<Vec3, MathError> {
        let (_, du, dv) = self.first_derivatives(u, v);
        let cross = du.cross(dv);

        if cross.length_squared() > 1e-30 {
            return cross.normalize();
        }

        // Degenerate point — try perturbing the parameter slightly.
        let (u0, u1) = self.domain_u();
        let (v0, v1) = self.domain_v();
        let eps_u = (u1 - u0) * 1e-6;
        let eps_v = (v1 - v0) * 1e-6;

        let perturbations = [
            (u + eps_u, v),
            (u - eps_u, v),
            (u, v + eps_v),
            (u, v - eps_v),
        ];

        for (pu, pv) in perturbations {
            let pu = pu.clamp(u0, u1);
            let pv = pv.clamp(v0, v1);
            let (_, pdu, pdv) = self.first_derivatives(pu, pv);
            let pcross = pdu.cross(pdv);
            if pcross.length_squared() > 1e-30 {
                return pcross.normalize();
            }
        }

        Err(MathError::ZeroVector)
    }

    /// Compute an axis-aligned bounding box from control point extrema.
    #[must_use]
    pub fn aabb(&self) -> Aabb3 {
        *self.caches.control_box.get_or_init(|| {
            Aabb3::from_points(
                self.control_points
                    .iter()
                    .flat_map(|row| row.iter().copied()),
            )
        })
    }

    /// The surface split into Bezier patches for casting rays at it
    /// ([`RaySurface::new`](crate::nurbs::intersection::RaySurface::new)),
    /// made on first use and shared from then on: a surface is never
    /// changed once made. `None` where `RaySurface::new` declines it.
    #[must_use]
    pub fn ray_surface(&self) -> Option<std::sync::Arc<crate::nurbs::intersection::RaySurface>> {
        self.caches
            .ray_surface
            .get_or_init(|| {
                crate::nurbs::intersection::RaySurface::new(self).map(std::sync::Arc::new)
            })
            .clone()
    }

    /// `(u, v)` of `point`'s projection by `project`, kept for the same
    /// point (its coordinates bit for bit): a surface's projection of a
    /// point does not change, and several classifiers project the same
    /// boundary points onto it in turn.
    pub(crate) fn projected(
        &self,
        point: Point3,
        project: impl FnOnce() -> (f64, f64),
    ) -> (f64, f64) {
        let key = [
            point.x().to_bits(),
            point.y().to_bits(),
            point.z().to_bits(),
        ];
        if let Some(&uv) = self
            .caches
            .projections
            .0
            .lock()
            .ok()
            .as_ref()
            .and_then(|m| m.get(&key))
        {
            return uv;
        }
        let uv = project();
        if let Ok(mut m) = self.caches.projections.0.lock()
            && m.len() < ProjectionMemo::LIMIT
        {
            m.insert(key, uv);
        }
        uv
    }

    /// The surface split into Bezier patches
    /// ([`surface_to_bezier_patches`](crate::nurbs::decompose::surface_to_bezier_patches)),
    /// kept among the last few split anywhere (at most 50,000 patches in
    /// all) and found again by the surface's data bit for bit: a
    /// boolean's result rebuilds its faces' surfaces, so a body cut by one
    /// tool after another would otherwise split the same surface for each.
    /// Not kept on the surface, which can outlive its use by far.
    ///
    /// # Errors
    ///
    /// As `surface_to_bezier_patches`.
    pub fn bezier_patches(
        &self,
    ) -> Result<std::sync::Arc<Vec<crate::nurbs::decompose::BezierPatch>>, MathError> {
        let key = self.content_key();
        let found = SPLIT
            .lock()
            .ok()
            .and_then(|mut kept| split_kept(&mut kept, key, self));
        if let Some(patches) = found {
            return Ok(patches);
        }
        let patches =
            std::sync::Arc::new(crate::nurbs::decompose::surface_to_bezier_patches(self)?);
        Ok(match SPLIT.lock() {
            Ok(mut kept) => keep_split(&mut kept, (key, self.data_only(), patches), SPLIT_PATCHES),
            Err(_) => patches,
        })
    }

    /// A hash of the surface's degrees, knots, control points and weights,
    /// bit for bit.
    fn content_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.degree_u, self.degree_v).hash(&mut h);
        for &k in self.knots_u.iter().chain(&self.knots_v) {
            k.to_bits().hash(&mut h);
        }
        for p in self.control_points.iter().flatten() {
            [p.x().to_bits(), p.y().to_bits(), p.z().to_bits()].hash(&mut h);
        }
        for &w in self.weights.iter().flatten() {
            w.to_bits().hash(&mut h);
        }
        h.finish()
    }

    /// Whether `other` has this surface's data, bit for bit.
    fn same_bits(&self, other: &Self) -> bool {
        let bits = |a: &[f64], b: &[f64]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        let point_bits = |a: &Point3, b: &Point3| {
            a.x().to_bits() == b.x().to_bits()
                && a.y().to_bits() == b.y().to_bits()
                && a.z().to_bits() == b.z().to_bits()
        };
        self.degree_u == other.degree_u
            && self.degree_v == other.degree_v
            && bits(&self.knots_u, &other.knots_u)
            && bits(&self.knots_v, &other.knots_v)
            && self.control_points.len() == other.control_points.len()
            && self
                .control_points
                .iter()
                .zip(&other.control_points)
                .all(|(a, b)| a.len() == b.len() && a.iter().zip(b).all(|(p, q)| point_bits(p, q)))
            && self.weights.len() == other.weights.len()
            && self
                .weights
                .iter()
                .zip(&other.weights)
                .all(|(a, b)| bits(a, b))
    }

    /// The surface's data without anything it keeps.
    fn data_only(&self) -> Self {
        Self {
            degree_u: self.degree_u,
            degree_v: self.degree_v,
            knots_u: self.knots_u.clone(),
            knots_v: self.knots_v.clone(),
            control_points: self.control_points.clone(),
            weights: self.weights.clone(),
            caches: Box::default(),
        }
    }

    /// The surface's coarse grid for projections, evaluated on first use.
    pub(crate) fn coarse_grid(&self) -> &crate::nurbs::projection::CoarseGrid {
        self.caches
            .coarse_grid
            .get_or_init(|| std::sync::Arc::new(crate::nurbs::projection::CoarseGrid::new(self)))
    }

    /// Create a cached evaluator for repeated evaluation.
    ///
    /// The evaluator lazily precomputes polynomial coefficients for Horner
    /// evaluation, amortising the setup cost over many evaluations.
    #[must_use]
    pub fn evaluator(&self) -> SurfaceEvaluator<'_> {
        SurfaceEvaluator::new(self)
    }
}

/// A NURBS surface's span and basis functions at one `u`: see
/// [`NurbsSurface::u_basis`].
#[derive(Debug, Clone, Copy)]
pub struct UBasis {
    /// The parameter they were worked out at.
    pub u: f64,
    span: usize,
    ders: [f64; 2 * (basis::MAX_STACK_OUTPUT + 1)],
}

/// A NURBS surface along a line of constant `v`: see [`NurbsSurface::at_v`].
#[derive(Debug, Clone)]
pub struct NurbsRow<'a> {
    surface: &'a NurbsSurface,
    span_v: usize,
    basis_v: Vec<f64>,
    ders_v: Vec<f64>,
}

impl NurbsRow<'_> {
    /// The point at `u`, as [`NurbsSurface::evaluate`] gives it.
    #[must_use]
    pub fn evaluate(&self, u: f64) -> Point3 {
        self.surface.evaluate_on_row(u, self.span_v, &self.basis_v)
    }

    /// The point at `u` as [`Self::evaluate`] gives it, with its first
    /// partials as [`Self::first_derivatives`] gives them, at about the
    /// cost of the partials alone.
    #[must_use]
    pub fn point_and_partials(&self, u: f64) -> (Point3, Vec3, Vec3) {
        self.surface
            .point_and_partials_on_row(u, self.span_v, &self.basis_v, &self.ders_v)
    }

    /// [`Self::point_and_partials`] at the `u` that `basis` was worked out
    /// at, from it.
    #[must_use]
    pub fn point_and_partials_with(&self, basis: &UBasis) -> (Point3, Vec3, Vec3) {
        let (point, _, su, sv) = self.surface.first_order_from::<true>(
            basis.span,
            &basis.ders,
            self.span_v,
            &self.basis_v,
            &self.ders_v,
        );
        (point, su, sv)
    }

    /// The point and first partials at `u`, as
    /// [`NurbsSurface::first_derivatives`] gives them.
    #[must_use]
    pub fn first_derivatives(&self, u: f64) -> (Vec3, Vec3, Vec3) {
        let pu = self.surface.degree_u;
        let n_rows = self.surface.control_points.len();
        let span_u = basis::find_span(n_rows, pu, u, &self.surface.knots_u);
        let du = 1.min(pu);
        let mut ders_u_stack = [0.0f64; 2 * (basis::MAX_STACK_OUTPUT + 1)];
        let mut ders_u_heap = Vec::new();
        let ders_u = basis::scratch(&mut ders_u_stack, &mut ders_u_heap, (du + 1) * (pu + 1));
        basis::ders_basis_funs_into(span_u, u, pu, du, &self.surface.knots_u, ders_u);
        let (_, s, su, sv) =
            self.surface
                .first_order_from::<false>(span_u, ders_u, self.span_v, &[], &self.ders_v);
        (s, su, sv)
    }
}

/// [`rational_quotient`] to order 1, `aw` laid out as it lays it out:
/// `(S, S_u, S_v)`, by its operations less those that leave a value as it
/// is (a product by a binomial of 1, a difference with a sum of none).
fn first_order_quotient(top_u: usize, top_v: usize, aw: &[[f64; 4]; 4]) -> (Vec3, Vec3, Vec3) {
    let w0 = aw[0][3];
    let over_w0 = |v3: [f64; 3]| {
        if w0 == 0.0 {
            Vec3::new(v3[0], v3[1], v3[2])
        } else {
            Vec3::new(v3[0] / w0, v3[1] / w0, v3[2] / w0)
        }
    };
    let zero = Vec3::new(0.0, 0.0, 0.0);
    let s = over_w0([aw[0][0], aw[0][1], aw[0][2]]);
    let less =
        |a: [f64; 4], w: f64| over_w0([a[0] - w * s.x(), a[1] - w * s.y(), a[2] - w * s.z()]);
    let sv = if top_v >= 1 {
        less(aw[1], aw[1][3])
    } else {
        zero
    };
    let su = if top_u >= 1 {
        less(aw[2], aw[2][3])
    } else {
        zero
    };
    (s, su, sv)
}

/// The rational quotient rule (NURBS Book A4.4): `skl`'s derivatives of the
/// surface from `aw`'s homogeneous ones, to order `d`, `top_u` and `top_v`
/// in `u` and `v`.
#[allow(clippy::cast_precision_loss)]
fn rational_quotient(d: usize, top_u: usize, top_v: usize, aw: &[[f64; 4]], skl: &mut [Vec3]) {
    let n = d + 1;
    let w0 = aw[0][3];

    for k in 0..=top_u {
        for l in 0..=top_v.min(d - k) {
            let [x, y, z, _] = aw[k * n + l];
            let mut v3 = [x, y, z];

            for j in 1..=l {
                let bin = binomial(l, j) as f64;
                v3[0] -= bin * aw[j][3] * skl[k * n + l - j].x();
                v3[1] -= bin * aw[j][3] * skl[k * n + l - j].y();
                v3[2] -= bin * aw[j][3] * skl[k * n + l - j].z();
            }

            for i in 1..=k {
                let bin = binomial(k, i) as f64;
                v3[0] -= bin * aw[i * n][3] * skl[(k - i) * n + l].x();
                v3[1] -= bin * aw[i * n][3] * skl[(k - i) * n + l].y();
                v3[2] -= bin * aw[i * n][3] * skl[(k - i) * n + l].z();

                let mut v2 = [0.0f64; 3];
                for j in 1..=l {
                    let bin2 = binomial(l, j) as f64;
                    v2[0] += bin2 * aw[i * n + j][3] * skl[(k - i) * n + l - j].x();
                    v2[1] += bin2 * aw[i * n + j][3] * skl[(k - i) * n + l - j].y();
                    v2[2] += bin2 * aw[i * n + j][3] * skl[(k - i) * n + l - j].z();
                }
                v3[0] -= bin * v2[0];
                v3[1] -= bin * v2[1];
                v3[2] -= bin * v2[2];
            }

            if w0 == 0.0 {
                skl[k * n + l] = Vec3::new(v3[0], v3[1], v3[2]);
            } else {
                skl[k * n + l] = Vec3::new(v3[0] / w0, v3[1] / w0, v3[2] / w0);
            }
        }
    }
}

use super::basis::binomial;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::cast_lossless, clippy::suboptimal_flops)]
mod tests {
    use super::*;
    use crate::madd::Madd;

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

    /// A plane patch whose weights differ along `u` is
    /// `S(u, v) = (3u / (1 + 2u), v, 0)`: past its degrees its second
    /// derivative along `u` is -1.5 at `u = 0.5`, and the others are zero.
    #[test]
    fn rational_derivatives_past_the_degrees() {
        let s = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![3.0, 3.0]],
        )
        .expect("valid patch");
        let d = s.derivatives(0.5, 0.3, 2);
        let cases = [
            (1, 0, Vec3::new(0.75, 0.0, 0.0)),
            (0, 1, Vec3::new(0.0, 1.0, 0.0)),
            (2, 0, Vec3::new(-1.5, 0.0, 0.0)),
            (1, 1, Vec3::new(0.0, 0.0, 0.0)),
            (0, 2, Vec3::new(0.0, 0.0, 0.0)),
        ];
        for (k, l, want) in cases {
            assert!(
                (d[k][l] - want).length() < 1e-12,
                "S_({k},{l}) = {:?}",
                d[k][l]
            );
        }
    }

    /// `first_derivatives` is `derivatives(u, v, 1)` to the bit, on a
    /// rational patch and a polynomial one, at knots and between them; and
    /// past the stack buffers' order the table keeps its lower terms.
    #[test]
    fn first_derivatives_are_the_order_one_table() {
        let rational = NurbsSurface::new(
            2,
            1,
            vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            (0..4)
                .map(|i| {
                    let x = f64::from(i);
                    vec![Point3::new(x, 0.0, x * x * 0.1), Point3::new(x, 1.0, 0.3)]
                })
                .collect(),
            vec![
                vec![1.0, 2.0],
                vec![0.5, 1.5],
                vec![3.0, 1.0],
                vec![1.0, 0.7],
            ],
        )
        .expect("valid patch");
        let bits = |p: Vec3| [p.x().to_bits(), p.y().to_bits(), p.z().to_bits()];
        for s in [&rational, &bicubic_surface()] {
            for (u, v) in [
                (0.0, 0.0),
                (0.5, 0.25),
                (0.3, 1.0),
                (1.0, 0.7),
                (0.77, 0.13),
            ] {
                let table = s.derivatives(u, v, 1);
                let (p, su, sv) = s.first_derivatives(u, v);
                assert_eq!(bits(p), bits(table[0][0]), "S at ({u}, {v})");
                assert_eq!(bits(su), bits(table[1][0]), "S_u at ({u}, {v})");
                assert_eq!(bits(sv), bits(table[0][1]), "S_v at ({u}, {v})");
                let (pu, pv) = crate::traits::ParametricSurface::partials(s, u, v);
                assert_eq!(bits(pu), bits(table[1][0]), "partials' S_u at ({u}, {v})");
                assert_eq!(bits(pv), bits(table[0][1]), "partials' S_v at ({u}, {v})");
                let deep = s.derivatives(u, v, 3);
                for (k, l) in [(0, 0), (1, 0), (0, 1)] {
                    assert_eq!(bits(deep[k][l]), bits(table[k][l]), "order 3 ({k}, {l})");
                }
            }
        }
    }

    /// Along a line of constant `v`, the row's points and partials are the
    /// surface's own, bit for bit, and the surface's first derivatives are
    /// its order-1 table's: rational and polynomial patches, degrees 1 to 5,
    /// at knots, ends and between.
    #[test]
    fn a_row_evaluates_as_the_surface_does() {
        let rational = NurbsSurface::new(
            2,
            1,
            vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            (0..4)
                .map(|i| {
                    let x = f64::from(i);
                    vec![Point3::new(x, 0.0, x * x * 0.1), Point3::new(x, 1.0, 0.3)]
                })
                .collect(),
            vec![
                vec![1.0, 2.0],
                vec![0.5, 1.5],
                vec![3.0, 1.0],
                vec![1.0, 0.7],
            ],
        )
        .expect("valid patch");
        // A net of `n` by `m` points, clamped knots with interior ones at
        // uneven steps, weights varying or not.
        let net = |pu: usize, pv: usize, n: usize, m: usize, rational: bool| {
            let knots = |p: usize, n: usize| {
                let mut k = vec![0.0; p + 1];
                for i in 1..n - p {
                    let t = i as f64 / (n - p) as f64;
                    k.push(t * t.madd(0.3, 0.7));
                }
                k.extend(vec![1.0; p + 1]);
                k
            };
            let points = (0..n)
                .map(|i| {
                    (0..m)
                        .map(|j| {
                            let (x, y) = (i as f64, j as f64);
                            Point3::new(x, y, (x * 0.7 + y * 1.3).sin())
                        })
                        .collect()
                })
                .collect();
            let weights = (0..n)
                .map(|i| {
                    (0..m)
                        .map(|j| {
                            if rational {
                                1.0 + 0.5 * ((i * 3 + j * 5) % 7) as f64 / 7.0
                            } else {
                                1.0
                            }
                        })
                        .collect()
                })
                .collect();
            NurbsSurface::new(pu, pv, knots(pu, n), knots(pv, m), points, weights)
                .expect("valid net")
        };
        let surfaces = [
            rational,
            bicubic_surface(),
            net(1, 3, 5, 6, false),
            net(3, 1, 6, 4, true),
            net(5, 2, 9, 5, true),
            net(4, 4, 8, 8, false),
        ];
        let bits = |p: Vec3| [p.x().to_bits(), p.y().to_bits(), p.z().to_bits()];
        let point_bits = |p: Point3| [p.x().to_bits(), p.y().to_bits(), p.z().to_bits()];
        for s in &surfaces {
            let (u0, u1) = s.domain_u();
            let (v0, v1) = s.domain_v();
            let mut us: Vec<f64> = (0..=40)
                .map(|i| (u1 - u0).madd(f64::from(i) / 40.0, u0))
                .collect();
            us.extend(s.knots_u().iter().copied());
            let mut vs: Vec<f64> = (0..=12)
                .map(|i| (v1 - v0).madd(f64::from(i) / 12.0, v0))
                .collect();
            vs.extend(s.knots_v().iter().copied());
            for &v in &vs {
                let row = s.at_v(v);
                for &u in &us {
                    let p = s.evaluate(u, v);
                    let (q, su, sv) = s.first_derivatives(u, v);
                    let table = s.derivatives(u, v, 1);
                    assert_eq!(
                        [bits(q), bits(su), bits(sv)],
                        [bits(table[0][0]), bits(table[1][0]), bits(table[0][1])],
                        "first derivatives at ({u}, {v})"
                    );
                    let (rp, rsu, rsv) = row.point_and_partials(u);
                    let at = format!("({u}, {v}), degrees {:?}", (s.degree_u(), s.degree_v()));
                    let (fp, fsu, fsv) = s.point_and_first_derivatives(u, v);
                    assert_eq!(point_bits(fp), point_bits(p), "fused point at {at}");
                    assert_eq!(
                        [bits(fsu), bits(fsv)],
                        [bits(su), bits(sv)],
                        "fused partials at {at}"
                    );
                    assert_eq!(point_bits(rp), point_bits(p), "point at {at}");
                    assert_eq!(bits(rsu), bits(su), "S_u at {at}");
                    assert_eq!(bits(rsv), bits(sv), "S_v at {at}");
                    assert_eq!(
                        point_bits(row.evaluate(u)),
                        point_bits(p),
                        "row point at {at}"
                    );
                    let (rq, ru, rv) = row.first_derivatives(u);
                    assert_eq!(
                        [bits(rq), bits(ru), bits(rv)],
                        [bits(q), bits(su), bits(sv)],
                        "row partials at {at}"
                    );
                }
            }
        }
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

    #[test]
    fn nurbs_partial_matches_finite_difference() {
        use crate::traits::ParametricSurface;

        let s = bicubic_surface();
        let u = 0.5;
        let v = 0.5;
        let h = 1e-6;

        // Central finite difference for du
        let p_plus = s.evaluate(u + h, v);
        let p_minus = s.evaluate(u - h, v);
        let fd_u = Vec3::new(
            (p_plus.x() - p_minus.x()) / (2.0 * h),
            (p_plus.y() - p_minus.y()) / (2.0 * h),
            (p_plus.z() - p_minus.z()) / (2.0 * h),
        );
        let du = ParametricSurface::partial_u(&s, u, v);
        assert!(
            (du.x() - fd_u.x()).abs() < 1e-4,
            "du.x: {} vs {}",
            du.x(),
            fd_u.x()
        );
        assert!(
            (du.y() - fd_u.y()).abs() < 1e-4,
            "du.y: {} vs {}",
            du.y(),
            fd_u.y()
        );
        assert!(
            (du.z() - fd_u.z()).abs() < 1e-4,
            "du.z: {} vs {}",
            du.z(),
            fd_u.z()
        );

        // Central finite difference for dv
        let p_plus = s.evaluate(u, v + h);
        let p_minus = s.evaluate(u, v - h);
        let fd_v = Vec3::new(
            (p_plus.x() - p_minus.x()) / (2.0 * h),
            (p_plus.y() - p_minus.y()) / (2.0 * h),
            (p_plus.z() - p_minus.z()) / (2.0 * h),
        );
        let dv = ParametricSurface::partial_v(&s, u, v);
        assert!(
            (dv.x() - fd_v.x()).abs() < 1e-4,
            "dv.x: {} vs {}",
            dv.x(),
            fd_v.x()
        );
        assert!(
            (dv.y() - fd_v.y()).abs() < 1e-4,
            "dv.y: {} vs {}",
            dv.y(),
            fd_v.y()
        );
        assert!(
            (dv.z() - fd_v.z()).abs() < 1e-4,
            "dv.z: {} vs {}",
            dv.z(),
            fd_v.z()
        );
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

    /// A bilinear sheet of `spans` knot spans along `u`, lifted by `lift`:
    /// it splits into `spans` Bezier patches.
    fn sheet(spans: usize, lift: f64) -> NurbsSurface {
        let mut knots_u = vec![0.0, 0.0];
        knots_u.extend((1..spans).map(|i| i as f64));
        knots_u.extend([spans as f64, spans as f64]);
        let rows = (0..=spans)
            .map(|i| {
                vec![
                    Point3::new(i as f64, 0.0, lift),
                    Point3::new(i as f64, 1.0, lift),
                ]
            })
            .collect();
        NurbsSurface::new(
            1,
            1,
            knots_u,
            vec![0.0, 0.0, 1.0, 1.0],
            rows,
            vec![vec![1.0, 1.0]; spans + 1],
        )
        .expect("valid sheet")
    }

    fn entry(surface: &NurbsSurface) -> SplitEntry {
        let patches =
            crate::nurbs::decompose::surface_to_bezier_patches(surface).expect("clamped knots");
        (
            surface.content_key(),
            surface.data_only(),
            std::sync::Arc::new(patches),
        )
    }

    #[test]
    fn kept_patches_are_the_fresh_split_bit_for_bit() {
        let surface = sheet(3, 0.25);
        let fresh =
            crate::nurbs::decompose::surface_to_bezier_patches(&surface).expect("clamped knots");
        // Twice: the second call reads what the first kept.
        for _ in 0..2 {
            let kept = surface.bezier_patches().expect("clamped knots");
            assert_eq!(kept.len(), fresh.len());
            for (a, b) in kept.iter().zip(&fresh) {
                assert!(a.surface.same_bits(&b.surface));
                assert_eq!(a.u_range.0.to_bits(), b.u_range.0.to_bits());
                assert_eq!(a.u_range.1.to_bits(), b.u_range.1.to_bits());
                assert_eq!(a.v_range.0.to_bits(), b.v_range.0.to_bits());
                assert_eq!(a.v_range.1.to_bits(), b.v_range.1.to_bits());
            }
        }
    }

    #[test]
    fn a_surface_made_again_from_the_same_data_finds_its_kept_patches() {
        let mut kept = Vec::new();
        let first = keep_split(&mut kept, entry(&sheet(3, 0.5)), 100);
        // The same data, made anew: the entry already kept is the one used.
        let again = keep_split(&mut kept, entry(&sheet(3, 0.5)), 100);
        assert!(std::sync::Arc::ptr_eq(&first, &again));
        assert_eq!(kept.len(), 1);
        let found = split_kept(&mut kept, sheet(3, 0.5).content_key(), &sheet(3, 0.5));
        assert!(found.is_some_and(|p| std::sync::Arc::ptr_eq(&p, &first)));
    }

    #[test]
    fn surfaces_of_different_data_keep_their_own_patches() {
        let mut kept = Vec::new();
        let low = keep_split(&mut kept, entry(&sheet(3, 0.5)), 100);
        let high = keep_split(&mut kept, entry(&sheet(3, 0.75)), 100);
        assert!(!std::sync::Arc::ptr_eq(&low, &high));
        assert_eq!(kept.len(), 2);
        let other = sheet(3, 1.0);
        assert!(split_kept(&mut kept, other.content_key(), &other).is_none());
    }

    #[test]
    fn the_least_lately_used_patches_go_first_to_make_room() {
        let (a, b, c) = (sheet(4, 0.0), sheet(4, 1.0), sheet(4, 2.0));
        let mut kept = Vec::new();
        keep_split(&mut kept, entry(&a), 10);
        keep_split(&mut kept, entry(&b), 10);
        // Using `a` again leaves `b` the least lately used.
        assert!(split_kept(&mut kept, a.content_key(), &a).is_some());
        keep_split(&mut kept, entry(&c), 10);
        assert!(kept.iter().map(|e| e.2.len()).sum::<usize>() <= 10);
        assert!(split_kept(&mut kept, b.content_key(), &b).is_none());
        assert!(split_kept(&mut kept, a.content_key(), &a).is_some());
        assert!(split_kept(&mut kept, c.content_key(), &c).is_some());

        // Both fit beside what is kept: neither puts the other out.
        let mut kept = Vec::new();
        keep_split(&mut kept, entry(&a), 10);
        keep_split(&mut kept, entry(&sheet(2, 5.0)), 10);
        keep_split(&mut kept, entry(&sheet(2, 5.0)), 10);
        assert_eq!(kept.len(), 2);
        assert!(split_kept(&mut kept, a.content_key(), &a).is_some());

        // A surface over the limit by itself is used and not kept.
        let mut kept = Vec::new();
        keep_split(&mut kept, entry(&a), 10);
        let big = keep_split(&mut kept, entry(&sheet(11, 0.0)), 10);
        assert_eq!(big.len(), 11);
        assert_eq!(kept.len(), 1);
    }
}
