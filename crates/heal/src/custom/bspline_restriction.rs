//! B-spline restriction — limit degree and segment count.
//!
//! Checks all NURBS curves and surfaces in a solid against configurable
//! limits for polynomial degree and number of segments (spans). Entities
//! that exceed the limits are counted and logged. Actual degree reduction
//! or re-approximation is not performed (that is an advanced NURBS
//! operation requiring iterative fitting).

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

use crate::HealError;

/// Configuration for B-spline restriction checking.
#[derive(Debug, Clone)]
pub struct RestrictionOptions {
    /// Maximum allowed polynomial degree for curves and surfaces.
    pub max_degree: usize,
    /// Maximum allowed number of spans (unique knot intervals) for
    /// curves and surfaces.
    pub max_segments: usize,
    /// Tolerance for geometric approximation (reserved for future
    /// degree reduction).
    pub tolerance: f64,
}

impl Default for RestrictionOptions {
    fn default() -> Self {
        Self {
            max_degree: 25,
            max_segments: 100,
            tolerance: 1e-7,
        }
    }
}

/// Tolerance for comparing knot values when counting unique spans.
const KNOT_EPS: f64 = 1e-15;

/// Check all NURBS curves and surfaces in a solid against restriction limits.
///
/// Walks all edges and faces of the solid. For each NURBS curve or
/// surface, checks the polynomial degree and the number of spans (unique
/// internal knot intervals) against the limits in `options`.
///
/// Returns the number of entities that exceed the limits. Each violation
/// is logged via [`log::warn!`].
///
/// # Errors
///
/// Returns [`HealError`] if topology lookups fail.
pub fn check_bspline_restrictions(
    topo: &Topology,
    solid_id: SolidId,
    options: &RestrictionOptions,
) -> Result<usize, HealError> {
    let mut violations = 0usize;

    // ── Check edges ──────────────────────────────────────────────
    let solid_data = topo.solid(solid_id)?;
    let shell_id = solid_data.outer_shell();
    let shell = topo.shell(shell_id)?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    let mut seen_edges = std::collections::HashSet::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                if !seen_edges.insert(eid.index()) {
                    continue;
                }

                let edge = topo.edge(eid)?;
                if let EdgeCurve::NurbsCurve(ref nc) = *edge.curve() {
                    let degree = nc.degree();
                    let segments = count_curve_segments(nc);

                    if degree > options.max_degree {
                        log::warn!(
                            "edge (index {}) has NURBS degree {} (limit: {})",
                            eid.index(),
                            degree,
                            options.max_degree
                        );
                        violations += 1;
                    }
                    if segments > options.max_segments {
                        log::warn!(
                            "edge (index {}) has {} NURBS segments (limit: {})",
                            eid.index(),
                            segments,
                            options.max_segments
                        );
                        violations += 1;
                    }
                }
            }
        }

        // ── Check face surface ───────────────────────────────────
        let face_data = topo.face(fid)?;
        if let FaceSurface::Nurbs(ref ns) = *face_data.surface() {
            let degree_u = ns.degree_u();
            let degree_v = ns.degree_v();
            let segments_u = count_surface_segments_u(ns);
            let segments_v = count_surface_segments_v(ns);

            if degree_u > options.max_degree {
                log::warn!(
                    "face (index {}) has NURBS degree_u {} (limit: {})",
                    fid.index(),
                    degree_u,
                    options.max_degree
                );
                violations += 1;
            }
            if degree_v > options.max_degree {
                log::warn!(
                    "face (index {}) has NURBS degree_v {} (limit: {})",
                    fid.index(),
                    degree_v,
                    options.max_degree
                );
                violations += 1;
            }
            if segments_u > options.max_segments {
                log::warn!(
                    "face (index {}) has {} NURBS segments in u (limit: {})",
                    fid.index(),
                    segments_u,
                    options.max_segments
                );
                violations += 1;
            }
            if segments_v > options.max_segments {
                log::warn!(
                    "face (index {}) has {} NURBS segments in v (limit: {})",
                    fid.index(),
                    segments_v,
                    options.max_segments
                );
                violations += 1;
            }
        }
    }

    Ok(violations)
}

/// Count the number of spans (segments) in a NURBS curve.
///
/// A span is a unique knot interval `[u_i, u_{i+1}]` where `u_i < u_{i+1}`.
fn count_curve_segments(curve: &brepkit_math::nurbs::curve::NurbsCurve) -> usize {
    count_unique_knot_intervals(curve.knots(), curve.degree())
}

/// Count the number of u-spans in a NURBS surface.
fn count_surface_segments_u(surface: &brepkit_math::nurbs::surface::NurbsSurface) -> usize {
    count_unique_knot_intervals(surface.knots_u(), surface.degree_u())
}

/// Count the number of v-spans in a NURBS surface.
fn count_surface_segments_v(surface: &brepkit_math::nurbs::surface::NurbsSurface) -> usize {
    count_unique_knot_intervals(surface.knots_v(), surface.degree_v())
}

/// Count unique knot intervals in a knot vector (excluding clamped ends).
fn count_unique_knot_intervals(knots: &[f64], degree: usize) -> usize {
    if knots.len() < 2 * (degree + 1) {
        return 1;
    }

    let start = degree;
    let end = knots.len() - degree - 1;
    if end <= start {
        return 1;
    }

    let mut count = 0;
    let mut prev = knots[start];
    for &k in &knots[(start + 1)..=end] {
        if (k - prev).abs() > KNOT_EPS {
            count += 1;
            prev = k;
        }
    }

    count.max(1)
}
