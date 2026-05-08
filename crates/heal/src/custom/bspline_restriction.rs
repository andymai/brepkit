//! B-spline restriction — limit degree and segment count.
//!
//! Checks all NURBS curves and surfaces in a solid against configurable
//! limits for polynomial degree and number of segments (spans). Entities
//! that exceed the limits are counted and logged. Actual degree reduction
//! or re-approximation is not performed (that is an advanced NURBS
//! operation requiring iterative fitting).

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::solid_faces;
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
    // Walk outer + inner (cavity) shells. NURBS degree/segment
    // restrictions apply to every edge and surface in the solid,
    // including those bounding cavity volumes — restricting only
    // the outer shell would silently leave inner-shell NURBS
    // violations uncounted.
    let face_ids = solid_faces(topo, solid_id)?;

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_math::vec::Point3;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use crate::construct::convert_surface::sphere_to_nurbs;

    fn add_nurbs_sphere_face(
        topo: &mut Topology,
        center: Point3,
    ) -> brepkit_topology::face::FaceId {
        let sphere = SphericalSurface::new(center, 1.0).unwrap();
        let nurbs = sphere_to_nurbs(&sphere).unwrap();
        let v = topo.add_vertex(Vertex::new(center, 1e-7));
        let edge_id = topo.add_edge(Edge::new(v, v, EdgeCurve::Line));
        let wire = Wire::new(vec![OrientedEdge::new(edge_id, true)], true).unwrap();
        let wid = topo.add_wire(wire);
        topo.add_face(Face::new(wid, vec![], FaceSurface::Nurbs(nurbs)))
    }

    #[test]
    fn check_bspline_restrictions_walks_inner_shells() {
        // The sphere-to-NURBS converter produces a degree-2 surface in
        // both u and v, so any solid with a NURBS sphere face violates a
        // limit of `max_degree = 1`. Place such faces on BOTH the outer
        // and an inner shell, then verify we count violations from both.
        let mut topo = Topology::new();

        let outer_face = add_nurbs_sphere_face(&mut topo, Point3::new(0.0, 0.0, 0.0));
        let inner_face = add_nurbs_sphere_face(&mut topo, Point3::new(5.0, 0.0, 0.0));

        let outer_shell = topo.add_shell(Shell::new(vec![outer_face]).unwrap());
        let inner_shell = topo.add_shell(Shell::new(vec![inner_face]).unwrap());
        let solid_id = topo.add_solid(Solid::new(outer_shell, vec![inner_shell]));

        let options = RestrictionOptions {
            max_degree: 1,
            max_segments: 1000,
            tolerance: 1e-7,
        };
        let violations = check_bspline_restrictions(&topo, solid_id, &options).unwrap();

        // Each face has degree-2 in both u and v → 2 violations per
        // face × 2 faces = 4 total. Without inner-shell walking we'd
        // get only 2.
        assert_eq!(
            violations, 4,
            "expected 4 violations (2 per face × 2 faces), got {violations} — inner-shell walking missing?"
        );
    }
}
