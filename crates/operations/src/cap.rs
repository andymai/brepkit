//! Shared loft/sweep/pipe/revolve end-cap construction.
//!
//! A swept/lofted/revolved end is closed by filling its section *boundary* — a
//! ring of chord edges — rather than by reusing the section's own surface.
//! brepkit tessellates and integrates a non-planar face over its full u/v
//! extent rather than clipping to a chord-polygon wire, so a reused parent
//! surface would overfill past the section. Instead: a planar ring gets an exact
//! `Plane` cap (the planar tessellator clips it to the polygon), and a
//! non-planar 4-sided ring is filled by a bilinear patch whose boundary
//! iso-curves are exactly the ring chords (`domain == wire`, so it can't
//! overfill).

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::dot_normal_point;

/// Collect the 3D positions of a ring of vertices (a cap's outer boundary).
///
/// # Errors
///
/// Returns an error if a vertex id is missing from the arena.
pub fn ring_point_positions(
    topo: &Topology,
    ring: &[VertexId],
) -> Result<Vec<Point3>, crate::OperationsError> {
    ring.iter()
        .map(|&vid| -> Result<Point3, crate::OperationsError> { Ok(topo.vertex(vid)?.point()) })
        .collect()
}

/// A cap ring whose vertices deviate from their best-fit plane by less than this
/// fraction of the ring's size is treated as planar (capped by an exact
/// `Plane`); a larger deviation is filled by a bilinear patch.
const CAP_PLANARITY_TOL: f64 = 1e-6;

/// Characteristic size of a cap ring: the largest distance from its first
/// vertex to any other, used to scale the planarity test.
fn ring_scale(verts: &[Point3]) -> f64 {
    let c = verts[0];
    verts.iter().map(|p| (*p - c).length()).fold(0.0, f64::max)
}

/// Whether the ring lies (within tolerance) in the plane through `cap_verts[0]`
/// with normal `outward`.
fn ring_is_planar(cap_verts: &[Point3], outward: Vec3) -> bool {
    let plane_pt = cap_verts[0];
    let max_dev = cap_verts
        .iter()
        .map(|p| (*p - plane_pt).dot(outward).abs())
        .fold(0.0, f64::max);
    max_dev <= CAP_PLANARITY_TOL * ring_scale(cap_verts)
}

/// Bilinear (degree-1) NURBS patch through a 4-corner ring, in ring order.
///
/// Its four boundary iso-curves are the straight segments between consecutive
/// corners — exactly the ring's chord edges — so the cap shares its boundary
/// with the side faces and tessellates/integrates clipped to the section.
fn bilinear_cap_patch(corners: &[Point3]) -> Result<NurbsSurface, brepkit_math::MathError> {
    NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![vec![corners[0], corners[1]], vec![corners[3], corners[2]]],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
}

/// The ring's outward cap normal: its Newell normal, flipped to agree with
/// `toward` (the side the cap should face, e.g. away from the swept body).
///
/// # Errors
///
/// Returns an error if the ring is degenerate (zero-area Newell normal).
pub fn outward_normal(verts: &[Point3], toward: Vec3) -> Result<Vec3, crate::OperationsError> {
    let n = crate::winding::newell_normal(verts).normalize()?;
    Ok(if n.dot(toward) < 0.0 { -n } else { n })
}

/// Build one end-cap face that fills the ring boundary (and any holes).
///
/// `outer_ring_edges` is the section's outer boundary in ring order;
/// `inner_wires` are pre-built hole loops (empty for hole-free sections, and
/// only supported on a planar cap). `outward` is the section's outward normal;
/// `start_role` builds the reversed-ring wire so the cap faces away from the
/// body.
///
/// A planar ring → exact `Plane` cap. A non-planar 4-sided hole-free ring →
/// bilinear patch. A non-planar ring with more than four edges, or with holes,
/// is unsupported.
///
/// # Errors
///
/// Returns an error if the wire is invalid, the bilinear patch cannot be built,
/// or the ring is an unsupported non-planar shape.
pub fn build_cap_face(
    topo: &mut Topology,
    outer_ring_edges: &[EdgeId],
    inner_wires: Vec<WireId>,
    cap_verts: &[Point3],
    outward: Vec3,
    start_role: bool,
) -> Result<FaceId, crate::OperationsError> {
    let n = outer_ring_edges.len();
    // `cap_verts` must be one position per ring edge (and a ring needs ≥ 3) —
    // later indexing (`cap_verts[0]`, the 4 bilinear corners) relies on this.
    if n < 3 || cap_verts.len() != n {
        return Err(crate::OperationsError::InvalidInput {
            reason: "cap ring must have at least 3 vertices matching its edge count".into(),
        });
    }
    let edges: Vec<OrientedEdge> = if start_role {
        (0..n)
            .rev()
            .map(|i| OrientedEdge::new(outer_ring_edges[i], false))
            .collect()
    } else {
        (0..n)
            .map(|i| OrientedEdge::new(outer_ring_edges[i], true))
            .collect()
    };
    let wid = topo.add_wire(Wire::new(edges, true).map_err(crate::OperationsError::Topology)?);

    if ring_is_planar(cap_verts, outward) {
        let surface = FaceSurface::Plane {
            normal: outward,
            d: dot_normal_point(outward, cap_verts[0]),
        };
        return Ok(topo.add_face(Face::new(wid, inner_wires, surface)));
    }

    if !inner_wires.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "cap with holes on a non-planar section boundary is not supported".into(),
        });
    }
    if n != 4 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "cap for a non-planar section boundary with more than 4 edges is not supported"
                .into(),
        });
    }

    let surf = bilinear_cap_patch(cap_verts).map_err(crate::OperationsError::Math)?;
    // A near-flat bilinear lid: its center normal is stable and aligned with the
    // ring axis, so probe there and flip if it opposes `outward`.
    let reversed = surf
        .normal(0.5, 0.5)
        .map(|nrm| nrm.dot(outward) < 0.0)
        .unwrap_or(false);
    let mut face = Face::new(wid, vec![], FaceSurface::Nurbs(surf));
    if reversed {
        face.set_reversed(true);
    }
    Ok(topo.add_face(face))
}
