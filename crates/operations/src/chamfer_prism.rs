//! Chamfers of separate straight edges cut as triangular prisms.
//!
//! A straight convex edge between two planes, ending at each end on a plane
//! square to it, is chamfered exactly by cutting away the prism its bevel
//! sweeps: the triangle between the edge and its two contacts, run along the
//! edge from end to end. The cut ends on the end planes as the chamfer does,
//! however the faces beyond them run (a rim notched right where the chamfer
//! ends, its notch no deeper than the chamfer is wide, leaves the bevel's end
//! reaching past the notch's wall, which the trimmer cannot close).

use std::collections::HashSet;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

/// How far square the end planes must stand to the edge, as `1 - |n . t|`.
const SQUARE_TOL: f64 = 1e-9;

/// One edge's prism: the triangle at its start, and the run along it.
struct Prism {
    triangle: [Point3; 3],
    along: Vec3,
    length: f64,
    /// The area of the bevel's section, the triangle between the edge and
    /// its contacts.
    section: f64,
}

/// The outward normal of a plane face.
fn plane_normal(topo: &Topology, face: FaceId) -> Result<Option<Vec3>, crate::OperationsError> {
    let face = topo.face(face)?;
    let FaceSurface::Plane { normal, .. } = face.surface() else {
        return Ok(None);
    };
    Ok(Some(if face.is_reversed() {
        -*normal
    } else {
        *normal
    }))
}

/// The direction into `face` square to `edge`, from the way the face's wire
/// runs along it: the face lies to the left of its wires seen from the side
/// its surface's normal points to (whichever way the face is turned).
fn into_face(
    topo: &Topology,
    face_id: FaceId,
    edge: EdgeId,
) -> Result<Option<Vec3>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let FaceSurface::Plane { normal, .. } = face.surface() else {
        return Ok(None);
    };
    let normal = *normal;
    let edge_data = topo.edge(edge)?;
    let (a, b) = (
        topo.vertex(edge_data.start())?.point(),
        topo.vertex(edge_data.end())?.point(),
    );
    let mut ways = Vec::new();
    for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wire)?.edges() {
            if oe.edge() == edge {
                ways.push(if oe.is_forward() { b - a } else { a - b });
            }
        }
    }
    let [way] = ways[..] else {
        return Ok(None);
    };
    Ok(normal.cross(way).normalize().ok())
}

/// The prism of a picked edge, `d1` into `reference` and `d2` into its
/// other face, or `None` when the edge is not a straight convex edge between
/// two planes ending on planes square to it.
#[allow(clippy::too_many_arguments)]
fn prism(
    topo: &Topology,
    faces: &[FaceId],
    adjacency: &brepkit_topology::adjacency::AdjacencyIndex,
    edge: EdgeId,
    reference: Option<FaceId>,
    d1: f64,
    d2: f64,
) -> Result<Option<Prism>, crate::OperationsError> {
    let edge_data = topo.edge(edge)?;
    if !matches!(edge_data.curve(), EdgeCurve::Line) {
        return Ok(None);
    }
    let (start, end) = (edge_data.start(), edge_data.end());
    let (a, b) = (topo.vertex(start)?.point(), topo.vertex(end)?.point());
    let length = (b - a).length();
    let Ok(along) = (b - a).normalize() else {
        return Ok(None);
    };
    let &[f1, f2] = adjacency.faces_for_edge(edge) else {
        return Ok(None);
    };
    let (f1, f2) = if reference == Some(f2) {
        (f2, f1)
    } else {
        (f1, f2)
    };
    let (Some(n1), Some(n2)) = (plane_normal(topo, f1)?, plane_normal(topo, f2)?) else {
        return Ok(None);
    };
    let (Some(w1), Some(w2)) = (into_face(topo, f1, edge)?, into_face(topo, f2, edge)?) else {
        return Ok(None);
    };
    // Convex: each face runs off the edge away from the other's outside.
    if w1.dot(n2) >= -1e-9 || w2.dot(n1) >= -1e-9 {
        return Ok(None);
    }
    // Each end on a plane square to the edge that bounds the material
    // beyond it.
    for (vertex, out) in [(start, -along), (end, along)] {
        let mut square = false;
        for &face in faces {
            if face == f1 || face == f2 {
                continue;
            }
            let Some(n) = plane_normal(topo, face)? else {
                continue;
            };
            if 1.0 - n.dot(out) > SQUARE_TOL {
                continue;
            }
            if brepkit_topology::explorer::face_vertices(topo, face)?.contains(&vertex) {
                square = true;
                break;
            }
        }
        if !square {
            return Ok(None);
        }
    }
    let (c1, c2) = (w1 * d1, w2 * d2);
    // The apex a little outside the corner, so the cut leaves no film along
    // the edge; its sides cross the faces along the contacts.
    let Ok(out) = (n1 + n2).normalize() else {
        return Ok(None);
    };
    let apex = a + out * (1e-3 * d1.min(d2));
    Ok(Some(Prism {
        triangle: [apex, a + c1, a + c2],
        along,
        length,
        section: 0.5 * c1.cross(c2).length(),
    }))
}

/// Chamfers `edges`, each `d1` into its reference face (or into either face
/// with `None`) and `d2` into the other, by cutting away each edge's bevel
/// prism. `None` unless every edge is a straight convex edge between two
/// planes, ending at both ends on planes square to it, and no two share a
/// vertex; or when a cut fails, leaves an invalid solid, or takes off other
/// than the prisms' own volume.
///
/// # Errors
///
/// Returns an error on a failed topology lookup.
pub fn chamfer_by_prisms(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[(EdgeId, Option<FaceId>)],
    d1: f64,
    d2: f64,
) -> Result<Option<SolidId>, crate::OperationsError> {
    if edges.is_empty() || d1 <= 0.0 || d2 <= 0.0 {
        return Ok(None);
    }
    let faces = brepkit_topology::explorer::solid_faces(topo, solid)?;
    let adjacency = brepkit_topology::adjacency::AdjacencyIndex::build_from_faces(topo, &faces)?;
    let mut ends = HashSet::new();
    let mut prisms = Vec::with_capacity(edges.len());
    let mut seen = HashSet::new();
    for &(edge, reference) in edges {
        if !seen.insert(edge) {
            continue;
        }
        let data = topo.edge(edge)?;
        if !ends.insert(data.start()) || !ends.insert(data.end()) {
            return Ok(None);
        }
        let Some(p) = prism(topo, &faces, &adjacency, edge, reference, d1, d2)? else {
            return Ok(None);
        };
        prisms.push(p);
    }
    let before = crate::measure::solid_volume(topo, solid, 0.01)?;
    let removed: f64 = prisms.iter().map(|p| p.section * p.length).sum();
    let opts = crate::boolean::BooleanOptions {
        mesh_fallback: false,
        ..crate::boolean::BooleanOptions::default()
    };
    let mut result = solid;
    for p in &prisms {
        let base = brepkit_topology::builder::make_planar_face(topo, &p.triangle, 1e-7)?;
        let tool = crate::extrude::extrude(topo, base, p.along, p.length)?;
        match crate::boolean::boolean_with_options(
            topo,
            crate::boolean::BooleanOp::Cut,
            result,
            tool,
            opts,
        ) {
            Ok(cut) => result = cut,
            Err(e) => {
                log::debug!("chamfer by prisms: a cut failed: {e}");
                return Ok(None);
            }
        }
    }
    if !crate::validate::validate_solid(topo, result)?.is_valid() {
        log::debug!("chamfer by prisms: the result is not valid");
        return Ok(None);
    }
    let after = crate::measure::solid_volume(topo, result, 0.01)?;
    let tol = 1e-4 * removed + 1e-7 * before;
    if ((before - after) - removed).abs() > tol {
        log::debug!(
            "chamfer by prisms: took off {}, the prisms hold {removed}",
            before - after
        );
        return Ok(None);
    }
    Ok(Some(result))
}
