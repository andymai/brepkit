//! Sections along loose tool edges.
//!
//! A tool edge bounds no face, so no face-face phase ever meets it. Where it
//! runs through a face of the solid, the edge itself is that face's section:
//! its leaf pave blocks, cut where it crosses the solid's edges, stand in for
//! the curve an intersection would have found, and the solid's edges are
//! already paved there for the faces on their far side.

use brepkit_math::aabb::Aabb3;
use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::planar::PlanarRegion;
use brepkit_topology::solid::SolidId;

use crate::builder::plane_frame::PlaneFrame;
use crate::classifier::FaceLoops2d;
use crate::ds::{GfaArena, Interference, IntersectionCurveDS};
use crate::error::AlgoError;

/// Pieces a leaf is held to a surface at.
const SAMPLES: u32 = 8;

/// How far off a surface an edge may run, in linear tolerances, and still lie
/// on it: the band fitted geometry keeps from a face it bounds.
const ON_SURFACE_FACTOR: f64 = 100.0;

/// A plane face's region as the curves it is, where it can be read so. A sampled
/// polygon misses the sliver between each chord and its arc, and an edge close by
/// a rim can run all its length in there.
fn plane_region(
    topo: &Topology,
    face: FaceId,
    tol: Tolerance,
) -> Result<Option<(PlaneFrame, PlanarRegion)>, AlgoError> {
    let data = topo.face(face)?;
    let FaceSurface::Plane { normal, .. } = data.surface() else {
        return Ok(None);
    };
    let normal = *normal;
    let corners = topo
        .wire(data.outer_wire())?
        .edges()
        .iter()
        .map(|oriented| Ok(topo.vertex(topo.edge(oriented.edge())?.start())?.point()))
        .collect::<Result<Vec<_>, AlgoError>>()?;
    let frame = PlaneFrame::from_plane_face(normal, &corners);
    let region = brepkit_topology::planar::face_region_2d(
        topo,
        face,
        frame.origin(),
        frame.u_axis(),
        frame.v_axis(),
        tol.linear,
    )?;
    Ok(region.map(|region| (frame, region)))
}

/// Register every leaf of `tool_edges` lying in a face of `body` as a section
/// of that face alone. Runs after the blocks are cut and before they become
/// edges.
pub(super) fn perform(
    topo: &Topology,
    body: SolidId,
    tool_edges: &[EdgeId],
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    if tool_edges.is_empty() {
        return Ok(());
    }
    let faces = brepkit_topology::explorer::solid_faces(topo, body)?;
    let loops = faces
        .iter()
        .map(|&face| FaceLoops2d::new(topo, face))
        .collect::<Result<Vec<_>, _>>()?;
    let band = ON_SURFACE_FACTOR * tol.linear;
    let regions = faces
        .iter()
        .map(|&face| plane_region(topo, face, tol))
        .collect::<Result<Vec<_>, _>>()?;

    for &edge_id in tool_edges {
        let edge = topo.edge(edge_id)?;
        let curve = edge.curve().clone();
        let (sp, ep) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let roots = arena
            .edge_pave_blocks
            .get(&edge_id)
            .cloned()
            .unwrap_or_default();
        for leaf in arena.collect_leaf_pave_blocks(&roots) {
            let Some(pb) = arena.pave_blocks.get(leaf) else {
                continue;
            };
            let (t0, t1) = pb.parameter_range();
            let samples: Vec<_> = (0..=SAMPLES)
                .map(|k| {
                    let t = (t1 - t0).mul_add(f64::from(k) / f64::from(SAMPLES), t0);
                    curve.evaluate_with_endpoints(t, sp, ep)
                })
                .collect();
            let middle = samples[SAMPLES as usize / 2];
            for ((&face, region), exact) in faces.iter().zip(&loops).zip(&regions) {
                let on_surface = samples
                    .iter()
                    .all(|&p| region.distance_to_surface(p).is_some_and(|d| d <= band));
                let inside = exact.as_ref().map_or_else(
                    || region.to_uv(middle).is_some_and(|q| region.contains(q)),
                    |(frame, exact)| exact.contains_clear(frame.project(middle), tol.linear),
                );
                if !(on_surface && inside) {
                    continue;
                }
                let curve_index = arena.curves.len();
                arena.curves.push(IntersectionCurveDS {
                    curve: curve.clone(),
                    face_a: face,
                    face_b: face,
                    bbox: Aabb3::from_points(samples.iter().copied()),
                    pave_blocks: vec![leaf],
                    t_range: (t0, t1),
                });
                arena.interference.ff.push(Interference::FF {
                    f1: face,
                    f2: face,
                    curve_index,
                });
                log::debug!("tool edge {edge_id:?} leaf {leaf:?} runs in face {face:?}");
            }
        }
    }
    Ok(())
}
