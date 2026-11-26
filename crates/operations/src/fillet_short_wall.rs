//! A round on the whole rim of a plane cap whose walls are shorter than the
//! radius.
//!
//! The part is a prism: a plane cap, a plane floor parallel to it, and walls
//! square to both, planes and cylinders along the cap's normal. When the
//! round's radius `r` exceeds the walls' height `h`, a ball touching the cap
//! and a wall would touch the wall below the floor, so the blend engine
//! refuses the edges. A ball of radius `r` can still roll touching the cap
//! and passing over the floor's rim: its centre is `r` below the cap and
//! `s = sqrt(r² - (r - h)²)` in from the wall, and the wall goes whole.
//!
//! That round is the one the blend engine makes on the same prism with its
//! walls set out by `δ = r - s` and made taller than `r`: the ball touches the
//! cap and the set-out wall, so its centre is `r` below the cap and `r` in
//! from the set-out wall, which is `s` in from the original wall. Cut back to
//! the floor, the rounded prism passes through the floor's rim, and between
//! the floor and the cap it lies inside the original walls.

use std::collections::HashSet;

use brepkit_blend::BlendResult;
use brepkit_blend::fillet_builder::FilletBuilder;
use brepkit_math::frame::Frame3;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::OperationsError;
use crate::boolean::{BooleanOp, boolean};

/// A face's outward normal when it is a plane, with a point on it.
fn outward_plane(topo: &Topology, face: FaceId) -> Result<Option<(Vec3, Point3)>, OperationsError> {
    let f = topo.face(face)?;
    let FaceSurface::Plane { normal, .. } = f.surface() else {
        return Ok(None);
    };
    let normal = if f.is_reversed() { -*normal } else { *normal };
    let wire = topo.wire(f.outer_wire())?;
    let Some(first) = wire.edges().first() else {
        return Ok(None);
    };
    let p = topo.vertex(topo.edge(first.edge())?.start())?.point();
    Ok(Some((normal, p)))
}

/// The edges of a face's outer wire.
fn outer_edges(topo: &Topology, face: FaceId) -> Result<HashSet<EdgeId>, OperationsError> {
    let f = topo.face(face)?;
    Ok(topo
        .wire(f.outer_wire())?
        .edges()
        .iter()
        .map(brepkit_topology::wire::OrientedEdge::edge)
        .collect())
}

/// The prism's cap, floor and the walls' height, when `edges` are the whole
/// rim of a plane cap of a prism whose walls are shorter than `radius`.
fn short_walled_cap(
    topo: &Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
) -> Result<Option<(FaceId, Vec3, Point3, f64)>, OperationsError> {
    const ANGLE: f64 = 1e-9;
    if !topo.solid(solid)?.inner_shells().is_empty() {
        return Ok(None);
    }
    let picked: HashSet<EdgeId> = edges.iter().copied().collect();
    let faces = solid_faces(topo, solid)?;
    let mut cap = None;
    for &f in &faces {
        if topo.face(f)?.inner_wires().is_empty()
            && outer_edges(topo, f)? == picked
            && let Some((n, p)) = outward_plane(topo, f)?
        {
            cap = Some((f, n, p));
            break;
        }
    }
    let Some((cap, n, on_cap)) = cap else {
        return Ok(None);
    };
    let mut floor = None;
    for &f in faces.iter().filter(|&&f| f != cap) {
        let face = topo.face(f)?;
        match face.surface() {
            FaceSurface::Plane { normal, .. } if normal.cross(n).length() < ANGLE => {
                let Some((m, p)) = outward_plane(topo, f)? else {
                    return Ok(None);
                };
                if m.dot(n) > 0.0 || floor.is_some() || !face.inner_wires().is_empty() {
                    return Ok(None);
                }
                floor = Some(p);
            }
            FaceSurface::Plane { normal, .. } if normal.dot(n).abs() < ANGLE => {}
            FaceSurface::Cylinder(c) if c.axis().cross(n).length() < ANGLE => {}
            _ => return Ok(None),
        }
    }
    let Some(on_floor) = floor else {
        return Ok(None);
    };
    let height = n.dot(on_cap - on_floor);
    let scale = radius.max(height);
    if height <= 1e-9 * scale || radius <= height * (1.0 + 1e-9) {
        return Ok(None);
    }
    Ok(Some((cap, n, on_floor, height)))
}

/// Round the whole rim of a prism's plane cap when its walls are shorter
/// than `radius`: the ball touches the cap and rolls over the floor's rim,
/// and the walls go. `None` when `edges` are not such a rim.
///
/// # Errors
/// Returns an error if a step of the construction fails or its result is
/// not a valid solid.
pub fn fillet_over_short_walls(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
) -> Result<Option<BlendResult>, OperationsError> {
    let Some((cap, n, on_floor, height)) = short_walled_cap(topo, solid, edges, radius)? else {
        return Ok(None);
    };
    let set_out = radius - (radius * radius - (radius - height).powi(2)).sqrt();
    // The prism made taller than the radius below the cap, its walls and
    // ends set out by `set_out`, then lowered back to the cap.
    let tall = crate::extrude::extrude(topo, cap, -n, height + radius)?;
    let outer = crate::offset_v2::offset_solid_v2(topo, tall, set_out)?;
    let lift = -n * set_out;
    crate::transform::transform_solid(
        topo,
        outer,
        &Mat4::translation(lift.x(), lift.y(), lift.z()),
    )?;
    let on_cap = on_floor + n * height;
    let mut top = None;
    for f in solid_faces(topo, outer)? {
        if let Some((m, p)) = outward_plane(topo, f)?
            && m.dot(n) > 1.0 - 1e-9
            && n.dot(p - on_cap).abs() < 1e-7 * radius.max(1.0)
        {
            top = Some(f);
        }
    }
    let Some(top) = top else {
        return Err(OperationsError::InvalidInput {
            reason: "the set-out prism has no cap at the cap's plane".into(),
        });
    };
    let rim: Vec<EdgeId> = outer_edges(topo, top)?.into_iter().collect();
    let mut builder = FilletBuilder::new(topo, outer);
    builder.add_edges(&rim, radius);
    let rounded = builder.build()?;
    if !rounded.failed.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "the set-out prism's rim was rounded on {} of {} edges",
                rounded.succeeded.len(),
                rim.len()
            ),
        });
    }
    // Everything beyond the floor goes: a box past the floor's plane, wider
    // than the part.
    let bbox = crate::measure::solid_bounding_box(topo, solid)?;
    let span = (bbox.max - bbox.min).length() + 2.0 * radius;
    let frame = Frame3::from_normal(on_floor, -n)?;
    let corner = on_floor - frame.x * span - frame.y * span;
    let block = crate::primitives::make_box(topo, 2.0 * span, 2.0 * span, 2.0 * span)?;
    let place = Mat4([
        [frame.x.x(), frame.y.x(), frame.z.x(), corner.x()],
        [frame.x.y(), frame.y.y(), frame.z.y(), corner.y()],
        [frame.x.z(), frame.y.z(), frame.z.z(), corner.z()],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    crate::transform::transform_solid(topo, block, &place)?;
    let result = boolean(topo, BooleanOp::Cut, rounded.solid, block)?;
    let report = crate::validate::validate_solid(topo, result)?;
    if !report.is_valid() {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "the round over the short walls is not valid: {:?}",
                report.issues
            ),
        });
    }
    Ok(Some(BlendResult {
        solid: result,
        succeeded: edges.to_vec(),
        failed: Vec::new(),
        is_partial: false,
    }))
}
