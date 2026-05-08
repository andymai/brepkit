//! Face fixing — wire orientation, small-area removal, duplicate detection.
//!
//! The fix sequence:
//! 1. Fix all wires in the face (delegate to `fix_wire`)
//! 2. Fix `SameParameter` for each edge on this face — rebuild PCurves
//!    that deviate from their 3D curves by more than tolerance
//! 3. Fix wire orientation (outer wire CCW from surface normal)
//! 4. Small area check (bbox diagonal < tolerance → mark for removal)
//! 5. Duplicate face detection (stub)

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::FixResult;
use super::config::{FixConfig, FixMode};
use crate::HealError;
use crate::context::HealContext;
use crate::status::Status;

/// Fix a single face: fix wires, wire orientation, small area, duplicates.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn fix_face(
    topo: &mut Topology,
    face_id: FaceId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let mut result = FixResult::ok();

    // 1. Fix all wires in this face (outer + inner).
    let face = topo.face(face_id)?;
    let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();

    for wid in &wire_ids {
        let wire_result = super::wire::fix_wire_on_face(topo, *wid, face_id, ctx, config)?;
        result.merge(&wire_result);
    }

    // 2. SameParameter: each edge on this face must have a PCurve that
    // matches its 3D curve within tolerance, otherwise downstream UV
    // operations (CDT triangulation, intersection) drift.
    //
    // Note on reshape ordering: wire fixing in step 1 records edge
    // replacements/removals in `ctx.reshape` but does NOT apply them
    // to `topo` (that happens at end-of-pipeline via `ReShape::apply`).
    // We deliberately skip edges already marked for removal here —
    // their PCurves are irrelevant. Edges replaced by other edges are
    // resolved through `reshape.resolve_edge` so we operate on the
    // current canonical edge.
    if config.fix_same_parameter != FixMode::Off {
        // Collect edges first (we'll mutate `topo` inside the loop).
        // Propagate wire-lookup errors with `?` for consistency with
        // the rest of `fix_face`: at this point all `wire_ids` were
        // already accessible in step 1, so an error here would
        // indicate an unexpected topology mutation inside
        // `fix_wire_on_face` and shouldn't be silently swallowed.
        let mut edge_ids: Vec<brepkit_topology::edge::EdgeId> = Vec::new();
        for &wid in &wire_ids {
            let w = topo.wire(wid)?;
            edge_ids.extend(
                w.edges()
                    .iter()
                    .map(brepkit_topology::wire::OrientedEdge::edge),
            );
        }
        // Resolve any reshape-recorded edge replacements, skip removed
        // edges, and dedupe (the same edge can appear in outer + inner
        // wires, and `ReShape::resolve_edge` can collapse multiple
        // replaced edges onto the same target).
        let mut seen = std::collections::HashSet::new();
        edge_ids.retain(|&eid| match ctx.reshape.resolve_edge(eid) {
            Some(canonical) => seen.insert(canonical),
            None => false,
        });
        for eid in edge_ids {
            let r = super::edge::fix_same_parameter_on_face(topo, eid, face_id, ctx, config)?;
            result.merge(&r);
        }
    }

    // 3. Fix wire orientation relative to the face normal.
    if config.fix_wire_orientation != FixMode::Off {
        let r = fix_wire_orientation(topo, face_id, ctx, config)?;
        result.merge(&r);
    }

    // 3. Small area check.
    if config.fix_small_area != FixMode::Off {
        let r = fix_small_area(topo, face_id, ctx, config)?;
        result.merge(&r);
    }

    // 4. Duplicate face detection stub.
    if config.fix_duplicate_faces != FixMode::Off {
        let r = fix_duplicate_faces(ctx, config);
        result.merge(&r);
    }

    Ok(result)
}

// ── Fix implementations ─────────────────────────────────────────────────

/// Fix wire orientation: outer wire should be CCW when viewed from the
/// surface normal.
///
/// For planar faces this computes the signed area of the outer wire
/// projected onto the face normal. If the area is negative (CW), the
/// face normal is flipped.
///
/// Ported from `operations::heal::fix_face_orientations`.
#[allow(clippy::too_many_lines)]
fn fix_wire_orientation(
    topo: &mut Topology,
    face_id: FaceId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let face = topo.face(face_id)?;
    let surface = face.surface().clone();
    let outer_wire_id = face.outer_wire();

    // Compute the face centroid from outer wire vertices.
    let wire = topo.wire(outer_wire_id)?;
    let edges = wire.edges();
    if edges.is_empty() {
        return Ok(FixResult::ok());
    }

    // Collect vertex positions from the outer wire.
    let mut positions: Vec<Point3> = Vec::new();
    for oe in edges {
        let edge = topo.edge(oe.edge())?;
        let start_pos = topo.vertex(oe.oriented_start(edge))?.point();
        positions.push(start_pos);
    }

    if positions.len() < 3 {
        return Ok(FixResult::ok());
    }

    // Compute face normal at a representative point.
    let face_normal = match &surface {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => {
            // For non-planar faces, use Newell's method to get the polygon
            // normal, which serves as a proxy for the surface normal
            // direction.
            newell_normal(&positions)
        }
    };

    // Compute signed area of the wire polygon projected onto the normal.
    let signed_area = projected_signed_area(&positions, &face_normal);

    // If signed_area < 0 the wire is CW (wrong orientation).
    let is_cw = signed_area < 0.0;

    if !config.fix_wire_orientation.should_fix(is_cw) {
        return Ok(FixResult::ok());
    }

    if !is_cw {
        return Ok(FixResult::ok());
    }

    // Fix: for planar faces, flip the normal.
    // For non-planar faces, toggle the reversed flag.
    if let FaceSurface::Plane { normal, d } = &surface {
        let flipped_normal = -*normal;
        let flipped_d = -*d;
        let face_mut = topo.face_mut(face_id)?;
        face_mut.set_surface(FaceSurface::Plane {
            normal: flipped_normal,
            d: flipped_d,
        });
    } else {
        // For analytic/NURBS surfaces, toggle the reversed flag.
        let face_data = topo.face(face_id)?;
        let was_reversed = face_data.is_reversed();
        let face_mut = topo.face_mut(face_id)?;
        face_mut.set_reversed(!was_reversed);
    }

    ctx.info(format!(
        "Face {face_id:?}: flipped orientation (wire was CW, signed_area={signed_area:.4e})",
    ));

    Ok(FixResult {
        status: Status::DONE1,
        actions_taken: 1,
    })
}

/// Check if the face is too small and mark for removal.
fn fix_small_area(
    topo: &Topology,
    face_id: FaceId,
    ctx: &mut HealContext,
    config: &FixConfig,
) -> Result<FixResult, HealError> {
    let analysis = crate::analysis::face::analyze_face(topo, face_id, &ctx.tolerance)?;

    if !config.fix_small_area.should_fix(analysis.is_small) {
        return Ok(FixResult::ok());
    }

    ctx.info(format!(
        "Face {face_id:?}: small face (bbox_diagonal={:.2e}), marking for removal",
        analysis.bbox_diagonal,
    ));
    ctx.reshape.remove_face(face_id);

    Ok(FixResult {
        status: Status::DONE2,
        actions_taken: 1,
    })
}

/// Stub: duplicate face detection.
fn fix_duplicate_faces(ctx: &mut HealContext, config: &FixConfig) -> FixResult {
    if !config.fix_duplicate_faces.should_fix(false) {
        return FixResult::ok();
    }
    ctx.warn("Duplicate face fix: not yet implemented (TODO)".to_string());
    FixResult::ok()
}

// ── Geometry helpers ────────────────────────────────────────────────────

/// Compute the normal of a polygon via Newell's method.
///
/// Returns a unit vector or `Vec3::Z` for degenerate polygons.
fn newell_normal(positions: &[Point3]) -> Vec3 {
    let n = positions.len();
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;

    for i in 0..n {
        let curr = positions[i];
        let next = positions[(i + 1) % n];
        nx += (curr.y() - next.y()) * (curr.z() + next.z());
        ny += (curr.z() - next.z()) * (curr.x() + next.x());
        nz += (curr.x() - next.x()) * (curr.y() + next.y());
    }

    let normal = Vec3::new(nx, ny, nz);
    normal.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0))
}

/// Compute the signed area of a polygon projected onto a plane with the
/// given normal.
///
/// Positive → CCW when viewed from the normal direction.
/// Negative → CW.
fn projected_signed_area(positions: &[Point3], normal: &Vec3) -> f64 {
    let n = positions.len();
    if n < 3 {
        return 0.0;
    }

    // Use the centroid as the reference point for the cross-product fan.
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    #[allow(clippy::cast_precision_loss)]
    let inv_n = 1.0 / n as f64;
    for p in positions {
        cx += p.x();
        cy += p.y();
        cz += p.z();
    }
    let centroid = Point3::new(cx * inv_n, cy * inv_n, cz * inv_n);

    let mut area2 = 0.0;
    for i in 0..n {
        let a = positions[i] - centroid;
        let b = positions[(i + 1) % n] - centroid;
        let cross = a.cross(b);
        area2 += normal.dot(cross);
    }

    area2 * 0.5
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::context::HealContext;
    use crate::fix::config::{FixConfig, FixMode};
    use brepkit_topology::test_utils::make_unit_square_face;

    fn default_config() -> FixConfig {
        // Disable face-level steps that aren't being exercised
        // (orientation, small-area, duplicate detection). Wire-level
        // fixes still run with their `FixConfig::default()` values
        // (Auto), since SameParameter operates AFTER wire fixing and
        // we want the realistic flow.
        FixConfig {
            fix_wire_orientation: FixMode::Off,
            fix_small_area: FixMode::Off,
            fix_duplicate_faces: FixMode::Off,
            fix_same_parameter: FixMode::Auto,
            ..FixConfig::default()
        }
    }

    #[test]
    fn fix_face_creates_missing_pcurves_via_same_parameter() {
        // make_unit_square_face produces a planar face where the 4 edges
        // have NO PCurves attached (test_utils doesn't register them).
        // After fix_face with fix_same_parameter, every edge on the face
        // should now have a registered PCurve.
        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);

        // Pre-condition: no PCurves yet.
        let edges_before: Vec<_> = {
            let face = topo.face(face_id).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            wire.edges()
                .iter()
                .map(brepkit_topology::wire::OrientedEdge::edge)
                .collect()
        };
        for &eid in &edges_before {
            assert!(
                !topo.pcurves().contains(eid, face_id),
                "edge {eid:?} should not have a PCurve before fix_face"
            );
        }

        let mut ctx = HealContext::new();
        let cfg = default_config();
        let result = fix_face(&mut topo, face_id, &mut ctx, &cfg).unwrap();

        // Post-condition: every edge has a PCurve.
        for &eid in &edges_before {
            assert!(
                topo.pcurves().contains(eid, face_id),
                "edge {eid:?} should have a PCurve after fix_face"
            );
        }
        // And SameParameter ran for each edge → some actions taken.
        assert!(result.actions_taken >= edges_before.len());
    }

    #[test]
    fn fix_face_skips_same_parameter_when_off() {
        // Symmetric: with fix_same_parameter = Off, no PCurves are created.
        let mut topo = Topology::new();
        let face_id = make_unit_square_face(&mut topo);
        let edges: Vec<_> = {
            let face = topo.face(face_id).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            wire.edges()
                .iter()
                .map(brepkit_topology::wire::OrientedEdge::edge)
                .collect()
        };

        let mut ctx = HealContext::new();
        let cfg = FixConfig {
            fix_same_parameter: FixMode::Off,
            ..default_config()
        };
        let _ = fix_face(&mut topo, face_id, &mut ctx, &cfg).unwrap();

        for &eid in &edges {
            assert!(
                !topo.pcurves().contains(eid, face_id),
                "edge {eid:?} should still have no PCurve when fix_same_parameter=Off"
            );
        }
    }
}
