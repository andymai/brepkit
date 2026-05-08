//! Small face removal — detect and remove degenerate faces.
//!
//! Faces whose outer-wire bounding-box diagonal is smaller than the
//! working tolerance are considered degenerate slivers (common after
//! boolean operations at near-tangent intersections) and are removed
//! via the [`ReShape`](crate::reshape::ReShape) tracker.

use brepkit_math::vec::Vec3;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::shell::ShellId;
use brepkit_topology::solid::SolidId;

use super::FixResult;
use super::config::FixConfig;
use crate::HealError;
use crate::context::HealContext;
use crate::status::Status;

/// Remove or merge small faces in a solid.
///
/// For each face in *every* shell of the solid (outer plus any inner
/// cavity shells), computes the bounding-box diagonal of the outer
/// wire's vertex positions. If the diagonal is smaller than
/// `ctx.tolerance.linear`, the face is recorded for removal via
/// [`ctx.reshape.remove_face`](crate::reshape::ReShape::remove_face).
///
/// Each shell is processed independently — the per-shell "don't
/// remove every face" guard preserves shell topology even when the
/// outer shell contains no small faces but a cavity shell is entirely
/// degenerate.
///
/// The removal is not applied immediately — it is deferred until
/// [`ReShape::apply`](crate::reshape::ReShape::apply) is called by the
/// top-level [`fix_shape`](super::fix_shape) function.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn fix_small_faces(
    topo: &mut Topology,
    solid_id: SolidId,
    ctx: &mut HealContext,
    _config: &FixConfig,
) -> Result<FixResult, HealError> {
    let solid_data = topo.solid(solid_id)?;
    let shell_ids: Vec<ShellId> = std::iter::once(solid_data.outer_shell())
        .chain(solid_data.inner_shells().iter().copied())
        .collect();

    let mut total_removed = 0usize;
    for shell_id in shell_ids {
        total_removed += process_shell(topo, shell_id, ctx)?;
    }

    if total_removed == 0 {
        Ok(FixResult::ok())
    } else {
        ctx.info(format!("marked {total_removed} small faces for removal"));
        Ok(FixResult {
            status: Status::DONE3,
            actions_taken: total_removed,
        })
    }
}

/// Process one shell: collect small faces, but skip removal if every
/// face in the shell would be removed (we'd otherwise produce an
/// empty shell, which is invalid topology).
fn process_shell(
    topo: &Topology,
    shell_id: ShellId,
    ctx: &mut HealContext,
) -> Result<usize, HealError> {
    let tol = ctx.tolerance.linear;
    let face_ids: Vec<FaceId> = topo.shell(shell_id)?.faces().to_vec();

    let mut to_remove: Vec<FaceId> = Vec::new();
    for &fid in &face_ids {
        if face_diagonal(topo, fid)? < tol {
            to_remove.push(fid);
        }
    }

    if to_remove.is_empty() {
        return Ok(0);
    }
    if to_remove.len() >= face_ids.len() {
        ctx.warn(format!(
            "all {} faces in shell {shell_id:?} are small; skipping removal to preserve shell",
            face_ids.len()
        ));
        return Ok(0);
    }

    for fid in &to_remove {
        ctx.reshape.remove_face(*fid);
    }
    Ok(to_remove.len())
}

/// Bounding-box diagonal of a face's outer-wire vertex positions.
fn face_diagonal(topo: &Topology, fid: FaceId) -> Result<f64, HealError> {
    let face = topo.face(fid)?;
    let wire = topo.wire(face.outer_wire())?;

    let mut min_pt = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut max_pt = Vec3::new(f64::MIN, f64::MIN, f64::MIN);

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        for &vid in &[edge.start(), edge.end()] {
            let pos = topo.vertex(vid)?.point();
            min_pt = Vec3::new(
                min_pt.x().min(pos.x()),
                min_pt.y().min(pos.y()),
                min_pt.z().min(pos.z()),
            );
            max_pt = Vec3::new(
                max_pt.x().max(pos.x()),
                max_pt.y().max(pos.y()),
                max_pt.z().max(pos.z()),
            );
        }
    }

    Ok((max_pt - min_pt).length())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use crate::context::HealContext;
    use crate::fix::FixConfig;

    /// Add a triangular face to `topo` whose vertices are at the three
    /// supplied points. Returns the face's ID.
    fn add_triangle_face(topo: &mut Topology, a: Point3, b: Point3, c: Point3) -> FaceId {
        let va = topo.add_vertex(Vertex::new(a, 1e-7));
        let vb = topo.add_vertex(Vertex::new(b, 1e-7));
        let vc = topo.add_vertex(Vertex::new(c, 1e-7));
        let eab = topo.add_edge(Edge::new(va, vb, EdgeCurve::Line));
        let ebc = topo.add_edge(Edge::new(vb, vc, EdgeCurve::Line));
        let eca = topo.add_edge(Edge::new(vc, va, EdgeCurve::Line));
        let wire = Wire::new(
            vec![
                OrientedEdge::new(eab, true),
                OrientedEdge::new(ebc, true),
                OrientedEdge::new(eca, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ))
    }

    #[test]
    fn fix_small_faces_walks_inner_shells() {
        // Build a solid with two shells, each containing one normal-sized
        // face and one degenerate-small face. The outer-shell-only
        // implementation would only flag the outer-shell small face; the
        // multi-shell implementation should flag both.
        let mut topo = Topology::new();

        // Big face (~1 unit diagonal) and small face (1e-9 diagonal,
        // well below the default 1e-7 tolerance).
        let big_outer = add_triangle_face(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        let small_outer = add_triangle_face(
            &mut topo,
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(5.0 + 1e-9, 0.0, 0.0),
            Point3::new(5.0, 1e-9, 0.0),
        );
        let big_inner = add_triangle_face(
            &mut topo,
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(1.0, 0.0, 5.0),
            Point3::new(0.0, 1.0, 5.0),
        );
        let small_inner = add_triangle_face(
            &mut topo,
            Point3::new(5.0, 0.0, 5.0),
            Point3::new(5.0 + 1e-9, 0.0, 5.0),
            Point3::new(5.0, 1e-9, 5.0),
        );

        let outer_shell = topo.add_shell(Shell::new(vec![big_outer, small_outer]).unwrap());
        let inner_shell = topo.add_shell(Shell::new(vec![big_inner, small_inner]).unwrap());
        let solid_id = topo.add_solid(Solid::new(outer_shell, vec![inner_shell]));

        let mut ctx = HealContext::new();
        let result = fix_small_faces(&mut topo, solid_id, &mut ctx, &FixConfig::default()).unwrap();

        assert_eq!(
            result.actions_taken, 2,
            "should flag both small faces (one per shell), got {}",
            result.actions_taken
        );
    }

    #[test]
    fn fix_small_faces_per_shell_guard_preserves_fully_degenerate_shell() {
        // If every face on a shell is small, the guard should refuse to
        // remove any face from THAT shell — but other shells are still
        // processed. Outer shell has 1 big + 1 small; inner shell has 2
        // small (no big). Inner shell guard fires; outer shell still
        // gets its 1 small face marked.
        let mut topo = Topology::new();

        let big_outer = add_triangle_face(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        let small_outer = add_triangle_face(
            &mut topo,
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(5.0 + 1e-9, 0.0, 0.0),
            Point3::new(5.0, 1e-9, 0.0),
        );
        let small_inner_1 = add_triangle_face(
            &mut topo,
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(1e-9, 0.0, 5.0),
            Point3::new(0.0, 1e-9, 5.0),
        );
        let small_inner_2 = add_triangle_face(
            &mut topo,
            Point3::new(5.0, 0.0, 5.0),
            Point3::new(5.0 + 1e-9, 0.0, 5.0),
            Point3::new(5.0, 1e-9, 5.0),
        );

        let outer_shell = topo.add_shell(Shell::new(vec![big_outer, small_outer]).unwrap());
        let inner_shell = topo.add_shell(Shell::new(vec![small_inner_1, small_inner_2]).unwrap());
        let solid_id = topo.add_solid(Solid::new(outer_shell, vec![inner_shell]));

        let mut ctx = HealContext::new();
        let result = fix_small_faces(&mut topo, solid_id, &mut ctx, &FixConfig::default()).unwrap();

        // Only the outer shell's small face should be marked — the
        // inner shell would lose all faces, so the guard preserves it.
        assert_eq!(
            result.actions_taken, 1,
            "expected 1 (outer shell only), got {} — inner-shell guard should have preserved it",
            result.actions_taken
        );
    }
}
