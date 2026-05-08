//! Built-in pipeline operators.
//!
//! These wrap the analysis/fix/upgrade functions as [`super::operator::HealOperator`]
//! implementations for use in [`super::process::HealProcess`] pipelines.

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use super::operator::HealOperator;
use super::registry::OperatorRegistry;
use crate::HealError;
use crate::context::HealContext;
use crate::fix::FixResult;
use crate::fix::config::FixConfig;

/// Register all built-in operators into a registry.
pub fn register_builtins(registry: &mut OperatorRegistry) {
    registry.register("fix_shape", Box::new(FixShapeOp));
    registry.register("unify_same_domain", Box::new(UnifySameDomainOp));
    registry.register("direct_faces", Box::new(DirectFacesOp));
    registry.register("same_parameter", Box::new(SameParameterOp));
    registry.register("merge_vertices", Box::new(MergeVerticesOp));
    registry.register("drop_small_edges", Box::new(DropSmallEdgesOp));
    registry.register("drop_small_faces", Box::new(DropSmallFacesOp));
    registry.register("remove_internal_wires", Box::new(RemoveInternalWiresOp));
    registry.register("sew_shells", Box::new(SewShellsOp));
    registry.register("split_common_vertex", Box::new(SplitCommonVertexOp));
    registry.register("convert_to_bspline", Box::new(ConvertToBSplineOp));
    registry.register("convert_to_elementary", Box::new(ConvertToElementaryOp));
    registry.register("fix_wireframe", Box::new(FixWireframeOp));
}

// ── fix_shape ──────────────────────────────────────────────────────

/// Full recursive shape fix (solid → shell → face → wire → edge).
#[derive(Debug)]
struct FixShapeOp;

impl HealOperator for FixShapeOp {
    fn name(&self) -> &'static str {
        "fix_shape"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        _ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let config = FixConfig::default();
        let (new_solid, result) = crate::fix::fix_shape(topo, solid_id, &config)?;
        Ok((new_solid, result))
    }
}

// ── unify_same_domain ──────────────────────────────────────────────

/// Merge adjacent faces sharing the same surface.
#[derive(Debug)]
struct UnifySameDomainOp;

impl HealOperator for UnifySameDomainOp {
    fn name(&self) -> &'static str {
        "unify_same_domain"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        _ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let options = crate::upgrade::unify_same_domain::UnifyOptions::default();
        let (new_solid, unify) =
            crate::upgrade::unify_same_domain::unify_same_domain(topo, solid_id, &options)?;
        let actions = unify.faces_merged + unify.edges_merged;
        let status = if actions > 0 {
            crate::status::Status::DONE1
        } else {
            crate::status::Status::OK
        };
        Ok((
            new_solid,
            FixResult {
                status,
                actions_taken: actions,
            },
        ))
    }
}

// ── direct_faces ───────────────────────────────────────────────────

/// Orient all faces so normals point outward.
#[derive(Debug)]
struct DirectFacesOp;

impl HealOperator for DirectFacesOp {
    fn name(&self) -> &'static str {
        "direct_faces"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let config = FixConfig {
            fix_orientation: crate::fix::FixMode::On,
            ..Default::default()
        };
        // Only run shell orientation fix.
        let solid_data = topo.solid(solid_id)?;
        let shell_id = solid_data.outer_shell();
        let result = crate::fix::shell::fix_shell(topo, shell_id, ctx, &config)?;
        Ok((solid_id, result))
    }
}

// ── same_parameter ─────────────────────────────────────────────────

/// Fix PCurve/3D curve consistency for all edges.
#[derive(Debug)]
struct SameParameterOp;

impl HealOperator for SameParameterOp {
    fn name(&self) -> &'static str {
        "same_parameter"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let config = FixConfig {
            fix_same_parameter: crate::fix::FixMode::On,
            ..Default::default()
        };
        let solid_data = topo.solid(solid_id)?;
        let shell = topo.shell(solid_data.outer_shell())?;
        let face_ids: Vec<_> = shell.faces().to_vec();

        let mut aggregate = FixResult::ok();
        for &fid in &face_ids {
            let face = topo.face(fid)?;
            let wire = topo.wire(face.outer_wire())?;
            let edge_ids: Vec<_> = wire
                .edges()
                .iter()
                .map(brepkit_topology::wire::OrientedEdge::edge)
                .collect();
            for eid in edge_ids {
                let r = crate::fix::edge::fix_edge(topo, eid, ctx, &config)?;
                aggregate.merge(&r);
            }
        }

        Ok((solid_id, aggregate))
    }
}

// ── merge_vertices ─────────────────────────────────────────────────

/// Merge coincident vertices across the solid.
#[derive(Debug)]
struct MergeVerticesOp;

impl HealOperator for MergeVerticesOp {
    fn name(&self) -> &'static str {
        "merge_vertices"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let config = FixConfig {
            fix_coincident_vertices: crate::fix::FixMode::On,
            ..Default::default()
        };
        let result = crate::fix::solid::fix_solid(topo, solid_id, ctx, &config)?;
        let new_solid = ctx.reshape.apply(topo, solid_id)?;
        Ok((new_solid, result))
    }
}

// ── drop_small_edges ───────────────────────────────────────────────

/// Remove edges shorter than tolerance.
#[derive(Debug)]
struct DropSmallEdgesOp;

impl HealOperator for DropSmallEdgesOp {
    fn name(&self) -> &'static str {
        "drop_small_edges"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let config = FixConfig {
            fix_small_edges: crate::fix::FixMode::On,
            ..Default::default()
        };
        let result = crate::fix::solid::fix_solid(topo, solid_id, ctx, &config)?;
        let new_solid = ctx.reshape.apply(topo, solid_id)?;
        Ok((new_solid, result))
    }
}

// ── drop_small_faces ───────────────────────────────────────────────

/// Remove faces with area below tolerance.
#[derive(Debug)]
struct DropSmallFacesOp;

impl HealOperator for DropSmallFacesOp {
    fn name(&self) -> &'static str {
        "drop_small_faces"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let config = FixConfig {
            fix_small_faces: crate::fix::FixMode::On,
            ..Default::default()
        };
        let result = crate::fix::small_face::fix_small_faces(topo, solid_id, ctx, &config)?;
        let new_solid = ctx.reshape.apply(topo, solid_id)?;
        Ok((new_solid, result))
    }
}

// ── remove_internal_wires ──────────────────────────────────────────

/// Drop internal (hole) wires from all faces.
#[derive(Debug)]
struct RemoveInternalWiresOp;

impl HealOperator for RemoveInternalWiresOp {
    fn name(&self) -> &'static str {
        "remove_internal_wires"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        _ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let removed = crate::upgrade::remove_internal_wires::remove_internal_wires(topo, solid_id)?;
        let status = if removed > 0 {
            crate::status::Status::DONE1
        } else {
            crate::status::Status::OK
        };
        Ok((
            solid_id,
            FixResult {
                status,
                actions_taken: removed,
            },
        ))
    }
}

// ── sew_shells ─────────────────────────────────────────────────────

/// Sew free boundaries in shells.
#[derive(Debug)]
struct SewShellsOp;

impl HealOperator for SewShellsOp {
    fn name(&self) -> &'static str {
        "sew_shells"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let solid_data = topo.solid(solid_id)?;
        let shell_id = solid_data.outer_shell();
        let sewn = crate::upgrade::shell_sewing::sew_shell(topo, shell_id, ctx.tolerance.linear)?;
        let status = if sewn > 0 {
            crate::status::Status::DONE1
        } else {
            crate::status::Status::OK
        };
        Ok((
            solid_id,
            FixResult {
                status,
                actions_taken: sewn,
            },
        ))
    }
}

// ── split_common_vertex ────────────────────────────────────────────

/// Split vertices shared by too many non-adjacent edges.
#[derive(Debug)]
struct SplitCommonVertexOp;

impl HealOperator for SplitCommonVertexOp {
    fn name(&self) -> &'static str {
        "split_common_vertex"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let config = FixConfig {
            fix_split_common_vertex: crate::fix::FixMode::On,
            ..Default::default()
        };
        let result =
            crate::fix::split_vertex::fix_split_common_vertex(topo, solid_id, ctx, &config)?;
        Ok((solid_id, result))
    }
}

// ── convert_to_bspline ─────────────────────────────────────────────

/// Convert all geometry to B-Spline representation.
#[derive(Debug)]
struct ConvertToBSplineOp;

impl HealOperator for ConvertToBSplineOp {
    fn name(&self) -> &'static str {
        "convert_to_bspline"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        _ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let converted =
            crate::custom::convert_to_bspline::convert_solid_to_bspline(topo, solid_id)?;
        let status = if converted > 0 {
            crate::status::Status::DONE1
        } else {
            crate::status::Status::OK
        };
        Ok((
            solid_id,
            FixResult {
                status,
                actions_taken: converted,
            },
        ))
    }
}

// ── convert_to_elementary ──────────────────────────────────────────

/// Recognize and replace NURBS with elementary surfaces.
#[derive(Debug)]
struct ConvertToElementaryOp;

impl HealOperator for ConvertToElementaryOp {
    fn name(&self) -> &'static str {
        "convert_to_elementary"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let converted = crate::custom::convert_to_elementary::convert_to_elementary(
            topo,
            solid_id,
            &ctx.tolerance,
        )?;
        let status = if converted > 0 {
            crate::status::Status::DONE1
        } else {
            crate::status::Status::OK
        };
        Ok((
            solid_id,
            FixResult {
                status,
                actions_taken: converted,
            },
        ))
    }
}

// ── fix_wireframe ──────────────────────────────────────────────────

/// Repair missing or misaligned edges in shells.
#[derive(Debug)]
struct FixWireframeOp;

impl HealOperator for FixWireframeOp {
    fn name(&self) -> &'static str {
        "fix_wireframe"
    }

    fn execute(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
        ctx: &mut HealContext,
    ) -> Result<(SolidId, FixResult), HealError> {
        let config = FixConfig {
            fix_wireframe: crate::fix::FixMode::On,
            ..Default::default()
        };
        let solid_data = topo.solid(solid_id)?;
        let shell_id = solid_data.outer_shell();
        let result = crate::fix::wireframe::fix_wireframe(topo, shell_id, ctx, &config)?;
        Ok((solid_id, result))
    }
}
