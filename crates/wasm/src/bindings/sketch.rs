//! 2D sketch constraint solver bindings.

#![allow(clippy::missing_errors_doc)]

use wasm_bindgen::prelude::*;

use crate::error::WasmError;
use crate::helpers::{json_f64, json_usize, parse_sketch_constraint};
use crate::kernel::BrepKernel;
use crate::state::SketchState;

/// Build a `GcsSystem` from a `SketchState`, returning the system along with
/// the `PointId` handles needed for result readback.
///
/// Shared between `sketch_solve` (when arcs are present) and `sketch_dof`.
/// Result of building a GCS from sketch state.
#[allow(dead_code)]
struct GcsBuildResult {
    sys: brepkit_operations::sketch::GcsSystem,
    point_ids: Vec<brepkit_operations::sketch::PointId>,
    arc_ids: Vec<brepkit_operations::sketch::ArcId>,
    circle_ids: Vec<brepkit_operations::sketch::CircleId>,
}

fn build_gcs_from_state(sk: &SketchState) -> Result<GcsBuildResult, JsError> {
    use brepkit_operations::sketch::GcsConstraint;

    let mut sys = brepkit_operations::sketch::GcsSystem::new();

    // Add points
    let point_ids: Vec<brepkit_operations::sketch::PointId> = sk
        .points
        .iter()
        .map(|p| {
            sys.add_point(brepkit_operations::sketch::PointData {
                x: p.x,
                y: p.y,
                fixed: p.fixed,
            })
        })
        .collect();

    // Add arcs
    let mut arc_ids = Vec::with_capacity(sk.arcs.len());
    for &(center, start, end) in &sk.arcs {
        let aid = sys
            .add_arc(point_ids[center], point_ids[start], point_ids[end])
            .map_err(|e| WasmError::InvalidInput {
                reason: format!("failed to add arc: {e}"),
            })?;
        arc_ids.push(aid);
    }

    // Add circles
    let mut circle_ids = Vec::with_capacity(sk.circles.len());
    for &(center, radius) in &sk.circles {
        let cid =
            sys.add_circle(point_ids[center], radius)
                .map_err(|e| WasmError::InvalidInput {
                    reason: format!("failed to add circle: {e}"),
                })?;
        circle_ids.push(cid);
    }

    // Implicit line cache for point-pair-based constraints
    let mut line_cache: std::collections::HashMap<
        (usize, usize),
        brepkit_operations::sketch::LineId,
    > = std::collections::HashMap::new();

    let mut get_or_create_line = |sys: &mut brepkit_operations::sketch::GcsSystem,
                                  ids: &[brepkit_operations::sketch::PointId],
                                  a: usize,
                                  b: usize|
     -> Option<brepkit_operations::sketch::LineId> {
        if let std::collections::hash_map::Entry::Vacant(e) = line_cache.entry((a, b)) {
            if let Ok(lid) = sys.add_line(ids[a], ids[b]) {
                e.insert(lid);
            }
        }
        line_cache.get(&(a, b)).copied()
    };

    // Convert legacy constraints
    for c in &sk.constraints {
        let _ = match c {
            brepkit_operations::sketch::Constraint::Coincident(a, b) => {
                sys.add_constraint(GcsConstraint::Coincident(point_ids[*a], point_ids[*b]))
            }
            brepkit_operations::sketch::Constraint::Distance(a, b, d) => {
                sys.add_constraint(GcsConstraint::Distance(point_ids[*a], point_ids[*b], *d))
            }
            brepkit_operations::sketch::Constraint::FixX(p, v) => {
                sys.add_constraint(GcsConstraint::FixX(point_ids[*p], *v))
            }
            brepkit_operations::sketch::Constraint::FixY(p, v) => {
                sys.add_constraint(GcsConstraint::FixY(point_ids[*p], *v))
            }
            brepkit_operations::sketch::Constraint::Horizontal(a, b) => {
                if let Some(l) = get_or_create_line(&mut sys, &point_ids, *a, *b) {
                    sys.add_constraint(GcsConstraint::Horizontal(l))
                } else {
                    continue;
                }
            }
            brepkit_operations::sketch::Constraint::Vertical(a, b) => {
                if let Some(l) = get_or_create_line(&mut sys, &point_ids, *a, *b) {
                    sys.add_constraint(GcsConstraint::Vertical(l))
                } else {
                    continue;
                }
            }
            brepkit_operations::sketch::Constraint::Angle(a, b, c, d, theta) => {
                let l1 = get_or_create_line(&mut sys, &point_ids, *a, *b);
                let l2 = get_or_create_line(&mut sys, &point_ids, *c, *d);
                if let (Some(l1), Some(l2)) = (l1, l2) {
                    sys.add_constraint(GcsConstraint::Angle(l1, l2, *theta))
                } else {
                    continue;
                }
            }
            brepkit_operations::sketch::Constraint::Perpendicular(a, b, c, d) => {
                let l1 = get_or_create_line(&mut sys, &point_ids, *a, *b);
                let l2 = get_or_create_line(&mut sys, &point_ids, *c, *d);
                if let (Some(l1), Some(l2)) = (l1, l2) {
                    sys.add_constraint(GcsConstraint::Perpendicular(l1, l2))
                } else {
                    continue;
                }
            }
            brepkit_operations::sketch::Constraint::Parallel(a, b, c, d) => {
                let l1 = get_or_create_line(&mut sys, &point_ids, *a, *b);
                let l2 = get_or_create_line(&mut sys, &point_ids, *c, *d);
                if let (Some(l1), Some(l2)) = (l1, l2) {
                    sys.add_constraint(GcsConstraint::Parallel(l1, l2))
                } else {
                    continue;
                }
            }
        };
    }

    // Resolve deferred (arc-referencing) constraints now that we have real IDs
    for val in &sk.deferred_constraints {
        let gc = resolve_deferred_constraint(
            val,
            &point_ids,
            &arc_ids,
            &circle_ids,
            &mut line_cache,
            &mut sys,
        )?;
        let _ = sys
            .add_constraint(gc)
            .map_err(|e| WasmError::InvalidInput {
                reason: format!("failed to add deferred constraint: {e}"),
            })?;
    }

    Ok(GcsBuildResult {
        sys,
        point_ids,
        arc_ids,
        circle_ids,
    })
}

/// Resolve a deferred constraint JSON value into a real `GcsConstraint`
/// using the entity IDs created during `build_gcs_from_state`.
fn resolve_deferred_constraint(
    val: &serde_json::Value,
    point_ids: &[brepkit_operations::sketch::PointId],
    arc_ids: &[brepkit_operations::sketch::ArcId],
    circle_ids: &[brepkit_operations::sketch::CircleId],
    line_cache: &mut std::collections::HashMap<(usize, usize), brepkit_operations::sketch::LineId>,
    sys: &mut brepkit_operations::sketch::GcsSystem,
) -> Result<brepkit_operations::sketch::GcsConstraint, JsError> {
    use brepkit_operations::sketch::GcsConstraint;

    let ty = val["type"].as_str().unwrap_or("");

    let get_point = |key: &str| -> Result<brepkit_operations::sketch::PointId, JsError> {
        let idx = json_usize(val, key)?;
        point_ids.get(idx).copied().ok_or_else(|| {
            WasmError::InvalidInput {
                reason: format!("point index {idx} out of range"),
            }
            .into()
        })
    };

    let get_arc = |key: &str| -> Result<brepkit_operations::sketch::ArcId, JsError> {
        let idx = json_usize(val, key)?;
        arc_ids.get(idx).copied().ok_or_else(|| {
            WasmError::InvalidInput {
                reason: format!("arc index {idx} out of range"),
            }
            .into()
        })
    };

    let get_circle = |key: &str| -> Result<brepkit_operations::sketch::CircleId, JsError> {
        let idx = json_usize(val, key)?;
        circle_ids.get(idx).copied().ok_or_else(|| {
            WasmError::InvalidInput {
                reason: format!("circle index {idx} out of range"),
            }
            .into()
        })
    };

    let get_or_create_line = |p1: usize,
                              p2: usize,
                              cache: &mut std::collections::HashMap<
        (usize, usize),
        brepkit_operations::sketch::LineId,
    >,
                              s: &mut brepkit_operations::sketch::GcsSystem|
     -> Result<brepkit_operations::sketch::LineId, JsError> {
        if p1 >= point_ids.len() || p2 >= point_ids.len() {
            return Err(WasmError::InvalidInput {
                reason: "point index out of range in deferred constraint".to_string(),
            }
            .into());
        }
        let key = (p1, p2);
        if let Some(&lid) = cache.get(&key) {
            return Ok(lid);
        }
        let lid =
            s.add_line(point_ids[p1], point_ids[p2])
                .map_err(|e| WasmError::InvalidInput {
                    reason: format!("failed to create line: {e}"),
                })?;
        cache.insert(key, lid);
        Ok(lid)
    };

    match ty {
        "pointOnArc" => {
            let pt = get_point("point")?;
            let arc = get_arc("arc")?;
            Ok(GcsConstraint::PointOnArc(pt, arc))
        }
        "tangentLineArc" => {
            let line_arr = val["line"]
                .as_array()
                .ok_or_else(|| WasmError::InvalidInput {
                    reason: "tangentLineArc requires 'line' as [p1, p2]".into(),
                })?;
            if line_arr.len() != 2 {
                return Err(WasmError::InvalidInput {
                    reason: "tangentLineArc 'line' must have exactly 2 point indices".into(),
                }
                .into());
            }
            #[allow(clippy::cast_possible_truncation)]
            let lp1 = line_arr[0]
                .as_u64()
                .ok_or_else(|| WasmError::InvalidInput {
                    reason: "tangentLineArc line[0] must be an integer".into(),
                })? as usize;
            #[allow(clippy::cast_possible_truncation)]
            let lp2 = line_arr[1]
                .as_u64()
                .ok_or_else(|| WasmError::InvalidInput {
                    reason: "tangentLineArc line[1] must be an integer".into(),
                })? as usize;
            let lid = get_or_create_line(lp1, lp2, line_cache, sys)?;
            let arc = get_arc("arc")?;
            let pt = get_point("point")?;
            Ok(GcsConstraint::TangentLineArc(lid, arc, pt))
        }
        "tangentArcArc" => {
            let arc1 = get_arc("arc1")?;
            let arc2 = get_arc("arc2")?;
            let pt = get_point("point")?;
            Ok(GcsConstraint::TangentArcArc(arc1, arc2, pt))
        }
        "equalRadiusArcArc" => {
            let arc1 = get_arc("arc1")?;
            let arc2 = get_arc("arc2")?;
            Ok(GcsConstraint::EqualRadiusArcArc(arc1, arc2))
        }
        "equalRadiusArcCircle" => {
            let arc = get_arc("arc")?;
            let circle = get_circle("circle")?;
            Ok(GcsConstraint::EqualRadiusArcCircle(arc, circle))
        }
        "arcLength" => {
            let arc = get_arc("arc")?;
            let v = json_f64(val, "value")?;
            Ok(GcsConstraint::ArcLength(arc, v))
        }
        "concentricArcArc" => {
            let arc1 = get_arc("arc1")?;
            let arc2 = get_arc("arc2")?;
            Ok(GcsConstraint::ConcentricArcArc(arc1, arc2))
        }
        "concentricArcCircle" => {
            let arc = get_arc("arc")?;
            let circle = get_circle("circle")?;
            Ok(GcsConstraint::ConcentricArcCircle(arc, circle))
        }
        "pointOnCircle" => {
            let pt = get_point("point")?;
            let circle = get_circle("circle")?;
            Ok(GcsConstraint::PointOnCircle(pt, circle))
        }
        _ => Err(WasmError::InvalidInput {
            reason: format!("unknown constraint type: {ty}"),
        }
        .into()),
    }
}

#[wasm_bindgen]
impl BrepKernel {
    /// Create a new empty sketch. Returns a sketch index.
    #[wasm_bindgen(js_name = "sketchNew")]
    pub fn sketch_new(&mut self) -> u32 {
        self.sketches.push(SketchState::default());
        #[allow(clippy::cast_possible_truncation)]
        let idx = (self.sketches.len() - 1) as u32;
        idx
    }

    /// Add a point to a sketch. Returns the point index.
    #[wasm_bindgen(js_name = "sketchAddPoint")]
    pub fn sketch_add_point(
        &mut self,
        sketch: u32,
        x: f64,
        y: f64,
        fixed: bool,
    ) -> Result<u32, JsError> {
        let sk = self
            .sketches
            .get_mut(sketch as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "sketch",
                index: sketch as usize,
            })?;
        let pt = if fixed {
            brepkit_operations::sketch::SketchPoint::fixed(x, y)
        } else {
            brepkit_operations::sketch::SketchPoint::new(x, y)
        };
        sk.points.push(pt);
        #[allow(clippy::cast_possible_truncation)]
        Ok((sk.points.len() - 1) as u32)
    }

    /// Add an arc to a sketch (defined by center, start, end point indices).
    /// Returns the arc index.
    #[wasm_bindgen(js_name = "sketchAddArc")]
    pub fn sketch_add_arc(
        &mut self,
        sketch: u32,
        center_idx: u32,
        start_idx: u32,
        end_idx: u32,
    ) -> Result<u32, JsError> {
        let sk = self
            .sketches
            .get_mut(sketch as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "sketch",
                index: sketch as usize,
            })?;
        let center = center_idx as usize;
        let start = start_idx as usize;
        let end = end_idx as usize;
        if center >= sk.points.len() || start >= sk.points.len() || end >= sk.points.len() {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "arc point index out of range (center={center}, start={start}, \
                     end={end}, points={})",
                    sk.points.len()
                ),
            }
            .into());
        }
        sk.arcs.push((center, start, end));
        #[allow(clippy::cast_possible_truncation)]
        Ok((sk.arcs.len() - 1) as u32)
    }

    /// Add a circle to a sketch.
    ///
    /// `center_idx` must be a valid point index. Returns the circle index
    /// (0-based) for use in circle-referencing constraints.
    #[wasm_bindgen(js_name = "sketchAddCircle")]
    pub fn sketch_add_circle(
        &mut self,
        sketch: u32,
        center_idx: u32,
        radius: f64,
    ) -> Result<u32, JsError> {
        let sk = self
            .sketches
            .get_mut(sketch as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "sketch",
                index: sketch as usize,
            })?;
        let center = center_idx as usize;
        if center >= sk.points.len() {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "circle center index out of range (center={center}, points={})",
                    sk.points.len()
                ),
            }
            .into());
        }
        if radius <= 0.0 || !radius.is_finite() {
            return Err(WasmError::InvalidInput {
                reason: format!("circle radius must be positive and finite, got {radius}"),
            }
            .into());
        }
        sk.circles.push((center, radius));
        #[allow(clippy::cast_possible_truncation)]
        Ok((sk.circles.len() - 1) as u32)
    }

    /// Add a constraint to a sketch from a JSON string.
    ///
    /// Supports all legacy constraint types plus arc-referencing constraints:
    /// `tangentLineArc`, `tangentArcArc`, `pointOnArc`, `equalRadiusArcArc`,
    /// `arcLength`, `concentricArcArc`.
    #[wasm_bindgen(js_name = "sketchAddConstraint")]
    pub fn sketch_add_constraint(&mut self, sketch: u32, json: &str) -> Result<(), JsError> {
        let sk = self
            .sketches
            .get_mut(sketch as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "sketch",
                index: sketch as usize,
            })?;
        let val: serde_json::Value =
            serde_json::from_str(json).map_err(|e| WasmError::InvalidInput {
                reason: format!("invalid constraint JSON: {e}"),
            })?;

        // Try legacy constraint first; fall back to deferred (arc-aware) storage
        match parse_sketch_constraint(&val) {
            Ok(constraint) => {
                sk.constraints.push(constraint);
            }
            Err(e) => {
                // Only defer known arc constraint types — don't silently swallow parse errors
                let ty = val.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let arc_types: &[&str] = &[
                    "tangentLineArc",
                    "tangentArcArc",
                    "pointOnArc",
                    "pointOnCircle",
                    "equalRadiusArcArc",
                    "equalRadiusArcCircle",
                    "arcLength",
                    "concentricArcArc",
                    "concentricArcCircle",
                ];
                if arc_types.contains(&ty) {
                    sk.deferred_constraints.push(val);
                } else {
                    return Err(WasmError::InvalidInput {
                        reason: format!("failed to parse constraint: {e:?}"),
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    /// Solve the sketch constraints.
    ///
    /// Returns a JSON string with converged status, iteration count, point
    /// positions, and arc definitions.
    #[wasm_bindgen(js_name = "sketchSolve")]
    pub fn sketch_solve(
        &mut self,
        sketch: u32,
        max_iterations: u32,
        tolerance: f64,
    ) -> Result<String, JsError> {
        let sk = self
            .sketches
            .get_mut(sketch as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "sketch",
                index: sketch as usize,
            })?;

        // If no arcs and no deferred constraints, use the fast legacy path
        if sk.arcs.is_empty() && sk.deferred_constraints.is_empty() {
            let mut sketch_obj = brepkit_operations::sketch::Sketch {
                points: std::mem::take(&mut sk.points),
                constraints: std::mem::take(&mut sk.constraints),
            };
            let result = sketch_obj.solve(max_iterations as usize, tolerance);
            sk.points = sketch_obj.points;
            sk.constraints = sketch_obj.constraints;
            let (converged, iterations, max_residual) = match &result {
                Ok(r) => (r.converged, r.iterations, Some(r.max_residual)),
                Err(_) => (false, max_iterations as usize, None),
            };
            let pts: Vec<serde_json::Value> = sk
                .points
                .iter()
                .map(|p| serde_json::json!([p.x, p.y]))
                .collect();
            return Ok(serde_json::json!({
                "converged": converged,
                "iterations": iterations,
                "maxResidual": max_residual,
                "points": pts,
                "arcs": [],
            })
            .to_string());
        }

        // Full GcsSystem path (supports arcs + deferred constraints)
        let gcs = build_gcs_from_state(sk)?;
        let mut sys = gcs.sys;
        let result = sys.solve(max_iterations as usize, tolerance);
        let (converged, iterations, max_residual) = match &result {
            Ok(r) => (r.converged, r.iterations, Some(r.max_residual)),
            Err(_) => (false, max_iterations as usize, None),
        };

        // Write solved positions back
        for (i, pid) in gcs.point_ids.iter().enumerate() {
            if let Some(data) = sys.point(*pid) {
                sk.points[i].x = data.x;
                sk.points[i].y = data.y;
            }
        }

        let pts: Vec<serde_json::Value> = sk
            .points
            .iter()
            .map(|p| serde_json::json!([p.x, p.y]))
            .collect();
        let arcs: Vec<serde_json::Value> = sk
            .arcs
            .iter()
            .map(|(c, s, e)| serde_json::json!({"center": c, "start": s, "end": e}))
            .collect();
        let circles: Vec<serde_json::Value> = sk
            .circles
            .iter()
            .enumerate()
            .map(|(i, &(center, _))| {
                // Read solved radius from GCS
                let radius = gcs
                    .circle_ids
                    .get(i)
                    .and_then(|cid| sys.circle(*cid))
                    .map_or(0.0, |c| c.radius);
                serde_json::json!({"center": center, "radius": radius})
            })
            .collect();
        Ok(serde_json::json!({
            "converged": converged,
            "iterations": iterations,
            "maxResidual": max_residual,
            "points": pts,
            "arcs": arcs,
            "circles": circles,
        })
        .to_string())
    }

    /// Compute degrees of freedom for a sketch.
    ///
    /// Returns a JSON string: `{"dof": n, "rank": n, "numParams": n, "numEquations": n}`.
    #[wasm_bindgen(js_name = "sketchDof")]
    pub fn sketch_dof(&mut self, sketch: u32) -> Result<String, JsError> {
        let sk = self
            .sketches
            .get_mut(sketch as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "sketch",
                index: sketch as usize,
            })?;
        let GcsBuildResult { mut sys, .. } = build_gcs_from_state(sk)?;
        let dof = sys.dof();
        Ok(serde_json::json!({
            "dof": dof.dof,
            "rank": dof.rank,
            "numParams": dof.num_params,
            "numEquations": dof.num_equations,
        })
        .to_string())
    }
}
