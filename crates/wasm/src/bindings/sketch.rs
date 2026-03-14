//! 2D sketch constraint solver bindings.

#![allow(clippy::missing_errors_doc)]

use wasm_bindgen::prelude::*;

use crate::error::WasmError;
use crate::helpers::parse_sketch_constraint;
use crate::kernel::BrepKernel;
use crate::state::SketchState;

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

    /// Add a constraint to a sketch from a JSON string.
    ///
    /// Formats: `{"type":"coincident","p1":0,"p2":1}`,
    /// `{"type":"distance","p1":0,"p2":1,"value":5.0}`,
    /// `{"type":"fixX","point":0,"value":1.0}`, etc.
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
        let constraint = parse_sketch_constraint(&val)?;
        sk.constraints.push(constraint);
        Ok(())
    }

    /// Solve the sketch constraints.
    ///
    /// Returns a JSON string: `{"converged": bool, "iterations": n, "maxResidual": f, "points": [[x,y], ...]}`.
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
        let mut sketch_obj = brepkit_operations::sketch::Sketch {
            points: std::mem::take(&mut sk.points),
            constraints: std::mem::take(&mut sk.constraints),
        };
        let result = sketch_obj.solve(max_iterations as usize, tolerance);
        // Store back the updated points
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
        Ok(serde_json::json!({
            "converged": converged,
            "iterations": iterations,
            "maxResidual": max_residual,
            "points": pts,
        })
        .to_string())
    }

    /// Compute degrees of freedom for a sketch.
    ///
    /// Returns a JSON string: `{"dof": n, "rank": n, "numParams": n, "numEquations": n}`.
    #[wasm_bindgen(js_name = "sketchDof")]
    pub fn sketch_dof(&mut self, sketch: u32) -> Result<String, JsError> {
        use brepkit_operations::sketch::GcsConstraint;
        let sk = self
            .sketches
            .get_mut(sketch as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "sketch",
                index: sketch as usize,
            })?;
        // Build a temporary GcsSystem from the legacy data
        let mut sys = brepkit_operations::sketch::GcsSystem::new();
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

        // We need lines for line-based constraints, create them implicitly
        let mut line_cache: std::collections::HashMap<
            (usize, usize),
            brepkit_operations::sketch::LineId,
        > = std::collections::HashMap::new();

        // Helper: get or create an implicit line for a point pair
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

        // Convert constraints from legacy enum to GcsConstraint
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
