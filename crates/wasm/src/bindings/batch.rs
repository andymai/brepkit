//! Batch execution and dispatch bindings.

#![allow(clippy::missing_errors_doc, clippy::too_many_lines)]

use wasm_bindgen::prelude::*;

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{self, BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure;
use brepkit_operations::revolve::revolve;
use brepkit_operations::sweep::sweep;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::edge::EdgeCurve;

use crate::error::WasmError;
use crate::handles::{
    compound_id_to_u32, edge_id_to_u32, face_id_to_u32, solid_id_to_u32, wire_id_to_u32,
};
use crate::helpers::{TOL, classify_to_string, get_f64, get_u32, panic_message, try_fillet};
use crate::kernel::BrepKernel;

#[wasm_bindgen]
impl BrepKernel {
    // ── Batch execution ──────────────────────────────────────────

    /// Execute a batch of operations, crossing the JS/WASM boundary once.
    ///
    /// Accepts a JSON string containing an array of operation objects:
    /// ```json
    /// [
    ///   {"op": "makeBox", "args": {"width": 2.0, "height": 2.0, "depth": 2.0}},
    ///   {"op": "fuse", "args": {"solidA": 0, "solidB": 1}},
    ///   {"op": "volume", "args": {"solid": 2, "deflection": 0.1}}
    /// ]
    /// ```
    ///
    /// Returns a JSON string with an array of results:
    /// ```json
    /// [
    ///   {"ok": 0},
    ///   {"ok": 2},
    ///   {"error": "invalid solid id"}
    /// ]
    /// ```
    ///
    /// Operations are executed sequentially; an error in one does not
    /// prevent execution of subsequent operations.
    #[wasm_bindgen(js_name = "executeBatch")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn execute_batch(&mut self, json: &str) -> String {
        let ops: Vec<serde_json::Value> = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(e) => {
                return serde_json::json!([{"error": format!("invalid JSON: {e}")}]).to_string();
            }
        };

        let results: Vec<serde_json::Value> = ops
            .iter()
            .map(|entry| {
                let op = match entry["op"].as_str() {
                    Some(s) => s,
                    None => return serde_json::json!({"error": "missing or invalid 'op' field"}),
                };
                let args = &entry["args"];
                match self.dispatch_op(op, args) {
                    Ok(val) => serde_json::json!({"ok": val}),
                    Err(msg) => serde_json::json!({"error": msg}),
                }
            })
            .collect();

        serde_json::Value::Array(results).to_string()
    }
}

impl BrepKernel {
    /// Extract a `NurbsCurve` from an edge, or error if it's a line.
    pub(crate) fn extract_nurbs_curve(&self, edge: u32) -> Result<NurbsCurve, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        match edge_data.curve() {
            EdgeCurve::NurbsCurve(c) => Ok(c.clone()),
            EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                Err(WasmError::InvalidInput {
                    reason: "edge is not a NURBS curve".into(),
                }
                .into())
            }
        }
    }

    /// Create an edge from a `NurbsCurve`, using its endpoints.
    pub(crate) fn nurbs_curve_to_edge(
        &mut self,
        points: &[Point3],
        curve: NurbsCurve,
    ) -> brepkit_topology::edge::EdgeId {
        let start = points[0];
        let end = points[points.len() - 1];
        brepkit_topology::builder::make_nurbs_edge(self.topo_mut(), start, end, curve, TOL)
    }

    /// Create an edge from a `NurbsCurve`, evaluating its endpoints.
    pub(crate) fn nurbs_curve_to_edge_from_curve(
        &mut self,
        curve: &NurbsCurve,
    ) -> brepkit_topology::edge::EdgeId {
        brepkit_topology::builder::make_nurbs_edge_from_curve(self.topo_mut(), curve, TOL)
    }

    /// Create a face from a `NurbsSurface` with a rectangular domain wire.
    pub(crate) fn nurbs_surface_to_face(
        &mut self,
        surface: NurbsSurface,
    ) -> Result<brepkit_topology::face::FaceId, JsError> {
        Ok(brepkit_topology::builder::make_nurbs_face(
            self.topo_mut(),
            surface,
            TOL,
        )?)
    }

    /// Dispatch a single batch operation by name.
    #[allow(clippy::too_many_lines)]
    fn dispatch_op(
        &mut self,
        op: &str,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        match op {
            "makeBox" => {
                let w = get_f64(args, "width")?;
                let h = get_f64(args, "height")?;
                let d = get_f64(args, "depth")?;
                let solid = brepkit_operations::primitives::make_box(self.topo_mut(), w, h, d)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "makeCylinder" => {
                let r = get_f64(args, "radius")?;
                let h = get_f64(args, "height")?;
                let solid = brepkit_operations::primitives::make_cylinder(self.topo_mut(), r, h)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "makeSphere" => {
                let r = get_f64(args, "radius")?;
                let segments = get_u32(args, "segments").unwrap_or(16);
                let solid = brepkit_operations::primitives::make_sphere(
                    self.topo_mut(),
                    r,
                    segments as usize,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "makeCone" => {
                let br = get_f64(args, "bottomRadius")?;
                let tr = get_f64(args, "topRadius")?;
                let h = get_f64(args, "height")?;
                let solid = brepkit_operations::primitives::make_cone(self.topo_mut(), br, tr, h)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "makeTorus" => {
                let major = get_f64(args, "majorRadius")?;
                let minor = get_f64(args, "minorRadius")?;
                let segments = get_u32(args, "segments").unwrap_or(16);
                let solid = brepkit_operations::primitives::make_torus(
                    self.topo_mut(),
                    major,
                    minor,
                    segments as usize,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "makeEllipsoid" => {
                let rx = get_f64(args, "rx")?;
                let ry = get_f64(args, "ry")?;
                let rz = get_f64(args, "rz")?;
                if rx <= 0.0 || ry <= 0.0 || rz <= 0.0 {
                    return Err("rx, ry, rz must be positive".to_string());
                }
                let solid = brepkit_operations::primitives::make_sphere(self.topo_mut(), 1.0, 16)
                    .map_err(|e| e.to_string())?;
                let mat = brepkit_math::mat::Mat4::scale(rx, ry, rz);
                transform_solid(self.topo_mut(), solid, &mat).map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "fuse" => {
                let a = get_u32(args, "solidA")?;
                let b = get_u32(args, "solidB")?;
                let a_id = self.resolve_solid(a).map_err(|e| e.to_string())?;
                let b_id = self.resolve_solid(b).map_err(|e| e.to_string())?;
                let result = boolean(self.topo_mut(), BooleanOp::Fuse, a_id, b_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "cut" => {
                let a = get_u32(args, "solidA")?;
                let b = get_u32(args, "solidB")?;
                let a_id = self.resolve_solid(a).map_err(|e| e.to_string())?;
                let b_id = self.resolve_solid(b).map_err(|e| e.to_string())?;
                let result = boolean(self.topo_mut(), BooleanOp::Cut, a_id, b_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "intersect" => {
                let a = get_u32(args, "solidA")?;
                let b = get_u32(args, "solidB")?;
                let a_id = self.resolve_solid(a).map_err(|e| e.to_string())?;
                let b_id = self.resolve_solid(b).map_err(|e| e.to_string())?;
                let result = boolean(self.topo_mut(), BooleanOp::Intersect, a_id, b_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "detectCoincidentFaces" => {
                let a = get_u32(args, "solidA")?;
                let b = get_u32(args, "solidB")?;
                let a_id = self.resolve_solid(a).map_err(|e| e.to_string())?;
                let b_id = self.resolve_solid(b).map_err(|e| e.to_string())?;
                let pairs = brepkit_algo::diagnostic::detect_coincident_faces(
                    self.topo(),
                    a_id,
                    b_id,
                    brepkit_math::tolerance::Tolerance::default(),
                )
                .map_err(|e| e.to_string())?;
                Ok(crate::bindings::booleans::coincident_face_pairs_to_json(
                    &pairs,
                ))
            }
            "compoundCut" => {
                let target = get_u32(args, "target")?;
                let target_id = self.resolve_solid(target).map_err(|e| e.to_string())?;
                let tool_arr = args["tools"]
                    .as_array()
                    .ok_or("missing or invalid 'tools' array")?;
                let tools: Vec<brepkit_topology::solid::SolidId> = tool_arr
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let h = v
                            .as_u64()
                            .ok_or_else(|| format!("tools[{i}] is not a number"))
                            .map(|n| n as u32)?;
                        self.resolve_solid(h).map_err(|e| e.to_string())
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let result = boolean::compound_cut(
                    self.topo_mut(),
                    target_id,
                    &tools,
                    boolean::BooleanOptions::default(),
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "transform" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let matrix = args["matrix"]
                    .as_array()
                    .ok_or("missing or invalid 'matrix'")?;
                if matrix.len() != 16 {
                    return Err(format!(
                        "matrix must have 16 elements, got {}",
                        matrix.len()
                    ));
                }
                let elems: Vec<f64> = matrix
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        v.as_f64()
                            .ok_or_else(|| format!("matrix[{i}] is not a number"))
                    })
                    .collect::<Result<_, _>>()?;
                let rows = std::array::from_fn(|i| std::array::from_fn(|j| elems[i * 4 + j]));
                let mat = Mat4(rows);
                transform_solid(self.topo_mut(), solid_id, &mat).map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid_id)))
            }
            "volume" => {
                let s = get_u32(args, "solid")?;
                let deflection = get_f64(args, "deflection").unwrap_or(0.1);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let v = measure::solid_volume(&self.topo, solid_id, deflection)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(v))
            }
            "surfaceArea" => {
                let s = get_u32(args, "solid")?;
                let deflection = get_f64(args, "deflection").unwrap_or(0.1);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let a = measure::solid_surface_area(&self.topo, solid_id, deflection)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(a))
            }
            "boundingBox" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let aabb =
                    measure::solid_bounding_box(&self.topo, solid_id).map_err(|e| e.to_string())?;
                Ok(serde_json::json!([
                    aabb.min.x(),
                    aabb.min.y(),
                    aabb.min.z(),
                    aabb.max.x(),
                    aabb.max.y(),
                    aabb.max.z()
                ]))
            }
            "centerOfMass" => {
                let s = get_u32(args, "solid")?;
                let deflection = get_f64(args, "deflection").unwrap_or(0.1);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let com = measure::solid_center_of_mass(&self.topo, solid_id, deflection)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!([com.x(), com.y(), com.z()]))
            }
            "solidEdges" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let edges = brepkit_topology::explorer::solid_edges(&self.topo, solid_id)
                    .map_err(|e| e.to_string())?;
                let handles: Vec<u32> = edges.iter().map(|&e| edge_id_to_u32(e)).collect();
                Ok(serde_json::json!(handles))
            }
            "solidToSolidDistance" => {
                let a = get_u32(args, "solidA")?;
                let b = get_u32(args, "solidB")?;
                let a_id = self.resolve_solid(a).map_err(|e| e.to_string())?;
                let b_id = self.resolve_solid(b).map_err(|e| e.to_string())?;
                let result =
                    brepkit_operations::distance::solid_to_solid_distance(&self.topo, a_id, b_id)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!([
                    result.distance,
                    result.point_a.x(),
                    result.point_a.y(),
                    result.point_a.z(),
                    result.point_b.x(),
                    result.point_b.y(),
                    result.point_b.z(),
                ]))
            }
            "copySolid" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let copy = brepkit_operations::copy::copy_solid(self.topo_mut(), solid_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(copy)))
            }
            "copyAndTransformSolid" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let matrix = args["matrix"]
                    .as_array()
                    .ok_or("missing or invalid 'matrix'")?;
                if matrix.len() != 16 {
                    return Err(format!(
                        "matrix must have 16 elements, got {}",
                        matrix.len()
                    ));
                }
                let elems: Vec<f64> = matrix
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        v.as_f64()
                            .ok_or_else(|| format!("matrix[{i}] is not a number"))
                    })
                    .collect::<Result<_, _>>()?;
                let rows = std::array::from_fn(|i| std::array::from_fn(|j| elems[i * 4 + j]));
                let mat = Mat4(rows);
                let copy = brepkit_operations::copy::copy_and_transform_solid(
                    self.topo_mut(),
                    solid_id,
                    &mat,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(copy)))
            }
            // ── Batch 8: new batch-dispatched operations ──────────────
            "extrude" => {
                let f = get_u32(args, "face")?;
                let dx = get_f64(args, "dx").unwrap_or(0.0);
                let dy = get_f64(args, "dy").unwrap_or(0.0);
                let dz = get_f64(args, "dz").unwrap_or(1.0);
                let dist = get_f64(args, "distance").unwrap_or(1.0);
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let dir = Vec3::new(dx, dy, dz);
                let solid =
                    extrude(self.topo_mut(), face_id, dir, dist).map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "revolve" => {
                let f = get_u32(args, "face")?;
                let angle_degrees = get_f64(args, "angle")?;
                let ox = get_f64(args, "originX").unwrap_or(0.0);
                let oy = get_f64(args, "originY").unwrap_or(0.0);
                let oz = get_f64(args, "originZ").unwrap_or(0.0);
                let ax = get_f64(args, "axisX").unwrap_or(0.0);
                let ay = get_f64(args, "axisY").unwrap_or(0.0);
                let az = get_f64(args, "axisZ").unwrap_or(1.0);
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                // Convert degrees to radians to match the direct WASM binding.
                let solid = revolve(
                    self.topo_mut(),
                    face_id,
                    Point3::new(ox, oy, oz),
                    Vec3::new(ax, ay, az),
                    angle_degrees.to_radians(),
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "sweep" => {
                let f = get_u32(args, "face")?;
                let e = get_u32(args, "pathEdge")?;
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let edge_id = self.resolve_edge(e).map_err(|e| e.to_string())?;
                let edge_data = self.topo.edge(edge_id).map_err(|e| e.to_string())?;
                let curve = match edge_data.curve() {
                    EdgeCurve::NurbsCurve(c) => c.clone(),
                    EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                        return Err("sweep path must be a NURBS edge".into());
                    }
                };
                let solid = sweep(self.topo_mut(), face_id, &curve).map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "chamfer" => {
                let s = get_u32(args, "solid")?;
                let dist = get_f64(args, "distance")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let edge_handles: Vec<u32> = args["edges"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let edge_ids: Vec<_> = edge_handles
                    .iter()
                    .map(|&h| self.resolve_edge(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = brepkit_operations::chamfer::chamfer(
                    self.topo_mut(),
                    solid_id,
                    &edge_ids,
                    dist,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "fillet" => {
                let s = get_u32(args, "solid")?;
                let radius = get_f64(args, "radius")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let edge_handles: Vec<u32> = args["edges"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let edge_ids: Vec<_> = edge_handles
                    .iter()
                    .map(|&h| self.resolve_edge(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let fillet_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    try_fillet(self.topo_mut(), solid_id, &edge_ids, radius)
                }));
                let result = match fillet_result {
                    Ok(inner) => inner.map_err(|e| e.to_string())?,
                    Err(panic_info) => {
                        return Err(panic_message(&panic_info, "Fillet"));
                    }
                };
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "filletVariable" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let specs = args["specs"]
                    .as_array()
                    .ok_or_else(|| "missing 'specs' array".to_string())?;
                let mut edge_laws = Vec::with_capacity(specs.len());
                for spec in specs {
                    let edge_handle = spec["edge"]
                        .as_u64()
                        .ok_or_else(|| "missing 'edge' in fillet spec".to_string())?
                        as u32;
                    let edge_id = self.resolve_edge(edge_handle).map_err(|e| e.to_string())?;
                    let start_val = spec["start"]
                        .as_f64()
                        .or_else(|| spec["startRadius"].as_f64());
                    let end_val = spec["end"].as_f64().or_else(|| spec["endRadius"].as_f64());
                    let law_str =
                        spec["law"]
                            .as_str()
                            .unwrap_or_else(|| match (start_val, end_val) {
                                (Some(sv), Some(ev)) if (sv - ev).abs() > f64::EPSILON => "linear",
                                _ => "constant",
                            });
                    let law = match law_str {
                        "linear" => brepkit_operations::fillet::FilletRadiusLaw::Linear {
                            start: start_val.unwrap_or(1.0),
                            end: end_val.unwrap_or(1.0),
                        },
                        "scurve" => brepkit_operations::fillet::FilletRadiusLaw::SCurve {
                            start: start_val.unwrap_or(1.0),
                            end: end_val.unwrap_or(1.0),
                        },
                        _ => {
                            let r = spec["radius"].as_f64().or(start_val).unwrap_or(1.0);
                            brepkit_operations::fillet::FilletRadiusLaw::Constant(r)
                        }
                    };
                    edge_laws.push((edge_id, law));
                }
                let result = brepkit_operations::fillet::fillet_variable(
                    self.topo_mut(),
                    solid_id,
                    &edge_laws,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "filletV2" => {
                let s = get_u32(args, "solid")?;
                let radius = get_f64(args, "radius")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let edge_handles: Vec<u32> = args["edges"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let edge_ids: Vec<_> = edge_handles
                    .iter()
                    .map(|&h| self.resolve_edge(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = brepkit_operations::blend_ops::fillet_v2(
                    self.topo_mut(),
                    solid_id,
                    &edge_ids,
                    radius,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result.solid)))
            }
            "chamferV2" => {
                let s = get_u32(args, "solid")?;
                let d1 = get_f64(args, "d1")?;
                let d2 = get_f64(args, "d2")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let edge_handles: Vec<u32> = args["edges"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let edge_ids: Vec<_> = edge_handles
                    .iter()
                    .map(|&h| self.resolve_edge(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = brepkit_operations::blend_ops::chamfer_v2(
                    self.topo_mut(),
                    solid_id,
                    &edge_ids,
                    d1,
                    d2,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result.solid)))
            }
            "chamferDistanceAngle" => {
                let s = get_u32(args, "solid")?;
                let distance = get_f64(args, "distance")?;
                let angle = get_f64(args, "angle")?;
                if angle >= std::f64::consts::FRAC_PI_2 {
                    return Err("angle must be less than π/2".into());
                }
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let edge_handles: Vec<u32> = args["edges"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let edge_ids: Vec<_> = edge_handles
                    .iter()
                    .map(|&h| self.resolve_edge(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = brepkit_operations::blend_ops::chamfer_distance_angle(
                    self.topo_mut(),
                    solid_id,
                    &edge_ids,
                    distance,
                    angle,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result.solid)))
            }
            "shell" => {
                let s = get_u32(args, "solid")?;
                let thickness = get_f64(args, "thickness")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let face_handles: Vec<u32> = args["faces"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let face_ids: Vec<_> = face_handles
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = brepkit_operations::shell_op::shell(
                    self.topo_mut(),
                    solid_id,
                    thickness,
                    &face_ids,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "mirror" => {
                let s = get_u32(args, "solid")?;
                let px = get_f64(args, "px").unwrap_or(0.0);
                let py = get_f64(args, "py").unwrap_or(0.0);
                let pz = get_f64(args, "pz").unwrap_or(0.0);
                let nx = get_f64(args, "nx").unwrap_or(1.0);
                let ny = get_f64(args, "ny").unwrap_or(0.0);
                let nz = get_f64(args, "nz").unwrap_or(0.0);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let result = brepkit_operations::mirror::mirror(
                    self.topo_mut(),
                    solid_id,
                    Point3::new(px, py, pz),
                    Vec3::new(nx, ny, nz),
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "unifyFaces" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                brepkit_operations::heal::unify_faces(self.topo_mut(), solid_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid_id)))
            }
            "healSolid" => {
                let s = get_u32(args, "solid")?;
                let tol = get_f64(args, "tolerance").unwrap_or(1e-7);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                brepkit_operations::heal::heal_solid(self.topo_mut(), solid_id, tol)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid_id)))
            }
            "repairSolid" => {
                let s = get_u32(args, "solid")?;
                let tol = get_f64(args, "tolerance").unwrap_or(1e-7);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let report = brepkit_operations::heal::repair_solid(self.topo_mut(), solid_id, tol)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "solid": solid_id_to_u32(solid_id),
                    "errorsBefore": report.before.error_count(),
                    "errorsAfter": report.after.error_count(),
                    "totalRepairs": report.total_repairs(),
                }))
            }
            "classifyPoint" => {
                let s = get_u32(args, "solid")?;
                let x = get_f64(args, "x")?;
                let y = get_f64(args, "y")?;
                let z = get_f64(args, "z")?;
                let tol = get_f64(args, "tolerance").unwrap_or(1e-7);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let pt = Point3::new(x, y, z);
                let result = brepkit_operations::classify::classify_point(
                    &self.topo, solid_id, pt, 0.1, tol,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(classify_to_string(result)))
            }
            "loft" => {
                let face_handles: Vec<u32> = args["faces"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let face_ids: Vec<_> = face_handles
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = brepkit_operations::loft::loft(self.topo_mut(), &face_ids)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "loftSmooth" => {
                let face_handles: Vec<u32> = args["faces"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let face_ids: Vec<_> = face_handles
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = brepkit_operations::loft::loft_smooth(self.topo_mut(), &face_ids)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "circularPattern" => {
                let s = get_u32(args, "solid")?;
                let ax = get_f64(args, "ax").unwrap_or(0.0);
                let ay = get_f64(args, "ay").unwrap_or(0.0);
                let az = get_f64(args, "az").unwrap_or(1.0);
                let count = get_u32(args, "count")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let axis = Vec3::new(ax, ay, az);
                let compound = brepkit_operations::pattern::circular_pattern(
                    self.topo_mut(),
                    solid_id,
                    axis,
                    count as usize,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(compound_id_to_u32(compound)))
            }
            "gridPattern" => {
                let s = get_u32(args, "solid")?;
                let dxx = get_f64(args, "dirXx").unwrap_or(1.0);
                let dxy = get_f64(args, "dirXy").unwrap_or(0.0);
                let dxz = get_f64(args, "dirXz").unwrap_or(0.0);
                let dyx = get_f64(args, "dirYx").unwrap_or(0.0);
                let dyy = get_f64(args, "dirYy").unwrap_or(1.0);
                let dyz = get_f64(args, "dirYz").unwrap_or(0.0);
                let sx = get_f64(args, "spacingX")?;
                let sy = get_f64(args, "spacingY")?;
                let cx = get_u32(args, "countX")?;
                let cy = get_u32(args, "countY")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let compound = brepkit_operations::pattern::grid_pattern(
                    self.topo_mut(),
                    solid_id,
                    Vec3::new(dxx, dxy, dxz),
                    Vec3::new(dyx, dyy, dyz),
                    sx,
                    sy,
                    cx as usize,
                    cy as usize,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(compound_id_to_u32(compound)))
            }
            "defeature" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let face_handles: Vec<u32> = args["faces"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let face_ids: Vec<_> = face_handles
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result =
                    brepkit_operations::defeature::defeature(self.topo_mut(), solid_id, &face_ids)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "copyWire" => {
                let w = get_u32(args, "wire")?;
                let wire_id = self.resolve_wire(w).map_err(|e| e.to_string())?;
                let copy = brepkit_operations::copy::copy_wire(self.topo_mut(), wire_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(wire_id_to_u32(copy)))
            }
            "transformWire" => {
                let w = get_u32(args, "wire")?;
                let wire_id = self.resolve_wire(w).map_err(|e| e.to_string())?;
                let matrix = args["matrix"]
                    .as_array()
                    .ok_or("missing or invalid 'matrix'")?;
                if matrix.len() != 16 {
                    return Err(format!(
                        "matrix must have 16 elements, got {}",
                        matrix.len()
                    ));
                }
                let elems: Vec<f64> = matrix
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        v.as_f64()
                            .ok_or_else(|| format!("matrix[{i}] is not a number"))
                    })
                    .collect::<Result<_, _>>()?;
                if let Some(pos) = elems.iter().position(|v| !v.is_finite()) {
                    return Err(format!("matrix element at index {pos} is not finite"));
                }
                let rows = std::array::from_fn(|i| std::array::from_fn(|j| elems[i * 4 + j]));
                let mat = Mat4(rows);
                brepkit_operations::transform::transform_wire(self.topo_mut(), wire_id, &mat)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(null))
            }
            "transformFace" => {
                let f = get_u32(args, "face")?;
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let matrix = args["matrix"]
                    .as_array()
                    .ok_or("missing or invalid 'matrix'")?;
                if matrix.len() != 16 {
                    return Err(format!(
                        "matrix must have 16 elements, got {}",
                        matrix.len()
                    ));
                }
                let elems: Vec<f64> = matrix
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        v.as_f64()
                            .ok_or_else(|| format!("matrix element {i} is not a number"))
                    })
                    .collect::<Result<_, _>>()?;
                if let Some(pos) = elems.iter().position(|v| !v.is_finite()) {
                    return Err(format!("matrix element at index {pos} is not finite"));
                }
                let rows = std::array::from_fn(|i| std::array::from_fn(|j| elems[i * 4 + j]));
                let mat = Mat4(rows);
                brepkit_operations::transform::transform_face(self.topo_mut(), face_id, &mat)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(null))
            }
            "offsetFace" => {
                let f = get_u32(args, "face")?;
                let dist = get_f64(args, "distance")?;
                let samples = get_u32(args, "samples").unwrap_or(16);
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let result = brepkit_operations::offset_face::offset_face(
                    self.topo_mut(),
                    face_id,
                    dist,
                    samples as usize,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(face_id_to_u32(result)))
            }
            "offsetSolid" => {
                let s = get_u32(args, "solid")?;
                let dist = get_f64(args, "distance")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                #[allow(deprecated)]
                let result =
                    brepkit_operations::offset_solid::offset_solid(self.topo_mut(), solid_id, dist)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "offsetSolidV2" => {
                let s = get_u32(args, "solid")?;
                let dist = get_f64(args, "distance")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let result =
                    brepkit_operations::offset_v2::offset_solid_v2(self.topo_mut(), solid_id, dist)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "section" => {
                let s = get_u32(args, "solid")?;
                let px = get_f64(args, "px").unwrap_or(0.0);
                let py = get_f64(args, "py").unwrap_or(0.0);
                let pz = get_f64(args, "pz").unwrap_or(0.0);
                let nx = get_f64(args, "nx").unwrap_or(0.0);
                let ny = get_f64(args, "ny").unwrap_or(0.0);
                let nz = get_f64(args, "nz").unwrap_or(1.0);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let result = brepkit_operations::section::section(
                    self.topo_mut(),
                    solid_id,
                    Point3::new(px, py, pz),
                    Vec3::new(nx, ny, nz),
                )
                .map_err(|e| e.to_string())?;
                let face_ids: Vec<u32> = result.faces.iter().map(|&f| face_id_to_u32(f)).collect();
                Ok(serde_json::json!(face_ids))
            }
            "split" => {
                let s = get_u32(args, "solid")?;
                let px = get_f64(args, "px").unwrap_or(0.0);
                let py = get_f64(args, "py").unwrap_or(0.0);
                let pz = get_f64(args, "pz").unwrap_or(0.0);
                let nx = get_f64(args, "nx").unwrap_or(0.0);
                let ny = get_f64(args, "ny").unwrap_or(0.0);
                let nz = get_f64(args, "nz").unwrap_or(1.0);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let result = brepkit_operations::split::split(
                    self.topo_mut(),
                    solid_id,
                    Point3::new(px, py, pz),
                    Vec3::new(nx, ny, nz),
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "positive": solid_id_to_u32(result.positive),
                    "negative": solid_id_to_u32(result.negative),
                }))
            }
            "sewFaces" => {
                let face_handles: Vec<u32> = args["faces"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let tol = get_f64(args, "tolerance").unwrap_or(1e-6);
                let face_ids: Vec<_> = face_handles
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let solid = brepkit_operations::sew::sew_faces(self.topo_mut(), &face_ids, tol)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "thicken" => {
                let f = get_u32(args, "face")?;
                let thickness = get_f64(args, "thickness")?;
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let result =
                    brepkit_operations::thicken::thicken(self.topo_mut(), face_id, thickness)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "pipe" => {
                let f = get_u32(args, "face")?;
                let e = get_u32(args, "pathEdge")?;
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let edge_id = self.resolve_edge(e).map_err(|e| e.to_string())?;
                let edge_data = self.topo.edge(edge_id).map_err(|e| e.to_string())?;
                let curve = match edge_data.curve() {
                    EdgeCurve::NurbsCurve(c) => c.clone(),
                    EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                        return Err("pipe path must be a NURBS edge".into());
                    }
                };
                let solid = brepkit_operations::pipe::pipe(self.topo_mut(), face_id, &curve, None)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "linearPattern" => {
                let s = get_u32(args, "solid")?;
                let dx = get_f64(args, "dx").unwrap_or(1.0);
                let dy = get_f64(args, "dy").unwrap_or(0.0);
                let dz = get_f64(args, "dz").unwrap_or(0.0);
                let spacing = get_f64(args, "spacing")?;
                let count = get_u32(args, "count")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let compound = brepkit_operations::pattern::linear_pattern(
                    self.topo_mut(),
                    solid_id,
                    Vec3::new(dx, dy, dz),
                    spacing,
                    count as usize,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(compound_id_to_u32(compound)))
            }
            "draft" => {
                let s = get_u32(args, "solid")?;
                let angle = get_f64(args, "angle")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let face_handles: Vec<u32> = args["faces"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let face_ids: Vec<_> = face_handles
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let dx = get_f64(args, "dirX").unwrap_or(0.0);
                let dy = get_f64(args, "dirY").unwrap_or(0.0);
                let dz = get_f64(args, "dirZ").unwrap_or(1.0);
                let npx = get_f64(args, "neutralX").unwrap_or(0.0);
                let npy = get_f64(args, "neutralY").unwrap_or(0.0);
                let npz = get_f64(args, "neutralZ").unwrap_or(0.0);
                let dir = Vec3::new(dx, dy, dz);
                let neutral = Point3::new(npx, npy, npz);
                let result = brepkit_operations::draft::draft(
                    self.topo_mut(),
                    solid_id,
                    &face_ids,
                    dir,
                    neutral,
                    angle,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "makeTangentArc3d" => {
                let sx = get_f64(args, "startX")?;
                let sy = get_f64(args, "startY")?;
                let sz = get_f64(args, "startZ")?;
                let tx = get_f64(args, "tangentX")?;
                let ty = get_f64(args, "tangentY")?;
                let tz = get_f64(args, "tangentZ")?;
                let ex = get_f64(args, "endX")?;
                let ey = get_f64(args, "endY")?;
                let ez = get_f64(args, "endZ")?;
                let eid = self
                    .make_tangent_arc_3d_impl(sx, sy, sz, tx, ty, tz, ex, ey, ez)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(eid))
            }
            "liftCurve2dToPlane" => {
                let ct = get_u32(args, "curveType")?;
                let params_arr = args["curveParams"]
                    .as_array()
                    .ok_or("missing or invalid 'curveParams'")?;
                let cp: Vec<f64> = params_arr
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        v.as_f64()
                            .ok_or_else(|| format!("curveParams[{i}] is not a number"))
                    })
                    .collect::<Result<_, _>>()?;
                let ox = get_f64(args, "originX")?;
                let oy = get_f64(args, "originY")?;
                let oz = get_f64(args, "originZ")?;
                let xx = get_f64(args, "xAxisX")?;
                let xy = get_f64(args, "xAxisY")?;
                let xz = get_f64(args, "xAxisZ")?;
                let nx = get_f64(args, "normalX")?;
                let ny = get_f64(args, "normalY")?;
                let nz = get_f64(args, "normalZ")?;
                let t0 = get_f64(args, "tStart")?;
                let t1 = get_f64(args, "tEnd")?;
                let eid = self
                    .lift_curve2d_to_plane_impl(ct, cp, ox, oy, oz, xx, xy, xz, nx, ny, nz, t0, t1)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(eid))
            }
            "offsetWire" => {
                let f = get_u32(args, "face")?;
                let dist = get_f64(args, "distance")?;
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let wire_id =
                    brepkit_operations::offset_wire::offset_wire(self.topo_mut(), face_id, dist)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(wire_id_to_u32(wire_id)))
            }
            "offsetWireWithJoinType" => {
                let f = get_u32(args, "face")?;
                let dist = get_f64(args, "distance")?;
                let jt_str = args["joinType"]
                    .as_str()
                    .ok_or("missing or invalid 'joinType' string")?;
                let jt =
                    super::operations::parse_join_type_str(jt_str).map_err(|e| e.to_string())?;
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let wire_id = brepkit_operations::offset_wire::offset_wire_with_join(
                    self.topo_mut(),
                    face_id,
                    dist,
                    jt,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(wire_id_to_u32(wire_id)))
            }
            _ => Err(format!("unknown operation: {op}")),
        }
    }
}
