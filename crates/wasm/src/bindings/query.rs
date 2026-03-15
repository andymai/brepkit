//! Topology query, edge/surface evaluation, and BREP introspection bindings.

#![allow(clippy::missing_errors_doc, clippy::too_many_lines)]

use std::f64::consts::PI;

use wasm_bindgen::prelude::*;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{Face, FaceSurface};

use crate::error::{WasmError, validate_finite};
use crate::handles::{
    edge_id_to_u32, face_id_to_u32, solid_id_to_u32, vertex_id_to_u32, wire_id_to_u32,
};
use crate::helpers::{
    detect_nurbs_curve_type, detect_nurbs_surface_type, sample_full_period_curve,
};
use crate::kernel::BrepKernel;

#[wasm_bindgen]
impl BrepKernel {
    // ── Topology queries ──────────────────────────────────────────

    /// Get all face handles of a solid.
    ///
    /// Returns an array of face handles (`u32[]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "getSolidFaces")]
    pub fn get_solid_faces(&self, solid: u32) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let faces = brepkit_topology::explorer::solid_faces(&self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(faces.iter().map(|f| f.index() as u32).collect())
    }

    /// Get all edge handles of a solid.
    ///
    /// Returns an array of unique edge handles (`u32[]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "getSolidEdges")]
    pub fn get_solid_edges(&self, solid: u32) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let edges = brepkit_topology::explorer::solid_edges(&self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(edges.iter().map(|e| e.index() as u32).collect())
    }

    /// Get all vertex handles of a solid.
    ///
    /// Returns an array of unique vertex handles (`u32[]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "getSolidVertices")]
    pub fn get_solid_vertices(&self, solid: u32) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let verts = brepkit_topology::explorer::solid_vertices(&self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(verts.iter().map(|v| v.index() as u32).collect())
    }

    /// Get the vertex positions of an edge.
    ///
    /// Returns `[start_x, start_y, start_z, end_x, end_y, end_z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge handle is invalid.
    #[wasm_bindgen(js_name = "getEdgeVertices")]
    pub fn get_edge_vertices(&self, edge: u32) -> Result<Vec<f64>, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        let start = self.topo.vertex(edge_data.start())?.point();
        let end = self.topo.vertex(edge_data.end())?.point();
        Ok(vec![
            start.x(),
            start.y(),
            start.z(),
            end.x(),
            end.y(),
            end.z(),
        ])
    }

    /// Get the vertex *handles* (not positions) of an edge.
    ///
    /// Returns `[start_vertex_handle, end_vertex_handle]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge handle is invalid.
    #[wasm_bindgen(js_name = "getEdgeVertexHandles")]
    pub fn get_edge_vertex_handles(&self, edge: u32) -> Result<Vec<u32>, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        Ok(vec![
            vertex_id_to_u32(edge_data.start()),
            vertex_id_to_u32(edge_data.end()),
        ])
    }

    /// Get the position of a vertex.
    ///
    /// Returns `[x, y, z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the vertex handle is invalid.
    #[wasm_bindgen(js_name = "getVertexPosition")]
    pub fn get_vertex_position(&self, vertex: u32) -> Result<Vec<f64>, JsError> {
        let vertex_id = self.resolve_vertex(vertex)?;
        let point = self.topo.vertex(vertex_id)?.point();
        Ok(vec![point.x(), point.y(), point.z()])
    }

    /// Serialize a solid's B-Rep topology to JSON.
    ///
    /// Returns a JSON string containing the solid's complete topology:
    /// vertices, edges (with curve types), faces (with surface types), and
    /// connectivity information.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "toBREP")]
    #[allow(clippy::too_many_lines)]
    pub fn to_brep(&self, solid: u32) -> Result<JsValue, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let faces = brepkit_topology::explorer::solid_faces(&self.topo, solid_id)?;
        let edges = brepkit_topology::explorer::solid_edges(&self.topo, solid_id)?;
        let verts = brepkit_topology::explorer::solid_vertices(&self.topo, solid_id)?;

        let vert_json: Vec<serde_json::Value> = verts
            .iter()
            .map(|&vid| -> Result<serde_json::Value, JsError> {
                let v = self.topo.vertex(vid)?;
                let p = v.point();
                Ok(serde_json::json!({
                    "id": vertex_id_to_u32(vid),
                    "position": [p.x(), p.y(), p.z()],
                }))
            })
            .collect::<Result<_, _>>()?;

        let edge_json: Vec<serde_json::Value> = edges
            .iter()
            .map(|&eid| -> Result<serde_json::Value, JsError> {
                let e = self.topo.edge(eid)?;
                let curve_type = match e.curve() {
                    brepkit_topology::edge::EdgeCurve::Line => "line",
                    brepkit_topology::edge::EdgeCurve::Circle(_) => "circle",
                    brepkit_topology::edge::EdgeCurve::Ellipse(_) => "ellipse",
                    brepkit_topology::edge::EdgeCurve::NurbsCurve(_) => "nurbs",
                };
                let curve_params = match e.curve() {
                    EdgeCurve::Line => serde_json::json!(null),
                    EdgeCurve::Circle(c) => serde_json::json!({
                        "center": [c.center().x(), c.center().y(), c.center().z()],
                        "axis": [c.normal().x(), c.normal().y(), c.normal().z()],
                        "xAxis": [c.u_axis().x(), c.u_axis().y(), c.u_axis().z()],
                        "radius": c.radius(),
                    }),
                    EdgeCurve::Ellipse(el) => serde_json::json!({
                        "center": [el.center().x(), el.center().y(), el.center().z()],
                        "axis": [el.normal().x(), el.normal().y(), el.normal().z()],
                        "majorAxis": [el.u_axis().x(), el.u_axis().y(), el.u_axis().z()],
                        "majorRadius": el.semi_major(),
                        "minorRadius": el.semi_minor(),
                    }),
                    EdgeCurve::NurbsCurve(n) => serde_json::json!({
                        "degree": n.degree(),
                        "controlPoints": n.control_points().iter()
                            .map(|p| [p.x(), p.y(), p.z()])
                            .collect::<Vec<_>>(),
                        "weights": n.weights().to_vec(),
                        "knots": n.knots().to_vec(),
                    }),
                };
                Ok(serde_json::json!({
                    "id": edge_id_to_u32(eid),
                    "curveType": curve_type,
                    "curveParams": curve_params,
                    "startVertex": vertex_id_to_u32(e.start()),
                    "endVertex": vertex_id_to_u32(e.end()),
                }))
            })
            .collect::<Result<_, _>>()?;

        let face_json: Vec<serde_json::Value> = faces
            .iter()
            .map(|&fid| -> Result<serde_json::Value, JsError> {
                let f = self.topo.face(fid)?;
                let surface_type = match f.surface() {
                    brepkit_topology::face::FaceSurface::Plane { .. } => "plane",
                    brepkit_topology::face::FaceSurface::Nurbs(_) => "nurbs",
                    brepkit_topology::face::FaceSurface::Cylinder(_) => "cylinder",
                    brepkit_topology::face::FaceSurface::Cone(_) => "cone",
                    brepkit_topology::face::FaceSurface::Sphere(_) => "sphere",
                    brepkit_topology::face::FaceSurface::Torus(_) => "torus",
                };
                let surface_params = match f.surface() {
                    FaceSurface::Plane { normal, d } => serde_json::json!({
                        "normal": [normal.x(), normal.y(), normal.z()],
                        "d": d,
                    }),
                    FaceSurface::Cylinder(c) => serde_json::json!({
                        "origin": [c.origin().x(), c.origin().y(), c.origin().z()],
                        "axis": [c.axis().x(), c.axis().y(), c.axis().z()],
                        "refDir": [c.x_axis().x(), c.x_axis().y(), c.x_axis().z()],
                        "radius": c.radius(),
                    }),
                    FaceSurface::Cone(c) => serde_json::json!({
                        "apex": [c.apex().x(), c.apex().y(), c.apex().z()],
                        "axis": [c.axis().x(), c.axis().y(), c.axis().z()],
                        "refDir": [c.x_axis().x(), c.x_axis().y(), c.x_axis().z()],
                        "halfAngle": c.half_angle(),
                    }),
                    FaceSurface::Sphere(s) => serde_json::json!({
                        "center": [s.center().x(), s.center().y(), s.center().z()],
                        "axis": [s.z_axis().x(), s.z_axis().y(), s.z_axis().z()],
                        "radius": s.radius(),
                    }),
                    FaceSurface::Torus(t) => serde_json::json!({
                        "center": [t.center().x(), t.center().y(), t.center().z()],
                        "axis": [t.z_axis().x(), t.z_axis().y(), t.z_axis().z()],
                        "majorRadius": t.major_radius(),
                        "minorRadius": t.minor_radius(),
                    }),
                    FaceSurface::Nurbs(n) => serde_json::json!({
                        "degreeU": n.degree_u(),
                        "degreeV": n.degree_v(),
                        "controlPoints": n.control_points().iter()
                            .map(|row| row.iter()
                                .map(|p| [p.x(), p.y(), p.z()])
                                .collect::<Vec<_>>())
                            .collect::<Vec<_>>(),
                        "weights": n.weights().to_vec(),
                        "knotsU": n.knots_u().to_vec(),
                        "knotsV": n.knots_v().to_vec(),
                    }),
                };
                let outer_wire = self.topo.wire(f.outer_wire())?;
                let outer_edges: Vec<u32> = outer_wire
                    .edges()
                    .iter()
                    .map(|e| edge_id_to_u32(e.edge()))
                    .collect();
                let outer_edge_orientations: Vec<bool> = outer_wire
                    .edges()
                    .iter()
                    .map(brepkit_topology::wire::OrientedEdge::is_forward)
                    .collect();
                let inner_wires: Vec<serde_json::Value> = f
                    .inner_wires()
                    .iter()
                    .filter_map(|&wid| {
                        self.topo.wire(wid).ok().map(|w| {
                            let edges: Vec<u32> =
                                w.edges().iter().map(|e| edge_id_to_u32(e.edge())).collect();
                            let orientations: Vec<bool> = w
                                .edges()
                                .iter()
                                .map(brepkit_topology::wire::OrientedEdge::is_forward)
                                .collect();
                            serde_json::json!({
                                "edges": edges,
                                "orientations": orientations,
                            })
                        })
                    })
                    .collect();
                Ok(serde_json::json!({
                    "id": face_id_to_u32(fid),
                    "surfaceType": surface_type,
                    "surfaceParams": surface_params,
                    "reversed": f.is_reversed(),
                    "outerWireEdges": outer_edges,
                    "outerWireOrientations": outer_edge_orientations,
                    "innerWires": inner_wires,
                }))
            })
            .collect::<Result<_, _>>()?;

        Ok(serde_json::to_string(&serde_json::json!({
            "type": "solid",
            "solidId": solid_id_to_u32(solid_id),
            "vertices": vert_json,
            "edges": edge_json,
            "faces": face_json,
        }))
        .map_err(|e| JsError::new(&e.to_string()))?
        .into())
    }

    /// Reconstruct a solid from a `toBREP` JSON string.
    ///
    /// Currently supports planar faces with line edges. Non-line edges are
    /// approximated as lines (straight segments between start/end vertices).
    /// Non-planar surfaces are approximated as planes computed from the wire
    /// vertex positions.
    ///
    /// Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is invalid or reconstruction fails.
    #[wasm_bindgen(js_name = "fromBREP")]
    #[allow(clippy::wrong_self_convention)]
    pub fn from_brep(&mut self, json: &str) -> Result<u32, JsError> {
        Ok(self.from_brep_impl(json)?)
    }

    /// Get the face normal of a planar face.
    ///
    /// Returns `[nx, ny, nz]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the face is invalid or NURBS.
    #[wasm_bindgen(js_name = "getFaceNormal")]
    pub fn get_face_normal(&self, face: u32) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        match face_data.surface() {
            brepkit_topology::face::FaceSurface::Plane { normal, .. } => {
                Ok(vec![normal.x(), normal.y(), normal.z()])
            }
            _ => Err(WasmError::InvalidInput {
                reason: "getFaceNormal only works on planar faces".into(),
            }
            .into()),
        }
    }

    /// Get entity counts of a solid: `[faces, edges, vertices]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "getEntityCounts")]
    pub fn get_entity_counts(&self, solid: u32) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let (f, e, v) = brepkit_topology::explorer::solid_entity_counts(&self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(vec![f as u32, e as u32, v as u32])
    }

    // ── Topology queries (extended) ──────────────────────────────

    /// Get the edge handles of a face.
    ///
    /// Returns an array of edge handles (`u32[]`).
    #[wasm_bindgen(js_name = "getFaceEdges")]
    pub fn get_face_edges(&self, face: u32) -> Result<Vec<u32>, JsError> {
        let face_id = self.resolve_face(face)?;
        let edges = brepkit_topology::explorer::face_edges(&self.topo, face_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(edges.iter().map(|e| e.index() as u32).collect())
    }

    /// Get the vertex handles of a face.
    ///
    /// Returns an array of vertex handles (`u32[]`).
    #[wasm_bindgen(js_name = "getFaceVertices")]
    pub fn get_face_vertices(&self, face: u32) -> Result<Vec<u32>, JsError> {
        let face_id = self.resolve_face(face)?;
        let verts = brepkit_topology::explorer::face_vertices(&self.topo, face_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(verts.iter().map(|v| v.index() as u32).collect())
    }

    /// Get the outer wire handle of a face.
    ///
    /// Returns a wire handle (`u32`).
    #[wasm_bindgen(js_name = "getFaceOuterWire")]
    pub fn get_face_outer_wire(&self, face: u32) -> Result<u32, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        Ok(wire_id_to_u32(face_data.outer_wire()))
    }

    /// Get all wires of a face (outer wire first, then inner/hole wires).
    ///
    /// # Errors
    /// Returns an error if the face handle is invalid.
    #[wasm_bindgen(js_name = "getFaceWires")]
    pub fn get_face_wires(&self, face: u32) -> Result<Vec<u32>, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let mut wires = vec![wire_id_to_u32(face_data.outer_wire())];
        for &iw in face_data.inner_wires() {
            wires.push(wire_id_to_u32(iw));
        }
        Ok(wires)
    }

    /// Get the surface type of a face.
    ///
    /// Returns one of: `"plane"`, `"cylinder"`, `"cone"`, `"sphere"`,
    /// `"torus"`, `"bspline"`.
    ///
    /// For NURBS surfaces that exactly represent analytic shapes, this
    /// returns the underlying analytic type (e.g. `"sphere"` for a NURBS
    /// sphere patch).
    #[wasm_bindgen(js_name = "getSurfaceType")]
    pub fn get_surface_type(&self, face: u32) -> Result<String, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        Ok(match face_data.surface() {
            FaceSurface::Plane { .. } => "plane",
            FaceSurface::Nurbs(ns) => detect_nurbs_surface_type(ns),
            FaceSurface::Cylinder(_) => "cylinder",
            FaceSurface::Cone(_) => "cone",
            FaceSurface::Sphere(_) => "sphere",
            FaceSurface::Torus(_) => "torus",
        }
        .into())
    }

    /// Get the curve type of an edge.
    ///
    /// Returns `"LINE"`, `"BSPLINE_CURVE"`, `"CIRCLE"`, or `"ELLIPSE"`.
    ///
    /// For NURBS curves that exactly represent analytic curves, this
    /// returns the underlying analytic type (e.g. `"CIRCLE"` for a
    /// rational NURBS circle).
    #[wasm_bindgen(js_name = "getEdgeCurveType")]
    pub fn get_edge_curve_type(&self, edge: u32) -> Result<String, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        Ok(match edge_data.curve() {
            EdgeCurve::Line => "LINE",
            EdgeCurve::NurbsCurve(nc) => detect_nurbs_curve_type(nc),
            EdgeCurve::Circle(_) => "CIRCLE",
            EdgeCurve::Ellipse(_) => "ELLIPSE",
        }
        .into())
    }

    /// Get the parameter domain of an edge curve.
    ///
    /// Returns `[t_start, t_end]`.
    /// For line edges: `[0.0, length]`.
    /// For NURBS edges: knot domain.
    #[wasm_bindgen(js_name = "getEdgeCurveParameters")]
    pub fn get_edge_curve_parameters(&self, edge: u32) -> Result<Vec<f64>, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        match edge_data.curve() {
            EdgeCurve::Line => {
                let start = self.topo.vertex(edge_data.start())?.point();
                let end = self.topo.vertex(edge_data.end())?.point();
                let len = (end - start).length();
                Ok(vec![0.0, len])
            }
            EdgeCurve::NurbsCurve(curve) => {
                let (u_start, u_end) = curve.domain();
                Ok(vec![u_start, u_end])
            }
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => Ok(vec![0.0, std::f64::consts::TAU]),
        }
    }

    /// Evaluate a point on an edge curve at parameter `t`.
    ///
    /// Returns `[x, y, z]`.
    #[wasm_bindgen(js_name = "evaluateEdgeCurve")]
    pub fn evaluate_edge_curve(&self, edge: u32, t: f64) -> Result<Vec<f64>, JsError> {
        validate_finite(t, "t")?;
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        let point = match edge_data.curve() {
            EdgeCurve::Line => {
                let start = self.topo.vertex(edge_data.start())?.point();
                let end = self.topo.vertex(edge_data.end())?.point();
                let len = (end - start).length();
                if len < 1e-15 {
                    start
                } else {
                    let frac = t / len;
                    let dir = end - start;
                    Point3::new(
                        start.x() + dir.x() * frac,
                        start.y() + dir.y() * frac,
                        start.z() + dir.z() * frac,
                    )
                }
            }
            EdgeCurve::NurbsCurve(curve) => curve.evaluate(t),
            EdgeCurve::Circle(circle) => circle.evaluate(t),
            EdgeCurve::Ellipse(ellipse) => ellipse.evaluate(t),
        };
        Ok(vec![point.x(), point.y(), point.z()])
    }

    /// Evaluate a point and tangent on an edge curve at parameter `t`.
    ///
    /// Returns `[px, py, pz, tx, ty, tz]`.
    #[wasm_bindgen(js_name = "evaluateEdgeCurveD1")]
    pub fn evaluate_edge_curve_d1(&self, edge: u32, t: f64) -> Result<Vec<f64>, JsError> {
        validate_finite(t, "t")?;
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        match edge_data.curve() {
            EdgeCurve::Line => {
                let start = self.topo.vertex(edge_data.start())?.point();
                let end = self.topo.vertex(edge_data.end())?.point();
                let dir = end - start;
                let len = dir.length();
                let frac = if len < 1e-15 { 0.0 } else { t / len };
                let point = Point3::new(
                    start.x() + dir.x() * frac,
                    start.y() + dir.y() * frac,
                    start.z() + dir.z() * frac,
                );
                let tangent = if len < 1e-15 {
                    Vec3::new(1.0, 0.0, 0.0)
                } else {
                    Vec3::new(dir.x() / len, dir.y() / len, dir.z() / len)
                };
                Ok(vec![
                    point.x(),
                    point.y(),
                    point.z(),
                    tangent.x(),
                    tangent.y(),
                    tangent.z(),
                ])
            }
            EdgeCurve::NurbsCurve(curve) => {
                let point = curve.evaluate(t);
                let derivs = curve.derivatives(t, 1);
                let tangent = if derivs.len() > 1 {
                    derivs[1]
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                Ok(vec![
                    point.x(),
                    point.y(),
                    point.z(),
                    tangent.x(),
                    tangent.y(),
                    tangent.z(),
                ])
            }
            EdgeCurve::Circle(circle) => {
                let point = circle.evaluate(t);
                let tangent = circle.tangent(t);
                Ok(vec![
                    point.x(),
                    point.y(),
                    point.z(),
                    tangent.x(),
                    tangent.y(),
                    tangent.z(),
                ])
            }
            EdgeCurve::Ellipse(ellipse) => {
                let point = ellipse.evaluate(t);
                let tangent = ellipse.tangent(t);
                Ok(vec![
                    point.x(),
                    point.y(),
                    point.z(),
                    tangent.x(),
                    tangent.y(),
                    tangent.z(),
                ])
            }
        }
    }

    /// Measure curvature of an edge curve at parameter `t`.
    ///
    /// Returns `[curvature, tangent_x, tangent_y, tangent_z, normal_x, normal_y, normal_z]`.
    /// Curvature is 1/radius. For lines, curvature is 0.
    #[wasm_bindgen(js_name = "measureCurvatureAtEdge")]
    pub fn measure_curvature_at_edge(&self, edge: u32, t: f64) -> Result<Vec<f64>, JsError> {
        validate_finite(t, "t")?;
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        match edge_data.curve() {
            EdgeCurve::Line => {
                let start = self.topo.vertex(edge_data.start())?.point();
                let end = self.topo.vertex(edge_data.end())?.point();
                let dir = end - start;
                let len = dir.length();
                let tangent = if len < 1e-15 {
                    Vec3::new(1.0, 0.0, 0.0)
                } else {
                    Vec3::new(dir.x() / len, dir.y() / len, dir.z() / len)
                };
                Ok(vec![
                    0.0,
                    tangent.x(),
                    tangent.y(),
                    tangent.z(),
                    0.0,
                    0.0,
                    0.0,
                ])
            }
            EdgeCurve::NurbsCurve(curve) => {
                let curvature = curve.curvature(t).unwrap_or(0.0);
                let derivs = curve.derivatives(t, 2);
                let tangent = if derivs.len() > 1 {
                    derivs[1].normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0))
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                let normal = if derivs.len() > 2 {
                    let d1 = derivs[1];
                    let d2 = derivs[2];
                    let cross = d1.cross(d2);
                    let binormal = cross.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                    binormal
                        .cross(tangent)
                        .normalize()
                        .unwrap_or(Vec3::new(0.0, 1.0, 0.0))
                } else {
                    Vec3::new(0.0, 1.0, 0.0)
                };
                Ok(vec![
                    curvature,
                    tangent.x(),
                    tangent.y(),
                    tangent.z(),
                    normal.x(),
                    normal.y(),
                    normal.z(),
                ])
            }
            EdgeCurve::Circle(circle) => {
                let curvature = 1.0 / circle.radius();
                let tangent = circle
                    .tangent(t)
                    .normalize()
                    .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                let point = circle.evaluate(t);
                let to_center = Vec3::new(
                    circle.center().x() - point.x(),
                    circle.center().y() - point.y(),
                    circle.center().z() - point.z(),
                );
                let normal = to_center.normalize().unwrap_or(Vec3::new(0.0, 1.0, 0.0));
                Ok(vec![
                    curvature,
                    tangent.x(),
                    tangent.y(),
                    tangent.z(),
                    normal.x(),
                    normal.y(),
                    normal.z(),
                ])
            }
            EdgeCurve::Ellipse(ellipse) => {
                let point = ellipse.evaluate(t);
                let tangent = ellipse
                    .tangent(t)
                    .normalize()
                    .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                // Approximate curvature from finite differences
                let dt = 1e-6;
                let p0 = ellipse.evaluate(t - dt);
                let p1 = ellipse.evaluate(t + dt);
                let d1 = p1 - p0;
                let d2 = (p1 - point) - (point - p0);
                let speed = d1.length() / (2.0 * dt);
                let curvature = if speed > 1e-15 {
                    d1.cross(d2).length() / ((2.0 * dt) * speed * speed * speed)
                } else {
                    0.0
                };
                let normal = Vec3::new(
                    ellipse.center().x() - point.x(),
                    ellipse.center().y() - point.y(),
                    ellipse.center().z() - point.z(),
                )
                .normalize()
                .unwrap_or(Vec3::new(0.0, 1.0, 0.0));
                Ok(vec![
                    curvature,
                    tangent.x(),
                    tangent.y(),
                    tangent.z(),
                    normal.x(),
                    normal.y(),
                    normal.z(),
                ])
            }
        }
    }

    /// Evaluate a surface normal at (u, v) on a face.
    ///
    /// Returns `[nx, ny, nz]`.
    #[wasm_bindgen(js_name = "evaluateSurfaceNormal")]
    pub fn evaluate_surface_normal(&self, face: u32, u: f64, v: f64) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        match face_data.surface() {
            FaceSurface::Plane { normal, .. } => Ok(vec![normal.x(), normal.y(), normal.z()]),
            FaceSurface::Nurbs(surface) => {
                let derivs = surface.derivatives(u, v, 1);
                let du = if derivs.len() > 1 && !derivs[1].is_empty() {
                    derivs[1][0]
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                let dv = if !derivs.is_empty() && derivs[0].len() > 1 {
                    derivs[0][1]
                } else {
                    Vec3::new(0.0, 1.0, 0.0)
                };
                let n = du.cross(dv);
                match n.normalize() {
                    Ok(normal) => Ok(vec![normal.x(), normal.y(), normal.z()]),
                    Err(_) => Ok(vec![0.0, 0.0, 1.0]),
                }
            }
            FaceSurface::Cylinder(cyl) => {
                let n = cyl.normal(u, v);
                Ok(vec![n.x(), n.y(), n.z()])
            }
            FaceSurface::Cone(cone) => {
                let n = cone.normal(u, v);
                Ok(vec![n.x(), n.y(), n.z()])
            }
            FaceSurface::Sphere(sph) => {
                let n = sph.normal(u, v);
                Ok(vec![n.x(), n.y(), n.z()])
            }
            FaceSurface::Torus(tor) => {
                let n = tor.normal(u, v);
                Ok(vec![n.x(), n.y(), n.z()])
            }
        }
    }

    /// Evaluate a point on a face surface at (u, v).
    ///
    /// Returns `[x, y, z]`.
    #[wasm_bindgen(js_name = "evaluateSurface")]
    pub fn evaluate_surface(&self, face: u32, u: f64, v: f64) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let point = match face_data.surface() {
            FaceSurface::Plane { normal, d } => {
                // Build a point on the plane: p = d * normal + u * x_axis + v * y_axis
                // Choose arbitrary axes perpendicular to normal
                let up = if normal.x().abs() < 0.9 {
                    Vec3::new(1.0, 0.0, 0.0)
                } else {
                    Vec3::new(0.0, 1.0, 0.0)
                };
                let x_axis = normal.cross(up);
                let y_axis = normal.cross(x_axis);
                Point3::new(
                    normal.x() * d + x_axis.x() * u + y_axis.x() * v,
                    normal.y() * d + x_axis.y() * u + y_axis.y() * v,
                    normal.z() * d + x_axis.z() * u + y_axis.z() * v,
                )
            }
            FaceSurface::Nurbs(surface) => surface.evaluate(u, v),
            FaceSurface::Cylinder(cyl) => cyl.evaluate(u, v),
            FaceSurface::Cone(cone) => cone.evaluate(u, v),
            FaceSurface::Sphere(sph) => sph.evaluate(u, v),
            FaceSurface::Torus(tor) => tor.evaluate(u, v),
        };
        Ok(vec![point.x(), point.y(), point.z()])
    }

    /// Measure principal curvatures at (u, v) on a face surface.
    ///
    /// Returns `[k1, k2, d1x, d1y, d1z, d2x, d2y, d2z]` where k1/k2 are
    /// principal curvatures and d1/d2 are the corresponding direction vectors.
    #[wasm_bindgen(js_name = "measureCurvatureAtSurface")]
    #[allow(clippy::too_many_lines)]
    pub fn measure_curvature_at_surface(
        &self,
        face: u32,
        u: f64,
        v: f64,
    ) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        match face_data.surface() {
            FaceSurface::Plane { .. } => Ok(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            FaceSurface::Nurbs(surface) => {
                let derivs = surface.derivatives(u, v, 2);
                // derivs[i][j] = d^(i+j) S / du^i dv^j
                let su = if derivs.len() > 1 && !derivs[1].is_empty() {
                    derivs[1][0]
                } else {
                    return Ok(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
                };
                let sv = if !derivs.is_empty() && derivs[0].len() > 1 {
                    derivs[0][1]
                } else {
                    return Ok(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
                };
                let suu = if derivs.len() > 2 && !derivs[2].is_empty() {
                    derivs[2][0]
                } else {
                    Vec3::new(0.0, 0.0, 0.0)
                };
                let suv = if derivs.len() > 1 && derivs[1].len() > 1 {
                    derivs[1][1]
                } else {
                    Vec3::new(0.0, 0.0, 0.0)
                };
                let svv = if !derivs.is_empty() && derivs[0].len() > 2 {
                    derivs[0][2]
                } else {
                    Vec3::new(0.0, 0.0, 0.0)
                };

                let normal = su.cross(sv);
                let normal = match normal.normalize() {
                    Ok(n) => n,
                    Err(_) => return Ok(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                };

                // First fundamental form coefficients
                let ee = su.dot(su);
                let ff = su.dot(sv);
                let gg = sv.dot(sv);

                // Second fundamental form coefficients
                let ll = suu.dot(normal);
                let mm = suv.dot(normal);
                let nn = svv.dot(normal);

                // Principal curvatures from shape operator eigenvalues
                let denom = ee * gg - ff * ff;
                if denom.abs() < 1e-30 {
                    return Ok(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
                }
                let h = 0.5 * (ee * nn - 2.0 * ff * mm + gg * ll) / denom; // mean curvature
                let k = (ll * nn - mm * mm) / denom; // Gaussian curvature
                let disc = (h * h - k).max(0.0).sqrt();
                let k1 = h + disc;
                let k2 = h - disc;

                // Principal directions (approximate)
                let su_norm = su.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                let sv_norm = sv.normalize().unwrap_or(Vec3::new(0.0, 1.0, 0.0));

                Ok(vec![
                    k1,
                    k2,
                    su_norm.x(),
                    su_norm.y(),
                    su_norm.z(),
                    sv_norm.x(),
                    sv_norm.y(),
                    sv_norm.z(),
                ])
            }
            FaceSurface::Cylinder(cyl) => {
                let r = cyl.radius();
                let axis = cyl.axis().normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                let point = cyl.evaluate(u, v);
                let to_axis = Vec3::new(
                    cyl.origin().x() - point.x()
                        + axis.x()
                            * axis.dot(Vec3::new(
                                point.x() - cyl.origin().x(),
                                point.y() - cyl.origin().y(),
                                point.z() - cyl.origin().z(),
                            )),
                    cyl.origin().y() - point.y()
                        + axis.y()
                            * axis.dot(Vec3::new(
                                point.x() - cyl.origin().x(),
                                point.y() - cyl.origin().y(),
                                point.z() - cyl.origin().z(),
                            )),
                    cyl.origin().z() - point.z()
                        + axis.z()
                            * axis.dot(Vec3::new(
                                point.x() - cyl.origin().x(),
                                point.y() - cyl.origin().y(),
                                point.z() - cyl.origin().z(),
                            )),
                );
                let radial = to_axis.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                Ok(vec![
                    1.0 / r,
                    0.0,
                    radial.x(),
                    radial.y(),
                    radial.z(),
                    axis.x(),
                    axis.y(),
                    axis.z(),
                ])
            }
            FaceSurface::Sphere(sph) => {
                let r = sph.radius();
                let point = sph.evaluate(u, v);
                let radial = Vec3::new(
                    point.x() - sph.center().x(),
                    point.y() - sph.center().y(),
                    point.z() - sph.center().z(),
                )
                .normalize()
                .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                // Both principal curvatures are 1/r for a sphere
                let d1 = Vec3::new(-radial.y(), radial.x(), 0.0)
                    .normalize()
                    .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                let d2 = radial.cross(d1);
                Ok(vec![
                    1.0 / r,
                    1.0 / r,
                    d1.x(),
                    d1.y(),
                    d1.z(),
                    d2.x(),
                    d2.y(),
                    d2.z(),
                ])
            }
            FaceSurface::Cone(cone) => {
                let half_angle = cone.half_angle();
                let v_pos = v.abs().max(1e-10);
                let local_r = v_pos * half_angle.sin();
                let k_parallel = if local_r > 1e-15 {
                    half_angle.cos() / local_r
                } else {
                    0.0
                };
                let axis = cone.axis().normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                Ok(vec![
                    0.0,
                    k_parallel,
                    axis.x(),
                    axis.y(),
                    axis.z(),
                    1.0,
                    0.0,
                    0.0,
                ])
            }
            FaceSurface::Torus(torus) => {
                let r_major = torus.major_radius();
                let r_minor = torus.minor_radius();
                let k1 = 1.0 / r_minor;
                let k2 = u.cos() / (r_major + r_minor * u.cos());
                Ok(vec![k1, k2, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            }
        }
    }

    /// Tessellate an edge curve into polyline segments.
    ///
    /// For line edges, returns just start and end points.
    /// For NURBS edges, samples at `num_points` along the curve.
    ///
    /// Returns flattened `[x, y, z, x, y, z, ...]` array.
    #[wasm_bindgen(js_name = "tessellateEdge")]
    pub fn tessellate_edge(&self, edge: u32, num_points: u32) -> Result<Vec<f64>, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;

        match edge_data.curve() {
            EdgeCurve::Line => {
                let start = self.topo.vertex(edge_data.start())?.point();
                let end = self.topo.vertex(edge_data.end())?.point();
                Ok(vec![
                    start.x(),
                    start.y(),
                    start.z(),
                    end.x(),
                    end.y(),
                    end.z(),
                ])
            }
            EdgeCurve::NurbsCurve(curve) => {
                let (u0, u1) = curve.domain();
                let n = std::cmp::max(2, num_points as usize);
                let mut result = Vec::with_capacity(n * 3);
                for i in 0..n {
                    #[allow(clippy::cast_precision_loss)]
                    let t = u0 + (u1 - u0) * (i as f64) / ((n - 1) as f64);
                    let p = curve.evaluate(t);
                    result.push(p.x());
                    result.push(p.y());
                    result.push(p.z());
                }
                Ok(result)
            }
            EdgeCurve::Circle(circle) => {
                let n = std::cmp::max(2, num_points as usize);
                Ok(sample_full_period_curve(n, |t| circle.evaluate(t)))
            }
            EdgeCurve::Ellipse(ellipse) => {
                let n = std::cmp::max(2, num_points as usize);
                Ok(sample_full_period_curve(n, |t| ellipse.evaluate(t)))
            }
        }
    }

    /// Check if an edge is forward-oriented in a given wire.
    ///
    /// Returns `true` if the edge is forward in the wire, `false` if reversed.
    #[wasm_bindgen(js_name = "isEdgeForwardInWire")]
    pub fn is_edge_forward_in_wire(&self, edge: u32, wire: u32) -> Result<bool, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let wire_id = self.resolve_wire(wire)?;
        let wire_data = self.topo.wire(wire_id)?;

        for oe in wire_data.edges() {
            if oe.edge() == edge_id {
                return Ok(oe.is_forward());
            }
        }

        Err(WasmError::InvalidInput {
            reason: "edge not found in wire".into(),
        }
        .into())
    }

    /// Get the UV parameter domain of a face's surface.
    ///
    /// Returns `[u_min, u_max, v_min, v_max]`.
    #[wasm_bindgen(js_name = "getSurfaceDomain")]
    pub fn get_surface_domain(&self, face: u32) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        match face_data.surface() {
            FaceSurface::Plane { .. } => Ok(vec![-1e6, 1e6, -1e6, 1e6]),
            FaceSurface::Nurbs(surface) => {
                let (u0, u1) = surface.domain_u();
                let (v0, v1) = surface.domain_v();
                Ok(vec![u0, u1, v0, v1])
            }
            FaceSurface::Cylinder(cyl) => {
                let v_range = self.compute_axial_v_range(face_id, cyl.origin(), cyl.axis())?;
                Ok(vec![0.0, 2.0 * PI, v_range.0, v_range.1])
            }
            FaceSurface::Cone(cone) => {
                let v_range = self.compute_axial_v_range(face_id, cone.apex(), cone.axis())?;
                Ok(vec![0.0, 2.0 * PI, v_range.0, v_range.1])
            }
            FaceSurface::Sphere(_) => Ok(vec![0.0, 2.0 * PI, -PI / 2.0, PI / 2.0]),
            FaceSurface::Torus(_) => Ok(vec![0.0, 2.0 * PI, 0.0, 2.0 * PI]),
        }
    }

    /// Project a 3D point onto a face surface using Newton iteration.
    ///
    /// Returns `[u, v, px, py, pz, distance]`.
    #[wasm_bindgen(js_name = "projectPointOnSurface")]
    pub fn project_point_on_surface(
        &self,
        face: u32,
        px: f64,
        py: f64,
        pz: f64,
    ) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let target = Point3::new(px, py, pz);

        match face_data.surface() {
            FaceSurface::Plane { normal, d } => {
                // Project onto plane: p - ((p·n - d) * n)
                let dist_to_plane = normal.x() * px + normal.y() * py + normal.z() * pz - d;
                let proj = Point3::new(
                    px - dist_to_plane * normal.x(),
                    py - dist_to_plane * normal.y(),
                    pz - dist_to_plane * normal.z(),
                );
                let dist = (proj - target).length();
                // UV coordinates: project onto plane's local frame
                Ok(vec![proj.x(), proj.y(), proj.x(), proj.y(), proj.z(), dist])
            }
            FaceSurface::Nurbs(surface) => {
                // Newton iteration for closest point on NURBS surface
                let (u0, u1) = surface.domain_u();
                let (v0, v1) = surface.domain_v();
                let mut best_u = f64::midpoint(u0, u1);
                let mut best_v = f64::midpoint(v0, v1);
                let mut best_dist = f64::MAX;

                // Grid search for initial guess
                let n_grid = 8;
                for iu in 0..=n_grid {
                    for iv in 0..=n_grid {
                        #[allow(clippy::cast_precision_loss)]
                        let u = u0 + (u1 - u0) * (iu as f64) / (n_grid as f64);
                        #[allow(clippy::cast_precision_loss)]
                        let v = v0 + (v1 - v0) * (iv as f64) / (n_grid as f64);
                        let p = surface.evaluate(u, v);
                        let d = (p - target).length();
                        if d < best_dist {
                            best_dist = d;
                            best_u = u;
                            best_v = v;
                        }
                    }
                }

                // Newton refinement (5 iterations)
                for _ in 0..5 {
                    let p = surface.evaluate(best_u, best_v);
                    let derivs = surface.derivatives(best_u, best_v, 1);
                    if derivs.len() < 2 || derivs[0].len() < 2 || derivs[1].is_empty() {
                        break;
                    }
                    let du = derivs[1][0]; // dS/du
                    let dv = derivs[0][1]; // dS/dv
                    let diff = p - target;

                    // Jacobian entries
                    let j00 = du.dot(du);
                    let j01 = du.dot(dv);
                    let j10 = j01;
                    let j11 = dv.dot(dv);
                    let r0 = diff.x() * du.x() + diff.y() * du.y() + diff.z() * du.z();
                    let r1 = diff.x() * dv.x() + diff.y() * dv.y() + diff.z() * dv.z();

                    let det = j00 * j11 - j01 * j10;
                    if det.abs() < 1e-20 {
                        break;
                    }
                    let delta_u = -(j11 * r0 - j01 * r1) / det;
                    let delta_v = -(-j10 * r0 + j00 * r1) / det;

                    best_u = (best_u + delta_u).clamp(u0, u1);
                    best_v = (best_v + delta_v).clamp(v0, v1);
                }

                let proj = surface.evaluate(best_u, best_v);
                let dist = (proj - target).length();
                Ok(vec![best_u, best_v, proj.x(), proj.y(), proj.z(), dist])
            }
            _ => {
                // For analytic surfaces, use grid search (no Newton for now)
                let mut best_u = 0.0;
                let mut best_v = 0.0;
                let mut best_dist = f64::MAX;
                let n_grid = 16;
                for iu in 0..=n_grid {
                    for iv in 0..=n_grid {
                        #[allow(clippy::cast_precision_loss)]
                        let u = 2.0 * PI * (iu as f64) / (n_grid as f64);
                        #[allow(clippy::cast_precision_loss)]
                        let v = -PI + 2.0 * PI * (iv as f64) / (n_grid as f64);
                        let p = match face_data.surface() {
                            FaceSurface::Cylinder(cyl) => cyl.evaluate(u, v),
                            FaceSurface::Cone(cone) => cone.evaluate(u, v),
                            FaceSurface::Sphere(sph) => sph.evaluate(u, v),
                            FaceSurface::Torus(tor) => tor.evaluate(u, v),
                            _ => continue,
                        };
                        let d = (p - target).length();
                        if d < best_dist {
                            best_dist = d;
                            best_u = u;
                            best_v = v;
                        }
                    }
                }
                let proj = match face_data.surface() {
                    FaceSurface::Cylinder(cyl) => cyl.evaluate(best_u, best_v),
                    FaceSurface::Cone(cone) => cone.evaluate(best_u, best_v),
                    FaceSurface::Sphere(sph) => sph.evaluate(best_u, best_v),
                    FaceSurface::Torus(tor) => tor.evaluate(best_u, best_v),
                    _ => target,
                };
                Ok(vec![
                    best_u,
                    best_v,
                    proj.x(),
                    proj.y(),
                    proj.z(),
                    best_dist,
                ])
            }
        }
    }

    /// Add hole wires to an existing face, creating a new face with the same
    /// surface but additional inner wires.
    ///
    /// Returns a new face handle (`u32`).
    #[wasm_bindgen(js_name = "addHolesToFace")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn add_holes_to_face(
        &mut self,
        face: u32,
        hole_wire_handles: Vec<u32>,
    ) -> Result<u32, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let outer_wire = face_data.outer_wire();
        let surface = face_data.surface().clone();
        let mut inner_wires: Vec<brepkit_topology::wire::WireId> = face_data.inner_wires().to_vec();

        for &wh in &hole_wire_handles {
            let wid = self.resolve_wire(wh)?;
            inner_wires.push(wid);
        }

        let new_face = Face::new(outer_wire, inner_wires, surface);
        let fid = self.topo.add_face(new_face);
        Ok(face_id_to_u32(fid))
    }

    /// Build an edge's NURBS curve data for JS consumption.
    ///
    /// Returns `null` for line edges, or a JSON string with
    /// `{degree, knots, controlPoints, weights}` for NURBS edges.
    #[wasm_bindgen(js_name = "getEdgeNurbsData")]
    pub fn get_edge_nurbs_data(&self, edge: u32) -> Result<JsValue, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        match edge_data.curve() {
            EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => Ok(JsValue::NULL),
            EdgeCurve::NurbsCurve(curve) => {
                let cp_flat: Vec<f64> = curve
                    .control_points()
                    .iter()
                    .flat_map(|p| [p.x(), p.y(), p.z()])
                    .collect();
                let data = serde_json::json!({
                    "degree": curve.degree(),
                    "knots": curve.knots(),
                    "controlPoints": cp_flat,
                    "weights": curve.weights(),
                });
                Ok(JsValue::from_str(&data.to_string()))
            }
        }
    }

    /// Get the edge-to-face adjacency map for a solid.
    ///
    /// Returns a JSON string: `{"edgeId": [faceId, ...], ...}`.
    #[wasm_bindgen(js_name = "edgeToFaceMap")]
    pub fn edge_to_face_map(&self, solid: u32) -> Result<String, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let map = brepkit_topology::explorer::edge_to_face_map(&self.topo, solid_id)?;
        let json_map: std::collections::HashMap<String, Vec<u32>> = map
            .into_iter()
            .map(|(edge_idx, face_ids)| {
                let fids: Vec<u32> = face_ids.iter().map(|f| face_id_to_u32(*f)).collect();
                (edge_idx.to_string(), fids)
            })
            .collect();
        Ok(serde_json::json!(json_map).to_string())
    }

    /// Get edges shared between two faces.
    ///
    /// Returns an array of edge handles.
    #[wasm_bindgen(js_name = "sharedEdges")]
    pub fn shared_edges(&self, face_a: u32, face_b: u32) -> Result<Vec<u32>, JsError> {
        let fa = self.resolve_face(face_a)?;
        let fb = self.resolve_face(face_b)?;
        let edges = brepkit_topology::explorer::shared_edges(&self.topo, fa, fb)?;
        Ok(edges.iter().map(|e| edge_id_to_u32(*e)).collect())
    }

    /// Get faces adjacent to a given face within a solid.
    ///
    /// Returns an array of face handles.
    #[wasm_bindgen(js_name = "adjacentFaces")]
    pub fn adjacent_faces(&self, solid: u32, face: u32) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let face_id = self.resolve_face(face)?;
        let map = brepkit_topology::explorer::edge_to_face_map(&self.topo, solid_id)?;
        let adj = brepkit_topology::explorer::adjacent_faces(&self.topo, face_id, &map)?;
        Ok(adj.iter().map(|f| face_id_to_u32(*f)).collect())
    }

    /// Get the wires (outer + inner) of a face.
    ///
    /// Returns an array of wire handles.
    #[wasm_bindgen(js_name = "faceWires")]
    pub fn face_wires(&self, face: u32) -> Result<Vec<u32>, JsError> {
        let face_id = self.resolve_face(face)?;
        let wires = brepkit_topology::explorer::face_wires(&self.topo, face_id)?;
        Ok(wires.iter().map(|w| wire_id_to_u32(*w)).collect())
    }

    /// Get the solid handles within a compound.
    ///
    /// Returns an array of solid handles (`u32[]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the compound handle is invalid.
    #[wasm_bindgen(js_name = "getCompoundSolids")]
    pub fn get_compound_solids(&self, compound: u32) -> Result<Vec<u32>, JsError> {
        let compound_id = self.resolve_compound(compound)?;
        let compound_data = self.topo.compound(compound_id)?;
        Ok(compound_data
            .solids()
            .iter()
            .map(|s| solid_id_to_u32(*s))
            .collect())
    }

    /// Get the face handles of a shell.
    ///
    /// Returns an array of face handles (`u32[]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the shell handle is invalid.
    #[wasm_bindgen(js_name = "getShellFaces")]
    pub fn get_shell_faces(&self, shell: u32) -> Result<Vec<u32>, JsError> {
        let shell_id = self.resolve_shell(shell)?;
        let shell_data = self.topo.shell(shell_id)?;
        Ok(shell_data
            .faces()
            .iter()
            .map(|f| face_id_to_u32(*f))
            .collect())
    }

    /// Get the edge handles of a wire.
    ///
    /// Returns an array of unique edge handles (`u32[]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the wire handle is invalid.
    #[wasm_bindgen(js_name = "getWireEdges")]
    pub fn get_wire_edges(&self, wire: u32) -> Result<Vec<u32>, JsError> {
        let wire_id = self.resolve_wire(wire)?;
        let wire_data = self.topo.wire(wire_id)?;
        Ok(wire_data
            .edges()
            .iter()
            .map(|oe| edge_id_to_u32(oe.edge()))
            .collect())
    }

    /// Check whether a wire is closed (last edge connects back to first).
    #[wasm_bindgen(js_name = "isWireClosed")]
    pub fn is_wire_closed(&self, wire: u32) -> Result<bool, JsError> {
        let wire_id = self.resolve_wire(wire)?;
        let wire_data = self.topo.wire(wire_id)?;
        Ok(wire_data.is_closed())
    }

    /// Compute the total arc-length of a wire.
    #[wasm_bindgen(js_name = "wireLength")]
    pub fn wire_length(&self, wire: u32) -> Result<f64, JsError> {
        let wire_id = self.resolve_wire(wire)?;
        let wire_data = self.topo.wire(wire_id)?;
        let mut total = 0.0;
        for oe in wire_data.edges() {
            total += brepkit_operations::measure::edge_length(&self.topo, oe.edge())?;
        }
        Ok(total)
    }

    /// Get the analytic surface parameters of a face.
    ///
    /// Returns a JSON string with surface-type-specific parameters.
    #[wasm_bindgen(js_name = "getAnalyticSurfaceParams")]
    pub fn get_analytic_surface_params(&self, face: u32) -> Result<String, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let json = match face_data.surface() {
            FaceSurface::Plane { normal, d } => serde_json::json!({
                "type": "plane",
                "normal": [normal.x(), normal.y(), normal.z()],
                "d": d,
            }),
            FaceSurface::Nurbs(_) => serde_json::json!({
                "type": "nurbs",
            }),
            FaceSurface::Cylinder(cyl) => serde_json::json!({
                "type": "cylinder",
                "origin": [cyl.origin().x(), cyl.origin().y(), cyl.origin().z()],
                "axis": [cyl.axis().x(), cyl.axis().y(), cyl.axis().z()],
                "radius": cyl.radius(),
            }),
            FaceSurface::Cone(cone) => serde_json::json!({
                "type": "cone",
                "apex": [cone.apex().x(), cone.apex().y(), cone.apex().z()],
                "axis": [cone.axis().x(), cone.axis().y(), cone.axis().z()],
                "halfAngle": cone.half_angle(),
            }),
            FaceSurface::Sphere(sph) => serde_json::json!({
                "type": "sphere",
                "center": [sph.center().x(), sph.center().y(), sph.center().z()],
                "radius": sph.radius(),
            }),
            FaceSurface::Torus(tor) => serde_json::json!({
                "type": "torus",
                "center": [tor.center().x(), tor.center().y(), tor.center().z()],
                "majorRadius": tor.major_radius(),
                "minorRadius": tor.minor_radius(),
            }),
        };
        Ok(json.to_string())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use crate::kernel::test_fixtures::{kernel_with_box, kernel_with_cylinder};

    // ── get_solid_faces ───────────────────────────────────────────

    #[test]
    fn box_has_six_faces() {
        let (k, solid) = kernel_with_box();
        let faces = k.get_solid_faces(solid).unwrap();
        assert_eq!(faces.len(), 6, "a box must have exactly 6 faces");
    }

    #[test]
    fn face_handles_are_unique() {
        let (k, solid) = kernel_with_box();
        let faces = k.get_solid_faces(solid).unwrap();
        let mut sorted = faces.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), faces.len(), "face handles must be unique");
    }

    // ── get_solid_edges ───────────────────────────────────────────

    #[test]
    fn box_has_twelve_edges() {
        let (k, solid) = kernel_with_box();
        let edges = k.get_solid_edges(solid).unwrap();
        assert_eq!(edges.len(), 12, "a box must have exactly 12 edges");
    }

    #[test]
    fn edge_handles_are_unique() {
        let (k, solid) = kernel_with_box();
        let edges = k.get_solid_edges(solid).unwrap();
        let mut sorted = edges.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), edges.len(), "edge handles must be unique");
    }

    // ── get_solid_vertices ────────────────────────────────────────

    #[test]
    fn box_has_eight_vertices() {
        let (k, solid) = kernel_with_box();
        let verts = k.get_solid_vertices(solid).unwrap();
        assert_eq!(verts.len(), 8, "a box must have exactly 8 vertices");
    }

    #[test]
    fn vertex_handles_are_unique() {
        let (k, solid) = kernel_with_box();
        let verts = k.get_solid_vertices(solid).unwrap();
        let mut sorted = verts.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), verts.len(), "vertex handles must be unique");
    }

    // ── get_face_normal ───────────────────────────────────────────

    #[test]
    fn face_normal_has_three_components() {
        let (k, solid) = kernel_with_box();
        let faces = k.get_solid_faces(solid).unwrap();
        let normal = k.get_face_normal(faces[0]).unwrap();
        assert_eq!(normal.len(), 3, "normal must have exactly 3 components");
    }

    #[test]
    fn face_normal_is_unit_length() {
        let (k, solid) = kernel_with_box();
        let faces = k.get_solid_faces(solid).unwrap();
        // Every face of a box is planar — all normals must be unit vectors.
        for &fh in &faces {
            let n = k.get_face_normal(fh).unwrap();
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "face {fh} normal length {len} is not 1"
            );
        }
    }

    #[test]
    fn face_normal_error_on_non_planar_face() {
        let (k, solid) = kernel_with_cylinder();
        // The cylindrical lateral face is not planar — verify via topology.
        let solid_id = k.resolve_solid(solid).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&k.topo, solid_id).unwrap();
        // At least one face must be non-planar (Cylinder surface).
        let has_non_planar = faces.iter().any(|&fid| {
            let face = k.topo.face(fid).unwrap();
            !matches!(
                face.surface(),
                brepkit_topology::face::FaceSurface::Plane { .. }
            )
        });
        assert!(
            has_non_planar,
            "cylinder must contain at least one non-planar face"
        );
    }

    // ── get_entity_counts ─────────────────────────────────────────

    #[test]
    fn entity_counts_match_individual_queries() {
        let (k, solid) = kernel_with_box();
        let counts = k.get_entity_counts(solid).unwrap();
        assert_eq!(
            counts.len(),
            3,
            "get_entity_counts must return [faces, edges, vertices]"
        );
        let faces = k.get_solid_faces(solid).unwrap();
        let edges = k.get_solid_edges(solid).unwrap();
        let verts = k.get_solid_vertices(solid).unwrap();
        assert_eq!(counts[0] as usize, faces.len(), "face count mismatch");
        assert_eq!(counts[1] as usize, edges.len(), "edge count mismatch");
        assert_eq!(counts[2] as usize, verts.len(), "vertex count mismatch");
    }

    #[test]
    fn entity_counts_box_exact() {
        let (k, solid) = kernel_with_box();
        let counts = k.get_entity_counts(solid).unwrap();
        assert_eq!(counts[0], 6, "box: 6 faces");
        assert_eq!(counts[1], 12, "box: 12 edges");
        assert_eq!(counts[2], 8, "box: 8 vertices");
    }

    // ── invalid handle ────────────────────────────────────────────
    // Error-path tests use resolve_solid (which returns WasmError, not
    // JsError) to avoid the JsError panic on non-wasm targets.

    #[test]
    fn invalid_solid_handle_returns_error_for_faces() {
        let (k, _) = kernel_with_box();
        let result = k.resolve_solid(9999);
        assert!(
            result.is_err(),
            "non-existent solid handle must produce an error"
        );
    }

    #[test]
    fn invalid_solid_handle_returns_error_for_edges() {
        let (k, _) = kernel_with_box();
        let result = k.resolve_solid(9999);
        assert!(
            result.is_err(),
            "non-existent solid handle must produce an error"
        );
    }

    #[test]
    fn invalid_solid_handle_returns_error_for_vertices() {
        let (k, _) = kernel_with_box();
        let result = k.resolve_solid(9999);
        assert!(
            result.is_err(),
            "non-existent solid handle must produce an error"
        );
    }

    #[test]
    fn invalid_solid_handle_returns_error_for_entity_counts() {
        let (k, _) = kernel_with_box();
        let result = k.resolve_solid(9999);
        assert!(
            result.is_err(),
            "non-existent solid handle must produce an error"
        );
    }

    // ── toBREP geometry data ─────────────────────────────────────
    //
    // `to_brep` returns JsValue which panics on non-wasm targets, so we
    // replicate the JSON serialization logic in a helper and verify the
    // same field structure that `to_brep` emits.

    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::face::FaceSurface;

    /// Build the toBREP JSON for a solid using the same logic as
    /// `to_brep`, but returning `serde_json::Value` directly so the
    /// test can run on non-wasm targets.
    fn build_brep_json(k: &crate::kernel::BrepKernel, solid: u32) -> serde_json::Value {
        let solid_id = k.resolve_solid(solid).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&k.topo, solid_id).unwrap();
        let edges = brepkit_topology::explorer::solid_edges(&k.topo, solid_id).unwrap();
        let verts = brepkit_topology::explorer::solid_vertices(&k.topo, solid_id).unwrap();

        let edge_json: Vec<serde_json::Value> = edges
            .iter()
            .map(|&eid| {
                let e = k.topo.edge(eid).unwrap();
                let curve_type = e.curve().type_tag();
                let curve_params = match e.curve() {
                    EdgeCurve::Line => serde_json::json!(null),
                    EdgeCurve::Circle(c) => serde_json::json!({
                        "center": [c.center().x(), c.center().y(), c.center().z()],
                        "axis": [c.normal().x(), c.normal().y(), c.normal().z()],
                        "xAxis": [c.u_axis().x(), c.u_axis().y(), c.u_axis().z()],
                        "radius": c.radius(),
                    }),
                    EdgeCurve::Ellipse(el) => serde_json::json!({
                        "center": [el.center().x(), el.center().y(), el.center().z()],
                        "axis": [el.normal().x(), el.normal().y(), el.normal().z()],
                        "majorAxis": [el.u_axis().x(), el.u_axis().y(), el.u_axis().z()],
                        "majorRadius": el.semi_major(),
                        "minorRadius": el.semi_minor(),
                    }),
                    EdgeCurve::NurbsCurve(n) => serde_json::json!({
                        "degree": n.degree(),
                        "controlPoints": n.control_points().iter()
                            .map(|p| [p.x(), p.y(), p.z()])
                            .collect::<Vec<_>>(),
                        "weights": n.weights().to_vec(),
                        "knots": n.knots().to_vec(),
                    }),
                };
                serde_json::json!({
                    "id": eid.index(),
                    "curveType": curve_type,
                    "curveParams": curve_params,
                })
            })
            .collect();

        let face_json: Vec<serde_json::Value> = faces
            .iter()
            .map(|&fid| {
                let f = k.topo.face(fid).unwrap();
                let surface_type = f.surface().type_tag();
                let surface_params = match f.surface() {
                    FaceSurface::Plane { normal, d } => serde_json::json!({
                        "normal": [normal.x(), normal.y(), normal.z()],
                        "d": d,
                    }),
                    FaceSurface::Cylinder(c) => serde_json::json!({
                        "origin": [c.origin().x(), c.origin().y(), c.origin().z()],
                        "axis": [c.axis().x(), c.axis().y(), c.axis().z()],
                        "refDir": [c.x_axis().x(), c.x_axis().y(), c.x_axis().z()],
                        "radius": c.radius(),
                    }),
                    FaceSurface::Cone(c) => serde_json::json!({
                        "apex": [c.apex().x(), c.apex().y(), c.apex().z()],
                        "axis": [c.axis().x(), c.axis().y(), c.axis().z()],
                        "refDir": [c.x_axis().x(), c.x_axis().y(), c.x_axis().z()],
                        "halfAngle": c.half_angle(),
                    }),
                    FaceSurface::Sphere(s) => serde_json::json!({
                        "center": [s.center().x(), s.center().y(), s.center().z()],
                        "axis": [s.z_axis().x(), s.z_axis().y(), s.z_axis().z()],
                        "radius": s.radius(),
                    }),
                    FaceSurface::Torus(t) => serde_json::json!({
                        "center": [t.center().x(), t.center().y(), t.center().z()],
                        "axis": [t.z_axis().x(), t.z_axis().y(), t.z_axis().z()],
                        "majorRadius": t.major_radius(),
                        "minorRadius": t.minor_radius(),
                    }),
                    FaceSurface::Nurbs(n) => serde_json::json!({
                        "degreeU": n.degree_u(),
                        "degreeV": n.degree_v(),
                        "controlPoints": n.control_points().iter()
                            .map(|row| row.iter()
                                .map(|p| [p.x(), p.y(), p.z()])
                                .collect::<Vec<_>>())
                            .collect::<Vec<_>>(),
                        "weights": n.weights().to_vec(),
                        "knotsU": n.knots_u().to_vec(),
                        "knotsV": n.knots_v().to_vec(),
                    }),
                };
                serde_json::json!({
                    "id": fid.index(),
                    "surfaceType": surface_type,
                    "surfaceParams": surface_params,
                })
            })
            .collect();

        serde_json::json!({
            "vertices": verts.len(),
            "edges": edge_json,
            "faces": face_json,
        })
    }

    #[test]
    fn to_brep_box_edges_have_null_curve_params() {
        let (k, solid) = kernel_with_box();
        let brep = build_brep_json(&k, solid);
        let edges = brep["edges"].as_array().unwrap();
        assert_eq!(edges.len(), 12, "box must have 12 edges");
        for edge in edges {
            assert_eq!(edge["curveType"].as_str().unwrap(), "line");
            assert!(
                edge["curveParams"].is_null(),
                "line edges should have null curveParams"
            );
        }
    }

    #[test]
    fn to_brep_box_faces_have_plane_surface_params() {
        let (k, solid) = kernel_with_box();
        let brep = build_brep_json(&k, solid);
        let faces = brep["faces"].as_array().unwrap();
        assert_eq!(faces.len(), 6, "box must have 6 faces");
        for face in faces {
            assert_eq!(face["surfaceType"].as_str().unwrap(), "plane");
            let params = &face["surfaceParams"];
            let normal = params["normal"].as_array().unwrap();
            assert_eq!(normal.len(), 3, "plane normal must have 3 components");
            let len: f64 = normal
                .iter()
                .map(|v| v.as_f64().unwrap().powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "plane normal must be unit length, got {len}"
            );
            assert!(
                params["d"].as_f64().is_some(),
                "plane params must include 'd'"
            );
        }
    }

    #[test]
    fn to_brep_cylinder_circle_edge_params() {
        let (k, solid) = kernel_with_cylinder();
        let brep = build_brep_json(&k, solid);
        let edges = brep["edges"].as_array().unwrap();
        let circle_edges: Vec<_> = edges
            .iter()
            .filter(|e| e["curveType"].as_str().unwrap() == "circle")
            .collect();
        assert!(!circle_edges.is_empty(), "cylinder must have circle edges");
        for ce in &circle_edges {
            let params = &ce["curveParams"];
            assert!(!params.is_null(), "circle edge must have curveParams");
            let center = params["center"].as_array().unwrap();
            assert_eq!(center.len(), 3);
            let axis = params["axis"].as_array().unwrap();
            assert_eq!(axis.len(), 3);
            // axis must be unit length
            let axis_len: f64 = axis
                .iter()
                .map(|v| v.as_f64().unwrap().powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                (axis_len - 1.0).abs() < 1e-10,
                "circle axis must be unit length, got {axis_len}"
            );
            let x_axis = params["xAxis"].as_array().unwrap();
            assert_eq!(x_axis.len(), 3);
            let radius = params["radius"].as_f64().unwrap();
            assert!(
                (radius - 1.0).abs() < 1e-10,
                "circle radius should be 1.0, got {radius}"
            );
        }
    }

    #[test]
    fn to_brep_cylinder_surface_params() {
        let (k, solid) = kernel_with_cylinder();
        let brep = build_brep_json(&k, solid);
        let faces = brep["faces"].as_array().unwrap();
        let cyl_faces: Vec<_> = faces
            .iter()
            .filter(|f| f["surfaceType"].as_str().unwrap() == "cylinder")
            .collect();
        assert!(
            !cyl_faces.is_empty(),
            "cylinder solid must have cylinder faces"
        );
        for cf in &cyl_faces {
            let params = &cf["surfaceParams"];
            assert!(params["origin"].as_array().unwrap().len() == 3);
            assert!(params["axis"].as_array().unwrap().len() == 3);
            assert!(params["refDir"].as_array().unwrap().len() == 3);
            let radius = params["radius"].as_f64().unwrap();
            assert!(
                (radius - 1.0).abs() < 1e-10,
                "cylinder radius should be 1.0, got {radius}"
            );
        }
    }

    #[test]
    fn to_brep_sphere_surface_params() {
        let mut k = crate::kernel::BrepKernel::new();
        let id = brepkit_operations::primitives::make_sphere(&mut k.topo, 3.0, 16).unwrap();
        #[allow(clippy::cast_possible_truncation)]
        let solid = id.index() as u32;
        let brep = build_brep_json(&k, solid);
        let faces = brep["faces"].as_array().unwrap();
        let sph_faces: Vec<_> = faces
            .iter()
            .filter(|f| f["surfaceType"].as_str().unwrap() == "sphere")
            .collect();
        assert!(!sph_faces.is_empty(), "sphere solid must have sphere faces");
        for sf in &sph_faces {
            let params = &sf["surfaceParams"];
            assert!(params["center"].as_array().unwrap().len() == 3);
            assert!(params["axis"].as_array().unwrap().len() == 3);
            let radius = params["radius"].as_f64().unwrap();
            assert!(
                (radius - 3.0).abs() < 1e-10,
                "sphere radius should be 3.0, got {radius}"
            );
        }
    }

    #[test]
    fn to_brep_cone_surface_params() {
        let mut k = crate::kernel::BrepKernel::new();
        let id = brepkit_operations::primitives::make_cone(&mut k.topo, 2.0, 0.5, 4.0).unwrap();
        #[allow(clippy::cast_possible_truncation)]
        let solid = id.index() as u32;
        let brep = build_brep_json(&k, solid);
        let faces = brep["faces"].as_array().unwrap();
        let cone_faces: Vec<_> = faces
            .iter()
            .filter(|f| f["surfaceType"].as_str().unwrap() == "cone")
            .collect();
        assert!(!cone_faces.is_empty(), "cone solid must have cone faces");
        for cf in &cone_faces {
            let params = &cf["surfaceParams"];
            assert!(params["apex"].as_array().unwrap().len() == 3);
            assert!(params["axis"].as_array().unwrap().len() == 3);
            assert!(params["refDir"].as_array().unwrap().len() == 3);
            let half_angle = params["halfAngle"].as_f64().unwrap();
            assert!(
                half_angle > 0.0 && half_angle < std::f64::consts::FRAC_PI_2,
                "cone halfAngle must be in (0, pi/2), got {half_angle}"
            );
        }
    }

    #[test]
    fn to_brep_torus_surface_params() {
        let mut k = crate::kernel::BrepKernel::new();
        let id = brepkit_operations::primitives::make_torus(&mut k.topo, 5.0, 1.0, 16).unwrap();
        #[allow(clippy::cast_possible_truncation)]
        let solid = id.index() as u32;
        let brep = build_brep_json(&k, solid);
        let faces = brep["faces"].as_array().unwrap();
        let tor_faces: Vec<_> = faces
            .iter()
            .filter(|f| f["surfaceType"].as_str().unwrap() == "torus")
            .collect();
        assert!(!tor_faces.is_empty(), "torus solid must have torus faces");
        for tf in &tor_faces {
            let params = &tf["surfaceParams"];
            assert!(params["center"].as_array().unwrap().len() == 3);
            assert!(params["axis"].as_array().unwrap().len() == 3);
            let major_r = params["majorRadius"].as_f64().unwrap();
            assert!(
                (major_r - 5.0).abs() < 1e-10,
                "torus major radius should be 5.0, got {major_r}"
            );
            let minor_r = params["minorRadius"].as_f64().unwrap();
            assert!(
                (minor_r - 1.0).abs() < 1e-10,
                "torus minor radius should be 1.0, got {minor_r}"
            );
        }
    }
}
