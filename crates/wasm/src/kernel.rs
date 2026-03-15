//! The `BrepKernel` — a WASM-exposed modeling context.
//!
//! JavaScript consumers create a single `BrepKernel` instance and call
//! methods on it to build and query geometry. All topological state is
//! owned by the kernel; JS only holds opaque `u32` handles.

#![allow(
    clippy::missing_errors_doc,
    clippy::too_many_arguments,
    clippy::redundant_closure,
    clippy::redundant_closure_for_method_calls,
    clippy::map_unwrap_or,
    clippy::expect_used,
    dead_code
)]

use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::curves2d::Line2D;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::surfaces::{
    ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
};
use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};
use wasm_bindgen::prelude::*;

use crate::error::{WasmError, validate_finite};
use crate::handles::{edge_id_to_u32, solid_id_to_u32};
use crate::helpers::TOL;
use crate::state::{Checkpoint, SketchState};

/// The B-Rep modeling kernel.
///
/// Owns all topological state. JavaScript holds this reference and
/// invokes methods to create, transform, and query geometry.
#[wasm_bindgen]
pub struct BrepKernel {
    pub(crate) topo: Topology,
    pub(crate) assemblies: Vec<brepkit_operations::assembly::Assembly>,
    pub(crate) sketches: Vec<SketchState>,
    pub(crate) checkpoints: Vec<Checkpoint>,
    pub(crate) poisoned: bool,
}

#[wasm_bindgen]
impl BrepKernel {
    /// Create a new, empty kernel.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            topo: Topology::new(),
            assemblies: Vec::new(),
            sketches: Vec::new(),
            checkpoints: Vec::new(),
            poisoned: false,
        }
    }
}

impl Default for BrepKernel {
    fn default() -> Self {
        Self::new()
    }
}

// ── Private helpers ────────────────────────────────────────────────

impl BrepKernel {
    /// Inner implementation for `make_tangent_arc_3d`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn make_tangent_arc_3d_impl(
        &mut self,
        start_x: f64,
        start_y: f64,
        start_z: f64,
        tangent_x: f64,
        tangent_y: f64,
        tangent_z: f64,
        end_x: f64,
        end_y: f64,
        end_z: f64,
    ) -> Result<u32, WasmError> {
        for (v, name) in [
            (start_x, "startX"),
            (start_y, "startY"),
            (start_z, "startZ"),
            (tangent_x, "tangentX"),
            (tangent_y, "tangentY"),
            (tangent_z, "tangentZ"),
            (end_x, "endX"),
            (end_y, "endY"),
            (end_z, "endZ"),
        ] {
            validate_finite(v, name)?;
        }

        let start = Point3::new(start_x, start_y, start_z);
        let end = Point3::new(end_x, end_y, end_z);
        let tangent = Vec3::new(tangent_x, tangent_y, tangent_z);

        let chord = end - start;
        if chord.length() < TOL {
            return Err(WasmError::InvalidInput {
                reason: "start and end points coincide".into(),
            });
        }

        let t_norm = tangent.normalize().map_err(|e| WasmError::InvalidInput {
            reason: format!("invalid tangent: {e}"),
        })?;

        // Tangent parallel to chord means the points are collinear.
        let cross = t_norm.cross(chord);
        if cross.length() < 1e-10 * chord.length() {
            let v_start = self.topo.add_vertex(Vertex::new(start, TOL));
            let v_end = self.topo.add_vertex(Vertex::new(end, TOL));
            let eid = self
                .topo
                .add_edge(Edge::new(v_start, v_end, EdgeCurve::Line));
            return Ok(edge_id_to_u32(eid));
        }

        // Arc geometry: find center and radius from the tangent constraint.
        let normal = cross.normalize().map_err(|e| WasmError::InvalidInput {
            reason: format!("degenerate arc plane: {e}"),
        })?;
        let perp = normal.cross(t_norm);
        let half_proj = chord.length_squared() / (2.0 * perp.dot(chord));
        let center = start + perp * half_proj;
        let radius = half_proj.abs();

        let u_axis = (start - center)
            .normalize()
            .map_err(|e| WasmError::InvalidInput {
                reason: format!("degenerate u_axis: {e}"),
            })?;
        let v_axis = normal.cross(u_axis);

        let circle = Circle3D::with_axes(center, normal, radius, u_axis, v_axis).map_err(|e| {
            WasmError::InvalidInput {
                reason: format!("invalid circle: {e}"),
            }
        })?;

        let v_start = self.topo.add_vertex(Vertex::new(start, TOL));
        let v_end = if (start - end).length() < TOL * 100.0 {
            v_start
        } else {
            self.topo.add_vertex(Vertex::new(end, TOL))
        };
        let eid = self
            .topo
            .add_edge(Edge::new(v_start, v_end, EdgeCurve::Circle(circle)));
        Ok(edge_id_to_u32(eid))
    }

    /// Inner implementation for `lift_curve2d_to_plane`.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(crate) fn lift_curve2d_to_plane_impl(
        &mut self,
        curve_type: u32,
        curve_params: Vec<f64>,
        origin_x: f64,
        origin_y: f64,
        origin_z: f64,
        x_axis_x: f64,
        x_axis_y: f64,
        x_axis_z: f64,
        normal_x: f64,
        normal_y: f64,
        normal_z: f64,
        t_start: f64,
        t_end: f64,
    ) -> Result<u32, WasmError> {
        validate_finite(origin_x, "originX")?;
        validate_finite(origin_y, "originY")?;
        validate_finite(origin_z, "originZ")?;
        validate_finite(x_axis_x, "xAxisX")?;
        validate_finite(x_axis_y, "xAxisY")?;
        validate_finite(x_axis_z, "xAxisZ")?;
        validate_finite(normal_x, "normalX")?;
        validate_finite(normal_y, "normalY")?;
        validate_finite(normal_z, "normalZ")?;
        validate_finite(t_start, "tStart")?;
        validate_finite(t_end, "tEnd")?;

        if curve_type > 3 {
            return Err(WasmError::InvalidInput {
                reason: format!("curve_type must be 0–3, got {curve_type}"),
            });
        }

        for (i, &v) in curve_params.iter().enumerate() {
            validate_finite(v, &format!("curveParams[{i}]"))?;
        }

        let normal = Vec3::new(normal_x, normal_y, normal_z)
            .normalize()
            .map_err(|e| WasmError::InvalidInput {
                reason: format!("invalid normal: {e}"),
            })?;
        let x_raw = Vec3::new(x_axis_x, x_axis_y, x_axis_z);
        let x_axis = (x_raw - normal * x_raw.dot(normal))
            .normalize()
            .map_err(|e| WasmError::InvalidInput {
                reason: format!("invalid x_axis (parallel to normal?): {e}"),
            })?;
        let y_axis = normal.cross(x_axis);
        let origin = Point3::new(origin_x, origin_y, origin_z);

        let lift = |x: f64, y: f64| -> Point3 { origin + x_axis * x + y_axis * y };

        match curve_type {
            0 => {
                if curve_params.len() != 4 {
                    return Err(WasmError::InvalidInput {
                        reason: format!(
                            "Line2D expects 4 params [ox,oy,dx,dy], got {}",
                            curve_params.len()
                        ),
                    });
                }
                let line2d = Line2D::new(
                    Point2::new(curve_params[0], curve_params[1]),
                    Vec2::new(curve_params[2], curve_params[3]),
                )
                .map_err(|e| WasmError::InvalidInput {
                    reason: format!("invalid Line2D: {e}"),
                })?;
                let p0 = line2d.evaluate(t_start);
                let p1 = line2d.evaluate(t_end);
                let start_3d = lift(p0.x(), p0.y());
                let end_3d = lift(p1.x(), p1.y());
                if (end_3d - start_3d).length() < TOL {
                    return Err(WasmError::InvalidInput {
                        reason: "degenerate line segment (start ≈ end)".into(),
                    });
                }
                let v_start = self.topo.add_vertex(Vertex::new(start_3d, TOL));
                let v_end = self.topo.add_vertex(Vertex::new(end_3d, TOL));
                let eid = self
                    .topo
                    .add_edge(Edge::new(v_start, v_end, EdgeCurve::Line));
                Ok(edge_id_to_u32(eid))
            }
            1 => {
                if curve_params.len() != 3 {
                    return Err(WasmError::InvalidInput {
                        reason: format!(
                            "Circle expects 3 params [cx,cy,r], got {}",
                            curve_params.len()
                        ),
                    });
                }
                let center_3d = lift(curve_params[0], curve_params[1]);
                let radius = curve_params[2];
                let circle = Circle3D::with_axes(center_3d, normal, radius, x_axis, y_axis)
                    .map_err(|e| WasmError::InvalidInput {
                        reason: format!("invalid Circle3D: {e}"),
                    })?;

                let start_3d = circle.evaluate(t_start);
                let end_3d = circle.evaluate(t_end);

                let full_circle = (t_end - t_start).abs() >= std::f64::consts::TAU - 1e-10;
                let v_start = self.topo.add_vertex(Vertex::new(start_3d, TOL));
                let v_end = if full_circle {
                    v_start
                } else {
                    self.topo.add_vertex(Vertex::new(end_3d, TOL))
                };
                let eid = self
                    .topo
                    .add_edge(Edge::new(v_start, v_end, EdgeCurve::Circle(circle)));
                Ok(edge_id_to_u32(eid))
            }
            2 => {
                if curve_params.len() != 5 {
                    return Err(WasmError::InvalidInput {
                        reason: format!(
                            "Ellipse expects 5 params [cx,cy,a,b,rot], got {}",
                            curve_params.len()
                        ),
                    });
                }
                let semi_major = curve_params[2];
                let semi_minor = curve_params[3];
                let rotation = curve_params[4];

                let center_3d = lift(curve_params[0], curve_params[1]);
                let (sin_r, cos_r) = rotation.sin_cos();
                let u3d = x_axis * cos_r + y_axis * sin_r;
                let v3d = y_axis * cos_r - x_axis * sin_r;
                let ellipse =
                    Ellipse3D::with_axes(center_3d, normal, semi_major, semi_minor, u3d, v3d)
                        .map_err(|e| WasmError::InvalidInput {
                            reason: format!("invalid Ellipse3D: {e}"),
                        })?;

                let start_3d = ellipse.evaluate(t_start);
                let end_3d = ellipse.evaluate(t_end);

                let full_ellipse = (t_end - t_start).abs() >= std::f64::consts::TAU - 1e-10;
                let v_start = self.topo.add_vertex(Vertex::new(start_3d, TOL));
                let v_end = if full_ellipse {
                    v_start
                } else {
                    self.topo.add_vertex(Vertex::new(end_3d, TOL))
                };
                let eid =
                    self.topo
                        .add_edge(Edge::new(v_start, v_end, EdgeCurve::Ellipse(ellipse)));
                Ok(edge_id_to_u32(eid))
            }
            3 => {
                if curve_params.len() < 2 {
                    return Err(WasmError::InvalidInput {
                        reason: "NURBS params too short (need at least degree + n_cp)".into(),
                    });
                }
                let raw_degree = curve_params[0];
                let raw_n_cp = curve_params[1];
                if !(1.0..=16.0).contains(&raw_degree) || raw_degree.fract() != 0.0 {
                    return Err(WasmError::InvalidInput {
                        reason: format!(
                            "NURBS degree must be an integer in [1, 16], got {raw_degree}"
                        ),
                    });
                }
                if !(1.0..=4096.0).contains(&raw_n_cp) || raw_n_cp.fract() != 0.0 {
                    return Err(WasmError::InvalidInput {
                        reason: format!(
                            "NURBS n_cp must be an integer in [1, 4096], got {raw_n_cp}"
                        ),
                    });
                }
                #[allow(clippy::cast_possible_truncation)]
                let degree = raw_degree as usize;
                #[allow(clippy::cast_possible_truncation)]
                let n_cp = raw_n_cp as usize;
                let n_knots = n_cp + degree + 1;
                let expected_len = 2 + n_knots + 3 * n_cp;
                if curve_params.len() != expected_len {
                    return Err(WasmError::InvalidInput {
                        reason: format!(
                            "NURBS params: expected {expected_len} elements \
                             (2 + {n_knots} knots + {} coords + {n_cp} weights), got {}",
                            2 * n_cp,
                            curve_params.len()
                        ),
                    });
                }
                let knots = curve_params[2..2 + n_knots].to_vec();
                let coords_start = 2 + n_knots;
                let weights_start = coords_start + 2 * n_cp;
                let control_points_3d: Vec<Point3> = curve_params[coords_start..weights_start]
                    .chunks_exact(2)
                    .map(|c| lift(c[0], c[1]))
                    .collect();
                let weights = curve_params[weights_start..weights_start + n_cp].to_vec();

                let curve = NurbsCurve::new(degree, knots, control_points_3d, weights)?;
                let start_3d = curve.evaluate(t_start);
                let end_3d = curve.evaluate(t_end);

                let v_start = self.topo.add_vertex(Vertex::new(start_3d, TOL));
                let v_end = if (start_3d - end_3d).length() < TOL * 100.0 {
                    v_start
                } else {
                    self.topo.add_vertex(Vertex::new(end_3d, TOL))
                };
                let eid =
                    self.topo
                        .add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(curve)));
                Ok(edge_id_to_u32(eid))
            }
            _ => Err(WasmError::InvalidInput {
                reason: format!("curve_type must be 0–3, got {curve_type}"),
            }),
        }
    }

    /// Build a closed planar face from an ordered sequence of points.
    pub(crate) fn make_planar_face(
        &mut self,
        points: &[Point3],
    ) -> Result<brepkit_topology::face::FaceId, WasmError> {
        let n = points.len();

        let verts: Vec<_> = points
            .iter()
            .map(|p| self.topo.add_vertex(Vertex::new(*p, TOL)))
            .collect();

        let edges: Vec<_> = (0..n)
            .map(|i| {
                let next = (i + 1) % n;
                self.topo
                    .add_edge(Edge::new(verts[i], verts[next], EdgeCurve::Line))
            })
            .collect();

        let oriented: Vec<_> = edges
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let wire = Wire::new(oriented, true)?;
        let wid = self.topo.add_wire(wire);

        let a = points[1] - points[0];
        let b = points[2] - points[0];
        let normal = a.cross(b).normalize()?;

        let d = normal.x().mul_add(
            points[0].x(),
            normal
                .y()
                .mul_add(points[0].y(), normal.z() * points[0].z()),
        );

        let face_id = self
            .topo
            .add_face(Face::new(wid, vec![], FaceSurface::Plane { normal, d }));

        Ok(face_id)
    }

    /// Compute the v-range for an analytic surface by projecting face wire
    /// vertices onto the surface axis.
    pub(crate) fn compute_axial_v_range(
        &self,
        face_id: brepkit_topology::face::FaceId,
        origin: Point3,
        axis: Vec3,
    ) -> Result<(f64, f64), JsError> {
        let face_data = self.topo.face(face_id)?;
        let wire = self.topo.wire(face_data.outer_wire())?;

        let mut v_min = f64::MAX;
        let mut v_max = f64::MIN;

        for oe in wire.edges() {
            let edge = self.topo.edge(oe.edge())?;
            for vid in [edge.start(), edge.end()] {
                let pt = self.topo.vertex(vid)?.point();
                let to_pt = Vec3::new(
                    pt.x() - origin.x(),
                    pt.y() - origin.y(),
                    pt.z() - origin.z(),
                );
                let v = axis.dot(to_pt);
                v_min = v_min.min(v);
                v_max = v_max.max(v);
            }
        }

        if v_min < v_max {
            Ok((v_min, v_max))
        } else {
            Ok((-1.0, 1.0))
        }
    }

    /// Compute a plane surface from the vertices of a wire.
    ///
    /// Uses the first three non-collinear vertex positions of the wire's
    /// edges to derive a plane normal and signed distance `d`.
    fn compute_plane_from_wire(
        &self,
        wire_id: brepkit_topology::wire::WireId,
    ) -> Result<FaceSurface, WasmError> {
        let wire = self.topo.wire(wire_id)?;
        let mut points = Vec::new();
        for oe in wire.edges() {
            let edge = self.topo.edge(oe.edge())?;
            let start_pos = self.topo.vertex(edge.start())?.point();
            points.push(start_pos);
        }
        if points.len() < 3 {
            return Err(WasmError::InvalidInput {
                reason: "need at least 3 vertices to compute a plane".into(),
            });
        }
        let e1 = points[1] - points[0];
        let e2 = points[2] - points[0];
        let normal = e1.cross(e2).normalize()?;
        let p0 = points[0];
        let d = normal
            .x()
            .mul_add(p0.x(), normal.y().mul_add(p0.y(), normal.z() * p0.z()));
        Ok(FaceSurface::Plane { normal, d })
    }

    /// Parse a JSON array of 3 floats into a `Vec3`.
    fn parse_vec3(arr: &[serde_json::Value], name: &str) -> Result<Vec3, WasmError> {
        let x = arr
            .first()
            .and_then(|v| v.as_f64())
            .ok_or_else(|| WasmError::InvalidInput {
                reason: format!("{name}[0] is not a number"),
            })?;
        let y = arr
            .get(1)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| WasmError::InvalidInput {
                reason: format!("{name}[1] is not a number"),
            })?;
        let z = arr
            .get(2)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| WasmError::InvalidInput {
                reason: format!("{name}[2] is not a number"),
            })?;
        Ok(Vec3::new(x, y, z))
    }

    /// Parse a JSON array of 3 floats into a `Point3`.
    fn parse_point3(arr: &[serde_json::Value], name: &str) -> Result<Point3, WasmError> {
        let v = Self::parse_vec3(arr, name)?;
        Ok(Point3::new(v.x(), v.y(), v.z()))
    }

    /// Internal implementation of `fromBREP` that returns `WasmError`
    /// for easier testing in native (non-WASM) contexts.
    #[allow(clippy::too_many_lines, clippy::wrong_self_convention)]
    pub(crate) fn from_brep_impl(&mut self, json: &str) -> Result<u32, WasmError> {
        let parsed: serde_json::Value =
            serde_json::from_str(json).map_err(|e| WasmError::InvalidInput {
                reason: format!("invalid BREP JSON: {e}"),
            })?;

        // 1. Reconstruct vertices
        let vertices = parsed["vertices"]
            .as_array()
            .ok_or_else(|| WasmError::InvalidInput {
                reason: "missing vertices array".into(),
            })?;
        let mut vertex_map: std::collections::HashMap<u32, brepkit_topology::vertex::VertexId> =
            std::collections::HashMap::new();

        for v in vertices {
            let id = v["id"].as_u64().ok_or_else(|| WasmError::InvalidInput {
                reason: "vertex missing id".into(),
            })? as u32;
            let pos = v["position"]
                .as_array()
                .ok_or_else(|| WasmError::InvalidInput {
                    reason: "vertex missing position".into(),
                })?;
            let x =
                pos.first()
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| WasmError::InvalidInput {
                        reason: "invalid vertex x coordinate".into(),
                    })?;
            let y = pos
                .get(1)
                .and_then(|v| v.as_f64())
                .ok_or_else(|| WasmError::InvalidInput {
                    reason: "invalid vertex y coordinate".into(),
                })?;
            let z = pos
                .get(2)
                .and_then(|v| v.as_f64())
                .ok_or_else(|| WasmError::InvalidInput {
                    reason: "invalid vertex z coordinate".into(),
                })?;
            let vid = self.topo.add_vertex(Vertex::new(Point3::new(x, y, z), TOL));
            vertex_map.insert(id, vid);
        }

        // 2. Reconstruct edges (line edges from start/end vertices)
        let edges = parsed["edges"]
            .as_array()
            .ok_or_else(|| WasmError::InvalidInput {
                reason: "missing edges array".into(),
            })?;
        let mut edge_map: std::collections::HashMap<u32, brepkit_topology::edge::EdgeId> =
            std::collections::HashMap::new();

        for e in edges {
            let id = e["id"].as_u64().ok_or_else(|| WasmError::InvalidInput {
                reason: "edge missing id".into(),
            })? as u32;
            let start_v = e["startVertex"]
                .as_u64()
                .ok_or_else(|| WasmError::InvalidInput {
                    reason: "edge missing startVertex".into(),
                })? as u32;
            let end_v = e["endVertex"]
                .as_u64()
                .ok_or_else(|| WasmError::InvalidInput {
                    reason: "edge missing endVertex".into(),
                })? as u32;

            let start_vid = *vertex_map
                .get(&start_v)
                .ok_or_else(|| WasmError::InvalidInput {
                    reason: format!("edge {id} references unknown start vertex {start_v}"),
                })?;
            let end_vid = *vertex_map
                .get(&end_v)
                .ok_or_else(|| WasmError::InvalidInput {
                    reason: format!("edge {id} references unknown end vertex {end_v}"),
                })?;

            let curve_type = e["curveType"].as_str().unwrap_or("line");
            let params = e.get("curveParams").and_then(|p| p.as_object());
            let curve = match curve_type {
                "line" => EdgeCurve::Line,
                "circle" => {
                    let p = params.ok_or_else(|| WasmError::InvalidInput {
                        reason: format!("edge {id}: circle missing curveParams"),
                    })?;
                    let center_arr =
                        p.get("center").and_then(|v| v.as_array()).ok_or_else(|| {
                            WasmError::InvalidInput {
                                reason: format!("edge {id}: circle missing center"),
                            }
                        })?;
                    let axis_arr = p.get("axis").and_then(|v| v.as_array()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: format!("edge {id}: circle missing axis"),
                        }
                    })?;
                    let radius = p.get("radius").and_then(|v| v.as_f64()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: format!("edge {id}: circle missing radius"),
                        }
                    })?;
                    let center = Self::parse_point3(center_arr, "center")?;
                    let axis = Self::parse_vec3(axis_arr, "axis")?;
                    // Use xAxis if available for exact round-trip, else derive from axis
                    let circle = if let Some(x_axis_arr) = p.get("xAxis").and_then(|v| v.as_array())
                    {
                        let x_axis = Self::parse_vec3(x_axis_arr, "xAxis")?;
                        let v_axis = axis.cross(x_axis);
                        Circle3D::with_axes(center, axis, radius, x_axis, v_axis)?
                    } else {
                        Circle3D::new(center, axis, radius)?
                    };
                    EdgeCurve::Circle(circle)
                }
                "ellipse" => {
                    let p = params.ok_or_else(|| WasmError::InvalidInput {
                        reason: format!("edge {id}: ellipse missing curveParams"),
                    })?;
                    let center_arr =
                        p.get("center").and_then(|v| v.as_array()).ok_or_else(|| {
                            WasmError::InvalidInput {
                                reason: format!("edge {id}: ellipse missing center"),
                            }
                        })?;
                    let axis_arr = p.get("axis").and_then(|v| v.as_array()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: format!("edge {id}: ellipse missing axis"),
                        }
                    })?;
                    let major_r =
                        p.get("majorRadius")
                            .and_then(|v| v.as_f64())
                            .ok_or_else(|| WasmError::InvalidInput {
                                reason: format!("edge {id}: ellipse missing majorRadius"),
                            })?;
                    let minor_r =
                        p.get("minorRadius")
                            .and_then(|v| v.as_f64())
                            .ok_or_else(|| WasmError::InvalidInput {
                                reason: format!("edge {id}: ellipse missing minorRadius"),
                            })?;
                    let center = Self::parse_point3(center_arr, "center")?;
                    let axis = Self::parse_vec3(axis_arr, "axis")?;
                    // Use majorAxis if available for exact round-trip
                    let ellipse =
                        if let Some(major_arr) = p.get("majorAxis").and_then(|v| v.as_array()) {
                            let u_axis = Self::parse_vec3(major_arr, "majorAxis")?;
                            let v_axis = axis.cross(u_axis);
                            Ellipse3D::with_axes(center, axis, major_r, minor_r, u_axis, v_axis)?
                        } else {
                            Ellipse3D::new(center, axis, major_r, minor_r)?
                        };
                    EdgeCurve::Ellipse(ellipse)
                }
                "nurbs" => {
                    let p = params.ok_or_else(|| WasmError::InvalidInput {
                        reason: format!("edge {id}: nurbs missing curveParams"),
                    })?;
                    let degree = p.get("degree").and_then(|v| v.as_u64()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: format!("edge {id}: nurbs missing degree"),
                        }
                    })? as usize;
                    let knots_arr = p.get("knots").and_then(|v| v.as_array()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: format!("edge {id}: nurbs missing knots"),
                        }
                    })?;
                    let knots: Vec<f64> = knots_arr
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            v.as_f64().ok_or_else(|| WasmError::InvalidInput {
                                reason: format!("edge {id}: knot[{i}] is not a number"),
                            })
                        })
                        .collect::<Result<_, _>>()?;
                    let weights_arr =
                        p.get("weights").and_then(|v| v.as_array()).ok_or_else(|| {
                            WasmError::InvalidInput {
                                reason: format!("edge {id}: nurbs missing weights"),
                            }
                        })?;
                    let weights: Vec<f64> = weights_arr
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            v.as_f64().ok_or_else(|| WasmError::InvalidInput {
                                reason: format!("edge {id}: weight[{i}] is not a number"),
                            })
                        })
                        .collect::<Result<_, _>>()?;
                    let cps_arr = p
                        .get("controlPoints")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| WasmError::InvalidInput {
                            reason: format!("edge {id}: nurbs missing controlPoints"),
                        })?;
                    let control_points: Vec<Point3> = cps_arr
                        .iter()
                        .enumerate()
                        .map(|(i, cp)| -> Result<Point3, WasmError> {
                            let arr = cp.as_array().ok_or_else(|| WasmError::InvalidInput {
                                reason: format!("edge {id}: controlPoints[{i}] is not an array"),
                            })?;
                            Self::parse_point3(arr, &format!("controlPoints[{i}]"))
                        })
                        .collect::<Result<_, _>>()?;
                    let nc = NurbsCurve::new(degree, knots, control_points, weights)?;
                    EdgeCurve::NurbsCurve(nc)
                }
                other => {
                    log::warn!(
                        "fromBREP: edge {id} has unsupported curve type '{other}', \
                         approximating as line"
                    );
                    EdgeCurve::Line
                }
            };

            let eid = self.topo.add_edge(Edge::new(start_vid, end_vid, curve));
            edge_map.insert(id, eid);
        }

        // 3. Reconstruct faces
        let faces = parsed["faces"]
            .as_array()
            .ok_or_else(|| WasmError::InvalidInput {
                reason: "missing faces array".into(),
            })?;
        let mut face_ids: Vec<brepkit_topology::face::FaceId> = Vec::new();

        for f in faces {
            let outer_edge_ids =
                f["outerWireEdges"]
                    .as_array()
                    .ok_or_else(|| WasmError::InvalidInput {
                        reason: "face missing outerWireEdges".into(),
                    })?;
            let outer_orientations = f
                .get("outerWireOrientations")
                .and_then(|v| v.as_array().cloned());

            // Build oriented edges for the outer wire
            let mut oriented_edges = Vec::new();
            for (i, eid_val) in outer_edge_ids.iter().enumerate() {
                let eid = eid_val.as_u64().ok_or_else(|| WasmError::InvalidInput {
                    reason: "invalid edge id in outerWireEdges".into(),
                })? as u32;
                let edge_id = *edge_map.get(&eid).ok_or_else(|| WasmError::InvalidInput {
                    reason: format!("wire references unknown edge {eid}"),
                })?;
                let forward = outer_orientations
                    .as_ref()
                    .and_then(|arr| arr.get(i))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                oriented_edges.push(OrientedEdge::new(edge_id, forward));
            }

            let wire = Wire::new(oriented_edges, true)?;
            let wire_id = self.topo.add_wire(wire);

            // Reconstruct surface
            let surface_type = f["surfaceType"].as_str().unwrap_or("plane");
            let reversed = f["reversed"].as_bool().unwrap_or(false);

            let surface_params = f.get("surfaceParams").and_then(|p| p.as_object());
            let surface = match surface_type {
                "plane" => {
                    if let Some(params) = surface_params {
                        if let (Some(normal_arr), Some(d)) = (
                            params.get("normal").and_then(|n| n.as_array()),
                            params.get("d").and_then(|d| d.as_f64()),
                        ) {
                            let nx = normal_arr.first().and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let ny = normal_arr.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let nz = normal_arr.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0);
                            FaceSurface::Plane {
                                normal: Vec3::new(nx, ny, nz),
                                d,
                            }
                        } else {
                            self.compute_plane_from_wire(wire_id)?
                        }
                    } else {
                        self.compute_plane_from_wire(wire_id)?
                    }
                }
                "cylinder" => {
                    let p = surface_params.ok_or_else(|| WasmError::InvalidInput {
                        reason: "cylinder face missing surfaceParams".into(),
                    })?;
                    let origin_arr =
                        p.get("origin").and_then(|v| v.as_array()).ok_or_else(|| {
                            WasmError::InvalidInput {
                                reason: "cylinder missing origin".into(),
                            }
                        })?;
                    let axis_arr = p.get("axis").and_then(|v| v.as_array()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: "cylinder missing axis".into(),
                        }
                    })?;
                    let radius = p.get("radius").and_then(|v| v.as_f64()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: "cylinder missing radius".into(),
                        }
                    })?;
                    let origin = Self::parse_point3(origin_arr, "origin")?;
                    let axis = Self::parse_vec3(axis_arr, "axis")?;
                    let cyl = if let Some(ref_arr) = p.get("refDir").and_then(|v| v.as_array()) {
                        let ref_dir = Self::parse_vec3(ref_arr, "refDir")?;
                        CylindricalSurface::with_ref_dir(origin, axis, radius, ref_dir)?
                    } else {
                        CylindricalSurface::new(origin, axis, radius)?
                    };
                    FaceSurface::Cylinder(cyl)
                }
                "cone" => {
                    let p = surface_params.ok_or_else(|| WasmError::InvalidInput {
                        reason: "cone face missing surfaceParams".into(),
                    })?;
                    let apex_arr = p.get("apex").and_then(|v| v.as_array()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: "cone missing apex".into(),
                        }
                    })?;
                    let axis_arr = p.get("axis").and_then(|v| v.as_array()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: "cone missing axis".into(),
                        }
                    })?;
                    let half_angle =
                        p.get("halfAngle").and_then(|v| v.as_f64()).ok_or_else(|| {
                            WasmError::InvalidInput {
                                reason: "cone missing halfAngle".into(),
                            }
                        })?;
                    let apex = Self::parse_point3(apex_arr, "apex")?;
                    let axis = Self::parse_vec3(axis_arr, "axis")?;
                    let cone = if let Some(ref_arr) = p.get("refDir").and_then(|v| v.as_array()) {
                        let ref_dir = Self::parse_vec3(ref_arr, "refDir")?;
                        ConicalSurface::with_ref_dir(apex, axis, half_angle, ref_dir)?
                    } else {
                        ConicalSurface::new(apex, axis, half_angle)?
                    };
                    FaceSurface::Cone(cone)
                }
                "sphere" => {
                    let p = surface_params.ok_or_else(|| WasmError::InvalidInput {
                        reason: "sphere face missing surfaceParams".into(),
                    })?;
                    let center_arr =
                        p.get("center").and_then(|v| v.as_array()).ok_or_else(|| {
                            WasmError::InvalidInput {
                                reason: "sphere missing center".into(),
                            }
                        })?;
                    let radius = p.get("radius").and_then(|v| v.as_f64()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: "sphere missing radius".into(),
                        }
                    })?;
                    let center = Self::parse_point3(center_arr, "center")?;
                    let sphere = SphericalSurface::new(center, radius)?;
                    FaceSurface::Sphere(sphere)
                }
                "torus" => {
                    let p = surface_params.ok_or_else(|| WasmError::InvalidInput {
                        reason: "torus face missing surfaceParams".into(),
                    })?;
                    let center_arr =
                        p.get("center").and_then(|v| v.as_array()).ok_or_else(|| {
                            WasmError::InvalidInput {
                                reason: "torus missing center".into(),
                            }
                        })?;
                    let axis_arr = p.get("axis").and_then(|v| v.as_array()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: "torus missing axis".into(),
                        }
                    })?;
                    let major_r =
                        p.get("majorRadius")
                            .and_then(|v| v.as_f64())
                            .ok_or_else(|| WasmError::InvalidInput {
                                reason: "torus missing majorRadius".into(),
                            })?;
                    let minor_r =
                        p.get("minorRadius")
                            .and_then(|v| v.as_f64())
                            .ok_or_else(|| WasmError::InvalidInput {
                                reason: "torus missing minorRadius".into(),
                            })?;
                    let center = Self::parse_point3(center_arr, "center")?;
                    let axis = Self::parse_vec3(axis_arr, "axis")?;
                    let torus = ToroidalSurface::with_axis(center, major_r, minor_r, axis)?;
                    FaceSurface::Torus(torus)
                }
                "nurbs" => {
                    let p = surface_params.ok_or_else(|| WasmError::InvalidInput {
                        reason: "nurbs face missing surfaceParams".into(),
                    })?;
                    let degree_u = p.get("degreeU").and_then(|v| v.as_u64()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: "nurbs surface missing degreeU".into(),
                        }
                    })? as usize;
                    let degree_v = p.get("degreeV").and_then(|v| v.as_u64()).ok_or_else(|| {
                        WasmError::InvalidInput {
                            reason: "nurbs surface missing degreeV".into(),
                        }
                    })? as usize;
                    let knots_u_arr =
                        p.get("knotsU").and_then(|v| v.as_array()).ok_or_else(|| {
                            WasmError::InvalidInput {
                                reason: "nurbs surface missing knotsU".into(),
                            }
                        })?;
                    let knots_u: Vec<f64> = knots_u_arr
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            v.as_f64().ok_or_else(|| WasmError::InvalidInput {
                                reason: format!("nurbs surface knotsU[{i}] is not a number"),
                            })
                        })
                        .collect::<Result<_, _>>()?;
                    let knots_v_arr =
                        p.get("knotsV").and_then(|v| v.as_array()).ok_or_else(|| {
                            WasmError::InvalidInput {
                                reason: "nurbs surface missing knotsV".into(),
                            }
                        })?;
                    let knots_v: Vec<f64> = knots_v_arr
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            v.as_f64().ok_or_else(|| WasmError::InvalidInput {
                                reason: format!("nurbs surface knotsV[{i}] is not a number"),
                            })
                        })
                        .collect::<Result<_, _>>()?;
                    let cps_grid = p
                        .get("controlPoints")
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| WasmError::InvalidInput {
                            reason: "nurbs surface missing controlPoints".into(),
                        })?;
                    let control_points: Vec<Vec<Point3>> = cps_grid
                        .iter()
                        .enumerate()
                        .map(|(ri, row)| -> Result<Vec<Point3>, WasmError> {
                            let row_arr =
                                row.as_array().ok_or_else(|| WasmError::InvalidInput {
                                    reason: format!("nurbs controlPoints[{ri}] is not an array"),
                                })?;
                            row_arr
                                .iter()
                                .enumerate()
                                .map(|(ci, cp)| -> Result<Point3, WasmError> {
                                    let arr =
                                        cp.as_array().ok_or_else(|| WasmError::InvalidInput {
                                            reason: format!(
                                                "nurbs controlPoints[{ri}][{ci}] is not an array"
                                            ),
                                        })?;
                                    Self::parse_point3(arr, &format!("controlPoints[{ri}][{ci}]"))
                                })
                                .collect()
                        })
                        .collect::<Result<_, _>>()?;
                    let weights_grid =
                        p.get("weights").and_then(|v| v.as_array()).ok_or_else(|| {
                            WasmError::InvalidInput {
                                reason: "nurbs surface missing weights".into(),
                            }
                        })?;
                    let weights: Vec<Vec<f64>> = weights_grid
                        .iter()
                        .enumerate()
                        .map(|(ri, row)| -> Result<Vec<f64>, WasmError> {
                            let row_arr =
                                row.as_array().ok_or_else(|| WasmError::InvalidInput {
                                    reason: format!("nurbs weights[{ri}] is not an array"),
                                })?;
                            row_arr
                                .iter()
                                .enumerate()
                                .map(|(ci, v)| {
                                    v.as_f64().ok_or_else(|| WasmError::InvalidInput {
                                        reason: format!(
                                            "nurbs surface weights[{ri}][{ci}] is not a number"
                                        ),
                                    })
                                })
                                .collect::<Result<_, _>>()
                        })
                        .collect::<Result<_, _>>()?;
                    let ns = NurbsSurface::new(
                        degree_u,
                        degree_v,
                        knots_u,
                        knots_v,
                        control_points,
                        weights,
                    )?;
                    FaceSurface::Nurbs(ns)
                }
                other => {
                    log::warn!(
                        "fromBREP: face has unsupported surface type '{other}', \
                         approximating as plane from vertices"
                    );
                    self.compute_plane_from_wire(wire_id)?
                }
            };

            // Handle inner wires (holes)
            let mut inner_wire_ids = Vec::new();
            if let Some(inner_wires) = f["innerWires"].as_array() {
                for iw in inner_wires {
                    // Support both old format (array of edge IDs) and new format
                    // (object with "edges" and "orientations")
                    let (edge_arr, orient_arr) = if let Some(obj) = iw.as_object() {
                        let e = obj
                            .get("edges")
                            .and_then(|v| v.as_array())
                            .cloned()
                            .unwrap_or_default();
                        let o = obj.get("orientations").and_then(|v| v.as_array()).cloned();
                        (e, o)
                    } else if let Some(arr) = iw.as_array() {
                        (arr.clone(), None)
                    } else {
                        continue;
                    };

                    let mut inner_oriented = Vec::new();
                    for (i, eid_val) in edge_arr.iter().enumerate() {
                        if let Some(eid) = eid_val.as_u64() {
                            if let Some(&edge_id) = edge_map.get(&(eid as u32)) {
                                let fwd = orient_arr
                                    .as_ref()
                                    .and_then(|arr| arr.get(i))
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(true);
                                inner_oriented.push(OrientedEdge::new(edge_id, fwd));
                            }
                        }
                    }
                    if !inner_oriented.is_empty() {
                        let inner_wire = Wire::new(inner_oriented, true)?;
                        inner_wire_ids.push(self.topo.add_wire(inner_wire));
                    }
                }
            }

            let mut face = Face::new(wire_id, inner_wire_ids, surface);
            if reversed {
                face.set_reversed(true);
            }

            face_ids.push(self.topo.add_face(face));
        }

        // 4. Build shell and solid
        if face_ids.is_empty() {
            return Err(WasmError::InvalidInput {
                reason: "fromBREP: no faces reconstructed".into(),
            });
        }

        let shell = brepkit_topology::shell::Shell::new(face_ids)?;
        let shell_id = self.topo.add_shell(shell);
        let solid = brepkit_topology::solid::Solid::new(shell_id, vec![]);
        let solid_id = self.topo.add_solid(solid);

        Ok(solid_id_to_u32(solid_id))
    }
}

// ── Test fixtures ─────────────────────────────────────────────────

#[cfg(test)]
pub(crate) mod test_fixtures {
    #![allow(clippy::unwrap_used)]
    use super::*;

    pub fn kernel_with_box() -> (BrepKernel, u32) {
        let mut k = BrepKernel::new();
        let id = brepkit_operations::primitives::make_box(&mut k.topo, 1.0, 1.0, 1.0).unwrap();
        #[allow(clippy::cast_possible_truncation)]
        (k, id.index() as u32)
    }

    pub fn kernel_with_two_boxes() -> (BrepKernel, u32, u32) {
        let mut k = BrepKernel::new();
        let a = brepkit_operations::primitives::make_box(&mut k.topo, 2.0, 2.0, 2.0).unwrap();
        let b = brepkit_operations::primitives::make_box(&mut k.topo, 1.0, 1.0, 1.0).unwrap();
        #[allow(clippy::cast_possible_truncation)]
        (k, a.index() as u32, b.index() as u32)
    }

    pub fn kernel_with_cylinder() -> (BrepKernel, u32) {
        let mut k = BrepKernel::new();
        let id = brepkit_operations::primitives::make_cylinder(&mut k.topo, 1.0, 2.0).unwrap();
        #[allow(clippy::cast_possible_truncation)]
        (k, id.index() as u32)
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod batch_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn batch_single_op() {
        let mut kernel = BrepKernel::new();
        let result = kernel
            .execute_batch(r#"[{"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}}]"#);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed[0]["ok"].is_number(),
            "expected ok number, got {parsed}"
        );
    }

    #[test]
    fn batch_multiple_ops() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 2, "height": 2, "depth": 2}},
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "volume", "args": {"solid": 0}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 3);
        assert!(parsed[0]["ok"].is_number());
        assert!(parsed[1]["ok"].is_number());
        assert!(parsed[2]["ok"].is_number());
    }

    #[test]
    fn batch_error_doesnt_stop_rest() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch(
            r#"[
                {"op": "unknownOp", "args": {}},
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed[0]["error"].is_string());
        assert!(parsed[1]["ok"].is_number());
    }

    #[test]
    fn batch_invalid_json() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch("not valid json");
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed[0]["error"]
                .as_str()
                .unwrap()
                .contains("invalid JSON")
        );
    }

    #[test]
    fn batch_missing_op_field() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch(r#"[{"args": {"width": 1}}]"#);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed[0]["error"].as_str().unwrap().contains("op"));
    }

    #[test]
    fn batch_boolean_ops() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 2, "height": 2, "depth": 2}},
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "fuse", "args": {"solidA": 0, "solidB": 1}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed[0]["ok"].is_number());
        assert!(parsed[1]["ok"].is_number());
        assert!(parsed[2]["ok"].is_number());
    }

    #[test]
    fn batch_bounding_box() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 2, "height": 4, "depth": 6}},
                {"op": "boundingBox", "args": {"solid": 0}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed[0]["ok"].is_number());
        let bbox = &parsed[1]["ok"];
        assert!(bbox.is_array());
        assert_eq!(bbox.as_array().unwrap().len(), 6);
    }

    #[test]
    fn batch_copy_solid() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "copySolid", "args": {"solid": 0}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed[0]["ok"].is_number());
        assert!(parsed[1]["ok"].is_number());
        assert_ne!(parsed[0]["ok"].as_u64(), parsed[1]["ok"].as_u64());
    }
}

#[cfg(test)]
mod tangent_arc_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn get_edge(k: &BrepKernel, handle: u32) -> &Edge {
        let id = k.resolve_edge(handle).unwrap();
        k.topo.edge(id).unwrap()
    }

    #[test]
    fn semicircle() {
        let mut k = BrepKernel::new();
        let eid = k
            .make_tangent_arc_3d_impl(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
            .unwrap();
        let edge = get_edge(&k, eid);
        assert!(matches!(edge.curve(), EdgeCurve::Circle(_)));
        if let EdgeCurve::Circle(c) = edge.curve() {
            assert!((c.radius() - 1.0).abs() < 1e-10);
            let center = c.center();
            assert!(center.x().abs() < 1e-10);
            assert!(center.y().abs() < 1e-10);
            assert!(center.z().abs() < 1e-10);
        }
    }

    #[test]
    fn quarter_circle() {
        let mut k = BrepKernel::new();
        let eid = k
            .make_tangent_arc_3d_impl(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0)
            .unwrap();
        let edge = get_edge(&k, eid);
        assert!(matches!(edge.curve(), EdgeCurve::Circle(_)));
        if let EdgeCurve::Circle(c) = edge.curve() {
            assert!((c.radius() - 1.0).abs() < 1e-10);
        }
        let s = k.topo.vertex(edge.start()).unwrap().point();
        let e = k.topo.vertex(edge.end()).unwrap().point();
        assert!((s.x() - 1.0).abs() < 1e-10);
        assert!((e.y() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tilted_3d_arc() {
        let mut k = BrepKernel::new();
        let eid = k
            .make_tangent_arc_3d_impl(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0)
            .unwrap();
        let edge = get_edge(&k, eid);
        assert!(matches!(edge.curve(), EdgeCurve::Circle(_)));
        if let EdgeCurve::Circle(c) = edge.curve() {
            assert!((c.radius() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn collinear_fallback() {
        let mut k = BrepKernel::new();
        let eid = k
            .make_tangent_arc_3d_impl(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0)
            .unwrap();
        assert!(matches!(get_edge(&k, eid).curve(), EdgeCurve::Line));
    }

    #[test]
    fn large_arc_gt_pi() {
        let mut k = BrepKernel::new();
        let eid = k
            .make_tangent_arc_3d_impl(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0)
            .unwrap();
        assert!(matches!(get_edge(&k, eid).curve(), EdgeCurve::Circle(_)));
    }

    #[test]
    fn coincident_points_error() {
        let mut k = BrepKernel::new();
        let err = k
            .make_tangent_arc_3d_impl(1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0)
            .unwrap_err();
        assert!(err.to_string().contains("coincide"));
    }

    #[test]
    fn zero_tangent_error() {
        let mut k = BrepKernel::new();
        let err = k
            .make_tangent_arc_3d_impl(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            .unwrap_err();
        assert!(err.to_string().contains("tangent"));
    }
}

#[cfg(test)]
mod lift_curve2d_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    #[test]
    fn line2d_on_xy_plane() {
        let mut k = BrepKernel::new();
        let eid = k
            .lift_curve2d_to_plane_impl(
                0,
                vec![1.0, 0.0, 1.0, 0.0],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                3.0,
            )
            .unwrap();
        let edge_id = k.resolve_edge(eid).unwrap();
        let edge = k.topo.edge(edge_id).unwrap();
        let s = k.topo.vertex(edge.start()).unwrap().point();
        let e = k.topo.vertex(edge.end()).unwrap().point();
        assert!((s.x() - 1.0).abs() < 1e-10);
        assert!(s.y().abs() < 1e-10);
        assert!((e.x() - 4.0).abs() < 1e-10);
        assert!(e.y().abs() < 1e-10);
        assert!(matches!(edge.curve(), EdgeCurve::Line));
    }

    #[test]
    fn circle2d_quarter_arc_xy() {
        let mut k = BrepKernel::new();
        let eid = k
            .lift_curve2d_to_plane_impl(
                1,
                vec![0.0, 0.0, 1.0],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                FRAC_PI_2,
            )
            .unwrap();
        let edge_id = k.resolve_edge(eid).unwrap();
        let edge = k.topo.edge(edge_id).unwrap();
        let s = k.topo.vertex(edge.start()).unwrap().point();
        let e = k.topo.vertex(edge.end()).unwrap().point();
        assert!((s.x() - 1.0).abs() < 1e-10);
        assert!(s.y().abs() < 1e-10);
        assert!(e.x().abs() < 1e-10);
        assert!((e.y() - 1.0).abs() < 1e-10);
        assert!(matches!(edge.curve(), EdgeCurve::Circle(_)));
    }

    #[test]
    fn circle2d_on_xz_plane() {
        let mut k = BrepKernel::new();
        let eid = k
            .lift_curve2d_to_plane_impl(
                1,
                vec![0.0, 0.0, 2.0],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                FRAC_PI_2,
            )
            .unwrap();
        let edge_id = k.resolve_edge(eid).unwrap();
        let edge = k.topo.edge(edge_id).unwrap();
        let s = k.topo.vertex(edge.start()).unwrap().point();
        let e = k.topo.vertex(edge.end()).unwrap().point();
        assert!((s.x() - 2.0).abs() < 1e-10);
        assert!(s.y().abs() < 1e-10);
        assert!(s.z().abs() < 1e-10);
        assert!(e.x().abs() < 1e-10);
        assert!(e.y().abs() < 1e-10);
        assert!((e.z() + 2.0).abs() < 1e-10);
    }

    #[test]
    fn circle2d_full_circle() {
        let mut k = BrepKernel::new();
        let eid = k
            .lift_curve2d_to_plane_impl(
                1,
                vec![0.0, 0.0, 1.0],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                TAU,
            )
            .unwrap();
        let edge_id = k.resolve_edge(eid).unwrap();
        let edge = k.topo.edge(edge_id).unwrap();
        assert_eq!(edge.start(), edge.end());
    }

    #[test]
    fn ellipse2d_with_rotation() {
        let mut k = BrepKernel::new();
        let eid = k
            .lift_curve2d_to_plane_impl(
                2,
                vec![0.0, 0.0, 2.0, 1.0, PI / 4.0],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                FRAC_PI_2,
            )
            .unwrap();
        let edge_id = k.resolve_edge(eid).unwrap();
        let edge = k.topo.edge(edge_id).unwrap();
        assert!(matches!(edge.curve(), EdgeCurve::Ellipse(_)));
        let s = k.topo.vertex(edge.start()).unwrap().point();
        let dist = (s.x().powi(2) + s.y().powi(2) + s.z().powi(2)).sqrt();
        assert!((dist - 2.0).abs() < 1e-10);
    }

    #[test]
    fn ellipse2d_full() {
        let mut k = BrepKernel::new();
        let eid = k
            .lift_curve2d_to_plane_impl(
                2,
                vec![0.0, 0.0, 3.0, 1.0, 0.0],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                TAU,
            )
            .unwrap();
        let edge_id = k.resolve_edge(eid).unwrap();
        let edge = k.topo.edge(edge_id).unwrap();
        assert_eq!(edge.start(), edge.end());
    }

    #[test]
    fn nurbs2d_degree1_line() {
        let mut k = BrepKernel::new();
        let eid = k
            .lift_curve2d_to_plane_impl(
                3,
                vec![1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 3.0, 4.0, 1.0, 1.0],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
            )
            .unwrap();
        let edge_id = k.resolve_edge(eid).unwrap();
        let edge = k.topo.edge(edge_id).unwrap();
        let s = k.topo.vertex(edge.start()).unwrap().point();
        let e = k.topo.vertex(edge.end()).unwrap().point();
        assert!(s.x().abs() < 1e-10);
        assert!(s.y().abs() < 1e-10);
        assert!((e.x() - 3.0).abs() < 1e-10);
        assert!((e.y() - 4.0).abs() < 1e-10);
        assert!(matches!(edge.curve(), EdgeCurve::NurbsCurve(_)));
    }

    #[test]
    fn invalid_curve_type() {
        let mut k = BrepKernel::new();
        let err = k
            .lift_curve2d_to_plane_impl(
                5,
                vec![],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
            )
            .unwrap_err();
        assert!(err.to_string().contains("curve_type"));
    }

    #[test]
    fn wrong_param_count() {
        let mut k = BrepKernel::new();
        let err = k
            .lift_curve2d_to_plane_impl(
                1,
                vec![0.0, 0.0],
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
            )
            .unwrap_err();
        assert!(err.to_string().contains("Circle expects 3 params"));
    }
}
