//! Shape creation bindings (vertices, edges, wires, faces, compounds).

#![allow(clippy::missing_errors_doc, clippy::too_many_arguments)]

use std::f64::consts::PI;

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};
use wasm_bindgen::prelude::*;

use crate::error::{WasmError, validate_finite, validate_positive};
use crate::handles::{
    edge_id_to_u32, face_id_to_u32, solid_id_to_u32, vertex_id_to_u32, wire_id_to_u32,
};
use crate::helpers::{TOL, parse_points};
use crate::kernel::BrepKernel;

#[wasm_bindgen]
impl BrepKernel {
    // ── Shape creation ─────────────────────────────────────────────

    /// Create a rectangular face on the XY plane centered at the origin.
    ///
    /// Returns a face handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if `width` or `height` is non-positive, NaN,
    /// or infinite, or if the face geometry cannot be constructed.
    #[wasm_bindgen(js_name = "makeRectangle")]
    pub fn make_rectangle(&mut self, width: f64, height: f64) -> Result<u32, JsError> {
        validate_positive(width, "width")?;
        validate_positive(height, "height")?;

        let hw = width / 2.0;
        let hh = height / 2.0;

        let points = [
            Point3::new(-hw, -hh, 0.0),
            Point3::new(hw, -hh, 0.0),
            Point3::new(hw, hh, 0.0),
            Point3::new(-hw, hh, 0.0),
        ];

        let face_id = self.make_planar_face(&points)?;
        Ok(face_id_to_u32(face_id))
    }

    /// Create a polygonal face from flat coordinate triples `[x,y,z, ...]`.
    ///
    /// Requires at least 3 points (9 `f64` values).
    /// Returns a face handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if `coords` length is not a multiple of 3,
    /// fewer than 3 points are provided, or the face normal is degenerate.
    #[wasm_bindgen(js_name = "makePolygon")]
    #[allow(clippy::needless_pass_by_value)] // wasm-bindgen requires owned Vec
    pub fn make_polygon(&mut self, coords: Vec<f64>) -> Result<u32, JsError> {
        if coords.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "coordinate array length must be a multiple of 3, got {}",
                    coords.len()
                ),
            }
            .into());
        }
        let n = coords.len() / 3;
        if n < 3 {
            return Err(WasmError::InvalidInput {
                reason: format!("polygon requires at least 3 points, got {n}"),
            }
            .into());
        }

        if let Some(pos) = coords.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("coordinate at index {pos} is not finite"),
            }
            .into());
        }

        let points: Vec<Point3> = coords
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();

        let face_id = self.make_planar_face(&points)?;
        Ok(face_id_to_u32(face_id))
    }

    /// Create a circular polygon approximation on the XY plane.
    ///
    /// The circle is centered at the origin with the given `radius`,
    /// approximated by `segments` straight edges.
    /// Returns a face handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 3 segments are specified.
    #[wasm_bindgen(js_name = "makeCircle")]
    pub fn make_circle(&mut self, radius: f64, segments: u32) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        if segments < 3 {
            return Err(WasmError::InvalidInput {
                reason: format!("circle requires at least 3 segments, got {segments}"),
            }
            .into());
        }

        let n = segments as usize;
        let mut points = Vec::with_capacity(n);
        for i in 0..n {
            #[allow(clippy::cast_precision_loss)]
            let angle = 2.0 * PI * (i as f64) / (n as f64);
            points.push(Point3::new(radius * angle.cos(), radius * angle.sin(), 0.0));
        }

        let face_id = self.make_planar_face(&points)?;
        Ok(face_id_to_u32(face_id))
    }

    // ── Shape construction (low-level) ────────────────────────────

    /// Create a vertex at the given position.
    ///
    /// Returns a vertex handle (`u32`).
    #[wasm_bindgen(js_name = "makeVertex")]
    pub fn make_vertex(&mut self, x: f64, y: f64, z: f64) -> Result<u32, JsError> {
        validate_finite(x, "x")?;
        validate_finite(y, "y")?;
        validate_finite(z, "z")?;
        let id = self.topo.add_vertex(Vertex::new(Point3::new(x, y, z), TOL));
        Ok(vertex_id_to_u32(id))
    }

    /// Create a straight-line edge between two points.
    ///
    /// Returns an edge handle (`u32`).
    #[wasm_bindgen(js_name = "makeLineEdge")]
    pub fn make_line_edge(
        &mut self,
        x1: f64,
        y1: f64,
        z1: f64,
        x2: f64,
        y2: f64,
        z2: f64,
    ) -> Result<u32, JsError> {
        let start = Point3::new(x1, y1, z1);
        let end = Point3::new(x2, y2, z2);
        let eid = brepkit_topology::builder::make_line_edge(&mut self.topo, start, end, TOL)?;
        Ok(edge_id_to_u32(eid))
    }

    /// Create a circular arc edge between two points.
    ///
    /// The arc lies on a circle with the given center, normal axis, and
    /// radius derived from `|start − center|`. The arc goes from start
    /// to end counter-clockwise when viewed along the normal.
    ///
    /// Returns an edge handle (`u32`).
    #[wasm_bindgen(js_name = "makeCircleArc3d")]
    pub fn make_circle_arc_3d(
        &mut self,
        start_x: f64,
        start_y: f64,
        start_z: f64,
        end_x: f64,
        end_y: f64,
        end_z: f64,
        center_x: f64,
        center_y: f64,
        center_z: f64,
        axis_x: f64,
        axis_y: f64,
        axis_z: f64,
    ) -> Result<u32, JsError> {
        let start_pt = Point3::new(start_x, start_y, start_z);
        let end_pt = Point3::new(end_x, end_y, end_z);
        let center = Point3::new(center_x, center_y, center_z);
        let axis = Vec3::new(axis_x, axis_y, axis_z);

        let n = axis.normalize().map_err(|e| WasmError::InvalidInput {
            reason: format!("invalid axis: {e}"),
        })?;

        // u_axis = normalized(start − center), v_axis = n × u
        let radial = start_pt - center;
        let radius = radial.length();
        if radius < 1e-12 {
            return Err(WasmError::InvalidInput {
                reason: "start point coincides with center".into(),
            }
            .into());
        }
        let u_axis = Vec3::new(
            radial.x() / radius,
            radial.y() / radius,
            radial.z() / radius,
        );
        let v_axis = n.cross(u_axis);

        let circle = brepkit_math::curves::Circle3D::with_axes(center, n, radius, u_axis, v_axis)
            .map_err(|e| WasmError::InvalidInput {
            reason: format!("invalid circle: {e}"),
        })?;

        let v_start = self.topo.add_vertex(Vertex::new(start_pt, TOL));
        let v_end = if (start_pt - end_pt).length() < TOL * 100.0 {
            v_start
        } else {
            self.topo.add_vertex(Vertex::new(end_pt, TOL))
        };
        let eid = self
            .topo
            .add_edge(Edge::new(v_start, v_end, EdgeCurve::Circle(circle)));
        Ok(edge_id_to_u32(eid))
    }

    /// Create a NURBS curve edge.
    ///
    /// Returns an edge handle (`u32`).
    #[wasm_bindgen(js_name = "makeNurbsEdge")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_nurbs_edge(
        &mut self,
        start_x: f64,
        start_y: f64,
        start_z: f64,
        end_x: f64,
        end_y: f64,
        end_z: f64,
        degree: u32,
        knots: Vec<f64>,
        control_points: Vec<f64>,
        weights: Vec<f64>,
    ) -> Result<u32, JsError> {
        if control_points.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "control_points length must be a multiple of 3, got {}",
                    control_points.len()
                ),
            }
            .into());
        }
        let cp: Vec<Point3> = control_points
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();
        let curve = NurbsCurve::new(degree as usize, knots, cp, weights)?;

        let start_pt = Point3::new(start_x, start_y, start_z);
        let end_pt = Point3::new(end_x, end_y, end_z);
        let v_start = self.topo.add_vertex(Vertex::new(start_pt, TOL));
        // When start ≈ end (closed curve), reuse the same vertex so
        // downstream code correctly identifies the edge as closed.
        let v_end = if (start_pt - end_pt).length() < TOL * 100.0 {
            v_start
        } else {
            self.topo.add_vertex(Vertex::new(end_pt, TOL))
        };
        let eid = self
            .topo
            .add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(curve)));
        Ok(edge_id_to_u32(eid))
    }

    /// Create a circular arc edge defined by start point, tangent direction
    /// at start, and end point.
    ///
    /// If the tangent is parallel to the start→end chord (collinear), falls
    /// back to a straight line edge.
    ///
    /// Returns an edge handle (`u32`).
    #[wasm_bindgen(js_name = "makeTangentArc3d")]
    pub fn make_tangent_arc_3d(
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
    ) -> Result<u32, JsError> {
        Ok(self.make_tangent_arc_3d_impl(
            start_x, start_y, start_z, tangent_x, tangent_y, tangent_z, end_x, end_y, end_z,
        )?)
    }
    /// Lift a 2D curve onto a 3D plane, producing an edge.
    ///
    /// `curve_type`: 0 = Line, 1 = Circle, 2 = Ellipse, 3 = NURBS.
    /// `curve_params` layout varies by type (see docs).
    /// The plane is defined by an origin, x-axis, and normal.
    /// `t_start`/`t_end` specify the parameter range on the 2D curve.
    ///
    /// Returns an edge handle (`u32`).
    #[wasm_bindgen(js_name = "liftCurve2dToPlane")]
    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        clippy::needless_pass_by_value
    )]
    pub fn lift_curve2d_to_plane(
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
    ) -> Result<u32, JsError> {
        Ok(self.lift_curve2d_to_plane_impl(
            curve_type,
            curve_params,
            origin_x,
            origin_y,
            origin_z,
            x_axis_x,
            x_axis_y,
            x_axis_z,
            normal_x,
            normal_y,
            normal_z,
            t_start,
            t_end,
        )?)
    }

    /// Create a closed wire from an ordered array of edge handles.
    ///
    /// Returns a wire handle (`u32`).
    #[wasm_bindgen(js_name = "makeWire")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_wire(&mut self, edge_handles: Vec<u32>, closed: bool) -> Result<u32, JsError> {
        let tol = brepkit_math::tolerance::Tolerance::new();

        let edge_ids: Vec<brepkit_topology::edge::EdgeId> = edge_handles
            .iter()
            .map(|&h| self.resolve_edge(h))
            .collect::<Result<_, WasmError>>()?;

        // Merge coincident vertices between adjacent edges.
        // When edge[i].end is at the same position as edge[i+1].start,
        // replace edge[i+1].start with edge[i].end so they share a vertex.
        if edge_ids.len() > 1 {
            for i in 0..edge_ids.len() {
                let next = if i + 1 < edge_ids.len() {
                    i + 1
                } else if closed {
                    0 // wrap around for closed wires
                } else {
                    continue;
                };
                if next == i {
                    continue; // single-edge closed wire
                }

                let end_vid = self.topo.edge(edge_ids[i])?.end();
                let start_vid = self.topo.edge(edge_ids[next])?.start();

                if end_vid == start_vid {
                    continue; // already shared
                }

                let end_pos = self.topo.vertex(end_vid)?.point();
                let start_pos = self.topo.vertex(start_vid)?.point();

                let dist = (end_pos - start_pos).length();
                if dist < tol.linear {
                    // Merge: replace the next edge's start with the current edge's end
                    self.topo.edge_mut(edge_ids[next])?.set_start(end_vid);
                }
            }
        }

        let oriented: Vec<OrientedEdge> = edge_ids
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let wire = Wire::new(oriented, closed)?;
        let wid = self.topo.add_wire(wire);
        Ok(wire_id_to_u32(wid))
    }

    /// Create a planar face from a wire (computes normal from first 3 vertices).
    ///
    /// Returns a face handle (`u32`).
    #[wasm_bindgen(js_name = "makeFaceFromWire")]
    pub fn make_face_from_wire(&mut self, wire: u32) -> Result<u32, JsError> {
        let wid = self.resolve_wire(wire)?;
        let fid = brepkit_topology::builder::make_face_from_wire(&mut self.topo, wid)?;
        Ok(face_id_to_u32(fid))
    }

    /// Create a solid from a shell.
    ///
    /// Returns a solid handle (`u32`).
    #[wasm_bindgen(js_name = "solidFromShell")]
    pub fn solid_from_shell(&mut self, shell: u32) -> Result<u32, JsError> {
        let shell_id = self.resolve_shell(shell)?;
        let solid = brepkit_topology::solid::Solid::new(shell_id, vec![]);
        let sid = self.topo.add_solid(solid);
        Ok(solid_id_to_u32(sid))
    }

    /// Create a compound from multiple solid handles.
    ///
    /// Returns a compound handle (stored as `u32`).
    #[wasm_bindgen(js_name = "makeCompound")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_compound(&mut self, solid_handles: Vec<u32>) -> Result<u32, JsError> {
        let solid_ids: Vec<brepkit_topology::solid::SolidId> = solid_handles
            .iter()
            .map(|&h| self.resolve_solid(h))
            .collect::<Result<_, _>>()?;
        let compound = brepkit_topology::compound::Compound::new(solid_ids);
        #[allow(clippy::cast_possible_truncation)]
        let cid = self.topo.add_compound(compound);
        Ok(cid.index() as u32)
    }

    /// Build a convex hull solid from a point cloud.
    ///
    /// Uses a simple Quickhull-inspired algorithm for 3D point sets.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 4 points are provided.
    #[wasm_bindgen(js_name = "convexHull")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn convex_hull(&mut self, coords: Vec<f64>) -> Result<u32, JsError> {
        if coords.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "coordinate array length must be a multiple of 3, got {}",
                    coords.len()
                ),
            }
            .into());
        }
        let points: Vec<Point3> = coords
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();
        if points.len() < 4 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "convex hull requires at least 4 points, got {}",
                    points.len()
                ),
            }
            .into());
        }

        // Build a proper convex hull using Quickhull.
        let hull = brepkit_math::convex_hull::convex_hull_3d(&points).ok_or_else(|| {
            WasmError::InvalidInput {
                reason: "points are coplanar or degenerate — cannot form a 3D convex hull".into(),
            }
        })?;

        // Convert hull to B-Rep: vertices → edges → faces → shell → solid.
        let vertex_ids: Vec<brepkit_topology::vertex::VertexId> = hull
            .vertices
            .iter()
            .map(|p| self.topo.add_vertex(Vertex::new(*p, TOL)))
            .collect();

        let mut face_ids = Vec::new();
        for &[a, b, c] in &hull.faces {
            let va = vertex_ids[a];
            let vb = vertex_ids[b];
            let vc = vertex_ids[c];

            let e0 = self.topo.add_edge(Edge::new(va, vb, EdgeCurve::Line));
            let e1 = self.topo.add_edge(Edge::new(vb, vc, EdgeCurve::Line));
            let e2 = self.topo.add_edge(Edge::new(vc, va, EdgeCurve::Line));

            let oriented = vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
            ];
            let wire = Wire::new(oriented, true)?;
            let wid = self.topo.add_wire(wire);

            // Compute face normal.
            let pa = hull.vertices[a];
            let pb = hull.vertices[b];
            let pc = hull.vertices[c];
            let ab = pb - pa;
            let ac = pc - pa;
            let normal = ab.cross(ac);
            let normal = match normal.normalize() {
                Ok(n) => n,
                Err(_) => Vec3::new(0.0, 0.0, 1.0),
            };
            let d = normal
                .x()
                .mul_add(pa.x(), normal.y().mul_add(pa.y(), normal.z() * pa.z()));

            let fid = self
                .topo
                .add_face(Face::new(wid, vec![], FaceSurface::Plane { normal, d }));
            face_ids.push(fid);
        }

        let shell = brepkit_topology::shell::Shell::new(face_ids)?;
        let shell_id = self.topo.add_shell(shell);
        let solid = brepkit_topology::solid::Solid::new(shell_id, vec![]);
        let solid_id = self.topo.add_solid(solid);

        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a closed polygon wire from flat coordinates.
    ///
    /// Returns a wire handle.
    #[wasm_bindgen(js_name = "makePolygonWire")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_polygon_wire(&mut self, coords: Vec<f64>) -> Result<u32, JsError> {
        let points = parse_points(&coords)?;
        if points.len() < 3 {
            return Err(WasmError::InvalidInput {
                reason: format!("polygon wire needs at least 3 points, got {}", points.len()),
            }
            .into());
        }
        let n = points.len();
        let verts: Vec<_> = points
            .iter()
            .map(|p| self.topo.add_vertex(Vertex::new(*p, TOL)))
            .collect();
        let edges: Vec<_> = (0..n)
            .map(|i| {
                self.topo
                    .add_edge(Edge::new(verts[i], verts[(i + 1) % n], EdgeCurve::Line))
            })
            .collect();
        let oriented: Vec<_> = edges
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let wire = Wire::new(oriented, true)?;
        let wid = self.topo.add_wire(wire);
        Ok(wire_id_to_u32(wid))
    }

    /// Create a regular polygon wire on the XY plane.
    ///
    /// Returns a wire handle.
    #[wasm_bindgen(js_name = "makeRegularPolygonWire")]
    pub fn make_regular_polygon_wire(&mut self, radius: f64, n_sides: u32) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        if n_sides < 3 {
            return Err(WasmError::InvalidInput {
                reason: format!("polygon needs at least 3 sides, got {n_sides}"),
            }
            .into());
        }
        let wid = brepkit_topology::builder::make_regular_polygon_wire(
            &mut self.topo,
            radius,
            n_sides as usize,
            TOL,
        )?;
        Ok(wire_id_to_u32(wid))
    }

    /// Create a circular face on the XY plane (using NURBS arcs).
    ///
    /// Returns a face handle.
    #[wasm_bindgen(js_name = "makeCircleFace")]
    pub fn make_circle_face(&mut self, radius: f64, segments: u32) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        if segments < 3 {
            return Err(WasmError::InvalidInput {
                reason: format!("circle face needs at least 3 segments, got {segments}"),
            }
            .into());
        }
        let fid = brepkit_topology::builder::make_circle_face(
            &mut self.topo,
            radius,
            segments as usize,
            TOL,
        )?;
        Ok(face_id_to_u32(fid))
    }
}
