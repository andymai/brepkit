//! Batch execution and dispatch bindings.

#![allow(clippy::missing_errors_doc, clippy::too_many_lines)]

use wasm_bindgen::prelude::*;

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{self, BooleanOp, boolean};
use brepkit_operations::measure;
use brepkit_operations::sweep::sweep;
use brepkit_topology::edge::EdgeCurve;

use crate::error::{WasmError, validate_finite, validate_tolerance};
use crate::handles::{
    edge_id_to_u32, face_id_to_u32, shell_id_to_u32, solid_id_to_u32, wire_id_to_u32,
};
use crate::helpers::{
    TOL, classify_to_string, get_f64, get_u32, handles_in, matrix_in, opt_f64, opt_handles_in,
    opt_u32, well_formed_nurbs, well_formed_nurbs_surface,
};
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

/// A `(u_range, v_range)` pair, each `(min, max)`.
type UvRanges = ((f64, f64), (f64, f64));

/// A revolve's axis from batch args: `originX..Z` (default the origin) and
/// `axisX..Z` (default +z).
fn axis_in(args: &serde_json::Value) -> Result<(Point3, Vec3), String> {
    Ok((
        Point3::new(
            opt_f64(args, "originX", 0.0)?,
            opt_f64(args, "originY", 0.0)?,
            opt_f64(args, "originZ", 0.0)?,
        ),
        Vec3::new(
            opt_f64(args, "axisX", 0.0)?,
            opt_f64(args, "axisY", 0.0)?,
            opt_f64(args, "axisZ", 1.0)?,
        ),
    ))
}

/// A cutting plane from batch args: `px..pz` (default the origin) and
/// `nx..nz` (default +z).
fn plane_in(args: &serde_json::Value) -> Result<(Point3, Vec3), String> {
    Ok((
        Point3::new(
            opt_f64(args, "px", 0.0)?,
            opt_f64(args, "py", 0.0)?,
            opt_f64(args, "pz", 0.0)?,
        ),
        Vec3::new(
            opt_f64(args, "nx", 0.0)?,
            opt_f64(args, "ny", 0.0)?,
            opt_f64(args, "nz", 1.0)?,
        ),
    ))
}

/// Build the in-plane axes used by `plane_to_nurbs`.
///
/// Must match `brepkit_heal::construct::convert_surface`'s private frame
/// so projected face corners reconstruct the plane rectangle consistently.
fn plane_frame_axes(normal: Vec3) -> (Vec3, Vec3) {
    let seed = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_axis = normal
        .cross(seed)
        .normalize()
        .unwrap_or_else(|_| Vec3::new(1.0, 0.0, 0.0));
    let v_axis = normal.cross(u_axis);
    (u_axis, v_axis)
}

impl BrepKernel {
    /// Extract a `NurbsCurve` from an edge.
    ///
    /// NURBS edges are returned directly. Line, Circle, and Ellipse edges
    /// are converted to their exact rational NURBS equivalent using the
    /// edge's bounding vertices (and the curve's analytic params for
    /// circles/ellipses).
    pub(crate) fn extract_nurbs_curve(&self, edge: u32) -> Result<NurbsCurve, WasmError> {
        use brepkit_geometry::convert::{circle_to_nurbs, ellipse_to_nurbs, line_to_nurbs};
        use std::f64::consts::TAU;

        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        let start_v = edge_data.start();
        let end_v = edge_data.end();
        let start_pt = self.topo.vertex(start_v)?.point();
        let end_pt = self.topo.vertex(end_v)?.point();

        match edge_data.curve() {
            EdgeCurve::NurbsCurve(c) => Ok(c.clone()),
            EdgeCurve::Line => {
                Ok(
                    line_to_nurbs(start_pt, end_pt).map_err(|e| WasmError::InvalidInput {
                        reason: format!("line_to_nurbs failed: {e}"),
                    })?,
                )
            }
            EdgeCurve::Circle(c) => {
                let (t_start, t_end) = if start_v == end_v {
                    (0.0, TAU)
                } else {
                    let ts = c.project(start_pt);
                    let mut te = c.project(end_pt);
                    if te <= ts {
                        te += TAU;
                    }
                    (ts, te)
                };
                Ok(
                    circle_to_nurbs(c, t_start, t_end).map_err(|e| WasmError::InvalidInput {
                        reason: format!("circle_to_nurbs failed: {e}"),
                    })?,
                )
            }
            EdgeCurve::Ellipse(e) => {
                let (t_start, t_end) = if start_v == end_v {
                    (0.0, TAU)
                } else {
                    let ts = e.project(start_pt);
                    let mut te = e.project(end_pt);
                    if te <= ts {
                        te += TAU;
                    }
                    (ts, te)
                };
                Ok(
                    ellipse_to_nurbs(e, t_start, t_end).map_err(|err| WasmError::InvalidInput {
                        reason: format!("ellipse_to_nurbs failed: {err}"),
                    })?,
                )
            }
        }
    }

    /// Extract a `NurbsSurface` from a face.
    ///
    /// NURBS faces are returned directly. Analytic surfaces are converted to
    /// their NURBS equivalent: planes and cylinders are geometrically exact;
    /// cones, spheres, and tori use the exact rational forms from
    /// `brepkit_heal::construct::convert_surface`. Plane and cone parameter
    /// ranges are derived from the face's boundary vertices.
    pub(crate) fn extract_nurbs_surface(&self, face: u32) -> Result<NurbsSurface, WasmError> {
        use brepkit_heal::construct::convert_surface;
        use brepkit_topology::face::FaceSurface;

        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;

        let map_err = |context: &str, e: brepkit_heal::HealError| WasmError::InvalidInput {
            reason: format!("{context}: {e}"),
        };

        match face_data.surface() {
            FaceSurface::Nurbs(s) => Ok(s.clone()),
            FaceSurface::Plane { normal, d } => {
                let (u_range, v_range) = self.plane_face_uv_bounds(face_id, *normal, *d)?;
                convert_surface::plane_to_nurbs(*normal, *d, u_range, v_range)
                    .map_err(|e| map_err("plane_to_nurbs failed", e))
            }
            FaceSurface::Cylinder(c) => {
                let v_range = self.analytic_face_v_bounds(face_id, face_data.surface())?;
                convert_surface::cylinder_to_nurbs(c, v_range)
                    .map_err(|e| map_err("cylinder_to_nurbs failed", e))
            }
            FaceSurface::Cone(c) => {
                let v_range = self.analytic_face_v_bounds(face_id, face_data.surface())?;
                convert_surface::cone_to_nurbs(c, v_range)
                    .map_err(|e| map_err("cone_to_nurbs failed", e))
            }
            FaceSurface::Sphere(s) => convert_surface::sphere_to_nurbs(s)
                .map_err(|e| map_err("sphere_to_nurbs failed", e)),
            FaceSurface::Torus(t) => {
                convert_surface::torus_to_nurbs(t).map_err(|e| map_err("torus_to_nurbs failed", e))
            }
        }
    }

    /// Derive the parametric rectangle of a planar face by sampling its outer
    /// boundary edges and projecting the samples onto the same local frame
    /// `plane_to_nurbs` uses.
    ///
    /// Sampling the edge curves (not just the bounding vertices) is required for
    /// circle- and ellipse-bounded faces such as cylinder/cone caps, whose
    /// outer wire may carry a single seam vertex while the disk spans a finite
    /// rectangle in the plane frame.
    #[allow(clippy::cast_precision_loss)]
    fn plane_face_uv_bounds(
        &self,
        face_id: brepkit_topology::face::FaceId,
        normal: Vec3,
        d: f64,
    ) -> Result<UvRanges, WasmError> {
        const EDGE_SAMPLES: usize = 16;

        let face_data = self.topo.face(face_id)?;
        let wire = self.topo.wire(face_data.outer_wire())?;
        let origin = Point3::new(0.0, 0.0, 0.0) + normal * d;
        let (u_axis, v_axis) = plane_frame_axes(normal);

        let mut u_min = f64::INFINITY;
        let mut u_max = f64::NEG_INFINITY;
        let mut v_min = f64::INFINITY;
        let mut v_max = f64::NEG_INFINITY;
        for oe in wire.edges() {
            let edge = self.topo.edge(oe.edge())?;
            let start = self.topo.vertex(edge.start())?.point();
            let end = self.topo.vertex(edge.end())?.point();
            let curve = edge.curve();
            let (t0, t1) = curve.domain_with_endpoints(start, end);
            for i in 0..=EDGE_SAMPLES {
                let t = t0 + (t1 - t0) * (i as f64 / EDGE_SAMPLES as f64);
                let p = curve.evaluate_with_endpoints(t, start, end);
                let rel = p - origin;
                let u = rel.dot(u_axis);
                let v = rel.dot(v_axis);
                u_min = u_min.min(u);
                u_max = u_max.max(u);
                v_min = v_min.min(v);
                v_max = v_max.max(v);
            }
        }
        if u_max <= u_min || v_max <= v_min {
            return Err(WasmError::InvalidInput {
                reason: "planar face has degenerate parametric extent".to_string(),
            });
        }
        Ok(((u_min, u_max), (v_min, v_max)))
    }

    /// Derive the axial/generator parameter range of an analytic face by
    /// projecting its boundary vertices onto the surface.
    fn analytic_face_v_bounds(
        &self,
        face_id: brepkit_topology::face::FaceId,
        surface: &brepkit_topology::face::FaceSurface,
    ) -> Result<(f64, f64), WasmError> {
        let verts = brepkit_topology::explorer::face_vertices(&self.topo, face_id)?;
        let mut v_min = f64::INFINITY;
        let mut v_max = f64::NEG_INFINITY;
        for vid in verts {
            let p = self.topo.vertex(vid)?.point();
            if let Some((_, v)) = surface.project_point(p) {
                v_min = v_min.min(v);
                v_max = v_max.max(v);
            }
        }
        if v_max <= v_min {
            return Err(WasmError::InvalidInput {
                reason: "analytic face has degenerate axial extent".to_string(),
            });
        }
        Ok((v_min, v_max))
    }

    /// Create an edge from a `NurbsCurve`, using its endpoints. The curve is
    /// fitted to, or computed from, the caller's numbers, and one no
    /// evaluation can walk is refused.
    pub(crate) fn nurbs_curve_to_edge(
        &mut self,
        points: &[Point3],
        curve: NurbsCurve,
    ) -> Result<brepkit_topology::edge::EdgeId, WasmError> {
        let curve = well_formed_nurbs(curve, "the fitted curve")?;
        let start = points[0];
        let end = points[points.len() - 1];
        Ok(brepkit_topology::builder::make_nurbs_edge(
            self.topo_mut(),
            start,
            end,
            curve,
            TOL,
        ))
    }

    /// Create an edge from a `NurbsCurve`, evaluating its endpoints. The
    /// curve is computed from the caller's numbers, and one no evaluation can
    /// walk is refused.
    pub(crate) fn nurbs_curve_to_edge_from_curve(
        &mut self,
        curve: NurbsCurve,
    ) -> Result<brepkit_topology::edge::EdgeId, WasmError> {
        let curve = well_formed_nurbs(curve, "the curve")?;
        Ok(brepkit_topology::builder::make_nurbs_edge_from_curve(
            self.topo_mut(),
            &curve,
            TOL,
        ))
    }

    /// Create a face from a `NurbsSurface` with a rectangular domain wire.
    /// The surface is fitted to, or computed from, the caller's numbers, and
    /// one no evaluation can walk is refused.
    pub(crate) fn nurbs_surface_to_face(
        &mut self,
        surface: NurbsSurface,
    ) -> Result<brepkit_topology::face::FaceId, WasmError> {
        let surface = well_formed_nurbs_surface(surface, "the surface")?;
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
                let solid = self.make_box_impl(w, h, d).map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "makeCylinder" => {
                let r = get_f64(args, "radius")?;
                let h = get_f64(args, "height")?;
                let solid = self.make_cylinder_impl(r, h).map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "makeSphere" => {
                let r = get_f64(args, "radius")?;
                let segments = opt_u32(args, "segments", 16)?;
                let solid = self
                    .make_sphere_impl(r, segments)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "makeCone" => {
                let br = get_f64(args, "bottomRadius")?;
                let tr = get_f64(args, "topRadius")?;
                let h = get_f64(args, "height")?;
                let solid = self.make_cone_impl(br, tr, h).map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "makeTorus" => {
                let major = get_f64(args, "majorRadius")?;
                let minor = get_f64(args, "minorRadius")?;
                let segments = opt_u32(args, "segments", 16)?;
                let solid = self
                    .make_torus_impl(major, minor, segments)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "makeEllipsoid" => {
                let rx = get_f64(args, "rx")?;
                let ry = get_f64(args, "ry")?;
                let rz = get_f64(args, "rz")?;
                let solid = self
                    .make_ellipsoid_impl(rx, ry, rz)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "makeCircleEdge" => {
                let edge = self
                    .make_circle_edge_impl(
                        [
                            opt_f64(args, "cx", 0.0)?,
                            opt_f64(args, "cy", 0.0)?,
                            opt_f64(args, "cz", 0.0)?,
                        ],
                        [
                            opt_f64(args, "nx", 0.0)?,
                            opt_f64(args, "ny", 0.0)?,
                            opt_f64(args, "nz", 1.0)?,
                        ],
                        get_f64(args, "radius")?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(edge))
            }
            "makeCircle" | "makeCircleFace" | "makeRegularPolygonWire" => {
                let radius = get_f64(args, "radius")?;
                let made = match op {
                    "makeCircle" => self.make_circle_impl(radius, get_u32(args, "segments")?),
                    "makeCircleFace" => {
                        self.make_circle_face_impl(radius, get_u32(args, "segments")?)
                    }
                    _ => self.make_regular_polygon_wire_impl(radius, get_u32(args, "nSides")?),
                };
                Ok(serde_json::json!(made.map_err(|e| e.to_string())?))
            }
            "sweepWithOptions" => {
                let scale_values = args["scaleValues"]
                    .as_array()
                    .map(|values| {
                        values
                            .iter()
                            .enumerate()
                            .map(|(i, v)| {
                                v.as_f64()
                                    .ok_or_else(|| format!("scaleValues[{i}] is not a number"))
                            })
                            .collect::<Result<Vec<f64>, String>>()
                    })
                    .transpose()?
                    .unwrap_or_default();
                let solid = self
                    .sweep_with_options_impl(
                        get_u32(args, "profile")?,
                        get_u32(args, "pathEdge")?,
                        args["contactMode"].as_str().unwrap_or("rmf"),
                        &scale_values,
                        opt_u32(args, "segments", 0)?,
                        args["cornerMode"].as_str().unwrap_or("smooth"),
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "untrimFace" => {
                let face = self
                    .untrim_face_impl(
                        get_u32(args, "face")?,
                        get_u32(args, "samplesPerCurve")?,
                        get_u32(args, "interiorSamples")?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(face))
            }
            "fuse" | "cut" | "intersect" => {
                let bool_op = match op {
                    "fuse" => BooleanOp::Fuse,
                    "cut" => BooleanOp::Cut,
                    _ => BooleanOp::Intersect,
                };
                let a = get_u32(args, "solidA")?;
                let b = get_u32(args, "solidB")?;
                let a_id = self.resolve_solid(a).map_err(|e| e.to_string())?;
                let b_id = self.resolve_solid(b).map_err(|e| e.to_string())?;
                let simplify = args["simplify"].as_bool().unwrap_or(false);
                let result = if simplify {
                    let opts = brepkit_operations::boolean::BooleanOptions {
                        unify_faces: true,
                        ..Default::default()
                    };
                    brepkit_operations::boolean::boolean_with_options(
                        self.topo_mut(),
                        bool_op,
                        a_id,
                        b_id,
                        opts,
                    )
                } else {
                    boolean(self.topo_mut(), bool_op, a_id, b_id)
                }
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "meshFallbackCount" => {
                #[allow(clippy::cast_precision_loss)]
                let count = brepkit_operations::boolean::mesh_fallback_count() as f64;
                Ok(serde_json::json!(count))
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
                let tools: Vec<brepkit_topology::solid::SolidId> = handles_in(args, "tools")?
                    .into_iter()
                    .map(|h| self.resolve_solid(h).map_err(|e| e.to_string()))
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
            "fuseAll" => {
                let solids: Vec<brepkit_topology::solid::SolidId> = handles_in(args, "solids")?
                    .into_iter()
                    .map(|h| self.resolve_solid(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, String>>()?;
                let compound = self
                    .topo_mut()
                    .add_compound(brepkit_topology::compound::Compound::new(solids));
                let result = brepkit_operations::compound_ops::fuse_all(self.topo_mut(), compound)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "transform" => {
                let s = get_u32(args, "solid")?;
                let matrix = matrix_in(args, "matrix")?;
                self.transform_solid_impl(s, &matrix)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(s))
            }
            "volume" => {
                let v = self
                    .volume_impl(get_u32(args, "solid")?, opt_f64(args, "deflection", 0.1)?)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(v))
            }
            "surfaceArea" => {
                let a = self
                    .surface_area_impl(get_u32(args, "solid")?, opt_f64(args, "deflection", 0.1)?)
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
                let com = self
                    .center_of_mass_impl(get_u32(args, "solid")?, opt_f64(args, "deflection", 0.1)?)
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
                let matrix = matrix_in(args, "matrix")?;
                let copy = self
                    .copy_and_transform_solid_impl(s, &matrix)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(copy))
            }
            // ── Batch 8: new batch-dispatched operations ──────────────
            "extrude" => {
                let direction = Vec3::new(
                    opt_f64(args, "dx", 0.0)?,
                    opt_f64(args, "dy", 0.0)?,
                    opt_f64(args, "dz", 1.0)?,
                );
                let solid = self
                    .extrude_impl(
                        get_u32(args, "face")?,
                        direction,
                        opt_f64(args, "distance", 1.0)?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "revolve" => {
                let (origin, axis) = axis_in(args)?;
                let solid = self
                    .revolve_impl(
                        get_u32(args, "face")?,
                        origin,
                        axis,
                        get_f64(args, "angle")?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "extrudeWire" => {
                let direction = Vec3::new(
                    opt_f64(args, "dx", 0.0)?,
                    opt_f64(args, "dy", 0.0)?,
                    opt_f64(args, "dz", 1.0)?,
                );
                let shell = self
                    .extrude_wire_impl(
                        get_u32(args, "wire")?,
                        direction,
                        opt_f64(args, "distance", 1.0)?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(shell))
            }
            "revolveWire" => {
                let (origin, axis) = axis_in(args)?;
                let shell = self
                    .revolve_wire_impl(
                        get_u32(args, "wire")?,
                        origin,
                        axis,
                        get_f64(args, "angle")?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(shell))
            }
            "loftWires" => {
                let wire_handles = handles_in(args, "wires")?;
                let wire_ids: Vec<_> = wire_handles
                    .iter()
                    .map(|&h| self.resolve_wire(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let shell = brepkit_operations::sheet::loft_wires(self.topo_mut(), &wire_ids)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(shell_id_to_u32(shell)))
            }
            "blendSurface" => {
                let chain = |this: &Self, key: &str| -> Result<Vec<_>, String> {
                    handles_in(args, key)?
                        .iter()
                        .map(|&h| this.resolve_edge(h).map_err(|e| e.to_string()))
                        .collect()
                };
                let edges_a = chain(self, "chainA")?;
                let edges_b = chain(self, "chainB")?;
                let face_a = self
                    .resolve_face(get_u32(args, "faceA")?)
                    .map_err(|e| e.to_string())?;
                let face_b = self
                    .resolve_face(get_u32(args, "faceB")?)
                    .map_err(|e| e.to_string())?;
                let options = crate::helpers::blend_surface_options(args)?;
                let face = brepkit_operations::blend_surface::blend_surface(
                    self.topo_mut(),
                    &edges_a,
                    face_a,
                    &edges_b,
                    face_b,
                    &options,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(face_id_to_u32(face)))
            }
            "matchSurface" => {
                let face = self
                    .resolve_face(get_u32(args, "face")?)
                    .map_err(|e| e.to_string())?;
                let edge = self
                    .resolve_edge(get_u32(args, "edge")?)
                    .map_err(|e| e.to_string())?;
                let target = self
                    .resolve_edge(get_u32(args, "targetEdge")?)
                    .map_err(|e| e.to_string())?;
                let continuity = crate::helpers::match_continuity(args)?;
                let result = if args["targetFace"].is_null() {
                    brepkit_operations::blend_surface::match_surface(
                        self.topo_mut(),
                        face,
                        edge,
                        target,
                        continuity,
                    )
                } else {
                    let on = self
                        .resolve_face(get_u32(args, "targetFace")?)
                        .map_err(|e| e.to_string())?;
                    brepkit_operations::blend_surface::match_surface_on(
                        self.topo_mut(),
                        face,
                        edge,
                        target,
                        on,
                        continuity,
                    )
                }
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(face_id_to_u32(result)))
            }
            "fairSurface" => {
                let face = self
                    .fair_surface_impl(get_u32(args, "face")?, get_f64(args, "tolerance")?)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(face))
            }
            "sweepWire" => {
                let w = get_u32(args, "wire")?;
                let edge_handles = handles_in(args, "pathEdges")?;
                let wire_id = self.resolve_wire(w).map_err(|e| e.to_string())?;
                let edge_ids: Vec<_> = edge_handles
                    .iter()
                    .map(|&h| self.resolve_edge(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let shell =
                    brepkit_operations::sheet::sweep_wire(self.topo_mut(), wire_id, &edge_ids)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(shell_id_to_u32(shell)))
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
            "multiSectionSweep" => {
                let faces = opt_handles_in(args, "faces")?;
                let params: Vec<f64> = match &args["params"] {
                    serde_json::Value::Null => Vec::new(),
                    value => value
                        .as_array()
                        .ok_or("'params' must be an array")?
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            let p = v
                                .as_f64()
                                .ok_or_else(|| format!("params[{i}] is not a number"))?;
                            validate_finite(p, "a section's parameter")
                                .map_err(|e| e.to_string())?;
                            Ok(p)
                        })
                        .collect::<Result<_, String>>()?,
                };
                if faces.len() != params.len() {
                    return Err("multiSectionSweep: faces and params length mismatch".into());
                }
                let spine_edge = get_u32(args, "spineEdge")?;
                let edge_id = self.resolve_edge(spine_edge).map_err(|e| e.to_string())?;
                let edge_data = self.topo.edge(edge_id).map_err(|e| e.to_string())?;
                let spine = match edge_data.curve() {
                    EdgeCurve::NurbsCurve(c) => c.clone(),
                    EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                        return Err("multiSectionSweep spine must be a NURBS edge".into());
                    }
                };
                let ruled = args["ruled"].as_bool().unwrap_or(true);
                let sections: Vec<(brepkit_topology::face::FaceId, f64)> = faces
                    .iter()
                    .zip(params.iter())
                    .map(|(&h, &p)| {
                        self.resolve_face(h)
                            .map(|f| (f, p))
                            .map_err(|e| e.to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let solid = brepkit_operations::sweep::multi_section_sweep(
                    self.topo_mut(),
                    &spine,
                    &sections,
                    ruled,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "guidedSweep" => {
                let face_id = self
                    .resolve_face(get_u32(args, "face")?)
                    .map_err(|e| e.to_string())?;
                let nurbs_of =
                    |this: &Self, edge: u32, label: &str| -> Result<NurbsCurve, String> {
                        let edge_id = this.resolve_edge(edge).map_err(|e| e.to_string())?;
                        let edge_data = this.topo.edge(edge_id).map_err(|e| e.to_string())?;
                        match edge_data.curve() {
                            EdgeCurve::NurbsCurve(c) => Ok(c.clone()),
                            EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                                Err(format!("guidedSweep {label} must be a NURBS edge"))
                            }
                        }
                    };
                let spine = nurbs_of(self, get_u32(args, "spineEdge")?, "spineEdge")?;
                let aux = nurbs_of(self, get_u32(args, "auxEdge")?, "auxEdge")?;
                let solid =
                    brepkit_operations::sweep::sweep_guided(self.topo_mut(), face_id, &spine, aux)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "minkowskiSum" => {
                let a = self
                    .resolve_solid(get_u32(args, "solidA")?)
                    .map_err(|e| e.to_string())?;
                let b = self
                    .resolve_solid(get_u32(args, "solidB")?)
                    .map_err(|e| e.to_string())?;
                let solid =
                    brepkit_operations::primitives::make_minkowski_sum(self.topo_mut(), a, b)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "projectEdges" => {
                let frame = (
                    Point3::new(
                        get_f64(args, "originX")?,
                        get_f64(args, "originY")?,
                        get_f64(args, "originZ")?,
                    ),
                    Vec3::new(
                        get_f64(args, "dirX")?,
                        get_f64(args, "dirY")?,
                        get_f64(args, "dirZ")?,
                    ),
                    Vec3::new(
                        get_f64(args, "xAxisX")?,
                        get_f64(args, "xAxisY")?,
                        get_f64(args, "xAxisZ")?,
                    ),
                );
                self.project_edges_impl(
                    get_u32(args, "solid")?,
                    frame,
                    args["hiddenLines"].as_bool().unwrap_or(true),
                    opt_f64(args, "deflection", 0.1)?,
                )
                .map_err(|e| e.to_string())
            }
            "chamfer" => {
                let solid = self
                    .chamfer_impl(
                        get_u32(args, "solid")?,
                        &opt_handles_in(args, "edges")?,
                        get_f64(args, "distance")?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "fillet" => {
                let solid = self
                    .fillet_impl(
                        get_u32(args, "solid")?,
                        &opt_handles_in(args, "edges")?,
                        get_f64(args, "radius")?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "filletVariable" => {
                let specs = args["specs"]
                    .as_array()
                    .ok_or_else(|| "missing 'specs' array".to_string())?;
                let solid = self
                    .fillet_variable_impl(get_u32(args, "solid")?, specs)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "filletV2" => {
                let solid = self
                    .fillet_v2_impl(
                        get_u32(args, "solid")?,
                        &opt_handles_in(args, "edges")?,
                        get_f64(args, "radius")?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "filletV2Corners" => {
                let solid = self
                    .fillet_v2_corners_impl(
                        get_u32(args, "solid")?,
                        &handles_in(args, "edges")?,
                        get_f64(args, "radius")?,
                        &args["corners"],
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "chamferV2" => {
                let solid = self
                    .chamfer_v2_impl(
                        get_u32(args, "solid")?,
                        &opt_handles_in(args, "edges")?,
                        get_f64(args, "d1")?,
                        get_f64(args, "d2")?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "chamferDistanceAngle" => {
                let solid = self
                    .chamfer_distance_angle_impl(
                        get_u32(args, "solid")?,
                        &opt_handles_in(args, "edges")?,
                        get_f64(args, "distance")?,
                        get_f64(args, "angle")?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "shell" => {
                let solid = self
                    .shell_impl(
                        get_u32(args, "solid")?,
                        get_f64(args, "thickness")?,
                        &opt_handles_in(args, "faces")?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "mirror" => {
                let s = get_u32(args, "solid")?;
                let point = Point3::new(
                    opt_f64(args, "px", 0.0)?,
                    opt_f64(args, "py", 0.0)?,
                    opt_f64(args, "pz", 0.0)?,
                );
                let normal = Vec3::new(
                    opt_f64(args, "nx", 1.0)?,
                    opt_f64(args, "ny", 0.0)?,
                    opt_f64(args, "nz", 0.0)?,
                );
                let result = self
                    .mirror_impl(s, point, normal)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(result))
            }
            "unifyFaces" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                brepkit_operations::heal::unify_faces(self.topo_mut(), solid_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid_id)))
            }
            "convertToBspline" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let count = brepkit_operations::heal::convert_to_bspline(self.topo_mut(), solid_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "solid": solid_id_to_u32(solid_id),
                    "converted": count,
                }))
            }
            "convertToElementary" => {
                let s = get_u32(args, "solid")?;
                let tol = opt_f64(args, "tolerance", crate::helpers::TOL)?;
                validate_tolerance(tol, "tolerance").map_err(|e| e.to_string())?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let count =
                    brepkit_operations::heal::convert_to_elementary(self.topo_mut(), solid_id, tol)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "solid": solid_id_to_u32(solid_id),
                    "converted": count,
                }))
            }
            "healSolid" => {
                let s = get_u32(args, "solid")?;
                let tol = opt_f64(args, "tolerance", 1e-7)?;
                validate_tolerance(tol, "tolerance").map_err(|e| e.to_string())?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                brepkit_operations::heal::heal_solid(self.topo_mut(), solid_id, tol)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid_id)))
            }
            "repairSolid" => {
                let s = get_u32(args, "solid")?;
                let tol = opt_f64(args, "tolerance", 1e-7)?;
                validate_tolerance(tol, "tolerance").map_err(|e| e.to_string())?;
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
                let point = Point3::new(
                    get_f64(args, "x")?,
                    get_f64(args, "y")?,
                    get_f64(args, "z")?,
                );
                let result = self
                    .classify_point_impl(
                        get_u32(args, "solid")?,
                        point,
                        opt_f64(args, "tolerance", 1e-7)?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(classify_to_string(result)))
            }
            "loft" => {
                let face_ids: Vec<_> = opt_handles_in(args, "faces")?
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = brepkit_operations::loft::loft(self.topo_mut(), &face_ids)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "loftSmooth" => {
                let face_ids: Vec<_> = opt_handles_in(args, "faces")?
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = brepkit_operations::loft::loft_smooth(self.topo_mut(), &face_ids)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "loftWithOptions" => {
                let face_ids: Vec<_> = handles_in(args, "faces")?
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let handle = self
                    .loft_from_options(&face_ids, args)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(handle))
            }
            "circularPattern" => {
                let axis = Vec3::new(
                    opt_f64(args, "ax", 0.0)?,
                    opt_f64(args, "ay", 0.0)?,
                    opt_f64(args, "az", 1.0)?,
                );
                let compound = self
                    .circular_pattern_impl(get_u32(args, "solid")?, axis, get_u32(args, "count")?)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(compound))
            }
            "gridPattern" => {
                let directions = [
                    Vec3::new(
                        opt_f64(args, "dirXx", 1.0)?,
                        opt_f64(args, "dirXy", 0.0)?,
                        opt_f64(args, "dirXz", 0.0)?,
                    ),
                    Vec3::new(
                        opt_f64(args, "dirYx", 0.0)?,
                        opt_f64(args, "dirYy", 1.0)?,
                        opt_f64(args, "dirYz", 0.0)?,
                    ),
                ];
                let compound = self
                    .grid_pattern_impl(
                        get_u32(args, "solid")?,
                        directions,
                        [get_f64(args, "spacingX")?, get_f64(args, "spacingY")?],
                        [get_u32(args, "countX")?, get_u32(args, "countY")?],
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(compound))
            }
            "defeature" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let face_ids: Vec<_> = opt_handles_in(args, "faces")?
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
            "copyFace" => {
                let f = get_u32(args, "face")?;
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let copy = brepkit_operations::copy::copy_face(self.topo_mut(), face_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(face_id_to_u32(copy)))
            }
            "transformWire" => {
                let matrix = matrix_in(args, "matrix")?;
                self.transform_wire_impl(get_u32(args, "wire")?, &matrix)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(null))
            }
            "transformFace" => {
                let matrix = matrix_in(args, "matrix")?;
                self.transform_face_impl(get_u32(args, "face")?, &matrix)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(null))
            }
            "offsetFace" => {
                let face = self
                    .offset_face_impl(
                        get_u32(args, "face")?,
                        get_f64(args, "distance")?,
                        opt_u32(args, "samples", 16)?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(face))
            }
            "offsetSolid" => {
                let solid = self
                    .offset_solid_impl(get_u32(args, "solid")?, get_f64(args, "distance")?)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "offsetSolidV2" => {
                let solid = self
                    .offset_solid_impl(get_u32(args, "solid")?, get_f64(args, "distance")?)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "section" => {
                let (point, normal) = plane_in(args)?;
                let faces = self
                    .section_impl(get_u32(args, "solid")?, point, normal)
                    .map_err(|e| e.to_string())?;
                let face_ids: Vec<u32> = faces.iter().map(|&f| face_id_to_u32(f)).collect();
                Ok(serde_json::json!(face_ids))
            }
            "split" => {
                let (point, normal) = plane_in(args)?;
                let (positive, negative) = self
                    .split_impl(get_u32(args, "solid")?, point, normal)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!({ "positive": positive, "negative": negative }))
            }
            "imprintFaces" => {
                use brepkit_operations::imprint::{ImprintTool, imprint_faces};

                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let mut tools = Vec::new();
                if !args["faces"].is_null() {
                    for handle in handles_in(args, "faces")? {
                        let face = self.resolve_face(handle).map_err(|e| e.to_string())?;
                        tools.push(ImprintTool::Face(face));
                    }
                }
                if !args["edges"].is_null() {
                    for handle in handles_in(args, "edges")? {
                        let edge = self.resolve_edge(handle).map_err(|e| e.to_string())?;
                        tools.push(ImprintTool::Edge(edge));
                    }
                }
                let result =
                    imprint_faces(self.topo_mut(), solid_id, &tools).map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "splitByFaces" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let faces = handles_in(args, "faces")?
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let pieces =
                    brepkit_operations::imprint::split_by_faces(self.topo_mut(), solid_id, &faces)
                        .map_err(|e| e.to_string())?;
                let handles: Vec<u32> = pieces.into_iter().map(solid_id_to_u32).collect();
                Ok(serde_json::json!(handles))
            }
            "sewFaces" => {
                let solid = self
                    .sew_faces_impl(
                        &opt_handles_in(args, "faces")?,
                        opt_f64(args, "tolerance", 1e-6)?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
            }
            "thicken" => {
                let solid = self
                    .thicken_impl(get_u32(args, "face")?, get_f64(args, "thickness")?)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
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
                let direction = Vec3::new(
                    opt_f64(args, "dx", 1.0)?,
                    opt_f64(args, "dy", 0.0)?,
                    opt_f64(args, "dz", 0.0)?,
                );
                let compound = self
                    .linear_pattern_impl(
                        get_u32(args, "solid")?,
                        direction,
                        get_f64(args, "spacing")?,
                        get_u32(args, "count")?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(compound))
            }
            "draft" => {
                let pull = Vec3::new(
                    opt_f64(args, "dirX", 0.0)?,
                    opt_f64(args, "dirY", 0.0)?,
                    opt_f64(args, "dirZ", 1.0)?,
                );
                let neutral = Point3::new(
                    opt_f64(args, "neutralX", 0.0)?,
                    opt_f64(args, "neutralY", 0.0)?,
                    opt_f64(args, "neutralZ", 0.0)?,
                );
                let solid = self
                    .draft_impl(
                        get_u32(args, "solid")?,
                        &opt_handles_in(args, "faces")?,
                        pull,
                        neutral,
                        get_f64(args, "angle")?,
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid))
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
            "curveDegreeElevate" => {
                let eid = self
                    .curve_degree_elevate_impl(get_u32(args, "edge")?, get_u32(args, "elevateBy")?)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(eid))
            }
            "makeNurbsEdge" => {
                let numbers = |key: &str| -> Result<Vec<f64>, String> {
                    args[key]
                        .as_array()
                        .ok_or_else(|| format!("missing or invalid '{key}'"))?
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            v.as_f64()
                                .ok_or_else(|| format!("{key}[{i}] is not a number"))
                        })
                        .collect()
                };
                let start = [
                    get_f64(args, "startX")?,
                    get_f64(args, "startY")?,
                    get_f64(args, "startZ")?,
                ];
                let end = [
                    get_f64(args, "endX")?,
                    get_f64(args, "endY")?,
                    get_f64(args, "endZ")?,
                ];
                let eid = self
                    .make_nurbs_edge_impl(
                        start,
                        end,
                        get_u32(args, "degree")?,
                        numbers("knots")?,
                        &numbers("controlPoints")?,
                        numbers("weights")?,
                    )
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
                let wire = self
                    .offset_wire_impl(get_u32(args, "face")?, get_f64(args, "distance")?, None)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(wire))
            }
            "offsetWireWithJoinType" => {
                let join = args["joinType"]
                    .as_str()
                    .ok_or("missing or invalid 'joinType' string")?;
                let wire = self
                    .offset_wire_impl(
                        get_u32(args, "face")?,
                        get_f64(args, "distance")?,
                        Some(join),
                    )
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(wire))
            }
            "offsetWire2DWithJoin" => {
                let join = args["joinType"]
                    .as_str()
                    .ok_or("missing or invalid 'joinType' string")?;
                let wire = self
                    .offset_wire_2d_impl(get_u32(args, "wire")?, get_f64(args, "distance")?, join)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(wire))
            }
            "getNurbsCurveData" => {
                let edge = get_u32(args, "edge")?;
                let curve = self.extract_nurbs_curve(edge).map_err(|e| e.to_string())?;
                Ok(super::nurbs::curve_data_json(&curve))
            }
            "getNurbsSurfaceData" => {
                let face = get_u32(args, "face")?;
                let surface = self
                    .extract_nurbs_surface(face)
                    .map_err(|e| e.to_string())?;
                Ok(super::nurbs::surface_data_json(&surface))
            }
            "getNurbsSurfaceDataParity" => {
                let face = get_u32(args, "face")?;
                self.free_form_surface_data_parity(face)
                    .map_err(|e| e.to_string())
            }
            _ => Err(format!("unknown operation: {op}")),
        }
    }
}
