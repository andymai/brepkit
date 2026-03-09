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

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure;
use brepkit_operations::revolve::revolve;
use brepkit_operations::sweep::sweep;
use brepkit_operations::tessellate;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};
use wasm_bindgen::prelude::*;

use crate::error::{WasmError, validate_finite, validate_positive};
use crate::shapes::JsMesh;

/// Default tolerance for vertices created by the kernel.
const TOL: f64 = 1e-7;

/// The B-Rep modeling kernel.
///
/// Owns all topological state. JavaScript holds this reference and
/// invokes methods to create, transform, and query geometry.
#[wasm_bindgen]
pub struct BrepKernel {
    topo: Topology,
    assemblies: Vec<brepkit_operations::assembly::Assembly>,
    sketches: Vec<SketchState>,
}

/// Internal state for an in-progress sketch.
#[derive(Default)]
struct SketchState {
    points: Vec<brepkit_operations::sketch::SketchPoint>,
    constraints: Vec<brepkit_operations::sketch::Constraint>,
}

/// Filter edges to only those shared by two planar faces in a solid.
fn filter_planar_edges(
    topo: &brepkit_topology::Topology,
    solid_id: brepkit_topology::solid::SolidId,
    edge_ids: &[brepkit_topology::edge::EdgeId],
) -> Result<Vec<brepkit_topology::edge::EdgeId>, JsError> {
    use std::collections::HashMap;
    let solid_data = topo.solid(solid_id)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut edge_faces: HashMap<usize, Vec<brepkit_topology::face::FaceId>> = HashMap::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            edge_faces.entry(oe.edge().index()).or_default().push(fid);
        }
    }

    let mut result = Vec::new();
    for &eid in edge_ids {
        if let Some(adj_faces) = edge_faces.get(&eid.index()) {
            let all_planar = adj_faces.iter().all(|&fid| {
                topo.face(fid)
                    .map(|f| {
                        matches!(
                            f.surface(),
                            brepkit_topology::face::FaceSurface::Plane { .. }
                        )
                    })
                    .unwrap_or(false)
            });
            if all_planar {
                result.push(eid);
            }
        }
    }
    Ok(result)
}

/// Attempt fillet with rolling-ball, falling back to flat bevel on failure.
#[allow(deprecated)]
fn try_fillet(
    topo: &mut brepkit_topology::Topology,
    solid_id: brepkit_topology::solid::SolidId,
    edge_ids: &[brepkit_topology::edge::EdgeId],
    radius: f64,
) -> Result<brepkit_topology::solid::SolidId, brepkit_operations::OperationsError> {
    brepkit_operations::fillet::fillet_rolling_ball(topo, solid_id, edge_ids, radius)
        .or_else(|_| brepkit_operations::fillet::fillet(topo, solid_id, edge_ids, radius))
}

/// Extract a human-readable message from a `catch_unwind` panic payload.
fn panic_message(payload: &Box<dyn std::any::Any + Send>, operation: &str) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        format!("{operation} operation panicked: {s}")
    } else if let Some(s) = payload.downcast_ref::<String>() {
        format!("{operation} operation panicked: {s}")
    } else {
        format!("{operation} operation panicked (unknown cause)")
    }
}

/// Sample a closed periodic curve (period = TAU) into a flat `[x, y, z, ...]` buffer.
///
/// Produces `n` evenly-spaced points in `[0, TAU)` using the supplied `evaluate` function.
fn sample_full_period_curve(
    n: usize,
    evaluate: impl Fn(f64) -> brepkit_math::vec::Point3,
) -> Vec<f64> {
    let mut result = Vec::with_capacity(n * 3);
    for i in 0..n {
        #[allow(clippy::cast_precision_loss)]
        let t = std::f64::consts::TAU * (i as f64) / ((n - 1) as f64);
        let p = evaluate(t);
        result.push(p.x());
        result.push(p.y());
        result.push(p.z());
    }
    result
}

#[wasm_bindgen]
impl BrepKernel {
    // ── Construction ────────────────────────────────────────────────

    /// Create a new, empty kernel.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            topo: Topology::new(),
            assemblies: Vec::new(),
            sketches: Vec::new(),
        }
    }

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

    // ── Primitive shapes ─────────────────────────────────────────

    /// Create a box solid with the given dimensions, centered at the origin.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is non-positive or non-finite.
    #[wasm_bindgen(js_name = "makeBox")]
    pub fn make_box_solid(&mut self, dx: f64, dy: f64, dz: f64) -> Result<u32, JsError> {
        validate_positive(dx, "dx")?;
        validate_positive(dy, "dy")?;
        validate_positive(dz, "dz")?;
        let solid_id = brepkit_operations::primitives::make_box(&mut self.topo, dx, dy, dz)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a cylinder solid centered at the origin, axis along +Z.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if radius or height is non-positive.
    #[wasm_bindgen(js_name = "makeCylinder")]
    pub fn make_cylinder_solid(&mut self, radius: f64, height: f64) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        validate_positive(height, "height")?;
        let solid_id =
            brepkit_operations::primitives::make_cylinder(&mut self.topo, radius, height)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a sphere solid centered at the origin.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if radius is non-positive or segments < 4.
    #[wasm_bindgen(js_name = "makeSphere")]
    pub fn make_sphere_solid(&mut self, radius: f64, segments: u32) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        let solid_id =
            brepkit_operations::primitives::make_sphere(&mut self.topo, radius, segments as usize)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a cone or frustum solid centered at the origin, axis along +Z.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if height is non-positive or both radii are zero.
    #[wasm_bindgen(js_name = "makeCone")]
    pub fn make_cone_solid(
        &mut self,
        bottom_radius: f64,
        top_radius: f64,
        height: f64,
    ) -> Result<u32, JsError> {
        validate_finite(bottom_radius, "bottom_radius")?;
        validate_finite(top_radius, "top_radius")?;
        validate_positive(height, "height")?;
        let solid_id = brepkit_operations::primitives::make_cone(
            &mut self.topo,
            bottom_radius,
            top_radius,
            height,
        )?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a torus solid centered at the origin in the XY plane.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if radii are non-positive or minor >= major.
    #[wasm_bindgen(js_name = "makeTorus")]
    pub fn make_torus_solid(
        &mut self,
        major_radius: f64,
        minor_radius: f64,
        segments: u32,
    ) -> Result<u32, JsError> {
        validate_positive(major_radius, "major_radius")?;
        validate_positive(minor_radius, "minor_radius")?;
        let solid_id = brepkit_operations::primitives::make_torus(
            &mut self.topo,
            major_radius,
            minor_radius,
            segments as usize,
        )?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create an ellipsoid solid centered at the origin.
    ///
    /// Built by creating a unit sphere and scaling it by `(rx, ry, rz)`.
    ///
    /// # Errors
    ///
    /// Returns an error if any radius is non-positive.
    #[wasm_bindgen(js_name = "makeEllipsoid")]
    pub fn make_ellipsoid_solid(&mut self, rx: f64, ry: f64, rz: f64) -> Result<u32, JsError> {
        validate_positive(rx, "rx")?;
        validate_positive(ry, "ry")?;
        validate_positive(rz, "rz")?;
        // Create a unit sphere, then scale it non-uniformly.
        let solid_id = brepkit_operations::primitives::make_sphere(&mut self.topo, 1.0, 16)?;
        let mat = brepkit_math::mat::Mat4::scale(rx, ry, rz);
        transform_solid(&mut self.topo, solid_id, &mat)?;
        Ok(solid_id_to_u32(solid_id))
    }

    // ── Section ───────────────────────────────────────────────────

    /// Section a solid with a plane, returning cross-section face handles.
    ///
    /// Returns an array of face handles (`u32[]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or the plane doesn't
    /// intersect the solid.
    #[wasm_bindgen(js_name = "section")]
    #[allow(clippy::too_many_arguments)]
    pub fn section_solid(
        &mut self,
        solid: u32,
        px: f64,
        py: f64,
        pz: f64,
        nx: f64,
        ny: f64,
        nz: f64,
    ) -> Result<Vec<u32>, JsError> {
        validate_finite(px, "px")?;
        validate_finite(py, "py")?;
        validate_finite(pz, "pz")?;
        validate_finite(nx, "nx")?;
        validate_finite(ny, "ny")?;
        validate_finite(nz, "nz")?;
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::section::section(
            &mut self.topo,
            solid_id,
            Point3::new(px, py, pz),
            Vec3::new(nx, ny, nz),
        )?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(result.faces.iter().map(|f| f.index() as u32).collect())
    }

    // ── Loft ──────────────────────────────────────────────────────

    /// Loft two or more profile faces into a solid.
    ///
    /// Takes an array of face handles. Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 faces or profiles have
    /// different vertex counts.
    #[wasm_bindgen(js_name = "loft")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn loft_faces(&mut self, faces: Vec<u32>) -> Result<u32, JsError> {
        let face_ids: Vec<brepkit_topology::face::FaceId> = faces
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let solid_id = brepkit_operations::loft::loft(&mut self.topo, &face_ids)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Loft profiles with smooth NURBS interpolation.
    ///
    /// Like `loft()`, but produces smooth NURBS side surfaces for 3+
    /// profiles instead of piecewise-planar quads. The surfaces
    /// interpolate through all intermediate profiles with C1+ continuity.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 profiles are given, profiles have
    /// different vertex counts, or surface fitting fails.
    #[wasm_bindgen(js_name = "loftSmooth")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn loft_smooth_faces(&mut self, faces: Vec<u32>) -> Result<u32, JsError> {
        let face_ids: Vec<brepkit_topology::face::FaceId> = faces
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let solid_id = brepkit_operations::loft::loft_smooth(&mut self.topo, &face_ids)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Loft profiles with options for start/end points and ruled mode.
    ///
    /// `options` is a JSON string with optional fields:
    /// - `startPoint: [x, y, z]` — apex point before first profile
    /// - `endPoint: [x, y, z]` — apex point after last profile
    /// - `ruled: bool` — true for ruled (linear) surfaces (default), false for smooth
    #[wasm_bindgen(js_name = "loftWithOptions")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn loft_with_options(&mut self, faces: Vec<u32>, options: &str) -> Result<u32, JsError> {
        let opts: serde_json::Value =
            serde_json::from_str(options).unwrap_or(serde_json::Value::Null);

        let mut face_ids: Vec<brepkit_topology::face::FaceId> = faces
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;

        // If startPoint is given, create a tiny degenerate triangle face at that point
        // and prepend it to the profiles.
        if let Some(sp) = opts.get("startPoint").and_then(|v| v.as_array()) {
            if sp.len() >= 3 {
                let x = sp[0].as_f64().unwrap_or(0.0);
                let y = sp[1].as_f64().unwrap_or(0.0);
                let z = sp[2].as_f64().unwrap_or(0.0);
                let apex_face = create_apex_face(&mut self.topo, Point3::new(x, y, z), &face_ids)?;
                face_ids.insert(0, apex_face);
            }
        }

        // If endPoint is given, create a tiny degenerate triangle face and append.
        if let Some(ep) = opts.get("endPoint").and_then(|v| v.as_array()) {
            if ep.len() >= 3 {
                let x = ep[0].as_f64().unwrap_or(0.0);
                let y = ep[1].as_f64().unwrap_or(0.0);
                let z = ep[2].as_f64().unwrap_or(0.0);
                let apex_face = create_apex_face(&mut self.topo, Point3::new(x, y, z), &face_ids)?;
                face_ids.push(apex_face);
            }
        }

        let ruled = opts.get("ruled").and_then(|v| v.as_bool()).unwrap_or(true);

        let solid_id = if ruled {
            brepkit_operations::loft::loft(&mut self.topo, &face_ids)?
        } else {
            brepkit_operations::loft::loft_smooth(&mut self.topo, &face_ids)?
        };
        Ok(solid_id_to_u32(solid_id))
    }

    // ── Shell ─────────────────────────────────────────────────────

    /// Hollow a solid with uniform wall thickness.
    ///
    /// `open_faces` is an array of face handles to remove (creating openings).
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if thickness is non-positive or the solid is invalid.
    #[wasm_bindgen(js_name = "shell")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn shell_solid(
        &mut self,
        solid: u32,
        thickness: f64,
        open_faces: Vec<u32>,
    ) -> Result<u32, JsError> {
        validate_positive(thickness, "thickness")?;
        let solid_id = self.resolve_solid(solid)?;
        let open_face_ids: Vec<brepkit_topology::face::FaceId> = open_faces
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let result = brepkit_operations::shell_op::shell(
            &mut self.topo,
            solid_id,
            thickness,
            &open_face_ids,
        )?;
        Ok(solid_id_to_u32(result))
    }

    // ── Chamfer ───────────────────────────────────────────────────

    /// Chamfer edges of a solid.
    ///
    /// `edge_handles` is an array of edge handles. Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if distance is non-positive or edges are invalid.
    #[wasm_bindgen(js_name = "chamfer")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn chamfer_solid(
        &mut self,
        solid: u32,
        edge_handles: Vec<u32>,
        distance: f64,
    ) -> Result<u32, JsError> {
        validate_positive(distance, "distance")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids: Vec<brepkit_topology::edge::EdgeId> = edge_handles
            .iter()
            .map(|&h| self.resolve_edge(h))
            .collect::<Result<_, _>>()?;
        let result =
            brepkit_operations::chamfer::chamfer(&mut self.topo, solid_id, &edge_ids, distance)?;
        Ok(solid_id_to_u32(result))
    }

    // ── Fillet ────────────────────────────────────────────────────

    /// Fillet (round) edges of a solid.
    ///
    /// `edge_handles` is an array of edge handles. Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if radius is non-positive or edges are invalid.
    #[wasm_bindgen(js_name = "fillet")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn fillet_solid(
        &mut self,
        solid: u32,
        edge_handles: Vec<u32>,
        radius: f64,
    ) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids: Vec<brepkit_topology::edge::EdgeId> = edge_handles
            .iter()
            .map(|&h| self.resolve_edge(h))
            .collect::<Result<_, _>>()?;
        // Use the rolling-ball fillet algorithm for true G1-continuous NURBS
        // blend surfaces. Falls back to the planar fillet if rolling-ball fails.
        // If the full set of edges fails (e.g. edges adjacent to NURBS faces from
        // a prior fillet), filter to edges between two planar faces and retry.
        //
        // Wrap in catch_unwind to prevent panics from propagating across the
        // WASM FFI boundary, which would abort the entire WASM instance.
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<u32, JsError> {
                let solid = if let Ok(s) = try_fillet(&mut self.topo, solid_id, &edge_ids, radius) {
                    s
                } else {
                    // Filter to edges where both adjacent faces are planar.
                    let planar_edges = filter_planar_edges(&self.topo, solid_id, &edge_ids)?;
                    if planar_edges.is_empty() {
                        solid_id
                    } else {
                        try_fillet(&mut self.topo, solid_id, &planar_edges, radius)
                            .map_err(|e| JsError::new(&e.to_string()))?
                    }
                };
                Ok(solid_id_to_u32(solid))
            }));
        match result {
            Ok(inner) => inner,
            Err(panic_info) => {
                let msg = panic_message(&panic_info, "Fillet");
                Err(JsError::new(&msg))
            }
        }
    }

    // ── Operations ─────────────────────────────────────────────────

    /// Extrude a planar face along a direction vector to create a solid.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid or the extrusion fails.
    #[wasm_bindgen(js_name = "extrude")]
    pub fn extrude_face(
        &mut self,
        face: u32,
        dir_x: f64,
        dir_y: f64,
        dir_z: f64,
        distance: f64,
    ) -> Result<u32, JsError> {
        validate_finite(dir_x, "dir_x")?;
        validate_finite(dir_y, "dir_y")?;
        validate_finite(dir_z, "dir_z")?;
        validate_finite(distance, "distance")?;

        let face_id = self.resolve_face(face)?;
        let direction = Vec3::new(dir_x, dir_y, dir_z);
        let solid_id = extrude(&mut self.topo, face_id, direction, distance)?;

        Ok(solid_id_to_u32(solid_id))
    }

    /// Revolve a planar face around an axis to create a solid of revolution.
    ///
    /// The axis is defined by an origin point `(ox, oy, oz)` and a direction
    /// `(dx, dy, dz)`. The angle is in degrees and must be in (0, 360].
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if any input is non-finite, the face handle is
    /// invalid, or the revolve operation fails.
    #[wasm_bindgen(js_name = "revolve")]
    #[allow(clippy::too_many_arguments)]
    pub fn revolve_face(
        &mut self,
        face: u32,
        ox: f64,
        oy: f64,
        oz: f64,
        dx: f64,
        dy: f64,
        dz: f64,
        angle_degrees: f64,
    ) -> Result<u32, JsError> {
        validate_finite(ox, "ox")?;
        validate_finite(oy, "oy")?;
        validate_finite(oz, "oz")?;
        validate_finite(dx, "dx")?;
        validate_finite(dy, "dy")?;
        validate_finite(dz, "dz")?;
        validate_finite(angle_degrees, "angle_degrees")?;
        if angle_degrees <= 0.0 || angle_degrees > 360.0 {
            return Err(WasmError::InvalidInput {
                reason: format!("angle_degrees must be in (0, 360], got {angle_degrees}"),
            }
            .into());
        }

        let face_id = self.resolve_face(face)?;
        let origin = Point3::new(ox, oy, oz);
        let direction = Vec3::new(dx, dy, dz);
        let angle_radians = angle_degrees.to_radians();

        let solid_id = revolve(&mut self.topo, face_id, origin, direction, angle_radians)?;

        Ok(solid_id_to_u32(solid_id))
    }

    /// Sweep a planar face along a NURBS curve path to create a solid.
    ///
    /// The path is specified as flat arrays for JS interop:
    /// - `path_degree` — polynomial degree of the path curve
    /// - `path_knots` — knot vector
    /// - `path_control_points` — flat `[x,y,z, ...]` control point coordinates
    /// - `path_weights` — per-control-point weights
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid, the NURBS arrays have
    /// inconsistent lengths, or the sweep operation fails.
    #[wasm_bindgen(js_name = "sweep")]
    #[allow(clippy::needless_pass_by_value)] // wasm-bindgen requires owned Vec
    pub fn sweep_face(
        &mut self,
        face: u32,
        path_degree: u32,
        path_knots: Vec<f64>,
        path_control_points: Vec<f64>,
        path_weights: Vec<f64>,
    ) -> Result<u32, JsError> {
        // Validate coordinate array length.
        if path_control_points.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "path_control_points length must be a multiple of 3, got {}",
                    path_control_points.len()
                ),
            }
            .into());
        }
        let num_pts = path_control_points.len() / 3;

        if path_weights.len() != num_pts {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "path_weights length ({}) must match number of control points ({num_pts})",
                    path_weights.len()
                ),
            }
            .into());
        }

        // Validate all values are finite.
        if let Some(pos) = path_knots.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("path_knots[{pos}] is not finite"),
            }
            .into());
        }
        if let Some(pos) = path_control_points.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("path_control_points[{pos}] is not finite"),
            }
            .into());
        }
        if let Some(pos) = path_weights.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("path_weights[{pos}] is not finite"),
            }
            .into());
        }

        let face_id = self.resolve_face(face)?;

        let control_points: Vec<Point3> = path_control_points
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();

        let path_curve = NurbsCurve::new(
            path_degree as usize,
            path_knots,
            control_points,
            path_weights,
        )?;

        let solid_id = sweep(&mut self.topo, face_id, &path_curve)?;

        Ok(solid_id_to_u32(solid_id))
    }

    /// Sweep a face along a path with smooth NURBS side surfaces.
    ///
    /// Like `sweep()`, but produces a single NURBS surface per edge strip
    /// instead of multiple flat quads, giving smooth geometry that
    /// tessellates to arbitrary quality.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if the face or path is invalid, or surface fitting fails.
    #[wasm_bindgen(js_name = "sweepSmooth")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn sweep_smooth_face(
        &mut self,
        face: u32,
        path_degree: u32,
        path_knots: Vec<f64>,
        path_control_points: Vec<f64>,
        path_weights: Vec<f64>,
    ) -> Result<u32, JsError> {
        if path_control_points.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "path_control_points length must be a multiple of 3, got {}",
                    path_control_points.len()
                ),
            }
            .into());
        }

        let face_id = self.resolve_face(face)?;
        let n_cp = path_control_points.len() / 3;
        let control_points: Vec<Point3> = (0..n_cp)
            .map(|i| {
                Point3::new(
                    path_control_points[i * 3],
                    path_control_points[i * 3 + 1],
                    path_control_points[i * 3 + 2],
                )
            })
            .collect();

        let weights = if path_weights.is_empty() {
            vec![1.0; n_cp]
        } else {
            path_weights
        };

        #[allow(clippy::cast_possible_truncation)]
        let path_curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            path_degree as usize,
            path_knots,
            control_points,
            weights,
        )?;

        let solid_id =
            brepkit_operations::sweep::sweep_smooth(&mut self.topo, face_id, &path_curve)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Apply a 4×4 affine transform to a solid (in place).
    ///
    /// The `matrix` must contain exactly 16 values in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid, the matrix doesn't
    /// have 16 elements, or the matrix is singular.
    #[wasm_bindgen(js_name = "transformSolid")]
    #[allow(clippy::needless_pass_by_value)] // wasm-bindgen requires owned Vec
    pub fn transform_solid_wasm(&mut self, solid: u32, matrix: Vec<f64>) -> Result<(), JsError> {
        if matrix.len() != 16 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "transform matrix must have 16 elements, got {}",
                    matrix.len()
                ),
            }
            .into());
        }

        if let Some(pos) = matrix.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("matrix element at index {pos} is not finite"),
            }
            .into());
        }

        let solid_id = self.resolve_solid(solid)?;

        let rows = std::array::from_fn(|i| std::array::from_fn(|j| matrix[i * 4 + j]));
        let mat = Mat4(rows);

        transform_solid(&mut self.topo, solid_id, &mat)?;
        Ok(())
    }

    /// Compose (multiply) two 4x4 transformation matrices.
    ///
    /// Returns the composed matrix as a flat 16-element array (row-major).
    /// This computes `a * b`, meaning `b` is applied first, then `a`.
    ///
    /// # Errors
    ///
    /// Returns an error if either matrix doesn't have 16 elements.
    #[wasm_bindgen(js_name = "composeTransforms")]
    #[allow(clippy::needless_pass_by_value, clippy::unused_self)]
    pub fn compose_transforms(
        &self,
        matrix_a: Vec<f64>,
        matrix_b: Vec<f64>,
    ) -> Result<Vec<f64>, JsError> {
        if matrix_a.len() != 16 {
            return Err(WasmError::InvalidInput {
                reason: format!("matrix A must have 16 elements, got {}", matrix_a.len()),
            }
            .into());
        }
        if matrix_b.len() != 16 {
            return Err(WasmError::InvalidInput {
                reason: format!("matrix B must have 16 elements, got {}", matrix_b.len()),
            }
            .into());
        }
        let rows_a = std::array::from_fn(|i| std::array::from_fn(|j| matrix_a[i * 4 + j]));
        let rows_b = std::array::from_fn(|i| std::array::from_fn(|j| matrix_b[i * 4 + j]));
        let result = Mat4(rows_a) * Mat4(rows_b);
        let mut out = Vec::with_capacity(16);
        for row in &result.0 {
            out.extend_from_slice(row);
        }
        Ok(out)
    }

    // ── Boolean operations ──────────────────────────────────────────

    /// Fuse (union) two solids into one.
    ///
    /// Returns a new solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty or non-manifold result.
    #[wasm_bindgen(js_name = "fuse")]
    pub fn fuse(&mut self, a: u32, b: u32) -> Result<u32, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let result = boolean(&mut self.topo, BooleanOp::Fuse, a_id, b_id)?;
        Ok(solid_id_to_u32(result))
    }

    /// Cut (subtract) solid `b` from solid `a`.
    ///
    /// Returns a new solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty or non-manifold result.
    #[wasm_bindgen(js_name = "cut")]
    pub fn cut(&mut self, a: u32, b: u32) -> Result<u32, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let result = boolean(&mut self.topo, BooleanOp::Cut, a_id, b_id)?;
        Ok(solid_id_to_u32(result))
    }

    /// Intersect two solids, keeping only their common volume.
    ///
    /// Returns a new solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty result.
    #[wasm_bindgen(js_name = "intersect")]
    pub fn intersect_solids(&mut self, a: u32, b: u32) -> Result<u32, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let result = boolean(&mut self.topo, BooleanOp::Intersect, a_id, b_id)?;
        Ok(solid_id_to_u32(result))
    }

    // ── Boolean operations with evolution tracking ─────────────────

    /// Fuse (union) two solids and return evolution tracking data.
    ///
    /// Returns a JSON string: `{"solid": <u32>, "evolution": {...}}`.
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty or non-manifold result.
    #[wasm_bindgen(js_name = "fuseWithEvolution")]
    pub fn fuse_with_evolution(&mut self, a: u32, b: u32) -> Result<JsValue, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let (result, evo) = brepkit_operations::boolean::boolean_with_evolution(
            &mut self.topo,
            BooleanOp::Fuse,
            a_id,
            b_id,
        )?;
        let json = format!(
            "{{\"solid\":{},\"evolution\":{}}}",
            solid_id_to_u32(result),
            evo.to_json()
        );
        Ok(JsValue::from_str(&json))
    }

    /// Cut (subtract) solid `b` from solid `a` and return evolution tracking data.
    ///
    /// Returns a JSON string: `{"solid": <u32>, "evolution": {...}}`.
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty or non-manifold result.
    #[wasm_bindgen(js_name = "cutWithEvolution")]
    pub fn cut_with_evolution(&mut self, a: u32, b: u32) -> Result<JsValue, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let (result, evo) = brepkit_operations::boolean::boolean_with_evolution(
            &mut self.topo,
            BooleanOp::Cut,
            a_id,
            b_id,
        )?;
        let json = format!(
            "{{\"solid\":{},\"evolution\":{}}}",
            solid_id_to_u32(result),
            evo.to_json()
        );
        Ok(JsValue::from_str(&json))
    }

    /// Intersect two solids and return evolution tracking data.
    ///
    /// Returns a JSON string: `{"solid": <u32>, "evolution": {...}}`.
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty result.
    #[wasm_bindgen(js_name = "intersectWithEvolution")]
    pub fn intersect_with_evolution(&mut self, a: u32, b: u32) -> Result<JsValue, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let (result, evo) = brepkit_operations::boolean::boolean_with_evolution(
            &mut self.topo,
            BooleanOp::Intersect,
            a_id,
            b_id,
        )?;
        let json = format!(
            "{{\"solid\":{},\"evolution\":{}}}",
            solid_id_to_u32(result),
            evo.to_json()
        );
        Ok(JsValue::from_str(&json))
    }

    // ── Export ─────────────────────────────────────────────────────

    /// Export a solid to 3MF format (ZIP archive as bytes).
    ///
    /// Returns a `Uint8Array` in JavaScript containing the `.3mf` file.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "export3mf")]
    pub fn export_3mf(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let bytes = brepkit_io::threemf::write_threemf(&self.topo, &[solid_id], deflection)?;
        Ok(bytes)
    }

    /// Export a solid to binary STL format.
    ///
    /// Returns a `Uint8Array` containing the `.stl` file.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "exportStl")]
    pub fn export_stl(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let bytes = brepkit_io::stl::writer::write_stl(
            &self.topo,
            &[solid_id],
            deflection,
            brepkit_io::stl::writer::StlFormat::Binary,
        )?;
        Ok(bytes)
    }

    /// Export a solid to STL ASCII format.
    ///
    /// Returns the ASCII STL as UTF-8 bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "exportStlAscii")]
    pub fn export_stl_ascii(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let bytes = brepkit_io::stl::writer::write_stl(
            &self.topo,
            &[solid_id],
            deflection,
            brepkit_io::stl::writer::StlFormat::Ascii,
        )?;
        Ok(bytes)
    }

    /// Export a solid to OBJ format (UTF-8 string as bytes).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "exportObj")]
    pub fn export_obj(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let obj_str = brepkit_io::obj::write_obj(&self.topo, &[solid_id], deflection)?;
        Ok(obj_str.into_bytes())
    }

    /// Export a solid to glTF binary (.glb) format.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "exportGlb")]
    pub fn export_glb(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let glb = brepkit_io::gltf::write_glb(&self.topo, &[solid_id], deflection)?;
        Ok(glb)
    }

    /// Export a solid to PLY format (binary little-endian).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "exportPly")]
    pub fn export_ply(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let ply = brepkit_io::ply::write_ply(
            &self.topo,
            &[solid_id],
            deflection,
            brepkit_io::ply::writer::PlyFormat::BinaryLittleEndian,
        )?;
        Ok(ply)
    }

    // ── Import ──────────────────────────────────────────────────────

    /// Import an OBJ file and return a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is malformed or mesh import fails.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "importObj")]
    pub fn import_obj(&mut self, data: &[u8]) -> Result<u32, JsError> {
        let text = std::str::from_utf8(data).map_err(|e| WasmError::InvalidInput {
            reason: format!("OBJ must be valid UTF-8: {e}"),
        })?;
        let mesh = brepkit_io::obj::read_obj(text)?;
        let solid_id = brepkit_io::stl::import::import_mesh(&mut self.topo, &mesh, 1e-7)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(solid_id.index() as u32)
    }

    /// Import a GLB (glTF binary) file and return a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is malformed or mesh import fails.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "importGlb")]
    pub fn import_glb(&mut self, data: &[u8]) -> Result<u32, JsError> {
        let mesh = brepkit_io::gltf::read_glb(data)?;
        let solid_id = brepkit_io::stl::import::import_mesh(&mut self.topo, &mesh, 1e-7)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(solid_id.index() as u32)
    }

    /// Import an STL file (binary or ASCII) and return a solid handle.
    ///
    /// The mesh triangles are converted to planar B-Rep faces with
    /// vertex merging.
    ///
    /// # Errors
    ///
    /// Returns an error if the STL data is malformed or empty.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "importStl")]
    pub fn import_stl(&mut self, data: &[u8]) -> Result<u32, JsError> {
        let mesh = brepkit_io::stl::reader::read_stl(data)?;
        let solid_id = brepkit_io::stl::import::import_mesh(&mut self.topo, &mesh, TOL)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Import a 3MF file and return solid handles.
    ///
    /// Returns handles for each object found in the 3MF archive.
    ///
    /// # Errors
    ///
    /// Returns an error if the 3MF data is malformed.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "import3mf")]
    pub fn import_3mf(&mut self, data: &[u8]) -> Result<Vec<u32>, JsError> {
        let meshes = brepkit_io::threemf::reader::read_threemf(data)?;
        let mut handles = Vec::new();
        for mesh in &meshes {
            let solid_id = brepkit_io::stl::import::import_mesh(&mut self.topo, mesh, TOL)?;
            handles.push(solid_id_to_u32(solid_id));
        }
        Ok(handles)
    }

    /// Import a triangle mesh from flat vertex/index arrays.
    ///
    /// `positions` is a flat `[x0,y0,z0, x1,y1,z1, ...]` array.
    /// `indices` is a flat `[i0,i1,i2, i3,i4,i5, ...]` array of triangle
    /// vertex indices. Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the arrays are malformed or mesh import fails.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "importIndexedMesh")]
    pub fn import_indexed_mesh(
        &mut self,
        positions: &[f64],
        indices: &[u32],
    ) -> Result<u32, JsError> {
        use brepkit_math::vec::Point3;

        if positions.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "positions length {} is not a multiple of 3",
                    positions.len()
                ),
            }
            .into());
        }
        if indices.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!("indices length {} is not a multiple of 3", indices.len()),
            }
            .into());
        }

        let verts: Vec<Point3> = positions
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();
        let normals = Vec::new();

        let mesh = brepkit_operations::tessellate::TriangleMesh {
            positions: verts,
            normals,
            indices: indices.to_vec(),
        };

        let solid_id = brepkit_io::stl::import::import_mesh(&mut self.topo, &mesh, TOL)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Export a solid to STEP AP203 format.
    ///
    /// Returns the STEP file as a UTF-8 encoded byte vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "exportStep")]
    pub fn export_step(&self, solid: u32) -> Result<Vec<u8>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let step_str = brepkit_io::step::writer::write_step(&self.topo, &[solid_id])?;
        Ok(step_str.into_bytes())
    }

    /// Import a STEP file and return solid handles.
    ///
    /// Returns handles for each solid found in the STEP file.
    ///
    /// # Errors
    ///
    /// Returns an error if the STEP data is malformed.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "importStep")]
    pub fn import_step(&mut self, data: &[u8]) -> Result<Vec<u32>, JsError> {
        let text = std::str::from_utf8(data)
            .map_err(|e| JsError::new(&format!("STEP data is not valid UTF-8: {e}")))?;
        let solid_ids = brepkit_io::step::reader::read_step(text, &mut self.topo)?;
        Ok(solid_ids.iter().map(|id| solid_id_to_u32(*id)).collect())
    }

    /// Offset a face by a distance along its surface normal.
    ///
    /// Returns the new offset face handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid or the operation fails.
    #[wasm_bindgen(js_name = "offsetFace")]
    pub fn offset_face_wasm(
        &mut self,
        face: u32,
        distance: f64,
        samples: u32,
    ) -> Result<u32, JsError> {
        validate_finite(distance, "distance")?;
        let face_id = self.resolve_face(face)?;
        let result = brepkit_operations::offset_face::offset_face(
            &mut self.topo,
            face_id,
            distance,
            samples as usize,
        )?;
        Ok(face_id_to_u32(result))
    }

    // ── IGES Import/Export ────────────────────────────────────────

    /// Export a solid to IGES format.
    ///
    /// Returns the IGES file as a UTF-8 encoded byte vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "exportIges")]
    pub fn export_iges(&self, solid: u32) -> Result<Vec<u8>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let iges_str = brepkit_io::iges::writer::write_iges(&self.topo, &[solid_id])?;
        Ok(iges_str.into_bytes())
    }

    /// Import an IGES file and return solid handles.
    ///
    /// # Errors
    ///
    /// Returns an error if the IGES data is malformed.
    #[cfg(feature = "io")]
    #[wasm_bindgen(js_name = "importIges")]
    pub fn import_iges(&mut self, data: &[u8]) -> Result<Vec<u32>, JsError> {
        let text = std::str::from_utf8(data)
            .map_err(|e| JsError::new(&format!("IGES data is not valid UTF-8: {e}")))?;
        let solid_ids = brepkit_io::iges::reader::read_iges(text, &mut self.topo)?;
        Ok(solid_ids.iter().map(|id| solid_id_to_u32(*id)).collect())
    }

    // ── Helical Sweep ───────────────────────────────────────────

    /// Create a helical sweep of a profile face.
    ///
    /// Sweeps the profile along a helix defined by axis, radius, pitch,
    /// and number of turns. Used for generating thread geometry.
    ///
    /// # Errors
    ///
    /// Returns an error if parameters are invalid or the sweep fails.
    #[wasm_bindgen(js_name = "helicalSweep")]
    #[allow(clippy::too_many_arguments)]
    pub fn helical_sweep_wasm(
        &mut self,
        profile: u32,
        axis_origin_x: f64,
        axis_origin_y: f64,
        axis_origin_z: f64,
        axis_dir_x: f64,
        axis_dir_y: f64,
        axis_dir_z: f64,
        radius: f64,
        pitch: f64,
        turns: f64,
    ) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        validate_positive(pitch, "pitch")?;
        let face_id = self.resolve_face(profile)?;

        let origin = brepkit_math::vec::Point3::new(axis_origin_x, axis_origin_y, axis_origin_z);
        let axis_dir = brepkit_math::vec::Vec3::new(axis_dir_x, axis_dir_y, axis_dir_z);

        let solid_id = brepkit_operations::helix::helical_sweep(
            &mut self.topo,
            face_id,
            origin,
            axis_dir,
            radius,
            pitch,
            turns,
            8,
        )?;
        Ok(solid_id_to_u32(solid_id))
    }

    // ── Copy / Mirror / Pattern ───────────────────────────────────

    /// Deep copy a solid, returning a new independent solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "copySolid")]
    pub fn copy_solid_wasm(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let copy = brepkit_operations::copy::copy_solid(&mut self.topo, solid_id)?;
        Ok(solid_id_to_u32(copy))
    }

    /// Deep copy a wire, returning a new independent wire handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the wire handle is invalid.
    #[wasm_bindgen(js_name = "copyWire")]
    pub fn copy_wire_wasm(&mut self, wire: u32) -> Result<u32, JsError> {
        let wire_id = self.resolve_wire(wire)?;
        let copy = brepkit_operations::copy::copy_wire(&mut self.topo, wire_id)?;
        Ok(wire_id_to_u32(copy))
    }

    /// Apply a 4×4 affine transform to a wire (in place).
    ///
    /// The `matrix` must contain exactly 16 values in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error if the wire handle is invalid, the matrix doesn't
    /// have 16 elements, or the matrix is singular.
    #[wasm_bindgen(js_name = "transformWire")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn transform_wire_wasm(&mut self, wire: u32, matrix: Vec<f64>) -> Result<(), JsError> {
        if matrix.len() != 16 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "transform matrix must have 16 elements, got {}",
                    matrix.len()
                ),
            }
            .into());
        }

        if let Some(pos) = matrix.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("matrix element at index {pos} is not finite"),
            }
            .into());
        }

        let wire_id = self.resolve_wire(wire)?;
        let rows = std::array::from_fn(|i| std::array::from_fn(|j| matrix[i * 4 + j]));
        let mat = Mat4(rows);
        brepkit_operations::transform::transform_wire(&mut self.topo, wire_id, &mat)?;
        Ok(())
    }

    /// Copy a solid and apply a 4×4 row-major affine transform in one pass.
    ///
    /// Equivalent to `copySolid` + `transformSolid` but performs both in a
    /// single topology traversal, avoiding redundant NURBS clones.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid, the matrix doesn't
    /// have 16 elements, or the matrix is singular.
    #[wasm_bindgen(js_name = "copyAndTransformSolid")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn copy_and_transform_solid_wasm(
        &mut self,
        solid: u32,
        matrix: Vec<f64>,
    ) -> Result<u32, JsError> {
        if matrix.len() != 16 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "transform matrix must have 16 elements, got {}",
                    matrix.len()
                ),
            }
            .into());
        }

        if let Some(pos) = matrix.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("matrix element at index {pos} is not finite"),
            }
            .into());
        }

        let solid_id = self.resolve_solid(solid)?;

        let rows = std::array::from_fn(|i| std::array::from_fn(|j| matrix[i * 4 + j]));
        let mat = Mat4(rows);

        let copy =
            brepkit_operations::copy::copy_and_transform_solid(&mut self.topo, solid_id, &mat)?;
        Ok(solid_id_to_u32(copy))
    }

    /// Mirror a solid across a plane.
    ///
    /// Returns a new solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or the normal is zero.
    #[wasm_bindgen(js_name = "mirror")]
    #[allow(clippy::too_many_arguments)]
    pub fn mirror_solid(
        &mut self,
        solid: u32,
        px: f64,
        py: f64,
        pz: f64,
        nx: f64,
        ny: f64,
        nz: f64,
    ) -> Result<u32, JsError> {
        validate_finite(px, "px")?;
        validate_finite(py, "py")?;
        validate_finite(pz, "pz")?;
        validate_finite(nx, "nx")?;
        validate_finite(ny, "ny")?;
        validate_finite(nz, "nz")?;
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::mirror::mirror(
            &mut self.topo,
            solid_id,
            Point3::new(px, py, pz),
            Vec3::new(nx, ny, nz),
        )?;
        Ok(solid_id_to_u32(result))
    }

    /// Create a linear pattern of a solid.
    ///
    /// Returns a compound handle containing all copies.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid.
    #[wasm_bindgen(js_name = "linearPattern")]
    #[allow(clippy::too_many_arguments)]
    pub fn linear_pattern_wasm(
        &mut self,
        solid: u32,
        dx: f64,
        dy: f64,
        dz: f64,
        spacing: f64,
        count: u32,
    ) -> Result<u32, JsError> {
        validate_finite(dx, "dx")?;
        validate_finite(dy, "dy")?;
        validate_finite(dz, "dz")?;
        validate_positive(spacing, "spacing")?;
        let solid_id = self.resolve_solid(solid)?;
        let compound = brepkit_operations::pattern::linear_pattern(
            &mut self.topo,
            solid_id,
            Vec3::new(dx, dy, dz),
            spacing,
            count as usize,
        )?;
        Ok(compound_id_to_u32(compound))
    }

    // ── Grid Pattern ──────────────────────────────────────────────

    /// Create a 2D grid pattern of a solid.
    ///
    /// Produces `count_x × count_y` copies arranged in a rectangular grid.
    #[wasm_bindgen(js_name = "gridPattern")]
    #[allow(clippy::too_many_arguments)]
    pub fn grid_pattern(
        &mut self,
        solid: u32,
        dir_x_x: f64,
        dir_x_y: f64,
        dir_x_z: f64,
        dir_y_x: f64,
        dir_y_y: f64,
        dir_y_z: f64,
        spacing_x: f64,
        spacing_y: f64,
        count_x: u32,
        count_y: u32,
    ) -> Result<u32, JsError> {
        validate_finite(dir_x_x, "dir_x_x")?;
        validate_finite(dir_x_y, "dir_x_y")?;
        validate_finite(dir_x_z, "dir_x_z")?;
        validate_finite(dir_y_x, "dir_y_x")?;
        validate_finite(dir_y_y, "dir_y_y")?;
        validate_finite(dir_y_z, "dir_y_z")?;
        validate_positive(spacing_x, "spacing_x")?;
        validate_positive(spacing_y, "spacing_y")?;
        let solid_id = self.resolve_solid(solid)?;
        let compound = brepkit_operations::pattern::grid_pattern(
            &mut self.topo,
            solid_id,
            Vec3::new(dir_x_x, dir_x_y, dir_x_z),
            Vec3::new(dir_y_x, dir_y_y, dir_y_z),
            spacing_x,
            spacing_y,
            count_x as usize,
            count_y as usize,
        )?;
        Ok(compound_id_to_u32(compound))
    }

    // ── Split ─────────────────────────────────────────────────────

    /// Split a solid into two halves along a plane.
    ///
    /// Returns `[positive_solid_handle, negative_solid_handle]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the plane doesn't intersect the solid.
    #[wasm_bindgen(js_name = "split")]
    #[allow(clippy::too_many_arguments)]
    pub fn split_solid(
        &mut self,
        solid: u32,
        px: f64,
        py: f64,
        pz: f64,
        nx: f64,
        ny: f64,
        nz: f64,
    ) -> Result<Vec<u32>, JsError> {
        validate_finite(px, "px")?;
        validate_finite(py, "py")?;
        validate_finite(pz, "pz")?;
        validate_finite(nx, "nx")?;
        validate_finite(ny, "ny")?;
        validate_finite(nz, "nz")?;
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::split::split(
            &mut self.topo,
            solid_id,
            Point3::new(px, py, pz),
            Vec3::new(nx, ny, nz),
        )?;
        Ok(vec![
            solid_id_to_u32(result.positive),
            solid_id_to_u32(result.negative),
        ])
    }

    // ── Draft ─────────────────────────────────────────────────────

    /// Apply draft angle to faces of a solid.
    ///
    /// `face_handles` is an array of face handles to draft.
    /// Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if angle is zero or faces are invalid.
    #[wasm_bindgen(js_name = "draft")]
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub fn draft_solid(
        &mut self,
        solid: u32,
        face_handles: Vec<u32>,
        pull_x: f64,
        pull_y: f64,
        pull_z: f64,
        neutral_x: f64,
        neutral_y: f64,
        neutral_z: f64,
        angle_degrees: f64,
    ) -> Result<u32, JsError> {
        validate_finite(angle_degrees, "angle_degrees")?;
        let solid_id = self.resolve_solid(solid)?;
        let face_ids: Vec<brepkit_topology::face::FaceId> = face_handles
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let result = brepkit_operations::draft::draft(
            &mut self.topo,
            solid_id,
            &face_ids,
            Vec3::new(pull_x, pull_y, pull_z),
            Point3::new(neutral_x, neutral_y, neutral_z),
            angle_degrees.to_radians(),
        )?;
        Ok(solid_id_to_u32(result))
    }

    // ── Pipe ──────────────────────────────────────────────────────

    /// Pipe sweep: sweep a profile along a NURBS path (no guide).
    ///
    /// Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the face or path is invalid.
    #[wasm_bindgen(js_name = "pipe")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn pipe_solid(
        &mut self,
        face: u32,
        path_degree: u32,
        path_knots: Vec<f64>,
        path_control_points: Vec<f64>,
        path_weights: Vec<f64>,
    ) -> Result<u32, JsError> {
        if path_control_points.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "path_control_points length must be a multiple of 3, got {}",
                    path_control_points.len()
                ),
            }
            .into());
        }

        let face_id = self.resolve_face(face)?;
        let control_points: Vec<Point3> = path_control_points
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();

        let path_curve = NurbsCurve::new(
            path_degree as usize,
            path_knots,
            control_points,
            path_weights,
        )?;

        let solid_id = brepkit_operations::pipe::pipe(&mut self.topo, face_id, &path_curve, None)?;
        Ok(solid_id_to_u32(solid_id))
    }

    // ── Tessellation ───────────────────────────────────────────────

    /// Tessellate a single face into a triangle mesh.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "tessellateFace")]
    pub fn tessellate_face(&self, face: u32, deflection: f64) -> Result<JsMesh, JsError> {
        validate_positive(deflection, "deflection")?;
        let face_id = self.resolve_face(face)?;
        let mesh = tessellate::tessellate(&self.topo, face_id, deflection)?;
        Ok(mesh.into())
    }

    /// Tessellate all faces of a solid into a single merged triangle mesh.
    ///
    /// Includes both the outer shell and any inner shells (voids).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "tessellateSolid")]
    pub fn tessellate_solid(&self, solid: u32, deflection: f64) -> Result<JsMesh, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;

        // Use watertight tessellation that shares edge vertices between
        // adjacent faces, eliminating cracks at face boundaries.
        let merged = tessellate::tessellate_solid(&self.topo, solid_id, deflection)?;

        Ok(merged.into())
    }

    /// Tessellate a solid with per-face triangle grouping.
    ///
    /// Returns a JSON string containing `{ positions, normals, indices, faceOffsets }`.
    /// `faceOffsets` is an array where `faceOffsets[i]` is the start index into
    /// `indices` for face `i`, and the last element is `indices.length`.
    #[wasm_bindgen(js_name = "tessellateSolidGrouped")]
    pub fn tessellate_solid_grouped(
        &self,
        solid: u32,
        deflection: f64,
    ) -> Result<JsValue, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let faces = brepkit_topology::explorer::solid_faces(&self.topo, solid_id)?;

        let mut all_positions: Vec<f64> = Vec::new();
        let mut all_normals: Vec<f64> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut face_offsets: Vec<u32> = Vec::new();

        for &face_id in &faces {
            #[allow(clippy::cast_possible_truncation)]
            let idx_offset = (all_positions.len() / 3) as u32;
            face_offsets.push(all_indices.len() as u32);

            if let Ok(mesh) = tessellate::tessellate(&self.topo, face_id, deflection) {
                for p in &mesh.positions {
                    all_positions.extend_from_slice(&[p.x(), p.y(), p.z()]);
                }
                for n in &mesh.normals {
                    all_normals.extend_from_slice(&[n.x(), n.y(), n.z()]);
                }
                for &idx in &mesh.indices {
                    all_indices.push(idx + idx_offset);
                }
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        face_offsets.push(all_indices.len() as u32);

        Ok(serde_json::to_string(&serde_json::json!({
            "positions": all_positions,
            "normals": all_normals,
            "indices": all_indices,
            "faceOffsets": face_offsets,
        }))
        .map_err(|e| JsError::new(&e.to_string()))?
        .into())
    }

    /// Tessellate a solid and include per-vertex UV coordinates.
    ///
    /// Returns a JSON string containing `{ positions, normals, indices, uvs }`.
    /// `uvs` is a flat array of `[u0, v0, u1, v1, ...]` values, two per vertex.
    /// For analytic and NURBS surfaces, these are the parametric (u, v) values.
    /// For planar faces, UVs are computed by projection onto the face plane.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "tessellateSolidUV")]
    pub fn tessellate_solid_uv(&self, solid: u32, deflection: f64) -> Result<JsValue, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let faces = brepkit_topology::explorer::solid_faces(&self.topo, solid_id)?;

        let mut all_positions: Vec<f64> = Vec::new();
        let mut all_normals: Vec<f64> = Vec::new();
        let mut all_uvs: Vec<f64> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        for &face_id in &faces {
            #[allow(clippy::cast_possible_truncation)]
            let idx_offset = (all_positions.len() / 3) as u32;

            let mesh_uv = tessellate::tessellate_with_uvs(&self.topo, face_id, deflection)?;
            for p in &mesh_uv.mesh.positions {
                all_positions.extend_from_slice(&[p.x(), p.y(), p.z()]);
            }
            for n in &mesh_uv.mesh.normals {
                all_normals.extend_from_slice(&[n.x(), n.y(), n.z()]);
            }
            for uv in &mesh_uv.uvs {
                all_uvs.extend_from_slice(uv);
            }
            for &idx in &mesh_uv.mesh.indices {
                all_indices.push(idx + idx_offset);
            }
        }

        Ok(serde_json::to_string(&serde_json::json!({
            "positions": all_positions,
            "normals": all_normals,
            "indices": all_indices,
            "uvs": all_uvs,
        }))
        .map_err(|e| JsError::new(&e.to_string()))?
        .into())
    }

    // ── Edge wireframe ────────────────────────────────────────────

    /// Sample all edges of a solid into polylines for wireframe rendering.
    ///
    /// Returns a `JsEdgeLines` containing flattened positions and per-edge
    /// offset indices. The `deflection` parameter controls sampling density.
    #[wasm_bindgen(js_name = "meshEdges")]
    pub fn mesh_edges(
        &self,
        solid: u32,
        deflection: f64,
    ) -> Result<crate::shapes::JsEdgeLines, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_lines = tessellate::sample_solid_edges(&self.topo, solid_id, deflection)?;
        Ok(edge_lines.into())
    }

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
                Ok(serde_json::json!({
                    "id": edge_id_to_u32(eid),
                    "curveType": curve_type,
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
                let outer_wire = self.topo.wire(f.outer_wire())?;
                let outer_edges: Vec<u32> = outer_wire
                    .edges()
                    .iter()
                    .map(|e| edge_id_to_u32(e.edge()))
                    .collect();
                let inner_wires: Vec<Vec<u32>> =
                    f.inner_wires()
                        .iter()
                        .filter_map(|&wid| {
                            self.topo.wire(wid).ok().map(|w| {
                                w.edges().iter().map(|e| edge_id_to_u32(e.edge())).collect()
                            })
                        })
                        .collect();
                Ok(serde_json::json!({
                    "id": face_id_to_u32(fid),
                    "surfaceType": surface_type,
                    "reversed": f.is_reversed(),
                    "outerWireEdges": outer_edges,
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

    // ── Measurement ───────────────────────────────────────────────

    /// Compute the axis-aligned bounding box of a solid.
    ///
    /// Returns `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or has no vertices.
    #[wasm_bindgen(js_name = "boundingBox")]
    pub fn bounding_box(&self, solid: u32) -> Result<Vec<f64>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let aabb = measure::solid_bounding_box(&self.topo, solid_id)?;
        Ok(vec![
            aabb.min.x(),
            aabb.min.y(),
            aabb.min.z(),
            aabb.max.x(),
            aabb.max.y(),
            aabb.max.z(),
        ])
    }

    /// Compute the volume of a solid.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "volume")]
    pub fn volume(&self, solid: u32, deflection: f64) -> Result<f64, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        Ok(measure::solid_volume(&self.topo, solid_id, deflection)?)
    }

    /// Compute the total surface area of a solid.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "surfaceArea")]
    pub fn surface_area(&self, solid: u32, deflection: f64) -> Result<f64, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        Ok(measure::solid_surface_area(
            &self.topo, solid_id, deflection,
        )?)
    }

    /// Compute the area of a single face.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "faceArea")]
    pub fn face_area_wasm(&self, face: u32, deflection: f64) -> Result<f64, JsError> {
        validate_positive(deflection, "deflection")?;
        let face_id = self.resolve_face(face)?;
        Ok(measure::face_area(&self.topo, face_id, deflection)?)
    }

    /// Compute the center of mass of a solid (uniform density).
    ///
    /// Returns `[x, y, z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid has zero volume or tessellation fails.
    #[wasm_bindgen(js_name = "centerOfMass")]
    pub fn center_of_mass(&self, solid: u32, deflection: f64) -> Result<Vec<f64>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let com = measure::solid_center_of_mass(&self.topo, solid_id, deflection)?;
        Ok(vec![com.x(), com.y(), com.z()])
    }

    /// Classify a point relative to a solid: inside, outside, or on boundary.
    ///
    /// Returns `"inside"`, `"outside"`, or `"boundary"`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "classifyPoint")]
    pub fn classify_point(
        &self,
        solid: u32,
        x: f64,
        y: f64,
        z: f64,
        tolerance: f64,
    ) -> Result<String, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let point = brepkit_math::vec::Point3::new(x, y, z);
        let result = brepkit_operations::classify::classify_point(
            &self.topo, solid_id, point, 0.1, tolerance,
        )?;
        Ok(match result {
            brepkit_operations::classify::PointClassification::Inside => "inside".into(),
            brepkit_operations::classify::PointClassification::Outside => "outside".into(),
            brepkit_operations::classify::PointClassification::OnBoundary => "boundary".into(),
        })
    }

    /// Compute the length of an edge.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge handle is invalid.
    #[wasm_bindgen(js_name = "edgeLength")]
    pub fn edge_length_wasm(&self, edge: u32) -> Result<f64, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        Ok(measure::edge_length(&self.topo, edge_id)?)
    }

    /// Compute the perimeter of a face.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid.
    #[wasm_bindgen(js_name = "facePerimeter")]
    pub fn face_perimeter_wasm(&self, face: u32) -> Result<f64, JsError> {
        let face_id = self.resolve_face(face)?;
        Ok(measure::face_perimeter(&self.topo, face_id)?)
    }

    /// Validate a solid, returning the number of errors found.
    ///
    /// Returns 0 if the solid is valid.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "validateSolid")]
    pub fn validate_solid_wasm(&self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let report = brepkit_operations::validate::validate_solid(&self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(report.error_count() as u32)
    }

    // ── Distance ──────────────────────────────────────────────────

    /// Compute minimum distance from a point to a solid.
    ///
    /// Returns `[distance, closest_x, closest_y, closest_z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "pointToSolidDistance")]
    pub fn point_to_solid_distance_wasm(
        &self,
        px: f64,
        py: f64,
        pz: f64,
        solid: u32,
    ) -> Result<Vec<f64>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::distance::point_to_solid_distance(
            &self.topo,
            Point3::new(px, py, pz),
            solid_id,
        )?;
        Ok(vec![
            result.distance,
            result.point_b.x(),
            result.point_b.y(),
            result.point_b.z(),
        ])
    }

    /// Compute minimum distance between two solids.
    ///
    /// Returns `[distance]`.
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid.
    #[wasm_bindgen(js_name = "solidToSolidDistance")]
    pub fn solid_to_solid_distance_wasm(&self, a: u32, b: u32) -> Result<f64, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let result = brepkit_operations::distance::solid_to_solid_distance(&self.topo, a_id, b_id)?;
        Ok(result.distance)
    }

    /// Compute minimum distance from a point to a face.
    ///
    /// Returns `[distance, closest_x, closest_y, closest_z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid.
    #[wasm_bindgen(js_name = "pointToFaceDistance")]
    pub fn point_to_face_distance_wasm(
        &self,
        px: f64,
        py: f64,
        pz: f64,
        face: u32,
    ) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let result = brepkit_operations::distance::point_to_face(
            &self.topo,
            Point3::new(px, py, pz),
            face_id,
        )?;
        Ok(vec![
            result.distance,
            result.point_b.x(),
            result.point_b.y(),
            result.point_b.z(),
        ])
    }

    /// Compute minimum distance from a point to an edge.
    ///
    /// Returns `[distance, closest_x, closest_y, closest_z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge handle is invalid.
    #[wasm_bindgen(js_name = "pointToEdgeDistance")]
    pub fn point_to_edge_distance_wasm(
        &self,
        px: f64,
        py: f64,
        pz: f64,
        edge: u32,
    ) -> Result<Vec<f64>, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let result = brepkit_operations::distance::point_to_edge(
            &self.topo,
            Point3::new(px, py, pz),
            edge_id,
        )?;
        Ok(vec![
            result.distance,
            result.point_b.x(),
            result.point_b.y(),
            result.point_b.z(),
        ])
    }

    // ── Sewing ────────────────────────────────────────────────────

    /// Sew loose faces into a connected solid.
    ///
    /// `face_handles` is an array of face handles. Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 faces or sewing fails.
    #[wasm_bindgen(js_name = "sewFaces")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn sew_faces_wasm(
        &mut self,
        face_handles: Vec<u32>,
        tolerance: f64,
    ) -> Result<u32, JsError> {
        let face_ids: Vec<brepkit_topology::face::FaceId> = face_handles
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let solid = brepkit_operations::sew::sew_faces(&mut self.topo, &face_ids, tolerance)?;
        Ok(solid_id_to_u32(solid))
    }

    /// Create a solid from a set of faces by sewing them together.
    ///
    /// Alias for `sewFaces` with a default tolerance. This is the equivalent
    /// of OCCT's `BRepBuilderAPI_MakeSolid`.
    #[wasm_bindgen(js_name = "makeSolid")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_solid_from_faces(&mut self, face_handles: Vec<u32>) -> Result<u32, JsError> {
        let face_ids: Vec<brepkit_topology::face::FaceId> = face_handles
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let tolerance = brepkit_math::tolerance::Tolerance::new().linear;
        let solid = brepkit_operations::sew::sew_faces(&mut self.topo, &face_ids, tolerance)?;
        Ok(solid_id_to_u32(solid))
    }

    /// Remove all holes from a face, returning a new face with only the outer wire.
    #[wasm_bindgen(js_name = "removeHolesFromFace")]
    pub fn remove_holes_from_face(&mut self, face: u32) -> Result<u32, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let outer_wire = face_data.outer_wire();
        let surface = face_data.surface().clone();
        let new_face = Face::new(outer_wire, vec![], surface);
        let fid = self.topo.faces.alloc(new_face);
        Ok(face_id_to_u32(fid))
    }

    /// Weld shells and faces into a single solid by sewing.
    ///
    /// Accepts an array of face handles from potentially different shells.
    /// Sews all faces together into a single solid.
    #[wasm_bindgen(js_name = "weldShellsAndFaces")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn weld_shells_and_faces(
        &mut self,
        face_handles: Vec<u32>,
        tolerance: f64,
    ) -> Result<u32, JsError> {
        let face_ids: Vec<brepkit_topology::face::FaceId> = face_handles
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let tol = if tolerance > 0.0 {
            tolerance
        } else {
            brepkit_math::tolerance::Tolerance::new().linear
        };
        let solid = brepkit_operations::sew::sew_faces(&mut self.topo, &face_ids, tol)?;
        Ok(solid_id_to_u32(solid))
    }

    // ── Shape construction (low-level) ────────────────────────────

    /// Create a vertex at the given position.
    ///
    /// Returns a vertex handle (`u32`).
    #[wasm_bindgen(js_name = "makeVertex")]
    pub fn make_vertex_wasm(&mut self, x: f64, y: f64, z: f64) -> Result<u32, JsError> {
        validate_finite(x, "x")?;
        validate_finite(y, "y")?;
        validate_finite(z, "z")?;
        let id = self
            .topo
            .vertices
            .alloc(Vertex::new(Point3::new(x, y, z), TOL));
        Ok(vertex_id_to_u32(id))
    }

    /// Create a straight-line edge between two points.
    ///
    /// Returns an edge handle (`u32`).
    #[wasm_bindgen(js_name = "makeLineEdge")]
    pub fn make_line_edge_wasm(
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
        let eid = brepkit_topology::builder::make_line_edge(&mut self.topo, start, end)?;
        Ok(edge_id_to_u32(eid))
    }

    /// Create a NURBS curve edge.
    ///
    /// Returns an edge handle (`u32`).
    #[wasm_bindgen(js_name = "makeNurbsEdge")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_nurbs_edge_wasm(
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
        let v_start = self.topo.vertices.alloc(Vertex::new(start_pt, TOL));
        // When start ≈ end (closed curve), reuse the same vertex so
        // downstream code correctly identifies the edge as closed.
        let v_end = if (start_pt - end_pt).length() < TOL * 100.0 {
            v_start
        } else {
            self.topo.vertices.alloc(Vertex::new(end_pt, TOL))
        };
        let eid = self
            .topo
            .edges
            .alloc(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(curve)));
        Ok(edge_id_to_u32(eid))
    }

    /// Create a closed wire from an ordered array of edge handles.
    ///
    /// Returns a wire handle (`u32`).
    #[wasm_bindgen(js_name = "makeWire")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_wire_wasm(&mut self, edge_handles: Vec<u32>, closed: bool) -> Result<u32, JsError> {
        let oriented: Vec<OrientedEdge> = edge_handles
            .iter()
            .map(|&h| {
                let eid = self.resolve_edge(h)?;
                Ok(OrientedEdge::new(eid, true))
            })
            .collect::<Result<_, WasmError>>()?;
        let wire = Wire::new(oriented, closed)?;
        let wid = self.topo.wires.alloc(wire);
        Ok(wire_id_to_u32(wid))
    }

    /// Create a planar face from a wire (computes normal from first 3 vertices).
    ///
    /// Returns a face handle (`u32`).
    #[wasm_bindgen(js_name = "makeFaceFromWire")]
    pub fn make_face_from_wire_wasm(&mut self, wire: u32) -> Result<u32, JsError> {
        let wid = self.resolve_wire(wire)?;
        let fid = brepkit_topology::builder::make_face_from_wire(&mut self.topo, wid)?;
        Ok(face_id_to_u32(fid))
    }

    /// Create a solid from a shell.
    ///
    /// Returns a solid handle (`u32`).
    #[wasm_bindgen(js_name = "solidFromShell")]
    pub fn solid_from_shell_wasm(&mut self, shell: u32) -> Result<u32, JsError> {
        let shell_id = self.resolve_shell(shell)?;
        let solid = brepkit_topology::solid::Solid::new(shell_id, vec![]);
        let sid = self.topo.solids.alloc(solid);
        Ok(solid_id_to_u32(sid))
    }

    /// Create a compound from multiple solid handles.
    ///
    /// Returns a compound handle (stored as `u32`).
    #[wasm_bindgen(js_name = "makeCompound")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_compound_wasm(&mut self, solid_handles: Vec<u32>) -> Result<u32, JsError> {
        let solid_ids: Vec<brepkit_topology::solid::SolidId> = solid_handles
            .iter()
            .map(|&h| self.resolve_solid(h))
            .collect::<Result<_, _>>()?;
        let compound = brepkit_topology::compound::Compound::new(solid_ids);
        #[allow(clippy::cast_possible_truncation)]
        let cid = self.topo.compounds.alloc(compound);
        Ok(cid.index() as u32)
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

    /// Heal a solid topology.
    ///
    /// Returns the number of issues fixed.
    #[wasm_bindgen(js_name = "healSolid")]
    pub fn heal_solid_wasm(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let report = brepkit_operations::heal::heal_solid(&mut self.topo, solid_id, TOL)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok((report.vertices_merged
            + report.degenerate_edges_removed
            + report.orientations_fixed
            + report.wire_gaps_closed
            + report.small_faces_removed
            + report.duplicate_faces_removed) as u32)
    }

    /// Validate, heal, and re-validate a solid in one pass.
    ///
    /// Returns the number of remaining validation errors after repair.
    /// A return value of 0 means the solid is valid after repair.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "repairSolid")]
    pub fn repair_solid_wasm(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let report = brepkit_operations::heal::repair_solid(&mut self.topo, solid_id, TOL)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(report.after.error_count() as u32)
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
                    let du = derivs[1][0]; // ∂S/∂u
                    let dv = derivs[0][1]; // ∂S/∂v
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
        let fid = self.topo.faces.alloc(new_face);
        Ok(face_id_to_u32(fid))
    }

    /// Sweep a face along a path defined by a chain of edges.
    ///
    /// Collects points from the edges, fits an interpolating NURBS curve,
    /// then sweeps the profile along that curve.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 edges or the fit fails.
    #[wasm_bindgen(js_name = "sweepAlongEdges")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn sweep_along_edges(&mut self, face: u32, edge_handles: Vec<u32>) -> Result<u32, JsError> {
        if edge_handles.is_empty() {
            return Err(WasmError::InvalidInput {
                reason: "sweepAlongEdges requires at least one edge".into(),
            }
            .into());
        }

        // Collect ordered points from the edge chain.
        let mut points = Vec::new();
        for &eh in &edge_handles {
            let eid = self.resolve_edge(eh)?;
            let edge_data = self.topo.edge(eid)?;
            let start = self.topo.vertex(edge_data.start())?.point();

            // Only push start if it's not a duplicate of the last point.
            if points
                .last()
                .is_none_or(|p: &Point3| (*p - start).length() > TOL)
            {
                points.push(start);
            }

            // For non-line edges, sample interior points for better fidelity.
            match edge_data.curve() {
                EdgeCurve::NurbsCurve(curve) => {
                    let (u0, u1) = curve.domain();
                    let n_samples = 4;
                    for i in 1..n_samples {
                        #[allow(clippy::cast_precision_loss)]
                        let frac = i as f64 / n_samples as f64;
                        let u = u0 + frac * (u1 - u0);
                        points.push(curve.evaluate(u));
                    }
                }
                EdgeCurve::Circle(circle) => {
                    let n_samples = 8;
                    for i in 1..n_samples {
                        #[allow(clippy::cast_precision_loss)]
                        let t = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
                        points.push(circle.evaluate(t));
                    }
                }
                EdgeCurve::Ellipse(ellipse) => {
                    let n_samples = 8;
                    for i in 1..n_samples {
                        #[allow(clippy::cast_precision_loss)]
                        let t = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
                        points.push(ellipse.evaluate(t));
                    }
                }
                EdgeCurve::Line => {}
            }

            let end = self.topo.vertex(edge_data.end())?.point();
            points.push(end);
        }

        if points.len() < 2 {
            return Err(WasmError::InvalidInput {
                reason: "sweepAlongEdges: need at least 2 distinct points".into(),
            }
            .into());
        }

        // Fit an interpolating NURBS curve through the points.
        let degree = std::cmp::min(3, points.len() - 1);
        let path_curve = brepkit_math::nurbs::fitting::interpolate(&points, degree)?;

        let face_id = self.resolve_face(face)?;
        let solid_id = sweep(&mut self.topo, face_id, &path_curve)?;
        Ok(solid_id_to_u32(solid_id))
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
    pub fn convex_hull_wasm(&mut self, coords: Vec<f64>) -> Result<u32, JsError> {
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
            .map(|p| self.topo.vertices.alloc(Vertex::new(*p, TOL)))
            .collect();

        let mut face_ids = Vec::new();
        for &[a, b, c] in &hull.faces {
            let va = vertex_ids[a];
            let vb = vertex_ids[b];
            let vc = vertex_ids[c];

            let e0 = self.topo.edges.alloc(Edge::new(va, vb, EdgeCurve::Line));
            let e1 = self.topo.edges.alloc(Edge::new(vb, vc, EdgeCurve::Line));
            let e2 = self.topo.edges.alloc(Edge::new(vc, va, EdgeCurve::Line));

            let oriented = vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
            ];
            let wire = Wire::new(oriented, true)?;
            let wid = self.topo.wires.alloc(wire);

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

            let fid =
                self.topo
                    .faces
                    .alloc(Face::new(wid, vec![], FaceSurface::Plane { normal, d }));
            face_ids.push(fid);
        }

        let shell = brepkit_topology::shell::Shell::new(face_ids)?;
        let shell_id = self.topo.shells.alloc(shell);
        let solid = brepkit_topology::solid::Solid::new(shell_id, vec![]);
        let solid_id = self.topo.solids.alloc(solid);

        Ok(solid_id_to_u32(solid_id))
    }

    /// Interpolate a NURBS curve through points and create an edge.
    ///
    /// Uses chord-length parameterization with the given degree.
    /// Returns an edge handle (`u32`).
    #[wasm_bindgen(js_name = "interpolatePoints")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn interpolate_points_wasm(
        &mut self,
        coords: Vec<f64>,
        degree: u32,
    ) -> Result<u32, JsError> {
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
        if points.len() < 2 {
            return Err(WasmError::InvalidInput {
                reason: format!("need at least 2 points, got {}", points.len()),
            }
            .into());
        }

        let deg = std::cmp::min(degree as usize, points.len() - 1);
        let curve = brepkit_math::nurbs::fitting::interpolate(&points, deg)?;

        let start = points[0];
        let end = points[points.len() - 1];
        let v_start = self.topo.vertices.alloc(Vertex::new(start, TOL));
        let v_end = self.topo.vertices.alloc(Vertex::new(end, TOL));
        let eid = self
            .topo
            .edges
            .alloc(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(curve)));
        Ok(edge_id_to_u32(eid))
    }

    /// Offset (shell) a solid by a distance.
    ///
    /// Returns a new solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the distance is zero or the solid is invalid.
    #[wasm_bindgen(js_name = "offsetSolid")]
    pub fn offset_solid_wasm(&mut self, solid: u32, distance: f64) -> Result<u32, JsError> {
        validate_finite(distance, "distance")?;
        let solid_id = self.resolve_solid(solid)?;
        let result =
            brepkit_operations::offset_solid::offset_solid(&mut self.topo, solid_id, distance)?;
        Ok(solid_id_to_u32(result))
    }

    /// Thicken a face into a solid by offsetting it by the given distance.
    ///
    /// Creates a solid from a face by extruding it along its normal by
    /// `thickness`. Positive values offset outward, negative inward.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid or thickness is zero.
    #[wasm_bindgen(js_name = "thicken")]
    pub fn thicken_face(&mut self, face: u32, thickness: f64) -> Result<u32, JsError> {
        validate_finite(thickness, "thickness")?;
        let face_id = self.resolve_face(face)?;
        let result = brepkit_operations::thicken::thicken(&mut self.topo, face_id, thickness)?;
        Ok(solid_id_to_u32(result))
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

// ── Private helpers ────────────────────────────────────────────────

impl BrepKernel {
    /// Build a closed planar face from an ordered sequence of points.
    fn make_planar_face(
        &mut self,
        points: &[Point3],
    ) -> Result<brepkit_topology::face::FaceId, WasmError> {
        let n = points.len();

        // Allocate vertices.
        let verts: Vec<_> = points
            .iter()
            .map(|p| self.topo.vertices.alloc(Vertex::new(*p, TOL)))
            .collect();

        // Allocate edges connecting consecutive vertices.
        let edges: Vec<_> = (0..n)
            .map(|i| {
                let next = (i + 1) % n;
                self.topo
                    .edges
                    .alloc(Edge::new(verts[i], verts[next], EdgeCurve::Line))
            })
            .collect();

        // Build oriented edge list and closed wire.
        let oriented: Vec<_> = edges
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let wire = Wire::new(oriented, true)?;
        let wid = self.topo.wires.alloc(wire);

        // Compute the face normal from the first three points.
        let a = points[1] - points[0];
        let b = points[2] - points[0];
        let normal = a.cross(b).normalize()?;

        // Plane equation: n · p = d
        let d = normal.x().mul_add(
            points[0].x(),
            normal
                .y()
                .mul_add(points[0].y(), normal.z() * points[0].z()),
        );

        let face_id =
            self.topo
                .faces
                .alloc(Face::new(wid, vec![], FaceSurface::Plane { normal, d }));

        Ok(face_id)
    }

    /// Resolve a `u32` face handle to a typed `FaceId`.
    /// Compute the v-range for an analytic surface by projecting face wire
    /// vertices onto the surface axis.
    fn compute_axial_v_range(
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

    fn resolve_face(&self, handle: u32) -> Result<brepkit_topology::face::FaceId, WasmError> {
        let index = handle as usize;
        self.topo
            .faces
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "face",
                index,
            })
    }

    /// Resolve a `u32` vertex handle to a typed `VertexId`.
    fn resolve_vertex(&self, handle: u32) -> Result<brepkit_topology::vertex::VertexId, WasmError> {
        let index = handle as usize;
        self.topo
            .vertices
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "vertex",
                index,
            })
    }

    /// Resolve a `u32` edge handle to a typed `EdgeId`.
    fn resolve_edge(&self, handle: u32) -> Result<brepkit_topology::edge::EdgeId, WasmError> {
        let index = handle as usize;
        self.topo
            .edges
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "edge",
                index,
            })
    }

    /// Resolve a `u32` solid handle to a typed `SolidId`.
    fn resolve_solid(&self, handle: u32) -> Result<brepkit_topology::solid::SolidId, WasmError> {
        let index = handle as usize;
        self.topo
            .solids
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "solid",
                index,
            })
    }

    /// Resolve a `u32` wire handle to a typed `WireId`.
    fn resolve_wire(&self, handle: u32) -> Result<brepkit_topology::wire::WireId, WasmError> {
        let index = handle as usize;
        self.topo
            .wires
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "wire",
                index,
            })
    }

    /// Resolve a `u32` shell handle to a typed `ShellId`.
    fn resolve_shell(&self, handle: u32) -> Result<brepkit_topology::shell::ShellId, WasmError> {
        let index = handle as usize;
        self.topo
            .shells
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "shell",
                index,
            })
    }

    /// Resolve a `u32` compound handle to a typed `CompoundId`.
    fn resolve_compound(
        &self,
        handle: u32,
    ) -> Result<brepkit_topology::compound::CompoundId, WasmError> {
        let index = handle as usize;
        self.topo
            .compounds
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "compound",
                index,
            })
    }
}

// ── New WASM bindings (Batches 1–6) ─────────────────────────────────

#[wasm_bindgen]
impl BrepKernel {
    // ── Batch 1: Math WASM bindings ────────────────────────────────

    /// Approximate a curve through points (least-squares).
    ///
    /// Returns an edge handle.
    #[wasm_bindgen(js_name = "approximateCurve")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn approximate_curve(
        &mut self,
        coords: Vec<f64>,
        degree: u32,
        num_control_points: u32,
    ) -> Result<u32, JsError> {
        let points = parse_points(&coords)?;
        if points.len() < 2 {
            return Err(WasmError::InvalidInput {
                reason: format!("need at least 2 points, got {}", points.len()),
            }
            .into());
        }
        let deg = std::cmp::min(degree as usize, points.len() - 1);
        let curve =
            brepkit_math::nurbs::fitting::approximate(&points, deg, num_control_points as usize)?;
        Ok(edge_id_to_u32(self.nurbs_curve_to_edge(&points, curve)))
    }

    /// Approximate a curve through points using LSPIA (progressive iteration).
    ///
    /// Returns an edge handle.
    #[wasm_bindgen(js_name = "approximateCurveLspia")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn approximate_curve_lspia(
        &mut self,
        coords: Vec<f64>,
        degree: u32,
        num_control_points: u32,
        tolerance: f64,
        max_iterations: u32,
    ) -> Result<u32, JsError> {
        let points = parse_points(&coords)?;
        if points.len() < 2 {
            return Err(WasmError::InvalidInput {
                reason: format!("need at least 2 points, got {}", points.len()),
            }
            .into());
        }
        let deg = std::cmp::min(degree as usize, points.len() - 1);
        let curve = brepkit_math::nurbs::fitting::approximate_lspia(
            &points,
            deg,
            num_control_points as usize,
            tolerance,
            max_iterations as usize,
        )?;
        Ok(edge_id_to_u32(self.nurbs_curve_to_edge(&points, curve)))
    }

    /// Interpolate a grid of points into a NURBS surface.
    ///
    /// `coords` is a flat array `[x,y,z, ...]` of `rows * cols` points.
    /// Returns a face handle.
    #[wasm_bindgen(js_name = "interpolateSurface")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn interpolate_surface(
        &mut self,
        coords: Vec<f64>,
        rows: u32,
        cols: u32,
        degree_u: u32,
        degree_v: u32,
    ) -> Result<u32, JsError> {
        let grid = parse_point_grid(&coords, rows as usize, cols as usize)?;
        let surface = brepkit_math::nurbs::surface_fitting::interpolate_surface(
            &grid,
            degree_u as usize,
            degree_v as usize,
        )?;
        Ok(face_id_to_u32(self.nurbs_surface_to_face(surface)?))
    }

    /// Approximate a grid of points into a NURBS surface using LSPIA.
    ///
    /// Returns a face handle.
    #[wasm_bindgen(js_name = "approximateSurfaceLspia")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn approximate_surface_lspia(
        &mut self,
        coords: Vec<f64>,
        rows: u32,
        cols: u32,
        degree_u: u32,
        degree_v: u32,
        num_cps_u: u32,
        num_cps_v: u32,
        tolerance: f64,
        max_iterations: u32,
    ) -> Result<u32, JsError> {
        let grid = parse_point_grid(&coords, rows as usize, cols as usize)?;
        let surface = brepkit_math::nurbs::surface_fitting::approximate_surface_lspia(
            &grid,
            degree_u as usize,
            degree_v as usize,
            num_cps_u as usize,
            num_cps_v as usize,
            tolerance,
            max_iterations as usize,
        )?;
        Ok(face_id_to_u32(self.nurbs_surface_to_face(surface)?))
    }

    /// Insert a knot into an edge's NURBS curve.
    ///
    /// Returns a new edge handle with the refined curve.
    #[wasm_bindgen(js_name = "curveKnotInsert")]
    pub fn curve_knot_insert(&mut self, edge: u32, knot: f64, times: u32) -> Result<u32, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        let refined =
            brepkit_math::nurbs::knot_ops::curve_knot_insert(&curve, knot, times as usize)?;
        Ok(edge_id_to_u32(
            self.nurbs_curve_to_edge_from_curve(&refined),
        ))
    }

    /// Remove a knot from an edge's NURBS curve.
    ///
    /// Returns a new edge handle with the simplified curve.
    #[wasm_bindgen(js_name = "curveKnotRemove")]
    pub fn curve_knot_remove(
        &mut self,
        edge: u32,
        knot: f64,
        tolerance: f64,
    ) -> Result<u32, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        let simplified = brepkit_math::nurbs::knot_ops::curve_knot_remove(&curve, knot, tolerance)?;
        Ok(edge_id_to_u32(
            self.nurbs_curve_to_edge_from_curve(&simplified),
        ))
    }

    /// Split an edge's NURBS curve at a parameter value.
    ///
    /// Returns two edge handles as `[u32; 2]`.
    #[wasm_bindgen(js_name = "curveSplit")]
    pub fn curve_split(&mut self, edge: u32, u: f64) -> Result<Vec<u32>, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        let (left, right) = brepkit_math::nurbs::knot_ops::curve_split(&curve, u)?;
        let e1 = self.nurbs_curve_to_edge_from_curve(&left);
        let e2 = self.nurbs_curve_to_edge_from_curve(&right);
        Ok(vec![edge_id_to_u32(e1), edge_id_to_u32(e2)])
    }

    /// Elevate the degree of an edge's NURBS curve.
    ///
    /// Returns a new edge handle.
    #[wasm_bindgen(js_name = "curveDegreeElevate")]
    pub fn curve_degree_elevate(&mut self, edge: u32, elevate_by: u32) -> Result<u32, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        let elevated =
            brepkit_math::nurbs::decompose::curve_degree_elevate(&curve, elevate_by as usize)?;
        Ok(edge_id_to_u32(
            self.nurbs_curve_to_edge_from_curve(&elevated),
        ))
    }

    // ── Batch 2: Topology query bindings ─────────────────────────────

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
            .map(|p| self.topo.vertices.alloc(Vertex::new(*p, TOL)))
            .collect();
        let edges: Vec<_> = (0..n)
            .map(|i| {
                self.topo
                    .edges
                    .alloc(Edge::new(verts[i], verts[(i + 1) % n], EdgeCurve::Line))
            })
            .collect();
        let oriented: Vec<_> = edges
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let wire = Wire::new(oriented, true)?;
        let wid = self.topo.wires.alloc(wire);
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
        )?;
        Ok(wire_id_to_u32(wid))
    }

    /// Create a circular face on the XY plane (using NURBS arcs).
    ///
    /// Returns a face handle.
    #[wasm_bindgen(js_name = "makeCircleFace")]
    pub fn make_circle_face_wasm(&mut self, radius: f64, segments: u32) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        if segments < 3 {
            return Err(WasmError::InvalidInput {
                reason: format!("circle face needs at least 3 segments, got {segments}"),
            }
            .into());
        }
        let fid =
            brepkit_topology::builder::make_circle_face(&mut self.topo, radius, segments as usize)?;
        Ok(face_id_to_u32(fid))
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

    // ── Batch 3: Simple operations bindings ──────────────────────────

    /// Create a circular pattern of a solid around an axis.
    ///
    /// Returns a compound handle.
    #[wasm_bindgen(js_name = "circularPattern")]
    pub fn circular_pattern(
        &mut self,
        solid: u32,
        ax: f64,
        ay: f64,
        az: f64,
        count: u32,
    ) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let axis = Vec3::new(ax, ay, az);
        let compound = brepkit_operations::pattern::circular_pattern(
            &mut self.topo,
            solid_id,
            axis,
            count as usize,
        )?;
        Ok(compound_id_to_u32(compound))
    }

    /// Merge coincident vertices in a solid.
    ///
    /// Returns the number of vertices merged.
    #[wasm_bindgen(js_name = "mergeCoincidentVertices")]
    pub fn merge_coincident_vertices(
        &mut self,
        solid: u32,
        tolerance: f64,
    ) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let count = brepkit_operations::heal::merge_coincident_vertices(
            &mut self.topo,
            solid_id,
            tolerance,
        )?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(count as u32)
    }

    /// Remove degenerate (zero-length) edges from a solid.
    ///
    /// Returns the number of edges removed.
    #[wasm_bindgen(js_name = "removeDegenerateEdges")]
    pub fn remove_degenerate_edges(&mut self, solid: u32, tolerance: f64) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let count =
            brepkit_operations::heal::remove_degenerate_edges(&mut self.topo, solid_id, tolerance)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(count as u32)
    }

    /// Fix face orientations to ensure consistent outward normals.
    ///
    /// Returns the number of faces fixed.
    #[wasm_bindgen(js_name = "fixFaceOrientations")]
    pub fn fix_face_orientations(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let count = brepkit_operations::heal::fix_face_orientations(&mut self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(count as u32)
    }

    /// Remove specified faces from a solid (defeaturing).
    ///
    /// `face_handles` is an array of face handles to remove.
    /// Returns a new solid handle.
    #[wasm_bindgen(js_name = "defeature")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn defeature(&mut self, solid: u32, face_handles: Vec<u32>) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let face_ids: Vec<_> = face_handles
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<Vec<_>, _>>()?;
        let result = brepkit_operations::defeature::defeature(&mut self.topo, solid_id, &face_ids)?;
        Ok(solid_id_to_u32(result))
    }

    /// Detect small features (faces below an area threshold).
    ///
    /// Returns an array of face handles.
    #[wasm_bindgen(js_name = "detectSmallFeatures")]
    pub fn detect_small_features(
        &self,
        solid: u32,
        area_threshold: f64,
        deflection: f64,
    ) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let faces = brepkit_operations::defeature::detect_small_features(
            &self.topo,
            solid_id,
            area_threshold,
            deflection,
        )?;
        Ok(faces.iter().map(|f| face_id_to_u32(*f)).collect())
    }

    /// Recognize geometric features in a solid.
    ///
    /// Returns a JSON string describing the recognized features.
    #[wasm_bindgen(js_name = "recognizeFeatures")]
    pub fn recognize_features(&self, solid: u32, deflection: f64) -> Result<String, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let features = brepkit_operations::feature_recognition::recognize_features(
            &self.topo, solid_id, deflection,
        )?;
        let json_features: Vec<serde_json::Value> =
            features.iter().map(|f| serialize_feature(f)).collect();
        Ok(serde_json::Value::Array(json_features).to_string())
    }

    // ── Batch 4: Complex operations bindings ─────────────────────────

    /// Apply variable-radius fillets to edges.
    ///
    /// `json` is a JSON string: `[{"edge": u32, "law": "constant"|"linear"|"scurve", "start": f64, "end": f64}]`
    /// Returns a new solid handle.
    #[wasm_bindgen(js_name = "filletVariable")]
    pub fn fillet_variable(&mut self, solid: u32, json: &str) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let specs: Vec<serde_json::Value> =
            serde_json::from_str(json).map_err(|e| WasmError::InvalidInput {
                reason: format!("invalid JSON: {e}"),
            })?;
        let mut edge_laws = Vec::with_capacity(specs.len());
        for spec in &specs {
            let edge_handle = spec["edge"]
                .as_u64()
                .ok_or_else(|| WasmError::InvalidInput {
                    reason: "missing 'edge' in fillet spec".into(),
                })? as u32;
            let edge_id = self.resolve_edge(edge_handle)?;
            let law_str = spec["law"].as_str().unwrap_or("constant");
            let law = match law_str {
                "linear" => {
                    let s = spec["start"].as_f64().unwrap_or(1.0);
                    let e = spec["end"].as_f64().unwrap_or(1.0);
                    brepkit_operations::fillet::FilletRadiusLaw::Linear { start: s, end: e }
                }
                "scurve" => {
                    let s = spec["start"].as_f64().unwrap_or(1.0);
                    let e = spec["end"].as_f64().unwrap_or(1.0);
                    brepkit_operations::fillet::FilletRadiusLaw::SCurve { start: s, end: e }
                }
                _ => {
                    let r = spec["radius"]
                        .as_f64()
                        .or_else(|| spec["start"].as_f64())
                        .unwrap_or(1.0);
                    brepkit_operations::fillet::FilletRadiusLaw::Constant(r)
                }
            };
            edge_laws.push((edge_id, law));
        }
        let result =
            brepkit_operations::fillet::fillet_variable(&mut self.topo, solid_id, &edge_laws)?;
        Ok(solid_id_to_u32(result))
    }

    /// Sweep a face along a NURBS path with advanced options.
    ///
    /// `contact_mode`: "rmf" (default), "fixed", or "constantNormal:x,y,z"
    /// `scale_values`: flat `[t0,s0,t1,s1,...]` pairs for piecewise-linear scale law.
    /// Returns a solid handle.
    #[wasm_bindgen(js_name = "sweepWithOptions")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn sweep_with_options(
        &mut self,
        profile: u32,
        path_edge: u32,
        contact_mode: &str,
        scale_values: Vec<f64>,
        segments: u32,
    ) -> Result<u32, JsError> {
        use brepkit_operations::sweep::{SweepContactMode, SweepOptions};

        let face_id = self.resolve_face(profile)?;
        let path_curve = self.extract_nurbs_curve(path_edge)?;

        let mode = if contact_mode == "fixed" {
            SweepContactMode::Fixed
        } else if let Some(rest) = contact_mode.strip_prefix("constantNormal:") {
            let parts: Vec<f64> = rest
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if parts.len() >= 3 {
                SweepContactMode::ConstantNormal(Vec3::new(parts[0], parts[1], parts[2]))
            } else {
                SweepContactMode::RotationMinimizing
            }
        } else {
            SweepContactMode::RotationMinimizing
        };

        let scale_law: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>> =
            if scale_values.len() >= 4 && scale_values.len() % 2 == 0 {
                let pairs: Vec<(f64, f64)> =
                    scale_values.chunks_exact(2).map(|c| (c[0], c[1])).collect();
                Some(Box::new(move |t: f64| -> f64 {
                    // Piecewise-linear interpolation
                    if pairs.is_empty() {
                        return 1.0;
                    }
                    if t <= pairs[0].0 {
                        return pairs[0].1;
                    }
                    if t >= pairs[pairs.len() - 1].0 {
                        return pairs[pairs.len() - 1].1;
                    }
                    for w in pairs.windows(2) {
                        if t >= w[0].0 && t <= w[1].0 {
                            let frac = (t - w[0].0) / (w[1].0 - w[0].0);
                            return w[0].1 + frac * (w[1].1 - w[0].1);
                        }
                    }
                    1.0
                }))
            } else {
                None
            };

        let options = SweepOptions {
            contact_mode: mode,
            scale_law,
            segments: segments as usize,
        };

        let result = brepkit_operations::sweep::sweep_with_options(
            &mut self.topo,
            face_id,
            &path_curve,
            &options,
        )?;
        Ok(solid_id_to_u32(result))
    }

    /// Classify a point relative to a solid using generalized winding numbers.
    ///
    /// Returns "inside", "outside", or "boundary".
    #[wasm_bindgen(js_name = "classifyPointWinding")]
    pub fn classify_point_winding(
        &self,
        solid: u32,
        x: f64,
        y: f64,
        z: f64,
        tolerance: f64,
    ) -> Result<String, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let point = Point3::new(x, y, z);
        let result = brepkit_operations::classify::classify_point_winding(
            &self.topo, solid_id, point, 0.1, tolerance,
        )?;
        Ok(classify_to_string(result))
    }

    /// Classify a point using robust dual-method (winding + ray casting).
    ///
    /// Returns "inside", "outside", or "boundary".
    #[wasm_bindgen(js_name = "classifyPointRobust")]
    pub fn classify_point_robust(
        &self,
        solid: u32,
        x: f64,
        y: f64,
        z: f64,
        tolerance: f64,
    ) -> Result<String, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let point = Point3::new(x, y, z);
        let result = brepkit_operations::classify::classify_point_robust(
            &self.topo, solid_id, point, 0.1, tolerance,
        )?;
        Ok(classify_to_string(result))
    }

    /// Perform a mesh boolean on raw triangle data.
    ///
    /// Returns a `JsMesh` with the result.
    #[wasm_bindgen(js_name = "meshBoolean")]
    #[allow(
        clippy::needless_pass_by_value,
        clippy::too_many_arguments,
        clippy::unused_self
    )]
    pub fn mesh_boolean(
        &self,
        positions_a: Vec<f64>,
        indices_a: Vec<u32>,
        positions_b: Vec<f64>,
        indices_b: Vec<u32>,
        op: &str,
        tolerance: f64,
    ) -> Result<JsMesh, JsError> {
        let mesh_a = build_triangle_mesh(&positions_a, &indices_a)?;
        let mesh_b = build_triangle_mesh(&positions_b, &indices_b)?;
        let bool_op = parse_boolean_op(op)?;
        let result =
            brepkit_operations::mesh_boolean::mesh_boolean(&mesh_a, &mesh_b, bool_op, tolerance)?;
        Ok(triangle_mesh_to_js(&result.mesh))
    }

    /// Fill a 4-sided boundary with a Coons patch surface.
    ///
    /// `boundary_coords` is flat `[x,y,z, ...]` for all 4 curves concatenated.
    /// `curve_lengths` is `[n0, n1, n2, n3]` — number of points per curve.
    /// Returns a face handle.
    #[wasm_bindgen(js_name = "fillCoonsPatch")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn fill_coons_patch(
        &mut self,
        boundary_coords: Vec<f64>,
        curve_lengths: Vec<u32>,
    ) -> Result<u32, JsError> {
        if curve_lengths.len() != 4 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "Coons patch requires exactly 4 boundary curves, got {}",
                    curve_lengths.len()
                ),
            }
            .into());
        }
        let points = parse_points(&boundary_coords)?;
        let mut curves: Vec<Vec<Point3>> = Vec::with_capacity(4);
        let mut offset = 0usize;
        for &len in &curve_lengths {
            let l = len as usize;
            if offset + l > points.len() {
                return Err(WasmError::InvalidInput {
                    reason: "curve_lengths exceed total coordinate count".into(),
                }
                .into());
            }
            curves.push(points[offset..offset + l].to_vec());
            offset += l;
        }
        let face_id = brepkit_operations::fill_face::fill_coons_patch(&mut self.topo, &curves)?;
        Ok(face_id_to_u32(face_id))
    }

    /// Untrim a NURBS face by fitting a new surface to the trimmed region.
    ///
    /// Returns a new face handle.
    #[wasm_bindgen(js_name = "untrimFace")]
    pub fn untrim_face(
        &mut self,
        face: u32,
        samples_per_curve: u32,
        interior_samples: u32,
    ) -> Result<u32, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let surface = match face_data.surface() {
            FaceSurface::Nurbs(s) => s.clone(),
            _ => {
                return Err(WasmError::InvalidInput {
                    reason: "untrim only works on NURBS faces".into(),
                }
                .into());
            }
        };
        // Build trim curves from wire edges projected to UV space
        let wire_id = face_data.outer_wire();
        let wire = self.topo.wire(wire_id)?;
        let mut trim_curves = Vec::new();
        for oe in wire.edges() {
            let edge = self.topo.edge(oe.edge())?;
            let v_start = self.topo.vertex(edge.start())?;
            let v_end = self.topo.vertex(edge.end())?;
            // Project endpoints to UV
            let uv_start = project_to_uv(&surface, v_start.point());
            let uv_end = project_to_uv(&surface, v_end.point());
            trim_curves.push(brepkit_operations::untrim::TrimCurve {
                curve: vec![uv_start, uv_end],
            });
        }
        let new_surface = brepkit_operations::untrim::untrim_face(
            &surface,
            &trim_curves,
            samples_per_curve as usize,
            interior_samples as usize,
        )?;
        Ok(face_id_to_u32(self.nurbs_surface_to_face(new_surface)?))
    }

    /// Offset a wire on a planar face.
    ///
    /// Returns a new wire handle.
    #[wasm_bindgen(js_name = "offsetWire")]
    pub fn offset_wire(&mut self, face: u32, distance: f64) -> Result<u32, JsError> {
        let face_id = self.resolve_face(face)?;
        let wire_id =
            brepkit_operations::offset_wire::offset_wire(&mut self.topo, face_id, distance)?;
        Ok(wire_id_to_u32(wire_id))
    }

    /// Get analytic surface parameters for a face.
    ///
    /// Returns a JSON string with type-dependent fields.
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

    // ── Batch 5: Assembly & Sketch bindings ──────────────────────────

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
            Ok(r) => (r.converged, r.iterations, r.max_residual),
            Err(_) => (false, max_iterations as usize, f64::NAN),
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

    /// Create a new empty assembly. Returns an assembly index.
    #[wasm_bindgen(js_name = "assemblyNew")]
    pub fn assembly_new(&mut self, name: &str) -> u32 {
        self.assemblies
            .push(brepkit_operations::assembly::Assembly::new(name));
        #[allow(clippy::cast_possible_truncation)]
        let idx = (self.assemblies.len() - 1) as u32;
        idx
    }

    /// Add a root component to an assembly.
    ///
    /// Returns the component ID.
    #[wasm_bindgen(js_name = "assemblyAddRoot")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn assembly_add_root(
        &mut self,
        assembly: u32,
        name: &str,
        solid: u32,
        matrix: Vec<f64>,
    ) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let mat = parse_mat4(&matrix)?;
        let asm = self
            .assemblies
            .get_mut(assembly as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "assembly",
                index: assembly as usize,
            })?;
        let cid = asm.add_root_component(name, solid_id, mat);
        #[allow(clippy::cast_possible_truncation)]
        Ok(cid as u32)
    }

    /// Add a child component to a parent in an assembly.
    ///
    /// Returns the component ID.
    #[wasm_bindgen(js_name = "assemblyAddChild")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn assembly_add_child(
        &mut self,
        assembly: u32,
        parent: u32,
        name: &str,
        solid: u32,
        matrix: Vec<f64>,
    ) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let mat = parse_mat4(&matrix)?;
        let asm = self
            .assemblies
            .get_mut(assembly as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "assembly",
                index: assembly as usize,
            })?;
        let cid = asm.add_child_component(parent as usize, name, solid_id, mat)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(cid as u32)
    }

    /// Flatten an assembly into `[(solid, matrix), ...]`.
    ///
    /// Returns a JSON string: `[{"solid": u32, "matrix": [16 floats]}, ...]`.
    #[wasm_bindgen(js_name = "assemblyFlatten")]
    pub fn assembly_flatten(&self, assembly: u32) -> Result<String, JsError> {
        let asm = self
            .assemblies
            .get(assembly as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "assembly",
                index: assembly as usize,
            })?;
        let flat = asm.flatten();
        let entries: Vec<serde_json::Value> = flat
            .iter()
            .map(|(solid_id, mat)| {
                serde_json::json!({
                    "solid": solid_id_to_u32(*solid_id),
                    "matrix": mat4_to_array(mat),
                })
            })
            .collect();
        Ok(serde_json::Value::Array(entries).to_string())
    }

    /// Get the bill of materials for an assembly.
    ///
    /// Returns a JSON string: `[{"name": "...", "solidIndex": n, "instanceCount": n}, ...]`.
    #[wasm_bindgen(js_name = "assemblyBom")]
    pub fn assembly_bom(&self, assembly: u32) -> Result<String, JsError> {
        let asm = self
            .assemblies
            .get(assembly as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "assembly",
                index: assembly as usize,
            })?;
        let bom = asm.bill_of_materials();
        let entries: Vec<serde_json::Value> = bom
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "name": entry.name,
                    "solidIndex": entry.solid_index,
                    "instanceCount": entry.instance_count,
                })
            })
            .collect();
        Ok(serde_json::Value::Array(entries).to_string())
    }

    // ── Batch 6: Polygon offset ──────────────────────────────────────

    /// Offset a 2D polygon by a signed distance.
    ///
    /// `coords` is a flat array `[x,y, x,y, ...]` of 2D points.
    /// Returns a flat array of offset polygon coordinates.
    #[wasm_bindgen(js_name = "offsetPolygon2d")]
    #[allow(clippy::needless_pass_by_value, clippy::unused_self)]
    pub fn offset_polygon_2d(
        &self,
        coords: Vec<f64>,
        distance: f64,
        tolerance: f64,
    ) -> Result<Vec<f64>, JsError> {
        if coords.len() % 2 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "2D coordinate array length must be even, got {}",
                    coords.len()
                ),
            }
            .into());
        }
        let points: Vec<brepkit_math::vec::Point2> = coords
            .chunks_exact(2)
            .map(|c| brepkit_math::vec::Point2::new(c[0], c[1]))
            .collect();
        let result = brepkit_math::polygon_offset::offset_polygon_2d(&points, distance, tolerance)?;
        Ok(result.iter().flat_map(|p| [p.x(), p.y()]).collect())
    }

    // ── 2D Blueprint Operations ────────────────────────────────────

    /// Test if a 2D point is inside a closed polygon.
    ///
    /// `polygon_coords` is a flat array `[x,y, x,y, ...]`.
    /// Returns `true` if the point is inside the polygon (winding number test).
    #[wasm_bindgen(js_name = "pointInPolygon2d")]
    #[allow(clippy::unused_self)]
    pub fn point_in_polygon_2d(
        &self,
        polygon_coords: Vec<f64>,
        px: f64,
        py: f64,
    ) -> Result<bool, JsError> {
        if polygon_coords.len() % 2 != 0 || polygon_coords.len() < 6 {
            return Err(WasmError::InvalidInput {
                reason: "polygon needs at least 3 points (6 coordinates)".into(),
            }
            .into());
        }
        let polygon: Vec<brepkit_math::vec::Point2> = polygon_coords
            .chunks_exact(2)
            .map(|c| brepkit_math::vec::Point2::new(c[0], c[1]))
            .collect();
        let point = brepkit_math::vec::Point2::new(px, py);
        Ok(brepkit_math::predicates::point_in_polygon(point, &polygon))
    }

    /// Test if two 2D polygons intersect (overlap).
    ///
    /// Both polygons are flat arrays `[x,y, x,y, ...]`.
    /// Returns `true` if any vertex of one polygon is inside the other
    /// or if any edges cross.
    #[wasm_bindgen(js_name = "polygonsIntersect2d")]
    #[allow(clippy::unused_self)]
    pub fn polygons_intersect_2d(
        &self,
        coords_a: Vec<f64>,
        coords_b: Vec<f64>,
    ) -> Result<bool, JsError> {
        let poly_a = parse_polygon_2d(&coords_a)?;
        let poly_b = parse_polygon_2d(&coords_b)?;
        Ok(polygons_overlap_2d(&poly_a, &poly_b))
    }

    /// Compute the boolean intersection of two 2D polygons.
    ///
    /// Both polygons are flat arrays `[x,y, x,y, ...]`.
    /// Returns a flat array of the intersection polygon coordinates,
    /// or an empty array if they don't intersect.
    ///
    /// Uses the Sutherland-Hodgman algorithm (convex clipper).
    #[wasm_bindgen(js_name = "intersectPolygons2d")]
    #[allow(clippy::unused_self)]
    pub fn intersect_polygons_2d(
        &self,
        coords_a: Vec<f64>,
        coords_b: Vec<f64>,
    ) -> Result<Vec<f64>, JsError> {
        let subject = parse_polygon_2d(&coords_a)?;
        let clip = parse_polygon_2d(&coords_b)?;
        let result = sutherland_hodgman_clip(&subject, &clip);
        Ok(result.iter().flat_map(|p| [p.x(), p.y()]).collect())
    }

    /// Find common (shared) edges between two adjacent 2D polygons.
    ///
    /// Both polygons are flat arrays `[x,y, x,y, ...]`.
    /// Returns a flat array of common segment endpoints `[x1,y1, x2,y2, ...]`,
    /// or an empty array if no common segments exist.
    #[wasm_bindgen(js_name = "commonSegment2d")]
    #[allow(clippy::unused_self)]
    pub fn common_segment_2d(
        &self,
        coords_a: Vec<f64>,
        coords_b: Vec<f64>,
    ) -> Result<Vec<f64>, JsError> {
        let poly_a = parse_polygon_2d(&coords_a)?;
        let poly_b = parse_polygon_2d(&coords_b)?;
        let tolerance = 1e-7;
        let result = find_common_segments(&poly_a, &poly_b, tolerance);
        Ok(result
            .iter()
            .flat_map(|(a, b)| [a.x(), a.y(), b.x(), b.y()])
            .collect())
    }

    /// Round corners of a 2D polygon by inserting arc-approximation vertices.
    ///
    /// `coords` is a flat array `[x,y, x,y, ...]`.
    /// `radius` is the fillet radius.
    /// Returns a flat array of the filleted polygon coordinates.
    #[wasm_bindgen(js_name = "fillet2d")]
    #[allow(clippy::unused_self)]
    pub fn fillet_2d(&self, coords: Vec<f64>, radius: f64) -> Result<Vec<f64>, JsError> {
        validate_positive(radius, "radius")?;
        let polygon = parse_polygon_2d(&coords)?;
        let result = fillet_polygon_2d(&polygon, radius);
        Ok(result.iter().flat_map(|p| [p.x(), p.y()]).collect())
    }

    /// Cut corners of a 2D polygon with flat bevels.
    ///
    /// `coords` is a flat array `[x,y, x,y, ...]`.
    /// `distance` is the chamfer distance from each corner.
    /// Returns a flat array of the chamfered polygon coordinates.
    #[wasm_bindgen(js_name = "chamfer2d")]
    #[allow(clippy::unused_self)]
    pub fn chamfer_2d(&self, coords: Vec<f64>, distance: f64) -> Result<Vec<f64>, JsError> {
        validate_positive(distance, "distance")?;
        let polygon = parse_polygon_2d(&coords)?;
        let result = chamfer_polygon_2d(&polygon, distance);
        Ok(result.iter().flat_map(|p| [p.x(), p.y()]).collect())
    }

    // ── Semantic APIs (Theme G) ──────────────────────────────────────

    /// Get the orientation of a shape.
    ///
    /// Returns `"forward"` for all faces (brepkit faces don't have an
    /// independent orientation flag; the normal direction is canonical).
    #[allow(clippy::unused_self)]
    #[must_use]
    #[wasm_bindgen(js_name = "getShapeOrientation")]
    pub fn get_shape_orientation(&self, _id: u32) -> String {
        // In brepkit, face normals are always canonical (outward-pointing).
        // There is no separate orientation flag like OCCT's TopAbs_Orientation.
        "forward".to_string()
    }

    /// Reverse the orientation of a face or edge.
    ///
    /// For faces: creates a new face with negated plane normal.
    /// For edges: creates a new edge with swapped start/end vertices.
    /// Returns the handle of the new reversed shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is neither a valid face nor edge.
    #[wasm_bindgen(js_name = "reverseShape")]
    pub fn reverse_shape(&mut self, id: u32) -> Result<u32, JsError> {
        // Try as face
        if let Ok(face_id) = self.resolve_face(id) {
            let face = self.topo.face(face_id)?;
            let outer_wire = face.outer_wire();
            let inner_wires: Vec<_> = face.inner_wires().to_vec();
            let new_surface = match face.surface() {
                FaceSurface::Plane { normal, d } => FaceSurface::Plane {
                    normal: -*normal,
                    d: -*d,
                },
                other => other.clone(),
            };
            let new_face = Face::new(outer_wire, inner_wires, new_surface);
            let new_fid = self.topo.faces.alloc(new_face);
            return Ok(face_id_to_u32(new_fid));
        }
        // Try as edge
        if let Ok(edge_id) = self.resolve_edge(id) {
            let edge = self.topo.edge(edge_id)?;
            let new_edge = Edge::new(edge.end(), edge.start(), edge.curve().clone());
            let new_eid = self.topo.edges.alloc(new_edge);
            return Ok(edge_id_to_u32(new_eid));
        }
        Err(WasmError::InvalidInput {
            reason: "reverseShape requires a face or edge handle".into(),
        }
        .into())
    }
}

// ── Private helpers for new bindings ────────────────────────────────

impl BrepKernel {
    /// Extract a `NurbsCurve` from an edge, or error if it's a line.
    fn extract_nurbs_curve(&self, edge: u32) -> Result<NurbsCurve, JsError> {
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
    fn nurbs_curve_to_edge(
        &mut self,
        points: &[Point3],
        curve: NurbsCurve,
    ) -> brepkit_topology::edge::EdgeId {
        let start = points[0];
        let end = points[points.len() - 1];
        let v_start = self.topo.vertices.alloc(Vertex::new(start, TOL));
        let v_end = self.topo.vertices.alloc(Vertex::new(end, TOL));
        self.topo
            .edges
            .alloc(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(curve)))
    }

    /// Create an edge from a `NurbsCurve`, evaluating its endpoints.
    fn nurbs_curve_to_edge_from_curve(
        &mut self,
        curve: &NurbsCurve,
    ) -> brepkit_topology::edge::EdgeId {
        let start = curve.evaluate(curve.knots()[0]);
        let end = curve.evaluate(*curve.knots().last().unwrap_or(&1.0));
        let v_start = self.topo.vertices.alloc(Vertex::new(start, TOL));
        let v_end = self.topo.vertices.alloc(Vertex::new(end, TOL));
        self.topo.edges.alloc(Edge::new(
            v_start,
            v_end,
            EdgeCurve::NurbsCurve(curve.clone()),
        ))
    }

    /// Create a face from a `NurbsSurface` with a rectangular domain wire.
    fn nurbs_surface_to_face(
        &mut self,
        surface: brepkit_math::nurbs::surface::NurbsSurface,
    ) -> Result<brepkit_topology::face::FaceId, JsError> {
        // Evaluate corner points from the surface domain
        let (u_min, u_max) = surface.domain_u();
        let (v_min, v_max) = surface.domain_v();
        let corners = [
            surface.evaluate(u_min, v_min),
            surface.evaluate(u_max, v_min),
            surface.evaluate(u_max, v_max),
            surface.evaluate(u_min, v_max),
        ];
        let verts: Vec<_> = corners
            .iter()
            .map(|p| self.topo.vertices.alloc(Vertex::new(*p, TOL)))
            .collect();
        let n = verts.len();
        let edges: Vec<_> = (0..n)
            .map(|i| {
                self.topo
                    .edges
                    .alloc(Edge::new(verts[i], verts[(i + 1) % n], EdgeCurve::Line))
            })
            .collect();
        let oriented: Vec<_> = edges
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let wire = Wire::new(oriented, true)?;
        let wid = self.topo.wires.alloc(wire);
        let face_id = self
            .topo
            .faces
            .alloc(Face::new(wid, vec![], FaceSurface::Nurbs(surface)));
        Ok(face_id)
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
                let solid = brepkit_operations::primitives::make_box(&mut self.topo, w, h, d)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "makeCylinder" => {
                let r = get_f64(args, "radius")?;
                let h = get_f64(args, "height")?;
                let solid = brepkit_operations::primitives::make_cylinder(&mut self.topo, r, h)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "makeSphere" => {
                let r = get_f64(args, "radius")?;
                let segments = get_u32(args, "segments").unwrap_or(16);
                let solid = brepkit_operations::primitives::make_sphere(
                    &mut self.topo,
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
                let solid = brepkit_operations::primitives::make_cone(&mut self.topo, br, tr, h)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "makeTorus" => {
                let major = get_f64(args, "majorRadius")?;
                let minor = get_f64(args, "minorRadius")?;
                let segments = get_u32(args, "segments").unwrap_or(16);
                let solid = brepkit_operations::primitives::make_torus(
                    &mut self.topo,
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
                let solid = brepkit_operations::primitives::make_sphere(&mut self.topo, 1.0, 16)
                    .map_err(|e| e.to_string())?;
                let mat = brepkit_math::mat::Mat4::scale(rx, ry, rz);
                transform_solid(&mut self.topo, solid, &mat).map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "fuse" => {
                let a = get_u32(args, "solidA")?;
                let b = get_u32(args, "solidB")?;
                let a_id = self.resolve_solid(a).map_err(|e| e.to_string())?;
                let b_id = self.resolve_solid(b).map_err(|e| e.to_string())?;
                let result = boolean(&mut self.topo, BooleanOp::Fuse, a_id, b_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "cut" => {
                let a = get_u32(args, "solidA")?;
                let b = get_u32(args, "solidB")?;
                let a_id = self.resolve_solid(a).map_err(|e| e.to_string())?;
                let b_id = self.resolve_solid(b).map_err(|e| e.to_string())?;
                let result = boolean(&mut self.topo, BooleanOp::Cut, a_id, b_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "intersect" => {
                let a = get_u32(args, "solidA")?;
                let b = get_u32(args, "solidB")?;
                let a_id = self.resolve_solid(a).map_err(|e| e.to_string())?;
                let b_id = self.resolve_solid(b).map_err(|e| e.to_string())?;
                let result = boolean(&mut self.topo, BooleanOp::Intersect, a_id, b_id)
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
                transform_solid(&mut self.topo, solid_id, &mat).map_err(|e| e.to_string())?;
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
            "copySolid" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let copy = brepkit_operations::copy::copy_solid(&mut self.topo, solid_id)
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
                    &mut self.topo,
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
                    extrude(&mut self.topo, face_id, dir, dist).map_err(|e| e.to_string())?;
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
                    &mut self.topo,
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
                let solid = sweep(&mut self.topo, face_id, &curve).map_err(|e| e.to_string())?;
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
                let result =
                    brepkit_operations::chamfer::chamfer(&mut self.topo, solid_id, &edge_ids, dist)
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
                    try_fillet(&mut self.topo, solid_id, &edge_ids, radius)
                }));
                let result = match fillet_result {
                    Ok(inner) => inner.map_err(|e| e.to_string())?,
                    Err(panic_info) => {
                        return Err(panic_message(&panic_info, "Fillet"));
                    }
                };
                Ok(serde_json::json!(solid_id_to_u32(result)))
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
                    &mut self.topo,
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
                    &mut self.topo,
                    solid_id,
                    Point3::new(px, py, pz),
                    Vec3::new(nx, ny, nz),
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "healSolid" => {
                let s = get_u32(args, "solid")?;
                let tol = get_f64(args, "tolerance").unwrap_or(1e-7);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                brepkit_operations::heal::heal_solid(&mut self.topo, solid_id, tol)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid_id)))
            }
            "repairSolid" => {
                let s = get_u32(args, "solid")?;
                let tol = get_f64(args, "tolerance").unwrap_or(1e-7);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let report = brepkit_operations::heal::repair_solid(&mut self.topo, solid_id, tol)
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
                let result = brepkit_operations::loft::loft(&mut self.topo, &face_ids)
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
                let result = brepkit_operations::loft::loft_smooth(&mut self.topo, &face_ids)
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
                    &mut self.topo,
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
                    &mut self.topo,
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
                    brepkit_operations::defeature::defeature(&mut self.topo, solid_id, &face_ids)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "copyWire" => {
                let w = get_u32(args, "wire")?;
                let wire_id = self.resolve_wire(w).map_err(|e| e.to_string())?;
                let copy = brepkit_operations::copy::copy_wire(&mut self.topo, wire_id)
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
                brepkit_operations::transform::transform_wire(&mut self.topo, wire_id, &mat)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(null))
            }
            "offsetFace" => {
                let f = get_u32(args, "face")?;
                let dist = get_f64(args, "distance")?;
                let samples = get_u32(args, "samples").unwrap_or(16);
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let result = brepkit_operations::offset_face::offset_face(
                    &mut self.topo,
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
                let result =
                    brepkit_operations::offset_solid::offset_solid(&mut self.topo, solid_id, dist)
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
                    &mut self.topo,
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
                    &mut self.topo,
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
                let solid = brepkit_operations::sew::sew_faces(&mut self.topo, &face_ids, tol)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "thicken" => {
                let f = get_u32(args, "face")?;
                let thickness = get_f64(args, "thickness")?;
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let result =
                    brepkit_operations::thicken::thicken(&mut self.topo, face_id, thickness)
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
                let solid = brepkit_operations::pipe::pipe(&mut self.topo, face_id, &curve, None)
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
                    &mut self.topo,
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
                    &mut self.topo,
                    solid_id,
                    &face_ids,
                    dir,
                    neutral,
                    angle,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            _ => Err(format!("unknown operation: {op}")),
        }
    }
}

// ── 2D Blueprint Helpers ──────────────────────────────────────────

fn parse_polygon_2d(coords: &[f64]) -> Result<Vec<brepkit_math::vec::Point2>, JsError> {
    if coords.len() % 2 != 0 || coords.len() < 6 {
        return Err(WasmError::InvalidInput {
            reason: "polygon needs at least 3 points (6 coordinates)".into(),
        }
        .into());
    }
    Ok(coords
        .chunks_exact(2)
        .map(|c| brepkit_math::vec::Point2::new(c[0], c[1]))
        .collect())
}

/// Check if two 2D polygons overlap using vertex containment + edge crossing.
fn polygons_overlap_2d(a: &[brepkit_math::vec::Point2], b: &[brepkit_math::vec::Point2]) -> bool {
    use brepkit_math::predicates::point_in_polygon;

    // Check if any vertex of A is inside B or vice versa.
    for p in a {
        if point_in_polygon(*p, b) {
            return true;
        }
    }
    for p in b {
        if point_in_polygon(*p, a) {
            return true;
        }
    }

    // Check edge crossings.
    for i in 0..a.len() {
        let a1 = a[i];
        let a2 = a[(i + 1) % a.len()];
        for j in 0..b.len() {
            let b1 = b[j];
            let b2 = b[(j + 1) % b.len()];
            if segments_intersect_2d(a1, a2, b1, b2) {
                return true;
            }
        }
    }
    false
}

/// Test if two 2D line segments intersect (proper crossing).
fn segments_intersect_2d(
    a1: brepkit_math::vec::Point2,
    a2: brepkit_math::vec::Point2,
    b1: brepkit_math::vec::Point2,
    b2: brepkit_math::vec::Point2,
) -> bool {
    let d1 = cross_2d(b1, b2, a1);
    let d2 = cross_2d(b1, b2, a2);
    let d3 = cross_2d(a1, a2, b1);
    let d4 = cross_2d(a1, a2, b2);

    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

fn cross_2d(
    a: brepkit_math::vec::Point2,
    b: brepkit_math::vec::Point2,
    c: brepkit_math::vec::Point2,
) -> f64 {
    (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x())
}

/// Sutherland-Hodgman polygon clipping algorithm.
fn sutherland_hodgman_clip(
    subject: &[brepkit_math::vec::Point2],
    clip: &[brepkit_math::vec::Point2],
) -> Vec<brepkit_math::vec::Point2> {
    use brepkit_math::vec::Point2;

    let mut output: Vec<Point2> = subject.to_vec();

    for i in 0..clip.len() {
        if output.is_empty() {
            return output;
        }
        let edge_start = clip[i];
        let edge_end = clip[(i + 1) % clip.len()];
        let input = output;
        output = Vec::new();

        for j in 0..input.len() {
            let current = input[j];
            let previous = input[(j + input.len() - 1) % input.len()];

            let curr_inside = cross_2d(edge_start, edge_end, current) >= 0.0;
            let prev_inside = cross_2d(edge_start, edge_end, previous) >= 0.0;

            if curr_inside {
                if !prev_inside {
                    if let Some(p) = line_intersect_2d(previous, current, edge_start, edge_end) {
                        output.push(p);
                    }
                }
                output.push(current);
            } else if prev_inside {
                if let Some(p) = line_intersect_2d(previous, current, edge_start, edge_end) {
                    output.push(p);
                }
            }
        }
    }

    output
}

/// Find the intersection point of two 2D line segments (as infinite lines).
fn line_intersect_2d(
    a1: brepkit_math::vec::Point2,
    a2: brepkit_math::vec::Point2,
    b1: brepkit_math::vec::Point2,
    b2: brepkit_math::vec::Point2,
) -> Option<brepkit_math::vec::Point2> {
    let dx_a = a2.x() - a1.x();
    let dy_a = a2.y() - a1.y();
    let dx_b = b2.x() - b1.x();
    let dy_b = b2.y() - b1.y();
    let denom = dx_a * dy_b - dy_a * dx_b;
    if denom.abs() < 1e-15 {
        return None;
    }
    let t = ((b1.x() - a1.x()) * dy_b - (b1.y() - a1.y()) * dx_b) / denom;
    Some(brepkit_math::vec::Point2::new(
        a1.x() + t * dx_a,
        a1.y() + t * dy_a,
    ))
}

/// Find common (collinear, overlapping) edges between two polygons.
fn find_common_segments(
    a: &[brepkit_math::vec::Point2],
    b: &[brepkit_math::vec::Point2],
    tolerance: f64,
) -> Vec<(brepkit_math::vec::Point2, brepkit_math::vec::Point2)> {
    let mut results = Vec::new();
    let tol_sq = tolerance * tolerance;

    for i in 0..a.len() {
        let a1 = a[i];
        let a2 = a[(i + 1) % a.len()];
        for j in 0..b.len() {
            let b1 = b[j];
            let b2 = b[(j + 1) % b.len()];

            // Check if edge A and edge B are collinear and overlapping.
            // Both endpoints of B must be close to line through A, or vice versa.
            let dist_b1 = point_to_line_dist_sq_2d(b1, a1, a2);
            let dist_b2 = point_to_line_dist_sq_2d(b2, a1, a2);

            if dist_b1 < tol_sq && dist_b2 < tol_sq {
                // Edges are collinear. Check for overlap by projecting onto A's direction.
                let dx = a2.x() - a1.x();
                let dy = a2.y() - a1.y();
                let len_sq = dx * dx + dy * dy;
                if len_sq < tol_sq {
                    continue;
                }
                let t1 = ((b1.x() - a1.x()) * dx + (b1.y() - a1.y()) * dy) / len_sq;
                let t2 = ((b2.x() - a1.x()) * dx + (b2.y() - a1.y()) * dy) / len_sq;
                let t_min = t1.min(t2).max(0.0);
                let t_max = t1.max(t2).min(1.0);
                if t_max - t_min > tolerance / len_sq.sqrt() {
                    results.push((
                        brepkit_math::vec::Point2::new(a1.x() + t_min * dx, a1.y() + t_min * dy),
                        brepkit_math::vec::Point2::new(a1.x() + t_max * dx, a1.y() + t_max * dy),
                    ));
                }
            }
        }
    }
    results
}

fn point_to_line_dist_sq_2d(
    p: brepkit_math::vec::Point2,
    a: brepkit_math::vec::Point2,
    b: brepkit_math::vec::Point2,
) -> f64 {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-30 {
        let ex = p.x() - a.x();
        let ey = p.y() - a.y();
        return ex * ex + ey * ey;
    }
    let cross = (p.x() - a.x()) * dy - (p.y() - a.y()) * dx;
    (cross * cross) / len_sq
}

/// Round all corners of a 2D polygon with arc approximations.
fn fillet_polygon_2d(
    polygon: &[brepkit_math::vec::Point2],
    radius: f64,
) -> Vec<brepkit_math::vec::Point2> {
    use brepkit_math::vec::Point2;

    let n = polygon.len();
    if n < 3 {
        return polygon.to_vec();
    }

    let arc_segments = 8; // Number of segments per fillet arc
    let mut result = Vec::with_capacity(n * (arc_segments + 1));

    for i in 0..n {
        let prev = polygon[(i + n - 1) % n];
        let curr = polygon[i];
        let next = polygon[(i + 1) % n];

        let d_prev = ((prev.x() - curr.x()).powi(2) + (prev.y() - curr.y()).powi(2)).sqrt();
        let d_next = ((next.x() - curr.x()).powi(2) + (next.y() - curr.y()).powi(2)).sqrt();

        let max_r = (d_prev.min(d_next) / 2.0).min(radius);

        if max_r < 1e-10 {
            result.push(curr);
            continue;
        }

        // Direction vectors from corner to adjacent vertices
        let dir_prev_x = (prev.x() - curr.x()) / d_prev;
        let dir_prev_y = (prev.y() - curr.y()) / d_prev;
        let dir_next_x = (next.x() - curr.x()) / d_next;
        let dir_next_y = (next.y() - curr.y()) / d_next;

        // Tangent points on edges
        let t1 = Point2::new(curr.x() + dir_prev_x * max_r, curr.y() + dir_prev_y * max_r);
        let t2 = Point2::new(curr.x() + dir_next_x * max_r, curr.y() + dir_next_y * max_r);

        // Generate arc points from t1 to t2
        for k in 0..=arc_segments {
            #[allow(clippy::cast_precision_loss)]
            let t = k as f64 / arc_segments as f64;
            let x = t2.x().mul_add(t, t1.x() * (1.0 - t));
            let y = t2.y().mul_add(t, t1.y() * (1.0 - t));

            // Push point toward the arc center for a circular approximation
            let mid_x = f64::midpoint(t1.x(), t2.x());
            let mid_y = f64::midpoint(t1.y(), t2.y());
            let to_corner_x = curr.x() - mid_x;
            let to_corner_y = curr.y() - mid_y;
            let corner_dist = (to_corner_x * to_corner_x + to_corner_y * to_corner_y).sqrt();

            if corner_dist > 1e-10 {
                // Compute the bulge: how much to push along the corner bisector
                let chord_half =
                    ((t2.x() - t1.x()).powi(2) + (t2.y() - t1.y()).powi(2)).sqrt() / 2.0;
                let sagitta = if max_r > chord_half {
                    max_r - (max_r * max_r - chord_half * chord_half).sqrt()
                } else {
                    0.0
                };

                // Blend factor: maximum at midpoint (t=0.5), zero at endpoints
                let blend = 4.0 * t * (1.0 - t); // parabolic blend
                let push = sagitta * blend;

                let nx = to_corner_x / corner_dist;
                let ny = to_corner_y / corner_dist;
                result.push(Point2::new(x + nx * push, y + ny * push));
            } else {
                result.push(Point2::new(x, y));
            }
        }
    }

    result
}

/// Cut all corners of a 2D polygon with flat bevels.
fn chamfer_polygon_2d(
    polygon: &[brepkit_math::vec::Point2],
    distance: f64,
) -> Vec<brepkit_math::vec::Point2> {
    use brepkit_math::vec::Point2;

    let n = polygon.len();
    if n < 3 {
        return polygon.to_vec();
    }

    let mut result = Vec::with_capacity(n * 2);

    for i in 0..n {
        let prev = polygon[(i + n - 1) % n];
        let curr = polygon[i];
        let next = polygon[(i + 1) % n];

        let d_prev = ((prev.x() - curr.x()).powi(2) + (prev.y() - curr.y()).powi(2)).sqrt();
        let d_next = ((next.x() - curr.x()).powi(2) + (next.y() - curr.y()).powi(2)).sqrt();

        let d = (d_prev.min(d_next) / 2.0).min(distance);

        if d < 1e-10 {
            result.push(curr);
            continue;
        }

        // Two chamfer points: one on previous edge, one on next edge
        result.push(Point2::new(
            curr.x() + (prev.x() - curr.x()) / d_prev * d,
            curr.y() + (prev.y() - curr.y()) / d_prev * d,
        ));
        result.push(Point2::new(
            curr.x() + (next.x() - curr.x()) / d_next * d,
            curr.y() + (next.y() - curr.y()) / d_next * d,
        ));
    }

    result
}

/// Create a tiny degenerate polygon face at a point, matching the vertex
/// count of the first existing profile. Used for loft start/end points.
fn create_apex_face(
    topo: &mut Topology,
    point: Point3,
    existing_profiles: &[brepkit_topology::face::FaceId],
) -> Result<brepkit_topology::face::FaceId, JsError> {
    // Determine target vertex count from the first profile.
    let n = if let Some(&fid) = existing_profiles.first() {
        let verts = brepkit_operations::boolean::face_polygon(topo, fid)
            .map_err(|e: brepkit_operations::OperationsError| JsError::new(&e.to_string()))?;
        verts.len().max(3)
    } else {
        3
    };

    // Create a tiny polygon at the apex point.
    let epsilon = 1e-6;
    let mut pts = Vec::with_capacity(n);
    #[allow(clippy::cast_precision_loss)]
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        pts.push(Point3::new(
            point.x() + epsilon * angle.cos(),
            point.y() + epsilon * angle.sin(),
            point.z(),
        ));
    }

    let wire_id = brepkit_topology::builder::make_polygon_wire(topo, &pts)
        .map_err(|e| JsError::new(&e.to_string()))?;
    let face_id = brepkit_topology::builder::make_face_from_wire(topo, wire_id)
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(face_id)
}

/// Detect if a NURBS curve represents an analytic curve type.
///
/// Checks if the curve is a circle or ellipse by sampling points
/// and verifying they are coplanar and equidistant from a center.
fn detect_nurbs_curve_type(nc: &brepkit_math::nurbs::NurbsCurve) -> &'static str {
    // A rational degree-2 NURBS with specific weight patterns can represent
    // conic sections. Check if all sampled points lie on a circle.
    if nc.degree() < 2 || !nc.is_rational() {
        return "BSPLINE_CURVE";
    }

    let (u_min, u_max) = nc.domain();
    let n_samples = 16;

    // Check if the curve is closed (start ≈ end) to avoid sampling the
    // duplicate endpoint, which would bias the center calculation.
    let start_pt = nc.evaluate(u_min);
    let end_pt = nc.evaluate(u_max);
    let is_closed = (start_pt - end_pt).length() < 1e-6;

    // Sample points along the curve. For closed curves, exclude the
    // last point (t=u_max) since it duplicates the first.
    let mut points = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        #[allow(clippy::cast_precision_loss)]
        let t = if is_closed {
            u_min + (u_max - u_min) * (i as f64) / (n_samples as f64)
        } else {
            u_min + (u_max - u_min) * (i as f64) / ((n_samples - 1) as f64)
        };
        points.push(nc.evaluate(t));
    }

    // Compute center as average of all sampled points
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut cz = 0.0_f64;
    for p in &points {
        cx += p.x();
        cy += p.y();
        cz += p.z();
    }
    #[allow(clippy::cast_precision_loss)]
    let n = points.len() as f64;
    let center = brepkit_math::vec::Point3::new(cx / n, cy / n, cz / n);

    // Check if all points are equidistant from center (circle test)
    let distances: Vec<f64> = points.iter().map(|p| (*p - center).length()).collect();
    let avg_dist = distances.iter().sum::<f64>() / n;

    if avg_dist < 1e-10 {
        return "BSPLINE_CURVE";
    }

    let tol = avg_dist * 1e-4; // 0.01% relative tolerance
    let is_circle = distances.iter().all(|d| (d - avg_dist).abs() < tol);

    if is_circle {
        // Check coplanarity — all points should lie in a plane through center
        let v0 = points[0] - center;
        let v1 = points[n_samples / 4] - center;
        let normal = v0.cross(v1);
        let normal_len = normal.length();
        if normal_len < 1e-10 {
            return "BSPLINE_CURVE";
        }
        let normal = brepkit_math::vec::Vec3::new(
            normal.x() / normal_len,
            normal.y() / normal_len,
            normal.z() / normal_len,
        );

        let coplanar = points
            .iter()
            .all(|p| ((*p - center).dot(normal)).abs() < tol);

        if coplanar {
            return "CIRCLE";
        }
    }

    // TODO: Could also detect ELLIPSE (non-uniform distances but elliptic pattern)
    "BSPLINE_CURVE"
}

/// Detect if a NURBS surface represents an analytic surface type.
///
/// Checks if the surface is a sphere, cylinder, cone, or torus by
/// sampling a grid of points and analyzing their geometric properties.
fn detect_nurbs_surface_type(ns: &brepkit_math::nurbs::surface::NurbsSurface) -> &'static str {
    let (u_min, u_max) = ns.domain_u();
    let (v_min, v_max) = ns.domain_v();
    let n = 8; // 8×8 grid = 64 sample points

    // Sample points on the surface
    let mut points = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            #[allow(clippy::cast_precision_loss)]
            let u = u_min + (u_max - u_min) * (i as f64) / ((n - 1) as f64);
            #[allow(clippy::cast_precision_loss)]
            let v = v_min + (v_max - v_min) * (j as f64) / ((n - 1) as f64);
            points.push(ns.evaluate(u, v));
        }
    }

    // Compute center as average
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut cz = 0.0_f64;
    for p in &points {
        cx += p.x();
        cy += p.y();
        cz += p.z();
    }
    #[allow(clippy::cast_precision_loss)]
    let np = points.len() as f64;
    let center = brepkit_math::vec::Point3::new(cx / np, cy / np, cz / np);

    // Check if all points equidistant from center (sphere test)
    let distances: Vec<f64> = points.iter().map(|p| (*p - center).length()).collect();
    let avg_dist = distances.iter().sum::<f64>() / np;

    if avg_dist < 1e-10 {
        return "bspline";
    }

    let tol = avg_dist * 1e-3; // 0.1% relative tolerance
    let is_sphere = distances.iter().all(|d| (d - avg_dist).abs() < tol);

    if is_sphere {
        return "sphere";
    }

    // Cylinder test: points should be equidistant from an axis line.
    // Try to find the axis by PCA (direction of maximum variance).
    // For a cylinder, points cluster around a line; cross-section is a circle.
    if let Some(axis_dir) = estimate_cylinder_axis(&points, center) {
        let projected_distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let v = *p - center;
                let along_axis = v.dot(axis_dir);
                let radial = brepkit_math::vec::Vec3::new(
                    v.x() - axis_dir.x() * along_axis,
                    v.y() - axis_dir.y() * along_axis,
                    v.z() - axis_dir.z() * along_axis,
                );
                radial.length()
            })
            .collect();

        let avg_r = projected_distances.iter().sum::<f64>() / np;
        if avg_r > 1e-10 {
            let r_tol = avg_r * 1e-3;
            let is_cylinder = projected_distances
                .iter()
                .all(|d| (d - avg_r).abs() < r_tol);
            if is_cylinder {
                return "cylinder";
            }
        }
    }

    "bspline"
}

/// Estimate the cylinder axis direction from a set of surface sample points
/// using a simple PCA-like approach (direction of maximum variance).
fn estimate_cylinder_axis(
    points: &[brepkit_math::vec::Point3],
    center: brepkit_math::vec::Point3,
) -> Option<brepkit_math::vec::Vec3> {
    // Build covariance matrix
    let mut cxx = 0.0_f64;
    let mut cxy = 0.0_f64;
    let mut cxz = 0.0_f64;
    let mut cyy = 0.0_f64;
    let mut cyz = 0.0_f64;
    let mut czz = 0.0_f64;

    for p in points {
        let dx = p.x() - center.x();
        let dy = p.y() - center.y();
        let dz = p.z() - center.z();
        cxx += dx * dx;
        cxy += dx * dy;
        cxz += dx * dz;
        cyy += dy * dy;
        cyz += dy * dz;
        czz += dz * dz;
    }

    // Power iteration to find the principal eigenvector
    let mut v = brepkit_math::vec::Vec3::new(1.0, 0.0, 0.0);
    for _ in 0..20 {
        let new_v = brepkit_math::vec::Vec3::new(
            v.x().mul_add(cxx, v.y().mul_add(cxy, v.z() * cxz)),
            v.x().mul_add(cxy, v.y().mul_add(cyy, v.z() * cyz)),
            v.x().mul_add(cxz, v.y().mul_add(cyz, v.z() * czz)),
        );
        let len = new_v.length();
        if len < 1e-15 {
            return None;
        }
        v = brepkit_math::vec::Vec3::new(new_v.x() / len, new_v.y() / len, new_v.z() / len);
    }
    Some(v)
}

const fn face_id_to_u32(id: brepkit_topology::face::FaceId) -> u32 {
    id.index() as u32
}

/// Convert a `SolidId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
const fn solid_id_to_u32(id: brepkit_topology::solid::SolidId) -> u32 {
    id.index() as u32
}

/// Convert a `VertexId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
const fn vertex_id_to_u32(id: brepkit_topology::vertex::VertexId) -> u32 {
    id.index() as u32
}

/// Convert an `EdgeId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
const fn edge_id_to_u32(id: brepkit_topology::edge::EdgeId) -> u32 {
    id.index() as u32
}

/// Convert a `WireId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
const fn wire_id_to_u32(id: brepkit_topology::wire::WireId) -> u32 {
    id.index() as u32
}

/// Convert a `ShellId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
const fn shell_id_to_u32(id: brepkit_topology::shell::ShellId) -> u32 {
    id.index() as u32
}

/// Convert a `CompoundId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
const fn compound_id_to_u32(id: brepkit_topology::compound::CompoundId) -> u32 {
    id.index() as u32
}

impl Default for BrepKernel {
    fn default() -> Self {
        Self::new()
    }
}

// ── Batch argument helpers ─────────────────────────────────────────

/// Extract a required `f64` value from a JSON object.
fn get_f64(args: &serde_json::Value, key: &str) -> Result<f64, String> {
    args[key]
        .as_f64()
        .ok_or_else(|| format!("missing or invalid '{key}'"))
}

/// Extract a required `u32` value from a JSON object.
fn get_u32(args: &serde_json::Value, key: &str) -> Result<u32, String> {
    args[key]
        .as_u64()
        .map(|v| v as u32)
        .ok_or_else(|| format!("missing or invalid '{key}'"))
}

// ── Free helper functions for new bindings ────────────────────────

/// Parse flat `[x,y,z, ...]` coordinates into `Vec<Point3>`.
fn parse_points(coords: &[f64]) -> Result<Vec<Point3>, JsError> {
    if coords.len() % 3 != 0 {
        return Err(WasmError::InvalidInput {
            reason: format!(
                "coordinate array length must be a multiple of 3, got {}",
                coords.len()
            ),
        }
        .into());
    }
    Ok(coords
        .chunks_exact(3)
        .map(|c| Point3::new(c[0], c[1], c[2]))
        .collect())
}

/// Parse flat coordinates into a 2D grid of points.
fn parse_point_grid(coords: &[f64], rows: usize, cols: usize) -> Result<Vec<Vec<Point3>>, JsError> {
    let points = parse_points(coords)?;
    if points.len() != rows * cols {
        return Err(WasmError::InvalidInput {
            reason: format!(
                "expected {} points ({}x{}), got {}",
                rows * cols,
                rows,
                cols,
                points.len()
            ),
        }
        .into());
    }
    Ok(points.chunks(cols).map(|row| row.to_vec()).collect())
}

/// Serialize a `Feature` enum to JSON.
fn serialize_feature(f: &brepkit_operations::feature_recognition::Feature) -> serde_json::Value {
    use brepkit_operations::feature_recognition::Feature;
    match f {
        Feature::Hole { faces, diameter } => serde_json::json!({
            "type": "hole",
            "faces": faces.iter().map(|f| face_id_to_u32(*f)).collect::<Vec<_>>(),
            "diameter": diameter,
        }),
        Feature::Chamfer {
            face,
            adjacent,
            angle,
        } => serde_json::json!({
            "type": "chamfer",
            "face": face_id_to_u32(*face),
            "adjacent": [face_id_to_u32(adjacent.0), face_id_to_u32(adjacent.1)],
            "angle": angle,
        }),
        Feature::FilletLike { face, area } => serde_json::json!({
            "type": "filletLike",
            "face": face_id_to_u32(*face),
            "area": area,
        }),
        Feature::Pocket { floor, walls } => serde_json::json!({
            "type": "pocket",
            "floor": face_id_to_u32(*floor),
            "walls": walls.iter().map(|f| face_id_to_u32(*f)).collect::<Vec<_>>(),
        }),
        Feature::Pattern {
            feature_indices,
            pattern_type,
            count,
            spacing,
        } => serde_json::json!({
            "type": "pattern",
            "featureIndices": feature_indices,
            "patternType": format!("{pattern_type:?}").to_lowercase(),
            "count": count,
            "spacing": spacing,
        }),
    }
}

/// Convert a `PointClassification` to a string.
fn classify_to_string(c: brepkit_operations::classify::PointClassification) -> String {
    match c {
        brepkit_operations::classify::PointClassification::Inside => "inside".into(),
        brepkit_operations::classify::PointClassification::Outside => "outside".into(),
        brepkit_operations::classify::PointClassification::OnBoundary => "boundary".into(),
    }
}

/// Build a `TriangleMesh` from flat position/index arrays.
fn build_triangle_mesh(
    positions: &[f64],
    indices: &[u32],
) -> Result<tessellate::TriangleMesh, JsError> {
    if positions.len() % 3 != 0 {
        return Err(WasmError::InvalidInput {
            reason: format!(
                "positions length must be a multiple of 3, got {}",
                positions.len()
            ),
        }
        .into());
    }
    let pts: Vec<Point3> = positions
        .chunks_exact(3)
        .map(|c| Point3::new(c[0], c[1], c[2]))
        .collect();
    // Compute normals as zero vectors (mesh_boolean recomputes them)
    let normals = vec![Vec3::new(0.0, 0.0, 0.0); pts.len()];
    Ok(tessellate::TriangleMesh {
        positions: pts,
        normals,
        indices: indices.to_vec(),
    })
}

/// Parse a boolean operation string to the enum.
fn parse_boolean_op(op: &str) -> Result<BooleanOp, JsError> {
    match op {
        "fuse" | "union" => Ok(BooleanOp::Fuse),
        "cut" | "difference" => Ok(BooleanOp::Cut),
        "intersect" | "intersection" => Ok(BooleanOp::Intersect),
        _ => Err(WasmError::InvalidInput {
            reason: format!("unknown boolean op: {op}"),
        }
        .into()),
    }
}

/// Convert a `TriangleMesh` to `JsMesh`.
fn triangle_mesh_to_js(mesh: &tessellate::TriangleMesh) -> JsMesh {
    JsMesh::from(mesh.clone())
}

/// Parse a sketch constraint from a JSON value.
fn parse_sketch_constraint(
    val: &serde_json::Value,
) -> Result<brepkit_operations::sketch::Constraint, JsError> {
    use brepkit_operations::sketch::Constraint;
    let ty = val["type"].as_str().unwrap_or("");
    match ty {
        "coincident" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            Ok(Constraint::Coincident(p1, p2))
        }
        "distance" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            let v = json_f64(val, "value")?;
            Ok(Constraint::Distance(p1, p2, v))
        }
        "fixX" => {
            let p = json_usize(val, "point")?;
            let v = json_f64(val, "value")?;
            Ok(Constraint::FixX(p, v))
        }
        "fixY" => {
            let p = json_usize(val, "point")?;
            let v = json_f64(val, "value")?;
            Ok(Constraint::FixY(p, v))
        }
        "vertical" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            Ok(Constraint::Vertical(p1, p2))
        }
        "horizontal" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            Ok(Constraint::Horizontal(p1, p2))
        }
        "angle" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            let v = json_f64(val, "value")?;
            Ok(Constraint::Angle(p1, p2, v))
        }
        "perpendicular" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            let p3 = json_usize(val, "p3")?;
            let p4 = json_usize(val, "p4")?;
            Ok(Constraint::Perpendicular(p1, p2, p3, p4))
        }
        "parallel" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            let p3 = json_usize(val, "p3")?;
            let p4 = json_usize(val, "p4")?;
            Ok(Constraint::Parallel(p1, p2, p3, p4))
        }
        _ => Err(WasmError::InvalidInput {
            reason: format!("unknown constraint type: {ty}"),
        }
        .into()),
    }
}

/// Extract a `usize` from a JSON value.
fn json_usize(val: &serde_json::Value, key: &str) -> Result<usize, JsError> {
    val[key].as_u64().map(|v| v as usize).ok_or_else(|| {
        WasmError::InvalidInput {
            reason: format!("missing or invalid '{key}'"),
        }
        .into()
    })
}

/// Extract an `f64` from a JSON value.
fn json_f64(val: &serde_json::Value, key: &str) -> Result<f64, JsError> {
    val[key].as_f64().ok_or_else(|| {
        WasmError::InvalidInput {
            reason: format!("missing or invalid '{key}'"),
        }
        .into()
    })
}

/// Parse a flat 16-element array into a `Mat4` (row-major).
fn parse_mat4(elems: &[f64]) -> Result<Mat4, JsError> {
    if elems.len() != 16 {
        return Err(WasmError::InvalidInput {
            reason: format!("matrix requires 16 elements, got {}", elems.len()),
        }
        .into());
    }
    let rows = std::array::from_fn(|i| std::array::from_fn(|j| elems[i * 4 + j]));
    Ok(Mat4(rows))
}

/// Convert a `Mat4` to a flat 16-element f64 array for JSON (row-major).
fn mat4_to_array(mat: &Mat4) -> Vec<f64> {
    let mut out = Vec::with_capacity(16);
    for row in &mat.0 {
        for &v in row {
            out.push(v);
        }
    }
    out
}

/// Project a 3D point onto a NURBS surface to get (u,v) parameters.
///
/// Uses a simple grid search + Newton refinement.
fn project_to_uv(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    point: Point3,
) -> brepkit_math::vec::Point2 {
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();
    let n = 10;
    let mut best_u = u_min;
    let mut best_v = v_min;
    let mut best_dist = f64::MAX;
    for i in 0..=n {
        for j in 0..=n {
            #[allow(clippy::cast_precision_loss)]
            let u = u_min + (u_max - u_min) * (i as f64) / (n as f64);
            #[allow(clippy::cast_precision_loss)]
            let v = v_min + (v_max - v_min) * (j as f64) / (n as f64);
            let pt = surface.evaluate(u, v);
            let dx = pt.x() - point.x();
            let dy = pt.y() - point.y();
            let dz = pt.z() - point.z();
            let dist = dx * dx + dy * dy + dz * dz;
            if dist < best_dist {
                best_dist = dist;
                best_u = u;
                best_v = v;
            }
        }
    }
    brepkit_math::vec::Point2::new(best_u, best_v)
}

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
        // First two should be ok with solid ids
        assert!(parsed[0]["ok"].is_number());
        assert!(parsed[1]["ok"].is_number());
        // Third should be a volume value
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
        // Copy should have a different handle
        assert_ne!(parsed[0]["ok"].as_u64(), parsed[1]["ok"].as_u64());
    }
}
