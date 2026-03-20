//! Modeling operation bindings (extrude, revolve, sweep, loft, fillet, etc.).

#![allow(
    clippy::missing_errors_doc,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use wasm_bindgen::prelude::*;

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};

use crate::error::{WasmError, validate_finite, validate_positive};
use crate::handles::{edge_id_to_u32, face_id_to_u32, solid_id_to_u32, wire_id_to_u32};
use crate::helpers::{
    TOL, classify_to_string, create_apex_face, filter_planar_edges, panic_message, parse_points,
    project_to_uv, try_fillet,
};
use crate::kernel::BrepKernel;

use brepkit_operations::extrude::extrude;
use brepkit_operations::offset_wire::JoinType;
use brepkit_operations::revolve::revolve;
use brepkit_operations::sweep::sweep;

/// Parse a join type string into a [`JoinType`] enum value.
///
/// Used by both the direct WASM binding and the batch dispatcher.
pub fn parse_join_type_str(s: &str) -> Result<JoinType, WasmError> {
    match s {
        "intersection" => Ok(JoinType::Intersection),
        "arc" => Ok(JoinType::Arc),
        "chamfer" => Ok(JoinType::Chamfer),
        _ => Err(WasmError::InvalidInput {
            reason: format!(
                "unknown join type '{s}', expected 'intersection', 'arc', or 'chamfer'"
            ),
        }),
    }
}

#[wasm_bindgen]
impl BrepKernel {
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
            self.topo_mut(),
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
        let solid_id = brepkit_operations::loft::loft(self.topo_mut(), &face_ids)?;
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
        let solid_id = brepkit_operations::loft::loft_smooth(self.topo_mut(), &face_ids)?;
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
                let apex_face = create_apex_face(self.topo_mut(), Point3::new(x, y, z), &face_ids)?;
                face_ids.insert(0, apex_face);
            }
        }

        // If endPoint is given, create a tiny degenerate triangle face and append.
        if let Some(ep) = opts.get("endPoint").and_then(|v| v.as_array()) {
            if ep.len() >= 3 {
                let x = ep[0].as_f64().unwrap_or(0.0);
                let y = ep[1].as_f64().unwrap_or(0.0);
                let z = ep[2].as_f64().unwrap_or(0.0);
                let apex_face = create_apex_face(self.topo_mut(), Point3::new(x, y, z), &face_ids)?;
                face_ids.push(apex_face);
            }
        }

        let ruled = opts
            .get("ruled")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);

        let solid_id = if ruled {
            brepkit_operations::loft::loft(self.topo_mut(), &face_ids)?
        } else {
            brepkit_operations::loft::loft_smooth(self.topo_mut(), &face_ids)?
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
            self.topo_mut(),
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
            brepkit_operations::chamfer::chamfer(self.topo_mut(), solid_id, &edge_ids, distance)?;
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
                let solid = if let Ok(s) = try_fillet(self.topo_mut(), solid_id, &edge_ids, radius)
                {
                    s
                } else {
                    // Filter to edges where both adjacent faces are planar.
                    let planar_edges = filter_planar_edges(&self.topo, solid_id, &edge_ids)?;
                    if planar_edges.is_empty() {
                        solid_id
                    } else {
                        try_fillet(self.topo_mut(), solid_id, &planar_edges, radius)
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
        let solid_id = extrude(self.topo_mut(), face_id, direction, distance)?;

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

        let solid_id = revolve(self.topo_mut(), face_id, origin, direction, angle_radians)?;

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

        let solid_id = sweep(self.topo_mut(), face_id, &path_curve)?;

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
            brepkit_operations::sweep::sweep_smooth(self.topo_mut(), face_id, &path_curve)?;
        Ok(solid_id_to_u32(solid_id))
    }

    // ── Offset Face ──────────────────────────────────────────────

    /// Offset a face by a distance along its surface normal.
    ///
    /// Returns the new offset face handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid or the operation fails.
    #[wasm_bindgen(js_name = "offsetFace")]
    pub fn offset_face(&mut self, face: u32, distance: f64, samples: u32) -> Result<u32, JsError> {
        validate_finite(distance, "distance")?;
        let face_id = self.resolve_face(face)?;
        let result = brepkit_operations::offset_face::offset_face(
            self.topo_mut(),
            face_id,
            distance,
            samples as usize,
        )?;
        Ok(face_id_to_u32(result))
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
    pub fn helical_sweep(
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
            self.topo_mut(),
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
            self.topo_mut(),
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
            self.topo_mut(),
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

        let solid_id = brepkit_operations::pipe::pipe(self.topo_mut(), face_id, &path_curve, None)?;
        Ok(solid_id_to_u32(solid_id))
    }

    // ── Sweep Along Edges ─────────────────────────────────────────

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
        let solid_id = sweep(self.topo_mut(), face_id, &path_curve)?;
        Ok(solid_id_to_u32(solid_id))
    }

    // ── Offset Solid ──────────────────────────────────────────────

    /// Offset (shell) a solid by a distance.
    ///
    /// Returns a new solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the distance is zero or the solid is invalid.
    #[wasm_bindgen(js_name = "offsetSolid")]
    pub fn offset_solid(&mut self, solid: u32, distance: f64) -> Result<u32, JsError> {
        validate_finite(distance, "distance")?;
        let solid_id = self.resolve_solid(solid)?;
        let result =
            brepkit_operations::offset_solid::offset_solid(self.topo_mut(), solid_id, distance)?;
        Ok(solid_id_to_u32(result))
    }

    /// Offset all faces of a solid outward or inward (V2 pipeline).
    ///
    /// Uses the new `brepkit-offset` engine with intersection-based joints.
    ///
    /// # Errors
    ///
    /// Returns an error if the distance is not finite or the solid is invalid.
    #[wasm_bindgen(js_name = "offsetSolidV2")]
    pub fn offset_solid_v2(&mut self, solid: u32, distance: f64) -> Result<u32, JsError> {
        validate_finite(distance, "distance")?;
        let sid = self.resolve_solid(solid)?;
        let result =
            brepkit_operations::offset_v2::offset_solid_v2(self.topo_mut(), sid, distance)?;
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
        let result = brepkit_operations::thicken::thicken(self.topo_mut(), face_id, thickness)?;
        Ok(solid_id_to_u32(result))
    }

    // ── Variable Fillet ───────────────────────────────────────────

    /// Apply variable-radius fillets to edges.
    ///
    /// `json` is a JSON string: `[{"edge": u32, "law": "constant"|"linear"|"scurve", "start": f64, "end": f64}]`
    ///
    /// Also accepts brepjs-style fields: `startRadius`/`endRadius` as aliases for `start`/`end`.
    /// When `law` is omitted and `startRadius` != `endRadius`, the law auto-detects as `"linear"`.
    ///
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
            // Accept both brepkit-native ("start"/"end") and brepjs ("startRadius"/"endRadius")
            let start_val = spec["start"]
                .as_f64()
                .or_else(|| spec["startRadius"].as_f64());
            let end_val = spec["end"].as_f64().or_else(|| spec["endRadius"].as_f64());

            // Auto-detect law: if no "law" field but start != end, use "linear"
            let law_str = spec["law"]
                .as_str()
                .unwrap_or_else(|| match (start_val, end_val) {
                    (Some(s), Some(e)) if (s - e).abs() > f64::EPSILON => "linear",
                    _ => "constant",
                });
            let law = match law_str {
                "linear" => {
                    let s = start_val.unwrap_or(1.0);
                    let e = end_val.unwrap_or(1.0);
                    brepkit_operations::fillet::FilletRadiusLaw::Linear { start: s, end: e }
                }
                "scurve" => {
                    let s = start_val.unwrap_or(1.0);
                    let e = end_val.unwrap_or(1.0);
                    brepkit_operations::fillet::FilletRadiusLaw::SCurve { start: s, end: e }
                }
                _ => {
                    let r = spec["radius"].as_f64().or(start_val).unwrap_or(1.0);
                    brepkit_operations::fillet::FilletRadiusLaw::Constant(r)
                }
            };
            edge_laws.push((edge_id, law));
        }
        let result =
            brepkit_operations::fillet::fillet_variable(self.topo_mut(), solid_id, &edge_laws)?;
        Ok(solid_id_to_u32(result))
    }

    /// Sweep a face along a NURBS path with advanced options.
    ///
    /// `contact_mode`: "rmf" (default), "fixed", or "constantNormal:x,y,z"
    /// `scale_values`: flat `[t0,s0,t1,s1,...]` pairs for piecewise-linear scale law.
    /// `corner_mode`: "smooth" (default), "miter", or "round"
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
        corner_mode: &str,
    ) -> Result<u32, JsError> {
        use brepkit_operations::sweep::{SweepContactMode, SweepCornerMode, SweepOptions};

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

        let cm = match corner_mode {
            "miter" => SweepCornerMode::Miter,
            "round" => SweepCornerMode::Round,
            _ => SweepCornerMode::Smooth,
        };

        let options = SweepOptions {
            contact_mode: mode,
            corner_mode: cm,
            scale_law,
            segments: segments as usize,
        };

        let result = brepkit_operations::sweep::sweep_with_options(
            self.topo_mut(),
            face_id,
            &path_curve,
            &options,
        )?;
        Ok(solid_id_to_u32(result))
    }

    // ── Point Classification ──────────────────────────────────────

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

    // ── Fill / Untrim / Offset Wire ───────────────────────────────

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
        let face_id = brepkit_operations::fill_face::fill_coons_patch(self.topo_mut(), &curves)?;
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
            brepkit_operations::offset_wire::offset_wire(self.topo_mut(), face_id, distance)?;
        Ok(wire_id_to_u32(wire_id))
    }

    /// Offset a wire on a planar face with a specific join type.
    ///
    /// `join_type` must be one of `"intersection"`, `"arc"`, or `"chamfer"`.
    /// Returns a new wire handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid, the join type string
    /// is unrecognized, or the offset operation fails.
    #[wasm_bindgen(js_name = "offsetWireWithJoinType")]
    pub fn offset_wire_with_join_type(
        &mut self,
        face: u32,
        distance: f64,
        join_type: &str,
    ) -> Result<u32, JsError> {
        let face_id = self.resolve_face(face)?;
        let jt = parse_join_type_str(join_type)?;
        let wire_id = brepkit_operations::offset_wire::offset_wire_with_join(
            self.topo_mut(),
            face_id,
            distance,
            jt,
        )?;
        Ok(wire_id_to_u32(wire_id))
    }

    // ── Orientation ───────────────────────────────────────────────

    /// Get the orientation of a shape.
    ///
    /// Returns `"forward"` for all faces (brepkit faces don't have an
    /// independent orientation flag; the normal direction is canonical).
    #[allow(clippy::unused_self)]
    #[must_use]
    #[wasm_bindgen(js_name = "getShapeOrientation")]
    pub fn get_shape_orientation(&self, _id: u32) -> String {
        // In brepkit, face normals are always canonical (outward-pointing).
        // There is no separate orientation flag.
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
            let new_fid = self.topo_mut().add_face(new_face);
            return Ok(face_id_to_u32(new_fid));
        }
        // Try as edge
        if let Ok(edge_id) = self.resolve_edge(id) {
            let edge = self.topo.edge(edge_id)?;
            let new_edge = Edge::new(edge.end(), edge.start(), edge.curve().clone());
            let new_eid = self.topo_mut().add_edge(new_edge);
            return Ok(edge_id_to_u32(new_eid));
        }
        Err(WasmError::InvalidInput {
            reason: "reverseShape requires a face or edge handle".into(),
        }
        .into())
    }

    // ── Blend V2 (walking engine) ────────────────────────────────

    /// Fillet edges using the v2 walking-based blend engine.
    ///
    /// Returns a new solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid or edge handles are invalid, or the
    /// blend computation fails.
    #[wasm_bindgen(js_name = "filletV2")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn fillet_v2(
        &mut self,
        solid: u32,
        edge_handles: Vec<u32>,
        radius: f64,
    ) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids: Vec<_> = edge_handles
            .iter()
            .map(|&h| self.resolve_edge(h))
            .collect::<Result<_, _>>()?;
        let result =
            brepkit_operations::blend_ops::fillet_v2(self.topo_mut(), solid_id, &edge_ids, radius)?;
        Ok(solid_id_to_u32(result.solid))
    }

    /// Chamfer edges with two distances using the v2 blend engine.
    ///
    /// Returns a new solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid or edge handles are invalid, or the
    /// blend computation fails.
    #[wasm_bindgen(js_name = "chamferV2")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn chamfer_v2(
        &mut self,
        solid: u32,
        edge_handles: Vec<u32>,
        d1: f64,
        d2: f64,
    ) -> Result<u32, JsError> {
        validate_positive(d1, "d1")?;
        validate_positive(d2, "d2")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids: Vec<_> = edge_handles
            .iter()
            .map(|&h| self.resolve_edge(h))
            .collect::<Result<_, _>>()?;
        let result = brepkit_operations::blend_ops::chamfer_v2(
            self.topo_mut(),
            solid_id,
            &edge_ids,
            d1,
            d2,
        )?;
        Ok(solid_id_to_u32(result.solid))
    }

    /// Chamfer edges with distance and angle using the v2 blend engine.
    ///
    /// Returns a new solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid or edge handles are invalid, or the
    /// blend computation fails.
    #[wasm_bindgen(js_name = "chamferDistanceAngle")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn chamfer_distance_angle(
        &mut self,
        solid: u32,
        edge_handles: Vec<u32>,
        distance: f64,
        angle: f64,
    ) -> Result<u32, JsError> {
        validate_positive(distance, "distance")?;
        validate_positive(angle, "angle")?;
        if angle >= std::f64::consts::FRAC_PI_2 {
            return Err(JsError::new("angle must be less than π/2"));
        }
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids: Vec<_> = edge_handles
            .iter()
            .map(|&h| self.resolve_edge(h))
            .collect::<Result<_, _>>()?;
        let result = brepkit_operations::blend_ops::chamfer_distance_angle(
            self.topo_mut(),
            solid_id,
            &edge_ids,
            distance,
            angle,
        )?;
        Ok(solid_id_to_u32(result.solid))
    }
}
