//! Modeling operation bindings (extrude, revolve, sweep, loft, fillet, etc.).

#![allow(
    clippy::missing_errors_doc,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use wasm_bindgen::prelude::*;

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::edge::Edge;
use brepkit_topology::face::{Face, FaceSurface};

use crate::error::{WasmError, validate_finite, validate_positive};
use crate::handles::{
    edge_id_to_u32, face_id_to_u32, shell_id_to_u32, solid_id_to_u32, wire_id_to_u32,
};
use brepkit_geometry::extrema::point_to_nurbs_surface;

use crate::helpers::{
    classify_to_string, loft_with_points, panic_message, parse_points, try_fillet,
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
    /// - `solid: bool` — true (default) returns a solid handle; false returns
    ///   a shell handle holding the loft's faces without the caps on its end
    ///   profiles
    #[wasm_bindgen(js_name = "loftWithOptions")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn loft_with_options(&mut self, faces: Vec<u32>, options: &str) -> Result<u32, JsError> {
        let opts: serde_json::Value =
            serde_json::from_str(options).unwrap_or(serde_json::Value::Null);

        let face_ids: Vec<brepkit_topology::face::FaceId> = faces
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let point = |key: &str| -> Option<Point3> {
            let p = opts.get(key)?.as_array()?;
            (p.len() >= 3).then(|| {
                Point3::new(
                    p[0].as_f64().unwrap_or(0.0),
                    p[1].as_f64().unwrap_or(0.0),
                    p[2].as_f64().unwrap_or(0.0),
                )
            })
        };
        let ruled = opts
            .get("ruled")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        let closed = opts
            .get("solid")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        let (start, end) = (point("startPoint"), point("endPoint"));
        let solid_id = loft_with_points(self.topo_mut(), &face_ids, start, end, ruled)?;
        if closed {
            return Ok(solid_id_to_u32(solid_id));
        }
        // A profile the loft carries on to a point is no end of the result.
        let ends: Vec<_> = [
            face_ids.first().filter(|_| start.is_none()),
            face_ids.last().filter(|_| end.is_none()),
        ]
        .into_iter()
        .flatten()
        .copied()
        .collect();
        let shell = brepkit_operations::loft::loft_sides(self.topo_mut(), solid_id, &ends)?;
        Ok(shell_id_to_u32(shell))
    }

    /// Loft two or more open wires of equal edge counts into an open shell.
    ///
    /// Returns a shell handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if a handle is invalid, fewer than two wires are given,
    /// their edge counts differ, a wire is closed, meets itself or breaks, two
    /// wires touch where they would be joined, or two lines cross or fold over.
    #[wasm_bindgen(js_name = "loftWires")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn loft_wires(&mut self, wires: Vec<u32>) -> Result<u32, JsError> {
        let wire_ids: Vec<brepkit_topology::wire::WireId> = wires
            .iter()
            .map(|&h| self.resolve_wire(h))
            .collect::<Result<_, _>>()?;
        let shell = brepkit_operations::sheet::loft_wires(self.topo_mut(), &wire_ids)?;
        Ok(shell_id_to_u32(shell))
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
        let result = crate::helpers::try_chamfer(self.topo_mut(), solid_id, &edge_ids, distance)?;
        Ok(solid_id_to_u32(result))
    }

    // ── Fillet ────────────────────────────────────────────────────

    /// Fillet (round) edges of a solid.
    ///
    /// `edge_handles` is an array of edge handles. Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if radius is non-positive, edges are invalid, or no
    /// fillet engine can produce a valid changed solid.
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
                let solid = match try_fillet(self.topo_mut(), solid_id, &edge_ids, radius) {
                    Ok(solid) => solid,
                    Err(error) => {
                        // Filter to edges where both adjacent faces are planar.
                        let planar_edges = brepkit_operations::query::filter_planar_edges(
                            &self.topo, solid_id, &edge_ids,
                        )?;
                        if planar_edges.is_empty() {
                            return Err(JsError::new(&error.to_string()));
                        }
                        try_fillet(self.topo_mut(), solid_id, &planar_edges, radius)
                            .map_err(|retry_error| JsError::new(&retry_error.to_string()))?
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

    /// Apply a constant-radius fillet and return face-evolution tracking data.
    ///
    /// Returns a JSON string `{"solid": <u32>, "evolution": {modified,
    /// generated, deleted}}` — the same shape as `fuseWithEvolution`. Blend
    /// faces appear under `generated` and surviving faces under `modified`.
    /// Provenance is matched geometrically (face normal + centroid), so it is
    /// unaffected by how the fillet renumbers faces.
    ///
    /// # Errors
    ///
    /// Returns an error if a handle is invalid, the radius is non-positive, or
    /// the fillet fails.
    #[wasm_bindgen(js_name = "filletWithEvolution")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn fillet_with_evolution(
        &mut self,
        solid: u32,
        edge_handles: Vec<u32>,
        radius: f64,
    ) -> Result<JsValue, JsError> {
        validate_positive(radius, "radius")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids: Vec<brepkit_topology::edge::EdgeId> = edge_handles
            .iter()
            .map(|&h| self.resolve_edge(h))
            .collect::<Result<_, _>>()?;

        // Wrap in catch_unwind like `fillet` does: a fillet panic must not
        // abort the whole WASM instance.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<String, JsError> {
                let input_faces =
                    brepkit_operations::boolean::collect_face_signatures(&self.topo, solid_id)?;
                let result = try_fillet(self.topo_mut(), solid_id, &edge_ids, radius)?;
                let output_faces =
                    brepkit_operations::boolean::collect_face_signatures(&self.topo, result)?;
                let evo = brepkit_operations::evolution::build_evolution_by_geometry(
                    &input_faces,
                    &output_faces,
                );
                Ok(format!(
                    "{{\"solid\":{},\"evolution\":{}}}",
                    solid_id_to_u32(result),
                    evo.to_json()
                ))
            },
        ));
        match result {
            Ok(inner) => inner.map(|json| JsValue::from_str(&json)),
            Err(panic_info) => Err(JsError::new(&panic_message(&panic_info, "Fillet"))),
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

    /// Extrude an open wire along a direction vector into an open shell.
    ///
    /// Returns a shell handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if an input is non-finite, the wire handle is invalid,
    /// the extrusion has no length, the wire is closed, meets itself or breaks,
    /// or a line runs along the extrusion.
    #[wasm_bindgen(js_name = "extrudeWire")]
    pub fn extrude_wire(
        &mut self,
        wire: u32,
        dir_x: f64,
        dir_y: f64,
        dir_z: f64,
        distance: f64,
    ) -> Result<u32, JsError> {
        validate_finite(dir_x, "dir_x")?;
        validate_finite(dir_y, "dir_y")?;
        validate_finite(dir_z, "dir_z")?;
        validate_finite(distance, "distance")?;

        let wire_id = self.resolve_wire(wire)?;
        let direction = Vec3::new(dir_x, dir_y, dir_z);
        let shell =
            brepkit_operations::sheet::extrude_wire(self.topo_mut(), wire_id, direction, distance)?;

        Ok(shell_id_to_u32(shell))
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

    /// Revolve an open wire around an axis into an open shell.
    ///
    /// The axis is defined by an origin point `(ox, oy, oz)` and a direction
    /// `(dx, dy, dz)`. The angle is in degrees and must be in (0, 360].
    ///
    /// Returns a shell handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if an input is non-finite, the wire handle is invalid,
    /// the axis has no length, the wire lies along the axis, an edge crosses the
    /// axis, or the wire is closed, meets itself or breaks.
    #[wasm_bindgen(js_name = "revolveWire")]
    #[allow(clippy::too_many_arguments)]
    pub fn revolve_wire(
        &mut self,
        wire: u32,
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

        let wire_id = self.resolve_wire(wire)?;
        let shell = brepkit_operations::sheet::revolve_wire(
            self.topo_mut(),
            wire_id,
            Point3::new(ox, oy, oz),
            Vec3::new(dx, dy, dz),
            angle_degrees.to_radians(),
        )?;

        Ok(shell_id_to_u32(shell))
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
        if !path_control_points.len().is_multiple_of(3) {
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

    /// Sweep through multiple section profiles along a spine, lofting the
    /// rotation-minimizing-frame-placed profiles.
    ///
    /// `face_handles` and `params` are parallel arrays: each planar profile and
    /// its parameter in `[0, 1]` along the spine (given as raw NURBS data).
    /// `ruled` selects ruled (planar bands) vs smooth (NURBS) lofted sides.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error for fewer than two sections, mismatched array lengths, a
    /// non-finite or out-of-range value, a non-planar profile, or loft failure.
    #[wasm_bindgen(js_name = "multiSectionSweep")]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    pub fn multi_section_sweep(
        &mut self,
        face_handles: Vec<u32>,
        params: Vec<f64>,
        spine_degree: u32,
        spine_knots: Vec<f64>,
        spine_control_points: Vec<f64>,
        spine_weights: Vec<f64>,
        ruled: bool,
    ) -> Result<u32, JsError> {
        if face_handles.len() != params.len() {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "face_handles ({}) and params ({}) must have equal length",
                    face_handles.len(),
                    params.len()
                ),
            }
            .into());
        }
        if !spine_control_points.len().is_multiple_of(3) {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "spine_control_points length must be a multiple of 3, got {}",
                    spine_control_points.len()
                ),
            }
            .into());
        }
        let num_pts = spine_control_points.len() / 3;
        if spine_weights.len() != num_pts {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "spine_weights length ({}) must match control point count ({num_pts})",
                    spine_weights.len()
                ),
            }
            .into());
        }
        for p in &params {
            validate_finite(*p, "param")?;
        }
        for (name, arr) in [
            ("spine_knots", &spine_knots),
            ("spine_control_points", &spine_control_points),
            ("spine_weights", &spine_weights),
        ] {
            if let Some(pos) = arr.iter().position(|v| !v.is_finite()) {
                return Err(WasmError::InvalidInput {
                    reason: format!("{name}[{pos}] is not finite"),
                }
                .into());
            }
        }

        let control_points: Vec<Point3> = spine_control_points
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();
        let spine = NurbsCurve::new(
            spine_degree as usize,
            spine_knots,
            control_points,
            spine_weights,
        )?;

        let sections: Vec<(brepkit_topology::face::FaceId, f64)> = face_handles
            .iter()
            .zip(params.iter())
            .map(|(&h, &p)| self.resolve_face(h).map(|f| (f, p)))
            .collect::<Result<_, _>>()?;

        let solid_id = brepkit_operations::sweep::multi_section_sweep(
            self.topo_mut(),
            &spine,
            &sections,
            ruled,
        )?;
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
        if !path_control_points.len().is_multiple_of(3) {
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

    /// A copy of a solid with its faces split where the given faces and edges
    /// meet them.
    ///
    /// Returns a solid handle. The copy has the solid's volume.
    ///
    /// # Errors
    ///
    /// Returns an error if a handle does not resolve, a face cannot be split
    /// along its marks, or the result would not be a valid solid of the same
    /// volume.
    #[wasm_bindgen(js_name = "imprintFaces")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn imprint_faces(
        &mut self,
        solid: u32,
        face_handles: Vec<u32>,
        edge_handles: Vec<u32>,
    ) -> Result<u32, JsError> {
        use brepkit_operations::imprint::ImprintTool;

        let solid_id = self.resolve_solid(solid)?;
        let mut tools = Vec::with_capacity(face_handles.len() + edge_handles.len());
        for &handle in &face_handles {
            tools.push(ImprintTool::Face(self.resolve_face(handle)?));
        }
        for &handle in &edge_handles {
            tools.push(ImprintTool::Edge(self.resolve_edge(handle)?));
        }
        let result = brepkit_operations::imprint::imprint_faces(self.topo_mut(), solid_id, &tools)?;
        Ok(solid_id_to_u32(result))
    }

    /// A solid in the pieces the given faces separate it into. A face that
    /// walls nothing off imprints instead, so a face that only dips in returns
    /// one solid.
    ///
    /// Returns solid handles.
    ///
    /// # Errors
    ///
    /// Returns an error if a handle does not resolve, a face cannot be split
    /// along its marks, or the pieces would not be valid solids summing to the
    /// solid's volume.
    #[wasm_bindgen(js_name = "splitByFaces")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn split_by_faces(
        &mut self,
        solid: u32,
        face_handles: Vec<u32>,
    ) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let faces: Vec<_> = face_handles
            .iter()
            .map(|&handle| self.resolve_face(handle))
            .collect::<Result<_, _>>()?;
        let pieces =
            brepkit_operations::imprint::split_by_faces(self.topo_mut(), solid_id, &faces)?;
        Ok(pieces.into_iter().map(solid_id_to_u32).collect())
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
        if !path_control_points.len().is_multiple_of(3) {
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
    /// A closed planar chain of lines and tangent arcs with an all-line
    /// perpendicular profile is swept analytically (exact plane / cylinder /
    /// cone faces); anything else falls back to fitting an interpolating
    /// NURBS curve through sampled chain points and sweeping along that.
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
        let edge_ids: Vec<brepkit_topology::edge::EdgeId> = edge_handles
            .iter()
            .map(|&eh| self.resolve_edge(eh))
            .collect::<Result<_, _>>()?;
        let face_id = self.resolve_face(face)?;
        let solid_id =
            brepkit_operations::sweep::sweep_along_edges(self.topo_mut(), face_id, &edge_ids)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Sweep an open wire along a chain of path edges into an open shell.
    ///
    /// A straight path extrudes the wire and one circle or arc revolves it;
    /// any other open, kink-free chain sweeps a NURBS face per wire edge.
    ///
    /// Returns a shell handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if a handle is invalid, the chain has no edges, breaks,
    /// closes or has a kink, or the wire is closed.
    #[wasm_bindgen(js_name = "sweepWire")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn sweep_wire(&mut self, wire: u32, path_edges: Vec<u32>) -> Result<u32, JsError> {
        let wire_id = self.resolve_wire(wire)?;
        let edge_ids: Vec<brepkit_topology::edge::EdgeId> = path_edges
            .iter()
            .map(|&h| self.resolve_edge(h))
            .collect::<Result<_, _>>()?;
        let shell = brepkit_operations::sheet::sweep_wire(self.topo_mut(), wire_id, &edge_ids)?;
        Ok(shell_id_to_u32(shell))
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
            brepkit_operations::offset_v2::offset_solid_v2(self.topo_mut(), solid_id, distance)?;
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
        let result = crate::helpers::try_fillet_variable(self.topo_mut(), solid_id, &edge_laws)?;
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
            if scale_values.len() >= 4 && scale_values.len().is_multiple_of(2) {
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
            aux_spine: None,
        };

        let result = brepkit_operations::sweep::sweep_with_options(
            self.topo_mut(),
            face_id,
            &path_curve,
            &options,
        )?;
        Ok(solid_id_to_u32(result))
    }

    /// Guided (two-rail) sweep: sweep `face` along a spine, orienting the
    /// profile so its up-vector tracks an auxiliary spine.
    ///
    /// The spine and auxiliary spine are each passed as raw NURBS data
    /// (`degree`, `knots`, flat `control_points`, `weights`). Returns a solid
    /// handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error for a non-finite or malformed curve, a non-planar
    /// profile, or a degenerate path.
    #[wasm_bindgen(js_name = "guidedSweep")]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    pub fn guided_sweep(
        &mut self,
        face: u32,
        spine_degree: u32,
        spine_knots: Vec<f64>,
        spine_control_points: Vec<f64>,
        spine_weights: Vec<f64>,
        aux_degree: u32,
        aux_knots: Vec<f64>,
        aux_control_points: Vec<f64>,
        aux_weights: Vec<f64>,
    ) -> Result<u32, JsError> {
        let build = |degree: u32,
                     knots: Vec<f64>,
                     cps: Vec<f64>,
                     weights: Vec<f64>,
                     label: &str|
         -> Result<NurbsCurve, JsError> {
            if degree < 1 {
                return Err(WasmError::InvalidInput {
                    reason: format!("{label}_degree must be at least 1"),
                }
                .into());
            }
            if !cps.len().is_multiple_of(3) {
                return Err(WasmError::InvalidInput {
                    reason: format!("{label}_control_points length must be a multiple of 3"),
                }
                .into());
            }
            if weights.len() != cps.len() / 3 {
                return Err(WasmError::InvalidInput {
                    reason: format!("{label}_weights length must match control point count"),
                }
                .into());
            }
            for (name, arr) in [
                ("knots", &knots),
                ("control_points", &cps),
                ("weights", &weights),
            ] {
                if let Some(pos) = arr.iter().position(|v| !v.is_finite()) {
                    return Err(WasmError::InvalidInput {
                        reason: format!("{label}_{name}[{pos}] is not finite"),
                    }
                    .into());
                }
            }
            let control_points: Vec<Point3> = cps
                .chunks_exact(3)
                .map(|c| Point3::new(c[0], c[1], c[2]))
                .collect();
            Ok(NurbsCurve::new(
                degree as usize,
                knots,
                control_points,
                weights,
            )?)
        };

        let spine = build(
            spine_degree,
            spine_knots,
            spine_control_points,
            spine_weights,
            "spine",
        )?;
        let aux = build(
            aux_degree,
            aux_knots,
            aux_control_points,
            aux_weights,
            "aux",
        )?;
        let face_id = self.resolve_face(face)?;
        let solid = brepkit_operations::sweep::sweep_guided(self.topo_mut(), face_id, &spine, aux)?;
        Ok(solid_id_to_u32(solid))
    }

    /// Convex Minkowski sum of two solids (`A ⊕ B`).
    ///
    /// Returns the convex hull of all pairwise vertex sums — exact for convex
    /// polytopes (boxes, or a tessellated-sphere rolling tool), a convex
    /// over-approximation otherwise. Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if either handle is invalid, either solid is empty, or
    /// the summed points are degenerate so no hull can be built.
    #[wasm_bindgen(js_name = "minkowskiSum")]
    pub fn minkowski_sum(&mut self, solid_a: u32, solid_b: u32) -> Result<u32, JsError> {
        let a = self.resolve_solid(solid_a)?;
        let b = self.resolve_solid(solid_b)?;
        let result = brepkit_operations::primitives::make_minkowski_sum(self.topo_mut(), a, b)?;
        Ok(solid_id_to_u32(result))
    }

    /// Project a solid's edges onto a view plane with hidden-line removal.
    ///
    /// Viewed along `dir` (orthographic) through `origin`, with in-plane x-axis
    /// `x_axis`. Returns a JSON string `{"visible": [[x,y,…]], "hidden": [[…]]}`
    /// — flat 2D polylines in view coordinates. `hidden_lines = false` drops the
    /// hidden set. Occlusion is an exact point-in-solid test.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid handle, a non-positive `deflection`, or a
    /// degenerate `dir`/`x_axis`.
    #[wasm_bindgen(js_name = "projectEdges")]
    #[allow(clippy::too_many_arguments)]
    pub fn project_edges(
        &self,
        solid: u32,
        origin_x: f64,
        origin_y: f64,
        origin_z: f64,
        dir_x: f64,
        dir_y: f64,
        dir_z: f64,
        x_axis_x: f64,
        x_axis_y: f64,
        x_axis_z: f64,
        hidden_lines: bool,
        deflection: f64,
    ) -> Result<JsValue, JsError> {
        validate_positive(deflection, "deflection")?;
        for (v, name) in [
            (origin_x, "origin_x"),
            (origin_y, "origin_y"),
            (origin_z, "origin_z"),
            (dir_x, "dir_x"),
            (dir_y, "dir_y"),
            (dir_z, "dir_z"),
            (x_axis_x, "x_axis_x"),
            (x_axis_y, "x_axis_y"),
            (x_axis_z, "x_axis_z"),
        ] {
            validate_finite(v, name)?;
        }
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::projection::project_edges(
            &self.topo,
            solid_id,
            Point3::new(origin_x, origin_y, origin_z),
            Vec3::new(dir_x, dir_y, dir_z),
            Vec3::new(x_axis_x, x_axis_y, x_axis_z),
            hidden_lines,
            deflection,
        )?;
        let flatten = |polys: &[Vec<brepkit_math::vec::Point2>]| -> Vec<Vec<f64>> {
            polys
                .iter()
                .map(|poly| poly.iter().flat_map(|p| [p.x(), p.y()]).collect())
                .collect()
        };
        let json = serde_json::json!({
            "visible": flatten(&result.visible),
            "hidden": flatten(&result.hidden),
        });
        Ok(JsValue::from_str(&json.to_string()))
    }

    /// Project several solids' edges together, as `projectEdges` does one
    /// solid's: each edge is hidden where any of the solids lies in front of
    /// it. Returns the same JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty list, an invalid handle, a non-positive
    /// `deflection`, or a degenerate `dir`/`x_axis`.
    #[wasm_bindgen(js_name = "projectSolidsEdges")]
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub fn project_solids_edges(
        &self,
        solids: Vec<u32>,
        origin_x: f64,
        origin_y: f64,
        origin_z: f64,
        dir_x: f64,
        dir_y: f64,
        dir_z: f64,
        x_axis_x: f64,
        x_axis_y: f64,
        x_axis_z: f64,
        hidden_lines: bool,
        deflection: f64,
    ) -> Result<JsValue, JsError> {
        validate_positive(deflection, "deflection")?;
        for (v, name) in [
            (origin_x, "origin_x"),
            (origin_y, "origin_y"),
            (origin_z, "origin_z"),
            (dir_x, "dir_x"),
            (dir_y, "dir_y"),
            (dir_z, "dir_z"),
            (x_axis_x, "x_axis_x"),
            (x_axis_y, "x_axis_y"),
            (x_axis_z, "x_axis_z"),
        ] {
            validate_finite(v, name)?;
        }
        let solid_ids = solids
            .iter()
            .map(|&h| self.resolve_solid(h))
            .collect::<Result<Vec<_>, _>>()?;
        let result = brepkit_operations::projection::project_solids_edges(
            &self.topo,
            &solid_ids,
            Point3::new(origin_x, origin_y, origin_z),
            Vec3::new(dir_x, dir_y, dir_z),
            Vec3::new(x_axis_x, x_axis_y, x_axis_z),
            hidden_lines,
            deflection,
        )?;
        let flatten = |polys: &[Vec<brepkit_math::vec::Point2>]| -> Vec<Vec<f64>> {
            polys
                .iter()
                .map(|poly| poly.iter().flat_map(|p| [p.x(), p.y()]).collect())
                .collect()
        };
        let json = serde_json::json!({
            "visible": flatten(&result.visible),
            "hidden": flatten(&result.hidden),
        });
        Ok(JsValue::from_str(&json.to_string()))
    }

    // ── Point Classification ──────────────────────────────────────

    /// Classify a point relative to a solid by the same ray cast as
    /// `classifyPoint`.
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

    /// Classify a point relative to a solid by the same ray cast as
    /// `classifyPoint`.
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

    /// Bridge two edge chains with a surface G1 or G2 continuous to the face
    /// beside each.
    ///
    /// Each chain lists its edges in running order, each stored either way.
    /// `options` is JSON: `{ "continuity": "g1" | "g2", "weights"?: [a, b] }`.
    /// Returns a face handle.
    ///
    /// # Errors
    ///
    /// Returns an error if a handle or the options are invalid, a weight is not
    /// positive, a chain breaks or leaves its face, or one chain is closed and
    /// the other open.
    #[wasm_bindgen(js_name = "blendSurface")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn blend_surface(
        &mut self,
        chain_a: Vec<u32>,
        face_a: u32,
        chain_b: Vec<u32>,
        face_b: u32,
        options: &str,
    ) -> Result<u32, JsError> {
        let opts: serde_json::Value =
            serde_json::from_str(options).map_err(|e| WasmError::InvalidInput {
                reason: format!("blend options are not JSON: {e}"),
            })?;
        let options = crate::helpers::blend_surface_options(&opts)
            .map_err(|reason| WasmError::InvalidInput { reason })?;
        let edges_a = chain_a
            .iter()
            .map(|&h| self.resolve_edge(h))
            .collect::<Result<Vec<_>, _>>()?;
        let edges_b = chain_b
            .iter()
            .map(|&h| self.resolve_edge(h))
            .collect::<Result<Vec<_>, _>>()?;
        let (face_a, face_b) = (self.resolve_face(face_a)?, self.resolve_face(face_b)?);
        let face = brepkit_operations::blend_surface::blend_surface(
            self.topo_mut(),
            &edges_a,
            face_a,
            &edges_b,
            face_b,
            &options,
        )?;
        Ok(face_id_to_u32(face))
    }

    /// Move one surface's edge on to another's at G0, G1 or G2, keeping the
    /// rest of the surface as close as the move allows.
    ///
    /// `options` is JSON: `{ "continuity": "g0" | "g1" | "g2", "targetFace"?: face }`.
    /// `targetFace` names the face a solid's edge is matched to, since it
    /// bounds two. Returns a new face handle.
    ///
    /// # Errors
    ///
    /// Returns an error if a handle or the options are invalid, `edge` is not on
    /// the face's outer boundary, or the target bounds no face or several when
    /// a surface is needed and none is named.
    #[wasm_bindgen(js_name = "matchSurface")]
    pub fn match_surface(
        &mut self,
        face: u32,
        edge: u32,
        target_edge: u32,
        options: &str,
    ) -> Result<u32, JsError> {
        let opts: serde_json::Value =
            serde_json::from_str(options).map_err(|e| WasmError::InvalidInput {
                reason: format!("match options are not JSON: {e}"),
            })?;
        let continuity = crate::helpers::match_continuity(&opts)
            .map_err(|reason| WasmError::InvalidInput { reason })?;
        let target_face = match &opts["targetFace"] {
            serde_json::Value::Null => None,
            value => {
                let handle = value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| WasmError::InvalidInput {
                        reason: "targetFace is not a face handle".into(),
                    })?;
                Some(self.resolve_face(handle)?)
            }
        };
        let face = self.resolve_face(face)?;
        let edge = self.resolve_edge(edge)?;
        let target = self.resolve_edge(target_edge)?;
        let result = match target_face {
            Some(on) => brepkit_operations::blend_surface::match_surface_on(
                self.topo_mut(),
                face,
                edge,
                target,
                on,
                continuity,
            )?,
            None => brepkit_operations::blend_surface::match_surface(
                self.topo_mut(),
                face,
                edge,
                target,
                continuity,
            )?,
        };
        Ok(face_id_to_u32(result))
    }

    /// Smooth a surface, staying within `tolerance` of it, with its boundary
    /// held. A surface with no bumps to remove comes back unchanged.
    ///
    /// Returns a new face handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the handle is invalid or `tolerance` is not positive.
    #[wasm_bindgen(js_name = "fairSurface")]
    pub fn fair_surface(&mut self, face: u32, tolerance: f64) -> Result<u32, JsError> {
        let face = self.resolve_face(face)?;
        let result =
            brepkit_operations::blend_surface::fair_surface(self.topo_mut(), face, tolerance)?;
        Ok(face_id_to_u32(result))
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
            let proj_start = point_to_nurbs_surface(v_start.point(), &surface);
            let uv_start = brepkit_math::vec::Point2::new(proj_start.u, proj_start.v);
            let proj_end = point_to_nurbs_surface(v_end.point(), &surface);
            let uv_end = brepkit_math::vec::Point2::new(proj_end.u, proj_end.v);
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

    /// Offset a planar wire directly by a distance with a specific join type.
    ///
    /// Builds a planar face from the wire internally, then offsets it with
    /// the requested corner join. This is the wire-based counterpart to
    /// [`offset_wire_with_join_type`](Self::offset_wire_with_join_type),
    /// which requires a face handle. Consumers that only hold a wire (such
    /// as 2D sketch offsets) can route a join type through this entry point
    /// without first constructing a face.
    ///
    /// `join_type` must be one of `"intersection"`, `"arc"`, or `"chamfer"`.
    /// Returns a new wire handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the wire handle is invalid, the wire is not
    /// planar, the join type string is unrecognized, or the offset
    /// operation fails.
    #[wasm_bindgen(js_name = "offsetWire2DWithJoin")]
    pub fn offset_wire_2d_with_join(
        &mut self,
        wire: u32,
        distance: f64,
        join_type: &str,
    ) -> Result<u32, JsError> {
        let wire_id = self.resolve_wire(wire)?;
        let jt = parse_join_type_str(join_type)?;
        let face_id =
            brepkit_topology::builder::make_planar_face_from_wire(self.topo_mut(), wire_id)?;
        let result = brepkit_operations::offset_wire::offset_wire_with_join(
            self.topo_mut(),
            face_id,
            distance,
            jt,
        )?;
        Ok(wire_id_to_u32(result))
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
    /// Returns an error if the solid or edge handles are invalid, the blend
    /// computation fails, or any edge could not be blended.
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
        // Wrap in catch_unwind like `fillet` does: a blend panic must not cross
        // the WASM FFI boundary, which would abort the entire WASM instance.
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<u32, JsError> {
                let result = brepkit_operations::blend_ops::fillet_v2(
                    self.topo_mut(),
                    solid_id,
                    &edge_ids,
                    radius,
                )?;
                Ok(solid_id_to_u32(brepkit_operations::blend_ops::all_blended(
                    result,
                )?))
            }));
        match outcome {
            Ok(inner) => inner,
            Err(panic_info) => Err(JsError::new(&panic_message(&panic_info, "FilletV2"))),
        }
    }

    /// Chamfer edges with two distances using the v2 blend engine.
    ///
    /// Returns a new solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid or edge handles are invalid, the blend
    /// computation fails, or any edge could not be blended.
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
        // Wrap in catch_unwind like `fillet` does: a blend panic must not cross
        // the WASM FFI boundary, which would abort the entire WASM instance.
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<u32, JsError> {
                let result = brepkit_operations::blend_ops::chamfer_v2(
                    self.topo_mut(),
                    solid_id,
                    &edge_ids,
                    d1,
                    d2,
                )?;
                Ok(solid_id_to_u32(brepkit_operations::blend_ops::all_blended(
                    result,
                )?))
            }));
        match outcome {
            Ok(inner) => inner,
            Err(panic_info) => Err(JsError::new(&panic_message(&panic_info, "ChamferV2"))),
        }
    }

    /// Chamfer edges with distance and angle using the v2 blend engine.
    ///
    /// Returns a new solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid or edge handles are invalid, the blend
    /// computation fails, or any edge could not be blended.
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
        Ok(solid_id_to_u32(brepkit_operations::blend_ops::all_blended(
            result,
        )?))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    /// A line 10 long then a half circle of radius 5, as a wire.
    fn line_and_half_circle(k: &mut crate::kernel::BrepKernel) -> u32 {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::vec::{Point3, Vec3};
        use brepkit_topology::edge::{Edge, EdgeCurve};

        let line = k.make_line_edge(0.0, 0.0, 0.0, 10.0, 0.0, 0.0).unwrap();
        let circle =
            Circle3D::new(Point3::new(15.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), 5.0).unwrap();
        let topo = k.topo_mut();
        let start = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            Point3::new(10.0, 0.0, 0.0),
            1e-7,
        ));
        let end = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            Point3::new(20.0, 0.0, 0.0),
            1e-7,
        ));
        let arc = topo.add_edge(Edge::new(start, end, EdgeCurve::Circle(circle)));
        let arc = crate::handles::edge_id_to_u32(arc);
        k.make_wire(vec![line, arc], false).unwrap()
    }

    fn shell_area(k: &crate::kernel::BrepKernel, shell: u32) -> f64 {
        let shell = k.resolve_shell(shell).unwrap();
        k.topo
            .shell(shell)
            .unwrap()
            .faces()
            .iter()
            .map(|&f| brepkit_operations::measure::face_area(&k.topo, f, 1e-3).unwrap())
            .sum()
    }

    /// The bindings return shells of the areas their closed forms give: the
    /// wire's length times the distance, the sphere its arc sweeps about the
    /// line it starts along, and the rectangle between two lines.
    #[test]
    fn open_wire_bindings_return_shells_of_the_closed_form_areas() {
        use std::f64::consts::PI;

        let length = 10.0 + 5.0 * PI;
        let mut k = crate::kernel::BrepKernel::new();
        let wire = line_and_half_circle(&mut k);
        let shell = k.extrude_wire(wire, 0.0, 0.0, 1.0, 20.0).unwrap();
        assert!((shell_area(&k, shell) - 20.0 * length).abs() < 1e-9);

        let shell = k
            .revolve_wire(wire, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 360.0)
            .unwrap();
        assert!((shell_area(&k, shell) - 100.0 * PI).abs() < 1e-6);

        let path = k.make_line_edge(0.0, 0.0, 0.0, 0.0, 0.0, 30.0).unwrap();
        let shell = k.sweep_wire(wire, vec![path]).unwrap();
        assert!((shell_area(&k, shell) - 30.0 * length).abs() < 1e-9);

        let low = k.make_line_edge(0.0, 0.0, 0.0, 10.0, 0.0, 0.0).unwrap();
        let high = k.make_line_edge(0.0, 5.0, 10.0, 10.0, 5.0, 10.0).unwrap();
        let (low, high) = (
            k.make_wire(vec![low], false).unwrap(),
            k.make_wire(vec![high], false).unwrap(),
        );
        let shell = k.loft_wires(vec![low, high]).unwrap();
        assert!((shell_area(&k, shell) - 10.0 * 125f64.sqrt()).abs() < 1e-9);
    }

    /// A closed wire is the solid operations' to take: every open-wire op in
    /// the batch dispatcher refuses it, and takes the same wire left open.
    #[test]
    fn open_wire_batch_ops_refuse_a_closed_wire() {
        let mut k = crate::kernel::BrepKernel::new();
        let a = k.make_line_edge(0.0, 0.0, 0.0, 4.0, 0.0, 0.0).unwrap();
        let b = k.make_line_edge(4.0, 0.0, 0.0, 4.0, 4.0, 0.0).unwrap();
        let c = k.make_line_edge(4.0, 4.0, 0.0, 0.0, 0.0, 0.0).unwrap();
        let closed = k.make_wire(vec![a, b, c], true).unwrap();
        let path = k.make_line_edge(0.0, 0.0, 0.0, 0.0, 0.0, 9.0).unwrap();
        let far = k.make_line_edge(0.0, 0.0, 5.0, 4.0, 0.0, 5.0).unwrap();
        let far = k.make_wire(vec![far], false).unwrap();
        let open = k.make_line_edge(0.0, 0.0, 0.0, 4.0, 0.0, 0.0).unwrap();
        let open = k.make_wire(vec![open], false).unwrap();
        let batch = |wire: u32| {
            format!(
                r#"[
                    {{"op": "extrudeWire", "args": {{"wire": {wire}, "dz": 1, "distance": 5}}}},
                    {{"op": "revolveWire", "args": {{"wire": {wire}, "originY": 7, "axisX": 1, "axisZ": 0, "angle": 90}}}},
                    {{"op": "sweepWire", "args": {{"wire": {wire}, "pathEdges": [{path}]}}}},
                    {{"op": "loftWires", "args": {{"wires": [{wire}, {far}]}}}}
                ]"#
            )
        };
        let refused: serde_json::Value =
            serde_json::from_str(&k.execute_batch(&batch(closed))).unwrap();
        for result in refused.as_array().unwrap() {
            let reason = result["error"].as_str().unwrap_or_default();
            assert!(reason.contains("closed"), "{result}");
        }
        let taken: serde_json::Value =
            serde_json::from_str(&k.execute_batch(&batch(open))).unwrap();
        for result in taken.as_array().unwrap() {
            let shell = u32::try_from(result["ok"].as_u64().expect("a shell")).unwrap();
            assert!(k.resolve_shell(shell).is_ok());
        }

        // A bad entry is refused, not dropped: a loft of what was left of the
        // list would be a different surface than was asked for.
        let bad = format!(
            r#"[
                {{"op": "loftWires", "args": {{"wires": [{open}, "bad", {far}]}}}},
                {{"op": "loftWires", "args": {{"wires": [{open}, 4294967296]}}}},
                {{"op": "loftWires", "args": {{"wires": 3}}}},
                {{"op": "sweepWire", "args": {{"wire": {open}, "pathEdges": [{path}, null]}}}},
                {{"op": "sweepWire", "args": {{"wire": {open}}}}}
            ]"#
        );
        let refused: serde_json::Value = serde_json::from_str(&k.execute_batch(&bad)).unwrap();
        for result in refused.as_array().unwrap() {
            let reason = result["error"].as_str().unwrap_or_default();
            assert!(
                reason.contains("not a handle") || reason.contains("array"),
                "{result}"
            );
        }
    }

    /// A dip, a face lying on the top and an edge along it mark a 20 x 20 x 10
    /// box through the batch dispatcher, and a face across it cuts it in two; a
    /// bad handle anywhere is refused.
    #[test]
    fn imprint_and_split_by_faces_through_the_batch() {
        let mut k = crate::kernel::BrepKernel::new();
        let body = k.make_box_solid(20.0, 20.0, 10.0).unwrap();
        let dip = k
            .make_polygon(vec![
                -5.0, 10.0, 9.0, 25.0, 10.0, 9.0, 25.0, 10.0, 11.0, -5.0, 10.0, 11.0,
            ])
            .unwrap();
        let lying = k
            .make_polygon(vec![
                -5.0, 10.0, 10.0, 25.0, 10.0, 10.0, 25.0, 25.0, 10.0, -5.0, 25.0, 10.0,
            ])
            .unwrap();
        let across = k
            .make_polygon(vec![
                -5.0, 10.0, -5.0, 25.0, 10.0, -5.0, 25.0, 10.0, 15.0, -5.0, 10.0, 15.0,
            ])
            .unwrap();
        let line = k.make_line_edge(0.0, 10.0, 10.0, 20.0, 10.0, 10.0).unwrap();

        let batch = format!(
            r#"[
                {{"op": "imprintFaces", "args": {{"solid": {body}, "faces": [{dip}]}}}},
                {{"op": "imprintFaces", "args": {{"solid": {body}, "faces": [{lying}]}}}},
                {{"op": "imprintFaces", "args": {{"solid": {body}, "edges": [{line}]}}}},
                {{"op": "imprintFaces", "args": {{"solid": {body}, "faces": [{across}], "edges": [{line}]}}}},
                {{"op": "splitByFaces", "args": {{"solid": {body}, "faces": [{dip}]}}}},
                {{"op": "splitByFaces", "args": {{"solid": {body}, "faces": [{across}]}}}}
            ]"#
        );
        let results: serde_json::Value = serde_json::from_str(&k.execute_batch(&batch)).unwrap();
        let results = results.as_array().unwrap();
        let handle = |value: &serde_json::Value| u32::try_from(value.as_u64().unwrap()).unwrap();
        for (result, faces) in results[..4].iter().zip([7, 7, 7, 10]) {
            let solid = handle(&result["ok"]);
            assert!(
                (k.volume(solid, 0.01).unwrap() - 2000.0 * 2.0).abs() < 1e-6,
                "{result}"
            );
            assert_eq!(k.get_solid_faces(solid).unwrap().len(), faces, "{result}");
        }
        let dipped = results[4]["ok"].as_array().expect("pieces");
        assert_eq!(dipped.len(), 1, "a dip separates nothing");
        assert!((k.volume(handle(&dipped[0]), 0.01).unwrap() - 4000.0).abs() < 1e-6);
        let halves = results[5]["ok"].as_array().expect("pieces");
        assert_eq!(halves.len(), 2, "a face across the box separates it");
        for half in halves {
            assert!((k.volume(handle(half), 0.01).unwrap() - 2000.0).abs() < 1e-6);
        }

        // The direct bindings agree.
        let marked = k.imprint_faces(body, vec![dip], vec![]).unwrap();
        assert_eq!(k.get_solid_faces(marked).unwrap().len(), 7);
        let marked = k.imprint_faces(body, vec![], vec![line]).unwrap();
        assert_eq!(k.get_solid_faces(marked).unwrap().len(), 7);
        assert_eq!(k.split_by_faces(body, vec![across]).unwrap().len(), 2);
        assert_eq!(k.split_by_faces(body, vec![dip]).unwrap().len(), 1);

        let bad = format!(
            r#"[
                {{"op": "imprintFaces", "args": {{"solid": 999999, "faces": [{dip}]}}}},
                {{"op": "imprintFaces", "args": {{"solid": {body}, "faces": [999999]}}}},
                {{"op": "imprintFaces", "args": {{"solid": {body}, "edges": [999999]}}}},
                {{"op": "imprintFaces", "args": {{"solid": {body}, "faces": [{dip}, "bad"]}}}},
                {{"op": "imprintFaces", "args": {{"solid": {body}, "edges": [{line}, null]}}}},
                {{"op": "imprintFaces", "args": {{"solid": {body}, "faces": 3}}}},
                {{"op": "splitByFaces", "args": {{"solid": 999999, "faces": [{dip}]}}}},
                {{"op": "splitByFaces", "args": {{"solid": {body}, "faces": [999999]}}}},
                {{"op": "splitByFaces", "args": {{"solid": {body}, "faces": [{dip}, 4294967296]}}}},
                {{"op": "splitByFaces", "args": {{"solid": {body}}}}}
            ]"#
        );
        let refused: serde_json::Value = serde_json::from_str(&k.execute_batch(&bad)).unwrap();
        for result in refused.as_array().unwrap() {
            assert!(result["error"].is_string(), "{result}");
        }
    }

    /// Drafting a box side by 5 degrees about its base through the batch
    /// dispatcher cuts the exact wedge `d h^2 tan(a) / 2`; an angle of 90
    /// degrees or more is refused.
    #[test]
    fn draft_batch_cuts_the_exact_wedge() {
        use brepkit_topology::face::FaceSurface;

        let mut k = crate::kernel::BrepKernel::new();
        let r = k
            .execute_batch(r#"[{"op": "makeBox", "args": {"width": 4, "height": 3, "depth": 2}}]"#);
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        let solid = u32::try_from(parsed[0]["ok"].as_u64().unwrap()).unwrap();
        let right = k
            .get_solid_faces(solid)
            .unwrap()
            .into_iter()
            .find(|&h| {
                let face = k.topo.face(k.resolve_face(h).unwrap()).unwrap();
                matches!(face.surface(), FaceSurface::Plane { normal, .. } if normal.x() > 0.5)
            })
            .unwrap();
        let angle = 5.0_f64.to_radians();
        let batch = format!(
            r#"[
                {{"op": "draft", "args": {{"solid": {solid}, "faces": [{right}], "angle": {angle}}}}},
                {{"op": "draft", "args": {{"solid": {solid}, "faces": [{right}], "angle": 1.6}}}}
            ]"#
        );
        let parsed: serde_json::Value = serde_json::from_str(&k.execute_batch(&batch)).unwrap();
        let drafted = u32::try_from(parsed[0]["ok"].as_u64().expect("draft result")).unwrap();
        assert!(parsed[1]["error"].is_string(), "{}", parsed[1]);
        let truth = 24.0 - 3.0 * 4.0 * angle.tan() / 2.0;
        // The public binding takes degrees and spells out the batch's
        // default pull and neutral point.
        let bound = k
            .draft_solid(solid, vec![right], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0)
            .unwrap();
        for handle in [drafted, bound] {
            let id = k.resolve_solid(handle).unwrap();
            let volume = brepkit_operations::measure::solid_volume(&k.topo, id, 0.001).unwrap();
            assert!(
                (volume - truth).abs() < 1e-9 * truth,
                "volume {volume}, expected {truth}"
            );
        }
    }

    /// A rod's top rim chamfered through the batch dispatcher and through
    /// the public binding removes the exact ring `pi d^2 (r - d/3)`.
    #[test]
    fn chamfer_batch_bevels_a_rod_rim_exactly() {
        use brepkit_topology::edge::EdgeCurve;

        let mut k = crate::kernel::BrepKernel::new();
        let r = k.execute_batch(
            r#"[
                {"op": "makeCylinder", "args": {"radius": 3, "height": 10}},
                {"op": "solidEdges", "args": {"solid": 0}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        let solid = u32::try_from(parsed[0]["ok"].as_u64().unwrap()).unwrap();
        let rim = parsed[1]["ok"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| u32::try_from(h.as_u64().unwrap()).unwrap())
            .find(|&h| {
                let edge = k.topo.edge(k.resolve_edge(h).unwrap()).unwrap();
                matches!(edge.curve(), EdgeCurve::Circle(_))
                    && k.topo.vertex(edge.start()).unwrap().point().z() > 9.0
            })
            .unwrap();
        let batch = format!(
            r#"[{{"op": "chamfer", "args": {{"solid": {solid}, "edges": [{rim}], "distance": 1.0}}}}]"#
        );
        let parsed: serde_json::Value = serde_json::from_str(&k.execute_batch(&batch)).unwrap();
        let chamfered = u32::try_from(parsed[0]["ok"].as_u64().expect("chamfer result")).unwrap();
        let bound = k.chamfer_solid(solid, vec![rim], 1.0).unwrap();
        let truth = std::f64::consts::PI * (90.0 - (3.0 - 1.0 / 3.0));
        for handle in [chamfered, bound] {
            let id = k.resolve_solid(handle).unwrap();
            let volume = brepkit_operations::measure::solid_volume(&k.topo, id, 0.001).unwrap();
            assert!(
                (volume - truth).abs() < 1e-9 * truth,
                "volume {volume}, expected {truth}"
            );
        }
    }

    use brepkit_math::vec::Point3;
    use brepkit_topology::builder::make_polygon_wire;

    use crate::handles::{solid_id_to_u32, wire_id_to_u32};
    use crate::helpers::TOL;
    use crate::kernel::BrepKernel;

    fn square_wire(k: &mut BrepKernel) -> u32 {
        let pts = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        ];
        let wid = make_polygon_wire(k.topo_mut(), &pts, TOL).unwrap();
        wire_id_to_u32(wid)
    }

    fn dispatch(k: &mut BrepKernel, op: &str, args: serde_json::Value) -> serde_json::Value {
        let batch = serde_json::json!([{ "op": op, "args": args }]);
        let out = k.execute_batch(&batch.to_string());
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&out).unwrap();
        parsed[0].clone()
    }

    /// A flat square sheet at height `z`, its edges in order round the wire.
    fn flat_sheet(k: &mut BrepKernel, x: f64, z: f64) -> (u32, Vec<u32>) {
        let pts = [
            Point3::new(x, 0.0, z),
            Point3::new(x + 4.0, 0.0, z),
            Point3::new(x + 4.0, 10.0, z),
            Point3::new(x, 10.0, z),
        ];
        let wid = make_polygon_wire(k.topo_mut(), &pts, TOL).unwrap();
        let face = brepkit_topology::builder::make_face_from_wire(k.topo_mut(), wid).unwrap();
        let edges = k
            .topo
            .wire(wid)
            .unwrap()
            .edges()
            .iter()
            .map(|oe| crate::handles::edge_id_to_u32(oe.edge()))
            .collect();
        (crate::handles::face_id_to_u32(face), edges)
    }

    #[test]
    fn batch_blends_two_sheets() {
        let mut k = BrepKernel::new();
        let (face_a, edges_a) = flat_sheet(&mut k, -4.0, 0.0);
        let (face_b, edges_b) = flat_sheet(&mut k, 6.0, 2.0);
        for continuity in ["g1", "g2"] {
            let out = dispatch(
                &mut k,
                "blendSurface",
                serde_json::json!({
                    "chainA": [edges_a[1]],
                    "faceA": face_a,
                    "chainB": [edges_b[3]],
                    "faceB": face_b,
                    "continuity": continuity,
                    "weights": [1.0, 1.5],
                }),
            );
            assert!(out["ok"].is_u64(), "{out}");
            let id = k.resolve_face(out["ok"].as_u64().unwrap() as u32).unwrap();
            assert_eq!(k.topo.face(id).unwrap().surface().type_tag(), "nurbs");
        }
    }

    #[test]
    fn batch_refuses_a_bad_blend() {
        let mut k = BrepKernel::new();
        let (face_a, edges_a) = flat_sheet(&mut k, -4.0, 0.0);
        let (face_b, edges_b) = flat_sheet(&mut k, 6.0, 2.0);
        let args = |continuity: &str, weights: serde_json::Value| {
            serde_json::json!({
                "chainA": [edges_a[1]],
                "faceA": face_a,
                "chainB": [edges_b[3]],
                "faceB": face_b,
                "continuity": continuity,
                "weights": weights,
            })
        };
        for bad in [
            args("g3", serde_json::Value::Null),
            args("g1", serde_json::json!([0.0, 1.0])),
            args("g1", serde_json::json!([1.0])),
            args("g1", serde_json::json!("heavy")),
        ] {
            let out = dispatch(&mut k, "blendSurface", bad.clone());
            assert!(out["error"].is_string(), "{bad} gave {out}");
        }
        let out = dispatch(
            &mut k,
            "blendSurface",
            serde_json::json!({
                "chainA": [edges_b[3]],
                "faceA": face_a,
                "chainB": [edges_b[3]],
                "faceB": face_b,
                "continuity": "g1",
            }),
        );
        assert!(out["error"].is_string(), "{out}");
    }

    #[test]
    fn batch_matches_an_edge() {
        let mut k = BrepKernel::new();
        let (target_face, target_edges) = flat_sheet(&mut k, -4.0, 0.0);
        let (source_face, source_edges) = flat_sheet(&mut k, 0.3, 0.2);
        for continuity in ["g0", "g1", "g2"] {
            let out = dispatch(
                &mut k,
                "matchSurface",
                serde_json::json!({
                    "face": source_face,
                    "edge": source_edges[3],
                    "targetEdge": target_edges[1],
                    "continuity": continuity,
                }),
            );
            assert!(out["ok"].is_u64(), "{continuity}: {out}");
            let named = dispatch(
                &mut k,
                "matchSurface",
                serde_json::json!({
                    "face": source_face,
                    "edge": source_edges[3],
                    "targetEdge": target_edges[1],
                    "targetFace": target_face,
                    "continuity": continuity,
                }),
            );
            assert!(named["ok"].is_u64(), "{continuity}: {named}");
        }
        let out = dispatch(
            &mut k,
            "matchSurface",
            serde_json::json!({
                "face": source_face,
                "edge": target_edges[1],
                "targetEdge": target_edges[1],
                "continuity": "g9",
            }),
        );
        assert!(out["error"].is_string(), "{out}");
    }

    #[test]
    fn batch_fairs_a_surface() {
        let mut k = BrepKernel::new();
        let (face, _) = flat_sheet(&mut k, 0.0, 0.0);
        let out = dispatch(
            &mut k,
            "fairSurface",
            serde_json::json!({ "face": face, "tolerance": 0.01 }),
        );
        assert!(out["ok"].is_u64(), "{out}");
        assert_ne!(out["ok"].as_u64().unwrap(), u64::from(face));
        let bad = dispatch(
            &mut k,
            "fairSurface",
            serde_json::json!({ "face": face, "tolerance": 0.0 }),
        );
        assert!(bad["error"].is_string(), "{bad}");
        let missing = dispatch(&mut k, "fairSurface", serde_json::json!({ "face": face }));
        assert!(missing["error"].is_string(), "{missing}");
    }

    fn wire_perimeter(k: &BrepKernel, wire_handle: u32) -> f64 {
        let wid = k.resolve_wire(wire_handle).unwrap();
        brepkit_operations::measure::wire_length(&k.topo, wid).unwrap()
    }

    #[test]
    fn multi_section_sweep_lofts_circles_along_line() {
        let mut k = BrepKernel::new();
        let big = k.make_circle_face(10.0, 24).unwrap();
        let small = k.make_circle_face(5.0, 24).unwrap();
        // Degree-1 line spine from the origin to (0, 0, 50).
        let solid = k
            .multi_section_sweep(
                vec![big, small],
                vec![0.0, 1.0],
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
                vec![1.0, 1.0],
                true,
            )
            .unwrap();
        let vol = k.volume(solid, 0.5).unwrap();
        assert!(
            vol > 0.0,
            "tapered tube should have positive volume, got {vol}"
        );
        // (Validation paths — <2 sections, length mismatch, out-of-range param —
        // are covered by the operations-layer tests; the error path can't be
        // exercised here because JsError can't be constructed off-wasm.)
    }

    #[test]
    fn multi_section_sweep_batch_dispatch_lofts_circles() {
        let mut k = BrepKernel::new();
        let big = k.make_circle_face(10.0, 24).unwrap();
        let small = k.make_circle_face(5.0, 24).unwrap();
        // A degree-1 NURBS line spine edge (the batch op takes the spine by edge).
        let spine = k
            .make_nurbs_edge(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                50.0,
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 50.0],
                vec![1.0, 1.0],
            )
            .unwrap();
        let out = dispatch(
            &mut k,
            "multiSectionSweep",
            serde_json::json!({
                "faces": [big, small],
                "params": [0.0, 1.0],
                "spineEdge": spine,
                "ruled": true,
            }),
        );
        assert!(
            out.get("ok").and_then(serde_json::Value::as_u64).is_some(),
            "expected an ok solid handle, got {out}"
        );
    }

    /// A profile of a square of side 10 holed by a circle of radius 2,
    /// lofted 10 up as brepjs lofts it: each loop on its own with
    /// `"solid": false`, then sewn to the two holed caps. The square's loft
    /// is its four sides, the circle's its one wall, and the sewn solid holds
    /// (100 - 4 pi) 10.
    #[test]
    fn an_open_loft_of_each_loop_sews_to_holed_caps() {
        use brepkit_math::vec::{Point3, Vec3};
        use brepkit_topology::builder::{make_circle_edge_with_ref, make_polygon_wire};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut k = BrepKernel::new();
        let square = |k: &mut BrepKernel, z: f64| {
            let corners = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)]
                .map(|(x, y)| Point3::new(x, y, z));
            make_polygon_wire(k.topo_mut(), &corners, 1e-7).unwrap()
        };
        let circle = |k: &mut BrepKernel, z: f64, normal: f64| {
            // Seams at +x, where the caps' holes and the lofts' rims meet.
            let e = make_circle_edge_with_ref(
                k.topo_mut(),
                Point3::new(0.0, 0.0, z),
                Vec3::new(0.0, 0.0, normal),
                2.0,
                Vec3::new(1.0, 0.0, 0.0),
                1e-7,
            )
            .unwrap();
            k.topo_mut()
                .add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap())
        };
        let plane = |z: f64| FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z,
        };
        let face = |k: &mut BrepKernel, wire, holes: Vec<_>, z: f64| {
            let id = k.topo_mut().add_face(Face::new(wire, holes, plane(z)));
            crate::handles::face_id_to_u32(id)
        };
        let loops = [
            (square(&mut k, 0.0), square(&mut k, 10.0)),
            (circle(&mut k, 0.0, 1.0), circle(&mut k, 10.0, 1.0)),
        ];
        let mut sides = Vec::new();
        for (count, (lower, upper)) in [4, 1].into_iter().zip(loops) {
            let lower = face(&mut k, lower, vec![], 0.0);
            let upper = face(&mut k, upper, vec![], 10.0);
            let shell = k
                .loft_with_options(vec![lower, upper], r#"{"ruled":true,"solid":false}"#)
                .unwrap();
            let shell = k.resolve_shell(shell).unwrap();
            let faces = k.topo.shell(shell).unwrap().faces().to_vec();
            assert_eq!(faces.len(), count, "the loft's sides alone");
            sides.extend(faces.into_iter().map(crate::handles::face_id_to_u32));
        }
        for z in [0.0, 10.0] {
            let (outer, hole) = (square(&mut k, z), circle(&mut k, z, -1.0));
            sides.push(face(&mut k, outer, vec![hole], z));
        }
        let solid = k.sew_faces(sides, 1e-6).unwrap();
        let volume = k.volume(solid, 0.001).unwrap();
        let want = 4.0f64.mul_add(-std::f64::consts::PI, 100.0) * 10.0;
        assert!(
            (volume - want).abs() < 1e-6 * want,
            "volume {volume}, want {want}"
        );
    }

    #[test]
    fn guided_sweep_produces_solid() {
        let mut k = BrepKernel::new();
        let profile = k.make_circle_face(2.0, 24).unwrap();
        // Spine: line (0,0,0)→(0,0,20). Aux: parallel guide offset +10 in X.
        let solid = k
            .guided_sweep(
                profile,
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 20.0],
                vec![1.0, 1.0],
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![10.0, 0.0, 0.0, 10.0, 0.0, 20.0],
                vec![1.0, 1.0],
            )
            .unwrap();
        let vol = k.volume(solid, 0.5).unwrap();
        assert!(vol > 0.0, "guided sweep volume, got {vol}");
    }

    #[test]
    fn guided_sweep_batch_dispatch() {
        let mut k = BrepKernel::new();
        let profile = k.make_circle_face(2.0, 24).unwrap();
        let mk_line = |k: &mut BrepKernel, x: f64| {
            k.make_nurbs_edge(
                x,
                0.0,
                0.0,
                x,
                0.0,
                20.0,
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![x, 0.0, 0.0, x, 0.0, 20.0],
                vec![1.0, 1.0],
            )
            .unwrap()
        };
        let spine = mk_line(&mut k, 0.0);
        let aux = mk_line(&mut k, 10.0);
        let out = dispatch(
            &mut k,
            "guidedSweep",
            serde_json::json!({ "face": profile, "spineEdge": spine, "auxEdge": aux }),
        );
        assert!(
            out.get("ok").and_then(serde_json::Value::as_u64).is_some(),
            "expected an ok solid handle, got {out}"
        );
    }

    #[test]
    fn minkowski_sum_binding_box10_box2_is_box12() {
        let mut k = BrepKernel::new();
        let a = brepkit_operations::primitives::make_box(k.topo_mut(), 10.0, 10.0, 10.0).unwrap();
        let b = brepkit_operations::primitives::make_box(k.topo_mut(), 2.0, 2.0, 2.0).unwrap();
        let sum = k
            .minkowski_sum(solid_id_to_u32(a), solid_id_to_u32(b))
            .unwrap();
        let vol = k.volume(sum, 0.1).unwrap();
        assert!(
            (vol - 1728.0).abs() < 0.5,
            "expected ~1728 (12³), got {vol}"
        );
    }

    #[test]
    fn minkowski_sum_batch_dispatch() {
        let mut k = BrepKernel::new();
        let a = brepkit_operations::primitives::make_box(k.topo_mut(), 4.0, 4.0, 4.0).unwrap();
        let b = brepkit_operations::primitives::make_box(k.topo_mut(), 2.0, 2.0, 2.0).unwrap();
        let out = dispatch(
            &mut k,
            "minkowskiSum",
            serde_json::json!({ "solidA": solid_id_to_u32(a), "solidB": solid_id_to_u32(b) }),
        );
        assert!(
            out.get("ok").and_then(serde_json::Value::as_u64).is_some(),
            "expected an ok solid handle, got {out}"
        );
    }

    #[test]
    fn project_edges_batch_dispatch_box_oblique() {
        let mut k = BrepKernel::new();
        let solid =
            brepkit_operations::primitives::make_box(k.topo_mut(), 10.0, 10.0, 10.0).unwrap();
        let out = dispatch(
            &mut k,
            "projectEdges",
            serde_json::json!({
                "solid": solid_id_to_u32(solid),
                "originX": -100.0, "originY": -100.0, "originZ": -100.0,
                "dirX": 1.0, "dirY": 1.0, "dirZ": 1.0,
                "xAxisX": 1.0, "xAxisY": -1.0, "xAxisZ": 0.0,
                "hiddenLines": true, "deflection": 0.1,
            }),
        );
        let ok = out.get("ok").expect("projectEdges batch should return ok");
        let nonempty = |key: &str| {
            ok.get(key)
                .and_then(serde_json::Value::as_array)
                .is_some_and(|a| !a.is_empty())
        };
        assert!(nonempty("visible"), "visible polylines expected, got {out}");
        assert!(nonempty("hidden"), "hidden polylines expected, got {out}");
    }

    #[test]
    fn offset_wire_2d_with_join_routes_arc_distinct_from_chamfer() {
        let mut k = BrepKernel::new();
        let w_int = square_wire(&mut k);
        let w_arc = square_wire(&mut k);
        let w_chamfer = square_wire(&mut k);

        let intersection = dispatch(
            &mut k,
            "offsetWire2DWithJoin",
            serde_json::json!({"wire": w_int, "distance": 2.0, "joinType": "intersection"}),
        );
        let arc = dispatch(
            &mut k,
            "offsetWire2DWithJoin",
            serde_json::json!({"wire": w_arc, "distance": 2.0, "joinType": "arc"}),
        );
        let chamfer = dispatch(
            &mut k,
            "offsetWire2DWithJoin",
            serde_json::json!({"wire": w_chamfer, "distance": 2.0, "joinType": "chamfer"}),
        );

        for (label, entry) in [
            ("intersection", &intersection),
            ("arc", &arc),
            ("chamfer", &chamfer),
        ] {
            assert!(entry.get("error").is_none(), "{label} errored: {entry}");
        }

        let int_wire = intersection["ok"].as_u64().unwrap() as u32;
        let arc_wire = arc["ok"].as_u64().unwrap() as u32;
        let chamfer_wire = chamfer["ok"].as_u64().unwrap() as u32;

        let int_len = wire_perimeter(&k, int_wire);
        let arc_len = wire_perimeter(&k, arc_wire);
        let chamfer_len = wire_perimeter(&k, chamfer_wire);

        assert!(int_len > 0.0 && arc_len > 0.0 && chamfer_len > 0.0);
        // Arc joins round the four convex corners, so the perimeter must
        // differ from the sharp/chamfer join shapes — proving the join type
        // is actually threaded through rather than silently ignored.
        assert!(
            (arc_len - chamfer_len).abs() > 1.0,
            "arc ({arc_len}) should differ from chamfer ({chamfer_len})"
        );
    }

    #[test]
    fn offset_wire_2d_with_join_rejects_unknown_join_type() {
        let mut k = BrepKernel::new();
        let w = square_wire(&mut k);
        let entry = dispatch(
            &mut k,
            "offsetWire2DWithJoin",
            serde_json::json!({"wire": w, "distance": 2.0, "joinType": "bogus"}),
        );
        assert!(
            entry.get("error").is_some(),
            "unknown join type should error: {entry}"
        );
    }
}
