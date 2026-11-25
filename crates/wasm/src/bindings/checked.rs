//! The checked bodies that a JS binding and its batch arm share.
//!
//! Each body checks the numbers it is given before it builds anything, so
//! the batch cannot pass a size, distance or count that the binding refuses.

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::fillet::FilletRadiusLaw;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;

use crate::bindings::operations::parse_join_type_str;
use crate::bindings::transforms::{Reach, scaled_unit};
use crate::error::{
    WasmError, validate_finite, validate_point, validate_positive, validate_size,
    validate_tolerance, validate_vector,
};
use crate::handles::{shell_id_to_u32, solid_id_to_u32, wire_id_to_u32};
use crate::helpers::panic_message;
use crate::kernel::BrepKernel;

/// A refusal for a panic caught in `operation`.
fn caught(payload: &Box<dyn std::any::Any + Send>, operation: &str) -> WasmError {
    WasmError::InvalidInput {
        reason: panic_message(payload, operation),
    }
}

/// The angle of a revolve in degrees, checked as the bindings take it.
fn revolve_radians(angle_degrees: f64) -> Result<f64, WasmError> {
    validate_finite(angle_degrees, "angle_degrees")?;
    if angle_degrees.abs() > 0.0 && angle_degrees.abs() <= 360.0 {
        Ok(angle_degrees.to_radians())
    } else {
        Err(WasmError::InvalidInput {
            reason: format!("angle_degrees must be in [-360, 0) or (0, 360], got {angle_degrees}"),
        })
    }
}

/// Check a plane from JS: a plane face stores its distance from the origin,
/// which the load checks hold within the size limit, and a point within it
/// can lie farther, up to the square root of 3 times it.
fn check_plane(point: Point3, normal: Vec3, what: &str) -> Result<(), WasmError> {
    let unit = scaled_unit(normal)?;
    validate_size((point - Point3::new(0.0, 0.0, 0.0)).dot(unit), what)
}

impl BrepKernel {
    fn edge_ids(&self, handles: &[u32]) -> Result<Vec<EdgeId>, WasmError> {
        handles.iter().map(|&h| self.resolve_edge(h)).collect()
    }

    fn face_ids(&self, handles: &[u32]) -> Result<Vec<FaceId>, WasmError> {
        handles.iter().map(|&h| self.resolve_face(h)).collect()
    }

    /// Check a sweep of a profile by `offset`: the offset, and the profile
    /// at its far end, stay within the size limit.
    fn check_swept(reach: &Reach, offset: Vec3) -> Result<(), WasmError> {
        validate_vector(offset, "the sweep's offset")?;
        let along = Mat4::translation(offset.x(), offset.y(), offset.z());
        reach.check_moved(&along, "the swept profile")
    }

    // ── Measurement ────────────────────────────────────────────────

    pub(crate) fn volume_impl(&self, solid: u32, deflection: f64) -> Result<f64, WasmError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        Ok(brepkit_operations::measure::solid_volume(
            &self.topo, solid_id, deflection,
        )?)
    }

    pub(crate) fn surface_area_impl(&self, solid: u32, deflection: f64) -> Result<f64, WasmError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        Ok(brepkit_operations::measure::solid_surface_area(
            &self.topo, solid_id, deflection,
        )?)
    }

    pub(crate) fn center_of_mass_impl(
        &self,
        solid: u32,
        deflection: f64,
    ) -> Result<Point3, WasmError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        Ok(brepkit_operations::measure::solid_center_of_mass(
            &self.topo, solid_id, deflection,
        )?)
    }

    pub(crate) fn classify_point_impl(
        &self,
        solid: u32,
        point: Point3,
        tolerance: f64,
    ) -> Result<brepkit_operations::classify::PointClassification, WasmError> {
        validate_point(point, "the point")?;
        validate_tolerance(tolerance, "tolerance")?;
        let solid_id = self.resolve_solid(solid)?;
        Ok(brepkit_operations::classify::classify_point(
            &self.topo, solid_id, point, 0.1, tolerance,
        )?)
    }

    // ── Sweeps ─────────────────────────────────────────────────────

    pub(crate) fn extrude_impl(
        &mut self,
        face: u32,
        direction: Vec3,
        distance: f64,
    ) -> Result<u32, WasmError> {
        validate_vector(direction, "the extrusion's direction")?;
        validate_size(distance, "distance")?;
        let face_id = self.resolve_face(face)?;
        Self::check_swept(&Reach::of_face(&self.topo, face_id)?, direction * distance)?;
        let solid =
            brepkit_operations::extrude::extrude(self.topo_mut(), face_id, direction, distance)?;
        Ok(solid_id_to_u32(solid))
    }

    pub(crate) fn extrude_wire_impl(
        &mut self,
        wire: u32,
        direction: Vec3,
        distance: f64,
    ) -> Result<u32, WasmError> {
        validate_vector(direction, "the extrusion's direction")?;
        validate_size(distance, "distance")?;
        let wire_id = self.resolve_wire(wire)?;
        Self::check_swept(
            &Reach::of_wires(&self.topo, [wire_id])?,
            direction * distance,
        )?;
        let shell =
            brepkit_operations::sheet::extrude_wire(self.topo_mut(), wire_id, direction, distance)?;
        Ok(shell_id_to_u32(shell))
    }

    pub(crate) fn revolve_impl(
        &mut self,
        face: u32,
        origin: Point3,
        axis: Vec3,
        angle_degrees: f64,
    ) -> Result<u32, WasmError> {
        validate_point(origin, "the axis' origin")?;
        validate_vector(axis, "the axis' direction")?;
        let angle = revolve_radians(angle_degrees)?;
        let face_id = self.resolve_face(face)?;
        Reach::of_face(&self.topo, face_id)?.check_turned(origin, "the revolved solid")?;
        let solid =
            brepkit_operations::revolve::revolve(self.topo_mut(), face_id, origin, axis, angle)?;
        Ok(solid_id_to_u32(solid))
    }

    pub(crate) fn revolve_wire_impl(
        &mut self,
        wire: u32,
        origin: Point3,
        axis: Vec3,
        angle_degrees: f64,
    ) -> Result<u32, WasmError> {
        validate_point(origin, "the axis' origin")?;
        validate_vector(axis, "the axis' direction")?;
        let angle = revolve_radians(angle_degrees)?;
        let wire_id = self.resolve_wire(wire)?;
        Reach::of_wires(&self.topo, [wire_id])?.check_turned(origin, "the revolved shell")?;
        let shell =
            brepkit_operations::sheet::revolve_wire(self.topo_mut(), wire_id, origin, axis, angle)?;
        Ok(shell_id_to_u32(shell))
    }

    // ── Cuts by a plane ────────────────────────────────────────────

    pub(crate) fn section_impl(
        &mut self,
        solid: u32,
        point: Point3,
        normal: Vec3,
    ) -> Result<Vec<FaceId>, WasmError> {
        validate_point(point, "the plane's point")?;
        validate_vector(normal, "the plane's normal")?;
        check_plane(point, normal, "the plane's distance from the origin")?;
        let solid_id = self.resolve_solid(solid)?;
        let result =
            brepkit_operations::section::section(self.topo_mut(), solid_id, point, normal)?;
        Ok(result.faces)
    }

    pub(crate) fn split_impl(
        &mut self,
        solid: u32,
        point: Point3,
        normal: Vec3,
    ) -> Result<(u32, u32), WasmError> {
        validate_point(point, "the plane's point")?;
        validate_vector(normal, "the plane's normal")?;
        check_plane(point, normal, "the plane's distance from the origin")?;
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::split::split(self.topo_mut(), solid_id, point, normal)?;
        Ok((
            solid_id_to_u32(result.positive),
            solid_id_to_u32(result.negative),
        ))
    }

    /// The draft of `faces`, by `angle` in radians.
    pub(crate) fn draft_impl(
        &mut self,
        solid: u32,
        faces: &[u32],
        pull: Vec3,
        neutral: Point3,
        angle: f64,
    ) -> Result<u32, WasmError> {
        validate_vector(pull, "the pull direction")?;
        validate_point(neutral, "the neutral plane's point")?;
        check_plane(
            neutral,
            pull,
            "the neutral plane's distance from the origin",
        )?;
        validate_finite(angle, "angle")?;
        let solid_id = self.resolve_solid(solid)?;
        let face_ids = self.face_ids(faces)?;
        let result = brepkit_operations::draft::draft(
            self.topo_mut(),
            solid_id,
            &face_ids,
            pull,
            neutral,
            angle,
        )?;
        Ok(solid_id_to_u32(result))
    }

    // ── Blends and shells ──────────────────────────────────────────

    pub(crate) fn shell_impl(
        &mut self,
        solid: u32,
        thickness: f64,
        open_faces: &[u32],
    ) -> Result<u32, WasmError> {
        validate_positive(thickness, "thickness")?;
        let solid_id = self.resolve_solid(solid)?;
        let face_ids = self.face_ids(open_faces)?;
        let result =
            brepkit_operations::shell_op::shell(self.topo_mut(), solid_id, thickness, &face_ids)?;
        Ok(solid_id_to_u32(result))
    }

    pub(crate) fn chamfer_impl(
        &mut self,
        solid: u32,
        edges: &[u32],
        distance: f64,
    ) -> Result<u32, WasmError> {
        validate_positive(distance, "distance")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids = self.edge_ids(edges)?;
        let result = crate::helpers::try_chamfer(self.topo_mut(), solid_id, &edge_ids, distance)?;
        Ok(solid_id_to_u32(result))
    }

    /// A fillet by the rolling-ball engine, and on failure by the planar
    /// fillet; if the whole set of edges fails, the edges between two
    /// planar faces alone. A panic is caught: it would abort the instance.
    pub(crate) fn fillet_impl(
        &mut self,
        solid: u32,
        edges: &[u32],
        radius: f64,
    ) -> Result<u32, WasmError> {
        validate_positive(radius, "radius")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids = self.edge_ids(edges)?;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::helpers::try_fillet_or_its_planar_edges(
                self.topo_mut(),
                solid_id,
                &edge_ids,
                radius,
            )
        }));
        match outcome {
            Ok(result) => Ok(solid_id_to_u32(result?)),
            Err(payload) => Err(caught(&payload, "Fillet")),
        }
    }

    /// A fillet whose radius varies along each edge as its spec says. A
    /// value a spec gives must be a number, and each radius positive.
    pub(crate) fn fillet_variable_impl(
        &mut self,
        solid: u32,
        specs: &[serde_json::Value],
    ) -> Result<u32, WasmError> {
        let invalid = |reason: String| WasmError::InvalidInput { reason };
        let solid_id = self.resolve_solid(solid)?;
        let mut edge_laws = Vec::with_capacity(specs.len());
        for spec in specs {
            let edge_handle = spec["edge"]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| invalid(format!("a fillet spec's edge is not a handle: {spec}")))?;
            let edge_id = self.resolve_edge(edge_handle)?;
            // A key that is present must hold a radius: `null` is how JSON
            // writes NaN and the infinities, not a missing value.
            let number = |keys: &[&str]| -> Result<Option<f64>, WasmError> {
                for key in keys {
                    if let Some(value) = spec.get(*key) {
                        let r = value.as_f64().ok_or_else(|| {
                            invalid(format!("a fillet spec's '{key}' is not a number: {value}"))
                        })?;
                        validate_positive(r, key)?;
                        return Ok(Some(r));
                    }
                }
                Ok(None)
            };
            let start = number(&["start", "startRadius"])?;
            let end = number(&["end", "endRadius"])?;
            let law = match spec.get("law") {
                None => match (start, end) {
                    (Some(s), Some(e)) if (s - e).abs() > f64::EPSILON => "linear",
                    _ => "constant",
                },
                Some(value) => match value.as_str() {
                    Some(law @ ("constant" | "linear" | "scurve")) => law,
                    _ => {
                        return Err(invalid(format!(
                            "a fillet spec's law must be 'constant', 'linear' or 'scurve', got {value}"
                        )));
                    }
                },
            };
            let law = match law {
                "linear" => FilletRadiusLaw::Linear {
                    start: start.unwrap_or(1.0),
                    end: end.unwrap_or(1.0),
                },
                "scurve" => FilletRadiusLaw::SCurve {
                    start: start.unwrap_or(1.0),
                    end: end.unwrap_or(1.0),
                },
                _ => FilletRadiusLaw::Constant(number(&["radius"])?.or(start).unwrap_or(1.0)),
            };
            edge_laws.push((edge_id, law));
        }
        let result = crate::helpers::try_fillet_variable(self.topo_mut(), solid_id, &edge_laws)?;
        Ok(solid_id_to_u32(result))
    }

    pub(crate) fn fillet_v2_impl(
        &mut self,
        solid: u32,
        edges: &[u32],
        radius: f64,
    ) -> Result<u32, WasmError> {
        validate_positive(radius, "radius")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids = self.edge_ids(edges)?;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<u32, WasmError> {
                let result = brepkit_operations::blend_ops::fillet_v2(
                    self.topo_mut(),
                    solid_id,
                    &edge_ids,
                    radius,
                )?;
                Ok(solid_id_to_u32(brepkit_operations::blend_ops::all_blended(
                    result,
                )?))
            },
        ));
        outcome.unwrap_or_else(|payload| Err(caught(&payload, "FilletV2")))
    }

    pub(crate) fn fillet_v2_corners_impl(
        &mut self,
        solid: u32,
        edges: &[u32],
        radius: f64,
        corners: &serde_json::Value,
    ) -> Result<u32, WasmError> {
        validate_positive(radius, "radius")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids = self.edge_ids(edges)?;
        let corners = self
            .parse_fillet_corners(corners)
            .map_err(|reason| WasmError::InvalidInput { reason })?;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<u32, WasmError> {
                let result = brepkit_operations::blend_ops::fillet_v2_with_corners(
                    self.topo_mut(),
                    solid_id,
                    &edge_ids,
                    radius,
                    &corners,
                )?;
                Ok(solid_id_to_u32(result))
            },
        ));
        outcome.unwrap_or_else(|payload| Err(caught(&payload, "FilletV2Corners")))
    }

    pub(crate) fn chamfer_v2_impl(
        &mut self,
        solid: u32,
        edges: &[u32],
        d1: f64,
        d2: f64,
    ) -> Result<u32, WasmError> {
        validate_positive(d1, "d1")?;
        validate_positive(d2, "d2")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids = self.edge_ids(edges)?;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<u32, WasmError> {
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
            },
        ));
        outcome.unwrap_or_else(|payload| Err(caught(&payload, "ChamferV2")))
    }

    pub(crate) fn chamfer_distance_angle_impl(
        &mut self,
        solid: u32,
        edges: &[u32],
        distance: f64,
        angle: f64,
    ) -> Result<u32, WasmError> {
        validate_positive(distance, "distance")?;
        validate_positive(angle, "angle")?;
        if angle >= std::f64::consts::FRAC_PI_2 {
            return Err(WasmError::InvalidInput {
                reason: "angle must be less than π/2".into(),
            });
        }
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids = self.edge_ids(edges)?;
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

    // ── Offsets ────────────────────────────────────────────────────

    pub(crate) fn offset_solid_impl(
        &mut self,
        solid: u32,
        distance: f64,
    ) -> Result<u32, WasmError> {
        validate_size(distance, "distance")?;
        let solid_id = self.resolve_solid(solid)?;
        let result =
            brepkit_operations::offset_v2::offset_solid_v2(self.topo_mut(), solid_id, distance)?;
        Ok(solid_id_to_u32(result))
    }

    pub(crate) fn thicken_impl(&mut self, face: u32, thickness: f64) -> Result<u32, WasmError> {
        validate_size(thickness, "thickness")?;
        let face_id = self.resolve_face(face)?;
        let result = brepkit_operations::thicken::thicken(self.topo_mut(), face_id, thickness)?;
        Ok(solid_id_to_u32(result))
    }

    /// The offset of a face's outer wire by `distance`, with corners joined
    /// as `join_type` says, or by intersection when it is `None`.
    pub(crate) fn offset_wire_impl(
        &mut self,
        face: u32,
        distance: f64,
        join_type: Option<&str>,
    ) -> Result<u32, WasmError> {
        validate_size(distance, "distance")?;
        let face_id = self.resolve_face(face)?;
        let wire = match join_type {
            None => {
                brepkit_operations::offset_wire::offset_wire(self.topo_mut(), face_id, distance)?
            }
            Some(join) => brepkit_operations::offset_wire::offset_wire_with_join(
                self.topo_mut(),
                face_id,
                distance,
                parse_join_type_str(join)?,
            )?,
        };
        Ok(wire_id_to_u32(wire))
    }

    /// The offset of a planar wire by `distance`, through a face built on it.
    pub(crate) fn offset_wire_2d_impl(
        &mut self,
        wire: u32,
        distance: f64,
        join_type: &str,
    ) -> Result<u32, WasmError> {
        validate_size(distance, "distance")?;
        let wire_id = self.resolve_wire(wire)?;
        let join = parse_join_type_str(join_type)?;
        let face_id =
            brepkit_topology::builder::make_planar_face_from_wire(self.topo_mut(), wire_id)?;
        let result = brepkit_operations::offset_wire::offset_wire_with_join(
            self.topo_mut(),
            face_id,
            distance,
            join,
        )?;
        Ok(wire_id_to_u32(result))
    }

    pub(crate) fn fair_surface_impl(
        &mut self,
        face: u32,
        tolerance: f64,
    ) -> Result<u32, WasmError> {
        validate_tolerance(tolerance, "tolerance")?;
        let face = self.resolve_face(face)?;
        let result =
            brepkit_operations::blend_surface::fair_surface(self.topo_mut(), face, tolerance)?;
        Ok(crate::handles::face_id_to_u32(result))
    }

    // ── Healing ────────────────────────────────────────────────────

    pub(crate) fn sew_faces_impl(
        &mut self,
        faces: &[u32],
        tolerance: f64,
    ) -> Result<u32, WasmError> {
        validate_tolerance(tolerance, "tolerance")?;
        let face_ids = self.face_ids(faces)?;
        let solid = brepkit_operations::sew::sew_faces(self.topo_mut(), &face_ids, tolerance)?;
        Ok(solid_id_to_u32(solid))
    }

    // ── Projection ─────────────────────────────────────────────────

    pub(crate) fn project_edges_impl(
        &self,
        solid: u32,
        frame: (Point3, Vec3, Vec3),
        hidden_lines: bool,
        deflection: f64,
    ) -> Result<serde_json::Value, WasmError> {
        let (origin, dir, x_axis) = frame;
        validate_positive(deflection, "deflection")?;
        validate_point(origin, "the view's origin")?;
        validate_vector(dir, "the view's direction")?;
        validate_vector(x_axis, "the view's x axis")?;
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::projection::project_edges(
            &self.topo,
            solid_id,
            origin,
            dir,
            x_axis,
            hidden_lines,
            deflection,
        )?;
        let flatten = |polys: &[Vec<brepkit_math::vec::Point2>]| -> Vec<Vec<f64>> {
            polys
                .iter()
                .map(|poly| poly.iter().flat_map(|p| [p.x(), p.y()]).collect())
                .collect()
        };
        Ok(serde_json::json!({
            "visible": flatten(&result.visible),
            "hidden": flatten(&result.hidden),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A turn about an axis whose origin lies 1e155 along it keeps a small
    /// profile within 1e155 of that origin, well within the size limit: its
    /// reach is found without squaring 1e155, which overflows. An empty
    /// profile has nothing to check.
    #[test]
    fn a_turn_about_a_far_origin_is_bounded_without_overflow() {
        let profile =
            Reach::of_points(vec![Point3::new(1.0, 0.0, 0.0), Point3::new(2.0, 0.0, 1.0)]);
        let far = Point3::new(0.0, 0.0, 1e155);
        assert!(profile.check_turned(far, "the turned profile").is_ok());
        assert!(
            profile
                .check_turned(Point3::new(0.0, 0.0, 9e299), "past")
                .is_err()
        );
        assert!(
            Reach::of_points(Vec::new())
                .check_turned(far, "nothing")
                .is_ok()
        );
    }
}
