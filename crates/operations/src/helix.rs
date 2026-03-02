//! Helical sweep: sweep a profile along a helical path to create
//! thread-like geometry (screws, springs, coils).
//!
//! The helix is approximated as a NURBS curve using rational quadratic
//! Bezier segments (same technique used for circular arcs). The profile
//! face is then swept along this path using the existing sweep operation.

use std::f64::consts::PI;

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// Create a helical sweep of a profile face.
///
/// The helix starts at the origin of the axis, spiraling around
/// `axis` with the given `radius` and `pitch` (height per revolution).
///
/// # Parameters
///
/// - `profile`: The face to sweep along the helix
/// - `axis_origin`: Starting point of the helix axis
/// - `axis_dir`: Direction of the helix axis (normalized internally)
/// - `radius`: Radius of the helix
/// - `pitch`: Vertical distance per full revolution
/// - `turns`: Number of complete turns
/// - `segments_per_turn`: NURBS approximation quality (default: 8)
///
/// # Errors
///
/// Returns an error if:
/// - `radius` or `pitch` is non-positive
/// - `turns` is less than a quarter turn
/// - NURBS curve construction or sweep fails
#[allow(clippy::too_many_arguments)]
pub fn helical_sweep(
    topo: &mut Topology,
    profile: FaceId,
    axis_origin: Point3,
    axis_dir: Vec3,
    radius: f64,
    pitch: f64,
    turns: f64,
    segments_per_turn: usize,
) -> Result<SolidId, OperationsError> {
    if radius <= 0.0 {
        return Err(OperationsError::InvalidInput {
            reason: format!("helix radius must be positive, got {radius}"),
        });
    }
    if pitch <= 0.0 {
        return Err(OperationsError::InvalidInput {
            reason: format!("helix pitch must be positive, got {pitch}"),
        });
    }
    if turns < 0.25 {
        return Err(OperationsError::InvalidInput {
            reason: format!("helix needs at least 0.25 turns, got {turns}"),
        });
    }

    let helix_curve = make_helix_curve(
        axis_origin,
        axis_dir,
        radius,
        pitch,
        turns,
        segments_per_turn,
    )?;

    crate::sweep::sweep(topo, profile, &helix_curve)
}

/// Build a NURBS curve approximating a helix.
///
/// Uses rational quadratic Bezier segments to approximate circular arcs,
/// stacked with a linear pitch component.
///
/// # Errors
///
/// Returns an error if the axis direction is zero or NURBS construction fails.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn make_helix_curve(
    origin: Point3,
    axis_dir: Vec3,
    radius: f64,
    pitch: f64,
    turns: f64,
    segments_per_turn: usize,
) -> Result<NurbsCurve, OperationsError> {
    let axis = axis_dir
        .normalize()
        .map_err(|_| OperationsError::InvalidInput {
            reason: "helix axis direction is zero".to_string(),
        })?;

    let (u_dir, v_dir) = build_local_frame(axis);

    let segs_per_turn = segments_per_turn.max(4);
    let total_segs = ((turns * segs_per_turn as f64).ceil() as usize).max(1);
    let total_segs_f = total_segs as f64;
    let angle_per_seg = (turns * 2.0 * PI) / total_segs_f;
    let height_per_seg = (turns * pitch) / total_segs_f;

    // Weight for the middle control point of each quadratic arc segment.
    let w = (angle_per_seg / 2.0).cos();

    let mut control_points = Vec::with_capacity(2 * total_segs + 1);
    let mut weights = Vec::with_capacity(2 * total_segs + 1);

    // Starting point.
    let start = helix_point(origin, axis, u_dir, v_dir, radius, 0.0, 0.0);
    control_points.push(start);
    weights.push(1.0);

    for seg in 0..total_segs {
        let seg_f = seg as f64;
        let theta_start = seg_f * angle_per_seg;
        let theta_mid = theta_start + angle_per_seg / 2.0;
        let theta_end = theta_start + angle_per_seg;

        let z_start = seg_f * height_per_seg;
        let z_mid = z_start + height_per_seg / 2.0;
        let z_end = z_start + height_per_seg;

        // Mid-arc control point: at tangent intersection, radius / cos(half_angle).
        let mid_radius = radius / w;
        let mid_pt = helix_point(origin, axis, u_dir, v_dir, mid_radius, theta_mid, z_mid);
        let end_pt = helix_point(origin, axis, u_dir, v_dir, radius, theta_end, z_end);

        control_points.push(mid_pt);
        weights.push(w);

        control_points.push(end_pt);
        weights.push(1.0);
    }

    // Build knot vector for degree 2.
    let n = control_points.len();
    let degree = 2;
    let mut knots = Vec::with_capacity(n + degree + 1);

    // Open knot vector: degree+1 zeros at start, interior knots, degree+1 ones at end.
    knots.extend(std::iter::repeat_n(0.0, degree + 1));
    for seg in 1..total_segs {
        let t = seg as f64 / total_segs_f;
        knots.push(t);
        knots.push(t);
    }
    knots.extend(std::iter::repeat_n(1.0, degree + 1));

    NurbsCurve::new(degree, knots, control_points, weights).map_err(|e| {
        OperationsError::InvalidInput {
            reason: format!("helix NURBS curve construction failed: {e}"),
        }
    })
}

/// Compute a point on a helix.
fn helix_point(
    origin: Point3,
    axis: Vec3,
    u_dir: Vec3,
    v_dir: Vec3,
    radius: f64,
    theta: f64,
    height: f64,
) -> Point3 {
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    Point3::new(
        origin.x()
            + u_dir.x().mul_add(
                radius * cos_t,
                v_dir.x().mul_add(radius * sin_t, axis.x() * height),
            ),
        origin.y()
            + u_dir.y().mul_add(
                radius * cos_t,
                v_dir.y().mul_add(radius * sin_t, axis.y() * height),
            ),
        origin.z()
            + u_dir.z().mul_add(
                radius * cos_t,
                v_dir.z().mul_add(radius * sin_t, axis.z() * height),
            ),
    )
}

/// Build a local coordinate frame from an axis direction.
fn build_local_frame(axis: Vec3) -> (Vec3, Vec3) {
    let ax = Vec3::new(1.0, 0.0, 0.0);
    let ay = Vec3::new(0.0, 1.0, 0.0);

    let candidate = if axis.dot(ax).abs() < 0.9 { ax } else { ay };
    let u = axis
        .cross(candidate)
        .normalize()
        .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
    let v = axis
        .cross(u)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 1.0, 0.0));

    (u, v)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_square_face;

    use super::*;

    #[test]
    fn make_helix_curve_basic() {
        let curve = make_helix_curve(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0, // radius
            1.0, // pitch
            1.0, // turns
            8,   // segments
        )
        .unwrap();

        // Degree 2 curve.
        assert_eq!(curve.degree(), 2);

        // Should have enough control points for 8 segments.
        // 2*8 + 1 = 17 control points.
        assert_eq!(curve.control_points().len(), 17);
    }

    #[test]
    fn helix_curve_start_point() {
        let origin = Point3::new(1.0, 2.0, 3.0);
        let curve = make_helix_curve(
            origin,
            Vec3::new(0.0, 0.0, 1.0),
            2.0, // radius
            1.0, // pitch
            1.0, // turns
            8,
        )
        .unwrap();

        let start = curve.evaluate(0.0);
        // Start should be at origin + radius along u_dir.
        // u_dir for axis=(0,0,1) should be perpendicular in XY plane.
        let dist_from_axis = (start.x() - origin.x()).hypot(start.y() - origin.y());
        assert!(
            (dist_from_axis - 2.0).abs() < 0.1,
            "start should be ~2.0 from axis, got {dist_from_axis}"
        );
    }

    #[test]
    fn helix_curve_end_height() {
        let curve = make_helix_curve(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            2.0, // pitch = 2.0
            3.0, // turns = 3.0 → total height = 6.0
            8,
        )
        .unwrap();

        let end = curve.evaluate(1.0);
        assert!(
            (end.z() - 6.0).abs() < 0.3,
            "end height should be ~6.0, got {}",
            end.z()
        );
    }

    #[test]
    fn helical_sweep_creates_solid() {
        let mut topo = Topology::new();
        let profile = make_unit_square_face(&mut topo);

        let result = helical_sweep(
            &mut topo,
            profile,
            Point3::new(5.0, 0.0, 0.0), // Offset from origin
            Vec3::new(0.0, 0.0, 1.0),
            3.0, // radius
            2.0, // pitch
            1.0, // turns
            8,
        );

        assert!(
            result.is_ok(),
            "helical sweep should succeed: {:?}",
            result.err()
        );

        let solid = result.unwrap();
        let s = topo.solid(solid).unwrap();
        let shell = topo.shell(s.outer_shell()).unwrap();
        assert!(
            shell.faces().len() >= 4,
            "helical sweep should produce multiple faces"
        );
    }

    #[test]
    fn helix_negative_radius_error() {
        let mut topo = Topology::new();
        let profile = make_unit_square_face(&mut topo);
        let result = helical_sweep(
            &mut topo,
            profile,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            -1.0,
            1.0,
            1.0,
            8,
        );
        assert!(result.is_err());
    }

    #[test]
    fn helix_zero_pitch_error() {
        let mut topo = Topology::new();
        let profile = make_unit_square_face(&mut topo);
        let result = helical_sweep(
            &mut topo,
            profile,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            0.0,
            1.0,
            8,
        );
        assert!(result.is_err());
    }

    #[test]
    fn helix_too_few_turns_error() {
        let mut topo = Topology::new();
        let profile = make_unit_square_face(&mut topo);
        let result = helical_sweep(
            &mut topo,
            profile,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            1.0,
            0.1,
            8,
        );
        assert!(result.is_err());
    }
}
