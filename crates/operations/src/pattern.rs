//! Pattern operations: linear and circular arrays of solids.
//!
//! Creates multiple copies of a solid arranged in a pattern.

use brepkit_math::mat::Mat4;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Vec3;
use brepkit_topology::Topology;
use brepkit_topology::compound::{Compound, CompoundId};
use brepkit_topology::solid::SolidId;

use crate::copy::copy_solid;
use crate::transform::transform_solid;

/// Create a linear pattern of a solid.
///
/// Produces `count` copies of the solid, each offset from the previous
/// by `spacing` along `direction`. The original solid is included as
/// the first element.
///
/// Returns a compound containing all copies.
///
/// # Errors
///
/// Returns an error if `count < 1`, `spacing` is non-positive,
/// the direction is zero-length, or copy/transform fails.
pub fn linear_pattern(
    topo: &mut Topology,
    solid: SolidId,
    direction: Vec3,
    spacing: f64,
    count: usize,
) -> Result<CompoundId, crate::OperationsError> {
    let tol = Tolerance::new();

    if count < 1 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "pattern count must be at least 1".into(),
        });
    }
    if spacing <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("pattern spacing must be positive, got {spacing}"),
        });
    }

    let dir = direction.normalize()?;

    let mut solids = Vec::with_capacity(count);
    solids.push(solid);

    for i in 1..count {
        let copy = copy_solid(topo, solid)?;
        #[allow(clippy::cast_precision_loss)]
        let offset = dir * (spacing * i as f64);
        let matrix = Mat4::translation(offset.x(), offset.y(), offset.z());
        transform_solid(topo, copy, &matrix)?;
        solids.push(copy);
    }

    let compound = Compound::new(solids);
    Ok(topo.add_compound(compound))
}

/// Create a circular pattern of a solid.
///
/// Produces `count` copies arrayed around an axis, evenly spaced over
/// a full 360 degrees. The original solid is included as the first element.
///
/// Returns a compound containing all copies.
///
/// # Errors
///
/// Returns an error if `count < 2`, the axis is zero-length, or
/// copy/transform fails.
pub fn circular_pattern(
    topo: &mut Topology,
    solid: SolidId,
    axis_direction: Vec3,
    count: usize,
) -> Result<CompoundId, crate::OperationsError> {
    if count < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "circular pattern needs at least 2 copies".into(),
        });
    }

    let axis = axis_direction.normalize()?;

    let mut solids = Vec::with_capacity(count);
    solids.push(solid);

    #[allow(clippy::cast_precision_loss)]
    let angle_step = 2.0 * std::f64::consts::PI / (count as f64);

    for i in 1..count {
        let copy = copy_solid(topo, solid)?;
        #[allow(clippy::cast_precision_loss)]
        let angle = angle_step * (i as f64);

        // Build rotation matrix around the axis.
        let matrix = rotation_matrix(axis, angle);
        transform_solid(topo, copy, &matrix)?;
        solids.push(copy);
    }

    let compound = Compound::new(solids);
    Ok(topo.add_compound(compound))
}

/// Create a 2D grid pattern of a solid.
///
/// Produces `count_x × count_y` copies arranged in a rectangular grid.
/// Each row is offset by `spacing_x` along `dir_x`, each column by
/// `spacing_y` along `dir_y`. The original solid occupies position (0, 0).
///
/// Returns a compound containing all copies.
///
/// # Errors
///
/// Returns an error if either count is less than 1, either spacing is
/// non-positive, either direction is zero-length, or copy/transform fails.
#[allow(clippy::too_many_arguments)]
pub fn grid_pattern(
    topo: &mut Topology,
    solid: SolidId,
    dir_x: Vec3,
    dir_y: Vec3,
    spacing_x: f64,
    spacing_y: f64,
    count_x: usize,
    count_y: usize,
) -> Result<CompoundId, crate::OperationsError> {
    let tol = Tolerance::new();

    if count_x < 1 || count_y < 1 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "grid pattern counts must be at least 1".into(),
        });
    }
    if spacing_x <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("grid spacing_x must be positive, got {spacing_x}"),
        });
    }
    if spacing_y <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("grid spacing_y must be positive, got {spacing_y}"),
        });
    }

    let dx = dir_x.normalize()?;
    let dy = dir_y.normalize()?;

    if dx.cross(dy).length() < tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: "dir_x and dir_y must not be parallel".into(),
        });
    }

    let mut solids = Vec::with_capacity(count_x * count_y);

    for iy in 0..count_y {
        for ix in 0..count_x {
            if ix == 0 && iy == 0 {
                solids.push(solid);
                continue;
            }

            let copy = copy_solid(topo, solid)?;

            #[allow(clippy::cast_precision_loss)]
            let offset = dx * (spacing_x * ix as f64) + dy * (spacing_y * iy as f64);

            let matrix = Mat4::translation(offset.x(), offset.y(), offset.z());
            transform_solid(topo, copy, &matrix)?;
            solids.push(copy);
        }
    }

    let compound = Compound::new(solids);
    Ok(topo.add_compound(compound))
}

/// Build a rotation matrix for a given axis and angle (Rodrigues' formula).
fn rotation_matrix(axis: Vec3, angle: f64) -> Mat4 {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let omc = 1.0 - cos_a; // one minus cos
    let ax = axis.x();
    let ay = axis.y();
    let az = axis.z();

    Mat4([
        [
            omc.mul_add(ax * ax, cos_a),
            ax.mul_add(ay * omc, -(sin_a * az)),
            ax.mul_add(az * omc, sin_a * ay),
            0.0,
        ],
        [
            ax.mul_add(ay * omc, sin_a * az),
            omc.mul_add(ay * ay, cos_a),
            ay.mul_add(az * omc, -(sin_a * ax)),
            0.0,
        ],
        [
            ax.mul_add(az * omc, -(sin_a * ay)),
            ay.mul_add(az * omc, sin_a * ax),
            omc.mul_add(az * az, cos_a),
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::Vec3;
    use brepkit_topology::Topology;

    use super::*;

    #[test]
    fn linear_pattern_3_boxes() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let compound = linear_pattern(&mut topo, solid, Vec3::new(1.0, 0.0, 0.0), 2.0, 3).unwrap();

        let comp = topo.compound(compound).unwrap();
        assert_eq!(comp.solids().len(), 3, "should have 3 copies");

        // Verify each copy has the right volume.
        let tol = Tolerance::loose();
        for &sid in comp.solids() {
            let vol = crate::measure::solid_volume(&topo, sid, 0.1).unwrap();
            assert!(
                tol.approx_eq(vol, 1.0),
                "each copy should have volume ~1.0, got {vol}"
            );
        }
    }

    #[test]
    fn linear_pattern_spacing_shifts_bboxes() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let compound = linear_pattern(&mut topo, solid, Vec3::new(1.0, 0.0, 0.0), 3.0, 3).unwrap();

        let comp = topo.compound(compound).unwrap();
        let tol = Tolerance::loose();

        // First copy at x=0, second at x=3, third at x=6.
        let bbox0 = crate::measure::solid_bounding_box(&topo, comp.solids()[0]).unwrap();
        let bbox1 = crate::measure::solid_bounding_box(&topo, comp.solids()[1]).unwrap();
        let bbox2 = crate::measure::solid_bounding_box(&topo, comp.solids()[2]).unwrap();

        // Box goes from [0,1], copies shifted by 3 and 6 along x.
        assert!(
            tol.approx_eq(bbox0.min.x(), 0.0),
            "first copy min_x should be ~0.0, got {}",
            bbox0.min.x()
        );
        assert!(
            tol.approx_eq(bbox1.min.x(), 3.0),
            "second copy min_x should be ~3.0, got {}",
            bbox1.min.x()
        );
        assert!(
            tol.approx_eq(bbox2.min.x(), 6.0),
            "third copy min_x should be ~6.0, got {}",
            bbox2.min.x()
        );
    }

    #[test]
    fn linear_pattern_single_returns_original() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let compound = linear_pattern(&mut topo, solid, Vec3::new(1.0, 0.0, 0.0), 1.0, 1).unwrap();

        let comp = topo.compound(compound).unwrap();
        assert_eq!(comp.solids().len(), 1);
        assert_eq!(comp.solids()[0].index(), solid.index());
    }

    #[test]
    fn linear_pattern_zero_spacing_error() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        assert!(linear_pattern(&mut topo, solid, Vec3::new(1.0, 0.0, 0.0), 0.0, 3).is_err());
    }

    #[test]
    fn circular_pattern_4_around_z() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Move box to x=3 so rotations are visible.
        let matrix = Mat4::translation(3.0, 0.0, 0.0);
        transform_solid(&mut topo, solid, &matrix).unwrap();

        let compound = circular_pattern(&mut topo, solid, Vec3::new(0.0, 0.0, 1.0), 4).unwrap();

        let comp = topo.compound(compound).unwrap();
        assert_eq!(comp.solids().len(), 4, "should have 4 copies");

        let tol = Tolerance::loose();
        for &sid in comp.solids() {
            let vol = crate::measure::solid_volume(&topo, sid, 0.1).unwrap();
            assert!(
                tol.approx_eq(vol, 1.0),
                "each copy should have volume ~1.0, got {vol}"
            );
        }
    }

    #[test]
    fn circular_pattern_single_error() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        assert!(circular_pattern(&mut topo, solid, Vec3::new(0.0, 0.0, 1.0), 1).is_err());
    }

    #[test]
    fn grid_pattern_3x2() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let compound = grid_pattern(
            &mut topo,
            solid,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0,
            3.0,
            3,
            2,
        )
        .unwrap();

        let comp = topo.compound(compound).unwrap();
        assert_eq!(comp.solids().len(), 6, "3×2 grid should have 6 copies");

        let tol = Tolerance::loose();
        for &sid in comp.solids() {
            let vol = crate::measure::solid_volume(&topo, sid, 0.1).unwrap();
            assert!(
                tol.approx_eq(vol, 1.0),
                "each copy should have volume ~1.0, got {vol}"
            );
        }
    }

    #[test]
    fn grid_pattern_positions() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let compound = grid_pattern(
            &mut topo,
            solid,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            5.0,
            5.0,
            2,
            2,
        )
        .unwrap();

        let comp = topo.compound(compound).unwrap();
        let tol = Tolerance::loose();

        // (0,0), (5,0), (0,5), (5,5)
        let bbox00 = crate::measure::solid_bounding_box(&topo, comp.solids()[0]).unwrap();
        let bbox10 = crate::measure::solid_bounding_box(&topo, comp.solids()[1]).unwrap();
        let bbox01 = crate::measure::solid_bounding_box(&topo, comp.solids()[2]).unwrap();
        let bbox11 = crate::measure::solid_bounding_box(&topo, comp.solids()[3]).unwrap();

        assert!(tol.approx_eq(bbox00.min.x(), 0.0));
        assert!(tol.approx_eq(bbox00.min.y(), 0.0));
        assert!(tol.approx_eq(bbox10.min.x(), 5.0));
        assert!(tol.approx_eq(bbox10.min.y(), 0.0));
        assert!(tol.approx_eq(bbox01.min.x(), 0.0));
        assert!(tol.approx_eq(bbox01.min.y(), 5.0));
        assert!(tol.approx_eq(bbox11.min.x(), 5.0));
        assert!(tol.approx_eq(bbox11.min.y(), 5.0));
    }

    #[test]
    fn grid_pattern_1x1_returns_original() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let compound = grid_pattern(
            &mut topo,
            solid,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            1.0,
            1.0,
            1,
            1,
        )
        .unwrap();

        let comp = topo.compound(compound).unwrap();
        assert_eq!(comp.solids().len(), 1);
        assert_eq!(comp.solids()[0].index(), solid.index());
    }

    #[test]
    fn grid_pattern_zero_count_error() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        assert!(
            grid_pattern(
                &mut topo,
                solid,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                1.0,
                1.0,
                0,
                3
            )
            .is_err()
        );
    }

    #[test]
    fn grid_pattern_zero_spacing_error() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        assert!(
            grid_pattern(
                &mut topo,
                solid,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                0.0,
                1.0,
                3,
                3
            )
            .is_err()
        );
    }

    #[test]
    fn grid_pattern_parallel_directions_error() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        // Both directions along X — should fail.
        assert!(
            grid_pattern(
                &mut topo,
                solid,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                1.0,
                1.0,
                3,
                3
            )
            .is_err()
        );
    }
}
