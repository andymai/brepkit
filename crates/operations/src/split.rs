//! Split a solid into two halves along a cutting plane.
//!
//! Divides a solid into two new solids along the specified cutting plane.

use brepkit_math::frame::Frame3;
use brepkit_math::mat::Mat4;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::boolean::{BooleanOp, boolean};
use crate::dot_normal_point;

/// Result of splitting a solid: two halves.
#[derive(Debug)]
pub struct SplitResult {
    /// The half on the positive side of the cutting plane (same side as normal).
    pub positive: SolidId,
    /// The half on the negative side of the cutting plane.
    pub negative: SolidId,
}

/// Split a solid into two halves along a plane.
///
/// The cutting plane is defined by a point and a normal. The `positive`
/// half contains geometry on the side the normal points toward; the
/// `negative` half contains the rest.
///
/// A box standing on the plane covers the solid's positive side: the
/// positive half is the solid's intersection with it and the negative half
/// what cutting it away leaves, each through the boolean engine, so curved
/// faces keep their exact geometry.
///
/// # Errors
///
/// Returns an error if the plane doesn't intersect the solid or a boolean
/// fails.
pub fn split(
    topo: &mut Topology,
    solid: SolidId,
    plane_point: Point3,
    plane_normal: Vec3,
) -> Result<SplitResult, crate::OperationsError> {
    let tol = Tolerance::new();
    let bbox = crate::measure::solid_outer_bounds(topo, solid)?;
    // The frame stands on the plane's point nearest the solid: a point far
    // along the plane would round the solid's extent away.
    let normal = Frame3::from_normal(plane_point, plane_normal)?.z;
    let centre = bbox.center();
    let anchor = centre
        - normal * (dot_normal_point(normal, centre) - dot_normal_point(normal, plane_point));
    let frame = Frame3::from_normal(anchor, normal)?;
    let corners = [bbox.min, bbox.max];
    let local: Vec<[f64; 3]> = (0..8)
        .map(|k: usize| {
            let p = Point3::new(
                corners[k & 1].x(),
                corners[(k >> 1) & 1].y(),
                corners[(k >> 2) & 1].z(),
            ) - anchor;
            [p.dot(frame.x), p.dot(frame.y), p.dot(frame.z)]
        })
        .collect();
    let lo = |i: usize| local.iter().map(|c| c[i]).fold(f64::INFINITY, f64::min);
    let hi = |i: usize| local.iter().map(|c| c[i]).fold(f64::NEG_INFINITY, f64::max);
    let misses = || crate::OperationsError::InvalidInput {
        reason: "cutting plane does not split the solid (entirely on one side)".into(),
    };
    if hi(2) <= tol.linear || lo(2) >= -tol.linear {
        return Err(misses());
    }
    let margin = 0.25 * (bbox.max - bbox.min).length() + 1.0;

    let mut half = |op: BooleanOp| -> Result<SolidId, crate::OperationsError> {
        let side = crate::primitives::make_box(
            topo,
            hi(0) - lo(0) + 2.0 * margin,
            hi(1) - lo(1) + 2.0 * margin,
            hi(2) + margin,
        )?;
        let corner = anchor + frame.x * (lo(0) - margin) + frame.y * (lo(1) - margin);
        let place = Mat4([
            [frame.x.x(), frame.y.x(), frame.z.x(), corner.x()],
            [frame.x.y(), frame.y.y(), frame.z.y(), corner.y()],
            [frame.x.z(), frame.y.z(), frame.z.z(), corner.z()],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        crate::transform::transform_solid(topo, side, &place)?;
        match boolean(topo, op, solid, side) {
            Ok(half) if topo.is_empty_solid(half) => Err(misses()),
            Err(crate::OperationsError::EmptyResult { .. }) => Err(misses()),
            other => other,
        }
    };
    let positive = half(BooleanOp::Intersect)?;
    let negative = half(BooleanOp::Cut)?;
    Ok(SplitResult { positive, negative })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    use super::*;

    #[test]
    fn split_cube_at_half_height() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let result = split(
            &mut topo,
            cube,
            Point3::new(0.0, 0.0, 0.5),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();

        let vol_pos = crate::measure::solid_volume(&topo, result.positive, 0.1).unwrap();
        let vol_neg = crate::measure::solid_volume(&topo, result.negative, 0.1).unwrap();

        assert!(
            vol_pos > 0.1,
            "positive half should have volume, got {vol_pos}"
        );
        assert!(
            vol_neg > 0.1,
            "negative half should have volume, got {vol_neg}"
        );

        let tol = Tolerance::loose();
        let total = vol_pos + vol_neg;
        assert!(
            tol.approx_eq(total, 1.0),
            "halves should sum to ~1.0, got {total} ({vol_pos} + {vol_neg})"
        );
    }

    #[test]
    fn split_box_at_quarter() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Box extends from (0,0,0) to (2,2,2). Cut at z=0.5 (quarter height).
        let result = split(
            &mut topo,
            solid,
            Point3::new(0.0, 0.0, 0.5),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();

        let vol_pos = crate::measure::solid_volume(&topo, result.positive, 0.1).unwrap();
        let vol_neg = crate::measure::solid_volume(&topo, result.negative, 0.1).unwrap();

        // Box from 0 to 2. Cut at z=0.5: positive is 3/4, negative is 1/4.
        let total = vol_pos + vol_neg;
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(total, 8.0),
            "halves should sum to ~8.0, got {total}"
        );
    }

    /// A plane tilted 30° through a cylinder's centre crosses only its wall
    /// and halves it, the halves mirror images through the centre. Each half
    /// keeps the cylinder's wall, is a valid closed solid, and holds the
    /// point on the axis on its own side of the plane but not the other's.
    #[test]
    fn split_cylinder_through_its_centre_on_a_tilt() {
        let mut topo = Topology::new();
        let cyl = crate::primitives::make_cylinder(&mut topo, 20.0, 60.0).unwrap();
        crate::transform::transform_solid(&mut topo, cyl, &Mat4::translation(0.0, 0.0, -30.0))
            .unwrap();
        let result = split(
            &mut topo,
            cyl,
            Point3::new(12.125, 0.0, 0.0),
            Vec3::new(0.0, -0.5, 0.866_025_403_784_438_6),
        )
        .unwrap();

        let half = std::f64::consts::PI * 400.0 * 30.0;
        let (above, below) = (Point3::new(0.0, 0.0, 15.0), Point3::new(0.0, 0.0, -15.0));
        for (piece, inside, outside) in [
            (result.positive, above, below),
            (result.negative, below, above),
        ] {
            assert!(
                crate::validate::validate_solid(&topo, piece)
                    .unwrap()
                    .is_valid()
            );
            let mesh = crate::tessellate::tessellate_solid(&topo, piece, 0.05).unwrap();
            assert_eq!(crate::tessellate::boundary_edge_count(&mesh), 0);
            assert_eq!(crate::tessellate::non_manifold_edge_count(&mesh), 0);
            let class = |p| crate::classify::classify_point(&topo, piece, p, 0.0, 1e-7).unwrap();
            assert_eq!(class(inside), crate::classify::PointClassification::Inside);
            assert_eq!(
                class(outside),
                crate::classify::PointClassification::Outside
            );
            let volume = crate::measure::solid_volume(&topo, piece, 0.01).unwrap();
            assert!(
                (volume - half).abs() < 1e-3 * half,
                "half holds {volume}, exact {half}"
            );
            let faces = brepkit_topology::explorer::solid_faces(&topo, piece).unwrap();
            assert!(faces.iter().any(|&f| matches!(
                topo.face(f).unwrap().surface(),
                brepkit_topology::face::FaceSurface::Cylinder(_)
            )));
        }
    }

    #[test]
    fn split_cube_by_a_plane_given_far_along_it() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let result = split(
            &mut topo,
            cube,
            Point3::new(1e17, 0.0, 0.5),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();
        for piece in [result.positive, result.negative] {
            let volume = crate::measure::solid_volume(&topo, piece, 0.01).unwrap();
            assert!((volume - 0.5).abs() < 1e-9, "half holds {volume}");
        }
    }

    /// The plane x + y = 0, given by a point 2⁵⁷ along it, touches the
    /// unit cube only along its edge on the z axis.
    #[test]
    fn split_refuses_a_far_given_plane_touching_an_edge() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let far = 2f64.powi(57);
        let result = split(
            &mut topo,
            cube,
            Point3::new(far, -far, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
        );
        assert!(result.is_err(), "the plane only touches the cube");
    }

    #[test]
    fn split_a_plane_crossing_only_a_spheres_box_misses_it() {
        let mut topo = Topology::new();
        let ball = crate::primitives::make_sphere(&mut topo, 1.0, 32).unwrap();
        let result = split(
            &mut topo,
            ball,
            Point3::new(0.7, 0.7, 0.7),
            Vec3::new(1.0, 1.0, 1.0),
        );
        assert!(result.is_err(), "the plane x + y + z = 2.1 misses the ball");
    }

    /// A block whose top is the Bezier lid `z = 100 u (1 − u)⁴ v (1 − v)`
    /// over the unit square, flat along its rim. The lid peaks at 2.048 at
    /// `(0.2, 0.5)`, between the samples that put it at 1.98.
    #[test]
    fn outer_bounds_reach_a_nurbs_lids_peak() {
        use brepkit_math::nurbs::surface::NurbsSurface;
        use brepkit_topology::face::FaceSurface;

        let mut topo = Topology::new();
        let block = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        crate::transform::transform_solid(&mut topo, block, &Mat4::translation(0.0, 0.0, -1.0))
            .unwrap();
        let top = brepkit_topology::explorer::solid_faces(&topo, block)
            .unwrap()
            .into_iter()
            .find(|&f| {
                matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal, d }
                    if normal.z() > 0.5 && d.abs() < 1e-9)
            })
            .unwrap();
        let rows = (0..6)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        let z = if (i, j) == (1, 1) { 10.0 } else { 0.0 };
                        Point3::new(f64::from(i) / 5.0, f64::from(j) / 2.0, z)
                    })
                    .collect()
            })
            .collect();
        let lid = NurbsSurface::new(
            5,
            2,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            rows,
            vec![vec![1.0; 3]; 6],
        )
        .unwrap();
        topo.face_mut(top)
            .unwrap()
            .set_surface(FaceSurface::Nurbs(lid));

        let top = crate::measure::solid_outer_bounds(&topo, block)
            .unwrap()
            .max
            .z();
        assert!(top >= 2.048, "the lid reaches 2.048, its extent {top}");
    }

    /// A block whose top is a biquadratic patch with row weights 1, -1.5
    /// and 4 and one control point, its middle, at z = -2. With no pole, it
    /// still rises to 3 above the unit square, where its control points stay
    /// at or below 0: they bound nothing, and the bound says so.
    #[test]
    fn outer_bounds_refuse_a_nurbs_face_with_a_negative_weight() {
        use brepkit_math::nurbs::surface::NurbsSurface;
        use brepkit_topology::face::FaceSurface;

        let mut topo = Topology::new();
        let block = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        crate::transform::transform_solid(&mut topo, block, &Mat4::translation(0.0, 0.0, -1.0))
            .unwrap();
        let top = brepkit_topology::explorer::solid_faces(&topo, block)
            .unwrap()
            .into_iter()
            .find(|&f| {
                matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal, d }
                    if normal.z() > 0.5 && d.abs() < 1e-9)
            })
            .unwrap();
        let rows = (0..3)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        let z = if (i, j) == (1, 1) { -2.0 } else { 0.0 };
                        Point3::new(f64::from(i) / 2.0, f64::from(j) / 2.0, z)
                    })
                    .collect()
            })
            .collect();
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let patch = NurbsSurface::new(
            2,
            2,
            knots.clone(),
            knots,
            rows,
            [1.0, -1.5, 4.0].map(|w| vec![w; 3]).to_vec(),
        )
        .unwrap();
        let peak = patch.evaluate(1.0 / 3.0, 0.5).z();
        assert!(peak > 2.99 && patch.aabb().max.z() <= 0.0, "{peak}");
        topo.face_mut(top)
            .unwrap()
            .set_surface(FaceSurface::Nurbs(patch));

        assert!(crate::measure::solid_outer_bounds(&topo, block).is_err());
    }

    #[test]
    fn split_plane_misses_solid() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Plane above the cube.
        let result = split(
            &mut topo,
            cube,
            Point3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!(result.is_err(), "plane above cube should fail");
    }

    #[test]
    fn split_along_x_axis() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let result = split(
            &mut topo,
            cube,
            Point3::new(0.5, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();

        let vol_pos = crate::measure::solid_volume(&topo, result.positive, 0.1).unwrap();
        let vol_neg = crate::measure::solid_volume(&topo, result.negative, 0.1).unwrap();

        let total = vol_pos + vol_neg;
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(total, 1.0),
            "halves should sum to ~1.0, got {total}"
        );
    }
}
