//! A torus of radii 5 and 4 cut by a plane tilted across its hole, shallower
//! than the tube's tangent planes: the plane meets the tube in two loops,
//! each winding once around the ring. The cut and the intersect are exact,
//! valid and watertight in every pose. Through the torus's centre the plane
//! halves it; off the centre the two pieces still make up the torus.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::{solid_center_of_mass, solid_volume};
    use brepkit_operations::mirror::mirror;
    use brepkit_operations::primitives::{make_box, make_torus};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::solid::SolidId;

    /// The turned pose.
    fn turn() -> Mat4 {
        Mat4::translation(1.0, -2.0, 0.5) * Mat4::rotation_x(0.6) * Mat4::rotation_z(0.4)
    }

    /// A point of the mirror plane, and its normal.
    fn mirror_plane() -> (Point3, Vec3) {
        (Point3::new(0.3, 0.0, 0.0), Vec3::new(1.0, 0.3, 0.2))
    }

    fn posed(topo: &mut Topology, s: SolidId, pose: &str) -> SolidId {
        match pose {
            "turned" => {
                transform_solid(topo, s, &turn()).unwrap();
                s
            }
            "mirrored" => {
                let (origin, normal) = mirror_plane();
                mirror(topo, s, origin, normal).unwrap()
            }
            _ => s,
        }
    }

    /// The upright point `depth` in from the tube's own circle at ring angle
    /// `u` and tube angle `v`, and which side of the plane it lies: negative
    /// below.
    fn tube_point(u: f64, v: f64, depth: f64, tilt: f64, lift: f64) -> (Point3, f64) {
        let radial = 5.0 + depth * v.cos();
        let p = Point3::new(radial * u.cos(), radial * u.sin(), depth * v.sin());
        (p, tilt.sin() * p.x() + tilt.cos() * (p.z() - lift))
    }

    /// An upright point where `posed` puts it.
    fn model_point(p: Point3, pose: &str) -> Point3 {
        match pose {
            "turned" => turn().mul_point(p),
            "mirrored" => {
                let (origin, normal) = mirror_plane();
                let normal = normal.normalize().unwrap();
                p - normal * (2.0 * normal.dot(p - origin))
            }
            _ => p,
        }
    }

    /// A point of the plane through `(0, 0, lift)` tilted `tilt` about y, and
    /// the normal pointing away from the half-space below it, where `posed`
    /// puts them.
    fn posed_plane(tilt: f64, lift: f64, pose: &str) -> (Point3, Vec3) {
        let on_plane = Point3::new(0.0, 0.0, lift);
        let tip = on_plane + Vec3::new(tilt.sin(), 0.0, tilt.cos());
        let at = model_point(on_plane, pose);
        (at, model_point(tip, pose) - at)
    }

    #[test]
    fn a_plane_tilted_across_a_torus_hole_cuts_it_exactly() {
        let torus_volume = 2.0 * PI * PI * 5.0 * 16.0;
        // (tilt about y, lift along z): the steeper planes stay through the
        // centre, where their loops still wind the ring up to the tube's
        // tangent tilt, asin(4 / 5).
        for (tilt, lift) in [
            (0.2_f64, 0.0_f64),
            (0.2, 0.7),
            (0.5, 0.0),
            (0.5, 0.7),
            (0.8, 0.0),
            (0.9, 0.0),
        ] {
            for pose in ["upright", "turned", "mirrored"] {
                let mut sum = 0.0;
                for op in [BooleanOp::Intersect, BooleanOp::Cut] {
                    let case = format!("tilt {tilt}, lift {lift}, {pose}, {op:?}");
                    let mut topo = Topology::new();
                    let torus = make_torus(&mut topo, 5.0, 4.0, 16).unwrap();
                    // Everything below the plane through (0, 0, lift)
                    // tilted `tilt` about y.
                    let below = make_box(&mut topo, 40.0, 40.0, 20.0).unwrap();
                    let m = Mat4::translation(0.0, 0.0, lift)
                        * Mat4::rotation_y(tilt)
                        * Mat4::translation(-20.0, -20.0, -20.0);
                    transform_solid(&mut topo, below, &m).unwrap();
                    let torus = posed(&mut topo, torus, pose);
                    let below = posed(&mut topo, below, pose);
                    let before = mesh_fallback_count();
                    let result = boolean(&mut topo, op, torus, below).unwrap();
                    assert_eq!(mesh_fallback_count(), before, "{case}: mesh fallback");
                    let report = validate_solid(&topo, result).unwrap();
                    assert!(report.is_valid(), "{case}: {:?}", report.issues);
                    let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
                    assert!(is_watertight(&mesh), "{case}: open or non-manifold mesh");
                    let volume = solid_volume(&topo, result, 0.001).unwrap();
                    // Round the tube, a point reads as held only on the side
                    // its op keeps: the band's seam runs slantwise, and the
                    // classifier has to find it across a whole turn.
                    for (&u, &v) in [0.3, 1.9, 3.3, 4.1, 5.7]
                        .iter()
                        .flat_map(|u| [0.5, 2.0, 3.0, 4.3].iter().map(move |v| (u, v)))
                    {
                        let (p, side) = tube_point(u, v, 2.0, tilt, lift);
                        if side.abs() < 0.1 {
                            continue;
                        }
                        let held = side < 0.0;
                        let read = classify_point(&topo, result, model_point(p, pose), 0.01, 1e-7)
                            .unwrap();
                        let want = if held == (op == BooleanOp::Intersect) {
                            PointClassification::Inside
                        } else {
                            PointClassification::Outside
                        };
                        assert_eq!(read, want, "{case}: tube ({u}, {v})");
                    }
                    // Each op keeps its own side: the kept piece's centre of
                    // mass lies on it.
                    let centre = solid_center_of_mass(&topo, result, 0.01).unwrap();
                    let (on_plane, along) = posed_plane(tilt, lift, pose);
                    let side = along.dot(centre - on_plane);
                    let keeps_below = op == BooleanOp::Intersect;
                    assert_eq!(
                        side < 0.0,
                        keeps_below,
                        "{case}: centre {centre:?}, side {side}"
                    );
                    if lift.abs() < 1e-12 {
                        let half = 0.5 * torus_volume;
                        assert!(
                            (volume - half).abs() < 1e-6 * half,
                            "{case}: volume {volume}, half the torus {half}"
                        );
                    }
                    sum += volume;
                }
                assert!(
                    (sum - torus_volume).abs() < 1e-6 * torus_volume,
                    "tilt {tilt}, lift {lift}, {pose}: the pieces sum to {sum}"
                );
            }
        }
    }
}
