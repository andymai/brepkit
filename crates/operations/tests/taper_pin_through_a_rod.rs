//! A taper pin's hole through a rod: a cone along y cut from a rod along z,
//! its two loops on the rod's wall half a turn apart. Read in one window of
//! `u` about one loop, the other loop spanned the window and the first read
//! as inside it.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
    use brepkit_operations::primitives::{make_cone, make_cylinder};
    use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;

    /// A rod of radius 5 and height 20 turned `spin` about its axis, less a
    /// cone of radii 1.5 and 0.8 along y through it at z = 10, its wide end
    /// at y = 8 when `flip`, else at y = -8.
    fn cut(spin: f64, flip: bool) {
        check_cut(spin, flip, true);
    }

    /// The cut of [`cut`]. With `exact`, the boolean must build it exactly;
    /// without, it may also refuse, but what it returns must be right.
    fn check_cut(spin: f64, flip: bool, exact: bool) {
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        transform_solid(&mut topo, rod, &Mat4::rotation_z(spin)).unwrap();
        let pin = make_cone(&mut topo, 1.5, 0.8, 16.0).unwrap();
        let (lay, start) = if flip {
            (Mat4::rotation_x(PI / 2.0), 8.0)
        } else {
            (Mat4::rotation_x(-PI / 2.0), -8.0)
        };
        transform_solid(&mut topo, pin, &(Mat4::translation(0.0, start, 10.0) * lay)).unwrap();
        let opts = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let name = format!("spin {spin} flip {flip}");
        let result = boolean::boolean_with_options(&mut topo, BooleanOp::Cut, rod, pin, opts);
        if !exact && result.is_err() {
            return;
        }
        assert!(result.is_ok(), "{name}: {result:?}");
        let result = result.unwrap();
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        let faces = brepkit_topology::explorer::solid_faces(&topo, result).unwrap();
        assert_eq!(faces.len(), 4, "{name}: faces");
        let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
        assert_eq!(boundary_edge_count(&mesh), 0, "{name}: open mesh");
        let o = ClassifyOptions::default();
        for p in [
            Point3::new(0.0, 4.9, 10.0),
            Point3::new(0.0, -4.9, 10.0),
            Point3::new(0.0, 0.0, 10.0),
        ] {
            assert_eq!(
                classify_point(&topo, result, p, &o).unwrap(),
                PointClassification::Outside,
                "{name}: {p:?} lies in the hole"
            );
        }
        for p in [Point3::new(3.0, 0.0, 10.0), Point3::new(0.0, 4.0, 12.0)] {
            assert_eq!(
                classify_point(&topo, result, p, &o).unwrap(),
                PointClassification::Inside,
                "{name}: {p:?} lies in the rod"
            );
        }
    }

    #[test]
    fn a_taper_pin_cuts_both_of_its_holes_in_a_rod() {
        for (spin, flip) in [
            (0.0, false),
            (0.0, true),
            (0.3, false),
            (0.3, true),
            (2.0, true),
            (2.6, false),
            (3.0, false),
        ] {
            cut(spin, flip);
        }
    }

    /// Poses that need the mesh fallback today. A cut there must not come
    /// back exact and wrong: at spin 1.4 the rod's wall and the cone's side
    /// each split into one piece, and a change to the face trace once
    /// assembled the rod alone, uncut.
    #[test]
    fn a_taper_pin_cut_the_engine_cannot_build_is_refused_not_wrong() {
        for (spin, flip) in [
            (1.4, false),
            (1.4, true),
            (2.0, false),
            (2.6, true),
            (3.0, true),
        ] {
            check_cut(spin, flip, false);
        }
    }
}
