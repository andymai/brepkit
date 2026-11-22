//! A drum and a frustum built by hand, whose wall has no seam: its outer
//! wire is the bottom rim and its hole the top rim. The mesh of each closes,
//! and a box cut from the top and a box cut through the wall are exact. The
//! boolean gives a copy of the operand a seam and leaves the operand as it
//! was.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
    use brepkit_operations::primitives::{make_box, make_cone, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// `solid`'s wall bounded by its two rims alone, its seam left out.
    fn without_seam(topo: &mut Topology, solid: SolidId) -> SolidId {
        let wall = solid_faces(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&f| !topo.face(f).unwrap().surface().is_planar())
            .unwrap();
        let rims: Vec<OrientedEdge> = topo
            .wire(topo.face(wall).unwrap().outer_wire())
            .unwrap()
            .edges()
            .iter()
            .copied()
            .filter(|oe| !matches!(topo.edge(oe.edge()).unwrap().curve(), EdgeCurve::Line))
            .collect();
        assert_eq!(rims.len(), 2);
        let bottom = topo.add_wire(Wire::new(vec![rims[0]], true).unwrap());
        let top = topo.add_wire(Wire::new(vec![rims[1]], true).unwrap());
        let face = topo.face_mut(wall).unwrap();
        face.set_outer_wire(bottom);
        *face.inner_wires_mut() = vec![top];
        solid
    }

    fn poses() -> Vec<(&'static str, Mat4)> {
        let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1);
        vec![
            ("upright", Mat4::identity()),
            ("turned", turn),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
            ("turned and mirrored", Mat4::scale(1.0, -1.0, 1.0) * turn),
        ]
    }

    fn exact() -> BooleanOptions {
        BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        }
    }

    fn assert_sound(topo: &Topology, solid: SolidId, volume: f64, what: &str) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        let measured = solid_volume(topo, solid, 0.001).unwrap();
        assert!(
            (measured - volume).abs() < 1e-7 * volume,
            "{what}: volume {measured}, expected {volume}"
        );
        for deflection in [0.01, 0.1] {
            let mesh = tessellate_solid(topo, solid, deflection).unwrap();
            assert!(is_watertight(&mesh), "{what}: open mesh at {deflection}");
        }
        // The rims' chords cut the wall's sag off: 32 of them on a radius 2
        // circle hold 0.6% less area.
        let meshed = oriented_solid_volume(topo, solid, 0.01).unwrap();
        assert!(
            (meshed - volume).abs() < 1e-2 * volume,
            "{what}: meshed volume {meshed}, expected {volume}"
        );
    }

    /// Whether each point, in the drum's frame before `pose`, reads inside.
    fn assert_reads(topo: &Topology, solid: SolidId, pose: &Mat4, points: &[([f64; 3], bool)]) {
        for &([x, y, z], inside) in points {
            let at = pose.mul_point(Point3::new(x, y, z));
            let read = classify_point(topo, solid, at, 0.01, 1e-7).unwrap();
            assert_eq!(
                read == PointClassification::Inside,
                inside,
                "({x}, {y}, {z}) reads {read:?}"
            );
        }
    }

    /// The part of a cylinder of radius `r` about z past `x = from` with
    /// `|y| < half`, per unit of height.
    fn segment_past(r: f64, from: f64, half: f64) -> f64 {
        let strip = half * (r * r - half * half).sqrt() + r * r * (half / r).asin();
        strip - 2.0 * half * from
    }

    #[test]
    fn a_drum_with_no_seam_meshes_closed_and_cuts_exactly() {
        let drum = PI * 4.0 * 3.0;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = make_cylinder(&mut topo, 2.0, 3.0).unwrap();
            let solid = without_seam(&mut topo, solid);
            let top = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
            transform_solid(&mut topo, top, &Mat4::translation(-0.5, -0.5, 2.5)).unwrap();
            let side = make_box(&mut topo, 1.5, 1.0, 1.0).unwrap();
            transform_solid(&mut topo, side, &Mat4::translation(1.5, -0.5, 1.0)).unwrap();
            for s in [solid, top, side] {
                transform_solid(&mut topo, s, &pose).unwrap();
            }
            assert_sound(&topo, solid, drum, name);
            assert_reads(
                &topo,
                solid,
                &pose,
                &[
                    ([0.0, 0.0, 1.5], true),
                    ([1.9, 0.3, 0.2], true),
                    ([2.1, 0.0, 1.5], false),
                    ([0.0, 0.0, 3.2], false),
                ],
            );

            let faces_before = solid_faces(&topo, solid).unwrap().len();
            let capped =
                boolean_with_options(&mut topo, BooleanOp::Cut, solid, top, exact()).unwrap();
            assert_sound(&topo, capped, drum - 0.5, &format!("{name}, top cut"));
            let notched =
                boolean_with_options(&mut topo, BooleanOp::Cut, solid, side, exact()).unwrap();
            assert_sound(
                &topo,
                notched,
                drum - segment_past(2.0, 1.5, 0.5),
                &format!("{name}, side cut"),
            );
            assert_reads(
                &topo,
                notched,
                &pose,
                &[
                    ([1.8, 0.0, 1.5], false),
                    ([1.8, 0.0, 0.5], true),
                    ([1.4, 0.0, 1.5], true),
                    ([1.8, 0.7, 1.5], true),
                ],
            );

            let wall = solid_faces(&topo, solid)
                .unwrap()
                .into_iter()
                .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
                .unwrap();
            assert_eq!(solid_faces(&topo, solid).unwrap().len(), faces_before);
            assert_eq!(
                topo.face(wall).unwrap().inner_wires().len(),
                1,
                "{name}: the operand keeps its wall as it was"
            );
        }
    }

    #[test]
    fn a_frustum_with_no_seam_meshes_closed_and_cuts_exactly() {
        let (r0, r1, h) = (2.0_f64, 1.0_f64, 3.0_f64);
        let frustum = PI * h * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = make_cone(&mut topo, r0, r1, h).unwrap();
            let solid = without_seam(&mut topo, solid);
            let top = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
            transform_solid(&mut topo, top, &Mat4::translation(-0.5, -0.5, 2.5)).unwrap();
            for s in [solid, top] {
                transform_solid(&mut topo, s, &pose).unwrap();
            }
            assert_sound(&topo, solid, frustum, name);
            let capped =
                boolean_with_options(&mut topo, BooleanOp::Cut, solid, top, exact()).unwrap();
            assert_sound(&topo, capped, frustum - 0.5, &format!("{name}, top cut"));
        }
    }
}
