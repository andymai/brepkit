//! The walking fillet's contact on a curved face: the pcurve that the trim
//! registers there gives, at each fraction of its span, the point of the
//! contact edge's curve at that fraction of the edge's span.
//!
//! A round of `r` on the top arc of a rod's sector (radius `R`, turned
//! through `θ`) takes off the region between the corner and the quarter
//! circle, `(1 - π/4) r²` in area, swept round the axis. Its centroid
//! stands `c = r (10 - 3π) / (3 (4 - π))` in from the wall, so Pappus gives
//! `θ (1 - π/4) r² (R - c)`.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{PI, TAU};

    use brepkit_blend::fillet_builder::FilletBuilder;
    use brepkit_check::validate::{CheckId, ValidateOptions};
    use brepkit_math::curves::Circle3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::make_planar_face_from_wire;
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::explorer::solid_edges;
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    const R: f64 = 2.0;

    fn poses() -> Vec<(&'static str, Mat4)> {
        vec![
            ("upright", Mat4::translation(0.0, 0.0, 0.0)),
            ("turned", Mat4::rotation_z(0.7)),
            (
                "tilted",
                Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    /// A sector of a rod of radius `R` and height 4 from `from` through
    /// `turns` of a turn, placed by `pose`, and its top arc.
    fn rod_sector(topo: &mut Topology, from: f64, turns: f64, pose: &Mat4) -> (SolidId, EdgeId) {
        let at = |th: f64| Point3::new(R * th.cos(), R * th.sin(), 0.0);
        let centre = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let first = topo.add_vertex(Vertex::new(at(from), 1e-7));
        let last = topo.add_vertex(Vertex::new(at(from + turns * TAU), 1e-7));
        let rim = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), R).unwrap();
        let edges = [
            Edge::new(centre, first, EdgeCurve::Line),
            Edge::new(first, last, EdgeCurve::Circle(rim)),
            Edge::new(last, centre, EdgeCurve::Line),
        ]
        .map(|edge| OrientedEdge::new(topo.add_edge(edge), true));
        let wire = topo.add_wire(Wire::new(edges.to_vec(), true).unwrap());
        let profile = make_planar_face_from_wire(topo, wire).unwrap();
        let sector = extrude(topo, profile, Vec3::new(0.0, 0.0, 1.0), 4.0).unwrap();
        let arc = solid_edges(topo, sector)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                matches!(edge.curve(), EdgeCurve::Circle(_))
                    && (topo.vertex(edge.start()).unwrap().point().z() - 4.0).abs() < 1e-9
            })
            .unwrap();
        transform_solid(topo, sector, pose).unwrap();
        (sector, arc)
    }

    fn same_parameter_issues(topo: &Topology, solid: SolidId) -> Vec<String> {
        brepkit_check::validate::validate_solid(topo, solid, &ValidateOptions::default())
            .unwrap()
            .issues
            .iter()
            .filter(|issue| issue.check == CheckId::EdgeSameParameter)
            .map(|issue| issue.description.clone())
            .collect()
    }

    /// The closed contact on the wall of a rod cut by a slanted plane: the
    /// round of the ellipse there is walked, and its contact on the wall is
    /// no circle of the wall.
    #[test]
    fn a_round_of_a_slanted_rods_rim_runs_its_wall_contact_along_its_pcurve() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let rod = brepkit_operations::primitives::make_cylinder(&mut topo, R, 5.0).unwrap();
            let cutter =
                brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            let place = Mat4::translation(0.0, 0.0, 4.0)
                * Mat4::rotation_x(0.3)
                * Mat4::translation(-5.0, -5.0, 0.0);
            transform_solid(&mut topo, cutter, &place).unwrap();
            let cut = brepkit_operations::boolean::boolean(
                &mut topo,
                brepkit_operations::boolean::BooleanOp::Cut,
                rod,
                cutter,
            )
            .unwrap();
            transform_solid(&mut topo, cut, &pose).unwrap();
            let rim = solid_edges(&topo, cut)
                .unwrap()
                .into_iter()
                .find(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Ellipse(_)))
                .unwrap();
            let mut builder = FilletBuilder::new(&mut topo, cut);
            builder.add_edges(&[rim], 0.3);
            let built = builder.build().unwrap();
            assert!(built.failed.is_empty(), "{name}: {:?}", built.failed);
            assert!(
                validate_solid(&topo, built.solid).unwrap().is_valid(),
                "{name}: not valid"
            );
            let off = same_parameter_issues(&topo, built.solid);
            assert!(off.is_empty(), "{name}: {off:?}");
        }
    }

    #[test]
    fn a_round_of_a_sectors_arc_runs_its_wall_contact_along_its_pcurve() {
        let r = 0.3;
        let c = r * (10.0 - 3.0 * PI) / (3.0 * (4.0 - PI));
        for (from, turns) in [(0.5, 0.25), (0.25 * PI, 0.75)] {
            for (name, pose) in poses() {
                let what = format!("{turns} turn {name}");
                let mut topo = Topology::new();
                let (sector, arc) = rod_sector(&mut topo, from, turns, &pose);
                let before = solid_volume(&topo, sector, 1e-4).unwrap();
                let mut builder = FilletBuilder::new(&mut topo, sector);
                builder.add_edges(&[arc], r);
                let built = builder.build().unwrap();
                assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
                let solid = built.solid;
                assert!(
                    validate_solid(&topo, solid).unwrap().is_valid(),
                    "{what}: not valid"
                );
                let report = brepkit_check::validate::validate_solid(
                    &topo,
                    solid,
                    &ValidateOptions::default(),
                )
                .unwrap();
                let off: Vec<_> = report
                    .issues
                    .iter()
                    .filter(|issue| issue.check == CheckId::EdgeSameParameter)
                    .collect();
                assert!(off.is_empty(), "{what}: {off:?}");
                let removed = turns * TAU * (1.0 - PI / 4.0) * r * r * (R - c);
                let after = solid_volume(&topo, solid, 1e-4).unwrap();
                assert!(
                    ((before - after) - removed).abs() < 1e-3 * removed,
                    "{what}: took off {} for {removed}",
                    before - after
                );
                let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{what}: open mesh");
            }
        }
    }
}
