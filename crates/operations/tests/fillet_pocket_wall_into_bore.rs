//! Rounds where a pocket's wall meets a bore: a block less a pocket, less a
//! bore whose axis stands on the pocket's wall. The two upright edges where
//! the wall meets the bore are convex, a quarter turn, and run from the
//! floor the pocket and the bore share to the block's top. The cuts leave
//! that floor in three faces of one plane, meeting at each edge's foot.
//!
//! Each round is a straight cylinder of radius `r` standing on the floor,
//! so it takes off the floor's area between the wall, the bore and the
//! round, times the pocket's depth.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use brepkit_blend::fillet_builder::FilletBuilder;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::blend_ops::{all_blended, fillet_v2};
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::explorer::solid_edges;
    use brepkit_topology::solid::SolidId;

    const WALL_X: f64 = 110.0;
    const BORE_Y: f64 = 40.0;
    const BORE_R: f64 = 8.0;
    const DEPTH: f64 = 5.0;

    /// The block less the pocket and the bore, and the two upright edges
    /// where the pocket's wall meets the bore.
    fn pocket_and_bore(topo: &mut Topology) -> (SolidId, Vec<EdgeId>) {
        let block = make_box(topo, 200.0, 80.0, 20.0).unwrap();
        let pocket = make_box(topo, 20.0, 20.0, DEPTH).unwrap();
        transform_solid(topo, pocket, &Mat4::translation(90.0, 30.0, 15.0)).unwrap();
        let bore = make_cylinder(topo, BORE_R, DEPTH).unwrap();
        transform_solid(topo, bore, &Mat4::translation(WALL_X, BORE_Y, 15.0)).unwrap();
        let first = boolean(topo, BooleanOp::Cut, block, pocket).unwrap();
        let solid = boolean(topo, BooleanOp::Cut, first, bore).unwrap();
        let corners: Vec<EdgeId> = solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .filter(|&e| {
                let edge = topo.edge(e).unwrap();
                let a = topo.vertex(edge.start()).unwrap().point();
                let b = topo.vertex(edge.end()).unwrap().point();
                let upright = |p: Point3| {
                    (p.x() - WALL_X).abs() < 1e-7 && ((p.y() - BORE_Y).abs() - BORE_R).abs() < 1e-7
                };
                upright(a) && upright(b) && (a.y() - b.y()).abs() < 1e-7
            })
            .collect();
        assert_eq!(corners.len(), 2);
        (solid, corners)
    }

    /// The floor's area one round takes: between the wall's line, the
    /// bore's circle and the round's circle, whose centre stands `r` off
    /// the wall and `R + r` off the bore's axis.
    fn floor_area_taken(r: f64) -> f64 {
        let big = BORE_R;
        let reach = ((big + r) * (big + r) - r * r).sqrt();
        // The corner, the contact on the wall, the round's centre and its
        // contact on the bore, from the bore's axis.
        let corner = (0.0, big);
        let on_wall = (0.0, reach);
        let centre = (r, reach);
        let on_bore = (r * big / (big + r), reach * big / (big + r));
        let quad = [corner, on_wall, centre, on_bore];
        let mut twice = 0.0;
        for k in 0..4 {
            let (p, q) = (quad[k], quad[(k + 1) % 4]);
            twice += p.0 * q.1 - q.0 * p.1;
        }
        let round_sector = 0.5 * r * r * (r / (big + r)).acos();
        let bore_turn = (r / (big + r)).asin();
        let bore_segment = 0.5 * big * big * (bore_turn - bore_turn.sin());
        0.5 * twice.abs() - round_sector - bore_segment
    }

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "tilted",
                Mat4::translation(15.0, -40.0, 60.0)
                    * Mat4::rotation_y(0.45)
                    * Mat4::rotation_z(-0.8),
            ),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
        ]
    }

    #[test]
    fn the_rounds_where_a_pocket_wall_meets_a_bore_build() {
        let r = 1.0;
        let want = 2.0 * DEPTH * floor_area_taken(r);
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let (solid, corners) = pocket_and_bore(&mut topo);
            transform_solid(&mut topo, solid, &pose).unwrap();
            let before = solid_volume(&topo, solid, 0.01).unwrap();
            let result = fillet_v2(&mut topo, solid, &corners, r)
                .and_then(all_blended)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{name}: invalid: {report:?}");
            let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
            assert!(is_watertight(&mesh), "{name}: mesh not watertight");
            let taken = before - solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (taken - want).abs() <= 1e-6 * want,
                "{name}: took {taken}, want {want}"
            );
        }
    }

    /// The builder alone builds the rounds on the block as the cuts leave
    /// it, its floor in pieces: each round's runout on the floor runs back
    /// along the bore's foot, which the trim keeps whole, and the foot is
    /// split where the contact meets it.
    #[test]
    fn the_builder_rounds_the_block_as_the_cuts_leave_it() {
        let r = 1.0;
        let mut topo = Topology::new();
        let (solid, corners) = pocket_and_bore(&mut topo);
        let before = solid_volume(&topo, solid, 0.01).unwrap();
        let mut builder = FilletBuilder::new(&mut topo, solid);
        builder.add_edges(&corners, r);
        let built = builder.build().expect("the builder must not fail");
        assert!(built.failed.is_empty(), "failed: {:?}", built.failed);
        assert_eq!(built.succeeded.len(), corners.len());
        let report = validate_solid(&topo, built.solid).unwrap();
        assert!(report.is_valid(), "invalid: {report:?}");
        let mesh = tessellate_solid(&topo, built.solid, 0.05).unwrap();
        assert!(is_watertight(&mesh), "mesh not watertight");
        let want = 2.0 * DEPTH * floor_area_taken(r);
        let taken = before - solid_volume(&topo, built.solid, 0.01).unwrap();
        assert!(
            (taken - want).abs() <= 1e-6 * want,
            "took {taken}, want {want}"
        );
    }
}
