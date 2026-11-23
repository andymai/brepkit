//! Each gated entry point to the walking blend, given a pick that the
//! fillet builder alone turns into an invalid solid with no failed edge:
//! the sloped top rim of a box whose upright corner edge is rounded, a
//! concave round over a bore's mouth, and a round over a bore that opens
//! onto a thin slit or a shelf between two slits.
//!
//! `fillet_v2`, `fillet_v2_variable`, `chamfer_v2`, `chamfer_v2_on` and
//! `chamfer_distance_angle` each refuse such a pick (an error, or the input
//! back with every pick failed) or build a valid, watertight solid. A
//! chamfer on the slit box with two distances, the longer down the wall,
//! stays clear of the bore and builds to its closed form.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use brepkit_blend::BlendResult;
    use brepkit_blend::fillet_builder::FilletBuilder;
    use brepkit_blend::radius_law::RadiusLaw;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::OperationsError;
    use brepkit_operations::blend_ops::{
        chamfer_distance_angle, chamfer_v2, chamfer_v2_on, fillet_v2, fillet_v2_variable,
    };
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::explorer::{edge_to_face_map, solid_edges};
    use brepkit_topology::face::FaceId;
    use brepkit_topology::solid::SolidId;

    fn edges_where(
        topo: &Topology,
        solid: SolidId,
        keep: impl Fn(Point3, Point3) -> bool,
    ) -> Vec<EdgeId> {
        solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .filter(|&e| {
                let edge = topo.edge(e).unwrap();
                let a = topo.vertex(edge.start()).unwrap().point();
                let b = topo.vertex(edge.end()).unwrap().point();
                keep(a, b)
            })
            .collect()
    }

    fn line_edge(topo: &Topology, solid: SolidId, a: [f64; 3], b: [f64; 3]) -> EdgeId {
        let (a, b) = (Point3::new(a[0], a[1], a[2]), Point3::new(b[0], b[1], b[2]));
        let found = edges_where(topo, solid, |p, q| {
            ((p - a).length() < 1e-9 && (q - b).length() < 1e-9)
                || ((p - b).length() < 1e-9 && (q - a).length() < 1e-9)
        });
        assert_eq!(found.len(), 1);
        found[0]
    }

    /// A 20 x 20 x 10 box, its upright edge at x = y = 20 rounded by 2, cut
    /// by the plane z = 8 + 0.05 y; and the five edges of its top.
    fn sloped_box(topo: &mut Topology) -> (SolidId, Vec<EdgeId>) {
        const SLOPE: f64 = 0.05;
        let block = make_box(topo, 20.0, 20.0, 10.0).unwrap();
        let upright = edges_where(topo, block, |a, b| {
            [a, b]
                .iter()
                .all(|p| (p.x() - 20.0).abs() < 1e-9 && (p.y() - 20.0).abs() < 1e-9)
        });
        let rounded = fillet_v2(topo, block, &upright, 2.0).unwrap();
        let cutter = make_box(topo, 100.0, 100.0, 100.0).unwrap();
        let place = Mat4::translation(0.0, 0.0, 8.0)
            * Mat4::rotation_x(SLOPE.atan())
            * Mat4::translation(-50.0, -50.0, 0.0);
        transform_solid(topo, cutter, &place).unwrap();
        let cut = boolean(topo, BooleanOp::Cut, rounded.solid, cutter).unwrap();
        let on_top = |p: Point3| (p.z() - 8.0 - SLOPE * p.y()).abs() < 1e-7;
        let rim = edges_where(topo, cut, |a, b| on_top(a) && on_top(b));
        assert_eq!(rim.len(), 5);
        (cut, rim)
    }

    /// An L bracket, a base 20 x 20 x 5 with a wall 5 thick to z = 20,
    /// drilled through the base at (9, 10) by a bore of radius 2; and the
    /// inner corner, concave.
    fn bracket(topo: &mut Topology) -> (SolidId, Vec<EdgeId>) {
        let base = make_box(topo, 20.0, 20.0, 5.0).unwrap();
        let wall = make_box(topo, 5.0, 20.0, 20.0).unwrap();
        let bracket = boolean(topo, BooleanOp::Fuse, base, wall).unwrap();
        let drill = make_cylinder(topo, 2.0, 7.0).unwrap();
        transform_solid(topo, drill, &Mat4::translation(9.0, 10.0, -1.0)).unwrap();
        let solid = boolean(topo, BooleanOp::Cut, bracket, drill).unwrap();
        let corner = line_edge(topo, solid, [5.0, 0.0, 5.0], [5.0, 20.0, 5.0]);
        (solid, vec![corner])
    }

    /// A box 40 x 20 x 15 slit through along y between each pair of heights
    /// in `slits`, from x = 5 to 35, and bored from the top down to z = 7.05
    /// at (20, 6) by a bore of radius 2; and its top edge along x at y = 0.
    fn slit_box(topo: &mut Topology, slits: &[(f64, f64)]) -> (SolidId, Vec<EdgeId>) {
        let mut solid = make_box(topo, 40.0, 20.0, 15.0).unwrap();
        for &(low, high) in slits {
            let slit = make_box(topo, 30.0, 22.0, high - low).unwrap();
            transform_solid(topo, slit, &Mat4::translation(5.0, -1.0, low)).unwrap();
            solid = boolean(topo, BooleanOp::Cut, solid, slit).unwrap();
        }
        let drill = make_cylinder(topo, 2.0, 16.0 - 7.05).unwrap();
        transform_solid(topo, drill, &Mat4::translation(20.0, 6.0, 7.05)).unwrap();
        let solid = boolean(topo, BooleanOp::Cut, solid, drill).unwrap();
        let edge = line_edge(topo, solid, [0.0, 0.0, 15.0], [40.0, 0.0, 15.0]);
        (solid, vec![edge])
    }

    fn thin_slit(topo: &mut Topology) -> (SolidId, Vec<EdgeId>) {
        slit_box(topo, &[(7.0, 7.1)])
    }

    fn shelf(topo: &mut Topology) -> (SolidId, Vec<EdgeId>) {
        slit_box(topo, &[(7.0, 7.1), (6.8, 6.95)])
    }

    /// Each edge with its face on the wall y = 0, the face its first
    /// distance is read on.
    fn on_the_wall(topo: &Topology, solid: SolidId, edges: &[EdgeId]) -> Vec<(EdgeId, FaceId)> {
        let uses = edge_to_face_map(topo, solid).unwrap();
        edges
            .iter()
            .map(|&edge| {
                let wall = uses[&edge.index()]
                    .iter()
                    .copied()
                    .find(|&f| {
                        topo.face(f)
                            .unwrap()
                            .effective_plane_normal()
                            .is_some_and(|n| n.y() < -0.5)
                    })
                    .unwrap();
                (edge, wall)
            })
            .collect()
    }

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(4.0, -9.5, 21.0) * Mat4::rotation_x(0.7) * Mat4::rotation_z(1.9),
            ),
            ("mirrored", Mat4::scale(1.0, 1.0, -1.0)),
        ]
    }

    /// The result is a refusal, or a valid, watertight solid with every
    /// pick blended. Returns the solid where one was built.
    fn no_bad_solid(
        label: &str,
        topo: &Topology,
        input: SolidId,
        picks: usize,
        got: Result<BlendResult, OperationsError>,
    ) -> Option<SolidId> {
        let result = got.ok()?;
        if result.solid == input {
            assert!(result.succeeded.is_empty(), "{label}: a pick succeeded");
            assert_eq!(result.failed.len(), picks, "{label}: the failed picks");
            return None;
        }
        assert!(result.failed.is_empty(), "{label}: {:?}", result.failed);
        let report = validate_solid(topo, result.solid).unwrap();
        assert!(report.is_valid(), "{label}: invalid: {report:?}");
        let mesh = tessellate_solid(topo, result.solid, 0.05).unwrap();
        assert!(is_watertight(&mesh), "{label}: the mesh is open");
        Some(result.solid)
    }

    type Input = fn(&mut Topology) -> (SolidId, Vec<EdgeId>);

    /// The builder alone gives an invalid solid with no failed edge for each
    /// pick here, and each gated entry point gives none, in each pose.
    #[test]
    fn no_entry_point_returns_the_builders_bad_solid() {
        let inputs: [(&str, Input, f64); 4] = [
            ("sloped rim", sloped_box, 1.0),
            ("bracket", bracket, 6.0),
            ("thin slit", thin_slit, 7.0),
            ("shelf", shelf, 7.0),
        ];
        for (name, make, r) in inputs {
            let mut topo = Topology::new();
            let (solid, picks) = make(&mut topo);
            let mut builder = FilletBuilder::new(&mut topo, solid);
            builder.add_edges(&picks, r);
            let alone = builder.build().unwrap();
            assert!(alone.failed.is_empty(), "{name}");
            assert!(
                !validate_solid(&topo, alone.solid).unwrap().is_valid(),
                "{name}: the builder alone gives a valid solid"
            );
            for (pose_name, pose) in poses() {
                for entry in 0..6 {
                    let mut topo = Topology::new();
                    let (solid, picks) = make(&mut topo);
                    transform_solid(&mut topo, solid, &pose).unwrap();
                    let label = format!("{name}, {pose_name}, entry {entry}");
                    let laws = |law: &dyn Fn() -> RadiusLaw| -> Vec<(EdgeId, RadiusLaw)> {
                        picks.iter().map(|&e| (e, law())).collect()
                    };
                    let got = match entry {
                        0 => fillet_v2(&mut topo, solid, &picks, r),
                        1 => fillet_v2_variable(&mut topo, solid, laws(&|| RadiusLaw::Constant(r))),
                        2 => fillet_v2_variable(
                            &mut topo,
                            solid,
                            laws(&|| RadiusLaw::Linear {
                                start: r,
                                end: 0.8 * r,
                            }),
                        ),
                        3 => chamfer_v2(&mut topo, solid, &picks, r, r),
                        4 => {
                            let pairs = edge_to_face_map(&topo, solid).unwrap();
                            let pairs: Vec<(EdgeId, FaceId)> =
                                picks.iter().map(|&e| (e, pairs[&e.index()][0])).collect();
                            chamfer_v2_on(&mut topo, solid, &pairs, r, 0.5 * r)
                        }
                        _ => chamfer_distance_angle(&mut topo, solid, &picks, r, 0.6),
                    };
                    no_bad_solid(&label, &topo, solid, picks.len(), got);
                }
            }
        }
    }

    /// On the slit box, 7 down the wall and 3.5 into the top stays clear of
    /// the bore, 4 from the edge: the chamfer takes off its prism, 7 x 3.5
    /// / 2 along the 40 of the edge. The bore's part inside the slit is the
    /// slit's.
    #[test]
    fn a_chamfer_clear_of_the_bore_builds_to_its_prism() {
        let bore = std::f64::consts::PI * 4.0 * (15.0 - 7.1);
        let prism = 0.5 * 7.0 * 3.5 * 40.0;
        for (name, make, slits) in [
            ("thin slit", thin_slit as Input, 60.0),
            ("shelf", shelf as Input, 150.0),
        ] {
            let want = 40.0 * 20.0 * 15.0 - slits - bore - prism;
            for (pose_name, pose) in poses() {
                let label = format!("{name}, {pose_name}");
                let mut topo = Topology::new();
                let (solid, picks) = make(&mut topo);
                let pairs = on_the_wall(&topo, solid, &picks);
                transform_solid(&mut topo, solid, &pose).unwrap();
                let got = chamfer_v2_on(&mut topo, solid, &pairs, 7.0, 3.5);
                let result = no_bad_solid(&label, &topo, solid, picks.len(), got)
                    .unwrap_or_else(|| panic!("{label}: refused"));
                let volume = solid_volume(&topo, result, 0.001).unwrap();
                assert!(
                    (volume - want).abs() <= 1e-7 * want,
                    "{label}: volume {volume}, want {want}"
                );
            }
        }
    }
}
