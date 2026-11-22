//! Rounds at the corner of a step, where a concave edge meets two convex ones.
//!
//! A block 40 on a side has the quarter `x > 20, z > 20` taken off along `y`.
//! At the step's corner `(20, 0, 20)` on the front `y = 0`, the concave edge
//! between the step's floor and its wall runs back along `y`, and two convex
//! edges run along the front: across the floor and up the wall.
//!
//! The concave round (radius `R`) fills the step's inner corner along its whole
//! length, and its end on the front is an arc about its axis. The convex rounds
//! (radius `r`) run along the front's edges up to that arc. Where both are
//! picked, they run on round the arc in a quarter of a torus about the concave
//! round's axis, major radius `R + r`, so the solid is the block with the
//! concave round's strip added and the front's outline, arc and all, rounded.
//! Where one is picked, it stops where the floor or the wall stops, square to
//! its edge. The back `y = 40` has the same corner, and a slot through a
//! plate has four, each at both ends of its concave edge.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, PI};

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::blend_ops::fillet_v2;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::{face_area, solid_volume};
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::explorer::{solid_edges, solid_entity_counts, solid_faces};
    use brepkit_topology::solid::SolidId;

    const SIDE: f64 = 40.0;
    const STEP: f64 = 20.0;

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(31.0, -470.0, 12.0)
                    * Mat4::rotation_x(0.7)
                    * Mat4::rotation_z(-1.2),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    /// The stepped block moved by `pose`, and its edges at the step's corners:
    /// the concave one, then on the front and on the back the one across the
    /// floor and the one up the wall.
    fn stepped(topo: &mut Topology, pose: &Mat4) -> (SolidId, [EdgeId; 5]) {
        let block = make_box(topo, SIDE, SIDE, SIDE).unwrap();
        let step = make_box(topo, STEP, SIDE, STEP).unwrap();
        transform_solid(topo, step, &Mat4::translation(STEP, 0.0, STEP)).unwrap();
        let solid = boolean(topo, BooleanOp::Cut, block, step).unwrap();
        transform_solid(topo, solid, pose).unwrap();
        let edge_to = |topo: &Topology, near: Point3, far: Point3| {
            let (corner, far) = (pose.mul_point(near), pose.mul_point(far));
            let close = |p: Point3, q: Point3| (p - q).length() < 1e-6;
            solid_edges(topo, solid)
                .unwrap()
                .into_iter()
                .find(|&id| {
                    let edge = topo.edge(id).unwrap();
                    let s = topo.vertex(edge.start()).unwrap().point();
                    let e = topo.vertex(edge.end()).unwrap().point();
                    (close(s, corner) && close(e, far)) || (close(s, far) && close(e, corner))
                })
                .unwrap()
        };
        let (front, back) = (Point3::new(STEP, 0.0, STEP), Point3::new(STEP, SIDE, STEP));
        let edges = [
            edge_to(topo, front, back),
            edge_to(topo, front, Point3::new(SIDE, 0.0, STEP)),
            edge_to(topo, front, Point3::new(STEP, 0.0, SIDE)),
            edge_to(topo, back, Point3::new(SIDE, SIDE, STEP)),
            edge_to(topo, back, Point3::new(STEP, SIDE, SIDE)),
        ];
        (solid, edges)
    }

    /// The area a round of radius `r` takes from a square corner.
    fn corner_area(r: f64) -> f64 {
        r * r * (1.0 - PI / 4.0)
    }

    /// How far that area's centroid lies from the corner, along either side.
    fn corner_centroid(r: f64) -> f64 {
        r * (10.0 - 3.0 * PI) / (3.0 * (4.0 - PI))
    }

    /// Valid, watertight, with no face of no area, and holding `volume`.
    fn assert_sound(topo: &Topology, solid: SolidId, volume: f64, within: f64, case: &str) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{case}: {:?}", report.issues);
        for deflection in [0.1, 0.01] {
            let mesh = tessellate_solid(topo, solid, deflection).unwrap();
            assert!(is_watertight(&mesh), "{case}: open mesh at {deflection}");
        }
        for face in solid_faces(topo, solid).unwrap() {
            let area = face_area(topo, face, 0.01).unwrap();
            assert!(area > 1e-6, "{case}: face {} has area {area}", face.index());
        }
        let held = solid_volume(topo, solid, 1e-4).unwrap();
        assert!(
            (held - volume).abs() <= within,
            "{case}: volume {held}, want {volume}"
        );
    }

    /// Rounds `picked` of the step's corner edges (see [`stepped`]) with the
    /// concave radius `big` and the convex radius `small`, in every pose,
    /// and checks the result against `volume`.
    fn check(picked: &[usize], big: f64, small: f64, volume: f64, within: f64) {
        for (name, pose) in poses() {
            let case = format!("{name}, {picked:?}, R {big}, r {small}");
            let mut topo = Topology::new();
            let (solid, edges) = stepped(&mut topo, &pose);
            let before = solid_entity_counts(&topo, solid).unwrap();
            let result = if picked.contains(&0) && (big - small).abs() > 1e-12 {
                let mut builder =
                    brepkit_blend::fillet_builder::FilletBuilder::new(&mut topo, solid);
                builder.add_edges(&[edges[0]], big);
                let convex: Vec<EdgeId> = picked
                    .iter()
                    .filter(|&&k| k != 0)
                    .map(|&k| edges[k])
                    .collect();
                builder.add_edges(&convex, small);
                builder.build().unwrap()
            } else {
                let chosen: Vec<EdgeId> = picked.iter().map(|&k| edges[k]).collect();
                fillet_v2(&mut topo, solid, &chosen, big).unwrap()
            };
            assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
            assert_eq!(result.succeeded.len(), picked.len(), "{case}");
            let solid = result.solid;
            assert_ne!(
                solid_entity_counts(&topo, solid).unwrap(),
                before,
                "{case}: unchanged"
            );
            assert_sound(&topo, solid, volume, within, &case);
            // The concave round's material, and the air its ball leaves.
            let at = |x: f64, y: f64, z: f64| {
                classify_point(
                    &topo,
                    solid,
                    pose.mul_point(Point3::new(x, y, z)),
                    0.01,
                    1e-7,
                )
                .unwrap()
            };
            let (c, mid) = (STEP + big, 0.5 * SIDE);
            let filled = STEP + 0.05 * big;
            assert_eq!(
                at(filled, mid, filled),
                PointClassification::Inside,
                "{case}"
            );
            let hollow = c - 0.5 * big;
            assert_eq!(
                at(hollow, mid, hollow),
                PointClassification::Outside,
                "{case}"
            );
            // The corner the convex round across the floor takes, and the
            // material under its ball.
            if picked.contains(&1) {
                let x = 0.5 * (c + SIDE);
                let gone = STEP - 0.05 * small;
                assert_eq!(
                    at(x, 0.05 * small, gone),
                    PointClassification::Outside,
                    "{case}"
                );
                assert_eq!(
                    at(x, small, STEP - small),
                    PointClassification::Inside,
                    "{case}"
                );
            }
        }
    }

    /// The block with the concave round's strip added along its whole length.
    fn filled(big: f64) -> f64 {
        SIDE * SIDE * SIDE - STEP * SIDE * STEP + SIDE * corner_area(big)
    }

    #[test]
    fn all_three_rounds_run_round_the_concave_rounds_end() {
        for (big, small) in [(2.0, 2.0), (3.0, 1.5)] {
            let along = 2.0 * (SIDE - STEP - big) * corner_area(small);
            // Pappus: the convex round's corner area swept a quarter turn
            // about the concave round's axis.
            let round_the_arc = FRAC_PI_2 * corner_area(small) * (big + corner_centroid(small));
            check(
                &[0, 1, 2],
                big,
                small,
                filled(big) - along - round_the_arc,
                1e-4,
            );
        }
    }

    #[test]
    fn convex_rounds_run_round_both_ends_of_the_concave_round() {
        for (big, small) in [(2.0, 2.0), (3.0, 1.5)] {
            let along = 4.0 * (SIDE - STEP - big) * corner_area(small);
            let round_the_arcs =
                2.0 * FRAC_PI_2 * corner_area(small) * (big + corner_centroid(small));
            check(
                &[0, 1, 2, 3, 4],
                big,
                small,
                filled(big) - along - round_the_arcs,
                1e-4,
            );
        }
    }

    /// A slot 3 by 4 through a plate 10 thick, every edge rounded: its four
    /// concave corners add their strips, and its top and bottom rims, run
    /// round those corners' ends, take the round's corner area along their
    /// straight runs and, by Pappus, a whole turn about the corners' axes.
    #[test]
    fn a_slot_through_a_plate_rounds_all_round() {
        let (wide, long, thick) = (3.0, 4.0, 10.0);
        for r in [0.5, 1.0] {
            let straight = 2.0 * (wide - 2.0 * r) + 2.0 * (long - 2.0 * r);
            let rim = corner_area(r) * (straight + 2.0 * PI * (r + corner_centroid(r)));
            let want = 20.0 * 20.0 * thick - wide * long * thick + 4.0 * thick * corner_area(r)
                - 2.0 * rim;
            for (name, pose) in poses() {
                let case = format!("{name}, r {r}");
                let mut topo = Topology::new();
                let plate = make_box(&mut topo, 20.0, 20.0, thick).unwrap();
                let slot = make_box(&mut topo, wide, long, thick + 2.0).unwrap();
                transform_solid(&mut topo, slot, &Mat4::translation(8.0, 8.0, -1.0)).unwrap();
                let solid = boolean(&mut topo, BooleanOp::Cut, plate, slot).unwrap();
                transform_solid(&mut topo, solid, &pose).unwrap();
                let back = pose.inverse().unwrap();
                let in_slot = |p: Point3| {
                    let p = back.mul_point(p);
                    p.x() > 7.9 && p.x() < 8.1 + wide && p.y() > 7.9 && p.y() < 8.1 + long
                };
                let edges: Vec<EdgeId> = solid_edges(&topo, solid)
                    .unwrap()
                    .into_iter()
                    .filter(|&e| {
                        let edge = topo.edge(e).unwrap();
                        in_slot(topo.vertex(edge.start()).unwrap().point())
                            && in_slot(topo.vertex(edge.end()).unwrap().point())
                    })
                    .collect();
                assert_eq!(edges.len(), 12, "{case}");
                let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
                assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
                assert_eq!(result.succeeded.len(), 12, "{case}");
                assert_sound(&topo, result.solid, want, 1e-4, &case);
            }
        }
    }

    #[test]
    fn a_convex_round_stops_where_the_concave_round_starts() {
        for picked in [[0, 1], [0, 2]] {
            for (big, small) in [(2.0, 2.0), (3.0, 1.5)] {
                let along = (SIDE - STEP - big) * corner_area(small);
                check(&picked, big, small, filled(big) - along, 1e-6);
            }
        }
    }
}
