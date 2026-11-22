//! Rounds over flat seams: an L block made by a fuse of two boxes keeps the
//! seams of the boxes on its top, bottom and two walls (two faces of one
//! plane that meet at 180 degrees). The rounds of its rim meet at the
//! inside corner, where the seams cut the top in three pieces, and must
//! build as on the same L extruded from its outline. A picked seam has no
//! angle to round and is passed over.
//!
//! The rim's rounds take off `(1 - pi / 4) r^2` per unit length of rim. At
//! each of the five outer corners the two rounds meet in a mitre, each
//! short by the moment of its cross-section about the corner's wall,
//! `A * ybar`; at the inside corner a quarter turn of a horn torus adds
//! `A * ybar * pi / 2` (the theorem of Pappus).

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::blend_ops::{all_blended, fillet_v2};
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::{make_planar_face_from_wire, make_polygon_wire};
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::explorer::solid_edges;
    use brepkit_topology::solid::SolidId;

    const HEIGHT: f64 = 3.0;
    const VOLUME: f64 = 192.0;

    fn fused_l(topo: &mut Topology) -> SolidId {
        let a = make_box(topo, 10.0, 4.0, HEIGHT).unwrap();
        let b = make_box(topo, 4.0, 10.0, HEIGHT).unwrap();
        boolean(topo, BooleanOp::Fuse, a, b).unwrap()
    }

    fn extruded_l(topo: &mut Topology) -> SolidId {
        let outline = [
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 4.0),
            (4.0, 4.0),
            (4.0, 10.0),
            (0.0, 10.0),
        ]
        .map(|(x, y)| Point3::new(x, y, 0.0));
        let wire = make_polygon_wire(topo, &outline, 1e-7).unwrap();
        let face = make_planar_face_from_wire(topo, wire).unwrap();
        extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), HEIGHT).unwrap()
    }

    /// Whether the edge from `a` to `b` lies on one of the fused boxes'
    /// seams: the lines x = 4 and y = 4 inside the L's square corner.
    fn on_seam(a: Point3, b: Point3) -> bool {
        let (x, y) = (0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y()));
        ((a.x() - 4.0).abs() < 1e-9 && (b.x() - 4.0).abs() < 1e-9 && y < 4.0 - 1e-9)
            || ((a.y() - 4.0).abs() < 1e-9 && (b.y() - 4.0).abs() < 1e-9 && x < 4.0 - 1e-9)
    }

    /// The edges of `solid` that `keep` takes, read by their end points.
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

    fn on_top(a: Point3, b: Point3) -> bool {
        (a.z() - HEIGHT).abs() < 1e-9 && (b.z() - HEIGHT).abs() < 1e-9
    }

    /// The volume the rim's rounds take off the L.
    fn rim_rounds_take(r: f64) -> f64 {
        let area = (1.0 - FRAC_PI_4) * r * r;
        let ybar = r * (10.0 - 3.0 * PI) / (12.0 - 3.0 * PI);
        area * (40.0 - 10.0 * ybar + FRAC_PI_2 * ybar)
    }

    /// Rounds `edges` of `solid` (already posed by `pose`) and checks the
    /// result is valid, watertight and of volume `want`. Returns the volume.
    fn rounds_to(
        label: &str,
        topo: &mut Topology,
        solid: SolidId,
        edges: &[EdgeId],
        r: f64,
        want: Option<f64>,
    ) -> f64 {
        let result = fillet_v2(topo, solid, edges, r)
            .and_then(all_blended)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_ne!(result, solid, "{label}: nothing rounded");
        let report = validate_solid(topo, result).unwrap();
        assert!(report.is_valid(), "{label}: invalid: {report:?}");
        let mesh = tessellate_solid(topo, result, 0.02).unwrap();
        assert!(is_watertight(&mesh), "{label}: mesh not watertight");
        let volume = solid_volume(topo, result, 0.001).unwrap();
        if let Some(want) = want {
            assert!(
                (volume - want).abs() <= 1e-6 * want,
                "{label}: volume {volume}, want {want}"
            );
        }
        volume
    }

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(-12.5, 30.0, 7.25)
                    * Mat4::rotation_z(0.6)
                    * Mat4::rotation_x(1.1),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    #[test]
    fn the_rim_of_a_fused_l_rounds_as_the_extruded_one() {
        for r in [0.5, 1.0] {
            for (pose_name, pose) in poses() {
                let label = format!("{pose_name} r {r}");
                let mut topo = Topology::new();
                let l = fused_l(&mut topo);
                let rim = edges_where(&topo, l, |a, b| on_top(a, b) && !on_seam(a, b));
                assert_eq!(rim.len(), 8, "{label}: the rim's pieces");
                transform_solid(&mut topo, l, &pose).unwrap();
                rounds_to(
                    &label,
                    &mut topo,
                    l,
                    &rim,
                    r,
                    Some(VOLUME - rim_rounds_take(r)),
                );
            }
        }
    }

    /// The extruded L's rim rounds to the same closed form: the oracle
    /// holds for the shape the fused L's rounds are measured against.
    #[test]
    fn the_rim_of_an_extruded_l_rounds_to_the_closed_form() {
        for r in [0.5, 1.0] {
            let mut topo = Topology::new();
            let l = extruded_l(&mut topo);
            let rim = edges_where(&topo, l, on_top);
            assert_eq!(rim.len(), 6);
            rounds_to(
                &format!("extruded r {r}"),
                &mut topo,
                l,
                &rim,
                r,
                Some(VOLUME - rim_rounds_take(r)),
            );
        }
    }

    /// The top's seams picked with its rim are passed over.
    #[test]
    fn picked_top_seams_are_passed_over() {
        let r = 0.5;
        let mut topo = Topology::new();
        let l = fused_l(&mut topo);
        let top = edges_where(&topo, l, on_top);
        assert_eq!(top.len(), 10);
        rounds_to(
            "rim and seams",
            &mut topo,
            l,
            &top,
            r,
            Some(VOLUME - rim_rounds_take(r)),
        );
    }

    /// Every edge of the fused L, seams and all, rounds as every edge of
    /// the extruded L does.
    #[test]
    fn every_edge_of_a_fused_l_rounds_as_the_extruded_one() {
        let r = 0.5;
        let mut topo = Topology::new();
        let extruded = extruded_l(&mut topo);
        let all = solid_edges(&topo, extruded).unwrap();
        assert_eq!(all.len(), 18);
        let want = rounds_to("extruded, every edge", &mut topo, extruded, &all, r, None);
        assert!(want < VOLUME - rim_rounds_take(r));
        for (pose_name, pose) in poses() {
            let mut topo = Topology::new();
            let l = fused_l(&mut topo);
            let all = solid_edges(&topo, l).unwrap();
            assert_eq!(all.len(), 28);
            transform_solid(&mut topo, l, &pose).unwrap();
            rounds_to(
                &format!("{pose_name}, every edge"),
                &mut topo,
                l,
                &all,
                r,
                Some(want),
            );
        }
    }

    /// A round of one piece of a rim edge split by a seam stays on that
    /// piece: it does not run on over the seam.
    #[test]
    fn a_round_of_one_piece_stops_at_the_seam() {
        let r = 0.5;
        let mut topo = Topology::new();
        let l = fused_l(&mut topo);
        let piece = edges_where(&topo, l, |a, b| {
            on_top(a, b) && a.y().abs() < 1e-9 && b.y().abs() < 1e-9 && a.x().min(b.x()) > 3.0
        });
        assert_eq!(piece.len(), 1);
        let area = (1.0 - FRAC_PI_4) * r * r;
        rounds_to(
            "one piece",
            &mut topo,
            l,
            &piece,
            r,
            Some(VOLUME - 6.0 * area),
        );
    }

    /// The rim less the piece of its y = 0 edge on the second box: the
    /// rounds meet at the inside corner, so the fillet is made on a copy
    /// with the seams merged. Each pick stays its own edge there, and the
    /// round of the other piece ends square at the seam's vertex, as it
    /// does where nothing is merged.
    #[test]
    fn a_retried_round_stays_on_its_picks() {
        let r = 0.5;
        let mut topo = Topology::new();
        let l = fused_l(&mut topo);
        let rim = edges_where(&topo, l, |a, b| {
            on_top(a, b)
                && !on_seam(a, b)
                && !(a.y().abs() < 1e-9 && b.y().abs() < 1e-9 && a.x().max(b.x()) < 4.0 + 1e-9)
        });
        assert_eq!(rim.len(), 7);
        let area = (1.0 - FRAC_PI_4) * r * r;
        let ybar = r * (10.0 - 3.0 * PI) / (12.0 - 3.0 * PI);
        let want = VOLUME - area * (36.0 - 8.0 * ybar + FRAC_PI_2 * ybar);
        rounds_to("rim less a piece", &mut topo, l, &rim, r, Some(want));
    }

    /// An edge of another solid lying on the fused L's rim, picked in place
    /// of the rim's two pieces there, is never reported as rounded.
    #[test]
    fn an_edge_of_another_solid_is_never_rounded() {
        let mut topo = Topology::new();
        let l = fused_l(&mut topo);
        let other = make_box(&mut topo, 10.0, 4.0, HEIGHT).unwrap();
        let foreign = edges_where(&topo, other, |a, b| {
            on_top(a, b) && a.y().abs() < 1e-9 && b.y().abs() < 1e-9
        });
        assert_eq!(foreign.len(), 1);
        let mut picks = edges_where(&topo, l, |a, b| {
            on_top(a, b) && !on_seam(a, b) && !(a.y().abs() < 1e-9 && b.y().abs() < 1e-9)
        });
        assert_eq!(picks.len(), 6);
        picks.push(foreign[0]);
        if let Ok(result) = fillet_v2(&mut topo, l, &picks, 0.5) {
            assert!(
                !result.succeeded.contains(&foreign[0]),
                "an edge of another solid was reported rounded"
            );
        }
    }

    /// The fused L with the top of its long arm dipping 1e-4 rad from
    /// x = 8 on: a shallow crease, which a loose test of two planes reads as
    /// one plane. Only faces of one plane within the linear tolerance are
    /// merged for the fillet across the seams, so any result keeps every
    /// vertex of a plane face on its plane.
    #[test]
    fn a_shallow_crease_is_not_merged_flat() {
        let tilt = 1e-4_f64;
        let mut topo = Topology::new();
        let l = fused_l(&mut topo);
        let cutter = make_box(&mut topo, 100.0, 100.0, 100.0).unwrap();
        let place = Mat4::translation(8.0, 0.0, HEIGHT)
            * Mat4::rotation_y(tilt)
            * Mat4::translation(-50.0, -50.0, 0.0);
        transform_solid(&mut topo, cutter, &place).unwrap();
        let creased = boolean(&mut topo, BooleanOp::Cut, l, cutter).unwrap();
        let top = |p: Point3| (p.z() - HEIGHT).abs() < 1e-3;
        let rim = edges_where(&topo, creased, |a, b| {
            top(a)
                && top(b)
                && !on_seam(a, b)
                && !((a.x() - 8.0).abs() < 1e-9 && (b.x() - 8.0).abs() < 1e-9)
        });
        let Ok(result) = fillet_v2(&mut topo, creased, &rim, 0.5) else {
            return;
        };
        if result.solid == creased {
            return;
        }
        for face in brepkit_topology::explorer::solid_faces(&topo, result.solid).unwrap() {
            let data = topo.face(face).unwrap();
            let brepkit_topology::face::FaceSurface::Plane { normal, d } = data.surface() else {
                continue;
            };
            for v in brepkit_topology::explorer::face_vertices(&topo, face).unwrap() {
                let p = topo.vertex(v).unwrap().point();
                let off = normal.dot(Vec3::new(p.x(), p.y(), p.z())) - d;
                assert!(
                    off.abs() < 1e-7,
                    "face {} vertex {p:?} off its plane by {off}",
                    face.index()
                );
            }
        }
    }
}
