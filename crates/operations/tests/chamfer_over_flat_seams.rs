//! Chamfers over flat seams: an L block made by a fuse of two boxes keeps
//! the seams of the boxes on its top, bottom and two walls (two faces of one
//! plane that meet at 180 degrees). The chamfers of its rim meet at the
//! inside corner, where the seams cut the top in three pieces, and must
//! build as on the same L extruded from its outline.
//!
//! The rim's chamfers take off `d^2 / 2` per unit length of rim. At each of
//! the five outer corners the two chamfers meet in a mitre, each short by
//! the moment of its cross-section about the corner's wall, `A * ybar` with
//! `ybar = d / 3`. At the inside corner each runs on past the corner by the
//! same moment.
//!
//! The walking chamfer of `chamfer_v2` refuses two chamfers that meet in a
//! mitre, so it refuses the rim of either L at its first outer corner,
//! before the seams count. The flat-bevel rebuild of `chamfer` builds the
//! extruded L's rim to the closed form. On the fused L it gives an open
//! shell with 3 boundary edges, of 186.75 at d 0.5 against 187.1667.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::blend_ops::{all_blended, chamfer_v2};
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::chamfer::chamfer;
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

    fn on_top(a: Point3, b: Point3) -> bool {
        (a.z() - HEIGHT).abs() < 1e-9 && (b.z() - HEIGHT).abs() < 1e-9
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

    /// The volume the rim's chamfers of `d` take off the L.
    fn rim_chamfers_take(d: f64) -> f64 {
        let area = 0.5 * d * d;
        let ybar = d / 3.0;
        area * (40.0 - 10.0 * ybar + 2.0 * ybar)
    }

    /// Chamfers `edges` of `solid` by `d` and checks the result is valid,
    /// watertight and of volume `want`. Returns the volume.
    fn chamfers_to(
        label: &str,
        topo: &mut Topology,
        solid: SolidId,
        edges: &[EdgeId],
        d: f64,
        want: f64,
    ) -> f64 {
        let result = chamfer_v2(topo, solid, edges, d, d)
            .and_then(all_blended)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_ne!(result, solid, "{label}: nothing chamfered");
        holds(label, topo, result, want)
    }

    /// Checks `result` is valid, watertight and of volume `want`. Returns
    /// the volume.
    fn holds(label: &str, topo: &Topology, result: SolidId, want: f64) -> f64 {
        let report = validate_solid(topo, result).unwrap();
        assert!(report.is_valid(), "{label}: invalid: {report:?}");
        let mesh = tessellate_solid(topo, result, 0.02).unwrap();
        assert!(is_watertight(&mesh), "{label}: mesh not watertight");
        let volume = solid_volume(topo, result, 0.001).unwrap();
        assert!(
            (volume - want).abs() <= 1e-6 * want,
            "{label}: volume {volume}, want {want}"
        );
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

    /// The flat-bevel rebuild chamfers the extruded L's rim to the closed
    /// form, in each pose.
    #[test]
    fn the_flat_bevels_of_an_extruded_ls_rim_meet_the_closed_form() {
        for d in [0.5, 1.0] {
            for (pose_name, pose) in poses() {
                let label = format!("{pose_name} d {d}");
                let mut topo = Topology::new();
                let l = extruded_l(&mut topo);
                let rim = edges_where(&topo, l, on_top);
                assert_eq!(rim.len(), 6);
                transform_solid(&mut topo, l, &pose).unwrap();
                let result = chamfer(&mut topo, l, &rim, d).unwrap();
                holds(&label, &topo, result, VOLUME - rim_chamfers_take(d));
            }
        }
    }

    /// `chamfer_v2` of the fused L's rim gives no solid it has not built
    /// whole: an error, or a valid, watertight solid of the closed form.
    #[test]
    fn chamfer_v2_of_a_fused_ls_rim_gives_no_wrong_solid() {
        for d in [0.5, 1.0] {
            for (pose_name, pose) in poses() {
                let label = format!("{pose_name} d {d}");
                let mut topo = Topology::new();
                let l = fused_l(&mut topo);
                let rim = edges_where(&topo, l, |a, b| on_top(a, b) && !on_seam(a, b));
                assert_eq!(rim.len(), 8, "{label}: the rim's pieces");
                transform_solid(&mut topo, l, &pose).unwrap();
                if let Ok(result) = chamfer_v2(&mut topo, l, &rim, d, d).and_then(all_blended) {
                    holds(&label, &topo, result, VOLUME - rim_chamfers_take(d));
                }
            }
        }
    }

    #[test]
    #[ignore = "ready repro: the flat-bevel rebuild of the rim across the seams leaves 3 boundary edges"]
    fn the_flat_bevels_of_a_fused_ls_rim_meet_the_closed_form() {
        let d = 0.5;
        let mut topo = Topology::new();
        let l = fused_l(&mut topo);
        let rim = edges_where(&topo, l, |a, b| on_top(a, b) && !on_seam(a, b));
        let result = chamfer(&mut topo, l, &rim, d).unwrap();
        holds("fused", &topo, result, VOLUME - rim_chamfers_take(d));
    }

    /// The extruded L's rim chamfers to the closed form.
    #[test]
    #[ignore = "ready repro: the walking chamfer refuses two chamfers that meet in a mitre"]
    fn the_rim_of_an_extruded_l_chamfers_to_the_closed_form() {
        for d in [0.5, 1.0] {
            let mut topo = Topology::new();
            let l = extruded_l(&mut topo);
            let rim = edges_where(&topo, l, on_top);
            assert_eq!(rim.len(), 6);
            chamfers_to(
                &format!("extruded d {d}"),
                &mut topo,
                l,
                &rim,
                d,
                VOLUME - rim_chamfers_take(d),
            );
        }
    }

    /// The fused L's rim, in eight pieces, chamfers as the extruded L's.
    #[test]
    #[ignore = "ready repro: the walking chamfer refuses two chamfers that meet in a mitre"]
    fn the_rim_of_a_fused_l_chamfers_as_the_extruded_one() {
        for d in [0.5, 1.0] {
            for (pose_name, pose) in poses() {
                let label = format!("{pose_name} d {d}");
                let mut topo = Topology::new();
                let l = fused_l(&mut topo);
                let rim = edges_where(&topo, l, |a, b| on_top(a, b) && !on_seam(a, b));
                assert_eq!(rim.len(), 8, "{label}: the rim's pieces");
                transform_solid(&mut topo, l, &pose).unwrap();
                chamfers_to(&label, &mut topo, l, &rim, d, VOLUME - rim_chamfers_take(d));
            }
        }
    }
}
