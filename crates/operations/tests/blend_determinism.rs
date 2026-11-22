//! Chamfers and fillets give the same result on every run: the same faces in
//! the same order, bounded by the same edges and vertices, at the same
//! bit-exact points.
//!
//! `std` hash maps seed every instance differently, so repeating an
//! operation in one process exercises the orders a later process would see.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::mat::Mat4;
    use brepkit_operations::blend_ops::{chamfer_v2, fillet_v2};
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::chamfer::{chamfer, chamfer_asymmetric};
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::transform::transform_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{EdgeCurve, EdgeId};
    use brepkit_topology::explorer::{solid_edges, solid_faces};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::VertexId;

    const RUNS: usize = 6;

    fn fingerprint(topo: &Topology, solid: SolidId) -> Vec<String> {
        let point = |v: VertexId| {
            let q = topo.vertex(v).unwrap().point();
            format!(
                "#{}({:016x},{:016x},{:016x})",
                v.index(),
                q.x().to_bits(),
                q.y().to_bits(),
                q.z().to_bits()
            )
        };
        let mut out = Vec::new();
        for fid in solid_faces(topo, solid).unwrap() {
            let face = topo.face(fid).unwrap();
            out.push(format!(
                "face #{} {} rev={}",
                fid.index(),
                face.surface().type_tag(),
                face.is_reversed()
            ));
            let wires =
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
            for wid in wires {
                out.push(format!("  wire #{}", wid.index()));
                for oe in topo.wire(wid).unwrap().edges() {
                    let edge = topo.edge(oe.edge()).unwrap();
                    out.push(format!(
                        "    edge #{} {} fwd={} {} -> {}",
                        oe.edge().index(),
                        edge.curve().type_tag(),
                        oe.is_forward(),
                        point(edge.start()),
                        point(edge.end())
                    ));
                }
            }
        }
        out
    }

    fn assert_repeatable(name: &str, build: impl Fn(&mut Topology) -> SolidId) {
        let mut first: Option<Vec<String>> = None;
        for run in 0..RUNS {
            let mut topo = Topology::default();
            let solid = build(&mut topo);
            let fp = fingerprint(&topo, solid);
            match &first {
                None => first = Some(fp),
                Some(want) => {
                    let at = want
                        .iter()
                        .zip(&fp)
                        .position(|(a, b)| a != b)
                        .unwrap_or_else(|| want.len().min(fp.len()));
                    assert!(
                        *want == fp,
                        "{name}: run {run} differs from run 0 at line {at} \
                         ({} vs {} lines)\n  run 0: {:?}\n  run {run}: {:?}",
                        want.len(),
                        fp.len(),
                        want.get(at),
                        fp.get(at)
                    );
                }
            }
        }
    }

    fn edges_where(
        t: &Topology,
        s: SolidId,
        keep: impl Fn(&Topology, EdgeId) -> bool,
    ) -> Vec<EdgeId> {
        solid_edges(t, s)
            .unwrap()
            .into_iter()
            .filter(|&e| keep(t, e))
            .collect()
    }

    fn run(t: &Topology, e: EdgeId) -> brepkit_math::vec::Vec3 {
        let e = t.edge(e).unwrap();
        t.vertex(e.end()).unwrap().point() - t.vertex(e.start()).unwrap().point()
    }

    fn on_level(t: &Topology, e: EdgeId, z: f64) -> bool {
        let e = t.edge(e).unwrap();
        [e.start(), e.end()]
            .iter()
            .all(|&v| (t.vertex(v).unwrap().point().z() - z).abs() < 1e-9)
    }

    fn is_circle(t: &Topology, e: EdgeId) -> bool {
        matches!(t.edge(e).unwrap().curve(), EdgeCurve::Circle(_))
    }

    /// A 20 x 12 x 4 plate with a bore of radius 2 through its middle.
    fn holed_plate(t: &mut Topology) -> SolidId {
        let plate = make_box(t, 20.0, 12.0, 4.0).unwrap();
        let bore = make_cylinder(t, 2.0, 8.0).unwrap();
        transform_solid(t, bore, &Mat4::translation(10.0, 6.0, -2.0)).unwrap();
        boolean(t, BooleanOp::Cut, plate, bore).unwrap()
    }

    /// Two 3 high bars fused into an L, with a concave edge at its inside
    /// corner.
    fn l_block(t: &mut Topology) -> SolidId {
        let a = make_box(t, 10.0, 4.0, 3.0).unwrap();
        let b = make_box(t, 4.0, 10.0, 3.0).unwrap();
        boolean(t, BooleanOp::Fuse, a, b).unwrap()
    }

    #[test]
    fn chamfer_box_all_edges() {
        assert_repeatable("chamfer box", |t| {
            let b = make_box(t, 10.0, 10.0, 10.0).unwrap();
            let edges = solid_edges(t, b).unwrap();
            chamfer(t, b, &edges, 1.5).unwrap()
        });
    }

    #[test]
    fn chamfer_asymmetric_box_all_edges() {
        assert_repeatable("chamfer_asymmetric box", |t| {
            let b = make_box(t, 10.0, 10.0, 10.0).unwrap();
            let edges = solid_edges(t, b).unwrap();
            chamfer_asymmetric(t, b, &edges, 1.5, 1.0).unwrap()
        });
    }

    #[test]
    fn chamfer_two_edges_beside_a_bore() {
        assert_repeatable("chamfer beside a bore", |t| {
            let s = holed_plate(t);
            let edges = edges_where(t, s, |t, e| {
                let d = run(t, e);
                on_level(t, e, 4.0) && d.y().abs() < 1e-9 && d.x().abs() > 1.0
            });
            assert_eq!(edges.len(), 2);
            chamfer(t, s, &edges, 0.5).unwrap()
        });
    }

    #[test]
    fn chamfer_bore_rims() {
        assert_repeatable("chamfer bore rims", |t| {
            let s = holed_plate(t);
            let rims = edges_where(t, s, is_circle);
            assert_eq!(rims.len(), 2);
            chamfer(t, s, &rims, 0.5).unwrap()
        });
    }

    #[test]
    fn chamfer_v2_box_upright_edges() {
        assert_repeatable("chamfer_v2 box", |t| {
            let b = make_box(t, 10.0, 10.0, 10.0).unwrap();
            let edges = edges_where(t, b, |t, e| {
                let d = run(t, e);
                d.x().abs() < 1e-9 && d.y().abs() < 1e-9
            });
            assert_eq!(edges.len(), 4);
            chamfer_v2(t, b, &edges, 1.5, 1.0).unwrap().solid
        });
    }

    #[test]
    fn chamfer_v2_cylinder_rims() {
        assert_repeatable("chamfer_v2 cylinder rims", |t| {
            let c = make_cylinder(t, 3.0, 5.0).unwrap();
            let edges = edges_where(t, c, is_circle);
            chamfer_v2(t, c, &edges, 0.5, 0.3).unwrap().solid
        });
    }

    #[test]
    fn fillet_v2_box_all_edges() {
        assert_repeatable("fillet_v2 box", |t| {
            let b = make_box(t, 20.0, 20.0, 20.0).unwrap();
            let edges = solid_edges(t, b).unwrap();
            fillet_v2(t, b, &edges, 1.0).unwrap().solid
        });
    }

    #[test]
    fn fillet_v2_box_corner_three_edges() {
        assert_repeatable("fillet_v2 box corner", |t| {
            let b = make_box(t, 10.0, 10.0, 10.0).unwrap();
            let corner = t.edge(solid_edges(t, b).unwrap()[0]).unwrap().start();
            let edges = edges_where(t, b, |t, e| {
                let e = t.edge(e).unwrap();
                e.start() == corner || e.end() == corner
            });
            assert_eq!(edges.len(), 3);
            fillet_v2(t, b, &edges, 1.0).unwrap().solid
        });
    }

    #[test]
    fn fillet_v2_l_block_all_edges() {
        assert_repeatable("fillet_v2 L block", |t| {
            let s = l_block(t);
            let edges = solid_edges(t, s).unwrap();
            fillet_v2(t, s, &edges, 0.5).unwrap().solid
        });
    }

    #[test]
    fn fillet_v2_holed_plate_top_edges() {
        assert_repeatable("fillet_v2 holed plate", |t| {
            let s = holed_plate(t);
            let edges = edges_where(t, s, |t, e| on_level(t, e, 4.0));
            fillet_v2(t, s, &edges, 0.5).unwrap().solid
        });
    }

    #[test]
    fn fillet_v2_cylinder_rims() {
        assert_repeatable("fillet_v2 cylinder rims", |t| {
            let c = make_cylinder(t, 3.0, 5.0).unwrap();
            let edges = edges_where(t, c, is_circle);
            fillet_v2(t, c, &edges, 0.5).unwrap().solid
        });
    }

    #[test]
    #[allow(deprecated)]
    fn fillet_bevel_box_all_edges() {
        assert_repeatable("fillet box", |t| {
            let b = make_box(t, 10.0, 10.0, 10.0).unwrap();
            let edges = solid_edges(t, b).unwrap();
            brepkit_operations::fillet::fillet(t, b, &edges, 1.0).unwrap()
        });
    }

    #[test]
    #[allow(deprecated)]
    fn fillet_rolling_ball_box_all_edges() {
        assert_repeatable("fillet_rolling_ball box", |t| {
            let b = make_box(t, 10.0, 10.0, 10.0).unwrap();
            let edges = solid_edges(t, b).unwrap();
            brepkit_operations::fillet::fillet_rolling_ball(t, b, &edges, 1.0).unwrap()
        });
    }
}
