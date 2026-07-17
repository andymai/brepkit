//! Guard for the deepened wall-opening union (the plane-face stranded-rim
//! case, the snapClip join-edges export root).
//!
//! The snap-slot cutter stack cuts the same seam wall twice: an earlier box
//! opens a rectangular window in the wall, and a later, deeper box re-opens
//! an overlapping window offset by 0.01 (its top floats 0.01 above the
//! earlier slot's floor). The later cutter's sections form a closed internal
//! loop on the wall face that OVERLAPS the existing inner wire; the
//! internal-loops splitter used to attach the loop as an independent second
//! hole, double-covering the 0.01 band — both rims stayed as unpaired edges
//! and the collinear band pieces traced twice (use=3 micro verticals), the
//! analytic gate rejected the cut, and the whole 44-hole snapClip chain fell
//! back to mesh booleans (F=615 → 8203 at op-cut-3).
//!
//! The union pre-pass in `split_face_with_internal_loops` now merges the pair:
//! the wall keeps ONE union outline and the removable piece is bounded by the
//! loop pieces outside the hole plus the hole pieces inside the loop.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;

use brepkit_algo::bop::BooleanOp;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::primitives::make_box;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn boxed(topo: &mut Topology, lo: [f64; 3], hi: [f64; 3]) -> SolidId {
    let s = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
    transform_solid(topo, s, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
    s
}

type Q = (i64, i64, i64);

fn pos_bad_edges(topo: &Topology, solid: SolidId) -> usize {
    let sc = 1.0e5;
    let q = |p: Point3| -> Q {
        (
            (p.x() * sc).round() as i64,
            (p.y() * sc).round() as i64,
            (p.z() * sc).round() as i64,
        )
    };
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    for &fid in &solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                *occ.entry(key).or_default() += 1;
            }
        }
    }
    occ.values().filter(|&&c| c != 2).count()
}

#[test]
fn deepened_wall_opening_merges_overlapping_hole() {
    let mut topo = Topology::new();
    let base = boxed(&mut topo, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);

    // First cut: tunnel through the x=10 wall, floor at z=4.
    let a = boxed(&mut topo, [5.0, 3.0, 4.0], [11.0, 7.0, 6.0]);
    let s1 = brepkit_algo::gfa::boolean(&mut topo, BooleanOp::Cut, base, a).unwrap();
    assert_eq!(pos_bad_edges(&topo, s1), 0, "first cut must be watertight");

    // Second cut: deeper opening through the same wall whose top floats 0.01
    // above the first tunnel's floor (the snap-slot ledge margin).
    let b = boxed(&mut topo, [5.0, 3.0, 1.0], [11.0, 7.0, 4.01]);
    let s2 = brepkit_algo::gfa::boolean(&mut topo, BooleanOp::Cut, s1, b).unwrap();

    assert_eq!(
        pos_bad_edges(&topo, s2),
        0,
        "stranded rims: wall kept both opening outlines over the 0.01 band"
    );

    // The wall must carry ONE merged opening, not two overlapping wires.
    let wall_holes: Vec<usize> = solid_faces(&topo, s2)
        .unwrap()
        .iter()
        .filter_map(|&fid| {
            let face = topo.face(fid).ok()?;
            let wall_eps = brepkit_math::tolerance::Tolerance::new().linear * 100.0;
            let on_wall = std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .flat_map(|wid| topo.wire(wid).unwrap().edges().to_vec())
                .all(|oe| {
                    let e = topo.edge(oe.edge()).unwrap();
                    (topo.vertex(e.start()).unwrap().point().x() - 10.0).abs() < wall_eps
                        && (topo.vertex(e.end()).unwrap().point().x() - 10.0).abs() < wall_eps
                });
            (on_wall && !face.inner_wires().is_empty()).then_some(face.inner_wires().len())
        })
        .collect();
    assert_eq!(wall_holes, vec![1], "wall must have exactly one union hole");

    // Volume: 1000 - 40 (tunnel) - 60.2 (deep cut) + 0.2 (their overlap).
    let vol = brepkit_operations::measure::solid_volume(&topo, s2, 0.01).unwrap();
    assert!(
        (vol - 900.0).abs() < 0.05,
        "volume {vol} deviates from expected 900"
    );
}
