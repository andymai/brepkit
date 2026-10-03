//! The gridfinity tool's split bins
//! (`binGenerator.scenario.split-robustness-topology.test.ts`), from operands
//! captured from its calls.
//!
//! One piece of a bin split across x takes its stacking lip in one fuse. The
//! piece's walls are 1.2 thick and end at z = 21, with r = 2.55 inner corners
//! about (-122, ±80). The lip arrives as two corner pieces, each standing on
//! an L-shaped ring at z = 18.4 along the wall top, its inner rim on those
//! corner arcs. The inner wall planes x = -124.55 and y = ±82.55 meet the
//! ring's plane in lines that run along the ring's inner rim, then on past the
//! corner arc to the square corner the walls would make. Both faces carry arcs
//! on their outlines (between the lip pieces each wall dips to z = 13.475
//! through r = 3.75 rounded corners, and the ring turns its corners on arcs),
//! so neither polygon clip could trim the line, and the overshoot cut a sliver
//! off the ring between the arc and that square corner: two free edges and two
//! edges on three faces.
//!
//! A bin with two rows of two compartments clips its four scoop ramps, one per
//! compartment, to its envelope. Each ramp is a block whose curve is a run of
//! planar facets along x, two against the front wall and two against the row
//! divider, and the envelope is a rounded box whose sides lie in the ramps' end faces
//! and whose r = 2.63 corner cylinders trim the two front ramps' square
//! corners. A facet's lower edge meets a corner cylinder 12.8 microns short of
//! the ramp's end, and the face splitter skipped split points whose parameter
//! lay within the tolerance of an end, a length read as a fraction: on the
//! 166 mm edge any point within 16.6 microns. The cylinder's section stayed
//! a pendant, was bridged to the corner along the edge itself, and the sliver
//! past the cylinder never separated: eight free edges.
//!
//! Both fell back to meshes, and the split chain after them ran mesh against
//! mesh.
//!
//! Data: `binsplit_piece.bin` (the piece) and `binsplit_piece_lips.bin` (the
//! two lip pieces), `binsplit_scoops.bin` (the four ramps) and
//! `binsplit_envelope.bin` (the envelope).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

/// Runs `op`, requiring it to stay exact and return a valid solid.
fn exact(topo: &mut Topology, op: BooleanOp, a: SolidId, b: SolidId) -> SolidId {
    let before = boolean::mesh_fallback_count();
    let result = boolean::boolean(topo, op, a, b).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        before,
        "{op:?} mesh fallback"
    );
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{op:?}: {:?}", report.issues);
    result
}

fn inside(topo: &Topology, solid: SolidId, p: [f64; 3]) -> Option<bool> {
    let q = brepkit_math::vec::Point3::new(p[0], p[1], p[2]);
    match classify_point(topo, solid, q, 0.01, 1e-7).ok()? {
        PointClassification::Inside => Some(true),
        PointClassification::Outside => Some(false),
        PointClassification::OnBoundary => None,
    }
}

/// `result` holds each grid point exactly where `expect` says, over the
/// points all three solids classify off their boundaries; returns how many.
fn matches_on_grid(
    topo: &Topology,
    result: SolidId,
    (a, b): (SolidId, SolidId),
    expect: impl Fn(bool, bool) -> bool,
    points: impl Iterator<Item = [f64; 3]>,
) -> usize {
    let mut checked = 0;
    for p in points {
        let (Some(r), Some(ia), Some(ib)) = (
            inside(topo, result, p),
            inside(topo, a, p),
            inside(topo, b, p),
        ) else {
            continue;
        };
        assert_eq!(r, expect(ia, ib), "at {p:?}");
        checked += 1;
    }
    checked
}

#[test]
fn split_bin_piece_takes_its_lip_exactly() {
    let mut topo = Topology::new();
    let piece = load(&mut topo, "binsplit_piece.bin");
    let lips = load(&mut topo, "binsplit_piece_lips.bin");
    let fused = exact(&mut topo, BooleanOp::Fuse, piece, lips);

    // The fuse is the piece less the lips plus the lips, a disjoint union.
    let cut = exact(&mut topo, BooleanOp::Cut, piece, lips);
    let volume = |s| solid_volume(&topo, s, 0.01).unwrap();
    let (vf, vc, vl) = (volume(fused), volume(cut), volume(lips));
    assert!(
        (vf - vc - vl).abs() < 1e-6 * vf,
        "fuse {vf}, cut {vc} + lips {vl}"
    );

    // Both corners, through the ring, the wall top and the lip above it.
    let grid = [-1.0, 1.0].into_iter().flat_map(|sign| {
        (0..7).flat_map(move |i| {
            (0..7).flat_map(move |j| {
                [18.25, 18.6, 19.5, 20.8, 21.3].map(|z| {
                    [
                        -125.6 + 0.7 * f64::from(i),
                        sign * (83.6 - 0.7 * f64::from(j)),
                        z,
                    ]
                })
            })
        })
    });
    let checked = matches_on_grid(&topo, fused, (piece, lips), |a, b| a || b, grid);
    assert!(checked > 400, "only {checked} points classified");
}

#[test]
fn split_bin_scoops_clip_to_the_envelope_exactly() {
    let mut topo = Topology::new();
    let scoops = load(&mut topo, "binsplit_scoops.bin");
    let envelope = load(&mut topo, "binsplit_envelope.bin");
    let common = exact(&mut topo, BooleanOp::Intersect, scoops, envelope);

    // The clip and what it trims make up the ramps.
    let trimmed = exact(&mut topo, BooleanOp::Cut, scoops, envelope);
    let volume = |s| solid_volume(&topo, s, 0.01).unwrap();
    let (vi, vc, vs) = (volume(common), volume(trimmed), volume(scoops));
    assert!(
        (vi + vc - vs).abs() < 1e-6 * vs,
        "common {vi} + trimmed {vc}, ramps {vs}"
    );

    // Both front corners, through the facets the corner cylinders trim.
    let grid = [-1.0, 1.0].into_iter().flat_map(|sign| {
        (0..8).flat_map(move |i| {
            (0..8).flat_map(move |j| {
                [2.0, 8.0, 11.9, 12.2, 16.0, 22.5].map(|z| {
                    [
                        sign * (166.6 - 0.4 * f64::from(i)),
                        -82.6 + 0.4 * f64::from(j),
                        z,
                    ]
                })
            })
        })
    });
    let checked = matches_on_grid(&topo, common, (scoops, envelope), |a, b| a && b, grid);
    assert!(checked > 500, "only {checked} points classified");
}
