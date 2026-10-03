//! The gridfinity tool's split bins
//! (`binGenerator.scenario.split-robustness-topology.test.ts`), from operands
//! captured from its calls.
//!
//! A bin with two rows of two compartments clips its four scoop ramps, one per
//! compartment, to its envelope. Each ramp is a block whose curve is a run of
//! planar facets along x, two against the front wall and two against the row
//! divider, and the envelope is a rounded box whose sides lie in the ramps' end
//! faces and whose r = 2.63 corner cylinders trim the two front ramps' square
//! corners. A facet's lower edge meets a corner cylinder 12.8 microns short of
//! the ramp's end, and the face splitter skipped split points whose parameter
//! lay within the tolerance of an end, a length read as a fraction: on the
//! 166 mm edge any point within 16.6 microns. The cylinder's section stayed a
//! pendant, was bridged to the corner along the edge itself, and the sliver
//! past the cylinder never separated: eight free edges. The clip fell back to a
//! mesh with 27 free edges, and the split chain after it ran mesh against mesh
//! (one fuse took 73 s).
//!
//! Data: `binsplit_scoops.bin` (the four ramps) and `binsplit_envelope.bin`
//! (the envelope).

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
