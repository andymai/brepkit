//! The gridfinity tool's hinged lid and bin (`hingeSwing.scenario.test.ts`,
//! "swings clear of the bin on the back wall"), captured on a brepkit-wasm
//! built from #1935's branch.
//!
//! The lid: compound-cut by its clearance bevel and four knuckle bores, the
//! scenario's first boolean to fall back to a mesh. Every later op then ran
//! against that blob: the three knuckle fuses fell back in turn and each
//! swing-pose intersect took 22 s in wasm. Each bore (radius 2.45 along `y`)
//! has its axis on the lid's pocket ceiling `z = -3.2`, crosses the ceiling's
//! edge at the pocket wall `x = -59`, and pokes 0.05 past the back face
//! `x = -62.75`, which the ceiling's plane splits in two.
//!
//! The bin: compound-cut by five clearance rods that only touch its lip. The
//! result came back exact but with one face inside out, which every later
//! boolean on the bin inherited.
//!
//! Data: `hinge_lid.bin` (the lid), `hinge_lid_clearance.bin` (the bevel's
//! box), `hinge_lid_bore_<1..4>.bin`, `hinge_lid_knuckle.bin` (the first
//! knuckle the tool fuses onto the cut lid), `hinge_bin.bin` and
//! `hinge_bin_clearance_<1..5>.bin`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// The material one bore removes: its disc less the sliver past the back face
/// and the quarter in the pocket below the ceiling, over the bore's length.
fn bore_truth() -> f64 {
    let r: f64 = 2.45;
    let segment = |h: f64| r * r * (h / r).acos() - h * (r * r - h * h).sqrt();
    let area = std::f64::consts::PI * r * r - segment(2.4) - segment(1.35) / 2.0;
    area * (37.8 - 26.571_428_571_428_57)
}

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn tools(topo: &mut Topology) -> Vec<SolidId> {
    std::iter::once("hinge_lid_clearance.bin".to_string())
        .chain((1..=4).map(|i| format!("hinge_lid_bore_{i}.bin")))
        .map(|name| load(topo, &name))
        .collect()
}

fn volume(topo: &Topology, solid: SolidId) -> f64 {
    solid_volume(topo, solid, 0.001).unwrap()
}

/// Runs `op`, requiring it to stay exact and return a valid solid.
fn exact(topo: &mut Topology, op: impl FnOnce(&mut Topology) -> SolidId) -> SolidId {
    let before = boolean::mesh_fallback_count();
    let result = op(topo);
    assert_eq!(boolean::mesh_fallback_count(), before, "mesh fallback");
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    result
}

#[test]
fn hinge_lid_fixture_is_faithful() {
    let mut topo = Topology::new();
    let lid = load(&mut topo, "hinge_lid.bin");
    let mut census: HashMap<&str, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(&topo, lid).unwrap() {
        *census
            .entry(topo.face(fid).unwrap().surface().type_tag())
            .or_insert(0) += 1;
    }
    assert_eq!(
        ["cone", "cylinder", "plane"].map(|k| census.get(k).copied().unwrap_or(0)),
        [4, 16, 24],
        "fixture drifted: {census:?}"
    );
    assert!(validate_solid(&topo, lid).unwrap().is_valid());
}

/// The back face's two halves each took the bore cap's arc past the face as
/// a piece of themselves (it sags 0.05 off the plane, a twentieth of its
/// chord, and ends on the other half), and the ceiling never met the caps:
/// their section ran 1.1 inside both faces, under the filter's sampling step.
#[test]
fn hinge_lid_bore_cut_is_exact() {
    let mut topo = Topology::new();
    let lid = load(&mut topo, "hinge_lid.bin");
    let bore = load(&mut topo, "hinge_lid_bore_1.bin");
    let cut = exact(&mut topo, |t| {
        boolean::boolean(t, BooleanOp::Cut, lid, bore).unwrap()
    });
    let removed = volume(&topo, lid) - volume(&topo, cut);
    assert!(
        (removed - bore_truth()).abs() < 1e-3,
        "removed {removed}, truth {}",
        bore_truth()
    );
}

/// The tool's call: all five tools at once, through the default options,
/// which unify same-domain faces afterwards. The unify step merged the lid's
/// stacked corner cylinders into faces whose wires listed their edges out of
/// order.
#[test]
fn hinge_lid_compound_cut_is_exact() {
    let mut topo = Topology::new();
    let lid = load(&mut topo, "hinge_lid.bin");
    let tools = tools(&mut topo);
    let compound = exact(&mut topo, |t| {
        boolean::compound_cut(t, lid, &tools, BooleanOptions::default()).unwrap()
    });
    let mut sequential = lid;
    for &tool in &tools {
        sequential = exact(&mut topo, |t| {
            boolean::boolean(t, BooleanOp::Cut, sequential, tool).unwrap()
        });
    }
    let (vc, vs) = (volume(&topo, compound), volume(&topo, sequential));
    assert!(
        (vc - vs).abs() < 1e-6 * vs,
        "compound {vc}, sequential {vs}"
    );
}

/// The tool's next call: the first knuckle fused onto the cut lid. The
/// knuckle's axis lies on the ceiling too, and its flat top runs in the
/// bevel's plane.
#[test]
fn hinge_lid_knuckle_fuse_is_exact() {
    let mut topo = Topology::new();
    let lid = load(&mut topo, "hinge_lid.bin");
    let tools = tools(&mut topo);
    let cut = exact(&mut topo, |t| {
        boolean::compound_cut(t, lid, &tools, BooleanOptions::default()).unwrap()
    });
    let knuckle = load(&mut topo, "hinge_lid_knuckle.bin");
    let fused = exact(&mut topo, |t| {
        boolean::boolean(t, BooleanOp::Fuse, cut, knuckle).unwrap()
    });
    let outside = exact(&mut topo, |t| {
        boolean::boolean(t, BooleanOp::Cut, knuckle, cut).unwrap()
    });
    let expected = volume(&topo, cut) + volume(&topo, outside);
    let got = volume(&topo, fused);
    assert!(
        (got - expected).abs() < 1e-6 * expected,
        "fused {got}, lid plus the knuckle past it {expected}"
    );
}

/// The unify step after the cut merged the two halves of a reversed strip on
/// the bin's front lip into a face whose wire ran the other way round, so its
/// four edges ran the same way as its neighbours'.
#[test]
fn hinge_bin_clearance_cut_stays_valid() {
    let mut topo = Topology::new();
    let bin = load(&mut topo, "hinge_bin.bin");
    let rods: Vec<_> = (1..=5)
        .map(|i| load(&mut topo, &format!("hinge_bin_clearance_{i}.bin")))
        .collect();
    let cut = exact(&mut topo, |t| {
        boolean::compound_cut(t, bin, &rods, BooleanOptions::default()).unwrap()
    });
    let (before, after) = (volume(&topo, bin), volume(&topo, cut));
    assert!(
        (before - after).abs() < 1e-6 * before,
        "the rods only touch the lip: {before} became {after}"
    );
}
