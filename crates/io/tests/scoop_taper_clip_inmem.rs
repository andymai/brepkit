//! A scoop clipped to a bin's tapered envelope, from the gridfinity tool's
//! interior fillet (`binGenerator.export.interiorFilletScoops.test.ts`, "a
//! scoop beside tapered side walls keeps the plain fillet").
//!
//! The scoop is a prism along x over [-44.03, 44.03] whose profile is a
//! floor at z 1.25, a lip, a cubic scoop up to the back wall at y -18.15,
//! the back wall, a top at z 23.25 and the outer wall at y -20.03. The
//! envelope is a box over the bin's footprint (x within 44.03, y within
//! 20.03, z 0 to 25.25) with r 3.03 vertical corners, whose x walls lean
//! in below z 6 to x 41.03 at z 0 (the corners there are oblique
//! extrusions of a quarter circle). The scoop's end profiles lie in the
//! envelope's upright x walls, and each passes 0.05 above the corner where
//! the wall, its leaning foot and the corner cylinder meet, touching the
//! cylinder there along its ruling.
//!
//! The clip fell back to a mesh, which every later boolean of the test's
//! fillet then took as an operand.
//!
//! The tool then fuses the clipped scoop into its bin, whose pocket (half
//! sizes 43.55 by 19.55, corners r 2.55, floor at z 2.25, its x walls leaning
//! in below z 6 like the envelope's, a stacking lip narrowing it above z
//! 20.65) lies inside the envelope, so the scoop's part outside the pocket is
//! inside the bin's walls and floor.
//!
//! Data: `taper_clip_scoop.bin` and `taper_clip_envelope.bin`; for the fuse,
//! `taper_scoop_bin.bin` and `taper_scoop_clipped.bin`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn exact(topo: &mut Topology, op: BooleanOp, a: SolidId, b: SolidId) -> SolidId {
    let fallbacks = boolean::mesh_fallback_count();
    let result = boolean::boolean(topo, op, a, b).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        fallbacks,
        "{op:?} fell back"
    );
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, result).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    assert!(
        uses.values().all(|&n| n == 2),
        "{op:?} left free or over-shared edges"
    );
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{op:?}: {:?}", report.issues);
    assert!(
        is_watertight(&tessellate_solid(topo, result, 0.01).unwrap()),
        "{op:?}: mesh not watertight"
    );
    result
}

/// Integrated numerically (to 1e-3): the scoop's profile area times its
/// length, the envelope's horizontal sections up its height, and the
/// scoop's sections times the envelope's x extent at each (y, z).
const SCOOP: f64 = 5_248.357;
const ENVELOPE: f64 = 88_153.937;
const COMMON: f64 = 5_078.933;

#[test]
fn a_scoop_clipped_by_a_tapered_envelope_keeps_its_closed_form() {
    let mut topo = Topology::new();
    let scoop = load(&mut topo, "taper_clip_scoop.bin");
    let envelope = load(&mut topo, "taper_clip_envelope.bin");
    let common = exact(&mut topo, BooleanOp::Intersect, scoop, envelope);
    let cut = exact(&mut topo, BooleanOp::Cut, scoop, envelope);
    let fused = exact(&mut topo, BooleanOp::Fuse, scoop, envelope);

    // The mesh volume at deflection 0.001 reads the curved faces a little
    // off (intersect 0.45, cut 0.15, fuse 0.17 here); each allowance is
    // twice that.
    let volume = |s: SolidId| solid_volume(&topo, s, 0.001).unwrap();
    for (label, got, want, allowance) in [
        ("intersect", volume(common), COMMON, 0.9),
        ("cut", volume(cut), SCOOP - COMMON, 0.3),
        ("fuse", volume(fused), SCOOP + ENVELOPE - COMMON, 0.35),
    ] {
        assert!(
            (got - want).abs() <= allowance,
            "{label} volume {got}, expected {want}"
        );
    }
}

/// The clipped scoop's part inside the pocket, integrated numerically (to
/// 1e-3) from the cubic and the pocket's sections, lip included.
const SCOOP_IN_POCKET: f64 = 3_023.950;

#[test]
fn a_clipped_scoop_fuses_into_its_tapered_bin_exactly() {
    let mut topo = Topology::new();
    let bin = load(&mut topo, "taper_scoop_bin.bin");
    let scoop = load(&mut topo, "taper_scoop_clipped.bin");
    let fused = exact(&mut topo, BooleanOp::Fuse, bin, scoop);
    let volume = |s: SolidId| solid_volume(&topo, s, 0.001).unwrap();
    let added = volume(fused) - volume(bin);
    assert!(
        (added - SCOOP_IN_POCKET).abs() <= 2e-4 * SCOOP_IN_POCKET,
        "the fuse adds {added}, expected {SCOOP_IN_POCKET}"
    );
    // The mesh's own volume agrees: no face meshes inside out.
    let meshed = oriented_solid_volume(&topo, fused, 0.001).unwrap();
    assert!(
        (meshed - volume(fused)).abs() <= 2e-4 * SCOOP_IN_POCKET,
        "the fuse meshes to {meshed}"
    );
}

/// The scoop's part in the bin's walls and floor. Its cubic reaches the
/// pocket's upright x walls in a 0.04 by 0.05 sliver at their corners, which
/// the classifier read against a flat polygon through the trimmed cubic's
/// wires. (The cut is not a manifold solid: the cubic dips below the floor
/// and rises back to it at the lip, so two wedges of floor touch along the
/// lip's line.)
#[test]
fn a_clipped_scoop_and_its_tapered_bin_intersect_exactly() {
    let mut topo = Topology::new();
    let bin = load(&mut topo, "taper_scoop_bin.bin");
    let scoop = load(&mut topo, "taper_scoop_clipped.bin");
    let common = exact(&mut topo, BooleanOp::Intersect, bin, scoop);
    let got = solid_volume(&topo, common, 0.001).unwrap();
    let want = COMMON - SCOOP_IN_POCKET;
    assert!(
        (got - want).abs() <= 2e-4 * SCOOP_IN_POCKET,
        "intersect volume {got}, expected {want}"
    );
}

/// The test's fillet material (the bin's walls and floor: half sizes 44.03
/// by 20.03 with r 3.03 corners, z 1.25 to 22.55, less a pocket with half
/// sizes 43.55 by 19.55, r 2.55 corners and its floor at z 2.25 rounded r
/// 2.45 by cylinders and spindle tori) against the clipped scoop. The
/// scoop's cubic bulges past the pocket's corner cylinders between two
/// crossings of their seam line, its envelope's leaning corners cross the
/// spindle tori's tubes, and its front taper lies flush on the material's
/// front wall. (The material less the scoop is the floor-wedge non-manifold
/// solid above.)
#[test]
fn the_fillet_material_and_the_clipped_scoop_meet_exactly() {
    let mut topo = Topology::new();
    let material = load(&mut topo, "taper_scoop_fillet_material.bin");
    let scoop = load(&mut topo, "taper_scoop_fillet_scoop.bin");
    let common = exact(&mut topo, BooleanOp::Intersect, material, scoop);
    let common_swapped = exact(&mut topo, BooleanOp::Intersect, scoop, material);
    let fused = exact(&mut topo, BooleanOp::Fuse, material, scoop);
    let scoop_only = exact(&mut topo, BooleanOp::Cut, scoop, material);

    // At deflection 0.001 the fuse and the scoop cut both read 0.37 below
    // these sums, the scoop's own volume reading that much high; the
    // allowance is twice that.
    let volume = |s: SolidId| solid_volume(&topo, s, 0.001).unwrap();
    let (a, b, c) = (volume(material), volume(scoop), volume(common));
    for (label, got, want) in [
        ("swapped intersect", volume(common_swapped), c),
        ("fuse", volume(fused), a + b - c),
        ("scoop cut", volume(scoop_only), b - c),
    ] {
        assert!(
            (got - want).abs() <= 0.75,
            "{label} volume {got}, expected {want}"
        );
    }
    let meshed = oriented_solid_volume(&topo, fused, 0.001).unwrap();
    assert!(
        (meshed - volume(fused)).abs() <= 0.75,
        "the fuse meshes to {meshed}"
    );

    // Points the operands place on either side: in the front wall behind
    // the scoop (up high and down at the floor), in a side wall clear of the
    // scoop, in the scoop above the wall top, and in the pocket's air.
    let inside_of = |s: SolidId, p: Point3| {
        classify_point(&topo, s, p, 0.01, 1e-7).unwrap() == PointClassification::Inside
    };
    for (p, in_material, in_scoop) in [
        (Point3::new(0.0, -19.8, 12.0), true, true),
        (Point3::new(0.0, -19.8, 3.0), true, true),
        (Point3::new(-43.8, 10.0, 12.0), true, false),
        (Point3::new(0.0, -19.0, 23.0), false, true),
        (Point3::new(0.0, 0.0, 10.0), false, false),
    ] {
        assert_eq!(
            (inside_of(material, p), inside_of(scoop, p)),
            (in_material, in_scoop),
            "operands at {p:?}"
        );
        assert_eq!(
            inside_of(common, p),
            in_material && in_scoop,
            "intersect at {p:?}"
        );
        assert_eq!(
            inside_of(fused, p),
            in_material || in_scoop,
            "fuse at {p:?}"
        );
        assert_eq!(
            inside_of(scoop_only, p),
            in_scoop && !in_material,
            "scoop cut at {p:?}"
        );
    }
}

/// The same scoop against the fillet material once its rim is rounded: the
/// envelope's leaning corner is tangent to the material's front wall along
/// a ruling, and the march re-traces only part of that ruling with stubs at
/// its foot, which split the wall off the ruling's own edge.
#[test]
fn the_rounded_fillet_material_and_the_clipped_scoop_meet_exactly() {
    let mut topo = Topology::new();
    let material = load(&mut topo, "rounded_fillet_material.bin");
    let scoop = load(&mut topo, "rounded_fillet_scoop.bin");
    let common = exact(&mut topo, BooleanOp::Intersect, material, scoop);
    let common_swapped = exact(&mut topo, BooleanOp::Intersect, scoop, material);
    let fused = exact(&mut topo, BooleanOp::Fuse, material, scoop);
    let scoop_only = exact(&mut topo, BooleanOp::Cut, scoop, material);

    // The scoop's NURBS faces mesh up to 0.5 off at deflection 0.001.
    let volume = |s: SolidId| oriented_solid_volume(&topo, s, 0.001).unwrap();
    let (a, b, c) = (volume(material), volume(scoop), volume(common));
    for (label, got, want) in [
        ("swapped intersect", volume(common_swapped), c),
        ("fuse", volume(fused), a + b - c),
        ("scoop cut", volume(scoop_only), b - c),
        ("fuse less scoop cut", volume(fused) - volume(scoop_only), a),
    ] {
        assert!(
            (got - want).abs() <= 0.75,
            "{label} volume {got}, expected {want}"
        );
    }
}
