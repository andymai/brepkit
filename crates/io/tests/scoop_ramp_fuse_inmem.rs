//! Two scoop ramps from the gridfinity tool's interior fillet
//! (`binGenerator.export.interiorFilletScoops.test.ts`, "a scoop on a raised
//! floor blends from its own floor"), fused into one cluster.
//!
//! Each ramp is a prism along x whose profile is a floor, a front lip, a
//! cubic scoop bulging back past its back wall, the back wall, a top and an
//! outer wall. The second ramp's profile is the first's moved 8 down, and the
//! two overlap by 0.96 along x, so each scoop crosses the other ramp's end
//! face. The fuse fell back to a mesh, which the tool's cluster fuse refuses.
//!
//! Data: `scoop_ramp_a.bin` (x -41.03 to 0.48, floor at z 9.25) and
//! `scoop_ramp_b.bin` (x -0.48 to 41.03, floor at z 1.25).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::measure::solid_volume;
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

/// The profiles' areas and their overlap's, integrated numerically from the
/// cubic (to 1e-6): the volumes are those areas times the lengths.
const AREA_A: f64 = 70.879_841;
const AREA_B: f64 = 85.919_841;
const OVERLAP_AREA: f64 = 45.401_47;

#[test]
fn scoop_ramps_fuse_cut_and_intersect_exactly() {
    let mut topo = Topology::new();
    let a = load(&mut topo, "scoop_ramp_a.bin");
    let b = load(&mut topo, "scoop_ramp_b.bin");
    let fused = exact(&mut topo, BooleanOp::Fuse, a, b);
    let cut = exact(&mut topo, BooleanOp::Cut, a, b);
    let common = exact(&mut topo, BooleanOp::Intersect, a, b);

    let (v_a, v_b) = (AREA_A * 41.51, AREA_B * 41.51);
    let overlap = OVERLAP_AREA * 0.96;
    // The mesh volume at deflection 0.001 reads the scoops' curved faces a
    // little off (fuse 0.81, cut 0.39, intersect 0.019 here); each allowance
    // is twice that.
    let volume = |s: SolidId| solid_volume(&topo, s, 0.001).unwrap();
    for (label, got, want, allowance) in [
        ("fuse", volume(fused), v_a + v_b - overlap, 1.6),
        ("cut", volume(cut), v_a - overlap, 0.8),
        ("intersect", volume(common), overlap, 0.04),
    ] {
        assert!(
            (got - want).abs() <= allowance,
            "{label} volume {got}, expected {want}"
        );
    }
}

/// A point above the second ramp's scoop, on the first ramp's end face, lies
/// outside the second ramp. The classifier read the curved scoop through a
/// flat polygon on its boundary, which two of three rays crossed.
#[test]
fn a_point_above_a_scoop_is_outside_its_ramp() {
    let mut topo = Topology::new();
    let b = load(&mut topo, "scoop_ramp_b.bin");
    let geoms = brepkit_algo::classifier::RayCastGeoms::new(&topo, b).unwrap();
    let class = brepkit_algo::classifier::classify_ray_cast_cached(
        &geoms,
        Point3::new(0.48, -14.95, 11.801),
    )
    .unwrap();
    assert_eq!(class, brepkit_algo::FaceClass::Outside);
}
