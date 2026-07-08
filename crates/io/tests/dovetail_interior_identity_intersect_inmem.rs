//! Faithful guard for the interior dovetail tile corner-rounding intersect.
//!
//! The gridfinity tool's `baseplateGenerator.scenario.dovetail.test.ts`
//! interior/middle-column tiles (all-join edges) exported STLs with
//! bnd=108/144 boundary edges. Captured stage probes localized the defect to
//! the `cornerClipIntersect` step: for an interior tile every corner is a
//! join edge, so the rounding profile degenerates to a plain 6-plane box
//! exactly matching the slab's outer bounds — the intersect is an identity
//! with FULLY COINCIDENT walls.
//!
//! The tool's export path runs `boolean_with_evolution` (face provenance for
//! feature tags), whose faithful raw-GFA branch mis-split that configuration:
//! 134 input faces collapsed to 38 with 32 free edges, and the result passed
//! `validate_boolean_result` because the coincident-wall drop leaves
//! position-duplicate free edges that the by-edge-id check (every id used
//! ≤ 2×) cannot see. `boolean()` was immune — its identical/containment
//! shortcut returns a copy. Fix: `boolean_with_evolution` now consults the
//! same `detect_trivial_relation` and routes identical/contained pairs
//! through `boolean()`'s shortcuts (the geometry-heuristic evolution maps a
//! copied result's faces exactly).
//!
//! Fixtures are the tool's EXACT serialized operands for the 4×4 interior
//! tile: `_slab` = the pocketed slab (134 faces, 64 cones), `_round` = the
//! degenerate all-plane rounding box.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean, boolean_with_evolution};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    brepkit_io::arena_io::deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

fn free_edges(topo: &Topology, solid: SolidId) -> usize {
    type Q = (i64, i64, i64);
    let s = 1.0e5;
    let q = |p: Point3| -> Q {
        (
            (p.x() * s).round() as i64,
            (p.y() * s).round() as i64,
            (p.z() * s).round() as i64,
        )
    };
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
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
    occ.values().filter(|&&c| c == 1).count()
}

fn check_identity_result(topo: &Topology, result: SolidId, label: &str) {
    let faces = solid_faces(topo, result).unwrap();
    let cones = faces
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() == "cone")
        .count();
    let free = free_edges(topo, result);
    assert_eq!(
        free,
        0,
        "{label}: identity intersect must be watertight; got {} faces, {cones} cones",
        faces.len()
    );
    assert_eq!(
        faces.len(),
        134,
        "{label}: identity intersect must preserve all 134 slab faces"
    );
    assert_eq!(
        cones, 64,
        "{label}: identity intersect must preserve all 64 pocket cones"
    );
}

#[test]
fn interior_tile_identity_intersect_via_boolean() {
    let mut topo = Topology::new();
    let slab = load("dovetail_interior_slab.bin", &mut topo);
    let round = load("dovetail_interior_round.bin", &mut topo);
    let result = boolean(&mut topo, BooleanOp::Intersect, slab, round).unwrap();
    check_identity_result(&topo, result, "boolean");
}

#[test]
fn interior_tile_identity_intersect_via_evolution() {
    let mut topo = Topology::new();
    let slab = load("dovetail_interior_slab.bin", &mut topo);
    let round = load("dovetail_interior_round.bin", &mut topo);
    let (result, evo) =
        boolean_with_evolution(&mut topo, BooleanOp::Intersect, slab, round).unwrap();
    check_identity_result(&topo, result, "boolean_with_evolution");
    // The copied result's faces must still carry provenance (the geometry
    // heuristic maps a copy 1:1) — feature tags depend on it.
    assert!(
        !evo.modified.is_empty(),
        "evolution map must attribute the copied faces"
    );
}
