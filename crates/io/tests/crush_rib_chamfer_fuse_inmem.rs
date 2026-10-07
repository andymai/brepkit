//! A magnet pocket with crush ribs fused with its entry chamfer, from the
//! gridfinity tool (`binGenerator.scenario.baseStyles.test.ts`, "magnet/screw
//! base with crush ribs and chamfer"): the pocket's four NURBS walls bulge out
//! to the chamfer cone's top rim at z 0.8, each touching it once.
//!
//! One rib touches the rim 1.25e-3 from the rim edge's vertex, which also
//! lies on that rib's wall. EF's crossing search found the touch inside its
//! bracket, but taking whichever of that point and the bracket's ends lay
//! nearer the wall moved it onto the vertex, where it was dropped as the
//! vertex's own contact: the rim lost its fourth touch point and the fuse
//! failed the Euler check and fell back to a mesh.
//!
//! Data: `crush_rib_pocket.bin` and `crush_rib_chamfer.bin`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::measure::oriented_solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn exact(topo: &mut Topology, op: BooleanOp, a: SolidId, b: SolidId, watertight: bool) -> SolidId {
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
    assert!(
        !watertight || is_watertight(&tessellate_solid(topo, result, 0.01).unwrap()),
        "{op:?}: mesh not watertight"
    );
    result
}

#[test]
fn a_crush_rib_pocket_and_its_chamfer_keep_every_rim_touch() {
    let mut topo = Topology::new();
    let pocket = load(&mut topo, "crush_rib_pocket.bin");
    let chamfer = load(&mut topo, "crush_rib_chamfer.bin");
    let volume = |topo: &Topology, s| oriented_solid_volume(topo, s, 0.001).unwrap();
    let (v_pocket, v_chamfer) = (volume(&topo, pocket), volume(&topo, chamfer));

    // The four touches pinch the fuse's rim annulus to the ribs at single
    // points: its mesh is not watertight there, validation reads the pinches
    // as an Euler defect, and its volume reads 2e-4 high.
    let fused = exact(&mut topo, BooleanOp::Fuse, pocket, chamfer, false);
    let common = exact(&mut topo, BooleanOp::Intersect, pocket, chamfer, true);
    let cut = exact(&mut topo, BooleanOp::Cut, pocket, chamfer, true);

    let (v_fused, v_common, v_cut) = (
        volume(&topo, fused),
        volume(&topo, common),
        volume(&topo, cut),
    );
    assert!(
        (v_cut + v_common - v_pocket).abs() / v_pocket < 1e-4,
        "cut {v_cut:.4} + intersect {v_common:.4} != pocket {v_pocket:.4}"
    );
    let parts = v_pocket + v_chamfer - v_common;
    assert!(
        (v_fused - parts).abs() / parts < 5e-4,
        "fuse {v_fused:.4} != pocket + chamfer - intersect {parts:.4}"
    );
}
