//! Faithful repro: the 2×1 compartmented-bin EXPORT goes non-manifold (#1753).
//!
//! Captured the live brepkit boolean operands (JS monkey-patch over
//! `BrepKernel.fuseWithEvolution` + `serializeSolid`, #915) at the FIRST
//! over-shared result while running the tool's `compartmentBuilder.scenario.
//! manifold` 2×1 case. The over-share is NOT the compartment-cavity cut (that
//! step is clean, free=0/over=0 — see `gridfinity_cavitycut_inmem.rs`) but the
//! FINAL socket-into-body Fuse done at export time:
//!   - `_body.bin`  : the featured bin body (box + cavities + lip), 94 faces
//!   - `_socket.bin`: the 2×2 base socket, 200 faces
//!
//! Root cause (diagnosed, NOT a weld near-miss): the body and socket meet on
//! the coplanar floor at z=5, but their OUTER CORNERS differ by design — the
//! box uses `BOX_CORNER_RADIUS = 3.75` (a cylinder arc) while the socket uses
//! `SOCKET_CORNER_RADIUS = 4` (a chamfer). At each of the 4 corners the body
//! footprint overhangs the socket by ~0.1mm, so the contact is a PARTIAL
//! coplanar same-domain overlap with mismatched curved corner boundaries.
//! `gfa::boolean` dissolves the straight perimeter contact correctly but
//! leaves 20 free corner edges (5 per corner): the body corner-cylinder
//! bottoms and socket corner-chamfer tops have no shared partner and no step
//! face is synthesised across the 0.1mm ledge. The free-edged shell is
//! rejected by `operations::boolean`'s acceptance gate → mesh fallback →
//! non-manifold (over-shared) result.
//!
//! Near-dup free-endpoint scan = 0 and the gap is 0.1mm (>> any weld
//! tolerance), so this is NOT the #859/#867 weld family. The fix is a deeper
//! GFA change: split the partial coplanar same-domain overlap into matching
//! sub-regions and synthesise the corner step faces. Left `#[ignore]`d.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp, boolean as ops_boolean};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

fn free_and_over(topo: &Topology, solid: SolidId) -> (usize, usize) {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e6;
    let q = |p: brepkit_math::vec::Point3| -> QPoint {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };
    let mut faces_per_edge: HashMap<(QPoint, QPoint), HashSet<FaceId>> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                faces_per_edge.entry(key).or_default().insert(fid);
            }
        }
    }
    let free = faces_per_edge.values().filter(|f| f.len() == 1).count();
    let over = faces_per_edge.values().filter(|f| f.len() > 2).count();
    (free, over)
}

#[test]
#[ignore = "repro: 2x1 compartment socket-fuse goes non-manifold (corner-radius mismatch partial coplanar overlap)"]
fn compartment_socket_fuse_is_manifold() {
    let mut topo = Topology::new();
    let body = load("compart2x1_socketfuse_body.bin", &mut topo);
    let socket = load("compart2x1_socketfuse_socket.bin", &mut topo);

    let (fb, ob) = free_and_over(&topo, body);
    let (fs, os) = free_and_over(&topo, socket);
    assert_eq!(
        (fb, ob),
        (0, 0),
        "body operand must start watertight/manifold"
    );
    assert_eq!(
        (fs, os),
        (0, 0),
        "socket operand must start watertight/manifold"
    );

    let result = ops_boolean(&mut topo, BooleanOp::Fuse, body, socket).unwrap();
    let (fr, or) = free_and_over(&topo, result);

    assert_eq!(
        (fr, or),
        (0, 0),
        "socket fuse must stay watertight + manifold (currently free={fr} over={or} via mesh fallback)"
    );
}
