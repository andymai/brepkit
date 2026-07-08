//! Faithful regression guard: the fractional-width halfSockets corner
//! crescent (the post-loft `1.5×6 halfSockets` family — bnd=104 on every
//! tilt variant of the compartment manifold matrix).
//!
//! Operands captured from the live gridfinity tool via the boolean-capture
//! probe kernel: the 1.5×6 halfSockets compartment bin (98 faces, corner
//! radius 3.75 arcs) and one corner half-socket from the socket-assembly
//! fuse tree (34 faces, analytic cone/cylinder feet with an r=4 outline
//! corner arc). The socket is translated +5 in z to its final-assembly
//! position, reproducing the corner geometry of the final bin × 36-socket
//! export fuse with a 20 KB operand instead of 740 KB.
//!
//! At each bin corner the socket's r=4 outline circle (tangent to both bin
//! wall lines) and the bin's r=3.75 corner arc bound a sliver ≈0.1–0.25 mm
//! wide on the z=5 interface plane. The arrangement split emits the sliver
//! region correctly, but `interior_point_3d` sampled the region's polygon
//! from the stored pcurves — and a wire can mix pcurve orientation
//! conventions (section arcs carry the curve's natural parameterization
//! plus a traversal flag; boundary arcs are fit in traversal order but keep
//! the topology orientation flag). The reversed boundary arcs sampled
//! backwards, folding the sliver polygon into a self-crossing zig-zag whose
//! "interior" point landed in the neighboring socket-imprint region — the
//! classifier then read solid material at the wrong point, marked the
//! sliver Inside, and dropped it. Five unpaired rim edges per corner
//! (socket r=4 arc + two bin r=3.75 arc pieces + two 0.25 mm wall stubs)
//! surfaced as mesh boundary edges in every fractional-width export.
//!
//! Fixed by sampling plane-face wire polygons from the 3D curves through
//! the face's `PlaneFrame` (orientation-unambiguous), never the pcurves —
//! the same arc-true pattern as `find_point_outside_holes`. That function's
//! hole polygons were also densified (3 → 15 interior samples): a
//! single-edge closed bore hole sampled at 4 points is an inscribed square
//! whose sagitta gap accepted ring seeds well inside the hole.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean_with_evolution};
use brepkit_operations::tessellate::{
    boundary_edge_count, non_manifold_edge_count, tessellate_solid_with_tolerance,
};
use brepkit_operations::transform::transform_solid;
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

fn edge_health(topo: &Topology, solid: SolidId) -> (usize, usize) {
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
    let mut faces_per_edge: HashMap<(Q, Q), HashSet<FaceId>> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                *occ.entry(key).or_default() += 1;
                faces_per_edge.entry(key).or_default().insert(fid);
            }
        }
    }
    let free = occ.values().filter(|&&c| c == 1).count();
    let over = faces_per_edge.values().filter(|f| f.len() > 2).count();
    (free, over)
}

#[test]
fn fracwidth_corner_socket_fuse_keeps_the_corner_crescent() {
    let mut topo = Topology::new();
    let bin = load("fracwidth_hs_bin.bin", &mut topo);
    let socket = load("fracwidth_hs_corner_socket.bin", &mut topo);
    transform_solid(&mut topo, socket, &Mat4::translation(0.0, 0.0, 5.0)).unwrap();

    // The tool's export fuses run through the provenance path.
    let (result, _) = boolean_with_evolution(&mut topo, BooleanOp::Fuse, bin, socket).unwrap();

    // Analytic result, not a mesh fallback: the socket's cone/cylinder feet
    // must survive (a fallback would be hundreds of all-plane faces).
    let fids = solid_faces(&topo, result).unwrap();
    assert!(
        fids.len() < 200,
        "expected an analytic fuse (~130 faces), got {} (mesh fallback?)",
        fids.len()
    );
    let curved = fids
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count();
    assert!(
        curved > 20,
        "expected the analytic cone/cylinder feet to survive, curved={curved}"
    );

    // B-Rep edge health: the pre-fix defect left 5 unpaired rim edges at the
    // fused corner (the dropped crescent's boundary).
    let (free, over) = edge_health(&topo, result);
    assert_eq!(free, 0, "corner-crescent rim edges must be paired");
    assert_eq!(over, 0, "no over-shared edges");

    // Watertight at the tool's export tessellation tier (0.01 mm / 5°).
    let mesh = tessellate_solid_with_tolerance(&topo, result, 0.01, 5f64.to_radians()).unwrap();
    assert_eq!(
        boundary_edge_count(&mesh),
        0,
        "export-tier mesh must be watertight"
    );
    assert_eq!(non_manifold_edge_count(&mesh), 0);
}
