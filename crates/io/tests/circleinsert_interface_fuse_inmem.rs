//! Regression for the coplanar-interface holed-cap fuse family (#1538 tail):
//! the "2x2 with circle insert" bin's pocket has cutDepth equal to the floor
//! depth, so the insert tool through-cuts the 1.2mm floor slab with its
//! bottom cap coincident with the bin bottom (z=0). The layout tool authors
//! the insert profile as a CW-wound 4-arc circle; before extrude normalized
//! profile winding, the extruded tool carried every wire wound against its
//! face flags (globally mirrored winding passes validation — pairwise edge
//! opposition survives a mirror), and the pocket cut then minted 8
//! same-direction rim arcs into the body. Fusing the socket assembly onto
//! that body emitted 60 free edges at the interface (raw GFA), and the ops
//! layer paid a mesh fallback whose output fed the export.
//!
//! The chain is replayed natively here: the captured pre-pocket bin body
//! (brepkit-wasm 3.2.28 capture; byte-identical across 3.2.24 and 3.2.28
//! captures) is cut with a CW-authored quartered-cylinder tool built through
//! `extrude` — the authoring the tool traffic actually ships — and the
//! result is fused with the captured socket assembly.
//!
//! Same family as `deepcutout_cut_inmem::deep_cutout_socket_fuse_is_watertight`
//! (there the hole is a rectangular through-cut meeting the z=0 interface).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

fn load(name: &str, topo: &mut Topology) -> brepkit_topology::solid::SolidId {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

/// The insert tool as the layout tool ships it: a 4-arc circle profile wound
/// CW, extruded r=10 from z=0 to z=5 at the bin center.
fn cw_quartered_cylinder_tool(topo: &mut Topology) -> brepkit_topology::solid::SolidId {
    let (r, z0) = (10.0, 0.0);
    let z = Vec3::new(0.0, 0.0, 1.0);
    let v = |topo: &mut Topology, x: f64, y: f64| {
        topo.add_vertex(Vertex::new(Point3::new(x, y, z0), 1e-7))
    };
    let v0 = v(topo, r, 0.0);
    let v1 = v(topo, 0.0, r);
    let v2 = v(topo, -r, 0.0);
    let v3 = v(topo, 0.0, -r);
    let arc = |topo: &mut Topology, a, b| {
        let circle = Circle3D::new(Point3::new(0.0, 0.0, z0), z, r).unwrap();
        topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle)))
    };
    let edges = [
        arc(topo, v0, v1),
        arc(topo, v1, v2),
        arc(topo, v2, v3),
        arc(topo, v3, v0),
    ];
    // CW traversal of the CCW-stored arcs: v0 -> v3 -> v2 -> v1 -> v0.
    let oes: Vec<OrientedEdge> = edges
        .iter()
        .rev()
        .map(|&e| OrientedEdge::new(e, false))
        .collect();
    let wid = topo.add_wire(Wire::new(oes, true).unwrap());
    let fid = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z0,
        },
    ));
    brepkit_operations::extrude::extrude(topo, fid, z, 5.0).unwrap()
}

// Operand notes, measured at capture:
// - `circleinsert_base.bin` (the bin before the insert pocket cut) validates
//   clean with orientation checking on.
// - `circleinsert_sockets.bin` is 4 disjoint feet in one shell (V-E+F = 8,
//   which `validate_solid` mis-reports as an Euler error — it expects one
//   component per shell). Do not "fix" a fixture to satisfy that report.

#[test]
fn circleinsert_pocket_cut_is_strictly_valid() {
    let mut topo = Topology::new();
    let base = load("circleinsert_base.bin", &mut topo);
    let tool = cw_quartered_cylinder_tool(&mut topo);

    let body = brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Cut, base, tool)
        .unwrap();
    let report = brepkit_operations::validate::validate_solid(&topo, body).unwrap();
    assert!(report.is_valid(), "pocket cut must validate: {report:?}");
}

#[test]
#[ignore = "open residue: with BOTH operands validation-clean the socket fuse still emits 84 free edges around the pocket mouth (z=0/1.2 rim arcs and inter-feet gap channels) plus 4 over-shared bin corner arcs at z=5; native repro `cargo run --release -p brepkit-io --example circleinsert_chain` (FREE_EDGES=1 dumps owners); see #1538"]
fn circleinsert_socket_fuse_is_strictly_valid() {
    let mut topo = Topology::new();
    let base = load("circleinsert_base.bin", &mut topo);
    let tool = cw_quartered_cylinder_tool(&mut topo);

    let body = brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Cut, base, tool)
        .unwrap();

    let sockets = load("circleinsert_sockets.bin", &mut topo);
    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, body, sockets)
            .unwrap();
    let report = brepkit_operations::validate::validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "socket fuse must validate: {report:?}");
}
