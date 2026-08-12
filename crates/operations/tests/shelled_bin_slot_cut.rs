//! #1536: cutting wall-slot boxes out of a shelled open-top rounded bin must
//! keep the pocket open, not re-encode it as an enclosed void.
//!
//! The face splitter's first-vertex hole matching attached the bin's whole
//! cavity-mouth loop to the tiny woven notch rectangle it shares two corners
//! with (first-match order + a strict ray-cast reading an exactly-on-corner
//! probe as inside). The rim annulus then lost its hole and was emitted as a
//! full disc, with the mouth re-emerging as a same-sense coincident ceiling —
//! a closed, manifold solid that reads ~6x its true volume (the tool's
//! "2x2 slotted no-lip bin loses its cavity" defect). The trigger needs the
//! rim's corner arcs: an all-line rim traces its mouth loop from a different
//! first vertex and never mis-attached.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;

use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const W: f64 = 83.6;
const R: f64 = 3.75;
const H: f64 = 21.0;
const T: f64 = 1.25;
const SLOT_DEPTH: f64 = 0.6;
const SLOT_WIDTH: f64 = 2.1;
const SLOT_EXT: f64 = 0.01;

fn rounded_rect_box(topo: &mut Topology) -> SolidId {
    let hw = W / 2.0;
    let c = hw - R;
    let t = 1e-7;
    let z = Vec3::new(0.0, 0.0, 1.0);
    let v = |topo: &mut Topology, x: f64, y: f64| {
        topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), t))
    };
    let v0 = v(topo, hw, -c);
    let v1 = v(topo, hw, c);
    let v2 = v(topo, c, hw);
    let v3 = v(topo, -c, hw);
    let v4 = v(topo, -hw, c);
    let v5 = v(topo, -hw, -c);
    let v6 = v(topo, -c, -hw);
    let v7 = v(topo, c, -hw);
    let arc = |topo: &mut Topology, a, b, cx: f64, cy: f64| {
        let circle = Circle3D::new(Point3::new(cx, cy, 0.0), z, R).unwrap();
        topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle)))
    };
    let edges = [
        topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line)),
        arc(topo, v1, v2, c, c),
        topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line)),
        arc(topo, v3, v4, -c, c),
        topo.add_edge(Edge::new(v4, v5, EdgeCurve::Line)),
        arc(topo, v5, v6, -c, -c),
        topo.add_edge(Edge::new(v6, v7, EdgeCurve::Line)),
        arc(topo, v7, v0, c, -c),
    ];
    let wire = Wire::new(
        edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let fid = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    brepkit_operations::extrude::extrude(topo, fid, Vec3::new(0.0, 0.0, 1.0), H).unwrap()
}

fn shelled_bin(topo: &mut Topology) -> SolidId {
    let solid_box = rounded_rect_box(topo);
    let top_faces: Vec<_> = {
        let s = topo.solid(solid_box).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        sh.faces()
            .iter()
            .copied()
            .filter(|&fid| {
                matches!(
                    topo.face(fid).unwrap().surface(),
                    FaceSurface::Plane { normal, d }
                        if normal.z() > 0.99 && (*d - H).abs() < 1e-6
                )
            })
            .collect()
    };
    brepkit_operations::shell_op::shell(topo, solid_box, T, &top_faces).unwrap()
}

fn slot_boxes(topo: &mut Topology) -> Vec<SolidId> {
    let inner_wall = W / 2.0 - T;
    let mut out = Vec::new();
    for side in [1.0, -1.0] {
        for yc in [-20.27_f64, 0.0, 20.27] {
            let sid = brepkit_operations::primitives::make_box(
                topo,
                SLOT_DEPTH + SLOT_EXT,
                SLOT_WIDTH,
                H - T,
            )
            .unwrap();
            let x0 = if side > 0.0 {
                inner_wall - SLOT_EXT
            } else {
                -(inner_wall + SLOT_DEPTH)
            };
            let m = Mat4::translation(x0, yc - SLOT_WIDTH / 2.0, T);
            brepkit_operations::transform::transform_solid(topo, sid, &m).unwrap();
            out.push(sid);
        }
    }
    out
}

fn edge_health(topo: &Topology, sid: SolidId) -> (usize, usize) {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for fid in solid_faces(topo, sid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    let free = uses.values().filter(|&&c| c == 1).count();
    let over = uses.values().filter(|&&c| c > 2).count();
    (free, over)
}

#[test]
fn slot_cut_on_shelled_rounded_bin_keeps_the_open_pocket() {
    let mut topo = Topology::new();
    let hollow = shelled_bin(&mut topo);
    let vol_hollow =
        brepkit_operations::measure::oriented_solid_volume(&topo, hollow, 0.05).unwrap();

    let mut cur = hollow;
    for slot in slot_boxes(&mut topo) {
        cur = boolean::boolean(&mut topo, BooleanOp::Cut, cur, slot).unwrap();
    }

    let sd = topo.solid(cur).unwrap();
    assert!(
        sd.inner_shells().is_empty(),
        "slot cut must not close the pocket into an enclosed void, got {} inner shell(s)",
        sd.inner_shells().len()
    );
    let (free, over) = edge_health(&topo, cur);
    assert_eq!((free, over), (0, 0), "result must stay closed and manifold");

    // Each slot removes exactly its in-wall material: 0.6 x 2.1 x (H - T).
    let removed = 6.0 * SLOT_DEPTH * SLOT_WIDTH * (H - T);
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, cur, 0.05).unwrap();
    assert!(
        (vol - (vol_hollow - removed)).abs() < 0.5,
        "volume {vol:.1} must be the hollow body {vol_hollow:.1} minus the slots {removed:.1}"
    );
}
