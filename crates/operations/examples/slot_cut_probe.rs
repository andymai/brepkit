//! #1536 repro probe: does cutting wall-slot boxes (tops coplanar with the
//! bin rim plane) out of a shelled open-top bin body flip the z-top plane
//! faces into a same-sense doubled cover?
//!
//! The captured `slotted_nolip_body.bin` shows the defect: the outer shell is
//! the full box (top disc included), the inner shell is the complete
//! cavity+slots void, and the cavity ceiling at the rim plane faces UP —
//! same-sense with the outer top disc — so the enclosed void nearly cancels
//! and the solid reads ~92k too big. This rebuilds the chain natively:
//! rounded-rect box -> shell_op (open top) -> cut 6 wall slots.
//!
//! `cargo run --release --example slot_cut_probe -p brepkit-operations`
#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]

use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
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

fn report(topo: &Topology, sid: SolidId, label: &str) {
    report_h(topo, sid, label, H);
}

fn report_h(topo: &Topology, sid: SolidId, label: &str, h: f64) {
    let origin = Point3::new(0.0, 0.0, 0.0);
    let sd = topo.solid(sid).unwrap();
    let shells: Vec<_> = std::iter::once(("outer".to_string(), sd.outer_shell()))
        .chain(
            sd.inner_shells()
                .iter()
                .enumerate()
                .map(|(i, &s)| (format!("inner{i}"), s)),
        )
        .collect();
    println!("== {label}");
    let mut total = 0.0;
    for (name, shell) in &shells {
        let faces = topo.shell(*shell).unwrap().faces().to_vec();
        let mut vol = 0.0;
        let mut top_up = 0usize;
        let mut top_down = 0usize;
        for fid in &faces {
            let mesh = brepkit_operations::tessellate::tessellate(topo, *fid, 0.05).unwrap();
            let (idx, pos) = (&mesh.indices, &mesh.positions);
            let mut fvol = 0.0;
            let mut at_top = !idx.is_empty();
            let mut nz_sum = 0.0;
            for tri in 0..idx.len() / 3 {
                let a = pos[idx[tri * 3] as usize];
                let b = pos[idx[tri * 3 + 1] as usize];
                let c = pos[idx[tri * 3 + 2] as usize];
                fvol += (a - origin).dot((b - origin).cross(c - origin));
                nz_sum += (b - a).cross(c - a).z();
                for p in [a, b, c] {
                    if (p.z() - h).abs() > 1e-6 {
                        at_top = false;
                    }
                }
            }
            vol += fvol / 6.0;
            if at_top {
                if nz_sum > 0.0 {
                    top_up += 1;
                } else {
                    top_down += 1;
                }
            }
        }
        total += vol;
        println!(
            "  {name:<7} faces={:<4} signed={vol:>11.1}  z=H faces: {top_up} up, {top_down} down",
            faces.len()
        );
    }
    let ov = brepkit_operations::measure::oriented_solid_volume(topo, sid, 0.05).unwrap();
    println!("  sum={total:.1}  oriented_solid_volume={ov:.1}");
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

fn shell_and_cut_one_slot(topo: &mut Topology, solid_box: SolidId, h: f64, label: &str) {
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
                        if normal.z() > 0.99 && (*d - h).abs() < 1e-6
                )
            })
            .collect()
    };
    let hollow = brepkit_operations::shell_op::shell(topo, solid_box, T, &top_faces).unwrap();
    report_h(topo, hollow, &format!("{label}: after shell_op"), h);
    let bb = brepkit_operations::measure::solid_bounding_box(topo, hollow).unwrap();
    let inner_wall_x = bb.max.x() - T;
    let y_mid = f64::midpoint(bb.min.y(), bb.max.y());
    let slot = brepkit_operations::primitives::make_box(topo, 0.61, 2.1, h - T).unwrap();
    let m = Mat4::translation(inner_wall_x - 0.01, y_mid - 1.05, T);
    brepkit_operations::transform::transform_solid(topo, slot, &m).unwrap();
    let result = if std::env::var("BK_RAW_GFA").is_ok() {
        brepkit_algo::gfa::boolean(topo, brepkit_algo::bop::BooleanOp::Cut, hollow, slot).unwrap()
    } else {
        boolean::boolean(topo, BooleanOp::Cut, hollow, slot).unwrap()
    };
    report_h(topo, result, &format!("{label}: after slot cut"), h);
}

fn minimal() {
    let mut topo = Topology::new();
    let solid_box = brepkit_operations::primitives::make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
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
                        if normal.z() > 0.99 && (*d - 10.0).abs() < 1e-6
                )
            })
            .collect()
    };
    let hollow =
        brepkit_operations::shell_op::shell(&mut topo, solid_box, 1.25, &top_faces).unwrap();
    report_h(&topo, hollow, "min: after shell_op", 10.0);
    let slot = brepkit_operations::primitives::make_box(&mut topo, 0.61, 2.1, 8.75).unwrap();
    let m = Mat4::translation(20.0 - 1.25 - 0.01, 10.0 - 1.05, 1.25);
    brepkit_operations::transform::transform_solid(&mut topo, slot, &m).unwrap();
    let result = boolean::boolean(&mut topo, BooleanOp::Cut, hollow, slot).unwrap();
    report_h(&topo, result, "min: after slot cut", 10.0);
}

fn main() {
    env_logger::init();
    let mode = std::env::args().nth(1).unwrap_or_else(|| "seq".to_string());
    if mode == "min" {
        minimal();
        return;
    }
    if mode == "rectfull" {
        let mut topo = Topology::new();
        let b = brepkit_operations::primitives::make_box(&mut topo, W, W, H).unwrap();
        shell_and_cut_one_slot(&mut topo, b, H, "rectfull");
        return;
    }
    if mode == "roundone" {
        let mut topo = Topology::new();
        let b = rounded_rect_box(&mut topo);
        shell_and_cut_one_slot(&mut topo, b, H, "roundone");
        return;
    }

    let mut topo = Topology::new();
    let solid_box = rounded_rect_box(&mut topo);
    report(&topo, solid_box, "solid box");

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
    let hollow = brepkit_operations::shell_op::shell(&mut topo, solid_box, T, &top_faces).unwrap();
    report(&topo, hollow, "after shell_op (open top)");

    let slots = slot_boxes(&mut topo);
    let before_fallbacks = boolean::mesh_fallback_count();

    let result = if mode == "compound" {
        boolean::compound_cut(
            &mut topo,
            hollow,
            &slots,
            boolean::BooleanOptions::default(),
        )
        .unwrap()
    } else {
        let mut cur = hollow;
        for (i, s) in slots.iter().enumerate() {
            cur = boolean::boolean(&mut topo, BooleanOp::Cut, cur, *s).unwrap();
            report(&topo, cur, &format!("after slot cut {i}"));
        }
        cur
    };
    report(&topo, result, &format!("final ({mode})"));
    println!(
        "mesh fallbacks during cuts: {}",
        boolean::mesh_fallback_count() - before_fallbacks
    );
}
