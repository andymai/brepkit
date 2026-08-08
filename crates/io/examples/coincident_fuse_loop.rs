//! Loop the exact-coincidence lip fuse in-process to catch the
//! nondeterministic bad outcome and attribute its defective edges.
//!
//! The maximal-coincidence configuration (lip outer wall + corner cylinders
//! exactly coincident with the box's, bottom coplanar with the rim) flips
//! between a good 58-face analytic result and a bad 64-face one (14 defective
//! edges at z=21: both coincident contact faces kept as an internal membrane)
//! depending on hash-iteration order. `N=<runs>` controls iterations,
//! `BK_LOOP_DEBUG=1` enables debug-level kernel logs per iteration.
//!
//! ```sh
//! N=40 cargo run --release -p brepkit-io --example coincident_fuse_loop
//! ```
#![allow(
    clippy::print_stdout,
    clippy::expect_used,
    clippy::unwrap_used,
    missing_docs
)]

use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

fn make_rounded_rect_spine(
    topo: &mut Topology,
    w: f64,
    d: f64,
    r: f64,
) -> Vec<brepkit_topology::edge::EdgeId> {
    let hw = w / 2.0;
    let hd = d / 2.0;
    let cx = hw - r;
    let cy = hd - r;
    let t = 1e-7;
    let z = Vec3::new(0.0, 0.0, 1.0);
    let v = |topo: &mut Topology, x: f64, y: f64| {
        topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), t))
    };
    let v0 = v(topo, hw, -cy);
    let v1 = v(topo, hw, cy);
    let v2 = v(topo, cx, hd);
    let v3 = v(topo, -cx, hd);
    let v4 = v(topo, -hw, cy);
    let v5 = v(topo, -hw, -cy);
    let v6 = v(topo, -cx, -hd);
    let v7 = v(topo, cx, -hd);
    let arc = |topo: &mut Topology, a, b, ccx: f64, ccy: f64| {
        let circle = Circle3D::new(Point3::new(ccx, ccy, 0.0), z, r).unwrap();
        topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle)))
    };
    vec![
        topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line)),
        arc(topo, v1, v2, cx, cy),
        topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line)),
        arc(topo, v3, v4, -cx, cy),
        topo.add_edge(Edge::new(v4, v5, EdgeCurve::Line)),
        arc(topo, v5, v6, -cx, -cy),
        topo.add_edge(Edge::new(v6, v7, EdgeCurve::Line)),
        arc(topo, v7, v0, cx, -cy),
    ]
}

fn make_lip_profile(topo: &mut Topology, start: Point3, x_dir: Vec3) -> FaceId {
    let uv = [
        (-2.6, 0.0),
        (-1.9, 0.7),
        (-1.9, 2.5),
        (0.0, 4.4),
        (0.0, 0.0),
    ];
    let t = 1e-7;
    let z = Vec3::new(0.0, 0.0, 1.0);
    let verts: Vec<_> = uv
        .iter()
        .map(|&(u, v)| topo.add_vertex(Vertex::new(start + x_dir * u + z * v, t)))
        .collect();
    let n = verts.len();
    let edges: Vec<_> = (0..n)
        .map(|i| topo.add_edge(Edge::new(verts[i], verts[(i + 1) % n], EdgeCurve::Line)))
        .collect();
    let wire = Wire::new(
        edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let normal = x_dir.cross(z).normalize().unwrap();
    let d = normal.dot(Vec3::new(start.x(), start.y(), start.z()));
    topo.add_face(Face::new(wid, vec![], FaceSurface::Plane { normal, d }))
}

fn run_once(iter: usize) -> (usize, usize) {
    let mut topo = Topology::new();
    let spine_for_face = make_rounded_rect_spine(&mut topo, 84.0, 84.0, 3.75);
    let base_wire = Wire::new(
        spine_for_face
            .iter()
            .map(|&e| OrientedEdge::new(e, true))
            .collect(),
        true,
    )
    .unwrap();
    let base_wid = topo.add_wire(base_wire);
    let base_face = topo.add_face(Face::new(
        base_wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let box_solid =
        brepkit_operations::extrude::extrude(&mut topo, base_face, Vec3::new(0.0, 0.0, 1.0), 21.0)
            .unwrap();
    let top_faces: Vec<FaceId> = {
        let s = topo.solid(box_solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        sh.faces()
            .iter()
            .copied()
            .filter(|&fid| {
                matches!(
                    topo.face(fid).unwrap().surface(),
                    FaceSurface::Plane { normal, d }
                        if normal.z() > 0.99 && (*d - 21.0).abs() < 1e-6
                )
            })
            .collect()
    };
    let hollow =
        brepkit_operations::shell_op::shell(&mut topo, box_solid, 1.2, &top_faces).unwrap();
    let spine = make_rounded_rect_spine(&mut topo, 84.0, 84.0, 3.75);
    let profile = make_lip_profile(
        &mut topo,
        Point3::new(42.0, -38.25, 21.0),
        Vec3::new(1.0, 0.0, 0.0),
    );
    let lip = brepkit_operations::sweep::sweep_along_edges(&mut topo, profile, &spine).unwrap();
    let fused =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, hollow, lip)
            .unwrap();
    let faces = brepkit_topology::explorer::solid_faces(&topo, fused).unwrap();
    let curved = faces
        .iter()
        .filter(|&&fid| !matches!(topo.face(fid).unwrap().surface(), FaceSurface::Plane { .. }))
        .count();

    // On the bad outcome, attribute every edge used by != 2 faces.
    let mut uses: std::collections::HashMap<
        brepkit_topology::edge::EdgeId,
        Vec<brepkit_topology::face::FaceId>,
    > = std::collections::HashMap::new();
    for &fid in &faces {
        let f = topo.face(fid).unwrap();
        for wid in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                uses.entry(oe.edge()).or_default().push(fid);
            }
        }
    }
    let mut bad_edges: Vec<_> = uses.iter().filter(|(_, u)| u.len() != 2).collect();
    bad_edges.sort_by_key(|(e, _)| e.index());
    if !bad_edges.is_empty() {
        println!(
            "iter {iter}: BAD RESULT F={} — {} defective edges",
            faces.len(),
            bad_edges.len()
        );
        for (e, owners) in bad_edges.iter().take(20) {
            let edge = topo.edge(**e).unwrap();
            let sp = topo.vertex(edge.start()).unwrap().point();
            let ep = topo.vertex(edge.end()).unwrap().point();
            let surfs: Vec<String> = owners
                .iter()
                .map(|&fid| match topo.face(fid).unwrap().surface() {
                    FaceSurface::Plane { normal, .. } => format!("P{:?}", normal),
                    FaceSurface::Cylinder(c) => format!("C r={:.2}", c.radius()),
                    FaceSurface::Cone(_) => "K".to_string(),
                    _ => "?".to_string(),
                })
                .collect();
            println!(
                "  edge {:?} uses={} ({:.2},{:.2},{:.2})->({:.2},{:.2},{:.2}) owners: {}",
                e,
                owners.len(),
                sp.x(),
                sp.y(),
                sp.z(),
                ep.x(),
                ep.y(),
                ep.z(),
                surfs.join(" | ")
            );
        }
    }
    (faces.len(), curved)
}

fn main() {
    struct P;
    impl log::Log for P {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.level() <= log::Level::Warn || std::env::var("BK_LOOP_DEBUG").is_ok()
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                println!("  [{}] {}", r.level(), r.args());
            }
        }
        fn flush(&self) {}
    }
    static LOGGER: P = P;
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(if std::env::var("BK_LOOP_DEBUG").is_ok() {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Warn
    });

    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let mut fallbacks = 0;
    for i in 0..n {
        let (f, curved) = run_once(i);
        let tag = if curved == 0 { "FALLBACK" } else { "analytic" };
        if curved == 0 {
            fallbacks += 1;
        }
        println!("iter {i}: F={f} curved={curved} {tag}");
    }
    println!("== {fallbacks}/{n} fallbacks");
}
