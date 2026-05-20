//! Diagnostic: time the intersect(box,sphere) path to find the bottleneck.
#![allow(clippy::unwrap_used, clippy::print_stdout, clippy::expect_used)]

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives;
use brepkit_topology::Topology;
use brepkit_topology::explorer;
use std::time::Instant;

#[test]
#[ignore = "diagnostic — inspect topology of r=7 case after shortcut"]
fn profile_box_sphere_r7_topology() {
    let mut topo = Topology::new();
    let bx = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let sp = primitives::make_sphere(&mut topo, 7.0, 16).unwrap();
    let r = boolean(&mut topo, BooleanOp::Intersect, bx, sp).unwrap();
    let fids = brepkit_topology::explorer::solid_faces(&topo, r).unwrap();
    let v = brepkit_operations::measure::solid_volume(&topo, r, 0.1).unwrap();
    println!("volume = {v}, faces = {}", fids.len());
    for fid in fids {
        let face = topo.face(fid).unwrap();
        let kind = match face.surface() {
            brepkit_topology::face::FaceSurface::Plane { normal, d } => {
                format!(
                    "Plane n=({:.2},{:.2},{:.2}) d={d:.2}",
                    normal.x(),
                    normal.y(),
                    normal.z()
                )
            }
            brepkit_topology::face::FaceSurface::Sphere(s) => {
                let c = s.center();
                format!(
                    "Sphere c=({:.2},{:.2},{:.2}) r={:.2}",
                    c.x(),
                    c.y(),
                    c.z(),
                    s.radius()
                )
            }
            _ => "?".into(),
        };
        let outer = topo.wire(face.outer_wire()).unwrap();
        println!(
            "  {fid:?} reversed={} {kind} edges={}",
            face.is_reversed(),
            outer.edges().len()
        );
        for oe in outer.edges() {
            let e = topo.edge(oe.edge()).unwrap();
            let sv = topo.vertex(e.start()).unwrap();
            let ev = topo.vertex(e.end()).unwrap();
            let curve = match e.curve() {
                brepkit_topology::edge::EdgeCurve::Line => "Line".to_string(),
                brepkit_topology::edge::EdgeCurve::Circle(c) => format!(
                    "Circle(c=({:.2},{:.2},{:.2}), r={:.2})",
                    c.center().x(),
                    c.center().y(),
                    c.center().z(),
                    c.radius()
                ),
                _ => "?".to_string(),
            };
            let sp = sv.point();
            let ep = ev.point();
            println!(
                "    {:?} fwd={} ({:.2},{:.2},{:.2}) → ({:.2},{:.2},{:.2}) {curve}",
                oe.edge(),
                oe.is_forward(),
                sp.x(),
                sp.y(),
                sp.z(),
                ep.x(),
                ep.y(),
                ep.z()
            );
        }
    }
}

#[test]
#[ignore = "diagnostic — direct GFA call to inspect 3-face result before fallback"]
fn profile_gfa_box_sphere_direct() {
    let _ = env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .is_test(true)
        .try_init();
    let mut topo = Topology::new();
    let a = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let sph = primitives::make_sphere(&mut topo, 8.0, 16).unwrap();
    println!(
        "box faces: {:?}",
        brepkit_topology::explorer::solid_faces(&topo, a).unwrap()
    );
    println!(
        "sphere faces: {:?}",
        brepkit_topology::explorer::solid_faces(&topo, sph).unwrap()
    );
    // Direct GFA call, bypassing the validation+fallback in boolean().
    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Intersect, a, sph)
            .expect("GFA should not error");
    let face_ids = brepkit_topology::explorer::solid_faces(&topo, result).unwrap();
    println!("GFA-direct intersect(box,sphere): {} faces", face_ids.len());
    for fid in face_ids {
        let face = topo.face(fid).unwrap();
        let kind = match face.surface() {
            brepkit_topology::face::FaceSurface::Plane { normal, d } => {
                format!("Plane(n={:?}, d={:.3})", normal, d)
            }
            brepkit_topology::face::FaceSurface::Sphere(s) => {
                format!("Sphere(c={:?}, r={:.3})", s.center(), s.radius())
            }
            other => format!("{other:?}"),
        };
        let outer = topo.wire(face.outer_wire()).unwrap();
        let nedges = outer.edges().len();
        let n_inner = face.inner_wires().len();
        println!("  {fid:?}: {kind} outer_edges={nedges} inner_wires={n_inner}");
    }
}

#[test]
#[ignore = "diagnostic — explicit-only profile of the box-sphere GFA regression"]
fn profile_intersect_box_sphere() {
    let _ = env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .is_test(true)
        .try_init();
    // Warm up.
    for _ in 0..3 {
        let mut topo = Topology::new();
        let a = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let sph = primitives::make_sphere(&mut topo, 8.0, 16).unwrap();
        let _ = boolean(&mut topo, BooleanOp::Intersect, a, sph).unwrap();
    }

    // Time 10 iterations matching the bench inner loop.
    let t0 = Instant::now();
    let mut last_result = None;
    for _ in 0..10 {
        let mut topo = Topology::new();
        let a = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let sph = primitives::make_sphere(&mut topo, 8.0, 16).unwrap();
        let r = boolean(&mut topo, BooleanOp::Intersect, a, sph).unwrap();
        let (f, e, v) = explorer::solid_entity_counts(&topo, r).unwrap();
        last_result = Some((f, e, v));
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (f, e, v) = last_result.unwrap();
    println!(
        "intersect(box,sphere) x10: {total_ms:.1}ms total = {:.1}ms/op",
        total_ms / 10.0
    );
    println!("result: f={f} e={e} v={v}");

    // One more run with surface-kind breakdown.
    let mut topo = Topology::new();
    let a = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let sph = primitives::make_sphere(&mut topo, 8.0, 16).unwrap();
    let r = boolean(&mut topo, BooleanOp::Intersect, a, sph).unwrap();
    let face_ids = brepkit_topology::explorer::solid_faces(&topo, r).unwrap();
    let mut counts = std::collections::BTreeMap::new();
    for fid in face_ids {
        let surf = topo.face(fid).unwrap().surface();
        let kind = match surf {
            brepkit_topology::face::FaceSurface::Plane { .. } => "Plane",
            brepkit_topology::face::FaceSurface::Sphere(_) => "Sphere",
            brepkit_topology::face::FaceSurface::Cylinder(_) => "Cylinder",
            brepkit_topology::face::FaceSurface::Cone(_) => "Cone",
            brepkit_topology::face::FaceSurface::Torus(_) => "Torus",
            brepkit_topology::face::FaceSurface::Nurbs(_) => "Nurbs",
        };
        *counts.entry(kind).or_insert(0_usize) += 1;
    }
    for (kind, n) in &counts {
        println!("  {kind}: {n}");
    }
}
