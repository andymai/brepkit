//! Diagnostic: time the intersect(box,sphere) path to find the bottleneck.
#![allow(clippy::unwrap_used, clippy::print_stdout)]

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives;
use brepkit_topology::Topology;
use brepkit_topology::explorer;
use std::time::Instant;

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
