//! Lightweight boolean-perf tracking benchmarks for CI regression alerts.
//!
//! Deliberately small and fast (reduced sample count, moderate sizes) so the
//! whole suite runs in well under a minute on a shared CI runner. The goal is a
//! stable *trend* over time on `main`, not precise local measurement — the
//! `github-action-benchmark` step compares against the stored baseline and
//! comments on a large regression. For sharp O(N²) detection see the
//! deterministic complexity guard (`boolean::tests::scaling_*`); this catches
//! broad slowdowns the work-counters can't see.
//!
//! Run locally: `cargo bench -p brepkit-operations --bench boolean_tracking`

#![allow(
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items,
    missing_docs
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};

/// Merge disjoint solids into one multi-region tool (the lowering of a compound
/// cutter). Inlined here because `compound_ops::merge_disjoint_solids` is
/// crate-internal and benches are an external target.
fn merge_disjoint(topo: &mut Topology, solids: &[SolidId]) -> SolidId {
    let mut faces = Vec::new();
    for &s in solids {
        let outer = topo.solid(s).unwrap().outer_shell();
        faces.extend_from_slice(topo.shell(outer).unwrap().faces());
    }
    let outer = topo.add_shell(Shell::new(faces).unwrap());
    topo.add_solid(Solid::new(outer, vec![]))
}

/// Two unit boxes overlapping on the diagonal (a non-trivial intersection
/// region for all three boolean ops).
fn two_boxes(topo: &mut Topology) -> (SolidId, SolidId) {
    let a = make_box(topo, 1.0, 1.0, 1.0).unwrap();
    let b = make_box(topo, 1.0, 1.0, 1.0).unwrap();
    transform_solid(topo, b, &Mat4::translation(0.5, 0.5, 0.5)).unwrap();
    (a, b)
}

fn bench_booleans(c: &mut Criterion) {
    let mut group = c.benchmark_group("boolean");
    // CI-friendly: few samples, short measurement — trend tracking, not precision.
    group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));

    for (name, op) in [
        ("cut_box_box", BooleanOp::Cut),
        ("fuse_box_box", BooleanOp::Fuse),
        ("intersect_box_box", BooleanOp::Intersect),
    ] {
        group.bench_function(name, |bencher| {
            bencher.iter(|| {
                let mut topo = Topology::new();
                let (a, b) = two_boxes(&mut topo);
                black_box(boolean(&mut topo, op, a, b).unwrap())
            });
        });
    }

    group.bench_function("cut_cylinder_through_box", |bencher| {
        bencher.iter(|| {
            let mut topo = Topology::new();
            let box_id = make_box(&mut topo, 4.0, 4.0, 2.0).unwrap();
            let cyl = make_cylinder(&mut topo, 1.0, 6.0).unwrap();
            transform_solid(&mut topo, cyl, &Mat4::translation(2.0, 2.0, -2.0)).unwrap();
            black_box(boolean(&mut topo, BooleanOp::Cut, box_id, cyl).unwrap())
        });
    });

    // Issue #987: a 6×6 perforated panel (36 through-holes). Tracks the
    // many-holes Cut path that the O(N²) fixes made near-linear.
    group.bench_function("perforated_cut_36", |bencher| {
        bencher.iter(|| {
            let mut topo = Topology::new();
            let pitch = 2.4;
            let span = 7.0 * pitch;
            let slab = make_box(&mut topo, span, span, 2.0).unwrap();
            transform_solid(&mut topo, slab, &Mat4::translation(0.0, 0.0, 1.0)).unwrap();
            let mut tools = Vec::new();
            for j in 0..6 {
                for i in 0..6 {
                    let h = make_box(&mut topo, 1.0, 1.0, 4.0).unwrap();
                    let m = Mat4::translation((i + 1) as f64 * pitch, (j + 1) as f64 * pitch, 0.0);
                    transform_solid(&mut topo, h, &m).unwrap();
                    tools.push(h);
                }
            }
            let tool = merge_disjoint(&mut topo, &tools);
            black_box(boolean(&mut topo, BooleanOp::Cut, slab, tool).unwrap())
        });
    });

    group.finish();
}

criterion_group!(benches, bench_booleans);
criterion_main!(benches);
