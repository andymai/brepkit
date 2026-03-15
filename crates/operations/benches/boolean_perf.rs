//! Boolean scaling benchmarks.
//!
//! Measures how boolean performance scales with:
//! 1. Number of sequential operations (N = 4, 16, 64 cylinder cuts)
//! 2. Face count of the target solid at the time of a single cut
//!
//! Run with: `cargo bench -p brepkit-operations --bench boolean_perf`

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
use brepkit_operations::primitives;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a box and cut `n` cylinder holes in a grid pattern, evenly spaced.
/// Returns the resulting solid.
fn box_with_cylinder_cuts(topo: &mut Topology, n: usize) -> brepkit_topology::solid::SolidId {
    let mut result = primitives::make_box(topo, 100.0, 100.0, 10.0).unwrap();

    if n == 0 {
        return result;
    }

    let cols = (n as f64).sqrt().ceil() as usize;
    let rows = n.div_ceil(cols);

    let x_spacing = 100.0 / (cols + 1) as f64;
    let y_spacing = 100.0 / (rows + 1) as f64;

    for i in 0..n {
        let col = i % cols;
        let row = i / cols;
        let x = x_spacing * (col + 1) as f64;
        let y = y_spacing * (row + 1) as f64;

        let cyl = primitives::make_cylinder(topo, 2.0, 20.0).unwrap();
        let mat = Mat4::translation(x, y, -5.0);
        transform_solid(topo, cyl, &mat).unwrap();
        result = boolean(topo, BooleanOp::Cut, result, cyl).unwrap();
    }
    result
}

// ---------------------------------------------------------------------------
// Benchmark 1: Sequential cylinder cuts — scaling with N
// ---------------------------------------------------------------------------

fn bench_sequential_cylinder_cuts(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_cylinder_cuts");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for &n in &[4, 16, 64] {
        group.bench_function(format!("N={n}"), |b| {
            b.iter(|| {
                let mut topo = Topology::new();
                black_box(box_with_cylinder_cuts(&mut topo, n));
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark 2: Single boolean at varying face counts
// ---------------------------------------------------------------------------

fn bench_single_boolean_at_face_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_boolean_at_face_count");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    // (label, number of pre-cuts to reach approximate initial face count)
    // F=6: bare box (0 pre-cuts)
    // F=18: ~4 pre-cuts (each cylinder cut adds ~3 faces: 1 curved + 1 bottom + net top)
    // F=54: ~16 pre-cuts
    let configs: &[(&str, usize)] = &[
        ("F=6 (bare box)", 0),
        ("F~18 (4 cuts)", 4),
        ("F~54 (16 cuts)", 16),
    ];

    for &(label, pre_cuts) in configs {
        group.bench_function(label, |b| {
            // Build the pre-cut solid once outside the iteration.
            let mut base_topo = Topology::new();
            let base_solid = box_with_cylinder_cuts(&mut base_topo, pre_cuts);

            // Find a position that does not collide with existing holes.
            // Place the extra cylinder at (50, 95) — center-x, near top edge.
            b.iter(|| {
                let mut topo = base_topo.clone();
                let cyl = primitives::make_cylinder(&mut topo, 2.0, 20.0).unwrap();
                let mat = Mat4::translation(50.0, 95.0, -5.0);
                transform_solid(&mut topo, cyl, &mat).unwrap();
                black_box(boolean(&mut topo, BooleanOp::Cut, base_solid, cyl).unwrap());
            });
        });
    }

    group.finish();
}

// ===========================================================================
// Criterion setup
// ===========================================================================

criterion_group!(
    boolean_scaling,
    bench_sequential_cylinder_cuts,
    bench_single_boolean_at_face_count,
);

criterion_main!(boolean_scaling);
