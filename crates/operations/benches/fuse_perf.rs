//! Fuse (union) scaling benchmarks.
//!
//! Measures `fuse_all` (balanced pair-wise reduction) vs sequential left-fold
//! for grids of adjacent boxes. Exercises the Fuse code path in `boolean()`.
//!
//! Run with: `cargo bench -p brepkit-operations --bench fuse_perf`

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
use brepkit_operations::compound_ops::fuse_all;
use brepkit_operations::primitives;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::compound::Compound;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build an N×N grid of unit boxes, each touching its neighbors (shared edges).
/// Returns (topology, vec_of_solid_ids).
fn build_box_grid(n: usize) -> (Topology, Vec<brepkit_topology::solid::SolidId>) {
    let mut topo = Topology::new();
    let mut solids = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            let bx = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
            let mat = Mat4::translation(col as f64, row as f64, 0.0);
            transform_solid(&mut topo, bx, &mat).unwrap();
            solids.push(bx);
        }
    }
    (topo, solids)
}

/// Build an N×N grid of slightly overlapping boxes (0.01 overlap on each edge).
fn build_overlapping_box_grid(n: usize) -> (Topology, Vec<brepkit_topology::solid::SolidId>) {
    let mut topo = Topology::new();
    let size = 1.01; // slight overlap
    let mut solids = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            let bx = primitives::make_box(&mut topo, size, size, 1.0).unwrap();
            let mat = Mat4::translation(col as f64, row as f64, 0.0);
            transform_solid(&mut topo, bx, &mat).unwrap();
            solids.push(bx);
        }
    }
    (topo, solids)
}

// ---------------------------------------------------------------------------
// Benchmark: fuse_all (balanced) vs sequential left-fold
// ---------------------------------------------------------------------------

fn bench_fuse_balanced(c: &mut Criterion) {
    let mut group = c.benchmark_group("fuse_balanced");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    for &side in &[2, 3, 4, 5] {
        let n = side * side;
        group.bench_function(format!("balanced_N={n}"), |b| {
            let (base_topo, solids) = build_overlapping_box_grid(side);
            b.iter(|| {
                let mut topo = base_topo.clone();
                let cid = topo.add_compound(Compound::new(solids.clone()));
                black_box(fuse_all(&mut topo, cid).unwrap());
            });
        });

        group.bench_function(format!("sequential_N={n}"), |b| {
            let (base_topo, solids) = build_overlapping_box_grid(side);
            b.iter(|| {
                let mut topo = base_topo.clone();
                let mut result = solids[0];
                for &s in &solids[1..] {
                    result = boolean(&mut topo, BooleanOp::Fuse, result, s).unwrap();
                }
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_fuse_touching_boxes(c: &mut Criterion) {
    let mut group = c.benchmark_group("fuse_touching");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    for &side in &[2, 3, 4] {
        let n = side * side;
        group.bench_function(format!("balanced_N={n}"), |b| {
            let (base_topo, solids) = build_box_grid(side);
            b.iter(|| {
                let mut topo = base_topo.clone();
                let cid = topo.add_compound(Compound::new(solids.clone()));
                black_box(fuse_all(&mut topo, cid).unwrap());
            });
        });
    }

    group.finish();
}

// ===========================================================================
// Criterion setup
// ===========================================================================

criterion_group!(fuse_scaling, bench_fuse_balanced, bench_fuse_touching_boxes,);

criterion_main!(fuse_scaling);
