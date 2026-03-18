//! Boolean dispatch vs pipeline pipeline comparison benchmarks.
//!
//! For each workload, benchmarks both:
//! - `boolean_pipeline()` — the v2-only parameter-space pipeline
//! - `boolean()` — the dispatch function (tries analytic → v2 → chord-based)
//!
//! When `boolean_pipeline()` returns `Err`, the v2 bench is skipped for that workload
//! (the dispatch result IS the v1 path).
//!
//! Run with: `cargo bench -p brepkit-operations --bench boolean_v1_v2`

#![allow(
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items,
    dead_code,
    missing_docs
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::boolean_pipeline::boolean_pipeline;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure;
use brepkit_operations::primitives;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;

type SetupFn = fn() -> (
    Topology,
    brepkit_topology::solid::SolidId,
    brepkit_topology::solid::SolidId,
);

// ---------------------------------------------------------------------------
// Criterion config
// ---------------------------------------------------------------------------

fn fast_config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check whether v2 can handle a workload by doing a dry run.
/// Returns true if `boolean_pipeline` succeeds.
fn pipeline_supported(
    op: BooleanOp,
    setup: &dyn Fn() -> (
        Topology,
        brepkit_topology::solid::SolidId,
        brepkit_topology::solid::SolidId,
    ),
) -> bool {
    let (mut topo, a, b) = setup();
    boolean_pipeline(&mut topo, op, a, b).is_ok()
}

/// Check whether dispatch can handle a workload by doing a dry run.
/// Returns true if `boolean` succeeds.
fn dispatch_supported(
    op: BooleanOp,
    setup: &dyn Fn() -> (
        Topology,
        brepkit_topology::solid::SolidId,
        brepkit_topology::solid::SolidId,
    ),
) -> bool {
    let (mut topo, a, b) = setup();
    boolean(&mut topo, op, a, b).is_ok()
}

/// Count faces on a solid's outer shell.
fn face_count(topo: &Topology, solid: brepkit_topology::solid::SolidId) -> usize {
    let shell_id = topo.solid(solid).unwrap().outer_shell();
    topo.shell(shell_id).unwrap().faces().len()
}

/// Check whether all faces on a solid's outer shell are analytic (not Nurbs).
fn all_faces_analytic(topo: &Topology, solid: brepkit_topology::solid::SolidId) -> bool {
    let shell_id = topo.solid(solid).unwrap().outer_shell();
    let shell = topo.shell(shell_id).unwrap();
    shell.faces().iter().all(|&fid| {
        let face = topo.face(fid).unwrap();
        !matches!(face.surface(), FaceSurface::Nurbs(_))
    })
}

// ---------------------------------------------------------------------------
// Workload setup functions
// ---------------------------------------------------------------------------

/// Two boxes overlapping along one axis: box(10,10,10) at origin + box(10,10,10) translated (5,0,0).
fn setup_box_box_1axis() -> (
    Topology,
    brepkit_topology::solid::SolidId,
    brepkit_topology::solid::SolidId,
) {
    let mut topo = Topology::new();
    let a = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let b = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let mat = Mat4::translation(5.0, 0.0, 0.0);
    transform_solid(&mut topo, b, &mat).unwrap();
    (topo, a, b)
}

/// Two boxes overlapping along three axes: box(10,10,10) at origin + box(10,10,10) translated (5,5,5).
fn setup_box_box_3axis() -> (
    Topology,
    brepkit_topology::solid::SolidId,
    brepkit_topology::solid::SolidId,
) {
    let mut topo = Topology::new();
    let a = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let b = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let mat = Mat4::translation(5.0, 5.0, 5.0);
    transform_solid(&mut topo, b, &mat).unwrap();
    (topo, a, b)
}

/// Two disjoint boxes: box(10,10,10) at origin + box(10,10,10) translated (50,0,0).
fn setup_box_box_disjoint() -> (
    Topology,
    brepkit_topology::solid::SolidId,
    brepkit_topology::solid::SolidId,
) {
    let mut topo = Topology::new();
    let a = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let b = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let mat = Mat4::translation(50.0, 0.0, 0.0);
    transform_solid(&mut topo, b, &mat).unwrap();
    (topo, a, b)
}

/// Cylinder through box: box(10,10,10) centered at origin, cylinder(r=3, h=20) through center.
fn setup_cyl_through_box() -> (
    Topology,
    brepkit_topology::solid::SolidId,
    brepkit_topology::solid::SolidId,
) {
    let mut topo = Topology::new();
    let a = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let cyl = primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();
    // Center the cylinder through the box (box goes 0..10 on each axis).
    let mat = Mat4::translation(5.0, 5.0, -5.0);
    transform_solid(&mut topo, cyl, &mat).unwrap();
    (topo, a, cyl)
}

/// Sphere inside box: box(10,10,10) at origin, sphere(r=4) at box center.
fn setup_sphere_inside_box() -> (
    Topology,
    brepkit_topology::solid::SolidId,
    brepkit_topology::solid::SolidId,
) {
    let mut topo = Topology::new();
    let a = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let sph = primitives::make_sphere(&mut topo, 4.0, 16).unwrap();
    let mat = Mat4::translation(5.0, 5.0, 5.0);
    transform_solid(&mut topo, sph, &mat).unwrap();
    (topo, a, sph)
}

// ---------------------------------------------------------------------------
// Macro for benching both v2-only and dispatch paths
// ---------------------------------------------------------------------------

/// Bench a single boolean workload for both v2-only and dispatch.
/// `$group`: criterion BenchmarkGroup
/// `$name`: benchmark name prefix
/// `$op`: BooleanOp variant
/// `$setup`: function returning (Topology, SolidId, SolidId)
macro_rules! bench_dispatch_vs_pipeline {
    ($group:expr, $name:expr, $op:expr, $setup:expr) => {
        // Bench dispatch path only if it can handle this workload.
        if dispatch_supported($op, &$setup) {
            $group.bench_function(concat!($name, "/dispatch"), |b| {
                b.iter(|| {
                    let (mut topo, a, b_solid) = $setup();
                    black_box(boolean(&mut topo, $op, a, b_solid).unwrap());
                });
            });
        }

        // Bench v2-only if it can handle this workload.
        if pipeline_supported($op, &$setup) {
            $group.bench_function(concat!($name, "/pipeline_only"), |b| {
                b.iter(|| {
                    let (mut topo, a, b_solid) = $setup();
                    black_box(boolean_pipeline(&mut topo, $op, a, b_solid).unwrap());
                });
            });
        }
    };
}

// ---------------------------------------------------------------------------
// Group 1: box_box
// ---------------------------------------------------------------------------

fn bench_box_box(c: &mut Criterion) {
    let mut group = c.benchmark_group("box_box");

    // overlapping_1axis — all 3 ops
    bench_dispatch_vs_pipeline!(group, "1axis/fuse", BooleanOp::Fuse, setup_box_box_1axis);
    bench_dispatch_vs_pipeline!(group, "1axis/cut", BooleanOp::Cut, setup_box_box_1axis);
    bench_dispatch_vs_pipeline!(
        group,
        "1axis/intersect",
        BooleanOp::Intersect,
        setup_box_box_1axis
    );

    // overlapping_3axis — all 3 ops
    bench_dispatch_vs_pipeline!(group, "3axis/fuse", BooleanOp::Fuse, setup_box_box_3axis);
    bench_dispatch_vs_pipeline!(group, "3axis/cut", BooleanOp::Cut, setup_box_box_3axis);
    bench_dispatch_vs_pipeline!(
        group,
        "3axis/intersect",
        BooleanOp::Intersect,
        setup_box_box_3axis
    );

    // disjoint — fuse only
    bench_dispatch_vs_pipeline!(
        group,
        "disjoint/fuse",
        BooleanOp::Fuse,
        setup_box_box_disjoint
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: box_cylinder / box_sphere
// ---------------------------------------------------------------------------

fn bench_box_cylinder(c: &mut Criterion) {
    let mut group = c.benchmark_group("box_cylinder");

    bench_dispatch_vs_pipeline!(
        group,
        "cyl_through_box/cut",
        BooleanOp::Cut,
        setup_cyl_through_box
    );
    bench_dispatch_vs_pipeline!(
        group,
        "sphere_inside_box/intersect",
        BooleanOp::Intersect,
        setup_sphere_inside_box
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: sequential staircase
// ---------------------------------------------------------------------------

/// Build a staircase: `n` sequential cuts of small boxes from a large box.
fn staircase_pipeline(n: usize) -> brepkit_topology::solid::SolidId {
    let mut topo = Topology::new();
    let mut result = primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
    for i in 0..n {
        let step = primitives::make_box(&mut topo, 5.0, 20.0, 5.0).unwrap();
        let x = i as f64 * 5.0;
        let z = 20.0 - (i as f64 + 1.0) * (20.0 / n as f64);
        let mat = Mat4::translation(x, 0.0, z);
        transform_solid(&mut topo, step, &mat).unwrap();
        result = boolean_pipeline(&mut topo, BooleanOp::Cut, result, step).unwrap();
    }
    result
}

fn staircase_dispatch(n: usize) -> brepkit_topology::solid::SolidId {
    let mut topo = Topology::new();
    let mut result = primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
    for i in 0..n {
        let step = primitives::make_box(&mut topo, 5.0, 20.0, 5.0).unwrap();
        let x = i as f64 * 5.0;
        let z = 20.0 - (i as f64 + 1.0) * (20.0 / n as f64);
        let mat = Mat4::translation(x, 0.0, z);
        transform_solid(&mut topo, step, &mat).unwrap();
        result = boolean(&mut topo, BooleanOp::Cut, result, step).unwrap();
    }
    result
}

fn bench_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential");

    group.bench_function("staircase_4/dispatch", |b| {
        b.iter(|| {
            black_box(staircase_dispatch(4));
        });
    });

    // Check v2 support with a dry run.
    let mut test_topo = Topology::new();
    let test_a = primitives::make_box(&mut test_topo, 20.0, 20.0, 20.0).unwrap();
    let test_b = primitives::make_box(&mut test_topo, 5.0, 20.0, 5.0).unwrap();
    let test_mat = Mat4::translation(0.0, 0.0, 15.0);
    transform_solid(&mut test_topo, test_b, &test_mat).unwrap();
    if boolean_pipeline(&mut test_topo, BooleanOp::Cut, test_a, test_b).is_ok() {
        group.bench_function("staircase_4/pipeline_only", |b| {
            b.iter(|| {
                black_box(staircase_pipeline(4));
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: correctness (not timed — single iteration with metrics)
// ---------------------------------------------------------------------------

fn bench_correctness(c: &mut Criterion) {
    let mut group = c.benchmark_group("correctness");
    // Single iteration — we're measuring quality, not speed.
    group.sample_size(10);

    let workloads: &[(&str, BooleanOp, SetupFn)] = &[
        ("1axis_fuse", BooleanOp::Fuse, setup_box_box_1axis),
        ("1axis_cut", BooleanOp::Cut, setup_box_box_1axis),
        ("3axis_intersect", BooleanOp::Intersect, setup_box_box_3axis),
        ("cyl_through_box_cut", BooleanOp::Cut, setup_cyl_through_box),
        (
            "sphere_inside_box_intersect",
            BooleanOp::Intersect,
            setup_sphere_inside_box,
        ),
    ];

    for &(name, op, setup) in workloads {
        // Dispatch path — compute metrics if supported.
        if dispatch_supported(op, &setup) {
            group.bench_function(format!("{name}/dispatch_metrics"), |b| {
                b.iter(|| {
                    let (mut topo, a, b_solid) = setup();
                    let result = boolean(&mut topo, op, a, b_solid).unwrap();
                    let vol = measure::solid_volume(&topo, result, 0.1).unwrap();
                    let faces = face_count(&topo, result);
                    let analytic = all_faces_analytic(&topo, result);
                    black_box((vol, faces, analytic));
                });
            });
        }

        // V2 path — compute metrics if supported.
        if pipeline_supported(op, &setup) {
            group.bench_function(format!("{name}/pipeline_metrics"), |b| {
                b.iter(|| {
                    let (mut topo, a, b_solid) = setup();
                    let result = boolean_pipeline(&mut topo, op, a, b_solid).unwrap();
                    let vol = measure::solid_volume(&topo, result, 0.1).unwrap();
                    let faces = face_count(&topo, result);
                    let analytic = all_faces_analytic(&topo, result);
                    black_box((vol, faces, analytic));
                });
            });
        }
    }

    group.finish();
}

// ===========================================================================
// Criterion setup
// ===========================================================================

criterion_group! {
    name = box_box_bench;
    config = fast_config();
    targets = bench_box_box,
}

criterion_group! {
    name = box_cylinder_bench;
    config = fast_config();
    targets = bench_box_cylinder,
}

criterion_group! {
    name = sequential_bench;
    config = fast_config();
    targets = bench_sequential,
}

criterion_group! {
    name = correctness_bench;
    config = fast_config();
    targets = bench_correctness,
}

criterion_main!(
    box_box_bench,
    box_cylinder_bench,
    sequential_bench,
    correctness_bench,
);
