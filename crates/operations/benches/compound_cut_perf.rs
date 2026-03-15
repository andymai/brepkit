//! Compound-cut scaling benchmarks.
//!
//! Measures `compound_cut` vs sequential `boolean(Cut)` for:
//! 1. Cylinder grids: 4, 16, 36, 64 tools
//! 2. Box grids (honeycomb pattern): 7, 19, 37 tools — exercises ConvexPolyhedron classifier
//!
//! Run with: `cargo bench -p brepkit-operations --bench compound_cut_perf`

#![allow(
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items,
    missing_docs
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean, compound_cut};
use brepkit_operations::primitives;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a box and N cylinder tools translated to a grid pattern.
/// Returns (topology, target_solid, vec_of_tool_solids).
fn build_cylinder_grid(
    n: usize,
) -> (
    Topology,
    brepkit_topology::solid::SolidId,
    Vec<brepkit_topology::solid::SolidId>,
) {
    let mut topo = Topology::new();
    let target = primitives::make_box(&mut topo, 100.0, 100.0, 10.0).unwrap();

    let cols = (n as f64).sqrt().ceil() as usize;
    let rows = if n > 0 { n.div_ceil(cols) } else { 0 };
    let x_spacing = 100.0 / (cols + 1) as f64;
    let y_spacing = 100.0 / (rows + 1) as f64;

    let mut tools = Vec::with_capacity(n);
    for i in 0..n {
        let col = i % cols;
        let row = i / cols;
        let x = x_spacing * (col + 1) as f64;
        let y = y_spacing * (row + 1) as f64;

        let cyl = primitives::make_cylinder(&mut topo, 2.0, 20.0).unwrap();
        let mat = Mat4::translation(x, y, -5.0);
        transform_solid(&mut topo, cyl, &mat).unwrap();
        tools.push(cyl);
    }
    (topo, target, tools)
}

/// Build a box tool at (cx, cy) approximating a hex prism.
/// Uses `make_box` so the tool has all-planar faces and exercises the
/// `ConvexPolyhedron` classifier in `compound_cut`.
fn make_hex_prism(
    topo: &mut Topology,
    cx: f64,
    cy: f64,
    circumradius: f64,
    height: f64,
    z_offset: f64,
) -> brepkit_topology::solid::SolidId {
    // Use a box with side length ≈ circumradius * √3 (inscribed hex width).
    let side = circumradius * 1.732;
    let bx = primitives::make_box(topo, side, side, height).unwrap();
    let mat = Mat4::translation(cx - side / 2.0, cy - side / 2.0, z_offset);
    transform_solid(topo, bx, &mat).unwrap();
    bx
}

/// Build a honeycomb-like pattern of cylinder tools on a box.
/// `rings` = 0 → 1 tool, rings = 1 → 7 tools, rings = 2 → 19 tools, rings = 3 → 37 tools.
fn build_honeycomb_grid(
    rings: usize,
) -> (
    Topology,
    brepkit_topology::solid::SolidId,
    Vec<brepkit_topology::solid::SolidId>,
) {
    let mut topo = Topology::new();
    let target = primitives::make_box(&mut topo, 100.0, 100.0, 10.0).unwrap();

    let cx = 50.0;
    let cy = 50.0;
    let r = 3.0; // tool radius
    let spacing = 8.0; // center-to-center

    let mut tools = Vec::new();

    // Center tool
    let cyl = make_hex_prism(&mut topo, cx, cy, r, 20.0, -5.0);
    tools.push(cyl);

    // Hex ring offsets
    for ring in 1..=rings {
        let n = ring;
        // 6 directions in hex grid
        let dirs: [(f64, f64); 6] = [
            (1.0, 0.0),
            (0.5, 0.866_025_403_784_438_6),
            (-0.5, 0.866_025_403_784_438_6),
            (-1.0, 0.0),
            (-0.5, -0.866_025_403_784_438_6),
            (0.5, -0.866_025_403_784_438_6),
        ];
        // Walk along each edge of the hex ring
        for (side, &(dx, dy)) in dirs.iter().enumerate() {
            let next = dirs[(side + 2) % 6]; // perpendicular walk direction
            for step in 0..n {
                let hx = cx + spacing * (n as f64 * dx + step as f64 * next.0);
                let hy = cy + spacing * (n as f64 * dy + step as f64 * next.1);

                // Only add if inside the box with margin
                if hx > r && hx < 100.0 - r && hy > r && hy < 100.0 - r {
                    let cyl = make_hex_prism(&mut topo, hx, hy, r, 20.0, -5.0);
                    tools.push(cyl);
                }
            }
        }
    }

    (topo, target, tools)
}

// ---------------------------------------------------------------------------
// Benchmark: compound_cut vs sequential for cylinder grids
// ---------------------------------------------------------------------------

fn bench_compound_cut_cylinders(c: &mut Criterion) {
    let mut group = c.benchmark_group("compound_cut_cylinders");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for &n in &[4, 16, 36, 64] {
        // compound_cut (single-pass)
        group.bench_function(format!("compound_N={n}"), |b| {
            let (base_topo, target, tools) = build_cylinder_grid(n);
            b.iter(|| {
                let mut topo = base_topo.clone();
                black_box(
                    compound_cut(&mut topo, target, &tools, BooleanOptions::default()).unwrap(),
                );
            });
        });

        // sequential (one boolean per tool)
        group.bench_function(format!("sequential_N={n}"), |b| {
            let (base_topo, target, tools) = build_cylinder_grid(n);
            b.iter(|| {
                let mut topo = base_topo.clone();
                let mut result = target;
                for &tool in &tools {
                    result = boolean(&mut topo, BooleanOp::Cut, result, tool).unwrap();
                }
                black_box(result);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: compound_cut vs sequential for honeycomb patterns
// ---------------------------------------------------------------------------

fn bench_compound_cut_honeycomb(c: &mut Criterion) {
    let mut group = c.benchmark_group("compound_cut_honeycomb");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for &rings in &[1, 2, 3, 5] {
        let (base_topo, target, tools) = build_honeycomb_grid(rings);
        let n = tools.len();

        group.bench_function(format!("compound_rings={rings}_N={n}"), |b| {
            b.iter(|| {
                let mut topo = base_topo.clone();
                black_box(
                    compound_cut(&mut topo, target, &tools, BooleanOptions::default()).unwrap(),
                );
            });
        });

        group.bench_function(format!("sequential_rings={rings}_N={n}"), |b| {
            let (base_topo, target, tools) = build_honeycomb_grid(rings);
            b.iter(|| {
                let mut topo = base_topo.clone();
                let mut result = target;
                for &tool in &tools {
                    result = boolean(&mut topo, BooleanOp::Cut, result, tool).unwrap();
                }
                black_box(result);
            });
        });
    }

    group.finish();
}

// ===========================================================================
// Criterion setup
// ===========================================================================

criterion_group!(
    compound_cut_scaling,
    bench_compound_cut_cylinders,
    bench_compound_cut_honeycomb,
);

criterion_main!(compound_cut_scaling);
