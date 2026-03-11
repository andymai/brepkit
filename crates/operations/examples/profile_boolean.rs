//! Profiling harness for boolean + tessellation operations.
//!
//! Run with per-phase timing:
//!   cargo run --profile profiling --example profile_boolean -- honeycomb
//!   cargo run --profile profiling --example profile_boolean -- cylinders
//!   cargo run --profile profiling --example profile_boolean -- fuse
//!   cargo run --profile profiling --example profile_boolean -- large-honeycomb
//!   cargo run --profile profiling --example profile_boolean -- scale    # scaling study
//!   cargo run --profile profiling --example profile_boolean -- xl       # 200+ tool stress
//!   cargo run --profile profiling --example profile_boolean -- tess     # tessellation focus
//!
//! For flamegraphs:
//!   cargo flamegraph --profile profiling --example profile_boolean -o flame.svg -- honeycomb

#![allow(clippy::unwrap_used, missing_docs)]

use std::time::Instant;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOptions, compound_cut};
use brepkit_operations::compound_ops::fuse_all;
use brepkit_operations::primitives;
use brepkit_operations::tessellate::tessellate_solid;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::compound::Compound;

// ---------------------------------------------------------------------------
// Grid builders (shared with benchmarks)
// ---------------------------------------------------------------------------

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

fn make_hex_prism(
    topo: &mut Topology,
    cx: f64,
    cy: f64,
    circumradius: f64,
    height: f64,
    z_offset: f64,
) -> brepkit_topology::solid::SolidId {
    let side = circumradius * 1.732;
    let bx = primitives::make_box(topo, side, side, height).unwrap();
    let mat = Mat4::translation(cx - side / 2.0, cy - side / 2.0, z_offset);
    transform_solid(topo, bx, &mat).unwrap();
    bx
}

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
    let r = 3.0;
    let spacing = 8.0;

    let mut tools = Vec::new();
    let cyl = make_hex_prism(&mut topo, cx, cy, r, 20.0, -5.0);
    tools.push(cyl);

    for ring in 1..=rings {
        let n = ring;
        let dirs: [(f64, f64); 6] = [
            (1.0, 0.0),
            (0.5, 0.866_025_403_784_438_6),
            (-0.5, 0.866_025_403_784_438_6),
            (-1.0, 0.0),
            (-0.5, -0.866_025_403_784_438_6),
            (0.5, -0.866_025_403_784_438_6),
        ];
        for (side, &(dx, dy)) in dirs.iter().enumerate() {
            let next = dirs[(side + 2) % 6];
            for step in 0..n {
                let hx = cx + spacing * (n as f64 * dx + step as f64 * next.0);
                let hy = cy + spacing * (n as f64 * dy + step as f64 * next.1);
                if hx > r && hx < 100.0 - r && hy > r && hy < 100.0 - r {
                    let cyl = make_hex_prism(&mut topo, hx, hy, r, 20.0, -5.0);
                    tools.push(cyl);
                }
            }
        }
    }

    (topo, target, tools)
}

fn build_overlapping_box_grid(side: usize) -> (Topology, Vec<brepkit_topology::solid::SolidId>) {
    let mut topo = Topology::new();
    let size = 1.01;
    let mut solids = Vec::with_capacity(side * side);
    for row in 0..side {
        for col in 0..side {
            let bx = primitives::make_box(&mut topo, size, size, 1.0).unwrap();
            let mat = Mat4::translation(col as f64, row as f64, 0.0);
            transform_solid(&mut topo, bx, &mat).unwrap();
            solids.push(bx);
        }
    }
    (topo, solids)
}

fn build_touching_box_grid(side: usize) -> (Topology, Vec<brepkit_topology::solid::SolidId>) {
    let mut topo = Topology::new();
    let mut solids = Vec::with_capacity(side * side);
    for row in 0..side {
        for col in 0..side {
            let bx = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
            let mat = Mat4::translation(col as f64, row as f64, 0.0);
            transform_solid(&mut topo, bx, &mat).unwrap();
            solids.push(bx);
        }
    }
    (topo, solids)
}

// ---------------------------------------------------------------------------
// Workloads
// ---------------------------------------------------------------------------

fn print_bool_result(bool_ms: f64, face_count: usize) {
    println!("  Done in {bool_ms:.1}ms — {face_count} faces");
}

fn print_tess_result(
    bool_ms: f64,
    face_count: usize,
    topo: &Topology,
    result: brepkit_topology::solid::SolidId,
) {
    let t1 = Instant::now();
    let mesh = tessellate_solid(topo, result, 0.1).unwrap();
    let tess_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!(
        "  bool={bool_ms:.1}ms  tess={tess_ms:.1}ms  total={:.1}ms — {face_count} faces, {} tris, {} verts",
        bool_ms + tess_ms,
        mesh.indices.len() / 3,
        mesh.positions.len(),
    );
}

fn run_honeycomb(rings: usize, tess: bool) {
    let (mut topo, target, tools) = build_honeycomb_grid(rings);
    println!(
        "Honeycomb rings={rings}: {} tools, starting compound_cut...",
        tools.len()
    );
    let t0 = Instant::now();
    let result = compound_cut(&mut topo, target, &tools, BooleanOptions::default()).unwrap();
    let bool_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let shell_id = topo.solid(result).unwrap().outer_shell();
    let face_count = topo.shell(shell_id).unwrap().faces().len();

    if tess {
        print_tess_result(bool_ms, face_count, &topo, result);
    } else {
        print_bool_result(bool_ms, face_count);
    }
}

fn run_cylinders(n: usize, tess: bool) {
    let (mut topo, target, tools) = build_cylinder_grid(n);
    println!("Cylinder grid: {n} tools, starting compound_cut...");
    let t0 = Instant::now();
    let result = compound_cut(&mut topo, target, &tools, BooleanOptions::default()).unwrap();
    let bool_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let shell_id = topo.solid(result).unwrap().outer_shell();
    let face_count = topo.shell(shell_id).unwrap().faces().len();

    if tess {
        print_tess_result(bool_ms, face_count, &topo, result);
    } else {
        print_bool_result(bool_ms, face_count);
    }
}

fn run_fuse(side: usize) {
    let (mut topo, solids) = build_overlapping_box_grid(side);
    let n = solids.len();
    println!("Fuse overlapping: {side}×{side} = {n} boxes...");
    let cid = topo.compounds.alloc(Compound::new(solids));
    let t0 = Instant::now();
    let result = fuse_all(&mut topo, cid).unwrap();
    let elapsed = t0.elapsed();

    let shell_id = topo.solid(result).unwrap().outer_shell();
    let face_count = topo.shell(shell_id).unwrap().faces().len();
    let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.1).unwrap();
    println!(
        "  Done in {:.1}ms — {} faces, volume={:.2}",
        elapsed.as_secs_f64() * 1000.0,
        face_count,
        vol
    );
}

fn run_fuse_touching(side: usize) {
    let (mut topo, solids) = build_touching_box_grid(side);
    let n = solids.len();
    println!("Fuse touching: {side}×{side} = {n} boxes...");
    let cid = topo.compounds.alloc(Compound::new(solids));
    let t0 = Instant::now();
    let result = fuse_all(&mut topo, cid).unwrap();
    let elapsed = t0.elapsed();

    let shell_id = topo.solid(result).unwrap().outer_shell();
    let face_count = topo.shell(shell_id).unwrap().faces().len();
    let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.1).unwrap();
    println!(
        "  Done in {:.1}ms — {} faces, volume={:.2}",
        elapsed.as_secs_f64() * 1000.0,
        face_count,
        vol
    );
}

fn main() {
    // Initialize logger — use RUST_LOG=debug for per-phase timing.
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let workload = args.get(1).map(String::as_str).unwrap_or("all");

    match workload {
        "honeycomb" => {
            run_honeycomb(3, false);
        }
        "large-honeycomb" => {
            run_honeycomb(5, false);
        }
        "cylinders" => {
            run_cylinders(64, false);
        }
        "fuse" => {
            run_fuse(4);
            run_fuse_touching(4);
        }
        "xl" => {
            // Stress tests: 200+ tools
            run_honeycomb(8, false);
            run_cylinders(256, false);
            run_fuse(8);
            run_fuse_touching(8);
        }
        "scale" => {
            // Scaling study: increasing tool counts
            println!("=== Honeycomb scaling ===");
            for rings in [3, 5, 8, 10] {
                run_honeycomb(rings, false);
            }
            println!("\n=== Cylinder scaling ===");
            for n in [16, 64, 144, 256] {
                run_cylinders(n, false);
            }
            println!("\n=== Fuse overlapping scaling ===");
            for side in [3, 4, 6, 8, 12, 16, 22] {
                run_fuse(side);
            }
            println!("\n=== Fuse touching scaling ===");
            for side in [3, 4, 6, 8, 12, 16, 22] {
                run_fuse_touching(side);
            }
        }
        "tess" => {
            println!("=== Tessellation profiling (bool + tess) ===");
            println!("\n--- Honeycomb (planar-dominated) ---");
            for rings in [3, 5, 8] {
                run_honeycomb(rings, true);
            }
            println!("\n--- Cylinders (analytic curved faces) ---");
            for n in [16, 64, 144, 256] {
                run_cylinders(n, true);
            }
        }
        "all" => {
            run_honeycomb(3, false);
            run_cylinders(64, false);
            run_honeycomb(5, false);
            run_fuse(4);
            run_fuse_touching(4);
        }
        other => {
            eprintln!(
                "Unknown workload: {other}\n\
                 Usage: profile_boolean [honeycomb|large-honeycomb|cylinders|fuse|xl|scale|all]"
            );
            std::process::exit(1);
        }
    }
}
