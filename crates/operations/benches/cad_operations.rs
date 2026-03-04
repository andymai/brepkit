//! Performance benchmarks for core CAD operations.
//!
//! These benchmarks mirror the operations and parameters in
//! `brepjs/benchmarks/kernel-comparison.bench.test.ts` for 1:1 comparison
//! against OCCT. Each benchmark name matches the JS counterpart.
//!
//! Run with: `cargo bench -p brepkit-operations`

#![allow(
    clippy::unwrap_used,
    clippy::missing_docs_in_private_items,
    missing_docs
)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Vec3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::copy::copy_solid;
use brepkit_operations::fillet::fillet;
use brepkit_operations::measure;
use brepkit_operations::pattern;
use brepkit_operations::primitives;
use brepkit_operations::shell_op;
use brepkit_operations::tessellate;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Collect all unique edge IDs from a solid's outer shell.
fn collect_edges(
    topo: &Topology,
    solid: brepkit_topology::solid::SolidId,
) -> Vec<brepkit_topology::edge::EdgeId> {
    let shell_id = topo.solid(solid).unwrap().outer_shell();
    let shell = topo.shell(shell_id).unwrap();
    let mut edges = Vec::new();
    for &fid in shell.faces() {
        let face = topo.face(fid).unwrap();
        let wire = topo.wire(face.outer_wire()).unwrap();
        for oe in wire.edges() {
            if !edges.contains(&oe.edge()) {
                edges.push(oe.edge());
            }
        }
    }
    edges
}

/// Tessellate all faces of a solid into a combined mesh.
fn tessellate_solid(
    topo: &Topology,
    solid: brepkit_topology::solid::SolidId,
    deflection: f64,
) -> tessellate::TriangleMesh {
    let shell_id = topo.solid(solid).unwrap().outer_shell();
    let shell = topo.shell(shell_id).unwrap();
    let mut combined = tessellate::TriangleMesh::default();
    for &fid in shell.faces() {
        if let Ok(mesh) = tessellate::tessellate(topo, fid, deflection) {
            let offset = combined.positions.len() as u32;
            combined.positions.extend_from_slice(&mesh.positions);
            combined.normals.extend_from_slice(&mesh.normals);
            combined
                .indices
                .extend(mesh.indices.iter().map(|i| i + offset));
        }
    }
    combined
}

// ===========================================================================
// Primitives — matches kernel-comparison "Primitives" group
// ===========================================================================

/// `makeBox(10,20,30)` ×100 — matches JS benchmark loop count.
fn bench_make_box_x100(c: &mut Criterion) {
    c.bench_function("makeBox(10,20,30) x100", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let mut topo = Topology::new();
                black_box(primitives::make_box(&mut topo, 10.0, 20.0, 30.0).unwrap());
            }
        });
    });
}

/// `makeCylinder(5,20)` ×100 — matches JS: `k.makeCylinder(5, 20)`.
fn bench_make_cylinder_x100(c: &mut Criterion) {
    c.bench_function("makeCylinder(5,20) x100", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let mut topo = Topology::new();
                black_box(primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap());
            }
        });
    });
}

/// `makeSphere(10)` ×100 — matches JS: `k.makeSphere(10)`.
fn bench_make_sphere_x100(c: &mut Criterion) {
    c.bench_function("makeSphere(10) x100", |b| {
        b.iter(|| {
            for _ in 0..100 {
                let mut topo = Topology::new();
                black_box(primitives::make_sphere(&mut topo, 10.0, 16).unwrap());
            }
        });
    });
}

// ===========================================================================
// Booleans — matches kernel-comparison "Booleans" group
// ===========================================================================

/// `fuse(box(10,10,10), translate(box(5,5,5), 5,5,5))` ×10.
fn bench_fuse_box_box_x10(c: &mut Criterion) {
    c.bench_function("fuse(box,box) x10", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let mut topo = Topology::new();
                let a = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
                let b_solid = primitives::make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
                let mat = Mat4::translation(5.0, 5.0, 5.0);
                transform_solid(&mut topo, b_solid, &mat).unwrap();
                black_box(boolean(&mut topo, BooleanOp::Fuse, a, b_solid).unwrap());
            }
        });
    });
}

/// `cut(box(10,10,10), cylinder(3,20))` ×10.
fn bench_cut_box_cyl_x10(c: &mut Criterion) {
    c.bench_function("cut(box,cyl) x10", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let mut topo = Topology::new();
                let a = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
                let cyl = primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();
                black_box(boolean(&mut topo, BooleanOp::Cut, a, cyl).unwrap());
            }
        });
    });
}

/// `intersect(box(10,10,10), sphere(8))` ×10.
fn bench_intersect_box_sphere_x10(c: &mut Criterion) {
    c.bench_function("intersect(box,sphere) x10", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let mut topo = Topology::new();
                let a = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
                let sph = primitives::make_sphere(&mut topo, 8.0, 16).unwrap();
                black_box(boolean(&mut topo, BooleanOp::Intersect, a, sph).unwrap());
            }
        });
    });
}

// ===========================================================================
// Transforms — matches kernel-comparison "Transforms" group
// ===========================================================================

/// `translate` ×1000 — chain of small translations on a box.
fn bench_translate_x1000(c: &mut Criterion) {
    c.bench_function("translate x1000", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            let mut solid = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
            let mat = Mat4::translation(0.01, 0.0, 0.0);
            for _ in 0..1000 {
                let copy = copy_solid(&mut topo, solid).unwrap();
                transform_solid(&mut topo, copy, &mat).unwrap();
                solid = copy;
            }
            black_box(solid);
        });
    });
}

/// `rotate` ×100 — chain of 3.6° rotations around Z.
fn bench_rotate_x100(c: &mut Criterion) {
    c.bench_function("rotate x100", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            let mut solid = primitives::make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
            let angle_rad = 3.6_f64.to_radians();
            for _ in 0..100 {
                let copy = copy_solid(&mut topo, solid).unwrap();
                let mat = Mat4::rotation_z(angle_rad);
                transform_solid(&mut topo, copy, &mat).unwrap();
                solid = copy;
            }
            black_box(solid);
        });
    });
}

// ===========================================================================
// Meshing — matches kernel-comparison "Meshing" group
// ===========================================================================

/// `mesh box (tol=0.1)` — coarse tessellation of box(10,10,10).
fn bench_mesh_box_coarse(c: &mut Criterion) {
    c.bench_function("mesh box (tol=0.1)", |b| {
        let mut topo = Topology::new();
        let solid = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        b.iter(|| {
            black_box(tessellate_solid(&topo, solid, 0.1));
        });
    });
}

/// `mesh sphere (tol=0.01)` — fine tessellation of sphere(10).
/// This is the key comparison: OCCT does 63.6 ms for this.
fn bench_mesh_sphere_fine(c: &mut Criterion) {
    c.bench_function("mesh sphere (tol=0.01)", |b| {
        let mut topo = Topology::new();
        let solid = primitives::make_sphere(&mut topo, 10.0, 16).unwrap();
        b.iter(|| {
            black_box(tessellate_solid(&topo, solid, 0.01));
        });
    });
}

// ===========================================================================
// Measurement — matches kernel-comparison "Measurement" group
// ===========================================================================

/// `volume` ×100 on box(10,10,10).
fn bench_volume_x100(c: &mut Criterion) {
    c.bench_function("volume x100", |b| {
        let mut topo = Topology::new();
        let solid = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        b.iter(|| {
            for _ in 0..100 {
                black_box(measure::solid_volume(&topo, solid, 0.1).unwrap());
            }
        });
    });
}

/// `boundingBox` ×100 on box(10,10,10).
fn bench_bounding_box_x100(c: &mut Criterion) {
    c.bench_function("boundingBox x100", |b| {
        let mut topo = Topology::new();
        let solid = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        b.iter(|| {
            for _ in 0..100 {
                black_box(measure::solid_bounding_box(&topo, solid).unwrap());
            }
        });
    });
}

// ===========================================================================
// End-to-end — matches kernel-comparison "End-to-end model" group
// ===========================================================================

/// `box(20,20,20)` + chamfer all edges r=1.
fn bench_box_chamfer_all(c: &mut Criterion) {
    c.bench_function("box+chamfer", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            let solid = primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
            let edges = collect_edges(&topo, solid);
            black_box(chamfer(&mut topo, solid, &edges, 1.0).unwrap());
        });
    });
}

/// `box(20,20,20)` + fillet all edges r=1.
fn bench_box_fillet_all(c: &mut Criterion) {
    c.bench_function("box+fillet", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            let solid = primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
            let edges = collect_edges(&topo, solid);
            black_box(fillet(&mut topo, solid, &edges, 1.0).unwrap());
        });
    });
}

/// Multi-boolean: plate(50,50,10) with 4 cylinder holes.
fn bench_multi_boolean(c: &mut Criterion) {
    c.bench_function("multi-boolean model", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            let mut result = primitives::make_box(&mut topo, 50.0, 50.0, 10.0).unwrap();

            // Punch holes: JS does x,y in {-15,-5,5,15} = 4×4 = 16 holes
            let positions: &[f64] = &[-15.0, -5.0, 5.0, 15.0];
            for &x in positions {
                for &y in positions {
                    let cyl = primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();
                    let mat = Mat4::translation(x, y, -5.0);
                    transform_solid(&mut topo, cyl, &mat).unwrap();
                    result = boolean(&mut topo, BooleanOp::Cut, result, cyl).unwrap();
                }
            }
            black_box(result);
        });
    });
}

// ===========================================================================
// Additional: single-iteration benchmarks (for micro-level comparison)
// ===========================================================================

/// Single `make_sphere(10)` — amortization-free timing.
fn bench_make_sphere_single(c: &mut Criterion) {
    c.bench_function("makeSphere(10) single", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            black_box(primitives::make_sphere(&mut topo, 10.0, 16).unwrap());
        });
    });
}

/// Single `make_cylinder(5,20)` — amortization-free timing.
fn bench_make_cylinder_single(c: &mut Criterion) {
    c.bench_function("makeCylinder(5,20) single", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            black_box(primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap());
        });
    });
}

// ===========================================================================
// Optimization A/B comparisons
// ===========================================================================

/// Phase 2: copy+transform vs fused copy_and_transform_solid
fn bench_translate_fused_x1000(c: &mut Criterion) {
    use brepkit_operations::copy::copy_and_transform_solid;

    let mut group = c.benchmark_group("translate_x1000_ab");

    group.bench_function("copy+transform (old)", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            let mut solid = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
            let mat = Mat4::translation(0.01, 0.0, 0.0);
            for _ in 0..1000 {
                let copy = copy_solid(&mut topo, solid).unwrap();
                transform_solid(&mut topo, copy, &mat).unwrap();
                solid = copy;
            }
            black_box(solid);
        });
    });

    group.bench_function("copy_and_transform (new)", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            let mut solid = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
            let mat = Mat4::translation(0.01, 0.0, 0.0);
            for _ in 0..1000 {
                solid = copy_and_transform_solid(&mut topo, solid, &mat).unwrap();
            }
            black_box(solid);
        });
    });

    group.finish();
}

/// Phase 3: intersect(box, sphere) — single iteration to see analytic fast path
fn bench_intersect_box_sphere_single(c: &mut Criterion) {
    c.bench_function("intersect(box,sphere) single", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            let bx = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            let sp = primitives::make_sphere(&mut topo, 7.0, 16).unwrap();
            let result = boolean(&mut topo, BooleanOp::Intersect, bx, sp).ok();
            black_box(result);
        });
    });
}

// ===========================================================================
// Shell — hollow solid creation
// ===========================================================================

/// `shell(box(20,20,20), thickness=1, open_top)` — gridfinity bin base shape.
fn bench_shell_box(c: &mut Criterion) {
    c.bench_function("shell(box, t=1, open_top)", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            let solid = primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
            // Find top face (highest z normal).
            let shell_id = topo.solid(solid).unwrap().outer_shell();
            let faces: Vec<_> = topo.shell(shell_id).unwrap().faces().to_vec();
            let top_face = *faces.last().unwrap(); // convention: top face is last
            black_box(shell_op::shell(&mut topo, solid, 1.0, &[top_face]).unwrap());
        });
    });
}

// ===========================================================================
// Patterns — linear, circular, grid
// ===========================================================================

/// `linear_pattern(box, +X, spacing=5, count=10)`.
fn bench_linear_pattern_10(c: &mut Criterion) {
    c.bench_function("linearPattern(box, 10)", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            let solid = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
            black_box(
                pattern::linear_pattern(&mut topo, solid, Vec3::new(1.0, 0.0, 0.0), 5.0, 10)
                    .unwrap(),
            );
        });
    });
}

/// `grid_pattern(box, 3×3, spacing=5)` — baseplate-like grid.
fn bench_grid_pattern_3x3(c: &mut Criterion) {
    c.bench_function("gridPattern(box, 3x3)", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            let solid = primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
            black_box(
                pattern::grid_pattern(
                    &mut topo,
                    solid,
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(0.0, 1.0, 0.0),
                    5.0,
                    5.0,
                    3,
                    3,
                )
                .unwrap(),
            );
        });
    });
}

// ===========================================================================
// Gridfinity — end-to-end bin/baseplate generation
// ===========================================================================

/// Simplified 1×1 gridfinity bin: box + shell + chamfer base edges.
fn bench_gridfinity_1x1_bin(c: &mut Criterion) {
    c.bench_function("gridfinity 1x1 bin (box+shell+chamfer)", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            // 42mm × 42mm × 21mm (1u height)
            let solid = primitives::make_box(&mut topo, 42.0, 42.0, 21.0).unwrap();

            // Open-top shell (thickness 1.6mm wall)
            let shell_id = topo.solid(solid).unwrap().outer_shell();
            let faces: Vec<_> = topo.shell(shell_id).unwrap().faces().to_vec();
            let top_face = *faces.last().unwrap();
            let shelled = shell_op::shell(&mut topo, solid, 1.6, &[top_face]).unwrap();

            // Chamfer bottom edges (substitute for fillet until vertex blending lands)
            let bottom_edges = collect_edges(&topo, shelled);
            if let Ok(result) = chamfer(&mut topo, shelled, &bottom_edges[..4], 0.8) {
                black_box(result);
            } else {
                black_box(shelled);
            }
        });
    });
}

/// 3×3 baseplate: box + grid pattern + cylinder holes (boolean cuts).
fn bench_gridfinity_3x3_baseplate(c: &mut Criterion) {
    c.bench_function("gridfinity 3x3 baseplate (grid+holes)", |b| {
        b.iter(|| {
            let mut topo = Topology::new();
            // Single baseplate unit: 42mm × 42mm × 4.65mm
            let unit = primitives::make_box(&mut topo, 42.0, 42.0, 4.65).unwrap();

            // 3×3 grid (spacing = 42mm)
            let grid = pattern::grid_pattern(
                &mut topo,
                unit,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                42.0,
                42.0,
                3,
                3,
            )
            .unwrap();

            // Punch 4 magnet holes in the first unit
            let first_solid = topo.compound(grid).unwrap().solids()[0];
            let mut result = first_solid;
            let hole_offsets: &[(f64, f64)] = &[(4.0, 4.0), (38.0, 4.0), (4.0, 38.0), (38.0, 38.0)];
            for &(x, y) in hole_offsets {
                let cyl = primitives::make_cylinder(&mut topo, 3.0, 5.0).unwrap();
                let mat = Mat4::translation(x, y, -0.5);
                transform_solid(&mut topo, cyl, &mat).unwrap();
                result = boolean(&mut topo, BooleanOp::Cut, result, cyl).unwrap();
            }
            black_box(result);
        });
    });
}

// ===========================================================================
// Criterion groups — organized to mirror JS benchmark sections
// ===========================================================================

criterion_group!(
    primitives_bench,
    bench_make_box_x100,
    bench_make_cylinder_x100,
    bench_make_sphere_x100,
    bench_make_sphere_single,
    bench_make_cylinder_single,
);

criterion_group!(
    booleans_bench,
    bench_fuse_box_box_x10,
    bench_cut_box_cyl_x10,
    bench_intersect_box_sphere_x10,
);

criterion_group!(transforms_bench, bench_translate_x1000, bench_rotate_x100,);

criterion_group!(meshing_bench, bench_mesh_box_coarse, bench_mesh_sphere_fine,);

criterion_group!(
    measurement_bench,
    bench_volume_x100,
    bench_bounding_box_x100,
);

criterion_group!(shell_bench, bench_shell_box,);

criterion_group!(
    pattern_bench,
    bench_linear_pattern_10,
    bench_grid_pattern_3x3,
);

criterion_group!(
    endtoend_bench,
    bench_box_chamfer_all,
    bench_box_fillet_all,
    bench_multi_boolean,
);

criterion_group!(
    gridfinity_bench,
    bench_gridfinity_1x1_bin,
    bench_gridfinity_3x3_baseplate,
);

criterion_group!(
    optimization_bench,
    bench_translate_fused_x1000,
    bench_intersect_box_sphere_single,
);

criterion_main!(
    primitives_bench,
    booleans_bench,
    transforms_bench,
    meshing_bench,
    measurement_bench,
    shell_bench,
    pattern_bench,
    endtoend_bench,
    gridfinity_bench,
    optimization_bench,
);
