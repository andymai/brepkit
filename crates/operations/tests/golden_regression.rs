//! Golden file regression tests for brepkit.
//!
//! These tests compare operation output against known-good reference data
//! stored in `tests/golden/data/`. Run with `UPDATE_GOLDEN=1` to regenerate.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::format_push_string,
    clippy::panic
)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use brepkit_math::mat::Mat4;
use brepkit_topology::Topology;

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{solid_bounding_box, solid_center_of_mass, solid_volume};
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere};
use brepkit_operations::tessellate::tessellate_solid;
use brepkit_operations::transform::transform_solid;

// ── Golden file helpers ─────────────────────────────────────────────

fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/data")
        .join(name)
}

fn assert_golden(name: &str, actual: &str) {
    let path = golden_path(name);
    if std::env::var("UPDATE_GOLDEN").is_ok() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, actual).unwrap();
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "Golden file not found: {}\nRun with UPDATE_GOLDEN=1 to create it.",
            path.display()
        )
    });
    assert_eq!(
        actual.trim(),
        expected.trim(),
        "Golden file mismatch: {name}"
    );
}

fn round6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

// ── Measurement snapshot ────────────────────────────────────────────

struct Measurements {
    vertex_count: usize,
    triangle_count: usize,
    volume: f64,
    bbox_min: [f64; 3],
    bbox_max: [f64; 3],
    com: [f64; 3],
}

fn measure_solid(topo: &Topology, solid: brepkit_topology::solid::SolidId) -> Measurements {
    let mesh = tessellate_solid(topo, solid, 0.1).unwrap();
    let vol = solid_volume(topo, solid, 0.1).unwrap();
    let bbox = solid_bounding_box(topo, solid).unwrap();
    let com = solid_center_of_mass(topo, solid, 0.1).unwrap();

    Measurements {
        vertex_count: mesh.positions.len(),
        triangle_count: mesh.indices.len() / 3,
        volume: vol,
        bbox_min: [bbox.min.x(), bbox.min.y(), bbox.min.z()],
        bbox_max: [bbox.max.x(), bbox.max.y(), bbox.max.z()],
        com: [com.x(), com.y(), com.z()],
    }
}

fn format_measurements(header: &str, m: &Measurements) -> String {
    format!(
        "# {header}\n\
         vertices: {}\n\
         triangles: {}\n\
         volume: {:.6}\n\
         bbox_min: ({:.6}, {:.6}, {:.6})\n\
         bbox_max: ({:.6}, {:.6}, {:.6})\n\
         com: ({:.6}, {:.6}, {:.6})\n",
        m.vertex_count,
        m.triangle_count,
        round6(m.volume),
        round6(m.bbox_min[0]),
        round6(m.bbox_min[1]),
        round6(m.bbox_min[2]),
        round6(m.bbox_max[0]),
        round6(m.bbox_max[1]),
        round6(m.bbox_max[2]),
        round6(m.com[0]),
        round6(m.com[1]),
        round6(m.com[2]),
    )
}

// ── Sorted vertex dump (for small meshes) ───────────────────────────

fn format_sorted_vertices(topo: &Topology, solid: brepkit_topology::solid::SolidId) -> String {
    let mesh = tessellate_solid(topo, solid, 0.1).unwrap();
    let mut verts: Vec<[f64; 3]> = mesh
        .positions
        .iter()
        .map(|p| [round6(p.x()), round6(p.y()), round6(p.z())])
        .collect();
    // Deduplicate
    verts.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then(a[1].total_cmp(&b[1]))
            .then(a[2].total_cmp(&b[2]))
    });
    verts.dedup();
    let mut out = String::new();
    for v in &verts {
        let _ = writeln!(out, "v {:.6} {:.6} {:.6}", v[0], v[1], v[2]);
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════
// Golden Tests
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn golden_box_10x20x30() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 20.0, 30.0).unwrap();

    let m = measure_solid(&topo, solid);
    let mut output = format_measurements("box 10x20x30", &m);
    output.push_str(&format_sorted_vertices(&topo, solid));

    assert_golden("box_10x20x30.golden", &output);
}

#[test]
fn golden_cylinder_r5_h20() {
    let mut topo = Topology::new();
    let solid = make_cylinder(&mut topo, 5.0, 20.0).unwrap();

    let m = measure_solid(&topo, solid);
    let output = format_measurements("cylinder r=5 h=20", &m);

    assert_golden("cylinder_r5_h20.golden", &output);
}

#[test]
fn golden_sphere_r3() {
    let mut topo = Topology::new();
    let solid = make_sphere(&mut topo, 3.0, 16).unwrap();

    let m = measure_solid(&topo, solid);
    let output = format_measurements("sphere r=3", &m);

    assert_golden("sphere_r3.golden", &output);
}

#[test]
fn golden_cone_r2_r0_h5() {
    let mut topo = Topology::new();
    let solid = make_cone(&mut topo, 2.0, 0.0, 5.0).unwrap();

    let m = measure_solid(&topo, solid);
    let output = format_measurements("cone r_bottom=2 r_top=0 h=5", &m);

    assert_golden("cone_r2_r0_h5.golden", &output);
}

#[test]
fn golden_boolean_box_minus_cylinder() {
    let mut topo = Topology::new();
    let box_solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let cyl = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    // Move cylinder to center of box
    transform_solid(&mut topo, cyl, &Mat4::translation(5.0, 5.0, 0.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Cut, box_solid, cyl).unwrap();

    let m = measure_solid(&topo, result);
    let output = format_measurements("box 10x10x10 minus cylinder r=3 h=10 at center", &m);

    assert_golden("boolean_box_minus_cylinder.golden", &output);
}

#[test]
fn golden_boolean_intersect_two_boxes() {
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    // Offset second box by (5, 5, 5) — overlap is a 5x5x5 cube
    transform_solid(&mut topo, b, &Mat4::translation(5.0, 5.0, 5.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();

    let m = measure_solid(&topo, result);
    let mut output = format_measurements("intersect box [0,10]^3 with box [5,15]^3 → [5,10]^3", &m);
    output.push_str(&format_sorted_vertices(&topo, result));

    assert_golden("boolean_intersect_two_boxes.golden", &output);
}

#[test]
fn golden_tessellation_sphere() {
    let mut topo = Topology::new();
    let solid = make_sphere(&mut topo, 2.0, 16).unwrap();

    let mesh = tessellate_solid(&topo, solid, 0.1).unwrap();

    // Compute min/max vertex distances from origin
    let distances: Vec<f64> = mesh
        .positions
        .iter()
        .map(|p| (p.x() * p.x() + p.y() * p.y() + p.z() * p.z()).sqrt())
        .collect();
    let min_dist = distances.iter().copied().fold(f64::INFINITY, f64::min);
    let max_dist = distances.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let output = format!(
        "# sphere r=2 tessellation (deflection=0.1)\n\
         vertices: {}\n\
         triangles: {}\n\
         min_distance_from_origin: {:.6}\n\
         max_distance_from_origin: {:.6}\n",
        mesh.positions.len(),
        mesh.indices.len() / 3,
        round6(min_dist),
        round6(max_dist),
    );

    assert_golden("tessellation_sphere.golden", &output);
}

#[test]
fn golden_boolean_fuse_two_boxes() {
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 6.0, 6.0, 6.0).unwrap();
    let b = make_box(&mut topo, 6.0, 6.0, 6.0).unwrap();
    // Offset second box by (3, 0, 0) — creates an L-shaped union
    transform_solid(&mut topo, b, &Mat4::translation(3.0, 0.0, 0.0)).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();

    let m = measure_solid(&topo, result);
    let output = format_measurements("fuse box [0,6]^3 with box [3,0,0]+[6,6,6]", &m);

    assert_golden("boolean_fuse_two_boxes.golden", &output);
}
