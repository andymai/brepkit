//! Regression: sequential disjoint hex-prism cuts through one wall of a
//! hollow thin-walled box must each remove exactly the analytic prism
//! volume. Historically cut 2 onward was erratic (wrong deltas, gouges)
//! and from cut ~13 every cut silently no-opped while returning Ok.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Vec3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::primitives;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_planar_face_from_wire, make_regular_polygon_wire};

const HEX_R: f64 = 1.8;
const WEB: f64 = 0.8;
const WALL_T: f64 = 1.2;

fn make_hex_prism(topo: &mut Topology, r: f64, depth: f64) -> brepkit_topology::solid::SolidId {
    let wire = make_regular_polygon_wire(topo, r, 6, 1e-7).expect("hex wire");
    let face = make_planar_face_from_wire(topo, wire).expect("hex face");
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), depth).expect("hex prism")
}

fn make_hollow_bin(topo: &mut Topology) -> brepkit_topology::solid::SolidId {
    let outer = primitives::make_box(topo, 84.0, 84.0, 30.0).expect("outer");
    let cavity =
        primitives::make_box(topo, 84.0 - 2.0 * WALL_T, 84.0 - 2.0 * WALL_T, 30.0).expect("cavity");
    transform_solid(topo, cavity, &Mat4::translation(WALL_T, WALL_T, WALL_T)).expect("mv cavity");
    boolean(topo, BooleanOp::Cut, outer, cavity).expect("hollow shell")
}

/// Honeycomb centers on the front wall (y=0), prism axis along +y,
/// piercing the full 1.2mm wall thickness.
fn hex_centers(n: usize) -> Vec<(f64, f64)> {
    let col_sp = 3.0_f64.sqrt() * HEX_R + WEB;
    let row_sp = 1.5 * HEX_R + WEB;
    let mut centers = Vec::with_capacity(n.min(256));
    let mut row = 0usize;
    'outer: loop {
        let z = 5.0 + row as f64 * row_sp;
        if z > 27.0 {
            break;
        }
        let x_off = if row % 2 == 1 { col_sp / 2.0 } else { 0.0 };
        let mut col = 0usize;
        loop {
            let x = 5.0 + x_off + col as f64 * col_sp;
            if x > 79.0 {
                break;
            }
            if centers.len() >= n {
                break 'outer;
            }
            centers.push((x, z));
            col += 1;
        }
        row += 1;
    }
    centers
}

fn run_sequential_hex_cuts(n: usize) {
    let depth = WALL_T * 4.0;
    let hex_area = 1.5 * 3.0_f64.sqrt() * HEX_R * HEX_R;
    let expected_delta = hex_area * WALL_T;

    let mut topo = Topology::new();
    let mut result = make_hollow_bin(&mut topo);
    let mut prev_vol =
        brepkit_operations::measure::solid_volume(&topo, result, 0.1).expect("shell volume");

    let mut failures = Vec::new();
    for (i, (x, z)) in hex_centers(n).iter().enumerate() {
        let prism = make_hex_prism(&mut topo, HEX_R, depth);
        let rot = Mat4::rotation_x(-std::f64::consts::FRAC_PI_2);
        let mat = Mat4::translation(*x, WALL_T / 2.0 - depth / 2.0, *z) * rot;
        transform_solid(&mut topo, prism, &mat).expect("mv prism");
        result = boolean(&mut topo, BooleanOp::Cut, result, prism)
            .unwrap_or_else(|e| panic!("cut {} failed: {e}", i + 1));
        let vol =
            brepkit_operations::measure::solid_volume(&topo, result, 0.1).expect("cut volume");
        let delta = prev_vol - vol;
        if (delta - expected_delta).abs() > 0.05 {
            failures.push(format!(
                "cut {}: delta {delta:.3} expected {expected_delta:.3}",
                i + 1
            ));
        }
        prev_vol = vol;
    }
    assert!(
        failures.is_empty(),
        "{} of {n} cuts removed the wrong volume:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn hexwall_sequential_cuts_20() {
    run_sequential_hex_cuts(20);
}

/// Full honeycomb wall: every center the spacing pattern yields (133 with
/// these parameters). Also guards against the historical multi-minute
/// mesh-fallback churn — this must complete in seconds.
#[test]
fn hexwall_sequential_cuts_full() {
    run_sequential_hex_cuts(usize::MAX);
}
