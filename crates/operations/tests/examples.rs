//! # brepkit Examples
//!
//! Curated examples demonstrating common CAD workflows with brepkit.
//! Each test function is a self-contained, copy-paste-ready example.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;

/// Helper: extract the plane normal of a face, or return Z_AXIS for non-planar.
fn face_normal(topo: &Topology, face: brepkit_topology::face::FaceId) -> Vec3 {
    match topo.face(face).unwrap().surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => Vec3::new(0.0, 0.0, 0.0),
    }
}

// ═══════════════════════════════════════════════════════════════════
// Example 1: Basic Box with Fillet
// ═══════════════════════════════════════════════════════════════════

/// Create a 20×10×5 box and fillet all 12 edges with radius 1.
#[test]
fn example_box_with_fillet() {
    let mut topo = Topology::new();

    // Create a box: width=20, depth=10, height=5
    let solid = brepkit_operations::primitives::make_box(&mut topo, 20.0, 10.0, 5.0).unwrap();

    // Collect all edge IDs from the solid
    let edges = brepkit_topology::explorer::solid_edges(&topo, solid).unwrap();

    // Fillet every edge with radius 1.0
    let filleted =
        brepkit_operations::fillet::fillet_rolling_ball(&mut topo, solid, &edges, 1.0).unwrap();

    // Verify: filleted box has more faces than original (6 planar + 12 fillet + 8 blend)
    let shell = topo.solid(filleted).unwrap().outer_shell();
    let face_count = topo.shell(shell).unwrap().faces().len();
    assert!(
        face_count > 6,
        "filleted box should have more than 6 faces, got {face_count}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Example 2: Gridfinity Bin (Showcase)
// ═══════════════════════════════════════════════════════════════════

/// Build a simple gridfinity-style bin: box → shell (hollow) → chamfer base.
#[test]
fn example_gridfinity_bin() {
    let mut topo = Topology::new();

    // Outer box: 42×42×21 mm (standard gridfinity 1×1 bin)
    let solid = brepkit_operations::primitives::make_box(&mut topo, 42.0, 42.0, 21.0).unwrap();

    // Find the top face (highest Z normal) to open it
    let shell_id = topo.solid(solid).unwrap().outer_shell();
    let faces: Vec<_> = topo.shell(shell_id).unwrap().faces().to_vec();
    let top_face = faces
        .iter()
        .copied()
        .max_by(|&a, &b| {
            let na = face_normal(&topo, a);
            let nb = face_normal(&topo, b);
            na.z().partial_cmp(&nb.z()).unwrap()
        })
        .unwrap();

    // Shell: remove top face, offset inward by 1.5 mm wall thickness
    let hollowed = brepkit_operations::shell_op::shell(&mut topo, solid, 1.5, &[top_face]).unwrap();

    // Chamfer bottom edges (first 4 edges)
    let bottom_edges = brepkit_topology::explorer::solid_edges(&topo, hollowed).unwrap();
    let _chamfered =
        brepkit_operations::chamfer::chamfer(&mut topo, hollowed, &bottom_edges[..4], 0.8).unwrap();

    // Verify it has positive volume
    let vol = brepkit_operations::measure::solid_volume(&topo, hollowed, 0.1).unwrap();
    assert!(vol > 0.0, "bin should have positive volume");
}

// ═══════════════════════════════════════════════════════════════════
// Example 3: Boolean Operations (Union, Cut, Intersect)
// ═══════════════════════════════════════════════════════════════════

/// Demonstrate union and cut boolean operations.
#[test]
fn example_boolean_operations() {
    use brepkit_operations::boolean::{BooleanOp, boolean};

    let mut topo = Topology::new();

    // Create two overlapping boxes
    let box_a = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let box_b = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    // Translate box_b by (5,5,5) so they partially overlap
    let translate = Mat4::translation(5.0, 5.0, 5.0);
    brepkit_operations::transform::transform_solid(&mut topo, box_b, &translate).unwrap();

    // Union: combined volume of both boxes
    let fused = boolean(&mut topo, BooleanOp::Fuse, box_a, box_b).unwrap();
    let fused_vol = brepkit_operations::measure::solid_volume(&topo, fused, 0.1).unwrap();

    // Each box is 1000, overlap is 5×5×5=125, so union ≈ 1875
    assert!(
        (fused_vol - 1875.0).abs() < 10.0,
        "fused volume should be ~1875, got {fused_vol}"
    );

    // Cut: box_a minus box_b
    let box_a2 = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let box_b2 = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    brepkit_operations::transform::transform_solid(&mut topo, box_b2, &translate).unwrap();
    let cut = boolean(&mut topo, BooleanOp::Cut, box_a2, box_b2).unwrap();
    let cut_vol = brepkit_operations::measure::solid_volume(&topo, cut, 0.1).unwrap();
    assert!(
        (cut_vol - 875.0).abs() < 10.0,
        "cut volume should be ~875, got {cut_vol}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Example 4: Extrude a Profile
// ═══════════════════════════════════════════════════════════════════

/// Create a profile face from a box and extrude it upward.
#[test]
fn example_extrude_profile() {
    let mut topo = Topology::new();

    // Create a thin box as profile source
    let profile_box = brepkit_operations::primitives::make_box(&mut topo, 2.0, 2.0, 0.1).unwrap();
    let shell_id = topo.solid(profile_box).unwrap().outer_shell();
    let faces: Vec<_> = topo.shell(shell_id).unwrap().faces().to_vec();

    // Find bottom face (most negative Z normal)
    let bottom_face = faces
        .iter()
        .copied()
        .min_by(|&a, &b| {
            let na = face_normal(&topo, a);
            let nb = face_normal(&topo, b);
            na.z().partial_cmp(&nb.z()).unwrap()
        })
        .unwrap();

    // Extrude the bottom face upward by 10 units
    let extruded = brepkit_operations::extrude::extrude(
        &mut topo,
        bottom_face,
        Vec3::new(0.0, 0.0, 1.0),
        10.0,
    )
    .unwrap();

    // Volume should be 2 × 2 × 10 = 40
    let vol = brepkit_operations::measure::solid_volume(&topo, extruded, 0.1).unwrap();
    assert!(
        (vol - 40.0).abs() < 1.0,
        "extruded volume should be ~40, got {vol}"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Example 5: Validate and Heal a Solid
// ═══════════════════════════════════════════════════════════════════

/// Demonstrate the validate → heal → validate pipeline.
#[test]
fn example_validate_and_heal() {
    let mut topo = Topology::new();

    // Create a box and validate it
    let solid = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    let report = brepkit_operations::validate::validate_solid(&topo, solid).unwrap();
    assert!(
        report.issues.is_empty(),
        "fresh box should have no validation errors"
    );

    // The repair_solid convenience function chains validate → heal → validate
    let repair = brepkit_operations::heal::repair_solid(&mut topo, solid, 1e-7).unwrap();
    assert!(
        repair.after.issues.is_empty(),
        "repaired box should still have no errors"
    );
}

// ═══════════════════════════════════════════════════════════════════
// Example 6: Pattern (Linear and Circular)
// ═══════════════════════════════════════════════════════════════════

/// Create copies using linear and circular patterns.
#[test]
fn example_linear_and_circular_pattern() {
    let mut topo = Topology::new();

    // Create a cylinder: radius=2, height=5
    let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, 2.0, 5.0).unwrap();

    // Linear pattern: 5 copies spaced 10 mm along X
    let row = brepkit_operations::pattern::linear_pattern(
        &mut topo,
        cyl,
        Vec3::new(1.0, 0.0, 0.0),
        10.0, // spacing
        5,    // count
    )
    .unwrap();

    // A compound should contain 5 solids
    let count = topo.compound(row).unwrap().solids().len();
    assert_eq!(count, 5, "linear pattern should have 5 copies");

    // Circular pattern: 6 copies around Z axis
    let pin = brepkit_operations::primitives::make_cylinder(&mut topo, 1.0, 3.0).unwrap();
    // Move pin off-center so circular pattern makes a ring
    let offset = Mat4::translation(10.0, 0.0, 0.0);
    brepkit_operations::transform::transform_solid(&mut topo, pin, &offset).unwrap();

    let ring = brepkit_operations::pattern::circular_pattern(
        &mut topo,
        pin,
        Vec3::new(0.0, 0.0, 1.0),
        6, // count
    )
    .unwrap();

    let ring_count = topo.compound(ring).unwrap().solids().len();
    assert_eq!(ring_count, 6, "circular pattern should have 6 copies");
}

// ═══════════════════════════════════════════════════════════════════
// Example 7: Measurement (Volume, Center of Mass)
// ═══════════════════════════════════════════════════════════════════

/// Measure properties of a known solid (box 10×10×10 = volume 1000).
#[test]
fn example_measurement() {
    let mut topo = Topology::new();

    // Create a 10×10×10 box
    let solid = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    // Volume: 10 × 10 × 10 = 1000
    let volume = brepkit_operations::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        (volume - 1000.0).abs() < 1.0,
        "box volume should be ~1000, got {volume}"
    );

    // Center of mass: should be at (5, 5, 5)
    let com = brepkit_operations::measure::solid_center_of_mass(&topo, solid, 0.1).unwrap();
    assert!(
        (com.x() - 5.0).abs() < 0.1,
        "CoM x should be ~5, got {}",
        com.x()
    );
    assert!(
        (com.y() - 5.0).abs() < 0.1,
        "CoM y should be ~5, got {}",
        com.y()
    );
    assert!(
        (com.z() - 5.0).abs() < 0.1,
        "CoM z should be ~5, got {}",
        com.z()
    );
}

// ═══════════════════════════════════════════════════════════════════
// Example 8: Transform and Mirror
// ═══════════════════════════════════════════════════════════════════

/// Translate and mirror a solid.
#[test]
fn example_transform_and_mirror() {
    let mut topo = Topology::new();

    // Create a box at origin
    let solid = brepkit_operations::primitives::make_box(&mut topo, 10.0, 5.0, 3.0).unwrap();

    // Translate by (100, 0, 0)
    let translate = Mat4::translation(100.0, 0.0, 0.0);
    brepkit_operations::transform::transform_solid(&mut topo, solid, &translate).unwrap();

    // Center of mass should be shifted
    let com = brepkit_operations::measure::solid_center_of_mass(&topo, solid, 0.1).unwrap();
    assert!(
        (com.x() - 105.0).abs() < 0.5,
        "translated CoM x should be ~105, got {}",
        com.x()
    );

    // Mirror across the YZ plane (x=0)
    let mirrored = brepkit_operations::mirror::mirror(
        &mut topo,
        solid,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    )
    .unwrap();

    // Mirrored solid's CoM should be at x ≈ -105
    let mirror_com =
        brepkit_operations::measure::solid_center_of_mass(&topo, mirrored, 0.1).unwrap();
    assert!(
        (mirror_com.x() + 105.0).abs() < 0.5,
        "mirrored CoM x should be ~-105, got {}",
        mirror_com.x()
    );
}

// ═══════════════════════════════════════════════════════════════════
// Example 9: Multi-Solid Assembly
// ═══════════════════════════════════════════════════════════════════

/// Create a simple assembly of positioned parts.
#[test]
fn example_multi_solid_assembly() {
    let mut topo = Topology::new();

    // Base plate: wide, flat box
    let base = brepkit_operations::primitives::make_box(&mut topo, 100.0, 100.0, 5.0).unwrap();

    // Four corner posts (cylinders)
    let mut posts = Vec::new();
    for (dx, dy) in [(5.0, 5.0), (85.0, 5.0), (5.0, 85.0), (85.0, 85.0)] {
        let post = brepkit_operations::primitives::make_cylinder(&mut topo, 3.0, 30.0).unwrap();
        let t = Mat4::translation(dx, dy, 5.0);
        brepkit_operations::transform::transform_solid(&mut topo, post, &t).unwrap();
        posts.push(post);
    }

    // Verify all parts have positive volume
    let base_vol = brepkit_operations::measure::solid_volume(&topo, base, 0.1).unwrap();
    assert!(base_vol > 0.0, "base should have positive volume");
    for &post in &posts {
        let post_vol = brepkit_operations::measure::solid_volume(&topo, post, 0.1).unwrap();
        assert!(post_vol > 0.0, "post should have positive volume");
    }
}

// ═══════════════════════════════════════════════════════════════════
// Example 10: Custom Profile Extrusion (L-Shape)
// ═══════════════════════════════════════════════════════════════════

/// Create an L-shaped cross-section using boolean cut, then tessellate.
#[test]
fn example_custom_profile_extrusion() {
    use brepkit_operations::boolean::{BooleanOp, boolean};

    let mut topo = Topology::new();

    // L-shape: big box minus a corner box
    let outer = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 20.0).unwrap();
    let cutout = brepkit_operations::primitives::make_box(&mut topo, 5.0, 5.0, 20.0).unwrap();

    // Move cutout to the corner (5,5,0)
    let t = Mat4::translation(5.0, 5.0, 0.0);
    brepkit_operations::transform::transform_solid(&mut topo, cutout, &t).unwrap();

    // Boolean cut to create L-shape
    let l_shape = boolean(&mut topo, BooleanOp::Cut, outer, cutout).unwrap();

    // L-shape volume: 10×10×20 - 5×5×20 = 1500
    let vol = brepkit_operations::measure::solid_volume(&topo, l_shape, 0.1).unwrap();
    assert!(
        (vol - 1500.0).abs() < 10.0,
        "L-shape volume should be ~1500, got {vol}"
    );

    // Verify the solid has the expected topology (no tessellation needed).
    let s = topo.solid(l_shape).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    assert!(
        sh.faces().len() >= 6,
        "L-shape should have at least 6 faces, got {}",
        sh.faces().len()
    );
}

// ═══════════════════════════════════════════════════════════════════
// Example 8: Checkpoint and Restore
// ═══════════════════════════════════════════════════════════════════

/// Snapshot topology before a boolean, then restore after a failed attempt.
#[test]
fn example_checkpoint_restore() {
    let mut topo = Topology::new();

    // Create a box
    let box_id = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let box_vol = brepkit_operations::measure::solid_volume(&topo, box_id, 0.1).unwrap();
    assert!((box_vol - 1000.0).abs() < 1.0);

    // Take a checkpoint (clone the topology)
    let snapshot = topo.clone();
    let snapshot_vertex_count = snapshot.vertices().len();

    // Perform a boolean that adds many entities
    let cyl_id = brepkit_operations::primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();
    let transform = Mat4::translation(5.0, 5.0, -5.0);
    brepkit_operations::transform::transform_solid(&mut topo, cyl_id, &transform).unwrap();
    let _result = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Cut,
        box_id,
        cyl_id,
    )
    .unwrap();

    // Arena has grown significantly
    assert!(topo.vertices().len() > snapshot_vertex_count + 10);

    // Restore from snapshot — arena shrinks back
    topo = snapshot;
    assert_eq!(topo.vertices().len(), snapshot_vertex_count);

    // Original box is still valid and unchanged
    let restored_vol = brepkit_operations::measure::solid_volume(&topo, box_id, 0.1).unwrap();
    assert!((restored_vol - 1000.0).abs() < 1.0);
}
