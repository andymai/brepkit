//! Shared test assertion helpers for geometric and topological validation.
//!
//! These helpers provide descriptive, tolerance-aware assertions for
//! volumes, areas, positions, and topological invariants. They are
//! designed to be used across all test modules in `brepkit-operations`.

#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::explorer;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

// ── Volume / area assertions ──────────────────────────────────────

/// Assert that a solid's volume is within `rel_tol` of `expected`.
///
/// When `expected` is near zero (< 1e-15), `rel_tol` is treated as an
/// absolute tolerance instead (i.e., asserts `|volume| < rel_tol`).
///
/// # Panics
///
/// Panics if the relative error exceeds `rel_tol` or if volume
/// computation fails.
pub fn assert_volume_near(topo: &Topology, solid: SolidId, expected: f64, rel_tol: f64) {
    let vol = crate::measure::solid_volume(topo, solid, 0.05).unwrap();
    let rel_error = if expected.abs() < 1e-15 {
        vol.abs()
    } else {
        (vol - expected).abs() / expected.abs()
    };
    assert!(
        rel_error < rel_tol,
        "volume mismatch: got {vol:.6}, expected {expected:.6} \
         (error: {:.4}%, tolerance: {:.4}%)",
        rel_error * 100.0,
        rel_tol * 100.0,
    );
}

/// Assert that a face's area is within `rel_tol` of `expected`.
///
/// # Panics
///
/// Panics if the relative error exceeds `rel_tol` or if area
/// computation fails.
pub fn assert_area_near(topo: &Topology, face: FaceId, expected: f64, rel_tol: f64) {
    let area = crate::measure::face_area(topo, face, 0.1).unwrap();
    let rel_error = if expected.abs() < 1e-15 {
        area.abs()
    } else {
        (area - expected).abs() / expected.abs()
    };
    assert!(
        rel_error < rel_tol,
        "area mismatch: got {area:.6}, expected {expected:.6} \
         (error: {:.4}%, tolerance: {:.4}%)",
        rel_error * 100.0,
        rel_tol * 100.0,
    );
}

// ── Point assertions ──────────────────────────────────────────────

/// Assert that two points are within `abs_tol` of each other.
///
/// # Panics
///
/// Panics if the Euclidean distance exceeds `abs_tol`.
pub fn assert_point_near(actual: Point3, expected: Point3, abs_tol: f64) {
    let dx = actual.x() - expected.x();
    let dy = actual.y() - expected.y();
    let dz = actual.z() - expected.z();
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    assert!(
        dist < abs_tol,
        "point mismatch: got ({:.6}, {:.6}, {:.6}), \
         expected ({:.6}, {:.6}, {:.6}), distance={dist:.2e}",
        actual.x(),
        actual.y(),
        actual.z(),
        expected.x(),
        expected.y(),
        expected.z(),
    );
}

// ── Topological invariant assertions ──────────────────────────────

/// Compute the Euler characteristic V - E + F for a solid.
///
/// For a closed orientable surface of genus g:
/// - genus 0 (sphere-like): χ = 2
/// - genus 1 (torus-like): χ = 0
/// - genus 2 (double torus): χ = -2
///
/// # Panics
///
/// Panics if topology lookups fail.
#[allow(clippy::cast_possible_wrap)]
pub fn euler_characteristic(topo: &Topology, solid: SolidId) -> i64 {
    let (f, e, v) = explorer::solid_entity_counts(topo, solid).unwrap();
    (v as i64) - (e as i64) + (f as i64)
}

/// Assert that a solid has Euler characteristic 2 (genus-0, sphere-like).
///
/// This is the expected value for any simply-connected closed solid
/// (box, cylinder, cone, sphere, any convex solid, boolean result
/// of convex inputs without holes).
///
/// # Panics
///
/// Panics if the Euler characteristic is not 2.
pub fn assert_euler_genus0(topo: &Topology, solid: SolidId) {
    let chi = euler_characteristic(topo, solid);
    assert_eq!(
        chi, 2,
        "expected Euler characteristic V-E+F = 2 (genus-0), got {chi}"
    );
}

/// Assert that a solid's shell is manifold (every edge shared by exactly 2 faces).
///
/// # Panics
///
/// Panics if the solid is not manifold or if topology lookups fail.
pub fn assert_manifold(topo: &Topology, solid: SolidId) {
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    brepkit_topology::validation::validate_shell_manifold(sh, topo.faces(), topo.wires())
        .expect("solid should be manifold");
}

// ── Boolean conservation ──────────────────────────────────────────

/// Assert the inclusion-exclusion principle: V(A) + V(B) = V(A∪B) + V(A∩B).
///
/// This is a fundamental conservation law for boolean operations.
///
/// # Panics
///
/// Panics if the identity is violated beyond `rel_tol`.
pub fn assert_volume_conservation(
    vol_a: f64,
    vol_b: f64,
    vol_fused: f64,
    vol_intersected: f64,
    rel_tol: f64,
) {
    let lhs = vol_a + vol_b;
    let rhs = vol_fused + vol_intersected;
    let rel_error = if lhs.abs() < 1e-15 {
        rhs.abs()
    } else {
        (lhs - rhs).abs() / lhs.abs()
    };
    assert!(
        rel_error < rel_tol,
        "volume conservation violated: V(A)+V(B) = {lhs:.6}, \
         V(A∪B)+V(A∩B) = {rhs:.6} (error: {:.4}%, tolerance: {:.4}%)\n\
         V(A)={vol_a:.6}, V(B)={vol_b:.6}, V(A∪B)={vol_fused:.6}, V(A∩B)={vol_intersected:.6}",
        rel_error * 100.0,
        rel_tol * 100.0,
    );
}

// ── CW profile assertions ──────────────────────────────────────

/// Assert that a CW-wound profile produces a solid with the expected volume.
///
/// Creates a CW unit square face, passes it to `build_solid`, and asserts
/// the resulting volume is within `rel_tol` of `expected_vol`.
///
/// # Panics
///
/// Panics if the volume deviates beyond `rel_tol` or if any operation fails.
pub fn assert_cw_profile_produces_valid_solid<F>(build_solid: F, expected_vol: f64, rel_tol: f64)
where
    F: Fn(&mut Topology, FaceId) -> SolidId,
{
    let mut topo = Topology::new();
    let face = brepkit_topology::test_utils::make_cw_unit_square_face(&mut topo);
    let solid = build_solid(&mut topo, face);
    assert_volume_near(&topo, solid, expected_vol, rel_tol);
}
