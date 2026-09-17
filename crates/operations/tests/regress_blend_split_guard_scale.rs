//! Regression: the trimmer's split-location filter must bound the coarse
//! scan's minimum by a model-space length, not by the carrier's parametric
//! span.
//!
//! A line's carrier parameter runs over `[0, 1]` whatever its model length,
//! so the old bound was pinned at 1.0 model unit while the coarse samples
//! spread out with the edge. On a 254 mm prism filleted at r = 25.4 mm the
//! terminal contact sat 1.59 mm from its nearest sample and was rejected: the
//! terminal spokes were never split, the caps were never notched, and the
//! result passed every id-level check but tessellated with 32 open edges. The
//! same relative geometry at 25.4 mm sits 0.16 mm from its nearest sample and
//! always passed, which is why the existing fixtures never caught this.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid_with_tolerance};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

/// Samples the trimmer's coarse scan takes across the carrier (`0..=64`).
const COARSE_STEPS: f64 = 64.0;

/// The prism's four vertical corners.
fn vertical_edges(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&id| {
            let edge = topo.edge(id).unwrap();
            let start = topo.vertex(edge.start()).unwrap().point();
            let end = topo.vertex(edge.end()).unwrap().point();
            (start.x() - end.x()).abs() < 1e-9
                && (start.y() - end.y()).abs() < 1e-9
                && (start.z() - end.z()).abs() > 1e-9
        })
        .collect()
}

/// Distance from the terminal contact's position on the cap edge — `radius`
/// along the edge from the corner — to the nearest coarse sample of that
/// edge's carrier. This is the quantity the trimmer's filter compared against
/// its bound before refinement; it scales with the edge's model length while
/// the (line-carrier) parametric span stays at 1.0.
fn distance_to_nearest_sample(size: f64, radius: f64) -> f64 {
    let spacing = size / COARSE_STEPS;
    let steps = radius / spacing;
    (steps - steps.round()).abs() * spacing
}

/// Fillet the prism's four vertical corners at `size / 10` and require the
/// result to be watertight. Returns the fixture's distance-to-nearest-sample
/// so the caller can assert which side of the old bound it sits on.
fn assert_prism_corners_are_watertight(size: f64) -> f64 {
    let radius = size / 10.0;
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, size, size, size).unwrap();
    let edges = vertical_edges(&topo, solid);
    assert_eq!(edges.len(), 4, "size={size}: prism vertical corners");

    let result = fillet_v2(&mut topo, solid, &edges, radius)
        .unwrap_or_else(|error| panic!("size={size}, r={radius}: fillet errored: {error}"));
    assert_eq!(
        result.succeeded.len(),
        4,
        "size={size}, r={radius}: blended corners"
    );
    let failures: Vec<String> = result
        .failed
        .iter()
        .map(|(edge, reason)| format!("{edge:?}: {reason}"))
        .collect();
    assert!(
        failures.is_empty(),
        "size={size}, r={radius}: failed edges: {failures:?}"
    );

    let validation = validate_solid(&topo, result.solid).unwrap();
    assert!(
        validation.is_valid(),
        "size={size}, r={radius}: {:?}",
        validation.issues
    );

    let mesh =
        tessellate_solid_with_tolerance(&topo, result.solid, 0.01, 5.0_f64.to_radians()).unwrap();
    assert_eq!(
        boundary_edge_count(&mesh),
        0,
        "size={size}, r={radius}: corner fillets left open mesh edges — the \
         terminal spokes were not split, so the caps were never notched"
    );

    distance_to_nearest_sample(size, radius)
}

/// Control: the one-inch prism at the same relative geometry as the 254 mm
/// regression below. Its coarse samples are 0.40 mm apart, so the terminal
/// contact sits inside the 1.0 model-unit bound the unfixed filter used and
/// the corners come out watertight either way — the fixture must keep passing
/// after the fix, and it passing before the fix is precisely why the defect
/// is scale-dependent.
#[test]
fn prism_corner_fillets_are_watertight_on_a_one_inch_prism() {
    let sample_distance = assert_prism_corners_are_watertight(25.4);
    assert!(
        sample_distance < 1.0,
        "control fixture is no longer inside the unfixed filter's bound \
         ({sample_distance} mm from the nearest coarse sample)"
    );
}

/// Regression: the same relative geometry on a 254 mm prism, where the coarse
/// scan samples the carriers every 3.97 mm and the terminal contact sits
/// 1.59 mm from the nearest sample — outside the 1.0 bound. Before the fix no
/// terminal spoke was split, the caps were never notched, and the support
/// faces kept their doubled-back runout tails: the result passed every
/// id-level check but tessellated with 32 open edges.
#[test]
fn prism_corner_fillets_are_watertight_on_a_254_mm_prism() {
    let sample_distance = assert_prism_corners_are_watertight(254.0);
    assert!(
        sample_distance > 1.0,
        "regression fixture is no longer outside the unfixed filter's bound \
         ({sample_distance} mm from the nearest coarse sample)"
    );
}
