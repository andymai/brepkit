//! Regression: the trimmer's split-location filter must bound the coarse
//! scan's minimum by a model-space length, not by the carrier's parametric
//! span.
//!
//! `rebuild_mapped_parametric_face` decides whether a boundary vertex lies on
//! a candidate edge's carrier by scanning 65 samples across the carrier's
//! parameter domain and comparing the smallest sample distance (`best_d`, a
//! model-space length) against a rejection bound. A line's carrier parameter
//! runs over `[0, 1]` whatever the line's model length, so bounding `best_d`
//! by the parametric span alone pins the bound at 1.0 model unit while the
//! coarse samples spread out as the edge grows: an on-carrier point can sit
//! up to half a sample step (`edge_length / 128`) from the nearest sample.
//! The bound must therefore be the larger of the parametric span and the
//! endpoint chord (a lower bound on the edge's true extent).
//!
//! The consequence is visible on the prism corner-fillet terminal split
//! (#1654's notch). On a 254 mm prism at r = 25.4 mm — the same relative
//! geometry as a 25.4 mm prism at r = 2.54 mm — the coarse samples are
//! 3.97 mm apart, so the terminal contact, which lands r along the cap edge,
//! sits 1.59 mm from the nearest sample: past the 1.0 bound, so the terminal
//! spokes were never split, the notch silently never fired, and each support
//! face kept its doubled-back runout tail (closed by edge id, but 8 planar
//! faces overlaying the caps and 32 open mesh edges). At 25.4 mm the samples
//! are 0.40 mm apart and the same point sits 0.16 mm from its nearest sample,
//! which is why the pre-existing fixtures never caught this.

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
