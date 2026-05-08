//! End-to-end integration tests for the v2 fillet and chamfer engine.
//!
//! Each test creates fresh geometry from primitives and exercises the
//! walking-based blend pipeline (`blend_ops::fillet_v2`, `chamfer_v2`,
//! `chamfer_distance_angle`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::traits::ParametricSurface;
use brepkit_operations::blend_ops::{chamfer_distance_angle, chamfer_v2, fillet_v2};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;

const BOX_VOLUME: f64 = 1000.0; // 10 x 10 x 10

/// Create a 10x10x10 box and fillet a single edge.
#[test]
fn fillet_box_single_edge() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    let edges = solid_edges(&topo, solid).unwrap();
    assert!(!edges.is_empty(), "box must have edges");

    let result = fillet_v2(&mut topo, solid, &edges[..1], 1.0).unwrap();

    // Fillet adds faces (original 6 + at least 1 blend surface).
    let faces = solid_faces(&topo, result.solid).unwrap();
    assert!(
        faces.len() > 6,
        "filleted box should have more than 6 faces"
    );

    // At least the one edge should have succeeded.
    assert!(
        !result.succeeded.is_empty(),
        "at least one edge should succeed"
    );

    // Volume changes when edges are filleted.
    let vol = solid_volume(&topo, result.solid, 0.01).unwrap();
    assert!(
        (vol - BOX_VOLUME).abs() > 0.01,
        "filleted volume {vol} should differ from original {BOX_VOLUME}"
    );
}

/// Fillet 4 edges of a box (e.g. the first 4 found).
#[test]
fn fillet_box_multiple_edges() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    let edges = solid_edges(&topo, solid).unwrap();
    let n = edges.len().min(4);
    let target = &edges[..n];

    let result = fillet_v2(&mut topo, solid, target, 0.5).unwrap();

    assert!(
        !result.succeeded.is_empty(),
        "at least some edges should succeed"
    );

    let vol = solid_volume(&topo, result.solid, 0.01).unwrap();
    assert!(
        (vol - BOX_VOLUME).abs() > 0.01,
        "filleted volume {vol} should differ from original {BOX_VOLUME}"
    );
}

/// Symmetric chamfer on a single edge.
#[test]
fn chamfer_box_symmetric() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    let edges = solid_edges(&topo, solid).unwrap();
    let result = chamfer_v2(&mut topo, solid, &edges[..1], 1.0, 1.0).unwrap();

    let faces = solid_faces(&topo, result.solid).unwrap();
    assert!(
        faces.len() > 6,
        "chamfered box should have more than 6 faces"
    );

    let vol = solid_volume(&topo, result.solid, 0.01).unwrap();
    assert!(
        (vol - BOX_VOLUME).abs() > 0.01,
        "chamfered volume {vol} should differ from original {BOX_VOLUME}"
    );
}

/// Distance-angle chamfer on a single edge.
#[test]
fn chamfer_box_distance_angle() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    let edges = solid_edges(&topo, solid).unwrap();
    let result = chamfer_distance_angle(
        &mut topo,
        solid,
        &edges[..1],
        1.0,
        std::f64::consts::FRAC_PI_4,
    )
    .unwrap();

    assert!(
        !result.succeeded.is_empty(),
        "distance-angle chamfer should succeed on at least one edge"
    );
}

/// Zero radius should be rejected.
#[test]
fn fillet_zero_radius_error() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let edges = solid_edges(&topo, solid).unwrap();

    let err = fillet_v2(&mut topo, solid, &edges[..1], 0.0);
    assert!(err.is_err(), "zero radius should return an error");
}

/// Zero distance should be rejected.
#[test]
fn chamfer_zero_distance_error() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let edges = solid_edges(&topo, solid).unwrap();

    let err = chamfer_v2(&mut topo, solid, &edges[..1], 0.0, 1.0);
    assert!(err.is_err(), "zero distance should return an error");
}

/// Empty edge list should be rejected.
#[test]
fn fillet_empty_edges_error() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    let err = fillet_v2(&mut topo, solid, &[], 1.0);
    assert!(err.is_err(), "empty edges should return an error");
}

/// Plane-cylinder fillet on a primitive cylinder. The bottom-cap circle
/// edge sits where the plane (cap) meets the cylinder lateral; the analytic
/// dispatcher should produce an exact toroidal fillet face for the convex
/// "post on plate" geometry.
#[test]
fn fillet_cylinder_base_circle_produces_torus() {
    let mut topo = Topology::new();
    // Cylinder of radius 2, height 4 — convex base circle is the spine.
    let solid = make_cylinder(&mut topo, 2.0, 4.0).unwrap();

    // The two `EdgeCurve::Circle` edges on a primitive cylinder are the top
    // and bottom rims; pick whichever the explorer surfaces first.
    let circle_edges: Vec<_> = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&eid| matches!(topo.edge(eid).unwrap().curve(), EdgeCurve::Circle(_)))
        .collect();
    assert!(
        !circle_edges.is_empty(),
        "cylinder must have at least one circular rim edge"
    );

    let result = fillet_v2(&mut topo, solid, &circle_edges[..1], 0.3).unwrap();

    assert!(
        !result.succeeded.is_empty(),
        "cylinder rim fillet must produce at least one stripe; failed = {:?}",
        result.failed
    );

    // The new blend face should be exactly a Torus when the analytic path
    // fired (vs a NURBS approximation when the walker fallback was used).
    let new_faces: Vec<_> = solid_faces(&topo, result.solid).unwrap();
    let torus = new_faces.iter().find_map(|&fid| {
        if let FaceSurface::Torus(t) = topo.face(fid).unwrap().surface() {
            Some(t.clone())
        } else {
            None
        }
    });
    let torus = torus.expect("analytic fast path should produce a Torus face");

    // Torus geometry: minor radius == fillet radius, major radius ==
    // r_cylinder + r_fillet (convex case), axis parallel to cylinder axis.
    assert!(
        (torus.minor_radius() - 0.3).abs() < 1e-9,
        "torus minor radius should equal fillet radius 0.3, got {}",
        torus.minor_radius()
    );
    assert!(
        (torus.major_radius() - 2.3).abs() < 1e-9,
        "torus major radius should equal cylinder radius + fillet radius (2.3), got {}",
        torus.major_radius()
    );

    // Stricter geometric assertion: the torus must actually be tangent to
    // both surfaces at the predicted contact points. Sample the small-circle
    // u-frame densely and check that the predicted plate-contact and
    // cylinder-contact 3D positions are reproduced. Without this we'd accept
    // any torus with the right radii regardless of placement.
    let r_c = 2.0;
    let r_fillet = 0.3;
    // Plate contact = top-of-tube (z toward +axis_dir): radial = r_c + r at z=0.
    // Cylinder contact = inner-equator: radial = r_c at z = -r (one fillet
    // radius below the plate, since the rolling ball center sits at z=-r).
    // Both contacts lie in the plane spanned by the cylinder's local x_axis
    // direction; for the brepkit primitive `Frame3::from_normal(z)` picks
    // `x_axis = (0, 1, 0)`, so contact points appear in the +y direction.
    let want_plate = brepkit_math::vec::Point3::new(0.0, r_c + r_fillet, 0.0);
    let want_cyl = brepkit_math::vec::Point3::new(0.0, r_c, -r_fillet);
    let mut closest_plate = f64::INFINITY;
    let mut closest_cyl = f64::INFINITY;
    for i in 0..1440 {
        let v = (f64::from(i) / 1440.0) * std::f64::consts::TAU;
        let p = ParametricSurface::evaluate(&torus, 0.0, v);
        closest_plate = closest_plate.min((p - want_plate).length());
        closest_cyl = closest_cyl.min((p - want_cyl).length());
    }
    assert!(
        closest_plate < 1e-6,
        "torus should touch plate at {want_plate:?}; closest sample was {closest_plate:.6}"
    );
    assert!(
        closest_cyl < 1e-6,
        "torus should touch cylinder at {want_cyl:?}; closest sample was {closest_cyl:.6}"
    );
}

/// Plane-cone fillet on a frustum primitive's bottom rim. Verifies the
/// analytic dispatcher produces an exact toroidal blend whose major radius
/// is `r_p + r·cot(α/2)` (collapsing to `r_p + r` at α = π/2, which matches
/// the plane-cylinder limit), and that both contacts are tangent to the
/// expected surfaces.
#[test]
fn fillet_cone_bottom_rim_produces_torus() {
    let mut topo = Topology::new();
    // Regular frustum: bottom_radius=3 > top_radius=1 with height 4 ⇒ make_cone
    // builds a virtual apex above the frustum and axis pointing -z. The
    // bottom rim corner is the convex spine for the fillet.
    let solid = make_cone(&mut topo, 3.0, 1.0, 4.0).unwrap();
    let fillet_r = 0.3;

    // Find the bottom rim (Circle edge at z=0, radius=3).
    let bottom_rim = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .find(|&eid| {
            if let EdgeCurve::Circle(c) = topo.edge(eid).unwrap().curve() {
                (c.radius() - 3.0).abs() < 1e-6
            } else {
                false
            }
        })
        .expect("frustum bottom rim should exist with radius 3");

    let result = fillet_v2(&mut topo, solid, &[bottom_rim], fillet_r).unwrap();
    assert!(
        !result.succeeded.is_empty(),
        "cone bottom-rim fillet must produce at least one stripe; failed = {:?}",
        result.failed
    );

    let new_faces = solid_faces(&topo, result.solid).unwrap();
    let torus = new_faces
        .iter()
        .find_map(|&fid| {
            if let FaceSurface::Torus(t) = topo.face(fid).unwrap().surface() {
                Some(t.clone())
            } else {
                None
            }
        })
        .expect("plane-cone fillet should produce a Torus face");

    // Half-angle: make_cone(3, 1, 4) builds the virtual apex 2 units above
    // the top, so the apex is at z=6, the frustum tip-to-base axial extent
    // is 6, and α = atan2(6, 3).
    let alpha = 6.0_f64.atan2(3.0);
    let r_p = 3.0;
    let expected_major = r_p + fillet_r * (alpha * 0.5).tan().recip(); // r_p + r·cot(α/2)

    assert!(
        (torus.minor_radius() - fillet_r).abs() < 1e-9,
        "torus minor should equal fillet radius {fillet_r}, got {}",
        torus.minor_radius()
    );
    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-6,
        "torus major should be r_p + r·cot(α/2) = {expected_major:.6}, got {}",
        torus.major_radius()
    );

    // Strict geometric check — the torus must touch the plate (z=0) and the
    // analytical cone surface at the predicted contact points.
    // Frame derivation for the cone primitive: ConicalSurface::new with
    // axis=(0,0,-1), Frame3::from_normal picks x_axis = axis × seed where
    // seed = (1,0,0) since axis.x() = 0; that gives x = (-z) × (1,0,0) =
    // (0, -1, 0). So the radial=cos(0)·x + sin(0)·y direction at u=0 is
    // (0, -1, 0).
    let want_plate = brepkit_math::vec::Point3::new(0.0, -expected_major, 0.0);
    // Cone contact is at axial = -r·(1 + cos(α)) (below the plate, where
    // the analytical cone surface — extended past the frustum's base —
    // intersects the rolling-ball trajectory).
    let cone_contact_axial = -fillet_r * (1.0 + alpha.cos());
    let cone_contact_radial = expected_major - fillet_r * alpha.sin();
    let want_cone = brepkit_math::vec::Point3::new(0.0, -cone_contact_radial, cone_contact_axial);
    // Evaluate at the exact predicted v parameters rather than sampling — the
    // contact happens at v values that aren't necessarily on a 1440-sample
    // grid, and discretization noise would otherwise mask geometry errors.
    // For axis_dir = -n_p_inward = -z (the choice the impl makes), plate
    // contact is at v = 3π/2 (cos=0, sin=-1) and cone contact is at the v
    // satisfying cos(v) = -sin(α), sin(v) = cos(α) — i.e. v = atan2(cos(α), -sin(α)).
    let v_plate = 3.0 * std::f64::consts::FRAC_PI_2;
    let p_plate = ParametricSurface::evaluate(&torus, 0.0, v_plate);
    let v_cone = alpha.cos().atan2(-alpha.sin());
    let p_cone = ParametricSurface::evaluate(&torus, 0.0, v_cone);
    assert!(
        (p_plate - want_plate).length() < 1e-9,
        "torus at v=3π/2 should touch plate at {want_plate:?}; got {p_plate:?}"
    );
    assert!(
        (p_cone - want_cone).length() < 1e-9,
        "torus at v=atan2(cos α, -sin α) should touch cone at {want_cone:?}; got {p_cone:?}"
    );
}
