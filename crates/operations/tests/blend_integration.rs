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

/// Plane-cylinder chamfer on a primitive cylinder's bottom rim. The
/// analytic dispatcher should produce an exact conical chamfer face
/// passing through both contact circles at the predicted positions.
#[test]
fn chamfer_cylinder_base_circle_produces_cone() {
    let mut topo = Topology::new();
    let r_c = 2.0;
    let height = 4.0;
    let solid = make_cylinder(&mut topo, r_c, height).unwrap();
    let d = 0.4;

    let bottom_rim = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .find(|&eid| {
            let edge = topo.edge(eid).unwrap();
            // Pick the Circle edge whose vertex is at z=0 (the bottom rim).
            matches!(edge.curve(), EdgeCurve::Circle(_))
                && topo.vertex(edge.start()).unwrap().point().z().abs() < 1e-9
        })
        .expect("cylinder bottom rim must exist");

    let result = chamfer_v2(&mut topo, solid, &[bottom_rim], d, d).unwrap();
    assert!(
        !result.succeeded.is_empty(),
        "cylinder rim chamfer must produce at least one stripe; failed = {:?}",
        result.failed
    );

    // The new blend face should be a Cone (not a NURBS approximation) when
    // the analytic chamfer fast path fired.
    let new_faces = solid_faces(&topo, result.solid).unwrap();
    let cone = new_faces
        .iter()
        .find_map(|&fid| {
            if let FaceSurface::Cone(c) = topo.face(fid).unwrap().surface() {
                Some(c.clone())
            } else {
                None
            }
        })
        .expect("plane-cylinder chamfer should produce a Cone face via the analytic fast path");

    // Symmetric d1 = d2 = 0.4 ⇒ half-angle = atan2(0.4, 0.4) = π/4 (45°).
    assert!(
        (cone.half_angle() - std::f64::consts::FRAC_PI_4).abs() < 1e-12,
        "cone half-angle should be π/4 for symmetric chamfer; got {}",
        cone.half_angle()
    );

    // Verify both contacts are points on the cone:
    //   - plate contact at radial r_c - d, on the plate (z = 0)
    //   - cylinder contact at radial r_c, axially d into the material
    //     (z = +d for cylinder primitive with material at z ≥ 0).
    // Frame3::from_normal(z=+1) gives x_axis = (0, 1, 0), so the contacts
    // appear in the +y direction at u = 0.
    let want_plate = brepkit_math::vec::Point3::new(0.0, r_c - d, 0.0);
    let want_cyl = brepkit_math::vec::Point3::new(0.0, r_c, d);
    // For each predicted contact, find the cone (u, v) and verify the cone
    // evaluates to the same point. We solve analytically:
    //   - At cone u=0 the radial axis is +y (matches Frame3 derivation),
    //     so position(0, v) = apex + v · (cos(α)·y + sin(α)·cone_axis).
    //   - cone_axis = -axial_into_material = -z (away from cylinder material).
    //   - apex axial = +d2 · (r_c - d1) / d1 (above the plate, into material).
    //     With d1 = d2 = d: apex_axial = r_c - d.
    let alpha = std::f64::consts::FRAC_PI_4;
    // (apex_axial = (r_c - d) · d / d = r_c - d for symmetric chamfer)
    // For plate contact at radial = r_c - d, z = 0:
    //   v_plate · cos(α) = r_c - d (radial component)
    //   v_plate · sin(α) = -apex_axial = -(r_c - d) (axial drop from apex to plate)
    //   ⇒ v_plate = (r_c - d) / cos(α). For α=π/4: v_plate = (r_c - d) · √2.
    let v_plate = (r_c - d) / alpha.cos();
    let p_plate = ParametricSurface::evaluate(&cone, 0.0, v_plate);
    assert!(
        (p_plate - want_plate).length() < 1e-9,
        "cone at v={v_plate:.6} should touch plate at {want_plate:?}; got {p_plate:?}"
    );
    // Cylinder contact at radial = r_c, z = +d (above the plate, on the
    // cylinder material side). The cone here has axis +z (opening upward
    // from apex below the plate), so cone z = apex_z + v·sin(α) and we want
    // that to equal +d.
    let v_cyl = r_c / alpha.cos();
    let p_cyl = ParametricSurface::evaluate(&cone, 0.0, v_cyl);
    assert!(
        (p_cyl - want_cyl).length() < 1e-9,
        "cone at v={v_cyl:.6} should touch cylinder at {want_cyl:?}; got {p_cyl:?}"
    );

    // Also guard against the bug where the chamfer's separately-built
    // contact circle gets placed on the wrong side of the plate. Walk the
    // chamfer cone's outer wire and confirm no `Circle3D` edge is
    // positioned at `z = -d` (the misplaced location); any preserved
    // contact circle on the cone must sit at `z = +d` instead.
    let cone_face = new_faces
        .iter()
        .copied()
        .find(|&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cone(_)))
        .unwrap();
    let cone_wire = topo.face(cone_face).unwrap().outer_wire();
    let has_misplaced = topo.wire(cone_wire).unwrap().edges().iter().any(|oe| {
        if let EdgeCurve::Circle(c) = topo.edge(oe.edge()).unwrap().curve() {
            (c.center().z() + d).abs() < 1e-6 && (c.radius() - r_c).abs() < 1e-6
        } else {
            false
        }
    });
    assert!(
        !has_misplaced,
        "no contact circle should sit at z = -d (cylinder contact misplaced)"
    );
}

/// Plane-cone chamfer on a frustum primitive's bottom rim. Verifies the
/// analytic dispatcher produces an exact conical chamfer face whose
/// half-angle matches `atan2(d1 - d2·cos α, d2·sin α)` and whose contacts
/// land at the predicted positions on both the plate and the cone.
#[test]
fn chamfer_cone_bottom_rim_produces_cone() {
    let mut topo = Topology::new();
    // Regular frustum: bottom_radius=3 > top_radius=1, height=4.
    // Half-angle α = atan2(6, 3) (the cone primitive uses the virtual-apex
    // height; see `make_cone` for the derivation).
    let solid = make_cone(&mut topo, 3.0, 1.0, 4.0).unwrap();
    let d = 0.4;

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

    let result = chamfer_v2(&mut topo, solid, &[bottom_rim], d, d).unwrap();
    assert!(
        !result.succeeded.is_empty(),
        "cone bottom-rim chamfer must produce at least one stripe; failed = {:?}",
        result.failed
    );

    // The new face should be a Cone (the chamfer surface — distinct from
    // the original frustum lateral cone, which is also a Cone).
    let new_faces = solid_faces(&topo, result.solid).unwrap();
    let alpha = 6.0_f64.atan2(3.0);
    let r_p = 3.0;
    // For symmetric d1 = d2 = d, brepkit's ConicalSurface measures the
    // half-angle from the AXIS to the generator, giving β = π/2 − α/2.
    let expected_half_angle = std::f64::consts::FRAC_PI_2 - alpha * 0.5;

    let chamfer_cone = new_faces
        .iter()
        .find_map(|&fid| {
            if let FaceSurface::Cone(c) = topo.face(fid).unwrap().surface() {
                // Distinguish the chamfer cone from the frustum's original
                // lateral cone by half-angle. The frustum cone has α; the
                // chamfer cone has β = π/2 − α/2.
                if (c.half_angle() - expected_half_angle).abs() < 1e-6 {
                    return Some(c.clone());
                }
            }
            None
        })
        .expect("chamfer cone face with half-angle = π/2 - α/2 should exist");

    assert!(
        (chamfer_cone.half_angle() - expected_half_angle).abs() < 1e-9,
        "chamfer cone half-angle should be π/2 - α/2 = {expected_half_angle:.6}, got {}",
        chamfer_cone.half_angle()
    );

    // Frustum cone primitive's frame: axis = (0, 0, -1), so
    // `Frame3::from_normal` gives x_axis = (0, -1, 0). The bottom rim
    // "u = 0" point is therefore in the −y direction. The chamfer cone
    // built with the same x_axis as ref_dir inherits that convention.
    let want_plate = brepkit_math::vec::Point3::new(0.0, -(r_p - d), 0.0);
    let cone_contact_axial = d * alpha.sin();
    let cone_contact_radial = r_p - d * alpha.cos();
    let want_cone = brepkit_math::vec::Point3::new(0.0, -cone_contact_radial, cone_contact_axial);

    // Contact v parameters on the chamfer cone, given:
    //   chamfer_axis = +z (toward cylinder material from apex below);
    //   x_axis = (0, -1, 0) (so radial at u=0 points in -y).
    // Position(0, v) = apex + v · (cos(β)·(0, -1, 0) + sin(β)·(0, 0, 1))
    //                = (0, -v·cos(β), apex.z + v·sin(β))
    // Plate contact: -v·cos(β) = -(r_p - d) → v = (r_p - d)/cos β.
    // Cone-side contact: -v·cos(β) = -(r_p - d·cos α) → v = (r_p - d·cos α)/cos β.
    let v_plate = (r_p - d) / expected_half_angle.cos();
    let p_plate = ParametricSurface::evaluate(&chamfer_cone, 0.0, v_plate);
    assert!(
        (p_plate - want_plate).length() < 1e-9,
        "chamfer cone at v={v_plate:.6} should touch plate at {want_plate:?}; got {p_plate:?}"
    );
    let v_cone = (r_p - d * alpha.cos()) / expected_half_angle.cos();
    let p_cone = ParametricSurface::evaluate(&chamfer_cone, 0.0, v_cone);
    assert!(
        (p_cone - want_cone).length() < 1e-9,
        "chamfer cone at v={v_cone:.6} should touch cone at {want_cone:?}; got {p_cone:?}"
    );
}
