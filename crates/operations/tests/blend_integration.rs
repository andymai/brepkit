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
use brepkit_operations::validate::validate_solid;
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

/// The chamfer builds on the input's faces in place; the input box must
/// come out of it closed, at its own volume, with its own faces.
#[test]
fn chamfer_leaves_its_input_as_it_was() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let (faces, edges) = (
        solid_faces(&topo, solid).unwrap(),
        solid_edges(&topo, solid).unwrap(),
    );

    let result = chamfer_v2(&mut topo, solid, &edges[..1], 1.0, 1.0).unwrap();
    assert!(result.failed.is_empty());
    let chamfered = solid_volume(&topo, result.solid, 0.01).unwrap();
    assert!((chamfered - (BOX_VOLUME - 5.0)).abs() < 1e-6, "{chamfered}");

    let report = validate_solid(&topo, solid).unwrap();
    assert!(report.is_valid(), "input: {:?}", report.issues);
    assert_eq!(solid_faces(&topo, solid).unwrap(), faces);
    assert_eq!(solid_edges(&topo, solid).unwrap(), edges);
    let volume = solid_volume(&topo, solid, 0.01).unwrap();
    assert!((volume - BOX_VOLUME).abs() < 1e-6, "input volume {volume}");
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "result: {:?}", report.issues);
}

/// A 10 cube less a 3 cube cavity at `corner`, and its top edge along x
/// at y = 0.
fn hollow_block(
    topo: &mut Topology,
    corner: (f64, f64, f64),
) -> (
    brepkit_topology::solid::SolidId,
    brepkit_topology::edge::EdgeId,
) {
    use brepkit_math::mat::Mat4;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::transform::transform_solid;

    let block = make_box(topo, 10.0, 10.0, 10.0).unwrap();
    let cube = make_box(topo, 3.0, 3.0, 3.0).unwrap();
    transform_solid(topo, cube, &Mat4::translation(corner.0, corner.1, corner.2)).unwrap();
    let hollow = boolean(topo, BooleanOp::Cut, block, cube).unwrap();
    assert_eq!(topo.solid(hollow).unwrap().inner_shells().len(), 1);
    let edge = solid_edges(topo, hollow)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            [a, b]
                .iter()
                .all(|p| p.y().abs() < 1e-9 && (p.z() - 10.0).abs() < 1e-9)
        })
        .unwrap();
    (hollow, edge)
}

/// A block with an enclosed cubic cavity keeps the cavity when one of its
/// outer edges is filleted.
#[test]
fn fillet_keeps_a_cavity() {
    let radius = 1.0;
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, (3.5, 3.5, 3.5));
    let result = fillet_v2(&mut topo, hollow, &[edge], radius).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    assert_eq!(topo.solid(result.solid).unwrap().inner_shells().len(), 1);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let want = BOX_VOLUME - 27.0 - (1.0 - std::f64::consts::FRAC_PI_4) * radius * radius * 10.0;
    let volume = solid_volume(&topo, result.solid, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-4 * want,
        "volume {volume}, want {want}"
    );
}

/// The result's cavity is its own: moving the result in place leaves the
/// input's cavity where it was.
#[test]
fn a_filleted_blocks_cavity_is_its_own() {
    use brepkit_math::mat::Mat4;
    use brepkit_operations::transform::transform_solid;

    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, (3.5, 3.5, 3.5));
    let result = fillet_v2(&mut topo, hollow, &[edge], 1.0).unwrap();
    transform_solid(&mut topo, result.solid, &Mat4::translation(0.0, 0.0, 100.0)).unwrap();
    let cavity = topo.solid(hollow).unwrap().inner_shells()[0];
    for &face in topo.shell(cavity).unwrap().faces() {
        for v in brepkit_topology::explorer::face_vertices(&topo, face).unwrap() {
            let z = topo.vertex(v).unwrap().point().z();
            assert!(
                (3.5 - 1e-9..=6.5 + 1e-9).contains(&z),
                "input cavity moved to z {z}"
            );
        }
    }
}

/// A cavity 0.05 behind the front face and 0.5 below the top: a fillet of
/// radius 1 along their edge would cut into it, so it is refused and the
/// block left as it was.
#[test]
fn fillet_refuses_to_reach_a_cavity() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, (3.5, 0.05, 6.5));
    let volume = solid_volume(&topo, hollow, 0.001).unwrap();
    let result = fillet_v2(&mut topo, hollow, &[edge], 1.0).unwrap();
    assert_eq!(result.solid, hollow);
    assert!(result.succeeded.is_empty());
    assert_eq!(result.failed.len(), 1, "{:?}", result.failed);
    let report = validate_solid(&topo, hollow).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(topo.solid(hollow).unwrap().inner_shells().len(), 1);
    let after = solid_volume(&topo, hollow, 0.001).unwrap();
    assert!(
        (after - volume).abs() < 1e-9 * volume,
        "{after} vs {volume}"
    );
}

/// A rod of height 10 with a 3 cube cavity at its middle: a radius 1
/// fillet of its top rim takes material only within 1 of z = 10, so the
/// cavity stays, however wide the rod. The ring taken off, a square corner
/// less a quarter disc, sweeps round the axis at its centroid (Pappus).
#[test]
fn fillet_of_a_rim_well_clear_of_a_cavity_keeps_it() {
    use brepkit_math::mat::Mat4;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::transform::transform_solid;
    use std::f64::consts::{FRAC_PI_4, PI, TAU};

    // The wide rod's reach is what is tested; its fine volume is slow.
    for (radius, measure) in [(10.0, true), (1000.0, false)] {
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, radius, 10.0).unwrap();
        let cube = make_box(&mut topo, 3.0, 3.0, 3.0).unwrap();
        transform_solid(&mut topo, cube, &Mat4::translation(-1.5, -1.5, 3.5)).unwrap();
        let hollow = boolean(&mut topo, BooleanOp::Cut, rod, cube).unwrap();
        assert_eq!(topo.solid(hollow).unwrap().inner_shells().len(), 1);
        let rim = solid_edges(&topo, hollow)
            .unwrap()
            .into_iter()
            .find(|&e| {
                matches!(topo.edge(e).unwrap().curve(),
                    EdgeCurve::Circle(c) if (c.center().z() - 10.0).abs() < 1e-9)
            })
            .unwrap();
        let result = fillet_v2(&mut topo, hollow, &[rim], 1.0).unwrap();
        assert!(
            result.failed.is_empty(),
            "radius {radius}: {:?}",
            result.failed
        );
        assert_eq!(topo.solid(result.solid).unwrap().inner_shells().len(), 1);
        let report = validate_solid(&topo, result.solid).unwrap();
        assert!(report.is_valid(), "radius {radius}: {:?}", report.issues);
        if !measure {
            continue;
        }
        let centroid = (10.0 - 3.0 * PI) / (12.0 - 3.0 * PI);
        let want = (1.0 - FRAC_PI_4) * TAU * (radius - centroid);
        // The torus band's volume reads short at a coarse deflection, so
        // the removed ring is measured fine.
        let removed = solid_volume(&topo, hollow, 1e-4).unwrap()
            - solid_volume(&topo, result.solid, 1e-4).unwrap();
        assert!(
            (removed - want).abs() < 1e-2 * want,
            "radius {radius}: removed {removed}, want {want}"
        );
    }
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

/// Plane-cylinder fillet on a primitive cylinder. The bottom-cap circle edge
/// sits where the disc cap (a bare disc bounded by the cylinder) meets the
/// lateral; the analytic dispatcher should produce an exact toroidal fillet
/// that rounds the rim INWARD (gh #967): `major = r_c − r`, torus centre one
/// fillet radius into the material (z = +r). The result must be a watertight
/// solid that removes only the thin rim sliver.
#[test]
fn fillet_cylinder_base_circle_produces_torus() {
    let mut topo = Topology::new();
    // Cylinder of radius 2, height 4 — a rim circle is the spine.
    let solid = make_cylinder(&mut topo, 2.0, 4.0).unwrap();
    let base_vol = solid_volume(&topo, solid, 0.005).unwrap();

    // The two `EdgeCurve::Circle` edges on a primitive cylinder are the top
    // and bottom rims; pick the bottom rim (z = 0).
    let bottom_rim = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .find(|&eid| {
            let e = topo.edge(eid).unwrap();
            matches!(e.curve(), EdgeCurve::Circle(_))
                && topo.vertex(e.start()).unwrap().point().z().abs() < 1e-9
        })
        .expect("cylinder must have a bottom rim circle edge");

    let result = fillet_v2(&mut topo, solid, &[bottom_rim], 0.3).unwrap();

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

    let r_c = 2.0;
    let r_fillet = 0.3;
    // Torus geometry: minor radius == fillet radius; major radius == r_c − r
    // (inward rim); axis parallel to the cylinder axis.
    assert!(
        (torus.minor_radius() - r_fillet).abs() < 1e-9,
        "torus minor radius should equal fillet radius {r_fillet}, got {}",
        torus.minor_radius()
    );
    assert!(
        (torus.major_radius() - (r_c - r_fillet)).abs() < 1e-9,
        "torus major radius should equal r_c − r_fillet ({}), got {}",
        r_c - r_fillet,
        torus.major_radius()
    );

    // Stricter geometric assertion: the torus must actually be tangent to both
    // surfaces at the predicted contact points. The rim rounds inward, so the
    // plate contact is at radial r_c − r (z = 0, inside the rim) and the
    // cylinder contact is at radial r_c, z = +r (one fillet radius up into the
    // body). For the brepkit primitive, `Frame3::from_normal(z)` picks
    // `x_axis = (0, 1, 0)`, so contacts appear in the +y direction.
    let want_plate = brepkit_math::vec::Point3::new(0.0, r_c - r_fillet, 0.0);
    let want_cyl = brepkit_math::vec::Point3::new(0.0, r_c, r_fillet);
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

    // The assembled solid must be watertight and remove only the thin rim
    // sliver — a small fraction of the cylinder volume.
    let sd = topo.solid(result.solid).unwrap();
    let sh = topo.shell(sd.outer_shell()).unwrap();
    brepkit_topology::validation::validate_shell_closed(sh, &topo)
        .expect("rim fillet result must be watertight");
    brepkit_topology::validation::validate_shell_manifold(sh, &topo)
        .expect("rim fillet result must be manifold");
    let vol = solid_volume(&topo, result.solid, 0.005).unwrap();
    assert!(
        vol < base_vol && vol > base_vol * 0.99,
        "rim fillet should remove only a thin sliver: base={base_vol:.3}, filleted={vol:.3}"
    );
}

/// A frustum's bottom rim: `plane_cone_fillet` rolls this rim's ball below
/// the base, so its wall contact lands on the cone's extension past the
/// frustum's base, and the closed-rim rebuild refuses it rather than build a
/// band off the wall.
#[test]
fn fillet_cone_bottom_rim_with_its_contact_off_the_frustum_is_refused() {
    let mut topo = Topology::new();
    let solid = make_cone(&mut topo, 3.0, 1.0, 4.0).unwrap();
    let bottom_rim = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .find(|&eid| {
            matches!(topo.edge(eid).unwrap().curve(), EdgeCurve::Circle(c) if (c.radius() - 3.0).abs() < 1e-6)
        })
        .expect("frustum bottom rim should exist with radius 3");

    let error = fillet_v2(&mut topo, solid, &[bottom_rim], 0.3)
        .err()
        .expect("a wall contact off the frustum must be refused");
    assert!(
        format!("{error:?}").contains("wall contact would land past"),
        "{error:?}"
    );
}

/// The blend chamfer cannot trim a cap or a wall along a closed rim, so a
/// cylinder's or a frustum's bottom rim is recorded as failed and the solid
/// comes back as it was, closed. The cone each rim's stripe would take is
/// checked in the chamfer builder's own tests.
#[test]
fn chamfer_v2_records_a_closed_rim_as_failed() {
    let mut topo = Topology::new();
    let primitives = [
        ("cylinder", make_cylinder(&mut topo, 2.0, 4.0).unwrap()),
        ("frustum", make_cone(&mut topo, 3.0, 1.0, 4.0).unwrap()),
    ];
    for (name, solid) in primitives {
        let bottom_rim = solid_edges(&topo, solid)
            .unwrap()
            .into_iter()
            .find(|&eid| {
                let edge = topo.edge(eid).unwrap();
                matches!(edge.curve(), EdgeCurve::Circle(c) if c.center().z().abs() < 1e-9)
            })
            .unwrap();
        let before = solid_volume(&topo, solid, 0.01).unwrap();

        let result = chamfer_v2(&mut topo, solid, &[bottom_rim], 0.4, 0.4).unwrap();
        assert_eq!(result.solid, solid, "{name}");
        assert!(result.succeeded.is_empty(), "{name}");
        assert_eq!(result.failed.len(), 1, "{name}: {:?}", result.failed);
        assert_eq!(result.failed[0].0, bottom_rim, "{name}");
        let report = validate_solid(&topo, solid).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        let after = solid_volume(&topo, solid, 0.01).unwrap();
        assert!(
            (after - before).abs() < 1e-9 * before,
            "{name}: {after} vs {before}"
        );
    }
}
