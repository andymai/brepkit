//! OttoCAD N340: exact native reproduction of the captured third mixed-radius fillet.
//!
//! Source provenance: `user-third-unequal-source.ottocad`, SHA-256
//! `f4403a2d032a9e99f15b996be1dc8467d795e7ea9d818c0e734d7ddda6c6c206`,
//! accepted head `7a5db668-1d13-4455-b551-139862bd0370`. The accepted graph creates
//! a 40 x 25 mm rectangular sketch, extrudes it 30 mm, then accepts the recorded
//! bottom-X and current vertical targets at 0.1 inch (2.54 mm) each. The refused
//! current edge49 is `(0,5.08,0)-(0,25,0)` and requests 0.05 inch (1.27 mm).

#![allow(
    clippy::expect_used,
    clippy::print_stderr,
    clippy::unwrap_used,
    deprecated
)]

use brepkit_check::validate::{Severity as CheckSeverity, ValidateOptions};
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::extrude::extrude;
use brepkit_operations::fillet::fillet_rolling_ball;
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces, solid_vertices};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;

const ACCEPTED_RADIUS_MM: f64 = 2.54;
const THIRD_RADIUS_MM: f64 = 1.27;

fn ordered_source(topo: &mut Topology, width: f64, depth: f64, height: f64) -> SolidId {
    let wire = make_polygon_wire(
        topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(width, 0.0, 0.0),
            Point3::new(width, depth, 0.0),
            Point3::new(0.0, depth, 0.0),
        ],
        Tolerance::new().linear,
    )
    .unwrap();
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), height).unwrap()
}

fn captured_ordered_source(topo: &mut Topology) -> SolidId {
    ordered_source(topo, 40.0, 25.0, 30.0)
}

fn edge_at(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let tolerance = Tolerance::new().linear;
    let matches: Vec<_> = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&id| {
            let edge = topo.edge(id).unwrap();
            let start = topo.vertex(edge.start()).unwrap().point();
            let end = topo.vertex(edge.end()).unwrap().point();
            ((start - a).length() < tolerance && (end - b).length() < tolerance)
                || ((start - b).length() < tolerance && (end - a).length() < tolerance)
        })
        .collect();
    assert_eq!(matches.len(), 1, "unique current target {a:?}-{b:?}");
    matches[0]
}

fn assert_valid_stage(topo: &Topology, solid: SolidId, expected_volume: f64) {
    let validation = validate_solid(topo, solid).unwrap();
    let volume = solid_volume(topo, solid, 0.01).unwrap();
    eprintln!(
        "N340 stage {solid:?}: V/E/F={}/{}/{} boundary={} volume={volume:.15} issues={:?}",
        solid_vertices(topo, solid).unwrap().len(),
        solid_edges(topo, solid).unwrap().len(),
        solid_faces(topo, solid).unwrap().len(),
        topo.build_adjacency(solid).unwrap().boundary_edges().len(),
        validation.issues
    );
    assert!(
        validation.is_valid(),
        "invalid stage: {:?}",
        validation.issues
    );
    assert!(volume.is_finite() && volume > 0.0);
    assert!((volume - expected_volume).abs() < 1.0e-9, "volume={volume}");
}

fn snapshot(topo: &Topology, solid: SolidId) -> Vec<String> {
    let mut state = Vec::new();
    for face_id in solid_faces(topo, solid).unwrap() {
        let face = topo.face(face_id).unwrap();
        state.push(format!("{face_id:?} {face:?}"));
        for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
        {
            let wire = topo.wire(wire_id).unwrap();
            state.push(format!("{wire_id:?} {wire:?}"));
            for oriented in wire.edges() {
                let edge_id = oriented.edge();
                let edge = topo.edge(edge_id).unwrap();
                state.push(format!(
                    "{edge_id:?} {edge:?} {:?} {:?}",
                    topo.vertex(edge.start()).unwrap(),
                    topo.vertex(edge.end()).unwrap()
                ));
            }
        }
    }
    state
}

fn effective_normal(
    topo: &Topology,
    face_id: brepkit_topology::face::FaceId,
    point: Point3,
) -> Vec3 {
    let face = topo.face(face_id).unwrap();
    let normal = match face.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        surface => {
            let projected = surface.project_point(point);
            assert!(
                projected.is_some(),
                "seam sample {point:?} does not project to face {face_id:?}: {surface:?}"
            );
            let (u, v) = projected.unwrap();
            surface.normal(u, v)
        }
    };
    if face.is_reversed() { -normal } else { normal }
}

fn assert_nurbs_seams_are_g1(topo: &Topology, solid: SolidId) {
    let adjacency = topo.build_adjacency(solid).unwrap();
    let mut seams = 0;
    let mut minimum_dot = 1.0_f64;
    for edge_id in solid_edges(topo, solid).unwrap() {
        let faces = adjacency.faces_for_edge(edge_id);
        let [left, right] = faces else { continue };
        if !matches!(topo.face(*left).unwrap().surface(), FaceSurface::Nurbs(_))
            && !matches!(topo.face(*right).unwrap().surface(), FaceSurface::Nurbs(_))
        {
            continue;
        }
        seams += 1;
        let edge = topo.edge(edge_id).unwrap();
        let start = topo.vertex(edge.start()).unwrap().point();
        let end = topo.vertex(edge.end()).unwrap().point();
        let (lo, hi) = edge.curve().domain_with_endpoints(start, end);
        for sample in 1..20 {
            let t = lo + (hi - lo) * f64::from(sample) / 20.0;
            let point = edge.curve().evaluate_with_endpoints(t, start, end);
            let dot =
                effective_normal(topo, *left, point).dot(effective_normal(topo, *right, point));
            minimum_dot = minimum_dot.min(dot);
            assert!(
                dot > 1.0 - 1.0e-7,
                "non-G1 seam {edge_id:?} sample={sample} dot={dot}"
            );
        }
    }
    eprintln!("N340 tangent seams={seams}, minimum sampled normal dot={minimum_dot:.16}");
    assert_eq!(seams, 7, "four retained N339 seams plus three runout seams");
}

fn runout_face(topo: &Topology, solid: SolidId) -> brepkit_topology::face::FaceId {
    let matches: Vec<_> = solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&face_id| {
            matches!(
                topo.face(face_id).unwrap().surface(),
                FaceSurface::Nurbs(surface)
                    if surface.degree_u() == 2 && surface.degree_v() == 3
            )
        })
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "expected one degree-(2,3) mixed-radius runout, got {matches:?}"
    );
    matches[0]
}

fn runout_samples(topo: &Topology, solid: SolidId) -> Vec<Point3> {
    let face = topo.face(runout_face(topo, solid)).unwrap();
    let FaceSurface::Nurbs(surface) = face.surface() else {
        unreachable!()
    };
    (0..=8)
        .flat_map(|i| {
            (0..=8).map(move |j| surface.evaluate(f64::from(i) / 8.0, f64::from(j) / 8.0))
        })
        .collect()
}

fn assert_runout_geometry(topo: &Topology, solid: SolidId) {
    let face = topo.face(runout_face(topo, solid)).unwrap();
    let FaceSurface::Nurbs(surface) = face.surface() else {
        unreachable!()
    };
    let mut previous_y = f64::NEG_INFINITY;
    for j in 0..=50 {
        let v = f64::from(j) / 50.0;
        let scale = 3.0 * v * v - 2.0 * v * v * v;
        let expected_y = 2.0 * ACCEPTED_RADIUS_MM + 2.0 * THIRD_RADIUS_MM * v;
        let radius = THIRD_RADIUS_MM * scale;
        let center = Point3::new(radius, expected_y, radius);
        for i in 0..=50 {
            let u = f64::from(i) / 50.0;
            let point = surface.evaluate(u, v);
            assert!((point.y() - expected_y).abs() < 1.0e-9);
            assert!(((point - center).length() - radius).abs() < 1.0e-9);
            assert!(point.x() >= -1.0e-10 && point.z() >= -1.0e-10);
            if j > 0 && i > 0 && i < 50 {
                let mut normal = surface.normal(u, v).unwrap();
                if face.is_reversed() {
                    normal = -normal;
                }
                let outward = Vec3::new(point.x() - radius, 0.0, point.z() - radius)
                    .normalize()
                    .unwrap();
                assert!(normal.dot(outward) > 0.0, "runout fold at ({u},{v})");
            }
        }
        assert!(
            expected_y > previous_y,
            "strict axial monotonicity rules out a folded-back runout"
        );
        previous_y = expected_y;
    }

    let old_patches: Vec<_> = solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .filter_map(|face_id| match topo.face(face_id).unwrap().surface() {
            FaceSurface::Nurbs(old) if old.degree_u() == 5 && old.degree_v() == 5 => Some(old),
            _ => None,
        })
        .collect();
    assert_eq!(old_patches.len(), 1);
    let mut maximum_old_y = f64::NEG_INFINITY;
    for i in 0..=50 {
        for j in 0..=50 {
            maximum_old_y = maximum_old_y.max(
                old_patches[0]
                    .evaluate(f64::from(i) / 50.0, f64::from(j) / 50.0)
                    .y(),
            );
        }
    }
    assert!(
        maximum_old_y <= 2.0 * ACCEPTED_RADIUS_MM + 1.0e-9,
        "old setback and new runout occupy disjoint axial interiors"
    );
}

#[test]
#[ignore = "diagnostic engine comparison; no route is adopted without geometry qualification"]
fn compare_mixed_corner_engine_routes_on_fresh_sources() {
    use brepkit_blend::radius_law::RadiusLaw;
    use brepkit_operations::blend_ops::{fillet_v2, fillet_v2_variable};
    use brepkit_operations::fillet::{FilletRadiusLaw, fillet_variable};

    let report = |label: &str,
                  topo: &Topology,
                  result: SolidId,
                  succeeded: usize,
                  failed: usize| {
        let validation = validate_solid(topo, result).unwrap();
        eprintln!(
            "N340 engine={label}: succeeded={succeeded} failed={failed} V/E/F={}/{}/{} boundary={} volume={:.15} issues={:?}",
            solid_vertices(topo, result).unwrap().len(),
            solid_edges(topo, result).unwrap().len(),
            solid_faces(topo, result).unwrap().len(),
            topo.build_adjacency(result).unwrap().boundary_edges().len(),
            solid_volume(topo, result, 0.01).unwrap(),
            validation.issues
        );
    };

    // Compare the alternative engine on an independently rebuilt copy of the
    // exact accepted two-blend source. A failed engine never feeds another route.
    {
        let mut topo = Topology::new();
        let source = captured_ordered_source(&mut topo);
        let first_edge = edge_at(
            &topo,
            source,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(40.0, 0.0, 0.0),
        );
        let first =
            fillet_rolling_ball(&mut topo, source, &[first_edge], ACCEPTED_RADIUS_MM).unwrap();
        let second_edge = edge_at(
            &topo,
            first,
            Point3::new(0.0, 0.0, ACCEPTED_RADIUS_MM),
            Point3::new(0.0, 0.0, 30.0),
        );
        let second =
            fillet_rolling_ball(&mut topo, first, &[second_edge], ACCEPTED_RADIUS_MM).unwrap();
        let third_edge = edge_at(
            &topo,
            second,
            Point3::new(0.0, 2.0 * ACCEPTED_RADIUS_MM, 0.0),
            Point3::new(0.0, 25.0, 0.0),
        );
        match fillet_v2(&mut topo, second, &[third_edge], THIRD_RADIUS_MM) {
            Ok(result) => report(
                "v2-after-accepted-pair",
                &topo,
                result.solid,
                result.succeeded.len(),
                result.failed.len(),
            ),
            Err(error) => eprintln!("N340 engine=v2-after-accepted-pair: error={error}"),
        }
    }

    // Compare both per-edge-radius APIs on independently rebuilt sharp source
    // snapshots with the three package targets, rather than feeding either one
    // topology already mutated by the other attempt.
    {
        let mut topo = Topology::new();
        let source = captured_ordered_source(&mut topo);
        let edges = [
            edge_at(
                &topo,
                source,
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(40.0, 0.0, 0.0),
            ),
            edge_at(
                &topo,
                source,
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 30.0),
            ),
            edge_at(
                &topo,
                source,
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 25.0, 0.0),
            ),
        ];
        let laws = vec![
            (edges[0], RadiusLaw::Constant(ACCEPTED_RADIUS_MM)),
            (edges[1], RadiusLaw::Constant(ACCEPTED_RADIUS_MM)),
            (edges[2], RadiusLaw::Constant(THIRD_RADIUS_MM)),
        ];
        match fillet_v2_variable(&mut topo, source, laws) {
            Ok(result) => report(
                "v2-variable-grouped-sharp-source",
                &topo,
                result.solid,
                result.succeeded.len(),
                result.failed.len(),
            ),
            Err(error) => eprintln!("N340 engine=v2-variable-grouped-sharp-source: error={error}"),
        }
    }
    {
        let mut topo = Topology::new();
        let source = captured_ordered_source(&mut topo);
        let edges = [
            edge_at(
                &topo,
                source,
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(40.0, 0.0, 0.0),
            ),
            edge_at(
                &topo,
                source,
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 30.0),
            ),
            edge_at(
                &topo,
                source,
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 25.0, 0.0),
            ),
        ];
        let laws = [
            (edges[0], FilletRadiusLaw::Constant(ACCEPTED_RADIUS_MM)),
            (edges[1], FilletRadiusLaw::Constant(ACCEPTED_RADIUS_MM)),
            (edges[2], FilletRadiusLaw::Constant(THIRD_RADIUS_MM)),
        ];
        match fillet_variable(&mut topo, source, &laws) {
            Ok(result) => report("legacy-variable-grouped-sharp-source", &topo, result, 3, 0),
            Err(error) => {
                eprintln!("N340 engine=legacy-variable-grouped-sharp-source: error={error}");
            }
        }
    }
}

#[test]
fn exact_third_mixed_radius_after_two_accepted_blends_is_valid() {
    let mut topo = Topology::new();
    let source = captured_ordered_source(&mut topo);
    assert_valid_stage(&topo, source, 30_000.0);
    let source_state = snapshot(&topo, source);

    let first_target = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let first =
        fillet_rolling_ball(&mut topo, source, &[first_target], ACCEPTED_RADIUS_MM).unwrap();
    assert_valid_stage(&topo, first, 29_944.330_706_290_57);
    let first_state = snapshot(&topo, first);

    // Re-resolve on the first replacement; native edge ids are deliberately not reused.
    let second_target = edge_at(
        &topo,
        first,
        Point3::new(0.0, 0.0, ACCEPTED_RADIUS_MM),
        Point3::new(0.0, 0.0, 30.0),
    );
    let second =
        fillet_rolling_ball(&mut topo, first, &[second_target], ACCEPTED_RADIUS_MM).unwrap();
    assert_valid_stage(&topo, second, 29_901.371_084_328_155);
    let second_state = snapshot(&topo, second);

    // Re-resolve edge49's captured current geometry on the second replacement.
    let third_target = edge_at(
        &topo,
        second,
        Point3::new(0.0, 2.0 * ACCEPTED_RADIUS_MM, 0.0),
        Point3::new(0.0, 25.0, 0.0),
    );
    let third = fillet_rolling_ball(&mut topo, second, &[third_target], THIRD_RADIUS_MM).unwrap();
    let validation = validate_solid(&topo, third).unwrap();
    let adjacency = topo.build_adjacency(third).unwrap();
    let coarse_volume = solid_volume(&topo, third, 0.01).unwrap();
    eprintln!(
        "N340 third: V/E/F={}/{}/{} boundary={} non_manifold={} volume={coarse_volume:.15} issues={:?}",
        solid_vertices(&topo, third).unwrap().len(),
        solid_edges(&topo, third).unwrap().len(),
        solid_faces(&topo, third).unwrap().len(),
        adjacency.boundary_edges().len(),
        adjacency.non_manifold_edges().len(),
        validation.issues
    );
    assert!(
        validation.is_valid(),
        "third fillet must be a valid closed solid: {:?}",
        validation.issues
    );
    let full_check =
        brepkit_check::validate::validate_solid(&topo, third, &ValidateOptions::default()).unwrap();
    assert!(
        full_check
            .issues
            .iter()
            .all(|issue| issue.severity != CheckSeverity::Error),
        "full geometric/topological check: {:?}",
        full_check.issues
    );
    assert_eq!(
        (
            solid_vertices(&topo, third).unwrap().len(),
            solid_edges(&topo, third).unwrap().len(),
            solid_faces(&topo, third).unwrap().len(),
        ),
        (16, 25, 11)
    );
    assert!(oriented_solid_volume(&topo, third, 0.01).unwrap() > 0.0);

    let mut radii: Vec<_> = solid_faces(&topo, third)
        .unwrap()
        .into_iter()
        .filter_map(|face_id| match topo.face(face_id).unwrap().surface() {
            FaceSurface::Cylinder(cylinder) => Some(cylinder.radius()),
            _ => None,
        })
        .collect();
    radii.sort_by(f64::total_cmp);
    assert_eq!(radii.len(), 3);
    for (actual, expected) in
        radii
            .iter()
            .zip([THIRD_RADIUS_MM, ACCEPTED_RADIUS_MM, ACCEPTED_RADIUS_MM])
    {
        assert!((*actual - expected).abs() < Tolerance::new().linear);
    }
    assert_eq!(
        solid_faces(&topo, third)
            .unwrap()
            .into_iter()
            .filter(|&face_id| matches!(
                topo.face(face_id).unwrap().surface(),
                FaceSurface::Plane { .. }
            ))
            .count(),
        6,
        "no overlapping terminal cap may be added"
    );
    assert_runout_geometry(&topo, third);
    assert_nurbs_seams_are_g1(&topo, third);

    // The exact runout removes A = r^2(1-pi/4) per unit length. Its smoothstep
    // scale has integral int_0^1 (3s^2-2s^3)^2 ds = 13/35 over a 2r transition.
    let area = THIRD_RADIUS_MM.powi(2) * (1.0 - std::f64::consts::PI / 4.0);
    let target_length = 25.0 - 2.0 * ACCEPTED_RADIUS_MM;
    let runout_length = 2.0 * THIRD_RADIUS_MM;
    let expected_removed = area * (target_length - runout_length + runout_length * 13.0 / 35.0);
    let fine_removed =
        solid_volume(&topo, second, 0.0001).unwrap() - solid_volume(&topo, third, 0.0001).unwrap();
    eprintln!(
        "N340 mass: coarse volume={coarse_volume:.15}; fine removed={fine_removed:.15}; analytic removed={expected_removed:.15}"
    );
    assert!((fine_removed - expected_removed).abs() < 0.01);
    assert!(coarse_volume < solid_volume(&topo, second, 0.01).unwrap());

    assert_eq!(
        snapshot(&topo, source),
        source_state,
        "original source mutated"
    );
    assert_eq!(
        snapshot(&topo, first),
        first_state,
        "first accepted source mutated"
    );
    assert_eq!(
        snapshot(&topo, second),
        second_state,
        "second accepted source mutated"
    );
}

#[test]
fn mixed_runout_is_rigidly_covariant_and_supports_bounded_smaller_radii() {
    use brepkit_math::mat::Mat4;
    use brepkit_operations::transform::transform_solid;

    let transforms = [
        Mat4::identity(),
        Mat4::rotation_z(0.37) * Mat4::rotation_x(-0.21),
        Mat4::translation(150.0, -230.0, 42.0) * Mat4::rotation_y(0.63),
    ];
    let radius_pairs = [(2.54, 1.27), (2.0, 0.5), (1.5, 1.0)];
    let mut references = std::collections::HashMap::new();

    for (pose, transform) in transforms.iter().enumerate() {
        for (case, (accepted_radius, third_radius)) in radius_pairs.into_iter().enumerate() {
            let mut topo = Topology::new();
            let source = captured_ordered_source(&mut topo);
            transform_solid(&mut topo, source, transform).unwrap();
            let source_state = snapshot(&topo, source);
            let first_target = edge_at(
                &topo,
                source,
                transform.mul_point(Point3::new(0.0, 0.0, 0.0)),
                transform.mul_point(Point3::new(40.0, 0.0, 0.0)),
            );
            let first =
                fillet_rolling_ball(&mut topo, source, &[first_target], accepted_radius).unwrap();
            let first_state = snapshot(&topo, first);
            let second_target = edge_at(
                &topo,
                first,
                transform.mul_point(Point3::new(0.0, 0.0, accepted_radius)),
                transform.mul_point(Point3::new(0.0, 0.0, 30.0)),
            );
            let second =
                fillet_rolling_ball(&mut topo, first, &[second_target], accepted_radius).unwrap();
            let second_state = snapshot(&topo, second);
            let third_target = edge_at(
                &topo,
                second,
                transform.mul_point(Point3::new(0.0, 2.0 * accepted_radius, 0.0)),
                transform.mul_point(Point3::new(0.0, 25.0, 0.0)),
            );
            let third =
                fillet_rolling_ball(&mut topo, second, &[third_target], third_radius).unwrap();
            let validation = validate_solid(&topo, third).unwrap();
            assert!(
                validation.is_valid(),
                "pose={pose} case={case}: {:?}",
                validation.issues
            );
            assert!(oriented_solid_volume(&topo, third, 0.01).unwrap() > 0.0);
            assert_nurbs_seams_are_g1(&topo, third);

            let mut radii: Vec<_> = solid_faces(&topo, third)
                .unwrap()
                .into_iter()
                .filter_map(|face_id| match topo.face(face_id).unwrap().surface() {
                    FaceSurface::Cylinder(cylinder) => Some(cylinder.radius()),
                    _ => None,
                })
                .collect();
            radii.sort_by(f64::total_cmp);
            assert_eq!(radii, [third_radius, accepted_radius, accepted_radius]);

            let samples = runout_samples(&topo, third);
            if pose == 0 {
                references.insert(case, samples);
            } else {
                let expected = &references[&case];
                assert_eq!(samples.len(), expected.len());
                for (&actual, &untransformed) in samples.iter().zip(expected) {
                    assert!(
                        (actual - transform.mul_point(untransformed)).length()
                            < Tolerance::new().linear * 100.0,
                        "pose={pose} case={case}: runout lost rigid covariance"
                    );
                }
            }
            assert_eq!(snapshot(&topo, source), source_state);
            assert_eq!(snapshot(&topo, first), first_state);
            assert_eq!(snapshot(&topo, second), second_state);
        }
    }
}

#[test]
fn three_fresh_exact_sequences_have_one_geometry_fingerprint() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut fingerprints = Vec::new();
    let mut volumes = Vec::new();
    for pass in 0..3 {
        let mut topo = Topology::new();
        let source = captured_ordered_source(&mut topo);
        let first_target = edge_at(
            &topo,
            source,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(40.0, 0.0, 0.0),
        );
        let first =
            fillet_rolling_ball(&mut topo, source, &[first_target], ACCEPTED_RADIUS_MM).unwrap();
        let second_target = edge_at(
            &topo,
            first,
            Point3::new(0.0, 0.0, ACCEPTED_RADIUS_MM),
            Point3::new(0.0, 0.0, 30.0),
        );
        let second =
            fillet_rolling_ball(&mut topo, first, &[second_target], ACCEPTED_RADIUS_MM).unwrap();
        let third_target = edge_at(
            &topo,
            second,
            Point3::new(0.0, 2.0 * ACCEPTED_RADIUS_MM, 0.0),
            Point3::new(0.0, 25.0, 0.0),
        );
        let third =
            fillet_rolling_ball(&mut topo, second, &[third_target], THIRD_RADIUS_MM).unwrap();
        assert!(validate_solid(&topo, third).unwrap().is_valid());
        let mesh = brepkit_operations::tessellate::tessellate_solid(&topo, third, 0.001).unwrap();
        let mut triangles = mesh
            .indices
            .chunks_exact(3)
            .map(|indices| {
                let quantize = |index| {
                    let point = mesh.positions[index as usize];
                    [
                        (point.x() * 1_000_000.0).round() as i64,
                        (point.y() * 1_000_000.0).round() as i64,
                        (point.z() * 1_000_000.0).round() as i64,
                    ]
                };
                let mut points = [
                    quantize(indices[0]),
                    quantize(indices[1]),
                    quantize(indices[2]),
                ];
                points.sort_unstable();
                points
            })
            .collect::<Vec<_>>();
        triangles.sort_unstable();
        assert!(!triangles.is_empty());
        let mut hasher = DefaultHasher::new();
        triangles.hash(&mut hasher);
        let fingerprint = hasher.finish();
        let volume = solid_volume(&topo, third, 0.001).unwrap();
        eprintln!(
            "N340 native replay={pass}: fingerprint={fingerprint:016x} volume={volume:.15} triangles={}",
            mesh.indices.len() / 3
        );
        fingerprints.push(fingerprint);
        volumes.push(volume);
    }
    assert!(fingerprints.windows(2).all(|pair| pair[0] == pair[1]));
    assert!(
        volumes
            .windows(2)
            .all(|pair| (pair[0] - pair[1]).abs() < f64::EPSILON)
    );
}

#[test]
fn mixed_runout_clearance_and_unrecognized_invalid_output_preserve_accepted_source() {
    let mut topo = Topology::new();
    let source = ordered_source(&mut topo, 40.0, 7.0, 30.0);
    let first_target = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let first =
        fillet_rolling_ball(&mut topo, source, &[first_target], ACCEPTED_RADIUS_MM).unwrap();
    let second_target = edge_at(
        &topo,
        first,
        Point3::new(0.0, 0.0, ACCEPTED_RADIUS_MM),
        Point3::new(0.0, 0.0, 30.0),
    );
    let second =
        fillet_rolling_ball(&mut topo, first, &[second_target], ACCEPTED_RADIUS_MM).unwrap();
    assert!(validate_solid(&topo, second).unwrap().is_valid());
    let state = snapshot(&topo, second);
    let third_target = edge_at(
        &topo,
        second,
        Point3::new(0.0, 2.0 * ACCEPTED_RADIUS_MM, 0.0),
        Point3::new(0.0, 7.0, 0.0),
    );
    let error =
        fillet_rolling_ball(&mut topo, second, &[third_target], THIRD_RADIUS_MM).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("runout requires more than twice"),
        "{error}"
    );
    assert_eq!(
        snapshot(&topo, second),
        state,
        "clearance refusal mutated accepted source"
    );

    // N342 (pin v3.4.0-ottocad.4): an equal third radius was, through
    // v3.4.0-ottocad.3, deliberately outside the smaller-radius runout
    // recognizer, and the unchanged ordinary engine returned a historically
    // invalid open shell here (V=16 E=25 F=10, 5 boundary edges) -- this
    // control asserted exactly that `!report.is_valid()`. N342 repairs this
    // specific case (all three radii equal) with a different, simpler exact
    // construction -- a genuine sphere octant, not an extension of N340's
    // runout -- see `crates/operations/src/fillet/reblend.rs`'s
    // `try_spherical_corner` and `docs/N342-*.md`. This control now asserts
    // the resulting positive behavior; its history is preserved in this
    // comment rather than deleting the test, matching
    // `regress_n341_naive_control.rs`'s own precedent.
    let mut topo = Topology::new();
    let source = captured_ordered_source(&mut topo);
    let first_target = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let first =
        fillet_rolling_ball(&mut topo, source, &[first_target], ACCEPTED_RADIUS_MM).unwrap();
    let second_target = edge_at(
        &topo,
        first,
        Point3::new(0.0, 0.0, ACCEPTED_RADIUS_MM),
        Point3::new(0.0, 0.0, 30.0),
    );
    let second =
        fillet_rolling_ball(&mut topo, first, &[second_target], ACCEPTED_RADIUS_MM).unwrap();
    let state = snapshot(&topo, second);
    let third_target = edge_at(
        &topo,
        second,
        Point3::new(0.0, 2.0 * ACCEPTED_RADIUS_MM, 0.0),
        Point3::new(0.0, 25.0, 0.0),
    );
    let equal_radius_corner =
        fillet_rolling_ball(&mut topo, second, &[third_target], ACCEPTED_RADIUS_MM).unwrap();
    let report = validate_solid(&topo, equal_radius_corner).unwrap();
    assert!(
        report.is_valid(),
        "N342 equal-radius corner: {:?}",
        report.issues
    );
    assert!(
        topo.build_adjacency(equal_radius_corner)
            .unwrap()
            .boundary_edges()
            .is_empty()
    );
    assert_eq!(
        (
            solid_vertices(&topo, equal_radius_corner).unwrap().len(),
            solid_edges(&topo, equal_radius_corner).unwrap().len(),
            solid_faces(&topo, equal_radius_corner).unwrap().len(),
        ),
        (13, 21, 10),
        "N342 sphere-octant corner topology"
    );
    let sphere_faces: Vec<_> = solid_faces(&topo, equal_radius_corner)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
        .collect();
    assert_eq!(
        sphere_faces.len(),
        1,
        "exactly one exact sphere-octant face"
    );
    let FaceSurface::Sphere(sphere) = topo.face(sphere_faces[0]).unwrap().surface() else {
        unreachable!()
    };
    assert!((sphere.radius() - ACCEPTED_RADIUS_MM).abs() < Tolerance::new().linear);
    assert_eq!(
        snapshot(&topo, second),
        state,
        "accepted two-strip source mutated by the equal-radius corner build"
    );
    assert!(oriented_solid_volume(&topo, equal_radius_corner, 0.01).unwrap() > 0.0);

    let excessive = fillet_rolling_ball(&mut topo, second, &[third_target], 100.0);
    assert!(excessive.is_err());
    assert_eq!(
        snapshot(&topo, second),
        state,
        "excessive-radius refusal mutated source"
    );
}

/// N342b: `MixedRadiusRunout::recognize` shares `matches_n339_patch`'s same
/// single-fixed-order boundary match `try_spherical_corner` had (see
/// `regress_fillet_third_equal_radius_corner.rs`'s own permutation sweep and
/// `reblend.rs`'s `matches_n339_patch_either`) -- confirmed by direct
/// instrumentation to be independently order-dependent for THIS smaller-
/// third-radius runout too, not merely a copy-paste risk: three of the six
/// fillet orders reproduced the identical `V=16,E=25,F=10` open-shell defect
/// before this fix, exactly matching N342's own equal-radius signature.
/// Fixed in the same `reblend.rs` change (the `swapped` reorder of
/// `boundaries` before `Self { first, second, supports, .. }` is built).
/// This sweep locks in order-independence across all 6 fillet orders and
/// all 8 cube corners for the smaller-third-radius runout, since N340's own
/// original coverage (like N342's) only ever exercised one order.
#[test]
fn mixed_third_edge_is_valid_for_every_order_and_every_corner() {
    const W: f64 = 40.0;
    const D: f64 = 25.0;
    const H: f64 = 30.0;
    const LENGTHS: [f64; 3] = [W, D, H];

    fn edge_pts(corner: [f64; 3], axis: usize, trim: f64) -> (Point3, Point3) {
        let mut near = corner;
        let mut far = corner;
        let dir = if corner[axis] == 0.0 { 1.0 } else { -1.0 };
        near[axis] = corner[axis] + dir * trim;
        far[axis] = if corner[axis] == 0.0 {
            LENGTHS[axis]
        } else {
            0.0
        };
        (
            Point3::new(near[0], near[1], near[2]),
            Point3::new(far[0], far[1], far[2]),
        )
    }

    let orders: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let corners: [[f64; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [W, 0.0, 0.0],
        [0.0, D, 0.0],
        [0.0, 0.0, H],
        [W, D, 0.0],
        [W, 0.0, H],
        [0.0, D, H],
        [W, D, H],
    ];

    let mut cases = 0;
    for &corner in &corners {
        for &order in &orders {
            let mut topo = Topology::new();
            let mut solid = captured_ordered_source(&mut topo);
            for (step, &axis) in order.iter().enumerate() {
                let radius = if step == 2 {
                    THIRD_RADIUS_MM
                } else {
                    ACCEPTED_RADIUS_MM
                };
                let trim = f64::from(u8::try_from(step).unwrap()) * ACCEPTED_RADIUS_MM;
                let (near, far) = edge_pts(corner, axis, trim);
                let target = edge_at(&topo, solid, near, far);
                let result = fillet_rolling_ball(&mut topo, solid, &[target], radius);
                if let Err(e) = &result {
                    eprintln!("corner={corner:?} order={order:?} step={step}: fillet error {e:?}");
                }
                solid = result.unwrap();
            }

            let validation = validate_solid(&topo, solid).unwrap();
            assert!(
                validation.is_valid(),
                "corner={corner:?} order={order:?}: {:?}",
                validation.issues
            );
            assert_eq!(
                (
                    solid_vertices(&topo, solid).unwrap().len(),
                    solid_edges(&topo, solid).unwrap().len(),
                    solid_faces(&topo, solid).unwrap().len(),
                ),
                (16, 25, 11),
                "corner={corner:?} order={order:?}"
            );

            let mut radii: Vec<_> = solid_faces(&topo, solid)
                .unwrap()
                .into_iter()
                .filter_map(|face_id| match topo.face(face_id).unwrap().surface() {
                    FaceSurface::Cylinder(cylinder) => Some(cylinder.radius()),
                    _ => None,
                })
                .collect();
            radii.sort_by(f64::total_cmp);
            assert_eq!(radii.len(), 3, "corner={corner:?} order={order:?}");
            for (actual, expected) in
                radii
                    .iter()
                    .zip([THIRD_RADIUS_MM, ACCEPTED_RADIUS_MM, ACCEPTED_RADIUS_MM])
            {
                assert!(
                    (*actual - expected).abs() < Tolerance::new().linear,
                    "corner={corner:?} order={order:?}"
                );
            }

            assert_nurbs_seams_are_g1(&topo, solid);
            let oriented = oriented_solid_volume(&topo, solid, 0.01).unwrap();
            assert!(
                oriented.is_finite() && oriented > 0.0,
                "corner={corner:?} order={order:?}"
            );
            cases += 1;
        }
    }
    eprintln!("N340 order/corner sweep: {cases} cases valid, all 6 orders x 8 corners");
    assert_eq!(cases, 48);
}
