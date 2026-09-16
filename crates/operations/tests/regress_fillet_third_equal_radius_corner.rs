//! OttoCAD N342: the third mutually-perpendicular edge at a box corner,
//! requested at the SAME radius as the two already-accepted fillets.
//!
//! User-reproduced sequence (2026-09-07): fillet edge at 0.1 in, accept;
//! fillet an adjacent edge at 0.1 in, accept; fillet the corner's third
//! mutually-perpendicular edge at 0.1 in. On the unmodified `v3.4.0-ottocad.3`
//! pin this reproduces the literal reported failure:
//!
//! ```text
//! V=16, E=25, F=10, 5 boundary edges;
//! Euler characteristic V-E+F = 1 is invalid
//! ```
//!
//! This is a DIFFERENT defect from N340's (smaller third radius growing a
//! runout from N339's setback patch's own singular point): with all three
//! radii equal, the exact corner is a genuine sphere octant -- rolling a
//! ball of the same radius into a convex right-angle corner traces a sphere
//! tangent to all three quarter-cylinder strips along a full analytic
//! circle. The accepted two-strip N339 patch is not extended; it is wholly
//! replaced. See `crates/operations/src/fillet/reblend.rs`'s
//! `try_spherical_corner`/`build_spherical_corner_solid`/
//! `replace_approximate_corner_with_sphere` and `docs/N342-*.md` for the
//! derivation and the tessellator fix this construction also required.

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

const R_MM: f64 = 2.54; // 0.1 inch, the user's literal report.

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
    // Same 40 x 25 x 30 mm extrusion N339/N340/N341 use.
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

/// Every seam bordering the sphere face, plus every cylinder/plane tangent
/// line, must be exactly G1 (the sphere and each cylinder share a common
/// tangent plane along the whole seam, not merely touch at a point -- see
/// module doc). The three far-end cap arcs (each cylinder meeting the
/// UNRELATED flat end face at its own untouched, ordinary far end) are
/// genuine sharp corners, not tangent seams, and are excluded by name.
fn assert_corner_seams_are_g1(topo: &Topology, solid: SolidId) -> (usize, f64) {
    let adjacency = topo.build_adjacency(solid).unwrap();
    let sphere_face = solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
        .expect("exactly one sphere face");

    let mut tangent_seams = 0;
    let mut minimum_dot = 1.0_f64;
    for edge_id in solid_edges(topo, solid).unwrap() {
        let faces = adjacency.faces_for_edge(edge_id);
        let [left, right] = faces else { continue };
        let touches_sphere = *left == sphere_face || *right == sphere_face;
        let both_cylinder_or_sphere = [*left, *right].iter().all(|&f| {
            matches!(
                topo.face(f).unwrap().surface(),
                FaceSurface::Cylinder(_) | FaceSurface::Sphere(_)
            )
        });
        // Tangent seams: sphere<->cylinder (the corner's own three joins)
        // and cylinder<->plane axial tangent lines. Cylinder<->plane far-cap
        // arcs, and any ordinary box edge unrelated to these three fillet
        // strips, are genuine sharp corners (or simply irrelevant) and must
        // NOT be asserted G1 -- a bare `EdgeCurve::Line` check alone would
        // also catch the box's own untouched far-wall edges.
        let touches_a_fillet_cylinder = [*left, *right]
            .iter()
            .any(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)));
        let is_axial_tangent_line = touches_a_fillet_cylinder
            && [*left, *right].iter().all(|&f| {
                matches!(
                    topo.face(f).unwrap().surface(),
                    FaceSurface::Cylinder(_) | FaceSurface::Plane { .. }
                )
            })
            && matches!(
                topo.edge(edge_id).unwrap().curve(),
                brepkit_topology::edge::EdgeCurve::Line
            );
        if !(is_axial_tangent_line || (touches_sphere && both_cylinder_or_sphere)) {
            continue;
        }
        tangent_seams += 1;
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
                dot > 1.0 - 1.0e-6,
                "non-G1 seam {edge_id:?} sample={sample} dot={dot} point={point:?} left={left:?} right={right:?}"
            );
        }
    }
    (tangent_seams, minimum_dot)
}

// Red baseline (historical, not a live test): `try_spherical_corner` is
// wired directly into `try_adjoining_strip`, the same reblend path every
// `fillet_rolling_ball` call already goes through, so once N342 lands there
// is no way to reach "the plain unmodified route" through the public API
// any more to keep a permanently-red control here (N339/N340 do not keep
// one either, for the same reason). The red baseline was captured directly
// against the exact reported sequence and radii, on the unmodified
// `v3.4.0-ottocad.3` pin (`git stash` on this same clone, reverting only
// `crates/operations/src/fillet/reblend.rs`, before any N342 source change):
//
// ```text
// N342 RED baseline: V/E/F=16/25/10 boundary=5 valid=false
// issues=[Euler characteristic V-E+F = 1 is invalid (expected V-E+F = 2+L
// with L=0 inner loops, got V=16, E=25, F=10), 5 boundary edge(s) found
// (shell is not closed)]
// ```
//
// This is the literal signature from the user's report, reproduced exactly
// before any repair code was written.

#[test]
fn equal_radius_third_edge_after_two_accepted_blends_is_valid() {
    let mut topo = Topology::new();
    let source = captured_ordered_source(&mut topo);
    let source_state = snapshot(&topo, source);

    let first_target = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let first = fillet_rolling_ball(&mut topo, source, &[first_target], R_MM).unwrap();
    assert!(validate_solid(&topo, first).unwrap().is_valid());
    let first_state = snapshot(&topo, first);

    let second_target = edge_at(
        &topo,
        first,
        Point3::new(0.0, 0.0, R_MM),
        Point3::new(0.0, 0.0, 30.0),
    );
    let second = fillet_rolling_ball(&mut topo, first, &[second_target], R_MM).unwrap();
    assert!(validate_solid(&topo, second).unwrap().is_valid());
    let second_state = snapshot(&topo, second);

    // N342's own route: the SAME public entry point (`fillet_rolling_ball`
    // -> `try_adjoining_strip` -> `try_spherical_corner`), on the accepted
    // two-strip source, at the SAME radius as both inherited strips.
    let third_target = edge_at(
        &topo,
        second,
        Point3::new(0.0, 2.0 * R_MM, 0.0),
        Point3::new(0.0, 25.0, 0.0),
    );
    let third = fillet_rolling_ball(&mut topo, second, &[third_target], R_MM).unwrap();

    let validation = validate_solid(&topo, third).unwrap();
    let adjacency = topo.build_adjacency(third).unwrap();
    let coarse_volume = solid_volume(&topo, third, 0.01).unwrap();
    eprintln!(
        "N342 GREEN: V/E/F={}/{}/{} boundary={} volume={coarse_volume:.15} issues={:?}",
        solid_vertices(&topo, third).unwrap().len(),
        solid_edges(&topo, third).unwrap().len(),
        solid_faces(&topo, third).unwrap().len(),
        adjacency.boundary_edges().len(),
        validation.issues,
    );
    assert!(validation.is_valid(), "{:?}", validation.issues);
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
        (13, 21, 10)
    );

    // Exact intended radii on all three strips.
    let mut cyl_radii: Vec<_> = solid_faces(&topo, third)
        .unwrap()
        .into_iter()
        .filter_map(|f| match topo.face(f).unwrap().surface() {
            FaceSurface::Cylinder(c) => Some(c.radius()),
            _ => None,
        })
        .collect();
    cyl_radii.sort_by(f64::total_cmp);
    assert_eq!(cyl_radii.len(), 3);
    for r in cyl_radii {
        assert!((r - R_MM).abs() < Tolerance::new().linear);
    }
    let sphere_faces: Vec<_> = solid_faces(&topo, third)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
        .collect();
    assert_eq!(
        sphere_faces.len(),
        1,
        "exactly one exact analytic sphere face"
    );
    let FaceSurface::Sphere(sphere) = topo.face(sphere_faces[0]).unwrap().surface() else {
        unreachable!()
    };
    assert!((sphere.radius() - R_MM).abs() < Tolerance::new().linear);
    // The sphere octant hypothesis: center offset R along each of the three
    // face normals from the original sharp corner (0,0,0) -- i.e. (R,R,R).
    assert!((sphere.center() - Point3::new(R_MM, R_MM, R_MM)).length() < 1.0e-9);
    assert_eq!(
        solid_faces(&topo, third)
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { .. }))
            .count(),
        6,
        "no overlapping terminal cap may be added"
    );

    // G1 normal-continuity sampling on every real tangent seam (sphere<->
    // cylinder and cylinder<->plane axial tangent lines): 3 sphere seams +
    // 6 axial tangent lines = 9. Report min dot and sample count.
    let (tangent_seams, minimum_dot) = assert_corner_seams_are_g1(&topo, third);
    eprintln!("N342 tangent seams={tangent_seams}, minimum sampled normal dot={minimum_dot:.16}");
    assert_eq!(
        tangent_seams, 9,
        "3 sphere<->cylinder + 6 cylinder<->plane axial tangent lines"
    );

    // No folds/overlaps, correct outward orientation, positive finite volume.
    let oriented = oriented_solid_volume(&topo, third, 0.01).unwrap();
    assert!(oriented.is_finite() && oriented > 0.0);
    assert!(coarse_volume.is_finite() && coarse_volume > 0.0);
    // Oriented (divergence-theorem) and absolute tessellated volume should
    // closely agree for a correctly, consistently outward-oriented solid; a
    // folded or inward-facing patch would show up as a large discrepancy
    // here (verified during development: a wrong sphere pole placement
    // produced grossly inflated, badly mismatched values).
    assert!(
        (oriented - coarse_volume).abs() < 5.0,
        "oriented={oriented} vs coarse={coarse_volume}: excessive mismatch suggests a fold or bad orientation"
    );

    // Cross-check against the independently derived closed-form removed
    // volume: three quarter-cylinder strips of length (edge length - R)
    // each, plus the sphere-octant complement of the corner cube --
    // R^2(1-pi/4)*sum(L_i - R) + R^3(1-pi/6), L = {40, 30, 25}.
    let area = R_MM.powi(2) * (1.0 - std::f64::consts::PI / 4.0);
    let strip_removed = area * ((40.0 - R_MM) + (30.0 - R_MM) + (25.0 - R_MM));
    let corner_removed = R_MM.powi(3) * (1.0 - std::f64::consts::PI / 6.0);
    let analytic_removed = strip_removed + corner_removed;
    let fine_volume = solid_volume(&topo, third, 0.00001).unwrap();
    let fine_removed = 30_000.0 - fine_volume;
    eprintln!(
        "N342 mass: fine volume={fine_volume:.15}; fine removed={fine_removed:.15}; analytic removed={analytic_removed:.15}"
    );
    assert!(
        (fine_removed - analytic_removed).abs() < 0.01,
        "fine_removed={fine_removed} analytic_removed={analytic_removed}"
    );
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
fn equal_radius_third_edge_is_rigidly_covariant_across_poses_and_scales() {
    use brepkit_math::mat::Mat4;
    use brepkit_operations::transform::transform_solid;

    let transforms = [
        Mat4::identity(),
        Mat4::rotation_z(0.41) * Mat4::rotation_x(0.19),
        Mat4::translation(-80.0, 130.0, 27.0) * Mat4::rotation_y(-0.57),
    ];
    let radii = [R_MM, 1.0, 1.5];

    for (pose, transform) in transforms.iter().enumerate() {
        for (case, &r) in radii.iter().enumerate() {
            let mut topo = Topology::new();
            let source = captured_ordered_source(&mut topo);
            transform_solid(&mut topo, source, transform).unwrap();
            let first_target = edge_at(
                &topo,
                source,
                transform.mul_point(Point3::new(0.0, 0.0, 0.0)),
                transform.mul_point(Point3::new(40.0, 0.0, 0.0)),
            );
            let first = fillet_rolling_ball(&mut topo, source, &[first_target], r).unwrap();
            let second_target = edge_at(
                &topo,
                first,
                transform.mul_point(Point3::new(0.0, 0.0, r)),
                transform.mul_point(Point3::new(0.0, 0.0, 30.0)),
            );
            let second = fillet_rolling_ball(&mut topo, first, &[second_target], r).unwrap();
            let third_target = edge_at(
                &topo,
                second,
                transform.mul_point(Point3::new(0.0, 2.0 * r, 0.0)),
                transform.mul_point(Point3::new(0.0, 25.0, 0.0)),
            );
            let third = fillet_rolling_ball(&mut topo, second, &[third_target], r).unwrap();
            let validation = validate_solid(&topo, third).unwrap();
            assert!(
                validation.is_valid(),
                "pose={pose} case={case}: {:?}",
                validation.issues
            );
            assert!(oriented_solid_volume(&topo, third, 0.01).unwrap() > 0.0);
            let (tangent_seams, minimum_dot) = assert_corner_seams_are_g1(&topo, third);
            assert_eq!(tangent_seams, 9, "pose={pose} case={case}");
            assert!(
                minimum_dot > 1.0 - 1.0e-6,
                "pose={pose} case={case}: min_dot={minimum_dot}"
            );
        }
    }
}

/// N342 defect fix: the equal-radius third corner used to be built via
/// `try_spherical_corner`'s single fixed-order boundary match against
/// `matches_n339_patch`. That match's own two boundary directions are
/// discovered by walking topology storage order, which tracks which of the
/// first two edges the user filleted first -- not the corner's geometry --
/// so for a given physical corner, three of the six possible fillet orders
/// (the ones whose first two edges were filleted in the "wrong" relative
/// order) silently reproduced the exact reported user failure
/// (`V=16,E=25,F=10`, 5 boundary edges) while the other three passed. Fixed
/// by `matches_n339_patch_either` in `reblend.rs`, which tries both
/// enumeration orders instead of just one. This sweep locks in the fix
/// across every one of the 6 fillet orders and all 8 cube corners (48
/// cases) so the same order-dependence cannot silently regress: N342's
/// original coverage above exercised exactly one order
/// (`[X, Z, Y]`, one of the three that happened to already work) and did
/// not catch this.
#[test]
fn equal_radius_third_edge_is_valid_for_every_order_and_every_corner() {
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

    fn sphere_center(corner: [f64; 3], r: f64) -> Point3 {
        let mut center = corner;
        for (axis, value) in center.iter_mut().enumerate() {
            let dir = if corner[axis] == 0.0 { 1.0 } else { -1.0 };
            *value += dir * r;
        }
        Point3::new(center[0], center[1], center[2])
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
                let trim = f64::from(u8::try_from(step).unwrap()) * R_MM;
                let (near, far) = edge_pts(corner, axis, trim);
                let target = edge_at(&topo, solid, near, far);
                let result = fillet_rolling_ball(&mut topo, solid, &[target], R_MM);
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
                (13, 21, 10),
                "corner={corner:?} order={order:?}"
            );

            let mut cyl_radii: Vec<_> = solid_faces(&topo, solid)
                .unwrap()
                .into_iter()
                .filter_map(|f| match topo.face(f).unwrap().surface() {
                    FaceSurface::Cylinder(c) => Some(c.radius()),
                    _ => None,
                })
                .collect();
            cyl_radii.sort_by(f64::total_cmp);
            assert_eq!(cyl_radii.len(), 3, "corner={corner:?} order={order:?}");
            for r in cyl_radii {
                assert!(
                    (r - R_MM).abs() < Tolerance::new().linear,
                    "corner={corner:?} order={order:?}"
                );
            }

            let sphere_faces: Vec<_> = solid_faces(&topo, solid)
                .unwrap()
                .into_iter()
                .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
                .collect();
            assert_eq!(
                sphere_faces.len(),
                1,
                "corner={corner:?} order={order:?}: exactly one exact analytic sphere face"
            );
            let FaceSurface::Sphere(sphere) = topo.face(sphere_faces[0]).unwrap().surface() else {
                unreachable!()
            };
            assert!(
                (sphere.radius() - R_MM).abs() < Tolerance::new().linear,
                "corner={corner:?} order={order:?}"
            );
            assert!(
                (sphere.center() - sphere_center(corner, R_MM)).length() < 1.0e-9,
                "corner={corner:?} order={order:?}: sphere center {:?} expected {:?}",
                sphere.center(),
                sphere_center(corner, R_MM)
            );

            let (tangent_seams, minimum_dot) = assert_corner_seams_are_g1(&topo, solid);
            assert_eq!(
                tangent_seams, 9,
                "corner={corner:?} order={order:?}: 3 sphere<->cylinder + 6 cylinder<->plane axial tangent lines"
            );
            assert!(
                minimum_dot > 1.0 - 1.0e-6,
                "corner={corner:?} order={order:?}: min_dot={minimum_dot}"
            );

            let oriented = oriented_solid_volume(&topo, solid, 0.01).unwrap();
            let coarse = solid_volume(&topo, solid, 0.01).unwrap();
            assert!(
                oriented.is_finite() && oriented > 0.0,
                "corner={corner:?} order={order:?}"
            );
            assert!(
                coarse.is_finite() && coarse > 0.0,
                "corner={corner:?} order={order:?}"
            );
            assert!(
                (oriented - coarse).abs() < 5.0,
                "corner={corner:?} order={order:?}: oriented={oriented} vs coarse={coarse}"
            );
            cases += 1;
        }
    }
    eprintln!("N342 order/corner sweep: {cases} cases valid, all 6 orders x 8 corners");
    assert_eq!(cases, 48);
}
