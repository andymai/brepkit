//! N433: all four top edges of the 1 inch cube build at the exact limit
//! `r = S/2` (the pillow), and across the whole admissible range `(0, S/2]`.
//!
//! The bulk `fillet_v2` route used to fail at the exact limit inside the
//! miter-trim construction: each shared-face (top) contact is trimmed once
//! from each of its two corners' ends, so at `r = S/2` the *second* trim
//! finds exactly `radius` of contact left, and trimming a curve by its whole
//! remaining parameter range produces a single-control-point NURBS the
//! builder rejects (`invalid knot vector: expected 3 knots, got 2`). The
//! contact has to collapse instead: removed from its own stripe's wire
//! (leaving that stripe bounded by its bottom contact and its two corner
//! creases) and restricted in place to the corners' shared meeting vertex in
//! the shared face's wire, whose then point-only boundary the result-shell
//! assembly drops. All four corners' meeting vertices coincide at the limit
//! (`(S/2, S/2, S)`), so the four stripes meet at one point and the solid is
//! closed with 9 faces, 16 edges, 9 vertices and Euler characteristic 2.
//!
//! The exact volume of the four-edge fillet on a cube `[0,S]^3` at radius
//! `r` is the cube less four single-fillet removals plus the exact overlap
//! corrections (four corners, each shared by exactly two removals):
//!
//! ```text
//! V(r) = S^3 - 4 S r^2 (1 - pi/4) + 4 (5/3 - pi/2) r^3
//! V(S/2) = (5/6) S^3 = 13655.887 mm^3
//! ```
//!
//! At the limit each of the four stripe patches is the quarter cylinder
//! `(pi/2) r S` trimmed at both ends by `r^2 (pi/2 - 1)`, which reduces to
//! exactly `S^2 / 2 = 322.58 mm^2`; all four must contribute positively.
//!
//! These tests pin that contract at the limit (both edge-selection orders),
//! the per-stripe patch area and its positive orientation contribution, a
//! sweep across `(0, S/2]` with the `~12.446`-to-`S/2` band re-checked, and
//! the `r = S/2 + 1e-6` control, which must still refuse with the source
//! preserved.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces, solid_vertices};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::validation::validate_shell_closed;

/// 1 inch in millimetres: `r = S/2` is exactly the four-edge limit.
const S: f64 = 25.4;

/// The limit radius.
const HALF: f64 = S / 2.0;

/// Half a micron: the "just above the limit" control radius increment.
const ABOVE: f64 = 1e-6;

/// Mesh deflection for the patch-area and mesh-volume measurements.
const DEFLECTION: f64 = 0.01;

/// The four top-face corners, in wire order.
const CORNERS: [Point3; 4] = [
    Point3::new(0.0, 0.0, S),
    Point3::new(S, 0.0, S),
    Point3::new(S, S, S),
    Point3::new(0.0, S, S),
];

fn inch_cube(topo: &mut Topology) -> SolidId {
    let wire = make_polygon_wire(
        topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(S, 0.0, 0.0),
            Point3::new(S, S, 0.0),
            Point3::new(0.0, S, 0.0),
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
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), S).unwrap()
}

/// The unique source edge with these endpoints, in either direction.
fn edge_at(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let tolerance = Tolerance::new().linear;
    let matches: Vec<EdgeId> = solid_edges(topo, solid)
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
    assert_eq!(matches.len(), 1, "expected one edge {a:?}-{b:?}");
    matches[0]
}

/// All four top edges of the cube, in wire order (or reversed).
fn top_edges(topo: &Topology, solid: SolidId, reversed: bool) -> Vec<EdgeId> {
    let mut edges = vec![
        edge_at(topo, solid, CORNERS[0], CORNERS[1]),
        edge_at(topo, solid, CORNERS[1], CORNERS[2]),
        edge_at(topo, solid, CORNERS[2], CORNERS[3]),
        edge_at(topo, solid, CORNERS[3], CORNERS[0]),
    ];
    if reversed {
        edges.reverse();
    }
    edges
}

/// Every datum of the source solid, for the refuse-without-mutation controls.
fn source_snapshot(topo: &Topology, solid: SolidId) -> Vec<String> {
    let mut rows = vec![format!("{:?}", topo.solid(solid).unwrap())];
    let data = topo.solid(solid).unwrap();
    for shell in std::iter::once(data.outer_shell()).chain(data.inner_shells().iter().copied()) {
        rows.push(format!("{shell:?}:{:?}", topo.shell(shell).unwrap()));
    }
    for face in solid_faces(topo, solid).unwrap() {
        let face_data = topo.face(face).unwrap();
        rows.push(format!("{face:?}:{face_data:?}"));
        for wire in
            std::iter::once(face_data.outer_wire()).chain(face_data.inner_wires().iter().copied())
        {
            rows.push(format!("{wire:?}:{:?}", topo.wire(wire).unwrap()));
        }
    }
    for edge in solid_edges(topo, solid).unwrap() {
        let edge_data = topo.edge(edge).unwrap();
        rows.push(format!("{edge:?}:{edge_data:?}"));
        for vertex in [edge_data.start(), edge_data.end()] {
            rows.push(format!("{vertex:?}:{:?}", topo.vertex(vertex).unwrap()));
        }
    }
    rows.sort();
    rows
}

/// The exact volume of the four-edge (pillow) fillet at radius `r` on a
/// `[0,S]^3` cube.
fn closed_form_volume(r: f64) -> f64 {
    let removal = S * r * r * (1.0 - std::f64::consts::FRAC_PI_4);
    let corner_overlap = (5.0 / 3.0 - std::f64::consts::FRAC_PI_2) * r * r * r;
    S * S * S - (4.0 * removal - 4.0 * corner_overlap)
}

fn assert_valid_closed_oriented(topo: &Topology, solid: SolidId, case: &str) {
    let shell = topo.solid(solid).unwrap().outer_shell();
    let closed = validate_shell_closed(topo.shell(shell).unwrap(), topo);
    assert!(closed.is_ok(), "{case}: shell is not closed: {closed:?}");
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{case}: {report:?}");
    let volume = solid_volume(topo, solid, DEFLECTION).unwrap();
    assert!(volume > 0.0, "{case}: not outward-oriented ({volume})");
}

/// Per-face signed volume contribution and triangle area, in `solid_faces`
/// order, measured off the watertight grouped mesh the display and export
/// routes use.
fn face_measurements(topo: &Topology, solid: SolidId) -> Vec<(f64, f64)> {
    let faces = solid_faces(topo, solid).unwrap();
    let (mesh, offsets) =
        tessellate_solid_grouped_with_tolerance(topo, solid, DEFLECTION, f64::to_radians(5.0))
            .unwrap();
    (0..faces.len())
        .map(|index| {
            let mut contribution = 0.0;
            let mut area = 0.0;
            for triangle in (offsets[index] as usize / 3)..(offsets[index + 1] as usize / 3) {
                let a = mesh.positions[mesh.indices[triangle * 3] as usize];
                let b = mesh.positions[mesh.indices[triangle * 3 + 1] as usize];
                let c = mesh.positions[mesh.indices[triangle * 3 + 2] as usize];
                let (a, b, c) = (
                    Vec3::new(a.x(), a.y(), a.z()),
                    Vec3::new(b.x(), b.y(), b.z()),
                    Vec3::new(c.x(), c.y(), c.z()),
                );
                contribution += a.dot(b.cross(c)) / 6.0;
                area += (b - a).cross(c - a).length() / 2.0;
            }
            (contribution, area)
        })
        .collect()
}

/// Build the four-edge fillet at `radius` and return the result solid.
fn build_pillow(topo: &mut Topology, radius: f64, reversed: bool) -> SolidId {
    let source = inch_cube(topo);
    let edges = top_edges(topo, source, reversed);
    let result = fillet_v2(topo, source, &edges, radius)
        .unwrap_or_else(|error| panic!("r={radius}: the four-edge fillet must build: {error:?}"));
    assert_eq!(
        result.succeeded.len(),
        4,
        "r={radius}: {} of 4 edges succeeded, failed {}",
        result.succeeded.len(),
        result.failed.len()
    );
    assert!(!result.is_partial, "r={radius}: partial result");
    result.solid
}

/// The limit builds in either selection order, as one valid, closed, oriented
/// solid of the exact limit topology: 9 faces (bottom, four sides, four
/// stripes — the zero-area top face is legitimately gone), 16 edges each used
/// exactly twice and 9 vertices, all four stripes meeting at one shared
/// meeting vertex with no non-manifold seam.
#[test]
fn four_top_edges_at_the_limit_build_on_oracle_in_both_orders() {
    let mut seam_vertices = Vec::new();
    for reversed in [false, true] {
        let mut topo = Topology::new();
        let solid = build_pillow(&mut topo, HALF, reversed);
        let case = format!("r=S/2 reversed={reversed}");
        assert_valid_closed_oriented(&topo, solid, &case);

        let faces = solid_faces(&topo, solid).unwrap();
        let edges = solid_edges(&topo, solid).unwrap();
        let vertices = solid_vertices(&topo, solid).unwrap();
        assert_eq!(
            (faces.len(), edges.len(), vertices.len()),
            (9, 16, 9),
            "{case}: the pillow's limit inventory"
        );

        // Every edge is used by exactly two faces, and each of the four
        // crease ellipses is shared by exactly two stripe cylinders that
        // meet at the one shared meeting vertex: the vanished top face
        // leaves no free or non-manifold seam behind.
        let mut uses = std::collections::HashMap::<EdgeId, usize>::new();
        for &face in &faces {
            let data = topo.face(face).unwrap();
            for oe in topo.wire(data.outer_wire()).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
        assert!(
            uses.values().all(|&count| count == 2),
            "{case}: every edge must be used exactly twice, got {uses:?}"
        );
        let mut creases = 0;
        let mut meeting_vertices = Vec::new();
        for &edge in &edges {
            let data = topo.edge(edge).unwrap();
            if matches!(data.curve(), EdgeCurve::Ellipse(_)) {
                creases += 1;
                let owners: Vec<_> = faces
                    .iter()
                    .copied()
                    .filter(|&face| {
                        topo.wire(topo.face(face).unwrap().outer_wire())
                            .unwrap()
                            .edges()
                            .iter()
                            .any(|oe| oe.edge() == edge)
                    })
                    .collect();
                assert_eq!(owners.len(), 2, "{case}: crease {edge:?} owners");
                assert!(
                    owners.iter().all(|&face| matches!(
                        topo.face(face).unwrap().surface(),
                        FaceSurface::Cylinder(_)
                    )),
                    "{case}: crease {edge:?} must join two stripe cylinders"
                );
                for vertex in [data.start(), data.end()] {
                    let point = topo.vertex(vertex).unwrap().point();
                    if (point - Point3::new(HALF, HALF, S)).length() < Tolerance::new().linear {
                        meeting_vertices.push(vertex);
                    }
                }
            }
        }
        assert_eq!(creases, 4, "{case}: four miter creases");
        assert_eq!(
            meeting_vertices.len(),
            4,
            "{case}: every crease must end at the meeting point"
        );
        assert!(
            meeting_vertices.iter().all(|v| *v == meeting_vertices[0]),
            "{case}: all four creases must share one meeting vertex (no seam)"
        );
        seam_vertices.push(topo.vertex(meeting_vertices[0]).unwrap().point());

        // Each stripe patch is the exact trimmed quarter cylinder
        // `(pi/2) r S - 2 r^2 (pi/2 - 1) = S^2/2` and contributes positively.
        let patch = S * S / 2.0;
        let measurements = face_measurements(&topo, solid);
        let mut stripe_areas = Vec::new();
        for (index, &face) in faces.iter().enumerate() {
            if matches!(topo.face(face).unwrap().surface(), FaceSurface::Cylinder(_)) {
                let (contribution, area) = measurements[index];
                assert!(
                    contribution > 0.0,
                    "{case}: stripe must contribute positively, got {contribution}"
                );
                stripe_areas.push(area);
            }
        }
        assert_eq!(stripe_areas.len(), 4, "{case}: four stripes");
        for area in &stripe_areas {
            assert!(
                (area - patch).abs() <= patch * 1e-3,
                "{case}: stripe patch {area} is not S^2/2 = {patch}"
            );
        }

        // The limit volume is the closed form.
        let exact = 5.0 / 6.0 * S * S * S;
        let volume = solid_volume(&topo, solid, DEFLECTION).unwrap();
        assert!(
            (volume - exact).abs() <= exact * 5e-4,
            "{case}: volume {volume} is not the closed form {exact}"
        );
    }
    assert_eq!(
        seam_vertices.len(),
        2,
        "both orders produce a meeting vertex"
    );
    assert!(
        (seam_vertices[0] - seam_vertices[1]).length() < Tolerance::new().linear,
        "both selection orders must meet at the same point: {:?} vs {:?}",
        seam_vertices[0],
        seam_vertices[1]
    );
}

/// The `~12.446`-to-`S/2` band N431 flagged and the rest of `(0, S/2]`:
/// every radius builds valid/closed/oriented, and its volume sits on the
/// closed form. No point in range may refuse or come back `Ok`-but-open.
#[test]
fn the_admissible_range_never_refuses_or_leaves_an_open_shell() {
    // 0.5 mm steps across the range, 0.005 mm steps across the flagged band
    // (N431's `~12.446` transition sits inside it), the measured transition
    // points themselves, and the case-7 control radius.
    let mut radii: Vec<f64> = (1..=25).map(|i| 0.5 * f64::from(i)).collect();
    radii.extend((0..=40).map(|i| 12.45 + 0.005 * f64::from(i)));
    radii.extend([
        12.4,
        12.425,
        12.445,
        12.446,
        12.446_000_07,
        12.446_000_1,
        12.5,
        12.55,
        12.6,
        12.65,
        12.69,
        12.699,
        12.699_9,
        12.699_974_6,
    ]);
    radii.extend([HALF - 1e-6, HALF - 1e-9, HALF]);
    radii.sort_by(f64::total_cmp);
    radii.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    assert!(radii.len() >= 60, "sweep must cover the range finely");
    for radius in radii {
        let mut topo = Topology::new();
        let source = inch_cube(&mut topo);
        let edges = top_edges(&topo, source, false);
        let case = format!("r={radius}");
        let result = fillet_v2(&mut topo, source, &edges, radius)
            .unwrap_or_else(|error| panic!("{case}: must build, refused: {error:?}"));
        assert_eq!(result.succeeded.len(), 4, "{case}: all four edges");
        assert!(!result.is_partial, "{case}: no partial result");
        assert_valid_closed_oriented(&topo, result.solid, &case);
        let volume = solid_volume(&topo, result.solid, DEFLECTION).unwrap();
        let exact = closed_form_volume(radius);
        let delta = (volume - exact).abs() / exact;
        assert!(
            delta <= 5e-4,
            "{case}: volume {volume} is {delta} off the closed form {exact}"
        );
    }
}

/// Past the limit the bulk route still refuses, before any mutation, naming
/// the adjacent edge it does not fit.
#[test]
fn just_above_the_limit_refuses_with_the_source_preserved() {
    let mut topo = Topology::new();
    let source = inch_cube(&mut topo);
    let edges = top_edges(&topo, source, false);
    let before = source_snapshot(&topo, source);
    let error = fillet_v2(&mut topo, source, &edges, HALF + ABOVE)
        .err()
        .expect("r = S/2 + 1e-6 must refuse");
    let message = format!("{error:?}");
    assert!(
        message.contains("RadiusTooLarge"),
        "refusal must name the overshoot: {message}"
    );
    assert!(message.contains("max_radius: 12.7"), "{message}");
    assert_eq!(before, source_snapshot(&topo, source));
}

/// The limit's mesh volume agrees with its analytic volume as closely as the
/// just-below case does (`solid_volume` clamps its own deflection; the
/// display mesh is coarser and must stay within 0.1% of the exact closed
/// form at the limit as just below).
#[test]
fn the_limit_mesh_volume_agrees_with_its_analytic_volume() {
    let mut topo = Topology::new();
    let solid = build_pillow(&mut topo, HALF, false);
    let exact = 5.0 / 6.0 * S * S * S;
    let computed = solid_volume(&topo, solid, DEFLECTION).unwrap();
    let limit_delta = (computed - exact).abs() / exact;
    assert!(
        limit_delta <= 1e-3,
        "limit volume {computed} is {limit_delta} off {exact}"
    );

    let mut below = Topology::new();
    let solid_below = build_pillow(&mut below, HALF - 1e-6, false);
    let exact_below = closed_form_volume(HALF - 1e-6);
    let computed_below = solid_volume(&below, solid_below, DEFLECTION).unwrap();
    let below_delta = (computed_below - exact_below).abs() / exact_below;
    assert!(
        below_delta <= 1e-3,
        "just-below volume {computed_below} is {below_delta} off {exact_below}"
    );
    assert!(
        limit_delta <= below_delta + 2e-4,
        "the limit must be as close to its closed form as just below is: \
         {limit_delta} vs {below_delta}"
    );
}

/// The limit stripe patch is bounded by the collapsed three-edge wire — one
/// NURBS side contact and the two exact miter creases of its own two ends —
/// each crease shared by exactly the two stripe cylinders meeting there, and
/// the solid's only planar faces are the bottom and the four sides: the
/// zero-area top face is absent rather than left as a degenerate seam.
#[test]
fn the_limit_stripe_wire_is_the_collapsed_three_edge_loop() {
    let mut topo = Topology::new();
    let solid = build_pillow(&mut topo, HALF, false);
    let faces = solid_faces(&topo, solid).unwrap();
    let mut planes = 0;
    let mut stripes = 0;
    for &face in &faces {
        let data = topo.face(face).unwrap();
        let wire = topo.wire(data.outer_wire()).unwrap();
        match data.surface() {
            FaceSurface::Plane { .. } => planes += 1,
            FaceSurface::Cylinder(_) => {
                stripes += 1;
                let (mut nurbs, mut ellipses, mut others) = (0, 0, 0);
                for oe in wire.edges() {
                    match topo.edge(oe.edge()).unwrap().curve() {
                        EdgeCurve::NurbsCurve(_) => nurbs += 1,
                        EdgeCurve::Ellipse(_) => ellipses += 1,
                        _ => others += 1,
                    }
                }
                assert_eq!(
                    (wire.edges().len(), nurbs, ellipses, others),
                    (3, 1, 2, 0),
                    "stripe {face:?} must be the collapsed 3-edge patch"
                );
            }
            other => panic!("unexpected surface {other:?}"),
        }
    }
    assert_eq!((planes, stripes), (5, 4), "five planes and four stripes");
}
