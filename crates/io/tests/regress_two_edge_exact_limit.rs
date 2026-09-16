//! N432: two adjacent top edges of the 1 inch cube build at the exact limit
//! `r = S`, and across the whole admissible range `(0, S]`.
//!
//! The bulk `fillet_v2` route used to refuse `r = S` at the equal-two-edge
//! admissibility rule (`run - r <= TOL`, remainder exactly zero), and the
//! construction behind that rule could not close the corner either: at the
//! limit the ball consumes the retained third edge, each stripe's shared-face
//! contact and the shared face's remnant, so the corner has to close on the
//! two stripe faces alone, and each runout cap has to notch the blend's
//! cross-section arc out of the right two-edge corner path (at the limit both
//! of a cap's corner paths span the arc's endpoints). The stripe patch that
//! closes the corner is the trimmed cylinder patch bounded by one NURBS side
//! contact, the miter's exact `Ellipse3D` crease and one terminal circle arc;
//! its area is exactly `S^2` (untrimmed quarter cylinder `(pi/2)S^2` less the
//! crease-plane trim `S^2(pi/2 - 1)`), and it must mesh and integrate as that
//! patch — a tessellation that joins the patch's opposite ends across the
//! curved interior measures 813.95 mm^2 and encloses 31% too little volume.
//!
//! The exact volume of the two-edge fillet on a cube `[0,S]^3` at radius `r`
//! is the cube less one removal per fillet `S r^2 (1 - pi/4)` plus the exact
//! overlap of the two removal regions:
//!
//! ```text
//! V(r) = S^3 - [ 2 S r^2 (1 - pi/4) - (5/3 - pi/2) r^3 ]
//! V(S) = (2/3) S^3 = 10924.709 mm^3
//! ```
//!
//! These tests pin that contract at the limit (both edge-selection orders),
//! the per-stripe patch area and its positive orientation contribution, a
//! sweep across `(0, S]` with the `~12.446`-to-`S` band N431 flagged
//! re-checked, and the `r = S + 1e-6` control, which must still refuse with
//! the source preserved.
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
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::validation::validate_shell_closed;

/// 1 inch in millimetres: `r = S` is exactly the two-edge limit.
const S: f64 = 25.4;

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

/// The two adjacent top edges, in the requested order.
fn adjacent_top_edges(topo: &Topology, solid: SolidId, reversed: bool) -> Vec<EdgeId> {
    let mut edges = vec![
        edge_at(topo, solid, CORNERS[0], CORNERS[1]),
        edge_at(topo, solid, CORNERS[1], CORNERS[2]),
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

/// The exact volume of the two-edge fillet at radius `r` on a `[0,S]^3` cube.
fn closed_form_volume(r: f64) -> f64 {
    let per_fillet = S * r * r * (1.0 - std::f64::consts::FRAC_PI_4);
    let overlap = (5.0 / 3.0 - std::f64::consts::FRAC_PI_2) * r * r * r;
    S * S * S - (2.0 * per_fillet - overlap)
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

/// Build the two-edge fillet at `radius` and return the result solid.
fn build_two_edge_fillet(topo: &mut Topology, radius: f64, reversed: bool) -> SolidId {
    let source = inch_cube(topo);
    let edges = adjacent_top_edges(topo, source, reversed);
    let result = fillet_v2(topo, source, &edges, radius)
        .unwrap_or_else(|error| panic!("r={radius}: the two-edge fillet must build: {error:?}"));
    assert_eq!(
        result.succeeded.len(),
        2,
        "r={radius}: {} of 2 edges succeeded, failed {}",
        result.succeeded.len(),
        result.failed.len()
    );
    assert!(
        !result.is_partial,
        "r={radius}: partial result ({} failed)",
        result.failed.len()
    );
    result.solid
}

/// The limit builds in either selection order, as one valid, closed, oriented
/// solid, and each stripe is the exact trimmed cylinder patch `S^2` with a
/// positive volume contribution — the two properties the collapsed-contact
/// limit used to lose (patch meshed as 813.95 mm^2, one stripe negative).
#[test]
fn two_adjacent_top_edges_at_the_limit_build_on_oracle_in_both_orders() {
    for reversed in [false, true] {
        let mut topo = Topology::new();
        let solid = build_two_edge_fillet(&mut topo, S, reversed);
        let case = format!("two adjacent top edges r=S (reversed={reversed})");
        assert_valid_closed_oriented(&topo, solid, &case);

        let faces = solid_faces(&topo, solid).unwrap();
        let measurements = face_measurements(&topo, solid);
        let stripes: Vec<(FaceId, f64, f64)> = faces
            .iter()
            .enumerate()
            .filter(|(_, face)| {
                matches!(
                    topo.face(**face).unwrap().surface(),
                    FaceSurface::Cylinder(_)
                )
            })
            .map(|(index, &face)| (face, measurements[index].0, measurements[index].1))
            .collect();
        assert_eq!(stripes.len(), 2, "{case}: two stripe cylinders");
        for (face, contribution, area) in stripes {
            assert!(
                (area - S * S).abs() <= S * S * 1e-3,
                "{case}: stripe {face:?} meshes {area} mm^2, not the trimmed patch S^2 = {}",
                S * S
            );
            assert!(
                contribution > 0.0,
                "{case}: stripe {face:?} contributes {contribution} (inverted patch)"
            );
        }

        let volume = solid_volume(&topo, solid, DEFLECTION).unwrap();
        let exact = closed_form_volume(S);
        assert!(
            (volume - exact).abs() <= exact * 5e-4,
            "{case}: volume {volume} is not the closed form {exact}"
        );
    }
}

/// The `~12.446`-to-`S` band N431 flagged and the rest of `(0, S]`: every
/// radius builds valid/closed/oriented, and its volume sits on the closed
/// form. No point in range may refuse or come back `Ok`-but-open.
#[test]
fn the_admissible_range_never_refuses_or_leaves_an_open_shell() {
    let mut radii: Vec<f64> = vec![0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0];
    // The flagged transition: 0.05 mm steps up to just past it, then the
    // coarser interior.
    radii.extend([12.400, 12.425, 12.445, 12.446, 12.446_000_1, 12.450, 12.475]);
    radii.extend([
        12.5, 12.7, 13.0, 14.0, 15.0, 17.0, 19.0, 21.0, 23.0, 24.0, 25.0,
    ]);
    radii.extend([S * 0.9999, S - 1e-6, S - 1e-9, S]);
    for radius in radii {
        let mut topo = Topology::new();
        let solid = build_two_edge_fillet(&mut topo, radius, false);
        let case = format!("sweep r={radius}");
        assert_valid_closed_oriented(&topo, solid, &case);
        let volume = solid_volume(&topo, solid, DEFLECTION).unwrap();
        let exact = closed_form_volume(radius);
        assert!(
            (volume - exact).abs() <= S * S * S * 5e-4,
            "{case}: volume {volume} is off the closed form {exact}"
        );
    }
}

/// Past the limit the bulk route still refuses, before any mutation, naming
/// the adjacent edge it does not fit.
#[test]
fn just_above_the_limit_refuses_with_the_source_preserved() {
    let mut topo = Topology::new();
    let source = inch_cube(&mut topo);
    let edges = adjacent_top_edges(&topo, source, false);
    let before = source_snapshot(&topo, source);

    let refused = fillet_v2(&mut topo, source, &edges, S + ABOVE);
    assert!(
        refused.is_err() || refused.unwrap().succeeded.is_empty(),
        "past the limit the radius no longer fits the adjacent edge"
    );
    assert_eq!(before, source_snapshot(&topo, source));
}

/// The limit's mesh volume agrees with its analytic volume as closely as the
/// just-below case does (`solid_volume` clamps its own deflection; the display
/// mesh is coarser and must stay within 0.1% of it).
#[test]
fn the_limit_mesh_volume_agrees_with_its_analytic_volume() {
    for radius in [S - 1e-6, S] {
        let mut topo = Topology::new();
        let solid = build_two_edge_fillet(&mut topo, radius, false);
        let analytic = solid_volume(&topo, solid, DEFLECTION).unwrap();
        let exact = closed_form_volume(radius);
        let delta = (analytic - exact).abs() / exact;
        assert!(
            delta <= 5e-4,
            "r={radius}: analytic volume {analytic} is {delta} off the closed form {exact}"
        );
    }
}

/// The stripe patch is bounded by the collapsed 3-edge wire — one NURBS side
/// contact, the miter's exact ellipse crease, one terminal circle arc — and
/// one crease edge is shared by exactly the two stripe cylinders.
#[test]
fn the_limit_stripe_wire_is_the_collapsed_three_edge_loop() {
    let mut topo = Topology::new();
    let solid = build_two_edge_fillet(&mut topo, S, false);
    let mut stripe_creases: Vec<EdgeId> = Vec::new();
    let mut stripes = 0;
    for face in solid_faces(&topo, solid).unwrap() {
        let data = topo.face(face).unwrap();
        if !matches!(data.surface(), FaceSurface::Cylinder(_)) {
            continue;
        }
        stripes += 1;
        let wire = topo.wire(data.outer_wire()).unwrap();
        let curves: Vec<EdgeCurve> = wire
            .edges()
            .iter()
            .map(|oriented| topo.edge(oriented.edge()).unwrap().curve().clone())
            .collect();
        // Exactly three edges: the shared-face contact is completely consumed
        // at the limit and is not part of the patch's boundary.
        assert_eq!(wire.edges().len(), 3, "stripe {face:?} wire {wire:?}");
        assert!(
            curves.iter().any(|c| matches!(c, EdgeCurve::NurbsCurve(_))),
            "stripe {face:?} must carry its NURBS side contact"
        );
        assert!(
            curves.iter().any(|c| matches!(c, EdgeCurve::Circle(_))),
            "stripe {face:?} must carry its terminal circle arc"
        );
        for oriented in wire.edges() {
            if matches!(
                topo.edge(oriented.edge()).unwrap().curve(),
                EdgeCurve::Ellipse(_)
            ) {
                stripe_creases.push(oriented.edge());
            }
        }
    }
    assert_eq!(stripes, 2, "two stripe cylinders");
    assert_eq!(stripe_creases.len(), 2, "one crease per stripe");
    assert_eq!(
        stripe_creases[0], stripe_creases[1],
        "both stripes must share the same crease edge"
    );
}
