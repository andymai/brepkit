//! N430: exact-limit fillet radii on the 1 inch (25.4 mm) cube.
//!
//! The degenerate-face guard used to refuse `r = S` outright because the blend
//! legitimately consumes a whole source face there, leaving a zero-area
//! *planar* remnant — indistinguishable from the closed-circular-rim collapse
//! by edge topology, but not by the face's own supporting surface. The guard
//! now excludes the planar remnant and requires the remainder to stay closed;
//! a zero-area *non-planar* (generated blend) face keeps the original refusal.
//! N429 measured which of the three reported limits this can close:
//!
//! | case | limit | outcome |
//! |---|---|---|
//! | single edge | `r = S` | builds: 5 faces, valid, closed, oriented |
//! | two adjacent top edges | `r = S` | refuses: the strips terminate where the face vanished |
//! | four top edges | `r = S/2` | refuses: the bulk setback equality, and per-edge the vanished remnant |
//!
//! Every case pins its refusal wording: the vanishing-remnant reason must not
//! borrow the closed-circular-edge claim, which was never true of these
//! requests. `cross_one_row_fillet_inmem`'s `rolling_reference_*` pair pins the
//! other side of the discriminator (the genuine rim collapse, byte-identical).
#![allow(clippy::unwrap_used, clippy::expect_used, deprecated)]

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::fillet::fillet_rolling_ball;
use brepkit_operations::measure::oriented_solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::validation::validate_shell_closed;

/// 1 inch in millimetres — the fixture the three exact-limit refusals were
/// reported on: `r = S` is exactly the single- and two-edge limit and
/// `r = S/2` exactly the four-edge (pillow) limit.
const S: f64 = 25.4;

/// Half a micron: the "just above the limit" control radius increment.
const ABOVE: f64 = 1e-6;

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
    brepkit_operations::extrude::extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), S).unwrap()
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

/// The top edges `[0, count)` in wire order.
fn top_edges(topo: &Topology, solid: SolidId, count: usize) -> Vec<EdgeId> {
    (0..count)
        .map(|i| edge_at(topo, solid, CORNERS[i], CORNERS[(i + 1) % 4]))
        .collect()
}

/// The longest edge still lying on the source segment `a..b` (a previous chain
/// step re-trims the neighbouring top edges, so the exact endpoint match above
/// no longer holds).
fn longest_edge_on_segment(
    topo: &Topology,
    solid: SolidId,
    a: Point3,
    b: Point3,
) -> Option<EdgeId> {
    let tol = 1e-6;
    let direction = (b - a) * (1.0 / (b - a).length());
    let mut best: Option<(EdgeId, f64)> = None;
    for id in solid_edges(topo, solid).unwrap() {
        let edge = topo.edge(id).unwrap();
        let (Ok(start), Ok(end)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            continue;
        };
        let (p, q) = (start.point(), end.point());
        let on_line = |point: Point3| -> bool {
            ((point - a) - direction * (point - a).dot(direction)).length() < tol
        };
        if on_line(p) && on_line(q) {
            let length = (q - p).length();
            if length > tol && best.is_none_or(|(_, best_len)| length > best_len) {
                best = Some((id, length));
            }
        }
    }
    best.map(|(id, _)| id)
}

fn assert_valid_closed_oriented(topo: &Topology, solid: SolidId, case: &str) {
    let shell = topo.solid(solid).unwrap().outer_shell();
    let closed = validate_shell_closed(topo.shell(shell).unwrap(), topo);
    assert!(closed.is_ok(), "{case}: shell is not closed: {closed:?}");
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{case}: {report:?}");
    let volume = oriented_solid_volume(topo, solid, 0.005).unwrap();
    assert!(volume > 0.0, "{case}: not outward-oriented ({volume})");
}

#[test]
fn single_edge_at_r_equals_side_builds_valid_closed_oriented() {
    let mut topo = Topology::new();
    let solid = inch_cube(&mut topo);
    let edge = edge_at(
        &topo,
        solid,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, S),
    );
    let result = fillet_rolling_ball(&mut topo, solid, &[edge], S)
        .expect("the single-edge exact limit must build");

    // The top face and the edge-side face vanish; the blend strip and the four
    // remaining source faces are returned.
    assert_eq!(solid_faces(&topo, result).unwrap().len(), 5);
    assert_valid_closed_oriented(&topo, result, "single edge r=S");

    // Analytic exact-limit volume: the extrusion of the region inside the
    // fillet arc, `pi/4 * S^3` at this radius.
    let volume = oriented_solid_volume(&topo, result, 0.005).unwrap();
    let exact = std::f64::consts::PI / 4.0 * S * S * S;
    assert!(
        (volume - exact).abs() < 5.0,
        "volume {volume} is not the exact-limit volume {exact}"
    );
}

#[test]
fn single_edge_just_above_r_equals_side_still_refuses() {
    let mut topo = Topology::new();
    let solid = inch_cube(&mut topo);
    let edge = edge_at(
        &topo,
        solid,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, S),
    );
    let error = fillet_rolling_ball(&mut topo, solid, &[edge], S + ABOVE)
        .expect_err("past the limit the radius no longer fits the adjacent edge");
    let message = error.to_string();
    assert!(
        message.contains("exceeds adjacent edge length"),
        "unexpected just-above-limit rejection: {message}"
    );
    assert!(!message.contains("vanishing planar remnant"), "{message}");
}

#[test]
fn two_adjacent_top_edges_at_r_equals_side_refuse_honestly() {
    let mut topo = Topology::new();
    let solid = inch_cube(&mut topo);
    let edges = top_edges(&topo, solid, 2);
    let error = fillet_rolling_ball(&mut topo, solid, &edges, S)
        .expect_err("the two-edge exact limit is not constructible and must refuse");
    let message = error.to_string();
    assert!(message.contains("vanishing planar remnant"), "{message}");
    assert!(message.contains("open shell"), "{message}");
    assert!(!message.contains("closed circular edges"), "{message}");
    assert!(!message.contains("combined setback"), "{message}");
}

#[test]
fn two_adjacent_top_edges_just_above_r_equals_side_still_refuse() {
    let mut topo = Topology::new();
    let solid = inch_cube(&mut topo);
    let edges = top_edges(&topo, solid, 2);
    let error = fillet_rolling_ball(&mut topo, solid, &edges, S + ABOVE)
        .expect_err("past the two-edge limit the radius no longer fits the adjacent edge");
    let message = error.to_string();
    assert!(
        message.contains("exceeds adjacent edge length"),
        "unexpected just-above-limit rejection: {message}"
    );
}

#[test]
fn four_top_edges_at_half_side_refuse_honestly() {
    let mut topo = Topology::new();
    let solid = inch_cube(&mut topo);
    let edges = top_edges(&topo, solid, 4);

    // The bulk call is refused by the combined-setback equality first (the
    // pillow's true limit); the guard is never reached here.
    let error = fillet_rolling_ball(&mut topo, solid, &edges, S / 2.0)
        .expect_err("the four-edge pillow limit must refuse");
    let message = error.to_string();
    assert!(message.contains("combined setback"), "{message}");

    // `dispatch::blend_batch` re-resolves and fillets one target at a time
    // when the bulk call fails — the route that used to surface the
    // closed-circular-edge claim for this case. Its per-edge steps must refuse
    // with the vanishing-remnant reason instead.
    let mut topo = Topology::new();
    let mut current = inch_cube(&mut topo);
    let mut chain_error: Option<String> = None;
    for step in 0..4 {
        let Some(edge) =
            longest_edge_on_segment(&topo, current, CORNERS[step], CORNERS[(step + 1) % 4])
        else {
            break;
        };
        match fillet_rolling_ball(&mut topo, current, &[edge], S / 2.0) {
            Ok(next) => current = next,
            Err(error) => {
                chain_error = Some(error.to_string());
                break;
            }
        }
    }
    let chain_message = chain_error.expect("the pillow-limit chain must refuse");
    assert!(
        chain_message.contains("vanishing planar remnant"),
        "{chain_message}"
    );
    assert!(
        !chain_message.contains("closed circular edges"),
        "{chain_message}"
    );
}

#[test]
fn four_top_edges_just_above_half_side_still_refuse() {
    let mut topo = Topology::new();
    let solid = inch_cube(&mut topo);
    let edges = top_edges(&topo, solid, 4);
    let error = fillet_rolling_ball(&mut topo, solid, &edges, S / 2.0 + ABOVE)
        .expect_err("past the pillow limit the setbacks exceed the edge length");
    let message = error.to_string();
    assert!(message.contains("combined setback"), "{message}");
}
