//! Rounds of a D prism's vertical edges, alone or with its top's outline.
//!
//! The part is a D: the profile `x = 0`, `y = -B`, `y = B` closed by an arc
//! of a circle of radius `R` about `(C, 0)` bulging out to `x > X0`,
//! extruded `H` up `z`. Its two vertical edges at `x = X0` are where a flat
//! side meets the arc's cylinder along a ruling.
//!
//! A round of such an edge is a cylinder about a line parallel to the
//! axis, a radius from the side and `R - r` from the axis: it was walked as
//! a fitted band, which a corner could not set back. Its end on the top or
//! bottom face is an arc of that face's outline round the wall, which the
//! cap notch took only for straight corner edges, so the cap was left whole
//! and the mesh open where the band ended.
//!
//! With the top's outline rounded too, each top corner is a ball: at the
//! flat corners the three planes fix its centre; at the arc corners the
//! round along the top arc is a torus, and the ball sits where its tube's
//! centre circle meets the vertical round's axis. The corners were not set
//! back (a torus stripe had no straight centreline), and the result failed.
//!
//! The volumes are closed forms: rounding every vertical edge leaves the
//! profile's inward offset by `r` grown back by `r`, and rounding the top's
//! outline as well takes off the quarter rounds a rolling ball sweeps along
//! that outline.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const B: f64 = 5.0;
const R: f64 = 8.0;
const X0: f64 = 10.0;
const H: f64 = 6.0;

/// The arc's centre on the x axis.
fn centre() -> f64 {
    X0 - (R * R - B * B).sqrt()
}

/// The D, its arc counter-clockwise about `+z` from `(X0, -B)` to `(X0, B)`.
fn d_prism(topo: &mut Topology) -> SolidId {
    let corners = [(X0, -B), (X0, B), (0.0, B), (0.0, -B)];
    let ids: Vec<_> = corners
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
        .collect();
    let arc = Circle3D::new(Point3::new(centre(), 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), R).unwrap();
    let edges: Vec<_> = (0..ids.len())
        .map(|i| {
            let curve = if i == 0 {
                EdgeCurve::Circle(arc.clone())
            } else {
                EdgeCurve::Line
            };
            let edge = Edge::new(ids[i], ids[(i + 1) % ids.len()], curve);
            OrientedEdge::new(topo.add_edge(edge), true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), H).unwrap()
}

/// The x of the round's axis at an arc corner: `r` in from the side and
/// `R - r` from the arc's axis.
fn arc_corner_x(r: f64) -> f64 {
    centre() + ((R - r).powi(2) - (B - r).powi(2)).sqrt()
}

/// The profile's area, and the area and perimeter of its offset in by `r`.
fn profile(r: f64) -> (f64, f64, f64) {
    let c = centre();
    let area = 2.0 * B * X0 + R * R * ((X0 - c) / R).acos() - (X0 - c) * B;
    let (rp, b) = (R - r, B - r);
    let d = arc_corner_x(r) - c;
    let alpha = (d / rp).acos();
    let inner = 2.0 * b * (c + d - r) + rp * rp * alpha - d * b;
    let perimeter = 2.0 * b + 2.0 * (c + d - r) + 2.0 * rp * alpha;
    (area, inner, perimeter)
}

/// The volume rounding every vertical edge at `r` takes off.
fn verticals_removed(r: f64) -> f64 {
    let (area, inner, perimeter) = profile(r);
    H * (area - (inner + perimeter * r + PI * r * r))
}

/// The volume rounding every vertical edge and the top's outline takes off.
fn verticals_and_top_removed(r: f64) -> f64 {
    let (area, inner, perimeter) = profile(r);
    let kept = (H - r) * (inner + perimeter * r + PI * r * r)
        + inner * r
        + perimeter * PI * r * r / 4.0
        + 2.0 * PI * r.powi(3) / 3.0;
    area * H - kept
}

fn is_top(p: Point3, q: Point3) -> bool {
    (p.z() - H).abs() < 1e-9 && (q.z() - H).abs() < 1e-9
}

fn is_vertical_at(x: f64, p: Point3, q: Point3) -> bool {
    (p.x() - x).abs() < 1e-9 && (q.x() - x).abs() < 1e-9 && (p.z() - q.z()).abs() > 1.0
}

struct Rounded {
    topo: Topology,
    solid: SolidId,
    removed: f64,
    before: f64,
}

fn round(label: &str, r: f64, pick: impl Fn(Point3, Point3) -> bool) -> Rounded {
    let mut topo = Topology::new();
    let solid = d_prism(&mut topo);
    let edges: Vec<EdgeId> = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            let [p, q] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
            pick(p, q)
        })
        .collect();
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
    assert!(
        result.failed.is_empty(),
        "{label} r {r}: {:?}",
        result.failed
    );
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{label} r {r}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{label} r {r}: mesh open");
    let removed = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    Rounded {
        topo,
        solid: result.solid,
        removed,
        before,
    }
}

fn faces_of(rounded: &Rounded) -> Vec<FaceSurface> {
    solid_faces(&rounded.topo, rounded.solid)
        .unwrap()
        .into_iter()
        .map(|f| rounded.topo.face(f).unwrap().surface().clone())
        .collect()
}

fn assert_removed(label: &str, r: f64, rounded: &Rounded, expected: f64) {
    assert!(
        (rounded.removed - expected).abs() <= 1e-6 * rounded.before,
        "{label} r {r}: removed {}, expected {expected}",
        rounded.removed
    );
}

#[test]
fn a_round_along_the_arc_walls_ruling_is_a_cylinder() {
    for r in [0.5, 1.0] {
        let label = "one arc corner";
        let rounded = round(label, r, |p, q| is_vertical_at(X0, p, q) && p.y() > 0.0);
        let faces = faces_of(&rounded);
        assert_eq!(faces.len(), 7, "{label} r {r}: the round alone is new");
        let bands: Vec<_> = faces
            .iter()
            .filter_map(|s| match s {
                FaceSurface::Cylinder(c) if (c.radius() - r).abs() < 1e-9 => Some(c.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(bands.len(), 1, "{label} r {r}: the round is a cylinder");
        let axis = bands[0].axis();
        assert!(
            axis.z().abs() > 1.0 - 1e-12,
            "{label} r {r}: along the wall"
        );
        let origin = bands[0].origin();
        assert!(
            (origin.x() - arc_corner_x(r)).abs() < 1e-9 && (origin.y() - (B - r)).abs() < 1e-9,
            "{label} r {r}: axis through {origin:?}"
        );
        // One corner of the four the verticals round; the flat corners each
        // take a quarter square less a quarter disc.
        let flat = (1.0 - PI / 4.0) * r * r * H;
        assert_removed(
            label,
            r,
            &rounded,
            (verticals_removed(r) - 2.0 * flat) / 2.0,
        );
    }
}

#[test]
fn every_vertical_edge() {
    for r in [0.5, 1.0] {
        let label = "every vertical";
        let rounded = round(label, r, |p, q| {
            is_vertical_at(X0, p, q) || is_vertical_at(0.0, p, q)
        });
        assert_eq!(faces_of(&rounded).len(), 10, "{label} r {r}");
        assert_removed(label, r, &rounded, verticals_removed(r));
    }
}

#[test]
fn the_top_outline_and_every_vertical_edge_meet_in_balls() {
    for r in [0.5, 1.0] {
        let label = "top and verticals";
        let rounded = round(label, r, |p, q| {
            is_top(p, q) || is_vertical_at(X0, p, q) || is_vertical_at(0.0, p, q)
        });
        let faces = faces_of(&rounded);
        assert_eq!(faces.len(), 18, "{label} r {r}");
        let balls: Vec<_> = faces
            .iter()
            .filter_map(|s| match s {
                FaceSurface::Sphere(s) => Some((s.center(), s.radius())),
                _ => None,
            })
            .collect();
        assert_eq!(balls.len(), 4, "{label} r {r}: {balls:?}");
        let x = arc_corner_x(r);
        for (ex, ey) in [(r, -(B - r)), (r, B - r), (x, -(B - r)), (x, B - r)] {
            let expected = Point3::new(ex, ey, H - r);
            assert!(
                balls.iter().any(|(centre, radius)| {
                    (radius - r).abs() < 1e-9 && (*centre - expected).length() < 1e-7
                }),
                "{label} r {r}: no ball at {expected:?}: {balls:?}"
            );
        }
        assert_removed(label, r, &rounded, verticals_and_top_removed(r));
    }
}

#[test]
fn the_top_outline_and_the_arc_corners() {
    for r in [0.5, 1.0] {
        let label = "top and arc corners";
        let rounded = round(label, r, |p, q| is_top(p, q) || is_vertical_at(X0, p, q));
        let faces = faces_of(&rounded);
        assert_eq!(faces.len(), 14, "{label} r {r}");
        let balls = faces
            .iter()
            .filter(|s| matches!(s, FaceSurface::Sphere(_)))
            .count();
        assert_eq!(balls, 2, "{label} r {r}: a ball at each arc corner");
    }
}
