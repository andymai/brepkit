//! Revolving a profile with a half-circle side about an axis it clears, part
//! way or a full turn: the side sweeps a torus band, and the solid measures
//! its Pappus volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::revolve::revolve;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// The rectangle over `1 < x < 2`, `0 < z < 1` in `y = 0`, one side a half
/// circle bulging out of it: the `x = 2` side away from the axis, or the
/// `x = 1` side toward it. The wire runs counterclockwise about `-y`, or the
/// other way round (each edge then used backward). Returns the face and its
/// Pappus volume per radian.
fn profile(topo: &mut Topology, toward_axis: bool, backward: bool) -> (FaceId, f64) {
    let v = |topo: &mut Topology, x: f64, z: f64| {
        topo.add_vertex(Vertex::new(Point3::new(x, 0.0, z), 1e-7))
    };
    let corners = [
        v(topo, 1.0, 0.0),
        v(topo, 2.0, 0.0),
        v(topo, 2.0, 1.0),
        v(topo, 1.0, 1.0),
    ];
    // The half circle on side `k` (from corner k to k + 1), counterclockwise
    // about `-y` like the wire, so it bulges out of the rectangle.
    let (side, center) = if toward_axis {
        (3, Point3::new(1.0, 0.0, 0.5))
    } else {
        (1, Point3::new(2.0, 0.0, 0.5))
    };
    let mut edges = Vec::new();
    for k in 0..4 {
        let curve = if k == side {
            EdgeCurve::Circle(Circle3D::new(center, Vec3::new(0.0, -1.0, 0.0), 0.5).unwrap())
        } else {
            EdgeCurve::Line
        };
        edges.push(topo.add_edge(Edge::new(corners[k], corners[(k + 1) % 4], curve)));
    }
    let oriented: Vec<OrientedEdge> = if backward {
        edges
            .iter()
            .rev()
            .map(|&e| OrientedEdge::new(e, false))
            .collect()
    } else {
        edges.iter().map(|&e| OrientedEdge::new(e, true)).collect()
    };
    let wire = topo.add_wire(Wire::new(oriented, true).unwrap());
    let normal = if backward { 1.0 } else { -1.0 };
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, normal, 0.0),
            d: 0.0,
        },
    ));
    let half = PI / 8.0;
    let offset = 2.0 / (3.0 * PI);
    let half_x = if toward_axis {
        1.0 - offset
    } else {
        2.0 + offset
    };
    (face, half.mul_add(half_x, 1.5))
}

/// Revolved 90, 180, 270 or 300 degrees, or a full turn, about z: each solid
/// is valid, meshes watertight and measures its Pappus volume within `1e-9`,
/// the side bulging away from the axis or toward it, the profile wound either
/// way. A partial turn's later rings carry the profile's own curves turned
/// about the axis (a half circle's chord in their place bounded the end cap
/// and the next band by the secant), and each torus band takes its side from
/// the arc's midpoint, where the chord's centre sits on the tube's core.
#[test]
fn a_half_circle_side_revolves_to_its_pappus_volume() {
    for toward_axis in [false, true] {
        for backward in [false, true] {
            for degrees in [90.0_f64, 180.0, 270.0, 300.0, 360.0] {
                let label = format!(
                    "{} the axis, {}, {degrees} degrees",
                    if toward_axis { "toward" } else { "away from" },
                    if backward {
                        "wound backward"
                    } else {
                        "wound forward"
                    }
                );
                let mut topo = Topology::new();
                let (face, per_radian) = profile(&mut topo, toward_axis, backward);
                let solid = revolve(
                    &mut topo,
                    face,
                    Point3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    degrees.to_radians(),
                )
                .unwrap();
                let report = validate_solid(&topo, solid).unwrap();
                assert!(report.is_valid(), "{label}: {:?}", report.issues);
                let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                let truth = per_radian * degrees.to_radians();
                let volume = solid_volume(&topo, solid, 0.01).unwrap();
                assert!(
                    (volume - truth).abs() < 1e-9 * truth,
                    "{label}: volume {volume}, truth {truth}"
                );
            }
        }
    }
}

/// The triangle with a leg on the axis, its apex above the rim (corners
/// `(0, 0, 0)`, `(1, 0, 0)`, `(0, 0, 1)`) or below it (`(0, 0, 0)`,
/// `(1, 0, 1)`, `(0, 0, 1)`), wound either way, revolved a full turn about
/// z: each pointed cone is valid, meshes watertight and measures its Pappus
/// volume within `1e-9`. The wall meets its one rim at the slanted edge's
/// start or its end, the way the profile winds.
#[test]
fn a_triangle_on_the_axis_revolves_to_a_pointed_cone() {
    for apex_above in [true, false] {
        for backward in [false, true] {
            let label = format!(
                "apex {}, {}",
                if apex_above { "above" } else { "below" },
                if backward {
                    "wound backward"
                } else {
                    "wound forward"
                }
            );
            let mut topo = Topology::new();
            let rim_z = if apex_above { 0.0 } else { 1.0 };
            let mut corners = [
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, rim_z),
                Point3::new(0.0, 0.0, 1.0),
            ];
            if backward {
                corners.reverse();
            }
            let ids = corners.map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
            let edges: Vec<OrientedEdge> = (0..3)
                .map(|k| {
                    let e = Edge::new(ids[k], ids[(k + 1) % 3], EdgeCurve::Line);
                    OrientedEdge::new(topo.add_edge(e), true)
                })
                .collect();
            let wire = topo.add_wire(Wire::new(edges, true).unwrap());
            // Counterclockwise about -y when wound forward.
            let normal = if backward == apex_above { -1.0 } else { 1.0 };
            let face = topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, normal, 0.0),
                    d: 0.0,
                },
            ));
            let solid = revolve(
                &mut topo,
                face,
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                2.0 * PI,
            )
            .unwrap();
            let report = validate_solid(&topo, solid).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let truth = PI / 3.0;
            let volume = solid_volume(&topo, solid, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-9 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
        }
    }
}

/// A named profile: its corners in the `(x, z)` half-plane and its Pappus
/// volume per radian.
type Profile<'a> = (&'a str, &'a [(f64, f64)], f64);

/// Profiles touching the axis in `y = 0`, their corners listed counterclockwise
/// about `-y`: a triangle with a leg on the axis (a pointed cone), a square
/// with a side on it (a cylinder), and a triangle touching it at one corner
/// (a cone whose apex is the corner), each wound either way and revolved a
/// quarter, half, three quarters or a full turn about z. Each solid is valid,
/// meshes watertight and measures its Pappus volume within `1e-9`. A vertex
/// on the axis stays one vertex and sweeps no circle, so the bands beside it
/// close as wedges and an edge along the axis sweeps nothing.
#[test]
fn a_profile_touching_the_axis_revolves_part_way() {
    let profiles: [Profile<'_>; 3] = [
        (
            "pointed cone",
            &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            1.0 / 6.0,
        ),
        (
            "cylinder",
            &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            0.5,
        ),
        (
            "cone on its tip",
            &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            1.0 / 3.0,
        ),
    ];
    for (name, corners, per_radian) in &profiles {
        for backward in [false, true] {
            for degrees in [90.0_f64, 180.0, 270.0, 360.0] {
                let label = format!(
                    "{name}, {}, {degrees} degrees",
                    if backward {
                        "wound backward"
                    } else {
                        "wound forward"
                    }
                );
                let mut topo = Topology::new();
                let mut points: Vec<Point3> = corners
                    .iter()
                    .map(|&(x, z)| Point3::new(x, 0.0, z))
                    .collect();
                if backward {
                    points.reverse();
                }
                let ids: Vec<_> = points
                    .iter()
                    .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
                    .collect();
                let n = ids.len();
                let edges: Vec<OrientedEdge> = (0..n)
                    .map(|k| {
                        let e = Edge::new(ids[k], ids[(k + 1) % n], EdgeCurve::Line);
                        OrientedEdge::new(topo.add_edge(e), true)
                    })
                    .collect();
                let wire = topo.add_wire(Wire::new(edges, true).unwrap());
                let normal = if backward { 1.0 } else { -1.0 };
                let face = topo.add_face(Face::new(
                    wire,
                    vec![],
                    FaceSurface::Plane {
                        normal: Vec3::new(0.0, normal, 0.0),
                        d: 0.0,
                    },
                ));
                let solid = revolve(
                    &mut topo,
                    face,
                    Point3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    degrees.to_radians(),
                )
                .unwrap();
                let report = validate_solid(&topo, solid).unwrap();
                assert!(report.is_valid(), "{label}: {:?}", report.issues);
                let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                let truth = per_radian * degrees.to_radians();
                let volume = solid_volume(&topo, solid, 0.01).unwrap();
                assert!(
                    (volume - truth).abs() < 1e-9 * truth,
                    "{label}: volume {volume}, truth {truth}"
                );
            }
        }
    }
}

/// The half disc of radius 1 in `y = 0` bounded by the half circle through
/// `(1, 0, 0)` and its diameter on the z axis, and the quarter disc bounded by
/// the arc from `(1, 0, 0)` to `(0, 0, 1)` and two radii, one on the axis, each
/// wound either way and revolved a quarter, half, three quarters or a full
/// turn about z: each solid is valid, meshes watertight and measures its
/// Pappus volume within `1e-9`. An arc centred on the axis sweeps a sphere
/// band, though the half circle ends on the axis at both ends (only a line
/// there lies along the axis) and its chord sweeps nothing. The full turn
/// takes the segmented revolve too.
#[test]
fn a_disc_sector_revolves_to_a_ball_part_way() {
    for half in [true, false] {
        for backward in [false, true] {
            for degrees in [90.0_f64, 180.0, 270.0, 360.0] {
                let label = format!(
                    "{} disc, {}, {degrees} degrees",
                    if half { "half" } else { "quarter" },
                    if backward {
                        "wound backward"
                    } else {
                        "wound forward"
                    }
                );
                let mut topo = Topology::new();
                let mut v =
                    |x: f64, z: f64| topo.add_vertex(Vertex::new(Point3::new(x, 0.0, z), 1e-7));
                let (from, top, center) = (
                    v(if half { 0.0 } else { 1.0 }, if half { -1.0 } else { 0.0 }),
                    v(0.0, 1.0),
                    v(0.0, 0.0),
                );
                // Counterclockwise about -y, through +x.
                let circle =
                    Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 1.0)
                        .unwrap();
                let mut edges = vec![
                    topo.add_edge(Edge::new(from, top, EdgeCurve::Circle(circle))),
                    topo.add_edge(Edge::new(
                        top,
                        if half { from } else { center },
                        EdgeCurve::Line,
                    )),
                ];
                if !half {
                    edges.push(topo.add_edge(Edge::new(center, from, EdgeCurve::Line)));
                }
                let oriented: Vec<OrientedEdge> = if backward {
                    edges
                        .iter()
                        .rev()
                        .map(|&e| OrientedEdge::new(e, false))
                        .collect()
                } else {
                    edges.iter().map(|&e| OrientedEdge::new(e, true)).collect()
                };
                let wire = topo.add_wire(Wire::new(oriented, true).unwrap());
                let normal = if backward { 1.0 } else { -1.0 };
                let face = topo.add_face(Face::new(
                    wire,
                    vec![],
                    FaceSurface::Plane {
                        normal: Vec3::new(0.0, normal, 0.0),
                        d: 0.0,
                    },
                ));
                let solid = revolve(
                    &mut topo,
                    face,
                    Point3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    degrees.to_radians(),
                )
                .unwrap();
                let report = validate_solid(&topo, solid).unwrap();
                assert!(report.is_valid(), "{label}: {:?}", report.issues);
                let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                let per_radian = if half { 2.0 / 3.0 } else { 1.0 / 3.0 };
                let truth = per_radian * degrees.to_radians();
                let volume = solid_volume(&topo, solid, 0.01).unwrap();
                assert!(
                    (volume - truth).abs() < 1e-9 * truth,
                    "{label}: volume {volume}, truth {truth}"
                );
            }
        }
    }
}

/// The part of the spherical shell `1 < r < 2` about the origin between 10
/// and 40 degrees from +z, in `y = 0`, its inner arc stored from 10 to 40
/// degrees and run back, wound either way and revolved a quarter, half,
/// three quarters or a full turn about z: each solid is valid, meshes
/// watertight and measures its volume, `(2^3 - 1) (cos 10° - cos 40°) / 3`
/// per radian, within `1e-8`. Both arcs sweep sphere bands clear of the
/// axis; each later segment reads its band's side from the arc turned back
/// to the profile by the segment's own angle, since a centre on the axis
/// has no direction to measure it by.
#[test]
fn a_spherical_shell_sector_revolves_part_way() {
    let (lo, hi) = (10.0_f64.to_radians(), 40.0_f64.to_radians());
    let at = |r: f64, polar: f64| Point3::new(r * polar.sin(), 0.0, r * polar.cos());
    for backward in [false, true] {
        for degrees in [90.0_f64, 180.0, 270.0, 360.0] {
            let label = format!(
                "{}, {degrees} degrees",
                if backward {
                    "wound backward"
                } else {
                    "wound forward"
                }
            );
            let mut topo = Topology::new();
            let [a, b, c, d] = [at(2.0, lo), at(2.0, hi), at(1.0, hi), at(1.0, lo)]
                .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
            // Counterclockwise about +y runs from +z toward +x.
            let circle = |r: f64| {
                EdgeCurve::Circle(
                    Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), r).unwrap(),
                )
            };
            let outer = topo.add_edge(Edge::new(a, b, circle(2.0)));
            let down = topo.add_edge(Edge::new(b, c, EdgeCurve::Line));
            let inner = topo.add_edge(Edge::new(d, c, circle(1.0)));
            let up = topo.add_edge(Edge::new(d, a, EdgeCurve::Line));
            let mut oriented = vec![
                OrientedEdge::new(outer, true),
                OrientedEdge::new(down, true),
                OrientedEdge::new(inner, false),
                OrientedEdge::new(up, true),
            ];
            if backward {
                oriented = oriented
                    .into_iter()
                    .rev()
                    .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
                    .collect();
            }
            let wire = topo.add_wire(Wire::new(oriented, true).unwrap());
            let normal = if backward { -1.0 } else { 1.0 };
            let face = topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, normal, 0.0),
                    d: 0.0,
                },
            ));
            let solid = revolve(
                &mut topo,
                face,
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                degrees.to_radians(),
            )
            .unwrap();
            let report = validate_solid(&topo, solid).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let truth = 7.0 / 3.0 * (lo.cos() - hi.cos()) * degrees.to_radians();
            let volume = solid_volume(&topo, solid, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-8 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
        }
    }
}
