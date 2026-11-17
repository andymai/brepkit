//! Two rounds of different radii at a convex corner whose third edge stays
//! sharp cross each other near the corner. They must meet along the curve
//! where their surfaces cross, and the wider round must end on the
//! narrower round's wall, so the solid is the part less the material each
//! round takes. The volume of that solid is worked out here apart from the
//! B-Rep, and points near the corner are read against the same rule.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use brepkit_blend::radius_law::RadiusLaw;
use brepkit_math::mat::Mat4;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2_variable;
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;

const POSES: [&str; 3] = ["upright", "turned", "mirrored"];

/// A solid in a pose, and the function that carries its points there.
fn posed(
    topo: &mut Topology,
    solid: SolidId,
    name: &str,
) -> (SolidId, impl Fn(Point3) -> Point3 + use<>) {
    let turn = Mat4::translation(3.0, -1.0, 2.0) * Mat4::rotation_z(0.4) * Mat4::rotation_x(0.6);
    let solid = match name {
        "turned" => {
            transform_solid(topo, solid, &turn).unwrap();
            solid
        }
        "mirrored" => mirror(
            topo,
            solid,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap(),
        _ => solid,
    };
    let (turned, mirrored) = (name == "turned", name == "mirrored");
    let carry = move |p: Point3| {
        if turned {
            turn.mul_point(p)
        } else if mirrored {
            Point3::new(-p.x(), p.y(), p.z())
        } else {
            p
        }
    };
    (solid, carry)
}

fn edge_at(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let found: Vec<_> = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&id| {
            let edge = topo.edge(id).unwrap();
            let (s, e) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            ((s - a).length() < 1e-6 && (e - b).length() < 1e-6)
                || ((s - b).length() < 1e-6 && (e - a).length() < 1e-6)
        })
        .collect();
    assert_eq!(found.len(), 1, "one edge from {a:?} to {b:?}");
    found[0]
}

/// How far into the corner of two faces at right angles a round of radius
/// `r` reaches along one face, at the depth `t` from the other.
fn reach(r: f64, t: f64) -> f64 {
    if t >= r {
        0.0
    } else {
        r - (r * r - (r - t).powi(2)).max(0.0).sqrt()
    }
}

/// The cross-section a round of radius `r` takes off a square corner.
fn corner(r: f64) -> f64 {
    r * r * (1.0 - FRAC_PI_4)
}

/// The volume two rounds of radii `r1` and `r2` both take at a box corner,
/// along two edges that share a face: at depth `t` from that face each takes
/// a strip of its reach, and the two strips share a rectangle. Read with
/// `t = lo (1 - cos phi)` for the smaller radius `lo`, which keeps the
/// integrand smooth.
fn both_take(r1: f64, r2: f64) -> f64 {
    const STEPS: u32 = 40_000;
    let (lo, hi) = (r1.min(r2), r1.max(r2));
    let step = FRAC_PI_2 / f64::from(STEPS);
    (0..STEPS)
        .map(|k| {
            let phi = (f64::from(k) + 0.5) * step;
            let t = lo * (1.0 - phi.cos());
            lo * (1.0 - phi.sin()) * reach(hi, t) * lo * phi.sin() * step
        })
        .sum()
}

/// The box corner at the origin and its three edges, along x, y and z.
const AXES: [Vec3; 3] = [
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
];

/// Whether the rounds of radius `radii[k]` along the box edges `edges[k]`
/// at the origin take `p`: each takes the points of its corner farther
/// from its edge's line than its arc.
fn taken_at_box_corner(edges: [usize; 2], radii: [f64; 2], p: Point3) -> bool {
    let c = [p.x(), p.y(), p.z()];
    edges.iter().zip(radii).any(|(&edge, r)| {
        let (u, v) = match edge {
            0 => (c[1], c[2]),
            1 => (c[0], c[2]),
            _ => (c[0], c[1]),
        };
        u < r && v < r && (u - r).hypot(v - r) > r
    })
}

/// Rounds the box edges `edges` at the origin with `radii`, in a pose, and
/// checks the result against the box less what the rounds take.
fn check_box_corner(dims: [f64; 3], edges: [usize; 2], radii: [f64; 2], name: &str) {
    let what = format!("{dims:?} edges {edges:?} radii {radii:?} {name}");
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, dims[0], dims[1], dims[2]).unwrap();
    let (solid, carry) = posed(&mut topo, solid, name);
    let origin = Point3::new(0.0, 0.0, 0.0);
    let laws = edges
        .iter()
        .zip(radii)
        .map(|(&k, r)| {
            let far = origin + AXES[k] * dims[k];
            (
                edge_at(&topo, solid, carry(origin), carry(far)),
                RadiusLaw::Constant(r),
            )
        })
        .collect();
    let result = fillet_v2_variable(&mut topo, solid, laws).unwrap();
    assert!(
        result.succeeded.len() == 2 && result.failed.is_empty() && !result.is_partial,
        "{what}: both rounds must be built: {:?}",
        result.failed
    );
    let s = result.solid;
    let report = validate_solid(&topo, s).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    assert!(
        is_watertight(&tessellate_solid(&topo, s, 0.01).unwrap()),
        "{what}: open mesh"
    );
    let faces = solid_faces(&topo, s).unwrap();
    let rounds = faces
        .iter()
        .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();
    assert_eq!(
        (faces.len(), rounds),
        (8, 2),
        "{what}: the six walls and the two rounds, with no patch at the corner"
    );
    let truth = dims[0] * dims[1] * dims[2]
        - corner(radii[0]) * dims[edges[0]]
        - corner(radii[1]) * dims[edges[1]]
        + both_take(radii[0], radii[1]);
    let volume = solid_volume(&topo, s, 0.001).unwrap();
    assert!(
        (volume - truth).abs() < 1e-8 * truth,
        "{what}: volume {volume}, expected {truth}"
    );
    let reach = radii[0].max(radii[1]);
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let p = Point3::new(
                    (0.1 + 0.3 * f64::from(i)) * reach,
                    (0.1 + 0.3 * f64::from(j)) * reach,
                    (0.1 + 0.3 * f64::from(k)) * reach,
                );
                let expected = if taken_at_box_corner(edges, radii, p) {
                    PointClassification::Outside
                } else {
                    PointClassification::Inside
                };
                let read = classify_point(&topo, s, carry(p), 0.01, 1e-7).unwrap();
                assert_eq!(read, expected, "{what}: {p:?}");
            }
        }
    }
}

/// The issue's box, and the other two pairs of edges at its corner, with
/// radii far apart, a hundredth apart and four to one.
#[test]
fn rounds_of_two_radii_at_a_box_corner_meet_along_their_crease() {
    for edges in [[0, 1], [0, 2], [1, 2]] {
        for radii in [[3.0, 1.0], [1.0, 3.0], [3.0, 3.01], [8.0, 2.0]] {
            for name in POSES {
                check_box_corner([20.0, 20.0, 20.0], edges, radii, name);
            }
        }
    }
}

/// On a slab 6 thick, the wider round spans almost the whole thickness.
#[test]
fn a_round_almost_as_wide_as_the_slab_meets_a_narrow_one() {
    for edges in [[0, 1], [0, 2], [1, 2]] {
        for radii in [[5.99, 1.0], [2.0, 5.9]] {
            for name in POSES {
                check_box_corner([20.0, 14.0, 6.0], edges, radii, name);
            }
        }
    }
}

/// Radii a little more than a vertex's tolerance apart: the crease ends
/// within that tolerance of the end of the narrower round's contact on its
/// wall, and the wider round's end there is shorter than a thousandth.
#[test]
fn rounds_of_nearly_one_radius_meet_along_their_crease() {
    for gap in [1e-4, 1e-5, 1e-6, 1.5e-7] {
        for name in POSES {
            check_box_corner([20.0, 20.0, 20.0], [0, 1], [3.0, 3.0 + gap], name);
        }
    }
}

/// A pocket in the narrower round's wall where the wider round's end takes
/// that wall away would keep its walls with nothing round it: the build
/// refuses the corner. A pocket clear of the corner leaves it to be built.
#[test]
fn a_pocket_in_the_wall_under_the_crossing_is_refused() {
    let pocketed = |topo: &mut Topology, low: Point3, size: Vec3| {
        let solid = make_box(topo, 20.0, 20.0, 20.0).unwrap();
        let pocket = make_box(topo, size.x(), size.y(), size.z()).unwrap();
        transform_solid(topo, pocket, &Mat4::translation(low.x(), low.y(), low.z())).unwrap();
        brepkit_operations::boolean::boolean(
            topo,
            brepkit_operations::boolean::BooleanOp::Cut,
            solid,
            pocket,
        )
        .unwrap()
    };
    let origin = Point3::new(0.0, 0.0, 0.0);
    let laws = |topo: &Topology, solid: SolidId| {
        vec![
            (
                edge_at(topo, solid, origin, Point3::new(20.0, 0.0, 0.0)),
                RadiusLaw::Constant(3.0),
            ),
            (
                edge_at(topo, solid, origin, Point3::new(0.0, 20.0, 0.0)),
                RadiusLaw::Constant(1.0),
            ),
        ]
    };

    let mut topo = Topology::new();
    let solid = pocketed(
        &mut topo,
        Point3::new(-0.1, 0.05, 1.3),
        Vec3::new(0.2, 0.1, 0.3),
    );
    let edges = laws(&topo, solid);
    match fillet_v2_variable(&mut topo, solid, edges) {
        Err(brepkit_operations::OperationsError::Blend(
            brepkit_blend::BlendError::PlanningFailure { reason },
        )) => assert!(reason.contains("cross at vertex"), "{reason}"),
        Err(other) => panic!("expected the crossing's refusal, got {other}"),
        Ok(result) => panic!(
            "a pocket under the crossing must be refused, got {} successes",
            result.succeeded.len()
        ),
    }

    let mut topo = Topology::new();
    let solid = pocketed(
        &mut topo,
        Point3::new(-0.1, 5.0, 5.0),
        Vec3::new(0.2, 1.0, 1.0),
    );
    let edges = laws(&topo, solid);
    let result = fillet_v2_variable(&mut topo, solid, edges).unwrap();
    assert!(
        result.failed.is_empty() && !result.is_partial,
        "{:?}",
        result.failed
    );
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert!(
        is_watertight(&tessellate_solid(&topo, result.solid, 0.01).unwrap()),
        "open mesh"
    );
    let truth = 8000.0 - 0.1 - corner(3.0) * 20.0 - corner(1.0) * 20.0 + both_take(3.0, 1.0);
    let volume = solid_volume(&topo, result.solid, 0.001).unwrap();
    assert!(
        (volume - truth).abs() < 1e-8 * truth,
        "volume {volume}, expected {truth}"
    );
}

/// A pocket in the narrower round's wall whose corner lies on the wider
/// round's end, the rest of it in the kept wall: kept, the hole would touch
/// the wall's new outline with no edge in common, and the build refuses it.
#[test]
fn a_pocket_that_touches_the_wider_rounds_end_is_refused() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
    // On the wall x = 0, the r = 3 round's end is the circle about
    // (y, z) = (3, 3), and the pocket's corner nearest the floor lies on it,
    // above the r = 1 round's contact at z = 1.
    let (y, z) = (3.0 - 3.0 * 0.4_f64.cos(), 3.0 - 3.0 * 0.4_f64.sin());
    let pocket = make_box(&mut topo, 0.2, 0.05, 0.05).unwrap();
    transform_solid(&mut topo, pocket, &Mat4::translation(-0.1, y, z)).unwrap();
    let pocketed = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Cut,
        solid,
        pocket,
    )
    .unwrap();
    let origin = Point3::new(0.0, 0.0, 0.0);
    let laws = vec![
        (
            edge_at(&topo, pocketed, origin, Point3::new(20.0, 0.0, 0.0)),
            RadiusLaw::Constant(3.0),
        ),
        (
            edge_at(&topo, pocketed, origin, Point3::new(0.0, 20.0, 0.0)),
            RadiusLaw::Constant(1.0),
        ),
    ];
    match fillet_v2_variable(&mut topo, pocketed, laws) {
        Err(brepkit_operations::OperationsError::Blend(
            brepkit_blend::BlendError::PlanningFailure { reason },
        )) => assert!(reason.contains("cross at vertex"), "{reason}"),
        Err(other) => panic!("expected the crossing's refusal, got {other}"),
        Ok(result) => panic!(
            "a pocket on the round's end must be refused, got {} successes",
            result.succeeded.len()
        ),
    }
}

/// A box sheared along `y` by `1 / sqrt(3)` of its height: its floor meets
/// the wall along the `x` edge at 60 degrees, so the r = 8 round along that
/// edge meets its wall 12 above the floor. A pocket in the wall `x = 0`
/// 9 above the floor, a hundredth off the sheared wall, lies in the corner
/// that round takes, and the build refuses it. A pocket 15 above the floor
/// stays clear of the round.
#[test]
fn a_pocket_under_a_round_of_an_acute_corner_is_refused() {
    let shift = 1.0 / 3.0_f64.sqrt();
    let build = |low: f64| {
        let mut topo = Topology::new();
        let outline = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(20.0, 0.0, 0.0),
            Point3::new(20.0, 20.0, 0.0),
            Point3::new(0.0, 20.0, 0.0),
        ];
        let wire = make_polygon_wire(&mut topo, &outline, Tolerance::new().linear).unwrap();
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: 0.0,
            },
        ));
        let solid = extrude(&mut topo, face, Vec3::new(0.0, shift, 1.0) * 20.0, 20.0).unwrap();
        let pocket = make_box(&mut topo, 0.2, 0.03, 0.01).unwrap();
        transform_solid(
            &mut topo,
            pocket,
            &Mat4::translation(-0.1, low * shift + 0.01, low),
        )
        .unwrap();
        let pocketed = brepkit_operations::boolean::boolean(
            &mut topo,
            brepkit_operations::boolean::BooleanOp::Cut,
            solid,
            pocket,
        )
        .unwrap();
        let laws = vec![
            (
                edge_at(&topo, pocketed, outline[0], outline[1]),
                RadiusLaw::Constant(8.0),
            ),
            (
                edge_at(&topo, pocketed, outline[0], outline[3]),
                RadiusLaw::Constant(2.0),
            ),
        ];
        let result = fillet_v2_variable(&mut topo, pocketed, laws);
        (topo, result)
    };
    match build(9.0).1 {
        Err(brepkit_operations::OperationsError::Blend(
            brepkit_blend::BlendError::PlanningFailure { reason },
        )) => assert!(reason.contains("cross at vertex"), "{reason}"),
        Err(other) => panic!("expected the crossing's refusal, got {other}"),
        Ok(result) => panic!(
            "a pocket under the crossing must be refused, got {} successes",
            result.succeeded.len()
        ),
    }
    let (topo, result) = build(15.0);
    let result = result.unwrap();
    assert!(
        result.succeeded.len() == 2 && result.failed.is_empty() && !result.is_partial,
        "{:?}",
        result.failed
    );
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert!(
        is_watertight(&tessellate_solid(&topo, result.solid, 0.01).unwrap()),
        "open mesh"
    );
}

/// A wall that curves away from the corner holds no plane for the wider
/// round to end on, and the build refuses the corner instead of returning
/// rounds that cross through each other.
#[test]
fn rounds_that_cross_beside_a_curved_wall_are_refused() {
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};
    for (ra, rb) in [(2.0, 1.0), (1.0, 2.0)] {
        let mut topo = Topology::new();
        let mut vertex =
            |x: f64, y: f64| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7));
        let corners = [
            vertex(0.0, 0.0),
            vertex(20.0, 0.0),
            vertex(12.0, 8.0),
            vertex(0.0, 8.0),
        ];
        // The floor's outline leaves (20, 0) along a quarter circle about
        // (12, 0), square to its first edge.
        let arc = brepkit_math::curves::Circle3D::new(
            Point3::new(12.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            8.0,
        )
        .unwrap();
        let curves = [
            EdgeCurve::Line,
            EdgeCurve::Circle(arc),
            EdgeCurve::Line,
            EdgeCurve::Line,
        ];
        let edges = curves
            .into_iter()
            .enumerate()
            .map(|(k, curve)| {
                let edge = Edge::new(corners[k], corners[(k + 1) % 4], curve);
                OrientedEdge::new(topo.add_edge(edge), true)
            })
            .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: 0.0,
            },
        ));
        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 10.0).unwrap();
        let corner = Point3::new(20.0, 0.0, 0.0);
        let laws = vec![
            (
                edge_at(&topo, solid, Point3::new(0.0, 0.0, 0.0), corner),
                RadiusLaw::Constant(ra),
            ),
            (
                edge_at(&topo, solid, corner, Point3::new(12.0, 8.0, 0.0)),
                RadiusLaw::Constant(rb),
            ),
        ];
        match fillet_v2_variable(&mut topo, solid, laws) {
            Err(brepkit_operations::OperationsError::Blend(
                brepkit_blend::BlendError::PlanningFailure { reason },
            )) => assert!(reason.contains("cross at vertex"), "{reason}"),
            Err(other) => panic!("expected the crossing's refusal, got {other}"),
            Ok(result) => panic!(
                "radii {ra} and {rb} beside a curved wall must be refused, got {} successes",
                result.succeeded.len()
            ),
        }
    }
}

/// The part of the convex polygon `poly` (counter-clockwise) on the side
/// `a x + b y <= c`.
fn clip(poly: &[(f64, f64)], (a, b, c): (f64, f64, f64)) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        let (fp, fq) = (a * p.0 + b * p.1 - c, a * q.0 + b * q.1 - c);
        if fp <= 0.0 {
            out.push(p);
        }
        if (fp < 0.0 && fq > 0.0) || (fq < 0.0 && fp > 0.0) {
            let t = fp / (fp - fq);
            out.push((p.0 + t * (q.0 - p.0), p.1 + t * (q.1 - p.1)));
        }
    }
    out
}

fn area(poly: &[(f64, f64)]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    0.5 * (0..n)
        .map(|i| {
            let (p, q) = (poly[i], poly[(i + 1) % n]);
            p.0 * q.1 - q.0 * p.1
        })
        .sum::<f64>()
        .abs()
}

/// The part of `poly` within `width` of the line of its edge `i`.
fn strip(poly: &[(f64, f64)], outline: &[(f64, f64)], i: usize, width: f64) -> Vec<(f64, f64)> {
    let (p, q) = (outline[i], outline[(i + 1) % outline.len()]);
    let length = (q.0 - p.0).hypot(q.1 - p.1);
    let (nx, ny) = ((q.1 - p.1) / length, -(q.0 - p.0) / length);
    clip(poly, (-nx, -ny, width - (nx * p.0 + ny * p.1)))
}

/// The volume rounds of radii `ra` and `rb` along the floor edges `ia` and
/// `ib` of a convex prism take: at each height, the part of its outline
/// within each round's reach of that round's edge.
fn prism_taken(outline: &[(f64, f64)], (ia, ra): (usize, f64), (ib, rb): (usize, f64)) -> f64 {
    const STEPS: u32 = 20_000;
    let top = ra.max(rb);
    let step = FRAC_PI_2 / f64::from(STEPS);
    (0..STEPS)
        .map(|k| {
            let phi = (f64::from(k) + 0.5) * step;
            let z = top * (1.0 - phi.cos());
            let (wa, wb) = (reach(ra, z), reach(rb, z));
            let a = strip(outline, outline, ia, wa);
            let b = strip(outline, outline, ib, wb);
            let ab = strip(&a, outline, ib, wb);
            let taken = if wa > 0.0 { area(&a) } else { 0.0 }
                + if wb > 0.0 { area(&b) } else { 0.0 }
                - if wa > 0.0 && wb > 0.0 { area(&ab) } else { 0.0 };
            taken * top * phi.sin() * step
        })
        .sum()
}

/// A corner of a prism at an acute and at an obtuse angle. The wider round
/// ends where its cylinder meets the plane of the narrower round's wall, an
/// ellipse.
#[test]
fn rounds_of_two_radii_at_a_prism_corner_of_any_angle() {
    let wedge = vec![(0.0, 0.0), (20.0, 0.0), (10.0, 15.0)];
    let obtuse = vec![(0.0, 0.0), (20.0, 0.0), (26.0, 15.0), (0.0, 15.0)];
    for (outline, radii) in [
        (&wedge, (2.0, 1.0)),
        (&wedge, (1.0, 2.0)),
        (&obtuse, (2.0, 1.0)),
        (&obtuse, (1.0, 2.0)),
    ] {
        for name in POSES {
            let what = format!("{outline:?} radii {radii:?} {name}");
            let height = 10.0;
            let mut topo = Topology::new();
            let points: Vec<Point3> = outline
                .iter()
                .map(|&(x, y)| Point3::new(x, y, 0.0))
                .collect();
            let wire = make_polygon_wire(&mut topo, &points, Tolerance::new().linear).unwrap();
            let face = topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 0.0, -1.0),
                    d: 0.0,
                },
            ));
            let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), height).unwrap();
            let (solid, carry) = posed(&mut topo, solid, name);
            let laws = vec![
                (
                    edge_at(&topo, solid, carry(points[0]), carry(points[1])),
                    RadiusLaw::Constant(radii.0),
                ),
                (
                    edge_at(&topo, solid, carry(points[1]), carry(points[2])),
                    RadiusLaw::Constant(radii.1),
                ),
            ];
            let result = fillet_v2_variable(&mut topo, solid, laws).unwrap();
            assert!(
                result.succeeded.len() == 2 && result.failed.is_empty() && !result.is_partial,
                "{what}: both rounds must be built: {:?}",
                result.failed
            );
            let s = result.solid;
            let report = validate_solid(&topo, s).unwrap();
            assert!(report.is_valid(), "{what}: {:?}", report.issues);
            assert!(
                is_watertight(&tessellate_solid(&topo, s, 0.01).unwrap()),
                "{what}: open mesh"
            );
            let truth = area(outline) * height - prism_taken(outline, (0, radii.0), (1, radii.1));
            let volume = solid_volume(&topo, s, 0.001).unwrap();
            assert!(
                (volume - truth).abs() < 1e-8 * truth,
                "{what}: volume {volume}, expected {truth}"
            );
        }
    }
}
