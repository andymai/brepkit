//! Chamfers of a disc's top outline where a square notch is cut from it:
//! the rim's arcs and the notch's two sides, which meet the rim at convex
//! corners and each other at an inside corner. At the inside corner the two
//! sides' chamfers meet beyond both their edges' ends, in a square no
//! strip along an edge covers. The walls stand square to the top, so the
//! cross-section `h` below the top is the outline inset by `d - h`, and the
//! chamfer takes the integral of the top's area less the inset's over the
//! inset from 0 to `d`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::{PI, TAU};

use brepkit_math::mat::Mat4;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

const H: f64 = 3.0;

/// One piece of a flat outline, run the way its loop runs: a segment, or an
/// arc about `center` sweeping `sweep` radians from the angle `from`
/// (counter-clockwise when positive).
#[derive(Clone, Copy)]
enum Run2 {
    Segment([f64; 2], [f64; 2]),
    Arc {
        center: [f64; 2],
        radius: f64,
        from: f64,
        sweep: f64,
    },
}

impl Run2 {
    fn ends(self) -> [[f64; 2]; 2] {
        match self {
            Self::Segment(a, b) => [a, b],
            Self::Arc {
                center,
                radius,
                from,
                sweep,
            } => [from, from + sweep]
                .map(|t| [center[0] + radius * t.cos(), center[1] + radius * t.sin()]),
        }
    }

    /// The area term `(x dy - y dx) / 2` along the piece.
    fn green(self) -> f64 {
        match self {
            Self::Segment(a, b) => 0.5 * (a[0] * b[1] - b[0] * a[1]),
            Self::Arc {
                center: [cx, cy],
                radius: r,
                from,
                sweep,
            } => {
                let to = from + sweep;
                0.5 * (r * r * sweep
                    + r * (cx * (to.sin() - from.sin()) - cy * (to.cos() - from.cos())))
            }
        }
    }

    /// The piece moved `s` to its left.
    fn inset(self, s: f64) -> Self {
        match self {
            Self::Segment(a, b) => {
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let l = dx.hypot(dy);
                let (lx, ly) = (-dy / l * s, dx / l * s);
                Self::Segment([a[0] + lx, a[1] + ly], [b[0] + lx, b[1] + ly])
            }
            Self::Arc {
                center,
                radius,
                from,
                sweep,
            } => Self::Arc {
                center,
                radius: radius - s * sweep.signum(),
                from,
                sweep,
            },
        }
    }

    /// Where the piece's curve meets `other`'s, nearest `near`.
    fn crossing(self, other: Self, near: [f64; 2]) -> [f64; 2] {
        let line_circle = |a: [f64; 2], b: [f64; 2], c: [f64; 2], r: f64| {
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let l = dx.hypot(dy);
            let (ux, uy) = (dx / l, dy / l);
            let (wx, wy) = (a[0] - c[0], a[1] - c[1]);
            let bb = wx * ux + wy * uy;
            let root = (bb * bb - (wx * wx + wy * wy - r * r)).sqrt();
            [-bb - root, -bb + root]
                .map(|t| [a[0] + ux * t, a[1] + uy * t])
                .into_iter()
                .min_by(|p, q| {
                    let dp = (p[0] - near[0]).hypot(p[1] - near[1]);
                    let dq = (q[0] - near[0]).hypot(q[1] - near[1]);
                    dp.total_cmp(&dq)
                })
                .unwrap()
        };
        match (self, other) {
            (Self::Segment(a, b), Self::Arc { center, radius, .. })
            | (Self::Arc { center, radius, .. }, Self::Segment(a, b)) => {
                line_circle(a, b, center, radius)
            }
            (Self::Segment(a, b), Self::Segment(c, d)) => {
                let (ux, uy, vx, vy) = (b[0] - a[0], b[1] - a[1], d[0] - c[0], d[1] - c[1]);
                let den = ux * vy - uy * vx;
                let t = ((c[0] - a[0]) * vy - (c[1] - a[1]) * vx) / den;
                [a[0] + ux * t, a[1] + uy * t]
            }
            (Self::Arc { .. }, Self::Arc { .. }) => panic!("no arc-arc corner here"),
        }
    }

    /// The piece with its start (or end) moved to `p` on its curve.
    fn with_end(self, p: [f64; 2], at_start: bool) -> Self {
        match self {
            Self::Segment(a, b) => {
                if at_start {
                    Self::Segment(p, b)
                } else {
                    Self::Segment(a, p)
                }
            }
            Self::Arc {
                center,
                radius,
                from,
                sweep,
            } => {
                let t = (p[1] - center[1]).atan2(p[0] - center[0]);
                // The nearest angle to the old one, a small move.
                let near = |old: f64| old + (t - old + PI).rem_euclid(TAU) - PI;
                if at_start {
                    let new_from = near(from);
                    Self::Arc {
                        center,
                        radius,
                        from: new_from,
                        sweep: sweep - (new_from - from),
                    }
                } else {
                    Self::Arc {
                        center,
                        radius,
                        from,
                        sweep: near(from + sweep) - from,
                    }
                }
            }
        }
    }
}

/// The area inside a counter-clockwise loop of `runs` inset by `s`: each
/// piece moved `s` to its left, two pieces meeting at an angle cut back to
/// where they cross.
fn inset_area(runs: &[Run2], s: f64) -> f64 {
    let mut moved: Vec<Run2> = runs.iter().map(|r| r.inset(s)).collect();
    let n = moved.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (end, start) = (moved[i].ends()[1], moved[j].ends()[0]);
        if (end[0] - start[0]).hypot(end[1] - start[1]) > 1e-12 {
            let vertex = runs[i].ends()[1];
            let p = moved[i].crossing(moved[j], vertex);
            moved[i] = moved[i].with_end(p, false);
            moved[j] = moved[j].with_end(p, true);
        }
    }
    moved.iter().map(|r| r.green()).sum()
}

/// The outer loop of a face on a plane `z = const`, wound counter-clockwise
/// about `+z`.
fn flat_loop(topo: &Topology, face: FaceId) -> Vec<Run2> {
    let wire = topo.wire(topo.face(face).unwrap().outer_wire()).unwrap();
    let flat = |p: Point3| [p.x(), p.y()];
    wire.edges()
        .iter()
        .map(|oe| {
            let edge = topo.edge(oe.edge()).unwrap();
            let (mut a, mut b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            if !oe.is_forward() {
                std::mem::swap(&mut a, &mut b);
            }
            match edge.curve() {
                EdgeCurve::Line => Run2::Segment(flat(a), flat(b)),
                EdgeCurve::Circle(c) => {
                    let center = flat(c.center());
                    let angle = |p: Point3| (p.y() - center[1]).atan2(p.x() - center[0]);
                    let (from, to) = (angle(a), angle(b));
                    let ccw = (c.normal().z() > 0.0) == oe.is_forward();
                    let sweep = if ccw {
                        (to - from).rem_euclid(TAU)
                    } else {
                        -(from - to).rem_euclid(TAU)
                    };
                    Run2::Arc {
                        center,
                        radius: c.radius(),
                        from,
                        sweep,
                    }
                }
                other => panic!("unexpected {}", other.type_tag()),
            }
        })
        .collect()
}

/// A disc of radius 10, `H` high, less the quarter beyond `(2, 2)`, with a
/// blind pocket of radius 0.1 at `pocket` when given, placed by `pose`.
fn notched_disc(topo: &mut Topology, pocket: Option<[f64; 2]>, pose: &Mat4) -> SolidId {
    notched_at(topo, 2.0, pocket, pose)
}

/// A disc of radius 10, `H` high, less the quarter beyond `(corner,
/// corner)`, with a blind pocket of radius 0.1 at `pocket` when given,
/// placed by `pose`.
fn notched_at(topo: &mut Topology, corner: f64, pocket: Option<[f64; 2]>, pose: &Mat4) -> SolidId {
    let disc = make_cylinder(topo, 10.0, H).unwrap();
    let notch = make_box(topo, 10.0, 10.0, H + 2.0).unwrap();
    transform_solid(topo, notch, &Mat4::translation(corner, corner, -1.0)).unwrap();
    let mut solid = boolean(topo, BooleanOp::Cut, disc, notch).unwrap();
    if let Some([x, y]) = pocket {
        let faces = solid_faces(topo, solid).unwrap().len();
        let drill = make_cylinder(topo, 0.1, 0.6).unwrap();
        transform_solid(topo, drill, &Mat4::translation(x, y, H - 0.5)).unwrap();
        solid = boolean(topo, BooleanOp::Cut, solid, drill).unwrap();
        assert!(solid_faces(topo, solid).unwrap().len() > faces);
    }
    transform_solid(topo, solid, pose).unwrap();
    solid
}

/// The top face, read in the disc's own frame.
fn top(topo: &Topology, solid: SolidId, pose: &Mat4) -> FaceId {
    let inverse = pose.inverse().unwrap();
    solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&f| {
            let face = topo.face(f).unwrap();
            matches!(face.surface(), FaceSurface::Plane { .. })
                && topo
                    .wire(face.outer_wire())
                    .unwrap()
                    .edges()
                    .iter()
                    .all(|oe| {
                        let e = topo.edge(oe.edge()).unwrap();
                        [e.start(), e.end()].iter().all(|&v| {
                            (inverse.mul_point(topo.vertex(v).unwrap().point()).z() - H).abs()
                                < 1e-9
                        })
                    })
        })
        .unwrap()
}

/// The top outline's edges: the top face's outer wire.
fn outline(topo: &Topology, solid: SolidId, pose: &Mat4) -> Vec<EdgeId> {
    let face = topo.face(top(topo, solid, pose)).unwrap();
    topo.wire(face.outer_wire())
        .unwrap()
        .edges()
        .iter()
        .map(brepkit_topology::wire::OrientedEdge::edge)
        .collect()
}

/// Upright, turned about one axis and about all three, and mirrored.
fn poses() -> Vec<(&'static str, Mat4)> {
    vec![
        ("upright", Mat4::translation(0.0, 0.0, 0.0)),
        ("turned", Mat4::rotation_z(0.7)),
        (
            "tilted",
            Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
        ),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ]
}

/// The chamfered disc's volume: the top's area times the height, less the
/// integral of the top's area less the inset's over the inset.
fn chamfered(d: f64) -> f64 {
    let mut topo = Topology::new();
    let pose = Mat4::translation(0.0, 0.0, 0.0);
    let solid = notched_disc(&mut topo, None, &pose);
    let runs = flat_loop(&topo, top(&topo, solid, &pose));
    let whole = inset_area(&runs, 0.0);
    assert!(whole > 0.0, "the outline runs counter-clockwise");
    let taken: f64 = gauss_legendre_points(20)
        .iter()
        .map(|g| {
            let s = 0.5 * d * (g.x + 1.0);
            0.5 * d * g.w * (whole - inset_area(&runs, s))
        })
        .sum();
    whole * H - taken
}

/// The outline at 0.5: the notch's sides meet at the inside corner in a
/// mitre past both their ends, and meet the rim's arcs at the convex ones.
#[test]
fn a_notched_discs_outline_takes_a_chamfer_round_its_inside_corner() {
    let d = 0.5;
    let expected = chamfered(d);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let solid = notched_disc(&mut topo, None, &pose);
        let edges = outline(&topo, solid, &pose);

        let result = chamfer(&mut topo, solid, &edges, d).unwrap();

        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        assert!(
            is_watertight(&tessellate_solid(&topo, result, 0.01).unwrap()),
            "{name}"
        );
        let volume = solid_volume(&topo, result, 0.001).unwrap();
        assert!(
            (volume - expected).abs() < 1e-9 * expected,
            "{name}: volume {volume}, expected {expected}"
        );
    }
}

/// A pocket in the top inside the square the inside corner's chamfers take
/// past both their edges' ends: the chamfer would cut into the pocket, so
/// the pick is refused.
#[test]
fn a_pocket_in_the_inside_corners_square_refuses_the_chamfer() {
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let solid = notched_disc(&mut topo, Some([1.75, 1.75]), &pose);
        let edges = outline(&topo, solid, &pose);
        assert!(chamfer(&mut topo, solid, &edges, 0.5).is_err(), "{name}");
    }
}

/// The notched disc with a blind hole of radius 0.1 drilled up from its
/// bottom to `top`, under the rim at radius 9.65, placed by `pose`.
fn drilled_disc(topo: &mut Topology, top: f64, pose: &Mat4) -> SolidId {
    let solid = notched_disc(topo, None, &Mat4::translation(0.0, 0.0, 0.0));
    let faces = solid_faces(topo, solid).unwrap().len();
    let drill = make_cylinder(topo, 0.1, top + 1.0).unwrap();
    let (s, c) = 2.5_f64.sin_cos();
    transform_solid(topo, drill, &Mat4::translation(9.65 * c, 9.65 * s, -1.0)).unwrap();
    let drilled = boolean(topo, BooleanOp::Cut, solid, drill).unwrap();
    assert!(
        solid_faces(topo, drilled).unwrap().len() > faces,
        "the hole cuts"
    );
    transform_solid(topo, drilled, pose).unwrap();
    drilled
}

/// A hole drilled up from the bottom into the rim's chamfer touches neither
/// the top nor the wall, but the chamfer would cut through its end: the
/// pick is refused. A hole that stops short of the chamfer leaves it whole,
/// even where its end lies within the chamfer's distance of the rim (at
/// 2.6 its end reaches `r + z = 12.39`, where the chamfer's cone is at
/// 12.5).
#[test]
fn a_hole_from_below_into_the_rims_chamfer_refuses_it() {
    let d = 0.5;
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let into = drilled_disc(&mut topo, 2.8, &pose);
        let edges = outline(&topo, into, &pose);
        assert!(chamfer(&mut topo, into, &edges, d).is_err(), "{name}");

        for top in [2.0, 2.6] {
            let kept = chamfered(d) - PI * 0.01 * top;
            let short = drilled_disc(&mut topo, top, &pose);
            let edges = outline(&topo, short, &pose);
            let result = chamfer(&mut topo, short, &edges, d).unwrap();
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{name} {top}: {:?}", report.issues);
            let volume = solid_volume(&topo, result, 0.001).unwrap();
            assert!(
                (volume - kept).abs() < 1e-9 * kept,
                "{name} {top}: volume {volume}, expected {kept}"
            );
        }
    }
}

/// A notch whose inside corner sits at `(7, 7)`, its sides 0.14 long: a
/// chamfer of 0.5 would run each side's contacts back past their own ends,
/// so the pick is refused.
#[test]
fn a_notch_whose_sides_are_shorter_than_the_chamfer_refuses_it() {
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let solid = notched_at(&mut topo, 7.0, None, &pose);
        let edges = outline(&topo, solid, &pose);
        assert!(chamfer(&mut topo, solid, &edges, 0.5).is_err(), "{name}");
    }
}

/// A rod of radius 5, 10 high, turned 0.3 about its axis so that its seam
/// lies inside the rim below, less the quadrant `x, y > 0` above `z = 3`:
/// the quadrant's rim arc at `z = 3` has its wall below it, though the same
/// wall rises to 10 elsewhere. Its chamfer takes the ring's triangle over a
/// quarter turn, square at both ends where the steps stand on radial planes.
#[test]
fn a_rim_whose_wall_rises_past_it_elsewhere_takes_its_chamfer_below() {
    let (r, d) = (5.0_f64, 0.5_f64);
    let plain = PI * r * r * (3.0 + 0.75 * 7.0);
    let taken = 0.5 * d * d * 0.5 * PI * (r - d / 3.0);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, r, 10.0).unwrap();
        transform_solid(&mut topo, rod, &Mat4::rotation_z(0.3)).unwrap();
        let cut = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        transform_solid(&mut topo, cut, &Mat4::translation(0.0, 0.0, 3.0)).unwrap();
        let stepped = boolean(&mut topo, BooleanOp::Cut, rod, cut).unwrap();
        let rim: Vec<EdgeId> = brepkit_topology::explorer::solid_edges(&topo, stepped)
            .unwrap()
            .into_iter()
            .filter(|&e| {
                let edge = topo.edge(e).unwrap();
                let a = topo.vertex(edge.start()).unwrap().point();
                let b = topo.vertex(edge.end()).unwrap().point();
                matches!(edge.curve(), EdgeCurve::Circle(_))
                    && (a.z() - 3.0).abs() < 1e-9
                    && (b.z() - 3.0).abs() < 1e-9
            })
            .collect();
        assert_eq!(rim.len(), 2, "{name}: the rim, split at the seam");
        transform_solid(&mut topo, stepped, &pose).unwrap();

        let result = chamfer(&mut topo, stepped, &rim, d).unwrap();

        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        assert!(
            is_watertight(&tessellate_solid(&topo, result, 0.01).unwrap()),
            "{name}"
        );
        let volume = solid_volume(&topo, result, 0.001).unwrap();
        let expected = plain - taken;
        assert!(
            (volume - expected).abs() < 1e-9 * expected,
            "{name}: volume {volume}, expected {expected}"
        );
    }
}
