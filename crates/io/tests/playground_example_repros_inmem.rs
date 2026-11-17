//! Captured operands of three calls from the brepjs playground examples
//! (#2372). Each test states what the call must give; a test whose call
//! still fails is ignored until its issue closes.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::{PI, TAU};
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::mat::Mat4;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

fn load(name: &str, topo: &mut Topology) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn assert_holds(topo: &Topology, solid: SolidId) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "invalid: {report:?}");
    assert!(
        is_watertight(&tessellate_solid(topo, solid, 0.05).unwrap()),
        "mesh not watertight"
    );
}

fn class_of(topo: &Topology, solid: SolidId, at: [f64; 3]) -> PointClassification {
    let tol = brepkit_math::tolerance::Tolerance::new();
    classify_point(
        topo,
        solid,
        Point3::new(at[0], at[1], at[2]),
        0.01,
        tol.linear,
    )
    .unwrap()
}

/// Whether `solid` holds `at`; `None` on its boundary.
fn holds(topo: &Topology, solid: SolidId, at: [f64; 3]) -> Option<bool> {
    match class_of(topo, solid, at) {
        PointClassification::Inside => Some(true),
        PointClassification::Outside => Some(false),
        PointClassification::OnBoundary => None,
    }
}

/// A box of grid points: `n` steps along each axis from `lo` to `hi`.
struct Grid {
    lo: [f64; 3],
    hi: [f64; 3],
    n: usize,
}

impl Grid {
    fn points(&self) -> impl Iterator<Item = [f64; 3]> + '_ {
        let n = self.n;
        let at = move |k: usize, i: usize| {
            #[allow(clippy::cast_precision_loss)]
            let f = (i as f64 + 0.5) / n as f64;
            self.lo[k] + (self.hi[k] - self.lo[k]) * f
        };
        (0..n).flat_map(move |i| {
            (0..n).flat_map(move |j| (0..n).map(move |k| [at(0, i), at(1, j), at(2, k)]))
        })
    }
}

/// The exact fuse of `a` and `b`: a valid, watertight solid that keeps a
/// curved face, holds `in_a` (in `a` alone) and `in_b` (in `b` alone), and
/// does not hold `off`. Its volume is the operands' less their common part,
/// and over `grid` it holds a point exactly where either operand does.
fn fuse_exactly(topo: &mut Topology, a: SolidId, b: SolidId, probes: [[f64; 3]; 3], grid: &Grid) {
    fuse_exactly_posed(topo, a, b, probes, grid, &Mat4::identity());
}

/// [`fuse_exactly`] for operands moved by `pose`, the probes and the grid
/// given where they lie before the move.
fn fuse_exactly_posed(
    topo: &mut Topology,
    a: SolidId,
    b: SolidId,
    probes: [[f64; 3]; 3],
    grid: &Grid,
    pose: &Mat4,
) {
    use PointClassification::{Inside, Outside};
    let moved = |p: [f64; 3]| {
        let q = pose.mul_point(Point3::new(p[0], p[1], p[2]));
        [q.x(), q.y(), q.z()]
    };
    let [in_a, in_b, off] = probes.map(moved);
    for (solid, own, other) in [(a, in_a, in_b), (b, in_b, in_a)] {
        assert_eq!(class_of(topo, solid, own), Inside, "probe {own:?}");
        assert_eq!(class_of(topo, solid, other), Outside, "probe {other:?}");
        assert_eq!(class_of(topo, solid, off), Outside, "probe {off:?}");
    }
    let exact = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    let fused = boolean_with_options(topo, BooleanOp::Fuse, a, b, exact).unwrap();
    assert_holds(topo, fused);
    assert!(
        solid_faces(topo, fused).unwrap().iter().any(|&f| !topo
            .face(f)
            .unwrap()
            .surface()
            .is_planar()),
        "no curved face is left"
    );
    assert_eq!(class_of(topo, fused, in_a), Inside, "material of the first");
    assert_eq!(
        class_of(topo, fused, in_b),
        Inside,
        "material of the second"
    );
    assert_eq!(class_of(topo, fused, off), Outside, "a point off the two");

    let common = boolean_with_options(topo, BooleanOp::Intersect, a, b, exact).unwrap();
    let volume = |s| solid_volume(topo, s, 0.01).unwrap();
    let (va, vb, vc, vf) = (volume(a), volume(b), volume(common), volume(fused));
    assert!(
        (vf - (va + vb - vc)).abs() < 1e-6 * vf,
        "fused {vf}, operands {va} + {vb} less their common {vc}"
    );

    let mut read = 0;
    for p in grid.points().map(moved) {
        let (Some(f), Some(ia), Some(ib)) =
            (holds(topo, fused, p), holds(topo, a, p), holds(topo, b, p))
        else {
            continue;
        };
        assert_eq!(f, ia || ib, "at {p:?}");
        read += 1;
    }
    assert!(read > grid.n.pow(3) / 2, "only {read} points read");
}

/// `mars-rover`: a wheel's thin shell (radii 249.25 and 250) with four other
/// parts of the wheel fused into it, and a grouser: a box turned 9 degrees
/// about the radius, that passes through the shell's wall.
#[test]
fn a_grouser_fuses_onto_a_wheel() {
    let mut topo = Topology::new();
    let wheel = load("mars_rover_wheel.bin", &mut topo);
    let grouser = load("mars_rover_grouser.bin", &mut topo);
    // In the shell's wall away from the grouser, in the grouser past the
    // shell, and in the bore. The grid holds where the grouser crosses the
    // shell's wall and the ring inside it at z = 60 to 64.
    fuse_exactly(
        &mut topo,
        wheel,
        grouser,
        [[0.0, 249.6, 0.0], [255.0, 15.84, 100.0], [0.0, 0.0, 0.0]],
        &Grid {
            lo: [248.6, 3.0, 50.0],
            hi: [258.0, 16.0, 95.0],
            n: 20,
        },
    );
}

/// [`a_grouser_fuses_onto_a_wheel`] with the wheel and the grouser turned
/// together so that the wheel's axis leans off every coordinate axis. The
/// circle of radius 249.25 on each face of the ring crosses the grouser's
/// face nearest the axis at 2.6 degrees, and its arc from there to the
/// grouser's side stands up to 0.166 off that face.
#[test]
fn a_grouser_fuses_onto_a_tilted_wheel() {
    for pose in [
        Mat4::rotation_x(0.37),
        Mat4::rotation_y(-0.81),
        Mat4::rotation_x(0.37) * Mat4::rotation_y(-0.81) * Mat4::rotation_z(1.13),
    ] {
        let mut topo = Topology::new();
        let wheel = load("mars_rover_wheel.bin", &mut topo);
        let grouser = load("mars_rover_grouser.bin", &mut topo);
        for solid in [wheel, grouser] {
            brepkit_operations::transform::transform_solid(&mut topo, solid, &pose).unwrap();
        }
        fuse_exactly_posed(
            &mut topo,
            wheel,
            grouser,
            [[0.0, 249.6, 0.0], [255.0, 15.84, 100.0], [0.0, 0.0, 0.0]],
            &Grid {
                lo: [248.6, 3.0, 50.0],
                hi: [258.0, 16.0, 95.0],
                n: 20,
            },
            &pose,
        );
    }
}

/// `robot-arm`: the turret's cap, a cylinder with a cone on it, and a yoke
/// whose arms have rounded edges. The yoke's flat bottom lies in the plane
/// where the cylinder meets the cone.
#[test]
fn a_yoke_with_rounded_arms_fuses_onto_the_turrets_cap() {
    let mut topo = Topology::new();
    let cap = load("robot_arm_shoulder_cap.bin", &mut topo);
    let yoke = load("robot_arm_shoulder_yoke.bin", &mut topo);
    // In the cap under the yoke, in the middle of one arm of the yoke, and
    // beside that arm. The grid holds the cap's cone where one round of the
    // yoke and its side cylinder pass through it.
    fuse_exactly(
        &mut topo,
        cap,
        yoke,
        [[0.0, 0.0, 170.0], [0.0, -49.0, 250.0], [0.0, -60.0, 250.0]],
        &Grid {
            lo: [28.0, -52.0, 176.0],
            hi: [64.0, 0.0, 194.0],
            n: 16,
        },
    );
}

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

/// `pocket-watch`: a bridge plate whose top outline, at z = 3.15, is 6 lines
/// and 7 circle arcs, two of the lines meeting an arc at an angle. Its side
/// walls stand square to the top and meet along straight edges stored as
/// degree-1 NURBS curves. The cross-section `h` below the top is the outline
/// inset by `d - h`, so the chamfer takes the integral of the top's area
/// less the inset's over the inset from 0 to `d`.
#[test]
fn a_plates_top_outline_of_lines_and_arcs_takes_a_chamfer() {
    let mut topo = Topology::new();
    let plate = load("pocket_watch_bridge.bin", &mut topo);
    let outline: Vec<_> = solid_edges(&topo, plate)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()]
                .iter()
                .all(|&v| (topo.vertex(v).unwrap().point().z() - 3.15).abs() < 1e-9)
        })
        .collect();
    assert_eq!(outline.len(), 13);
    let top = solid_faces(&topo, plate)
        .unwrap()
        .into_iter()
        .find(|&f| match topo.face(f).unwrap().surface() {
            FaceSurface::Plane { normal, d } => normal.z() > 0.5 && (d - 3.15).abs() < 1e-9,
            _ => false,
        })
        .unwrap();
    let runs = flat_loop(&topo, top);
    let d = 0.1;
    let whole = inset_area(&runs, 0.0);
    let taken: f64 = gauss_legendre_points(20)
        .iter()
        .map(|g| {
            let s = 0.5 * d * (g.x + 1.0);
            0.5 * d * g.w * (whole - inset_area(&runs, s))
        })
        .sum();

    let before = solid_volume(&topo, plate, 1e-3).unwrap();
    let chamfered = chamfer(&mut topo, plate, &outline, d).unwrap();
    assert_holds(&topo, chamfered);
    assert_eq!(solid_faces(&topo, chamfered).unwrap().len(), 15 + 13);
    let after = solid_volume(&topo, chamfered, 1e-3).unwrap();
    assert!(
        (before - after - taken).abs() < 1e-9 * before,
        "the chamfer takes {} where {taken} is right",
        before - after
    );
}

/// `axial-fan`: the frame's shell (flanges and a duct of bore 38, with the
/// lead slot cut through the duct's wall from z = 3 to 4.5) and the wide
/// strut, 4.4 wide from z = 0 to 3, under the slot. The strut's footprint on
/// the bore meets the slot's window along the window's bottom edge.
#[test]
fn a_strut_fuses_into_the_fans_frame_under_its_slot() {
    let mut topo = Topology::new();
    let shell = load("axial_fan_frame_shell.bin", &mut topo);
    let strut = load("axial_fan_wide_strut.bin", &mut topo);
    // In the duct's wall, in the strut inside the bore, and on the axis.
    fuse_exactly(
        &mut topo,
        shell,
        strut,
        [[38.7, 0.0, 12.0], [13.435, -21.92, 1.5], [0.0, 0.0, 12.0]],
        &Grid {
            lo: [18.6, -35.1, 0.5],
            hi: [26.6, -27.1, 5.5],
            n: 16,
        },
    );
}

/// `universal-joint`: a yoke's fork (two ears and a bridge, 68 planes) and
/// the turned outside it is trimmed to, a cylinder of radius 47 that tapers
/// along a cone to radius 34 at x = -62. The bridge's corners stand outside
/// the cone.
#[test]
fn a_yokes_fork_trims_to_its_turned_outside_exactly() {
    use PointClassification::{Inside, Outside};
    let exact = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    let mut topo = Topology::new();
    let fork = load("universal_joint_fork.bin", &mut topo);
    let outside = load("universal_joint_turned_outside.bin", &mut topo);
    let yoke = boolean_with_options(&mut topo, BooleanOp::Intersect, fork, outside, exact).unwrap();
    assert_holds(&topo, yoke);
    assert!(
        solid_faces(&topo, yoke).unwrap().iter().any(|&f| topo
            .face(f)
            .unwrap()
            .surface()
            .type_tag()
            == "cone"),
        "the taper is gone"
    );
    // In the bridge inside the cone, at a bridge corner outside it, and in
    // an ear inside the cylinder.
    for (p, want) in [
        ([-55.0, 0.0, 30.0], Inside),
        ([-55.0, 0.0, 44.0], Outside),
        ([0.0, 18.0, 40.0], Inside),
    ] {
        assert_eq!(class_of(&topo, yoke, p), want, "at {p:?}");
    }
    let grid = Grid {
        lo: [-63.0, -22.0, -46.0],
        hi: [-37.0, 22.0, 46.0],
        n: 14,
    };
    let mut read = 0;
    for p in grid.points() {
        let (Some(y), Some(f), Some(o)) = (
            holds(&topo, yoke, p),
            holds(&topo, fork, p),
            holds(&topo, outside, p),
        ) else {
            continue;
        };
        assert_eq!(y, f && o, "at {p:?}");
        read += 1;
    }
    assert!(read > grid.n.pow(3) / 2, "only {read} points read");
}
