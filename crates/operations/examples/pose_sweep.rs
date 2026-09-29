//! Booleans between primitives in several poses, one line per result, for
//! diffing a branch's output against main's before opening a PR:
//!
//! ```text
//! cargo run --release --example pose_sweep -p brepkit-operations > main.txt
//! # on the branch
//! cargo run --release --example pose_sweep -p brepkit-operations > branch.txt
//! diff main.txt branch.txt
//! ```
//!
//! Each line reads the case, the operation (`a-b`, `a&b`, `b-a`), the pose
//! of the whole scene, the face count (`fallback` for the mesh boolean's
//! all-plane solid, `error` when the boolean fails), whether `validate_solid`
//! accepts the result (`valid`) and its mesh is closed (`closed`), its volume
//! and its error against a closed form where the case has one, the check
//! crate's volume against the same closed form (or against the operations
//! volume where there is none), and how many
//! points of a grid over the operands the engine's ray cast reads against
//! their analytic distances, points near either surface skipped. A case
//! whose name carries a pose (`ball rx0.35 | ...`) poses that operand alone.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    missing_docs
)]

use std::f64::consts::PI;

use brepkit_algo::FaceClass;
use brepkit_algo::classifier::{RayCastGeoms, classify_ray_cast_cached};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

/// Grid points per axis, and the distance from either surface within which
/// a point is not read.
const GRID: u32 = 13;
const NEAR: f64 = 0.02;

#[derive(Clone, Copy)]
enum Prim {
    /// `make_sphere(r, segments)` about the origin.
    Ball { r: f64, segments: usize },
    /// `make_box` from `lo` to `hi`.
    Block { lo: [f64; 3], hi: [f64; 3] },
    /// `make_cylinder(r, z1 - z0)` about the vertical through `at`.
    Rod {
        r: f64,
        at: (f64, f64),
        z: (f64, f64),
    },
    /// `make_cone(r0, r1, h)` on the origin.
    Cone { r0: f64, r1: f64, h: f64 },
}

impl Prim {
    /// Signed distance (negative inside; exact enough near the surface to
    /// skip points there).
    fn distance(self, p: Point3) -> f64 {
        match self {
            Self::Ball { r, .. } => (p - Point3::new(0.0, 0.0, 0.0)).length() - r,
            Self::Block { lo, hi } => {
                let c = [p.x(), p.y(), p.z()];
                (0..3).fold(f64::NEG_INFINITY, |d, k| {
                    d.max(lo[k] - c[k]).max(c[k] - hi[k])
                })
            }
            Self::Rod { r, at, z } => ((p.x() - at.0).hypot(p.y() - at.1) - r)
                .max(z.0 - p.z())
                .max(p.z() - z.1),
            Self::Cone { r0, r1, h } => {
                let slope = (r0 - r1) / h;
                let side = (p.x().hypot(p.y()) - slope.mul_add(-p.z(), r0)) / slope.hypot(1.0);
                side.max(-p.z()).max(p.z() - h)
            }
        }
    }

    fn build(self, topo: &mut Topology) -> SolidId {
        match self {
            Self::Ball { r, segments } => make_sphere(topo, r, segments).unwrap(),
            Self::Block { lo, hi } => {
                let b = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
                transform_solid(topo, b, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
                b
            }
            Self::Rod { r, at, z } => {
                let c = make_cylinder(topo, r, z.1 - z.0).unwrap();
                transform_solid(topo, c, &Mat4::translation(at.0, at.1, z.0)).unwrap();
                c
            }
            Self::Cone { r0, r1, h } => make_cone(topo, r0, r1, h).unwrap(),
        }
    }

    /// Corners of a box holding it.
    fn corners(self) -> Vec<Point3> {
        let (lo, hi) = match self {
            Self::Ball { r, .. } => ([-r; 3], [r; 3]),
            Self::Block { lo, hi } => (lo, hi),
            Self::Rod { r, at, z } => ([at.0 - r, at.1 - r, z.0], [at.0 + r, at.1 + r, z.1]),
            Self::Cone { r0, r1, h } => {
                let r = r0.max(r1);
                ([-r, -r, 0.0], [r, r, h])
            }
        };
        (0..8)
            .map(|k| {
                let pick = |i: usize| if k >> i & 1 == 0 { lo[i] } else { hi[i] };
                Point3::new(pick(0), pick(1), pick(2))
            })
            .collect()
    }
}

/// A primitive placed by `pose`.
#[derive(Clone, Copy)]
struct Part {
    prim: Prim,
    pose: Mat4,
}

fn part(prim: Prim) -> Part {
    Part {
        prim,
        pose: Mat4::identity(),
    }
}

fn posed(prim: Prim, pose: Mat4) -> Part {
    Part { prim, pose }
}

/// An operand: the union of its parts.
struct Operand(Vec<Part>);

impl Operand {
    fn distance(&self, scene: Mat4, p: Point3) -> f64 {
        self.0
            .iter()
            .map(|part| {
                let back = (scene * part.pose).inverse().unwrap();
                part.prim.distance(back.mul_point(p))
            })
            .fold(f64::INFINITY, f64::min)
    }

    fn build(&self, topo: &mut Topology, scene: Mat4) -> SolidId {
        let mut solid: Option<SolidId> = None;
        for part in &self.0 {
            let s = part.prim.build(topo);
            transform_solid(topo, s, &(scene * part.pose)).unwrap();
            solid = Some(match solid {
                None => s,
                Some(acc) => boolean(topo, BooleanOp::Fuse, acc, s).unwrap(),
            });
        }
        solid.unwrap()
    }

    fn corners(&self, scene: Mat4) -> Vec<Point3> {
        self.0
            .iter()
            .flat_map(|part| {
                let pose = scene * part.pose;
                part.prim
                    .corners()
                    .into_iter()
                    .map(move |c| pose.mul_point(c))
            })
            .collect()
    }
}

struct Case {
    name: &'static str,
    a: Operand,
    b: Operand,
    /// Volumes of `a - b`, `a & b` and `b - a`.
    truth: Option<[f64; 3]>,
}

/// The frustum `make_cone(5, 2, 10)` within the box `|x|, |y| < 3`: its
/// section is a disc while `r = 5 - 0.3 z` is at most 3, the disc less four
/// segments up to `3 sqrt(2)`, then the whole square, integrated in `z`
/// (Simpson over each stretch, 20000 panels).
const FRUSTUM_IN_BOX_3: f64 = 296.604_274_040_955_94;

/// The cap of a ball of radius 3 of height `h`.
fn cap(h: f64) -> f64 {
    PI * h * h * (9.0 - h) / 3.0
}

/// The four caps of a ball of radius 3 past the walls `|x|, |y| = 3 - h`.
fn caps(h: f64) -> f64 {
    4.0 * PI * h * h * (9.0 - h) / 3.0
}

/// The ball of radius 3 less, within and outside a square column of half
/// width `a` from `z = -5` to `top` (through the ball when `top >= 3`).
fn ball_column(a: f64, top: f64) -> [f64; 3] {
    let ball = 36.0 * PI;
    let h = (3.0 - top).max(0.0);
    let less = caps(3.0 - a) + PI * h * h * (9.0 - h) / 3.0;
    [less, ball - less, 4.0 * a * a * (5.0 + top) - (ball - less)]
}

/// The ball of radius 3 less, within and outside the column `|x|, |y| < a`,
/// `|z| < 5`, whose corners lie inside the ball: its chord `2 sqrt(9 - r^2)`
/// integrated over the square (Simpson, 1000 panels a side).
fn ball_narrow_column(a: f64) -> [f64; 3] {
    let n = 1000_u32;
    let step = 2.0 * a / f64::from(n);
    let weight = |k: u32| match k {
        0 => 1.0,
        k if k == n => 1.0,
        k if k % 2 == 1 => 4.0,
        _ => 2.0,
    };
    let mut sum = 0.0;
    for i in 0..=n {
        let x = step.mul_add(f64::from(i), -a);
        for j in 0..=n {
            let y = step.mul_add(f64::from(j), -a);
            sum += weight(i) * weight(j) * 2.0 * (9.0 - x * x - y * y).sqrt();
        }
    }
    let within = sum * step * step / 9.0;
    [36.0 * PI - within, within, 40.0 * a * a - within]
}

/// The ball of radius 3 and the column of half width 2.5 through it fused
/// with a rod of radius `r` along `z` at `(at, 0)` past the wall.
fn ball_column_rod(r: f64, at: f64) -> [f64; 3] {
    let simpson = |n: u32, lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| {
        let step = (hi - lo) / f64::from(n);
        let mut sum = f(lo) + f(hi);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step.mul_add(f64::from(k), lo));
        }
        sum * step / 3.0
    };
    let bore = simpson(200, 0.0, r, &|q: f64| {
        q * simpson(200, 0.0, 2.0 * PI, &|th: f64| {
            let (x, y) = (q.mul_add(th.cos(), at), q * th.sin());
            2.0 * (9.0 - x * x - y * y).max(0.0).sqrt()
        })
    });
    let [less, within, _] = ball_column(2.5, 5.0);
    let within = within + bore;
    [
        less - bore,
        within,
        PI.mul_add(r * r * 10.0, 250.0) - within,
    ]
}

/// A ball of radius `r` about the origin less, within and outside the union
/// of `blocks` (disjoint boxes, each from `lo` to `hi`): each slice across
/// `z` meets a box in a disc-and-rectangle area taken in closed form,
/// integrated by 5-point Gauss over 2000 panels between the heights where a
/// wall or a vertical edge of the box meets the slice's rim.
fn ball_and_blocks(r: f64, blocks: &[([f64; 3], [f64; 3])]) -> [f64; 3] {
    const GAUSS: [(f64, f64); 5] = [
        (-0.906_179_845_938_664, 0.236_926_885_056_189_1),
        (-0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
        (0.0, 0.568_888_888_888_888_9),
        (0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
        (0.906_179_845_938_664, 0.236_926_885_056_189_1),
    ];
    let within_block = |lo: [f64; 3], hi: [f64; 3]| {
        let slice = |rho: f64| {
            let (a, b) = (lo[0].max(-rho), hi[0].min(rho));
            if a >= b {
                return 0.0;
            }
            let half = |x: f64| ((rho - x) * (rho + x)).max(0.0).sqrt();
            let under = |x: f64| {
                let x = x.clamp(-rho, rho);
                0.5 * x.mul_add(half(x), rho * rho * (x / rho).asin())
            };
            let mut cuts = vec![a, b];
            for y in [lo[1], hi[1]] {
                if y.abs() < rho {
                    cuts.extend([-half(y), half(y)].into_iter().filter(|&x| x > a && x < b));
                }
            }
            cuts.sort_by(f64::total_cmp);
            cuts.windows(2)
                .map(|w| {
                    let (p, q) = (w[0], w[1]);
                    let s = half(0.5 * (p + q));
                    if hi[1].min(s) <= lo[1].max(-s) {
                        return 0.0;
                    }
                    let arc = under(q) - under(p);
                    let top = if hi[1] < s { hi[1] * (q - p) } else { arc };
                    let bottom = if lo[1] > -s { lo[1] * (q - p) } else { -arc };
                    top - bottom
                })
                .sum::<f64>()
        };
        let (z0, z1) = (lo[2].max(-r), hi[2].min(r));
        if z0 >= z1 {
            return 0.0;
        }
        let mut reach: Vec<f64> = [lo[0], hi[0], lo[1], hi[1]].iter().map(|c| c * c).collect();
        for x in [lo[0], hi[0]] {
            for y in [lo[1], hi[1]] {
                reach.push(x.mul_add(x, y * y));
            }
        }
        let mut cuts = vec![z0, z1];
        for c in reach.into_iter().filter(|&c| c < r * r) {
            let z = (r * r - c).sqrt();
            cuts.extend([-z, z].into_iter().filter(|&z| z > z0 && z < z1));
        }
        cuts.sort_by(f64::total_cmp);
        let panels = 2000_u32;
        let mut total = 0.0;
        for w in cuts.windows(2) {
            let h = (w[1] - w[0]) / f64::from(panels);
            for k in 0..panels {
                let mid = h.mul_add(f64::from(k) + 0.5, w[0]);
                for (t, weight) in GAUSS {
                    let z = (0.5 * h).mul_add(t, mid);
                    total += weight * 0.5 * h * slice(((r - z) * (r + z)).max(0.0).sqrt());
                }
            }
        }
        total
    };
    let within: f64 = blocks.iter().map(|&(lo, hi)| within_block(lo, hi)).sum();
    let block: f64 = blocks
        .iter()
        .map(|(lo, hi)| (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]))
        .sum();
    [4.0 * PI * r.powi(3) / 3.0 - within, within, block - within]
}

#[allow(clippy::too_many_lines)]
fn cases() -> Vec<Case> {
    let ball = Prim::Ball {
        r: 3.0,
        segments: 32,
    };
    let column = |a: f64, top: f64| Prim::Block {
        lo: [-a, -a, -5.0],
        hi: [a, a, top],
    };
    let rod = Prim::Rod {
        r: 0.1,
        at: (2.75, 0.0),
        z: (-5.0, 5.0),
    };
    let slab = posed(
        Prim::Block {
            lo: [0.0; 3],
            hi: [20.0, 20.0, 10.0],
        },
        Mat4::translation(0.0, 0.0, 7.0)
            * Mat4::rotation_x(0.4)
            * Mat4::translation(-10.0, -10.0, 0.0),
    );
    let frustum = Prim::Cone {
        r0: 5.0,
        r1: 2.0,
        h: 10.0,
    };
    let block = |lo: [f64; 3], hi: [f64; 3]| Prim::Block { lo, hi };
    let wedge = block([0.0, 0.0, -5.0], [5.0; 3]);
    vec![
        Case {
            name: "ball | column 2.5",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(column(2.5, 5.0))]),
            truth: Some(ball_column(2.5, 5.0)),
        },
        Case {
            name: "ball | column 2.2",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(column(2.2, 5.0))]),
            truth: Some(ball_column(2.2, 5.0)),
        },
        Case {
            name: "ball | column 2.5 to 2.8",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(column(2.5, 2.8))]),
            truth: Some(ball_column(2.5, 2.8)),
        },
        Case {
            name: "ball | column 2.5 to 2",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(column(2.5, 2.0))]),
            truth: Some(ball_column(2.5, 2.0)),
        },
        Case {
            name: "ball | column 1.8",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(column(1.8, 5.0))]),
            truth: Some(ball_narrow_column(1.8)),
        },
        Case {
            name: "ball | column -2.5 to 2",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(Prim::Block {
                lo: [-2.5, -2.5, -5.0],
                hi: [2.0, 2.0, 5.0],
            })]),
            truth: None,
        },
        Case {
            name: "ball | column -2.6,-2.2 rz0.2",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![posed(
                Prim::Block {
                    lo: [-2.6, -2.2, -5.0],
                    hi: [1.9, 2.3, 5.0],
                },
                Mat4::rotation_z(0.2),
            )]),
            truth: None,
        },
        Case {
            name: "ball | column 2.05 from -1",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(Prim::Block {
                lo: [-2.05, -2.05, -1.0],
                hi: [2.05, 2.05, 9.0],
            })]),
            truth: None,
        },
        Case {
            name: "ball | x > 0.5",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(Prim::Block {
                lo: [0.5, -5.0, -5.0],
                hi: [10.5, 5.0, 5.0],
            })]),
            truth: Some([36.0 * PI - cap(2.5), cap(2.5), 1000.0 - cap(2.5)]),
        },
        Case {
            name: "ball | column 4.1x6.1 from -1",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(Prim::Block {
                lo: [-2.05, -2.05, -1.0],
                hi: [2.05, 4.05, 9.0],
            })]),
            truth: None,
        },
        Case {
            name: "ball | column 2.5 + rod",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(column(2.5, 5.0)), part(rod)]),
            truth: Some(ball_column_rod(0.1, 2.75)),
        },
        Case {
            name: "ball rx0.35 | column 2.5",
            a: Operand(vec![posed(ball, Mat4::rotation_x(0.35))]),
            b: Operand(vec![part(column(2.5, 5.0))]),
            truth: Some(ball_column(2.5, 5.0)),
        },
        Case {
            name: "ball rx0.35 | column 2.5 to 2.8",
            a: Operand(vec![posed(ball, Mat4::rotation_x(0.35))]),
            b: Operand(vec![part(column(2.5, 2.8))]),
            truth: Some(ball_column(2.5, 2.8)),
        },
        Case {
            name: "ball | column rx-0.2 to 2.93093",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![posed(column(2.5, 2.930_93), Mat4::rotation_x(-0.2))]),
            truth: Some(ball_column(2.5, 2.930_93)),
        },
        Case {
            name: "ball 1 | octant",
            a: Operand(vec![part(Prim::Ball {
                r: 1.0,
                segments: 16,
            })]),
            b: Operand(vec![part(Prim::Block {
                lo: [0.0; 3],
                hi: [2.0; 3],
            })]),
            truth: Some([7.0 * PI / 6.0, PI / 6.0, 8.0 - PI / 6.0]),
        },
        Case {
            name: "ball | half x > 0",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(block([0.0, -5.0, -5.0], [10.0, 5.0, 5.0]))]),
            truth: Some([18.0 * PI, 18.0 * PI, 1000.0 - 18.0 * PI]),
        },
        Case {
            name: "ball | wedge x, y > 0",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(wedge)]),
            truth: Some([27.0 * PI, 9.0 * PI, 250.0 - 9.0 * PI]),
        },
        Case {
            name: "ball | wedge rz1",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![posed(wedge, Mat4::rotation_z(1.0))]),
            truth: Some([27.0 * PI, 9.0 * PI, 250.0 - 9.0 * PI]),
        },
        Case {
            name: "ball rx0.35 | wedge x, y > 0",
            a: Operand(vec![posed(ball, Mat4::rotation_x(0.35))]),
            b: Operand(vec![part(wedge)]),
            truth: Some([27.0 * PI, 9.0 * PI, 250.0 - 9.0 * PI]),
        },
        Case {
            name: "ball | x > 0, y > -1",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(block([0.0, -1.0, -5.0], [5.0; 3]))]),
            truth: Some(ball_and_blocks(3.0, &[([0.0, -1.0, -5.0], [5.0; 3])])),
        },
        Case {
            name: "ball | x > 0.01, y > -1",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(block([0.01, -1.0, -5.0], [5.0; 3]))]),
            truth: Some(ball_and_blocks(3.0, &[([0.01, -1.0, -5.0], [5.0; 3])])),
        },
        Case {
            name: "ball | x > 0.02, y > -1",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(block([0.02, -1.0, -5.0], [5.0; 3]))]),
            truth: Some(ball_and_blocks(3.0, &[([0.02, -1.0, -5.0], [5.0; 3])])),
        },
        Case {
            name: "ball | octant rz1",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![posed(
                block([0.0; 3], [5.0; 3]),
                Mat4::rotation_z(1.0),
            )]),
            truth: Some([31.5 * PI, 4.5 * PI, 125.0 - 4.5 * PI]),
        },
        Case {
            name: "ball | corner 0,0,-1",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(block([0.0, 0.0, -1.0], [5.0; 3]))]),
            truth: Some(ball_and_blocks(3.0, &[([0.0, 0.0, -1.0], [5.0; 3])])),
        },
        Case {
            name: "ball | corner -0.5,0.5,0",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(block([-0.5, 0.5, 0.0], [5.0; 3]))]),
            truth: Some(ball_and_blocks(3.0, &[([-0.5, 0.5, 0.0], [5.0; 3])])),
        },
        Case {
            name: "ball | y > 0, z > 1",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(block([-5.0, 0.0, 1.0], [5.0; 3]))]),
            truth: Some(ball_and_blocks(3.0, &[([-5.0, 0.0, 1.0], [5.0; 3])])),
        },
        Case {
            name: "ball | box top 2.95 round the pole",
            a: Operand(vec![part(ball)]),
            b: Operand(vec![part(block([-0.7, -1.1, 0.1], [9.3, 8.9, 2.95]))]),
            truth: Some(ball_and_blocks(
                3.0,
                &[([-0.7, -1.1, 0.1], [9.3, 8.9, 2.95])],
            )),
        },
        Case {
            name: "ball 10 | stepped block",
            a: Operand(vec![part(Prim::Ball {
                r: 10.0,
                segments: 32,
            })]),
            b: Operand(vec![
                part(block([0.0, -1.0, -20.0], [20.0, 20.0, 4.0])),
                part(block([0.0, -1.0, 4.0], [3.0, 20.0, 20.0])),
            ]),
            truth: Some(ball_and_blocks(
                10.0,
                &[
                    ([0.0, -1.0, -20.0], [20.0, 20.0, 4.0]),
                    ([0.0, -1.0, 4.0], [3.0, 20.0, 20.0]),
                ],
            )),
        },
        Case {
            name: "frustum | box 3",
            a: Operand(vec![part(frustum)]),
            b: Operand(vec![part(Prim::Block {
                lo: [-3.0, -3.0, -1.0],
                hi: [3.0, 3.0, 11.0],
            })]),
            truth: Some([
                130.0 * PI - FRUSTUM_IN_BOX_3,
                FRUSTUM_IN_BOX_3,
                432.0 - FRUSTUM_IN_BOX_3,
            ]),
        },
        Case {
            name: "frustum | x > 1",
            a: Operand(vec![part(frustum)]),
            b: Operand(vec![part(Prim::Block {
                lo: [1.0, -6.0, -1.0],
                hi: [6.0, 6.0, 11.0],
            })]),
            truth: None,
        },
        Case {
            name: "cone | x > 1",
            a: Operand(vec![part(Prim::Cone {
                r0: 5.0,
                r1: 0.0,
                h: 10.0,
            })]),
            b: Operand(vec![part(Prim::Block {
                lo: [1.0, -6.0, -1.0],
                hi: [6.0, 6.0, 11.0],
            })]),
            truth: None,
        },
        Case {
            name: "frustum | slant slab",
            a: Operand(vec![part(frustum)]),
            b: Operand(vec![slab]),
            truth: None,
        },
        Case {
            name: "cylinder | slant slab",
            a: Operand(vec![part(Prim::Rod {
                r: 5.0,
                at: (0.0, 0.0),
                z: (0.0, 10.0),
            })]),
            b: Operand(vec![slab]),
            truth: None,
        },
    ]
}

fn main() {
    let scenes = [
        ("id", Mat4::identity()),
        ("rz0.5", Mat4::rotation_z(0.5)),
        ("rx0.35", Mat4::rotation_x(0.35)),
        ("ry0.3rz1", Mat4::rotation_z(1.0) * Mat4::rotation_y(0.3)),
        ("mx", Mat4::scale(-1.0, 1.0, 1.0)),
    ];
    let ops = [
        ("a-b", BooleanOp::Cut, false),
        ("a&b", BooleanOp::Intersect, false),
        ("b-a", BooleanOp::Cut, true),
    ];
    let started = std::time::Instant::now();
    for case in cases() {
        for (scene_name, scene) in scenes {
            for (k, (op_name, op, swap)) in ops.into_iter().enumerate() {
                let mut topo = Topology::new();
                let a = case.a.build(&mut topo, scene);
                let b = case.b.build(&mut topo, scene);
                let (x, y) = if swap { (b, a) } else { (a, b) };
                let head = format!("{:<34} {op_name} {scene_name:<9}", case.name);
                let Ok(result) = boolean(&mut topo, op, x, y) else {
                    println!("{head} error");
                    continue;
                };
                let faces = solid_faces(&topo, result).unwrap();
                let flat = faces
                    .iter()
                    .all(|&f| topo.face(f).unwrap().surface().type_tag() == "plane");
                let census = if flat && faces.len() > 50 {
                    "fallback".to_string()
                } else {
                    format!("{} faces", faces.len())
                };
                let valid = validate_solid(&topo, result).is_ok_and(|r| r.is_valid());
                let closed = tessellate_solid(&topo, result, 0.01).is_ok_and(|m| is_watertight(&m));
                let volume = solid_volume(&topo, result, 0.01).unwrap_or(f64::NAN);
                let checked = brepkit_check::properties::solid_volume(
                    &topo,
                    result,
                    &brepkit_check::properties::PropertiesOptions::default(),
                )
                .unwrap_or(f64::NAN);
                let error = |v: f64| {
                    case.truth.map_or_else(
                        || format!("{:.1e}", (v - volume).abs() / volume.abs().max(1e-12)),
                        |t| format!("{:.1e}", (v - t[k]).abs() / t[k]),
                    )
                };
                let (error, checked_error) = (
                    case.truth
                        .map_or_else(|| "-".to_string(), |_| error(volume)),
                    error(checked),
                );
                let (wrong, read) = misreads(&topo, result, &case, scene, k);
                println!(
                    "{head} {census:<10} {} {} vol {volume:.4} err {error} check err {checked_error} misread {wrong}/{read}",
                    if valid { "valid" } else { "INVALID" },
                    if closed { "closed" } else { "OPEN" },
                );
            }
        }
    }
    eprintln!("pose sweep: {:.1} s", started.elapsed().as_secs_f64());
}

/// Points of a grid over both operands the engine's ray cast reads against
/// the operands' distances, and the points read.
fn misreads(topo: &Topology, result: SolidId, case: &Case, scene: Mat4, op: usize) -> (u32, u32) {
    let Ok(geoms) = RayCastGeoms::new(topo, result) else {
        return (0, 0);
    };
    let corners: Vec<Point3> = case
        .a
        .corners(scene)
        .into_iter()
        .chain(case.b.corners(scene))
        .collect();
    let lo = corners.iter().fold([f64::INFINITY; 3], |m, c| {
        [m[0].min(c.x()), m[1].min(c.y()), m[2].min(c.z())]
    });
    let hi = corners.iter().fold([f64::NEG_INFINITY; 3], |m, c| {
        [m[0].max(c.x()), m[1].max(c.y()), m[2].max(c.z())]
    });
    let at =
        |k: usize, i: u32| (hi[k] - lo[k]).mul_add((f64::from(i) + 0.5) / f64::from(GRID), lo[k]);
    let (mut wrong, mut read) = (0, 0);
    for i in 0..GRID {
        for j in 0..GRID {
            for l in 0..GRID {
                let p = Point3::new(at(0, i), at(1, j), at(2, l));
                let (da, db) = (case.a.distance(scene, p), case.b.distance(scene, p));
                if da.abs() < NEAR || db.abs() < NEAR {
                    continue;
                }
                let (ina, inb) = (da < 0.0, db < 0.0);
                let truth = match op {
                    0 => ina && !inb,
                    1 => ina && inb,
                    _ => inb && !ina,
                };
                read += 1;
                let inside =
                    classify_ray_cast_cached(&geoms, p).is_ok_and(|c| c == FaceClass::Inside);
                if inside != truth {
                    wrong += 1;
                }
            }
        }
    }
    (wrong, read)
}
