//! A seeded random sweep of booleans on balls, rods and boxes, with each
//! result held against a volume found independently of the kernel.
//!
//! Each case starts from a ball or a cylinder. It then cuts, fuses or
//! intersects one to three tools in turn: rods along x, y or z, balls and
//! boxes, in one of five poses, with the mesh fallback off. [`oracle`] gives
//! the volume of the exact result from slices in z and lines in x, with the
//! exact interval in y of each solid on each line.
//!
//! Run a seed with
//!
//! ```text
//! SWEEP_SEED=1 SWEEP_N=150 cargo test --profile ci -p brepkit-operations \
//!     --test random_boolean_sweep -- --ignored --nocapture sweep
//! ```
//!
//! and add `SWEEP_ONLY=17` to run case 17 of that seed alone. Each case
//! prints one line: `SWEEP <case> <base> [<pose>] <verdict> steps=<codes>
//! final=<verdict> ... :: <solids> <ops>`.
//!
//! Each step of a case is scored: the result of boolean `k` is held against
//! the oracle for the solids and operations up to `k`. A step is `G` (good),
//! `B` (bad), `E` (an error), `0` (the exact result is empty, and so is the
//! boolean's) or `P` (a panic).
//!
//! - `GOOD`: each step whose exact result is not empty is valid, its mesh
//!   has no open edge, and its volume is within 2e-4 of the oracle's, or
//!   within 1e-9 of it: the oracle's cells read a sliver of 1e-8 to some
//!   parts in 1e4. Each step whose exact result is empty gave a solid with
//!   no faces or a volume within 1e-9 of zero.
//! - `BAD`: a boolean gave an exact solid that fails one of these, at any
//!   step. The line gives the step, `valid`, the faces by surface type,
//!   `not2` (edges that two faces do not share), `open` (open mesh edges),
//!   `rel` (the relative volume error) and the axes of the sphere faces, or
//!   for a step whose exact result is empty, `vol` (the solid's volume).
//! - `ERR@<k>`: boolean `k` returned an error. `EMPTYEXP`: the exact final
//!   result is empty, and so is the boolean's. `PANIC`: a boolean panicked.
//!
//! `final=` gives the verdict on the final result alone. The run ends with a
//! `SUMMARY` line that counts the verdicts of the cases, the verdicts on the
//! final results alone, and the bad steps.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::panic
)]

use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// One boolean with the mesh fallback off, so that every result is exact.
fn op(topo: &mut Topology, o: BooleanOp, a: SolidId, b: SolidId) -> Result<SolidId, String> {
    let opts = BooleanOptions {
        mesh_fallback: false,
        unify_faces: false,
        ..BooleanOptions::default()
    };
    boolean::boolean_with_options(topo, o, a, b, opts).map_err(|e| format!("{e}"))
}

/// The faces of `solid` counted by surface type.
fn census(topo: &Topology, solid: SolidId) -> String {
    let mut m: BTreeMap<&str, usize> = BTreeMap::new();
    for f in solid_faces(topo, solid).unwrap() {
        *m.entry(topo.face(f).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    m.iter()
        .map(|(k, v)| format!("{k}{v}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// The distinct axes of the sphere faces of `solid`.
fn sphere_axes(topo: &Topology, solid: SolidId) -> String {
    let mut out: Vec<String> = Vec::new();
    for f in solid_faces(topo, solid).unwrap() {
        if let FaceSurface::Sphere(sp) = topo.face(f).unwrap().surface() {
            let z = sp.z_axis();
            let t = format!("({:.2},{:.2},{:.2})", z.x(), z.y(), z.z());
            if !out.contains(&t) {
                out.push(t);
            }
        }
    }
    out.join("")
}

/// The edges of `solid` that its faces do not use exactly twice.
fn not_twice(topo: &Topology, solid: SolidId) -> usize {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for f in solid_faces(topo, solid).unwrap() {
        let face = topo.face(f).unwrap();
        for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wire).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    uses.values().filter(|&&n| n != 2).count()
}

/// The area two discs of radii `a` and `b`, `d` apart, share.
fn lens(a: f64, b: f64, d: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 || d >= a + b {
        return 0.0;
    }
    if d <= (a - b).abs() {
        return PI * a.min(b).powi(2);
    }
    let p = a
        * a
        * ((d * d + a * a - b * b) / (2.0 * d * a))
            .clamp(-1.0, 1.0)
            .acos();
    let q = b
        * b
        * ((d * d + b * b - a * a) / (2.0 * d * b))
            .clamp(-1.0, 1.0)
            .acos();
    let r = 0.5
        * ((-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b))
            .max(0.0)
            .sqrt();
    p + q - r
}

fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: u32) -> f64 {
    if b <= a {
        return 0.0;
    }
    let h = (b - a) / f64::from(n);
    let mut s = f(a) + f(b);
    for i in 1..n {
        s += if i % 2 == 1 { 4.0 } else { 2.0 } * f(a + h * f64::from(i));
    }
    s * h / 3.0
}

/// The volume a ball (radius `rb`, centre `c`) shares with a cylinder of
/// radius `r` along coordinate `axis`, whose axis passes through `pq` in the
/// other two coordinates (in cyclic order), between `t0` and `t1` along it:
/// the shared area of the two discs in each cross-section, over the ball's
/// latitude.
fn ov_ball_cyl(rb: f64, c: [f64; 3], r: f64, axis: usize, pq: (f64, f64), t0: f64, t1: f64) -> f64 {
    let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
    let d = (c[i] - pq.0).hypot(c[j] - pq.1);
    let s = c[axis];
    let th = |t: f64| ((t - s) / rb).clamp(-1.0, 1.0).asin();
    simpson(
        |a: f64| lens(rb * a.cos(), r, d) * rb * a.cos(),
        th(t0),
        th(t1),
        20_000,
    )
}

fn ball_v(r: f64) -> f64 {
    4.0 / 3.0 * PI * r.powi(3)
}

fn bx(topo: &mut Topology, lo: [f64; 3], hi: [f64; 3]) -> SolidId {
    let s = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
    transform_solid(topo, s, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
    s
}

/// The poses a case runs in: one picked at random for each.
fn poses() -> Vec<(&'static str, Mat4)> {
    let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1);
    vec![
        ("id", Mat4::identity()),
        ("turned", turn),
        ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
        ("turned+mirrored", Mat4::scale(1.0, -1.0, 1.0) * turn),
        ("roll-z", Mat4::rotation_z(2.0)),
    ]
}

/// A solid of a case, before its pose.
#[derive(Clone, Copy, Debug)]
enum Prim {
    /// A ball turned by `spin` about x, then y, then z, so that its poles and
    /// seam lie anywhere.
    Ball {
        c: [f64; 3],
        r: f64,
        spin: [f64; 3],
    },
    /// A cylinder along `axis` (0 x, 1 y, 2 z) through `pq` (the other two
    /// coordinates in cyclic order), from `t0` to `t1`, turned by `spin` about
    /// its axis.
    Cyl {
        axis: usize,
        pq: (f64, f64),
        r: f64,
        t0: f64,
        t1: f64,
        spin: f64,
    },
    Box {
        lo: [f64; 3],
        hi: [f64; 3],
    },
}

impl Prim {
    /// The interval in y of the solid on the line through `(x, z)`.
    fn yspan(&self, x: f64, z: f64) -> Option<(f64, f64)> {
        match *self {
            Self::Ball { c, r, .. } => {
                let q = r * r - (x - c[0]).powi(2) - (z - c[2]).powi(2);
                (q > 0.0).then(|| (c[1] - q.sqrt(), c[1] + q.sqrt()))
            }
            Self::Box { lo, hi } => {
                (x >= lo[0] && x <= hi[0] && z >= lo[2] && z <= hi[2]).then_some((lo[1], hi[1]))
            }
            Self::Cyl {
                axis,
                pq,
                r,
                t0,
                t1,
                ..
            } => match axis {
                // Along x through (y, z) = pq.
                0 => {
                    let q = r * r - (z - pq.1).powi(2);
                    (x >= t0 && x <= t1 && q > 0.0).then(|| (pq.0 - q.sqrt(), pq.0 + q.sqrt()))
                }
                // Along y through (z, x) = pq.
                1 => {
                    let q = (z - pq.0).powi(2) + (x - pq.1).powi(2);
                    (q < r * r).then_some((t0, t1))
                }
                // Along z through (x, y) = pq.
                _ => {
                    let q = r * r - (x - pq.0).powi(2);
                    (z >= t0 && z <= t1 && q > 0.0).then(|| (pq.1 - q.sqrt(), pq.1 + q.sqrt()))
                }
            },
        }
    }

    fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        match *self {
            Self::Ball { c, r, .. } => (
                [c[0] - r, c[1] - r, c[2] - r],
                [c[0] + r, c[1] + r, c[2] + r],
            ),
            Self::Box { lo, hi } => (lo, hi),
            Self::Cyl {
                axis,
                pq,
                r,
                t0,
                t1,
                ..
            } => {
                let mut lo = [0.0; 3];
                let mut hi = [0.0; 3];
                let (i, j) = ((axis + 1) % 3, (axis + 2) % 3);
                lo[axis] = t0;
                hi[axis] = t1;
                lo[i] = pq.0 - r;
                hi[i] = pq.0 + r;
                lo[j] = pq.1 - r;
                hi[j] = pq.1 + r;
                (lo, hi)
            }
        }
    }

    fn build(&self, t: &mut Topology) -> SolidId {
        match *self {
            Self::Ball { c, r, spin } => {
                let s = make_sphere(t, r, 32).unwrap();
                let m = Mat4::translation(c[0], c[1], c[2])
                    * Mat4::rotation_z(spin[2])
                    * Mat4::rotation_y(spin[1])
                    * Mat4::rotation_x(spin[0]);
                transform_solid(t, s, &m).unwrap();
                s
            }
            Self::Box { lo, hi } => bx(t, lo, hi),
            Self::Cyl {
                axis,
                pq,
                r,
                t0,
                t1,
                spin,
            } => {
                let s = make_cylinder(t, r, t1 - t0).unwrap();
                let along_z = Mat4::translation(pq.0, pq.1, t0) * Mat4::rotation_z(spin);
                let m = match axis {
                    2 => along_z,
                    0 => Mat4::rotation_z(PI / 2.0) * Mat4::rotation_x(PI / 2.0) * along_z,
                    // (p, q, t) to x = q, y = t, z = p.
                    _ => Mat4::rotation_x(-PI / 2.0) * Mat4::rotation_z(-PI / 2.0) * along_z,
                };
                transform_solid(t, s, &m).unwrap();
                s
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Op3 {
    Cut,
    Fuse,
    Common,
}

/// The volume of `((p0 op0 p1) op1 p2) ...`, by slices in z and lines in x,
/// with the exact interval in y of each solid on each line.
fn oracle(prims: &[Prim], ops: &[Op3]) -> f64 {
    // The cells cover only a box round the result: the first solid's,
    // joined with each fused solid's and cut down to each intersected
    // solid's. A small result then gets as many cells as a large one.
    let (mut lo, mut hi) = prims[0].bounds();
    for (k, o) in ops.iter().enumerate() {
        let (a, b) = prims[k + 1].bounds();
        for i in 0..3 {
            match o {
                Op3::Cut => {}
                Op3::Fuse => {
                    lo[i] = lo[i].min(a[i]);
                    hi[i] = hi[i].max(b[i]);
                }
                Op3::Common => {
                    lo[i] = lo[i].max(a[i]);
                    hi[i] = hi[i].min(b[i]);
                }
            }
        }
    }
    if (0..3).any(|i| hi[i] <= lo[i]) {
        return 0.0;
    }
    // Planar caps and box faces across x or z lie on the boundaries of the
    // cells, so no cell of the midpoint rule has a flat face through it.
    let cells = |axis: usize, n: usize| -> Vec<(f64, f64)> {
        let mut br = vec![lo[axis], hi[axis]];
        for p in prims {
            match *p {
                Prim::Box { lo: a, hi: b } => {
                    br.push(a[axis]);
                    br.push(b[axis]);
                }
                Prim::Cyl {
                    axis: ax, t0, t1, ..
                } if ax == axis => {
                    br.push(t0);
                    br.push(t1);
                }
                _ => {}
            }
        }
        br.retain(|&b| b >= lo[axis] && b <= hi[axis]);
        br.sort_by(f64::total_cmp);
        br.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        let span = hi[axis] - lo[axis];
        let mut out = Vec::with_capacity(n + br.len());
        for w in br.windows(2) {
            let len = w[1] - w[0];
            let k = ((len / span * n as f64).round() as usize).max(1);
            let h = len / k as f64;
            for i in 0..k {
                out.push((w[0] + (i as f64 + 0.5) * h, h));
            }
        }
        out
    };
    let (xs, zs) = (cells(0, 2400), cells(2, 2400));
    let mut total = 0.0;
    let mut ends: Vec<f64> = Vec::with_capacity(16);
    let mut spans: Vec<Option<(f64, f64)>> = vec![None; prims.len()];
    for &(z, dz) in &zs {
        for &(x, dx) in &xs {
            ends.clear();
            let mut any = false;
            for (k, p) in prims.iter().enumerate() {
                spans[k] = p.yspan(x, z);
                if let Some((a, b)) = spans[k] {
                    ends.push(a);
                    ends.push(b);
                    any = true;
                }
            }
            if !any {
                continue;
            }
            ends.sort_by(f64::total_cmp);
            for w in ends.windows(2) {
                if w[1] <= w[0] {
                    continue;
                }
                let y = 0.5 * (w[0] + w[1]);
                let inside = |k: usize| spans[k].is_some_and(|(a, b)| y > a && y < b);
                let mut m = inside(0);
                for (k, o) in ops.iter().enumerate() {
                    let t = inside(k + 1);
                    m = match o {
                        Op3::Cut => m && !t,
                        Op3::Fuse => m || t,
                        Op3::Common => m && t,
                    };
                }
                if m {
                    total += (w[1] - w[0]) * dx * dz;
                }
            }
        }
    }
    total
}

/// A linear congruential generator: the same cases from the same seed on
/// every machine.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }

    fn pick(&mut self, n: usize) -> usize {
        ((self.next() * n as f64) as usize).min(n - 1)
    }
}

// The angles run to 6.28, not TAU: every reported seed and case index
// rests on these exact values.
#[allow(clippy::approx_constant)]
fn random_case(rng: &mut Lcg) -> (Vec<Prim>, Vec<Op3>, &'static str) {
    let kind = rng.pick(3);
    let spin_ball = |rng: &mut Lcg| -> [f64; 3] {
        match rng.pick(3) {
            0 => [0.0, 0.0, 0.0],
            1 => [
                rng.range(-0.3, 0.3),
                rng.range(-0.3, 0.3),
                rng.range(0.0, 6.28),
            ],
            _ => [
                rng.range(0.0, 6.28),
                rng.range(0.0, 6.28),
                rng.range(0.0, 6.28),
            ],
        }
    };
    let mut prims = Vec::new();
    let tag;
    if kind < 2 {
        tag = "ball";
        prims.push(Prim::Ball {
            c: [0.0, 0.0, 0.0],
            r: 1.0,
            spin: spin_ball(rng),
        });
    } else {
        tag = "cyl";
        prims.push(Prim::Cyl {
            axis: 2,
            pq: (0.0, 0.0),
            r: 1.0,
            t0: 0.0,
            t1: 4.0,
            spin: rng.range(0.0, 6.28),
        });
    }
    let ntools = 1 + rng.pick(3);
    let mut ops = Vec::new();
    for _ in 0..ntools {
        let tool = match rng.pick(3) {
            0 => {
                let axis = rng.pick(3);
                let r = rng.range(0.04, 0.5);
                let pq = if tag == "ball" {
                    let near = rng.pick(2) == 0;
                    if near {
                        (rng.range(-0.15, 0.15), rng.range(-0.15, 0.15))
                    } else {
                        (rng.range(-0.9, 0.9), rng.range(-0.9, 0.9))
                    }
                } else {
                    (rng.range(-1.3, 1.3), rng.range(-1.3, 1.3))
                };
                let (t0, t1) = if tag == "ball" {
                    match rng.pick(3) {
                        0 => (-2.0, 2.0),
                        1 => (rng.range(-0.6, 0.9), 2.0),
                        _ => (-2.0, rng.range(-0.9, 0.6)),
                    }
                } else if axis == 2 {
                    (rng.range(-1.0, 3.5), rng.range(3.0, 5.0))
                } else {
                    (-2.0, 2.0)
                };
                let pq = if tag == "cyl" && axis == 0 {
                    // Along x through (y, z): z within the wall's height.
                    (rng.range(-1.2, 1.2), rng.range(0.2, 4.2))
                } else if tag == "cyl" && axis == 1 {
                    (rng.range(0.2, 4.2), rng.range(-1.2, 1.2))
                } else {
                    pq
                };
                Prim::Cyl {
                    axis,
                    pq,
                    r,
                    t0: t0.min(t1 - 0.1),
                    t1,
                    spin: rng.range(0.0, 6.28),
                }
            }
            1 => {
                let r = rng.range(0.1, 0.7);
                let c = if tag == "ball" {
                    let d = rng.range(0.6, 1.3);
                    let th = rng.range(0.0, 6.28);
                    let ph = if rng.pick(2) == 0 {
                        rng.range(-0.25, 0.25)
                    } else {
                        rng.range(-1.57, 1.57)
                    };
                    let ph = if rng.pick(2) == 0 {
                        ph
                    } else {
                        std::f64::consts::FRAC_PI_2 - ph.abs() * 0.2
                    };
                    [
                        d * ph.cos() * th.cos(),
                        d * ph.cos() * th.sin(),
                        d * ph.sin(),
                    ]
                } else {
                    let th = rng.range(0.0, 6.28);
                    let z = if rng.pick(2) == 0 {
                        4.0 + rng.range(-0.3, 0.3)
                    } else {
                        rng.range(0.0, 4.0)
                    };
                    let d = rng.range(0.7, 1.2);
                    [d * th.cos(), d * th.sin(), z]
                };
                Prim::Ball {
                    c,
                    r,
                    spin: spin_ball(rng),
                }
            }
            _ => {
                let mut lo = [0.0; 3];
                let mut hi = [0.0; 3];
                for k in 0..3 {
                    let a = rng.range(-1.3, 1.3)
                        + if tag == "cyl" && k == 2 {
                            2.0 + rng.range(-2.0, 2.0)
                        } else {
                            0.0
                        };
                    let w = rng.range(0.1, 1.5);
                    lo[k] = a;
                    hi[k] = a + w;
                }
                Prim::Box { lo, hi }
            }
        };
        prims.push(tool);
        ops.push(match rng.pick(5) {
            0 | 1 => Op3::Cut,
            2 | 3 => Op3::Fuse,
            _ => Op3::Common,
        });
    }
    (prims, ops, tag)
}

/// Holds the result `s` of a step against the oracle's volume `expected`:
/// whether it is good, and the numbers.
fn score(topo: &Topology, s: SolidId, expected: f64) -> (bool, String) {
    let valid = validate_solid(topo, s).is_ok_and(|r| r.is_valid());
    let vol = solid_volume(topo, s, 0.0005).unwrap_or(f64::NAN);
    let rel = (vol - expected) / expected;
    let open = tessellate_solid(topo, s, 0.01)
        .map(|m| boundary_edge_count(&m))
        .unwrap_or(usize::MAX);
    let good = valid && (rel.abs() < 2e-4 || (vol - expected).abs() < 1e-9) && open == 0;
    let detail = format!(
        "valid={valid} faces={} not2={} open={open} rel={rel:.2e} axes={}",
        census(topo, s),
        not_twice(topo, s),
        sphere_axes(topo, s)
    );
    (good, detail)
}

/// The verdict of a case's final step alone, as it reads in a case's line.
enum Final {
    Good,
    Bad,
    Err(usize, String),
    Empty,
}

/// Counts of verdicts over a run.
#[derive(Default)]
struct Tally {
    cases: BTreeMap<&'static str, usize>,
    finals: BTreeMap<&'static str, usize>,
    steps: usize,
    bad_steps: usize,
    bad_middle_steps: usize,
}

impl Tally {
    fn line(&self, seed: u64) -> String {
        let fmt = |m: &BTreeMap<&str, usize>| {
            ["GOOD", "BAD", "ERR", "EMPTYEXP", "PANIC"]
                .iter()
                .map(|k| format!("{k}={}", m.get(k).copied().unwrap_or(0)))
                .collect::<Vec<_>>()
                .join(" ")
        };
        format!(
            "SUMMARY seed={seed} cases: {} | final only: {} | steps={} bad={} bad-middle={}",
            fmt(&self.cases),
            fmt(&self.finals),
            self.steps,
            self.bad_steps,
            self.bad_middle_steps
        )
    }
}

/// Runs `SWEEP_N` cases (none by default) from `SWEEP_SEED` (1 by default),
/// or only case `SWEEP_ONLY` of them. See the module notes.
#[test]
#[ignore = "a random sweep that reports cases; run it by hand with SWEEP_N"]
#[allow(clippy::too_many_lines)]
fn sweep() {
    let n: u64 = std::env::var("SWEEP_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let seed: u64 = std::env::var("SWEEP_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let only: Option<u64> = std::env::var("SWEEP_ONLY")
        .ok()
        .and_then(|s| s.parse().ok());
    let mut rng = Lcg(seed);
    let poses = poses();
    let mut tally = Tally::default();
    for i in 0..n {
        let (prims, ops, tag) = random_case(&mut rng);
        let pose_k = rng.pick(poses.len());
        if only.is_some_and(|o| o != i) {
            continue;
        }
        let (pname, pose) = poses[pose_k];
        let mut codes = String::new();
        let mut first_bad: Option<(usize, String)> = None;
        let mut last_detail = String::new();
        let fin = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut topo = Topology::new();
            let ids: Vec<SolidId> = prims.iter().map(|p| p.build(&mut topo)).collect();
            for &s in &ids {
                transform_solid(&mut topo, s, &pose).unwrap();
            }
            let mut cur = ids[0];
            let mut fin = Final::Empty;
            for (k, o) in ops.iter().enumerate() {
                let bo = match o {
                    Op3::Cut => BooleanOp::Cut,
                    Op3::Fuse => BooleanOp::Fuse,
                    Op3::Common => BooleanOp::Intersect,
                };
                match op(&mut topo, bo, cur, ids[k + 1]) {
                    Ok(s) => cur = s,
                    Err(e) => {
                        codes.push('E');
                        return Final::Err(k, e.chars().take(60).collect());
                    }
                }
                let expected = oracle(&prims[..k + 2], &ops[..=k]);
                if expected < 1e-9 {
                    let faces = solid_faces(&topo, cur).map_or(usize::MAX, |f| f.len());
                    let vol = if faces == 0 {
                        0.0
                    } else {
                        solid_volume(&topo, cur, 0.0005).unwrap_or(f64::NAN)
                    };
                    if vol.abs() < 1e-9 {
                        codes.push('0');
                        fin = Final::Empty;
                    } else {
                        codes.push('B');
                        let detail = format!(
                            "empty exact result: faces={} vol={vol:.6e}",
                            census(&topo, cur)
                        );
                        if first_bad.is_none() {
                            first_bad = Some((k, detail.clone()));
                        }
                        fin = Final::Bad;
                        last_detail = detail;
                    }
                    continue;
                }
                let (good, detail) = score(&topo, cur, expected);
                codes.push(if good { 'G' } else { 'B' });
                if !good && first_bad.is_none() {
                    first_bad = Some((k, detail.clone()));
                }
                fin = if good { Final::Good } else { Final::Bad };
                last_detail = detail;
            }
            fin
        }));
        let fin_key = match &fin {
            Err(_) => {
                codes.push('P');
                "PANIC"
            }
            Ok(Final::Good) => "GOOD",
            Ok(Final::Bad) => "BAD",
            Ok(Final::Empty) => "EMPTYEXP",
            Ok(Final::Err(..)) => "ERR",
        };
        let fin_word = match &fin {
            Ok(Final::Err(k, _)) => format!("ERR@{k}"),
            _ => fin_key.to_string(),
        };
        let (case_key, verdict, detail) = if let Some((k, d)) = &first_bad {
            ("BAD", "BAD".to_string(), format!("step={k} {d}"))
        } else {
            let detail = match &fin {
                Ok(Final::Err(_, e)) => e.clone(),
                Ok(Final::Good) => last_detail.clone(),
                _ => String::new(),
            };
            (fin_key, fin_word.clone(), detail)
        };
        *tally.cases.entry(case_key).or_default() += 1;
        *tally.finals.entry(fin_key).or_default() += 1;
        tally.steps += codes.chars().filter(|&c| c == 'G' || c == 'B').count();
        tally.bad_steps += codes.chars().filter(|&c| c == 'B').count();
        tally.bad_middle_steps += codes
            .chars()
            .take(ops.len().saturating_sub(1))
            .filter(|&c| c == 'B')
            .count();
        println!(
            "SWEEP {i} {tag} [{pname}] {verdict} steps={codes} final={fin_word} {detail} :: {prims:?} {ops:?}"
        );
    }
    println!("{}", tally.line(seed));
}

/// The oracle against closed forms and one-dimensional integrals: a ball, a
/// box, a cylinder, a ball with a tunnel through it, a ball with a blind rod
/// cut out of it, and the small part of a rod that a cylinder's cap holds.
#[test]
fn oracle_sanity() {
    let ball = Prim::Ball {
        c: [0.0, 0.0, 0.0],
        r: 1.0,
        spin: [0.0; 3],
    };
    let tunnel = Prim::Cyl {
        axis: 0,
        pq: (0.0, 0.0),
        r: 0.3,
        t0: -2.0,
        t1: 2.0,
        spin: 0.0,
    };
    let blind = Prim::Cyl {
        axis: 1,
        pq: (0.2, 0.1),
        r: 0.3,
        t0: -2.0,
        t1: 0.5,
        spin: 0.0,
    };
    let wall = Prim::Cyl {
        axis: 2,
        pq: (0.0, 0.0),
        r: 1.0,
        t0: 0.0,
        t1: 4.0,
        spin: 0.0,
    };
    let block = Prim::Box {
        lo: [-0.3, 0.2, 0.5],
        hi: [0.9, 1.1, 1.75],
    };
    // A rod along y, 0.114 above the cap of the cylinder, reaches 0.012
    // below it: the part under the cap is a segment of the rod's section
    // across the unit disc, `2 sqrt(1 - x^2)` long in y.
    let (zc, xc, rs) = (4.114, -0.23, 0.126);
    let sliver = Prim::Cyl {
        axis: 1,
        pq: (zc, xc),
        r: rs,
        t0: -2.0,
        t1: 2.0,
        spin: 0.0,
    };
    let half_width = (rs * rs - (zc - 4.0) * (zc - 4.0)).sqrt();
    let sliver_volume = simpson(
        |x: f64| {
            let depth = 4.0 - zc + (rs * rs - (x - xc) * (x - xc)).max(0.0).sqrt();
            depth.max(0.0) * 2.0 * (1.0 - x * x).sqrt()
        },
        xc - half_width,
        xc + half_width,
        100_000,
    );
    let cases = [
        ("ball", oracle(&[ball], &[]), ball_v(1.0)),
        ("box", oracle(&[block], &[]), 1.2 * 0.9 * 1.25),
        ("cylinder", oracle(&[wall], &[]), PI * 4.0),
        (
            "ball less a tunnel",
            oracle(&[ball, tunnel], &[Op3::Cut]),
            ball_v(1.0) - ov_ball_cyl(1.0, [0.0; 3], 0.3, 0, (0.0, 0.0), -2.0, 2.0),
        ),
        (
            "the sliver of a rod under a cylinder's cap",
            oracle(&[wall, sliver], &[Op3::Common]),
            sliver_volume,
        ),
        (
            "ball less a blind rod",
            oracle(&[ball, blind], &[Op3::Cut]),
            // The rod along y through (z, x) = (0.2, 0.1) is a rod along z
            // through the origin for a ball at (0.1, 0.2) in its frame.
            ball_v(1.0) - ov_ball_cyl(1.0, [0.1, 0.2, 0.0], 0.3, 2, (0.0, 0.0), -2.0, 0.5),
        ),
    ];
    for (name, got, want) in cases {
        let rel = (got - want) / want;
        assert!(rel.abs() < 2e-5, "{name}: {got} against {want} ({rel:.2e})");
    }
}
