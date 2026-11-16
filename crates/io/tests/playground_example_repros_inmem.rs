//! Captured operands of three calls that fail in the brepjs playground
//! examples (#2372), each a ready repro for its own issue. Each test states
//! what the call must give, and is ignored until its issue closes.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::{all_blended, chamfer_v2};
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces};
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
fn fuse_exactly(
    topo: &mut Topology,
    a: SolidId,
    b: SolidId,
    [in_a, in_b, off]: [[f64; 3]; 3],
    grid: &Grid,
) {
    use PointClassification::{Inside, Outside};
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
    for p in grid.points() {
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

/// `pocket-watch`: a bridge plate whose top outline, at z = 3.15, is 6 lines
/// and 7 circle arcs.
#[test]
#[ignore = "TODO: enable when #2490 closes"]
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
    let chamfered = chamfer_v2(&mut topo, plate, &outline, 0.1, 0.1)
        .and_then(all_blended)
        .unwrap();
    assert_holds(&topo, chamfered);
}
