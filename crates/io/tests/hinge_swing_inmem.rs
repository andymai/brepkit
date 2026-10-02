//! The gridfinity tool's hinged lid and bin (`hingeSwing.scenario.test.ts`),
//! from operands captured from its calls: the tests check the captured lid,
//! one of its bores alone, and the tool's own calls on them.
//!
//! The lid is compound-cut by its clearance bevel and four knuckle bores. Each
//! bore (radius 2.45 along `y`) has its axis on the lid's pocket ceiling
//! `z = -3.2`, crosses the ceiling's edge at the pocket wall `x = -59`, and
//! pokes 0.05 past the back face `x = -62.75`, which the ceiling's plane splits
//! in two. The tool then fuses its knuckles on and cuts two keyhole pins from
//! the knuckled lid.
//!
//! The bin is compound-cut by five clearance rods that only touch its lip.
//!
//! Data: `hinge_lid.bin` (the lid), `hinge_lid_clearance.bin` (the bevel's
//! box), `hinge_lid_bore_<1..4>.bin`, `hinge_lid_knuckle.bin` (the first
//! knuckle the tool fuses onto the cut lid), `hinge_lid_knuckled.bin` (the lid
//! with its knuckles) with `hinge_lid_pin_short.bin` and
//! `hinge_lid_pin_long.bin` (the two keyhole pins), `hinge_bin.bin` and
//! `hinge_bin_clearance_<1..5>.bin`, and the finished bin and lid the tool
//! swings against each other, `hinge_swing_bin.bin` and
//! `hinge_swing_lid_closed.bin` (the lid shut) and `hinge_swing_lid_40.bin`
//! (swung 40 degrees open), and the same hinge on the left wall,
//! `hinge_left_bin.bin` with `hinge_left_lid_18.bin` (swung 18 degrees open),
//! and that bin with its knuckles, `hinge_left_bin_knuckled.bin`, with its two
//! keyhole pins, `hinge_left_pin_short.bin` and `hinge_left_pin_long.bin`,
//! and a lid on its bin, `hinge_seat_bin.bin` with `hinge_seat_lid.bin`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// The material one bore removes: its disc less the sliver past the back face
/// and the quarter in the pocket below the ceiling, over the bore's length.
fn bore_truth() -> f64 {
    let r: f64 = 2.45;
    let segment = |h: f64| r * r * (h / r).acos() - h * (r * r - h * h).sqrt();
    let area = std::f64::consts::PI * r * r - segment(2.4) - segment(1.35) / 2.0;
    area * (37.8 - 26.571_428_571_428_57)
}

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn tools(topo: &mut Topology) -> Vec<SolidId> {
    std::iter::once("hinge_lid_clearance.bin".to_string())
        .chain((1..=4).map(|i| format!("hinge_lid_bore_{i}.bin")))
        .map(|name| load(topo, &name))
        .collect()
}

/// A solid's point classifier: a point counts when the ray cast and the
/// generalized winding number over its tessellation (taken once) agree.
struct Probe {
    solid: SolidId,
    mesh: brepkit_operations::tessellate::TriangleMesh,
}

impl Probe {
    fn new(topo: &Topology, solid: SolidId) -> Self {
        let mesh = brepkit_operations::tessellate::tessellate_solid(topo, solid, 0.01).unwrap();
        Self { solid, mesh }
    }

    fn inside(&self, topo: &Topology, p: [f64; 3]) -> Option<bool> {
        use brepkit_operations::classify::{PointClassification, classify_point};
        let q = brepkit_math::vec::Point3::new(p[0], p[1], p[2]);
        let ray = match classify_point(topo, self.solid, q, 0.01, 1e-7).ok()? {
            PointClassification::Inside => true,
            PointClassification::Outside => false,
            PointClassification::OnBoundary => return None,
        };
        let mut solid_angle = 0.0;
        for tri in self.mesh.indices.chunks_exact(3) {
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| self.mesh.positions[i as usize] - q);
            let (la, lb, lc) = (a.length(), b.length(), c.length());
            let den = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
            solid_angle += 2.0 * a.dot(b.cross(c)).atan2(den);
        }
        let wind = (solid_angle / (4.0 * std::f64::consts::PI)).abs() > 0.5;
        (ray == wind).then_some(ray)
    }
}

/// Whether `p` is inside `solid`, by [`Probe`].
fn inside(topo: &Topology, solid: SolidId, p: [f64; 3]) -> Option<bool> {
    Probe::new(topo, solid).inside(topo, p)
}

fn volume(topo: &Topology, solid: SolidId) -> f64 {
    solid_volume(topo, solid, 0.001).unwrap()
}

/// Runs `op`, requiring it to stay exact and return a valid solid.
fn exact(topo: &mut Topology, op: impl FnOnce(&mut Topology) -> SolidId) -> SolidId {
    let before = boolean::mesh_fallback_count();
    let result = op(topo);
    assert_eq!(boolean::mesh_fallback_count(), before, "mesh fallback");
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    result
}

#[test]
fn hinge_lid_fixture_is_faithful() {
    let mut topo = Topology::new();
    let lid = load(&mut topo, "hinge_lid.bin");
    faithful(&topo, lid, [4, 16, 24]);
}

/// `solid` is valid and has `faces` cone, cylinder and plane faces.
fn faithful(topo: &Topology, solid: SolidId, faces: [usize; 3]) {
    let mut census: HashMap<&str, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        *census
            .entry(topo.face(fid).unwrap().surface().type_tag())
            .or_insert(0) += 1;
    }
    assert_eq!(
        ["cone", "cylinder", "plane"].map(|k| census.get(k).copied().unwrap_or(0)),
        faces,
        "fixture drifted: {census:?}"
    );
    assert!(validate_solid(topo, solid).unwrap().is_valid());
}

/// The back face's two halves each took the bore cap's arc past the face as
/// a piece of themselves (it sags 0.05 off the plane, a twentieth of its
/// chord, and ends on the other half), and the ceiling never met the caps:
/// their section ran 1.1 inside both faces, under the filter's sampling step.
#[test]
fn hinge_lid_bore_cut_is_exact() {
    let mut topo = Topology::new();
    let lid = load(&mut topo, "hinge_lid.bin");
    let bore = load(&mut topo, "hinge_lid_bore_1.bin");
    let cut = exact(&mut topo, |t| {
        boolean::boolean(t, BooleanOp::Cut, lid, bore).unwrap()
    });
    let removed = volume(&topo, lid) - volume(&topo, cut);
    assert!(
        (removed - bore_truth()).abs() < 1e-3,
        "removed {removed}, truth {}",
        bore_truth()
    );
}

/// The tool's call: all five tools at once, through the default options,
/// which unify same-domain faces afterwards. The unify step merged the lid's
/// stacked corner cylinders into faces whose wires listed their edges out of
/// order.
#[test]
fn hinge_lid_compound_cut_is_exact() {
    let mut topo = Topology::new();
    let lid = load(&mut topo, "hinge_lid.bin");
    let tools = tools(&mut topo);
    let compound = exact(&mut topo, |t| {
        boolean::compound_cut(t, lid, &tools, BooleanOptions::default()).unwrap()
    });
    let mut sequential = lid;
    for &tool in &tools {
        sequential = exact(&mut topo, |t| {
            boolean::boolean(t, BooleanOp::Cut, sequential, tool).unwrap()
        });
    }
    let (vc, vs) = (volume(&topo, compound), volume(&topo, sequential));
    assert!(
        (vc - vs).abs() < 1e-6 * vs,
        "compound {vc}, sequential {vs}"
    );
}

/// The tool's next call: the first knuckle fused onto the cut lid. The
/// knuckle's axis lies on the ceiling too, and its flat top runs in the
/// bevel's plane.
#[test]
fn hinge_lid_knuckle_fuse_is_exact() {
    let mut topo = Topology::new();
    let lid = load(&mut topo, "hinge_lid.bin");
    let tools = tools(&mut topo);
    let cut = exact(&mut topo, |t| {
        boolean::compound_cut(t, lid, &tools, BooleanOptions::default()).unwrap()
    });
    let knuckle = load(&mut topo, "hinge_lid_knuckle.bin");
    let fused = exact(&mut topo, |t| {
        boolean::boolean(t, BooleanOp::Fuse, cut, knuckle).unwrap()
    });
    let outside = exact(&mut topo, |t| {
        boolean::boolean(t, BooleanOp::Cut, knuckle, cut).unwrap()
    });
    let expected = volume(&topo, cut) + volume(&topo, outside);
    let got = volume(&topo, fused);
    assert!(
        (got - expected).abs() < 1e-6 * expected,
        "fused {got}, lid plus the knuckle past it {expected}"
    );
    // The knuckle's ends in the two bores, its overlap with the lid, and the
    // lid's plate are material; the bore's gap below the knuckle, cleared by
    // the bevel, and the air past the back face are not.
    for (p, kept) in [
        ([-60.35, -26.7, -2.5], true),
        ([-60.35, -16.2, -2.5], true),
        ([-59.5, -21.5, -1.2], true),
        ([0.0, 0.0, -1.6], true),
        ([-60.35, -21.5, -5.53], false),
        ([-63.0, -21.5, -3.2], false),
    ] {
        assert_eq!(inside(&topo, fused, p), Some(kept), "probe {p:?}");
    }
}

/// The short keyhole pin sits in a gap between two knuckles, coaxial with
/// them, its caps flat on their end faces: it only touches the lid, which the
/// cut leaves whole. Each end face reaches the cut as several coplanar pieces
/// from the knuckle fuses, so one of them covers more than one piece of the
/// pin's cap and same-domain pairing leaves a cap piece unpaired, wholly on
/// the lid's boundary.
#[test]
fn hinge_lid_short_pin_cut_is_exact() {
    let mut topo = Topology::new();
    let lid = load(&mut topo, "hinge_lid_knuckled.bin");
    let pin = load(&mut topo, "hinge_lid_pin_short.bin");
    let before = volume(&topo, lid);
    let cut = exact(&mut topo, |t| {
        boolean::boolean(t, BooleanOp::Cut, lid, pin).unwrap()
    });
    let got = volume(&topo, cut);
    assert!(
        (got - before).abs() < 1e-6 * before,
        "cut {got}, lid {before}"
    );
    // Either side of both caps, inside the pin's profile and beside it, the
    // cut holds what the lid holds: the knuckles' material and the gap's air.
    let lid_ref = load(&mut topo, "hinge_lid_knuckled.bin");
    let lid_ref = Probe::new(&topo, lid_ref);
    let cut = Probe::new(&topo, cut);
    let (mut material, mut air) = (0, 0);
    for x in [-58.76, -58.36, -48.105, -47.705] {
        for r in [0.5, 1.3] {
            for deg in [0.0_f64, 90.0, 180.0, 270.0] {
                let (s, c) = deg.to_radians().sin_cos();
                let p = [x, 39.35 + r * c, -3.2 + r * s];
                let expected = lid_ref.inside(&topo, p).unwrap();
                assert_eq!(cut.inside(&topo, p), Some(expected), "probe {p:?}");
                if expected {
                    material += 1;
                } else {
                    air += 1;
                }
            }
        }
    }
    assert_eq!((material, air), (8, 24));
}

/// The long keyhole pin (radius 1, along `x` from -47.905 to 58.56) runs
/// through the knuckles coaxial with them, its axis in the lid's bevel plane.
/// The bevel faces meet its wall along a ruling only in the gaps between
/// knuckles, but each section ran the wall's whole length: one face touching
/// the wall only at its end, and one reaching across a knuckle around a hole.
/// The wall split along rulings inside the knuckles and three knuckles lost
/// its lower quarter. The tool's call cuts both pins at once.
#[test]
fn hinge_lid_long_pin_cut_is_exact() {
    let mut topo = Topology::new();
    let lid = load(&mut topo, "hinge_lid_knuckled.bin");
    let short = load(&mut topo, "hinge_lid_pin_short.bin");
    let long = load(&mut topo, "hinge_lid_pin_long.bin");
    let lid_copy = brepkit_operations::copy::copy_solid(&mut topo, lid).unwrap();
    let cut = exact(&mut topo, |t| {
        boolean::boolean(t, BooleanOp::Cut, lid_copy, long).unwrap()
    });
    let both = exact(&mut topo, |t| {
        boolean::compound_cut(t, lid, &[short, long], BooleanOptions::default()).unwrap()
    });
    let (got, expected) = (volume(&topo, both), volume(&topo, cut));
    assert!(
        (got - expected).abs() < 1e-6 * expected,
        "both pins {got}, the long pin alone {expected}"
    );
    // Material stays where the lid had it off the pin: around the axis at
    // half and 1.3 of the pin's radius, below the tail, in knuckles and gaps.
    let lid_ref = load(&mut topo, "hinge_lid_knuckled.bin");
    let lid_ref = Probe::new(&topo, lid_ref);
    let (cut, both) = (Probe::new(&topo, cut), Probe::new(&topo, both));
    let (mut kept_seen, mut removed_seen) = (0, 0);
    for x in [
        -42.58, -31.94, -21.29, -10.65, 0.0, 10.65, 21.29, 31.94, 42.58, 53.23,
    ] {
        for r in [0.5, 1.3] {
            for deg in [200.0_f64, 250.0, 300.0, 340.0] {
                let (s, c) = deg.to_radians().sin_cos();
                let p = [x, 39.35 + r * c, -3.2 + r * s];
                let Some(in_lid) = lid_ref.inside(&topo, p) else {
                    continue;
                };
                let kept = in_lid && r > 1.0;
                assert_eq!(cut.inside(&topo, p), Some(kept), "probe {p:?}");
                assert_eq!(both.inside(&topo, p), Some(kept), "probe {p:?}");
                if kept {
                    kept_seen += 1;
                } else if in_lid {
                    removed_seen += 1;
                }
            }
        }
    }
    assert_eq!((kept_seen, removed_seen), (20, 20));
}

/// The lid shut on the bin: the knuckles meet end to end and each keyhole pin
/// sits in a bore of its own radius, so the two only touch. Neither keeps a
/// face inside the other, so the intersect selects nothing: their common
/// region is empty. The tool reads this intersect's volume as the swing's
/// interference.
#[test]
fn hinge_closed_lid_only_touches_the_bin() {
    only_touches(
        "hinge_swing_bin.bin",
        "hinge_swing_lid_closed.bin",
        [-62.75, 37.0, 42.7],
        [125.5, 5.0, 7.2],
        [48, 10, 10],
    );
}

/// Swung 40 degrees about the hinge axis, the lid's knuckle ends still lie in
/// the bin's knuckle end planes, but the lid's pieces of each end face no
/// longer match the bin's, so the same-domain pass leaves the bin's piece
/// unpaired. The lid's material lies in front of it: a contact, not a shared
/// boundary.
#[test]
fn hinge_lid_swung_40_degrees_only_touches_the_bin() {
    only_touches(
        "hinge_swing_bin.bin",
        "hinge_swing_lid_40.bin",
        [-62.75, 37.0, 42.7],
        [125.5, 5.0, 7.2],
        [48, 10, 10],
    );
}

/// The left-wall hinge swung 18 degrees, a pose of the tool's corner sweep:
/// the lid's knuckle ends still lie in the bin's knuckle end planes, and the
/// circle where the lid's wider keyhole bore meets a bin knuckle end runs
/// through the bin keyhole's tail, a hole in that face. Split only at the
/// face's outer rim, the arc kept its span across the hole and the face kept a
/// piece there that ray cast read inside the lid.
#[test]
fn hinge_left_lid_swung_18_degrees_only_touches_the_bin() {
    only_touches(
        "hinge_left_bin.bin",
        "hinge_left_lid_18.bin",
        [-62.75, -41.75, 42.7],
        [5.0, 83.5, 7.2],
        [10, 32, 10],
    );
}

/// The intersect of `bin_file` and `lid_file` is exact and empty, and a grid
/// of `counts` points over the box from `lo` of `size` finds points inside
/// each and none inside both.
fn only_touches(bin_file: &str, lid_file: &str, lo: [f64; 3], size: [f64; 3], counts: [i32; 3]) {
    use brepkit_math::vec::Point3;
    use brepkit_operations::classify::{PointClassification, classify_point};

    let mut topo = Topology::new();
    let bin = load(&mut topo, bin_file);
    let lid = load(&mut topo, lid_file);
    let before = boolean::mesh_fallback_count();
    let common = boolean::boolean(&mut topo, BooleanOp::Intersect, bin, lid).unwrap();
    assert_eq!(boolean::mesh_fallback_count(), before, "mesh fallback");
    assert!(topo.is_empty_solid(common));
    // The tool's own call, which tracks each face's fate: none survives.
    let inputs: Vec<usize> = [bin, lid]
        .iter()
        .flat_map(|&s| brepkit_topology::explorer::solid_faces(&topo, s).unwrap())
        .map(brepkit_topology::arena::Id::index)
        .collect();
    let (tracked, evolution) =
        boolean::boolean_with_evolution(&mut topo, BooleanOp::Intersect, bin, lid).unwrap();
    assert_eq!(boolean::mesh_fallback_count(), before, "mesh fallback");
    assert!(topo.is_empty_solid(tracked));
    assert!(evolution.modified.is_empty());
    assert!(inputs.iter().all(|f| evolution.deleted.contains(f)));
    // No point of the hinge strip lies in both, though many lie in each.
    let (mut in_bin, mut in_lid) = (0, 0);
    for i in 0..counts[0] {
        for j in 0..counts[1] {
            for k in 0..counts[2] {
                let f = |n: i32, of: i32| (f64::from(n) + 0.37) / f64::from(of);
                let p = Point3::new(
                    size[0].mul_add(f(i, counts[0]), lo[0]),
                    size[1].mul_add(f(j, counts[1]), lo[1]),
                    size[2].mul_add(f(k, counts[2]), lo[2]),
                );
                let a = classify_point(&topo, bin, p, 0.001, 1e-6).unwrap();
                let b = classify_point(&topo, lid, p, 0.001, 1e-6).unwrap();
                let (a, b) = (
                    a == PointClassification::Inside,
                    b == PointClassification::Inside,
                );
                assert!(!(a && b), "point {p:?} inside both");
                in_bin += usize::from(a);
                in_lid += usize::from(b);
            }
        }
    }
    assert!(in_bin > 0 && in_lid > 0);
}

/// The lid on its bin overlaps the bin's lip in six slivers along the hinge
/// wall. Where the bin's knuckle neck meets a lid knuckle's end face, the
/// section circle crosses none of that face's straight edges yet runs off it,
/// so it must be split at the end face's boundary as well as at the neck's.
#[test]
fn hinge_lid_on_its_bin_overlaps_the_lip_exactly() {
    let mut topo = Topology::new();
    let bin = load(&mut topo, "hinge_seat_bin.bin");
    let lid = load(&mut topo, "hinge_seat_lid.bin");
    faithful(&topo, bin, [12, 40, 90]);
    faithful(&topo, lid, [4, 49, 143]);
    let common = exact(&mut topo, |t| {
        boolean::boolean(t, BooleanOp::Intersect, bin, lid).unwrap()
    });
    // A 216,000-point scan of one sliver's box reads 5.140 mm3 inside both
    // operands, a sixth of 30.84.
    let v = volume(&topo, common);
    assert!((v - 30.881).abs() < 1e-3 * 30.881, "volume {v}");
    // Each sliver's box, point by point against both operands.
    let probes = [bin, lid, common].map(|s| Probe::new(&topo, s));
    let f = |n: i32, of: i32| (f64::from(n) + 0.37) / f64::from(of);
    for sliver in 0..6 {
        let x0 = 21.290_909f64.mul_add(f64::from(sliver), -58.7);
        let (mut both, mut total) = (0, 0);
        for i in 0..8 {
            for j in 0..6 {
                for k in 0..6 {
                    let p = [
                        10.95f64.mul_add(f(i, 8), x0),
                        1.0f64.mul_add(f(j, 6), 37.9),
                        1.2f64.mul_add(f(k, 6), 45.5),
                    ];
                    // The winding cross-check abstains within its mesh's
                    // deflection of the lid's curved faces.
                    let [Some(in_bin), Some(in_lid), Some(in_common)] =
                        probes.each_ref().map(|probe| probe.inside(&topo, p))
                    else {
                        continue;
                    };
                    assert_eq!(in_common, in_bin && in_lid, "at {p:?}");
                    both += usize::from(in_common);
                    total += 1;
                }
            }
        }
        assert!(
            both > 0 && both < total,
            "sliver {sliver}: {both} of {total}"
        );
    }
}

/// The left-wall bin's two keyhole pins meet end to end on a knuckle's end
/// face: the short pin (keyhole 0.925) bored into that knuckle, the long one
/// (keyhole 1.0) running on through the others. The end face's ring between
/// the two keyholes coincides with the long pin's cap, the pin in front of
/// it, so the cut keeps it. Read that way, the face's split must also see the
/// short pin's rulings end on its rim circle.
#[test]
fn hinge_left_bin_pin_cut_matches_its_tools() {
    let mut topo = Topology::new();
    let bin = load(&mut topo, "hinge_left_bin_knuckled.bin");
    let short = load(&mut topo, "hinge_left_pin_short.bin");
    let long = load(&mut topo, "hinge_left_pin_long.bin");
    let cut = exact(&mut topo, |t| {
        boolean::compound_cut(t, bin, &[short, long], BooleanOptions::default()).unwrap()
    });
    // The tools and the bin by `Probe`, so the oracle side is cross-checked;
    // the cut by the ray classifier alone, because the bore's export mesh
    // fans its walls (the roadmap's developable-band row) and the winding
    // cross-check cannot read points inside the short pin's bore.
    let probes = [bin, short, long].map(|s| Probe::new(&topo, s));
    let (mut kept, mut bored) = (0, 0);
    for i in 0..10 {
        for j in 0..20 {
            for k in 0..10 {
                let f = |n: i32, of: i32| (f64::from(n) + 0.37) / f64::from(of);
                let p = [
                    3.3f64.mul_add(f(i, 10), -62.0),
                    20.0f64.mul_add(f(j, 20), -40.0),
                    4.6f64.mul_add(f(k, 10), 45.0),
                ];
                let [in_bin, in_short, in_long] = probes
                    .each_ref()
                    .map(|probe| probe.inside(&topo, p).unwrap());
                let in_cut = {
                    use brepkit_operations::classify::{PointClassification, classify_point};
                    let q = brepkit_math::vec::Point3::new(p[0], p[1], p[2]);
                    classify_point(&topo, cut, q, 0.001, 1e-6).unwrap()
                        == PointClassification::Inside
                };
                assert_eq!(in_cut, in_bin && !in_short && !in_long, "at {p:?}");
                kept += usize::from(in_cut);
                bored += usize::from(in_bin && in_short);
            }
        }
    }
    assert!(kept > 0 && bored > 0, "kept {kept}, bored {bored}");
}

/// The unify step after the cut merged the two halves of a reversed strip on
/// the bin's front lip into a face whose wire ran the other way round, so its
/// four edges ran the same way as its neighbours'.
#[test]
fn hinge_bin_clearance_cut_stays_valid() {
    let mut topo = Topology::new();
    let bin = load(&mut topo, "hinge_bin.bin");
    let rods: Vec<_> = (1..=5)
        .map(|i| load(&mut topo, &format!("hinge_bin_clearance_{i}.bin")))
        .collect();
    let cut = exact(&mut topo, |t| {
        boolean::compound_cut(t, bin, &rods, BooleanOptions::default()).unwrap()
    });
    let (before, after) = (volume(&topo, bin), volume(&topo, cut));
    assert!(
        (before - after).abs() < 1e-6 * before,
        "the rods only touch the lip: {before} became {after}"
    );
}
