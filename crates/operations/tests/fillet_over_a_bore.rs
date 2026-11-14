//! A round running over a through-bore's mouth: a box drilled through near
//! one top edge, that edge rounded wider than the bore stands from it. The
//! round's contact on the top crosses the bore, which the trimmer refuses.
//! The round is made as on the box with the bore filled, and the bore is cut
//! back through it. The round takes off its strip, (1 - pi / 4) r^2 per
//! unit length, less the part of the strip already in the bore.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use brepkit_blend::fillet_builder::FilletBuilder;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::{all_blended, fillet_v2};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::imprint::{ImprintTool, imprint_faces};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_line_edge;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const LENGTH: f64 = 40.0;
const WIDTH: f64 = 20.0;
const HEIGHT: f64 = 15.0;

/// A box drilled through its height, the bore's axis at (`cx`, `cy`), and
/// its top edge along x at y = 0.
fn drilled_box(topo: &mut Topology, cx: f64, cy: f64, rho: f64) -> (SolidId, EdgeId) {
    let block = make_box(topo, LENGTH, WIDTH, HEIGHT).unwrap();
    let drill = make_cylinder(topo, rho, HEIGHT + 2.0).unwrap();
    transform_solid(topo, drill, &Mat4::translation(cx, cy, -1.0)).unwrap();
    let solid = boolean(topo, BooleanOp::Cut, block, drill).unwrap();
    let edge = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                p.y().abs() < 1e-9 && (p.z() - HEIGHT).abs() < 1e-9
            })
        })
        .unwrap();
    (solid, edge)
}

/// The part of the round's strip (depth `r - sqrt(r^2 - (r - y)^2)` below
/// the top at `y` from the side) inside the bore.
fn strip_in_bore(r: f64, cy: f64, rho: f64) -> f64 {
    let (lo, hi) = ((cy - rho).max(0.0), (cy + rho).min(r));
    if hi <= lo {
        return 0.0;
    }
    let n = 20_000;
    let h = (hi - lo) / f64::from(n);
    let f = |y: f64| {
        let chord = 2.0 * (rho * rho - (y - cy) * (y - cy)).max(0.0).sqrt();
        let depth = r - (r * r - (r - y) * (r - y)).max(0.0).sqrt();
        chord * depth
    };
    let mut sum = f(lo) + f(hi);
    for i in 1..n {
        let y = lo + h * f64::from(i);
        sum += f(y) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

fn check(cy: f64, rho: f64, r: f64, pose: &Mat4) {
    let label = format!("bore at {cy} radius {rho}, r {r}");
    let mut topo = Topology::new();
    let (solid, edge) = drilled_box(&mut topo, 0.5 * LENGTH, cy, rho);
    transform_solid(&mut topo, solid, pose).unwrap();
    let want = (1.0 - FRAC_PI_4) * r * r * LENGTH - strip_in_bore(r, cy, rho);
    assert_rounds_off(&label, &mut topo, solid, edge, r, want);
}

/// Rounds `edge` of `solid` by `r`, and checks the result is valid and
/// watertight, has one face more, and is short by `want`.
fn assert_rounds_off(
    label: &str,
    topo: &mut Topology,
    solid: SolidId,
    edge: EdgeId,
    r: f64,
    want: f64,
) {
    let before = solid_volume(topo, solid, 0.01).unwrap();
    let faces = solid_faces(topo, solid).unwrap().len();
    let result = fillet_v2(topo, solid, &[edge], r)
        .and_then(all_blended)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    let mesh = tessellate_solid(topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "{label}: mesh not watertight");
    assert_eq!(
        solid_faces(topo, result).unwrap().len(),
        faces + 1,
        "{label}: faces"
    );
    let removed = before - solid_volume(topo, result, 0.01).unwrap();
    assert!(
        (removed - want).abs() <= 1e-4 * want,
        "{label}: removed {removed}, want {want}"
    );
}

#[test]
fn a_round_over_a_bore_cuts_back_through_it() {
    // The bore wholly inside the round's reach, and straddling its contact.
    for (cy, rho, r) in [(5.0, 2.0, 8.0), (6.0, 2.0, 7.0), (4.0, 1.5, 5.0)] {
        check(cy, rho, r, &Mat4::identity());
    }
}

#[test]
fn a_turned_bore_rounds_alike() {
    let pose =
        Mat4::translation(-37.2, 5.5, 140.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9);
    check(6.0, 2.0, 7.0, &pose);
}

/// A mirror turns each face's boundary the other way round its normal.
#[test]
fn a_mirrored_bore_rounds_alike() {
    check(6.0, 2.0, 7.0, &Mat4::scale(-1.0, 1.0, 1.0));
}

/// `solid` less a cylinder of radius `rho` standing on (`cx`, `cy`) from
/// height `z0` to `z1`.
fn drill(
    topo: &mut Topology,
    solid: SolidId,
    centre: (f64, f64),
    rho: f64,
    z: (f64, f64),
) -> SolidId {
    let drill = make_cylinder(topo, rho, z.1 - z.0).unwrap();
    transform_solid(topo, drill, &Mat4::translation(centre.0, centre.1, z.0)).unwrap();
    boolean(topo, BooleanOp::Cut, solid, drill).unwrap()
}

/// Two bores on the top under the round are both cut back through it. Of a
/// bore under the round and one past it, only the first is filled, and the
/// trimmed top keeps the other as its hole.
#[test]
fn a_round_over_two_bores_cuts_back_each_it_reaches() {
    let r = 7.0;
    let strip = (1.0 - FRAC_PI_4) * r * r * LENGTH;
    for (far, under) in [((5.0, 1.5), true), ((14.0, 1.5), false)] {
        let mut topo = Topology::new();
        let (solid, _) = drilled_box(&mut topo, 12.0, 6.0, 2.0);
        let solid = drill(&mut topo, solid, (28.0, far.0), far.1, (-1.0, HEIGHT + 1.0));
        let edge = line_edge(&topo, solid, [0.0, 0.0, HEIGHT], [LENGTH, 0.0, HEIGHT]);
        let mut want = strip - strip_in_bore(r, 6.0, 2.0);
        if under {
            want -= strip_in_bore(r, far.0, far.1);
        }
        let label = format!("second bore at {}", far.0);
        assert_rounds_off(&label, &mut topo, solid, edge, r, want);
    }
}

/// A counterbore, wide from below up to z = 7 and narrow from there to the
/// top: the narrow part is a bore from the step to the top, and the round of
/// the top edge, which stays above the wide part, cuts it back.
#[test]
fn a_round_over_a_counterbores_narrow_part_cuts_it_back() {
    let (cy, rho, r) = (6.0, 2.0, 7.0);
    let mut topo = Topology::new();
    let (solid, _) = drilled_box(&mut topo, 0.5 * LENGTH, cy, rho);
    let solid = drill(&mut topo, solid, (0.5 * LENGTH, cy), 3.5, (-1.0, 7.0));
    let edge = line_edge(&topo, solid, [0.0, 0.0, HEIGHT], [LENGTH, 0.0, HEIGHT]);
    let want = (1.0 - FRAC_PI_4) * r * r * LENGTH - strip_in_bore(r, cy, rho);
    assert_rounds_off("counterbore", &mut topo, solid, edge, r, want);
}

/// The straight edge of `solid` from `a` to `b`.
fn line_edge(topo: &Topology, solid: SolidId, a: [f64; 3], b: [f64; 3]) -> EdgeId {
    let (a, b) = (Point3::new(a[0], a[1], a[2]), Point3::new(b[0], b[1], b[2]));
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            let (p, q) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            matches!(edge.curve(), EdgeCurve::Line)
                && (((p - a).length() < 1e-9 && (q - b).length() < 1e-9)
                    || ((p - b).length() < 1e-9 && (q - a).length() < 1e-9))
        })
        .unwrap()
}

fn inside(topo: &Topology, solid: SolidId, p: [f64; 3]) -> bool {
    classify_point(topo, solid, Point3::new(p[0], p[1], p[2]), 0.0, 1e-7).unwrap()
        == PointClassification::Inside
}

/// A U channel 30 x 20 x 15, its slot between the walls at x = 10 and
/// x = 20 down to z = 5, with a pin of radius 1.5 along x across the slot at
/// height `z`, and the slot's top edge on the wall at x = 10.
fn pinned_channel(topo: &mut Topology, z: f64) -> (SolidId, EdgeId) {
    let block = make_box(topo, 30.0, 20.0, 15.0).unwrap();
    let slot = make_box(topo, 10.0, 22.0, 11.0).unwrap();
    transform_solid(topo, slot, &Mat4::translation(10.0, -1.0, 5.0)).unwrap();
    let channel = boolean(topo, BooleanOp::Cut, block, slot).unwrap();
    let pin = make_cylinder(topo, 1.5, 12.0).unwrap();
    let across = Mat4::translation(9.0, 10.0, z) * Mat4::rotation_y(FRAC_PI_2);
    transform_solid(topo, pin, &across).unwrap();
    let solid = boolean(topo, BooleanOp::Fuse, channel, pin).unwrap();
    let edge = line_edge(topo, solid, [10.0, 0.0, 15.0], [10.0, 20.0, 15.0]);
    (solid, edge)
}

/// Rounds `edges` by `r` and checks the round is the direct fillet's, with
/// no bore filled: the same volume and face count.
fn round_as_the_direct_fillet(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    r: f64,
) -> SolidId {
    let mut builder = FilletBuilder::new(topo, solid);
    builder.add_edges(edges, r);
    let direct = all_blended(builder.build().unwrap()).unwrap();
    let result = fillet_v2(topo, solid, edges, r)
        .and_then(all_blended)
        .unwrap();
    let [made, want] = [result, direct].map(|s| {
        (
            solid_volume(topo, s, 0.01).unwrap(),
            solid_faces(topo, s).unwrap().len(),
        )
    });
    assert!(
        (made.0 - want.0).abs() <= 1e-9 * want.0 && made.1 == want.1,
        "made {made:?}, the direct fillet {want:?}"
    );
    result
}

/// A pin across a slot meets each wall in a circle, as a bore meets its
/// faces, but its material lies inside its wall: it is no bore, and a round
/// of the slot's edge keeps it. Clear of the band, the round takes off its
/// strip alone. Under the band, the round is the direct fillet's.
#[test]
fn a_pin_under_a_round_is_kept() {
    let mut topo = Topology::new();
    let (solid, edge) = pinned_channel(&mut topo, 10.0);
    let before = solid_volume(&topo, solid, 0.01).unwrap();
    let result = fillet_v2(&mut topo, solid, &[edge], 1.0)
        .and_then(all_blended)
        .unwrap();
    assert!(validate_solid(&topo, result).unwrap().is_valid());
    let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    assert!(inside(&topo, result, [15.0, 10.0, 10.0]), "pin lost");
    let removed = before - solid_volume(&topo, result, 0.01).unwrap();
    let want = (1.0 - FRAC_PI_4) * WIDTH;
    assert!(
        (removed - want).abs() <= 1e-6 * before,
        "removed {removed}, want {want}"
    );

    let mut topo = Topology::new();
    let (solid, edge) = pinned_channel(&mut topo, 13.0);
    let result = round_as_the_direct_fillet(&mut topo, solid, &[edge], 4.0);
    assert!(inside(&topo, result, [15.0, 10.0, 13.0]), "pin lost");
}

/// A spool: a hub of radius 3 between two flanges of radius 10 meets each
/// flange's inner face in a circle. Rounding a flange's rim by 2 takes off
/// the rim's ring and keeps the hub.
#[test]
fn a_spools_flange_rounds_off_and_keeps_its_hub() {
    let mut topo = Topology::new();
    let near = make_cylinder(&mut topo, 10.0, 3.0).unwrap();
    let far = make_cylinder(&mut topo, 10.0, 3.0).unwrap();
    transform_solid(&mut topo, far, &Mat4::translation(0.0, 0.0, 9.0)).unwrap();
    let hub = make_cylinder(&mut topo, 3.0, 8.0).unwrap();
    transform_solid(&mut topo, hub, &Mat4::translation(0.0, 0.0, 2.0)).unwrap();
    let flanges = boolean(&mut topo, BooleanOp::Fuse, near, far).unwrap();
    let spool = boolean(&mut topo, BooleanOp::Fuse, flanges, hub).unwrap();
    let rim = solid_edges(&topo, spool)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(c) if (c.radius() - 10.0).abs() < 1e-9)
                && (topo.vertex(edge.start()).unwrap().point().z() - 3.0).abs() < 1e-9
        })
        .unwrap();
    let before = solid_volume(&topo, spool, 0.001).unwrap();
    let result = fillet_v2(&mut topo, spool, &[rim], 2.0)
        .and_then(all_blended)
        .unwrap();
    assert!(validate_solid(&topo, result).unwrap().is_valid());
    let mesh = tessellate_solid(&topo, result, 0.02).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    assert!(inside(&topo, result, [0.0, 0.0, 6.0]), "hub lost");
    // The ring's section, the square of side r less a quarter disc, turned
    // about the axis at its centroid's radius.
    let r: f64 = 2.0;
    let centroid = r * 3.0f64.mul_add(-PI, 10.0) / (3.0 * (4.0 - PI));
    let want = 2.0 * PI * (10.0 - centroid) * (1.0 - FRAC_PI_4) * r * r;
    let removed = before - solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (removed - want).abs() <= 1e-6 * before,
        "removed {removed}, want {want}"
    );
}

/// An L bracket: a base 20 x 20 x 5 and a wall 5 thick standing on it to
/// z = 20, drilled through the base at (9, 10) by a bore of radius 2. The
/// inner corner, concave, rounded by 6 adds material up to 1.5 over the
/// bore's mouth, which a cut back a tenth of the bore's radius past the mouth
/// leaves as a ledge in the bore: the round is the direct fillet's.
#[test]
fn a_concave_round_over_a_bore_is_the_direct_fillets() {
    let mut topo = Topology::new();
    let base = make_box(&mut topo, 20.0, 20.0, 5.0).unwrap();
    let wall = make_box(&mut topo, 5.0, 20.0, 20.0).unwrap();
    let bracket = boolean(&mut topo, BooleanOp::Fuse, base, wall).unwrap();
    let drill = make_cylinder(&mut topo, 2.0, 7.0).unwrap();
    transform_solid(&mut topo, drill, &Mat4::translation(9.0, 10.0, -1.0)).unwrap();
    let solid = boolean(&mut topo, BooleanOp::Cut, bracket, drill).unwrap();
    let corner = line_edge(&topo, solid, [5.0, 0.0, 5.0], [5.0, 20.0, 5.0]);
    round_as_the_direct_fillet(&mut topo, solid, &[corner], 6.0);
}

/// The box drilled at (20, `cy`) by a bore of radius `rho`, its top marked
/// along y = 15 into two faces, and its top edge along x at y = 0.
fn marked_box(topo: &mut Topology, cy: f64, rho: f64) -> (SolidId, EdgeId) {
    let (drilled, _) = drilled_box(topo, 0.5 * LENGTH, cy, rho);
    let mark = make_line_edge(
        topo,
        Point3::new(-1.0, 15.0, HEIGHT),
        Point3::new(LENGTH + 1.0, 15.0, HEIGHT),
        1e-7,
    )
    .unwrap();
    let marked = imprint_faces(topo, drilled, &[ImprintTool::Edge(mark)]).unwrap();
    let edge = line_edge(topo, marked, [0.0, 0.0, HEIGHT], [LENGTH, 0.0, HEIGHT]);
    (marked, edge)
}

/// The marked box drilled at (20, 8) by a bore of radius 2. A round of 2
/// reaches 2 into the top, short of the bore 6 away, and one of 1 reaches 1:
/// each is the direct fillet's, and keeps the mark.
#[test]
fn a_round_short_of_a_bore_is_the_direct_fillets() {
    for r in [2.0, 1.0] {
        let mut topo = Topology::new();
        let (marked, edge) = marked_box(&mut topo, 8.0, 2.0);
        round_as_the_direct_fillet(&mut topo, marked, &[edge], r);
    }
}

/// The marked box drilled at (20, 6) by a bore of radius 2, under a round
/// of 7: the bore is cut back through the round, and the cut merges no
/// faces of the solid, so the top keeps its mark.
#[test]
fn a_round_over_a_bore_keeps_the_marks_round_it() {
    let (cy, rho, r) = (6.0, 2.0, 7.0);
    let mut topo = Topology::new();
    let (marked, edge) = marked_box(&mut topo, cy, rho);
    let before = solid_volume(&topo, marked, 0.01).unwrap();
    let result = fillet_v2(&mut topo, marked, &[edge], r)
        .and_then(all_blended)
        .unwrap();
    assert!(validate_solid(&topo, result).unwrap().is_valid());
    let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    let tops = solid_faces(&topo, result)
        .unwrap()
        .into_iter()
        .filter(|&f| {
            matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal, d }
                if (normal.z().abs() - 1.0).abs() < 1e-9 && (d.abs() - HEIGHT).abs() < 1e-9)
        })
        .count();
    assert_eq!(tops, 2, "the top's faces");
    let removed = before - solid_volume(&topo, result, 0.01).unwrap();
    let want = (1.0 - FRAC_PI_4) * r * r * LENGTH - strip_in_bore(r, cy, rho);
    assert!(
        (removed - want).abs() <= 1e-4 * want,
        "removed {removed}, want {want}"
    );
}

/// The box slit through along y between each pair of heights in `slits`,
/// from x = 5 to 35, and bored from the top down to z = 7.05 at (20, 6) by
/// a bore of radius 2, with its top edge along x at y = 0.
fn slit_box(topo: &mut Topology, slits: &[(f64, f64)]) -> (SolidId, EdgeId) {
    let mut solid = make_box(topo, LENGTH, WIDTH, HEIGHT).unwrap();
    for &(low, high) in slits {
        let slit = make_box(topo, 30.0, WIDTH + 2.0, high - low).unwrap();
        transform_solid(topo, slit, &Mat4::translation(5.0, -1.0, low)).unwrap();
        solid = boolean(topo, BooleanOp::Cut, solid, slit).unwrap();
    }
    let solid = drill(topo, solid, (0.5 * LENGTH, 6.0), 2.0, (7.05, HEIGHT + 1.0));
    let edge = line_edge(topo, solid, [0.0, 0.0, HEIGHT], [LENGTH, 0.0, HEIGHT]);
    (solid, edge)
}

/// A slit between z = 7 and 7.1, onto which the bore's lower mouth opens:
/// the slit is thinner than the tenth of the bore's radius the cut back
/// runs past the mouth, so the cut back would cut into the slit's floor.
/// The round is the direct fillet's.
#[test]
fn a_bore_onto_a_thin_slit_is_left_to_the_direct_fillet() {
    let mut topo = Topology::new();
    let (solid, edge) = slit_box(&mut topo, &[(7.0, 7.1)]);
    round_as_the_direct_fillet(&mut topo, solid, &[edge], 7.0);
}

/// The same slit with a second one below it, between z = 6.8 and 6.95: the
/// cut back ends at z = 6.9, in the second slit, but on its way crosses the
/// shelf between the two. The round is the direct fillet's, and the shelf
/// keeps its material under the bore.
#[test]
fn a_bore_onto_a_shelf_between_two_slits_is_left_to_the_direct_fillet() {
    let mut topo = Topology::new();
    let (solid, edge) = slit_box(&mut topo, &[(7.0, 7.1), (6.8, 6.95)]);
    let result = round_as_the_direct_fillet(&mut topo, solid, &[edge], 7.0);
    assert!(
        inside(&topo, result, [0.5 * LENGTH, 6.0, 6.975]),
        "shelf cut"
    );
}
