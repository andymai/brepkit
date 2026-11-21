//! Rounds whose contact on a ledge runs off the ledge's outline before the
//! round's end, near a tip tighter than the round.
//!
//! Each round ended in the plane square to its spine, closed by a runout
//! patch, and the ledge's outline ran out along the contact past its own
//! edge and back: a loop crossing itself. The walls under the ledge now
//! meet the band, which ends on its intersections with them, and the ledge
//! ends where the contact comes back on to it.
//!
//! - A plate whose B-spline wall stands up from a ledge, the wall's foot
//!   rounded at r = 30 (concave: the walls under the ledge run up to the
//!   band). At one end the contact runs off across the tip's arc, at the
//!   other past it and across the straight edge beyond.
//! - A plate's top rim rounded at r = 50 along a B-spline wall, the top
//!   narrowing to a tip at one end (convex: the walls under the top's
//!   outline are cut down to the band).
//! - A plate whose B-spline wall ends at a boss's cylinder wall, the
//!   wall's foot rounded at r = 30: the cylinder stands above the ledge
//!   beside the round's end and runs on under the ledge's outline, a
//!   little past the band's end section, so the cylinder itself runs up
//!   to the band, which is run on past its end to meet it.
//!
//! Read from STEP, as a file gives the part.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

/// The edge of `solid` passing nearest `p`.
fn edge_through(topo: &Topology, solid: SolidId, p: Point3) -> EdgeId {
    edge_near(topo, solid, p, 0.05)
}

/// The edge of `solid` passing nearest `p`, within `within` of it.
fn edge_near(topo: &Topology, solid: SolidId, p: Point3, within: f64) -> EdgeId {
    let off = |e: EdgeId| {
        let edge = topo.edge(e).unwrap();
        let (a, b) = (
            topo.vertex(edge.start()).unwrap().point(),
            topo.vertex(edge.end()).unwrap().point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        (0..=20_000)
            .map(|k| {
                let t = (t1 - t0).mul_add(f64::from(k) / 20_000.0, t0);
                (edge.curve().evaluate_with_endpoints(t, a, b) - p).length()
            })
            .fold(f64::MAX, f64::min)
    };
    let edges = solid_edges(topo, solid).unwrap();
    let best = edges
        .iter()
        .copied()
        .min_by(|&a, &b| off(a).total_cmp(&off(b)))
        .unwrap();
    assert!(off(best) < within, "no edge through {p:?}: {}", off(best));
    best
}

/// How many times the plane face's wires cross themselves, each edge
/// sampled in the plane.
fn self_crossings(topo: &Topology, solid: SolidId) -> usize {
    let mut hits = 0;
    for face_id in solid_faces(topo, solid).unwrap() {
        let face = topo.face(face_id).unwrap();
        let FaceSurface::Plane { normal, .. } = face.surface() else {
            continue;
        };
        let across = if normal.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let u = normal.cross(across).normalize().unwrap();
        let v = normal.cross(u);
        for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let mut poly: Vec<(f64, f64)> = Vec::new();
            for oe in topo.wire(wire).unwrap().edges() {
                let edge = topo.edge(oe.edge()).unwrap();
                let a = topo.vertex(edge.start()).unwrap().point();
                let b = topo.vertex(edge.end()).unwrap().point();
                let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                for k in 0..32 {
                    let f = f64::from(k) / 32.0;
                    let f = if oe.is_forward() { f } else { 1.0 - f };
                    let p = edge
                        .curve()
                        .evaluate_with_endpoints((t1 - t0).mul_add(f, t0), a, b);
                    let p = Vec3::new(p.x(), p.y(), p.z());
                    poly.push((p.dot(u), p.dot(v)));
                }
            }
            let n = poly.len();
            let side = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
                (b.0 - a.0).mul_add(c.1 - a.1, -((b.1 - a.1) * (c.0 - a.0)))
            };
            for i in 0..n {
                for j in i + 2..n {
                    if i == 0 && j == n - 1 {
                        continue;
                    }
                    let (a, b, c, d) = (poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]);
                    // Strictly on either side: samples of one straight
                    // edge lie on a line, within rounding.
                    let ab = (b.0 - a.0).hypot(b.1 - a.1);
                    let cd = (d.0 - c.0).hypot(d.1 - c.1);
                    let eps = 1e-9 * ab.max(cd).powi(2).max(1e-300);
                    let apart = |x: f64, y: f64| (x > eps && y < -eps) || (x < -eps && y > eps);
                    if apart(side(a, b, c), side(a, b, d)) && apart(side(c, d, a), side(c, d, b)) {
                        hits += 1;
                    }
                }
            }
        }
    }
    hits
}

#[test]
fn the_walls_under_a_ledge_run_up_to_a_round_run_off_it() {
    let mut topo = Topology::new();
    let plate = load(&mut topo, "leg_ledge_tips.step");
    let picks = [
        Point3::new(-499.785_853_451_442_2, -237.953_139_994_871_66, 0.0),
        Point3::new(-179.578_193_242_226_77, -92.161_784_578_229_76, 0.0),
        Point3::new(-625.108_818_585_219_4, 48.109_225_697_939_96, 0.0),
    ]
    .map(|p| edge_through(&topo, plate, p));
    let result = fillet_v2(&mut topo, plate, &picks, 30.0).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(
        self_crossings(&topo, result.solid),
        0,
        "a loop crosses itself"
    );
    // The part's fourteen faces and the band, no runout patches.
    assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), 15);
    let mesh = tessellate_solid(&topo, result.solid, 0.5).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    // Another kernel's result of the same fillet, as brepkit measures it
    // finely. The band is a B-spline face, which the measure takes off its
    // mesh: at 0.1 it reads 9e-5 light.
    let volume = solid_volume(&topo, result.solid, 0.1).unwrap();
    let expected = 91_719_505.63;
    assert!(
        (volume - expected).abs() < 1e-4 * expected,
        "volume {volume}, expected {expected}"
    );
}

#[test]
fn the_walls_under_a_tip_are_cut_down_to_a_rim_round_run_off_it() {
    let mut topo = Topology::new();
    let plate = load(&mut topo, "leg_ankle_tip.step");
    let picks = [Point3::new(
        -125.183_950_877_462_46,
        -467.764_179_096_078_07,
        120.0,
    )]
    .map(|p| edge_through(&topo, plate, p));
    let result = fillet_v2(&mut topo, plate, &picks, 50.0).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(
        self_crossings(&topo, result.solid),
        0,
        "a loop crosses itself"
    );
    // The part's fifteen faces and the band, no runout patch.
    assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), 16);
    let mesh = tessellate_solid(&topo, result.solid, 0.5).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    // Another kernel's measure of its result of the same fillet.
    let volume = solid_volume(&topo, result.solid, 0.1).unwrap();
    let expected = 37_674_025.42;
    assert!(
        (volume - expected).abs() < 1e-5 * expected,
        "volume {volume}, expected {expected}"
    );
}

#[test]
fn a_wall_beside_a_round_s_end_runs_on_under_the_ledge_and_up_to_the_band() {
    let mut topo = Topology::new();
    let plate = load(&mut topo, "leg_ledge_wall_corner.step");
    // As a pick gives them: points beside the edges (one 0.9 off its
    // B-spline edge).
    let picks = [
        Point3::new(-627.894_148_889_057_9, 41.256_629_256_441_26, 0.0),
        Point3::new(-545.043_640_911_132_8, -235.758_747_827_397_15, 0.0),
        Point3::new(-272.577_438_603_194_2, -137.983_627_301_885_04, 0.0),
    ]
    .map(|p| edge_near(&topo, plate, p, 1.0));
    let result = fillet_v2(&mut topo, plate, &picks, 30.0).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(
        self_crossings(&topo, result.solid),
        0,
        "a loop crosses itself"
    );
    // The part's seventeen faces and the band, no runout patches.
    assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), 18);
    let mesh = tessellate_solid(&topo, result.solid, 0.5).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    // Another kernel's measure of its result of the same fillet. The band
    // is a B-spline face, which the measure takes off its mesh: at 0.1 it
    // reads 1e-4 light.
    let volume = solid_volume(&topo, result.solid, 0.1).unwrap();
    let expected = 93_180_093.95;
    assert!(
        (volume - expected).abs() < 2e-4 * expected,
        "volume {volume}, expected {expected}"
    );
}
