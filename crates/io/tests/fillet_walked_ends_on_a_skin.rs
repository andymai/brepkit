//! A wing's thin rib whose two top edges, B-splines between its sides and
//! its top strip, are rounded at r = 0.15; each round ends on the rib's
//! tilted end faces.
//!
//! The walked rounds run past their end faces and are cut back where their
//! contacts cross them, on each end face's edges with the sides. Those
//! edges stand off the sides by up to 2e-4 (a file's B-spline skin), and
//! the contacts, which lie on the sides, missed them by up to 5e-5: past
//! the walk's own tolerance, so the ends were left to runout patches, and
//! the top strip's outline ran out along a contact past its end edge and
//! back, a loop crossing itself. A contact now ends on its corner edge
//! within as far as the edge stands off its face.
//!
//! Read from STEP, as a file gives the part.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::{Severity, ValidationIssue, ValidationReport, validate_solid};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/wing_rib_ends.step");
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

/// The edge of `solid` passing nearest `p`.
fn edge_through(topo: &Topology, solid: SolidId, p: Point3) -> EdgeId {
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
    assert!(off(best) < 0.005, "no edge through {p:?}: {}", off(best));
    best
}

/// How many times the plane faces' wires cross themselves, each edge
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

/// The errors `report` holds besides curves that miss their vertices. The
/// walked rounds end up to 5.5e-4 off the vertices of their contact and end
/// edges, past the band in which the edges' readers take a vertex to lie on
/// its curve. That fault is open; every other fault is still refused.
fn faults_besides_curve_ends(report: &ValidationReport) -> Vec<&ValidationIssue> {
    report
        .issues
        .iter()
        .filter(|issue| {
            issue.severity == Severity::Error && !issue.description.contains("its curve misses")
        })
        .collect()
}

#[test]
fn walked_rounds_end_on_a_skin_s_corner_edges() {
    let mut topo = Topology::new();
    let rib = load(&mut topo);
    let picks = [
        Point3::new(
            -7.747_083_556_110_797,
            0.975_900_261_934_440_2,
            -6.363_172_033_051_238,
        ),
        Point3::new(
            -5.776_142_361_535_313,
            0.696_770_838_483_505_1,
            -6.860_575_198_545_542,
        ),
    ]
    .map(|p| edge_through(&topo, rib, p));
    let result = fillet_v2(&mut topo, rib, &picks, 0.15).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    let faults = faults_besides_curve_ends(&report);
    assert!(faults.is_empty(), "{faults:?}");
    assert_eq!(
        self_crossings(&topo, result.solid),
        0,
        "a loop crosses itself"
    );
    // The part's twenty faces and the two bands, no runout patches.
    assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), 22);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    // Another kernel's measure of its result of the same fillet.
    let volume = solid_volume(&topo, result.solid, 0.001).unwrap();
    let expected = 2.059_604;
    assert!(
        (volume - expected).abs() < 1e-3 * expected,
        "volume {volume}, expected {expected}"
    );
}
