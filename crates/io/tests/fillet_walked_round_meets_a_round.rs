//! A walked round meeting another round at a corner whose third edge is
//! left sharp, past the other round's strip.
//!
//! A flat cut into a tapered peg's end leaves a lobe whose inner face `F`
//! meets the end face `E` square and the peg's wall at an acute angle. Its
//! outline on `F` is rounded at r = 1: a straight round along `F`'s edge
//! with `E` (a cylinder) and a walked round along its edges with the wall.
//! The wall leans in over `F`, so the walked round's section at the corner
//! stops short of `E` on the wall: it reaches farther along `E` than the
//! straight round's strip. It is carried on to `E`, its contact on the wall
//! ending on the wall's rim with `E`, the straight round ends on it along
//! a crease, and `E` is notched by the walked round's last section. Before
//! this, both rounds ran to the corner and a patch closed the gap, the
//! straight round's contact on `E` running past `E`'s rim (`E`'s loop
//! crossing itself).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
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

/// The mesh edges used by one triangle only, the mesh's points welded at
/// 1e-5, as a consumer would.
fn welded_open_edges(topo: &Topology, solid: SolidId, deflection: f64) -> usize {
    let (mesh, _) = brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance(
        topo, solid, deflection, 0.5,
    )
    .unwrap();
    #[allow(clippy::cast_possible_truncation)]
    let key = |p: Point3| {
        (
            (p.x() * 1e5).round() as i64,
            (p.y() * 1e5).round() as i64,
            (p.z() * 1e5).round() as i64,
        )
    };
    let mut weld = std::collections::HashMap::new();
    let ids: Vec<usize> = mesh
        .positions
        .iter()
        .map(|&p| {
            let n = weld.len();
            *weld.entry(key(p)).or_insert(n)
        })
        .collect();
    let mut uses = std::collections::HashMap::new();
    for t in mesh.indices.chunks_exact(3) {
        for (i, j) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (a, b) = (ids[i as usize], ids[j as usize]);
            if a != b {
                *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
    }
    uses.values().filter(|&&n| n == 1).count()
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
fn a_walked_round_runs_on_to_the_end_face_past_the_straight_round() {
    let mut topo = Topology::new();
    let peg = load(&mut topo, "peg_flat_corner_rounds.step");
    let picks = [
        Point3::new(4.5, 30.0, 0.0),
        Point3::new(4.5, 25.001_192_399_121_788, -10.711_280_128_285_926),
        Point3::new(4.5, 25.001_192_399_121_784, 10.711_280_128_285_923),
    ]
    .map(|p| edge_near(&topo, peg, p, 0.05));
    let result = fillet_v2(&mut topo, peg, &picks, 1.0).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(
        self_crossings(&topo, result.solid),
        0,
        "a loop crosses itself"
    );
    // The part's thirteen faces and the three rounds: no corner patches.
    assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), 16);
    for deflection in [0.5, 0.05] {
        let open = welded_open_edges(&topo, result.solid, deflection);
        assert_eq!(open, 0, "{open} open mesh edges at {deflection}");
    }
    // Another kernel's measure of its result of the same fillet.
    let volume = solid_volume(&topo, result.solid, 0.01).unwrap();
    let expected = 41_389.252;
    assert!(
        (volume - expected).abs() < 1e-5 * expected,
        "volume {volume}, expected {expected}"
    );
}
