//! A plate 22 thick whose outline is one closed B-spline bending at two
//! tips to a radius under 8, its wall split at half height into two
//! faces. Both rims rounded at r = 8.
//!
//! No ball rolls round those tips: the round's contact on each cap, the
//! outline inset by the radius, runs backwards there and crosses itself,
//! and the fillet was refused. The round now carries on round each tip and
//! is cut off where it folds back over itself, along a crease from the
//! crossing on the cap down to where the fold ends. Read from STEP, as a
//! file gives the part.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;
use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const R: f64 = 8.0;
const HALF: f64 = 11.0;

fn load(topo: &mut Topology) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tight_tip_plate.step");
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

/// The plate's closed B-spline rim at height `z`, and the outline's
/// tightest bend radius.
fn rim_at(topo: &Topology, solid: SolidId, z: f64) -> (EdgeId, f64) {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find_map(|e| {
            let edge = topo.edge(e).unwrap();
            let EdgeCurve::NurbsCurve(nurbs) = edge.curve() else {
                return None;
            };
            if edge.start() != edge.end()
                || (topo.vertex(edge.start()).unwrap().point().z() - z).abs() > 1e-6
            {
                return None;
            }
            let (u0, u1) = nurbs.domain();
            let tightest = (0..=20_000)
                .map(|k| {
                    let d = nurbs.derivatives((u1 - u0).mul_add(f64::from(k) / 20_000.0, u0), 2);
                    d[1].length().powi(3) / d[1].cross(d[2]).length()
                })
                .fold(f64::MAX, f64::min);
            Some((e, tightest))
        })
        .expect("a closed rim at that height")
}

/// Each rim's round takes off its section, `r^2 (1 - pi / 4)`, swept along
/// the path of its centroid, `c` in from the wall: `L - 2 pi c` round the
/// outline of length `L` (Pappus; the wall stands square to the plate, so
/// `L` is its area over its height). The folds change that by under
/// `1e-3 r^3` a rim.
#[test]
fn a_plate_whose_outline_bends_tighter_than_the_round_is_rounded_on_both_rims() {
    let mut topo = Topology::new();
    let solid = load(&mut topo);
    let walls: f64 = solid_faces(&topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_)))
        .map(|f| face_area(&topo, f, 1e-4).unwrap())
        .sum();
    let perimeter = walls / (2.0 * HALF);
    let centroid = R * 3.0f64.mul_add(-PI, 10.0) / (3.0 * (4.0 - PI));
    let removed = 2.0 * R * R * (1.0 - PI / 4.0) * 2.0f64.mul_add(-PI * centroid, perimeter);

    let ((bottom, bend), (top, _)) = (rim_at(&topo, solid, -HALF), rim_at(&topo, solid, HALF));
    assert!(bend < R - 0.5, "the outline bends to {bend} only");
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let result = fillet_v2(&mut topo, solid, &[bottom, top], R).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(
        solid_faces(&topo, result.solid).unwrap().len(),
        16,
        "the two caps, the two walls, and each rim's round in two between \
         the tips and two faces round each tip"
    );
    let mesh = tessellate_solid(&topo, result.solid, 0.05).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    let allowance = 2.0 * R * R * R * 1e-3;
    assert!(
        (lost - removed).abs() <= allowance,
        "lost {lost}, the smooth round {removed}, allowed {allowance}"
    );
}
