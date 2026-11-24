//! A rod's top rim chamfered where a block stands on the rod's top, narrower
//! than the rod one way and overhanging it a little the other. The rim is in
//! arcs: convex ones on the two crescents of the rod's top beside the block,
//! concave ones where the rod meets the block's overhanging underside. Each
//! crescent's bevel is cut off square at the block's walls rising from it;
//! the concave arcs, whose underside is narrower than the chamfer all along
//! them, are left as they are, as another kernel leaves them, and so is the
//! rod's seam when it is picked too.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// The rod's radius and height.
const R: f64 = 2.5;
const H: f64 = 5.0;
/// The block's half-width across x (inside the rod) and the chamfer.
const HALF: f64 = 1.7;
const D: f64 = 0.7;

/// A rod of radius `R`, `H` tall, with a block `2 HALF` by `2 half_y` by 5
/// standing on its top, centred on its axis.
fn rod_under_a_block(topo: &mut Topology, half_y: f64) -> SolidId {
    let rod = make_cylinder(topo, R, H).unwrap();
    let block = make_box(topo, 2.0 * HALF, 2.0 * half_y, 5.0).unwrap();
    transform_solid(topo, block, &Mat4::translation(-HALF, -half_y, H)).unwrap();
    boolean(topo, BooleanOp::Fuse, rod, block).unwrap()
}

/// The rod's rim arcs, and its seam (the line along its wall) when `seam`.
fn rim(topo: &Topology, solid: SolidId, seam: bool) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            match edge.curve() {
                EdgeCurve::Circle(c) => {
                    (c.center().z() - H).abs() < 1e-9 && (c.radius() - R).abs() < 1e-9
                }
                EdgeCurve::Line => {
                    seam && (a.z() - b.z()).abs() > H - 1e-9
                        && (a.x().hypot(a.y()) - R).abs() < 1e-9
                }
                _ => false,
            }
        })
        .collect()
}

/// The crescents' bevel: `(r - (R - D))` deep over the part of each ring of
/// radius `r` outside the block, `4 acos(HALF / r)` of a turn.
fn crescents_bevel() -> f64 {
    let n = 4000;
    let (lo, hi) = (R - D, R);
    let f = |r: f64| (r - lo) * r * 4.0 * (HALF / r).acos();
    let h = (hi - lo) / f64::from(n);
    let mut sum = f(lo) + f(hi);
    for k in 1..n {
        sum += f(lo + h * f64::from(k)) * if k % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

fn chamfered(seam: bool) {
    let mut topo = Topology::new();
    let part = rod_under_a_block(&mut topo, R + 0.2);
    let edges = rim(&topo, part, seam);
    assert!(edges.len() >= 4, "the rim in arcs: {}", edges.len());
    let before = solid_volume(&topo, part, 0.01).unwrap();
    let result = chamfer(&mut topo, part, &edges, D).unwrap();
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    let removed = before - solid_volume(&topo, result, 0.01).unwrap();
    let want = crescents_bevel();
    assert!(
        (removed - want).abs() < 1e-6 * want,
        "took off {removed}, want {want}"
    );
    let cones = solid_faces(&topo, result)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cone(_)))
        .count();
    assert_eq!(cones, 2, "a cone on each crescent");
}

#[test]
fn a_rim_under_an_overhanging_block_is_chamfered_on_its_crescents() {
    chamfered(false);
}

#[test]
fn the_rods_seam_picked_with_the_rim_is_left() {
    chamfered(true);
}

#[test]
fn a_concave_arc_the_chamfer_fits_beside_is_not_left() {
    // The block overhangs the rod by more than the chamfer: a chamfer fits
    // under it, so leaving its arcs would be a chamfer silently not made.
    let mut topo = Topology::new();
    let part = rod_under_a_block(&mut topo, R + 1.5);
    let edges = rim(&topo, part, false);
    let before = solid_volume(&topo, part, 0.01).unwrap();
    if let Ok(result) = chamfer(&mut topo, part, &edges, D) {
        let removed = before - solid_volume(&topo, result, 0.01).unwrap();
        assert!(
            (removed - crescents_bevel()).abs() > 1e-3,
            "the concave arcs were left: took off {removed}"
        );
    }
}
