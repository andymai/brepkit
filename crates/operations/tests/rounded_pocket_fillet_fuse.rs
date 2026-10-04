//! An interior fillet added to a rounded pocket the way the gridfinity tool's
//! bins build one: the pocket's air, drawn as a rounded rectangle and rounded
//! along its floor edges, is cut from the same outline grown into the walls
//! and floor, and that rounding material is fused into the pocket. The
//! fillet's corner tori rest on the floor and touch the coaxial corner wall
//! along the tube's outer equator. An L-shaped pocket adds a reflex corner,
//! whose torus wraps the wall's arc from outside.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

const CORNER: f64 = 2.55;
const FLOOR: f64 = 2.25;
const HEIGHT: f64 = 12.0;
const FILLET_TOP: f64 = 8.0;
const SKIN: f64 = 0.48;

fn edges_where(
    topo: &Topology,
    solid: SolidId,
    keep: impl Fn(Point3, Point3) -> bool,
) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            let a = topo.vertex(edge.start()).unwrap().point();
            let b = topo.vertex(edge.end()).unwrap().point();
            keep(a, b)
        })
        .collect()
}

fn block(topo: &mut Topology, min: [f64; 3], max: [f64; 3]) -> SolidId {
    let solid = make_box(topo, max[0] - min[0], max[1] - min[1], max[2] - min[2]).unwrap();
    transform_solid(topo, solid, &Mat4::translation(min[0], min[1], min[2])).unwrap();
    solid
}

fn round_verticals(topo: &mut Topology, solid: SolidId, edges: &[EdgeId], radius: f64) -> SolidId {
    let rounded = fillet_v2(topo, solid, edges, radius).unwrap();
    assert!(rounded.failed.is_empty(), "{:?}", rounded.failed);
    rounded.solid
}

/// A pocket outline: the rectangle `[-15, 15] x [-10, 10]`, or that
/// rectangle less its corner `[5, 15] x [2, 10]`. Each has a straight run of
/// 100 before its corners are rounded.
#[derive(Clone, Copy)]
enum Outline {
    Rectangle,
    L,
}

impl Outline {
    /// The outline grown by `grow`, its corners concentric with the
    /// pocket's, as a prism between two heights.
    fn prism(self, topo: &mut Topology, z0: f64, z1: f64, grow: f64) -> SolidId {
        let g = grow;
        let full = block(topo, [-15.0 - g, -10.0 - g, z0], [15.0 + g, 10.0 + g, z1]);
        let solid = match self {
            Self::Rectangle => full,
            Self::L => {
                let notch = block(topo, [5.0 + g, 2.0 + g, z0 - 1.0], [16.0, 11.0, z1 + 1.0]);
                boolean(topo, BooleanOp::Cut, full, notch).unwrap()
            }
        };
        let reflex =
            |a: Point3| (a.x() - (5.0 + g)).abs() < 1e-9 && (a.y() - (2.0 + g)).abs() < 1e-9;
        let convex = edges_where(topo, solid, |a, b| {
            (a.z() - b.z()).abs() > 1e-6 && !reflex(a)
        });
        let solid = round_verticals(topo, solid, &convex, CORNER + g);
        let reflex = edges_where(topo, solid, |a, b| {
            (a.z() - b.z()).abs() > 1e-6 && reflex(a)
        });
        if reflex.is_empty() {
            solid
        } else {
            round_verticals(topo, solid, &reflex, CORNER - g)
        }
    }

    fn corners(self) -> (f64, f64) {
        match self {
            Self::Rectangle => (4.0, 0.0),
            Self::L => (5.0, 1.0),
        }
    }
}

/// The volume a floor fillet of radius `r` takes from the outline's air: its
/// cross-section `(1 - pi/4) r^2` along the straight runs, and around each
/// corner arc (Pappus) with the section's centroid `r (5/6 - pi/4) / (1 -
/// pi/4)` in from the wall, toward the corner's axis at a convex corner and
/// away from it at a reflex one.
fn rounded_off(outline: Outline, r: f64) -> f64 {
    let area = (1.0 - FRAC_PI_4) * r * r;
    let centroid = r * (5.0 / 6.0 - FRAC_PI_4) / (1.0 - FRAC_PI_4);
    let (convex, reflex) = outline.corners();
    let straight = 2.0f64.mul_add(-CORNER * (convex + reflex), 100.0);
    let corners = convex.mul_add(CORNER - centroid, reflex * (CORNER + centroid));
    area * FRAC_PI_2.mul_add(corners, straight)
}

fn fuse_rounding_into_pocket(outline: Outline, radius: f64) {
    let mut topo = Topology::new();
    let outer = block(&mut topo, [-20.0, -15.0, 0.0], [20.0, 15.0, HEIGHT]);
    let pocket = outline.prism(&mut topo, FLOOR, HEIGHT + 1.0, 0.0);
    let bin = boolean(&mut topo, BooleanOp::Cut, outer, pocket).unwrap();

    let air = outline.prism(&mut topo, FLOOR, FILLET_TOP + 1.0, 0.0);
    let floor_edges = edges_where(&topo, air, |a, b| {
        (a.z() - FLOOR).abs() < 1e-9 && (b.z() - FLOOR).abs() < 1e-9
    });
    let rounded = fillet_v2(&mut topo, air, &floor_edges, radius).unwrap();
    assert!(
        rounded.failed.is_empty(),
        "r = {radius}: {:?}",
        rounded.failed
    );
    let grown = outline.prism(&mut topo, FLOOR - 1.0, FILLET_TOP, SKIN);
    let material = boolean(&mut topo, BooleanOp::Cut, grown, rounded.solid).unwrap();

    let fallbacks = mesh_fallback_count();
    let filled = boolean(&mut topo, BooleanOp::Fuse, bin, material).unwrap();
    assert_eq!(
        mesh_fallback_count(),
        fallbacks,
        "r = {radius}: mesh fallback"
    );
    let report = validate_solid(&topo, filled).unwrap();
    assert!(report.is_valid(), "r = {radius}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, filled, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {radius}: mesh not watertight");

    // The material adds to the bin exactly what the fillet took from the air.
    let truth = rounded_off(outline, radius);
    let added =
        solid_volume(&topo, filled, 0.001).unwrap() - solid_volume(&topo, bin, 0.001).unwrap();
    assert!(
        (added - truth).abs() < 1e-6 * solid_volume(&topo, filled, 0.001).unwrap(),
        "r = {radius}: added {added}, truth {truth}"
    );
}

#[test]
fn a_fillet_below_the_corner_radius_fuses_into_its_pocket_exactly() {
    fuse_rounding_into_pocket(Outline::Rectangle, 1.0);
}

/// Past half the corner radius the corner tori are spindles.
#[test]
fn a_fillet_nearly_filling_its_corners_fuses_into_its_pocket_exactly() {
    fuse_rounding_into_pocket(Outline::Rectangle, 2.45);
}

#[test]
fn an_l_shaped_pocket_takes_its_fillet_exactly() {
    fuse_rounding_into_pocket(Outline::L, 1.0);
}

/// The classifier reads points against the spindle tori themselves: their
/// rims' flat polygon stood in front of the pocket's opposite wall.
#[test]
fn an_l_shaped_pocket_takes_a_fillet_nearly_filling_its_corners_exactly() {
    fuse_rounding_into_pocket(Outline::L, 2.45);
}
