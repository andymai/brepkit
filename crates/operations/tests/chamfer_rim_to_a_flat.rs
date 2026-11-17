//! A round rim that ends on a flat parallel to its axis: the top arc of a
//! shaft with a flat ground along its side, and the top rim of a hole with a
//! slot cut from it out to the plate's edge. The rim's chamfer is a cone,
//! which meets the flat in a hyperbola from the flat's top edge down to the
//! ruling where the flat meets the round wall.
//!
//! The result's volume is read off its cross-sections across the axis. At
//! height `z` the chamfer moves the rim's circle to the radius `rho(z)` on
//! the cone, so each cross-section is the plain one with that radius in
//! place of the rim's.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::chamfer_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::adjacency::AdjacencyIndex;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// The shaft: radius 5, 6 high, its flat at `x = 3`.
const SHAFT: (f64, f64, f64) = (5.0, 6.0, 3.0);

/// The plate: 20 square and 4 thick about a hole of radius 3, with a slot
/// 2 wide from the hole out along `+x`.
const PLATE: (f64, f64, f64, f64) = (10.0, 4.0, 3.0, 1.0);

/// Upright, turned about one axis and about all three, and mirrored.
fn poses() -> Vec<(&'static str, Mat4)> {
    vec![
        ("upright", Mat4::translation(0.0, 0.0, 0.0)),
        ("turned", Mat4::rotation_z(0.7)),
        (
            "tilted",
            Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
        ),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ]
}

/// `f` integrated from `a` to `b`, over which it is smooth.
fn integral(a: f64, b: f64, f: impl Fn(f64) -> f64) -> f64 {
    gauss_legendre_points(20)
        .iter()
        .map(|g| 0.5 * (b - a) * g.w * f(0.5 * (b - a) * g.x + 0.5 * (a + b)))
        .sum()
}

/// The shaft with its flat, placed by `pose`.
fn shaft(topo: &mut Topology, pose: &Mat4) -> SolidId {
    let (r, h, flat) = SHAFT;
    let rod = make_cylinder(topo, r, h).unwrap();
    let cutter = make_box(topo, 10.0, 20.0, h + 2.0).unwrap();
    transform_solid(topo, cutter, &Mat4::translation(flat, -10.0, -1.0)).unwrap();
    let shaft = boolean(topo, BooleanOp::Cut, rod, cutter).unwrap();
    transform_solid(topo, shaft, pose).unwrap();
    shaft
}

/// The plate with its hole and slot, placed by `pose`.
fn plate(topo: &mut Topology, pose: &Mat4) -> SolidId {
    let (half, h, r, c) = PLATE;
    let block = make_box(topo, 2.0 * half, 2.0 * half, h).unwrap();
    transform_solid(topo, block, &Mat4::translation(-half, -half, 0.0)).unwrap();
    let drill = make_cylinder(topo, r, h + 2.0).unwrap();
    transform_solid(topo, drill, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    let holed = boolean(topo, BooleanOp::Cut, block, drill).unwrap();
    let slot = make_box(topo, half + 5.0, 2.0 * c, h + 2.0).unwrap();
    transform_solid(topo, slot, &Mat4::translation(0.0, -c, -1.0)).unwrap();
    let plate = boolean(topo, BooleanOp::Cut, holed, slot).unwrap();
    transform_solid(topo, plate, pose).unwrap();
    plate
}

/// The arcs of radius `r` at height `h`, read in the solid's own frame.
fn rim(topo: &Topology, solid: SolidId, pose: &Mat4, r: f64, h: f64) -> Vec<EdgeId> {
    let inverse = pose.inverse().unwrap();
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            let p = inverse.mul_point(topo.vertex(edge.start()).unwrap().point());
            matches!(edge.curve(), EdgeCurve::Circle(c) if (c.radius() - r).abs() < 1e-9)
                && (p.z() - h).abs() < 1e-9
        })
        .collect()
}

/// The rim's distance into its cap and along its wall when `chamfer_v2`
/// sets `d1` on its first face in the solid's face order.
fn setback(topo: &Topology, solid: SolidId, edge: EdgeId, (d1, d2): (f64, f64)) -> (f64, f64) {
    let first = AdjacencyIndex::build(topo, solid)
        .unwrap()
        .faces_for_edge(edge)[0];
    match topo.face(first).unwrap().surface() {
        FaceSurface::Cylinder(_) => (d2, d1),
        _ => (d1, d2),
    }
}

/// The radius of the rim's circle at height `z`, moved `cap` into the cap
/// at the top `h` and `wall` down the wall, outward from a hole.
fn rho(r: f64, h: f64, (cap, wall): (f64, f64), outward: bool, z: f64) -> f64 {
    let gone = (z - (h - wall)).max(0.0) * cap / wall;
    if outward { r + gone } else { r - gone }
}

/// The shaft's volume: at each height the disc of radius `rho` less its
/// segment beyond the flat.
fn shaft_volume(setback: (f64, f64)) -> f64 {
    let (r, h, flat) = SHAFT;
    let section = |z: f64| {
        let rho = rho(r, h, setback, false, z);
        PI * rho * rho - (rho * rho * (flat / rho).acos() - flat * (rho * rho - flat * flat).sqrt())
    };
    integral(0.0, h - setback.1, section) + integral(h - setback.1, h, section)
}

/// The plate's volume: at each height the square less the hole of radius
/// `rho` and the slot, which overlap where the slot leaves the hole.
fn plate_volume(setback: (f64, f64)) -> f64 {
    let (half, h, r, c) = PLATE;
    let section = |z: f64| {
        let rho = rho(r, h, setback, true, z);
        let overlap = c * (rho * rho - c * c).sqrt() + rho * rho * (c / rho).asin();
        4.0 * half * half - (PI * rho * rho + 2.0 * c * half - overlap)
    };
    integral(0.0, h - setback.1, section) + integral(h - setback.1, h, section)
}

/// How many of the faces of `solid` are cones, cylinders and planes.
fn census(topo: &Topology, solid: SolidId) -> (usize, usize, usize) {
    let mut counts = (0, 0, 0);
    for f in solid_faces(topo, solid).unwrap() {
        match topo.face(f).unwrap().surface() {
            FaceSurface::Cone(_) => counts.0 += 1,
            FaceSurface::Cylinder(_) => counts.1 += 1,
            FaceSurface::Plane { .. } => counts.2 += 1,
            other => panic!("unexpected {}", other.type_tag()),
        }
    }
    counts
}

/// Valid, watertight, oriented outward, `expected` in volume, and holding
/// each of `kept` but none of `cut`, given in the solid's own frame.
fn assert_exact(
    topo: &Topology,
    result: SolidId,
    expected: f64,
    pose: &Mat4,
    (kept, cut): (&[[f64; 3]], &[[f64; 3]]),
    what: &str,
) {
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    let exact = solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (exact - expected).abs() < 1e-9 * expected,
        "{what}: volume {exact}, expected {expected}"
    );
    assert!(
        is_watertight(&tessellate_solid(topo, result, 0.01).unwrap()),
        "{what}: its mesh is open"
    );
    let meshed = oriented_solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (meshed - expected).abs() < 1e-3 * expected,
        "{what}: meshed volume {meshed}, expected {expected}"
    );
    let class = |p: [f64; 3]| {
        let at = pose.mul_point(Point3::new(p[0], p[1], p[2]));
        classify_point(topo, result, at, &ClassifyOptions::default()).unwrap()
    };
    for &p in kept {
        assert_eq!(
            class(p),
            PointClassification::Inside,
            "{what}: {p:?} is kept"
        );
    }
    for &p in cut {
        assert_eq!(
            class(p),
            PointClassification::Outside,
            "{what}: {p:?} is cut"
        );
    }
}

/// The shaft's top arc chamfered at 0.5: the cone cuts a sliver from the
/// flat between its top edge and the rulings at its sides.
#[test]
fn a_shafts_top_arc_ends_on_its_flat() {
    let d = 0.5;
    let expected = shaft_volume((d, d));
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let solid = shaft(&mut topo, &pose);
        let edges = rim(&topo, solid, &pose, SHAFT.0, SHAFT.1);
        assert!(!edges.is_empty(), "{name}");

        let result = chamfer(&mut topo, solid, &edges, d).unwrap();

        assert_exact(
            &topo,
            result,
            expected,
            &pose,
            (
                &[[0.0, 4.4, 5.9], [2.95, 3.95, 5.4], [2.95, 2.0, 5.95]],
                &[[0.0, 4.95, 5.95], [2.95, 3.95, 5.95], [2.95, -3.95, 5.95]],
            ),
            name,
        );
        // One cone over each of the rim's arcs.
        assert_eq!(census(&topo, result), (edges.len(), 1, 3), "{name}");
    }
}

/// The shaft's top arc through `chamfer_v2` with two distances.
#[test]
fn a_shafts_top_arc_takes_two_distances() {
    let distances = (0.5, 0.3);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let solid = shaft(&mut topo, &pose);
        let edges = rim(&topo, solid, &pose, SHAFT.0, SHAFT.1);
        let expected = shaft_volume(setback(&topo, solid, edges[0], distances));

        let result = chamfer_v2(&mut topo, solid, &edges, distances.0, distances.1).unwrap();

        assert!(result.failed.is_empty(), "{name}: {:?}", result.failed);
        assert_exact(
            &topo,
            result.solid,
            expected,
            &pose,
            (&[[0.0, 4.3, 5.9]], &[[0.0, 4.97, 5.97]]),
            name,
        );
    }
}

/// The hole's top rim, cut by the slot, chamfered at 0.5: a countersink
/// that runs out on the slot's two sides.
#[test]
fn a_holes_rim_ends_on_the_sides_of_its_slot() {
    let d = 0.5;
    let expected = plate_volume((d, d));
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let solid = plate(&mut topo, &pose);
        let edges = rim(&topo, solid, &pose, PLATE.2, PLATE.1);
        assert!(!edges.is_empty(), "{name}");

        let result = chamfer(&mut topo, solid, &edges, d).unwrap();

        let x = 8.0_f64.sqrt();
        assert_exact(
            &topo,
            result,
            expected,
            &pose,
            (
                &[
                    [0.0, 3.6, 3.9],
                    [x + 0.05, 1.05, 3.4],
                    [x + 1.0, 1.05, 3.95],
                ],
                &[
                    [0.0, 3.05, 3.95],
                    [x + 0.05, 1.05, 3.95],
                    [x + 0.05, -1.05, 3.95],
                ],
            ),
            name,
        );
        assert_eq!(census(&topo, result).0, edges.len(), "{name}");
    }
}

/// The hole's rim through `chamfer_v2` with two distances.
#[test]
fn a_holes_rim_takes_two_distances() {
    let distances = (0.5, 0.3);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let solid = plate(&mut topo, &pose);
        let edges = rim(&topo, solid, &pose, PLATE.2, PLATE.1);
        let expected = plate_volume(setback(&topo, solid, edges[0], distances));

        let result = chamfer_v2(&mut topo, solid, &edges, distances.0, distances.1).unwrap();

        assert!(result.failed.is_empty(), "{name}: {:?}", result.failed);
        assert_exact(
            &topo,
            result.solid,
            expected,
            &pose,
            (&[[0.0, 3.7, 3.9]], &[[0.0, 3.03, 3.97]]),
            name,
        );
    }
}
