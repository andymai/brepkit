//! A disc with two flats down its sides has its top rim rounded: two pieces
//! of one torus over the arcs and a cylinder over each flat, meeting at
//! mitres. A plane across the rounds, below the top, cuts all four pieces,
//! and each torus piece only along its own arc of the plane's circle, also
//! when one piece runs more than half way round and its box holds the
//! other's arc.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::split::split;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 60.0;
const HEIGHT: f64 = 40.0;
const ROUND: f64 = 10.0;

fn block(topo: &mut Topology, from: [f64; 3], to: [f64; 3]) -> SolidId {
    let b = make_box(topo, to[0] - from[0], to[1] - from[1], to[2] - from[2]).unwrap();
    transform_solid(topo, b, &Mat4::translation(from[0], from[1], from[2])).unwrap();
    b
}

/// The disc, flattened at `flat` from its axis on the sides `sides` (an
/// axis, 0 for x or 1 for y, and a sign), its top rim rounded.
fn rounded_disc(topo: &mut Topology, flat: f64, sides: [(usize, f64); 2]) -> SolidId {
    let mut disc = make_cylinder(topo, RADIUS, HEIGHT).unwrap();
    for (axis, sign) in sides {
        let (a, b) = (sign * flat, sign * (RADIUS + 5.0));
        let mut from = [-RADIUS - 5.0, -RADIUS - 5.0, -5.0];
        let mut to = [RADIUS + 5.0, RADIUS + 5.0, HEIGHT + 5.0];
        (from[axis], to[axis]) = (a.min(b), a.max(b));
        let cutter = block(topo, from, to);
        disc = boolean(topo, BooleanOp::Cut, disc, cutter).unwrap();
    }
    let top = solid_faces(topo, disc)
        .unwrap()
        .into_iter()
        .find(|&f| match topo.face(f).unwrap().surface() {
            FaceSurface::Plane { normal, d } => {
                normal.z().abs() > 0.5 && (d * normal.z() - HEIGHT).abs() < 1e-9
            }
            _ => false,
        })
        .unwrap();
    let rim: Vec<_> = topo
        .wire(topo.face(top).unwrap().outer_wire())
        .unwrap()
        .edges()
        .iter()
        .map(brepkit_topology::wire::OrientedEdge::edge)
        .collect();
    assert!(rim.len() >= 4, "the arcs and two flats");
    fillet_v2(topo, disc, &rim, ROUND).unwrap().solid
}

fn kinds(topo: &Topology, solid: SolidId) -> (usize, usize, usize) {
    let faces = solid_faces(topo, solid).unwrap();
    let count = |want: fn(&FaceSurface) -> bool| {
        faces
            .iter()
            .filter(|&&f| want(topo.face(f).unwrap().surface()))
            .count()
    };
    (
        faces.len(),
        count(|s| matches!(s, FaceSurface::Torus(_))),
        count(|s| matches!(s, FaceSurface::Cylinder(_))),
    )
}

#[test]
fn a_plane_across_the_rounds_cuts_each_piece() {
    let opposite = [(0, -1.0), (0, 1.0)];
    let beside = [(0, 1.0), (1, 1.0)];
    for (flat, sides) in [(58.0, opposite), (50.0, opposite), (58.0, beside)] {
        let mut topo = Topology::new();
        let solid = rounded_disc(&mut topo, flat, sides);
        let (_, tori, _) = kinds(&topo, solid);
        assert!(tori >= 2, "flat {flat} {sides:?}: the rim's arcs");
        let whole = solid_volume(&topo, solid, 1e-3).unwrap();
        let at = Point3::new(0.0, 0.0, HEIGHT - 0.4 * ROUND);
        let halves = split(&mut topo, solid, at, Vec3::new(0.0, 0.0, 1.0)).unwrap();
        let mut total = 0.0;
        for (half, name) in [(halves.positive, "top"), (halves.negative, "bottom")] {
            let report = validate_solid(&topo, half).unwrap();
            assert!(
                report.is_valid(),
                "flat {flat} {sides:?}, {name}: {:?}",
                report.issues
            );
            let mesh = tessellate_solid(&topo, half, 0.05).unwrap();
            assert!(
                is_watertight(&mesh),
                "flat {flat} {sides:?}, {name}: open mesh"
            );
            let (faces, cut_tori, cylinders) = kinds(&topo, half);
            // Exact pieces, not a mesh: the cap or the base, the cut, the
            // torus pieces and the two flats' rounds, and below, the wall's
            // arcs and two flats.
            assert!(faces <= 12, "flat {flat} {sides:?}, {name}: {faces} faces");
            assert_eq!(
                cut_tori, tori,
                "flat {flat} {sides:?}, {name}: torus pieces"
            );
            assert!(
                cylinders >= 2,
                "flat {flat} {sides:?}, {name}: {cylinders} cylinders"
            );
            total += solid_volume(&topo, half, 1e-3).unwrap();
        }
        assert!(
            (total - whole).abs() < 1e-6 * whole,
            "flat {flat} {sides:?}: halves hold {total}, the whole {whole}"
        );
    }
}
