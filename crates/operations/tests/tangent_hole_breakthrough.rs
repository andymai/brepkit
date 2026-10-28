//! A tilted hole whose side runs along one wall of a plate and whose end
//! breaks out through another: the cut keeps its cylinder and takes the
//! volume the hole's elliptical slices give.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{
    BooleanOp, BooleanOptions, boolean, compound_cut, mesh_fallback_count,
};
use brepkit_operations::classify::classify_point;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const PLATE: f64 = 12.0;
const THICK: f64 = 2.0;
const R: f64 = 0.5;

/// A cylinder of radius `R` through the plate's mid-plane at `(x, y)`, its
/// axis turned `tilt` about x.
fn hole(topo: &mut Topology, x: f64, y: f64, tilt: f64) -> SolidId {
    let c = make_cylinder(topo, R, 8.0).unwrap();
    let place = Mat4::translation(x, y, THICK / 2.0)
        * Mat4::rotation_x(tilt)
        * Mat4::translation(0.0, 0.0, -4.0);
    transform_solid(topo, c, &place).unwrap();
    c
}

/// The volume a hole centred at `y` takes from the plate. Each horizontal
/// slice is an ellipse with semi-axes `R` and `R / cos(tilt)` whose centre
/// moves `tan(tilt)` in y per unit of height, less any cap past the wall at
/// `y = PLATE`. Swapping the slices end for end turns the tilt over, so
/// its sign does not matter.
fn removed(y: f64, tilt: f64) -> f64 {
    let (a, b) = (R, R / tilt.cos());
    let slice = |z: f64| {
        let h = (PLATE - (y + (z - THICK / 2.0) * tilt.tan())) / b;
        let cap = if h >= 1.0 {
            0.0
        } else {
            a * b * ((1.0 - h) * (1.0 + h)).sqrt().mul_add(-h, h.acos())
        };
        PI * a * b - cap
    };
    // Composite Simpson's rule; the cap's onset is a kink of order 3/2.
    let n = 200_000;
    let step = THICK / f64::from(n);
    let mut sum = slice(0.0) + slice(THICK);
    for i in 1..n {
        sum += if i % 2 == 1 { 4.0 } else { 2.0 } * slice(f64::from(i) * step);
    }
    sum * step / 3.0
}

fn census(topo: &Topology, solid: SolidId) -> BTreeMap<&'static str, usize> {
    let mut kinds = BTreeMap::new();
    for f in solid_faces(topo, solid).unwrap() {
        *kinds
            .entry(topo.face(f).unwrap().surface().type_tag())
            .or_insert(0) += 1;
    }
    kinds
}

#[test]
fn a_hole_along_one_wall_and_out_through_another_cuts_exactly() {
    for tilt in [0.2, 0.35] {
        let mut topo = Topology::default();
        let plate = make_box(&mut topo, PLATE, PLATE, THICK).unwrap();
        // Its side runs along x = 12 from top to bottom, and its lower end
        // leaves through y = 12.
        let tool = hole(&mut topo, PLATE - R, PLATE - R, tilt);
        let fallbacks = mesh_fallback_count();
        let cut = boolean(&mut topo, BooleanOp::Cut, plate, tool).unwrap();
        assert_eq!(
            mesh_fallback_count(),
            fallbacks,
            "tilt {tilt}: mesh fallback"
        );
        assert_eq!(
            census(&topo, cut),
            BTreeMap::from([("cylinder", 1), ("plane", 7)]),
            "tilt {tilt}"
        );
        assert!(
            validate_solid(&topo, cut).unwrap().is_valid(),
            "tilt {tilt}"
        );
        let want = PLATE * PLATE * THICK - removed(PLATE - R, tilt);
        let got = solid_volume(&topo, cut, 0.001).unwrap();
        assert!(
            (got - want).abs() < 1e-6,
            "tilt {tilt}: volume {got} against {want}"
        );

        // Material in the corner the hole cuts off at the bottom, above it
        // at the top, and beside its grazing line; the hole's middle, and
        // the hole beside that line.
        let at = |x: f64, y: f64, z: f64| {
            format!(
                "{:?}",
                classify_point(&topo, cut, Point3::new(x, y, z), 0.1, 1e-7).unwrap()
            )
        };
        assert_eq!(at(11.995, 11.995, 0.02), "Inside", "tilt {tilt}");
        assert_eq!(at(11.99, 11.9, 1.9), "Inside", "tilt {tilt}");
        assert_eq!(at(11.99, 11.1, 1.0), "Inside", "tilt {tilt}");
        assert_eq!(at(11.5, 11.5, 1.0), "Outside", "tilt {tilt}");
        assert_eq!(at(11.95, 11.5, 1.0), "Outside", "tilt {tilt}");
    }
}

#[test]
fn a_grid_of_tilted_holes_reaching_two_walls_cuts_exactly() {
    let tilt = 0.2;
    let mut topo = Topology::default();
    let plate = make_box(&mut topo, PLATE, PLATE, THICK).unwrap();
    // Six by six holes 2 apart from 1.5: the last column runs along x = 12
    // and the last row breaks out through y = 12.
    let mut tools = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            let (x, y) = (
                2.0f64.mul_add(f64::from(i), 1.5),
                2.0f64.mul_add(f64::from(j), 1.5),
            );
            tools.push(hole(&mut topo, x, y, tilt));
        }
    }
    let fallbacks = mesh_fallback_count();
    let cut = compound_cut(&mut topo, plate, &tools, BooleanOptions::default()).unwrap();
    assert_eq!(mesh_fallback_count(), fallbacks, "mesh fallback");
    assert_eq!(
        census(&topo, cut),
        BTreeMap::from([("cylinder", 36), ("plane", 7)])
    );
    assert!(validate_solid(&topo, cut).unwrap().is_valid());
    let inside = removed(PLATE / 2.0, tilt);
    let want = 6.0f64.mul_add(
        -removed(PLATE - R, tilt),
        30.0f64.mul_add(-inside, PLATE * PLATE * THICK),
    );
    let got = solid_volume(&topo, cut, 0.001).unwrap();
    assert!((got - want).abs() < 1e-5, "volume {got} against {want}");
}
