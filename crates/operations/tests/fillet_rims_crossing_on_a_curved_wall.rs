//! Rounds on both rims of a curved wall, wider together than the wall is
//! high.
//!
//! The part is a slab whose one side is a deep arc: the profile `y = -B`,
//! an arc of radius `R` bulging well out to `x > X0`, `y = B` and `x = 0`,
//! extruded `H` up `z`. Rounding the arc's top and bottom rims at `r` puts
//! each round's contact on the wall `r` from its own rim, so past `r = H / 2`
//! the contacts have passed each other and the two bands overlap. Such
//! rounds are refused. On a curved wall each contact was read by the chord
//! between its two ends, which stood well off the contact, so the passing
//! went unseen and the fillet built a solid that crossed itself, its volume
//! and area both wrong. The wall is the arc's cylinder, or the same arc as a
//! B-spline (a wall read from a file).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_geometry::convert::curve_to_nurbs::circle_to_nurbs;
use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const B: f64 = 5.0;
const R: f64 = 6.0;
const X0: f64 = 10.0;
const H: f64 = 4.0;

/// The slab, its arc counter-clockwise about `+z` from `(X0, -B)` to
/// `(X0, B)`, as a circle or as a B-spline.
fn slab(topo: &mut Topology, spline: bool) -> SolidId {
    let corners = [(X0, -B), (X0, B), (0.0, B), (0.0, -B)];
    let ids: Vec<_> = corners
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
        .collect();
    let centre = Point3::new(X0 - (R * R - B * B).sqrt(), 0.0, 0.0);
    let circle = Circle3D::new(centre, Vec3::new(0.0, 0.0, 1.0), R).unwrap();
    let arc = if spline {
        let (t0, mut t1) = (
            circle.project(Point3::new(X0, -B, 0.0)),
            circle.project(Point3::new(X0, B, 0.0)),
        );
        if t1 < t0 {
            t1 += std::f64::consts::TAU;
        }
        EdgeCurve::NurbsCurve(circle_to_nurbs(&circle, t0, t1).unwrap())
    } else {
        EdgeCurve::Circle(circle)
    };
    let edges: Vec<_> = (0..ids.len())
        .map(|i| {
            let curve = if i == 0 { arc.clone() } else { EdgeCurve::Line };
            let edge = Edge::new(ids[i], ids[(i + 1) % ids.len()], curve);
            OrientedEdge::new(topo.add_edge(edge), true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), H).unwrap()
}

/// The arc's rims: the curved edges of `solid`, at the top and the bottom.
fn rims(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
    let found: Vec<EdgeId> = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&edge_id| !matches!(topo.edge(edge_id).unwrap().curve(), EdgeCurve::Line))
        .collect();
    assert_eq!(found.len(), 2, "the slab has two curved rims");
    found
}

#[test]
fn rounds_passing_each_other_on_a_curved_wall_are_refused() {
    for spline in [false, true] {
        for r in [0.55 * H, 0.6 * H] {
            let mut topo = Topology::new();
            let solid = slab(&mut topo, spline);
            let edges = rims(&topo, solid);
            if let Ok(result) = fillet_v2(&mut topo, solid, &edges, r) {
                assert!(
                    !result.failed.is_empty(),
                    "spline {spline} r {r}: rounds wider together than the wall were built"
                );
            }
        }
    }
}

#[test]
fn rounds_leaving_wall_between_them_build() {
    for spline in [false, true] {
        let r = 0.4 * H;
        let mut topo = Topology::new();
        let solid = slab(&mut topo, spline);
        let before = solid_volume(&topo, solid, 1e-3).unwrap();
        let edges = rims(&topo, solid);
        let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
        assert!(
            result.failed.is_empty(),
            "spline {spline}: {:?}",
            result.failed
        );
        let report = validate_solid(&topo, result.solid).unwrap();
        assert!(report.is_valid(), "spline {spline}: {:?}", report.issues);
        let after = solid_volume(&topo, result.solid, 1e-3).unwrap();
        assert!(
            after < before,
            "spline {spline}: the rounds took nothing off"
        );
    }
}
