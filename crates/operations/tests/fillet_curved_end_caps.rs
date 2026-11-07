//! A round of a straight edge whose end meets a curved end face.
//!
//! The part is a D: the profile `x = 0`, `y = -B`, `y = B` closed by an arc
//! of a circle of radius `R` about `(C, 0)` bulging out to `x > X0`,
//! extruded `H` up `z`. The edges along `x` where the top or bottom meets
//! a side `y = B` end on the arc's cylinder, which is neither square to
//! them nor flat. The round now runs on to that face: each contact to
//! where its line meets the cylinder, on the cylinder's corner edges (the
//! top or bottom arc and the vertical line), and the band ends on its
//! intersection with the cylinder, which the cylinder is notched along. It
//! had ended in the section plane, closed by a runout patch, and the result
//! failed the assembly's incidence audit.
//!
//! The bottom face walks its arc against the circle's sense. The trim
//! splits that arc where the round's contact ends, and the piece kept had
//! run the long way round from the split: the bottom read a whole disc too
//! large.
//!
//! The round takes off, along the edge, the corner's strip outside the
//! round, out to the cylinder: at a distance `t` in from the side face the
//! strip is `r - sqrt(r^2 - t^2)` deep and runs out to `C + sqrt(R^2 -
//! (B - r + t)^2)`, which integrates (with `t = r sin u`) to the removed
//! volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_2;

use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const B: f64 = 5.0;
const R: f64 = 8.0;
const X0: f64 = 10.0;
const H: f64 = 6.0;

/// The arc's centre on the x axis.
fn centre() -> f64 {
    X0 - (R * R - B * B).sqrt()
}

/// The D, its arc counter-clockwise about `+z` from `(X0, -B)` to `(X0, B)`.
fn d_prism(topo: &mut Topology) -> SolidId {
    let corners = [(X0, -B), (X0, B), (0.0, B), (0.0, -B)];
    let ids: Vec<_> = corners
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
        .collect();
    let arc = Circle3D::new(Point3::new(centre(), 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), R).unwrap();
    let edges: Vec<_> = (0..ids.len())
        .map(|i| {
            let curve = if i == 0 {
                EdgeCurve::Circle(arc.clone())
            } else {
                EdgeCurve::Line
            };
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

/// The straight edge of `solid` along `x` at `y = B` and height `z`.
fn side_edge(topo: &Topology, solid: SolidId, z: f64) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            let [p, q] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
            matches!(edge.curve(), EdgeCurve::Line)
                && [p, q]
                    .iter()
                    .all(|p| (p.y() - B).abs() < 1e-9 && (p.z() - z).abs() < 1e-9)
                && (p.x() - q.x()).abs() > 1.0
        })
        .expect("the edge")
}

/// The volume the round of radius `r` takes off.
fn removed(r: f64) -> f64 {
    // Simpson's rule in `u`, where the integrand is smooth.
    const N: usize = 2000;
    let c = centre();
    let f = |u: f64| {
        let t = r * u.sin();
        let depth = r * (1.0 - u.cos());
        let y = B - r + t;
        depth * (c + (R * R - y * y).sqrt()) * r * u.cos()
    };
    #[allow(clippy::cast_precision_loss)]
    let h = FRAC_PI_2 / N as f64;
    let mut sum = f(0.0) + f(FRAC_PI_2);
    for k in 1..N {
        #[allow(clippy::cast_precision_loss)]
        let u = h * k as f64;
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(u);
    }
    sum * h / 3.0
}

fn assert_round_ends_on_the_curve(z: f64, r: f64) {
    let mut topo = Topology::new();
    let solid = d_prism(&mut topo);
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let faces = solid_faces(&topo, solid).unwrap().len();
    let edge = side_edge(&topo, solid, z);
    let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
    assert!(result.failed.is_empty(), "z {z} r {r}: {:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "z {z} r {r}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "z {z} r {r}: mesh open");
    assert_eq!(
        solid_faces(&topo, result.solid).unwrap().len(),
        faces + 1,
        "z {z} r {r}: the round alone is new"
    );
    let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    let expected = removed(r);
    assert!(
        (lost - expected).abs() <= 1e-6 * before,
        "z {z} r {r}: removed {lost}, the strip out to the cylinder {expected}"
    );
}

#[test]
fn top_edge_ending_on_the_cylinder() {
    for r in [1.0, 2.0] {
        assert_round_ends_on_the_curve(H, r);
    }
}

#[test]
fn bottom_edge_ending_on_the_cylinder() {
    for r in [1.0, 2.0] {
        assert_round_ends_on_the_curve(0.0, r);
    }
}
