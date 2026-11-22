//! A slab's end rounded at both corners past half its width.
//!
//! The slab runs `X` along `x`, its end face at `x = X` is `W` wide in `y`
//! between its sides, and its top leans: it stands `H + S x` high. The two
//! vertical edges of the end are rounded by `r`, with `W / 2 < r < W`, so
//! each round's contact on the end face lies past the other's: the two
//! rounds cross there. The first round is made as it would be alone; the
//! second rests on its side and pivots on the first's contact on the end
//! face, so the end face goes and the two rounds meet along that line.
//!
//! Across the end, the first round keeps `x <= X - r + sqrt(r^2 - (y - r)^2)`
//! for `y <= r`, and the second `x <= X - h + sqrt(r^2 - (y - W + r)^2)`
//! beyond, its centre `h = sqrt(r^2 - (W - 2 r)^2)` from the end face. The
//! volume taken off is the removed area under the leaning top.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const X: f64 = 400.0;
const W: f64 = 300.0;
const H: f64 = 60.0;
const S: f64 = 0.1;

/// The slab: its side profile in `y = 0`, extruded `W` along `y`.
fn slab(topo: &mut Topology) -> SolidId {
    let points = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, H),
        Point3::new(X, 0.0, S.mul_add(X, H)),
        Point3::new(X, 0.0, 0.0),
    ];
    let ids: Vec<_> = points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let edges: Vec<_> = (0..ids.len())
        .map(|i| {
            let edge = Edge::new(ids[i], ids[(i + 1) % ids.len()], EdgeCurve::Line);
            OrientedEdge::new(topo.add_edge(edge), true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: 0.0 },
    ));
    extrude(topo, face, normal, W).unwrap()
}

/// The end's vertical edge on the side `y`, placed by `pose`.
fn end_edge(topo: &Topology, solid: SolidId, y: f64, pose: &Mat4) -> EdgeId {
    let (a, b) = (
        pose.mul_point(Point3::new(X, y, 0.0)),
        pose.mul_point(Point3::new(X, y, S.mul_add(X, H))),
    );
    let near = |p: Point3, q: Point3| (p - q).length() < 1e-9;
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            let [p, q] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
            (near(p, a) && near(q, b)) || (near(p, b) && near(q, a))
        })
        .expect("the end's edge")
}

/// The volume the two rounds take off, by Simpson's rule across the end,
/// in each round's angle about its axis (the arcs meet their sides square).
fn removed(r: f64) -> f64 {
    const N: usize = 2000;
    let h = (r * r - (W - 2.0 * r).powi(2)).sqrt();
    // The removed strip at `y` ending at `x`, under the leaning top.
    let strip = |x: f64| H * (X - x) + S * (X * X - x * x) / 2.0;
    let simpson = |f: &dyn Fn(f64) -> f64, a: f64, b: f64| {
        let step = (b - a) / N as f64;
        let mut sum = f(a) + f(b);
        for i in 1..N {
            sum += f((i as f64).mul_add(step, a)) * if i % 2 == 1 { 4.0 } else { 2.0 };
        }
        sum * step / 3.0
    };
    // The first round, `y = r (1 - cos t)` for `t` up to a quarter turn.
    let first = |t: f64| strip(X - r + r * t.sin()) * r * t.sin();
    // The second, `y = W - r + r cos t` from the end face's line, `y = r`.
    let second = |t: f64| strip(X - h + r * t.sin()) * r * t.sin();
    let reach = ((2.0 * r - W) / r).acos();
    simpson(&first, 0.0, std::f64::consts::FRAC_PI_2) + simpson(&second, 0.0, reach)
}

#[test]
fn a_slab_end_rounded_past_half_its_width_takes_the_second_round_pivoting() {
    let poses = [
        ("upright", Mat4::identity()),
        (
            "turned",
            Mat4::translation(30.0, -20.0, 10.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6),
        ),
    ];
    for (name, pose) in &poses {
        for r in [160.0, 200.0] {
            for first in [0.0, W] {
                let mut topo = Topology::new();
                let solid = slab(&mut topo);
                transform_solid(&mut topo, solid, pose).unwrap();
                let before = solid_volume(&topo, solid, 1e-3).unwrap();
                let picks = [
                    end_edge(&topo, solid, first, pose),
                    end_edge(&topo, solid, W - first, pose),
                ];
                let case = format!("{name}, r = {r}, first from y = {first}");
                let result = fillet_v2(&mut topo, solid, &picks, r).unwrap();
                assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
                let faces = solid_faces(&topo, result.solid).unwrap().len();
                assert_eq!(faces, 7, "{case}: the end face goes, the two rounds come");
                let report = validate_solid(&topo, result.solid).unwrap();
                assert!(report.is_valid(), "{case}: {:?}", report.issues);
                let mesh = tessellate_solid(&topo, result.solid, 0.1).unwrap();
                assert!(is_watertight(&mesh), "{case}: mesh open");
                let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
                let expected = removed(r);
                assert!(
                    (lost - expected).abs() <= 1e-6 * expected,
                    "{case}: removed {lost}, the pivoting round takes off {expected}"
                );
            }
        }
    }
}
