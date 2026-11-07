//! Fillets of both faces' outlines of a slab whose outline is a tombstone:
//! a semicircle of radius `R` on a rectangle `2R` wide and `D` deep. Each
//! outline has two square corners, where the two straight fillets meet in a
//! mitre over the slab's sharp side edge, and two points where a straight
//! side runs on round the arc, where the straight and the round fillets
//! meet on one section. Both outlines are rounded at once, so every side
//! face carries two contacts and every side edge has a mitre or a tangent
//! junction at each end.
//!
//! As in `fillet_floor_outlines.rs`, a fillet of radius `r` round a convex
//! outline of perimeter `P` removes `P r^2 (1 - pi / 4) - K r^3 (5 / 3 -
//! pi / 2)`, where `K` sums `tan(t / 2)` over its corners turning by `t` and
//! `phi / 2` over its arcs turning by `phi`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

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

const R: f64 = 18.5;
const D: f64 = 23.5;
const T: f64 = 37.0;

/// A pose of the outline in its plane: turned by `turn` about the origin,
/// moved by `shift`, every point rounded to `digits` decimals as a file
/// stores it (so its square corners are square only to about 1e-9).
#[derive(Clone, Copy)]
struct Pose {
    turn: f64,
    shift: (f64, f64),
    digits: Option<i32>,
}

impl Pose {
    fn place(self, (x, y): (f64, f64)) -> Point3 {
        let (s, c) = self.turn.sin_cos();
        let mut p = [
            c.mul_add(x, -s * y) + self.shift.0,
            s.mul_add(x, c * y) + self.shift.1,
        ];
        if let Some(digits) = self.digits {
            let scale = 10f64.powi(digits);
            for v in &mut p {
                *v = (*v * scale).round() / scale;
            }
        }
        Point3::new(p[0], p[1], 0.0)
    }
}

/// The slab: the outline in `z = 0`, its arc counter-clockwise about `+z`
/// from `(R, 0)` over the top to `(-R, 0)`, extruded `T` up.
fn slab(topo: &mut Topology, pose: Pose) -> SolidId {
    let corners = [(R, 0.0), (-R, 0.0), (-R, -D), (R, -D)];
    let ids: Vec<_> = corners
        .iter()
        .map(|&corner| topo.add_vertex(Vertex::new(pose.place(corner), 1e-7)))
        .collect();
    let center = pose.place((0.0, 0.0));
    let arc = Circle3D::new(center, Vec3::new(0.0, 0.0, 1.0), R).unwrap();
    let mut edges = Vec::new();
    for (i, &a) in ids.iter().enumerate() {
        let b = ids[(i + 1) % ids.len()];
        let curve = if i == 0 {
            EdgeCurve::Circle(arc.clone())
        } else {
            EdgeCurve::Line
        };
        edges.push(OrientedEdge::new(
            topo.add_edge(Edge::new(a, b, curve)),
            true,
        ));
    }
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), T).unwrap()
}

/// The edges of `solid` lying wholly in `z = 0` or `z = T`.
fn outline_edges(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            let z = |v| topo.vertex(v).unwrap().point().z();
            let (a, b) = (z(edge.start()), z(edge.end()));
            (a - b).abs() < 1e-9 && (a.abs() < 1e-9 || (a - T).abs() < 1e-9)
        })
        .collect()
}

/// Rounds both outlines of the slab in `pose` with radius `r` and checks the
/// result: one face per fillet, side and cap (no corner patches), valid,
/// watertight, and short of the slab by the two strips, to 1e-5.
fn assert_outlines_round_off(pose: Pose, r: f64) {
    let mut topo = Topology::new();
    let solid = slab(&mut topo, pose);
    let edges = outline_edges(&topo, solid);
    assert_eq!(edges.len(), 8, "the slab's two outlines");
    let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let faces = solid_faces(&topo, result.solid).unwrap();
    let kinds: Vec<&str> = faces
        .iter()
        .map(|&f| topo.face(f).unwrap().surface().type_tag())
        .collect();
    assert_eq!(
        faces.len(),
        14,
        "r = {r}: 5 planes, the wall, 6 straight and 2 round fillets: {kinds:?}"
    );
    assert!(
        !kinds.contains(&"nurbs"),
        "r = {r}: a corner patch: {kinds:?}"
    );
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
    let perimeter = 2.0f64.mul_add(D, PI.mul_add(R, 2.0 * R));
    let corners = 2.0 + PI / 2.0;
    let strip = (perimeter * r * r).mul_add(
        1.0 - PI / 4.0,
        -corners * r * r * r * (5.0 / 3.0 - PI / 2.0),
    );
    let truth = 2.0f64.mul_add(-strip, (2.0 * R).mul_add(D, PI * R * R / 2.0) * T);
    let after = solid_volume(&topo, result.solid, 0.001).unwrap();
    assert!(
        (after - truth).abs() < 1e-5 * truth,
        "r = {r}: volume {after}, expected {truth}"
    );
}

#[test]
fn a_tombstone_slabs_outlines_round_off() {
    for r in [1.0, 3.0] {
        assert_outlines_round_off(
            Pose {
                turn: 0.0,
                shift: (0.0, 0.0),
                digits: None,
            },
            r,
        );
    }
}

#[test]
fn a_tombstone_slab_stored_to_eight_digits_rounds_off() {
    for turn in [0.4229, 1.0, 2.5, 4.0] {
        assert_outlines_round_off(
            Pose {
                turn,
                shift: (37.0, 483.0),
                digits: Some(8),
            },
            1.0,
        );
    }
}
