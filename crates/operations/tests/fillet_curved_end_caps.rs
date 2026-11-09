//! A round of a straight edge whose end meets a curved end face.
//!
//! The part is a D: the profile `x = 0`, `y = -B`, `y = B` closed by an arc
//! of a circle of radius `R` about `(C, 0)` bulging out to `x > X0`,
//! extruded `H` up `z`. The edges along `x` where the top or bottom meets
//! a side `y = B` end on the arc's cylinder, which is neither square to
//! them nor flat. The round runs on to that face: each contact to where its
//! line meets the cylinder, on the cylinder's corner edges (the top or
//! bottom arc and the vertical line), and the band ends on its intersection
//! with the cylinder, which the cylinder is notched along.
//!
//! The bottom face walks its arc against the circle's sense. The trim
//! splits that arc where the round's contact ends, and the piece kept runs
//! the short way round from the split, so the bottom keeps its own area.
//!
//! The round takes off, along the edge, the corner's strip outside the
//! round, out to the cylinder: at a distance `t` in from the side face the
//! strip is `r - sqrt(r^2 - t^2)` deep and runs out to `C + sqrt(R^2 -
//! (B - r + t)^2)`, which integrates (with `t = r sin u`) to the removed
//! volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_2;

use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
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

const B: f64 = 5.0;
const R: f64 = 8.0;
const X0: f64 = 10.0;
const H: f64 = 6.0;

/// A D's half width `b` and its arc's radius.
#[derive(Clone, Copy, Debug)]
struct Profile {
    b: f64,
    arc: f64,
}

/// The D of most tests.
const D: Profile = Profile { b: B, arc: R };

/// A D whose arc meets its sides almost tangentially: the round's end on
/// the cylinder bends hard near the side face.
const NEAR_TANGENT: Profile = Profile { b: 5.0, arc: 5.01 };

impl Profile {
    /// The arc's centre on the x axis.
    fn centre(self) -> f64 {
        X0 - self.arc.mul_add(self.arc, -self.b * self.b).sqrt()
    }

    /// The D, its arc counter-clockwise about `+z` from `(X0, -b)` to
    /// `(X0, b)`.
    fn prism(self, topo: &mut Topology) -> SolidId {
        let b = self.b;
        let corners = [(X0, -b), (X0, b), (0.0, b), (0.0, -b)];
        let ids: Vec<_> = corners
            .iter()
            .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
            .collect();
        let arc = Circle3D::new(
            Point3::new(self.centre(), 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            self.arc,
        )
        .unwrap();
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

    /// The straight edge of `solid` along `x` at `y = b` and height `z`.
    fn side_edge(self, topo: &Topology, solid: SolidId, z: f64) -> EdgeId {
        solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&edge_id| {
                let edge = topo.edge(edge_id).unwrap();
                let [p, q] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
                matches!(edge.curve(), EdgeCurve::Line)
                    && [p, q]
                        .iter()
                        .all(|p| (p.y() - self.b).abs() < 1e-9 && (p.z() - z).abs() < 1e-9)
                    && (p.x() - q.x()).abs() > 1.0
            })
            .expect("the edge")
    }

    /// The volume the round of radius `r` takes off.
    fn removed(self, r: f64) -> f64 {
        // Simpson's rule in `u`, where the integrand is smooth.
        const N: usize = 2000;
        let c = self.centre();
        let f = |u: f64| {
            let t = r * u.sin();
            let depth = r * (1.0 - u.cos());
            let y = self.b - r + t;
            depth * (c + self.arc.mul_add(self.arc, -y * y).sqrt()) * r * u.cos()
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
}

/// The farthest any edge of `solid` lies off a face it bounds, from 64
/// samples along each.
fn edges_off_faces(topo: &Topology, solid: SolidId) -> f64 {
    let mut farthest = 0.0_f64;
    for face_id in solid_faces(topo, solid).unwrap() {
        let face = topo.face(face_id).unwrap();
        let surface = face.surface();
        for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
        {
            for oriented in topo.wire(wire_id).unwrap().edges() {
                let edge = topo.edge(oriented.edge()).unwrap();
                let [a, b] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
                let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                for k in 0..=64 {
                    let t = (t1 - t0).mul_add(f64::from(k) / 64.0, t0);
                    let p = edge.curve().evaluate_with_endpoints(t, a, b);
                    let off = if let FaceSurface::Plane { normal, d } = surface {
                        (normal.dot(Vec3::new(p.x(), p.y(), p.z())) - d).abs()
                    } else {
                        let (u, v) = surface.project_point(p).unwrap();
                        (surface.evaluate(u, v).unwrap() - p).length()
                    };
                    farthest = farthest.max(off);
                }
            }
        }
    }
    farthest
}

/// The D turned about two axes and moved off the origin.
fn turned() -> Mat4 {
    Mat4::translation(37.0, 483.0, -11.0) * Mat4::rotation_z(0.4229) * Mat4::rotation_x(1.0)
}

/// Rounds the edge of the D `profile` at height `z` with radius `r`, the D
/// moved by `pose`, and checks the result: valid, watertight, the round its
/// only new face, every edge on the faces it bounds, and the strip out to
/// the cylinder taken off.
fn assert_round_ends_on_the_curve(profile: Profile, z: f64, r: f64, pose: Option<Mat4>) {
    let mut topo = Topology::new();
    let solid = profile.prism(&mut topo);
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let faces = solid_faces(&topo, solid).unwrap().len();
    let edge = profile.side_edge(&topo, solid, z);
    if let Some(pose) = pose {
        transform_solid(&mut topo, solid, &pose).unwrap();
    }
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
    let off = edges_off_faces(&topo, result.solid);
    assert!(off <= 1e-6, "z {z} r {r}: an edge lies {off} off a face");
    let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    let expected = profile.removed(r);
    assert!(
        (lost - expected).abs() <= 1e-6 * before,
        "z {z} r {r}: removed {lost}, the strip out to the cylinder {expected}"
    );
}

#[test]
fn top_edge_ending_on_the_cylinder() {
    for r in [1.0, 2.0] {
        assert_round_ends_on_the_curve(D, H, r, None);
    }
}

#[test]
fn bottom_edge_ending_on_the_cylinder() {
    for r in [1.0, 2.0] {
        assert_round_ends_on_the_curve(D, 0.0, r, None);
    }
}

#[test]
fn edges_of_a_turned_d_ending_on_the_cylinder() {
    for z in [0.0, H] {
        assert_round_ends_on_the_curve(D, z, 2.0, Some(turned()));
    }
}

#[test]
fn edges_ending_on_a_nearly_tangent_cylinder() {
    for z in [0.0, H] {
        assert_round_ends_on_the_curve(NEAR_TANGENT, z, 2.0, None);
    }
}

/// The D with a blind radial hole in its arc wall, `HOLE` in radius and 1
/// deep, its mouth centred where the arc is at `y`, at height `z`.
fn d_prism_with_hole(topo: &mut Topology, y: f64, z: f64) -> SolidId {
    const HOLE: f64 = 0.15;
    let solid = D.prism(topo);
    let turn = (y / R).asin();
    let drill = make_cylinder(topo, HOLE, 1.5).unwrap();
    let place = Mat4::translation(
        (R - 1.0).mul_add(turn.cos(), D.centre()),
        (R - 1.0) * turn.sin(),
        z,
    ) * Mat4::rotation_z(turn)
        * Mat4::rotation_y(FRAC_PI_2);
    transform_solid(topo, drill, &place).unwrap();
    boolean(topo, BooleanOp::Cut, solid, drill).unwrap()
}

#[test]
fn a_round_whose_notch_would_cross_a_hole_in_its_curved_cap_is_refused() {
    // The hole's mouth lies in the corner the round of the top edge takes
    // off the arc wall: the notch would cut through the hole's wall.
    let mut topo = Topology::new();
    let solid = d_prism_with_hole(&mut topo, 4.5, 5.6);
    let edge = D.side_edge(&topo, solid, H);
    let refused = match fillet_v2(&mut topo, solid, &[edge], 2.0) {
        Err(_) => true,
        Ok(result) => !result.failed.is_empty(),
    };
    assert!(refused, "the notch crosses the hole's mouth");
}

#[test]
fn a_hole_clear_of_the_notch_keeps_the_round() {
    let r = 2.0;
    let mut topo = Topology::new();
    let solid = d_prism_with_hole(&mut topo, 0.0, 3.0);
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let edge = D.side_edge(&topo, solid, H);
    let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "mesh open");
    let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    let expected = D.removed(r);
    assert!(
        (lost - expected).abs() <= 1e-6 * before,
        "removed {lost}, the strip out to the cylinder {expected}"
    );
}
