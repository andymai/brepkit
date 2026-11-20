//! A round of an arc edge whose ends meet tilted end faces.
//!
//! The part is a D: an arc of radius `R` about the `z` axis, from angle
//! `-alpha` to `alpha`, closed by its chord at `x = c = R cos(alpha)`, and
//! extruded `H` up `z`. Its top arc edge, where the top meets the arc's
//! cylinder, is rounded: a torus about the axis, its sections the tube's
//! circles in the planes through the axis. At each end the round meets the
//! chord's wall, tilted to the end section by `sin(alpha)`.
//!
//! The round's end stood in the section plane and was closed by a runout
//! patch. Each contact, a circle about the axis, now runs on (or back) to
//! the chord's plane, on the wall's corner edges, and the band ends on its
//! section by that plane, which the wall is notched along.
//!
//! The volume taken off is the round's profile (the corner square of side
//! `r` less its quarter disc, in `(rho, z)`) swept about the axis between
//! the two chord planes: at radius `rho` the material spans
//! `2 acos(c / rho)` of a turn, so the volume is the profile's integral of
//! `2 rho acos(c / rho)`.
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

const R: f64 = 10.0;
const H: f64 = 6.0;

/// The D for `alpha`, its arc counter-clockwise about `+z`.
fn d_prism(topo: &mut Topology, alpha: f64) -> SolidId {
    let corners = [
        Point3::new(R * alpha.cos(), -R * alpha.sin(), 0.0),
        Point3::new(R * alpha.cos(), R * alpha.sin(), 0.0),
    ];
    let ids: Vec<_> = corners
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let arc = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), R).unwrap();
    let edges = [
        Edge::new(ids[0], ids[1], EdgeCurve::Circle(arc)),
        Edge::new(ids[1], ids[0], EdgeCurve::Line),
    ]
    .map(|edge| OrientedEdge::new(topo.add_edge(edge), true));
    let wire = topo.add_wire(Wire::new(edges.to_vec(), true).unwrap());
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: 0.0 },
    ));
    extrude(topo, face, normal, H).unwrap()
}

/// The prism's top arc edge.
fn top_arc(topo: &Topology, solid: SolidId) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(_))
                && (topo.vertex(edge.start()).unwrap().point().z() - H).abs() < 1e-9
        })
        .expect("the top arc")
}

/// The volume the round takes off: the profile's integral of
/// `2 rho acos(c / rho)`, over `rho = R - r + r sin(theta)`, where the
/// profile's height at `rho` is `r (1 - cos(theta))`.
fn removed(alpha: f64, r: f64) -> f64 {
    const STEPS: usize = 4000;
    let c = R * alpha.cos();
    let f = |theta: f64| {
        let rho = r.mul_add(theta.sin(), R - r);
        let height = r * (1.0 - theta.cos());
        2.0 * rho * (c / rho).acos() * height * r * theta.cos()
    };
    #[allow(clippy::cast_precision_loss)]
    let h = FRAC_PI_2 / STEPS as f64;
    let mut sum = f(0.0) + f(FRAC_PI_2);
    for k in 1..STEPS {
        #[allow(clippy::cast_precision_loss)]
        let theta = h * k as f64;
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(theta);
    }
    sum * h / 3.0
}

#[test]
fn a_torus_round_runs_on_to_tilted_end_faces() {
    for alpha_deg in [45.0_f64, 60.0] {
        let alpha = alpha_deg.to_radians();
        for r in [1.0, 2.0] {
            let case = format!("alpha = {alpha_deg}, r = {r}");
            let mut topo = Topology::new();
            let solid = d_prism(&mut topo, alpha);
            let edge = top_arc(&topo, solid);
            let before = solid_volume(&topo, solid, 1e-3).unwrap();
            let faces_before = solid_faces(&topo, solid).unwrap().len();

            let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
            assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
            let report = validate_solid(&topo, result.solid).unwrap();
            assert!(report.is_valid(), "{case}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{case}: mesh not watertight");
            let faces = solid_faces(&topo, result.solid).unwrap();
            assert_eq!(
                faces.len(),
                faces_before + 1,
                "{case}: the round alone is added, no runout patch"
            );
            assert!(
                faces.iter().any(|&face| matches!(
                    topo.face(face).unwrap().surface(),
                    FaceSurface::Torus(_)
                )),
                "{case}: the round is a torus"
            );

            let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
            let want = removed(alpha, r);
            assert!(
                (lost - want).abs() <= 1e-5 * want.max(1.0),
                "{case}: lost {lost}, the round's sweep {want}"
            );
        }
    }
}
