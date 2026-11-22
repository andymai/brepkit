//! A sphere or torus face that is not a box in its surface's two angles, on
//! a solid with a spline face that the exact boundary integral does not
//! read, so that the volume is summed face by face.
//!
//! - The unit ball less the cap below `z = -0.5`, its rim rounded: the round
//!   is a spline band, and the ball's lower face is a collar between the
//!   round and the ball's equator polygon.
//! - A torus less a box notch, with an oblique cone (a ruled spline face to
//!   a point) on a notch wall. Its band is seamed, or it has an outer and an
//!   inner wire round the tube, as a STEP file without the seam gives it.
//!
//! Each solid measures its closed form near the origin, far from it, and
//! turned.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::loft::loft_to_point;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_sphere, make_torus};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_circle_edge, make_planar_face_from_wire};
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;
use brepkit_topology::wire::{OrientedEdge, Wire};

fn poses() -> [(&'static str, Mat4); 4] {
    let turn = Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3);
    [
        ("upright", Mat4::identity()),
        ("moved", Mat4::translation(20.0, 0.0, 0.0)),
        ("turned", turn),
        (
            "turned and moved",
            Mat4::translation(20.0, -7.0, 5.0) * turn,
        ),
    ]
}

/// The integral of `f` over `[lo, hi]` by 20-point Gauss rules on `panels`
/// equal panels.
fn integral(panels: u32, lo: f64, hi: f64, f: &dyn Fn(f64) -> f64) -> f64 {
    let width = (hi - lo) / f64::from(panels);
    (0..panels)
        .map(|k| {
            let mid = (f64::from(k) + 0.5).mul_add(width, lo);
            gauss_legendre_points(20)
                .iter()
                .map(|g| g.w * 0.5 * width * f((0.5 * width).mul_add(g.x, mid)))
                .sum::<f64>()
        })
        .sum()
}

/// The unit ball less the cap below `z = -0.5`, its rim rounded at `r`.
fn rounded_cap(topo: &mut Topology, r: f64) -> SolidId {
    let ball = make_sphere(topo, 1.0, 32).unwrap();
    let below = make_box(topo, 4.0, 4.0, 4.0).unwrap();
    transform_solid(topo, below, &Mat4::translation(-2.0, -2.0, -4.5)).unwrap();
    let cut = boolean(topo, BooleanOp::Cut, ball, below).unwrap();
    let rim: Vec<_> = solid_edges(topo, cut)
        .unwrap()
        .into_iter()
        .filter(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Circle(_)))
        .collect();
    assert_eq!(rim.len(), 1, "the cut leaves one rim circle");
    fillet_v2(topo, cut, &rim, r).unwrap().solid
}

/// The rounded cap's volume: the ball less its cap of height 0.5, less the
/// ring the round takes off, by Pappus. Over the meridian section (`rho`
/// out, `z` up) the ring's section lies between the plane, the ball and
/// the round's arc about `(rho_c, z_c)`; its integral of `rho` over its area
/// is the integral of `rho^2 / 2` along its boundary, against `dz`.
fn rounded_cap_volume(r: f64) -> f64 {
    let capped = 4.0 / 3.0 * PI - PI * 0.25 * (3.0 - 0.5) / 3.0;
    let z_c = -0.5 + r;
    let rho_c = (1.0 - r).mul_add(1.0 - r, -z_c * z_c).sqrt();
    let touch = z_c.atan2(rho_c);
    let rim = (-0.5_f64).atan2(0.75_f64.sqrt());
    let along_ball = integral(4, rim, touch, &|t: f64| 0.5 * t.cos().powi(3));
    let along_round = -integral(4, -FRAC_PI_2, touch, &|p: f64| {
        let rho = r.mul_add(p.cos(), rho_c);
        0.5 * rho * rho * r * p.cos()
    });
    capped - 2.0 * PI * (along_ball + along_round)
}

#[test]
fn a_ball_cap_with_its_rim_rounded_measures_its_closed_form() {
    for r in [0.05, 0.1, 0.2] {
        let truth = rounded_cap_volume(r);
        for (pose, m) in poses() {
            let mut topo = Topology::new();
            let solid = rounded_cap(&mut topo, r);
            transform_solid(&mut topo, solid, &m).unwrap();
            let volume = solid_volume(&topo, solid, 0.002).unwrap();
            assert!(
                (volume - truth).abs() < 1e-8 * truth,
                "round {r}, {pose}: volume {volume}, truth {truth}"
            );
        }
    }
}

/// The torus of radii 10 and 3 less the box `[6, 14] x [-4, 4] x [-4, 4]`.
fn notched_torus(topo: &mut Topology) -> SolidId {
    let torus = make_torus(topo, 10.0, 3.0, 32).unwrap();
    let notch = make_box(topo, 8.0, 8.0, 8.0).unwrap();
    transform_solid(topo, notch, &Mat4::translation(6.0, -4.0, -4.0)).unwrap();
    boolean(topo, BooleanOp::Cut, torus, notch).unwrap()
}

/// The notched torus's volume: the torus less its part in the box. Over
/// the tube's section (`rho = 10 + 3 sin t`), each circle about the axis
/// keeps in the box the angle `2 min(acos(6 / rho), asin(4 / rho))`, which
/// changes its form at `rho = sqrt(52)`.
fn notched_torus_volume() -> f64 {
    let kept = |t: f64| {
        let rho = 3.0f64.mul_add(t.sin(), 10.0);
        let angle = (6.0 / rho).acos().min((4.0 / rho).asin());
        36.0 * rho * t.cos() * t.cos() * angle
    };
    let corner = ((52.0_f64.sqrt() - 10.0) / 3.0).asin();
    2.0 * PI * PI * 10.0 * 9.0
        - integral(50, -FRAC_PI_2, corner, &kept)
        - integral(50, corner, FRAC_PI_2, &kept)
}

/// The cone from the disc of radius 0.8 about `(9.5, 4, 0)` on the notch's
/// `y = 4` wall to the point `(10, 2, 0.3)`, off its axis: its side is a
/// ruled spline to a point. Its volume is its base times its height over 3.
fn oblique_cone(topo: &mut Topology) -> (SolidId, f64) {
    let rim = make_circle_edge(
        topo,
        Point3::new(9.5, 4.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        0.8,
        1e-7,
    )
    .unwrap();
    let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
    let base = make_planar_face_from_wire(topo, wire).unwrap();
    let cone = loft_to_point(topo, base, Point3::new(10.0, 2.0, 0.3)).unwrap();
    (cone, PI * 0.64 * 2.0 / 3.0)
}

/// Rebuilds the solid's torus band, one loop running up the tube's seam and
/// back, as two loops round the tube without the seam: the outer wire and
/// one inner wire.
fn unseam_the_band(topo: &mut Topology, solid: SolidId) {
    let band = solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Torus(_)))
        .unwrap();
    let edges = topo
        .wire(topo.face(band).unwrap().outer_wire())
        .unwrap()
        .edges()
        .to_vec();
    let on_seam = |oe: &OrientedEdge| edges.iter().filter(|o| o.edge() == oe.edge()).count() == 2;
    let n = edges.len();
    let start = (0..n)
        .find(|&i| on_seam(&edges[i]) && !on_seam(&edges[(i + 1) % n]))
        .unwrap()
        + 1;
    let mut loops: Vec<Vec<OrientedEdge>> = vec![Vec::new()];
    for k in 0..n {
        let oe = edges[(start + k) % n];
        if !on_seam(&oe) {
            loops.last_mut().unwrap().push(oe);
        } else if !loops.last().unwrap().is_empty() {
            loops.push(Vec::new());
        }
    }
    loops.retain(|l| !l.is_empty());
    assert_eq!(loops.len(), 2, "the band runs between two loops");
    let outer = topo.add_wire(Wire::new(loops[0].clone(), true).unwrap());
    let inner = topo.add_wire(Wire::new(loops[1].clone(), true).unwrap());
    let face = topo.face_mut(band).unwrap();
    face.set_outer_wire(outer);
    face.inner_wires_mut().push(inner);
}

#[test]
fn a_notched_torus_with_a_cone_on_its_wall_measures_its_closed_form() {
    for unseamed in [false, true] {
        for (pose, m) in poses() {
            let label = format!("unseamed {unseamed}, {pose}");
            let mut topo = Topology::new();
            let notched = notched_torus(&mut topo);
            let (cone, cone_volume) = oblique_cone(&mut topo);
            let solid = boolean(&mut topo, BooleanOp::Fuse, notched, cone).unwrap();
            if unseamed {
                unseam_the_band(&mut topo, solid);
            }
            transform_solid(&mut topo, solid, &m).unwrap();
            let truth = notched_torus_volume() + cone_volume;
            let volume = solid_volume(&topo, solid, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-7 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
        }
    }
}
