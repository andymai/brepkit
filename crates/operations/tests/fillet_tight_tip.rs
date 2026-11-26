//! A slab whose outline is an ellipse, its rims rounded at a radius past
//! the ellipse's tightest bend, `b^2 / a` at the ends of its major axis.
//!
//! No ball rolls round those ends: the round's contact on the cap, the
//! outline inset by the radius, runs backwards there and crosses itself.
//! The round carries on round each end and is cut off where it folds back
//! over itself, along a crease from the crossing on the cap down to where
//! the fold ends.
//!
//! The removed volume is the round's section, `r^2 (1 - pi / 4)`, swept
//! along the path of its centroid, `c` in from the wall: the outline less
//! `c` times its turning. That holds past the folds too, to within
//! `1e-4 r^3` a rim.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_geometry::convert::ellipse_to_nurbs;
use brepkit_math::curves::Ellipse3D;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const SEMI_MAJOR: f64 = 8.0;
const SEMI_MINOR: f64 = 5.0;
const HEIGHT: f64 = 8.0;

/// The outline in `z = 0`, starting `start` round from the major axis.
fn outline(start: f64) -> NurbsCurve {
    let ellipse = Ellipse3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        SEMI_MAJOR,
        SEMI_MINOR,
    )
    .unwrap();
    ellipse_to_nurbs(&ellipse, start, start + 2.0 * PI).unwrap()
}

/// The outline's length, summed from fine chords.
fn length(spline: &NurbsCurve) -> f64 {
    let (u0, u1) = spline.domain();
    let n: u32 = 200_000;
    let at = |k: u32| spline.evaluate((u1 - u0).mul_add(f64::from(k) / f64::from(n), u0));
    (0..n).map(|k| (at(k + 1) - at(k)).length()).sum()
}

/// The slab: the outline's disc `HEIGHT` thick, its wall one B-spline face
/// ruled between the outline and its copy `HEIGHT` up.
fn slab(topo: &mut Topology, spline: &NurbsCurve) -> SolidId {
    let up = Vec3::new(0.0, 0.0, HEIGHT);
    let lifted = NurbsCurve::new(
        spline.degree(),
        spline.knots().to_vec(),
        spline.control_points().iter().map(|&p| p + up).collect(),
        spline.weights().to_vec(),
    )
    .unwrap();
    let wall_surface = NurbsSurface::new(
        spline.degree(),
        1,
        spline.knots().to_vec(),
        vec![0.0, 0.0, 1.0, 1.0],
        spline
            .control_points()
            .iter()
            .map(|&p| vec![p, p + up])
            .collect(),
        spline.weights().iter().map(|&w| vec![w, w]).collect(),
    )
    .unwrap();
    let start = spline.evaluate(spline.domain().0);
    let vb = topo.add_vertex(Vertex::new(start, 1e-7));
    let vt = topo.add_vertex(Vertex::new(start + up, 1e-7));
    let bottom = topo.add_edge(Edge::new(vb, vb, EdgeCurve::NurbsCurve(spline.clone())));
    let top = topo.add_edge(Edge::new(vt, vt, EdgeCurve::NurbsCurve(lifted)));
    let seam = topo.add_edge(Edge::new(vb, vt, EdgeCurve::Line));
    let wire = |topo: &mut Topology, edges: &[(EdgeId, bool)]| {
        let edges = edges
            .iter()
            .map(|&(e, forward)| OrientedEdge::new(e, forward))
            .collect();
        topo.add_wire(Wire::new(edges, true).unwrap())
    };
    let wall_wire = wire(
        topo,
        &[(bottom, true), (seam, true), (top, false), (seam, false)],
    );
    let wall = topo.add_face(Face::new(
        wall_wire,
        vec![],
        FaceSurface::Nurbs(wall_surface),
    ));
    let floor_wire = wire(topo, &[(bottom, false)]);
    let floor = topo.add_face(Face::new(
        floor_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ));
    let lid_wire = wire(topo, &[(top, true)]);
    let lid = topo.add_face(Face::new(
        lid_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: HEIGHT,
        },
    ));
    let shell = topo.add_shell(Shell::new(vec![wall, floor, lid]).unwrap());
    topo.add_solid(Solid::new(shell, vec![]))
}

/// The slab's closed rim at height `z`.
fn rim_at(topo: &Topology, solid: SolidId, z: f64) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            edge.start() == edge.end()
                && (topo.vertex(edge.start()).unwrap().point().z() - z).abs() < 1e-9
        })
        .expect("a closed rim at that height")
}

/// Rounds the slab's rims at `zs` at radius `r`, and checks the result.
fn assert_tight_tips_round(zs: &[f64], r: f64) {
    // The wall's seam at the end of the minor axis, on the round.
    let spline = outline(PI / 2.0);
    assert!(r > SEMI_MINOR * SEMI_MINOR / SEMI_MAJOR);
    let mut topo = Topology::new();
    let solid = slab(&mut topo, &spline);
    let edges: Vec<EdgeId> = zs.iter().map(|&z| rim_at(&topo, solid, z)).collect();
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
    assert!(
        result.failed.is_empty(),
        "r {r}, rims {zs:?}: {:?}",
        result.failed
    );
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "r {r}, rims {zs:?}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(
        is_watertight(&mesh),
        "r {r}, rims {zs:?}: mesh not watertight"
    );
    // The floor, the lid and the wall, and for each rim the round in two
    // between the ends and two faces round each end, meeting at its crease.
    let faces = solid_faces(&topo, result.solid).unwrap().len();
    assert_eq!(faces, 3 + 6 * zs.len(), "r {r}, rims {zs:?}");

    let c = r * 3.0f64.mul_add(-PI, 10.0) / (3.0 * (4.0 - PI));
    #[allow(clippy::cast_precision_loss)]
    let removed =
        zs.len() as f64 * r * r * (1.0 - PI / 4.0) * 2.0f64.mul_add(-PI * c, length(&spline));
    let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    #[allow(clippy::cast_precision_loss)]
    // The bands' volume is read from their mesh, to about 1e-3 r^3 a rim.
    let allowance = zs.len() as f64 * r * r * r * 1e-3;
    assert!(
        (lost - removed).abs() <= allowance,
        "r {r}, rims {zs:?}: lost {lost}, the smooth round {removed}, allowed {allowance}"
    );
}

#[test]
fn a_round_past_the_outline_s_tightest_bend_folds_off_at_the_tips() {
    for r in [3.25, 3.5] {
        assert_tight_tips_round(&[HEIGHT], r);
        assert_tight_tips_round(&[0.0, HEIGHT], r);
    }
}

#[test]
fn a_tip_across_the_wall_s_seam_is_refused() {
    // The wall's seam at the end of the major axis, on a tip.
    let spline = outline(0.0);
    let mut topo = Topology::new();
    let solid = slab(&mut topo, &spline);
    let rim = rim_at(&topo, solid, HEIGHT);
    match fillet_v2(&mut topo, solid, &[rim], 3.5) {
        Err(_) => {}
        Ok(result) => assert!(!result.failed.is_empty(), "rounded a tip across the seam"),
    }
}
