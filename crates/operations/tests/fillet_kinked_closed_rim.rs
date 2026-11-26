//! A slab whose outline is one closed B-spline that meets itself at a reflex
//! angle: an ellipse's B-spline with its first (and last) control point
//! pushed in towards the centre, which makes a dimple there. The wall is one
//! B-spline face ruled up from the outline, with a seam line at the dimple.
//!
//! The sections a round walked once round the rim starts and ends with are
//! square to the outline's two tangents at the dimple, so the band closed on
//! neither: the rim is now rounded as two halves, which pivot on the seam
//! line through a horn torus and share a section halfway round.
//!
//! Away from the kink the removed volume is the round's section,
//! `r^2 (1 - pi / 4)`, swept along the path of its centroid, `c` in from the
//! wall: the outline less `c` times its turning. The kink turns the outline
//! by its angle `k` at a point, which the smooth stretches make up; the
//! corner's own share differs from that by under `r^3 k`.
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
const HEIGHT: f64 = 4.0;

/// The outline in `z = 0`: the ellipse's B-spline, its end control point
/// moved `push` out from the centre (in, for a negative `push`).
fn outline(push: f64) -> NurbsCurve {
    let ellipse = Ellipse3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        SEMI_MAJOR,
        SEMI_MINOR,
    )
    .unwrap();
    let spline = ellipse_to_nurbs(&ellipse, 0.0, 2.0 * PI).unwrap();
    let mut points = spline.control_points().to_vec();
    let last = points.len() - 1;
    let start = points[0] - Point3::new(0.0, 0.0, 0.0);
    let out = start * (push / start.length());
    for index in [0, last] {
        points[index] = points[index] + out;
    }
    NurbsCurve::new(
        spline.degree(),
        spline.knots().to_vec(),
        points,
        spline.weights().to_vec(),
    )
    .unwrap()
}

/// The angle between the outline's tangents where it meets itself.
fn kink(spline: &NurbsCurve) -> f64 {
    let (u0, u1) = spline.domain();
    let start = spline.derivatives(u0, 1)[1];
    let end = spline.derivatives(u1, 1)[1];
    (start.dot(end) / (start.length() * end.length())).acos()
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

/// Rounds the rims at `zs` of the slab whose outline is pushed `push` at
/// radius `r`, and checks the result.
fn assert_kinked_rims_round(push: f64, zs: &[f64], r: f64) {
    let spline = outline(push);
    let k = kink(&spline);
    assert!(k > 0.02, "push {push}: the outline kinks by {k} only");
    let mut topo = Topology::new();
    let solid = slab(&mut topo, &spline);
    let edges: Vec<EdgeId> = zs.iter().map(|&z| rim_at(&topo, solid, z)).collect();
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
    assert!(
        result.failed.is_empty(),
        "push {push}, r {r}, rims {zs:?}: {:?}",
        result.failed
    );
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(
        report.is_valid(),
        "push {push}, r {r}, rims {zs:?}: {:?}",
        report.issues
    );
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(
        is_watertight(&mesh),
        "push {push}, r {r}: mesh not watertight"
    );
    // The floor, the lid, the wall cut in two along its ruling halfway
    // round, and for each rim a round in two halves and, at a dimple, the
    // horn torus they pivot through.
    let faces = solid_faces(&topo, result.solid).unwrap().len();
    let per_rim = if push > 0.0 { 2 } else { 3 };
    assert_eq!(faces, 4 + per_rim * zs.len(), "push {push}, r {r}");

    let c = r * 3.0f64.mul_add(-PI, 10.0) / (3.0 * (4.0 - PI));
    #[allow(clippy::cast_precision_loss)]
    let removed =
        zs.len() as f64 * r * r * (1.0 - PI / 4.0) * 2.0f64.mul_add(-PI * c, length(&spline));
    let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    #[allow(clippy::cast_precision_loss)]
    let allowance = zs.len() as f64 * r * r * r * k;
    assert!(
        (lost - removed).abs() <= allowance,
        "push {push}, r {r}: lost {lost}, the smooth round {removed}, allowed {allowance}"
    );
}

#[test]
fn a_tip_where_the_outline_meets_itself_mitres_the_round() {
    for r in [0.5, 1.0] {
        assert_kinked_rims_round(0.4, &[HEIGHT], r);
        assert_kinked_rims_round(0.4, &[0.0, HEIGHT], r);
    }
}

#[test]
fn a_dimple_where_the_outline_meets_itself_pivots_the_round() {
    for r in [0.5, 1.0] {
        assert_kinked_rims_round(-0.4, &[HEIGHT], r);
        assert_kinked_rims_round(-0.4, &[0.0, HEIGHT], r);
    }
}
