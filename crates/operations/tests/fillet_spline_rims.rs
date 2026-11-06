//! Fillets of a closed B-spline rim where a flat cap meets a B-spline wall:
//! a slab with an elliptic outline whose wall is one B-spline surface closed
//! round the outline, as files write a plate cut from a spline outline. The walker rolls the ball all the way
//! round the rim and across the wall's seam.
//!
//! A fillet of radius `r` along a convex outline of length `L` removes a
//! strip whose section is the square of side `r` at the rim less a quarter
//! disc; its centroid lies `c` in from both faces, so it sweeps a path of
//! length `L - 2πc` (the outline offset inward by `c`), wherever the
//! outline's radius of curvature exceeds `c`.
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
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const SEMI_MAJOR: f64 = 8.0;
const SEMI_MINOR: f64 = 5.0;
const HEIGHT: f64 = 4.0;

/// The outline: an ellipse in `z = 0` as one closed B-spline edge.
fn outline() -> Ellipse3D {
    Ellipse3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        SEMI_MAJOR,
        SEMI_MINOR,
    )
    .unwrap()
}

/// The slab: the outline's disc `HEIGHT` thick, its wall one B-spline face
/// ruled between the outline and its copy `HEIGHT` up, closed round the
/// outline, with a seam line where the outline starts.
fn slab(topo: &mut Topology) -> SolidId {
    let spline = ellipse_to_nurbs(&outline(), 0.0, 2.0 * PI).unwrap();
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
    let bottom = topo.add_edge(Edge::new(vb, vb, EdgeCurve::NurbsCurve(spline)));
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

/// The closed B-spline edge of `solid` at height `z`.
fn rim_at(topo: &Topology, solid: SolidId, z: f64) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::NurbsCurve(_))
                && edge.start() == edge.end()
                && (topo.vertex(edge.start()).unwrap().point().z() - z).abs() < 1e-9
        })
        .expect("a closed B-spline rim at that height")
}

/// The outline's length, summed from fine chords.
fn perimeter() -> f64 {
    let ellipse = outline();
    let n: u32 = 200_000;
    let t = |k: u32| 2.0 * PI * f64::from(k) / f64::from(n);
    (0..n)
        .map(|k| (ellipse.evaluate(t(k + 1)) - ellipse.evaluate(t(k))).length())
        .sum()
}

/// The section's centroid lies this far from the rim's corner, along each
/// of the two faces.
fn centroid_from_corner(r: f64) -> f64 {
    r * 3.0f64.mul_add(-PI, 10.0) / (3.0 * (4.0 - PI))
}

/// Fillets the rim at height `z` with radius `r` and checks the result is
/// valid, watertight, rounded off by its walls, and short by the strip the
/// fillet removes, to 0.1%.
fn assert_rim_rounds_off(z: f64, r: f64) {
    assert_rims_round_off(&[z], r);
}

/// As [`assert_rim_rounds_off`], filleting the rims at all of `zs` at once.
fn assert_rims_round_off(zs: &[f64], r: f64) {
    let mut topo = Topology::new();
    let solid = slab(&mut topo);
    assert!(
        crate_faces(&topo, solid)
            .iter()
            .any(|s| matches!(s, FaceSurface::Nurbs(n) if n.is_periodic_u() || n.is_periodic_v())),
        "the wall is a B-spline closed round the outline: {:?}",
        crate_faces(&topo, solid)
            .iter()
            .map(|s| match s {
                FaceSurface::Nurbs(n) => format!(
                    "nurbs {}x{} periodic u {} v {}",
                    n.control_points().len(),
                    n.control_points()[0].len(),
                    n.is_periodic_u(),
                    n.is_periodic_v()
                ),
                other => other.type_tag().to_owned(),
            })
            .collect::<Vec<_>>()
    );
    let input_mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
    assert!(
        is_watertight(&input_mesh),
        "the slab's own mesh is not watertight"
    );
    let edges: Vec<EdgeId> = zs.iter().map(|&z| rim_at(&topo, solid, z)).collect();
    let before = solid_volume(&topo, solid, 0.001).unwrap();
    let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    assert_ne!(result.solid, solid, "r = {r}: the fillet changed nothing");
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(
        is_watertight(&mesh),
        "r = {r}: mesh not watertight ({} open, {} over-used)",
        brepkit_operations::tessellate::boundary_edge_count(&mesh),
        brepkit_operations::tessellate::non_manifold_edge_count(&mesh)
    );
    #[allow(clippy::cast_precision_loss)]
    let removed = zs.len() as f64
        * (1.0 - PI / 4.0)
        * r
        * r
        * 2.0f64.mul_add(-PI * centroid_from_corner(r), perimeter());
    let after = solid_volume(&topo, result.solid, 0.001).unwrap();
    let truth = before - removed;
    assert!(
        (after - truth).abs() < 1e-3 * truth,
        "r = {r}, z = {zs:?}: volume {after}, expected {truth} (removed {removed})"
    );
}

fn crate_faces(topo: &Topology, solid: SolidId) -> Vec<FaceSurface> {
    brepkit_topology::explorer::solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .map(|f| topo.face(f).unwrap().surface().clone())
        .collect()
}

#[test]
fn an_elliptic_slabs_top_rim_rounds_off() {
    for r in [0.5, 1.0, 2.0] {
        assert_rim_rounds_off(HEIGHT, r);
    }
}

#[test]
fn an_elliptic_slabs_bottom_rim_rounds_off() {
    for r in [0.5, 1.0, 2.0] {
        assert_rim_rounds_off(0.0, r);
    }
}

#[test]
fn an_elliptic_slabs_two_rims_round_off_together() {
    for r in [0.5, 1.0] {
        assert_rims_round_off(&[0.0, HEIGHT], r);
    }
}

/// Rounds on both rims whose radii add up to more than the wall's height
/// would cross on the wall: refused, not built as faces that overlap. At
/// exactly half the height (r = 2) the two contacts meet along one line and
/// leave the wall no width, which is refused as well.
#[test]
fn crossing_rounds_on_both_rims_are_refused() {
    for r in [2.0, 2.5] {
        let mut topo = Topology::new();
        let solid = slab(&mut topo);
        let edges = [rim_at(&topo, solid, 0.0), rim_at(&topo, solid, HEIGHT)];
        match fillet_v2(&mut topo, solid, &edges, r) {
            Err(_) => {}
            Ok(result) => assert!(
                !result.failed.is_empty(),
                "r = {r}: two rounds of {r} on a wall {HEIGHT} high were built"
            ),
        }
    }
}
