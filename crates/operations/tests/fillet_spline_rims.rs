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
    split_slab(topo, None)
}

/// The slab, its top rim in two pieces when `split` gives a fraction of the
/// outline's parameter to cut it at, away from the wall's seam (as a file
/// gives a rim a neighbour's edge met), the lid's wire then running from
/// the cut.
fn split_slab(topo: &mut Topology, split: Option<f64>) -> SolidId {
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
    let (d0, d1) = lifted.domain();
    let top_pieces = split.map(|f| {
        let t = (d1 - d0).mul_add(f, d0);
        let (left, right) = brepkit_math::nurbs::knot_ops::curve_split(&lifted, t).unwrap();
        let vp = topo.add_vertex(Vertex::new(lifted.evaluate(t), 1e-7));
        // The piece from the cut made first: a fillet walks a closed chain
        // from its first edge's start.
        let second = topo.add_edge(Edge::new(vp, vt, EdgeCurve::NurbsCurve(right)));
        let first = topo.add_edge(Edge::new(vt, vp, EdgeCurve::NurbsCurve(left)));
        (first, second)
    });
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
    let wall_wire = match top_pieces {
        Some((first, second)) => wire(
            topo,
            &[
                (bottom, true),
                (seam, true),
                (second, false),
                (first, false),
                (seam, false),
            ],
        ),
        None => wire(
            topo,
            &[(bottom, true), (seam, true), (top, false), (seam, false)],
        ),
    };
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
    let lid_wire = match top_pieces {
        Some((first, second)) => wire(topo, &[(second, true), (first, true)]),
        None => wire(topo, &[(top, true)]),
    };
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
    assert_slab_rounds_off(None, r, zs.len(), |topo, solid| {
        zs.iter().map(|&z| rim_at(topo, solid, z)).collect()
    });
}

/// Fillets the edges `pick` takes of a slab (its top rim cut at `split`)
/// with radius `r`, `rims` whole rims of it, and checks the result as
/// [`assert_rim_rounds_off`] does.
fn assert_slab_rounds_off(
    split: Option<f64>,
    r: f64,
    rims: usize,
    pick: impl Fn(&Topology, SolidId) -> Vec<EdgeId>,
) {
    let mut topo = Topology::new();
    let solid = split_slab(&mut topo, split);
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
    let edges = pick(&topo, solid);
    let before = solid_volume(&topo, solid, 0.001).unwrap();
    let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    assert_ne!(result.solid, solid, "r = {r}: the fillet changed nothing");
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    assert_edges_lie_on_their_faces(&topo, result.solid, r);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(
        is_watertight(&mesh),
        "r = {r}: mesh not watertight ({} open, {} over-used)",
        brepkit_operations::tessellate::boundary_edge_count(&mesh),
        brepkit_operations::tessellate::non_manifold_edge_count(&mesh)
    );
    #[allow(clippy::cast_precision_loss)]
    let removed = rims as f64
        * (1.0 - PI / 4.0)
        * r
        * r
        * 2.0f64.mul_add(-PI * centroid_from_corner(r), perimeter());
    let after = solid_volume(&topo, result.solid, 0.001).unwrap();
    let truth = before - removed;
    assert!(
        (after - truth).abs() < 1e-3 * truth,
        "r = {r}, split {split:?}: volume {after}, expected {truth} (removed {removed})"
    );
}

/// Every edge of `solid` lies on each face that uses it, sampled at three
/// points inside it. A closed band's seam runs across the round as its
/// section's arc: a straight seam would stand off the band by
/// `r (1 - cos 45°)`, and another reader then meshes the band up to it.
fn assert_edges_lie_on_their_faces(topo: &Topology, solid: SolidId, r: f64) {
    let faces_of = brepkit_topology::explorer::edge_to_face_map(topo, solid).unwrap();
    for edge_id in solid_edges(topo, solid).unwrap() {
        let edge = topo.edge(edge_id).unwrap();
        let (a, b) = (
            topo.vertex(edge.start()).unwrap().point(),
            topo.vertex(edge.end()).unwrap().point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        for face_id in &faces_of[&edge_id.index()] {
            let surface = topo.face(*face_id).unwrap().surface();
            for f in [0.25, 0.5, 0.75] {
                let p = edge
                    .curve()
                    .evaluate_with_endpoints((t1 - t0).mul_add(f, t0), a, b);
                let off = if let FaceSurface::Plane { normal, d } = surface {
                    (normal.dot(p - Point3::new(0.0, 0.0, 0.0)) - d).abs()
                } else {
                    let (u, v) = surface.project_point(p).unwrap();
                    (surface.evaluate(u, v).unwrap() - p).length()
                };
                assert!(
                    off < 1e-3,
                    "r = {r}: a {} edge stands {off} off its {} face",
                    edge.curve().type_tag(),
                    surface.type_tag()
                );
            }
        }
    }
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

/// The top rim in two pieces, cut away from the wall's seam, and rounded
/// whole: the walk starts at the cut, so the round's closed contact on the
/// wall crosses the wall's seam away from its vertex. The wall's trim cuts
/// the contact there, and its seam with it; it had bridged the contact's
/// vertex to the seam by an edge run out and back, which leaves the wall's
/// mesh open there.
#[test]
fn a_top_rim_in_pieces_rounds_off_across_the_wall_s_seam() {
    for split in [0.3, 0.6] {
        for r in [0.5, 1.0] {
            for reversed in [false, true] {
                assert_slab_rounds_off(Some(split), r, 1, |topo, solid| {
                    let mut pieces: Vec<EdgeId> = solid_edges(topo, solid)
                        .unwrap()
                        .into_iter()
                        .filter(|&e| {
                            let edge = topo.edge(e).unwrap();
                            matches!(edge.curve(), EdgeCurve::NurbsCurve(_))
                                && edge.start() != edge.end()
                        })
                        .collect();
                    assert_eq!(pieces.len(), 2);
                    if reversed {
                        pieces.reverse();
                    }
                    pieces
                });
            }
        }
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
/// cross on the wall. The first picked, the floor's, is made as it would be
/// alone, its contact on the wall `r` up; the lid's ball rests on the lid
/// and pivots on that contact, its centre `w = sqrt(r^2 - (2 r - H)^2)` in
/// from the wall, and the wall goes. At height `z` the section is the
/// outline set in by `d(z)`, of area `A - P d + pi d^2`.
#[test]
fn crossing_rounds_on_both_rims_pivot() {
    let length = perimeter();
    for r in [2.5, 3.0] {
        for floor_first in [true, false] {
            let mut topo = Topology::new();
            let solid = slab(&mut topo);
            let before = solid_volume(&topo, solid, 0.01).unwrap();
            let (floor, lid) = (rim_at(&topo, solid, 0.0), rim_at(&topo, solid, HEIGHT));
            let edges = if floor_first {
                [floor, lid]
            } else {
                [lid, floor]
            };
            let case = format!("r = {r}, floor first {floor_first}");
            let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
            assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
            let report = validate_solid(&topo, result.solid).unwrap();
            assert!(report.is_valid(), "{case}: {:?}", report.issues);
            let faces = brepkit_topology::explorer::solid_faces(&topo, result.solid).unwrap();
            assert_eq!(faces.len(), 4, "{case}: lid, floor and two bands");
            assert_edges_lie_on_their_faces(&topo, result.solid, r);
            let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{case}: mesh not watertight");
            let w = r.mul_add(r, -(2.0 * r - HEIGHT).powi(2)).sqrt();
            let set_in = |d: f64| length * d - PI * d * d;
            // The first round: `r - z = r sin t`, set in `r - r cos t`.
            let first = simpson(0.0, PI / 2.0, |t| set_in(r - r * t.cos()) * r * t.cos());
            // The second: `z - (H - r) = r sin s` from the pivot, set in
            // `w - r cos s`.
            let from = ((2.0 * r - HEIGHT) / r).asin();
            let second = simpson(from, PI / 2.0, |s| set_in(w - r * s.cos()) * r * s.cos());
            let removed = before - solid_volume(&topo, result.solid, 0.01).unwrap();
            assert!(
                (removed - (first + second)).abs() < 1e-6 * (first + second),
                "{case}: removed {removed}, the pivoting round takes off {}",
                first + second
            );
            // The first picked rim's face is set in by `r`, the other's by `w`.
            let plane_at = |z: f64| {
                faces
                    .iter()
                    .copied()
                    .find(|&f| {
                        matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal, d }
                            if (normal.z().abs() - 1.0).abs() < 1e-9 && (d.abs() - z).abs() < 1e-9)
                    })
                    .expect("a plane face at that height")
            };
            let ellipse = PI * SEMI_MAJOR * SEMI_MINOR;
            let inset = |d: f64| ellipse - set_in(d);
            let (floor_in, lid_in) = if floor_first { (r, w) } else { (w, r) };
            for (z, d) in [(0.0, floor_in), (HEIGHT, lid_in)] {
                let area =
                    brepkit_operations::measure::face_area(&topo, plane_at(z), 0.001).unwrap();
                assert!(
                    (area - inset(d)).abs() < 1e-3 * inset(d),
                    "{case}: the face at z = {z} has area {area}, set in by {d} it has {}",
                    inset(d)
                );
            }
        }
    }
}

/// Simpson's rule over `[a, b]`.
fn simpson(a: f64, b: f64, f: impl Fn(f64) -> f64) -> f64 {
    const N: u32 = 2000;
    let step = (b - a) / f64::from(N);
    let mut sum = f(a) + f(b);
    for i in 1..N {
        sum += f(f64::from(i).mul_add(step, a)) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * step / 3.0
}

/// Rounds of exactly half the wall's height on both rims (a full round)
/// meet along the wall's middle and leave it no width: the wall goes, and
/// the two bands share their contact. At height `z` from the middle the
/// section is the outline set in by `d = r - sqrt(r^2 - z^2)`, of area
/// `A - P d + pi d^2` (the outline is convex and bends no tighter than `d`),
/// so the solid holds `2 r A - (2 - pi / 2) P r^2 + pi (10 / 3 - pi) r^3`.
#[test]
fn rounds_of_half_the_wall_on_both_rims_make_a_full_round() {
    let r = HEIGHT / 2.0;
    let mut topo = Topology::new();
    let solid = slab(&mut topo);
    let edges = [rim_at(&topo, solid, 0.0), rim_at(&topo, solid, HEIGHT)];
    let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let faces = crate_faces(&topo, result.solid);
    assert_eq!(faces.len(), 4, "lid, floor and two bands: {faces:?}");
    assert_edges_lie_on_their_faces(&topo, result.solid, r);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(
        is_watertight(&mesh),
        "mesh not watertight ({} open, {} over-used)",
        brepkit_operations::tessellate::boundary_edge_count(&mesh),
        brepkit_operations::tessellate::non_manifold_edge_count(&mesh)
    );
    let area = PI * SEMI_MAJOR * SEMI_MINOR;
    let truth = (2.0 * r).mul_add(
        area,
        (PI * PI.mul_add(-1.0, 10.0 / 3.0) * r)
            .mul_add(r * r, -(2.0 - PI / 2.0) * perimeter() * r * r),
    );
    let volume = solid_volume(&topo, result.solid, 0.001).unwrap();
    assert!(
        (volume - truth).abs() < 1e-3 * truth,
        "volume {volume}, the full round {truth}"
    );
}

/// The outline's tightest bend, at the ends of its major axis, has radius
/// `b^2 / a`. A ball wider than that cannot round the tips: its contact on
/// the lid runs to a cusp and back there (the outline set in by `r` crosses
/// itself), and the lid was trimmed to a loop crossing itself, a valid-looking
/// solid with several times the volume removed. Such a round is refused; one
/// just inside the bend still rounds off.
#[test]
fn a_round_past_the_outlines_tightest_bend_is_refused() {
    let tightest = SEMI_MINOR * SEMI_MINOR / SEMI_MAJOR;
    assert_rim_rounds_off(HEIGHT, 0.96 * tightest);
    for r in [1.12 * tightest, 1.25 * tightest] {
        let mut topo = Topology::new();
        let solid = slab(&mut topo);
        let rim = rim_at(&topo, solid, HEIGHT);
        if let Ok(result) = fillet_v2(&mut topo, solid, &[rim], r) {
            assert!(
                !result.failed.is_empty(),
                "r = {r}, past the tips' bend {tightest}: the round was built"
            );
        }
    }
}
