//! A part extruded from an outline with a B-spline side is cut so that its
//! wall keeps only pieces of its surface, and a box is then fused to it
//! whose edges run through the part of the surface the cut took away.
//!
//! Each wall piece is bounded by its pcurves in the second boolean, so the
//! box's edges find no crossing on the surface beyond the piece: they come
//! out whole where they pass the part that was cut away.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::{solid_faces, solid_vertices};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);

/// The part's spline side: a cubic from (0, 0) dipping to y = -22.5 and
/// back up to (60, 0).
fn spline() -> NurbsCurve {
    NurbsCurve::new(
        3,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [(0.0, 0.0), (10.0, -30.0), (50.0, -30.0), (60.0, 0.0)]
            .iter()
            .map(|&(x, y)| Point3::new(x, y, 0.0))
            .collect(),
        vec![1.0; 4],
    )
    .unwrap()
}

/// The profile between the spline and the line back along `y = 0`, three
/// up.
fn prism(topo: &mut Topology) -> SolidId {
    let c = spline();
    let a = topo.add_vertex(Vertex::new(c.evaluate(0.0), 1e-7));
    let b = topo.add_vertex(Vertex::new(c.evaluate(1.0), 1e-7));
    let wall = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(c)));
    let back = topo.add_edge(Edge::new(b, a, EdgeCurve::Line));
    let wire = topo.add_wire(
        Wire::new(
            vec![OrientedEdge::new(wall, true), OrientedEdge::new(back, true)],
            true,
        )
        .unwrap(),
    );
    // The spline dips below its chord, so the loop runs clockwise seen from
    // above: the face looks down.
    let face = topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane {
            normal: -UP,
            d: 0.0,
        },
    ));
    extrude(topo, face, UP, 3.0).unwrap()
}

fn cuboid(topo: &mut Topology, lo: [f64; 3], hi: [f64; 3]) -> SolidId {
    let b = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
    transform_solid(topo, b, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
    b
}

#[test]
fn a_box_fused_past_a_cut_away_part_of_a_spline_wall_keeps_its_edges_whole() {
    let poses = [
        ("upright", Mat4::translation(0.0, 0.0, 0.0)),
        (
            "turned",
            Mat4::translation(5.0, -3.0, 2.0) * Mat4::rotation_z(0.7),
        ),
        (
            "mirrored",
            Mat4::translation(1.0e3, 2.0e3, -50.0) * Mat4::scale(-1.0, 1.0, 1.0),
        ),
    ];
    let (lo, hi) = ([20.0, -40.0, 1.0], [46.0, -5.0, 2.0]);
    for (tag, pose) in poses {
        let mut topo = Topology::new();
        let part = prism(&mut topo);
        let notch = cuboid(&mut topo, [37.1, -27.0, -1.0], [100.0, -3.0, 4.0]);
        let tool = cuboid(&mut topo, lo, hi);
        for solid in [part, notch, tool] {
            transform_solid(&mut topo, solid, &pose).unwrap();
        }
        let piece = boolean(&mut topo, BooleanOp::Cut, part, notch).unwrap();
        let walls = solid_faces(&topo, piece)
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_)))
            .count();
        assert_eq!(walls, 2, "{tag}: the wall in two pieces");

        let before = mesh_fallback_count();
        let fused = boolean(&mut topo, BooleanOp::Fuse, piece, tool).unwrap();
        assert_eq!(mesh_fallback_count(), before, "{tag}: mesh fallback");
        let report = validate_solid(&topo, fused).unwrap();
        assert!(report.is_valid(), "{tag}: {:?}", report.issues);

        // A vertex of the fuse is a corner of the box or lies on the piece.
        let corners: Vec<Point3> = (0..8)
            .map(|k: usize| {
                let pick = |bit: usize| {
                    if k >> bit & 1 == 1 { hi[bit] } else { lo[bit] }
                };
                pose.mul_point(Point3::new(pick(0), pick(1), pick(2)))
            })
            .collect();
        for vertex in solid_vertices(&topo, fused).unwrap() {
            let p = topo.vertex(vertex).unwrap().point();
            if corners.iter().any(|&c| (c - p).length() < 1e-6) {
                continue;
            }
            let class = classify_point(&topo, piece, p, 0.0, 1e-6).unwrap();
            assert_eq!(
                class,
                PointClassification::OnBoundary,
                "{tag}: a vertex at {p:?} off the piece"
            );
        }

        let cut = boolean(&mut topo, BooleanOp::Cut, piece, tool).unwrap();
        let common = boolean(&mut topo, BooleanOp::Intersect, piece, tool).unwrap();
        let volume = |s: SolidId| solid_volume(&topo, s, 1e-3).unwrap();
        let (whole, box_volume) = (volume(piece), 26.0 * 35.0);
        assert!(
            (volume(cut) + volume(common) - whole).abs() < 1e-6 * whole,
            "{tag}: the cut and the common part make the piece"
        );
        assert!(
            (volume(fused) + volume(common) - whole - box_volume).abs() < 1e-6 * whole,
            "{tag}: the fuse and the common part make the piece and the box"
        );
    }
}
