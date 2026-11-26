//! A strip on a floor running into a post that rises above it and a slab
//! beside the post: the strip's edge between its top and its wall is
//! chamfered, and at the strip's end it meets the post's side above the
//! top and the slab's end beside the wall, both in one plane square to the
//! edge. The chamfer bares a triangle of that plane between the two, and
//! the triangle joins them: the post's side and the slab's end become one
//! face. The result is valid, watertight, and takes off a prism of
//! d1 d2 / 2 per unit length along the edge; the second chamfer's wall
//! distance is the wall's height, so the wall goes too.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer_two_on;
use brepkit_operations::heal::unify_faces;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

/// The strip's length along x, from the post at x = 0.
const STRIP: f64 = 13.0;
/// The strip's wall height above the floor, and its top's width.
const WALL: f64 = 7.0;
const TOP: f64 = 10.0;

fn block(topo: &mut Topology, from: [f64; 3], to: [f64; 3]) -> SolidId {
    let b = make_box(topo, to[0] - from[0], to[1] - from[1], to[2] - from[2]).unwrap();
    transform_solid(topo, b, &Mat4::translation(from[0], from[1], from[2])).unwrap();
    b
}

/// The floor (y < 0); on it the post (x < 0, z < 0, up to twice the wall),
/// the slab (x < 0, z > 0, up to the wall) and the strip (x > 0, z < 0, up
/// to the wall). The strip's wall faces +z, its top +y.
fn part(topo: &mut Topology) -> SolidId {
    let pieces = [
        block(topo, [-10.0, -5.0, -TOP], [STRIP, 0.0, 20.0]),
        block(topo, [-10.0, 0.0, -TOP], [0.0, 2.0 * WALL, 0.0]),
        block(topo, [-10.0, 0.0, 0.0], [0.0, WALL, 20.0]),
        block(topo, [0.0, 0.0, -TOP], [STRIP, WALL, 0.0]),
    ];
    let mut solid = pieces[0];
    for &p in &pieces[1..] {
        solid = boolean(topo, BooleanOp::Fuse, solid, p).unwrap();
    }
    unify_faces(topo, solid).unwrap();
    solid
}

/// The strip's edge between its top and its wall (y = WALL, z = 0, x > 0),
/// and its top face.
fn strip_edge(topo: &Topology, solid: SolidId) -> (EdgeId, FaceId) {
    let edge = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let ed = topo.edge(e).unwrap();
            [ed.start(), ed.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                (p.y() - WALL).abs() < 1e-9 && p.z().abs() < 1e-9 && p.x() > -1e-9
            })
        })
        .unwrap();
    let top = solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&f| {
            let face = topo.face(f).unwrap();
            let on_edge = topo
                .wire(face.outer_wire())
                .unwrap()
                .edges()
                .iter()
                .any(|oe| oe.edge() == edge);
            on_edge
                && matches!(face.surface(), FaceSurface::Plane { normal, .. }
                    if (normal.y().abs() - 1.0).abs() < 1e-9)
        })
        .unwrap();
    (edge, top)
}

/// The faces in the plane x = 0.
fn faces_at_the_corner(topo: &Topology, solid: SolidId) -> usize {
    solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&f| match topo.face(f).unwrap().surface() {
            FaceSurface::Plane { normal, d } => {
                (normal.x().abs() - 1.0).abs() < 1e-9 && d.abs() < 1e-9
            }
            _ => false,
        })
        .count()
}

#[test]
fn the_bared_triangle_joins_the_post_and_the_slab() {
    for (d1, d2, wall_goes) in [(3.0, 2.0, false), (9.0, WALL, true)] {
        let mut topo = Topology::new();
        let solid = part(&mut topo);
        let faces_before = solid_faces(&topo, solid).unwrap().len();
        assert_eq!(
            faces_at_the_corner(&topo, solid),
            2,
            "the post's side and the slab's end"
        );
        let before = solid_volume(&topo, solid, 1e-3).unwrap();
        let (edge, top) = strip_edge(&topo, solid);
        let result = chamfer_two_on(&mut topo, solid, &[(edge, top)], d1, d2).unwrap();
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "d {d1}/{d2}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
        assert!(is_watertight(&mesh), "d {d1}/{d2}: open mesh");
        let after = solid_volume(&topo, result, 1e-3).unwrap();
        let removed = 0.5 * d1 * d2 * STRIP;
        assert!(
            ((before - after) - removed).abs() < 1e-6 * before,
            "d {d1}/{d2}: took off {}, the prism holds {removed}",
            before - after
        );
        assert_eq!(
            faces_at_the_corner(&topo, result),
            1,
            "d {d1}/{d2}: the triangle joins the two faces"
        );
        // The bevel added, the two corner faces one, and the wall gone when
        // the chamfer reaches the floor.
        let expected = faces_before - usize::from(wall_goes);
        assert_eq!(
            solid_faces(&topo, result).unwrap().len(),
            expected,
            "d {d1}/{d2}"
        );
    }
}
