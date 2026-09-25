//! A torus face that a boolean trims meshes on its own over the face itself:
//! its triangles hold the face's exact area to 1%, where the grid they replace
//! filled the face's `(u, v)` box (a notched ring meshed whole, a half ring
//! meshed to nothing, a patch inside a thin rod meshed as the whole ring).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::surfaces::ToroidalSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::face_area;
use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere, make_torus};
use brepkit_operations::tessellate::{tessellate, tessellate_with_uvs};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const RING: (f64, f64) = (4.0, 1.5);

type Tool = Box<dyn Fn(&mut Topology) -> SolidId>;

fn meshed_area(topo: &Topology, face: FaceId) -> f64 {
    let mesh = tessellate(topo, face, 0.005).unwrap();
    mesh.indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
            (b - a).cross(c - a).length() / 2.0
        })
        .sum()
}

fn torus_faces(topo: &Topology, piece: SolidId) -> Vec<FaceId> {
    solid_faces(topo, piece)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Torus(_)))
        .collect()
}

/// The ring against a tool placed by `place`, under `op`.
fn ring_with(
    op: BooleanOp,
    tool: impl Fn(&mut Topology) -> SolidId,
    place: Mat4,
) -> (Topology, SolidId) {
    let mut topo = Topology::new();
    let ring = make_torus(&mut topo, RING.0, RING.1, 32).unwrap();
    let t = tool(&mut topo);
    transform_solid(&mut topo, t, &place).unwrap();
    let piece = boolean(&mut topo, op, ring, t).unwrap();
    (topo, piece)
}

#[test]
fn trimmed_torus_faces_mesh_their_own_region() {
    let cube = |s: (f64, f64, f64)| move |t: &mut Topology| make_box(t, s.0, s.1, s.2).unwrap();
    let cases: Vec<(&str, Tool, Mat4)> = vec![
        (
            "box corner notch",
            Box::new(cube((4.0, 4.0, 4.0))),
            Mat4::translation(3.0, -2.0, -2.0),
        ),
        (
            "box through the ring",
            Box::new(cube((2.0, 20.0, 4.0))),
            Mat4::translation(-1.0, -10.0, -2.0),
        ),
        (
            "box over z > 0.4",
            Box::new(cube((20.0, 20.0, 10.0))),
            Mat4::translation(-10.0, -10.0, 0.4),
        ),
        (
            "box over x > 0",
            Box::new(cube((10.0, 20.0, 10.0))),
            Mat4::translation(0.0, -10.0, -5.0),
        ),
        (
            "box over x > 1",
            Box::new(cube((20.0, 20.0, 10.0))),
            Mat4::translation(1.0, -10.0, -5.0),
        ),
        (
            "ball in the hole",
            Box::new(|t: &mut Topology| make_sphere(t, 3.0, 32).unwrap()),
            Mat4::identity(),
        ),
        (
            "rod cutting the tube",
            Box::new(|t: &mut Topology| make_cylinder(t, 4.2, 10.0).unwrap()),
            Mat4::translation(0.0, 0.0, -5.0),
        ),
        (
            "thin rod across the tube",
            Box::new(|t: &mut Topology| make_cylinder(t, 0.5, 10.0).unwrap()),
            Mat4::translation(4.0, 0.0, -5.0),
        ),
        (
            "ring shifted up",
            Box::new(|t: &mut Topology| make_torus(t, 4.0, 1.5, 32).unwrap()),
            Mat4::translation(0.0, 0.0, 1.0),
        ),
    ];
    for (name, tool, place) in &cases {
        for op in [BooleanOp::Fuse, BooleanOp::Cut, BooleanOp::Intersect] {
            let (topo, piece) = ring_with(op, tool, *place);
            let faces = torus_faces(&topo, piece);
            assert!(!faces.is_empty(), "{name} {op:?}: no torus face");
            for face in faces {
                let exact = face_area(&topo, face, 0.005).unwrap();
                let meshed = meshed_area(&topo, face);
                assert!(
                    (meshed - exact).abs() < 1e-2 * exact,
                    "{name} {op:?}: mesh area {meshed}, exact {exact}"
                );
            }
        }
    }
}

/// Half the ring past a plane through its axis is `2 pi² R r` of surface, and
/// the part above `z = h` spans the tube angle between `asin(h / r)` and its
/// supplement, `2 pi r R (pi - 2 asin(h / r))`.
#[test]
fn half_rings_mesh_their_closed_form_area() {
    let (big, small) = RING;
    let cube = |s: (f64, f64, f64)| move |t: &mut Topology| make_box(t, s.0, s.1, s.2).unwrap();
    for (name, (topo, piece), truth) in [
        (
            "past x = 0",
            ring_with(
                BooleanOp::Intersect,
                cube((10.0, 20.0, 10.0)),
                Mat4::translation(0.0, -10.0, -5.0),
            ),
            2.0 * PI * PI * big * small,
        ),
        (
            "above z = 0.4",
            ring_with(
                BooleanOp::Intersect,
                cube((20.0, 20.0, 10.0)),
                Mat4::translation(-10.0, -10.0, 0.4),
            ),
            2.0 * PI * small * big * 2.0f64.mul_add(-(0.4 / small).asin(), PI),
        ),
    ] {
        let faces = torus_faces(&topo, piece);
        assert_eq!(faces.len(), 1, "{name}: one torus face");
        let exact = face_area(&topo, faces[0], 0.005).unwrap();
        assert!(
            (exact - truth).abs() < 1e-9 * truth,
            "{name}: area {exact}, truth {truth}"
        );
        let meshed = meshed_area(&topo, faces[0]);
        assert!(
            (meshed - truth).abs() < 1e-2 * truth,
            "{name}: mesh area {meshed}, truth {truth}"
        );
    }
}

/// A whole ring bounded, as files write it, by its two seam circles each run
/// both ways: it meshes as the whole ring, `4 pi² R r`.
#[test]
fn whole_ring_with_circle_seams_meshes_whole() {
    let (big, small) = RING;
    let mut topo = Topology::new();
    let o = Point3::new(0.0, 0.0, 0.0);
    let v0 = topo.add_vertex(Vertex::new(Point3::new(big + small, 0.0, 0.0), 1e-7));
    let tube = Circle3D::new(Point3::new(big, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), small).unwrap();
    let rim = Circle3D::new(o, Vec3::new(0.0, 0.0, 1.0), big + small).unwrap();
    let a = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Circle(tube)));
    let b = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Circle(rim)));
    let wire = Wire::new(
        vec![
            OrientedEdge::new(a, true),
            OrientedEdge::new(b, true),
            OrientedEdge::new(a, false),
            OrientedEdge::new(b, false),
        ],
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let torus = ToroidalSurface::new(o, big, small).unwrap();
    let face = topo.add_face(Face::new(wid, vec![], FaceSurface::Torus(torus)));
    let truth = 4.0 * PI * PI * big * small;
    let meshed = meshed_area(&topo, face);
    assert!(
        (meshed - truth).abs() < 1e-2 * truth,
        "mesh area {meshed}, truth {truth}"
    );
}

/// A trimmed torus face's per-face mesh keeps both parameters continuous
/// over every triangle, across the tube's seam as well as the ring's.
#[test]
fn trimmed_torus_face_uvs_stay_continuous() {
    let cube = |s: (f64, f64, f64)| move |t: &mut Topology| make_box(t, s.0, s.1, s.2).unwrap();
    for (name, place, size) in [
        (
            "past x = 0",
            Mat4::translation(0.0, -10.0, -5.0),
            (10.0, 20.0, 10.0),
        ),
        (
            "cube over the outer side",
            Mat4::translation(3.0, -2.0, -2.0),
            (4.0, 4.0, 4.0),
        ),
    ] {
        for op in [BooleanOp::Cut, BooleanOp::Intersect] {
            let (topo, piece) = ring_with(op, cube(size), place);
            for face in torus_faces(&topo, piece) {
                let mesh = tessellate_with_uvs(&topo, face, 0.01).unwrap();
                for t in mesh.mesh.indices.chunks_exact(3) {
                    for axis in 0..2 {
                        let vals = [0, 1, 2].map(|k| mesh.uvs[t[k] as usize][axis]);
                        let spread = vals.iter().copied().fold(f64::MIN, f64::max)
                            - vals.iter().copied().fold(f64::MAX, f64::min);
                        assert!(
                            spread < PI,
                            "{name} {op:?}: a triangle spans {spread} in {}",
                            ["u", "v"][axis]
                        );
                    }
                }
            }
        }
    }
}
