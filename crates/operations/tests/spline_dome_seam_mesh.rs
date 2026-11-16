//! A dome whose wall is one closed B-spline face, its seam run from the pole
//! to its rim and back, closed by a flat disc.
//!
//! The wall's only edges off its seam are its rim, a circle at v = 0, so the
//! rims span no v at all. The seam's samples are then laid by their own
//! projected v, not spaced evenly across that span, which would put every one
//! of them on the rim.
//!
//! The wall is the quadratic profile (R, 0), (R, H), (0, H) turned about z:
//! r = R (1 - t^2), z = H (2t - t^2), which holds 11/15 pi R^2 H.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use brepkit_math::curves::Circle3D;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::Solid;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const R: f64 = 10.0;
const H: f64 = 10.0;

/// The profile's radius and height at its three control points.
const PROFILE: [(f64, f64); 3] = [(R, 0.0), (R, H), (0.0, H)];

fn wall() -> NurbsSurface {
    let s = FRAC_1_SQRT_2;
    let circle = [
        (1.0, 0.0, 1.0),
        (1.0, 1.0, s),
        (0.0, 1.0, 1.0),
        (-1.0, 1.0, s),
        (-1.0, 0.0, 1.0),
        (-1.0, -1.0, s),
        (0.0, -1.0, 1.0),
        (1.0, -1.0, s),
        (1.0, 0.0, 1.0),
    ];
    let points = circle
        .iter()
        .map(|&(x, y, _)| {
            PROFILE
                .iter()
                .map(|&(r, z)| Point3::new(x * r, y * r, z))
                .collect()
        })
        .collect();
    let weights = circle.iter().map(|&(_, _, w)| vec![w; 3]).collect();
    NurbsSurface::new(
        2,
        2,
        vec![
            0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
        ],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        points,
        weights,
    )
    .unwrap()
}

fn mesh_volume(mesh: &brepkit_operations::tessellate::TriangleMesh) -> f64 {
    mesh.indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| {
                let p = mesh.positions[t[k] as usize];
                Vec3::new(p.x(), p.y(), p.z())
            });
            a.dot(b.cross(c)) / 6.0
        })
        .sum()
}

#[test]
fn a_dome_whose_rim_spans_no_v_lays_its_seam_by_projection() {
    let mut topo = Topology::new();
    let (pole, rim) = (Point3::new(0.0, 0.0, H), Point3::new(R, 0.0, 0.0));
    let [vp, va] = [pole, rim].map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
    let profile = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        PROFILE
            .iter()
            .rev()
            .map(|&(r, z)| Point3::new(r, 0.0, z))
            .collect(),
        vec![1.0; 3],
    )
    .unwrap();
    let seam = topo.add_edge(Edge::new(vp, va, EdgeCurve::NurbsCurve(profile)));
    let circle = Circle3D::new_with_ref(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        R,
        Vec3::new(1.0, 0.0, 0.0),
    )
    .unwrap();
    let rim_edge = topo.add_edge(Edge::new(va, va, EdgeCurve::Circle(circle)));
    let dome_wire = topo.add_wire(
        Wire::new(
            vec![
                OrientedEdge::new(seam, true),
                OrientedEdge::new(rim_edge, true),
                OrientedEdge::new(seam, false),
            ],
            true,
        )
        .unwrap(),
    );
    let dome = topo.add_face(Face::new(dome_wire, vec![], FaceSurface::Nurbs(wall())));
    let disc_wire =
        topo.add_wire(Wire::new(vec![OrientedEdge::new(rim_edge, false)], true).unwrap());
    let disc = topo.add_face(Face::new(
        disc_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ));
    let shell = topo.add_shell(Shell::new(vec![dome, disc]).unwrap());
    let solid = topo.add_solid(Solid::new(shell, Vec::new()));

    for deflection in [0.5, 0.25, 0.1] {
        let mesh = tessellate_solid(&topo, solid, deflection).unwrap();
        assert!(
            is_watertight(&mesh),
            "deflection {deflection}: mesh not watertight"
        );
    }
    let expected = 11.0 / 15.0 * PI * R * R * H;
    let volume = mesh_volume(&tessellate_solid(&topo, solid, 1e-3).unwrap());
    assert!(
        (volume - expected).abs() < 1e-2 * expected,
        "volume {volume}, closed form {expected}"
    );
}
