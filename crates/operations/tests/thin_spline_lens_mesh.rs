//! A thin lens between two spline sides, meshed coarsely.
//!
//! The lens runs 50 between its tips and is 0.075 at its thickest. Each
//! side is a cubic Bezier that sags less than the deflection on its halves,
//! so each keeps three samples. The lower side runs at nine tenths of the
//! upper's height and is parameterised unevenly: its middle sample sits above
//! the upper side's chord, so the two sides' chords cross though the curves
//! do not. The prism on the lens, its faces made
//! B-splines, must still mesh closed.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_heal::custom::convert_to_bspline::convert_solid_to_bspline;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::extrude::extrude;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const L: f64 = 50.0;
const H: f64 = 5.0;

fn bezier(points: [(f64, f64); 4], weights: [f64; 4]) -> NurbsCurve {
    NurbsCurve::new(
        3,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        points
            .iter()
            .map(|&(x, y)| Point3::new(x, y, 0.0))
            .collect(),
        weights.to_vec(),
    )
    .unwrap()
}

fn upper() -> NurbsCurve {
    bezier(
        [(0.0, 0.0), (L / 3.0, 1.0), (2.0 * L / 3.0, 1.0), (L, 0.0)],
        [1.0; 4],
    )
}

/// The upper side at nine tenths of its height, its parameter moved along
/// it by weights in a geometric run: its middle sample falls at a fifth of
/// the way, x = 10, above the upper side's chord there (0.432 against 0.3)
/// and under the upper side itself (0.48).
fn lower() -> NurbsCurve {
    let c = 0.25_f64;
    bezier(
        [(0.0, 0.0), (L / 3.0, 0.9), (2.0 * L / 3.0, 0.9), (L, 0.0)],
        [1.0, c, c * c, c * c * c],
    )
}

/// The lens's area: the region under the upper side less that under the
/// lower, each by the trapezoid rule along its curve.
fn lens_area() -> f64 {
    let under = |curve: &NurbsCurve| {
        let n = 200_000;
        let mut last = curve.evaluate(0.0);
        let mut area = 0.0;
        for i in 1..=n {
            let q = curve.evaluate(f64::from(i) / f64::from(n));
            area += 0.5 * (q.x() - last.x()) * (q.y() + last.y());
            last = q;
        }
        area
    };
    under(&upper()) - under(&lower())
}

fn mesh_volume(mesh: &brepkit_operations::tessellate::TriangleMesh) -> f64 {
    mesh.indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
            let (a, b, c) = (
                Vec3::new(a.x(), a.y(), a.z()),
                Vec3::new(b.x(), b.y(), b.z()),
                Vec3::new(c.x(), c.y(), c.z()),
            );
            a.dot(b.cross(c)) / 6.0
        })
        .sum()
}

#[test]
fn a_thin_lens_whose_side_chords_cross_meshes_closed() {
    let mut topo = Topology::new();
    let [a, b] = [Point3::new(0.0, 0.0, 0.0), Point3::new(L, 0.0, 0.0)]
        .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
    let low = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(lower())));
    let up = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(upper())));
    let wire = topo.add_wire(
        Wire::new(
            vec![OrientedEdge::new(low, true), OrientedEdge::new(up, false)],
            true,
        )
        .unwrap(),
    );
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: 0.0 },
    ));
    let solid = extrude(&mut topo, face, normal, H).unwrap();
    convert_solid_to_bspline(&mut topo, solid).unwrap();

    for deflection in [0.5, 0.25, 0.1] {
        let mesh = tessellate_solid(&topo, solid, deflection).unwrap();
        assert!(
            is_watertight(&mesh),
            "deflection {deflection}: mesh not watertight"
        );
    }
    // The lens is thinner than a coarse mesh's chords sag; a fine one holds
    // its volume.
    let expected = lens_area() * H;
    let volume = mesh_volume(&tessellate_solid(&topo, solid, 1e-3).unwrap());
    assert!(
        (volume - expected).abs() < 1e-2 * expected,
        "volume {volume}, lens {expected}"
    );
}
