//! A D-shaped prism standing in a frustum with one end just poking through
//! its wall: the cut takes the prism out exactly, the poke included.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cone;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_face_from_wire;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const FRUSTUM: f64 = 2.0 * std::f64::consts::PI * 15.75;

/// A prism 1 deep along y from y = -0.9, its profile a chord on x = 1.5
/// from z = 0.5 to 2.5 and an arch out to x = 2.5. Its y = -0.9 end reaches
/// 2.66 from the frustum's axis at z = 1.5, past the wall's 2.625 there.
fn d_prism(topo: &mut Topology, arch: EdgeCurve) -> SolidId {
    let (a, b) = (Point3::new(1.5, -0.9, 0.5), Point3::new(1.5, -0.9, 2.5));
    let (va, vb) = (
        topo.add_vertex(Vertex::new(a, 1e-7)),
        topo.add_vertex(Vertex::new(b, 1e-7)),
    );
    let arch = topo.add_edge(Edge::new(va, vb, arch));
    let chord = topo.add_edge(Edge::new(vb, va, EdgeCurve::Line));
    let wire = Wire::new(
        vec![
            OrientedEdge::new(arch, true),
            OrientedEdge::new(chord, true),
        ],
        true,
    )
    .unwrap();
    let wire = topo.add_wire(wire);
    let face = make_face_from_wire(topo, wire).unwrap();
    extrude(topo, face, Vec3::new(0.0, 1.0, 0.0), 1.0).unwrap()
}

/// `make_cone(3, 1.5, 6)` cut by the prism: exact, valid, holding the
/// frustum's volume less the prism's within `poke` (what the prism holds
/// beyond the wall), and inside exactly where the frustum is and the prism
/// is not.
fn assert_cut_takes_the_prism(arch: EdgeCurve, prism_volume: f64, poke: f64) {
    let mut topo = Topology::new();
    let frustum = make_cone(&mut topo, 3.0, 1.5, 6.0).unwrap();
    let prism = d_prism(&mut topo, arch);
    let before = mesh_fallback_count();
    let cut = boolean(&mut topo, BooleanOp::Cut, frustum, prism).unwrap();
    assert_eq!(mesh_fallback_count(), before, "the cut stays exact");
    let report = validate_solid(&topo, cut).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let volume = solid_volume(&topo, cut, 0.001).unwrap();
    let least = FRUSTUM - prism_volume;
    assert!(
        volume > least - 1e-6 && volume < least + poke,
        "cut volume {volume}, the frustum less the whole prism {least}"
    );
    let class = |solid: SolidId, p: Point3| classify_point(&topo, solid, p, 0.01, 1e-7).unwrap();
    for i in 0..=6 {
        for j in 0..=6 {
            for k in 0..=11 {
                let p = Point3::new(
                    0.2f64.mul_add(f64::from(i), 1.405),
                    0.2f64.mul_add(f64::from(j), -0.995),
                    0.2f64.mul_add(f64::from(k), 0.405),
                );
                let (f, t) = (class(frustum, p), class(prism, p));
                if [f, t].contains(&PointClassification::OnBoundary) {
                    continue;
                }
                let want = if f == PointClassification::Inside && t == PointClassification::Outside
                {
                    PointClassification::Inside
                } else {
                    PointClassification::Outside
                };
                assert_eq!(class(cut, p), want, "at {p:?}");
            }
        }
    }
}

/// A half-disc prism: its rim circle crosses the frustum's wall twice
/// between the samples that look for such crossings.
#[test]
fn a_half_disc_prism_poking_through_a_frustum_wall_cuts_exactly() {
    let rim = Circle3D::new(Point3::new(1.5, -0.9, 1.5), Vec3::new(0.0, -1.0, 0.0), 1.0).unwrap();
    assert_cut_takes_the_prism(EdgeCurve::Circle(rim), std::f64::consts::FRAC_PI_2, 2e-2);
}

/// A half-ellipse prism, 1.05 out along x and 1 along z: its rim ellipse
/// crosses the frustum's wall twice between the samples that look for such
/// crossings.
#[test]
fn a_half_ellipse_prism_poking_through_a_frustum_wall_cuts_exactly() {
    let rim = Ellipse3D::new_with_ref(
        Point3::new(1.5, -0.9, 1.5),
        Vec3::new(0.0, -1.0, 0.0),
        1.05,
        1.0,
        Vec3::new(1.0, 0.0, 0.0),
    )
    .unwrap();
    assert_cut_takes_the_prism(
        EdgeCurve::Ellipse(rim),
        std::f64::consts::FRAC_PI_2 * 1.05,
        2e-2,
    );
}

/// A half-ellipse prism 1.1 out along x: its near cap grazes the wall too,
/// past it by less than the sag of the chords that stand in for the cap's
/// rim, so the section across the cap ends at the rim's crossings with the
/// cone and no chord.
#[test]
fn a_half_ellipse_prism_grazing_the_wall_through_its_near_cap_cuts_exactly() {
    let rim = Ellipse3D::new_with_ref(
        Point3::new(1.5, -0.9, 1.5),
        Vec3::new(0.0, -1.0, 0.0),
        1.1,
        1.0,
        Vec3::new(1.0, 0.0, 0.0),
    )
    .unwrap();
    assert_cut_takes_the_prism(
        EdgeCurve::Ellipse(rim),
        std::f64::consts::FRAC_PI_2 * 1.1,
        5e-2,
    );
}

/// A parabolic-arch prism: its NURBS wall meets the cone in a section the
/// marcher traces against the cone itself.
#[test]
fn an_arch_prism_poking_through_a_frustum_wall_cuts_exactly() {
    let arch = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![
            Point3::new(1.5, -0.9, 0.5),
            Point3::new(3.5, -0.9, 1.5),
            Point3::new(1.5, -0.9, 2.5),
        ],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();
    assert_cut_takes_the_prism(EdgeCurve::NurbsCurve(arch), 4.0 / 3.0, 2e-2);
}
