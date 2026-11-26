//! A profile of lines with an ellipse arc among them, the arc given as the
//! rational B-spline a reference kernel writes for it, revolved about an
//! axis its profile touches. Each profile edge off the axis sweeps one face,
//! as the primitives have it: the lines their exact planes and cylinders, the
//! arc one rational B-spline wall.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, PI, TAU};

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::revolve::revolve;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_planar_face_from_wire;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);

/// A closed planar face through `points`, each edge a line but the ones
/// `curves` names by their first point's index.
fn profile(topo: &mut Topology, points: &[Point3], curves: &[(usize, NurbsCurve)]) -> FaceId {
    let vertices: Vec<_> = points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let edges = (0..points.len())
        .map(|i| {
            let curve = curves
                .iter()
                .find(|(at, _)| *at == i)
                .map_or(EdgeCurve::Line, |(_, c)| EdgeCurve::NurbsCurve(c.clone()));
            let edge = topo.add_edge(Edge::new(
                vertices[i],
                vertices[(i + 1) % points.len()],
                curve,
            ));
            OrientedEdge::new(edge, true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    make_planar_face_from_wire(topo, wire).unwrap()
}

/// The ellipse about (0, 0, 30) of radius 30 along -z and 16 along +y, from
/// 30 to 150 degrees: one rational quadratic span.
fn ellipse_arc() -> NurbsCurve {
    let (a, b) = (PI / 6.0, 5.0 * PI / 6.0);
    let z0 = 30.0 - 30.0 * (PI / 6.0).cos();
    NurbsCurve::new(
        2,
        vec![a, a, a, b, b, b],
        vec![
            Point3::new(0.0, 8.0, z0),
            Point3::new(0.0, 32.0, 30.0),
            Point3::new(0.0, 8.0, 60.0 - z0),
        ],
        vec![1.0, 0.5, 1.0],
    )
    .unwrap()
}

fn bulb_profile(topo: &mut Topology) -> FaceId {
    let z0 = 30.0 - 30.0 * (PI / 6.0).cos();
    let z1 = 60.0 - z0;
    profile(
        topo,
        &[
            Point3::new(0.0, 8.0, z0),
            Point3::new(0.0, 8.0, z1),
            Point3::new(0.0, 5.0, z1),
            Point3::new(0.0, 5.0, 90.0),
            Point3::new(0.0, 0.0, 90.0),
            Point3::new(0.0, 0.0, z0),
        ],
        &[(0, ellipse_arc())],
    )
}

/// The bulb's volume turned by `angle`: the ellipse's region past y = 8, the
/// slab between it and the axis, and the stem above.
fn bulb_volume(angle: f64) -> f64 {
    let z0 = 30.0 - 30.0 * (PI / 6.0).cos();
    let z1 = 60.0 - z0;
    // Pappus over the ellipse's segment past y = 8: integrate y^2 / 2 dz.
    let segment = {
        let n = 20_000;
        let h = (z1 - z0) / f64::from(n);
        (0..n)
            .map(|k| {
                let z = z0 + h * (f64::from(k) + 0.5);
                let t = 1.0 - ((z - 30.0) / 30.0).powi(2);
                let y = 16.0 * t.sqrt();
                0.5 * (y * y - 64.0) * h
            })
            .sum::<f64>()
    };
    let slab = 0.5 * 64.0 * (z1 - z0);
    let stem = 0.5 * 25.0 * (90.0 - z1);
    angle * (segment + slab + stem)
}

fn check(
    topo: &Topology,
    solid: brepkit_topology::solid::SolidId,
    angle: f64,
    faces: usize,
    walls: usize,
) {
    let label = format!("{angle}");
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{label}: {:?}", report.issues);
    assert!(
        is_watertight(&tessellate_solid(topo, solid, 0.01).unwrap()),
        "{label}"
    );
    let all = solid_faces(topo, solid).unwrap();
    assert_eq!(all.len(), faces, "{label}");
    let nurbs = all
        .iter()
        .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_)))
        .count();
    assert_eq!(nurbs, walls, "{label}: the ellipse's walls");
    let volume = bulb_volume(angle);
    let measured = solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (measured - volume).abs() < 1e-5 * volume,
        "{label}: {measured} against {volume}"
    );
    let oriented = oriented_solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (oriented - volume).abs() < 1e-3 * volume,
        "{label}: oriented {oriented} against {volume}"
    );
    for (point, inside) in [
        (Point3::new(0.0, 14.0, 30.0), true),
        (Point3::new(0.0, 17.0, 30.0), false),
        (Point3::new(0.0, 3.0, 80.0), true),
        (Point3::new(0.0, 6.0, 80.0), false),
    ] {
        // Half way round the turn, clear of a part turn's caps.
        let (sin, cos) = (0.5 * angle).sin_cos();
        let point = Point3::new(-point.y() * sin, point.y() * cos, point.z());
        let got = classify_point(topo, solid, point, 0.01, 1e-7).unwrap();
        let expected = if inside {
            PointClassification::Inside
        } else {
            PointClassification::Outside
        };
        assert_eq!(got, expected, "{label}: {point:?}");
    }
}

#[test]
fn a_full_turn_of_an_ellipse_arc_among_lines_sweeps_one_face_per_edge() {
    let mut topo = Topology::new();
    let face = bulb_profile(&mut topo);
    let solid = revolve(&mut topo, face, Point3::new(0.0, 0.0, 0.0), Z, TAU).unwrap();
    // The ellipse's wall, the step's annulus, the stem's cylinder and the
    // two discs.
    check(&topo, solid, TAU, 5, 1);
}

#[test]
fn a_part_turn_of_an_ellipse_arc_among_lines_keeps_its_lines_faces_whole() {
    // The ellipse's wall keeps a face per quarter turn; the planes and the
    // cylinder are one face each, beside the two caps.
    for (angle, walls) in [(FRAC_PI_2, 1), (PI, 2), (1.5 * PI, 3)] {
        let mut topo = Topology::new();
        let face = bulb_profile(&mut topo);
        let solid = revolve(&mut topo, face, Point3::new(0.0, 0.0, 0.0), Z, angle).unwrap();
        check(&topo, solid, angle, 6 + walls, walls);
    }
}

/// A half ellipse in the xz plane closed along the z axis, the arc as the
/// two rational quarter spans a reference kernel writes, makes a flat face.
#[test]
fn a_half_ellipse_closed_along_its_axis_is_flat() {
    let w = FRAC_1_SQRT_2;
    let arc = NurbsCurve::new(
        2,
        vec![
            -FRAC_PI_2, -FRAC_PI_2, -FRAC_PI_2, 0.0, 0.0, FRAC_PI_2, FRAC_PI_2, FRAC_PI_2,
        ],
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 20.0),
            Point3::new(10.0, 0.0, 40.0),
            Point3::new(0.0, 0.0, 40.0),
        ],
        vec![1.0, w, 1.0, w, 1.0],
    )
    .unwrap();
    let mut topo = Topology::new();
    let face = profile(
        &mut topo,
        &[Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 40.0)],
        &[(0, arc)],
    );
    let solid = revolve(&mut topo, face, Point3::new(0.0, 0.0, 0.0), Z, TAU).unwrap();
    let volume = 4.0 / 3.0 * PI * 10.0 * 10.0 * 20.0;
    let measured = solid_volume(&topo, solid, 0.001).unwrap();
    assert!(
        (measured - volume).abs() < 1e-5 * volume,
        "{measured} against {volume}"
    );
}
