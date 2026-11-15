//! A blend with large tangent weights bends back on the way across. Where
//! the two faces are at different heights it rises while it turns, so it
//! stays regular, and its area and its faired copy's area are the surface
//! integral. Where the two faces are in one plane it folds.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::panic
)]

mod surface_fixtures;

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::vec::Point3;
use brepkit_operations::OperationsError;
use brepkit_operations::blend_surface::{
    BlendContinuity, BlendSurfaceOptions, blend_surface, fair_surface,
};
use brepkit_operations::measure::face_area;
use brepkit_operations::tessellate::tessellate_with_uvs;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_planar_face_from_wire, make_polygon_wire};
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use surface_fixtures::Pose;
use surface_fixtures::fold::first_fold;

fn poses() -> [(&'static str, Pose); 3] {
    [
        ("upright", Pose::upright()),
        ("turned", Pose::turned()),
        ("mirrored", Pose::mirrored()),
    ]
}

/// A 4 by 10 flat sheet from `x` at height `z`.
fn sheet(topo: &mut Topology, pose: Pose, x: f64, z: f64) -> FaceId {
    let corners = [
        Point3::new(x, 0.0, z),
        Point3::new(x + 4.0, 0.0, z),
        Point3::new(x + 4.0, 10.0, z),
        Point3::new(x, 10.0, z),
    ]
    .map(|p| pose.point(p));
    let wire = make_polygon_wire(topo, &corners, 1e-7).unwrap();
    make_planar_face_from_wire(topo, wire).unwrap()
}

fn edge_at(topo: &Topology, pose: Pose, face: FaceId, mid: Point3) -> EdgeId {
    let mid = pose.point(mid);
    let wire = topo.face(face).unwrap().outer_wire();
    topo.wire(wire)
        .unwrap()
        .edges()
        .iter()
        .map(brepkit_topology::wire::OrientedEdge::edge)
        .find(|&edge| {
            let e = topo.edge(edge).unwrap();
            let a = topo.vertex(e.start()).unwrap().point();
            let b = topo.vertex(e.end()).unwrap().point();
            ((a + (b - a) * 0.5) - mid).length() < 1e-9
        })
        .expect("an edge with that middle")
}

/// The blend from the sheet at height 0 to the one a gap of 6 away at
/// height `rise`.
fn blend(
    topo: &mut Topology,
    pose: Pose,
    rise: f64,
    continuity: BlendContinuity,
    weight: f64,
) -> Result<FaceId, OperationsError> {
    let a = sheet(topo, pose, -4.0, 0.0);
    let b = sheet(topo, pose, 6.0, rise);
    let edge_a = edge_at(topo, pose, a, Point3::new(0.0, 5.0, 0.0));
    let edge_b = edge_at(topo, pose, b, Point3::new(6.0, 5.0, rise));
    blend_surface(
        topo,
        &[edge_a],
        a,
        &[edge_b],
        b,
        &BlendSurfaceOptions {
            continuity,
            weight_a: weight,
            weight_b: weight,
        },
    )
}

fn nurbs(topo: &Topology, face: FaceId) -> NurbsSurface {
    let FaceSurface::Nurbs(surface) = topo.face(face).unwrap().surface() else {
        panic!("not a spline face");
    };
    surface.clone()
}

/// `integral |Su x Sv| du dv` at 8 Gauss points on each eighth of each knot
/// span, and the least `|Su x Sv|` met.
fn integral_area(surface: &NurbsSurface) -> (f64, f64) {
    let spans = |knots: &[f64]| -> Vec<(f64, f64)> {
        let mut found = Vec::new();
        for pair in knots.windows(2) {
            if pair[1] - pair[0] > 1e-12 {
                for k in 0..8 {
                    let at = |i: f64| pair[0] + (pair[1] - pair[0]) * i / 8.0;
                    found.push((at(f64::from(k)), at(f64::from(k + 1))));
                }
            }
        }
        found
    };
    let points = gauss_legendre_points(8);
    let (mut sum, mut least) = (0.0, f64::INFINITY);
    for &(u0, u1) in &spans(surface.knots_u()) {
        for &(v0, v1) in &spans(surface.knots_v()) {
            for pu in points {
                for pv in points {
                    let u = 0.5 * (u0 + u1) + 0.5 * (u1 - u0) * pu.x;
                    let v = 0.5 * (v0 + v1) + 0.5 * (v1 - v0) * pv.x;
                    let d = surface.derivatives(u, v, 1);
                    let jacobian = d[1][0].cross(d[0][1]).length();
                    least = least.min(jacobian);
                    sum += jacobian * pu.w * pv.w * 0.25 * (u1 - u0) * (v1 - v0);
                }
            }
        }
    }
    (sum, least)
}

/// Each mesh vertex lies where its parameters say, and the mesh's area is
/// the surface's to the share a chord mesh at this deflection loses.
fn assert_mesh_is_the_surface(label: &str, topo: &Topology, face: FaceId, exact: f64) {
    let surface = nurbs(topo, face);
    for deflection in [0.5, 0.1, 0.05, 0.02, 0.01, 0.005] {
        let mesh = tessellate_with_uvs(topo, face, deflection).unwrap();
        for (uv, &position) in mesh.uvs.iter().zip(&mesh.mesh.positions) {
            let off = (surface.evaluate(uv[0], uv[1]) - position).length();
            assert!(
                off < 1e-5,
                "{label} at {deflection}: a vertex is {off} from its parameters' point"
            );
        }
        let area = face_area(topo, face, deflection).unwrap();
        assert!(
            (area - exact).abs() < 2e-3 * exact,
            "{label} at {deflection}: mesh area {area}, surface area {exact}"
        );
    }
}

#[test]
fn a_blend_and_its_faired_copy_read_their_surface_area_at_every_weight() {
    // Weight, and the area the issue's blend read: the third was right, and
    // its faired copy read 245.08.
    for (weight, expected) in [(1.0, 63.88), (3.0, 71.41), (6.0, 125.27)] {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let blended = blend(&mut topo, pose, 2.0, BlendContinuity::G1, weight).unwrap();
            let faired = fair_surface(&mut topo, blended, 0.05).unwrap();
            for (what, face) in [("blend", blended), ("faired", faired)] {
                let label = format!("{what}, weight {weight}, {name}");
                let (exact, least) = integral_area(&nurbs(&topo, face));
                assert!(least > 1.0, "{label}: the surface is singular ({least})");
                assert!(
                    (exact - expected).abs() < 0.02,
                    "{label}: surface area {exact}, expected {expected}"
                );
                assert_mesh_is_the_surface(&label, &topo, face, exact);
            }
        }
    }
}

#[test]
fn a_blend_between_two_heights_does_not_fold_at_any_weight() {
    for continuity in [BlendContinuity::G1, BlendContinuity::G2] {
        for weight in [0.5, 2.0, 2.8, 2.9, 6.0, 12.0, 40.0] {
            for (name, pose) in poses() {
                let mut topo = Topology::new();
                let face = blend(&mut topo, pose, 2.0, continuity, weight)
                    .unwrap_or_else(|e| panic!("{continuity:?} weight {weight} {name}: {e}"));
                let surface = nurbs(&topo, face);
                assert_eq!(
                    first_fold(&surface),
                    None,
                    "{continuity:?} weight {weight} {name}"
                );
                let faired = fair_surface(&mut topo, face, 0.05).unwrap();
                assert_eq!(
                    first_fold(&nurbs(&topo, faired)),
                    None,
                    "faired, {continuity:?} weight {weight} {name}"
                );
            }
        }
    }
}

/// In one plane the G1 blend is the cubic with control values `0, 2w, 6 -
/// 2w, 6` across the gap, which runs backwards in the middle once `w > 3`.
/// At 3 exactly it stops there and does not turn back.
#[test]
fn the_fold_measure_finds_a_blend_in_one_plane_that_turns_back() {
    for (name, pose) in poses() {
        for weight in [0.5, 1.0, 2.0, 2.9, 2.99] {
            let mut topo = Topology::new();
            let face = blend(&mut topo, pose, 0.0, BlendContinuity::G1, weight).unwrap();
            let surface = nurbs(&topo, face);
            assert_eq!(first_fold(&surface), None, "{name} weight {weight}");
            let (exact, _) = integral_area(&surface);
            assert!(
                (exact - 60.0).abs() < 1e-6,
                "{name} weight {weight}: {exact}"
            );
        }
        for weight in [3.01, 3.5, 6.0, 20.0] {
            let mut topo = Topology::new();
            let face = blend(&mut topo, pose, 0.0, BlendContinuity::G1, weight);
            // A refusal is also right: see the ignored test below.
            if let Ok(face) = face {
                let fold = first_fold(&nurbs(&topo, face));
                let fold = fold.unwrap_or_else(|| panic!("{name} weight {weight}: no fold"));
                // The cubic turns back at two places, one each side of the
                // middle, and no nearer an end than a tenth of the way.
                assert!((0.1..=0.9).contains(&fold.v), "{name} weight {weight}");
            }
        }
    }
}

/// `blend_surface` returns a surface that folds. Tracked in #2469.
#[test]
#[ignore = "blend_surface does not refuse a surface that folds (#2469)"]
fn a_blend_in_one_plane_that_folds_is_refused() {
    for (name, pose) in poses() {
        for weight in [3.01, 3.5, 6.0, 20.0] {
            let mut topo = Topology::new();
            let refused = blend(&mut topo, pose, 0.0, BlendContinuity::G1, weight);
            let Err(OperationsError::InvalidInput { reason }) = refused else {
                panic!("{name} weight {weight}: {refused:?}");
            };
            assert!(reason.contains("fold"), "{name} weight {weight}: {reason}");
        }
    }
}
