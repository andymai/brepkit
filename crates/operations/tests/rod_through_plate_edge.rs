//! A rod or a cone frustum standing through a plate's edge, fused with the
//! plate: the plate takes a window out of the tool's wall, and where the
//! window straddles the wall's seam the wall's outer wire carries it as a
//! notch between two rims and the seam's two copies, which is no box in
//! `(u, v)`. Each fuse matches its closed form, each wall its area, and each
//! wall's own mesh stays within the deflection of the surface, whichever way
//! the tool's seam points.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// The area of a disc of radius `r` about `x = cx` where `x > 0`.
fn disc_past_zero(r: f64, cx: f64) -> f64 {
    let d = (-cx / r).clamp(-1.0, 1.0);
    r * r * (d.acos() - d * (1.0 - d * d).sqrt())
}

/// The plate `[0, 10] x [0, 10] x [0, 2]` fused with `tool` placed at
/// `(cx, 5, -4)`, turned `spin` about its axis.
fn fuse(cone: bool, cx: f64, spin: f64) -> (Topology, SolidId) {
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
    let tool = if cone {
        make_cone(&mut topo, 1.2, 0.6, 10.0).unwrap()
    } else {
        make_cylinder(&mut topo, 1.0, 10.0).unwrap()
    };
    let place = Mat4::translation(cx, 5.0, -4.0) * Mat4::rotation_z(spin);
    transform_solid(&mut topo, tool, &place).unwrap();
    let fused = boolean(&mut topo, BooleanOp::Fuse, plate, tool).unwrap();
    (topo, fused)
}

/// A wall's own mesh at deflection `DEFLECTION`: its area, and how far the
/// midpoint of any of its edges falls inside the wall, whose radius at
/// height `z` is `radius(z)` about the vertical axis through `(cx, 5)`.
fn wall_mesh(
    topo: &Topology,
    face: brepkit_topology::face::FaceId,
    cx: f64,
    radius: &dyn Fn(f64) -> f64,
) -> (f64, f64) {
    let mesh = tessellate(topo, face, DEFLECTION).unwrap();
    let (mut area, mut sag) = (0.0, 0.0_f64);
    for t in mesh.indices.chunks(3) {
        let corners = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
        area += (corners[1] - corners[0])
            .cross(corners[2] - corners[0])
            .length()
            / 2.0;
        for k in 0..3 {
            let (a, b) = (corners[k], corners[(k + 1) % 3]);
            let (x, y, z) = (
                0.5 * (a.x() + b.x()),
                0.5 * (a.y() + b.y()),
                0.5 * (a.z() + b.z()),
            );
            sag = sag.max(radius(z) - (x - cx).hypot(y - 5.0));
        }
    }
    (area, sag)
}

const DEFLECTION: f64 = 0.002;

#[test]
fn a_rod_through_a_plate_edge_fuses_whole() {
    for cx in [0.0, 0.4] {
        let overlap = 2.0 * disc_past_zero(1.0, cx);
        let truth = 200.0 + 10.0 * PI - overlap;
        // The window's angular span, where the rod's wall lies over x > 0.
        let span = 2.0 * (-cx).clamp(-1.0, 1.0).acos();
        let wall = 2.0 * PI * 10.0 - 2.0 * span;
        // At cx = 0 turned 1 radian the fuse falls back to a mesh.
        let spins: &[f64] = if cx == 0.0 {
            &[0.0, PI]
        } else {
            &[0.0, 1.0, PI]
        };
        for &spin in spins {
            let label = format!("cx {cx} spin {spin}");
            let (topo, fused) = fuse(false, cx, spin);
            let faces = solid_faces(&topo, fused).unwrap();
            assert!(faces.len() <= 12, "{label}: fell back to a mesh");
            assert!(
                validate_solid(&topo, fused).unwrap().is_valid(),
                "{label}: invalid"
            );
            let mesh = tessellate_solid(&topo, fused, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let volume = solid_volume(&topo, fused, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-3 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            let walls: Vec<_> = faces
                .into_iter()
                .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
                .collect();
            assert_eq!(walls.len(), 1, "{label}: walls");
            let area = face_area(&topo, walls[0], 0.01).unwrap();
            assert!(
                (area - wall).abs() < 1e-9 * wall,
                "{label}: wall area {area}, truth {wall}"
            );
            let (meshed, sag) = wall_mesh(&topo, walls[0], cx, &|_| 1.0);
            assert!(
                (meshed - wall).abs() < 2e-3 * wall,
                "{label}: wall mesh area {meshed}, truth {wall}"
            );
            assert!(
                sag <= DEFLECTION * (1.0 + 1e-6),
                "{label}: wall mesh sags {sag}"
            );
        }
    }
}

#[test]
fn a_cone_through_a_plate_edge_fuses_whole() {
    // The cone's radius at height z (the plate spans z in [0, 2]).
    let radius = |z: f64| 0.06f64.mul_add(-(z + 4.0), 1.2);
    let cone = PI * 10.0 / 3.0 * (1.2f64.powi(2) + 1.2 * 0.6 + 0.6f64.powi(2));
    for cx in [0.0, 0.4] {
        let n = 2000_u32;
        let step = 2.0 / f64::from(n);
        let mut sum = disc_past_zero(radius(0.0), cx) + disc_past_zero(radius(2.0), cx);
        for k in 1..n {
            let z = step * f64::from(k);
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * disc_past_zero(radius(z), cx);
        }
        let truth = 200.0 + cone - sum * step / 3.0;
        for spin in [0.0, 1.0, PI] {
            let label = format!("cx {cx} spin {spin}");
            let (topo, fused) = fuse(true, cx, spin);
            let faces = solid_faces(&topo, fused).unwrap();
            assert!(faces.len() <= 12, "{label}: fell back to a mesh");
            assert!(
                validate_solid(&topo, fused).unwrap().is_valid(),
                "{label}: invalid"
            );
            let mesh = tessellate_solid(&topo, fused, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let volume = solid_volume(&topo, fused, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-3 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            for f in faces {
                if matches!(topo.face(f).unwrap().surface(), FaceSurface::Cone(_)) {
                    let area = face_area(&topo, f, 0.01).unwrap();
                    let (meshed, sag) = wall_mesh(&topo, f, cx, &radius);
                    assert!(
                        (meshed - area).abs() < 2e-3 * area,
                        "{label}: wall area {area}, mesh {meshed}"
                    );
                    assert!(
                        sag <= DEFLECTION * (1.0 + 1e-6),
                        "{label}: wall mesh sags {sag}"
                    );
                }
            }
        }
    }
}
