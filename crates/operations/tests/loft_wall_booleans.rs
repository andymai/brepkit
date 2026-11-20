//! Booleans against a lofted wall: one NURBS face round a vertical axis,
//! bounded by its two rim circles and a seam, as a file writes a loft, and
//! bulging out past both rims between them.
//!
//! The wall's edges hold none of its bulge, so a box built from them missed
//! it and the faces meeting the bulge were never intersected with it; the
//! classifier read the wall as the flat polygon of its rims; a plane's
//! section across the wall was trimmed only against straight edges, so a
//! disc's faces left their sections running on past the disc's rim; and a
//! piece of the disc's wall between two sections running rim to rim, with
//! no seam line in its boundary, was read as a hole; and the disc wall's far
//! rim, walked against its curve, was laid out a turn off its near one.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::surfaces::CylindricalSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// The profile's radius and height at its three control points: the wall
/// leaves its rims at radius 10 and swells to 20 halfway up.
const PROFILE: [(f64, f64); 3] = [(10.0, 0.0), (30.0, 10.0), (10.0, 20.0)];

/// The bulging wall turned about the z axis: a full rational quadratic
/// circle in `u` (nine control points), the quadratic profile in `v`.
fn wall_surface() -> NurbsSurface {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let ring = [
        (1.0, 0.0, 1.0),
        (1.0, 1.0, h),
        (0.0, 1.0, 1.0),
        (-1.0, 1.0, h),
        (-1.0, 0.0, 1.0),
        (-1.0, -1.0, h),
        (0.0, -1.0, 1.0),
        (1.0, -1.0, h),
        (1.0, 0.0, 1.0),
    ];
    let cps = ring
        .iter()
        .map(|&(x, y, _)| {
            PROFILE
                .iter()
                .map(|&(r, z)| Point3::new(r * x, r * y, z))
                .collect()
        })
        .collect();
    let weights = ring.iter().map(|&(_, _, w)| vec![w; 3]).collect();
    NurbsSurface::new(
        2,
        2,
        vec![
            0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
        ],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        cps,
        weights,
    )
    .unwrap()
}

/// The lofted solid: the wall and its two flat caps.
fn lofted(topo: &mut Topology) -> SolidId {
    let (r0, z0) = PROFILE[0];
    let (r1, z1) = PROFILE[2];
    let vb = topo.add_vertex(Vertex::new(Point3::new(r0, 0.0, z0), 1e-7));
    let vt = topo.add_vertex(Vertex::new(Point3::new(r1, 0.0, z1), 1e-7));
    let rim = |z: f64, r: f64| {
        Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, z),
            Vec3::new(0.0, 0.0, 1.0),
            r,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap()
    };
    let bottom = topo.add_edge(Edge::new(vb, vb, EdgeCurve::Circle(rim(z0, r0))));
    let top = topo.add_edge(Edge::new(vt, vt, EdgeCurve::Circle(rim(z1, r1))));
    let profile = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        PROFILE
            .iter()
            .map(|&(r, z)| Point3::new(r, 0.0, z))
            .collect(),
        vec![1.0; 3],
    )
    .unwrap();
    let seam = topo.add_edge(Edge::new(vb, vt, EdgeCurve::NurbsCurve(profile)));
    let wire = |topo: &mut Topology, edges: &[(_, bool)]| {
        let w = Wire::new(
            edges
                .iter()
                .map(|&(e, fwd)| OrientedEdge::new(e, fwd))
                .collect(),
            true,
        )
        .unwrap();
        topo.add_wire(w)
    };
    let wall_wire = wire(
        topo,
        &[(bottom, true), (seam, true), (top, false), (seam, false)],
    );
    let wall = topo.add_face(Face::new(
        wall_wire,
        vec![],
        FaceSurface::Nurbs(wall_surface()),
    ));
    let bottom_wire = wire(topo, &[(bottom, false)]);
    let floor = topo.add_face(Face::new(
        bottom_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: -z0,
        },
    ));
    let top_wire = wire(topo, &[(top, true)]);
    let lid = topo.add_face(Face::new(
        top_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z1,
        },
    ));
    let shell = topo.add_shell(Shell::new(vec![wall, floor, lid]).unwrap());
    topo.add_solid(Solid::new(shell, vec![]))
}

/// A disc of radius `r` and thickness `t` about the x axis, centred at
/// `centre`: its wall's seam at the surface's u = 3pi/2, its rim circles
/// starting there, as the STEP reader starts a closed circle at its vertex.
fn disc(topo: &mut Topology, centre: Point3, r: f64, t: f64) -> SolidId {
    let axis = Vec3::new(1.0, 0.0, 0.0);
    let base = centre - axis * (0.5 * t);
    let cyl = CylindricalSurface::new(base, axis, r).unwrap();
    let seam_u = 1.5 * std::f64::consts::PI;
    let vb = topo.add_vertex(Vertex::new(cyl.evaluate(seam_u, 0.0), 1e-7));
    let vt = topo.add_vertex(Vertex::new(cyl.evaluate(seam_u, t), 1e-7));
    let rim = |v: f64| {
        let c = base + axis * v;
        Circle3D::new_with_ref(c, axis, r, cyl.evaluate(seam_u, v) - c).unwrap()
    };
    let near = topo.add_edge(Edge::new(vb, vb, EdgeCurve::Circle(rim(0.0))));
    let far = topo.add_edge(Edge::new(vt, vt, EdgeCurve::Circle(rim(t))));
    let seam = topo.add_edge(Edge::new(vb, vt, EdgeCurve::Line));
    let wire = |topo: &mut Topology, edges: &[(_, bool)]| {
        let w = Wire::new(
            edges
                .iter()
                .map(|&(e, fwd)| OrientedEdge::new(e, fwd))
                .collect(),
            true,
        )
        .unwrap();
        topo.add_wire(w)
    };
    let wall_wire = wire(
        topo,
        &[(far, false), (seam, false), (near, true), (seam, true)],
    );
    let wall = topo.add_face(Face::new(wall_wire, vec![], FaceSurface::Cylinder(cyl)));
    let near_wire = wire(topo, &[(near, false)]);
    let near_face = topo.add_face(Face::new(
        near_wire,
        vec![],
        FaceSurface::Plane {
            normal: -axis,
            d: -axis.dot(base - Point3::new(0.0, 0.0, 0.0)),
        },
    ));
    let far_wire = wire(topo, &[(far, true)]);
    let far_face = topo.add_face(Face::new(
        far_wire,
        vec![],
        FaceSurface::Plane {
            normal: axis,
            d: axis.dot(base - Point3::new(0.0, 0.0, 0.0)) + t,
        },
    ));
    let shell = topo.add_shell(Shell::new(vec![wall, near_face, far_face]).unwrap());
    topo.add_solid(Solid::new(shell, vec![]))
}

/// The wall's radius at height z: r(s) = 10 + 40 s (1 - s), z = 20 s.
fn wall_radius(z: f64) -> f64 {
    let s = z / 20.0;
    40.0f64.mul_add(s * (1.0 - s), 10.0)
}

#[test]
fn a_disc_through_a_loft_wall_intersects_exactly() {
    let mut topo = Topology::new();
    let loft = lofted(&mut topo);
    // A disc 2 thick about the x axis, radius 9, centred at y = 12 at the
    // bulge's height: it reaches y = 21 past the wall (radius 20 there) and
    // pokes through it again near its top and bottom.
    let (r, t, yc, zc) = (9.0, 2.0, 12.0, 10.0);
    let d = disc(&mut topo, Point3::new(0.0, yc, zc), r, t);
    let v_disc = solid_volume(&topo, d, 0.001).unwrap();
    let exact_disc = std::f64::consts::PI * r * r * t;
    assert!(
        (v_disc - exact_disc).abs() < 1e-3 * exact_disc,
        "disc volume {v_disc}"
    );

    let res = boolean(&mut topo, BooleanOp::Intersect, d, loft).unwrap();
    let faces = solid_faces(&topo, res).unwrap();
    let nurbs = faces
        .iter()
        .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_)))
        .count();
    assert!(nurbs >= 1, "the intersection fell back to a mesh");

    // For each (x, z), y runs over the disc's chord, |y - yc| <= sqrt(r^2 -
    // (z - zc)^2), within the wall's, |y| <= sqrt(wall(z)^2 - x^2).
    let n = 400;
    let (dx, dz) = (t / f64::from(n), 2.0 * r / f64::from(n));
    let exact: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| {
            let x = (f64::from(i) + 0.5).mul_add(dx, -0.5 * t);
            let z = (f64::from(j) + 0.5).mul_add(dz, zc - r);
            let half = (r * r - (z - zc).powi(2)).max(0.0).sqrt();
            let wall = (wall_radius(z).powi(2) - x * x).max(0.0).sqrt();
            ((yc + half).min(wall) - (yc - half).max(-wall)).max(0.0) * dx * dz
        })
        .sum();
    let v = solid_volume(&topo, res, 0.001).unwrap();
    assert!(
        (v - exact).abs() < 2e-3 * exact,
        "volume {v}, expected {exact}"
    );
}

#[test]
fn a_box_at_a_loft_bulge_intersects_exactly() {
    let mut topo = Topology::new();
    let loft = lofted(&mut topo);
    // The wall's radius at height z: r(s) = 10 + 40 s (1 - s), z = 20 s.
    let radius = |z: f64| {
        let s = z / 20.0;
        40.0f64.mul_add(s * (1.0 - s), 10.0)
    };
    // The loft's volume, pi r^2 integrated over its height.
    let exact_loft = {
        let n = 2000;
        let dz = 20.0 / f64::from(n);
        (0..n)
            .map(|k| {
                let z = (f64::from(k) + 0.5) * dz;
                std::f64::consts::PI * radius(z).powi(2) * dz
            })
            .sum::<f64>()
    };
    let v_loft = solid_volume(&topo, loft, 0.001).unwrap();
    assert!(
        (v_loft - exact_loft).abs() < 1e-3 * exact_loft,
        "loft volume {v_loft}, expected {exact_loft}"
    );

    // A box from y = 14 to 30, |x| <= 3, z from 8 to 12: beyond both rims
    // (radius 10), meeting the wall where it swells to 19.2 and more, clear
    // of its seam (at y = 0, x > 0).
    let block = make_box(&mut topo, 6.0, 16.0, 4.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(-3.0, 14.0, 8.0)).unwrap();

    let res = boolean(&mut topo, BooleanOp::Intersect, block, loft).unwrap();
    let faces = solid_faces(&topo, res).unwrap();
    let nurbs = faces
        .iter()
        .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_)))
        .count();
    assert_eq!(nurbs, 1, "the intersection fell back to a mesh");
    assert_eq!(
        faces.len(),
        6,
        "five box faces cut by the wall, and the wall"
    );

    // The part of the box inside the wall: for each (x, z), y runs from 14
    // to sqrt(r(z)^2 - x^2).
    let n = 400;
    let (dx, dz) = (6.0 / f64::from(n), 4.0 / f64::from(n));
    let exact: f64 = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| {
            let x = (f64::from(i) + 0.5).mul_add(dx, -3.0);
            let z = (f64::from(j) + 0.5).mul_add(dz, 8.0);
            ((radius(z).powi(2) - x * x).sqrt() - 14.0).max(0.0) * dx * dz
        })
        .sum();
    let v = solid_volume(&topo, res, 0.001).unwrap();
    assert!(
        (v - exact).abs() < 1e-3 * exact,
        "volume {v}, expected {exact}"
    );
}
