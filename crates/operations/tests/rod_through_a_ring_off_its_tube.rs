//! A rod of radius 0.6 along `y` through a ring (4, 1.5) at `x = 0.5`,
//! `z = 1`: off the tube's centre and through its top, so it crosses the
//! tube on each side of the ring and leaves it through its upper surface.
//! The ring is dented first by a unit box on the tube's top across the ring
//! from the rod, which leaves its face the whole ring less the dent.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_cylinder, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const POSES: [&str; 3] = ["upright", "turned", "mirrored"];

fn turn() -> Mat4 {
    Mat4::translation(3.0, -2.0, 1.5) * Mat4::rotation_x(0.61) * Mat4::rotation_z(-0.43)
}

fn mirror_plane() -> (Point3, Vec3) {
    (Point3::new(2.0, -1.0, 0.5), Vec3::new(0.3, 1.0, 0.2))
}

fn posed(topo: &mut Topology, solid: SolidId, pose: &str) -> SolidId {
    match pose {
        "turned" => {
            transform_solid(topo, solid, &turn()).unwrap();
            solid
        }
        "mirrored" => {
            let (at, normal) = mirror_plane();
            mirror(topo, solid, at, normal).unwrap()
        }
        _ => solid,
    }
}

fn placed(p: Point3, pose: &str) -> Point3 {
    match pose {
        "turned" => turn().mul_point(p),
        "mirrored" => {
            let (at, normal) = mirror_plane();
            let unit = normal.normalize().unwrap();
            p - unit * (2.0 * (p - at).dot(unit))
        }
        _ => p,
    }
}

/// The volume the rod shares with the ring: over the rod's disc in the
/// plane `y = 0`, the length along `y` of the ring at `(x, z)`, where the
/// ring's tube spans the distances from the axis `4 ± √(1.5² − z²)`.
fn rod_in_ring() -> f64 {
    const N: u32 = 800;
    let (cx, cz, r) = (0.5, 1.0, 0.6);
    let mut sum = 0.0;
    for i in 0..N {
        let rho = r * ((f64::from(i) + 0.5) / f64::from(N)).sqrt();
        for j in 0..N {
            let theta = 2.0 * PI * (f64::from(j) + 0.5) / f64::from(N);
            let (x, z) = (rho.mul_add(theta.cos(), cx), rho.mul_add(theta.sin(), cz));
            let reach = 2.25 - z * z;
            if reach <= 0.0 {
                continue;
            }
            let w = reach.sqrt();
            let far = (4.0 + w).powi(2) - x * x;
            let near = ((4.0 - w).powi(2) - x * x).max(0.0);
            sum += 2.0 * (far.sqrt() - near.sqrt());
        }
    }
    sum * PI * r * r / f64::from(N * N)
}

/// The volume the dent, the box from `(-4.5, -0.5, 1)` to `(-3.5, 0.5, 2)`,
/// shares with the ring: over the box's base, the height of the tube above
/// `z = 1` there, the tube's top lying at `√(1.5² − (ρ − 4)²)` below the
/// box's top at `z = 2`.
fn dent_in_ring() -> f64 {
    const N: u32 = 2000;
    let step = 1.0 / f64::from(N);
    let mut sum = 0.0;
    for i in 0..N {
        let x = step.mul_add(f64::from(i) + 0.5, -4.5);
        for j in 0..N {
            let y = step.mul_add(f64::from(j) + 0.5, -0.5);
            let off = x.hypot(y) - 4.0;
            let reach = off.mul_add(-off, 2.25);
            if reach > 1.0 {
                sum += reach.sqrt() - 1.0;
            }
        }
    }
    sum * step * step
}

/// The rod's cut, after the dent's where `dented`: exact, valid, watertight,
/// with the faces `census`, at the ring's volume less the pieces'
/// integrals, carved where the rod (and the dent) run and kept round them.
fn cut_the_rod(dented: bool, census: &[(&str, usize)]) {
    let dent_volume = if dented { dent_in_ring() } else { 0.0 };
    let truth = 2.0 * PI * PI * 4.0 * 1.5 * 1.5 - dent_volume - rod_in_ring();
    let exact = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    for pose in POSES {
        let mut topo = Topology::new();
        let mut ring = make_torus(&mut topo, 4.0, 1.5, 32).unwrap();
        let dent = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        transform_solid(&mut topo, dent, &Mat4::translation(-4.5, -0.5, 1.0)).unwrap();
        let rod = make_cylinder(&mut topo, 0.6, 20.0).unwrap();
        transform_solid(
            &mut topo,
            rod,
            &(Mat4::translation(0.5, 10.0, 1.0) * Mat4::rotation_x(FRAC_PI_2)),
        )
        .unwrap();
        ring = posed(&mut topo, ring, pose);
        let (dent, rod) = (posed(&mut topo, dent, pose), posed(&mut topo, rod, pose));
        if dented {
            ring = boolean_with_options(&mut topo, BooleanOp::Cut, ring, dent, exact)
                .map_err(|e| format!("{pose}: {e}"))
                .unwrap();
        }
        let result = boolean_with_options(&mut topo, BooleanOp::Cut, ring, rod, exact)
            .map_err(|e| format!("{pose}: {e}"))
            .unwrap();
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{pose}: {:?}", report.issues);
        assert!(
            is_watertight(&tessellate_solid(&topo, result, 0.01).unwrap()),
            "{pose}: open mesh"
        );
        let mut faces: BTreeMap<&str, usize> = BTreeMap::new();
        for face in solid_faces(&topo, result).unwrap() {
            *faces
                .entry(topo.face(face).unwrap().surface().type_tag())
                .or_default() += 1;
        }
        assert_eq!(faces, census.iter().copied().collect(), "{pose}: faces");
        let volume = solid_volume(&topo, result, 0.001).unwrap();
        assert!(
            (volume - truth).abs() < 1e-6 * truth,
            "{pose}: volume {volume}, truth {truth}"
        );
        let holds = |p: Point3| classify_point(&topo, result, placed(p, pose), 0.01, 1e-7).unwrap();
        let mut carved = vec![Point3::new(0.5, 4.0, 1.0), Point3::new(0.5, -4.0, 0.8)];
        if dented {
            carved.push(Point3::new(-4.0, 0.0, 1.2));
        }
        for p in carved {
            assert_eq!(
                holds(p),
                PointClassification::Outside,
                "{pose}: {p:?} is carved"
            );
        }
        for p in [
            Point3::new(-0.5, 4.0, -0.5),
            Point3::new(0.5, -4.0, -0.5),
            Point3::new(-4.0, 0.0, 0.0),
        ] {
            assert_eq!(
                holds(p),
                PointClassification::Inside,
                "{pose}: {p:?} is kept"
            );
        }
    }
}

#[test]
fn a_rod_through_a_rings_tube_off_its_centre_cuts_exactly() {
    cut_the_rod(false, &[("cylinder", 2), ("torus", 1)]);
}

#[test]
fn a_rod_through_a_dented_rings_tube_off_its_centre_cuts_exactly() {
    cut_the_rod(true, &[("cylinder", 2), ("plane", 5), ("torus", 1)]);
}
