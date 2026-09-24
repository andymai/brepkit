//! A cylindrical bore drilled sideways into a rod: through and blind, along
//! x (where the hole straddles the rod's seam, with the bore's own seam on
//! the rod's seam or turned away from it) and along y (clear of the seam).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

const ROD_RADIUS: f64 = 1.5;
const ROD_HEIGHT: f64 = 4.0;
const BORE_RADIUS: f64 = 0.3;

enum Axis {
    X { spin: f64 },
    Y,
}

/// The rod minus the bore: the removed volume integrates, over the bore's
/// disc, the length of the bore's chord inside the rod (`length(y)` for the
/// disc point at offset `y` across the bore). `y = r sin θ` keeps the
/// integrand smooth at the disc's edge.
fn truth(length: impl Fn(f64) -> f64) -> f64 {
    const N: u32 = 2000;
    let h = PI / f64::from(N);
    let mut sum = 0.0;
    for k in 0..=N {
        let theta = -FRAC_PI_2 + h * f64::from(k);
        let weight = if k == 0 || k == N {
            1.0
        } else if k % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let chord = 2.0 * BORE_RADIUS * theta.cos();
        sum += weight * chord * BORE_RADIUS * theta.cos() * length(BORE_RADIUS * theta.sin());
    }
    PI * ROD_RADIUS * ROD_RADIUS * ROD_HEIGHT - sum * h / 3.0
}

impl Axis {
    /// A point `along` the bore's axis and `across` it in the rod's
    /// horizontal plane, at height `z`.
    fn point(&self, along: f64, across: f64, z: f64) -> Point3 {
        match self {
            Self::X { .. } => Point3::new(along, across, z),
            Self::Y => Point3::new(across, along, z),
        }
    }
}

/// `carved` lists positions along the bore's axis inside the rod that the
/// bore removes; `kept` lists `(along, across, z)` points of rod material.
fn drill(
    axis: &Axis,
    start: f64,
    length: f64,
    expected_volume: f64,
    planes: usize,
    carved: &[f64],
    kept: &[(f64, f64, f64)],
) {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, ROD_RADIUS, ROD_HEIGHT).unwrap();
    let bore = make_cylinder(&mut topo, BORE_RADIUS, length).unwrap();
    let place = match axis {
        Axis::X { spin } => {
            Mat4::translation(start, 0.0, 2.0)
                * Mat4::rotation_y(FRAC_PI_2)
                * Mat4::rotation_z(*spin)
        }
        Axis::Y => Mat4::translation(0.0, start, 2.0) * Mat4::rotation_x(-FRAC_PI_2),
    };
    transform_solid(&mut topo, bore, &place).unwrap();
    let result = boolean(&mut topo, BooleanOp::Cut, rod, bore).unwrap();

    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mut census: BTreeMap<&str, usize> = BTreeMap::new();
    for face in solid_faces(&topo, result).unwrap() {
        *census
            .entry(topo.face(face).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    assert_eq!(
        census,
        BTreeMap::from([("cylinder", 2), ("plane", planes)]),
        "the cut stays analytic"
    );

    let classify =
        |p: Point3| classify_point(&topo, result, p, &ClassifyOptions::default()).unwrap();
    for &along in carved {
        let p = axis.point(along, 0.0, 2.0);
        assert_eq!(classify(p), PointClassification::Outside, "{p:?} is carved");
    }
    for &(along, across, z) in kept {
        let p = axis.point(along, across, z);
        assert_eq!(classify(p), PointClassification::Inside, "{p:?} is kept");
    }

    let exact = solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (exact - expected_volume).abs() < 1e-7 * expected_volume,
        "volume {exact}, truth {expected_volume}"
    );
    let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
    assert!(is_watertight(&mesh), "open or non-manifold mesh");
    let meshed = oriented_solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (meshed - expected_volume).abs() < 2e-3 * expected_volume,
        "mesh volume {meshed}, truth {expected_volume}"
    );
}

fn through(axis: &Axis) {
    let volume = truth(|y| 2.0 * (ROD_RADIUS * ROD_RADIUS - y * y).sqrt());
    drill(
        axis,
        -3.0,
        6.0,
        volume,
        2,
        &[-1.4, -0.6, 0.0, 0.6, 1.4],
        &[
            (0.0, 0.8, 2.0),
            (0.0, -0.8, 2.0),
            (1.2, 0.0, 1.0),
            (-1.2, 0.0, 3.0),
        ],
    );
}

/// The bore starts inside the rod at 0.8 from its axis and leaves through
/// the wall.
fn blind(axis: &Axis) {
    let volume = truth(|y| (ROD_RADIUS * ROD_RADIUS - y * y).sqrt() - 0.8);
    drill(
        axis,
        0.8,
        2.0,
        volume,
        3,
        &[0.9, 1.2, 1.45],
        &[
            (0.5, 0.0, 2.0),
            (-1.2, 0.0, 2.0),
            (1.2, 0.5, 2.0),
            (1.2, 0.0, 1.5),
        ],
    );
}

#[test]
fn through_bore_across_the_seam() {
    through(&Axis::X { spin: 0.0 });
}

#[test]
fn through_bore_across_the_seam_turned() {
    through(&Axis::X { spin: FRAC_PI_4 });
}

#[test]
fn through_bore_clear_of_the_seam() {
    through(&Axis::Y);
}

#[test]
fn blind_bore_across_the_seam() {
    blind(&Axis::X { spin: 0.0 });
}

#[test]
fn blind_bore_across_the_seam_turned() {
    blind(&Axis::X { spin: FRAC_PI_4 });
}

#[test]
fn blind_bore_clear_of_the_seam() {
    blind(&Axis::Y);
}
