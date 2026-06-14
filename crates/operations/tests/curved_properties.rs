//! Regression guard for `brepkit_check` property integration on curved
//! analytic surfaces.
//!
//! `solid_volume` / `solid_area` integrate each face numerically via
//! `integrate_face`. Full-revolution and closed/pole faces (cone, sphere,
//! torus) previously broke the polygon-trimmed quadrature — a cone integrated
//! ~5x low, a torus to zero, a sphere errored on its poles — because the
//! projected UV boundary collapses on the apex/pole/seam. These tests build the
//! actual primitives and assert the numerical result against the analytic
//! closed form, so a regression in the integrator is caught directly (the
//! existing `check` unit tests only exercise the closed-form formulas).

#![allow(clippy::unwrap_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_check::properties::{PropertiesOptions, solid_area, solid_volume};
use brepkit_operations::primitives::{make_cone, make_cylinder, make_sphere, make_torus};
use brepkit_topology::Topology;

fn assert_close(got: f64, expected: f64, rel_tol: f64, what: &str) {
    let rel = (got - expected).abs() / expected.abs().max(1.0);
    assert!(
        rel <= rel_tol,
        "{what}: got {got:.6}, expected {expected:.6} (rel {rel:.2e} > {rel_tol:.0e})"
    );
}

#[test]
fn curved_primitive_volumes_match_analytic() {
    let opt = PropertiesOptions::default();
    let mut t = Topology::new();

    // Cone (r=0.5, h=2): V = pi r^2 h / 3 = pi/6. Fully analytic.
    let cone = make_cone(&mut t, 0.5, 0.0, 2.0).unwrap();
    assert_close(
        solid_volume(&t, cone, &opt).unwrap(),
        PI / 6.0,
        1e-3,
        "cone volume",
    );

    // Sphere (r=0.5): V = 4/3 pi r^3 = pi/6. Two hemisphere faces.
    let sphere = make_sphere(&mut t, 0.5, 64).unwrap();
    assert_close(
        solid_volume(&t, sphere, &opt).unwrap(),
        PI / 6.0,
        1e-3,
        "sphere volume",
    );

    // Torus (R=1, r=0.3): V = 2 pi^2 R r^2. Single doubly-periodic face.
    let torus = make_torus(&mut t, 1.0, 0.3, 64).unwrap();
    assert_close(
        solid_volume(&t, torus, &opt).unwrap(),
        2.0 * PI * PI * 1.0 * 0.09,
        2e-3,
        "torus volume",
    );

    // Cylinder (r=0.5, h=2): V = pi r^2 h. Lateral is exact; the faceted caps
    // leave a small (<1%) deficit, so the tolerance is looser here.
    let cyl = make_cylinder(&mut t, 0.5, 2.0).unwrap();
    assert_close(
        solid_volume(&t, cyl, &opt).unwrap(),
        PI * 0.25 * 2.0,
        1e-2,
        "cylinder volume",
    );
}

#[test]
fn curved_primitive_areas_match_analytic() {
    let opt = PropertiesOptions::default();
    let mut t = Topology::new();

    // Sphere surface area = 4 pi r^2.
    let sphere = make_sphere(&mut t, 0.5, 64).unwrap();
    assert_close(
        solid_area(&t, sphere, &opt).unwrap(),
        4.0 * PI * 0.25,
        2e-3,
        "sphere area",
    );

    // Torus surface area = 4 pi^2 R r.
    let torus = make_torus(&mut t, 1.0, 0.3, 64).unwrap();
    assert_close(
        solid_area(&t, torus, &opt).unwrap(),
        4.0 * PI * PI * 1.0 * 0.3,
        2e-3,
        "torus area",
    );
}
