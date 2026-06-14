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
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere, make_torus};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

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

/// Cutting a strictly-contained tool from a blank yields a hollow solid whose
/// cavity is a topologically-faithful copy of the tool: exactly one inner
/// shell, and `Euler(result) == Euler(blank) + Euler(tool)`. A per-face cavity
/// copy duplicates shared boundary edges and breaks this (the box case gave
/// Euler 8 instead of 4) while leaving the per-face volume correct — so this
/// guards the topology that the metric oracles in the parity corpus cannot.
#[test]
fn contained_cut_cavity_topology_is_faithful() {
    use brepkit_math::mat::Mat4;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::euler_characteristic;

    type ToolBuilder = fn(&mut Topology) -> SolidId;
    let cases: [(&str, ToolBuilder); 5] = [
        ("cone", |t| {
            let c = make_cone(t, 0.5, 0.0, 2.0).unwrap();
            transform_solid(t, c, &Mat4::translation(1.5, 1.5, 0.5)).unwrap();
            c
        }),
        ("cylinder", |t| {
            let c = make_cylinder(t, 0.5, 2.0).unwrap();
            transform_solid(t, c, &Mat4::translation(1.5, 1.5, 0.5)).unwrap();
            c
        }),
        ("sphere", |t| {
            let c = make_sphere(t, 0.5, 32).unwrap();
            transform_solid(t, c, &Mat4::translation(1.5, 1.5, 1.5)).unwrap();
            c
        }),
        ("box", |t| {
            let c = make_box(t, 1.0, 1.0, 1.0).unwrap();
            transform_solid(t, c, &Mat4::translation(1.0, 1.0, 1.0)).unwrap();
            c
        }),
        ("torus", |t| {
            let c = make_torus(t, 0.6, 0.2, 32).unwrap();
            transform_solid(t, c, &Mat4::translation(1.5, 1.5, 1.5)).unwrap();
            c
        }),
    ];

    for (label, build_tool) in cases {
        // Standalone Euler of the tool (genus-agnostic expected value).
        let mut tt = Topology::new();
        let standalone_tool = build_tool(&mut tt);
        let tool_euler = euler_characteristic(&tt, standalone_tool).unwrap();

        let mut topo = Topology::new();
        let blank = make_box(&mut topo, 3.0, 3.0, 3.0).unwrap();
        let blank_euler = euler_characteristic(&topo, blank).unwrap();
        let tool = build_tool(&mut topo);
        let result = boolean(&mut topo, BooleanOp::Cut, blank, tool).unwrap();

        let inner = topo.solid(result).unwrap().inner_shells().len();
        assert_eq!(
            inner, 1,
            "{label}: expected exactly one cavity shell, got {inner}"
        );

        let euler = euler_characteristic(&topo, result).unwrap();
        assert_eq!(
            euler,
            blank_euler + tool_euler,
            "{label}: cavity Euler {euler} != blank {blank_euler} + tool {tool_euler}"
        );
    }
}
