//! Parity robustness corpus — curved-primitive booleans (cylinder, cone,
//! sphere, torus against boxes and each other).
//!
//! Oracles are exact analytic volumes (πr²h, ⁴⁄₃πr³, ⅓πr²h, 2π²Rr²). Most
//! cases use *containment* configurations: a curved primitive fully inside a
//! box, so the intersection is the primitive's volume, the union is the box's,
//! and the subtraction is a box with a curved cavity (an inner shell). These
//! isolate the curved boolean + curved-cavity paths from intersection-curve
//! noise. A few interpenetration probes (coaxial cylinders, sphere∩box) target
//! the same-domain machinery prior work flagged as fragile.
//!
//! Tolerance is looser than the planar corpus because curved area/volume go
//! through quadrature; it is still tight enough that a mangled boolean misses.
//! `Gap` rows are calibrated from observed behavior — see the scoreboard.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::missing_panics_doc)]

mod parity_support;

use std::f64::consts::PI;

use parity_support::{Case, Expect, Op, Oracle, Prim, Xf, run_corpus};

const NONE: &[Xf] = &[];
const BOX3: Prim = Prim::Box(3.0, 3.0, 3.0);
/// Relative tolerance for curved volumes. Covers quadrature error and primitive
/// faceting (a built cylinder's volume runs ~0.2% under πr²h). Real boolean
/// failures here miss by 2%–100%, so this stays well clear of them.
const CTOL: f64 = 5e-3;

// Centering transforms that place a primitive fully inside BOX3 ([0,3]³).
const C_AXIS: &[Xf] = &[Xf::Translate(1.5, 1.5, 0.5)]; // base-at-z0 primitives (cyl/cone)
const C_CTR: &[Xf] = &[Xf::Translate(1.5, 1.5, 1.5)]; // centered primitives (sphere)

#[test]
fn curved_boolean_corpus() {
    #[rustfmt::skip]
    let cases: &[Case] = &[
        // ── Cylinder (r=0.5,h=2, V=0.5π) fully inside a 3-cube ──────────
        Case { name: "cyl_in_box_common_vol", a: (BOX3, NONE), b: (Prim::Cyl(0.5,2.0), C_AXIS), op: Op::Common, oracle: Oracle::Volume(0.5 * PI),  tol: CTOL, expect: Expect::Pass },
        Case { name: "cyl_in_box_fuse_vol",   a: (BOX3, NONE), b: (Prim::Cyl(0.5,2.0), C_AXIS), op: Op::Fuse,   oracle: Oracle::Volume(27.0),         tol: CTOL, expect: Expect::Pass },
        Case { name: "cyl_in_box_cut_vol",    a: (BOX3, NONE), b: (Prim::Cyl(0.5,2.0), C_AXIS), op: Op::Cut,    oracle: Oracle::Volume(27.0 - 0.5 * PI), tol: CTOL, expect: Expect::Pass },

        // ── Sphere (r=0.5, V=π/6) fully inside a 3-cube ─────────────────
        Case { name: "sphere_in_box_common_vol", a: (BOX3, NONE), b: (Prim::Sphere(0.5), C_CTR), op: Op::Common, oracle: Oracle::Volume(PI / 6.0),  tol: CTOL, expect: Expect::Pass },
        Case { name: "sphere_in_box_fuse_vol",   a: (BOX3, NONE), b: (Prim::Sphere(0.5), C_CTR), op: Op::Fuse,   oracle: Oracle::Volume(27.0),         tol: CTOL, expect: Expect::Pass },
        Case { name: "sphere_in_box_cut_vol",    a: (BOX3, NONE), b: (Prim::Sphere(0.5), C_CTR), op: Op::Cut,    oracle: Oracle::Volume(27.0 - PI / 6.0), tol: CTOL, expect: Expect::Pass },

        // ── Cone (rb=0.5,rt=0,h=2, V=π/6) fully inside a 3-cube ─────────
        Case { name: "cone_in_box_common_vol", a: (BOX3, NONE), b: (Prim::Cone(0.5,0.0,2.0), C_AXIS), op: Op::Common, oracle: Oracle::Volume(PI / 6.0),  tol: CTOL, expect: Expect::Pass },
        Case { name: "cone_in_box_fuse_vol",   a: (BOX3, NONE), b: (Prim::Cone(0.5,0.0,2.0), C_AXIS), op: Op::Fuse,   oracle: Oracle::Volume(27.0),         tol: CTOL, expect: Expect::Pass },
        Case { name: "cone_in_box_cut_vol",    a: (BOX3, NONE), b: (Prim::Cone(0.5,0.0,2.0), C_AXIS), op: Op::Cut,    oracle: Oracle::Volume(27.0 - PI / 6.0), tol: CTOL, expect: Expect::Gap("cone cut: tool not subtracted (full box remains)") },

        // ── Torus (R=1,r=0.3, V=0.18π²) fully inside a flat 3×3×1 box ───
        // Cut yields a box with a toroidal (genus-1) cavity.
        Case { name: "torus_in_box_common_vol", a: (Prim::Box(3.0,3.0,1.0), &[Xf::Translate(-1.5,-1.5,-0.5)]), b: (Prim::Torus(1.0,0.3), NONE), op: Op::Common, oracle: Oracle::Volume(0.18 * PI * PI), tol: CTOL, expect: Expect::Pass },
        Case { name: "torus_in_box_fuse_vol",   a: (Prim::Box(3.0,3.0,1.0), &[Xf::Translate(-1.5,-1.5,-0.5)]), b: (Prim::Torus(1.0,0.3), NONE), op: Op::Fuse,   oracle: Oracle::Volume(9.0),         tol: CTOL, expect: Expect::Pass },
        Case { name: "torus_in_box_cut_vol",    a: (Prim::Box(3.0,3.0,1.0), &[Xf::Translate(-1.5,-1.5,-0.5)]), b: (Prim::Torus(1.0,0.3), NONE), op: Op::Cut,    oracle: Oracle::Volume(9.0 - 0.18 * PI * PI), tol: CTOL, expect: Expect::Gap("torus cut: tool not subtracted (full box remains)") },

        // ── Interpenetration probes (same-domain machinery) ────────────
        // Coaxial cylinders overlapping along the axis: a z∈[0,2], b z∈[1,3].
        // Union is a cyl r1 h3 (V=3π); intersection a cyl r1 h1 (V=π).
        Case { name: "coaxial_cyl_fuse_vol",   a: (Prim::Cyl(1.0,2.0), NONE), b: (Prim::Cyl(1.0,2.0), &[Xf::Translate(0.0,0.0,1.0)]), op: Op::Fuse,   oracle: Oracle::Volume(3.0 * PI), tol: CTOL, expect: Expect::Pass },
        Case { name: "coaxial_cyl_common_vol", a: (Prim::Cyl(1.0,2.0), NONE), b: (Prim::Cyl(1.0,2.0), &[Xf::Translate(0.0,0.0,1.0)]), op: Op::Common, oracle: Oracle::Volume(PI), tol: CTOL, expect: Expect::Pass },
        // Unit sphere (centered) partially overlapping a unit cube at +octant.
        // Overlap is ⅛ of the ball (V=π/6); union V = ⁴⁄₃π + 1 − π/6.
        Case { name: "sphere_box_partial_common_vol", a: (Prim::Sphere(1.0), NONE), b: (Prim::Box(1.0,1.0,1.0), NONE), op: Op::Common, oracle: Oracle::Volume(PI / 6.0), tol: CTOL, expect: Expect::Gap("sphere/box partial intersect: volume ~46% low") },
        Case { name: "sphere_box_partial_fuse_vol",   a: (Prim::Sphere(1.0), NONE), b: (Prim::Box(1.0,1.0,1.0), NONE), op: Op::Fuse,   oracle: Oracle::Volume(4.0 / 3.0 * PI + 1.0 - PI / 6.0), tol: CTOL, expect: Expect::Gap("sphere/box partial fuse: volume ~2.2% low") },
    ];

    run_corpus(cases);
}
