//! Parity robustness corpus — all-planar boolean cases (box/box, box/prism).
//!
//! Every case carries an exact analytic oracle (surface area or volume), so
//! these assert to near-machine precision and form a tight regression net for
//! the configurations where boolean engines historically break: coincident
//! faces, full containment (hollow results), and edge/corner contact producing
//! non-convex solids.
//!
//! Oracle values are derived purely from the geometry (a box's area is the sum
//! of its face areas; a hollow cut's volume is outer minus cavity). Each case
//! is tagged `Pass` (asserted) or `Gap` (a documented robustness gap, allowed
//! to miss). See `parity_support` for the harness and scoreboard semantics.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::missing_panics_doc)]

mod parity_support;

use parity_support::{Case, Expect, Op, Oracle, Prim, Xf, run_corpus};

const NONE: &[Xf] = &[];
const HALF_X: &[Xf] = &[Xf::Translate(0.5, 0.0, 0.0)];
const STACK_Z: &[Xf] = &[Xf::Translate(0.0, 0.0, 1.0)];
const INSET: &[Xf] = &[Xf::Translate(0.5, 0.5, 0.5)];
const CORNER: &[Xf] = &[Xf::Translate(0.5, 0.5, 0.0)];
/// Unit square in XY, wound counter-clockwise (outward normal +Z).
const UNIT_SQUARE: &[(f64, f64)] = &[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];

const CUBE: Prim = Prim::Box(1.0, 1.0, 1.0);

#[test]
fn planar_boolean_corpus() {
    #[rustfmt::skip]
    let cases: &[Case] = &[
        // ── Family A: two identical unit cubes ──────────────────────────
        // Union/intersection collapse to the same cube; either subtraction
        // is empty. The canonical coincident-everything degenerate.
        Case { name: "identical_cubes_fuse_area", a: (CUBE, NONE), b: (CUBE, NONE), op: Op::Fuse, oracle: Oracle::Area(6.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "identical_cubes_fuse_volume", a: (CUBE, NONE), b: (CUBE, NONE), op: Op::Fuse, oracle: Oracle::Volume(1.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "identical_cubes_common_area", a: (CUBE, NONE), b: (CUBE, NONE), op: Op::Common, oracle: Oracle::Area(6.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "identical_cubes_cut_empty", a: (CUBE, NONE), b: (CUBE, NONE), op: Op::Cut, oracle: Oracle::Empty, tol: 0.0, expect: Expect::Pass },
        Case { name: "identical_cubes_cutrev_empty", a: (CUBE, NONE), b: (CUBE, NONE), op: Op::CutRev, oracle: Oracle::Empty, tol: 0.0, expect: Expect::Pass },

        // ── Family B: 50% overlap along X (b shifted +0.5x) ─────────────
        // Result solids are all axis-aligned boxes with exact dimensions.
        Case { name: "halfx_fuse_area", a: (CUBE, NONE), b: (CUBE, HALF_X), op: Op::Fuse, oracle: Oracle::Area(8.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "halfx_fuse_volume", a: (CUBE, NONE), b: (CUBE, HALF_X), op: Op::Fuse, oracle: Oracle::Volume(1.5), tol: 1e-6, expect: Expect::Pass },
        Case { name: "halfx_cut_area", a: (CUBE, NONE), b: (CUBE, HALF_X), op: Op::Cut, oracle: Oracle::Area(4.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "halfx_common_area", a: (CUBE, NONE), b: (CUBE, HALF_X), op: Op::Common, oracle: Oracle::Area(4.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "halfx_common_volume", a: (CUBE, NONE), b: (CUBE, HALF_X), op: Op::Common, oracle: Oracle::Volume(0.5), tol: 1e-6, expect: Expect::Pass },
        Case { name: "halfx_cutrev_area", a: (CUBE, NONE), b: (CUBE, HALF_X), op: Op::CutRev, oracle: Oracle::Area(4.0), tol: 1e-6, expect: Expect::Pass },

        // ── Family C: face-coincident stack along Z (b shifted +1z) ─────
        // The two cubes share the z=1 face exactly — a coincident-face fuse.
        Case { name: "facez_fuse_area", a: (CUBE, NONE), b: (CUBE, STACK_Z), op: Op::Fuse, oracle: Oracle::Area(10.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "facez_fuse_volume", a: (CUBE, NONE), b: (CUBE, STACK_Z), op: Op::Fuse, oracle: Oracle::Volume(2.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "facez_cut_unchanged_area", a: (CUBE, NONE), b: (CUBE, STACK_Z), op: Op::Cut, oracle: Oracle::Area(6.0), tol: 1e-6, expect: Expect::Pass },
        // Common of face-touching cubes has zero volume → empty solid.
        Case { name: "facez_common_empty", a: (CUBE, NONE), b: (CUBE, STACK_Z), op: Op::Common, oracle: Oracle::Empty, tol: 0.0, expect: Expect::Pass },

        // ── Family D: full containment (small cube inside a 2-cube) ─────
        // Cut produces a hollow solid (cavity in an inner shell) — exercises
        // the all-faces measure. Outer area 24 + cavity area 6 = 30;
        // volume 8 − 1 = 7.
        Case { name: "contain_fuse_area", a: (Prim::Box(2.0, 2.0, 2.0), NONE), b: (CUBE, INSET), op: Op::Fuse, oracle: Oracle::Area(24.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "contain_common_area", a: (Prim::Box(2.0, 2.0, 2.0), NONE), b: (CUBE, INSET), op: Op::Common, oracle: Oracle::Area(6.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "contain_cut_hollow_area", a: (Prim::Box(2.0, 2.0, 2.0), NONE), b: (CUBE, INSET), op: Op::Cut, oracle: Oracle::Area(30.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "contain_cut_hollow_volume", a: (Prim::Box(2.0, 2.0, 2.0), NONE), b: (CUBE, INSET), op: Op::Cut, oracle: Oracle::Volume(7.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "contain_cutrev_empty", a: (Prim::Box(2.0, 2.0, 2.0), NONE), b: (CUBE, INSET), op: Op::CutRev, oracle: Oracle::Empty, tol: 0.0, expect: Expect::Pass },

        // ── Family E: corner overlap (b shifted +0.5x +0.5y) ────────────
        // Two cubes interpenetrating in a 0.5×0.5×1 column — non-convex
        // (L-shaped) fuse and notched cut. GAP: fuse under-removes the
        // overlap (volume 11/6 ≈ 1.833 vs true 1.75) and the notched cut
        // carries a spurious 0.25 of face area (5.75 vs 5.5), although its
        // volume (0.75) and the intersection (area 2.5) are correct. The
        // disagreement between Fuse and Common over the same overlap points
        // at the fuse/cut assembly, not the intersection.
        Case { name: "corner_fuse_area", a: (CUBE, NONE), b: (CUBE, CORNER), op: Op::Fuse, oracle: Oracle::Area(9.5), tol: 1e-6, expect: Expect::Gap("corner interpenetration: fuse over-counts overlap") },
        Case { name: "corner_fuse_volume", a: (CUBE, NONE), b: (CUBE, CORNER), op: Op::Fuse, oracle: Oracle::Volume(1.75), tol: 1e-6, expect: Expect::Gap("corner interpenetration: fuse volume 11/6 vs 7/4") },
        Case { name: "corner_common_area", a: (CUBE, NONE), b: (CUBE, CORNER), op: Op::Common, oracle: Oracle::Area(2.5), tol: 1e-6, expect: Expect::Pass },
        Case { name: "corner_cut_notch_area", a: (CUBE, NONE), b: (CUBE, CORNER), op: Op::Cut, oracle: Oracle::Area(5.5), tol: 1e-6, expect: Expect::Gap("corner interpenetration: notched cut has a spurious face (+0.25 area)") },
        Case { name: "corner_cut_notch_volume", a: (CUBE, NONE), b: (CUBE, CORNER), op: Op::Cut, oracle: Oracle::Volume(0.75), tol: 1e-6, expect: Expect::Pass },

        // ── Family F: prism input path (extruded unit square == cube) ───
        // Validates the PrismZ builder against the same oracles as box/box.
        Case { name: "prism_box_halfx_fuse_area", a: (Prim::PrismZ(UNIT_SQUARE, 1.0), NONE), b: (CUBE, HALF_X), op: Op::Fuse, oracle: Oracle::Area(8.0), tol: 1e-6, expect: Expect::Pass },
        Case { name: "prism_box_identical_common_area", a: (Prim::PrismZ(UNIT_SQUARE, 1.0), NONE), b: (CUBE, NONE), op: Op::Common, oracle: Oracle::Area(6.0), tol: 1e-6, expect: Expect::Pass },

        // ── Family G: curved calibration probe ──────────────────────────
        // Unit sphere (centered) fused with a unit cube at the +octant.
        // Reference whole-boundary area ≈ 14.6394. Tagged Gap so it never
        // fails CI; the scoreboard reports whether brepkit matches it and
        // at what tolerance — used to set the curved-family default.
        Case { name: "sphere_unit_box_fuse_area", a: (Prim::Sphere(1.0), NONE), b: (CUBE, NONE), op: Op::Fuse, oracle: Oracle::Area(14.6394), tol: 2e-2, expect: Expect::Gap("curved calibration probe") },
    ];

    run_corpus(cases);
}
