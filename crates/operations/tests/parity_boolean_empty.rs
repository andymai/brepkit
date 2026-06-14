//! Parity robustness corpus — operations that must yield an *empty* result.
//!
//! Empty results are a notorious robustness cliff: a kernel must recognize that
//! a subtraction cancels, a tool misses, or two solids meet only on a shared
//! boundary (zero volume), and return a clean empty shape rather than a sliver,
//! a panic, or a malformed solid. The harness accepts either convention brepkit
//! uses (an `EmptyResult` error or an `Ok` empty solid).
//!
//! Cases tagged `Pass` are asserted; `Gap` documents a known miss. See
//! `parity_support` for the harness and scoreboard semantics.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(clippy::missing_panics_doc)]

mod parity_support;

use parity_support::{Case, Expect, Op, Oracle, Prim, Xf, run_corpus};

const NONE: &[Xf] = &[];
const CUBE: Prim = Prim::Box(1.0, 1.0, 1.0);
const EMPTY: Oracle = Oracle::Empty;

#[test]
fn empty_result_corpus() {
    #[rustfmt::skip]
    let cases: &[Case] = &[
        // ── Self-cancelling subtraction (identical solids) ──────────────
        Case { name: "identical_box_cut",        a: (Prim::Box(2.0,3.0,4.0), NONE), b: (Prim::Box(2.0,3.0,4.0), NONE), op: Op::Cut,    oracle: EMPTY, tol: 0.0, expect: Expect::Pass },
        Case { name: "identical_box_cutrev",     a: (Prim::Box(2.0,3.0,4.0), NONE), b: (Prim::Box(2.0,3.0,4.0), NONE), op: Op::CutRev, oracle: EMPTY, tol: 0.0, expect: Expect::Pass },
        Case { name: "identical_sphere_cut",     a: (Prim::Sphere(1.0), NONE),      b: (Prim::Sphere(1.0), NONE),      op: Op::Cut,    oracle: EMPTY, tol: 0.0, expect: Expect::Pass },
        Case { name: "identical_cyl_cut",        a: (Prim::Cyl(0.7,2.0), NONE),     b: (Prim::Cyl(0.7,2.0), NONE),     op: Op::Cut,    oracle: EMPTY, tol: 0.0, expect: Expect::Pass },

        // ── Subset subtraction (A ⊆ B  ⇒  A − B = ∅) ────────────────────
        // small cube fully inside a 3-cube
        Case { name: "subset_box_cut",           a: (CUBE, &[Xf::Translate(1.0,1.0,1.0)]), b: (Prim::Box(3.0,3.0,3.0), NONE), op: Op::Cut, oracle: EMPTY, tol: 0.0, expect: Expect::Pass },
        // small box fully inside a sphere
        Case { name: "subset_box_in_sphere_cut", a: (Prim::Box(0.2,0.2,0.2), &[Xf::Translate(-0.1,-0.1,-0.1)]), b: (Prim::Sphere(1.0), NONE), op: Op::Cut, oracle: EMPTY, tol: 0.0, expect: Expect::Pass },
        // small sphere fully inside a 3-cube (sphere centered at the box center)
        Case { name: "subset_sphere_in_box_cut", a: (Prim::Sphere(0.5), &[Xf::Translate(1.5,1.5,1.5)]), b: (Prim::Box(3.0,3.0,3.0), NONE), op: Op::Cut, oracle: EMPTY, tol: 0.0, expect: Expect::Pass },

        // ── Disjoint intersection (no shared volume) ────────────────────
        Case { name: "disjoint_box_common_near", a: (CUBE, NONE), b: (CUBE, &[Xf::Translate(2.0,0.0,0.0)]), op: Op::Common, oracle: EMPTY, tol: 0.0, expect: Expect::Pass },
        Case { name: "disjoint_box_common_far",  a: (CUBE, NONE), b: (CUBE, &[Xf::Translate(5.0,5.0,5.0)]), op: Op::Common, oracle: EMPTY, tol: 0.0, expect: Expect::Pass },
        Case { name: "disjoint_cyl_common",      a: (Prim::Cyl(0.5,1.0), NONE), b: (Prim::Cyl(0.5,1.0), &[Xf::Translate(3.0,0.0,0.0)]), op: Op::Common, oracle: EMPTY, tol: 0.0, expect: Expect::Pass },
        Case { name: "disjoint_sphere_common",   a: (Prim::Sphere(1.0), NONE), b: (Prim::Sphere(1.0), &[Xf::Translate(3.0,0.0,0.0)]), op: Op::Common, oracle: EMPTY, tol: 0.0, expect: Expect::Pass },

        // ── Boundary-only contact: intersection has zero volume ─────────
        // Two cubes meeting on a shared face / edge / vertex only. The common
        // is geometrically a face / edge / point — i.e. empty as a solid.
        Case { name: "facetouch_box_common",     a: (CUBE, NONE), b: (CUBE, &[Xf::Translate(1.0,0.0,0.0)]), op: Op::Common, oracle: EMPTY, tol: 0.0, expect: Expect::Pass },
        Case { name: "edgetouch_box_common",     a: (CUBE, NONE), b: (CUBE, &[Xf::Translate(1.0,1.0,0.0)]), op: Op::Common, oracle: EMPTY, tol: 0.0, expect: Expect::Pass },
        Case { name: "vertextouch_box_common",   a: (CUBE, NONE), b: (CUBE, &[Xf::Translate(1.0,1.0,1.0)]), op: Op::Common, oracle: EMPTY, tol: 0.0, expect: Expect::Pass },
    ];

    run_corpus(cases);
}
