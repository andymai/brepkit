//! Shared harness for the boolean/blend robustness parity corpus.
//!
//! Each case (`Case`) is pure data: two input solids built from primitives plus
//! transforms, a boolean operation, and an expected invariant (surface area,
//! volume, or empty result). The expected value is an independent analytic
//! oracle — for all-planar inputs it is exact, so the corpus doubles as a tight
//! regression net; for curved inputs it is a calibrated reference value.
//!
//! `run_corpus` executes a slice of cases and asserts each oracle, but fails the
//! test only on *unexpected* outcomes: a case tagged `Expect::Pass` that misses
//! its oracle is a regression, while a case tagged `Expect::Gap` documents a
//! known robustness gap and is allowed to miss. A `Gap` that starts passing is
//! reported (so it can be promoted) without failing CI. The printed scoreboard
//! makes parity a tracked number.
//!
//! This module is compiled into each corpus test binary; only a subset of its
//! API is used by any single file, hence `#![allow(dead_code)]`.

#![allow(dead_code)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
// The scoreboard is intentional test diagnostics, printed via `--nocapture`.
#![allow(clippy::print_stderr)]
#![allow(
    missing_docs,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::must_use_candidate
)]

use brepkit_check::properties::face_integrator::integrate_face;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::OperationsError;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere, make_torus};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid_relaxed;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_planar_face;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

/// Equatorial segment count for sphere inputs. High enough that the analytic
/// surface area is essentially exact and boolean intersection curves are well
/// resolved.
const SPHERE_SEGMENTS: usize = 64;
/// Segment count for torus inputs.
const TORUS_SEGMENTS: usize = 64;
/// Vertex-merge tolerance for prism (extruded-polygon) faces.
const FACE_TOL: f64 = 1e-7;
/// Gauss-Legendre quadrature order for area/volume integration.
const GAUSS_ORDER: usize = 5;

#[derive(Clone, Copy)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    fn rotation(self, radians: f64) -> Mat4 {
        match self {
            Self::X => Mat4::rotation_x(radians),
            Self::Y => Mat4::rotation_y(radians),
            Self::Z => Mat4::rotation_z(radians),
        }
    }
}

/// A placement transform applied to a freshly-built primitive, in list order.
#[derive(Clone, Copy)]
pub enum Xf {
    Translate(f64, f64, f64),
    /// Rotate about a principal axis through the origin (degrees).
    Rotate {
        axis: Axis,
        degrees: f64,
    },
    /// Rotate about a principal axis through an arbitrary point (degrees).
    RotateAbout {
        axis: Axis,
        point: (f64, f64, f64),
        degrees: f64,
    },
}

/// A primitive constructor with the kernel's placement conventions: box corner
/// at the origin; cylinder/cone base at z=0; sphere/torus centered at origin.
#[derive(Clone, Copy)]
pub enum Prim {
    Box(f64, f64, f64),
    Sphere(f64),
    Cyl(f64, f64),
    Cone(f64, f64, f64),
    Torus(f64, f64),
    /// Extrude an XY polygon (wound CCW) by a height along +Z.
    PrismZ(&'static [(f64, f64)], f64),
}

#[derive(Clone, Copy)]
pub enum Op {
    Fuse,
    Cut,
    Common,
    /// Reverse cut: `b - a`.
    CutRev,
}

#[derive(Clone, Copy)]
pub enum Oracle {
    Area(f64),
    Volume(f64),
    /// The operation must yield an empty result (no volume).
    Empty,
}

#[derive(Clone, Copy)]
pub enum Expect {
    Pass,
    /// Known robustness gap; carries a short reason. Allowed to miss its oracle.
    Gap(&'static str),
}

#[derive(Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub a: (Prim, &'static [Xf]),
    pub b: (Prim, &'static [Xf]),
    pub op: Op,
    pub oracle: Oracle,
    /// Relative tolerance for the oracle comparison.
    pub tol: f64,
    pub expect: Expect,
}

fn build_err(e: OperationsError) -> String {
    format!("build: {e:?}")
}

fn apply_xf(topo: &mut Topology, s: SolidId, xf: Xf) -> Result<(), String> {
    let go = |topo: &mut Topology, m: &Mat4| {
        transform_solid(topo, s, m).map_err(|e| format!("transform: {e:?}"))
    };
    match xf {
        Xf::Translate(x, y, z) => go(topo, &Mat4::translation(x, y, z)),
        Xf::Rotate { axis, degrees } => go(topo, &axis.rotation(degrees.to_radians())),
        Xf::RotateAbout {
            axis,
            point: (px, py, pz),
            degrees,
        } => {
            // Rotation about an axis through `point`: translate the point to the
            // origin, rotate, translate back. Applied as successive in-place
            // transforms so no Mat4 multiplication is required.
            go(topo, &Mat4::translation(-px, -py, -pz))?;
            go(topo, &axis.rotation(degrees.to_radians()))?;
            go(topo, &Mat4::translation(px, py, pz))
        }
    }
}

fn build(topo: &mut Topology, &(prim, xforms): &(Prim, &'static [Xf])) -> Result<SolidId, String> {
    let s = match prim {
        Prim::Box(x, y, z) => make_box(topo, x, y, z).map_err(build_err)?,
        Prim::Sphere(r) => make_sphere(topo, r, SPHERE_SEGMENTS).map_err(build_err)?,
        Prim::Cyl(r, h) => make_cylinder(topo, r, h).map_err(build_err)?,
        Prim::Cone(rb, rt, h) => make_cone(topo, rb, rt, h).map_err(build_err)?,
        Prim::Torus(maj, min) => make_torus(topo, maj, min, TORUS_SEGMENTS).map_err(build_err)?,
        Prim::PrismZ(poly, h) => {
            let pts: Vec<Point3> = poly.iter().map(|&(x, y)| Point3::new(x, y, 0.0)).collect();
            let face = make_planar_face(topo, &pts, FACE_TOL)
                .map_err(|e| format!("planar face: {e:?}"))?;
            extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), h).map_err(build_err)?
        }
    };
    for &xf in xforms {
        apply_xf(topo, s, xf)?;
    }
    Ok(s)
}

fn run_op(topo: &mut Topology, op: Op, a: SolidId, b: SolidId) -> Result<SolidId, OperationsError> {
    match op {
        Op::Fuse => boolean(topo, BooleanOp::Fuse, a, b),
        Op::Cut => boolean(topo, BooleanOp::Cut, a, b),
        Op::Common => boolean(topo, BooleanOp::Intersect, a, b),
        Op::CutRev => boolean(topo, BooleanOp::Cut, b, a),
    }
}

/// Total boundary area over *all* faces (outer shell + any cavity shells), so
/// hollow results measure their full surface, matching whole-boundary oracles.
fn area(topo: &Topology, s: SolidId) -> Result<f64, String> {
    let faces = solid_faces(topo, s).map_err(|e| format!("solid_faces: {e:?}"))?;
    let mut total = 0.0;
    for fid in faces {
        total += integrate_face(topo, fid, GAUSS_ORDER)
            .map_err(|e| format!("area: {e:?}"))?
            .area;
    }
    Ok(total)
}

/// Net volume via the divergence theorem over *all* faces. Cavity-shell faces
/// carry inward orientation, so their contributions subtract — a box with a
/// cubic cavity nets (outer − cavity) volume automatically.
fn volume(topo: &Topology, s: SolidId) -> Result<f64, String> {
    let faces = solid_faces(topo, s).map_err(|e| format!("solid_faces: {e:?}"))?;
    let mut total = 0.0;
    for fid in faces {
        total += integrate_face(topo, fid, GAUSS_ORDER)
            .map_err(|e| format!("volume: {e:?}"))?
            .volume;
    }
    Ok(total)
}

fn ensure_nonempty_valid(topo: &Topology, s: SolidId) -> Result<(), String> {
    if topo.is_empty_solid(s) {
        return Err("result is empty (expected a solid)".to_string());
    }
    let report = validate_solid_relaxed(topo, s).map_err(|e| format!("validate: {e:?}"))?;
    if !report.is_valid() {
        return Err(format!("invalid result ({} errors)", report.error_count()));
    }
    Ok(())
}

fn check_rel(got: f64, expected: f64, tol: f64, what: &str) -> Result<(), String> {
    let denom = got.abs().max(expected.abs()).max(1.0);
    let rel = (got - expected).abs() / denom;
    if rel <= tol {
        Ok(())
    } else {
        Err(format!(
            "{what} {got:.6} vs expected {expected:.6} (rel {rel:.2e} > tol {tol:.0e})"
        ))
    }
}

/// Build both inputs, run the operation, and check the oracle. `Ok(())` means
/// the case matched its oracle (and, for non-empty results, validated).
fn evaluate(topo: &mut Topology, c: &Case) -> Result<(), String> {
    let a = build(topo, &c.a)?;
    let b = build(topo, &c.b)?;
    let res = run_op(topo, c.op, a, b);
    match c.oracle {
        Oracle::Empty => match res {
            Err(OperationsError::EmptyResult { .. }) => Ok(()),
            Ok(s) if topo.is_empty_solid(s) => Ok(()),
            Ok(s) => Err(format!(
                "expected empty, got non-empty (area {:.6})",
                area(topo, s)?
            )),
            Err(e) => Err(format!("expected empty, got error: {e:?}")),
        },
        Oracle::Area(expected) => {
            let s = res.map_err(|e| format!("op failed: {e:?}"))?;
            ensure_nonempty_valid(topo, s)?;
            check_rel(area(topo, s)?, expected, c.tol, "area")
        }
        Oracle::Volume(expected) => {
            let s = res.map_err(|e| format!("op failed: {e:?}"))?;
            ensure_nonempty_valid(topo, s)?;
            check_rel(volume(topo, s)?, expected, c.tol, "volume")
        }
    }
}

/// Execute a corpus and assert it. Fails only on unexpected regressions
/// (`Expect::Pass` cases that miss their oracle). Prints a parity scoreboard.
pub fn run_corpus(cases: &[Case]) {
    let mut pass = 0usize;
    let mut newly_passing: Vec<&str> = Vec::new();
    let mut gaps: Vec<String> = Vec::new();
    let mut regressions: Vec<String> = Vec::new();

    for c in cases {
        let mut topo = Topology::new();
        let outcome = evaluate(&mut topo, c);
        match (&outcome, c.expect) {
            (Ok(()), Expect::Pass) => pass += 1,
            (Ok(()), Expect::Gap(_)) => newly_passing.push(c.name),
            (Err(why), Expect::Pass) => regressions.push(format!("[FAIL] {}: {}", c.name, why)),
            (Err(why), Expect::Gap(reason)) => {
                gaps.push(format!("{} — {why}  [{reason}]", c.name));
            }
        }
    }

    eprintln!(
        "── parity scoreboard ──  {pass} pass · {} known-gap · {} newly-passing · {} unexpected-fail",
        gaps.len(),
        newly_passing.len(),
        regressions.len()
    );
    for detail in &gaps {
        eprintln!("   • known gap: {detail}");
    }
    for name in &newly_passing {
        eprintln!("   ↑ newly passing — promote Gap→Pass: {name}");
    }

    assert!(
        regressions.is_empty(),
        "unexpected parity regressions ({}):\n{}",
        regressions.len(),
        regressions.join("\n")
    );
}
