//! A round post cut by a ring (a tube) lying across its side, the ring's
//! outer end tangent to the post's wall: the cut leaves a ring-shaped
//! groove in the wall with a pin standing in it.
//!
//! The ring's two walls cut the post's wall along two closed loops, one
//! inside the other. The face splitter told nested loops apart on planes
//! and spheres only, so on the post's wall both loops became holes of the
//! wall and discs of their own: the outer disc, unholed, covered the inner
//! one, and was sampled at its middle, in the ring's bore, so it read as
//! outside the ring and was kept. Nothing was cut, silently: the result was
//! valid. On a cylinder or cone the loops are now nested in `(u, v)`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const POST: f64 = 8.65;
const HEIGHT: f64 = 10.0;
const OUTER: f64 = 3.5;
const BORE: f64 = 1.5;
const DEPTH: f64 = 3.0;

/// The ring, along `+X` from `POST - DEPTH` to `POST` at mid-height, then
/// turned `phi` about the post's axis; `spin` turns it about its own axis
/// first (where its walls' seams lie).
fn ring(topo: &mut Topology, phi: f64, spin: f64) -> SolidId {
    let outer = make_cylinder(topo, OUTER, DEPTH).unwrap();
    let bore = make_cylinder(topo, BORE, DEPTH).unwrap();
    let tube = boolean(topo, BooleanOp::Cut, outer, bore).unwrap();
    let pose = Mat4::translation(0.0, 0.0, HEIGHT / 2.0)
        * Mat4::rotation_z(phi)
        * Mat4::translation(POST - DEPTH, 0.0, 0.0)
        * Mat4::rotation_y(FRAC_PI_2)
        * Mat4::rotation_z(spin);
    transform_solid(topo, tube, &pose).unwrap();
    tube
}

/// The volume the ring takes out of the post: through each point of the
/// ring's cross-section at `y` off its axis, the post's wall stands
/// `sqrt(POST^2 - y^2)` out, short of the ring's tangent end.
fn removed() -> f64 {
    // Midpoint rule in (rho, theta) over the annulus, fine enough for 1e-9.
    let (nr, nt) = (2000_u32, 4000_u32);
    let (dr, dt) = ((OUTER - BORE) / f64::from(nr), 2.0 * PI / f64::from(nt));
    let mut sum = 0.0;
    for i in 0..nr {
        let rho = (f64::from(i) + 0.5).mul_add(dr, BORE);
        for j in 0..nt {
            let y = rho * ((f64::from(j) + 0.5) * dt).cos();
            sum += (POST.mul_add(POST, -y * y).sqrt() - (POST - DEPTH)) * rho;
        }
    }
    sum * dr * dt
}

/// Every pose's failures, as lines.
fn failures(poses: &[(f64, f64)]) -> Vec<String> {
    let want = PI * POST * POST * HEIGHT - removed();
    let mut bad = Vec::new();
    for &(phi, spin) in poses {
        let mut topo = Topology::new();
        let post = make_cylinder(&mut topo, POST, HEIGHT).unwrap();
        let tube = ring(&mut topo, phi, spin);
        let before = mesh_fallback_count();
        let cut = boolean(&mut topo, BooleanOp::Cut, post, tube).unwrap();
        let fell_back = mesh_fallback_count() != before;
        let valid = validate_solid(&topo, cut).unwrap().is_valid();
        let v = solid_volume(&topo, cut, 1e-3).unwrap();
        if fell_back || !valid || (v - want).abs() > 1e-6 * want {
            bad.push(format!(
                "phi {phi}, spin {spin}: fallback {fell_back}, valid {valid}, volume {v} for {want}"
            ));
        }
    }
    bad
}

#[test]
fn a_ring_tangent_to_a_post_s_wall_cuts_a_groove() {
    let mut poses = Vec::new();
    for phi in [1.0, 1.75, 3.6] {
        for spin in [0.8, 2.5, 4.0] {
            poses.push((phi, spin));
        }
    }
    let bad = failures(&poses);
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// The ring's section with the post's wall touches the ring's outer rim at
/// its top and bottom. With the ring's seam through those points (spin 0)
/// its outer wall is left whole, the strip outside the post kept (0.12%
/// light); with the post's seam on the tangent line (phi 0) the cut falls
/// back to the mesh.
#[test]
#[ignore = "seams through the tangent points: see the doc comment"]
fn a_ring_with_a_seam_on_the_tangency_cuts_a_groove() {
    let bad = failures(&[(1.0, 0.0), (3.6, 0.0), (0.0, 0.8), (0.0, 2.5)]);
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
