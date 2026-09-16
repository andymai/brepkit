//! N341 history: session 1's attempt-1 control proved that deleting
//! `reblend::recognize`'s equal-radius guard alone, with no other change,
//! is not a repair -- `recognize` still returned `None` for every face
//! (other equality checks remained), so the naive removal never even
//! reached corner construction and reproduced the unmodified baseline's
//! open shell. That evidence lived in this file as a deliberately-red
//! negative control (`report.is_valid()` was expected to be false, or, if
//! ever true, to expose radius drift).
//!
//! Session 3 shipped the real repair: `recognize` now accepts the inherited
//! strip at its own radius (`Strip::radius`) instead of requiring equality
//! to the newly requested radius, and `SetbackCorner`/`setback_patch::surface_mixed`
//! build the exact two-radius corner patch from both radii independently
//! (see that module's doc for the derivation). That is a materially
//! different code path from "delete the guard and change nothing else" --
//! it is the concrete next-prerequisite construction the guard-removal
//! control always said was still missing, now implemented and numerically
//! qualified in `setback_patch`'s own tests.
//!
//! This file now asserts the POSITIVE, current behavior instead of the old
//! negative one: the previously-accepted strip's radius must survive
//! unchanged, the newly requested strip must carry its own (different)
//! radius, and both must be simultaneously present as distinct cylindrical
//! faces in the closed result. Keeping this file (rather than deleting it)
//! preserves the same source/edge geometry the original control used as an
//! independent regression alongside `regress_fillet_mixed_radius_adjoining.rs`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    deprecated
)]

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::extrude::extrude;
use brepkit_operations::fillet::fillet_rolling_ball;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;

fn source(topo: &mut Topology) -> SolidId {
    let wire = make_polygon_wire(
        topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(40.0, 0.0, 0.0),
            Point3::new(40.0, 25.0, 0.0),
            Point3::new(0.0, 25.0, 0.0),
        ],
        Tolerance::new().linear,
    )
    .unwrap();
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), 30.0).unwrap()
}

fn edge_at(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let tolerance = Tolerance::new().linear;
    let matches: Vec<_> = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&id| {
            let edge = topo.edge(id).unwrap();
            let start = topo.vertex(edge.start()).unwrap().point();
            let end = topo.vertex(edge.end()).unwrap().point();
            ((start - a).length() < tolerance && (end - b).length() < tolerance)
                || ((start - b).length() < tolerance && (end - a).length() < tolerance)
        })
        .collect();
    assert_eq!(matches.len(), 1);
    matches[0]
}

#[test]
#[allow(deprecated)]
fn repaired_route_preserves_the_accepted_radius_and_carries_the_new_one() {
    let _ = env_logger::try_init();
    let mut topo = Topology::new();
    let source = source(&mut topo);
    let e1 = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let r1 = 2.54;
    let r2 = 1.27;
    let first = fillet_rolling_ball(&mut topo, source, &[e1], r1).unwrap();
    let e2 = edge_at(
        &topo,
        first,
        Point3::new(0.0, 0.0, r1),
        Point3::new(0.0, 0.0, 30.0),
    );
    let second = fillet_rolling_ball(&mut topo, first, &[e2], r2).unwrap();
    let report = validate_solid(&topo, second).unwrap();
    eprintln!(
        "N341 repaired-route valid={} issues={:?}",
        report.is_valid(),
        report.issues
    );
    assert!(
        report.is_valid(),
        "N341 repair must close the shell: {:?}",
        report.issues
    );

    // Both the previously-accepted strip's own radius (r1) and the newly
    // requested strip's own radius (r2) must be simultaneously present as
    // distinct cylindrical faces -- neither strip's radius may drift to
    // match the other, which is exactly the failure mode the original
    // guard-removal control (this file's history, above) demonstrated.
    let mut radii = Vec::new();
    for fid in solid_faces(&topo, second).unwrap() {
        if let FaceSurface::Cylinder(c) = topo.face(fid).unwrap().surface() {
            radii.push(c.radius());
        }
    }
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("N341 repaired-route cylinder radii present: {radii:?}");
    let has_r1 = radii.iter().any(|r| (r - r1).abs() < 1e-6);
    let has_r2 = radii.iter().any(|r| (r - r2).abs() < 1e-6);
    assert!(
        has_r1,
        "accepted r1={r1} radius must survive unchanged: {radii:?}"
    );
    assert!(
        has_r2,
        "requested r2={r2} radius must be present: {radii:?}"
    );

    let volume = solid_volume(&topo, second, 0.01).unwrap();
    eprintln!("N341 repaired-route volume={volume:?}");
    assert!(volume.is_finite() && volume > 0.0 && volume < 30_000.0);
}
