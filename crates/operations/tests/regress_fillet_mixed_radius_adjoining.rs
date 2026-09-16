//! N341: second fillet of an adjoining pair at an UNEQUAL radius from the
//! first (accepted) fillet. N339 repaired only the equal-radius case; its
//! `reblend::recognize` guard requires the inherited strip's cylinder radius
//! to match the newly requested radius, so a mixed-radius pair falls through
//! to the ordinary `rolling_ball::build` route, which cannot trim a
//! cylindrical face touched only at a vertex and leaves an open shell.
//!
//! This is deliberately the SECOND fillet of the pair (not the third blend
//! N340 repaired): one edge accepted at R1, then an ADJACENT edge requested
//! at R2 != R1.

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
use brepkit_topology::explorer::solid_edges;
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
    assert_eq!(matches.len(), 1, "unique target {a:?}-{b:?}");
    matches[0]
}

fn euler_report(topo: &Topology, solid: SolidId) -> String {
    let adjacency = topo.build_adjacency(solid).unwrap();
    let boundary = adjacency.boundary_edges().len();
    let v = solid_edges(topo, solid)
        .unwrap()
        .iter()
        .flat_map(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()]
        })
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let e = solid_edges(topo, solid).unwrap().len();
    let f = topo
        .shell(topo.solid(solid).unwrap().outer_shell())
        .unwrap()
        .faces()
        .len();
    format!("V={v} E={e} F={f} boundary={boundary}")
}

fn run_pair(r1: f64, r2: f64, label: &str) {
    let _ = env_logger::try_init();
    let mut topo = Topology::new();
    let source = source(&mut topo);
    let e1 = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let first = fillet_rolling_ball(&mut topo, source, &[e1], r1).unwrap();
    let report1 = validate_solid(&topo, first).unwrap();
    eprintln!(
        "N341[{label}] first r={r1} {} valid={} vol={}",
        euler_report(&topo, first),
        report1.is_valid(),
        solid_volume(&topo, first, 0.01).unwrap()
    );
    assert!(report1.is_valid(), "{:?}", report1.issues);

    let e2 = edge_at(
        &topo,
        first,
        Point3::new(0.0, 0.0, r1),
        Point3::new(0.0, 0.0, 30.0),
    );
    let second = fillet_rolling_ball(&mut topo, first, &[e2], r2).unwrap();
    let report2 = validate_solid(&topo, second).unwrap();
    eprintln!(
        "N341[{label}] second r={r2} {} valid={} vol={:?}",
        euler_report(&topo, second),
        report2.is_valid(),
        solid_volume(&topo, second, 0.01)
    );
    // This is the RED baseline assertion: it must currently FAIL, reproducing
    // the user's literal report ("Euler characteristic ... is invalid").
    assert!(
        report2.is_valid(),
        "N341[{label}] mixed-radius second fillet must produce a closed shell: {:?}",
        report2.issues
    );
    let volume = solid_volume(&topo, second, 0.01).unwrap();
    assert!(volume.is_finite() && volume > 0.0 && volume < 30_000.0);
}

#[test]
#[allow(deprecated)]
fn mixed_radius_larger_then_smaller_inch_like() {
    // 0.1 in then 0.05 in, expressed directly in mm (2.54, 1.27) to match the
    // user's likely units without involving OttoCAD's unit conversion layer.
    run_pair(2.54, 1.27, "2.54-then-1.27");
}

#[test]
#[allow(deprecated)]
fn mixed_radius_smaller_then_larger_inch_like() {
    run_pair(1.27, 1.6, "1.27-then-1.6");
}

#[test]
#[allow(deprecated)]
fn mixed_radius_larger_then_smaller_mm() {
    run_pair(2.0, 1.0, "2.0-then-1.0");
}

#[test]
#[allow(deprecated)]
fn mixed_radius_smaller_then_larger_mm() {
    run_pair(1.0, 1.3, "1.0-then-1.3");
}

#[test]
#[allow(deprecated)]
fn mixed_radius_via_fillet_v2_sequential() {
    // Attempt-2 evidence: does the newer fillet_v2 engine already tolerate a
    // mixed-radius adjoining pair applied sequentially (accept, then fillet
    // an adjacent edge at a different radius)? N339's ledger recorded v2
    // failing an EQUAL-radius sequential case on this exact source; recheck
    // with mixed radii before assuming v2 is a viable alternate route.
    use brepkit_operations::blend_ops::fillet_v2;
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
    let first = fillet_v2(&mut topo, source, &[e1], r1).unwrap();
    assert!(first.failed.is_empty(), "{:?}", first.failed);
    let report1 = validate_solid(&topo, first.solid).unwrap();
    eprintln!(
        "N341 fillet_v2 first r={r1} {} valid={}",
        euler_report(&topo, first.solid),
        report1.is_valid()
    );
    let e2 = edge_at(
        &topo,
        first.solid,
        Point3::new(0.0, 0.0, r1),
        Point3::new(0.0, 0.0, 30.0),
    );
    let second = fillet_v2(&mut topo, first.solid, &[e2], r2);
    match second {
        Ok(result) => {
            let report2 = validate_solid(&topo, result.solid).unwrap();
            eprintln!(
                "N341 fillet_v2 second r={r2} succeeded={:?} failed={:?} {} valid={}",
                result.succeeded,
                result.failed,
                euler_report(&topo, result.solid),
                report2.is_valid()
            );
        }
        Err(e) => {
            eprintln!("N341 fillet_v2 second r={r2} Err={e:?}");
        }
    }
}
