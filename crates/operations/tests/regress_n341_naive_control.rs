//! N341 attempt-1 control: deleting reblend::recognize's equal-radius guard
//! alone, with no other change, is not a repair. This test is EXPECTED to
//! demonstrate wrong output (either non-closed, or closed but with the
//! wrong radius on the previously-accepted strip) -- it exists only to
//! provide evidence for the attempt ledger and must not be left green
//! against production semantics.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

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
fn naive_guard_removal_produces_wrong_radius_not_a_repair() {
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
        "N341 naive-control valid={} issues={:?}",
        report.is_valid(),
        report.issues
    );
    if report.is_valid() {
        // If it validates as closed at all, prove the inherited strip's
        // cylinder radius silently changed from r1 to r2 -- geometry drift,
        // not a repair. Collect all cylindrical face radii present.
        let mut radii = Vec::new();
        for fid in solid_faces(&topo, second).unwrap() {
            if let FaceSurface::Cylinder(c) = topo.face(fid).unwrap().surface() {
                radii.push(c.radius());
            }
        }
        radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "N341 naive-control cylinder radii present: {radii:?} (expected both {r1} and {r2} present if the accepted strip's radius was preserved)"
        );
        let has_r1 = radii.iter().any(|r| (r - r1).abs() < 1e-6);
        assert!(
            !has_r1,
            "naive guard removal unexpectedly preserved the accepted r1={r1} radius -- re-examine before trusting this as a control"
        );
    }
    eprintln!(
        "N341 naive-control volume={:?}",
        solid_volume(&topo, second, 0.01)
    );
}
