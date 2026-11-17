//! A near-zero-length residual boundary edge, collinear with and immediately
//! followed by a *different*, independently selected fillet contour, must not
//! be treated as the hard material boundary that bounds a sharp equal-radius
//! two-edge miter corner's runout.
//!
//! The fixture is a pentagonal prism whose bottom-face outline is
//!
//!   P1(0,0) --r1--> P2(Lx,0) --stub--> P3(Lx,stub) --r2--> P4(Lx,Ly) --> P5(0,Ly) --r1--> P1
//!
//! `P1` is a genuine sharp orthogonal equal-radius (`r1`) miter corner
//! between the two selected edges `P5-P1` and `P1-P2`, each backed by its own
//! vertical wall face plus the shared bottom floor face, which is exactly
//! [`brepkit_blend::corner::find_equal_two_edge_miter_junctions`]'s qualifying
//! shape. Contour `P1-P2`'s far runout terminal is `P2`; its only unselected
//! neighbor is the tiny `P2-P3` stub (length `stub`, well under `r1`), whose
//! far vertex `P3` is itself the near terminal of the independently selected
//! contour `P3-P4` (radius `r2`, collinear with `P2-P3`). Real material
//! plainly continues past the stub, since `stub + |P3P4|` is comfortably
//! larger than `r1`, so the corner is admissible: the seam is measured as the
//! stub plus its collinear continuation.
//!
//! The end of the `r1` round on the plane `x = Lx` spans the stub and runs
//! on to the face past it, where the `r2` round takes the corner of the
//! floor too. The two rounds cross: they meet along a crease, each wall
//! face on `x = Lx` loses the corner the `r1` round takes, and the stub goes.
//!
//! The oracle here is deliberately strong: every selected edge must report
//! success, no edge may fail, the result may not be partial, the fillet
//! surfaces must actually be present in the result, the solid must validate
//! (no two faces on one plane cover the same area), and its tessellation
//! must be watertight. An `Ok` result carrying `succeeded == []` (the
//! untouched input) fails every one of those checks.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    clippy::panic,
    deprecated
)]

use std::collections::HashMap;

use brepkit_blend::radius_law::RadiusLaw;
use brepkit_math::chord::DEFAULT_ANGULAR_TOL;
use brepkit_math::mat::Mat4;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2_variable;
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::tessellate::{
    boundary_edge_count, non_manifold_edge_count, tessellate_solid_with_tolerance,
};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// `Lx`, `Ly`: the pentagon's overall extent. `stub`: the tiny residual edge
/// length. `r1`: the miter corner's radius, equal on both selected edges,
/// deliberately much larger than `stub` alone but comfortably smaller than
/// `stub + |P3P4|`. `r2`: the independent contour's own (unrelated, smaller)
/// radius. `height`: the prism's extrude height, large enough that it never
/// itself constrains any of these radii. `p4y` sets the far end of the
/// independent contour, and so how much real material continues past the
/// stub.
fn seam_stub_prism(topo: &mut Topology, p4y: f64) -> (SolidId, [Point3; 5]) {
    seam_stub_prism_of_height(topo, p4y, 20.0)
}

/// [`seam_stub_prism`] of any height.
fn seam_stub_prism_of_height(topo: &mut Topology, p4y: f64, height: f64) -> (SolidId, [Point3; 5]) {
    let (lx, ly, stub) = (20.0, 20.0, 0.05);
    let p1 = Point3::new(0.0, 0.0, 0.0);
    let p2 = Point3::new(lx, 0.0, 0.0);
    let p3 = Point3::new(lx, stub, 0.0);
    let p4 = Point3::new(lx, p4y, 0.0);
    let p5 = Point3::new(0.0, ly, 0.0);
    let wire = make_polygon_wire(topo, &[p1, p2, p3, p4, p5], Tolerance::new().linear).unwrap();
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ));
    let solid = extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), height).unwrap();
    (solid, [p1, p2, p3, p4, p5])
}

/// The same fixture with the independent continuation *curved*: the outline
/// leaves `P3` along a circular arc whose tangent there is exactly collinear
/// with the `P2-P3` stub, but whose end-to-end chord is not. The seam is real
/// (material continues along the seam direction) and must be recognized from
/// that tangent, not from the chord.
fn seam_stub_prism_with_curved_continuation(topo: &mut Topology) -> (SolidId, [Point3; 4]) {
    let (stub, height, radius) = (0.05, 20.0, 8.0);
    let p1 = Point3::new(0.0, 0.0, 0.0);
    let p2 = Point3::new(20.0, 0.0, 0.0);
    let p3 = Point3::new(20.0, stub, 0.0);
    // Arc centre (20 - radius, stub): through P3, tangent +y there, curving
    // counter-clockwise through a quarter turn to P4.
    let center = Point3::new(20.0 - radius, stub, 0.0);
    let p4 = Point3::new(center.x(), center.y() + radius, 0.0);
    let p5 = Point3::new(0.0, 20.0, 0.0);

    let vertex =
        |topo: &mut Topology, p: Point3| topo.add_vertex(Vertex::new(p, Tolerance::new().linear));
    let (v1, v2, v3, v4, v5) = (
        vertex(topo, p1),
        vertex(topo, p2),
        vertex(topo, p3),
        vertex(topo, p4),
        vertex(topo, p5),
    );
    let line = |topo: &mut Topology, a, b| topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
    let e1 = line(topo, v1, v2);
    let e2 = line(topo, v2, v3);
    let e3 = {
        let circle =
            brepkit_math::curves::Circle3D::new(center, Vec3::new(0.0, 0.0, 1.0), radius).unwrap();
        topo.add_edge(Edge::new(v3, v4, EdgeCurve::Circle(circle)))
    };
    let e4 = line(topo, v4, v5);
    let e5 = line(topo, v5, v1);
    let wire = topo.add_wire(
        Wire::new(
            vec![
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
                OrientedEdge::new(e3, true),
                OrientedEdge::new(e4, true),
                OrientedEdge::new(e5, true),
            ],
            true,
        )
        .unwrap(),
    );
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ));
    let solid = extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), height).unwrap();
    (solid, [p1, p2, p3, p4])
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
    assert_eq!(matches.len(), 1, "unique target edge {a:?}-{b:?}");
    matches[0]
}

fn free_edge_count(topo: &Topology, solid: SolidId) -> usize {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        let mut wires = vec![face.outer_wire()];
        wires.extend_from_slice(face.inner_wires());
        for wid in wires {
            for oe in topo.wire(wid).unwrap().edges() {
                *counts.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    counts.values().filter(|&&c| c == 1).count()
}

fn same_sense_pair_count(topo: &Topology, solid: SolidId) -> usize {
    let mut senses: HashMap<usize, Vec<bool>> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        let rev = face.is_reversed();
        let mut wires = vec![face.outer_wire()];
        wires.extend_from_slice(face.inner_wires());
        for wid in wires {
            for oe in topo.wire(wid).unwrap().edges() {
                senses
                    .entry(oe.edge().index())
                    .or_default()
                    .push(oe.is_forward() ^ rev);
            }
        }
    }
    senses
        .values()
        .filter(|u| u.len() == 2 && u[0] == u[1])
        .count()
}

/// The blend-face tessellation gap, pinned per fixture so it cannot grow
/// silently.
const SEAM_PRISM_RUNOUT_PATCH_GAP: usize = 0;
/// The near-boundary fixture (`P4` at y = 12.0, a 12.05mm run against
/// r = 8). One of its ends meets a tilted end face, which takes the round's
/// notch (the round run on to it).
const NEAR_BOUNDARY_RUNOUT_PATCH_GAP: usize = 0;

/// A built result is only a pass when every requested edge reports success,
/// nothing is partial, the fillet surfaces are really present, the solid
/// validates, its B-Rep is closed and orientation-consistent, and its
/// tessellation is watertight up to the pinned blend-face gap above. An `Ok`
/// carrying `succeeded == []` (the untouched input) fails all of it.
fn assert_genuinely_filleted(
    topo: &Topology,
    result: &brepkit_blend::BlendResult,
    requested: usize,
    radii: &[f64],
    tessellation_gap: usize,
    label: &str,
) {
    assert_eq!(
        result.succeeded.len(),
        requested,
        "{label}: every selected edge must report success (failed={:?}, partial={})",
        result.failed,
        result.is_partial
    );
    assert!(
        result.failed.is_empty(),
        "{label}: no selected edge may fail: {:?}",
        result.failed
    );
    assert!(
        !result.is_partial,
        "{label}: a successful construction is never partial"
    );
    let surfaces: Vec<f64> = solid_faces(topo, result.solid)
        .unwrap()
        .into_iter()
        .filter_map(|fid| match topo.face(fid).unwrap().surface() {
            FaceSurface::Cylinder(c) => Some(c.radius()),
            _ => None,
        })
        .collect();
    for radius in radii {
        assert!(
            surfaces
                .iter()
                .any(|candidate| (candidate - radius).abs() < 1e-9),
            "{label}: the result must actually carry a radius-{radius} fillet surface (found {surfaces:?})"
        );
    }
    let non_planar = solid_faces(topo, result.solid)
        .unwrap()
        .into_iter()
        .filter(|&fid| !topo.face(fid).unwrap().surface().is_planar())
        .count();
    assert!(
        non_planar > 0,
        "{label}: a filleted result must carry at least one blend surface"
    );
    let report = brepkit_operations::validate::validate_solid(topo, result.solid).unwrap();
    assert!(
        report.is_valid(),
        "{label}: fillet result must validate: {:?}",
        report.issues
    );
    assert_eq!(
        free_edge_count(topo, result.solid),
        0,
        "{label}: result must be closed"
    );
    assert_eq!(
        same_sense_pair_count(topo, result.solid),
        0,
        "{label}: result must be orientation-consistent"
    );
    // The bar is a watertight tessellation, not edge-id closure alone: the
    // wasm `try_fillet` gate requires the mesh to close.
    let mesh =
        tessellate_solid_with_tolerance(topo, result.solid, 0.25, DEFAULT_ANGULAR_TOL).unwrap();
    assert_eq!(
        boundary_edge_count(&mesh),
        tessellation_gap,
        "{label}: tessellation boundary half-edges must equal the pinned blend-face gap"
    );
    assert_eq!(
        non_manifold_edge_count(&mesh),
        0,
        "{label}: tessellation must be manifold"
    );
}

/// A control that pins the tessellation bar itself: a plain single-edge
/// rolling-ball fillet and a bare two-edge miter corner both mesh watertight
/// on this same tree, so a non-zero count for the seam prism is a property
/// of that shape's terminal runout patches, not of the check.
#[test]
fn tessellation_watertight_controls() {
    let mut topo = Topology::new();
    let size = 20.0;
    let solid = brepkit_operations::primitives::make_box(&mut topo, size, size, size).unwrap();
    let top_edge = edge_at(
        &topo,
        solid,
        Point3::new(0.0, 0.0, size),
        Point3::new(size, 0.0, size),
    );
    let single =
        brepkit_operations::fillet::fillet_rolling_ball(&mut topo, solid, &[top_edge], 2.0)
            .unwrap();
    let mesh = tessellate_solid_with_tolerance(&topo, single, 0.25, DEFAULT_ANGULAR_TOL).unwrap();
    assert_eq!(
        boundary_edge_count(&mesh),
        0,
        "control: single-edge fillet must mesh watertight"
    );

    let mut topo = Topology::new();
    let solid = brepkit_operations::primitives::make_box(&mut topo, size, size, size).unwrap();
    let corner = Point3::new(0.0, 0.0, 0.0);
    let edges = [
        edge_at(&topo, solid, corner, Point3::new(size, 0.0, 0.0)),
        edge_at(&topo, solid, corner, Point3::new(0.0, size, 0.0)),
    ];
    let mitered = brepkit_operations::blend_ops::fillet_v2(&mut topo, solid, &edges, 8.0)
        .unwrap()
        .solid;
    let mesh = tessellate_solid_with_tolerance(&topo, mitered, 0.25, DEFAULT_ANGULAR_TOL).unwrap();
    assert_eq!(
        boundary_edge_count(&mesh),
        0,
        "control: bare two-edge miter corner must mesh watertight"
    );
}

#[test]
fn seam_stub_miter_runout_is_watertight() {
    let mut topo = Topology::new();
    let (solid, [p1, p2, p3, p4, p5]) = seam_stub_prism(&mut topo, 20.0);

    let p1p2 = edge_at(&topo, solid, p1, p2);
    let p5p1 = edge_at(&topo, solid, p5, p1);
    let p3p4 = edge_at(&topo, solid, p3, p4);
    // The stub is a real edge of the input, and the whole point of the shape.
    let _stub = edge_at(&topo, solid, p2, p3);

    let r1 = 8.0;
    let r2 = 2.0;
    let edge_laws = vec![
        (p1p2, RadiusLaw::Constant(r1)),
        (p5p1, RadiusLaw::Constant(r1)),
        (p3p4, RadiusLaw::Constant(r2)),
    ];

    let result = fillet_v2_variable(&mut topo, solid, edge_laws)
        .expect("an admissible seam-stub miter runout must not be refused");
    assert_genuinely_filleted(
        &topo,
        &result,
        3,
        &[r1, r2],
        SEAM_PRISM_RUNOUT_PATCH_GAP,
        "seam-stub miter runout",
    );
}

/// With only 3.0mm of real material past the stub the request is genuinely
/// oversized, and the honest answer is the *pre-mutation* `RadiusTooLarge`
/// refusal naming the combined run, not a mutation-time failure and not a
/// silently built solid. The seam is not free material: the bound is
/// `stub + collinear continuation`.
#[test]
fn short_continuation_run_is_refused_on_the_combined_run() {
    let mut topo = Topology::new();
    // |P3P4| = 2.95, so stub + continuation = 3.0 < r1 = 8.
    let (solid, [p1, p2, p3, p4, p5]) = seam_stub_prism(&mut topo, 3.0);

    let edge_laws = vec![
        (edge_at(&topo, solid, p1, p2), RadiusLaw::Constant(8.0)),
        (edge_at(&topo, solid, p5, p1), RadiusLaw::Constant(8.0)),
        (edge_at(&topo, solid, p3, p4), RadiusLaw::Constant(1.0)),
    ];

    match fillet_v2_variable(&mut topo, solid, edge_laws) {
        Err(brepkit_operations::OperationsError::Blend(
            brepkit_blend::BlendError::RadiusTooLarge { max_radius, .. },
        )) => {
            assert!(
                (max_radius - 3.0).abs() < 1e-9,
                "the refusal must bound the combined stub + continuation run (3.0), got {max_radius}"
            );
        }
        Err(other) => {
            panic!("a 3.0mm run against r=8 must be refused before any mutation, got {other}")
        }
        Ok(result) => panic!(
            "a 3.0mm run against r=8 must be refused, got a solid with {} successes",
            result.succeeded.len()
        ),
    }
}

/// The same bound, exercised from both sides: a continuation run that leaves
/// *just* less than the radius of material is refused, and one that leaves
/// just more builds and closes. That resolves the short-run self-intersection
/// question structurally, because a runout needs a full radius of material to
/// land on and below that the construction is never attempted, so there is no
/// solid whose `validate_solid` could miss an overlap.
#[test]
fn continuation_run_boundary_is_the_radius() {
    let mut topo = Topology::new();
    let (refused, [p1, p2, p3, p4, p5]) = seam_stub_prism(&mut topo, 7.90);
    let refused_laws = vec![
        (edge_at(&topo, refused, p1, p2), RadiusLaw::Constant(8.0)),
        (edge_at(&topo, refused, p5, p1), RadiusLaw::Constant(8.0)),
        (edge_at(&topo, refused, p3, p4), RadiusLaw::Constant(2.0)),
    ];
    match fillet_v2_variable(&mut topo, refused, refused_laws) {
        Err(brepkit_operations::OperationsError::Blend(
            brepkit_blend::BlendError::RadiusTooLarge { max_radius, .. },
        )) => {
            // stub 0.05 + continuation 7.85 = 7.90 < 8.
            assert!((max_radius - 7.90).abs() < 0.01, "got {max_radius}");
        }
        Err(other) => panic!("a run below the radius must be refused, got {other}"),
        Ok(result) => panic!(
            "a run below the radius must be refused, got a solid with {} successes",
            result.succeeded.len()
        ),
    }

    let mut topo = Topology::new();
    let (built, [p1, p2, p3, p4, p5]) = seam_stub_prism(&mut topo, 12.0);
    let built_laws = vec![
        (edge_at(&topo, built, p1, p2), RadiusLaw::Constant(8.0)),
        (edge_at(&topo, built, p5, p1), RadiusLaw::Constant(8.0)),
        (edge_at(&topo, built, p3, p4), RadiusLaw::Constant(2.0)),
    ];
    let result = fillet_v2_variable(&mut topo, built, built_laws)
        .expect("a run above the radius must build");
    assert_genuinely_filleted(
        &topo,
        &result,
        3,
        &[8.0, 2.0],
        NEAR_BOUNDARY_RUNOUT_PATCH_GAP,
        "run just above the radius",
    );
}

/// The volume the three rounds leave on the seam prism with `P4` at
/// `(20, 20)`, worked out apart from the B-Rep. The prism loses the `r1`
/// rounds along `P1-P2` and `P5-P1` (each a prism of the square corner less
/// a quarter disc; at `P1` they share a part of `r1^3 (5/3 - pi/2)`) and the
/// `r2` round along `P3-P4`. The `r1` round along `P1-P2` also takes a part
/// of the `r2` round's corner: at height `z` that corner is `r2 (1 - sin phi)`
/// wide (`z = r2 (1 - cos phi)`), and the `r1` round takes it from `y = stub`
/// to where its arc is at that height.
fn seam_prism_volume(r1: f64, r2: f64) -> f64 {
    20.0_f64.powi(3) - seam_prism_taken(r1, r2)
}

/// The volume the three rounds take off the seam prism (see
/// [`seam_prism_volume`]), whatever its height above `r1`.
fn seam_prism_taken(r1: f64, r2: f64) -> f64 {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
    const STEPS: u32 = 20_000;
    let (side, stub) = (20.0_f64, 0.05);
    let corner = |r: f64| r * r * (1.0 - FRAC_PI_4);
    let shared = r1.powi(3) * (5.0 / 3.0 - FRAC_PI_2);
    let step = FRAC_PI_2 / f64::from(STEPS);
    let mut both = 0.0;
    for k in 0..STEPS {
        let phi = (f64::from(k) + 0.5) * step;
        let z = r2 * (1.0 - phi.cos());
        let width = r2 * (1.0 - phi.sin());
        let reach = r1 - (r1 * r1 - (r1 - z).powi(2)).sqrt();
        both += width * (reach - stub) * r2 * phi.sin() * step;
    }
    (2.0 * corner(r1) * side - shared) + corner(r2) * (side - stub) - both
}

/// The seam prism in a pose, with a function that carries its points there.
fn posed_seam_prism(
    topo: &mut Topology,
    height: f64,
    name: &str,
) -> (SolidId, [Point3; 5], impl Fn(Point3) -> Point3 + use<>) {
    let turn = Mat4::translation(3.0, -1.0, 2.0) * Mat4::rotation_z(0.4) * Mat4::rotation_x(0.6);
    let (solid, points) = seam_stub_prism_of_height(topo, 20.0, height);
    let solid = match name {
        "turned" => {
            transform_solid(topo, solid, &turn).unwrap();
            solid
        }
        "mirrored" => mirror(
            topo,
            solid,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap(),
        _ => solid,
    };
    let mirrored = name == "mirrored";
    let turned = name == "turned";
    let carry = move |p: Point3| -> Point3 {
        if turned {
            turn.mul_point(p)
        } else if mirrored {
            Point3::new(-p.x(), p.y(), p.z())
        } else {
            p
        }
    };
    (solid, points, carry)
}

/// The crossing of the rounds past the stub does not depend on the height of
/// the walls they end on. On a prism 4000 high, the `r1` round meets the edge
/// between the two wall faces 7.1 above the floor, a small part of that
/// edge's length.
#[test]
fn a_tall_wall_takes_the_crossing_as_a_short_one() {
    let (r1, r2, height) = (8.0, 2.0, 4000.0);
    for name in ["upright", "turned", "mirrored"] {
        let mut topo = Topology::new();
        let (solid, [p1, p2, p3, p4, p5], carry) = posed_seam_prism(&mut topo, height, name);
        let edge_laws = vec![
            (
                edge_at(&topo, solid, carry(p1), carry(p2)),
                RadiusLaw::Constant(r1),
            ),
            (
                edge_at(&topo, solid, carry(p5), carry(p1)),
                RadiusLaw::Constant(r1),
            ),
            (
                edge_at(&topo, solid, carry(p3), carry(p4)),
                RadiusLaw::Constant(r2),
            ),
        ];
        let result = fillet_v2_variable(&mut topo, solid, edge_laws).unwrap();
        assert_genuinely_filleted(&topo, &result, 3, &[r1, r2], 0, name);
        let taken = 400.0 * height - solid_volume(&topo, result.solid, 0.001).unwrap();
        let truth = seam_prism_taken(r1, r2);
        assert!(
            (taken - truth).abs() < 1e-5 * truth,
            "{name}: the rounds take {taken}, expected {truth}"
        );
    }
}

/// The rounds cross past the stub at the floor and at the top: the second
/// crossing ends its `r1` round on the stub face as the first one left it.
#[test]
fn crossings_at_both_ends_of_the_stub_wall() {
    let (r1, r2, height) = (8.0, 2.0, 20.0);
    let truth = 400.0 * height - 2.0 * seam_prism_taken(r1, r2);
    for name in ["upright", "turned", "mirrored"] {
        let mut topo = Topology::new();
        let (solid, [p1, p2, p3, p4, p5], carry) = posed_seam_prism(&mut topo, height, name);
        let up = |p: Point3| carry(Point3::new(p.x(), p.y(), height));
        let mut edge_laws = Vec::new();
        for level in [&carry as &dyn Fn(Point3) -> Point3, &up] {
            edge_laws.extend([
                (
                    edge_at(&topo, solid, level(p1), level(p2)),
                    RadiusLaw::Constant(r1),
                ),
                (
                    edge_at(&topo, solid, level(p5), level(p1)),
                    RadiusLaw::Constant(r1),
                ),
                (
                    edge_at(&topo, solid, level(p3), level(p4)),
                    RadiusLaw::Constant(r2),
                ),
            ]);
        }
        let result = fillet_v2_variable(&mut topo, solid, edge_laws).unwrap();
        assert_genuinely_filleted(&topo, &result, 6, &[r1, r2], 0, name);
        let volume = solid_volume(&topo, result.solid, 0.001).unwrap();
        assert!(
            (volume - truth).abs() < 1e-6 * truth,
            "{name}: volume {volume}, expected {truth}"
        );
        for (p, expected) in [
            (Point3::new(19.5, 1.0, 1.0), PointClassification::Outside),
            (Point3::new(19.5, 1.0, 19.0), PointClassification::Outside),
            (Point3::new(19.99, 0.02, 10.0), PointClassification::Inside),
        ] {
            let read = classify_point(&topo, result.solid, carry(p), 0.01, 1e-7).unwrap();
            assert_eq!(read, expected, "{name}: {p:?}");
        }
    }
}

/// The `r1` round along `P1-P2` and the `r2` round along `P3-P4` cross past
/// the stub, upright, turned and mirrored. The solid keeps the volume the
/// rounds leave, the corner both rounds take is outside it, and the
/// material below their crease and above the stub is inside it.
#[test]
fn rounds_that_cross_past_the_stub_keep_the_rounded_volume() {
    let (r1, r2) = (8.0, 2.0);
    let truth = seam_prism_volume(r1, r2);
    for name in ["upright", "turned", "mirrored"] {
        let turn =
            Mat4::translation(3.0, -1.0, 2.0) * Mat4::rotation_z(0.4) * Mat4::rotation_x(0.6);
        let carry = |p: Point3| -> Point3 {
            match name {
                "turned" => turn.mul_point(p),
                "mirrored" => Point3::new(-p.x(), p.y(), p.z()),
                _ => p,
            }
        };
        let mut topo = Topology::new();
        let (solid, [p1, p2, p3, p4, p5]) = seam_stub_prism(&mut topo, 20.0);
        let solid = match name {
            "turned" => {
                transform_solid(&mut topo, solid, &turn).unwrap();
                solid
            }
            "mirrored" => mirror(
                &mut topo,
                solid,
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
            )
            .unwrap(),
            _ => solid,
        };
        let edge_laws = vec![
            (
                edge_at(&topo, solid, carry(p1), carry(p2)),
                RadiusLaw::Constant(r1),
            ),
            (
                edge_at(&topo, solid, carry(p5), carry(p1)),
                RadiusLaw::Constant(r1),
            ),
            (
                edge_at(&topo, solid, carry(p3), carry(p4)),
                RadiusLaw::Constant(r2),
            ),
        ];
        let result = fillet_v2_variable(&mut topo, solid, edge_laws).unwrap();
        assert_genuinely_filleted(&topo, &result, 3, &[r1, r2], 0, name);

        let faces = solid_faces(&topo, result.solid).unwrap();
        let cylinders = faces
            .iter()
            .filter(|&&face| matches!(topo.face(face).unwrap().surface(), FaceSurface::Cylinder(_)))
            .count();
        assert_eq!(
            (faces.len(), cylinders),
            (10, 3),
            "{name}: seven planes and the three rounds, with no patch"
        );
        let volume = solid_volume(&topo, result.solid, 0.001).unwrap();
        assert!(
            (volume - truth).abs() < 1e-6 * truth,
            "{name}: volume {volume}, expected {truth}"
        );
        for (p, expected) in [
            (Point3::new(19.5, 1.0, 1.0), PointClassification::Outside),
            (Point3::new(19.8, 5.0, 0.5), PointClassification::Outside),
            (Point3::new(19.0, 5.0, 1.5), PointClassification::Inside),
            (Point3::new(19.99, 0.02, 10.0), PointClassification::Inside),
        ] {
            let read = classify_point(&topo, result.solid, carry(p), 0.01, 1e-7).unwrap();
            assert_eq!(read, expected, "{name}: {p:?}");
        }
    }
}

/// The seam is recognized from the continuation contour's *tangent at the
/// shared vertex*. This fixture's continuation leaves `P3` exactly along the
/// stub's direction on a circle arc and then curves away, so its chord points
/// somewhere else entirely. Judged by the chord the seam is missed and the
/// 0.05mm stub is measured alone (`RadiusTooLarge`); judged by the tangent
/// the run is `stub + arc length`, comfortably above the radius.
///
/// The round's end on the plane `x = 20` still spans the stub, and past it
/// the wall curves away from that plane, so the end cannot lie on the wall
/// there. The build refuses that end; it does not close it with a patch
/// that lies off the part.
#[test]
fn curved_continuation_is_recognized_from_its_tangent() {
    let mut topo = Topology::new();
    let (solid, [p1, p2, p3, p4]) = seam_stub_prism_with_curved_continuation(&mut topo);

    let edge_laws = vec![
        (edge_at(&topo, solid, p1, p2), RadiusLaw::Constant(8.0)),
        (
            edge_at(&topo, solid, p1, Point3::new(0.0, 20.0, 0.0)),
            RadiusLaw::Constant(8.0),
        ),
        (edge_at(&topo, solid, p3, p4), RadiusLaw::Constant(2.0)),
    ];

    match fillet_v2_variable(&mut topo, solid, edge_laws) {
        Err(brepkit_operations::OperationsError::Blend(
            brepkit_blend::BlendError::PlanningFailure { reason },
        )) => assert!(
            reason.contains("runs past a short edge"),
            "the end on the curved wall must be refused as such, got {reason}"
        ),
        Err(other) => panic!("the seam is admissible, so only its end may refuse, got {other}"),
        Ok(result) => panic!(
            "a round whose end runs past the stub on to a curved wall must be refused, \
             got a solid with {} successes",
            result.succeeded.len()
        ),
    }
}

/// The stub's own length alone (well under `r1`) must still refuse a corner
/// with no seam past it: the seam measure must not become a blanket bypass of
/// `RadiusTooLarge`. Here `P3-P4` is left unselected (no independent contour
/// continues past the stub), so the stub really is the hard runout boundary
/// and `r1` genuinely does not fit.
#[test]
fn stub_with_no_continuing_contour_is_still_refused() {
    let mut topo = Topology::new();
    let (solid, [p1, p2, _p3, _p4, p5]) = seam_stub_prism(&mut topo, 20.0);

    let p1p2 = edge_at(&topo, solid, p1, p2);
    let p5p1 = edge_at(&topo, solid, p5, p1);

    let r1 = 8.0;
    let edge_laws = vec![
        (p1p2, RadiusLaw::Constant(r1)),
        (p5p1, RadiusLaw::Constant(r1)),
    ];

    // No independent contour past the stub: this must still fail, either by
    // an explicit error or by never reporting a spuriously closed result. It
    // must never come back watertight, since that would mean a genuinely
    // oversized request was silently admitted.
    if let Ok(result) = fillet_v2_variable(&mut topo, solid, edge_laws) {
        assert_ne!(
            free_edge_count(&topo, result.solid),
            0,
            "a genuinely oversized request with no seam past the stub must not \
             be silently admitted as a closed solid"
        );
    }
}
