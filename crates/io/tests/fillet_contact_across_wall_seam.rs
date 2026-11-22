//! A plate 240 thick whose wall is one closed B-spline surface, the two
//! B-spline pieces of its bottom rim along that wall rounded at r = 50. One
//! round's contact on the wall runs across the wall's seam line.
//!
//! The trim kept that contact as one edge, from one side of the seam to the
//! other, which no run of the wall's outline between the seam's copies
//! could take, so the wall's trim failed and the fillet with it. The
//! contact is now cut where it crosses the seam: each piece lies on one
//! side of it, and the band takes both pieces as its side.
//!
//! Read from STEP, as a file gives the part.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology) -> SolidId {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/spline_plate_two_rounds.step");
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

/// The edge of `solid` passing nearest `p`.
fn edge_through(topo: &Topology, solid: SolidId, p: Point3) -> EdgeId {
    let off = |e: EdgeId| {
        let edge = topo.edge(e).unwrap();
        let (a, b) = (
            topo.vertex(edge.start()).unwrap().point(),
            topo.vertex(edge.end()).unwrap().point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        (0..=20_000)
            .map(|k| {
                let t = (t1 - t0).mul_add(f64::from(k) / 20_000.0, t0);
                (edge.curve().evaluate_with_endpoints(t, a, b) - p).length()
            })
            .fold(f64::MAX, f64::min)
    };
    let edges = solid_edges(topo, solid).unwrap();
    let best = edges
        .iter()
        .copied()
        .min_by(|&a, &b| off(a).total_cmp(&off(b)))
        .unwrap();
    assert!(off(best) < 0.05, "no edge through {p:?}: {}", off(best));
    best
}

/// Each pcurve of `solid` that does not run with its edge by more than the
/// edge itself stands off that pcurve's face: the same-parameter check of
/// each edge and face, read at 11 points along the edge, at 1e-4 past the
/// farthest of 201 points of the edge from that face's surface. (A walked
/// contact is a fit, and stands off its wall by up to 8.7e-4 on this part;
/// the band it bounds follows it exactly.)
fn pcurves_off_their_edges(topo: &Topology, solid: SolidId) -> Vec<String> {
    use brepkit_check::validate::check_edge_same_parameter;
    let mut off = Vec::new();
    for face in solid_faces(topo, solid).unwrap() {
        let data = topo.face(face).unwrap();
        let surface = data.surface();
        for wire in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
            for oe in topo.wire(wire).unwrap().edges() {
                let edge_id = oe.edge();
                if topo.pcurves().get(edge_id, face).is_none() {
                    continue;
                }
                let edge = topo.edge(edge_id).unwrap();
                let (a, b) = (
                    topo.vertex(edge.start()).unwrap().point(),
                    topo.vertex(edge.end()).unwrap().point(),
                );
                let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                let stands_off = (0..=200)
                    .filter_map(|k| {
                        let p = edge.curve().evaluate_with_endpoints(
                            (t1 - t0).mul_add(f64::from(k) / 200.0, t0),
                            a,
                            b,
                        );
                        let (u, v) = surface.project_point(p)?;
                        Some((surface.evaluate(u, v)? - p).length())
                    })
                    .fold(0.0_f64, f64::max);
                for issue in
                    check_edge_same_parameter(topo, edge_id, face, 1e-4 + stands_off).unwrap()
                {
                    off.push(format!(
                        "edge {} on face {}: {} (the edge stands {stands_off:.2e} off)",
                        edge_id.index(),
                        face.index(),
                        issue.description
                    ));
                }
            }
        }
    }
    off
}

#[test]
fn a_contact_across_the_wall_s_seam_is_cut_there() {
    let mut topo = Topology::new();
    let plate = load(&mut topo);
    let picks = [
        Point3::new(-415.159_424_679_209_7, 351.839_643_089_86, -120.0),
        Point3::new(240.998_040_990_944_65, 69.993_421_862_153_55, -120.0),
    ]
    .map(|p| edge_through(&topo, plate, p));
    let result = fillet_v2(&mut topo, plate, &picks, 50.0).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    // The part's six faces and the two bands.
    assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), 8);
    let off = pcurves_off_their_edges(&topo, result.solid);
    assert!(off.is_empty(), "{off:?}");
    let mesh = tessellate_solid(&topo, result.solid, 0.5).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    // As a reference kernel measures it.
    let volume = solid_volume(&topo, result.solid, 0.1).unwrap();
    let expected = 93_285_967.06;
    assert!(
        (volume - expected).abs() < 1e-5 * expected,
        "volume {volume}, expected {expected}"
    );
}

/// The same part at a hundredth and a ten-thousandth of its size: the
/// contact's pcurve on the wall is read along the contact, each point from
/// the one before, so a point the search of the whole wall misses at that
/// size does not fail the trim.
#[test]
fn a_contact_across_a_small_wall_s_seam_is_cut_there() {
    for scale in [1e-2, 1e-4] {
        let mut topo = Topology::new();
        let plate = load(&mut topo);
        let picks = [
            Point3::new(-415.159_424_679_209_7, 351.839_643_089_86, -120.0),
            Point3::new(240.998_040_990_944_65, 69.993_421_862_153_55, -120.0),
        ]
        .map(|p| edge_through(&topo, plate, p));
        transform_solid(&mut topo, plate, &Mat4::scale(scale, scale, scale)).unwrap();
        let result = fillet_v2(&mut topo, plate, &picks, 50.0 * scale).unwrap();
        assert!(result.failed.is_empty(), "{scale}: {:?}", result.failed);
        let report = validate_solid(&topo, result.solid).unwrap();
        assert!(report.is_valid(), "{scale}: {:?}", report.issues);
        assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), 8);
        let off = pcurves_off_their_edges(&topo, result.solid);
        assert!(off.is_empty(), "{scale}: {off:?}");
        let mesh = tessellate_solid(&topo, result.solid, 0.5 * scale).unwrap();
        assert!(is_watertight(&mesh), "{scale}: mesh not watertight");
        let volume = solid_volume(&topo, result.solid, 0.1 * scale).unwrap() / scale.powi(3);
        let expected = 93_285_967.06;
        assert!(
            (volume - expected).abs() < 1e-5 * expected,
            "{scale}: volume {volume}, expected {expected}"
        );
    }
}

/// The check above reads each face on its own. The wall's contact stands
/// 8.7e-4 off the wall; the band follows the same edge within 1e-7 (and
/// has no pcurve along it). Given one moved 5e-4 off the edge, the band's
/// pcurve is caught, though it is within the wall's allowance.
#[test]
fn a_band_s_pcurve_off_its_edge_is_caught_beside_a_wall_it_stands_off() {
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::Vec2;
    use brepkit_topology::pcurve::PCurve;

    let mut topo = Topology::new();
    let plate = load(&mut topo);
    let picks = [
        Point3::new(-415.159_424_679_209_7, 351.839_643_089_86, -120.0),
        Point3::new(240.998_040_990_944_65, 69.993_421_862_153_55, -120.0),
    ]
    .map(|p| edge_through(&topo, plate, p));
    let result = fillet_v2(&mut topo, plate, &picks, 50.0).unwrap();
    let solid = result.solid;
    assert!(pcurves_off_their_edges(&topo, solid).is_empty());
    // How far an edge stands off a face's surface.
    let stands_off = |topo: &Topology, edge_id: EdgeId, face| -> f64 {
        let edge = topo.edge(edge_id).unwrap();
        let surface = topo.face(face).unwrap().surface();
        let (a, b) = (
            topo.vertex(edge.start()).unwrap().point(),
            topo.vertex(edge.end()).unwrap().point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        (0..=200)
            .filter_map(|k| {
                let p = edge.curve().evaluate_with_endpoints(
                    (t1 - t0).mul_add(f64::from(k) / 200.0, t0),
                    a,
                    b,
                );
                let (u, v) = surface.project_point(p)?;
                Some((surface.evaluate(u, v)? - p).length())
            })
            .fold(0.0_f64, f64::max)
    };
    let adjacency = brepkit_topology::adjacency::AdjacencyIndex::build(&topo, solid).unwrap();
    let (edge_id, band) = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .find_map(|e| {
            let faces = adjacency.faces_for_edge(e);
            let wall = faces
                .iter()
                .any(|&f| topo.pcurves().get(e, f).is_some() && stands_off(&topo, e, f) > 6e-4);
            let band = faces
                .iter()
                .copied()
                .find(|&f| stands_off(&topo, e, f) < 1e-6);
            band.filter(|_| wall).map(|band| (e, band))
        })
        .expect("a contact the wall stands off and the band follows");
    let edge = topo.edge(edge_id).unwrap().clone();
    let ends = (
        topo.vertex(edge.start()).unwrap().point(),
        topo.vertex(edge.end()).unwrap().point(),
    );
    let surface = topo.face(band).unwrap().surface().clone();
    // The edge is a piece of the band's side row, run with the band's own
    // parameter along it: its pcurve on the band is the line between its
    // ends' parameters.
    let (start, end) = (
        surface.project_point(ends.0).unwrap(),
        surface.project_point(ends.1).unwrap(),
    );
    let along = Vec2::new(end.0 - start.0, end.1 - start.1);
    let pcurve = PCurve::new(
        Curve2D::Line(
            Line2D::new(brepkit_math::vec::Point2::new(start.0, start.1), along).unwrap(),
        ),
        0.0,
        along.length(),
    );
    let middle = pcurve.evaluate(0.5 * (pcurve.t_start() + pcurve.t_end()));
    let h = 1e-6;
    let speed = (surface.evaluate(middle.x() + h, middle.y()).unwrap()
        - surface.evaluate(middle.x(), middle.y()).unwrap())
    .length()
        / h;
    let shift = Vec2::new(5e-4 / speed, 0.0);
    let moved = Curve2D::Line(
        Line2D::new(
            brepkit_math::vec::Point2::new(start.0, start.1) + shift,
            along,
        )
        .unwrap(),
    );
    topo.pcurves_mut().set(edge_id, band, pcurve.clone());
    assert!(
        pcurves_off_their_edges(&topo, solid).is_empty(),
        "the band's own pcurve"
    );
    topo.pcurves_mut().set(
        edge_id,
        band,
        PCurve::new(moved, pcurve.t_start(), pcurve.t_end()),
    );
    let off = pcurves_off_their_edges(&topo, solid);
    let named = format!("edge {} on face {}", edge_id.index(), band.index());
    assert!(off.iter().any(|o| o.starts_with(&named)), "{off:?}");
}
