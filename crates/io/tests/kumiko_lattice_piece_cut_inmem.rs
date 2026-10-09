//! Mitsukude lattice on dividers from the gridfinity tool
//! (`binGenerator.scenario.dividerPatterns.test.ts`, "mitsukude lattice on
//! dividers"): the base of the 9-tool compound cut (op 5915 of the capture),
//! cut by single connected pieces of its lattice tools, alone or after the
//! piece beside them.
//!
//! The lattice pieces come from earlier booleans, and their vertices stand
//! up to 6e-4 off their own planes. Where a strut crosses the base's corner
//! cylinder, the sections meeting at one strut edge end apart unless each
//! ends on that edge's crossing of the cylinder, and a strut face whose
//! sections did not meet was split into nothing: the cut fell to the mesh
//! fallback. The wedge piece, a cylinder patch with five planes, read as a
//! whole capped pipe to the classifier, which put base faces 20 mm above the
//! wedge inside it.
//!
//! Data: `kumiko_lattice_cut_base.bin` (the cut base, 86 faces),
//! `kumiko_lattice_cut_strut.bin` (a 405-face planar lattice piece),
//! `kumiko_lattice_cut_wedge.bin` (a 6-face wedge at the corner seam),
//! `kumiko_lattice_cut_corner_strut.bin` and `kumiko_lattice_cut_wall_strut.bin`
//! (corner-band pieces crossing the inner corner cylinder),
//! `kumiko_lattice_cut_facet_strut.bin` (a corner strut whose facets end on
//! the outer corner cylinder), and pairs of a notch-cutting piece ending
//! 0.05 past a corner cylinder's tangent seam with the strut cut after it:
//! `notch_inner` with `seam_strut`, `notch_top` with `top_strut`, `notch_low`
//! with `wall_strut` and `notch_lip` with `shard`. `holed_base.bin` is the base
//! after 83 pieces, its corner cylinders notched and windowed by struts, and
//! `holed_strut.bin` the strut cut after them.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::copy::copy_solid;
use brepkit_operations::measure::{solid_bounding_box, solid_volume};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

/// Edge uses keyed by quantized geometry (endpoints and midpoint), so two ids
/// on one curve count as one edge.
#[allow(clippy::cast_possible_truncation)]
fn positional_edge_uses(topo: &Topology, solid: SolidId) -> HashMap<[i64; 9], usize> {
    let q = |p: Point3| {
        [
            (p.x() * 1e6).round() as i64,
            (p.y() * 1e6).round() as i64,
            (p.z() * 1e6).round() as i64,
        ]
    };
    let mut uses = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        let mut wires = vec![face.outer_wire()];
        wires.extend_from_slice(face.inner_wires());
        for wid in wires {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let (start, end) = (
                    topo.vertex(e.start()).unwrap().point(),
                    topo.vertex(e.end()).unwrap().point(),
                );
                let (t0, t1) = e.curve().domain_with_endpoints(start, end);
                let mid = q(e
                    .curve()
                    .evaluate_with_endpoints(0.5 * (t0 + t1), start, end));
                let (mut a, mut b) = (q(start), q(end));
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                *uses
                    .entry([a[0], a[1], a[2], b[0], b[1], b[2], mid[0], mid[1], mid[2]])
                    .or_insert(0) += 1;
            }
        }
    }
    uses
}

fn cylinder_count(topo: &Topology, solid: SolidId) -> usize {
    brepkit_topology::explorer::solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count()
}

/// Cut `base` by `tool`, requiring the exact path and a closed result.
fn exact_cut(topo: &mut Topology, base: SolidId, tool: SolidId) -> SolidId {
    let result = exact_closed_cut(topo, base, tool);
    assert_eq!(
        cylinder_count(topo, result),
        24,
        "the base keeps its cylinders"
    );
    result
}

/// Cut `base` by `tool`, requiring the exact path and a result closed by
/// position.
fn exact_closed_cut(topo: &mut Topology, base: SolidId, tool: SolidId) -> SolidId {
    let before = boolean::mesh_fallback_count();
    let result = boolean::boolean(topo, BooleanOp::Cut, base, tool).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        before,
        "the cut took the mesh fallback"
    );
    let uses = positional_edge_uses(topo, result);
    assert_eq!(
        uses.values().filter(|&&c| c != 2).count(),
        0,
        "the cut is not closed by position"
    );
    result
}

#[test]
fn kumiko_lattice_pieces_are_closed() {
    let mut topo = Topology::new();
    for name in [
        "kumiko_lattice_cut_base.bin",
        "kumiko_lattice_cut_strut.bin",
        "kumiko_lattice_cut_wedge.bin",
        "kumiko_lattice_cut_corner_strut.bin",
        "kumiko_lattice_cut_wall_strut.bin",
        "kumiko_lattice_cut_facet_strut.bin",
        "kumiko_lattice_cut_notch_inner.bin",
        "kumiko_lattice_cut_seam_strut.bin",
        "kumiko_lattice_cut_notch_top.bin",
        "kumiko_lattice_cut_top_strut.bin",
        "kumiko_lattice_cut_notch_low.bin",
        "kumiko_lattice_cut_notch_lip.bin",
        "kumiko_lattice_cut_shard.bin",
        "kumiko_lattice_cut_holed_base.bin",
        "kumiko_lattice_cut_holed_strut.bin",
    ] {
        let solid = load(&mut topo, name);
        let uses = positional_edge_uses(&topo, solid);
        assert_eq!(
            uses.values().filter(|&&c| c != 2).count(),
            0,
            "{name} is not closed by position"
        );
    }
}

/// Points over `piece`'s box that `result` classifies other than
/// `base` less `piece`, on an `n` grid.
fn misclassified(
    topo: &Topology,
    base: SolidId,
    piece: SolidId,
    result: SolidId,
    (nx, ny, nz): (usize, usize, usize),
) -> Vec<Point3> {
    let bb = solid_bounding_box(topo, piece).unwrap();
    let inside = |s: SolidId, p: Point3| {
        classify_point(topo, s, p, 0.01, 1e-7).unwrap() == PointClassification::Inside
    };
    let at = |lo: f64, hi: f64, n: usize, m: usize| {
        #[allow(clippy::cast_precision_loss)]
        let f = (m as f64 + 0.503) / n as f64;
        (hi - lo).mul_add(f, lo)
    };
    let mut wrong = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let p = Point3::new(
                    at(bb.min.x(), bb.max.x(), nx, i),
                    at(bb.min.y(), bb.max.y(), ny, j),
                    at(bb.min.z(), bb.max.z(), nz, k),
                );
                if inside(result, p) != (inside(base, p) && !inside(piece, p)) {
                    wrong.push(p);
                }
            }
        }
    }
    wrong
}

/// Cut the base by the named piece exactly and check it against point
/// classification of the operands.
fn cut_matches_operands(name: &str, grid: (usize, usize, usize)) {
    let mut topo = Topology::new();
    let base = load(&mut topo, "kumiko_lattice_cut_base.bin");
    let piece = load(&mut topo, name);
    let probe_base = copy_solid(&mut topo, base).unwrap();
    let probe_piece = copy_solid(&mut topo, piece).unwrap();
    let result = exact_cut(&mut topo, base, piece);
    let wrong = misclassified(&topo, probe_base, probe_piece, result, grid);
    assert!(wrong.is_empty(), "{name}: misclassified {wrong:?}");
}

/// Cut `target` by the named piece exactly and check it against point
/// classification of the operands.
fn piece_cut_matches_operands(
    topo: &mut Topology,
    target: SolidId,
    name: &str,
    grid: (usize, usize, usize),
) {
    let piece = load(topo, name);
    let probe_target = copy_solid(topo, target).unwrap();
    let probe_piece = copy_solid(topo, piece).unwrap();
    let result = exact_closed_cut(topo, target, piece);
    let wrong = misclassified(topo, probe_target, probe_piece, result, grid);
    assert!(wrong.is_empty(), "{name}: misclassified {wrong:?}");
}

/// Cut the base by the notch piece, then by the strut beside it.
fn second_cut_matches_operands(notch: &str, strut: &str, grid: (usize, usize, usize)) {
    let mut topo = Topology::new();
    let base = load(&mut topo, "kumiko_lattice_cut_base.bin");
    let first = load(&mut topo, notch);
    let target = exact_closed_cut(&mut topo, base, first);
    piece_cut_matches_operands(&mut topo, target, strut, grid);
}

/// The strut piece's cut matches point classification of its operands
/// across the corner the strut wraps.
#[test]
fn a_lattice_strut_crossing_the_corner_cylinder_cuts_exactly() {
    cut_matches_operands("kumiko_lattice_cut_strut.bin", (6, 6, 24));
}

/// A strut facet crosses the inner corner cylinder in a 0.48 window of a
/// 19.7-long ellipse, which a sampling scaled to the faces' boxes took two
/// samples of and read as a graze.
#[test]
fn a_strut_facet_crossing_the_inner_corner_cylinder_cuts_exactly() {
    cut_matches_operands("kumiko_lattice_cut_corner_strut.bin", (6, 6, 8));
}

/// A strut poking through the inner corner cylinder bounds a triangle of
/// sections on it, traced both ways: the island and the hole around it.
#[test]
fn a_strut_poking_through_the_inner_corner_cylinder_cuts_exactly() {
    cut_matches_operands("kumiko_lattice_cut_wall_strut.bin", (6, 6, 12));
}

/// A strut facet's section with the outer corner cylinder ran on past the
/// facet's edge by its plane's boundary margin: the window's end was bisected
/// from a sample only that margin admitted, and the section chain it closes
/// was pruned with a 0.025 gap.
#[test]
fn a_strut_facet_ending_on_the_outer_corner_cylinder_cuts_exactly() {
    cut_matches_operands("kumiko_lattice_cut_facet_strut.bin", (6, 6, 12));
}

/// A notch 0.05 deep cut at the inner corner cylinder's tangent seam reads
/// as off that cylinder: a strut's section starting at the seam inside the
/// notch crossed its edge without a vertex.
#[test]
fn a_strut_beside_a_notch_in_the_inner_corner_cylinder_cuts_exactly() {
    second_cut_matches_operands(
        "kumiko_lattice_cut_notch_inner.bin",
        "kumiko_lattice_cut_seam_strut.bin",
        (6, 6, 16),
    );
}

/// The strut's top plane lies flush with the notch's ceiling, and its section
/// with the corner cylinder re-traces the notch's arc there: the curved
/// splitter traced that duplicate beside the boundary edge and swallowed the
/// patch the strut cuts.
#[test]
fn a_strut_flush_with_a_notch_ceiling_cuts_exactly() {
    second_cut_matches_operands(
        "kumiko_lattice_cut_notch_top.bin",
        "kumiko_lattice_cut_top_strut.bin",
        (6, 6, 12),
    );
}

/// Four planes meet within 0.003 where the notch's ceiling, the wall and the
/// strut's faces cross the seam: an edge crossing taken 0.0015 outside a
/// strut face, and a strut edge leaf crossing the ceiling at 11 degrees taken
/// as lying in it, planted vertices off the faces they bound.
#[test]
fn a_strut_crossing_a_notch_ceiling_at_the_seam_cuts_exactly() {
    second_cut_matches_operands(
        "kumiko_lattice_cut_notch_low.bin",
        "kumiko_lattice_cut_wall_strut.bin",
        (6, 6, 12),
    );
}

/// A notch's 0.004-long arc on the corner cylinder, walked in reverse, read
/// as its complement: the seam vertex just past its end split it into a
/// detour through the seam.
#[test]
fn a_shard_beside_a_notch_arc_at_the_seam_cuts_exactly() {
    second_cut_matches_operands(
        "kumiko_lattice_cut_notch_lip.bin",
        "kumiko_lattice_cut_shard.bin",
        (6, 6, 6),
    );
}

/// A corner cylinder notched and windowed by earlier struts reads its
/// notches and windows off its wires, not its `v` window and angular gap.
#[test]
fn a_strut_on_a_holed_corner_cylinder_cuts_exactly() {
    let mut topo = Topology::new();
    let target = load(&mut topo, "kumiko_lattice_cut_holed_base.bin");
    piece_cut_matches_operands(
        &mut topo,
        target,
        "kumiko_lattice_cut_holed_strut.bin",
        (6, 6, 16),
    );
}

/// The wedge piece's cut removes exactly the part of the wedge inside the
/// base.
#[test]
fn a_wedge_at_the_corner_seam_cuts_exactly() {
    let mut topo = Topology::new();
    let base = load(&mut topo, "kumiko_lattice_cut_base.bin");
    let wedge = load(&mut topo, "kumiko_lattice_cut_wedge.bin");
    let (base2, wedge2) = (
        copy_solid(&mut topo, base).unwrap(),
        copy_solid(&mut topo, wedge).unwrap(),
    );
    let base_volume = solid_volume(&topo, base, 0.01).unwrap();
    let cut = exact_cut(&mut topo, base, wedge);
    let common = boolean::boolean(&mut topo, BooleanOp::Intersect, base2, wedge2).unwrap();
    let (cut_volume, common_volume) = (
        solid_volume(&topo, cut, 0.01).unwrap(),
        solid_volume(&topo, common, 0.01).unwrap(),
    );
    assert!(common_volume > 0.1, "the wedge reaches into the base");
    assert!(
        (base_volume - cut_volume - common_volume).abs() < 1e-6,
        "{base_volume} - {cut_volume} != {common_volume}"
    );
}
