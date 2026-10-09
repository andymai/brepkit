//! Kumiko corner wrap from the gridfinity tool (`slideRailBuilder.test.ts`,
//! "is not carved away by a kumiko wrap either"): the base of the 8-tool
//! compound cut (op 5915 of the capture), cut by single connected pieces of
//! its lattice tools.
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
//! `kumiko_lattice_cut_wedge.bin` (a 6-face wedge at the corner seam).

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
    assert_eq!(
        cylinder_count(topo, result),
        24,
        "the base keeps its cylinders"
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

/// The strut piece's cut matches point classification of its operands
/// across the corner the strut wraps.
#[test]
fn a_lattice_strut_crossing_the_corner_cylinder_cuts_exactly() {
    let mut topo = Topology::new();
    let base = load(&mut topo, "kumiko_lattice_cut_base.bin");
    let strut = load(&mut topo, "kumiko_lattice_cut_strut.bin");
    let probe_base = copy_solid(&mut topo, base).unwrap();
    let probe_strut = copy_solid(&mut topo, strut).unwrap();
    let result = exact_cut(&mut topo, base, strut);

    let bb = solid_bounding_box(&topo, probe_strut).unwrap();
    let inside = |topo: &Topology, s: SolidId, p: Point3| {
        classify_point(topo, s, p, 0.01, 1e-7).unwrap() == PointClassification::Inside
    };
    let (nx, ny, nz) = (6, 6, 24);
    let mut mismatches = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let at = |lo: f64, hi: f64, n: usize, m: usize| {
                    #[allow(clippy::cast_precision_loss)]
                    let f = (m as f64 + 0.503) / n as f64;
                    (hi - lo).mul_add(f, lo)
                };
                let p = Point3::new(
                    at(bb.min.x(), bb.max.x(), nx, i),
                    at(bb.min.y(), bb.max.y(), ny, j),
                    at(bb.min.z(), bb.max.z(), nz, k),
                );
                let want = inside(&topo, probe_base, p) && !inside(&topo, probe_strut, p);
                if inside(&topo, result, p) != want {
                    mismatches.push(p);
                }
            }
        }
    }
    assert!(mismatches.is_empty(), "misclassified: {mismatches:?}");
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
