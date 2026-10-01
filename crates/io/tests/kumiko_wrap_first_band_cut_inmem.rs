//! The gridfinity tool's kumiko corner wrap (`slideRailBuilder.test.ts`, "is
//! not carved away by a kumiko wrap either"), captured on a brepkit-wasm built
//! from main: the first corner band the export compound-cuts by its 19 slot
//! boxes (194 faces, 160 of them NURBS). On that build the compound cut ran
//! 1,698 s and trapped.
//!
//! Data: `kumiko_wrap_first_band.bin` (cut base),
//! `kumiko_wrap_first_band_box_<1..19>.bin` (the captured tool list), and the
//! tool's next call on the result: `kumiko_wrap_first_band_slab_base.bin` (the
//! 19-box result in place) and `kumiko_wrap_first_band_slab.bin`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// Volumes are measured at this deflection: the band's NURBS walls lose
/// tenths of a unit at coarser ones.
const DEFLECTION: f64 = 0.002;

/// The boxes cut from the band one after another, and the volume after each;
/// each agrees within 0.001 with the previous volume less the exact
/// intersection of the band with the next box.
const CHAIN: [usize; 2] = [1, 2];
const CHAIN_VOLUMES: [f64; CHAIN.len()] = [337.043, 334.799];

/// The band less all 19 boxes; it agrees within 0.012 with the band less its
/// intersection with the fused boxes.
const COMPOUND_VOLUME: f64 = 267.526;

/// A point inside the band and inside each box (1, 2 and 17), which the cut
/// must remove, and points of band material that no box reaches.
const REMOVED: [(usize, [f64; 3]); 3] = [
    (1, [2.4191, 0.0573, 8.0013]),
    (2, [3.0923, 1.5415, 6.6517]),
    (17, [2.5138, 2.2473, 33.4310]),
];
const KEPT: [[f64; 3]; 2] = [[0.1529, 1.8115, 6.4787], [0.1529, 1.8115, 21.5934]];

/// The 19-box result less the slab; it and the exact intersection with the
/// slab sum to the band within 0.003.
const SLAB_CUT_VOLUME: f64 = 186.411;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

/// A solid with its tessellation, for point probes.
struct Probed {
    solid: SolidId,
    mesh: brepkit_operations::tessellate::TriangleMesh,
}

impl Probed {
    fn new(topo: &Topology, solid: SolidId) -> Self {
        let mesh = brepkit_operations::tessellate::tessellate_solid(topo, solid, 0.05).unwrap();
        Self { solid, mesh }
    }

    /// Whether `p` is inside, when the ray cast and the generalized winding
    /// number over the tessellation agree.
    fn inside(&self, topo: &Topology, p: [f64; 3]) -> Option<bool> {
        use brepkit_operations::classify::{PointClassification, classify_point};
        let q = brepkit_math::vec::Point3::new(p[0], p[1], p[2]);
        let ray = match classify_point(topo, self.solid, q, 0.05, 1e-7).ok()? {
            PointClassification::Inside => true,
            PointClassification::Outside => false,
            PointClassification::OnBoundary => return None,
        };
        let mut solid_angle = 0.0;
        for tri in self.mesh.indices.chunks_exact(3) {
            let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| self.mesh.positions[i as usize] - q);
            let (la, lb, lc) = (a.length(), b.length(), c.length());
            let den = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
            solid_angle += 2.0 * a.dot(b.cross(c)).atan2(den);
        }
        let wind = (solid_angle / (4.0 * std::f64::consts::PI)).abs() > 0.5;
        (ray == wind).then_some(ray)
    }
}

/// Edges used other than twice across the solid's faces.
fn bad_edge_uses(topo: &Topology, solid: SolidId) -> usize {
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    uses.values().filter(|&&n| n != 2).count()
}

#[test]
fn kumiko_wrap_first_band_fixture_is_faithful() {
    let mut topo = Topology::new();
    let band = load(&mut topo, "kumiko_wrap_first_band.bin");
    let mut census: HashMap<&str, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(&topo, band).unwrap() {
        *census
            .entry(topo.face(fid).unwrap().surface().type_tag())
            .or_insert(0) += 1;
    }
    assert_eq!(
        ["cylinder", "nurbs", "plane"].map(|k| census.get(k).copied().unwrap_or(0)),
        [14, 160, 20],
        "fixture drifted: {census:?}"
    );
    assert_eq!(bad_edge_uses(&topo, band), 0);
}

/// Box 1 clips a corner off one strut-wall patch, and box 2 then clips the
/// corner where box 1's face, the bore and that patch meet. Each section ends
/// where a smooth boundary curve is split, which the face splitter's walker
/// read as a turn and closed the section on its own reverse; box 2's arc on
/// the bore lies on it only between box 1's elliptical rim and a groove.
#[test]
fn kumiko_wrap_first_band_chain_stays_exact() {
    let mut topo = Topology::new();
    let mut cur = load(&mut topo, "kumiko_wrap_first_band.bin");
    for (i, expected) in CHAIN.into_iter().zip(CHAIN_VOLUMES) {
        let b = load(&mut topo, &format!("kumiko_wrap_first_band_box_{i}.bin"));
        let before = boolean::mesh_fallback_count();
        cur = boolean::boolean(&mut topo, BooleanOp::Cut, cur, b).unwrap();
        assert_eq!(
            boolean::mesh_fallback_count(),
            before,
            "box {i}: mesh fallback"
        );
        assert_eq!(
            bad_edge_uses(&topo, cur),
            0,
            "box {i}: open or over-shared edges"
        );
        let vol =
            brepkit_operations::measure::oriented_solid_volume(&topo, cur, DEFLECTION).unwrap();
        assert!(
            (vol - expected).abs() <= 0.01,
            "box {i}: volume {vol:.3}, expected {expected:.3}"
        );
        for (b, p) in REMOVED.into_iter().filter(|&(b, _)| b == i) {
            let probed = Probed::new(&topo, cur);
            assert_eq!(
                probed.inside(&topo, p),
                Some(false),
                "box {b}: material left"
            );
        }
    }
}

/// The tool's own call: one `compound_cut` of the band by all 19 boxes stays
/// an exact B-Rep. Box 17 meets the band's radial end plane, whose boundary
/// carries the bore and the strut grooves; the plane x plane section with it
/// must end at that boundary, not run on into the bore.
#[test]
fn kumiko_wrap_first_band_compound_cut_stays_exact() {
    let mut topo = Topology::new();
    let band = load(&mut topo, "kumiko_wrap_first_band.bin");
    let tools: Vec<_> = (1..=19)
        .map(|i| load(&mut topo, &format!("kumiko_wrap_first_band_box_{i}.bin")))
        .collect();
    let before = boolean::mesh_fallback_count();
    let cut = boolean::compound_cut(&mut topo, band, &tools, BooleanOptions::default()).unwrap();
    assert_eq!(boolean::mesh_fallback_count(), before, "mesh fallback");
    assert_eq!(bad_edge_uses(&topo, cut), 0, "open or over-shared edges");
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, cut, DEFLECTION).unwrap();
    assert!(
        (vol - COMPOUND_VOLUME).abs() <= 0.01,
        "volume {vol:.3}, expected {COMPOUND_VOLUME:.3}"
    );
    let (band, cut) = (Probed::new(&topo, band), Probed::new(&topo, cut));
    for (b, p) in REMOVED {
        assert_eq!(
            band.inside(&topo, p),
            Some(true),
            "box {b}: probe off the band"
        );
        assert_eq!(cut.inside(&topo, p), Some(false), "box {b}: material left");
    }
    for p in KEPT {
        assert_eq!(
            band.inside(&topo, p),
            Some(true),
            "probe {p:?} off the band"
        );
        assert_eq!(
            cut.inside(&topo, p),
            Some(true),
            "material at {p:?} removed"
        );
    }
}

/// The tool's next call on the band: the 19-box result, moved into place, cut
/// by a slab across its middle. The slab's top face meets the band's outer
/// wall in a circle that crosses a strut groove's mouth, where the wall does
/// not exist; the arc there joined two cap regions and one was emitted twice.
#[test]
fn kumiko_wrap_first_band_slab_cut_stays_exact() {
    let mut topo = Topology::new();
    let band = load(&mut topo, "kumiko_wrap_first_band_slab_base.bin");
    let slab = load(&mut topo, "kumiko_wrap_first_band_slab.bin");
    let before = boolean::mesh_fallback_count();
    let cut = boolean::boolean(&mut topo, BooleanOp::Cut, band, slab).unwrap();
    assert_eq!(boolean::mesh_fallback_count(), before, "mesh fallback");
    assert_eq!(bad_edge_uses(&topo, cut), 0, "open or over-shared edges");
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, cut, DEFLECTION).unwrap();
    assert!(
        (vol - SLAB_CUT_VOLUME).abs() <= 0.01,
        "volume {vol:.3}, expected {SLAB_CUT_VOLUME:.3}"
    );
    let (band, cut) = (Probed::new(&topo, band), Probed::new(&topo, cut));
    for (p, kept) in [
        ([-63.5541, 38.3465, 12.4829], false),
        ([-63.5541, 38.3465, 5.9610], true),
    ] {
        assert_eq!(
            band.inside(&topo, p),
            Some(true),
            "probe {p:?} off the band"
        );
        assert_eq!(cut.inside(&topo, p), Some(kept), "probe {p:?}");
    }
}
