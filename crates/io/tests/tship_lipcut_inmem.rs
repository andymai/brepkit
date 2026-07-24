//! Captured-operand regression for the gridfinity custom-shape (3×3 T) lip
//! frustum cut — the shared root of the 22-scenario custom-shape export
//! family.
//!
//! Operands captured from the live tool on 2.127.28 (both frustums
//! analytic). The T's stem-side walls sampled their interior points 8µm
//! above the coincident bottom cap (the longest boundary edge lies ON that
//! plane), where the ray-cast classifier turned unstable — mirror-image
//! walls classified Inside/Outside asymmetrically, one wall dropped, and
//! the cut fell to an OPEN mesh fallback (the T custom-shape export
//! failures, bd=88). Deeper-first interior sampling keeps the walls.
//!
//! A second defect in the SAME cut hid behind a manifold-looking result: the
//! band-bottom RING (outer ±62.75 T, inner ±61.55 T hole) was mis-classified
//! Inside the tool and dropped, so the outer bottom passed through UNSPLIT
//! alongside the tool's bottom disc — two nested same-orientation discs (a
//! DOUBLED bottom that the by-edge-id manifold gate is blind to). The ring's
//! interior sample sits in the ~1.2mm annulus, coplanar with the tool's bottom
//! cap; `classify_coincident_coplanar`'s depth probe stepped from the wedge tip
//! toward the face centroid by fractions of that distance, overshooting the
//! thin band into the hole so no valid probe was found and ray-cast (unstable
//! here) decided. With the thin-band absolute-nudge probe the ring classifies
//! Outside and is kept, so the band is single-covered: volume 6090.8, not the
//! doubled 20108.8.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

#[test]
fn t_lip_frustum_cut_is_analytic_and_watertight() {
    let mut topo = Topology::new();
    let outer = load("tship_lip_outerfrustum.bin", &mut topo);
    let inner = load("tship_lip_innerfrustum.bin", &mut topo);

    let result = boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();

    let faces = solid_faces(&topo, result).unwrap();
    let mut uses: HashMap<brepkit_topology::edge::EdgeId, usize> = HashMap::new();
    let mut curved = 0;
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        if face.surface().type_tag() != "plane" {
            curved += 1;
        }
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    let free = uses.values().filter(|&&c| c == 1).count();
    let over = uses.values().filter(|&&c| c > 2).count();
    assert_eq!(free, 0, "lip cut must be closed, got {free} free edges");
    assert_eq!(over, 0, "lip cut must be manifold, got {over} over-shared");
    assert!(
        curved >= 40 && faces.len() < 150,
        "analytic result expected, got {curved} curved of {} faces",
        faces.len()
    );

    // Single-cover guard: the band bottom (z == -2.6) is a ring, so EXACTLY ONE
    // planar face on that plane carries an inner (hole) wire and the total
    // horizontal area there equals the outer-minus-inner bottom area — not two
    // nested same-orientation discs (the doubled bottom). A doubled bottom
    // passes the by-edge-id manifold gate above, so it must be caught here.
    let mut bottom_with_hole = 0usize;
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        if face.surface().type_tag() != "plane" {
            continue;
        }
        let wire = topo.wire(face.outer_wire()).unwrap();
        let zs: Vec<f64> = wire
            .edges()
            .iter()
            .flat_map(|oe| {
                let e = topo.edge(oe.edge()).unwrap();
                [e.start(), e.end()].map(|v| topo.vertex(v).unwrap().point().z())
            })
            .collect();
        let zmin = zs.iter().copied().fold(f64::MAX, f64::min);
        let zmax = zs.iter().copied().fold(f64::MIN, f64::max);
        if (zmax - zmin).abs() < 1e-6 && (zmin + 2.6).abs() < 1e-3 && !face.inner_wires().is_empty()
        {
            bottom_with_hole += 1;
        }
    }
    assert_eq!(
        bottom_with_hole, 1,
        "band bottom must be one ring face carrying the ±61.55 hole, got {bottom_with_hole}"
    );

    let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.05).unwrap();
    assert!(
        (vol - 6090.8).abs() < 40.0,
        "lip ring volume out of band: got {vol} (doubled bottom would give ~20108)"
    );
}
