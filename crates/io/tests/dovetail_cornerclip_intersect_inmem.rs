//! Faithful (currently failing) guard for the baseplate dovetail tile-join
//! perf + non-manifold defect — root-caused to the corner-rounding INTERSECT,
//! NOT the connector fuse.
//!
//! The gridfinity tool's `baseplateGenerator.scenario.dovetail.test.ts`
//! "preferIdenticalPieces stays watertight on a corner tile with 2 join edges"
//! (the 2×2 A1-canonical doubled-dovetail) reported ~597 non-manifold STL edges
//! and took ~11 minutes per tile with brepkit. Capturing the tool's literal
//! kernel operands (via the `serializeSolid` wasm binding, replayed through
//! `arena_io::deserialize_solid`) showed the slowness is NOT in the dovetail
//! tongue fuse. The connector fuse is slow only because it is handed a
//! 5867-facet MESH-FALLBACK slab. That slab is produced one step earlier by the
//! corner-rounding op:
//!
//!     intersect(slab_with_pockets, rounded_rect_profile_extrude)
//!
//! The rounded-rect extrude emits its straight side walls as planar B-splines
//! (NURBS), coincident with the slab's planar walls. In raw GFA this intersect
//! goes non-manifold: ~45 faces, ~46 free + ~10 over-shared edges. The
//! operations layer recognises the planar NURBS walls and flattens them to
//! `Plane` (which removes the over-share), but the underlying intersect of two
//! coincident-walled solids whose only difference is the rounded corners STILL
//! drops most of the boundary (free≈74, only ~9 faces survive — three of the
//! four corner cylinders and most wall faces vanish). The resulting open shell
//! fails the Euler/manifold check, so the operations layer falls back to a
//! co-refinement MESH boolean → 6116 all-planar facets. That mesh slab is what
//! makes every downstream connector boolean catastrophically slow and
//! non-manifold.
//!
//! ROOT (verified, near-duplicate-vertex scan = 0, so this is the splitter /
//! face-selection doubled-face class, NOT the #859/#907 weld-or-same-domain-
//! near-miss class): the GFA Intersect of two solids sharing large coincident
//! planar walls, differing only by an analytic rounded-corner transition
//! (plane wall → corner cylinder → plane wall), mis-classifies/mis-splits the
//! coincident walls and the corner cylinders. Most boundary faces are dropped
//! (an open shell) and a few are doubled (over-shared). Flattening the NURBS
//! walls is necessary but not sufficient — the coincident-wall + analytic-
//! corner intersect itself is broken. This is the same coincident-contact
//! splitter family as the dovetail groove CUT (#938) and the scoop cases, and
//! needs a dedicated splitter/selection fix, not a weld/dedup pass.
//!
//! The two `.bin` fixtures are the tool's EXACT operands for that intersect:
//! `_slab` = the rectangular slab cut by 4 cell pockets (38 planar faces, clean),
//! `_round` = the same-footprint rounded-rect prism (5 plane + 2 planar-NURBS
//! walls + 2 corner cylinders). Replaying `Intersect(slab, round)` through the
//! operations boolean reproduces the mesh fallback natively (no tool needed).
//!
//! Marked `#[ignore]` until the coincident-wall + analytic-corner intersect is
//! fixed: the assertions below encode the watertight+analytic+compact target.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    brepkit_io::arena_io::deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

fn edge_health(topo: &Topology, solid: SolidId) -> (usize, usize) {
    type Q = (i64, i64, i64);
    let s = 1.0e5;
    let q = |p: Point3| -> Q {
        (
            (p.x() * s).round() as i64,
            (p.y() * s).round() as i64,
            (p.z() * s).round() as i64,
        )
    };
    let mut faces_per_edge: HashMap<(Q, Q), HashSet<FaceId>> = HashMap::new();
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                faces_per_edge.entry(key).or_default().insert(fid);
                *occ.entry(key).or_default() += 1;
            }
        }
    }
    let free = occ.values().filter(|&&c| c == 1).count();
    let over = faces_per_edge.values().filter(|f| f.len() > 2).count();
    (free, over)
}

#[test]
#[ignore = "open: coincident-wall + analytic-corner intersect drops the boundary; \
            tool falls back to a 6116-facet mesh slab → slow non-manifold dovetail fuse"]
fn dovetail_corner_clip_intersect_is_watertight() {
    let mut topo = Topology::new();
    let slab = load("dovetail_cornerclip_slab.bin", &mut topo);
    let round = load("dovetail_cornerclip_round.bin", &mut topo);

    let result = boolean(&mut topo, BooleanOp::Intersect, slab, round).unwrap();

    let (free, over) = edge_health(&topo, result);
    let faces = solid_faces(&topo, result).unwrap();
    let curved = faces
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count();

    assert_eq!(
        over,
        0,
        "corner-clip intersect must be manifold (no over-shared edges); \
         got {} faces, {curved} curved, {free} free",
        faces.len()
    );
    assert_eq!(
        free,
        0,
        "corner-clip intersect must be watertight (no free edges); got {} faces",
        faces.len()
    );
    // A clean analytic result is compact: 4 corner cylinders + ~4 wall pieces +
    // top/bottom caps. A mesh fallback explodes to thousands of planar facets.
    assert!(
        faces.len() < 60,
        "expected a compact analytic result, got {} faces (mesh fallback?)",
        faces.len()
    );
    assert!(
        curved >= 4,
        "expected the 4 rounded-corner cylinders to survive analytically; \
         got {curved} curved faces (mesh fallback tessellates them away)"
    );
}
