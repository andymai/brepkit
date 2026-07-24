//! Regression: a planar face whose inner wires NEST must tessellate to its
//! true area, not to the outer boundary.
//!
//! Captured from the gridfinity layout tool's `3x3 O-shape (ring)` custom
//! shape, in its minimal broken configuration (flat base, no stacking lip).
//! The bin exported closed-but-non-manifold (88 folded edges) in 64ms — far
//! too fast for a boolean fallback, and indeed the B-Rep is fine: 47 faces,
//! 24 cylinder + 23 plane, no free and no over-shared edges.
//!
//! Everything went wrong on ONE face, the z=21 wall top, whose three inner
//! wires nest: the cavity opening contains the island band around the central
//! 1u hole, which contains that hole. Two separate defects stacked there:
//!
//!   1. Both solid tessellation paths seeded hole flood-removal at each inner
//!      wire's vertex CENTROID. A gridfinity bin is centred on the origin, so
//!      all three concentric wires share the centroid (0,0) — the first flood
//!      took the innermost cell and the other two found it already gone, so
//!      the 13152-unit cavity was never removed at all.
//!   2. Removing a cell per inner wire is wrong regardless of seeding: nesting
//!      alternates material and void, so only ODD-depth wires bound a hole.
//!      Seeding correctly but removing all three erased the island band too.
//!
//! Stored winding cannot classify them — a boolean can emit a hole wound like
//! its outer — so `hole_removal_seeds` decides by geometric nesting depth.
//!
//! Cell areas on that face (outer ~15736): outer..I27 = 595 material,
//! I27..I28 = 13153 void, I28..I29 = 198 material, inside I29 = 1790 void.
//! Before the fix the mesh covered 13944; the true area is ~793 plus the
//! chord approximation of the rounded corners.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_operations::tessellate::tessellate_solid_with_tolerance;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

#[test]
fn oring_nested_hole_face_tessellates_watertight() {
    let text = std::fs::read_to_string(fixture("oring_nested_holes.step")).unwrap();
    let mut topo = Topology::new();
    let solids = brepkit_io::step::reader::read_step(&text, &mut topo).unwrap();
    assert_eq!(solids.len(), 1);
    let sid = solids[0];

    // The B-Rep was never the problem — assert it stays analytic and manifold
    // so a future failure here is read as a boolean regression, not this bug.
    let faces = solid_faces(&topo, sid).unwrap();
    let curved = faces
        .iter()
        .filter(|&&fid| topo.face(fid).unwrap().surface().type_tag() != "plane")
        .count();
    assert!(
        curved >= 20 && faces.len() < 80,
        "expected an analytic result, got {curved} curved of {} faces",
        faces.len()
    );

    let mesh = tessellate_solid_with_tolerance(&topo, sid, 0.01, 0.5).unwrap();

    // Position-welded edge use: 1 == a hole, >2 == a fold. The by-edge-id
    // B-Rep gate cannot see either, which is why this asserts on the mesh.
    let q = |v: f64| (v * 1e4).round() as i64;
    let mut edge_use: HashMap<[i64; 6], usize> = HashMap::new();
    for tri in mesh.indices.chunks(3) {
        let p: Vec<[i64; 3]> = tri
            .iter()
            .map(|&i| {
                let pt = mesh.positions[i as usize];
                [q(pt.x()), q(pt.y()), q(pt.z())]
            })
            .collect();
        for i in 0..3 {
            let (a, b) = (p[i], p[(i + 1) % 3]);
            let k = if a <= b {
                [a[0], a[1], a[2], b[0], b[1], b[2]]
            } else {
                [b[0], b[1], b[2], a[0], a[1], a[2]]
            };
            *edge_use.entry(k).or_default() += 1;
        }
    }
    let boundary = edge_use.values().filter(|&&c| c == 1).count();
    let folded = edge_use.values().filter(|&&c| c > 2).count();
    assert_eq!(
        boundary, 0,
        "mesh must be closed, got {boundary} boundary edges"
    );
    assert_eq!(
        folded, 0,
        "mesh must be manifold, got {folded} folded edges"
    );

    // Area is the discriminating check: leaving the cavity filled covered
    // 13944, and erasing the island band covered only 590. Both pass the
    // edge-use asserts above on their own, so pin the area too.
    let mut top_area = 0.0;
    for tri in mesh.indices.chunks(3) {
        let p: Vec<_> = tri.iter().map(|&i| mesh.positions[i as usize]).collect();
        if p.iter().all(|q| (q.z() - 21.0).abs() < 1e-6) {
            top_area += (p[1] - p[0]).cross(p[2] - p[0]).length() * 0.5;
        }
    }
    assert!(
        (top_area - 800.0).abs() < 25.0,
        "wall-top ring should cover ~800 (595 outer band + 198 island band), got {top_area:.3}"
    );
}
