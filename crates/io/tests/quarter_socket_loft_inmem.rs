//! The quarter-socket loft that MINTS the mixed-detail chain's directed
//! orientation mismatches (stage-capture v2, call 080: `loftWithOptions`
//! result, preserved as `quarter_socket_loft38.bin`).
//!
//! The 14-face loft solid is COMBINATORIALLY clean (edge-sense pairing
//! passes) yet its export-tolerance mesh counts 38 unmatched DIRECTED
//! half-edges, all bordering ONE cylinder side wall — a coherently
//! flipped face: its effective orientation disagrees geometrically with
//! every neighbour while the wire senses still pair. A sibling loft
//! (call 207) mints 78 the same way; the chain's cut merges them into
//! the 116 of `mixed_socket_tess_inmem.rs`.
//!
//! ROOT FIXED: the loft's coaxial Cylinder/Cone arm emitted
//! `(surface, false)` unconditionally — radial-outward equals
//! solid-outward only for a CONVEX corner arc, and a CONCAVE rounding
//! (material outside the cylinder) needs the reversal. The chord-cross
//! `outward` cannot discriminate (a concave traversal flips chord and
//! radial normal together), so the check uses material-outward =
//! traversal tangent x connect direction at the arc midpoint. Verified
//! against the captured REAL profiles (`quarter_socket_loft_profile_
//! {a,b}.bin`, thin extrusions whose top caps carry the inputs): without
//! the fix the native loft reproduces exactly 38 mismatches; with it,
//! both the 38- and 78-loft configurations mesh directed-watertight.
//!
//! Oracle note: use DIRECTED half-edge pairing (authoritative). The
//! offset-classification outwardness audit false-positives near concave
//! cylinders (see `topsocket_cut_inmem.rs`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
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

fn directed_unmatched(topo: &Topology, solid: SolidId) -> usize {
    let mesh = brepkit_operations::tessellate::tessellate_solid_with_tolerance(
        topo,
        solid,
        0.01,
        5.0_f64.to_radians(),
    )
    .unwrap();
    let mut half: HashMap<(u32, u32), usize> = HashMap::new();
    for t in mesh.indices.chunks(3) {
        for k in 0..3 {
            *half.entry((t[k], t[(k + 1) % 3])).or_default() += 1;
        }
    }
    half.keys()
        .filter(|&&(x, y)| !half.contains_key(&(y, x)))
        .count()
}

#[test]
fn quarter_socket_loft_is_combinatorially_clean() {
    // The class signature: edge-sense pairing passes on the very solid
    // whose mesh fails directed pairing.
    let mut topo = Topology::new();
    let solid = load("quarter_socket_loft38.bin", &mut topo);
    let opts = brepkit_operations::validate::ValidationOptions {
        check_orientation: true,
        ..Default::default()
    };
    let report =
        brepkit_operations::validate::validate_solid_with_options(&topo, solid, &opts).unwrap();
    assert!(
        !report
            .issues
            .iter()
            .any(|i| i.description.contains("inconsistent face orientations")),
        "loft must be combinatorially clean, got {:?}",
        report.issues
    );
}

#[test]
fn quarter_socket_loft_regenerates_directed_watertight() {
    // The fix pin: loft the REAL captured profiles (top caps of the thin
    // extrusions) and require a directed-watertight mesh (was 38
    // mismatches on one cylinder side wall).
    let mut topo = Topology::new();
    let ea = load("quarter_socket_loft_profile_a.bin", &mut topo);
    let eb = load("quarter_socket_loft_profile_b.bin", &mut topo);
    let top_face = |topo: &Topology, sid: SolidId| {
        let faces = solid_faces(topo, sid).unwrap();
        let mut best: Option<(brepkit_topology::face::FaceId, f64)> = None;
        for &fid in &faces {
            let face = topo.face(fid).unwrap();
            if !matches!(
                face.surface(),
                brepkit_topology::face::FaceSurface::Plane { .. }
            ) {
                continue;
            }
            let mut z_min = f64::MAX;
            for oe in topo.wire(face.outer_wire()).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                for vid in [e.start(), e.end()] {
                    z_min = z_min.min(topo.vertex(vid).unwrap().point().z());
                }
            }
            if best.is_none_or(|(_, bz)| z_min > bz) {
                best = Some((fid, z_min));
            }
        }
        best.map(|(fid, _)| fid).unwrap()
    };
    let fa = top_face(&topo, ea);
    let fb = top_face(&topo, eb);
    let solid = brepkit_operations::loft::loft(&mut topo, &[fa, fb]).unwrap();
    assert_eq!(
        directed_unmatched(&topo, solid),
        0,
        "loft of the real quarter-socket profiles must be directed-watertight"
    );
}

#[test]
fn quarter_socket_loft_capture_documents_the_defective_era() {
    // The CAPTURED (pre-fix) loft output stays as minted: 38 unmatched
    // directed half-edges, all bordering one cylinder side wall.
    let mut topo = Topology::new();
    let solid = load("quarter_socket_loft38.bin", &mut topo);
    assert_eq!(directed_unmatched(&topo, solid), 38);

    let mesh_faces = solid_faces(&topo, solid).unwrap();
    assert_eq!(mesh_faces.len(), 14, "captured loft has 14 faces");
    let cylinders = mesh_faces
        .iter()
        .filter(|&&fid| {
            matches!(
                topo.face(fid).unwrap().surface(),
                brepkit_topology::face::FaceSurface::Cylinder(_)
            )
        })
        .count();
    assert!(cylinders >= 1, "the owner wall is an analytic cylinder");
}
