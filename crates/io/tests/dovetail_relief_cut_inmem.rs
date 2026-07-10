//! Faithful guard for the doubled-dovetail tongue relief cut.
//!
//! The gridfinity tool's doubled-dovetail baseplate interior tile exported
//! STLs with nm=21: each relieved nub (cut(trapezoid tongue prism, tapered
//! socket pocket)) arrived at the connector fuse already broken (bnd=13-15
//! nm=1-2 per nub, through BOTH `boolean()` and `boolean_with_evolution`,
//! from watertight operands). Four stacked roots, fixed bottom-up:
//!
//! 1. `restrict_curves_to_faces` graze refinement — the 24-sample in-both
//!    test dropped a real socket-mouth corner circle crossing (~8° subtended
//!    on a 2 mm tongue face); re-test at a density scaled to the smaller
//!    face extent before dropping.
//! 2. `trim_open_curve_to_plane_face_lines` — an open marched-NURBS conic
//!    (plane × cone) spans the whole cone extent; clip it to exact crossings
//!    with the plane face's straight boundary edges AND the cone partner's
//!    angular-window rulings, trimming the stored NURBS to each kept span
//!    (`domain_with_endpoints` returns the full knot domain, so untrimmed
//!    pieces project/tessellate the whole marched conic). Plus
//!    `find_splits_on_nurbs_section` T-junction splits by sampled projection.
//! 3. Ray-cast classifier cone support — the cutter's tapered corner patches
//!    fell to the flat Newell-polygon fallback, which mis-counted crossings
//!    for sub-face interior points ~0.2 mm inside the pocket walls; two
//!    in-chunk pieces classified Outside and were kept (10 faces, 8 bad-use
//!    edges). Fixed by the analytic `FaceGeom::Cone` path.
//! 4. Tessellation NURBS edge orientation — GFA section edges can store
//!    traversal-order vertices over an unreversed curve; the samplers trusted
//!    `oe.is_forward()` plus natural domain order, folding the boundary
//!    polyline back on itself (mesh nm=6-11 on a B-Rep-clean result). The
//!    samplers now orient by endpoint alignment (`nurbs_runs_end_to_start`).
//!    Normalizing vertex order at the minting site (`instantiate_wire_edge`)
//!    instead regressed the calibrated torus-box notch landscape — do not
//!    retry that route.
//!
//! Fixtures are the tool's EXACT serialized operands for two nub positions
//! (`m38` = bp -38, `m4` = bp -4 — the mirrored variant that exercised the
//! reversed-NURBS fold on the flank instead of the tip).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean, boolean_with_evolution};
use brepkit_operations::tessellate::tessellate_solid_with_tolerance;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    brepkit_io::arena_io::deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

fn free_edges(topo: &Topology, solid: SolidId) -> usize {
    type Q = (i64, i64, i64);
    let s = 1.0e5;
    let q = |p: Point3| -> Q {
        (
            (p.x() * s).round() as i64,
            (p.y() * s).round() as i64,
            (p.z() * s).round() as i64,
        )
    };
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                *occ.entry(key).or_default() += 1;
            }
        }
    }
    occ.values().filter(|&&c| c != 2).count()
}

fn mesh_health(topo: &Topology, solid: SolidId) -> (usize, usize) {
    type Q = (i64, i64, i64);
    let mesh = tessellate_solid_with_tolerance(topo, solid, 0.01, 5.0_f64.to_radians()).unwrap();
    let q = |v: f64| (v * 1.0e4).round() as i64;
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    for t in mesh.indices.chunks(3) {
        for k in 0..3 {
            let a = t[k] as usize;
            let b = t[(k + 1) % 3] as usize;
            let vs = &mesh.positions;
            let pa = (q(vs[a].x()), q(vs[a].y()), q(vs[a].z()));
            let pb = (q(vs[b].x()), q(vs[b].y()), q(vs[b].z()));
            let key = if pa <= pb { (pa, pb) } else { (pb, pa) };
            *occ.entry(key).or_default() += 1;
        }
    }
    let bnd = occ.values().filter(|&&c| c == 1).count();
    let nm = occ.values().filter(|&&c| c > 2).count();
    (bnd, nm)
}

fn check_relief_result(topo: &Topology, result: SolidId, label: &str) {
    let faces = solid_faces(topo, result).unwrap();
    let cones = faces
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() == "cone")
        .count();
    assert_eq!(
        free_edges(topo, result),
        0,
        "{label}: relieved nub must be watertight and manifold; got {} faces",
        faces.len()
    );
    assert_eq!(
        faces.len(),
        8,
        "{label}: relief cut must produce the 8-face nub (tongue minus the pocket-corner chunk)"
    );
    assert_eq!(
        cones, 1,
        "{label}: the pocket's corner cone wall must survive as one analytic face"
    );
    let (bnd, nm) = mesh_health(topo, result);
    assert_eq!(
        (bnd, nm),
        (0, 0),
        "{label}: nub mesh must be watertight/manifold"
    );
}

fn run_pair(tongue_name: &str, cutter_name: &str) {
    let mut topo = Topology::new();
    let tongue = load(tongue_name, &mut topo);
    let cutter = load(cutter_name, &mut topo);
    let result = boolean(&mut topo, BooleanOp::Cut, tongue, cutter).unwrap();
    check_relief_result(&topo, result, "boolean");

    let mut topo2 = Topology::new();
    let tongue2 = load(tongue_name, &mut topo2);
    let cutter2 = load(cutter_name, &mut topo2);
    let (result2, _) =
        boolean_with_evolution(&mut topo2, BooleanOp::Cut, tongue2, cutter2).unwrap();
    check_relief_result(&topo2, result2, "boolean_with_evolution");
}

#[test]
fn relief_cut_tip_nub() {
    run_pair(
        "dovetail_relief_tongue_m38.bin",
        "dovetail_relief_cutter_m38.bin",
    );
}

#[test]
fn relief_cut_flank_nub() {
    run_pair(
        "dovetail_relief_tongue_m4.bin",
        "dovetail_relief_cutter_m4.bin",
    );
}
