//! Faithful guard for the doubled-dovetail CORNER-tile tongue relief cut.
//!
//! The gridfinity tool's 2×2 A1-canonical corner baseplate with
//! `preferIdenticalPieces` exported STLs with nm=2: each doubled tongue —
//! cut(trapezoid tongue prism, corner socket pocket) — mesh-fell-back and the
//! fallback itself was open (bnd=5), poisoning the whole plate into a
//! 1400-face fallback export. The paired tongue sits offset by exactly the
//! socket corner radius, so it straddles the tangency meridian where the
//! socket wall plane hands off to the corner cylinder (and the mouth chamfer
//! to the corner cone) — the recurring tangential-contact conditioning class.
//! Three stacked roots, fixed bottom-up:
//!
//! 1. FF raw-curve AABB pre-filter under-sampling — the flank×cone marched
//!    conic spans the unbounded cone (~30mm) while its true in-both span is
//!    ~2mm; 16 uniform samples missed it and the pair vanished before the
//!    exact open-conic clip ever ran (the mirrored nub survived by sampling
//!    luck). Fixed by adaptive re-sampling scaled to the smaller positive
//!    face-AABB dimension.
//! 2. `trim_open_curve_to_plane_face_lines` clipped the conic to the plane
//!    face's boundary lines and the cone's angular-window rulings but NOT the
//!    cone patch's axial v-range — the kept piece overshot the top rim circle
//!    and dangled, so the splitter's pendant filter removed the whole section
//!    chain. Fixed by bisecting v(t) to exact rim crossings.
//! 3. `find_splits_on_circle` normalized split points against the CCW
//!    start→end span (`domain_with_endpoints`), but a rim quarter-arc
//!    traversed clockwise covers the COMPLEMENT arc — on-arc split points
//!    normalized outside [0,1] and the rim never split at the section
//!    endpoints. Fixed by disambiguating the true arc via the edge's own UV
//!    midpoint (rim pcurves are straight v=const segments) and mirroring
//!    parameters back into traversal order.
//!
//! Fixtures are the tool's EXACT serialized operands for both corner nubs
//! (right-edge and back-edge join of the 2×2 corner tile).

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

fn check_nub(topo: &Topology, result: SolidId, label: &str) {
    let faces = solid_faces(topo, result).unwrap();
    let count_of = |tag: &str| {
        faces
            .iter()
            .filter(|&&f| topo.face(f).unwrap().surface().type_tag() == tag)
            .count()
    };
    assert_eq!(
        free_edges(topo, result),
        0,
        "{label}: relieved corner nub must be watertight and manifold; got {} faces",
        faces.len()
    );
    assert_eq!(
        faces.len(),
        10,
        "{label}: relief cut must produce the 10-face nub (tongue minus the socket-mouth corner)"
    );
    assert_eq!(
        count_of("cone"),
        1,
        "{label}: the socket's corner cone wall must survive as one analytic face"
    );
    assert_eq!(
        count_of("cylinder"),
        1,
        "{label}: the socket's corner cylinder band must survive as one analytic face"
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
    check_nub(&topo, result, "boolean");

    let mut topo2 = Topology::new();
    let tongue2 = load(tongue_name, &mut topo2);
    let cutter2 = load(cutter_name, &mut topo2);
    let (result2, _) =
        boolean_with_evolution(&mut topo2, BooleanOp::Cut, tongue2, cutter2).unwrap();
    check_nub(&topo2, result2, "boolean_with_evolution");
}

#[test]
fn doubled_corner_nub_right_edge() {
    run_pair(
        "dovetail_dblcorner_tongue_0.bin",
        "dovetail_dblcorner_cutter_0.bin",
    );
}

#[test]
fn doubled_corner_nub_back_edge() {
    run_pair(
        "dovetail_dblcorner_tongue_1.bin",
        "dovetail_dblcorner_cutter_1.bin",
    );
}
