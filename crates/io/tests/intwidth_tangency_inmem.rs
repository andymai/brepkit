//! Faithful regression guard: the integer-width halfSockets wall-tangency
//! family (`compartmentBuilder.scenario.manifold` `1×6`/`2×6` halfSockets
//! variants → nm=76/136/140; `1×4 2×8-comps` → nm=12).
//!
//! Operands captured from the live gridfinity tool via the boolean-capture
//! probe kernel: the 1×6 halfSockets ±40 tilted-divider bin (95 faces) and
//! one wall-adjacent half-socket from the socket-assembly fuse tree
//! (34 faces), translated +5 in z to its final-assembly position.
//!
//! A half-socket outline's r=4 corner circles are exactly TANGENT to the
//! bin wall lines, and the outline's straight runs continue along those
//! walls from the tangency points — which therefore exist as exact operand
//! vertices. Two solvers recomputed those tangential intersections instead
//! of landing on the exact points:
//!
//! 1. `Circle3D::intersect_segment` solved the near-tangent quadratic into
//!    a root pair straddling the foot by sqrt(2r·δ) (a 1e-13 residual at
//!    r=4 shifts each root a full micron) — used by both phase EE and the
//!    FF closed-circle splitter.
//! 2. Phase EF's grazing edge×surface refinement lands anywhere inside the
//!    tolerance well (distance to the surface grows only quadratically
//!    around a tangency), microns from the junction.
//!
//! Both minted near-duplicate vertices ~1e-6 from the exact ones (above
//! vertex-merge tolerance), leaving micro line edges that three faces
//! shared — one of them out-and-back — so the analytic fuse failed the
//! non-manifold gate and fell back to the mesh boolean, whose output was
//! itself non-manifold (nm=76 at export).
//!
//! Fixed by (1) collapsing the near-tangent root pair to the
//! well-conditioned double root (the foot) when the chord implies
//! sub-tolerance penetration, and (2) snapping tangential EF crossings to
//! an existing pave vertex within the angle-scaled window when that vertex
//! lies on both the surface and the edge.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean_with_evolution};
use brepkit_operations::tessellate::{
    boundary_edge_count, non_manifold_edge_count, tessellate_solid_with_tolerance,
};
use brepkit_operations::transform::transform_solid;
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
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
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
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    let mut faces_per_edge: HashMap<(Q, Q), HashSet<FaceId>> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                *occ.entry(key).or_default() += 1;
                faces_per_edge.entry(key).or_default().insert(fid);
            }
        }
    }
    let free = occ.values().filter(|&&c| c == 1).count();
    let over = faces_per_edge.values().filter(|f| f.len() > 2).count();
    (free, over)
}

#[test]
fn intwidth_wall_socket_fuse_stays_analytic_and_watertight() {
    let mut topo = Topology::new();
    let bin = load("intwidth_hs_bin.bin", &mut topo);
    let socket = load("intwidth_hs_wall_socket.bin", &mut topo);
    transform_solid(&mut topo, socket, &Mat4::translation(0.0, 0.0, 5.0)).unwrap();

    // The tool's export fuses run through the provenance path.
    let (result, _) = boolean_with_evolution(&mut topo, BooleanOp::Fuse, bin, socket).unwrap();

    // Analytic result, not a mesh fallback (pre-fix: the near-duplicate
    // tangency vertices tripped the non-manifold gate and the fuse fell
    // back to a 2000+-face all-plane mesh that was itself non-manifold).
    let fids = solid_faces(&topo, result).unwrap();
    assert!(
        fids.len() < 200,
        "expected an analytic fuse (~130 faces), got {} (mesh fallback?)",
        fids.len()
    );
    let curved = fids
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count();
    assert!(
        curved > 20,
        "expected the analytic cone/cylinder feet to survive, curved={curved}"
    );

    // No micro edges: every edge paired by exactly two faces.
    let (free, over) = edge_health(&topo, result);
    assert_eq!(free, 0, "wall-tangency rim edges must be paired");
    assert_eq!(over, 0, "no 3-face micro edges at outline↔wall tangencies");

    // Watertight and manifold at the tool's export tier (0.01 mm / 5°).
    let mesh = tessellate_solid_with_tolerance(&topo, result, 0.01, 5f64.to_radians()).unwrap();
    assert_eq!(boundary_edge_count(&mesh), 0);
    assert_eq!(non_manifold_edge_count(&mesh), 0);
}
