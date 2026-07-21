//! Faithful regression guard: the lite magnet-pad fuse (the lightweight
//! export family's `4×4 stress` / `solid bin + magnet` root).
//!
//! Operands captured from the live gridfinity tool's lightweight base
//! builder and cropped to two of the base's sixteen disjoint feet (the lite
//! base at this pipeline stage is N unconnected closed solids in one shell)
//! plus the one magnet pad whose wall grazes the hot foot's socket corner.
//!
//! The pad cylinder (r=4.45) clips the socket profile's corner cone (45°,
//! apex 0.094 mm inside the pad wall) so shallowly that the cone×cylinder
//! intersection branches exit the cone patch through its ANGULAR-window
//! corner: the in-both span is a 0.097 mm piece of a 1.4 mm curve — far
//! below the extent-scaled refinement's assumed minimum crossing, so the
//! restrict pass dropped both branch curves. With the connecting pieces
//! missing, the wire builder dead-ended at the junctions and backtracked
//! into zero-area out-and-back slits (free=5, over=19), the whole fuse fell
//! back to a ~9755-face mesh, and every downstream drill inherited the
//! poison (the tool's bd=448 / bd=48 lightweight exports).
//!
//! The fix spans five layers (see `rescue_corner_crossing` in phase FF,
//! `Circle3D::intersect_circle` + the arc-boundary crossings and lens
//! midpoint split in `closed_circle_boundary_crossings`, the fit-error weld
//! plumbing in `curve_endpoints`/`find_splits_on_line`/the splitter's UV
//! reconciliation, the cylinder-face mirrored-winding retry, the
//! multi-component Fuse acceptance gate, and the endpoint-trimmed NURBS
//! domains in tessellation edge sampling). This test pins the end-to-end
//! outcome: an analytic, watertight, position-manifold fuse through the
//! public `boolean()` entry.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::tessellate::tessellate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

type Q = (i64, i64, i64);

fn quant(p: Point3) -> Q {
    let s = 1.0e5;
    (
        (p.x() * s).round() as i64,
        (p.y() * s).round() as i64,
        (p.z() * s).round() as i64,
    )
}

/// Position-quantized edge pairing over the B-Rep (free, over-shared).
fn edge_health(topo: &Topology, solid: SolidId) -> (usize, usize) {
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = quant(topo.vertex(e.start()).unwrap().point());
                let b = quant(topo.vertex(e.end()).unwrap().point());
                if a == b {
                    continue;
                }
                let key = if a <= b { (a, b) } else { (b, a) };
                *occ.entry(key).or_default() += 1;
            }
        }
    }
    (
        occ.values().filter(|&&c| c == 1).count(),
        occ.values().filter(|&&c| c > 2).count(),
    )
}

/// Position-welded mesh boundary/non-manifold counts (the export oracle).
fn mesh_health(mesh: &brepkit_operations::tessellate::TriangleMesh) -> (usize, usize) {
    let s = 1.0e4;
    let q = |i: u32| -> Q {
        let p = mesh.positions[i as usize];
        (
            (p.x() * s).round() as i64,
            (p.y() * s).round() as i64,
            (p.z() * s).round() as i64,
        )
    };
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let (ka, kb) = (q(a), q(b));
            if ka == kb {
                continue;
            }
            let key = if ka <= kb { (ka, kb) } else { (kb, ka) };
            *occ.entry(key).or_default() += 1;
        }
    }
    (
        occ.values().filter(|&&c| c == 1).count(),
        occ.values().filter(|&&c| c > 2).count(),
    )
}

#[test]
fn lite_pad_graze_fuse_stays_analytic_and_watertight() {
    let mut topo = Topology::new();
    let hot = load("lite_foot_hot.bin", &mut topo);
    let neighbor = load("lite_foot_neighbor.bin", &mut topo);
    let pad = load("lite_pad.bin", &mut topo);

    // Reassemble the two disjoint feet into ONE solid (one shell), matching
    // the lite base builder's output shape — the multi-component
    // configuration is load-bearing: the fuse result is legitimately two
    // disjoint closed manifolds, which only the multi-region acceptance
    // path can admit.
    let mut faces = solid_faces(&topo, hot).unwrap();
    faces.extend(solid_faces(&topo, neighbor).unwrap());
    let shell = topo.add_shell(Shell::new(faces).unwrap());
    let base = topo.add_solid(Solid::new(shell, Vec::new()));

    let result = boolean(&mut topo, BooleanOp::Fuse, base, pad).unwrap();

    // Analytic result, not the mesh fallback (which is ~1200 all-plane
    // faces for this crop and was itself poisoned downstream).
    let fids = solid_faces(&topo, result).unwrap();
    assert!(
        fids.len() < 200,
        "expected an analytic fuse (~125 faces), got {} (mesh fallback?)",
        fids.len()
    );
    let curved = fids
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count();
    assert!(
        curved > 20,
        "expected the socket cones/cylinders and pad wall to survive, curved={curved}"
    );

    // Position-manifold B-Rep: the pre-fix defect left the out-and-back
    // slit chains at both pad×cone junctions.
    let (free, over) = edge_health(&topo, result);
    assert_eq!((free, over), (0, 0), "free={free} over={over}");

    // Watertight mesh at export deflection: the tessellation half of the
    // fix (endpoint-trimmed NURBS sampling) is what keeps the trimmed
    // junction splines from ripping a crack along the parent curve.
    let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
    let (bd, nm) = mesh_health(&mesh);
    assert_eq!((bd, nm), (0, 0), "mesh bd={bd} nm={nm}");
}
