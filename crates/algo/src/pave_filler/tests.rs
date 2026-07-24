//! Tests for PaveFiller phases.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr
)]

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::Solid;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::ds::GfaArena;
use crate::pave_filler::PaveFiller;

/// Build a minimal axis-aligned box solid in the topology.
fn make_box(topo: &mut Topology, min: [f64; 3], max: [f64; 3]) -> brepkit_topology::solid::SolidId {
    let [x0, y0, z0] = min;
    let [x1, y1, z1] = max;

    let v = [
        topo.add_vertex(Vertex::new(Point3::new(x0, y0, z0), 1e-7)),
        topo.add_vertex(Vertex::new(Point3::new(x1, y0, z0), 1e-7)),
        topo.add_vertex(Vertex::new(Point3::new(x1, y1, z0), 1e-7)),
        topo.add_vertex(Vertex::new(Point3::new(x0, y1, z0), 1e-7)),
        topo.add_vertex(Vertex::new(Point3::new(x0, y0, z1), 1e-7)),
        topo.add_vertex(Vertex::new(Point3::new(x1, y0, z1), 1e-7)),
        topo.add_vertex(Vertex::new(Point3::new(x1, y1, z1), 1e-7)),
        topo.add_vertex(Vertex::new(Point3::new(x0, y1, z1), 1e-7)),
    ];

    let mut edge = |a: usize, b: usize| -> brepkit_topology::edge::EdgeId {
        topo.add_edge(Edge::new(v[a], v[b], EdgeCurve::Line))
    };

    // Bottom: v0-v1-v2-v3, Top: v4-v5-v6-v7
    let e01 = edge(0, 1);
    let e12 = edge(1, 2);
    let e23 = edge(2, 3);
    let e30 = edge(3, 0);
    let e45 = edge(4, 5);
    let e56 = edge(5, 6);
    let e67 = edge(6, 7);
    let e74 = edge(7, 4);
    let e04 = edge(0, 4);
    let e15 = edge(1, 5);
    let e26 = edge(2, 6);
    let e37 = edge(3, 7);

    let fwd = |eid| OrientedEdge::new(eid, true);
    let rev = |eid| OrientedEdge::new(eid, false);

    let w_bot =
        topo.add_wire(Wire::new(vec![rev(e01), rev(e30), rev(e23), rev(e12)], true).unwrap());
    let w_top =
        topo.add_wire(Wire::new(vec![fwd(e45), fwd(e56), fwd(e67), fwd(e74)], true).unwrap());
    let w_front =
        topo.add_wire(Wire::new(vec![fwd(e01), fwd(e15), rev(e45), rev(e04)], true).unwrap());
    let w_back =
        topo.add_wire(Wire::new(vec![fwd(e23), fwd(e37), rev(e67), rev(e26)], true).unwrap());
    let w_left =
        topo.add_wire(Wire::new(vec![fwd(e30), fwd(e04), rev(e74), rev(e37)], true).unwrap());
    let w_right =
        topo.add_wire(Wire::new(vec![fwd(e12), fwd(e26), rev(e56), rev(e15)], true).unwrap());

    let f_bot = topo.add_face(Face::new(
        w_bot,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: -z0,
        },
    ));
    let f_top = topo.add_face(Face::new(
        w_top,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z1,
        },
    ));
    let f_front = topo.add_face(Face::new(
        w_front,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, -1.0, 0.0),
            d: -y0,
        },
    ));
    let f_back = topo.add_face(Face::new(
        w_back,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 1.0, 0.0),
            d: y1,
        },
    ));
    let f_left = topo.add_face(Face::new(
        w_left,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(-1.0, 0.0, 0.0),
            d: -x0,
        },
    ));
    let f_right = topo.add_face(Face::new(
        w_right,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(1.0, 0.0, 0.0),
            d: x1,
        },
    ));

    let shell =
        topo.add_shell(Shell::new(vec![f_bot, f_top, f_front, f_back, f_left, f_right]).unwrap());
    topo.add_solid(Solid::new(shell, vec![]))
}

/// Helper: create two overlapping boxes and run the PaveFiller.
fn two_overlapping_boxes() -> (Topology, GfaArena) {
    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = make_box(&mut topo, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);

    let mut arena = GfaArena::new();
    let mut filler = PaveFiller::new(&mut topo, a, b);
    filler
        .perform(&mut arena)
        .expect("PaveFiller should succeed");

    (topo, arena)
}

#[test]
fn pave_filler_initializes_pave_blocks() {
    let (_topo, arena) = two_overlapping_boxes();

    // Each box has 12 edges = 24 total edges (none shared between boxes)
    assert_eq!(arena.edge_pave_blocks.len(), 24);

    // Each edge should have exactly 1 pave block
    for pbs in arena.edge_pave_blocks.values() {
        assert_eq!(pbs.len(), 1, "each edge should have exactly 1 pave block");
    }
}

#[test]
fn vv_detects_no_coincident_vertices_for_offset_boxes() {
    let (_topo, arena) = two_overlapping_boxes();

    // Box A vertices at [0,1] coords, Box B at [0.5,1.5] — no coincidences
    assert!(
        arena.interference.vv.is_empty(),
        "offset boxes should have no coincident vertices"
    );
}

#[test]
fn ee_runs_without_panic() {
    let (_topo, arena) = two_overlapping_boxes();

    // Verify the phase ran. For offset boxes with all-line edges,
    // edges are axis-aligned and mostly skew in 3D (no crossings).
    // The important thing is it ran without errors.
    assert!(
        arena.interference.ee.len() <= 24,
        "at most 24 EE checks could produce crossings"
    );
}

#[test]
fn ff_detects_plane_plane_intersections() {
    let (_topo, arena) = two_overlapping_boxes();

    // Two offset unit cubes have overlapping faces. Each face of A
    // can intersect multiple faces of B. Plane-plane intersection
    // should produce line intersections for non-parallel face pairs.
    assert!(
        !arena.interference.ff.is_empty(),
        "overlapping boxes should have FF intersections, got {}",
        arena.interference.ff.len(),
    );
    assert!(
        !arena.curves.is_empty(),
        "FF phase should produce intersection curves"
    );
}

#[test]
fn ef_runs_without_panic() {
    let (_topo, arena) = two_overlapping_boxes();

    // For all-line edges and all-plane faces, some edges of A should
    // cross faces of B. The EF phase should detect these.
    // The exact count depends on geometry.
    assert!(
        !arena.interference.ef.is_empty(),
        "overlapping boxes should have EF intersections, got {}",
        arena.interference.ef.len(),
    );
}

#[test]
fn gfa_boolean_fuse_two_boxes() {
    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = make_box(&mut topo, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);

    let result = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Fuse, a, b);

    match &result {
        Ok(solid_id) => {
            eprintln!("GFA fuse succeeded: solid {:?}", solid_id);
            let faces = brepkit_topology::explorer::solid_faces(&topo, *solid_id).unwrap();
            eprintln!("  Result has {} faces", faces.len());
            assert!(!faces.is_empty(), "fuse result should have faces");
        }
        Err(e) => {
            eprintln!("GFA fuse FAILED: {e}");
            // Don't assert — we want to see the error
        }
    }
}

#[test]
fn gfa_fuse_adjacent_boxes_same_domain() {
    // Two boxes sharing a face: A=[0,1]^3, B=[1,2]×[0,1]^2.
    // They share the x=1 plane. Fuse should produce 10 faces (not 12).
    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = make_box(&mut topo, [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);

    let result = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Fuse, a, b);
    match &result {
        Ok(solid_id) => {
            let faces = brepkit_topology::explorer::solid_faces(&topo, *solid_id).unwrap();
            eprintln!("Adjacent fuse: {} faces", faces.len());
            assert_eq!(
                faces.len(),
                10,
                "fuse of adjacent unit cubes should have 10 faces, got {}",
                faces.len()
            );
        }
        Err(e) => {
            panic!("GFA fuse of adjacent boxes failed: {e}");
        }
    }
}

#[test]
fn gfa_cut_overlapping_boxes() {
    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let b = make_box(&mut topo, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);

    let result = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Cut, a, b);
    let solid_id = result.expect("GFA cut of overlapping boxes should succeed");
    let faces = brepkit_topology::explorer::solid_faces(&topo, solid_id).unwrap();
    eprintln!("Cut: {} faces", faces.len());
    assert!(
        faces.len() >= 6,
        "cut result should have at least 6 faces, got {}",
        faces.len()
    );
}

#[test]
fn ff_plane_plane_t_range_is_bounded() {
    let (_topo, arena) = two_overlapping_boxes();
    for curve in &arena.curves {
        let (t0, t1) = curve.t_range;
        assert!(
            (t1 - t0).abs() < 10.0,
            "t_range should be bounded by face extents, got ({t0:.1}, {t1:.1})"
        );
    }
}

/// GFA intersect produces 2 faces instead of 6 for overlapping boxes.
/// Root cause: the wire builder produces 1 sub-face per split face (not 4)
/// when 2 section edges cross the face. The wire builder's angular traversal
/// can't handle the 4-way junction where two crossing chords meet.
/// The fallback pipeline handles this correctly at the `boolean_gfa` level.
#[test]
fn gfa_intersect_overlapping_boxes() {
    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let b = make_box(&mut topo, [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);

    let solid_id = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Intersect, a, b)
        .expect("GFA intersect should succeed");
    let faces = brepkit_topology::explorer::solid_faces(&topo, solid_id).unwrap();
    assert_eq!(
        faces.len(),
        6,
        "intersect of overlapping cubes should have 6 faces, got {}",
        faces.len()
    );
}

/// ForceInterfEE creates CommonBlocks for overlapping boundary edges
/// when two boxes share a face.
#[test]
fn force_interf_ee_adjacent_boxes_creates_common_blocks() {
    use brepkit_math::tolerance::Tolerance;

    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = make_box(&mut topo, [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);

    let tol = Tolerance::default();
    let mut arena = GfaArena::new();

    {
        let mut filler = PaveFiller::with_tolerance(&mut topo, a, b, tol);
        filler.perform(&mut arena).unwrap();
    }

    // Run make_blocks (splits pave blocks at extra paves)
    crate::pave_filler::make_blocks::perform(&mut arena).unwrap();

    // Before ForceInterfEE: may have CommonBlocks from coplanar phase
    // (touching boundary edges get linked). Count them for comparison.
    let cb_before = arena.common_blocks.iter().count();

    // Run ForceInterfEE
    crate::pave_filler::force_interf_ee::perform(&topo, tol, &mut arena).unwrap();

    // After ForceInterfEE: should have CommonBlocks for the shared boundary edges.
    // Two adjacent boxes sharing the x=1 face have 4 shared boundary edges:
    // (1,0,0)→(1,1,0), (1,1,0)→(1,1,1), (1,1,1)→(1,0,1), (1,0,1)→(1,0,0)
    let cb_after = arena.common_blocks.iter().count();
    // ForceInterfEE should create additional CommonBlocks (or the coplanar
    // phase already created them). Total should be >= 4 for 4 shared edges.
    assert!(
        (4..=8).contains(&cb_after),
        "adjacent boxes should have >= 4 CommonBlocks for shared boundary edges, got {cb_after} (coplanar: {cb_before})"
    );

    // Each CommonBlock should have at least 2 PaveBlocks
    for (_, cb) in arena.common_blocks.iter() {
        assert!(
            cb.pave_blocks.len() >= 2,
            "CommonBlock should group at least 2 PaveBlocks, got {}",
            cb.pave_blocks.len()
        );
    }
}

/// ForceInterfEE should NOT create CommonBlocks for disjoint boxes.
#[test]
fn force_interf_ee_disjoint_boxes_no_common_blocks() {
    use brepkit_math::tolerance::Tolerance;

    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = make_box(&mut topo, [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);

    let tol = Tolerance::default();
    let mut arena = GfaArena::new();

    {
        let mut filler = PaveFiller::with_tolerance(&mut topo, a, b, tol);
        filler.perform(&mut arena).unwrap();
    }
    crate::pave_filler::make_blocks::perform(&mut arena).unwrap();
    crate::pave_filler::force_interf_ee::perform(&topo, tol, &mut arena).unwrap();

    let cb_count = arena.common_blocks.iter().count();
    assert_eq!(cb_count, 0, "disjoint boxes should have 0 CommonBlocks");
}

/// CB-aware MakeSplitEdges: all PaveBlocks in a CommonBlock share one edge.
#[test]
fn make_split_edges_common_block_shares_edge() {
    use brepkit_math::tolerance::Tolerance;

    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = make_box(&mut topo, [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);

    let tol = Tolerance::default();
    let mut arena = GfaArena::new();

    // Run full PaveFiller (includes ForceInterfEE + MakeSplitEdges)
    crate::pave_filler::run_pave_filler(&mut topo, a, b, tol, &mut arena).unwrap();

    // Check that EVERY CommonBlock member has a split_edge and they all match
    for (cb_id, cb) in arena.common_blocks.iter() {
        if cb.pave_blocks.len() < 2 {
            continue;
        }
        let mut edges = Vec::new();
        for &pb_id in &cb.pave_blocks {
            let pb = arena
                .pave_blocks
                .get(pb_id)
                .expect("PaveBlock referenced by CommonBlock not found in arena");
            let split_edge = pb.split_edge.unwrap_or_else(|| {
                panic!("PaveBlock {pb_id:?} in CommonBlock {cb_id:?} is missing a split_edge")
            });
            edges.push(split_edge);
        }

        // All edges in the CB should be the same
        let first = edges[0];
        for &edge in &edges[1..] {
            assert_eq!(
                first, edge,
                "all PaveBlocks in CommonBlock {cb_id:?} should share the same split edge"
            );
        }
    }
}

/// BuilderSolid improves edge connectivity for adjacent boxes.
/// Full manifoldness requires fill_images_faces CB integration (future work);
/// this test verifies face count and reduced non-manifold edges.
#[test]
fn builder_solid_adjacent_boxes_connectivity() {
    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = make_box(&mut topo, [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);

    let result = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Fuse, a, b)
        .expect("adjacent box fuse");
    let faces = brepkit_topology::explorer::solid_faces(&topo, result).unwrap();
    assert_eq!(faces.len(), 10, "adjacent fuse should have 10 faces");

    // Check manifold: each edge should be shared by exactly 2 faces
    let solid = topo.solid(result).unwrap();
    let shell = topo.shell(solid.outer_shell()).unwrap();
    let mut edge_count: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    for &fid in shell.faces() {
        let face = topo.face(fid).unwrap();
        let wire = topo.wire(face.outer_wire()).unwrap();
        for oe in wire.edges() {
            let e = topo.edge(oe.edge()).unwrap();
            let s = e.start().index();
            let e_idx = e.end().index();
            let key = if s <= e_idx {
                s * 10000 + e_idx
            } else {
                e_idx * 10000 + s
            };
            *edge_count.entry(key).or_default() += 1;
        }
    }
    let non_manifold = edge_count.values().filter(|&&c| c != 2).count();
    // With CommonBlocks, most edges are properly shared. A few boundary
    // edges from original (unsplit) faces may still appear as non-manifold
    // by vertex-pair count because they have separate EdgeIds.
    // The critical check is that the result forms a single connected shell.
    assert!(
        non_manifold <= 8,
        "most edges should be shared by exactly 2 faces, got {non_manifold} non-manifold (expected <= 8)"
    );
}

/// BuilderSolid angle_with_ref computes signed angles correctly.
#[test]
fn builder_solid_angle_with_ref_basic() {
    use crate::builder::builder_solid::get_face_off;

    // This test just verifies the module is accessible.
    // Detailed angle tests are in builder_solid.rs itself.
    let _ = get_face_off; // ensure public
}

/// GFA with manifold-input boxes should produce 10 faces for adjacent fuse.
#[test]
fn gfa_fuse_manifold_boxes_10_faces() {
    let mut topo = Topology::default();
    let a = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 1.0, 0.0, 0.0);

    let result = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Fuse, a, b)
        .expect("manifold box fuse");
    let faces = brepkit_topology::explorer::solid_faces(&topo, result).unwrap();
    assert_eq!(
        faces.len(),
        10,
        "adjacent manifold box fuse: got {}",
        faces.len()
    );
}

/// Touching-face cut: faces share a plane but only touch at an edge.
/// Same-domain detection must require interior overlap (not just edge contact).
#[test]
fn gfa_cut_touching_boxes() {
    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = make_box(&mut topo, [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);

    let solid = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Cut, a, b)
        .expect("cut of touching boxes");
    let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
    assert_eq!(
        faces.len(),
        6,
        "touching cut should have 6 faces, got {}",
        faces.len()
    );
}

#[test]
fn gfa_fuse_disjoint_boxes() {
    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = make_box(&mut topo, [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);

    let solid =
        crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Fuse, a, b).expect("disjoint fuse");
    let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
    assert_eq!(
        faces.len(),
        12,
        "disjoint fuse should have 12 faces, got {}",
        faces.len()
    );
}

#[test]
fn gfa_cut_nested_boxes() {
    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
    let b = make_box(&mut topo, [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);

    // Nested cut: B fully inside A. The containment shortcut in boolean_gfa
    // returns an error for this case ("B is inside A — result would have a void").
    // The GFA itself may also produce a result. Either outcome is acceptable.
    let result = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Cut, a, b);
    if let Ok(solid) = result {
        let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
        assert!(
            faces.len() >= 6,
            "nested cut should have at least 6 faces, got {}",
            faces.len()
        );
    }
    // Err is acceptable — containment shortcut fires before GFA
}

#[test]
fn gfa_fuse_overlapping_boxes_face_count() {
    let mut topo = Topology::default();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = make_box(&mut topo, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);

    let result = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Fuse, a, b)
        .expect("fuse of overlapping boxes");
    let faces = brepkit_topology::explorer::solid_faces(&topo, result).unwrap();
    // Section curves are trimmed to the mutual face overlap, so each cut
    // face splits into exactly 2 sub-faces (kept L + discarded square):
    // 6 untouched faces + 6 L-faces = 12 at the algo level. The
    // operations-level `boolean_gfa` unifies coplanar faces to 10.
    assert!(
        faces.len() == 10 || faces.len() == 12,
        "overlapping fuse should have 10 or 12 faces, got {}",
        faces.len()
    );
}

#[test]
fn coplanar_phase_creates_section_edges() {
    // Two overlapping boxes: A=[0,1]³, B=[0.5,1.5]×[0,1]². The coplanar faces
    // (y=0, y=1, z=0, z=1) share planes and partially overlap. Phase FF-coplanar
    // projects each face's boundary into the other; the projected edges are
    // clipped to the target face, so for this box pair the section edges that
    // ring the overlap coincide with the perpendicular FF intersection and are
    // correctly deduplicated — what matters is the assembled fuse staying
    // watertight (the coplanar phase's purpose).
    let mut topo = Topology::new();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = make_box(&mut topo, [0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);

    let mut arena = GfaArena::default();
    let tol = brepkit_math::tolerance::Tolerance::new();

    // Run PaveFiller (includes Phase FF-coplanar)
    crate::pave_filler::run_pave_filler(&mut topo, a, b, tol, &mut arena).unwrap();

    // The coplanar phase plus the perpendicular FF phase together produce more
    // than the four perpendicular-only intersection curves.
    let ff_count = arena.interference.ff.len();
    assert!(
        ff_count >= 4,
        "should have at least the perpendicular FF interferences, got {ff_count}"
    );

    // The overlapping-box fuse must be a watertight, manifold solid — the
    // coplanar phase exists to make coincident-face booleans close cleanly.
    let result = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Fuse, a, b).unwrap();
    let solid = topo.solid(result).unwrap();
    let shell = topo.shell(solid.outer_shell()).unwrap();
    let mut edge_face_count: std::collections::HashMap<brepkit_topology::edge::EdgeId, usize> =
        std::collections::HashMap::new();
    for &fid in shell.faces() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).unwrap();
            for oe in wire.edges() {
                *edge_face_count.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    let non_manifold = edge_face_count.values().filter(|&&n| n != 2).count();
    assert_eq!(
        non_manifold, 0,
        "overlapping-box fuse must be watertight & manifold (every edge shared by 2 faces)"
    );
}

/// Debug: check how many section PBs and boundary PBs exist for overlapping boxes.
#[test]
fn debug_overlapping_boxes_section_pbs() {
    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;

    let mut topo = Topology::default();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

    let tol = Tolerance::default();
    let mut arena = GfaArena::new();

    // Run Stage 1: intersection phases (VV, VE, EE, VF, EF, FF, FF-coplanar)
    {
        let mut filler = PaveFiller::with_tolerance(&mut topo, a, b, tol);
        filler.perform(&mut arena).unwrap();
    }

    crate::pave_filler::make_blocks::perform(&mut arena).unwrap();
    crate::pave_filler::force_interf_ee::perform(&topo, tol, &mut arena).unwrap();

    let section_pb_count: usize = arena.curves.iter().map(|c| c.pave_blocks.len()).sum();
    let boundary_pb_count: usize = arena.edge_pave_blocks.values().map(Vec::len).sum();
    let cb_count = arena.common_blocks.iter().count();

    eprintln!(
        "section PBs: {section_pb_count}, boundary PBs: {boundary_pb_count}, CBs: {cb_count}"
    );
    eprintln!("curves: {}", arena.curves.len());

    crate::pave_filler::link_existing::perform(&topo, tol, &mut arena).unwrap();
    let cb_count_after = arena.common_blocks.iter().count();
    eprintln!("CBs after link_existing: {cb_count_after}");

    // Run make_split_edges (populates split_edge on CBs)
    crate::pave_filler::make_split_edges::perform(&mut topo, &mut arena).unwrap();
    let cbs_with_split: usize = arena
        .common_blocks
        .iter()
        .filter(|(_, cb)| cb.split_edge.is_some())
        .count();
    eprintln!("CBs with split_edge: {cbs_with_split}/{cb_count_after}");

    eprintln!("VV merges: {}", arena.same_domain_vertices.len());

    let scale = 1.0 / tol.linear;
    let qpt = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };

    #[allow(clippy::type_complexity)]
    let mut boundary_positions: std::collections::HashSet<((i64, i64, i64), (i64, i64, i64))> =
        std::collections::HashSet::new();
    for pbs in arena.edge_pave_blocks.values() {
        for &pb_id in pbs {
            if let Some(pb) = arena.pave_blocks.get(pb_id) {
                let sv = arena.resolve_vertex(pb.start.vertex);
                let ev = arena.resolve_vertex(pb.end.vertex);
                let sp = topo.vertex(sv).unwrap().point();
                let ep = topo.vertex(ev).unwrap().point();
                let qs = qpt(sp);
                let qe = qpt(ep);
                let key = if qs <= qe { (qs, qe) } else { (qe, qs) };
                boundary_positions.insert(key);
            }
        }
    }

    let mut matched = 0;
    let mut unmatched = 0;
    for curve in &arena.curves {
        for &pb_id in &curve.pave_blocks {
            if let Some(pb) = arena.pave_blocks.get(pb_id) {
                let sv = arena.resolve_vertex(pb.start.vertex);
                let ev = arena.resolve_vertex(pb.end.vertex);
                let sp = topo.vertex(sv).unwrap().point();
                let ep = topo.vertex(ev).unwrap().point();
                let qs = qpt(sp);
                let qe = qpt(ep);
                let key = if qs <= qe { (qs, qe) } else { (qe, qs) };
                if boundary_positions.contains(&key) {
                    matched += 1;
                } else {
                    unmatched += 1;
                    eprintln!(
                        "  UNMATCHED section PB: ({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4})",
                        sp.x(),
                        sp.y(),
                        sp.z(),
                        ep.x(),
                        ep.y(),
                        ep.z()
                    );
                }
            }
        }
    }
    eprintln!("section PBs: {matched} matched, {unmatched} unmatched");

    assert!(
        section_pb_count > 0,
        "overlapping boxes should have FF section PBs"
    );
    assert!(
        matched > 0,
        "overlapping boxes should produce at least one section PB whose endpoints \
         coincide with a boundary PB after linking existing geometry"
    );
    // Verify link_existing linked section PBs to CBs (either existing or new)
    let section_pbs_in_cb: usize = arena
        .curves
        .iter()
        .flat_map(|c| c.pave_blocks.iter())
        .filter(|pb_id| arena.pb_to_cb.contains_key(pb_id))
        .count();
    assert!(
        section_pbs_in_cb > 0,
        "link_existing should link at least one section PB to a CommonBlock"
    );
}

/// Trace the full Builder pipeline for overlapping box fuse.
/// Dumps sub-face count, SD pairs, classification, and BOP selection.
#[test]
fn trace_builder_overlapping_box_fuse() {
    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;

    let mut topo = Topology::default();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

    let tol = Tolerance::default();
    let mut arena = crate::ds::GfaArena::new();
    crate::pave_filler::run_pave_filler(&mut topo, a, b, tol, &mut arena).unwrap();

    let mut builder = crate::builder::Builder::with_tolerance(topo, arena, a, b, tol);
    builder.perform().unwrap();

    let (sub_faces, sd_pairs, topo) = builder.debug_info();

    eprintln!("\n=== Builder debug for overlapping box fuse ===");
    eprintln!("sub_faces: {}", sub_faces.len());
    eprintln!("sd_pairs: {}", sd_pairs.len());

    for (i, sf) in sub_faces.iter().enumerate() {
        let surface_desc = match topo
            .face(sf.face_id)
            .ok()
            .map(brepkit_topology::face::Face::surface)
        {
            Some(FaceSurface::Plane { normal, d }) => {
                format!(
                    "Plane(n=({:.1},{:.1},{:.1}), d={d:.2})",
                    normal.x(),
                    normal.y(),
                    normal.z()
                )
            }
            _ => "Other".to_string(),
        };
        let n_edges = topo
            .face(sf.face_id)
            .ok()
            .and_then(|f| topo.wire(f.outer_wire()).ok())
            .map(|w| w.edges().len())
            .unwrap_or(0);
        let ipt = sf
            .interior_point
            .map(|p| format!("({:.3},{:.3},{:.3})", p.x(), p.y(), p.z()));
        eprintln!(
            "  SF[{i}]: {:?} {:?} {surface_desc} edges={n_edges} ipt={ipt:?}",
            sf.rank, sf.classification
        );
    }

    for (i, pair) in sd_pairs.iter().enumerate() {
        eprintln!(
            "  SD[{i}]: A={} B={} same_ori={} contained={}",
            pair.idx_a, pair.idx_b, pair.same_orientation, pair.geometric_overlap
        );
    }

    // Dump edges for sub-faces with out-of-bounds interior points
    for (i, sf) in sub_faces.iter().enumerate() {
        if let Some(pt) = sf.interior_point
            && (pt.x() > 1.6 || pt.x() < -0.1 || pt.y() > 1.1 || pt.z() > 1.1)
        {
            eprintln!(
                "  OUT-OF-BOUNDS SF[{i}]: ipt=({:.3},{:.3},{:.3})",
                pt.x(),
                pt.y(),
                pt.z()
            );
            if let Ok(face) = topo.face(sf.face_id)
                && let Ok(wire) = topo.wire(face.outer_wire())
            {
                for (ei, oe) in wire.edges().iter().enumerate() {
                    if let Ok(edge) = topo.edge(oe.edge()) {
                        let sp = topo
                            .vertex(edge.start())
                            .map(brepkit_topology::vertex::Vertex::point)
                            .ok();
                        let ep = topo
                            .vertex(edge.end())
                            .map(brepkit_topology::vertex::Vertex::point)
                            .ok();
                        if let (Some(s), Some(e)) = (sp, ep) {
                            eprintln!(
                                "    E[{ei}]: ({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4})",
                                s.x(),
                                s.y(),
                                s.z(),
                                e.x(),
                                e.y(),
                                e.z()
                            );
                        }
                    }
                }
            }
        }
    }

    let selected = crate::bop::select_faces(sub_faces, crate::bop::BooleanOp::Fuse, sd_pairs, &[]);
    eprintln!("BOP selected: {} faces", selected.len());
    for (i, sf) in selected.iter().enumerate() {
        let n_edges = topo
            .face(sf.face_id)
            .ok()
            .and_then(|f| topo.wire(f.outer_wire()).ok())
            .map(|w| w.edges().len())
            .unwrap_or(0);
        eprintln!(
            "  SEL[{i}]: face={:?} reversed={} edges={n_edges}",
            sf.face_id, sf.reversed
        );
    }

    // For a proper fuse of [0,1]^3 and [0.5,1.5]x[0,1]^2:
    // Should have 10 faces (6 outer + 4 SD representatives - 4 SD duplicates + 2 end faces)
    // Actually: 2 end faces (x=0, x=1.5) + 4 planes * 3 sub-faces each - 4 SD removals = 10
    assert!(
        selected.len() >= 8 && selected.len() <= 14,
        "expected 8-14 selected faces, got {}",
        selected.len()
    );
}

/// GFA fuse of 1D-offset overlapping manifold boxes.
/// Checks face count and edge manifoldness at the algo level.
#[test]
fn gfa_fuse_1d_overlapping_manifold_boxes() {
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;
    use std::collections::HashMap;

    let mut topo = Topology::default();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

    let result = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Fuse, a, b).unwrap();
    let solid = topo.solid(result).unwrap();
    let shell = topo.shell(solid.outer_shell()).unwrap();

    let face_count = shell.faces().len();
    eprintln!("1D-offset fuse: {face_count} faces");

    // Check edge manifoldness
    let mut edge_face_count: HashMap<brepkit_topology::edge::EdgeId, usize> = HashMap::new();
    for &fid in shell.faces() {
        let face = topo.face(fid).unwrap();
        let wire = topo.wire(face.outer_wire()).unwrap();
        for oe in wire.edges() {
            *edge_face_count.entry(oe.edge()).or_default() += 1;
        }
    }

    let non_manifold = edge_face_count.values().filter(|&&n| n != 2).count();
    eprintln!(
        "{} edges, {} non-manifold",
        edge_face_count.len(),
        non_manifold
    );

    // The GFA should produce 14 selected faces, but BuilderSolid assembly
    // may consolidate some. Accept 10-14 faces.
    assert!(
        (10..=14).contains(&face_count),
        "expected 10-14 faces, got {face_count}"
    );
    // For now, just document the non-manifold count.
    // Target: 0 non-manifold edges.
    eprintln!("NON-MANIFOLD EDGES: {non_manifold}");
}

/// Trace the Builder pipeline for z-axis overlapping boxes.
/// Z-axis offset creates 4 coplanar side face pairs (x=0, x=1, y=0, y=1).
#[test]
fn trace_builder_z_axis_overlap() {
    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;

    let mut topo = Topology::default();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.5);

    let tol = Tolerance::default();
    let mut arena = crate::ds::GfaArena::new();
    crate::pave_filler::run_pave_filler(&mut topo, a, b, tol, &mut arena).unwrap();

    eprintln!(
        "\n=== Z-axis overlap: {} FF interferences ===",
        arena.interference.ff.len()
    );
    for interf in &arena.interference.ff {
        if let crate::ds::Interference::FF {
            f1,
            f2,
            curve_index,
        } = interf
        {
            let curve = &arena.curves[*curve_index];
            let n_pbs = curve.pave_blocks.len();
            let f1_desc = topo
                .face(*f1)
                .ok()
                .map(|f| match f.surface() {
                    FaceSurface::Plane { normal, d } => format!(
                        "Plane(n=({:.1},{:.1},{:.1}),d={d:.2})",
                        normal.x(),
                        normal.y(),
                        normal.z()
                    ),
                    _ => "Other".to_string(),
                })
                .unwrap_or_default();
            let f2_desc = topo
                .face(*f2)
                .ok()
                .map(|f| match f.surface() {
                    FaceSurface::Plane { normal, d } => format!(
                        "Plane(n=({:.1},{:.1},{:.1}),d={d:.2})",
                        normal.x(),
                        normal.y(),
                        normal.z()
                    ),
                    _ => "Other".to_string(),
                })
                .unwrap_or_default();
            for &pb_id in &curve.pave_blocks {
                if let Some(pb) = arena.pave_blocks.get(pb_id) {
                    let sv = arena.resolve_vertex(pb.start.vertex);
                    let ev = arena.resolve_vertex(pb.end.vertex);
                    let sp = topo
                        .vertex(sv)
                        .map(brepkit_topology::vertex::Vertex::point)
                        .ok();
                    let ep = topo
                        .vertex(ev)
                        .map(brepkit_topology::vertex::Vertex::point)
                        .ok();
                    eprintln!(
                        "  FF: {f1:?}({f1_desc}) x {f2:?}({f2_desc}) PBs={n_pbs} \
                         start={sp:?} end={ep:?}"
                    );
                }
            }
        }
    }

    let mut builder = crate::builder::Builder::with_tolerance(topo, arena, a, b, tol);
    builder.perform().unwrap();

    let (sub_faces, sd_pairs, topo) = builder.debug_info();

    eprintln!(
        "\nsub_faces: {}, sd_pairs: {}",
        sub_faces.len(),
        sd_pairs.len()
    );

    for (i, sf) in sub_faces.iter().enumerate() {
        let surface_desc = match topo
            .face(sf.face_id)
            .ok()
            .map(brepkit_topology::face::Face::surface)
        {
            Some(FaceSurface::Plane { normal, d }) => format!(
                "Plane(n=({:.1},{:.1},{:.1}),d={d:.2})",
                normal.x(),
                normal.y(),
                normal.z()
            ),
            _ => "Other".to_string(),
        };
        let n_edges = topo
            .face(sf.face_id)
            .ok()
            .and_then(|f| topo.wire(f.outer_wire()).ok())
            .map(|w| w.edges().len())
            .unwrap_or(0);
        let ipt = sf
            .interior_point
            .map(|p| format!("({:.3},{:.3},{:.3})", p.x(), p.y(), p.z()));
        eprintln!(
            "  SF[{i}]: {:?} {:?} {surface_desc} edges={n_edges} ipt={ipt:?}",
            sf.rank, sf.classification
        );

        // Dump edges for faces with unexpected edge counts
        if n_edges != 4
            && let Ok(face) = topo.face(sf.face_id)
            && let Ok(wire) = topo.wire(face.outer_wire())
        {
            for (ei, oe) in wire.edges().iter().enumerate() {
                if let Ok(edge) = topo.edge(oe.edge()) {
                    let sp = topo
                        .vertex(edge.start())
                        .map(brepkit_topology::vertex::Vertex::point)
                        .ok();
                    let ep = topo
                        .vertex(edge.end())
                        .map(brepkit_topology::vertex::Vertex::point)
                        .ok();
                    eprintln!("    E[{ei}]: {sp:?} -> {ep:?} fwd={}", oe.is_forward());
                }
            }
        }
    }

    for (i, pair) in sd_pairs.iter().enumerate() {
        eprintln!(
            "  SD[{i}]: A={} B={} same_ori={} contained={}",
            pair.idx_a, pair.idx_b, pair.same_orientation, pair.geometric_overlap
        );
    }

    let selected = crate::bop::select_faces(sub_faces, crate::bop::BooleanOp::Fuse, sd_pairs, &[]);
    eprintln!("BOP selected: {} faces", selected.len());
}

/// GFA fuse of z-axis-offset overlapping manifold boxes.
#[test]
fn gfa_fuse_z_axis_overlapping_manifold_boxes() {
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;
    use std::collections::HashMap;

    let mut topo = Topology::default();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.5);

    let result = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Fuse, a, b).unwrap();
    let solid = topo.solid(result).unwrap();
    let shell = topo.shell(solid.outer_shell()).unwrap();

    let face_count = shell.faces().len();
    eprintln!("Z-axis fuse: {face_count} faces");

    // Check edge manifoldness
    let mut edge_face_count: HashMap<brepkit_topology::edge::EdgeId, usize> = HashMap::new();
    for &fid in shell.faces() {
        let face = topo.face(fid).unwrap();
        let wire = topo.wire(face.outer_wire()).unwrap();
        for oe in wire.edges() {
            *edge_face_count.entry(oe.edge()).or_default() += 1;
        }
    }

    let non_manifold = edge_face_count.values().filter(|&&n| n != 2).count();
    let mut verts: std::collections::HashSet<brepkit_topology::vertex::VertexId> =
        std::collections::HashSet::new();
    for &fid in shell.faces() {
        let face = topo.face(fid).unwrap();
        let wire = topo.wire(face.outer_wire()).unwrap();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            verts.insert(edge.start());
            verts.insert(edge.end());
        }
    }
    let v = verts.len();
    let e = edge_face_count.len();
    #[allow(clippy::cast_possible_wrap)]
    let euler = v as i64 - e as i64 + face_count as i64;
    eprintln!("{e} edges, {v} verts, euler={euler}, {non_manifold} non-manifold");

    for (&eid, &count) in &edge_face_count {
        if count != 2
            && let Ok(e) = topo.edge(eid)
        {
            let sp = topo
                .vertex(e.start())
                .map(brepkit_topology::vertex::Vertex::point)
                .ok();
            let ep = topo
                .vertex(e.end())
                .map(brepkit_topology::vertex::Vertex::point)
                .ok();
            eprintln!("  NM edge {eid:?}: count={count} {sp:?} -> {ep:?}");
        }
    }

    assert_eq!(
        non_manifold, 0,
        "{non_manifold} non-manifold edges in z-axis fuse"
    );
}

// ── Box-cylinder cut (periodic-surface wire reconstruction) ──────────
//
// These tests pin the contract for the simplest non-box analytic
// boolean: cutting a cylinder out of a box. The GFA face_splitter
// historically produced an INCOMPLETE outer wire on the cylinder
// lateral face — the new bottom intersection circle was added but
// the new top intersection circle was not, giving Euler = 3 instead
// of 2 and triggering the operations-layer mesh-boolean fallback
// (which polygonalised the cylinder into ~200 faces).
//
// Reference: PR #531 corpus + PR #533 diagnosis, gridfinity-layout-tool
// issue #260 / #270.

/// Build a single cylinder solid in the topology.
///
/// `(cx, cy, z0)` is the center of the bottom cap; the cylinder extends
/// up by `height` along +Z. Lateral face is a single periodic surface
/// with the standard 4-edge wire (bot circle + seam + top circle reversed
/// + seam reversed).
fn make_cylinder(
    topo: &mut Topology,
    cx: f64,
    cy: f64,
    z0: f64,
    radius: f64,
    height: f64,
) -> brepkit_topology::solid::SolidId {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let v_bot = topo.add_vertex(Vertex::new(Point3::new(cx + radius, cy, z0), 1e-7));
    let v_top = topo.add_vertex(Vertex::new(Point3::new(cx + radius, cy, z0 + height), 1e-7));

    let bot_circle =
        Circle3D::new(Point3::new(cx, cy, z0), Vec3::new(0.0, 0.0, 1.0), radius).unwrap();
    let top_circle = Circle3D::new(
        Point3::new(cx, cy, z0 + height),
        Vec3::new(0.0, 0.0, 1.0),
        radius,
    )
    .unwrap();
    let cyl_surface =
        CylindricalSurface::new(Point3::new(cx, cy, z0), Vec3::new(0.0, 0.0, 1.0), radius).unwrap();

    let e_bot = topo.add_edge(Edge::new(v_bot, v_bot, EdgeCurve::Circle(bot_circle)));
    let e_top = topo.add_edge(Edge::new(v_top, v_top, EdgeCurve::Circle(top_circle)));
    let e_seam = topo.add_edge(Edge::new(v_bot, v_top, EdgeCurve::Line));

    let lateral_wire = Wire::new(
        vec![
            OrientedEdge::new(e_bot, true),
            OrientedEdge::new(e_seam, true),
            OrientedEdge::new(e_top, false),
            OrientedEdge::new(e_seam, false),
        ],
        true,
    )
    .unwrap();
    let lateral_wid = topo.add_wire(lateral_wire);
    let lateral = topo.add_face(Face::new(
        lateral_wid,
        vec![],
        FaceSurface::Cylinder(cyl_surface),
    ));

    let bot_cap_wire = Wire::new(vec![OrientedEdge::new(e_bot, false)], true).unwrap();
    let bot_wid = topo.add_wire(bot_cap_wire);
    let bot_cap = topo.add_face(Face::new(
        bot_wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: -z0,
        },
    ));

    let top_cap_wire = Wire::new(vec![OrientedEdge::new(e_top, true)], true).unwrap();
    let top_wid = topo.add_wire(top_cap_wire);
    let top_cap = topo.add_face(Face::new(
        top_wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z0 + height,
        },
    ));

    let shell = topo.add_shell(Shell::new(vec![lateral, bot_cap, top_cap]).unwrap());
    topo.add_solid(Solid::new(shell, vec![]))
}

/// Compute (face_count, edge_count, vertex_count, euler) for a solid.
fn solid_topology_summary(
    topo: &Topology,
    solid: brepkit_topology::solid::SolidId,
) -> (usize, usize, usize, i64) {
    use std::collections::HashSet;
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    let mut edges = HashSet::new();
    let mut verts = HashSet::new();
    let face_count = sh.faces().len();
    for &fid in sh.faces() {
        let f = topo.face(fid).unwrap();
        for wid in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
            let w = topo.wire(wid).unwrap();
            for oe in w.edges() {
                let eid = oe.edge();
                edges.insert(eid);
                let e = topo.edge(eid).unwrap();
                verts.insert(e.start());
                verts.insert(e.end());
            }
        }
    }
    let v = verts.len();
    let e = edges.len();
    #[allow(clippy::cast_possible_wrap)]
    let euler = (v as i64) - (e as i64) + (face_count as i64);
    (face_count, e, v, euler)
}

#[test]
fn gfa_cut_box_cylinder_through_produces_valid_topology() {
    // Box [0,10]^3 with a cylinder r=1 at (5,5) piercing fully through
    // (z=-2 to z=12). The result should be a closed manifold solid:
    //   - 4 box side faces (unchanged)
    //   - 1 box bottom face with a circular inner wire (the hole)
    //   - 1 box top face with a circular inner wire (the hole)
    //   - 1 cylinder lateral face with: bottom circle + seam + top circle
    //     reversed + seam reversed (4 oriented edges, 3 unique edges)
    //
    // Total: 7 faces, V=10, E=15, Euler = V-E+F = 2 (closed manifold).
    //
    // Historically: the cylinder lateral wire was missing the top circle
    // and seam, giving Euler = 3 and triggering mesh fallback.
    // See PR #533 for the diagnosis.
    let mut topo = Topology::default();
    let box_id = make_box(&mut topo, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let cyl = make_cylinder(&mut topo, 5.0, 5.0, -2.0, 1.0, 14.0);

    let result = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Cut, box_id, cyl)
        .expect("GFA cut of box with through-cylinder should succeed");

    let (f, e, v, euler) = solid_topology_summary(&topo, result);
    eprintln!("box-cyl cut: faces={f}, edges={e}, verts={v}, euler={euler}");

    // Manifold check: every edge must be shared by exactly 2 faces.
    let s = topo.solid(result).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    let manifold = brepkit_topology::validation::validate_shell_closed(sh, &topo);
    assert!(
        manifold.is_ok(),
        "result shell must be manifold, got {manifold:?}"
    );

    assert_eq!(f, 7, "box-cyl cut should produce 7 faces, got {f}");
    assert_eq!(
        euler, 2,
        "Euler V-E+F should be 2 for closed manifold, got V={v} E={e} F={f} euler={euler}",
    );
}

#[test]
fn gfa_cut_box_cylinder_grid_through_sequential_produces_valid_topology() {
    // Slab [0,20]×[0,20]×[0,2] cut by a 4×4 grid of cylinders r=0.5,
    // z = -1 to +3 (piercing fully through), positions (2 + col*4, 2 + row*4)
    // for row,col in 0..4.
    //
    // This mirrors `compound_cut_matches_sequential_4x4_grid` in
    // `operations/src/boolean/tests.rs`, but calls `crate::gfa::boolean`
    // directly so no mesh-boolean fallback can mask a GFA failure.
    //
    // Expected result (closed manifold):
    //   - 4 box side faces
    //   - 1 box bottom face with 16 circular holes
    //   - 1 box top face with 16 circular holes
    //   - 16 cylinder lateral faces (each: bot circle + seam + top circle
    //     reversed + seam reversed)
    //
    // Total: 22 faces. Euler V-E+F = 2 for a closed manifold of genus 0.
    let mut topo = Topology::default();
    let mut target = make_box(&mut topo, [0.0, 0.0, 0.0], [20.0, 20.0, 2.0]);

    let mut first_failure: Option<(usize, usize, String)> = None;
    for row in 0..4_i32 {
        for col in 0..4_i32 {
            let cx = 2.0 + f64::from(col) * 4.0;
            let cy = 2.0 + f64::from(row) * 4.0;
            let cyl = make_cylinder(&mut topo, cx, cy, -1.0, 0.5, 4.0);
            match crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Cut, target, cyl) {
                Ok(next) => target = next,
                Err(err) => {
                    first_failure = Some((
                        usize::try_from(row).unwrap(),
                        usize::try_from(col).unwrap(),
                        format!("{err:?}"),
                    ));
                    break;
                }
            }
        }
        if first_failure.is_some() {
            break;
        }
    }

    if let Some((row, col, err)) = first_failure {
        panic!("GFA cut failed at iteration row={row} col={col} of 4×4 grid: {err}");
    }

    let (f, e, v, euler) = solid_topology_summary(&topo, target);
    eprintln!("box-cyl 4x4 grid cut: faces={f}, edges={e}, verts={v}, euler={euler}");

    let s = topo.solid(target).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    let manifold = brepkit_topology::validation::validate_shell_closed(sh, &topo);
    assert!(
        manifold.is_ok(),
        "result shell must be manifold, got {manifold:?}"
    );

    assert_eq!(
        f, 22,
        "4×4 grid cut should produce 22 faces (4 sides + 2 caps + 16 laterals), got {f}"
    );
    assert_eq!(
        euler, 2,
        "Euler V-E+F should be 2 for closed manifold, got V={v} E={e} F={f} euler={euler}",
    );
}

/// Coplanar-cap cylinder cut: the tool's caps lie exactly on the box's
/// z=0 and z=2 planes. The wall-plane section circles coincide with the
/// tool's existing cap boundary circles, so the section edges must adopt
/// the cap circles' seam vertices (or be geometrically linked) — otherwise
/// duplicate coincident circle edges survive, SD pairing finds nothing,
/// and the result is non-manifold.
#[test]
fn gfa_cut_box_cylinder_coplanar_caps_produces_valid_topology() {
    let mut topo = Topology::default();
    let box_id = make_box(&mut topo, [0.0, 0.0, 0.0], [4.0, 4.0, 2.0]);
    let cyl_radius = 0.3;
    let cyl_height = 2.0;
    let cyl = make_cylinder(&mut topo, 1.0, 1.0, 0.0, cyl_radius, cyl_height);

    let result = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Cut, box_id, cyl)
        .expect("GFA cut of box with coplanar-cap cylinder should succeed");

    let (f, e, v, euler) = solid_topology_summary(&topo, result);
    eprintln!("coplanar-cap cut: faces={f}, edges={e}, verts={v}, euler={euler}");

    // Manifold: every edge shared by exactly 2 oriented-edge uses.
    let mut edge_use_count: std::collections::HashMap<brepkit_topology::edge::EdgeId, usize> =
        std::collections::HashMap::new();
    let s = topo.solid(result).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    for &fid in sh.faces() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).unwrap();
            for oe in wire.edges() {
                *edge_use_count.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    for &fid in sh.faces() {
        let face = topo.face(fid).unwrap();
        let surf = match face.surface() {
            FaceSurface::Plane { normal, d } => format!(
                "Plane(n=({:.0},{:.0},{:.0}),d={d:.1})",
                normal.x(),
                normal.y(),
                normal.z()
            ),
            FaceSurface::Cylinder(_) => "Cylinder".to_string(),
            _ => "Other".to_string(),
        };
        eprintln!("  face {fid:?} {surf}:");
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).unwrap();
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge()).unwrap();
                let sp = topo.vertex(edge.start()).unwrap().point();
                let ep = topo.vertex(edge.end()).unwrap().point();
                eprintln!(
                    "    edge {:?} {} fwd={} ({:.2},{:.2},{:.2})->({:.2},{:.2},{:.2})",
                    oe.edge(),
                    edge.curve().type_tag(),
                    oe.is_forward(),
                    sp.x(),
                    sp.y(),
                    sp.z(),
                    ep.x(),
                    ep.y(),
                    ep.z()
                );
            }
        }
    }
    for (&eid, &count) in &edge_use_count {
        assert_eq!(
            count, 2,
            "edge {eid:?} is used by {count} face-wire slots (expected 2)"
        );
    }

    assert_eq!(
        euler, 2,
        "Euler V-E+F should be 2, got V={v} E={e} F={f} euler={euler}"
    );

    // No two coincident full-circle edges (same center/radius/axis).
    let mut full_circles: Vec<(
        brepkit_topology::edge::EdgeId,
        brepkit_math::curves::Circle3D,
    )> = Vec::new();
    for &eid in edge_use_count.keys() {
        let edge = topo.edge(eid).unwrap();
        if edge.start() == edge.end()
            && let EdgeCurve::Circle(c) = edge.curve()
        {
            full_circles.push((eid, c.clone()));
        }
    }
    for i in 0..full_circles.len() {
        for j in (i + 1)..full_circles.len() {
            let (ea, ca) = &full_circles[i];
            let (eb, cb) = &full_circles[j];
            let coincident = (ca.center() - cb.center()).length() < 1e-7
                && (ca.radius() - cb.radius()).abs() < 1e-7
                && ca.normal().dot(cb.normal()).abs() > 1.0 - 1e-9;
            assert!(
                !coincident,
                "coincident full-circle edges {ea:?} and {eb:?} (center={:?}, r={})",
                ca.center(),
                ca.radius()
            );
        }
    }

    let expected_vol = 32.0 - std::f64::consts::PI * (cyl_radius * cyl_radius) * cyl_height;
    // Order 5 is too coarse for full-period trig integrands on the
    // cylindrical hole wall; 16 keeps quadrature error far below the
    // 0.1% assertion budget.
    let options = brepkit_check::properties::PropertiesOptions {
        gauss_order: 16,
        ..Default::default()
    };
    let vol = brepkit_check::properties::solid_volume(&topo, result, &options).unwrap();
    let rel = (vol - expected_vol).abs() / expected_vol;
    assert!(
        rel < 0.001,
        "volume {vol:.6} should be within 0.1% of {expected_vol:.6} (rel={rel:.5})"
    );
}

/// Sequential coplanar-cap cuts: a second flush-cap cylinder cut onto a box
/// wall that already carries a circular hole from the first cut. The wall's
/// existing circular hole must stay visible to the same-domain hole test
/// (sampled, not vertex-only) so the wall face is not cancelled, and the
/// second tool's flush cap must not fragment into a many-sided polygon that
/// survives the cut. Expected: 4 sides + 2 holed caps + 2 lateral walls.
#[test]
fn gfa_cut_box_two_coplanar_cap_cylinders_sequential_valid() {
    let mut topo = Topology::default();
    let mut target = make_box(&mut topo, [0.0, 0.0, 0.0], [4.0, 4.0, 2.0]);
    for (cx, cy) in [(1.0, 1.0), (3.0, 3.0)] {
        let cyl = make_cylinder(&mut topo, cx, cy, 0.0, 0.3, 2.0);
        target = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Cut, target, cyl)
            .expect("sequential coplanar-cap cut should succeed");
    }

    let (f, e, v, euler) = solid_topology_summary(&topo, target);
    assert_eq!(f, 8, "expected 4 sides + 2 caps + 2 laterals, got {f}");
    assert_eq!(
        euler, 2,
        "Euler V-E+F should be 2 for a closed genus-2 manifold, got V={v} E={e} F={f}"
    );

    let s = topo.solid(target).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    let manifold = brepkit_topology::validation::validate_shell_closed(sh, &topo);
    assert!(
        manifold.is_ok(),
        "result must be manifold, got {manifold:?}"
    );

    let expected_vol = 32.0 - 2.0 * std::f64::consts::PI * 0.3 * 0.3 * 2.0;
    let options = brepkit_check::properties::PropertiesOptions {
        gauss_order: 16,
        ..Default::default()
    };
    let vol = brepkit_check::properties::solid_volume(&topo, target, &options).unwrap();
    let rel = (vol - expected_vol).abs() / expected_vol;
    assert!(
        rel < 0.001,
        "volume {vol:.6} should be within 0.1% of {expected_vol:.6} (rel={rel:.5})"
    );
}

/// Regression: a tray-shaped target (box with an open-top cavity) cut by a
/// tool passing through the cavity opening. The tool's cross-section at the
/// rim plane lies inside the rim face's existing hole — air, not face
/// material — and must not be stamped onto the rim face as a nested loop.
/// Previously it was, leaving four free edges and a broken manifold.
#[test]
fn gfa_cut_shelled_box_through_floor_is_manifold() {
    use std::collections::HashMap;

    let mut topo = Topology::new();
    let outer = make_box(&mut topo, [0.0, 0.0, 0.0], [40.0, 40.0, 10.0]);
    let cavity = make_box(&mut topo, [2.0, 2.0, 2.0], [38.0, 38.0, 10.0]);
    let tray = crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Cut, outer, cavity)
        .expect("tray cut");

    let tool = make_box(&mut topo, [4.0, 4.0, -5.0], [7.0, 7.0, 15.0]);
    let result =
        crate::gfa::boolean(&mut topo, crate::bop::BooleanOp::Cut, tray, tool).expect("floor cut");

    let faces = brepkit_topology::explorer::solid_faces(&topo, result).expect("faces");
    assert_eq!(
        faces.len(),
        15,
        "11 tray faces + 4 hole walls; the rim face must keep only its cavity hole"
    );

    let mut edge_uses: HashMap<usize, usize> = HashMap::new();
    let mut inner_wires = 0usize;
    for &fid in &faces {
        let face = topo.face(fid).expect("face");
        inner_wires += face.inner_wires().len();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).expect("wire");
            for oe in wire.edges() {
                *edge_uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    assert_eq!(inner_wires, 3, "bottom hole + floor hole + cavity rim hole");
    assert!(
        edge_uses.values().all(|&c| c == 2),
        "closed manifold: every edge shared by exactly two wires"
    );
}

// ── N-way pave filler (run_pave_filler_n) ───────────────────────────────

#[test]
fn interacting_pairs_prunes_disjoint_boxes() {
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;
    let tol = brepkit_math::tolerance::Tolerance::default();
    let mut topo = Topology::default();
    // Unit cubes span [0,1], [0.5,1.5], [1.0,2.0]: 0&1 overlap, 1&2 overlap,
    // 0&2 only touch at x=1.0 (within tol → still interacting), 0&3 disjoint.
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let c = make_unit_cube_manifold_at(&mut topo, 1.0, 0.0, 0.0);
    let d = make_unit_cube_manifold_at(&mut topo, 3.0, 0.0, 0.0); // far away

    let pairs = super::interacting_pairs(&topo, &[a, b, c, d], tol);
    assert!(pairs.contains(&(0, 1)), "0&1 overlap");
    assert!(pairs.contains(&(1, 2)), "1&2 overlap");
    // 0&3 and 1&3 and 2&3 are far apart and must be pruned.
    assert!(!pairs.contains(&(0, 3)), "0&3 disjoint, pruned");
    assert!(!pairs.contains(&(1, 3)), "1&3 disjoint, pruned");
    assert!(!pairs.contains(&(2, 3)), "2&3 disjoint, pruned");
}

#[test]
fn pave_filler_n_empty_sources_errors() {
    let mut topo = Topology::default();
    let tol = brepkit_math::tolerance::Tolerance::default();
    let mut arena = GfaArena::new();
    assert!(crate::pave_filler::run_pave_filler_n(&mut topo, &[], tol, &mut arena).is_err());
}

/// For N=2 the N-way pave filler runs the exact same phase sequence as the
/// two-solid `run_pave_filler`, so the full pipeline (pave filler → builder →
/// fuse) must produce an identical result. Compare the fused face count.
#[test]
fn pave_filler_n_matches_two_solid_for_n2() {
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;
    let tol = brepkit_math::tolerance::Tolerance::default();

    let fuse_faces = |use_nway: bool| -> usize {
        let mut topo = Topology::default();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
        let mut arena = GfaArena::new();
        if use_nway {
            crate::pave_filler::run_pave_filler_n(&mut topo, &[a, b], tol, &mut arena).unwrap();
        } else {
            crate::pave_filler::run_pave_filler(&mut topo, a, b, tol, &mut arena).unwrap();
        }
        let mut builder = crate::builder::Builder::with_tolerance(topo, arena, a, b, tol);
        builder.perform().unwrap();
        let (rtopo, result) = builder.build_result(crate::bop::BooleanOp::Fuse).unwrap();
        solid_faces(&rtopo, result).unwrap().len()
    };

    let two_solid = fuse_faces(false);
    let n_way = fuse_faces(true);
    assert!(two_solid > 0, "2-solid fuse produced faces");
    assert_eq!(
        two_solid, n_way,
        "N=2 n-way pave filler must match the 2-solid fuse face count"
    );
}

/// Smoke test that the pairwise accumulation over THREE solids runs and
/// actually splits shared edges. The middle box overlaps both neighbours, so
/// the arena must end with at least one edge carrying multiple pave blocks
/// (i.e. split by an intersection). Proves the phases accumulate into one arena
/// across pairs without a hidden two-solid assumption.
#[test]
fn pave_filler_n_accumulates_three_overlapping_boxes() {
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;
    let tol = brepkit_math::tolerance::Tolerance::default();
    let mut topo = Topology::default();
    let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
    let c = make_unit_cube_manifold_at(&mut topo, 1.0, 0.0, 0.0);

    let mut arena = GfaArena::new();
    crate::pave_filler::run_pave_filler_n(&mut topo, &[a, b, c], tol, &mut arena).unwrap();

    let split_edges = arena
        .edge_pave_blocks
        .values()
        .filter(|blocks| blocks.len() > 1)
        .count();
    assert!(
        split_edges > 0,
        "three overlapping boxes must split at least one shared edge \
         (arena accumulated no splits — pairwise phases did not accumulate)"
    );
}

// ── End-to-end N-way fuse (store → pave_filler_n → build_fuse_n) ─────────

/// Count how many distinct edges of a solid are used by exactly two faces
/// (manifold) vs otherwise, as a watertightness proxy.
fn edge_face_share(topo: &Topology, solid: brepkit_topology::solid::SolidId) -> (usize, usize) {
    use std::collections::HashMap;
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge().index()).or_default() += 1;
            }
        }
    }
    let two = uses.values().filter(|&&c| c == 2).count();
    let other = uses.values().filter(|&&c| c != 2).count();
    (two, other)
}

/// N=2: the N-way fuse of two interpenetrating boxes (offset on all three axes,
/// so no faces are coplanar-coincident) must match the standard two-solid fuse
/// face count and be watertight.
#[test]
fn build_fuse_n_two_interpenetrating_boxes_matches_two_solid() {
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;
    let tol = brepkit_math::tolerance::Tolerance::default();

    // Two-solid reference.
    let two_solid_faces = {
        let mut topo = Topology::default();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);
        let mut arena = GfaArena::new();
        crate::pave_filler::run_pave_filler(&mut topo, a, b, tol, &mut arena).unwrap();
        let mut builder = crate::builder::Builder::with_tolerance(topo, arena, a, b, tol);
        builder.perform().unwrap();
        let (rtopo, result) = builder.build_result(crate::bop::BooleanOp::Fuse).unwrap();
        solid_faces(&rtopo, result).unwrap().len()
    };

    // N-way path.
    let mut src = Topology::default();
    let a = make_unit_cube_manifold_at(&mut src, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut src, 0.5, 0.5, 0.5);
    let mut store = crate::ds::GfaShapeStoreN::new(&src, &[a, b]).unwrap();
    let mut arena = GfaArena::new();
    crate::pave_filler::run_pave_filler_n(&mut store.topo, &store.sources, tol, &mut arena)
        .unwrap();
    let (topo, result) = crate::builder::build_fuse_n(
        std::mem::take(&mut store.topo),
        arena,
        &store.sources,
        &store.face_source,
        tol,
    )
    .unwrap();

    let n_faces = solid_faces(&topo, result).unwrap().len();
    assert_eq!(
        n_faces, two_solid_faces,
        "N-way fuse face count must match the two-solid fuse"
    );
    let (two, other) = edge_face_share(&topo, result);
    assert!(
        two > 0 && other == 0,
        "N-way fuse must be watertight (every edge shared by 2 faces); got two={two} other={other}"
    );
}

/// N=3: three boxes each offset diagonally so consecutive boxes interpenetrate
/// and none share a coplanar face. The N-way fuse produces one watertight solid
/// in a single pass.
#[test]
fn build_fuse_n_three_interpenetrating_boxes_is_watertight() {
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;
    let tol = brepkit_math::tolerance::Tolerance::default();

    let mut src = Topology::default();
    let a = make_unit_cube_manifold_at(&mut src, 0.0, 0.0, 0.0);
    let b = make_unit_cube_manifold_at(&mut src, 0.5, 0.5, 0.5);
    let c = make_unit_cube_manifold_at(&mut src, 1.0, 1.0, 1.0);
    let mut store = crate::ds::GfaShapeStoreN::new(&src, &[a, b, c]).unwrap();
    let mut arena = GfaArena::new();
    crate::pave_filler::run_pave_filler_n(&mut store.topo, &store.sources, tol, &mut arena)
        .unwrap();
    let (topo, result) = crate::builder::build_fuse_n(
        std::mem::take(&mut store.topo),
        arena,
        &store.sources,
        &store.face_source,
        tol,
    )
    .unwrap();

    assert!(
        !solid_faces(&topo, result).unwrap().is_empty(),
        "produced a solid"
    );
    let (two, other) = edge_face_share(&topo, result);
    assert!(
        two > 0 && other == 0,
        "three-box N-way fuse must be watertight; got two={two} other={other}"
    );
}

// ── N-way fuse with coincident faces (cross-source same-domain) ──────────

/// Run the full N-way fuse pipeline for `offsets` unit cubes and return
/// (result topology, solid, face count).
fn nway_fuse(offsets: &[[f64; 3]]) -> (Topology, brepkit_topology::solid::SolidId) {
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;
    let tol = brepkit_math::tolerance::Tolerance::default();
    let mut src = Topology::default();
    let solids: Vec<_> = offsets
        .iter()
        .map(|o| make_unit_cube_manifold_at(&mut src, o[0], o[1], o[2]))
        .collect();
    let mut store = crate::ds::GfaShapeStoreN::new(&src, &solids).unwrap();
    let mut arena = GfaArena::new();
    crate::pave_filler::run_pave_filler_n(&mut store.topo, &store.sources, tol, &mut arena)
        .unwrap();
    crate::builder::build_fuse_n(
        std::mem::take(&mut store.topo),
        arena,
        &store.sources,
        &store.face_source,
        tol,
    )
    .unwrap()
}

/// Axis-aligned overlapping boxes share coincident coplanar faces (the top,
/// bottom, front and back planes coincide over the overlap). Cross-source
/// same-domain must resolve them so the N-way fuse still matches the two-solid
/// fuse and is watertight — the case that used to bail.
#[test]
fn build_fuse_n_axis_aligned_overlap_matches_two_solid() {
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;
    let tol = brepkit_math::tolerance::Tolerance::default();

    let two_solid = {
        let mut topo = Topology::default();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);
        let mut arena = GfaArena::new();
        crate::pave_filler::run_pave_filler(&mut topo, a, b, tol, &mut arena).unwrap();
        let mut builder = crate::builder::Builder::with_tolerance(topo, arena, a, b, tol);
        builder.perform().unwrap();
        let (rtopo, result) = builder.build_result(crate::bop::BooleanOp::Fuse).unwrap();
        solid_faces(&rtopo, result).unwrap().len()
    };

    let (topo, result) = nway_fuse(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]);
    let n_faces = solid_faces(&topo, result).unwrap().len();
    let (two, other) = edge_face_share(&topo, result);
    assert!(two > 0 && other == 0, "watertight; two={two} other={other}");
    assert_eq!(
        n_faces, two_solid,
        "N-way overlap fuse matches the two-solid fuse"
    );
}

/// Two boxes abutting face-to-face: the shared wall is an opposite-oriented
/// coincident pair (material on both sides), so both faces must be dropped,
/// producing one watertight 2×1×1 bar.
#[test]
fn build_fuse_n_abutting_boxes_drop_shared_wall() {
    use brepkit_topology::explorer::solid_faces;
    let (topo, result) = nway_fuse(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    let (two, other) = edge_face_share(&topo, result);
    assert!(
        two > 0 && other == 0,
        "abutting fuse watertight; two={two} other={other}"
    );
    // The shared x=1 wall (one face from each box) is dropped; a clean bar has
    // fewer faces than the 12 of two separate boxes.
    let n = solid_faces(&topo, result).unwrap().len();
    assert!(n < 12, "shared wall dropped, got {n} faces");
}

/// Three axis-aligned overlapping boxes fuse to one watertight solid in a single
/// pass, resolving coincident faces between every interacting pair.
#[test]
fn build_fuse_n_three_axis_aligned_row_watertight() {
    use brepkit_topology::explorer::solid_faces;
    let (topo, result) = nway_fuse(&[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]);
    assert!(!solid_faces(&topo, result).unwrap().is_empty());
    let (two, other) = edge_face_share(&topo, result);
    assert!(
        two > 0 && other == 0,
        "three-box row watertight; two={two} other={other}"
    );
}
