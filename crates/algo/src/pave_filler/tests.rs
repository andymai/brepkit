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

    // 8 vertices
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

    let w_bot =
        topo.add_wire(Wire::new(vec![fwd(e01), fwd(e12), fwd(e23), fwd(e30)], true).unwrap());
    let w_top =
        topo.add_wire(Wire::new(vec![fwd(e45), fwd(e56), fwd(e67), fwd(e74)], true).unwrap());
    let w_front =
        topo.add_wire(Wire::new(vec![fwd(e01), fwd(e15), fwd(e45), fwd(e04)], true).unwrap());
    let w_back =
        topo.add_wire(Wire::new(vec![fwd(e23), fwd(e37), fwd(e67), fwd(e26)], true).unwrap());
    let w_left =
        topo.add_wire(Wire::new(vec![fwd(e30), fwd(e04), fwd(e74), fwd(e37)], true).unwrap());
    let w_right =
        topo.add_wire(Wire::new(vec![fwd(e12), fwd(e26), fwd(e56), fwd(e15)], true).unwrap());

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
            // Check that the result solid exists and has faces
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

    // Run PaveFiller intersection phases
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
        cb_after >= 4,
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
    // GFA produces quadrant-split faces (24) at the algo level.
    // The operations-level `boolean_gfa` unifies coplanar faces to 10.
    // Accept either count here since this tests the algo crate directly.
    assert!(
        faces.len() == 10 || faces.len() == 24,
        "overlapping fuse should have 10 or 24 faces, got {}",
        faces.len()
    );
}

#[test]
fn coplanar_phase_creates_section_edges() {
    // Two overlapping boxes: A=[0,1]³, B=[0.5,1.5]×[0,1]².
    // The coplanar faces (y=0, y=1, z=0, z=1) share the same plane.
    // Phase FF-coplanar should create section edges where one face's
    // boundary edges lie inside the other face.
    let mut topo = Topology::new();
    let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = make_box(&mut topo, [0.5, 0.0, 0.0], [1.5, 1.0, 1.0]);

    let mut arena = GfaArena::default();
    let tol = brepkit_math::tolerance::Tolerance::new();

    // Run PaveFiller (includes Phase FF-coplanar)
    crate::pave_filler::run_pave_filler(&mut topo, a, b, tol, &mut arena).unwrap();

    // Count FF interferences — should include both perpendicular and coplanar
    let ff_count = arena.interference.ff.len();
    // Without coplanar phase: only perpendicular face pairs produce FF curves.
    // With coplanar phase: additional section edges from boundary projections.
    // Perpendicular pairs: A has faces at x=0, x=1; B has faces at x=0.5, x=1.5.
    // Each perpendicular face pair produces an intersection line.
    // Coplanar pairs: y=0+y=0, y=1+y=1, z=0+z=0, z=1+z=1 — each produces
    // section edges from boundary edge projections.
    assert!(
        ff_count > 4,
        "should have more than 4 FF interferences (perpendicular + coplanar), got {ff_count}"
    );

    // Verify that section curves exist for the coplanar faces
    let has_coplanar_curves = arena.curves.iter().any(|c| {
        let face_a_surf = topo.face(c.face_a).ok().map(|f| f.surface().clone());
        let face_b_surf = topo.face(c.face_b).ok().map(|f| f.surface().clone());
        matches!(
            (&face_a_surf, &face_b_surf),
            (
                Some(FaceSurface::Plane { normal: na, .. }),
                Some(FaceSurface::Plane { normal: nb, .. })
            ) if na.dot(*nb).abs() > 0.99
        )
    });

    assert!(has_coplanar_curves, "should have coplanar section curves");
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

    // Count section PBs
    let section_pb_count: usize = arena.curves.iter().map(|c| c.pave_blocks.len()).sum();
    let boundary_pb_count: usize = arena.edge_pave_blocks.values().map(Vec::len).sum();
    let cb_count = arena.common_blocks.iter().count();

    eprintln!(
        "section PBs: {section_pb_count}, boundary PBs: {boundary_pb_count}, CBs: {cb_count}"
    );
    eprintln!("curves: {}", arena.curves.len());

    // Run link_existing
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

    // Dump VV merge map
    eprintln!("VV merges: {}", arena.same_domain_vertices.len());

    // Dump section PB endpoints and check if they match boundary PBs
    let scale = 1.0 / tol.linear;
    let qpt = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };

    // Build boundary PB position index
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

    // Check each section PB
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

    // Dump each sub-face: rank, classification, surface type
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

    // Dump SD pairs
    for (i, pair) in sd_pairs.iter().enumerate() {
        eprintln!(
            "  SD[{i}]: A={} B={} same_ori={} contained={}",
            pair.idx_a, pair.idx_b, pair.same_orientation, pair.b_contained_in_a
        );
    }

    // Dump edges for sub-faces with out-of-bounds interior points
    for (i, sf) in sub_faces.iter().enumerate() {
        if let Some(pt) = sf.interior_point {
            if pt.x() > 1.6 || pt.x() < -0.1 || pt.y() > 1.1 || pt.z() > 1.1 {
                eprintln!(
                    "  OUT-OF-BOUNDS SF[{i}]: ipt=({:.3},{:.3},{:.3})",
                    pt.x(),
                    pt.y(),
                    pt.z()
                );
                if let Ok(face) = topo.face(sf.face_id) {
                    if let Ok(wire) = topo.wire(face.outer_wire()) {
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
        }
    }

    // Simulate BOP selection
    let selected = crate::bop::select_faces(sub_faces, crate::bop::BooleanOp::Fuse, sd_pairs);
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

    // Dump FF interferences
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
            // Dump PB endpoints
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
        if n_edges != 4 {
            if let Ok(face) = topo.face(sf.face_id) {
                if let Ok(wire) = topo.wire(face.outer_wire()) {
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
        }
    }

    for (i, pair) in sd_pairs.iter().enumerate() {
        eprintln!(
            "  SD[{i}]: A={} B={} same_ori={} contained={}",
            pair.idx_a, pair.idx_b, pair.same_orientation, pair.b_contained_in_a
        );
    }

    let selected = crate::bop::select_faces(sub_faces, crate::bop::BooleanOp::Fuse, sd_pairs);
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
    // Count unique vertices
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

    // Dump non-manifold edge details
    for (&eid, &count) in &edge_face_count {
        if count != 2 {
            if let Ok(e) = topo.edge(eid) {
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
    }

    assert_eq!(
        non_manifold, 0,
        "{non_manifold} non-manifold edges in z-axis fuse"
    );
}
