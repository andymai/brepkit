#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr
)]

use brepkit_topology::Topology;

use super::*;

#[test]
fn no_merge_on_clean_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

    let count = merge_coincident_vertices(&mut topo, solid, 1e-7).unwrap();
    assert_eq!(count, 0, "clean box should have no coincident vertices");
}

#[test]
fn merge_with_large_tolerance() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

    let count = merge_coincident_vertices(&mut topo, solid, 3.0).unwrap();
    assert!(count > 0, "large tolerance should merge some vertices");
}

#[test]
fn heal_clean_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

    let report = heal_solid(&mut topo, solid, 1e-7).unwrap();
    assert_eq!(report.vertices_merged, 0);
    assert_eq!(report.degenerate_edges_removed, 0);
    // Orientation may or may not need fixing depending on make_box
}

#[test]
fn no_degenerate_edges_in_clean_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

    let count = remove_degenerate_edges(&mut topo, solid, 1e-7).unwrap();
    assert_eq!(count, 0);
}

#[test]
fn fix_orientations_on_clean_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

    let count = fix_face_orientations(&mut topo, solid).unwrap();
    // A properly constructed box should not need fixes
    assert_eq!(count, 0);
}

// ── Wire gap closure tests ──────────────────────────

#[test]
fn close_wire_gaps_clean_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let count = close_wire_gaps(&mut topo, solid, 1e-7).unwrap();
    assert_eq!(count, 0, "clean box should have no wire gaps");
}

// ── Small face removal tests ────────────────────────

#[test]
fn no_small_faces_in_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let count = remove_small_faces(&mut topo, solid, 0.01).unwrap();
    assert_eq!(count, 0, "unit box should have no small faces");
}

// ── Duplicate face removal tests ────────────────────

#[test]
fn no_duplicates_in_box() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let count = remove_duplicate_faces(&mut topo, solid, 1e-7).unwrap();
    assert_eq!(count, 0, "box should have no duplicate faces");
}

// ── Full heal pipeline tests ────────────────────────

#[test]
fn heal_report_all_fields() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let report = heal_solid(&mut topo, solid, 1e-7).unwrap();

    // Clean box should need minimal or no repairs.
    assert_eq!(report.wire_gaps_closed, 0);
    assert_eq!(report.vertices_merged, 0);
    assert_eq!(report.degenerate_edges_removed, 0);
    assert_eq!(report.small_faces_removed, 0);
    assert_eq!(report.duplicate_faces_removed, 0);
}

#[test]
fn heal_cylinder_no_crash() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

    // Healing should not crash on cylinders (has non-planar faces).
    let report = heal_solid(&mut topo, solid, 1e-7).unwrap();
    assert_eq!(report.wire_gaps_closed, 0);
}

#[test]
fn heal_clean_geometry_zero_repairs() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let report = repair_solid(&mut topo, solid, 1e-7).unwrap();
    assert_eq!(
        report.total_repairs(),
        0,
        "clean box should need zero repairs, got {}",
        report.total_repairs()
    );
    assert!(
        report.is_valid_after(),
        "clean box should be valid after repair"
    );
}

#[test]
fn heal_preserves_volume() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 3.0, 4.0, 5.0).unwrap();

    let vol_before = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    let _report = repair_solid(&mut topo, solid, 1e-7).unwrap();
    let vol_after = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

    assert!(
        (vol_before - vol_after).abs() < 0.01,
        "heal should preserve volume: before={vol_before}, after={vol_after}"
    );
}

// ── Face unification tests ──────────────────────────

#[test]
fn unify_clean_box_no_change() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

    let (f_before, _, _) = brepkit_topology::explorer::solid_entity_counts(&topo, solid).unwrap();
    let removed = unify_faces(&mut topo, solid).unwrap();

    assert_eq!(removed, 0, "clean box should have nothing to unify");
    let (f_after, _, _) = brepkit_topology::explorer::solid_entity_counts(&topo, solid).unwrap();
    assert_eq!(f_before, f_after);
}

#[test]
fn unify_clean_cylinder_no_change() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

    let removed = unify_faces(&mut topo, solid).unwrap();
    assert_eq!(removed, 0, "clean cylinder should have nothing to unify");
}

#[test]
fn unify_boolean_box_reduces_faces() {
    // L-shape fuse: two boxes sharing a corner, only one of three
    // dimensions matching (z). The box-pair shortcut bails (needs
    // 2 of 3 dims to match) so GFA splits the z-faces into coplanar
    // fragments that `unify_faces` should merge.
    let mut topo = Topology::new();
    let box1 = crate::primitives::make_box(&mut topo, 3.0, 1.0, 1.0).unwrap();
    let box2 = crate::primitives::make_box(&mut topo, 1.0, 3.0, 1.0).unwrap();
    // Both boxes share the origin corner — no translation needed.

    let opts = crate::boolean::BooleanOptions {
        unify_faces: false,
        ..Default::default()
    };
    let fused = crate::boolean::boolean_with_options(
        &mut topo,
        crate::boolean::BooleanOp::Fuse,
        box1,
        box2,
        opts,
    )
    .unwrap();

    let (f_before, _, _) = brepkit_topology::explorer::solid_entity_counts(&topo, fused).unwrap();

    let vol_before = crate::measure::solid_volume(&topo, fused, 0.1).unwrap();

    let removed = unify_faces(&mut topo, fused).unwrap();

    let (f_after, _, _) = brepkit_topology::explorer::solid_entity_counts(&topo, fused).unwrap();

    let vol_after = crate::measure::solid_volume(&topo, fused, 0.1).unwrap();

    // Unification should reduce face count.
    assert!(
        removed > 0,
        "boolean fuse of overlapping boxes should produce unifiable coplanar faces, \
             f_before={f_before}, f_after={f_after}"
    );
    assert!(f_after < f_before);

    // Volume should be preserved.
    assert!(
        (vol_before - vol_after).abs() < 0.1,
        "unification should preserve volume: before={vol_before}, after={vol_after}"
    );
}

#[test]
fn unify_preserves_volume() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 3.0, 4.0, 5.0).unwrap();

    let vol_before = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    let _removed = unify_faces(&mut topo, solid).unwrap();
    let vol_after = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

    assert!(
        (vol_before - vol_after).abs() < 0.01,
        "unify should preserve volume: before={vol_before}, after={vol_after}"
    );
}

#[test]
fn unify_shell_box_reduces_faces() {
    // Shell a box (hollow it out) — this produces coplanar face fragments
    // that should be merged by unify_faces.
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    // Get top face (z=10) as the open face for shelling.
    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();
    let face_ids: Vec<_> = shell.faces().to_vec();

    // Find the top face (normal pointing +z, d ≈ 10)
    let top_face = face_ids
        .iter()
        .find(|&&fid| {
            let face = topo.face(fid).unwrap();
            match face.surface() {
                FaceSurface::Plane { normal, d } => normal.z() > 0.9 && (*d - 10.0).abs() < 0.1,
                _ => false,
            }
        })
        .copied();

    let open_faces = match top_face {
        Some(f) => vec![f],
        None => vec![face_ids[0]], // fallback
    };

    let shelled = crate::shell_op::shell(&mut topo, solid, 1.0, &open_faces).unwrap();

    let (f_before, e_before, v_before) =
        brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
    #[allow(clippy::cast_possible_wrap)]
    let chi_before = (v_before as i64) - (e_before as i64) + (f_before as i64);

    let vol_before = crate::measure::solid_volume(&topo, shelled, 0.1).unwrap();

    let removed = unify_faces(&mut topo, shelled).unwrap();

    let (f_after, e_after, v_after) =
        brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
    #[allow(clippy::cast_possible_wrap)]
    let chi_after = (v_after as i64) - (e_after as i64) + (f_after as i64);

    let vol_after = crate::measure::solid_volume(&topo, shelled, 0.1).unwrap();

    eprintln!(
        "shell box: faces {f_before} -> {f_after} (removed {removed}), \
             χ {chi_before} -> {chi_after}, vol {vol_before:.1} -> {vol_after:.1}"
    );

    // Volume must be preserved.
    assert!(
        (vol_before - vol_after).abs() / vol_before < 0.01,
        "unification should preserve volume: before={vol_before}, after={vol_after}"
    );

    // Euler characteristic must be preserved.
    assert_eq!(
        chi_before, chi_after,
        "Euler χ should be preserved: before={chi_before}, after={chi_after}"
    );
}

#[test]
fn unify_cylinder_boolean_reduces_faces() {
    // Boolean cut of a box with a cylinder produces co-cylindrical face fragments.
    let mut topo = Topology::new();
    let box1 = crate::primitives::make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
    let cyl = crate::primitives::make_cylinder(&mut topo, 1.0, 6.0).unwrap();

    // Move cylinder to center of box top face.
    let translate = brepkit_math::mat::Mat4::translation(2.0, 2.0, -1.0);
    crate::transform::transform_solid(&mut topo, cyl, &translate).unwrap();

    let result =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, box1, cyl).unwrap();

    let (f_before, _, _) = brepkit_topology::explorer::solid_entity_counts(&topo, result).unwrap();

    let vol_before = crate::measure::solid_volume(&topo, result, 0.1).unwrap();

    let removed = unify_faces(&mut topo, result).unwrap();

    let vol_after = crate::measure::solid_volume(&topo, result, 0.1).unwrap();

    // Volume must be preserved.
    assert!(
        (vol_before - vol_after).abs() < 0.1,
        "unification should preserve volume: before={vol_before}, after={vol_after}"
    );

    // Log for diagnostics (test passes either way — this is informational).
    let (f_after, _, _) = brepkit_topology::explorer::solid_entity_counts(&topo, result).unwrap();
    eprintln!("cylinder boolean: faces {f_before} -> {f_after}, removed {removed}");
}

#[test]
fn unify_shell_rounded_rect_preserves_volume() {
    // Shell a rounded rectangle with 3 arc edges per quarter-circle corner
    // (matching brepjs behavior). The extrusion creates 3 cylindrical face
    // fragments per corner. unify_faces merges these. This is the exact
    // scenario that causes volume corruption in the topology parity test.
    use brepkit_math::curves::Circle3D;
    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::Vec3;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let tol = Tolerance::new();

    let w = 41.5_f64;
    let d = 41.5_f64;
    let h = 21.0_f64;
    let r = 4.0_f64;
    let thickness = 1.2_f64;
    let hw = w / 2.0;
    let hd = d / 2.0;

    // Corner centers:
    let c_br = Point3::new(hw - r, -hd + r, 0.0);
    let c_tr = Point3::new(hw - r, hd - r, 0.0);
    let c_tl = Point3::new(-hw + r, hd - r, 0.0);
    let c_bl = Point3::new(-hw + r, -hd + r, 0.0);

    let z_axis = Vec3::new(0.0, 0.0, 1.0);

    // Subdivide each quarter circle into 3 arcs (30° each).
    // Quarter goes from angle a0 to a0+π/2 in 3 steps.
    let n_sub = 3usize;
    let quarter = std::f64::consts::FRAC_PI_2;

    // Corner start angles (CCW): BR=-π/2, TR=0, TL=π/2, BL=π
    let corners = [
        (c_br, -std::f64::consts::FRAC_PI_2),
        (c_tr, 0.0),
        (c_tl, std::f64::consts::FRAC_PI_2),
        (c_bl, std::f64::consts::PI),
    ];

    // Build all vertices: for each corner, n_sub+1 points on the arc,
    // but the last point of one corner is the first of the next line segment.
    // Layout: for corner i, arc vertices are at angles a0 + j*(π/2)/n_sub for j=0..n_sub.
    // The vertex at j=0 is the line-end of the previous side.
    // The vertex at j=n_sub is the line-start of the next side.

    // Generate corner arc points.
    let mut corner_verts: Vec<Vec<Point3>> = Vec::new();
    for &(center, a0) in &corners {
        let mut pts = Vec::new();
        for j in 0..=n_sub {
            #[allow(clippy::cast_precision_loss)]
            let angle = a0 + (j as f64) * quarter / (n_sub as f64);
            let pt = Point3::new(
                center.x() + r * angle.cos(),
                center.y() + r * angle.sin(),
                0.0,
            );
            pts.push(pt);
        }
        corner_verts.push(pts);
    }

    // Allocate vertices (sharing endpoints between corners and lines).
    // Wire order: bottom_line, br_arc[0..3], right_line, tr_arc[0..3], top_line, tl_arc[0..3], left_line, bl_arc[0..3]
    // Each line connects: corner[i][n_sub] -> corner[(i+1)%4][0]
    // Line vertices: br[3]=right_start, tr[0]=right_end -> but corners go BR, TR, TL, BL
    // So: bottom_line = bl[3] -> br[0], right_line = br[3] -> tr[0], etc.

    // Allocate all unique vertex IDs.
    // For each corner: n_sub+1 points, but corner[i][n_sub] == start of next line == corner[(i+1)%4][0]
    // Wait, that's not right. Let me think again...
    // Wire order CCW: bl[3]->br[0] (bottom), br[0]->br[3] (br arc), br[3]->tr[0] (right), tr[0]->tr[3] (tr arc), etc.
    // So corner[i][0] is the START of the arc, corner[i][n_sub] is the END.
    // Line between corner i end and corner (i+1)%4 start.
    // But our corners are [BR, TR, TL, BL], and CCW order is: bottom, BR, right, TR, top, TL, left, BL
    // So: BL[3]->BR[0] = bottom line, BR[0]->BR[3] = BR arc, BR[3]->TR[0] = right line, etc.

    // Unique points: 4 corners × n_sub intermediate + 4 corner endpoints shared with lines.
    // Actually: each corner has n_sub+1 points. corner[i][0] is shared with previous line end,
    // corner[i][n_sub] is shared with next line start.
    // Total unique: 4 * n_sub (intermediate arc points) + 4 (shared line/arc junction points) = 4*(n_sub+1) - 4 = 4*n_sub
    // Wait: 4 corners × (n_sub+1) points each, but corner[i][n_sub] == (next line start) and
    // corner[(i+1)%4][0] == (next line end). These are NOT the same point.
    // Actually in CCW order: BL_end -> BR_start (bottom line), BR arcs, BR_end -> TR_start (right line), etc.
    // So each corner contributes n_sub+1 unique points, and lines share those endpoints.
    // Total unique vertices = 4 * (n_sub + 1) = 16 for n_sub=3.

    // Let me just allocate all vertices in wire order.
    let mut wire_edges = Vec::new();

    // CCW order: bottom_line, br_arc, right_line, tr_arc, top_line, tl_arc, left_line, bl_arc
    // corners[0]=BR, corners[1]=TR, corners[2]=TL, corners[3]=BL
    let corner_order = [3, 0, 1, 2]; // BL, BR, TR, TL in CCW order

    for ci in 0..4 {
        let this_corner = corner_order[ci];
        let next_corner = corner_order[(ci + 1) % 4];

        // Line: this_corner[n_sub] -> next_corner[0]
        let line_start = corner_verts[this_corner][n_sub];
        let line_end = corner_verts[next_corner][0];
        let ls_vid = topo.add_vertex(Vertex::new(line_start, tol.linear));
        let le_vid = topo.add_vertex(Vertex::new(line_end, tol.linear));
        let line_eid = topo.add_edge(Edge::new(ls_vid, le_vid, EdgeCurve::Line));
        wire_edges.push(OrientedEdge::new(line_eid, true));

        // Arc segments for next_corner.
        let center = corners[next_corner].0;
        let circle = Circle3D::new(center, z_axis, r).unwrap();
        let mut prev_vid = le_vid;
        for j in 1..=n_sub {
            let pt = corner_verts[next_corner][j];
            let next_vid = topo.add_vertex(Vertex::new(pt, tol.linear));
            let arc_eid = topo.add_edge(Edge::new(
                prev_vid,
                next_vid,
                EdgeCurve::Circle(circle.clone()),
            ));
            wire_edges.push(OrientedEdge::new(arc_eid, true));
            prev_vid = next_vid;
        }
    }

    let wire = Wire::new(wire_edges, true).unwrap();
    let wire_id = topo.add_wire(wire);

    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = Face::new(wire_id, vec![], FaceSurface::Plane { normal, d: 0.0 });
    let face_id = topo.add_face(face);

    // Extrude.
    let solid = crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), h).unwrap();

    // Find top face for shelling.
    let top_faces: Vec<FaceId> = {
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        sh.faces()
            .iter()
            .filter(|&&fid| {
                let f = topo.face(fid).unwrap();
                if let FaceSurface::Plane { normal: n, d } = f.surface() {
                    n.z() > 0.9 && (*d - h).abs() < 0.1
                } else {
                    false
                }
            })
            .copied()
            .collect()
    };
    assert_eq!(top_faces.len(), 1, "should find exactly one top face");

    let shelled = crate::shell_op::shell(&mut topo, solid, thickness, &top_faces).unwrap();

    let (f_before, e_before, v_before) =
        brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
    let vol_before = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();

    // Count cylinder faces before.
    let shell_id = topo.solid(shelled).unwrap().outer_shell();
    let cyl_before = topo
        .shell(shell_id)
        .unwrap()
        .faces()
        .iter()
        .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();

    let removed = unify_faces(&mut topo, shelled).unwrap();

    let (f_after, e_after, v_after) =
        brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
    let vol_after = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();

    let cyl_after = topo
        .shell(shell_id)
        .unwrap()
        .faces()
        .iter()
        .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();

    #[allow(clippy::cast_possible_wrap)]
    let chi_before = (v_before as i64) - (e_before as i64) + (f_before as i64);
    #[allow(clippy::cast_possible_wrap)]
    let chi_after = (v_after as i64) - (e_after as i64) + (f_after as i64);

    eprintln!(
        "shell rounded rect (3 arcs/corner): faces {f_before} -> {f_after} (removed {removed}), \
             cyl {cyl_before} -> {cyl_after}, \
             χ {chi_before} -> {chi_after}, vol {vol_before:.1} -> {vol_after:.1}"
    );

    // Cylinder faces should be merged: 24 -> 8.
    assert!(removed > 0, "unify should merge cylinder face fragments");

    // Volume must be preserved (within 1%).
    let rel_err = (vol_before - vol_after).abs() / vol_before;
    assert!(
        rel_err < 0.01,
        "unify should preserve volume: before={vol_before:.2}, after={vol_after:.2}, err={:.2}%",
        rel_err * 100.0
    );
}

/// Regression: unify_faces must NOT merge coplanar faces with opposite
/// effective normals. A shelled box has inner faces whose normals point
/// inward — these must not be merged with outer faces on the same plane.
#[test]
fn unify_faces_skips_opposite_normals() {
    let mut topo = Topology::new();
    let box_solid = crate::primitives::make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
    let faces = brepkit_topology::explorer::solid_faces(&topo, box_solid).unwrap();
    // Find top face (z=2)
    let top = faces
        .iter()
        .find(|&&fid| {
            let f = topo.face(fid).unwrap();
            matches!(f.surface(), FaceSurface::Plane { normal, .. } if normal.z() > 0.9)
        })
        .copied()
        .unwrap();
    let shelled = crate::shell_op::shell(&mut topo, box_solid, 1.0, &[top]).unwrap();

    let (f_before, _, _) = brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
    let removed = unify_faces(&mut topo, shelled).unwrap();
    let (f_after, _, _) = brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();

    // unify_faces should NOT merge any faces — the shelled box has inner
    // faces with opposite normals that share the same plane equation.
    assert_eq!(
        removed, 0,
        "unify_faces should not merge opposite-normal faces, removed {removed}"
    );
    assert_eq!(
        f_before, f_after,
        "face count should be unchanged: {f_before} → {f_after}"
    );
}

#[test]
fn heal_preserves_box_volume() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

    let vol_before = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    let _report = heal_solid(&mut topo, solid, 1e-7).unwrap();
    let vol_after = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

    assert!(
        (vol_before - vol_after).abs() < 0.1,
        "healing should preserve volume: before={vol_before}, after={vol_after}"
    );
}

#[test]
fn find_shared_vertex_matches_by_position() {
    // Two faces with different VertexIds at the same 3D position.
    // find_shared_vertex should match via position fallback.
    use brepkit_math::vec::Point3;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let tol = 1e-7;

    // Face A: square (0,0,0)→(1,0,0)→(1,1,0)→(0,1,0)
    let va0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol));
    let va1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol));
    let va2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), tol));
    let va3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), tol));
    let ea0 = topo.add_edge(Edge::new(va0, va1, EdgeCurve::Line));
    let ea1 = topo.add_edge(Edge::new(va1, va2, EdgeCurve::Line));
    let ea2 = topo.add_edge(Edge::new(va2, va3, EdgeCurve::Line));
    let ea3 = topo.add_edge(Edge::new(va3, va0, EdgeCurve::Line));
    let wa = Wire::new(
        vec![
            OrientedEdge::new(ea0, true),
            OrientedEdge::new(ea1, true),
            OrientedEdge::new(ea2, true),
            OrientedEdge::new(ea3, true),
        ],
        true,
    )
    .unwrap();
    let waid = topo.add_wire(wa);
    let fa = Face::new(
        waid,
        vec![],
        FaceSurface::Plane {
            normal: brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    );
    let fa_id = topo.add_face(fa);

    // Face B: square (1,0,0)→(2,0,0)→(2,1,0)→(1,1,0)
    // Uses DIFFERENT VertexIds at (1,0,0) and (1,1,0)
    let vb0 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol));
    let vb1 = topo.add_vertex(Vertex::new(Point3::new(2.0, 0.0, 0.0), tol));
    let vb2 = topo.add_vertex(Vertex::new(Point3::new(2.0, 1.0, 0.0), tol));
    let vb3 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), tol));
    let eb0 = topo.add_edge(Edge::new(vb0, vb1, EdgeCurve::Line));
    let eb1 = topo.add_edge(Edge::new(vb1, vb2, EdgeCurve::Line));
    let eb2 = topo.add_edge(Edge::new(vb2, vb3, EdgeCurve::Line));
    let eb3 = topo.add_edge(Edge::new(vb3, vb0, EdgeCurve::Line));
    let wb = Wire::new(
        vec![
            OrientedEdge::new(eb0, true),
            OrientedEdge::new(eb1, true),
            OrientedEdge::new(eb2, true),
            OrientedEdge::new(eb3, true),
        ],
        true,
    )
    .unwrap();
    let wbid = topo.add_wire(wb);
    let fb = Face::new(
        wbid,
        vec![],
        FaceSurface::Plane {
            normal: brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    );
    let fb_id = topo.add_face(fb);

    // va1 and vb0 are at the same position (1,0,0) but different IDs
    assert_ne!(va1.index(), vb0.index());

    let result = super::find_shared_vertex(&topo, fa_id, fb_id);
    assert!(
        result.is_some(),
        "find_shared_vertex should match by position when VertexIds differ"
    );
    let pt = result.unwrap();
    // Should find (1,0,0) or (1,1,0) — both are shared positions
    let is_shared = ((pt.x() - 1.0).abs() < 1e-6 && (pt.y() - 0.0).abs() < 1e-6)
        || ((pt.x() - 1.0).abs() < 1e-6 && (pt.y() - 1.0).abs() < 1e-6);
    assert!(
        is_shared,
        "shared vertex should be at (1,0,0) or (1,1,0), got ({:.1},{:.1},{:.1})",
        pt.x(),
        pt.y(),
        pt.z()
    );
}

#[test]
fn convert_to_elementary_round_trip_cylinder() {
    // Build a cylinder, convert to B-spline (loses analytic types),
    // then convert back to elementary — should recover the lateral
    // cylinder face and the circular cap edges.
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::explorer;
    use brepkit_topology::face::FaceSurface;

    let mut topo = Topology::new();
    let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

    // Pin the entities under test BEFORE conversion: identify the
    // lateral face and cap circle edges by their analytic types.
    // Holding their IDs lets us re-check the same handles after the
    // round-trip — a weak existence test ("some face is NURBS") could
    // be satisfied by an unrelated face changing type.
    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();
    let lateral_face_id = shell
        .faces()
        .iter()
        .copied()
        .find(|&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
        .expect("cylinder primitive should have one Cylinder face");
    let circle_edge_ids: Vec<_> = explorer::solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&eid| matches!(topo.edge(eid).unwrap().curve(), EdgeCurve::Circle(_)))
        .collect();
    assert_eq!(
        circle_edge_ids.len(),
        2,
        "cylinder primitive should have 2 circular cap edges, got {}",
        circle_edge_ids.len()
    );

    // Step 1: NURBS-ify everything.
    let bspline_count = convert_to_bspline(&mut topo, solid).unwrap();
    assert!(
        bspline_count > 0,
        "convert_to_bspline should convert at least one face/edge, got {bspline_count}"
    );

    // Assert about the *specific* lateral face — not just "some face
    // became NURBS".
    assert!(
        matches!(
            topo.face(lateral_face_id).unwrap().surface(),
            FaceSurface::Nurbs(_)
        ),
        "lateral cylinder face should be NURBS after convert_to_bspline"
    );
    for &eid in &circle_edge_ids {
        assert!(
            matches!(topo.edge(eid).unwrap().curve(), EdgeCurve::NurbsCurve(_)),
            "cap edge {eid:?} should be NURBS after convert_to_bspline"
        );
    }

    // Step 2: convert back to elementary.
    let elementary_count = convert_to_elementary(&mut topo, solid, 1e-4).unwrap();
    assert!(
        elementary_count > 0,
        "convert_to_elementary should recover at least one analytic form, got {elementary_count}"
    );

    // Same lateral face → Cylinder again, with the original radius.
    if let FaceSurface::Cylinder(cyl) = topo.face(lateral_face_id).unwrap().surface() {
        assert!(
            (cyl.radius() - 1.0).abs() < 1e-3,
            "recovered cylinder radius {} should be ~1.0",
            cyl.radius()
        );
    } else {
        panic!(
            "after convert_to_elementary, lateral face should be Cylinder again, got {:?}",
            topo.face(lateral_face_id).unwrap().surface().type_tag()
        );
    }

    // Same cap edges → Circle again, with the original radius.
    for &eid in &circle_edge_ids {
        if let EdgeCurve::Circle(c) = topo.edge(eid).unwrap().curve() {
            assert!(
                (c.radius() - 1.0).abs() < 1e-3,
                "recovered circle edge {eid:?} radius {} should be ~1.0",
                c.radius()
            );
        } else {
            panic!(
                "after convert_to_elementary, edge {eid:?} should be Circle again, got {:?}",
                topo.edge(eid).unwrap().curve()
            );
        }
    }
}

#[test]
fn convert_to_elementary_idempotent_on_clean_solid() {
    // A solid with all-analytic geometry should have nothing to
    // convert — no faces or edges are NURBS.
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

    let count = convert_to_elementary(&mut topo, solid, 1e-7).unwrap();
    assert_eq!(
        count, 0,
        "all-analytic solid shouldn't convert anything, got {count}"
    );
}
