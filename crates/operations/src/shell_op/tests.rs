#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    clippy::cast_possible_wrap
)]

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::test_utils::make_unit_cube_manifold;

use super::*;

/// Helper: get face IDs matching a given normal direction.
fn find_faces_by_normal(topo: &Topology, solid: SolidId, target_normal: Vec3) -> Vec<FaceId> {
    let tol = Tolerance::loose();
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    let mut result = Vec::new();
    for &fid in sh.faces() {
        let f = topo.face(fid).unwrap();
        if let FaceSurface::Plane { normal, .. } = f.surface()
            && tol.approx_eq(normal.x(), target_normal.x())
            && tol.approx_eq(normal.y(), target_normal.y())
            && tol.approx_eq(normal.z(), target_normal.z())
        {
            result.push(fid);
        }
    }
    result
}

#[test]
fn shell_closed_box() {
    let mut topo = Topology::new();
    let cube = make_unit_cube_manifold(&mut topo);

    let result = shell(&mut topo, cube, 0.1, &[]).unwrap();

    let s = topo.solid(result).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    // 6 outer + 6 inner = 12 faces (no rim faces since no openings).
    assert_eq!(sh.faces().len(), 12, "closed shell should have 12 faces");
}

#[test]
fn shell_open_top() {
    let mut topo = Topology::new();
    let cube = make_unit_cube_manifold(&mut topo);

    let top_faces = find_faces_by_normal(&topo, cube, Vec3::new(0.0, 0.0, 1.0));
    assert_eq!(top_faces.len(), 1, "should find exactly one +Z face");

    let result = shell(&mut topo, cube, 0.1, &top_faces).unwrap();

    let s = topo.solid(result).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    // 5 outer + 5 inner + 1 annular rim = 11 faces
    assert_eq!(sh.faces().len(), 11, "open-top shell should have 11 faces");

    // Check volume accuracy: 1 - 0.8*0.8*0.9 = 0.424
    let vol = crate::measure::solid_volume(&topo, result, 0.01).unwrap();
    let expected = 1.0 - 0.8 * 0.8 * 0.9;
    eprintln!("[shell_open_top] volume: {vol:.6}, expected: {expected:.6}");
}

#[test]
fn shell_volume_decrease() {
    let mut topo = Topology::new();
    let cube = make_unit_cube_manifold(&mut topo);

    let original_vol = crate::measure::solid_volume(&topo, cube, 0.1).unwrap();

    let result = shell(&mut topo, cube, 0.1, &[]).unwrap();
    let shell_vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();

    // The shelled solid should have less volume than the original
    // (we removed the interior).
    assert!(
        shell_vol < original_vol,
        "shell volume ({shell_vol}) should be less than original ({original_vol})"
    );
    assert!(
        shell_vol > 0.0,
        "shell volume should be positive, got {shell_vol}"
    );
}

#[test]
fn shell_zero_thickness_error() {
    let mut topo = Topology::new();
    let cube = make_unit_cube_manifold(&mut topo);
    assert!(shell(&mut topo, cube, 0.0, &[]).is_err());
}

#[test]
fn shell_negative_thickness_error() {
    let mut topo = Topology::new();
    let cube = make_unit_cube_manifold(&mut topo);
    assert!(shell(&mut topo, cube, -0.1, &[]).is_err());
}

#[test]
fn shell_two_open_faces_volume() {
    let mut topo = Topology::new();
    let cube = make_unit_cube_manifold(&mut topo);

    let top = find_faces_by_normal(&topo, cube, Vec3::new(0.0, 0.0, 1.0));
    let bot = find_faces_by_normal(&topo, cube, Vec3::new(0.0, 0.0, -1.0));
    let mut open_faces = top;
    open_faces.extend(bot);
    assert_eq!(open_faces.len(), 2);

    let result = shell(&mut topo, cube, 0.1, &open_faces).unwrap();
    let vol = crate::measure::solid_volume(&topo, result, 0.01).unwrap();
    // Expected: 1.0 - 0.8*0.8*1.0 = 0.36
    assert!(vol > 0.1, "tube shell volume should be positive, got {vol}");
    assert!(
        vol < 1.0,
        "tube shell volume should be < original 1.0, got {vol}"
    );
}

/// Simulates the gridfinity "1×1 flat no-lip" pipeline:
/// rounded rectangle → extrude → shell (open top).
/// Reports face count and volume for debugging parity issues.
#[test]
fn shell_rounded_rect_extrude_diagnostics() {
    use crate::primitives::make_box;

    let mut topo = Topology::new();

    // Gridfinity dimensions: 41.5×41.5×21mm, 4mm corner radius, 1.2mm wall thickness
    let w = 41.5;
    let d = 41.5;
    let h = 21.0;
    let thickness = 1.2;

    // Use a simple box (no rounded corners) to isolate shell behavior.
    let box_solid = make_box(&mut topo, w, d, h).unwrap();

    let box_shell_data = topo
        .shell(topo.solid(box_solid).unwrap().outer_shell())
        .unwrap();
    let extrude_face_count = box_shell_data.faces().len();
    eprintln!("[diag] Box extrude face count: {extrude_face_count}");
    assert_eq!(extrude_face_count, 6);

    let top_faces = find_faces_by_normal(&topo, box_solid, Vec3::new(0.0, 0.0, 1.0));
    assert_eq!(top_faces.len(), 1, "should find exactly one top face");

    let shelled = shell(&mut topo, box_solid, thickness, &top_faces).unwrap();
    let sh = topo
        .shell(topo.solid(shelled).unwrap().outer_shell())
        .unwrap();
    let shell_face_count = sh.faces().len();
    eprintln!("[diag] Box shell face count: {shell_face_count}");
    // 5 outer + 5 inner + 1 annular rim = 11
    assert_eq!(shell_face_count, 11, "box shell should have 11 faces");

    let box_vol = crate::measure::solid_volume(&topo, box_solid, 0.01).unwrap();
    let expected_box_vol = w * d * h;
    eprintln!("[diag] Box volume: {box_vol:.2}, expected: {expected_box_vol:.2}");

    let vol = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();
    let expected_vol = w * d * h - (w - 2.0 * thickness) * (d - 2.0 * thickness) * (h - thickness);
    let pct = (vol - expected_vol).abs() / expected_vol;
    eprintln!("[diag] Shell volume: {vol:.2}, expected: {expected_vol:.2}, diff: {pct:.4}");

    for &fid in sh.faces() {
        let f = topo.face(fid).unwrap();
        let kind = match f.surface() {
            FaceSurface::Plane { .. } => "Plane",
            FaceSurface::Cylinder(_) => "Cylinder",
            FaceSurface::Cone(_) => "Cone",
            FaceSurface::Sphere(_) => "Sphere",
            FaceSurface::Torus(_) => "Torus",
            FaceSurface::Nurbs(_) => "Nurbs",
        };
        let wire = topo.wire(f.outer_wire()).unwrap();
        eprintln!(
            "[diag]   Face {}: {kind}, {} edges",
            fid.index(),
            wire.edges().len()
        );
    }

    assert!(
        pct < 0.05,
        "shell volume should be within 5% of expected, got {pct:.4}"
    );
}

/// Gridfinity exact parameters (r=4mm corner radius) diagnostic.
#[test]
fn shell_gridfinity_exact_params() {
    use brepkit_math::curves::Circle3D;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let tol = Tolerance::new();

    // Exact gridfinity 1×1 flat no-lip parameters
    let w = 41.5_f64; // 1 × 42 - 0.5 clearance
    let d = 41.5_f64;
    let h = 21.0_f64; // 3 height units × 7mm
    let r = 4.0_f64; // CORNER_RADIUS = SOCKET_CORNER_RADIUS = 4mm
    let thickness = 1.2_f64;

    let hw = w / 2.0;
    let hd = d / 2.0;

    let v0 = Point3::new(hw - r, -hd, 0.0);
    let v1 = Point3::new(hw, -hd + r, 0.0);
    let v2 = Point3::new(hw, hd - r, 0.0);
    let v3 = Point3::new(hw - r, hd, 0.0);
    let v4 = Point3::new(-hw + r, hd, 0.0);
    let v5 = Point3::new(-hw, hd - r, 0.0);
    let v6 = Point3::new(-hw, -hd + r, 0.0);
    let v7 = Point3::new(-hw + r, -hd, 0.0);

    let vids: Vec<_> = [v0, v1, v2, v3, v4, v5, v6, v7]
        .iter()
        .map(|p| topo.add_vertex(Vertex::new(*p, tol.linear)))
        .collect();

    let c_br = Point3::new(hw - r, -hd + r, 0.0);
    let c_tr = Point3::new(hw - r, hd - r, 0.0);
    let c_tl = Point3::new(-hw + r, hd - r, 0.0);
    let c_bl = Point3::new(-hw + r, -hd + r, 0.0);

    let z_axis = Vec3::new(0.0, 0.0, 1.0);

    let mk_line = |topo: &mut Topology, s, e| topo.add_edge(Edge::new(s, e, EdgeCurve::Line));
    let mk_arc = |topo: &mut Topology, s, e, center: Point3| {
        let circle = Circle3D::new(center, z_axis, r).unwrap();
        topo.add_edge(Edge::new(s, e, EdgeCurve::Circle(circle)))
    };

    let e_bot = mk_line(&mut topo, vids[7], vids[0]);
    let e_br = mk_arc(&mut topo, vids[0], vids[1], c_br);
    let e_right = mk_line(&mut topo, vids[1], vids[2]);
    let e_tr = mk_arc(&mut topo, vids[2], vids[3], c_tr);
    let e_top = mk_line(&mut topo, vids[3], vids[4]);
    let e_tl = mk_arc(&mut topo, vids[4], vids[5], c_tl);
    let e_left = mk_line(&mut topo, vids[5], vids[6]);
    let e_bl = mk_arc(&mut topo, vids[6], vids[7], c_bl);

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e_bot, true),
            OrientedEdge::new(e_br, true),
            OrientedEdge::new(e_right, true),
            OrientedEdge::new(e_tr, true),
            OrientedEdge::new(e_top, true),
            OrientedEdge::new(e_tl, true),
            OrientedEdge::new(e_left, true),
            OrientedEdge::new(e_bl, true),
        ],
        true,
    )
    .unwrap();
    let wire_id = topo.add_wire(wire);
    let face = Face::new(
        wire_id,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    );
    let face_id = topo.add_face(face);

    let solid = crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), h).unwrap();

    let sh_before = topo
        .shell(topo.solid(solid).unwrap().outer_shell())
        .unwrap();
    let fc_before = sh_before.faces().len();
    eprintln!("[gf-exact] Faces after extrude: {fc_before}");

    let top = find_faces_by_normal(&topo, solid, Vec3::new(0.0, 0.0, 1.0));
    assert_eq!(top.len(), 1, "one top face");

    let shelled = shell(&mut topo, solid, thickness, &top).unwrap();
    let sh2 = topo
        .shell(topo.solid(shelled).unwrap().outer_shell())
        .unwrap();
    let fc_after = sh2.faces().len();
    eprintln!("[gf-exact] Faces after shell: {fc_after}");

    let (f, e, v) = brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
    let chi = v as i64 - e as i64 + f as i64;
    eprintln!("[gf-exact] F={f}, E={e}, V={v}, χ={chi}");

    let vol = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();
    eprintln!("[gf-exact] Volume: {vol:.2}");

    let result = crate::validate::validate_solid(&topo, shelled);
    eprintln!("[gf-exact] Validation: {result:?}");

    let removed = crate::heal::unify_faces(&mut topo, shelled).unwrap();
    let sh3 = topo
        .shell(topo.solid(shelled).unwrap().outer_shell())
        .unwrap();
    let fc_unified = sh3.faces().len();
    eprintln!("[gf-exact] After unify_faces (removed {removed}): {fc_unified} faces");

    let (f2, e2, v2) = brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
    let chi2 = v2 as i64 - e2 as i64 + f2 as i64;
    eprintln!("[gf-exact] After unify: F={f2}, E={e2}, V={v2}, χ={chi2}");
}

/// Rounded rectangle extrusion → shell: the gridfinity "1×1 flat no-lip" path.
/// This test creates a face with lines + circle arcs, extrudes, then shells.
#[test]
fn shell_rounded_rect_with_arcs() {
    use brepkit_math::curves::Circle3D;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let tol = Tolerance::new();

    // Parameters matching gridfinity 1×1.
    let w = 41.5_f64;
    let d = 41.5_f64;
    let h = 21.0_f64;
    let r = 2.6_f64; // corner radius
    let thickness = 1.2_f64;

    // Rounded rectangle on XY at z=0:
    //   4 line segments + 4 quarter-circle arcs.
    // Vertices at the tangent points (where lines meet arcs).
    let hw = w / 2.0;
    let hd = d / 2.0;

    // Tangent points (CCW from bottom-right):
    let v0 = Point3::new(hw - r, -hd, 0.0);
    let v1 = Point3::new(hw, -hd + r, 0.0);
    let v2 = Point3::new(hw, hd - r, 0.0);
    let v3 = Point3::new(hw - r, hd, 0.0);
    let v4 = Point3::new(-hw + r, hd, 0.0);
    let v5 = Point3::new(-hw, hd - r, 0.0);
    let v6 = Point3::new(-hw, -hd + r, 0.0);
    let v7 = Point3::new(-hw + r, -hd, 0.0);

    let vids: Vec<_> = [v0, v1, v2, v3, v4, v5, v6, v7]
        .iter()
        .map(|p| topo.add_vertex(Vertex::new(*p, tol.linear)))
        .collect();

    // Corner centers:
    let c_br = Point3::new(hw - r, -hd + r, 0.0);
    let c_tr = Point3::new(hw - r, hd - r, 0.0);
    let c_tl = Point3::new(-hw + r, hd - r, 0.0);
    let c_bl = Point3::new(-hw + r, -hd + r, 0.0);

    let z_axis = Vec3::new(0.0, 0.0, 1.0);

    let mk_line = |topo: &mut Topology, s, e| topo.add_edge(Edge::new(s, e, EdgeCurve::Line));
    let mk_arc = |topo: &mut Topology, s, e, center: Point3| {
        let circle = Circle3D::new(center, z_axis, r).unwrap();
        topo.add_edge(Edge::new(s, e, EdgeCurve::Circle(circle)))
    };

    let e_bot = mk_line(&mut topo, vids[7], vids[0]);
    let e_br = mk_arc(&mut topo, vids[0], vids[1], c_br);
    let e_right = mk_line(&mut topo, vids[1], vids[2]);
    let e_tr = mk_arc(&mut topo, vids[2], vids[3], c_tr);
    let e_top = mk_line(&mut topo, vids[3], vids[4]);
    let e_tl = mk_arc(&mut topo, vids[4], vids[5], c_tl);
    let e_left = mk_line(&mut topo, vids[5], vids[6]);
    let e_bl = mk_arc(&mut topo, vids[6], vids[7], c_bl);

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e_bot, true),
            OrientedEdge::new(e_br, true),
            OrientedEdge::new(e_right, true),
            OrientedEdge::new(e_tr, true),
            OrientedEdge::new(e_top, true),
            OrientedEdge::new(e_tl, true),
            OrientedEdge::new(e_left, true),
            OrientedEdge::new(e_bl, true),
        ],
        true,
    )
    .unwrap();
    let wire_id = topo.add_wire(wire);

    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = Face::new(wire_id, vec![], FaceSurface::Plane { normal, d: 0.0 });
    let face_id = topo.add_face(face);

    let solid = crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), h).unwrap();

    let sh = topo
        .shell(topo.solid(solid).unwrap().outer_shell())
        .unwrap();
    let extrude_fc = sh.faces().len();
    eprintln!("[rounded] Extrude faces: {extrude_fc}");
    // Expected: 2 caps + 8 sides (4 planar + 4 cylindrical) = 10
    assert_eq!(extrude_fc, 10, "extruded rounded rect should have 10 faces");

    let mut plane_count = 0;
    let mut cyl_count = 0;
    for &fid in sh.faces() {
        let f = topo.face(fid).unwrap();
        match f.surface() {
            FaceSurface::Plane { .. } => plane_count += 1,
            FaceSurface::Cylinder(_) => cyl_count += 1,
            _ => {}
        }
    }
    eprintln!("[rounded] Extrude: {plane_count} planar, {cyl_count} cylinder");
    assert_eq!(plane_count, 6, "4 flat sides + 2 caps = 6 planar");
    assert_eq!(cyl_count, 4, "4 corner cylinders");

    // Expected: A = w*d - 4*r^2*(1-pi/4), V = A*h
    let expected_area = w * d - 4.0 * r * r * (1.0 - std::f64::consts::FRAC_PI_4);
    let expected_vol = expected_area * h;
    let extrude_vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
    let rel_err = (extrude_vol - expected_vol).abs() / expected_vol;
    eprintln!(
        "[rounded] Extrude volume: {extrude_vol:.2} (expected {expected_vol:.2}, diff {:.4}%)",
        rel_err * 100.0
    );
    assert!(
        rel_err < 0.001,
        "extrude volume error {rel_err:.6} exceeds 0.1%"
    );

    let top = find_faces_by_normal(&topo, solid, Vec3::new(0.0, 0.0, 1.0));
    assert_eq!(top.len(), 1, "one top face");

    let shelled = shell(&mut topo, solid, thickness, &top).unwrap();
    let sh2 = topo
        .shell(topo.solid(shelled).unwrap().outer_shell())
        .unwrap();
    let shell_fc = sh2.faces().len();
    eprintln!("[rounded] Shell faces: {shell_fc}");

    let mut sp = 0;
    let mut sc = 0;
    for &fid in sh2.faces() {
        let f = topo.face(fid).unwrap();
        match f.surface() {
            FaceSurface::Plane { .. } => sp += 1,
            FaceSurface::Cylinder(_) => sc += 1,
            _ => {}
        }
        let w2 = topo.wire(f.outer_wire()).unwrap();
        let kind = match f.surface() {
            FaceSurface::Plane { .. } => "Plane",
            FaceSurface::Cylinder(_) => "Cyl",
            _ => "Other",
        };
        eprintln!(
            "[rounded]   Face {}: {kind}, {} edges",
            fid.index(),
            w2.edges().len()
        );
    }
    eprintln!("[rounded] Shell: {sp} planar, {sc} cylinder");

    {
        let rim_fid = *sh2.faces().last().unwrap();
        let rim_f = topo.face(rim_fid).unwrap();
        let outer_w = topo.wire(rim_f.outer_wire()).unwrap();
        eprintln!(
            "[rim-diag] Rim face {}: outer wire has {} edges, {} inner wires",
            rim_fid.index(),
            outer_w.edges().len(),
            rim_f.inner_wires().len()
        );
        for (i, oe) in outer_w.edges().iter().enumerate() {
            let e = topo.edge(oe.edge()).unwrap();
            let sv = topo.vertex(e.start()).unwrap().point();
            let ev = topo.vertex(e.end()).unwrap().point();
            let kind = match e.curve() {
                brepkit_topology::edge::EdgeCurve::Line => "Line",
                brepkit_topology::edge::EdgeCurve::Circle(_) => "Circle",
                _ => "Other",
            };
            eprintln!(
                "[rim-diag]   outer[{i}]: {kind} fwd={} ({:.2},{:.2},{:.2})->({:.2},{:.2},{:.2})",
                oe.is_forward(),
                sv.x(),
                sv.y(),
                sv.z(),
                ev.x(),
                ev.y(),
                ev.z()
            );
        }
        for (iw_idx, &iw_id) in rim_f.inner_wires().iter().enumerate() {
            let iw = topo.wire(iw_id).unwrap();
            eprintln!("[rim-diag] Inner wire {iw_idx}: {} edges", iw.edges().len());
            for (i, oe) in iw.edges().iter().enumerate() {
                let e = topo.edge(oe.edge()).unwrap();
                let sv = topo.vertex(e.start()).unwrap().point();
                let ev = topo.vertex(e.end()).unwrap().point();
                let kind = match e.curve() {
                    brepkit_topology::edge::EdgeCurve::Line => "Line",
                    brepkit_topology::edge::EdgeCurve::Circle(_) => "Circle",
                    _ => "Other",
                };
                eprintln!(
                    "[rim-diag]   inner[{i}]: {kind} fwd={} ({:.2},{:.2},{:.2})->({:.2},{:.2},{:.2})",
                    oe.is_forward(),
                    sv.x(),
                    sv.y(),
                    sv.z(),
                    ev.x(),
                    ev.y(),
                    ev.z()
                );
            }
        }
    }

    for &fid in sh2.faces() {
        let f = topo.face(fid).unwrap();
        let kind = match f.surface() {
            FaceSurface::Plane { .. } => "Plane",
            FaceSurface::Cylinder(_) => "Cyl",
            _ => "Other",
        };
        if f.is_reversed() {
            eprintln!("[rounded]   Face {}: {kind} REVERSED", fid.index());
        }
    }

    let vol = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();
    eprintln!("[rounded] Shell volume: {vol:.2}");

    let result = crate::validate::validate_solid(&topo, shelled);
    eprintln!("[rounded] Validation: {result:?}");
}

/// CW-wound rounded rectangle (brepjs convention) → extrude → shell.
/// This is the exact scenario where the shell bbox was expanding outward.
#[test]
fn shell_cw_rounded_rect_bounds_preserved() {
    use brepkit_math::curves::Circle3D;
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

    // CW winding (brepjs convention): BOTTOM→RIGHT→TOP→LEFT
    // Start at bottom-left tangent point, go right
    let pts = [
        Point3::new(-hw + r, -hd, 0.0), // 0: bottom-left straight start
        Point3::new(hw - r, -hd, 0.0),  // 1: bottom-right straight end
        Point3::new(hw, -hd + r, 0.0),  // 2: right-bottom straight start
        Point3::new(hw, hd - r, 0.0),   // 3: right-top straight end
        Point3::new(hw - r, hd, 0.0),   // 4: top-right straight start
        Point3::new(-hw + r, hd, 0.0),  // 5: top-left straight end
        Point3::new(-hw, hd - r, 0.0),  // 6: left-top straight start
        Point3::new(-hw, -hd + r, 0.0), // 7: left-bottom straight end
    ];
    let vids: Vec<_> = pts
        .iter()
        .map(|p| topo.add_vertex(Vertex::new(*p, tol.linear)))
        .collect();

    let c_br = Point3::new(hw - r, -hd + r, 0.0);
    let c_tr = Point3::new(hw - r, hd - r, 0.0);
    let c_tl = Point3::new(-hw + r, hd - r, 0.0);
    let c_bl = Point3::new(-hw + r, -hd + r, 0.0);
    let z_axis = Vec3::new(0.0, 0.0, 1.0);

    let mk_line = |topo: &mut Topology, s, e| topo.add_edge(Edge::new(s, e, EdgeCurve::Line));
    let mk_arc = |topo: &mut Topology, s, e, center: Point3| {
        let circle = Circle3D::new(center, z_axis, r).unwrap();
        topo.add_edge(Edge::new(s, e, EdgeCurve::Circle(circle)))
    };

    // CW order: bottom→br_arc→right→tr_arc→top→tl_arc→left→bl_arc
    let e_bot = mk_line(&mut topo, vids[0], vids[1]);
    let e_br = mk_arc(&mut topo, vids[1], vids[2], c_br);
    let e_right = mk_line(&mut topo, vids[2], vids[3]);
    let e_tr = mk_arc(&mut topo, vids[3], vids[4], c_tr);
    let e_top = mk_line(&mut topo, vids[4], vids[5]);
    let e_tl = mk_arc(&mut topo, vids[5], vids[6], c_tl);
    let e_left = mk_line(&mut topo, vids[6], vids[7]);
    let e_bl = mk_arc(&mut topo, vids[7], vids[0], c_bl);

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e_bot, true),
            OrientedEdge::new(e_br, true),
            OrientedEdge::new(e_right, true),
            OrientedEdge::new(e_tr, true),
            OrientedEdge::new(e_top, true),
            OrientedEdge::new(e_tl, true),
            OrientedEdge::new(e_left, true),
            OrientedEdge::new(e_bl, true),
        ],
        true,
    )
    .unwrap();
    let wire_id = topo.add_wire(wire);

    // CW winding → face normal should be -Z
    let face = Face::new(
        wire_id,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    );
    let face_id = topo.add_face(face);

    let solid = crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), h).unwrap();

    let top = find_faces_by_normal(&topo, solid, Vec3::new(0.0, 0.0, 1.0));
    assert_eq!(top.len(), 1, "one top face");

    let shelled = shell(&mut topo, solid, thickness, &top).unwrap();

    // Key assertion: bounding box should NOT expand beyond the original
    let bbox = crate::measure::solid_bounding_box(&topo, shelled).unwrap();
    let bbox_x = bbox.max.x() - bbox.min.x();
    let bbox_y = bbox.max.y() - bbox.min.y();
    eprintln!("[cw-shell] bbox X={bbox_x:.3}, Y={bbox_y:.3} (expected ~{w})");
    assert!(
        (bbox_x - w).abs() < 0.5,
        "bbox X should be ~{w}, got {bbox_x:.3} (expanded by {:.3})",
        bbox_x - w
    );
    assert!(
        (bbox_y - d).abs() < 0.5,
        "bbox Y should be ~{d}, got {bbox_y:.3} (expanded by {:.3})",
        bbox_y - d
    );
}

/// Regression oracle for arc-edge identity in shelled rounded rects.
///
/// When wire traversal runs u-decreasing around a corner cylinder, the
/// stored `EdgeCurve::Circle` arc must still be the intended 90-degree
/// corner arc, not its 270-degree complement. The complement corrupts
/// face areas, the synthesized rim annulus, the floor trim, and makes
/// tessellation non-watertight.
#[test]
#[allow(
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::items_after_statements
)]
fn shell_rounded_rect_watertight() {
    use std::collections::HashMap;

    use brepkit_math::curves::Circle3D;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let tol = Tolerance::new();

    // Gridfinity 1x1 bin body: 41.5 x 41.5, corner radius 3.75.
    let w = 41.5_f64;
    let d = 41.5_f64;
    let h = 21.0_f64;
    let r = 3.75_f64;
    let thickness = 1.2_f64;

    let hw = w / 2.0;
    let hd = d / 2.0;

    let pts = [
        Point3::new(hw - r, -hd, 0.0),
        Point3::new(hw, -hd + r, 0.0),
        Point3::new(hw, hd - r, 0.0),
        Point3::new(hw - r, hd, 0.0),
        Point3::new(-hw + r, hd, 0.0),
        Point3::new(-hw, hd - r, 0.0),
        Point3::new(-hw, -hd + r, 0.0),
        Point3::new(-hw + r, -hd, 0.0),
    ];
    let vids: Vec<_> = pts
        .iter()
        .map(|p| topo.add_vertex(Vertex::new(*p, tol.linear)))
        .collect();

    let c_br = Point3::new(hw - r, -hd + r, 0.0);
    let c_tr = Point3::new(hw - r, hd - r, 0.0);
    let c_tl = Point3::new(-hw + r, hd - r, 0.0);
    let c_bl = Point3::new(-hw + r, -hd + r, 0.0);
    let z_axis = Vec3::new(0.0, 0.0, 1.0);

    let mk_line = |topo: &mut Topology, s, e| topo.add_edge(Edge::new(s, e, EdgeCurve::Line));
    let mk_arc = |topo: &mut Topology, s, e, center: Point3| {
        let circle = Circle3D::new(center, z_axis, r).unwrap();
        topo.add_edge(Edge::new(s, e, EdgeCurve::Circle(circle)))
    };

    let e_bot = mk_line(&mut topo, vids[7], vids[0]);
    let e_br = mk_arc(&mut topo, vids[0], vids[1], c_br);
    let e_right = mk_line(&mut topo, vids[1], vids[2]);
    let e_tr = mk_arc(&mut topo, vids[2], vids[3], c_tr);
    let e_top = mk_line(&mut topo, vids[3], vids[4]);
    let e_tl = mk_arc(&mut topo, vids[4], vids[5], c_tl);
    let e_left = mk_line(&mut topo, vids[5], vids[6]);
    let e_bl = mk_arc(&mut topo, vids[6], vids[7], c_bl);

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e_bot, true),
            OrientedEdge::new(e_br, true),
            OrientedEdge::new(e_right, true),
            OrientedEdge::new(e_tr, true),
            OrientedEdge::new(e_top, true),
            OrientedEdge::new(e_tl, true),
            OrientedEdge::new(e_left, true),
            OrientedEdge::new(e_bl, true),
        ],
        true,
    )
    .unwrap();
    let wire_id = topo.add_wire(wire);

    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = Face::new(wire_id, vec![], FaceSurface::Plane { normal, d: 0.0 });
    let face_id = topo.add_face(face);

    let solid = crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), h).unwrap();

    let top = find_faces_by_normal(&topo, solid, Vec3::new(0.0, 0.0, 1.0));
    assert_eq!(top.len(), 1, "one top face");

    let shelled = shell(&mut topo, solid, thickness, &top).unwrap();

    // Analytic reference values.
    let r_in = r - thickness;
    let h_in = h - thickness;
    let outer_corner_area = std::f64::consts::FRAC_PI_2 * r * h; // 123.70
    let inner_corner_area = std::f64::consts::FRAC_PI_2 * r_in * h_in; // 79.31
    let outer_section = w * d - (4.0 - std::f64::consts::PI) * r * r;
    let inner_section =
        (w - 2.0 * thickness) * (d - 2.0 * thickness) - (4.0 - std::f64::consts::PI) * r_in * r_in;
    let rim_area = outer_section - inner_section; // 186.95
    let floor_area = inner_section; // 1523.23
    let expected_volume = outer_section * h - inner_section * h_in; // 5753.8

    let face_ids = brepkit_topology::explorer::solid_faces(&topo, shelled).unwrap();

    let mut outer_corners = 0;
    let mut inner_corners = 0;
    let mut rim_total = 0.0;
    let mut floor_total = 0.0;
    for &fid in &face_ids {
        let f = topo.face(fid).unwrap();
        let area = crate::measure::face_area(&topo, fid, 0.01).unwrap();
        match f.surface() {
            FaceSurface::Cylinder(_) => {
                if (area - outer_corner_area).abs() < 0.1 {
                    outer_corners += 1;
                } else if (area - inner_corner_area).abs() < 0.1 {
                    inner_corners += 1;
                } else {
                    unreachable!(
                        "cylinder face {} area {area:.2} matches neither outer corner \
                             {outer_corner_area:.2} nor inner corner {inner_corner_area:.2}",
                        fid.index()
                    );
                }
            }
            FaceSurface::Plane { normal, .. } if normal.z().abs() > 0.9 => {
                let ow = topo.wire(f.outer_wire()).unwrap();
                let oe0 = ow.edges()[0];
                let z0 = topo
                    .vertex(topo.edge(oe0.edge()).unwrap().start())
                    .unwrap()
                    .point()
                    .z();
                if (z0 - h).abs() < 1e-6 {
                    rim_total += area;
                } else if (z0 - thickness).abs() < 1e-6 {
                    floor_total += area;
                }
            }
            _ => {}
        }
    }
    assert_eq!(outer_corners, 4, "4 outer corner cylinder faces");
    assert_eq!(inner_corners, 4, "4 inner corner cylinder faces");
    assert!(
        (rim_total - rim_area).abs() < 0.5,
        "rim annulus area {rim_total:.2} != expected {rim_area:.2}"
    );
    assert!(
        (floor_total - floor_area).abs() < 0.5,
        "floor area {floor_total:.2} != expected {floor_area:.2}"
    );

    let vol = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();
    assert!(
        (vol - expected_volume).abs() < 1.0,
        "shell volume {vol:.2} != expected {expected_volume:.2}"
    );

    // Watertightness: every quantized mesh edge must be shared by
    // exactly two triangles at all deflections.
    type QuantPoint = (i64, i64, i64);
    for defl in [0.01_f64, 0.1, 0.5] {
        let mesh = crate::tessellate::tessellate_solid(&topo, shelled, defl).unwrap();
        let q = |x: f64| (x * 1e4).round() as i64;
        let mut edge_counts: HashMap<(QuantPoint, QuantPoint), u32> = HashMap::new();
        for tri in mesh.indices.chunks_exact(3) {
            let keys: Vec<_> = tri
                .iter()
                .map(|&i| {
                    let p = mesh.positions[i as usize];
                    (q(p.x()), q(p.y()), q(p.z()))
                })
                .collect();
            for i in 0..3 {
                let a = keys[i];
                let b = keys[(i + 1) % 3];
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_counts.entry(key).or_insert(0) += 1;
            }
        }
        let boundary = edge_counts.values().filter(|&&c| c == 1).count();
        let nonmanifold = edge_counts.values().filter(|&&c| c > 2).count();
        assert_eq!(
            (boundary, nonmanifold),
            (0, 0),
            "mesh at deflection {defl} has {boundary} boundary and {nonmanifold} \
                 non-manifold edges"
        );
    }
}
