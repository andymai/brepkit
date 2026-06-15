use super::*;

const TOL: f64 = 1e-10;

#[test]
fn fix_xy_converges() {
    let mut sys = GcsSystem::new();
    let p = sys.add_point(PointData {
        x: 5.0,
        y: 7.0,
        fixed: false,
    });
    sys.add_constraint(Constraint::FixX(p, 2.0)).unwrap();
    sys.add_constraint(Constraint::FixY(p, 3.0)).unwrap();

    let result = sys.solve(100, TOL).unwrap();
    assert!(result.converged);
    let pt = sys.point(p).unwrap();
    assert!((pt.x - 2.0).abs() < TOL);
    assert!((pt.y - 3.0).abs() < TOL);
}

#[test]
fn distance_constraint() {
    let mut sys = GcsSystem::new();
    let p0 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let p1 = sys.add_point(PointData {
        x: 0.5,
        y: 0.0,
        fixed: false,
    });
    sys.add_constraint(Constraint::Distance(p0, p1, 3.0))
        .unwrap();

    let result = sys.solve(100, TOL).unwrap();
    assert!(result.converged, "max_r = {}", result.max_residual);
    let pt0 = sys.point(p0).unwrap();
    let pt1 = sys.point(p1).unwrap();
    let dist = ((pt1.x - pt0.x).powi(2) + (pt1.y - pt0.y).powi(2)).sqrt();
    assert!(
        (dist - 3.0).abs() < 1e-6,
        "distance should be 3.0, got {dist}"
    );
}

#[test]
fn coincident_constraint() {
    let mut sys = GcsSystem::new();
    let p0 = sys.add_point(PointData {
        x: 1.0,
        y: 2.0,
        fixed: true,
    });
    let p1 = sys.add_point(PointData {
        x: 3.0,
        y: 4.0,
        fixed: false,
    });
    sys.add_constraint(Constraint::Coincident(p0, p1)).unwrap();

    let result = sys.solve(100, TOL).unwrap();
    assert!(result.converged);
    let pt = sys.point(p1).unwrap();
    assert!((pt.x - 1.0).abs() < TOL);
    assert!((pt.y - 2.0).abs() < TOL);
}

#[test]
fn horizontal_line() {
    let mut sys = GcsSystem::new();
    let p0 = sys.add_point(PointData {
        x: 0.0,
        y: 1.0,
        fixed: true,
    });
    let p1 = sys.add_point(PointData {
        x: 5.0,
        y: 3.0,
        fixed: false,
    });
    let l = sys.add_line(p0, p1).unwrap();
    sys.add_constraint(Constraint::Horizontal(l)).unwrap();

    let result = sys.solve(100, TOL).unwrap();
    assert!(result.converged);
    assert!((sys.point(p1).unwrap().y - sys.point(p0).unwrap().y).abs() < TOL);
}

#[test]
fn vertical_line() {
    let mut sys = GcsSystem::new();
    let p0 = sys.add_point(PointData {
        x: 2.0,
        y: 0.0,
        fixed: true,
    });
    let p1 = sys.add_point(PointData {
        x: 5.0,
        y: 7.0,
        fixed: false,
    });
    let l = sys.add_line(p0, p1).unwrap();
    sys.add_constraint(Constraint::Vertical(l)).unwrap();

    let result = sys.solve(100, TOL).unwrap();
    assert!(result.converged);
    assert!((sys.point(p1).unwrap().x - sys.point(p0).unwrap().x).abs() < TOL);
}

#[test]
fn perpendicular_lines() {
    let mut sys = GcsSystem::new();
    let p0 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let p1 = sys.add_point(PointData {
        x: 1.0,
        y: 0.0,
        fixed: true,
    });
    let p2 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let p3 = sys.add_point(PointData {
        x: 0.5,
        y: 0.5,
        fixed: false,
    });
    let l1 = sys.add_line(p0, p1).unwrap();
    let l2 = sys.add_line(p2, p3).unwrap();
    sys.add_constraint(Constraint::Perpendicular(l1, l2))
        .unwrap();

    let result = sys.solve(100, TOL).unwrap();
    assert!(result.converged);
    let pt3 = sys.point(p3).unwrap();
    // Line p0-p1 is along X. Perpendicular means p3.x - p2.x = 0
    assert!(pt3.x.abs() < TOL, "p3.x = {}", pt3.x);
}

#[test]
fn parallel_lines() {
    let mut sys = GcsSystem::new();
    let p0 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let p1 = sys.add_point(PointData {
        x: 1.0,
        y: 1.0,
        fixed: true,
    });
    let p2 = sys.add_point(PointData {
        x: 2.0,
        y: 0.0,
        fixed: true,
    });
    let p3 = sys.add_point(PointData {
        x: 3.0,
        y: 0.5,
        fixed: false,
    });
    let l1 = sys.add_line(p0, p1).unwrap();
    let l2 = sys.add_line(p2, p3).unwrap();
    sys.add_constraint(Constraint::Parallel(l1, l2)).unwrap();

    let result = sys.solve(100, TOL).unwrap();
    assert!(result.converged);
    let pt3 = sys.point(p3).unwrap();
    let dy = pt3.y - 0.0; // p2.y = 0
    let dx = pt3.x - 2.0; // p2.x = 2
    // Cross with (1,1) should be 0: dy - dx = 0
    assert!((dy - dx).abs() < TOL, "not parallel: dy={dy}, dx={dx}");
}

#[test]
fn rectangle_30x20() {
    let mut sys = GcsSystem::new();

    let p0 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    let p1 = sys.add_point(PointData {
        x: 25.0,
        y: 1.0,
        fixed: false,
    });
    let p2 = sys.add_point(PointData {
        x: 26.0,
        y: 18.0,
        fixed: false,
    });
    let p3 = sys.add_point(PointData {
        x: 1.0,
        y: 17.0,
        fixed: false,
    });

    let bottom = sys.add_line(p0, p1).unwrap();
    let right = sys.add_line(p1, p2).unwrap();
    let top = sys.add_line(p2, p3).unwrap();
    let left = sys.add_line(p3, p0).unwrap();

    sys.add_constraint(Constraint::FixX(p0, 0.0)).unwrap();
    sys.add_constraint(Constraint::FixY(p0, 0.0)).unwrap();

    sys.add_constraint(Constraint::Horizontal(bottom)).unwrap();
    sys.add_constraint(Constraint::Distance(p0, p1, 30.0))
        .unwrap();

    sys.add_constraint(Constraint::Vertical(right)).unwrap();
    sys.add_constraint(Constraint::Distance(p1, p2, 20.0))
        .unwrap();

    sys.add_constraint(Constraint::Horizontal(top)).unwrap();
    sys.add_constraint(Constraint::Distance(p2, p3, 30.0))
        .unwrap();

    sys.add_constraint(Constraint::Vertical(left)).unwrap();
    sys.add_constraint(Constraint::Distance(p3, p0, 20.0))
        .unwrap();

    let result = sys.solve(200, 1e-8).unwrap();
    assert!(
        result.converged,
        "rectangle: max_r = {}",
        result.max_residual
    );

    let eps = 1e-4;
    let pt0 = sys.point(p0).unwrap();
    let pt1 = sys.point(p1).unwrap();
    let pt2 = sys.point(p2).unwrap();
    let pt3 = sys.point(p3).unwrap();

    assert!(pt0.x.abs() < eps, "p0.x = {}", pt0.x);
    assert!(pt0.y.abs() < eps, "p0.y = {}", pt0.y);
    assert!((pt1.x - 30.0).abs() < eps, "p1.x = {}", pt1.x);
    assert!(pt1.y.abs() < eps, "p1.y = {}", pt1.y);
    assert!((pt2.x - 30.0).abs() < eps, "p2.x = {}", pt2.x);
    assert!((pt2.y - 20.0).abs() < eps, "p2.y = {}", pt2.y);
    assert!(pt3.x.abs() < eps, "p3.x = {}", pt3.x);
    assert!((pt3.y - 20.0).abs() < eps, "p3.y = {}", pt3.y);
}

#[test]
fn dof_analysis() {
    let mut sys = GcsSystem::new();
    let p = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });

    // Free point: 2 DOF
    let dof = sys.dof();
    assert_eq!(dof.dof, 2);

    // Fix X: 1 DOF
    let cx = sys.add_constraint(Constraint::FixX(p, 0.0)).unwrap();
    let dof = sys.dof();
    assert_eq!(dof.dof, 1);

    // Fix Y: 0 DOF
    sys.add_constraint(Constraint::FixY(p, 0.0)).unwrap();
    let dof = sys.dof();
    assert_eq!(dof.dof, 0);

    // Remove FixX: back to 1 DOF
    sys.remove_constraint(cx).unwrap();
    let dof = sys.dof();
    assert_eq!(dof.dof, 1);
}

#[test]
fn remove_point_in_use_fails() {
    let mut sys = GcsSystem::new();
    let p0 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    let p1 = sys.add_point(PointData {
        x: 1.0,
        y: 0.0,
        fixed: false,
    });
    let _l = sys.add_line(p0, p1).unwrap();

    assert!(sys.remove_point(p0).is_err());
}

#[test]
fn remove_line_in_use_fails() {
    let mut sys = GcsSystem::new();
    let p0 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    let p1 = sys.add_point(PointData {
        x: 1.0,
        y: 0.0,
        fixed: false,
    });
    let l = sys.add_line(p0, p1).unwrap();
    sys.add_constraint(Constraint::Horizontal(l)).unwrap();

    assert!(sys.remove_line(l).is_err());
}

#[test]
fn stale_constraint_handle() {
    let mut sys = GcsSystem::new();
    let p = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    let c = sys.add_constraint(Constraint::FixX(p, 0.0)).unwrap();
    sys.remove_constraint(c).unwrap();
    assert!(sys.remove_constraint(c).is_err());
}

#[test]
fn solve_after_removal() {
    let mut sys = GcsSystem::new();
    let p = sys.add_point(PointData {
        x: 5.0,
        y: 7.0,
        fixed: false,
    });
    let _cx = sys.add_constraint(Constraint::FixX(p, 2.0)).unwrap();
    let cy = sys.add_constraint(Constraint::FixY(p, 3.0)).unwrap();

    // Solve, then remove FixY, re-solve
    let r1 = sys.solve(100, TOL).unwrap();
    assert!(r1.converged);

    sys.remove_constraint(cy).unwrap();
    let r2 = sys.solve(100, TOL).unwrap();
    assert!(r2.converged);
    // X should still be at 2.0, Y should be unchanged from last solve
    assert!((sys.point(p).unwrap().x - 2.0).abs() < TOL);
}

#[test]
fn add_constraint_with_invalid_handle_fails() {
    let mut sys = GcsSystem::new();
    let p = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    sys.remove_point(p).unwrap();
    assert!(sys.add_constraint(Constraint::FixX(p, 0.0)).is_err());
}

#[test]
fn triangle_345() {
    let mut sys = GcsSystem::new();
    let p0 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let p1 = sys.add_point(PointData {
        x: 1.0,
        y: 0.0,
        fixed: false,
    });
    let p2 = sys.add_point(PointData {
        x: 0.5,
        y: 1.0,
        fixed: false,
    });

    let bottom = sys.add_line(p0, p1).unwrap();
    sys.add_constraint(Constraint::Horizontal(bottom)).unwrap();
    sys.add_constraint(Constraint::Distance(p0, p1, 3.0))
        .unwrap();
    sys.add_constraint(Constraint::Distance(p0, p2, 4.0))
        .unwrap();
    sys.add_constraint(Constraint::Distance(p1, p2, 5.0))
        .unwrap();

    let result = sys.solve(200, 1e-8).unwrap();
    assert!(
        result.converged,
        "triangle: max_r = {}",
        result.max_residual
    );

    let pt1 = sys.point(p1).unwrap();
    let d01 = (pt1.x.powi(2) + pt1.y.powi(2)).sqrt();
    assert!((d01 - 3.0).abs() < 1e-4, "d01 = {d01}");
}

#[test]
fn fixed_points_no_solve_needed() {
    let mut sys = GcsSystem::new();
    let p0 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let p1 = sys.add_point(PointData {
        x: 1.0,
        y: 0.0,
        fixed: true,
    });
    sys.add_constraint(Constraint::Distance(p0, p1, 1.0))
        .unwrap();

    let result = sys.solve(100, TOL).unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
}

#[test]
fn add_arc_basic() {
    let mut sys = GcsSystem::new();
    let c = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let s = sys.add_point(PointData {
        x: 1.0,
        y: 0.0,
        fixed: false,
    });
    let e = sys.add_point(PointData {
        x: 0.0,
        y: 1.0,
        fixed: false,
    });
    let arc = sys.add_arc(c, s, e).unwrap();
    assert_eq!(sys.arc_count(), 1);
    let data = sys.arc(arc).unwrap();
    assert_eq!(data.center, c);
    assert_eq!(data.start, s);
    assert_eq!(data.end, e);
}

#[test]
fn remove_arc_cleans_up() {
    let mut sys = GcsSystem::new();
    let c = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let s = sys.add_point(PointData {
        x: 1.0,
        y: 0.0,
        fixed: false,
    });
    let e = sys.add_point(PointData {
        x: 0.0,
        y: 1.0,
        fixed: false,
    });
    let arc = sys.add_arc(c, s, e).unwrap();
    let count_before = sys.constraint_count();
    assert!(count_before > 0, "internal constraint should exist");
    sys.remove_arc(arc).unwrap();
    assert_eq!(sys.arc_count(), 0);
    assert!(
        sys.constraint_count() < count_before,
        "internal constraint should be removed"
    );
}

#[test]
fn point_on_circle_converges() {
    let mut sys = GcsSystem::new();
    let center = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let circ = sys.add_circle(center, 2.0).unwrap();
    let pt = sys.add_point(PointData {
        x: 3.0,
        y: 0.0,
        fixed: false,
    });
    sys.add_constraint(Constraint::PointOnCircle(pt, circ))
        .unwrap();
    let result = sys.solve(100, 1e-10).unwrap();
    assert!(result.converged);
    let p = sys.point(pt).unwrap();
    let dist = (p.x * p.x + p.y * p.y).sqrt();
    assert!(
        (dist - 2.0).abs() < 1e-6,
        "point should be on circle, dist={dist}"
    );
}

#[test]
fn point_on_arc_converges() {
    let mut sys = GcsSystem::new();
    let c = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let s = sys.add_point(PointData {
        x: 2.0,
        y: 0.0,
        fixed: true,
    });
    let e = sys.add_point(PointData {
        x: 0.0,
        y: 2.0,
        fixed: false,
    });
    let arc = sys.add_arc(c, s, e).unwrap();
    let pt = sys.add_point(PointData {
        x: 3.0,
        y: 3.0,
        fixed: false,
    });
    sys.add_constraint(Constraint::PointOnArc(pt, arc)).unwrap();
    let result = sys.solve(100, 1e-10).unwrap();
    assert!(result.converged);
    let p = sys.point(pt).unwrap();
    let dist = (p.x * p.x + p.y * p.y).sqrt();
    assert!(
        (dist - 2.0).abs() < 1e-6,
        "point should be on arc circle, dist={dist}"
    );
}

#[test]
fn tangent_line_arc_converges() {
    let mut sys = GcsSystem::new();
    let p0 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let p1 = sys.add_point(PointData {
        x: 2.0,
        y: 0.0,
        fixed: true,
    });
    let line = sys.add_line(p0, p1).unwrap();
    let c = sys.add_point(PointData {
        x: 2.0,
        y: 1.0,
        fixed: false,
    });
    let s = sys.add_point(PointData {
        x: 2.0,
        y: 0.0,
        fixed: false,
    });
    let e = sys.add_point(PointData {
        x: 3.0,
        y: 1.0,
        fixed: false,
    });
    let arc = sys.add_arc(c, s, e).unwrap();
    sys.add_constraint(Constraint::Coincident(p1, s)).unwrap();
    sys.add_constraint(Constraint::TangentLineArc(line, arc, p1))
        .unwrap();
    let result = sys.solve(100, 1e-10).unwrap();
    assert!(result.converged, "tangent line-arc should converge");
    let sp = sys.point(s).unwrap();
    let cp = sys.point(c).unwrap();
    let radius_dir = (sp.x - cp.x, sp.y - cp.y);
    let dot = 1.0 * radius_dir.0 + 0.0 * radius_dir.1;
    assert!(dot.abs() < 1e-6, "line should be tangent to arc, dot={dot}");
}

#[test]
fn equal_radius_arc_arc_converges() {
    let mut sys = GcsSystem::new();
    let c1 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let s1 = sys.add_point(PointData {
        x: 2.0,
        y: 0.0,
        fixed: true,
    });
    let e1 = sys.add_point(PointData {
        x: 0.0,
        y: 2.0,
        fixed: false,
    });
    let arc1 = sys.add_arc(c1, s1, e1).unwrap();
    let c2 = sys.add_point(PointData {
        x: 5.0,
        y: 0.0,
        fixed: true,
    });
    let s2 = sys.add_point(PointData {
        x: 8.0,
        y: 0.0,
        fixed: false,
    });
    let e2 = sys.add_point(PointData {
        x: 5.0,
        y: 3.0,
        fixed: false,
    });
    let arc2 = sys.add_arc(c2, s2, e2).unwrap();
    sys.add_constraint(Constraint::EqualRadiusArcArc(arc1, arc2))
        .unwrap();
    let result = sys.solve(100, 1e-10).unwrap();
    assert!(result.converged);
    let r1 = {
        let s = sys.point(s1).unwrap();
        (s.x * s.x + s.y * s.y).sqrt()
    };
    let r2 = {
        let cp = sys.point(c2).unwrap();
        let sp = sys.point(s2).unwrap();
        ((sp.x - cp.x).powi(2) + (sp.y - cp.y).powi(2)).sqrt()
    };
    assert!(
        (r1 - r2).abs() < 1e-6,
        "radii should be equal: r1={r1}, r2={r2}"
    );
}

#[test]
fn concentric_arc_arc_converges() {
    let mut sys = GcsSystem::new();
    let c1 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let s1 = sys.add_point(PointData {
        x: 1.0,
        y: 0.0,
        fixed: false,
    });
    let e1 = sys.add_point(PointData {
        x: 0.0,
        y: 1.0,
        fixed: false,
    });
    let arc1 = sys.add_arc(c1, s1, e1).unwrap();
    let c2 = sys.add_point(PointData {
        x: 0.5,
        y: 0.5,
        fixed: false,
    });
    let s2 = sys.add_point(PointData {
        x: 2.5,
        y: 0.5,
        fixed: false,
    });
    let e2 = sys.add_point(PointData {
        x: 0.5,
        y: 2.5,
        fixed: false,
    });
    let arc2 = sys.add_arc(c2, s2, e2).unwrap();
    sys.add_constraint(Constraint::ConcentricArcArc(arc1, arc2))
        .unwrap();
    let result = sys.solve(100, 1e-10).unwrap();
    assert!(result.converged);
    let cp1 = sys.point(c1).unwrap();
    let cp2 = sys.point(c2).unwrap();
    assert!((cp1.x - cp2.x).abs() < 1e-6 && (cp1.y - cp2.y).abs() < 1e-6);
}

#[test]
fn slot_profile_line_arc_tangent() {
    let mut sys = GcsSystem::new();
    // 4 corner points for a 4-unit-long, 2-unit-wide slot
    let p0 = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let p1 = sys.add_point(PointData {
        x: 4.0,
        y: 0.0,
        fixed: false,
    });
    let p2 = sys.add_point(PointData {
        x: 4.0,
        y: 2.0,
        fixed: false,
    });
    let p3 = sys.add_point(PointData {
        x: 0.0,
        y: 2.0,
        fixed: false,
    });
    // Two horizontal lines
    let bottom_line = sys.add_line(p0, p1).unwrap();
    let top_line = sys.add_line(p3, p2).unwrap();
    // Right semicircle: center at (4, 1), connecting p1 to p2
    let rc = sys.add_point(PointData {
        x: 4.0,
        y: 1.0,
        fixed: false,
    });
    let right_arc = sys.add_arc(rc, p1, p2).unwrap();
    // Left semicircle: center at (0, 1), connecting p3 to p0
    let lc = sys.add_point(PointData {
        x: 0.0,
        y: 1.0,
        fixed: false,
    });
    let left_arc = sys.add_arc(lc, p3, p0).unwrap();
    // Tangent constraints at all 4 junctions
    sys.add_constraint(Constraint::TangentLineArc(bottom_line, right_arc, p1))
        .unwrap();
    sys.add_constraint(Constraint::TangentLineArc(top_line, right_arc, p2))
        .unwrap();
    sys.add_constraint(Constraint::TangentLineArc(top_line, left_arc, p3))
        .unwrap();
    sys.add_constraint(Constraint::TangentLineArc(bottom_line, left_arc, p0))
        .unwrap();
    // Dimension constraints
    sys.add_constraint(Constraint::Distance(p0, p1, 4.0))
        .unwrap();
    sys.add_constraint(Constraint::Horizontal(bottom_line))
        .unwrap();
    sys.add_constraint(Constraint::Parallel(bottom_line, top_line))
        .unwrap();
    sys.add_constraint(Constraint::Distance(p0, p3, 2.0))
        .unwrap();

    let result = sys.solve(200, 1e-8).unwrap();
    assert!(result.converged, "slot profile should converge: {result:?}");
}

#[test]
fn arc_endpoints_equidistant_from_center() {
    let mut sys = GcsSystem::new();
    let c = sys.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let s = sys.add_point(PointData {
        x: 1.0,
        y: 0.0,
        fixed: true,
    });
    // End point starts off-circle — solver should move it onto the circle
    // (start is fixed, so the dynamic radius is pinned at 1.0)
    let e = sys.add_point(PointData {
        x: 0.0,
        y: 1.5,
        fixed: false,
    });
    let _arc = sys.add_arc(c, s, e).unwrap();
    let result = sys.solve(100, 1e-10).unwrap();
    assert!(result.converged);
    let ep = sys.point(e).unwrap();
    let dist = (ep.x * ep.x + ep.y * ep.y).sqrt();
    assert!(
        (dist - 1.0).abs() < 1e-6,
        "end should be on unit circle, dist={dist}"
    );

    // Also verify the dynamic behavior: when start moves, end tracks it.
    // Create a new system where start is free and moved by a FixX constraint.
    let mut sys2 = GcsSystem::new();
    let c2 = sys2.add_point(PointData {
        x: 0.0,
        y: 0.0,
        fixed: true,
    });
    let s2 = sys2.add_point(PointData {
        x: 1.0,
        y: 0.0,
        fixed: false,
    });
    let e2 = sys2.add_point(PointData {
        x: 0.0,
        y: 1.0,
        fixed: false,
    });
    let _arc2 = sys2.add_arc(c2, s2, e2).unwrap();
    // Push start out to radius 2
    sys2.add_constraint(Constraint::Distance(c2, s2, 2.0))
        .unwrap();
    let result2 = sys2.solve(100, 1e-10).unwrap();
    assert!(result2.converged, "dynamic radius test should converge");
    let sp2 = sys2.point(s2).unwrap();
    let ep2 = sys2.point(e2).unwrap();
    let r_start = (sp2.x * sp2.x + sp2.y * sp2.y).sqrt();
    let r_end = (ep2.x * ep2.x + ep2.y * ep2.y).sqrt();
    assert!(
        (r_end - r_start).abs() < 1e-6,
        "end radius ({r_end}) should track start radius ({r_start})"
    );
}
