use super::*;

/// Build a simple snapshot with two points at given positions.
fn two_point_snap(x1: f64, y1: f64, x2: f64, y2: f64) -> (PointId, PointId, EntitySnapshot) {
    use super::super::entity::GenArena;
    use super::super::entity::PointData;
    let mut arena = GenArena::new();
    let p1 = arena.insert(PointData {
        x: x1,
        y: y1,
        fixed: false,
    });
    let p2 = arena.insert(PointData {
        x: x2,
        y: y2,
        fixed: false,
    });
    let snap = EntitySnapshot {
        points: [(p1, (x1, y1)), (p2, (x2, y2))].into_iter().collect(),
        lines: HashMap::new(),
        circles: HashMap::new(),
        arcs: HashMap::new(),
    };
    (p1, p2, snap)
}

#[test]
fn coincident_at_solution() {
    let (p1, p2, snap) = two_point_snap(3.0, 4.0, 3.0, 4.0);
    let c = Constraint::Coincident(p1, p2);
    let mut r = Vec::new();
    eval_residuals(&c, &snap, &mut r);
    assert_eq!(r.len(), 2);
    assert!((r[0]).abs() < 1e-15);
    assert!((r[1]).abs() < 1e-15);
}

#[test]
fn coincident_away_from_solution() {
    let (p1, p2, snap) = two_point_snap(0.0, 0.0, 1.0, 2.0);
    let c = Constraint::Coincident(p1, p2);
    let mut r = Vec::new();
    eval_residuals(&c, &snap, &mut r);
    assert!((r[0] - (-1.0)).abs() < 1e-15);
    assert!((r[1] - (-2.0)).abs() < 1e-15);
}

#[test]
fn distance_at_solution() {
    let (p1, p2, snap) = two_point_snap(0.0, 0.0, 3.0, 4.0);
    let c = Constraint::Distance(p1, p2, 5.0);
    let mut r = Vec::new();
    eval_residuals(&c, &snap, &mut r);
    assert!(r[0].abs() < 1e-14, "residual = {}", r[0]);
}

#[test]
fn fix_x_residual() {
    let (p1, _, snap) = two_point_snap(7.0, 3.0, 0.0, 0.0);
    let c = Constraint::FixX(p1, 5.0);
    let mut r = Vec::new();
    eval_residuals(&c, &snap, &mut r);
    assert!((r[0] - 2.0).abs() < 1e-15);
}

/// Verify analytic Jacobian against finite differences for a constraint.
fn check_jacobian_fd(c: &Constraint, snap: &EntitySnapshot, params: &[ParamRef]) {
    let param_index: HashMap<ParamRef, usize> =
        params.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    let n = params.len();
    let m = residual_count(c);

    // Analytic Jacobian
    let mut jac = vec![0.0; m * n];
    let mut jw = JacobianWriter {
        data: &mut jac,
        ncols: n,
        param_index: &param_index,
    };
    eval_jacobian(c, snap, &mut jw, 0);

    // Finite-difference Jacobian
    let eps = 1e-7;
    let mut r0 = Vec::new();
    eval_residuals(c, snap, &mut r0);

    for (col, pr) in params.iter().enumerate() {
        let mut perturbed_points = snap.points.clone();
        match pr {
            ParamRef::PointX(pid) => {
                if let Some(xy) = perturbed_points.get_mut(pid) {
                    xy.0 += eps;
                }
            }
            ParamRef::PointY(pid) => {
                if let Some(xy) = perturbed_points.get_mut(pid) {
                    xy.1 += eps;
                }
            }
            ParamRef::CircleRadius(cid) => {
                // Perturb circle radius — need a mutable copy of circles
                let mut perturbed_circles = snap.circles.clone();
                if let Some(entry) = perturbed_circles.get_mut(cid) {
                    entry.1 += eps;
                }
                let perturbed_snap_circ = EntitySnapshot {
                    points: perturbed_points,
                    lines: snap.lines.clone(),
                    circles: perturbed_circles,
                    arcs: snap.arcs.clone(),
                };
                let mut r1 = Vec::new();
                eval_residuals(c, &perturbed_snap_circ, &mut r1);
                for row in 0..m {
                    let fd = (r1[row] - r0[row]) / eps;
                    let analytic = jac[row * n + col];
                    let err = (fd - analytic).abs();
                    let scale = 1.0_f64.max(analytic.abs());
                    assert!(
                        err < 1e-5 * scale + 1e-8,
                        "Jacobian mismatch at ({row},{col}): analytic={analytic}, fd={fd}, err={err}"
                    );
                }
                continue;
            }
        }
        let perturbed_snap = EntitySnapshot {
            points: perturbed_points,
            lines: snap.lines.clone(),
            circles: snap.circles.clone(),
            arcs: snap.arcs.clone(),
        };
        let mut r1 = Vec::new();
        eval_residuals(c, &perturbed_snap, &mut r1);

        for row in 0..m {
            let fd = (r1[row] - r0[row]) / eps;
            let analytic = jac[row * n + col];
            let err = (fd - analytic).abs();
            let scale = 1.0_f64.max(analytic.abs());
            assert!(
                err < 1e-5 * scale + 1e-8,
                "Jacobian mismatch at ({row},{col}): analytic={analytic}, fd={fd}, err={err}"
            );
        }
    }
}

#[test]
fn jacobian_coincident() {
    let (p1, p2, snap) = two_point_snap(1.0, 2.0, 3.0, 5.0);
    let c = Constraint::Coincident(p1, p2);
    let params = vec![
        ParamRef::PointX(p1),
        ParamRef::PointY(p1),
        ParamRef::PointX(p2),
        ParamRef::PointY(p2),
    ];
    check_jacobian_fd(&c, &snap, &params);
}

#[test]
fn jacobian_distance() {
    let (p1, p2, snap) = two_point_snap(1.0, 2.0, 4.0, 6.0);
    let c = Constraint::Distance(p1, p2, 5.0);
    let params = vec![
        ParamRef::PointX(p1),
        ParamRef::PointY(p1),
        ParamRef::PointX(p2),
        ParamRef::PointY(p2),
    ];
    check_jacobian_fd(&c, &snap, &params);
}

#[test]
fn jacobian_fix_xy() {
    let (p1, _, snap) = two_point_snap(3.0, 7.0, 0.0, 0.0);
    check_jacobian_fd(&Constraint::FixX(p1, 5.0), &snap, &[ParamRef::PointX(p1)]);
    check_jacobian_fd(&Constraint::FixY(p1, 2.0), &snap, &[ParamRef::PointY(p1)]);
}

#[test]
fn jacobian_horizontal_vertical() {
    use super::super::entity::GenArena;
    use super::super::entity::{LineData, PointData};
    let mut pts = GenArena::new();
    let p1 = pts.insert(PointData {
        x: 1.0,
        y: 3.0,
        fixed: false,
    });
    let p2 = pts.insert(PointData {
        x: 5.0,
        y: 7.0,
        fixed: false,
    });
    let mut lines = GenArena::new();
    let l = lines.insert(LineData { p1, p2 });

    let snap = EntitySnapshot {
        points: [(p1, (1.0, 3.0)), (p2, (5.0, 7.0))].into_iter().collect(),
        lines: std::iter::once((l, (p1, p2))).collect(),
        circles: HashMap::new(),
        arcs: HashMap::new(),
    };
    let params = vec![
        ParamRef::PointX(p1),
        ParamRef::PointY(p1),
        ParamRef::PointX(p2),
        ParamRef::PointY(p2),
    ];
    check_jacobian_fd(&Constraint::Horizontal(l), &snap, &params);
    check_jacobian_fd(&Constraint::Vertical(l), &snap, &params);
}

#[test]
fn jacobian_parallel_perpendicular() {
    use super::super::entity::GenArena;
    use super::super::entity::{LineData, PointData};
    let mut pts = GenArena::new();
    let p1 = pts.insert(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    let p2 = pts.insert(PointData {
        x: 3.0,
        y: 1.0,
        fixed: false,
    });
    let p3 = pts.insert(PointData {
        x: 1.0,
        y: 2.0,
        fixed: false,
    });
    let p4 = pts.insert(PointData {
        x: 4.0,
        y: 5.0,
        fixed: false,
    });
    let mut lines = GenArena::new();
    let l1 = lines.insert(LineData { p1, p2 });
    let l2 = lines.insert(LineData { p1: p3, p2: p4 });

    let snap = EntitySnapshot {
        points: [
            (p1, (0.0, 0.0)),
            (p2, (3.0, 1.0)),
            (p3, (1.0, 2.0)),
            (p4, (4.0, 5.0)),
        ]
        .into_iter()
        .collect(),
        lines: [(l1, (p1, p2)), (l2, (p3, p4))].into_iter().collect(),
        circles: HashMap::new(),
        arcs: HashMap::new(),
    };
    let params = vec![
        ParamRef::PointX(p1),
        ParamRef::PointY(p1),
        ParamRef::PointX(p2),
        ParamRef::PointY(p2),
        ParamRef::PointX(p3),
        ParamRef::PointY(p3),
        ParamRef::PointX(p4),
        ParamRef::PointY(p4),
    ];
    check_jacobian_fd(&Constraint::Parallel(l1, l2), &snap, &params);
    check_jacobian_fd(&Constraint::Perpendicular(l1, l2), &snap, &params);
}

#[test]
fn jacobian_angle() {
    use super::super::entity::GenArena;
    use super::super::entity::{LineData, PointData};
    let mut pts = GenArena::new();
    let p1 = pts.insert(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    let p2 = pts.insert(PointData {
        x: 3.0,
        y: 1.0,
        fixed: false,
    });
    let p3 = pts.insert(PointData {
        x: 1.0,
        y: 0.0,
        fixed: false,
    });
    let p4 = pts.insert(PointData {
        x: 2.0,
        y: 4.0,
        fixed: false,
    });
    let mut lines = GenArena::new();
    let l1 = lines.insert(LineData { p1, p2 });
    let l2 = lines.insert(LineData { p1: p3, p2: p4 });

    let snap = EntitySnapshot {
        points: [
            (p1, (0.0, 0.0)),
            (p2, (3.0, 1.0)),
            (p3, (1.0, 0.0)),
            (p4, (2.0, 4.0)),
        ]
        .into_iter()
        .collect(),
        lines: [(l1, (p1, p2)), (l2, (p3, p4))].into_iter().collect(),
        circles: HashMap::new(),
        arcs: HashMap::new(),
    };
    let params = vec![
        ParamRef::PointX(p1),
        ParamRef::PointY(p1),
        ParamRef::PointX(p2),
        ParamRef::PointY(p2),
        ParamRef::PointX(p3),
        ParamRef::PointY(p3),
        ParamRef::PointX(p4),
        ParamRef::PointY(p4),
    ];
    check_jacobian_fd(&Constraint::Angle(l1, l2, 0.5), &snap, &params);
}

#[test]
fn jacobian_point_on_circle() {
    use super::super::entity::GenArena;
    use super::super::entity::{CircleData, PointData};
    let mut pts = GenArena::new();
    let center = pts.insert(PointData {
        x: 1.0,
        y: 2.0,
        fixed: false,
    });
    let pt = pts.insert(PointData {
        x: 4.0,
        y: 6.0,
        fixed: false,
    });
    let mut circles = GenArena::new();
    let circ = circles.insert(CircleData {
        center,
        radius: 3.0,
    });
    let snap = EntitySnapshot {
        points: [(center, (1.0, 2.0)), (pt, (4.0, 6.0))]
            .into_iter()
            .collect(),
        lines: HashMap::new(),
        circles: [(circ, (center, 3.0))].into_iter().collect(),
        arcs: HashMap::new(),
    };
    let c = Constraint::PointOnCircle(pt, circ);
    let params = vec![
        ParamRef::PointX(pt),
        ParamRef::PointY(pt),
        ParamRef::PointX(center),
        ParamRef::PointY(center),
        ParamRef::CircleRadius(circ),
    ];
    check_jacobian_fd(&c, &snap, &params);
}

#[test]
fn jacobian_point_on_arc() {
    use super::super::entity::GenArena;
    use super::super::entity::{ArcData, PointData};
    let mut pts = GenArena::new();
    let center = pts.insert(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    let start = pts.insert(PointData {
        x: 2.0,
        y: 0.0,
        fixed: false,
    });
    let end = pts.insert(PointData {
        x: 0.0,
        y: 2.0,
        fixed: false,
    });
    let pt = pts.insert(PointData {
        x: 1.5,
        y: 1.5,
        fixed: false,
    });
    let mut arcs = GenArena::new();
    let arc = arcs.insert(ArcData { center, start, end });
    let snap = EntitySnapshot {
        points: [
            (center, (0.0, 0.0)),
            (start, (2.0, 0.0)),
            (end, (0.0, 2.0)),
            (pt, (1.5, 1.5)),
        ]
        .into_iter()
        .collect(),
        lines: HashMap::new(),
        circles: HashMap::new(),
        arcs: [(arc, (center, start, end))].into_iter().collect(),
    };
    let c = Constraint::PointOnArc(pt, arc);
    let params = vec![
        ParamRef::PointX(pt),
        ParamRef::PointY(pt),
        ParamRef::PointX(center),
        ParamRef::PointY(center),
        ParamRef::PointX(start),
        ParamRef::PointY(start),
    ];
    check_jacobian_fd(&c, &snap, &params);
}

#[test]
fn jacobian_tangent_line_arc() {
    use super::super::entity::GenArena;
    use super::super::entity::{ArcData, LineData, PointData};
    let mut pts = GenArena::new();
    let p1 = pts.insert(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    let p2 = pts.insert(PointData {
        x: 2.0,
        y: 0.0,
        fixed: false,
    });
    let center = pts.insert(PointData {
        x: 2.0,
        y: 1.0,
        fixed: false,
    });
    let start = pts.insert(PointData {
        x: 2.0,
        y: 0.0,
        fixed: false,
    });
    let end = pts.insert(PointData {
        x: 3.0,
        y: 1.0,
        fixed: false,
    });
    // shared point is p2 (same position as start)
    let mut lines = GenArena::new();
    let line = lines.insert(LineData { p1, p2 });
    let mut arcs = GenArena::new();
    let arc = arcs.insert(ArcData { center, start, end });
    let snap = EntitySnapshot {
        points: [
            (p1, (0.0, 0.0)),
            (p2, (2.0, 0.0)),
            (center, (2.0, 1.0)),
            (start, (2.0, 0.0)),
            (end, (3.0, 1.0)),
        ]
        .into_iter()
        .collect(),
        lines: [(line, (p1, p2))].into_iter().collect(),
        circles: HashMap::new(),
        arcs: [(arc, (center, start, end))].into_iter().collect(),
    };
    let c = Constraint::TangentLineArc(line, arc, p2);
    let params = vec![
        ParamRef::PointX(p1),
        ParamRef::PointY(p1),
        ParamRef::PointX(p2),
        ParamRef::PointY(p2),
        ParamRef::PointX(center),
        ParamRef::PointY(center),
    ];
    check_jacobian_fd(&c, &snap, &params);
}

#[test]
fn jacobian_tangent_arc_arc() {
    use super::super::entity::GenArena;
    use super::super::entity::{ArcData, PointData};
    let mut pts = GenArena::new();
    let c1 = pts.insert(PointData {
        x: 0.0,
        y: 1.0,
        fixed: false,
    });
    let c2 = pts.insert(PointData {
        x: 2.0,
        y: 1.0,
        fixed: false,
    });
    let shared = pts.insert(PointData {
        x: 1.0,
        y: 0.0,
        fixed: false,
    });
    let s1 = pts.insert(PointData {
        x: 1.0,
        y: 0.0,
        fixed: false,
    });
    let e1 = pts.insert(PointData {
        x: -1.0,
        y: 1.0,
        fixed: false,
    });
    let s2 = pts.insert(PointData {
        x: 1.0,
        y: 0.0,
        fixed: false,
    });
    let e2 = pts.insert(PointData {
        x: 3.0,
        y: 1.0,
        fixed: false,
    });
    let mut arcs = GenArena::new();
    let arc1 = arcs.insert(ArcData {
        center: c1,
        start: s1,
        end: e1,
    });
    let arc2 = arcs.insert(ArcData {
        center: c2,
        start: s2,
        end: e2,
    });
    let snap = EntitySnapshot {
        points: [
            (c1, (0.0, 1.0)),
            (c2, (2.0, 1.0)),
            (shared, (1.0, 0.0)),
            (s1, (1.0, 0.0)),
            (e1, (-1.0, 1.0)),
            (s2, (1.0, 0.0)),
            (e2, (3.0, 1.0)),
        ]
        .into_iter()
        .collect(),
        lines: HashMap::new(),
        circles: HashMap::new(),
        arcs: [(arc1, (c1, s1, e1)), (arc2, (c2, s2, e2))]
            .into_iter()
            .collect(),
    };
    let c = Constraint::TangentArcArc(arc1, arc2, shared);
    let params = vec![
        ParamRef::PointX(shared),
        ParamRef::PointY(shared),
        ParamRef::PointX(c1),
        ParamRef::PointY(c1),
        ParamRef::PointX(c2),
        ParamRef::PointY(c2),
    ];
    check_jacobian_fd(&c, &snap, &params);
}

#[test]
fn jacobian_equal_radius_arc_arc() {
    use super::super::entity::GenArena;
    use super::super::entity::{ArcData, PointData};
    let mut pts = GenArena::new();
    let c1 = pts.insert(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    let s1 = pts.insert(PointData {
        x: 2.0,
        y: 0.0,
        fixed: false,
    });
    let e1 = pts.insert(PointData {
        x: 0.0,
        y: 2.0,
        fixed: false,
    });
    let c2 = pts.insert(PointData {
        x: 5.0,
        y: 0.0,
        fixed: false,
    });
    let s2 = pts.insert(PointData {
        x: 8.0,
        y: 0.0,
        fixed: false,
    });
    let e2 = pts.insert(PointData {
        x: 5.0,
        y: 3.0,
        fixed: false,
    });
    let mut arcs = GenArena::new();
    let arc1 = arcs.insert(ArcData {
        center: c1,
        start: s1,
        end: e1,
    });
    let arc2 = arcs.insert(ArcData {
        center: c2,
        start: s2,
        end: e2,
    });
    let snap = EntitySnapshot {
        points: [
            (c1, (0.0, 0.0)),
            (s1, (2.0, 0.0)),
            (e1, (0.0, 2.0)),
            (c2, (5.0, 0.0)),
            (s2, (8.0, 0.0)),
            (e2, (5.0, 3.0)),
        ]
        .into_iter()
        .collect(),
        lines: HashMap::new(),
        circles: HashMap::new(),
        arcs: [(arc1, (c1, s1, e1)), (arc2, (c2, s2, e2))]
            .into_iter()
            .collect(),
    };
    let c = Constraint::EqualRadiusArcArc(arc1, arc2);
    let params = vec![
        ParamRef::PointX(c1),
        ParamRef::PointY(c1),
        ParamRef::PointX(s1),
        ParamRef::PointY(s1),
        ParamRef::PointX(c2),
        ParamRef::PointY(c2),
        ParamRef::PointX(s2),
        ParamRef::PointY(s2),
    ];
    check_jacobian_fd(&c, &snap, &params);
}

#[test]
fn jacobian_equal_radius_arc_circle() {
    use super::super::entity::GenArena;
    use super::super::entity::{ArcData, CircleData, PointData};
    let mut pts = GenArena::new();
    let ac = pts.insert(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    let as_ = pts.insert(PointData {
        x: 2.0,
        y: 0.0,
        fixed: false,
    });
    let ae = pts.insert(PointData {
        x: 0.0,
        y: 2.0,
        fixed: false,
    });
    let cc = pts.insert(PointData {
        x: 5.0,
        y: 5.0,
        fixed: false,
    });
    let mut arcs = GenArena::new();
    let arc = arcs.insert(ArcData {
        center: ac,
        start: as_,
        end: ae,
    });
    let mut circles = GenArena::new();
    let circ = circles.insert(CircleData {
        center: cc,
        radius: 3.0,
    });
    let snap = EntitySnapshot {
        points: [
            (ac, (0.0, 0.0)),
            (as_, (2.0, 0.0)),
            (ae, (0.0, 2.0)),
            (cc, (5.0, 5.0)),
        ]
        .into_iter()
        .collect(),
        lines: HashMap::new(),
        circles: [(circ, (cc, 3.0))].into_iter().collect(),
        arcs: [(arc, (ac, as_, ae))].into_iter().collect(),
    };
    let c = Constraint::EqualRadiusArcCircle(arc, circ);
    let params = vec![
        ParamRef::PointX(ac),
        ParamRef::PointY(ac),
        ParamRef::PointX(as_),
        ParamRef::PointY(as_),
        ParamRef::CircleRadius(circ),
    ];
    check_jacobian_fd(&c, &snap, &params);
}

#[test]
fn jacobian_arc_length() {
    use super::super::entity::GenArena;
    use super::super::entity::{ArcData, PointData};
    let mut pts = GenArena::new();
    let center = pts.insert(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    let start = pts.insert(PointData {
        x: 2.0,
        y: 0.0,
        fixed: false,
    });
    let end = pts.insert(PointData {
        x: 0.0,
        y: 2.0,
        fixed: false,
    });
    let mut arcs = GenArena::new();
    let arc = arcs.insert(ArcData { center, start, end });
    let snap = EntitySnapshot {
        points: [(center, (0.0, 0.0)), (start, (2.0, 0.0)), (end, (0.0, 2.0))]
            .into_iter()
            .collect(),
        lines: HashMap::new(),
        circles: HashMap::new(),
        arcs: [(arc, (center, start, end))].into_iter().collect(),
    };
    let target = std::f64::consts::PI; // 90 degrees * r=2
    let c = Constraint::ArcLength(arc, target);
    let params = vec![
        ParamRef::PointX(center),
        ParamRef::PointY(center),
        ParamRef::PointX(start),
        ParamRef::PointY(start),
        ParamRef::PointX(end),
        ParamRef::PointY(end),
    ];
    check_jacobian_fd(&c, &snap, &params);
}

#[test]
fn jacobian_concentric_arc_arc() {
    use super::super::entity::GenArena;
    use super::super::entity::{ArcData, PointData};
    let mut pts = GenArena::new();
    let c1 = pts.insert(PointData {
        x: 1.0,
        y: 2.0,
        fixed: false,
    });
    let s1 = pts.insert(PointData {
        x: 3.0,
        y: 2.0,
        fixed: false,
    });
    let e1 = pts.insert(PointData {
        x: 1.0,
        y: 4.0,
        fixed: false,
    });
    let c2 = pts.insert(PointData {
        x: 3.0,
        y: 4.0,
        fixed: false,
    });
    let s2 = pts.insert(PointData {
        x: 4.0,
        y: 4.0,
        fixed: false,
    });
    let e2 = pts.insert(PointData {
        x: 3.0,
        y: 5.0,
        fixed: false,
    });
    let mut arcs = GenArena::new();
    let arc1 = arcs.insert(ArcData {
        center: c1,
        start: s1,
        end: e1,
    });
    let arc2 = arcs.insert(ArcData {
        center: c2,
        start: s2,
        end: e2,
    });
    let snap = EntitySnapshot {
        points: [
            (c1, (1.0, 2.0)),
            (s1, (3.0, 2.0)),
            (e1, (1.0, 4.0)),
            (c2, (3.0, 4.0)),
            (s2, (4.0, 4.0)),
            (e2, (3.0, 5.0)),
        ]
        .into_iter()
        .collect(),
        lines: HashMap::new(),
        circles: HashMap::new(),
        arcs: [(arc1, (c1, s1, e1)), (arc2, (c2, s2, e2))]
            .into_iter()
            .collect(),
    };
    let c = Constraint::ConcentricArcArc(arc1, arc2);
    let params = vec![
        ParamRef::PointX(c1),
        ParamRef::PointY(c1),
        ParamRef::PointX(c2),
        ParamRef::PointY(c2),
    ];
    check_jacobian_fd(&c, &snap, &params);
}

#[test]
fn jacobian_concentric_arc_circle() {
    use super::super::entity::GenArena;
    use super::super::entity::{ArcData, CircleData, PointData};
    let mut pts = GenArena::new();
    let ac = pts.insert(PointData {
        x: 1.0,
        y: 2.0,
        fixed: false,
    });
    let as_ = pts.insert(PointData {
        x: 3.0,
        y: 2.0,
        fixed: false,
    });
    let ae = pts.insert(PointData {
        x: 1.0,
        y: 4.0,
        fixed: false,
    });
    let cc = pts.insert(PointData {
        x: 3.0,
        y: 4.0,
        fixed: false,
    });
    let mut arcs = GenArena::new();
    let arc = arcs.insert(ArcData {
        center: ac,
        start: as_,
        end: ae,
    });
    let mut circles = GenArena::new();
    let circ = circles.insert(CircleData {
        center: cc,
        radius: 2.0,
    });
    let snap = EntitySnapshot {
        points: [
            (ac, (1.0, 2.0)),
            (as_, (3.0, 2.0)),
            (ae, (1.0, 4.0)),
            (cc, (3.0, 4.0)),
        ]
        .into_iter()
        .collect(),
        lines: HashMap::new(),
        circles: [(circ, (cc, 2.0))].into_iter().collect(),
        arcs: [(arc, (ac, as_, ae))].into_iter().collect(),
    };
    let c = Constraint::ConcentricArcCircle(arc, circ);
    let params = vec![
        ParamRef::PointX(ac),
        ParamRef::PointY(ac),
        ParamRef::PointX(cc),
        ParamRef::PointY(cc),
    ];
    check_jacobian_fd(&c, &snap, &params);
}

#[test]
fn jacobian_point_line_distance() {
    use super::super::entity::GenArena;
    use super::super::entity::{LineData, PointData};
    let mut pts = GenArena::new();
    let pt = pts.insert(PointData {
        x: 2.0,
        y: 3.0,
        fixed: false,
    });
    let lp1 = pts.insert(PointData {
        x: 0.0,
        y: 0.0,
        fixed: false,
    });
    let lp2 = pts.insert(PointData {
        x: 4.0,
        y: 1.0,
        fixed: false,
    });
    let mut lines = GenArena::new();
    let l = lines.insert(LineData { p1: lp1, p2: lp2 });

    let snap = EntitySnapshot {
        points: [(pt, (2.0, 3.0)), (lp1, (0.0, 0.0)), (lp2, (4.0, 1.0))]
            .into_iter()
            .collect(),
        lines: std::iter::once((l, (lp1, lp2))).collect(),
        circles: HashMap::new(),
        arcs: HashMap::new(),
    };
    let params = vec![
        ParamRef::PointX(pt),
        ParamRef::PointY(pt),
        ParamRef::PointX(lp1),
        ParamRef::PointY(lp1),
        ParamRef::PointX(lp2),
        ParamRef::PointY(lp2),
    ];
    check_jacobian_fd(&Constraint::PointLineDistance(pt, l, 1.5), &snap, &params);
}
