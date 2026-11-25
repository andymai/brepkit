#![allow(clippy::unwrap_used)]

use crate::madd::Madd;
use std::f64::consts::{FRAC_PI_2, PI};

use crate::tolerance::Tolerance;
use crate::vec::{Point3, Vec3};

use super::*;

// ── Line tests ─────────────────────────────────────────────────

#[test]
fn line_evaluate() {
    let line = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
    let tol = Tolerance::new();

    let p = line.evaluate(3.0);
    assert!(tol.approx_eq(p.x(), 3.0));
    assert!(tol.approx_eq(p.y(), 0.0));
}

#[test]
fn line_project() {
    let line = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
    let t = line.project(Point3::new(5.0, 3.0, 0.0));
    let tol = Tolerance::new();
    assert!(tol.approx_eq(t, 5.0));
}

#[test]
fn line_distance() {
    let line = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)).unwrap();
    let d = line.distance_to_point(Point3::new(5.0, 3.0, 4.0));
    let tol = Tolerance::new();
    assert!(tol.approx_eq(d, 5.0)); // 3-4-5 triangle
}

#[test]
fn line_zero_direction_error() {
    assert!(Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)).is_err());
}

// ── Circle tests ───────────────────────────────────────────────

#[test]
fn circle_evaluate_at_zero() {
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let tol = Tolerance::new();
    let p = circle.evaluate(0.0);
    // Should be at radius distance from center in the plane.
    let dist = (p - circle.center()).length();
    assert!(
        tol.approx_eq(dist, 1.0),
        "point should be on circle, dist={dist}"
    );
}

#[test]
fn circle_evaluate_quarter() {
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

    let tol = Tolerance::new();
    let p = circle.evaluate(FRAC_PI_2);
    let dist = (p - circle.center()).length();
    assert!(tol.approx_eq(dist, 2.0));
}

#[test]
fn circle_circumference() {
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();

    let tol = Tolerance::new();
    assert!(tol.approx_eq(circle.circumference(), 6.0 * PI));
}

#[test]
fn circle_project_roundtrip() {
    let circle = Circle3D::new(Point3::new(1.0, 2.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();

    let tol = Tolerance::new();
    let t_orig = 1.23;
    let p = circle.evaluate(t_orig);
    let t_proj = circle.project(p);
    assert!(
        tol.approx_eq(t_orig, t_proj),
        "project should roundtrip: {t_orig} vs {t_proj}"
    );
}

#[test]
fn circle_zero_radius_error() {
    assert!(Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.0).is_err());
}

#[test]
fn circle_intersect_segment_in_plane_two_points() {
    // Unit circle in xy plane, segment from (-2,0,0) to (2,0,0) — crosses at ±1 along x.
    let c = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let hits = c.intersect_segment(
        Point3::new(-2.0, 0.0, 0.0),
        Point3::new(2.0, 0.0, 0.0),
        1e-9,
    );
    assert_eq!(hits.len(), 2);
    // u_axis defaults to +x for a +z normal; so points are at t = 0 and π.
    let xs: Vec<f64> = hits.iter().map(|(p, _)| p.x()).collect();
    assert!(xs.iter().any(|x| (x - 1.0).abs() < 1e-9));
    assert!(xs.iter().any(|x| (x + 1.0).abs() < 1e-9));
}

#[test]
fn circle_intersect_segment_crosses_plane() {
    // Great circle in x=0 plane (sphere-equator type): radius 8, normal +x.
    // Segment from (1, 8, 0) to (-1, 8, 0): crosses x=0 at (0, 8, 0), which is on the circle.
    let c = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 8.0).unwrap();
    let hits = c.intersect_segment(
        Point3::new(1.0, 8.0, 0.0),
        Point3::new(-1.0, 8.0, 0.0),
        1e-7,
    );
    assert_eq!(hits.len(), 1);
    let (p, _t) = hits[0];
    assert!((p.x()).abs() < 1e-7);
    assert!((p.y() - 8.0).abs() < 1e-7);
    assert!((p.z()).abs() < 1e-7);
}

#[test]
fn circle_intersect_segment_polygon_vertex_on_circle() {
    // The box-sphere case: equator polygon edge from a vertex AT (0,8,0)
    // (which lies on a great circle from the x=0 plane) outward.
    let c = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 8.0).unwrap();
    // Polygon vertex 4 at (0, 8, 0), polygon vertex 5 at (-3.06..., 7.39..., 0).
    let v4 = Point3::new(0.0, 8.0, 0.0);
    let v5 = Point3::new(-8.0 * (5.0_f64).cos(), 8.0 * (5.0_f64).sin(), 0.0);
    let _ = v5; // silence unused warn — this test only needs the vertex case.
    let v3 = Point3::new(3.061_467_458_920_71_f64, 7.391_036_260_090_294_f64, 0.0);
    // Edge from v3 (x>0) to v4 (x=0) — the vertex is on the great circle's plane and circle.
    let hits = c.intersect_segment(v3, v4, 1e-7);
    assert_eq!(hits.len(), 1, "expected one hit at the polygon vertex");
    let (p, _) = hits[0];
    assert!((p - v4).length() < 1e-6, "hit should be at v4");
}

#[test]
fn circle_intersect_segment_long_ruling_ending_on_the_circle() {
    // A cylinder's ruling ending on its rim circle, the end rounded 2e-15 to
    // the far end's side of the circle's plane: the product of the two ends'
    // plane distances (about 2e-14) exceeds tol^2 though the end is on it.
    let c = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.925).unwrap();
    let end = Point3::new(0.925, 2e-15, 0.0);
    let hits = c.intersect_segment(Point3::new(0.925, 10.74, 0.0), end, 1e-7);
    assert_eq!(hits.len(), 1, "the ruling meets the circle at its end");
    assert!((hits[0].0 - end).length() < 1e-9);
}

#[test]
fn circle_intersect_segment_no_crossing() {
    // Segment fully on one side of circle's plane, no crossing.
    let c = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let hits = c.intersect_segment(Point3::new(0.0, 0.0, 1.0), Point3::new(0.0, 0.0, 2.0), 1e-9);
    assert!(hits.is_empty());
}

#[test]
fn circle_intersect_segment_near_tangent_collapses_to_foot() {
    // The gridfinity wall-tangency case: an r=4 socket-outline corner circle
    // tangent to a wall line, penetrated by a sub-tolerance residual. The
    // separate quadratic roots straddle the tangency by sqrt(2r·δ) — a 1e-13
    // residual already shifts each root a full micron — so the near-tangent
    // pair must collapse to the well-conditioned double root (the foot of
    // the center on the line).
    let tol = 1e-7;
    let c = Circle3D::new(
        Point3::new(-16.75, 37.75, 5.0),
        Vec3::new(0.0, 0.0, 1.0),
        4.0,
    )
    .unwrap();
    // Line at x = -20.75 + 1.25e-13 (penetrates the circle by ~1.25e-13,
    // giving a ±1e-6 root pair without the collapse).
    let x = -20.75 + 1.25e-13;
    let hits = c.intersect_segment(Point3::new(x, 20.0, 5.0), Point3::new(x, 50.0, 5.0), tol);
    assert_eq!(hits.len(), 1, "near-tangent pair must collapse to one hit");
    let (p, _) = hits[0];
    assert!(
        (p - Point3::new(x, 37.75, 5.0)).length() < tol,
        "hit must be the exact foot of the tangency, got ({}, {}, {})",
        p.x(),
        p.y(),
        p.z()
    );
}

#[test]
fn circle_intersect_segment_genuine_secant_keeps_two_hits() {
    // A real secant (penetration far above tolerance) must still yield two
    // distinct crossings — the tangent collapse only fires when the chord
    // implies sub-tolerance penetration.
    let c = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 4.0).unwrap();
    // Line at x = 3.9: penetration δ = 0.1, chord = 2·sqrt(16 − 15.21) ≈ 1.78.
    let hits = c.intersect_segment(
        Point3::new(3.9, -5.0, 0.0),
        Point3::new(3.9, 5.0, 0.0),
        1e-7,
    );
    assert_eq!(hits.len(), 2, "genuine secant must keep both crossings");
}

// ── Ellipse tests ──────────────────────────────────────────────

#[test]
fn ellipse_evaluate() {
    let ellipse = Ellipse3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        3.0,
        2.0,
    )
    .unwrap();

    let tol = Tolerance::new();
    // At t=0, should be at (3, 0, 0) direction.
    let p0 = ellipse.evaluate(0.0);
    let dist0 = (p0 - ellipse.center()).length();
    assert!(tol.approx_eq(dist0, 3.0), "at t=0 should be at semi_major");

    // At t=π/2, should be at semi_minor distance.
    let p1 = ellipse.evaluate(FRAC_PI_2);
    let dist1 = (p1 - ellipse.center()).length();
    assert!(
        tol.approx_eq(dist1, 2.0),
        "at t=π/2 should be at semi_minor"
    );
}

#[test]
fn ellipse_circumference_circle() {
    // When semi_major == semi_minor, it's a circle.
    let ellipse = Ellipse3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        1.0,
        1.0,
    )
    .unwrap();

    let tol = Tolerance::loose();
    let circ = ellipse.approximate_circumference();
    assert!(
        tol.approx_eq(circ, 2.0 * PI),
        "circle circumference should be 2π, got {circ}"
    );
}

#[test]
fn ellipse_zero_axis_error() {
    assert!(
        Ellipse3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
            1.0
        )
        .is_err()
    );
}

/// The distance between `p` and `q`, by hypot, so it neither overflows nor
/// underflows where the coordinates' squares would.
fn apart(p: Point3, q: Point3) -> f64 {
    let d = p - q;
    d.x().hypot(d.y()).hypot(d.z())
}

/// The distance from `p` to the nearest of a dense scan of `ellipse`, each
/// best sample refined by golden section between its neighbours.
fn nearest_by_scan(ellipse: &Ellipse3D, p: Point3) -> f64 {
    const N: u32 = 4_000;
    let off = |t: f64| apart(ellipse.evaluate(t), p);
    let step = 2.0 * PI / f64::from(N);
    let best = (0..N)
        .map(|k| step.madd(f64::from(k), -PI))
        .min_by(|&a, &b| off(a).total_cmp(&off(b)))
        .unwrap();
    let (mut lo, mut hi) = (best - step, best + step);
    let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
    for _ in 0..200 {
        let (a, b) = ((hi - lo).madd(-golden, hi), (hi - lo).madd(golden, lo));
        if off(a) < off(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    off(best).min(off(0.5 * (lo + hi)))
}

/// A point off the ellipse along its normal projects to the point it was
/// moved from, which the angle of the point drawn onto a circle is not;
/// and every point, inside the ellipse, on either axis, at its centre or
/// far off, projects to a point no farther than the nearest of a dense
/// scan.
#[test]
fn ellipse_project_finds_the_nearest_point() {
    let ellipse = Ellipse3D::new(
        Point3::new(1.0, -2.0, 0.5),
        Vec3::new(0.0, 0.0, 1.0),
        3.0,
        2.0,
    )
    .unwrap();
    for k in -7..8 {
        let t = 0.4 * f64::from(k);
        let along = ellipse.tangent(t);
        let out = along.cross(ellipse.normal()).normalize().unwrap();
        for off in [0.0, 0.3, -0.3] {
            let p = ellipse.evaluate(t) + out * off;
            let back = ellipse.project(p);
            let gap = (back - t + PI).rem_euclid(2.0 * PI) - PI;
            assert!(gap.abs() < 1e-12, "t {t} off {off}: {back}");
        }
    }
    let normal = Vec3::new(0.3, -0.2, 1.0).normalize().unwrap();
    let centre = Point3::new(0.5, 1.0, -2.0);
    let frame = crate::frame::Frame3::from_normal(centre, normal).unwrap();
    let ellipses = [
        Ellipse3D::new(centre, normal, 3.0, 2.0).unwrap(),
        Ellipse3D::new(centre, normal, 5.0, 1.0).unwrap(),
        Ellipse3D::new(centre, normal, 2.0, 2.0).unwrap(),
        // The longer semi-axis along `v`.
        Ellipse3D::with_axes(centre, frame.z, 1.0, 3.0, frame.x, frame.y).unwrap(),
    ];
    let across = [-1e6, -2.9, -1.2, -0.7, -0.3, 0.0, 0.3, 0.7, 1.2, 2.9, 1e3];
    for ellipse in &ellipses {
        let (a, b) = (ellipse.semi_major(), ellipse.semi_minor());
        let (u, v) = (ellipse.u_axis(), ellipse.v_axis());
        for x in across {
            for y in across {
                let p = centre + u * (a * x) + v * (b * y);
                let t = ellipse.project(p);
                assert!(t > -PI && t <= PI, "({x}, {y}) on {a} x {b}: {t}");
                let reached = apart(ellipse.evaluate(t), p);
                let nearest = nearest_by_scan(ellipse, p);
                assert!(reached.is_finite() && nearest.is_finite());
                let scale = apart(p, centre).max(a.max(b));
                assert!(
                    reached <= nearest + 1e-12 * scale,
                    "({x}, {y}) on {a} x {b}: {reached}, the scan finds {nearest}"
                );
            }
        }
    }
    // A point near the centre on the longer axis has two nearest points,
    // off the axis, where the curve's own angle is no nearest point.
    let wide = &ellipses[0];
    let p = centre + wide.u_axis() * 0.5;
    let t = wide.project(p);
    assert!((t - 0.3_f64.acos()).abs() < 1e-12, "{t}");
    assert!(((wide.evaluate(t) - p).length_squared() - 3.8).abs() < 1e-12);
    // A point so far off that the squares of its coordinates overflow: its
    // nearest point is where the normal runs along `(1, 1)`, `tan t = b / a`.
    let far = centre + wide.u_axis() * 1e200 + wide.v_axis() * 1e200;
    let t = wide.project(far);
    assert!((t - 2.0_f64.atan2(3.0)).abs() < 1e-12, "{t}");
    // At the centre of a circle every point is nearest: `0`. At the centre
    // of a longer ellipse, the end of its shorter semi-axis.
    assert!(ellipses[2].project(centre).abs() < 1e-15);
    assert!((ellipses[0].project(centre) - FRAC_PI_2).abs() < 1e-15);
    assert!((ellipses[3].project(centre)).abs() < 1e-15);
}

/// The same ellipse and point, scaled from `1e-200` to `1e200`, where the
/// squares of the semi-axes leave the range of a double, read the same angle
/// to a few units in the last place, and points `1e-300` and `1e300`
/// semi-axes from the centre read a number at every scale. Ellipses whose
/// shorter semi-axis is below the rounding of the longer, down to the least
/// double, read a number and a nearest point at every scale too.
#[test]
fn ellipse_project_reads_alike_at_every_scale() {
    let normal = Vec3::new(0.3, -0.2, 1.0).normalize().unwrap();
    let at_scale = |k: f64| {
        Ellipse3D::new(Point3::new(0.5 * k, k, -2.0 * k), normal, 3.0 * k, 2.0 * k).unwrap()
    };
    let points = [
        (1.7, 0.4),
        (-0.3, 0.2),
        (0.0, 0.5),
        (2.5, 0.0),
        (0.5, 0.0),
        (-4.0, -3.0),
        (1e3, -7e2),
        (1e-3, 2e-3),
    ];
    for (x, y) in points {
        let read = |k: f64| {
            let ellipse = at_scale(k);
            let p = ellipse.center() + ellipse.u_axis() * (x * k) + ellipse.v_axis() * (y * k);
            ellipse.project(p)
        };
        // On the longer axis within the evolute two points are nearest, and
        // the rounding of the scaled point picks the side.
        let side = |t: f64| if y == 0.0 { t.abs() } else { t };
        let one = side(read(1.0));
        for k in [1e-200, 1e-150, 1e150, 1e200] {
            let t = side(read(k));
            assert!(
                (t - one).abs() <= 8.0 * f64::EPSILON * one.abs().max(1.0),
                "({x}, {y}) at {k}: {t}, at 1: {one}"
            );
        }
    }
    // Far off and near the centre, the centre at the origin so the point's
    // own digits survive the subtraction.
    for k in [1e-200, 1e-150, 1.0, 1e150, 1e200] {
        let ellipse = Ellipse3D::new(Point3::new(0.0, 0.0, 0.0), normal, 3.0 * k, 2.0 * k).unwrap();
        let (u, v) = (ellipse.u_axis(), ellipse.v_axis());
        for (reach, expected) in [
            (1e300, 2.0_f64.atan2(3.0)),
            (1e150, 2.0_f64.atan2(3.0)),
            (1e-150, FRAC_PI_2),
            (1e-300, FRAC_PI_2),
        ] {
            let scale = reach * k;
            if !scale.is_normal() {
                continue;
            }
            let p = Point3::new(0.0, 0.0, 0.0) + u * scale + v * scale;
            let t = ellipse.project(p);
            assert!(t.is_finite(), "{reach} at {k}: {t}");
            assert!(
                (t - expected).abs() <= 8.0 * f64::EPSILON,
                "{reach} at {k}: {t}, expected {expected}"
            );
        }
    }
    let origin = Point3::new(0.0, 0.0, 0.0);
    let frame = crate::frame::Frame3::from_normal(origin, normal).unwrap();
    let thin = [
        (1e150_f64, 1e-150_f64),
        (1.0, 1e-300),
        (1e300, 1.0),
        (1.0, 5e-324),
        (1e-150, 1e-300),
        (1e-150, 1e150),
        (5e-324, 1.0),
    ];
    for (a, b) in thin {
        for k in [1e-100_f64, 1.0, 1e100] {
            let (ak, bk) = (a * k, b * k);
            if !(ak.is_finite() && bk.is_finite() && ak > 0.0 && bk > 0.0) {
                continue;
            }
            let ellipse = Ellipse3D::with_axes(origin, frame.z, ak, bk, frame.x, frame.y).unwrap();
            let long = ak.max(bk);
            for (x, y) in [
                (0.5, 0.0),
                (0.5, 0.5),
                (0.5, -0.5),
                (0.0, 0.0),
                (0.0, 0.7),
                (2.0, 0.0),
                (-0.7, -3.0),
                (-1.0, 0.0),
            ] {
                let (along, across) = if ak >= bk { (x, y) } else { (y, x) };
                let p = origin + frame.x * (along * long) + frame.y * (across * long);
                let t = ellipse.project(p);
                assert!(
                    t.is_finite() && t > -PI && t <= PI,
                    "({x}, {y}) on {ak} x {bk}: {t}"
                );
                let reached = apart(ellipse.evaluate(t), p);
                let nearest = nearest_by_scan(&ellipse, p);
                assert!(
                    reached.is_finite() && nearest.is_finite(),
                    "({x}, {y}) on {ak} x {bk}: {reached}, {nearest}"
                );
                assert!(
                    reached <= nearest + 1e-12 * long,
                    "({x}, {y}) on {ak} x {bk}: {reached}, the scan finds {nearest}"
                );
            }
        }
    }
    // A thin ellipse and a point far enough off across it that the short
    // axis turns the nearest point round toward the end, the longer axis
    // along either frame axis: the angle is the root of the stationary
    // condition, bisected. The frame is square to the last digit, which a
    // point `1e20` off needs.
    let stationary = |(a, b): (f64, f64), (x, y): (f64, f64), (lo, hi): (f64, f64)| {
        let slope = |t: f64| {
            let (s, c) = t.sin_cos();
            (b * c).madd(b.madd(s, -y), -a * s * a.madd(c, -x))
        };
        let (mut lo, mut hi) = (lo, hi);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if (slope(mid) > 0.0) == (slope(lo) > 0.0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    for ((a, b), (x, y), bracket) in [
        ((1.0, 1e-17), (0.5, 1e20), (1.0, FRAC_PI_2)),
        ((1e-17, 1.0), (1e20, 0.5), (0.0, 0.5)),
    ] {
        let (u, v) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let ellipse = Ellipse3D::with_axes(origin, u.cross(v), a, b, u, v).unwrap();
        let t = ellipse.project(origin + u * x + v * y);
        let root = stationary((a, b), (x, y), bracket);
        assert!(
            (t - root).abs() < 1e-12,
            "({x}, {y}) on {a} x {b}: {t}, the root {root}"
        );
    }
}

/// A long arc of an ellipse 2e8 across and 10 wide, centred 1e8 away, as a
/// plane leaning 5e-8 off a wall's axis cuts the wall, turned: its centre
/// rounds by 1e-8 in the turn, and each point of the arc still projects
/// onto itself, not 0.1 along the curve.
#[test]
fn ellipse_project_holds_a_long_thin_arc_turned() {
    let major = Vec3::new(0.0, -5e-8, 1.0).normalize().unwrap();
    let minor = Vec3::new(1.0, 0.0, 0.0);
    let upright = Ellipse3D::with_axes(
        Point3::new(0.0, 0.0, 1e8),
        major.cross(minor),
        2e8,
        10.0,
        major,
        minor,
    )
    .unwrap();
    let turn = crate::mat::Mat4::rotation_x(0.4)
        * crate::mat::Mat4::rotation_y(1.1)
        * crate::mat::Mat4::rotation_z(0.3);
    let o = Point3::new(0.0, 0.0, 0.0);
    let dir = |v: Vec3| turn.mul_point(o + v) - o;
    let turned = Ellipse3D::with_axes(
        turn.mul_point(upright.center()),
        dir(upright.normal()),
        2e8,
        10.0,
        dir(major),
        dir(minor),
    )
    .unwrap();
    let end = 2.0 * PI / 3.0;
    for k in 0..=10 {
        let t = (-5.8e-6_f64).madd(f64::from(k) / 10.0, end);
        let p = turn.mul_point(upright.evaluate(t));
        let back = turned.evaluate(turned.project(p));
        assert!((back - p).length() < 1e-6, "{k}: {:e}", (back - p).length());
    }
}

// ── Parabola tests ──────────────────────────────────────────

#[test]
fn parabola_vertex() {
    let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let v = p.evaluate(0.0);
    assert!(Tolerance::default().approx_eq(v.x(), 0.0));
    assert!(Tolerance::default().approx_eq(v.y(), 0.0));
    assert!(Tolerance::default().approx_eq(v.z(), 0.0));
}

#[test]
fn parabola_symmetry() {
    let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    // z-component should be the same for +t and -t (symmetric about axis)
    let pos = p.evaluate(2.0);
    let neg = p.evaluate(-2.0);
    assert!(Tolerance::default().approx_eq(pos.z(), neg.z()));
}

#[test]
fn parabola_tangent_at_vertex() {
    let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let tang = p.tangent(0.0);
    // At vertex, tangent should be purely along u_axis (perpendicular to axis)
    assert!(Tolerance::default().approx_eq(tang.z(), 0.0));
    assert!(tang.length() > 0.5);
}

#[test]
fn parabola_curvature_at_vertex() {
    let f = 2.0;
    let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), f).unwrap();
    // Curvature at vertex = 1/(2f)
    let k = p.curvature(0.0);
    assert!(Tolerance::default().approx_eq(k, 1.0 / (2.0 * f)));
}

#[test]
fn parabola_focus() {
    let f = 3.0;
    let p = Parabola3D::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 1.0), f).unwrap();
    let focus = p.focus();
    assert!(Tolerance::default().approx_eq(focus.z(), 3.0 + f));
}

#[test]
fn parabola_zero_focal_length_error() {
    assert!(Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.0,).is_err());
}

// ── Hyperbola tests ─────────────────────────────────────────

#[test]
fn hyperbola_vertex() {
    let h = Hyperbola3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        3.0,
        2.0,
    )
    .unwrap();
    // At t=0: P = center + a*cosh(0)*u + b*sinh(0)*v = center + a*u
    let p = h.evaluate(0.0);
    let dist = (p - h.center()).length();
    assert!(Tolerance::default().approx_eq(dist, 3.0));
}

#[test]
fn hyperbola_tangent_at_vertex() {
    let h = Hyperbola3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        3.0,
        2.0,
    )
    .unwrap();
    // At t=0: tangent = a*sinh(0)*u + b*cosh(0)*v = b*v
    let tang = h.tangent(0.0);
    assert!(Tolerance::default().approx_eq(tang.length(), 2.0));
}

#[test]
fn hyperbola_eccentricity() {
    let h = Hyperbola3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        3.0,
        4.0,
    )
    .unwrap();
    // e = sqrt(1 + (b/a)^2) = sqrt(1 + 16/9) = sqrt(25/9) = 5/3
    assert!(Tolerance::default().approx_eq(h.eccentricity(), 5.0 / 3.0));
}

#[test]
fn hyperbola_foci_distance() {
    let a = 3.0;
    let b = 4.0;
    let h = Hyperbola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), a, b).unwrap();
    let (f1, f2) = h.foci();
    // Distance from center to each focus = c = sqrt(a^2 + b^2) = 5
    let c = a.hypot(b);
    assert!(Tolerance::default().approx_eq((f1 - h.center()).length(), c));
    assert!(Tolerance::default().approx_eq((f2 - h.center()).length(), c));
}

#[test]
fn hyperbola_zero_axis_error() {
    assert!(
        Hyperbola3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
            1.0,
        )
        .is_err()
    );
}

// ── Line3D::tangent tests ──────────────────────────

#[test]
fn line3d_tangent_is_unit_vector() {
    // Even when constructed with a non-unit direction, tangent() must return a unit vector.
    let line = Line3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 4.0, 0.0)).unwrap();
    let tang = line.tangent();
    let tol = Tolerance::new();
    assert!(
        tol.approx_eq(tang.length(), 1.0),
        "tangent should be unit length, got {}",
        tang.length()
    );
}

#[test]
fn line3d_tangent_matches_direction() {
    let line = Line3D::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 5.0)).unwrap();
    let tang = line.tangent();
    let dir = line.direction();
    let tol = Tolerance::new();
    assert!(tol.approx_eq(tang.x(), dir.x()));
    assert!(tol.approx_eq(tang.y(), dir.y()));
    assert!(tol.approx_eq(tang.z(), dir.z()));
}

// ── Circle3D::project tests ────────────────────────

#[test]
fn circle3d_project_at_pi() {
    // At t=π, evaluate gives the antipodal point; projecting back should return π.
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let pt = circle.evaluate(PI);
    let t_proj = circle.project(pt);
    let tol = Tolerance::new();
    // project returns atan2, which is in (-π, π]; -π and π are the same angle
    assert!(tol.approx_eq(t_proj.abs(), PI), "expected ±π, got {t_proj}");
}

#[test]
fn circle3d_project_at_negative_angle() {
    // project should recover a negative angle correctly (atan2 range is (-π, π])
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
    let t_orig = -1.0_f64;
    let pt = circle.evaluate(t_orig);
    let t_proj = circle.project(pt);
    let tol = Tolerance::new();
    assert!(
        tol.approx_eq(t_orig, t_proj),
        "expected {t_orig}, got {t_proj}"
    );
}

#[test]
fn circle3d_project_at_three_quarter() {
    // t = 3π/2 maps to -π/2 under atan2; round-trip through evaluate→project
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();
    let t_orig = 3.0 * PI / 2.0;
    let pt = circle.evaluate(t_orig);
    let t_proj = circle.project(pt);
    // atan2 wraps to -π/2, so recovered angle differs by 2π
    let tol = Tolerance::new();
    let diff = (t_proj - t_orig).abs() % (2.0 * PI);
    let diff_wrapped = diff.min(2.0f64.madd(PI, -diff));
    assert!(
        diff_wrapped < tol.linear * 1000.0,
        "angle mismatch after wrapping: t_orig={t_orig}, t_proj={t_proj}"
    );
}

// ── Parabola3D::evaluate at non-zero t ─────────────

#[test]
fn parabola_evaluate_nonzero_t() {
    // P(t) = vertex + (t²/4f)*axis + t*u_axis
    // For axis = (0,0,1), focal_length=1, at t=2:
    //   along_axis = 4/4 = 1   → z-component = 1
    //   u_axis component = 2    → in the plane perpendicular to z
    let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let pt = p.evaluate(2.0);
    let tol = Tolerance::new();
    // z (along-axis) component
    assert!(
        tol.approx_eq(pt.z(), 1.0),
        "z-component should be 1.0, got {}",
        pt.z()
    );
    // distance from axis in the XY plane should be |t| = 2
    let radial = pt.x().hypot(pt.y());
    assert!(
        tol.approx_eq(radial, 2.0),
        "radial distance should be 2.0, got {radial}"
    );
}

#[test]
fn parabola_evaluate_negative_t() {
    // Symmetry: z-component the same for +t and -t, radial same, u_axis direction flips
    let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let pos = p.evaluate(3.0);
    let neg = p.evaluate(-3.0);
    let tol = Tolerance::new();
    assert!(tol.approx_eq(pos.z(), neg.z()), "z should match for ±t");
    // The x/y components should be opposite in sign (u_axis flips)
    let radial_pos = pos.x().hypot(pos.y());
    let radial_neg = neg.x().hypot(neg.y());
    assert!(
        tol.approx_eq(radial_pos, radial_neg),
        "radial distance should match"
    );
}

#[test]
fn parabola_tangent_nonzero_t() {
    // tangent(t) = (t / 2f) * axis + u_axis
    // For focal_length=2, axis=(0,0,1), t=4:
    //   d_axis = 4/(2*2) = 1  → z-component of tangent = 1
    //   u_axis component = 1
    let p = Parabola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
    let tang = p.tangent(4.0);
    let tol = Tolerance::new();
    assert!(
        tol.approx_eq(tang.z(), 1.0),
        "z-component of tangent should be 1.0, got {}",
        tang.z()
    );
    // u_axis is in XY plane, its component should give length 1
    let radial = tang.x().hypot(tang.y());
    assert!(
        tol.approx_eq(radial, 1.0),
        "XY component of tangent should have length 1, got {radial}"
    );
}

// ── Hyperbola3D::evaluate away from vertex ─────────

#[test]
fn hyperbola_evaluate_nonzero_t() {
    // P(t) = center + a*cosh(t)*u + b*sinh(t)*v
    // The distance from center at parameter t satisfies:
    //   |P - center|² = a²*cosh²(t) + b²*sinh²(t)
    let a = 2.0_f64;
    let b = 1.0_f64;
    let t = 1.0_f64;
    let h = Hyperbola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), a, b).unwrap();
    let pt = h.evaluate(t);
    let dist_sq = (pt - h.center()).length_squared();
    let expected_sq = (a * t.cosh()).madd(a * t.cosh(), (b * t.sinh()).powi(2));
    assert!(
        (dist_sq - expected_sq).abs() < 1e-10,
        "|P-center|² should be {expected_sq}, got {dist_sq}"
    );
}

#[test]
fn hyperbola_evaluate_large_t_stays_on_curve() {
    // Verify the hyperbolic identity: (x/a)² - (y/b)² = 1
    // where x, y are the in-plane u/v components.
    // We use a finite-difference approximation to recover u/v components.
    let a = 3.0_f64;
    let b = 2.0_f64;
    let h = Hyperbola3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), a, b).unwrap();
    // The distance from center equals sqrt(a²cosh²(t) + b²sinh²(t))
    for t in [0.5_f64, 1.0, 2.0] {
        let pt = h.evaluate(t);
        let dist = (pt - h.center()).length();
        let expected = (a * t.cosh()).hypot(b * t.sinh());
        assert!(
            (dist - expected).abs() < 1e-10,
            "distance at t={t}: expected {expected}, got {dist}"
        );
    }
}

#[test]
fn hyperbola_tangent_nonzero_t() {
    // Check tangent via finite difference: tangent ≈ (P(t+ε) - P(t-ε)) / (2ε)
    let h = Hyperbola3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        2.0,
        1.0,
    )
    .unwrap();
    let t = 1.0_f64;
    let eps = 1e-6;
    let tang_analytic = h.tangent(t);
    let pt_plus = h.evaluate(t + eps);
    let pt_minus = h.evaluate(t - eps);
    let tang_fd_x = (pt_plus.x() - pt_minus.x()) / (2.0 * eps);
    let tang_fd_y = (pt_plus.y() - pt_minus.y()) / (2.0 * eps);
    let tang_fd_z = (pt_plus.z() - pt_minus.z()) / (2.0 * eps);
    let tol = 1e-5;
    assert!(
        (tang_analytic.x() - tang_fd_x).abs() < tol,
        "x: analytic={}, fd={tang_fd_x}",
        tang_analytic.x()
    );
    assert!(
        (tang_analytic.y() - tang_fd_y).abs() < tol,
        "y: analytic={}, fd={tang_fd_y}",
        tang_analytic.y()
    );
    assert!(
        (tang_analytic.z() - tang_fd_z).abs() < tol,
        "z: analytic={}, fd={tang_fd_z}",
        tang_analytic.z()
    );
}

#[test]
fn circle_intersect_circle_two_points() {
    // Two unit-radius coplanar circles with centers 1 apart: two crossings at
    // x = 0.5, y = ±sqrt(3)/2.
    let a = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let b = Circle3D::new(Point3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let hits = a.intersect_circle(&b, 1e-9);
    assert_eq!(hits.len(), 2);
    for (p, t) in &hits {
        assert!((p.x() - 0.5).abs() < 1e-12);
        assert!((p.y().abs() - 3.0_f64.sqrt() / 2.0).abs() < 1e-12);
        // Returned parameter evaluates back to the point on `a`.
        let q = a.evaluate(*t);
        assert!((q - *p).length() < 1e-12);
    }
}

/// The rim of a rod of radius 3 mitred by the plane z = 3 + 0.5 x: the
/// ellipse centred on the axis at z = 3, its major axis up the slope.
fn mitre_rim() -> Ellipse3D {
    let root5 = 5.0_f64.sqrt();
    Ellipse3D::with_axes(
        Point3::new(0.0, 0.0, 3.0),
        Vec3::new(-1.0 / root5, 0.0, 2.0 / root5),
        3.0 * 1.25_f64.sqrt(),
        3.0,
        Vec3::new(2.0 / root5, 0.0, 1.0 / root5),
        Vec3::new(0.0, 1.0, 0.0),
    )
    .unwrap()
}

#[test]
fn circle_intersect_ellipse_crossing_its_plane() {
    // The wall's circle at z = 4.3 crosses the rim where 3 + 0.5 x = 4.3.
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 4.3), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
    let hits = circle.intersect_ellipse(&mitre_rim(), 1e-9);
    assert_eq!(hits.len(), 2);
    let y = (9.0_f64 - 2.6 * 2.6).sqrt();
    for (p, t) in &hits {
        assert!((p.x() - 2.6).abs() < 1e-12, "{p:?}");
        assert!((p.y().abs() - y).abs() < 1e-12, "{p:?}");
        assert!((p.z() - 4.3).abs() < 1e-12, "{p:?}");
        assert!((circle.evaluate(*t) - *p).length() < 1e-12);
    }
    assert!(hits[0].0.y() * hits[1].0.y() < 0.0);
}

#[test]
fn circle_intersect_ellipse_at_and_past_its_peak() {
    let rim = mitre_rim();
    let at = |z: f64| {
        Circle3D::new(Point3::new(0.0, 0.0, z), Vec3::new(0.0, 0.0, 1.0), 3.0)
            .unwrap()
            .intersect_ellipse(&rim, 1e-9)
    };
    let touch = at(4.5);
    assert_eq!(touch.len(), 1);
    assert!((touch[0].0 - Point3::new(3.0, 0.0, 4.5)).length() < 1e-6);
    assert!(at(4.6).is_empty());
    // A circle of another radius in a crossed plane meets the ellipse nowhere.
    let wide = Circle3D::new(Point3::new(0.0, 0.0, 4.3), Vec3::new(0.0, 0.0, 1.0), 3.5).unwrap();
    assert!(wide.intersect_ellipse(&rim, 1e-9).is_empty());
}

#[test]
fn circle_intersect_ellipse_in_its_plane() {
    // The ellipse with semi-axes 2 and 1 crosses the circle of radius 1.5
    // four times.
    let ellipse = Ellipse3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        2.0,
        1.0,
    )
    .unwrap();
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.5).unwrap();
    let hits = circle.intersect_ellipse(&ellipse, 1e-9);
    assert_eq!(hits.len(), 4);
    for (p, t) in &hits {
        let d = *p - ellipse.center();
        let (x, y) = (d.dot(ellipse.u_axis()) / 2.0, d.dot(ellipse.v_axis()));
        assert!(
            (x.madd(x, y * y) - 1.0).abs() < 1e-9,
            "{p:?} is off the ellipse"
        );
        assert!(
            ((*p - circle.center()).length() - 1.5).abs() < 1e-9,
            "{p:?}"
        );
        assert!((circle.evaluate(*t) - *p).length() < 1e-9);
    }
    // In a parallel plane off it, none.
    let above = Circle3D::new(Point3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0), 1.5).unwrap();
    assert!(above.intersect_ellipse(&ellipse, 1e-9).is_empty());
}

#[test]
fn circle_intersect_circle_lite_pad_configuration() {
    // The lite magnet-pad junction: pad circle r=4.45 about (-50,-76) crossing
    // the socket cone's z=-3.8 rim circle r=1.05 about (-47.45,-78.55). The
    // crossings are the two triple-junction points of the pad fuse.
    let pad = Circle3D::new(
        Point3::new(-50.0, -76.0, -3.8),
        Vec3::new(0.0, 0.0, 1.0),
        4.45,
    )
    .unwrap();
    let rim = Circle3D::new(
        Point3::new(-47.45, -78.55, -3.8),
        Vec3::new(0.0, 0.0, 1.0),
        1.05,
    )
    .unwrap();
    let hits = pad.intersect_circle(&rim, 1e-7);
    assert_eq!(hits.len(), 2);
    for (p, _) in &hits {
        let dp = ((p.x() + 50.0).powi(2) + (p.y() + 76.0).powi(2)).sqrt();
        let dr = ((p.x() + 47.45).powi(2) + (p.y() + 78.55).powi(2)).sqrt();
        assert!((dp - 4.45).abs() < 1e-9, "on pad circle: {dp}");
        assert!((dr - 1.05).abs() < 1e-9, "on rim circle: {dr}");
    }
}

#[test]
fn circle_intersect_circle_near_tangent_collapses_to_foot() {
    // Externally near-tangent pair: centers 2 + delta apart with delta far
    // below the tolerance well. The noise root pair collapses to the single
    // well-conditioned foot on the center line.
    let a = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let b = Circle3D::new(
        Point3::new(2.0 - 1e-12, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        1.0,
    )
    .unwrap();
    let hits = a.intersect_circle(&b, 1e-7);
    assert_eq!(hits.len(), 1);
    let (p, _) = hits[0];
    assert!((p.x() - 1.0).abs() < 1e-6);
    assert!(p.y().abs() < 1e-6);
}

#[test]
fn circle_intersect_circle_on_one_sphere() {
    // The latitude z = 1 of a radius-3 sphere and its great circle in y = 0
    // meet at (±sqrt(8), 0, 1), from either side.
    let latitude = Circle3D::new(
        Point3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, 1.0),
        8.0_f64.sqrt(),
    )
    .unwrap();
    let great = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 3.0).unwrap();
    for (a, b) in [(&latitude, &great), (&great, &latitude)] {
        let hits = a.intersect_circle(b, 1e-9);
        assert_eq!(hits.len(), 2);
        for (p, t) in &hits {
            assert!((p.x().abs() - 8.0_f64.sqrt()).abs() < 1e-12);
            assert!(p.y().abs() < 1e-12);
            assert!((p.z() - 1.0).abs() < 1e-12);
            assert!((a.evaluate(*t) - *p).length() < 1e-12);
        }
    }
}

#[test]
fn circle_intersect_circle_disjoint_and_non_coplanar_empty() {
    let a = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    // Far apart.
    let b = Circle3D::new(Point3::new(5.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    assert!(a.intersect_circle(&b, 1e-9).is_empty());
    // Parallel but offset planes.
    let c = Circle3D::new(Point3::new(1.0, 0.0, 0.5), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    assert!(a.intersect_circle(&c, 1e-9).is_empty());
    // Skew planes whose common line misses one of the circles.
    let d = Circle3D::new(Point3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 1.0).unwrap();
    assert!(a.intersect_circle(&d, 1e-9).is_empty());
    // Concentric.
    let e = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.5).unwrap();
    assert!(a.intersect_circle(&e, 1e-9).is_empty());
}

type Sampler<'a> = &'a dyn Fn(f64) -> Point3;
type ArcBox<'a> = &'a dyn Fn(f64, f64) -> crate::aabb::Aabb3;

/// A tilted circle's and a tilted ellipse's boxes, whole and over arcs that
/// hold some of the full turn's extremes (both ways, and across angle zero),
/// hold every one of 20,000 samples and touch the samples' extremes.
#[test]
fn tilted_conic_boxes_match_their_samples() {
    let circle = Circle3D::new(Point3::new(1.0, -2.0, 0.5), Vec3::new(1.0, 2.0, 3.0), 1.7).unwrap();
    let ellipse = Ellipse3D::new(
        Point3::new(-0.5, 0.3, 2.0),
        Vec3::new(-2.0, 0.5, 1.0),
        2.3,
        0.9,
    )
    .unwrap();
    let eval: [(Sampler<'_>, ArcBox<'_>); 2] = [
        (&|t| circle.evaluate(t), &|a, b| circle.arc_aabb(a, b)),
        (&|t| ellipse.evaluate(t), &|a, b| ellipse.arc_aabb(a, b)),
    ];
    for (k, (at, arc)) in eval.iter().enumerate() {
        for (t0, t1) in [
            (0.0, 2.0 * PI),
            (0.3, 1.9),
            (1.9, 0.3),
            (-2.5, 0.4),
            (4.0, 7.5),
            (2.0, 5.5),
        ] {
            let b = arc(t0, t1);
            let n = 20_000;
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for i in 0..=n {
                let p = at(t0 + (t1 - t0) * f64::from(i) / f64::from(n));
                for (axis, c) in [p.x(), p.y(), p.z()].into_iter().enumerate() {
                    lo[axis] = lo[axis].min(c);
                    hi[axis] = hi[axis].max(c);
                }
            }
            let (bmin, bmax) = (
                [b.min.x(), b.min.y(), b.min.z()],
                [b.max.x(), b.max.y(), b.max.z()],
            );
            for axis in 0..3 {
                assert!(
                    bmin[axis] <= lo[axis] + 1e-12 && bmax[axis] >= hi[axis] - 1e-12,
                    "conic {k} ({t0}, {t1}) axis {axis}: box misses a sample"
                );
                assert!(
                    lo[axis] - bmin[axis] < 1e-6 && bmax[axis] - hi[axis] < 1e-6,
                    "conic {k} ({t0}, {t1}) axis {axis}: box too loose"
                );
            }
        }
        let whole = if k == 0 {
            circle.aabb()
        } else {
            ellipse.aabb()
        };
        let full = arc(0.0, 2.0 * PI);
        assert!((whole.min - full.min).length() < 1e-12 && (whole.max - full.max).length() < 1e-12);
    }
}

/// Directions given at any length, or a little off square, are normalized
/// and squared: the focus `focus()` reports is where the evaluated
/// parabola's focus is, and the hyperbola's vertex sits a real semi-axis
/// along its normalized `u_axis`.
#[test]
fn conics_with_axes_normalize_their_frames() {
    let origin = Point3::new(0.0, 0.0, 0.0);
    let parabola = Parabola3D::with_axes(
        origin,
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(1.0, 0.0, 0.1),
        1.0,
    )
    .unwrap();
    assert!((parabola.focus() - Point3::new(0.0, 0.0, 1.0)).length() < 1e-15);
    assert!((parabola.evaluate(2.0) - Point3::new(2.0, 0.0, 1.0)).length() < 1e-15);
    let hyperbola = Hyperbola3D::with_axes(
        origin,
        2.0,
        1.0,
        Vec3::new(3.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    )
    .unwrap();
    assert!((hyperbola.evaluate(0.0) - Point3::new(2.0, 0.0, 0.0)).length() < 1e-15);
    assert!((hyperbola.normal() - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-15);
    let (near, _) = hyperbola.foci();
    assert!((near - Point3::new(5.0_f64.sqrt(), 0.0, 0.0)).length() < 1e-15);
    assert!(
        Parabola3D::with_axes(
            origin,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 3.0),
            1.0
        )
        .is_err()
    );
}

/// Each constructor refuses a size or a direction that is not finite and in
/// range. A test of `value <= 0.0` alone lets NaN through, since every
/// comparison with NaN is false.
#[test]
fn constructors_refuse_sizes_that_are_not_finite_and_in_range() {
    let (o, z, x, y) = (
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        assert!(Circle3D::new(o, z, bad).is_err(), "{bad}");
        assert!(Circle3D::new_with_ref(o, z, bad, x).is_err(), "{bad}");
        assert!(Circle3D::with_axes(o, z, bad, x, y).is_err(), "{bad}");
        for (a, b) in [(bad, 1.0), (2.0, bad)] {
            assert!(Ellipse3D::new(o, z, a, b).is_err(), "{a} {b}");
            assert!(Ellipse3D::new_with_ref(o, z, a, b, x).is_err(), "{a} {b}");
            assert!(Ellipse3D::with_axes(o, z, a, b, x, y).is_err(), "{a} {b}");
            assert!(Hyperbola3D::new(o, z, a, b).is_err(), "{a} {b}");
            assert!(Hyperbola3D::with_axes(o, a, b, x, y).is_err(), "{a} {b}");
        }
        assert!(Parabola3D::new(o, z, bad).is_err(), "{bad}");
        assert!(Parabola3D::with_axes(o, z, x, bad).is_err(), "{bad}");
    }
    for bad in [f64::NAN, f64::INFINITY] {
        assert!(Line3D::new(o, Vec3::new(bad, 0.0, 0.0)).is_err(), "{bad}");
    }
    assert!(Circle3D::new(o, z, 2.0).is_ok());
    assert!(Ellipse3D::with_axes(o, z, 2.0, 1.0, x, y).is_ok());
    assert!(Parabola3D::new(o, z, 1.0).is_ok());
    assert!(Hyperbola3D::new(o, z, 2.0, 1.0).is_ok());
    assert!(Line3D::new(o, x).is_ok());
}

/// An ellipse of semi-axes 2 and 1 about the normal (0, 0, 1), with the
/// reference (1e-9, 0, 1): the reference's part square to the normal is
/// (1e-9, 0, 0), exact, so the major axis lies along x. Turned two ways,
/// the major axis turns with it.
#[test]
fn an_ellipse_keeps_the_major_axis_of_a_reference_near_its_normal() {
    use crate::mat::Mat4;
    let turn = |m: &Mat4, v: Vec3| {
        let origin = Point3::new(0.0, 0.0, 0.0);
        m.mul_point(origin + v) - m.mul_point(origin)
    };
    for pose in [
        Mat4::identity(),
        Mat4::rotation_x(0.7) * Mat4::rotation_z(-1.3),
        Mat4::rotation_y(2.1) * Mat4::rotation_x(0.4),
    ] {
        let normal = turn(&pose, Vec3::new(0.0, 0.0, 1.0));
        let reference = turn(&pose, Vec3::new(1e-9, 0.0, 1.0));
        let e = Ellipse3D::new_with_ref(Point3::new(0.0, 0.0, 0.0), normal, 2.0, 1.0, reference)
            .unwrap();
        let want = turn(&pose, Vec3::new(1.0, 0.0, 0.0));
        assert!(
            (e.u_axis() - want).length() < 1e-6,
            "{:?} for {want:?}",
            e.u_axis()
        );
    }
}
