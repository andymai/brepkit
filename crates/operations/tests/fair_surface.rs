//! Fairing, measured.
//!
//! Bending energy here is `integral (k1^2 + k2^2) dA`, the sum of the squared
//! principal curvatures over the surface, read from the fundamental forms at
//! Gauss points on each knot span. It is a property of the shape alone, so it
//! does not lean on the parametric energy the operation itself minimises.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::panic
)]

mod surface_fixtures;

use brepkit_math::nurbs::fitting::interpolate;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::vec::Point3;
use brepkit_operations::OperationsError;
use brepkit_operations::blend_surface::fair_surface;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};
use surface_fixtures::{Builder, Pose, net_sheet, sheet};

type Height<'a> = &'a dyn Fn(usize, usize, f64, f64) -> f64;

/// `integral (k1^2 + k2^2) dA` over the surface.
fn bending_energy(surface: &NurbsSurface) -> f64 {
    let spans = |knots: &[f64]| -> Vec<(f64, f64)> {
        let mut found = Vec::new();
        for pair in knots.windows(2) {
            if pair[1] - pair[0] > 1e-12 {
                for k in 0..3 {
                    let at = |i: f64| pair[0] + (pair[1] - pair[0]) * i / 3.0;
                    found.push((at(f64::from(k)), at(f64::from(k + 1))));
                }
            }
        }
        found
    };
    let (us, vs) = (spans(surface.knots_u()), spans(surface.knots_v()));
    let points = gauss_legendre_points(4);
    let mut sum = 0.0;
    for &(u0, u1) in &us {
        for &(v0, v1) in &vs {
            for pu in points {
                for pv in points {
                    let u = 0.5 * (u0 + u1) + 0.5 * (u1 - u0) * pu.x;
                    let v = 0.5 * (v0 + v1) + 0.5 * (v1 - v0) * pv.x;
                    let weight = pu.w * pv.w * 0.25 * (u1 - u0) * (v1 - v0);
                    let d = surface.derivatives(u, v, 2);
                    let (su, sv) = (d[1][0], d[0][1]);
                    let cross = su.cross(sv);
                    let area = cross.length();
                    if area <= 1e-14 {
                        continue;
                    }
                    let n = cross * (1.0 / area);
                    let (e, f, g) = (su.dot(su), su.dot(sv), sv.dot(sv));
                    let (l, m, nn) = (n.dot(d[2][0]), n.dot(d[1][1]), n.dot(d[0][2]));
                    let det = e * g - f * f;
                    let mean = (e * nn - 2.0 * f * m + g * l) / (2.0 * det);
                    let gauss = (l * nn - m * m) / det;
                    sum += (4.0 * mean * mean - 2.0 * gauss) * area * weight;
                }
            }
        }
    }
    sum
}

/// The farthest apart two surfaces get at the same parameters, densely.
fn farthest(a: &NurbsSurface, b: &NurbsSurface, per_side: i32) -> f64 {
    let mut worst = 0.0_f64;
    let (u0, u1) = a.domain_u();
    let (v0, v1) = a.domain_v();
    for i in 0..=per_side {
        for j in 0..=per_side {
            let u = u0 + (u1 - u0) * f64::from(i) / f64::from(per_side);
            let v = v0 + (v1 - v0) * f64::from(j) / f64::from(per_side);
            worst = worst.max((a.evaluate(u, v) - b.evaluate(u, v)).length());
        }
    }
    worst
}

/// The farthest apart two surfaces get along the sides of their domain.
fn boundary_moved(a: &NurbsSurface, b: &NurbsSurface) -> f64 {
    let (u0, u1) = a.domain_u();
    let (v0, v1) = a.domain_v();
    let mut worst = 0.0_f64;
    for i in 0..=400 {
        let t = f64::from(i) / 400.0;
        let (u, v) = (u0 + (u1 - u0) * t, v0 + (v1 - v0) * t);
        for (a_at, b_at) in [
            (a.evaluate(u, v0), b.evaluate(u, v0)),
            (a.evaluate(u, v1), b.evaluate(u, v1)),
            (a.evaluate(u0, v), b.evaluate(u0, v)),
            (a.evaluate(u1, v), b.evaluate(u1, v)),
        ] {
            worst = worst.max((a_at - b_at).length());
        }
    }
    worst
}

fn nurbs_of(topo: &Topology, face: FaceId) -> NurbsSurface {
    match topo.face(face).unwrap().surface() {
        FaceSurface::Nurbs(surface) => surface.clone(),
        other => panic!("wanted a spline, got {}", other.type_tag()),
    }
}

fn uniform_knots(count: usize, degree: usize) -> Vec<f64> {
    let spans = count - degree;
    let mut knots = vec![0.0; degree];
    for i in 0..=spans {
        knots.push(i as f64 / spans as f64);
    }
    knots.extend(std::iter::repeat_n(1.0, degree));
    knots
}

/// A cubic sheet over a 10 x 10 square whose net follows `height`.
fn cubic_sheet(
    topo: &mut Topology,
    pose: Pose,
    count: usize,
    height: impl Fn(usize, usize, f64, f64) -> f64,
) -> FaceId {
    let net: Vec<Vec<Point3>> = (0..count)
        .map(|i| {
            let x = 10.0 * i as f64 / (count - 1) as f64;
            (0..count)
                .map(|j| {
                    let y = 10.0 * j as f64 / (count - 1) as f64;
                    Point3::new(x, y, height(i, j, x, y))
                })
                .collect()
        })
        .collect();
    let weights = vec![vec![1.0; count]; count];
    let knots = uniform_knots(count, 3);
    let mut b = Builder { topo, pose };
    net_sheet(&mut b, (3, 3), (knots.clone(), knots), &net, &weights).0
}

/// A dome with bumps on every other control point.
fn bumpy(amplitude: f64) -> impl Fn(usize, usize, f64, f64) -> f64 {
    move |i, j, x, y| {
        let dome = 1.2 * (x * (10.0 - x) + y * (10.0 - y)) / 25.0;
        let bump = if (i + j) % 2 == 0 {
            amplitude
        } else {
            -amplitude
        };
        let inside = i > 0 && j > 0 && i < 14 && j < 14;
        dome + if inside { bump } else { 0.0 }
    }
}

#[test]
fn a_bumpy_surface_comes_out_smoother_within_tolerance_with_its_boundary_held() {
    let tolerance = 0.12;
    for (name, pose) in [
        ("upright", Pose::upright()),
        ("turned", Pose::turned()),
        ("mirrored", Pose::mirrored()),
    ] {
        let mut topo = Topology::new();
        let face = cubic_sheet(&mut topo, pose, 15, bumpy(0.1));
        let before = nurbs_of(&topo, face);
        let result = fair_surface(&mut topo, face, tolerance).unwrap();
        assert_ne!(face, result);
        let after = nurbs_of(&topo, result);
        let (energy_before, energy_after) = (bending_energy(&before), bending_energy(&after));
        let moved = farthest(&before, &after, 200);
        let held = boundary_moved(&before, &after);
        // The dome under the bumps is what ought to be left.
        let mut clean_topo = Topology::new();
        let clean_face = cubic_sheet(&mut clean_topo, pose, 15, bumpy(0.0));
        let clean = bending_energy(&nurbs_of(&clean_topo, clean_face));
        println!(
            "fair bumpy {name}: bending energy {energy_before:.4} -> {energy_after:.4} \
             ({:.1}%), smooth dome under the bumps {clean:.4}, farthest {moved:.4} of tolerance \
             {tolerance}, boundary moved {held:.2e}",
            100.0 * energy_after / energy_before
        );
        assert!(
            energy_after < 1.05 * clean,
            "{energy_after} against {clean}"
        );
        assert!(moved <= tolerance, "{moved}");
        assert!(held < 1e-12, "{held}");
        assert!(moved > 0.01, "something has to have moved: {moved}");
        // The source is untouched.
        assert_eq!(
            nurbs_of(&topo, face).control_points()[3][3],
            before.control_points()[3][3]
        );
    }
}

#[test]
fn a_tight_tolerance_is_kept_and_still_lowers_the_energy() {
    for tolerance in [0.03, 0.01, 0.001] {
        let mut topo = Topology::new();
        let face = cubic_sheet(&mut topo, Pose::turned(), 15, bumpy(0.1));
        let before = nurbs_of(&topo, face);
        let result = fair_surface(&mut topo, face, tolerance).unwrap();
        let after = nurbs_of(&topo, result);
        let moved = farthest(&before, &after, 200);
        let (energy_before, energy_after) = (bending_energy(&before), bending_energy(&after));
        println!(
            "fair at tolerance {tolerance}: {:.1}% of the energy, farthest {moved:.5}",
            100.0 * energy_after / energy_before
        );
        assert!(moved <= tolerance, "{tolerance}: {moved}");
        assert!(
            moved >= 0.5 * tolerance.min(0.02),
            "{tolerance}: only used {moved}"
        );
        assert!(energy_after < energy_before, "{tolerance}");
    }
}

#[test]
fn a_fair_surface_barely_changes() {
    let wave = |_: usize, _: usize, x: f64, y: f64| {
        (x * std::f64::consts::PI / 10.0).sin() * (y * std::f64::consts::PI / 10.0).sin() * 2.0
    };
    let bowl =
        |_: usize, _: usize, x: f64, y: f64| ((x - 5.0) * (x - 5.0) + (y - 5.0) * (y - 5.0)) * 0.05;
    let saddle = |_: usize, _: usize, x: f64, y: f64| (x - 5.0) * (y - 5.0) * 0.08;
    let flat = |_: usize, _: usize, _: f64, _: f64| 0.0;
    let tilted = |_: usize, _: usize, x: f64, y: f64| 0.2 * x - 0.1 * y;
    let two_lobes = |_: usize, _: usize, x: f64, y: f64| {
        (x * std::f64::consts::TAU / 10.0).sin() * (y * std::f64::consts::PI / 10.0).sin() * 1.5
    };
    let tolerance = 0.2;
    let cases: [(&str, Height<'_>); 6] = [
        ("wave", &wave),
        ("two lobes", &two_lobes),
        ("bowl", &bowl),
        ("saddle", &saddle),
        ("flat", &flat),
        ("tilted", &tilted),
    ];
    for (name, height) in cases {
        for count in [6, 15] {
            let mut topo = Topology::new();
            let face = cubic_sheet(&mut topo, Pose::turned(), count, height);
            let before = nurbs_of(&topo, face);
            let result = fair_surface(&mut topo, face, tolerance).unwrap();
            let after = nurbs_of(&topo, result);
            let moved = farthest(&before, &after, 100);
            println!("fair already fair {name} {count}: moved {moved:.2e}, tolerance {tolerance}");
            assert!(moved <= 0.1 * tolerance, "{name} {count}: {moved}");
        }
    }
}

/// A cylinder spline, rational and all: fair, so it comes back as it was.
#[test]
fn a_rational_cylinder_barely_changes() {
    let mut topo = Topology::new();
    let sheet_face = sheet(&mut topo, "rational spline", Pose::turned());
    let before = nurbs_of(&topo, sheet_face.face);
    let result = fair_surface(&mut topo, sheet_face.face, 0.1).unwrap();
    let after = nurbs_of(&topo, result);
    let moved = farthest(&before, &after, 100);
    assert!(moved < 1e-3, "{moved}");
}

/// A rational surface with bumps stays rational, its weights kept.
#[test]
fn a_rational_bumpy_surface_comes_out_smoother() {
    let count = 15;
    let height = bumpy(0.1);
    let net: Vec<Vec<Point3>> = (0..count)
        .map(|i| {
            let x = 10.0 * i as f64 / (count - 1) as f64;
            (0..count)
                .map(|j| {
                    let y = 10.0 * j as f64 / (count - 1) as f64;
                    Point3::new(x, y, height(i, j, x, y))
                })
                .collect()
        })
        .collect();
    let weights: Vec<Vec<f64>> = (0..count)
        .map(|i| {
            (0..count)
                .map(|j| 1.0 + 0.2 * (i as f64 * 0.7).sin() * (j as f64 * 0.5).cos())
                .collect()
        })
        .collect();
    let knots = uniform_knots(count, 3);
    let mut topo = Topology::new();
    let mut b = Builder {
        topo: &mut topo,
        pose: Pose::turned(),
    };
    let (face, _) = net_sheet(&mut b, (3, 3), (knots.clone(), knots), &net, &weights);
    let before = nurbs_of(&topo, face);
    assert!(before.is_rational());
    let tolerance = 0.12;
    let result = fair_surface(&mut topo, face, tolerance).unwrap();
    let after = nurbs_of(&topo, result);
    assert_eq!(after.weights(), before.weights());
    let moved = farthest(&before, &after, 200);
    let held = boundary_moved(&before, &after);
    let (energy_before, energy_after) = (bending_energy(&before), bending_energy(&after));
    println!(
        "fair rational: energy {energy_before:.4} -> {energy_after:.4}, moved {moved:.4}, \
         boundary {held:.2e}"
    );
    assert!(energy_after < 0.5 * energy_before, "{energy_after}");
    assert!(moved <= tolerance, "{moved}");
    assert!(held < 1e-12, "{held}");
}

/// A trimmed face, with a hole: the surface is held along both wires and
/// wherever it does not matter, and only its middle moves.
#[test]
fn a_trimmed_face_with_a_hole_is_held_along_both_wires() {
    let mut topo = Topology::new();
    let knots = uniform_knots(25, 3);
    let count = 25;
    let net: Vec<Vec<Point3>> = (0..count)
        .map(|i| {
            let x = 10.0 * i as f64 / (count - 1) as f64;
            (0..count)
                .map(|j| {
                    let y = 10.0 * j as f64 / (count - 1) as f64;
                    let bump = if (i + j) % 2 == 0 { 0.05 } else { -0.05 };
                    Point3::new(x, y, 0.01 * x * y + bump)
                })
                .collect()
        })
        .collect();
    let weights = vec![vec![1.0; count]; count];
    let surface = NurbsSurface::new(3, 3, knots.clone(), knots, net, weights).unwrap();

    let on_surface = |u: f64, v: f64| surface.evaluate(u, v);
    let rim = |from: (f64, f64), to: (f64, f64)| -> Vec<Point3> {
        (0..=40)
            .map(|k| {
                let t = f64::from(k) / 40.0;
                on_surface(from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t)
            })
            .collect()
    };
    let corners = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)];
    let outer_points: Vec<Vec<Point3>> = (0..4)
        .map(|k| rim(corners[k], corners[(k + 1) % 4]))
        .collect();
    let hole_corners = [(0.45, 0.45), (0.45, 0.55), (0.55, 0.55), (0.55, 0.45)];
    let hole_points: Vec<Vec<Point3>> = (0..4)
        .map(|k| rim(hole_corners[k], hole_corners[(k + 1) % 4]))
        .collect();
    let mut wire_of = |loops: &[Vec<Point3>]| {
        let mut vertices = Vec::new();
        for points in loops {
            vertices.push(topo.add_vertex(Vertex::new(points[0], 1e-7)));
        }
        let n = vertices.len();
        let edges: Vec<_> = loops
            .iter()
            .enumerate()
            .map(|(k, points)| {
                let curve = interpolate(points, 3).unwrap();
                topo.add_edge(Edge::new(
                    vertices[k],
                    vertices[(k + 1) % n],
                    EdgeCurve::NurbsCurve(curve),
                ))
            })
            .collect();
        topo.add_wire(
            Wire::new(
                edges
                    .into_iter()
                    .map(|e| OrientedEdge::new(e, true))
                    .collect(),
                true,
            )
            .unwrap(),
        )
    };
    let outer = wire_of(&outer_points);
    let hole = wire_of(&hole_points);
    let face = topo.add_face(Face::new(
        outer,
        vec![hole],
        FaceSurface::Nurbs(surface.clone()),
    ));
    let tolerance = 0.06;
    let result = fair_surface(&mut topo, face, tolerance).unwrap();
    let after = nurbs_of(&topo, result);

    let wire_moved = |points: &[Vec<Point3>], corners: &[(f64, f64); 4]| -> f64 {
        let mut worst = 0.0_f64;
        for k in 0..4 {
            let (from, to) = (corners[k], corners[(k + 1) % 4]);
            for i in 0..points[k].len() {
                let t = i as f64 / (points[k].len() - 1) as f64;
                let (u, v) = (from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t);
                worst = worst.max((after.evaluate(u, v) - surface.evaluate(u, v)).length());
            }
        }
        worst
    };
    let outer_moved = wire_moved(&outer_points, &corners);
    let hole_moved = wire_moved(&hole_points, &hole_corners);
    assert!(outer_moved < 1e-9, "{outer_moved}");
    assert!(hole_moved < 1e-9, "{hole_moved}");
    let moved = farthest(&surface, &after, 200);
    println!(
        "fair trimmed: moved {moved:.5}, outer wire {outer_moved:.2e}, hole {hole_moved:.2e}, \
         energy {:.3} -> {:.3}",
        bending_energy(&surface),
        bending_energy(&after)
    );
    assert!(moved <= tolerance);
    assert!(moved > 2e-4, "{moved}");
    assert!(bending_energy(&after) < bending_energy(&surface));

    let result_face = topo.face(result).unwrap();
    assert_eq!(result_face.inner_wires().len(), 1);
    assert_ne!(result_face.outer_wire(), outer);

    // Outside the outer wire nothing was spent.
    let out = (after.evaluate(0.02, 0.5) - surface.evaluate(0.02, 0.5)).length();
    assert!(out < 1e-9, "{out}");
    let inner = (after.evaluate(0.5, 0.5) - surface.evaluate(0.5, 0.5)).length();
    assert!(inner < 1e-9, "{inner}");
}

#[test]
fn an_analytic_face_comes_back_as_it_was() {
    let mut topo = Topology::new();
    for kind in [
        "plane",
        "cylinder across",
        "cone rim",
        "sphere meridian",
        "torus tube",
    ] {
        let source = sheet(&mut topo, kind, Pose::turned());
        let result = fair_surface(&mut topo, source.face, 0.1).unwrap();
        assert_ne!(result, source.face, "{kind}");
        assert_eq!(
            topo.face(result).unwrap().surface().type_tag(),
            topo.face(source.face).unwrap().surface().type_tag(),
            "{kind}"
        );
    }
}

#[test]
fn a_bad_tolerance_is_refused() {
    let mut topo = Topology::new();
    let face = cubic_sheet(&mut topo, Pose::upright(), 6, bumpy(0.0));
    for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let result = fair_surface(&mut topo, face, tolerance);
        assert!(
            matches!(result, Err(OperationsError::InvalidInput { .. })),
            "{tolerance}"
        );
    }
}

/// A patch of one span has one wavelength to its name, so it stays as it is.
#[test]
fn a_small_net_is_left_within_tolerance() {
    let mut topo = Topology::new();
    let height = |i: usize, j: usize, _: f64, _: f64| {
        if (i, j) == (1, 1) { 2.0 } else { 0.0 }
    };
    let face = cubic_sheet(&mut topo, Pose::upright(), 4, height);
    let before = nurbs_of(&topo, face);
    let result = fair_surface(&mut topo, face, 0.3).unwrap();
    let after = nurbs_of(&topo, result);
    assert!(bending_energy(&after) <= bending_energy(&before) * (1.0 + 1e-9));
    assert!(farthest(&before, &after, 100) <= 0.3);
    assert!(boundary_moved(&before, &after) < 1e-12);
}

/// One point pulled out of a fine net is a bump, however lonely.
#[test]
fn one_pulled_point_is_a_bump() {
    let mut topo = Topology::new();
    let height = |i: usize, j: usize, _: f64, _: f64| {
        if (i, j) == (8, 12) { 1.0 } else { 0.0 }
    };
    let face = cubic_sheet(&mut topo, Pose::turned(), 21, height);
    let before = nurbs_of(&topo, face);
    let tolerance = 0.5;
    let result = fair_surface(&mut topo, face, tolerance).unwrap();
    let after = nurbs_of(&topo, result);
    let (energy_before, energy_after) = (bending_energy(&before), bending_energy(&after));
    let moved = farthest(&before, &after, 300);
    println!("one pulled point: energy {energy_before:.3} -> {energy_after:.3}, moved {moved:.3}");
    assert!(
        energy_after < 0.5 * energy_before,
        "{energy_after} from {energy_before}"
    );
    assert!(moved <= tolerance, "{moved}");
    assert!(boundary_moved(&before, &after) < 1e-12);
}

/// A plane with its net bunched up unevenly stays that plane: the fairing moves
/// along the normal, and an uneven net has nothing to give there.
#[test]
fn an_unevenly_netted_plane_stays_flat() {
    let mut topo = Topology::new();
    let n = 12;
    let warp = |t: f64| t + 0.08 * (t * std::f64::consts::TAU).sin();
    let net: Vec<Vec<Point3>> = (0..n)
        .map(|i| {
            let x = 10.0 * warp(i as f64 / (n - 1) as f64);
            (0..n)
                .map(|j| Point3::new(x, 10.0 * warp(j as f64 / (n - 1) as f64), 1.5))
                .collect()
        })
        .collect();
    let weights = vec![vec![1.0; n]; n];
    let knots = uniform_knots(n, 3);
    let mut b = Builder {
        topo: &mut topo,
        pose: Pose::upright(),
    };
    let (face, _) = net_sheet(&mut b, (3, 3), (knots.clone(), knots), &net, &weights);
    let before = nurbs_of(&topo, face);
    let result = fair_surface(&mut topo, face, 0.5).unwrap();
    let after = nurbs_of(&topo, result);
    assert!(farthest(&before, &after, 60) < 1e-9);
}

/// A tube of a spline, closed across `u`: its seam is held, and it stays shut.
#[test]
fn a_closed_tube_keeps_its_seam() {
    let around = 16;
    let along = 12;
    let net: Vec<Vec<Point3>> = (0..=around)
        .map(|i| {
            let a = std::f64::consts::TAU * f64::from(i % around) / f64::from(around);
            (0..along)
                .map(|j| {
                    let bump = if (i % around + j) % 2 == 0 {
                        0.04
                    } else {
                        -0.04
                    };
                    let inside = j > 0 && j < along - 1 && i % around != 0;
                    let r = 5.0 + if inside { bump } else { 0.0 };
                    Point3::new(
                        r * a.cos(),
                        r * a.sin(),
                        12.0 * f64::from(j) / f64::from(along - 1),
                    )
                })
                .collect()
        })
        .collect();
    let weights = vec![vec![1.0; along as usize]; (around + 1) as usize];
    let knots_u = uniform_knots((around + 1) as usize, 3);
    let knots_v = uniform_knots(along as usize, 3);
    let surface = NurbsSurface::new(3, 3, knots_u, knots_v, net, weights).unwrap();
    assert!(surface.is_periodic_u());

    let mut topo = Topology::new();
    let (start, top) = (surface.evaluate(0.0, 0.0), surface.evaluate(0.0, 1.0));
    let row = |v: f64| -> Vec<Point3> {
        (0..=120)
            .map(|k| surface.evaluate(f64::from(k) / 120.0, v))
            .collect()
    };
    let low = topo.add_vertex(Vertex::new(start, 1e-7));
    let high = topo.add_vertex(Vertex::new(top, 1e-7));
    let rim_low = topo.add_edge(Edge::new(
        low,
        low,
        EdgeCurve::NurbsCurve(interpolate(&row(0.0), 3).unwrap()),
    ));
    let rim_high = topo.add_edge(Edge::new(
        high,
        high,
        EdgeCurve::NurbsCurve(interpolate(&row(1.0), 3).unwrap()),
    ));
    let seam = topo.add_edge(Edge::new(low, high, EdgeCurve::Line));
    let wire = topo.add_wire(
        Wire::new(
            vec![
                OrientedEdge::new(rim_low, true),
                OrientedEdge::new(seam, true),
                OrientedEdge::new(rim_high, false),
                OrientedEdge::new(seam, false),
            ],
            true,
        )
        .unwrap(),
    );
    let face = topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Nurbs(surface.clone()),
    ));
    let tolerance = 0.1;
    let result = fair_surface(&mut topo, face, tolerance).unwrap();
    let after = nurbs_of(&topo, result);
    for k in 0..=50 {
        let v = f64::from(k) / 50.0;
        let shut = (after.evaluate(0.0, v) - after.evaluate(1.0, v)).length();
        assert!(shut < 1e-12, "the seam opens by {shut}");
    }
    let moved = farthest(&surface, &after, 200);
    let (before_energy, after_energy) = (bending_energy(&surface), bending_energy(&after));
    println!("fair tube: energy {before_energy:.4} -> {after_energy:.4}, moved {moved:.5}");
    assert!(moved <= tolerance && moved > 1e-3, "{moved}");
    assert!(after_energy < before_energy);
}

/// Size does not change what counts as a bump: bending energy has no units, so
/// the same surface at another size comes out with the same share of it.
#[test]
fn fairing_does_not_care_about_size() {
    let run = |size: f64| {
        let mut topo = Topology::new();
        let pose = Pose::turned().then(Pose::scaled(size));
        let face = cubic_sheet(&mut topo, pose, 15, bumpy(0.1));
        let before = nurbs_of(&topo, face);
        let result = fair_surface(&mut topo, face, 0.12 * size).unwrap();
        let after = nurbs_of(&topo, result);
        (
            bending_energy(&after) / bending_energy(&before),
            farthest(&before, &after, 60) / size,
        )
    };
    let (share, moved) = run(1.0);
    for size in [1e-3, 1e3] {
        let (other_share, other_moved) = run(size);
        assert!(
            (other_share - share).abs() < 1e-6,
            "{size}: {other_share} {share}"
        );
        assert!(
            (other_moved - moved).abs() < 1e-6,
            "{size}: {other_moved} {moved}"
        );
    }
}

/// The measure reads what it should: a half-cylinder of radius `r` and length
/// `h` holds `pi h / r` of it.
#[test]
fn the_energy_measure_reads_a_known_shape() {
    let r = 5.0;
    let columns = [
        (Point3::new(r, 0.0, 0.0), 1.0),
        (Point3::new(r, 0.0, r), std::f64::consts::FRAC_1_SQRT_2),
        (Point3::new(0.0, 0.0, r), 1.0),
    ];
    let net: Vec<Vec<Point3>> = columns
        .iter()
        .map(|(p, _)| {
            [0.0, 10.0]
                .iter()
                .map(|&y| Point3::new(p.x(), y, p.z()))
                .collect()
        })
        .collect();
    let weights: Vec<Vec<f64>> = columns.iter().map(|(_, w)| vec![*w; 2]).collect();
    let surface = NurbsSurface::new(
        2,
        1,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        net,
        weights,
    )
    .unwrap();
    let want = std::f64::consts::FRAC_PI_2 * r * 10.0 / (r * r);
    let got = bending_energy(&surface);
    assert!((got - want).abs() < 1e-5 * want, "{got} against {want}");
}
