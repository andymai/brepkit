//! Lofts along guide rails and under end conditions: each result is valid,
//! meshes closed, reads the same volume exactly and from its mesh, and holds
//! its closed form where there is one, in three poses. A rail's samples lie on
//! the surface, a condition's derivative is read off the surface by finite
//! differences along the whole profile, and a boolean takes the result on.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_check::properties::exact_solid_volume;
use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::projection::project_point_to_surface;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::OperationsError;
use brepkit_operations::loft::{
    LoftEnd, LoftOptions, LoftRail, loft, loft_sides, loft_with_options,
};
use brepkit_operations::tessellate::{TriangleMesh, is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const TOL: f64 = 1e-7;

fn poses() -> [(&'static str, Mat4); 3] {
    [
        ("upright", Mat4::identity()),
        (
            "turned",
            Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6),
        ),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ]
}

fn dir(pose: Mat4, v: Vec3) -> Vec3 {
    pose.mul_point(Point3::new(v.x(), v.y(), v.z())) - pose.mul_point(Point3::new(0.0, 0.0, 0.0))
}

fn p(x: f64, y: f64, z: f64) -> Point3 {
    Point3::new(x, y, z)
}

/// A planar face on `corners` in order, carried by `pose`.
fn polygon(topo: &mut Topology, pose: Mat4, corners: &[Point3]) -> FaceId {
    let points: Vec<Point3> = corners.iter().map(|&c| pose.mul_point(c)).collect();
    let verts: Vec<_> = points
        .iter()
        .map(|&c| topo.add_vertex(Vertex::new(c, TOL)))
        .collect();
    let n = verts.len();
    let edges = (0..n)
        .map(|i| {
            let e = topo.add_edge(Edge::new(verts[i], verts[(i + 1) % n], EdgeCurve::Line));
            OrientedEdge::new(e, true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let normal = (points[1] - points[0])
        .cross(points[2] - points[0])
        .normalize()
        .unwrap();
    let d = normal.dot(points[0] - p(0.0, 0.0, 0.0));
    topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d }))
}

/// A square of half-size `h` about the z axis at height `z`.
fn square(topo: &mut Topology, pose: Mat4, h: f64, z: f64) -> FaceId {
    polygon(
        topo,
        pose,
        &[p(-h, -h, z), p(h, -h, z), p(h, h, z), p(-h, h, z)],
    )
}

/// A disc of radius `r` about `centre`, its normal z, carried by `pose`.
fn disc(topo: &mut Topology, pose: Mat4, centre: Point3, r: f64) -> FaceId {
    let normal = dir(pose, Vec3::new(0.0, 0.0, 1.0));
    let at = pose.mul_point(centre);
    let rim = brepkit_topology::builder::make_circle_edge(topo, at, normal, r, TOL).unwrap();
    let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
    let d = normal.dot(at - p(0.0, 0.0, 0.0));
    topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d }))
}

/// The Bezier curve on `points`, carried by `pose`.
fn bezier(pose: Mat4, points: &[Point3]) -> NurbsCurve {
    let degree = points.len() - 1;
    let mut knots = vec![0.0; degree + 1];
    knots.extend(vec![1.0; degree + 1]);
    NurbsCurve::new(
        degree,
        knots,
        points.iter().map(|&c| pose.mul_point(c)).collect(),
        vec![1.0; points.len()],
    )
    .unwrap()
}

fn nurbs_edge(topo: &mut Topology, curve: NurbsCurve) -> EdgeId {
    let (t0, t1) = curve.domain();
    let a = topo.add_vertex(Vertex::new(curve.evaluate(t0), TOL));
    let b = topo.add_vertex(Vertex::new(curve.evaluate(t1), TOL));
    topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(curve)))
}

fn line_edge(topo: &mut Topology, pose: Mat4, a: Point3, b: Point3) -> EdgeId {
    let a = topo.add_vertex(Vertex::new(pose.mul_point(a), TOL));
    let b = topo.add_vertex(Vertex::new(pose.mul_point(b), TOL));
    topo.add_edge(Edge::new(a, b, EdgeCurve::Line))
}

fn edge_rails(topo: &mut Topology, curves: &[NurbsCurve]) -> Vec<LoftRail> {
    curves
        .iter()
        .map(|c| LoftRail::Edge(nurbs_edge(topo, c.clone())))
        .collect()
}

/// The four square corners' signs, in the order [`square`] runs them.
const CORNERS: [(f64, f64); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];

fn signed_mesh_volume(mesh: &TriangleMesh) -> f64 {
    let origin = mesh.positions[0];
    mesh.indices
        .chunks_exact(3)
        .map(|t| {
            let (a, b, c) = (
                mesh.positions[t[0] as usize] - origin,
                mesh.positions[t[1] as usize] - origin,
                mesh.positions[t[2] as usize] - origin,
            );
            a.dot(b.cross(c)) / 6.0
        })
        .sum()
}

/// The solid is valid and meshes closed, and its exact volume agrees with the
/// volume of its mesh, which an outward-facing solid reads positive. The exact
/// volume.
fn assert_solid(topo: &Topology, solid: SolidId, case: &str) -> f64 {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{case}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, solid, 0.002).unwrap();
    assert!(is_watertight(&mesh), "{case}: the mesh is open");
    let exact = exact_solid_volume(topo, solid).unwrap();
    assert!(exact.is_some(), "{case}: no exact volume");
    let exact = exact.unwrap();
    let meshed = signed_mesh_volume(&mesh);
    assert!(
        ((exact - meshed) / exact).abs() < 1e-3,
        "{case}: exact volume {exact}, meshed {meshed}"
    );
    exact
}

fn assert_close(got: f64, want: f64, relative: f64, case: &str) {
    assert!(
        ((got - want) / want).abs() < relative,
        "{case}: {got}, want {want}"
    );
}

// ── Measures ────────────────────────────────────────────────────────────────

/// Composite Simpson over `[a, b]`, exact on the polynomials used here.
fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let n = 2000;
    let h = (b - a) / f64::from(n);
    let mut sum = f(a) + f(b);
    for k in 1..n {
        let weight = if k % 2 == 1 { 4.0 } else { 2.0 };
        sum += weight * f(a + h * f64::from(k));
    }
    sum * h / 3.0
}

/// The volume with a square section of half-size `a(v)` at height `z(v)`.
fn square_stack_volume(a: impl Fn(f64) -> f64, dz: impl Fn(f64) -> f64) -> f64 {
    integrate(|v| 4.0 * a(v) * a(v) * dz(v), 0.0, 1.0)
}

/// The surfaces of the solid's faces that are NURBS.
fn nurbs_sides(topo: &Topology, solid: SolidId) -> Vec<NurbsSurface> {
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .filter_map(|&f| match topo.face(f).unwrap().surface() {
            FaceSurface::Nurbs(s) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

/// How far `q` lies from the nearest of the solid's faces' surfaces.
fn from_surfaces(topo: &Topology, solid: SolidId, q: Point3) -> f64 {
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .map(|&f| match topo.face(f).unwrap().surface() {
            FaceSurface::Plane { normal, d } => {
                (normal.dot(q - p(0.0, 0.0, 0.0)) - d).abs() / normal.length()
            }
            FaceSurface::Nurbs(s) => project_point_to_surface(s, q, 1e-11).unwrap().distance,
            other => {
                let (u, v) = other.project_point(q).unwrap();
                (other.evaluate(u, v).unwrap() - q).length()
            }
        })
        .fold(f64::INFINITY, f64::min)
}

/// How far `q` lies from the nearest of the solid's edges: a line's segment, a
/// spline's curve, and an arc's curve sampled finely, which reads no nearer
/// than it lies.
fn from_edges(topo: &Topology, solid: SolidId, q: Point3) -> f64 {
    solid_edges(topo, solid)
        .unwrap()
        .iter()
        .map(|&e| {
            let edge = topo.edge(e).unwrap();
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            match edge.curve() {
                EdgeCurve::Line => {
                    let w = ((q - a).dot(b - a) / (b - a).dot(b - a)).clamp(0.0, 1.0);
                    (q - a - (b - a) * w).length()
                }
                EdgeCurve::NurbsCurve(n) => {
                    brepkit_math::nurbs::projection::project_point_to_curve(n, q, 1e-12)
                        .unwrap()
                        .distance
                }
                other => {
                    let (t0, t1) = other.domain_with_endpoints(a, b);
                    (0..=4000)
                        .map(|k| {
                            let t = (t1 - t0).mul_add(f64::from(k) / 4000.0, t0);
                            (other.evaluate_with_endpoints(t, a, b) - q).length()
                        })
                        .fold(f64::INFINITY, f64::min)
                }
            }
        })
        .fold(f64::INFINITY, f64::min)
}

/// How far `q` lies from the triangles of `mesh`.
fn from_mesh(mesh: &TriangleMesh, q: Point3) -> f64 {
    let on_triangle = |a: Point3, b: Point3, c: Point3| -> f64 {
        let (ab, ac, aq) = (b - a, c - a, q - a);
        let (d1, d2) = (ab.dot(aq), ac.dot(aq));
        let (aa, bb, ab_ac) = (ab.dot(ab), ac.dot(ac), ab.dot(ac));
        let det = aa * bb - ab_ac * ab_ac;
        if det > 0.0 {
            let s = (bb * d1 - ab_ac * d2) / det;
            let t = (aa * d2 - ab_ac * d1) / det;
            if s >= 0.0 && t >= 0.0 && s + t <= 1.0 {
                return (aq - ab * s - ac * t).length();
            }
        }
        let on_segment = |x: Point3, y: Point3| {
            let w = ((q - x).dot(y - x) / (y - x).dot(y - x).max(1e-300)).clamp(0.0, 1.0);
            (q - x - (y - x) * w).length()
        };
        on_segment(a, b).min(on_segment(b, c)).min(on_segment(c, a))
    };
    mesh.indices
        .chunks_exact(3)
        .map(|t| {
            on_triangle(
                mesh.positions[t[0] as usize],
                mesh.positions[t[1] as usize],
                mesh.positions[t[2] as usize],
            )
        })
        .fold(f64::INFINITY, f64::min)
}

/// Every sample lies on the solid: within `reach` of a face's surface, and
/// within a few deflections of its mesh, which reads the trimmed solid rather
/// than the faces' whole surfaces.
fn assert_on_solid(topo: &Topology, solid: SolidId, samples: &[Point3], reach: f64, case: &str) {
    let worst = samples
        .iter()
        .map(|&q| from_surfaces(topo, solid, q))
        .fold(0.0, f64::max);
    assert!(
        worst <= reach,
        "{case}: a sample lies {worst} off a surface"
    );
    let mesh = tessellate_solid(topo, solid, 0.1).unwrap();
    for &q in samples.iter().step_by(4) {
        let off = from_mesh(&mesh, q);
        assert!(off < 0.3, "{case}: {q:?} lies {off} from the mesh");
    }
}

fn curve_samples(curve: &NurbsCurve, count: u32) -> Vec<Point3> {
    let (t0, t1) = curve.domain();
    (0..=count)
        .map(|k| curve.evaluate((t1 - t0).mul_add(f64::from(k) / f64::from(count), t0)))
        .collect()
}

fn segment_samples(a: Point3, b: Point3, count: u32) -> Vec<Point3> {
    (0..=count)
        .map(|k| a + (b - a) * (f64::from(k) / f64::from(count)))
        .collect()
}

/// A square's outline at height `z`, carried by `pose`.
fn square_outline(pose: Mat4, h: f64, z: f64) -> Vec<Point3> {
    (0..4)
        .flat_map(|i| {
            let (a, b) = (CORNERS[i], CORNERS[(i + 1) % 4]);
            segment_samples(
                pose.mul_point(p(a.0 * h, a.1 * h, z)),
                pose.mul_point(p(b.0 * h, b.1 * h, z)),
                16,
            )
        })
        .collect()
}

/// A circle's outline, carried by `pose`.
fn circle_outline(pose: Mat4, centre: Point3, r: f64) -> Vec<Point3> {
    (0..64)
        .map(|k| {
            let a = f64::from(k) * std::f64::consts::TAU / 64.0;
            pose.mul_point(p(
                centre.x() + r * a.cos(),
                centre.y() + r * a.sin(),
                centre.z(),
            ))
        })
        .collect()
}

/// `surface` differentiated across its `v` from three points inside it, at `u`
/// and the end of `v` that lies on the profile. `None` if neither does. The
/// derivative runs the way `v` does.
fn across(
    surface: &NurbsSurface,
    on_profile: &dyn Fn(Point3) -> bool,
    u: f64,
) -> Option<(Point3, Vec3)> {
    let (v0, v1) = surface.domain_v();
    let h = (v1 - v0) * 1e-4;
    let at = |v: f64| surface.evaluate(u, v);
    if on_profile(at(v0)) {
        let (a, b, c) = (at(v0), at(v0 + h), at(v0 + 2.0 * h));
        return Some((a, ((b - a) * 4.0 - (c - a)) * (1.0 / (2.0 * h))));
    }
    if on_profile(at(v1)) {
        let (a, b, c) = (at(v1), at(v1 - h), at(v1 - 2.0 * h));
        return Some((a, ((b - a) * 4.0 - (c - a)) * (-1.0 / (2.0 * h))));
    }
    None
}

/// The derivative across the profile of every curved side at 41 places along
/// it: the point there and the derivative.
fn derivatives_along(
    topo: &Topology,
    solid: SolidId,
    on_profile: &dyn Fn(Point3) -> bool,
) -> Vec<(Point3, Vec3)> {
    let mut out = Vec::new();
    for side in nurbs_sides(topo, solid) {
        let (u0, u1) = side.domain_u();
        for k in 0..=40 {
            let u = (u1 - u0).mul_add(f64::from(k) / 40.0, u0);
            out.push(across(&side, on_profile, u).expect("a side reaches the profile"));
        }
    }
    assert!(!out.is_empty(), "no curved side to measure");
    out
}

/// How far the derivatives along `solid`'s profile lean off the unit `want`,
/// in radians, at their worst.
fn worst_angle_off(pairs: &[(Point3, Vec3)], want: impl Fn(Point3, Vec3) -> Vec3) -> f64 {
    pairs
        .iter()
        .map(|&(q, t)| {
            let unit = t.normalize().expect("a derivative");
            unit.cross(want(q, t)).length().atan2(unit.dot(want(q, t)))
        })
        .fold(0.0, f64::max)
}

/// The test for lying on the plane through `on` square to `normal`.
fn on_plane(on: Point3, normal: Vec3) -> impl Fn(Point3) -> bool {
    move |q: Point3| (q - on).dot(normal).abs() < 1e-6
}

fn classify(topo: &Topology, solid: SolidId, q: Point3) -> PointClassification {
    classify_point(topo, solid, q, &ClassifyOptions::default()).unwrap()
}

fn assert_refused(result: Result<SolidId, OperationsError>, words: &str) {
    assert!(
        matches!(&result, Err(OperationsError::InvalidInput { reason }) if reason.contains(words)),
        "not refused for {words}: {result:?}"
    );
}

// ── Rails ───────────────────────────────────────────────────────────────────

/// A loft of two squares along four straight rails down their corners is the
/// loft without them: the frustum's volume, in six planes and twelve lines.
#[test]
fn straight_rails_down_a_frustums_corners_leave_it_as_it_is() {
    let (h, low, high) = (10.0, 10.0, 5.0);
    let want = h / 3.0 * (400.0 + 100.0 + 200.0);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let a = square(&mut topo, pose, low, 0.0);
        let b = square(&mut topo, pose, high, h);
        let rails = CORNERS
            .map(|(x, y)| {
                LoftRail::Edge(line_edge(
                    &mut topo,
                    pose,
                    p(x * low, y * low, 0.0),
                    p(x * high, y * high, h),
                ))
            })
            .to_vec();
        let options = LoftOptions {
            rails,
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[a, b], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert_close(exact, want, 1e-12, name);

        let plain = loft(&mut topo, &[a, b]).unwrap();
        let census = |topo: &Topology, s: SolidId| {
            let faces = solid_faces(topo, s).unwrap();
            let planes = faces
                .iter()
                .filter(|&&f| topo.face(f).unwrap().surface().is_planar())
                .count();
            let edges = solid_edges(topo, s).unwrap();
            let lines = edges
                .iter()
                .filter(|&&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Line))
                .count();
            (faces.len(), planes, edges.len(), lines)
        };
        assert_eq!(census(&topo, solid), census(&topo, plain), "{name}");
        assert_eq!(census(&topo, solid), (6, 6, 12, 12), "{name}");
    }
}

/// The half-size along a quadratic rail from `a0` through the control `c` to
/// `a1`.
fn quadratic(a0: f64, c: f64, a1: f64) -> impl Fn(f64) -> f64 {
    move |v| (1.0 - v) * (1.0 - v) * a0 + 2.0 * v * (1.0 - v) * c + v * v * a1
}

/// Four rails bowing out between two squares: each height is a square of the
/// rails' half-size, so the solid holds their integral and a point between the
/// straight frustum and the rails is inside only because of them.
#[test]
fn rails_that_bow_out_carve_the_sections_they_ask_for() {
    let (h, low, mid, high) = (10.0, 6.0, 14.0, 4.0);
    let a = quadratic(low, mid, high);
    let want = square_stack_volume(&a, |_| h);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let (first, last) = (
            square(&mut topo, pose, low, 0.0),
            square(&mut topo, pose, high, h),
        );
        let curves = CORNERS.map(|(x, y)| {
            bezier(
                pose,
                &[
                    p(x * low, y * low, 0.0),
                    p(x * mid, y * mid, h / 2.0),
                    p(x * high, y * high, h),
                ],
            )
        });
        let options = LoftOptions {
            rails: edge_rails(&mut topo, &curves),
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert_close(exact, want, 1e-9, name);
        for curve in &curves {
            assert_on_solid(&topo, solid, &curve_samples(curve, 40), 1e-9, name);
        }
        assert_on_solid(&topo, solid, &square_outline(pose, low, 0.0), 1e-9, name);
        assert_on_solid(&topo, solid, &square_outline(pose, high, h), 1e-9, name);

        let half = a(0.5);
        let straight = f64::midpoint(low, high);
        assert!(straight < 0.9 * half, "the probe needs the rails to matter");
        let on_diagonal = |r: f64, z: f64| pose.mul_point(p(r, r, z));
        for (q, want_class) in [
            (on_diagonal(0.0, h / 2.0), PointClassification::Inside),
            (
                on_diagonal(f64::midpoint(straight, half), h / 2.0),
                PointClassification::Inside,
            ),
            (
                on_diagonal(1.02 * half, h / 2.0),
                PointClassification::Outside,
            ),
            (on_diagonal(0.0, h + 0.5), PointClassification::Outside),
            (on_diagonal(0.0, -0.5), PointClassification::Outside),
        ] {
            assert_eq!(classify(&topo, solid, q), want_class, "{name}: at {q:?}");
        }
    }
}

/// Two quadratic Beziers joined at parameter `at`, into one curve with a full
/// knot there.
fn joined(a: &NurbsCurve, b: &NurbsCurve, at: f64) -> NurbsCurve {
    let mut points = a.control_points().to_vec();
    points.extend_from_slice(&b.control_points()[1..]);
    NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, at, at, 1.0, 1.0, 1.0],
        points,
        vec![1.0; 5],
    )
    .unwrap()
}

/// Four rails each wandering its own way through three squares, meeting the
/// middle square at different fractions of their parameters. The surface passes
/// through each of them and through all three squares.
#[test]
fn rails_that_wander_pass_through_every_profile_and_rail() {
    let levels = [(8.0, 0.0), (5.0, 7.0), (7.0, 12.0)];
    let wobble = [(1.5, 0.5), (-1.0, 2.0), (0.5, -2.0), (-2.0, -0.5)];
    let fractions = [0.4, 0.5, 0.6, 0.45];
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let faces: Vec<FaceId> = levels
            .iter()
            .map(|&(h, z)| square(&mut topo, pose, h, z))
            .collect();
        let mut curves = Vec::new();
        for i in 0..4 {
            let (x, y) = CORNERS[i];
            let at = |&(h, z): &(f64, f64)| p(x * h, y * h, z);
            let (wx, wy) = wobble[i];
            let lower = bezier(
                pose,
                &[
                    at(&levels[0]),
                    p(x * 8.0 + wx, y * 8.0 + wy, 2.5),
                    at(&levels[1]),
                ],
            );
            let upper = bezier(
                pose,
                &[
                    at(&levels[1]),
                    p(x * 5.0 - wx, y * 5.0 - wy, 9.5),
                    at(&levels[2]),
                ],
            );
            curves.push(joined(&lower, &upper, fractions[i]));
        }
        let options = LoftOptions {
            rails: edge_rails(&mut topo, &curves),
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &faces, &options).unwrap();
        assert_solid(&topo, solid, name);
        for curve in &curves {
            assert_on_solid(&topo, solid, &curve_samples(curve, 60), 1e-8, name);
        }
        for &(h, z) in &levels {
            assert_on_solid(&topo, solid, &square_outline(pose, h, z), 1e-8, name);
        }
        assert_eq!(
            classify(&topo, solid, pose.mul_point(p(0.0, 0.0, 6.0))),
            PointClassification::Inside,
            "{name}"
        );
        assert_eq!(
            classify(&topo, solid, pose.mul_point(p(9.0, 9.0, 6.0))),
            PointClassification::Outside,
            "{name}"
        );
    }
}

/// One rail, or two across a diagonal, on a square: the other vertices take the
/// rails' deviation by distance round the square.
#[test]
fn fewer_rails_than_vertices_carry_the_others_with_them() {
    let (h, low, high) = (10.0, 6.0, 3.0);
    for (name, pose) in poses() {
        for count in [1_usize, 2] {
            let case = format!("{name}, {count} rails");
            let mut topo = Topology::new();
            let (first, last) = (
                square(&mut topo, pose, low, 0.0),
                square(&mut topo, pose, high, h),
            );
            let curves: Vec<NurbsCurve> = (0..count)
                .map(|i| {
                    let (x, y) = CORNERS[i * 2];
                    bezier(
                        pose,
                        &[
                            p(x * low, y * low, 0.0),
                            p(x * low + 3.0, y * low - 2.0, h / 2.0),
                            p(x * high, y * high, h),
                        ],
                    )
                })
                .collect();
            let options = LoftOptions {
                rails: edge_rails(&mut topo, &curves),
                ..LoftOptions::default()
            };
            let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
            let exact = assert_solid(&topo, solid, &case);
            if count == 1 {
                // One rail moves every vertex by one deviation: each section
                // only slides, so it holds the frustum's volume.
                assert_close(exact, h / 3.0 * (144.0 + 36.0 + 72.0), 1e-9, &case);
            }
            for curve in &curves {
                assert_on_solid(&topo, solid, &curve_samples(curve, 40), 1e-9, &case);
            }
            assert_on_solid(&topo, solid, &square_outline(pose, low, 0.0), 1e-9, &case);
            assert_on_solid(&topo, solid, &square_outline(pose, high, h), 1e-9, &case);
        }
    }
}

/// A rail may be a wire of edges with its edges stored either way and the wire
/// run from either end, or one edge stored from the last profile to the first;
/// the loft follows each. Straight runs through a square of half-size 9 at
/// z = 5 make two frustums, though no section lies there to say so.
#[test]
fn a_rail_is_followed_whichever_way_it_is_stored() {
    let (h, low, mid, high, z) = (10.0, 6.0, 9.0, 4.0, 5.0);
    let want = 5.0 / 3.0 * (144.0 + 324.0 + 216.0) + 5.0 / 3.0 * (324.0 + 64.0 + 144.0);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let (first, last) = (
            square(&mut topo, pose, low, 0.0),
            square(&mut topo, pose, high, h),
        );
        let mut rails = Vec::new();
        for (i, &(x, y)) in CORNERS.iter().enumerate() {
            let (a, b, c) = (
                pose.mul_point(p(x * low, y * low, 0.0)),
                pose.mul_point(p(x * mid, y * mid, z)),
                pose.mul_point(p(x * high, y * high, h)),
            );
            let [va, vb, vc] = [a, b, c].map(|q| topo.add_vertex(Vertex::new(q, TOL)));
            let line = |topo: &mut Topology, s, e| topo.add_edge(Edge::new(s, e, EdgeCurve::Line));
            let chain = |topo: &mut Topology, runs: Vec<OrientedEdge>| {
                LoftRail::Wire(topo.add_wire(Wire::new(runs, false).unwrap()))
            };
            rails.push(match i {
                0 => {
                    let (lower, upper) = (line(&mut topo, va, vb), line(&mut topo, vb, vc));
                    let runs = vec![
                        OrientedEdge::new(lower, true),
                        OrientedEdge::new(upper, true),
                    ];
                    chain(&mut topo, runs)
                }
                1 => {
                    let (lower, upper) = (line(&mut topo, vb, va), line(&mut topo, vc, vb));
                    let runs = vec![
                        OrientedEdge::new(upper, true),
                        OrientedEdge::new(lower, true),
                    ];
                    chain(&mut topo, runs)
                }
                2 => {
                    let (lower, upper) = (line(&mut topo, vb, va), line(&mut topo, vc, vb));
                    let runs = vec![
                        OrientedEdge::new(lower, false),
                        OrientedEdge::new(upper, false),
                    ];
                    chain(&mut topo, runs)
                }
                _ => {
                    // Its knot by length, as a wire's edges share its parameter.
                    let (near, far) = ((b - a).length(), (c - b).length());
                    let curve = NurbsCurve::new(
                        1,
                        vec![0.0, 0.0, far / (near + far), 1.0, 1.0],
                        vec![c, b, a],
                        vec![1.0; 3],
                    )
                    .unwrap();
                    LoftRail::Edge(nurbs_edge(&mut topo, curve))
                }
            });
        }
        let options = LoftOptions {
            rails,
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert_close(exact, want, 1e-9, name);
        assert_eq!(
            classify(&topo, solid, pose.mul_point(p(8.5, 8.5, 5.0))),
            PointClassification::Inside,
            "{name}"
        );
    }
}

/// A square's edges cut in two against an octagon's, between four rails that
/// meet at the square's corners and at every second corner of the octagon.
#[test]
fn rails_pair_profiles_of_different_vertex_counts() {
    let (h, low, high) = (10.0, 6.0, 8.0);
    let octagon: Vec<Point3> = (0..8)
        .map(|k| {
            let (x, y) = CORNERS[k / 2];
            let (nx, ny) = CORNERS[(k / 2 + 1) % 4];
            if k % 2 == 0 {
                p(x * high, y * high, h)
            } else {
                p(0.5 * (x + nx) * 1.2 * high, 0.5 * (y + ny) * 1.2 * high, h)
            }
        })
        .collect();
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let first = square(&mut topo, pose, low, 0.0);
        let last = polygon(&mut topo, pose, &octagon);
        let curves = CORNERS.map(|(x, y)| {
            bezier(
                pose,
                &[
                    p(x * low, y * low, 0.0),
                    p(x * 9.0, y * 9.0, h / 2.0),
                    p(x * high, y * high, h),
                ],
            )
        });
        let options = LoftOptions {
            rails: edge_rails(&mut topo, &curves),
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        assert_eq!(solid_faces(&topo, solid).unwrap().len(), 10, "{name}");
        assert_solid(&topo, solid, name);
        for curve in &curves {
            assert_on_solid(&topo, solid, &curve_samples(curve, 40), 1e-9, name);
        }
        assert_on_solid(&topo, solid, &square_outline(pose, low, 0.0), 1e-9, name);
        for (k, &corner) in octagon.iter().enumerate() {
            let next = octagon[(k + 1) % 8];
            let outline = segment_samples(pose.mul_point(corner), pose.mul_point(next), 8);
            assert_on_solid(&topo, solid, &outline, 1e-9, name);
        }
    }
}

/// A disc's rim cut into four arcs, its four quarter points running up four
/// circular rails to a smaller disc: the rails are rational, and so are the
/// rims, so the sum carries the rims' weights through one degree up. Each arc
/// keeps its circle at the caps.
#[test]
fn round_rails_between_round_profiles_pass_through_both() {
    let (r, top, h) = (6.0, 3.0, 8.0);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let low = quartered_disc(&mut topo, pose, r, 0.0);
        let high = quartered_disc(&mut topo, pose, top, h);
        let mut rails = Vec::new();
        let mut samples = Vec::new();
        for k in 0..4 {
            let angle = f64::from(k) * std::f64::consts::FRAC_PI_2;
            let (c, s) = (angle.cos(), angle.sin());
            // A circle through the rail's ends, bowing out by 1.5 at mid height.
            let (a, b) = (p(c * r, s * r, 0.0), p(c * top, s * top, h));
            let middle = p(
                c * (f64::midpoint(r, top) + 1.5),
                s * (f64::midpoint(r, top) + 1.5),
                h / 2.0,
            );
            let arc = circle_through(pose, a, middle, b);
            let (va, vb) = (
                topo.add_vertex(Vertex::new(pose.mul_point(a), TOL)),
                topo.add_vertex(Vertex::new(pose.mul_point(b), TOL)),
            );
            let edge = topo.add_edge(Edge::new(va, vb, EdgeCurve::Circle(arc.clone())));
            rails.push(LoftRail::Edge(edge));
            let (t0, t1) = topo
                .edge(edge)
                .unwrap()
                .curve()
                .domain_with_endpoints(pose.mul_point(a), pose.mul_point(b));
            samples
                .extend((0..=24).map(|i| arc.evaluate((t1 - t0).mul_add(f64::from(i) / 24.0, t0))));
        }
        let options = LoftOptions {
            rails,
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[low, high], &options).unwrap();
        assert_solid(&topo, solid, name);
        assert_on_solid(&topo, solid, &samples, 2e-8, name);
        assert_on_solid(
            &topo,
            solid,
            &circle_outline(pose, p(0.0, 0.0, 0.0), r),
            1e-9,
            name,
        );
        assert_on_solid(
            &topo,
            solid,
            &circle_outline(pose, p(0.0, 0.0, h), top),
            1e-9,
            name,
        );
        let circles = solid_edges(&topo, solid)
            .unwrap()
            .iter()
            .filter(|&&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Circle(_)))
            .count();
        assert_eq!(circles, 8, "{name}: the caps' arcs stay circles");
    }
}

/// The disc of radius `r` about the z axis at height `z`, its rim four quarter
/// arcs.
fn quartered_disc(topo: &mut Topology, pose: Mat4, r: f64, z: f64) -> FaceId {
    let normal = dir(pose, Vec3::new(0.0, 0.0, 1.0));
    let centre = pose.mul_point(p(0.0, 0.0, z));
    let circle = brepkit_math::curves::Circle3D::new(centre, normal, r).unwrap();
    let corners: Vec<_> = (0..4)
        .map(|k| {
            let a = f64::from(k) * std::f64::consts::FRAC_PI_2;
            let q = pose.mul_point(p(r * a.cos(), r * a.sin(), z));
            topo.add_vertex(Vertex::new(q, TOL))
        })
        .collect();
    let mirrored = outward_of(pose).dot(normal) < 0.0;
    let edges = (0..4)
        .map(|k| {
            let (a, b) = (corners[k], corners[(k + 1) % 4]);
            // A circle runs anticlockwise about its normal.
            let e = if mirrored {
                topo.add_edge(Edge::new(b, a, EdgeCurve::Circle(circle.clone())))
            } else {
                topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle.clone())))
            };
            OrientedEdge::new(e, !mirrored)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let d = normal.dot(centre - p(0.0, 0.0, 0.0));
    topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d }))
}

/// The circle through three points, carried by `pose`, running from `a` through
/// `m` to `b`.
fn circle_through(pose: Mat4, a: Point3, m: Point3, b: Point3) -> brepkit_math::curves::Circle3D {
    let (a, m, b) = (pose.mul_point(a), pose.mul_point(m), pose.mul_point(b));
    let (u, v) = (m - a, b - a);
    let normal = u.cross(v);
    let to_centre =
        (v.cross(normal) * u.dot(u) + normal.cross(u) * v.dot(v)) * (0.5 / normal.dot(normal));
    let centre = a + to_centre;
    let radius = (a - centre).length();
    let (x, y) = (a - centre, m - centre);
    // Its normal the way a to m turns: a circle runs anticlockwise about it.
    let way = x.cross(y).normalize().unwrap();
    brepkit_math::curves::Circle3D::new(centre, way, radius).unwrap()
}

/// Quadratic rails through a middle square, each reaching it at its own fraction
/// of its parameter. At one fraction they ask for no crease, so one face runs
/// across the middle square and each height is a square of their half-size. At
/// different fractions the parameter's rate has to jump there, the sides crease
/// along it though every rail runs smoothly through, and they end there, still
/// passing through every rail and profile.
#[test]
fn rails_through_a_middle_profile_crease_it_only_at_different_fractions() {
    let half = quadratic(6.0, 9.0, 4.0);
    let want = integrate(|t| 40.0 * half(t) * half(t), 0.0, 1.0);
    let rail = |pose: Mat4, (x, y): (f64, f64), at: f64| {
        let o = p(0.0, 0.0, 0.0);
        let corner = |h: f64, z: f64| p(x * h, y * h, z) - o;
        let w = 1.0 - at;
        let control =
            (corner(7.0, 5.0) - corner(6.0, 0.0) * (w * w) - corner(4.0, 10.0) * (at * at))
                * (1.0 / (2.0 * at * w));
        bezier(
            pose,
            &[o + corner(6.0, 0.0), o + control, o + corner(4.0, 10.0)],
        )
    };
    for (name, pose) in poses() {
        let up = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let middle = pose.mul_point(p(0.0, 0.0, 5.0));
        for (even, fractions) in [(true, [0.5; 4]), (false, [0.4, 0.5, 0.6, 0.45])] {
            let case = format!("{name}, fractions {fractions:?}");
            let mut topo = Topology::new();
            let faces = [
                square(&mut topo, pose, 6.0, 0.0),
                square(&mut topo, pose, 7.0, 5.0),
                square(&mut topo, pose, 4.0, 10.0),
            ];
            let curves: Vec<NurbsCurve> = (0..4)
                .map(|i| rail(pose, CORNERS[i], fractions[i]))
                .collect();
            let options = LoftOptions {
                rails: edge_rails(&mut topo, &curves),
                ..LoftOptions::default()
            };
            let solid = loft_with_options(&mut topo, &faces, &options).unwrap();
            let exact = assert_solid(&topo, solid, &case);
            for curve in &curves {
                assert_on_solid(&topo, solid, &curve_samples(curve, 40), 1e-9, &case);
            }
            for (h, z) in [(6.0, 0.0), (7.0, 5.0), (4.0, 10.0)] {
                assert_on_solid(&topo, solid, &square_outline(pose, h, z), 1e-9, &case);
            }
            let count = solid_faces(&topo, solid).unwrap().len();
            if even {
                assert_close(exact, want, 1e-9, &case);
                assert_eq!(count, 6, "{case}: one face across the middle profile");
            } else {
                assert_eq!(count, 10, "{case}: faces ending on the middle profile");
                let pairs = derivatives_along(&topo, solid, &on_plane(middle, up));
                let mut crease = 0.0_f64;
                for (i, &(q, t)) in pairs.iter().enumerate() {
                    for &(r, s) in &pairs[i + 1..] {
                        if (q - r).length() < 1e-9 {
                            let (a, b) = (t.normalize().unwrap(), s.normalize().unwrap());
                            crease = crease.max(a.cross(b).length().atan2(a.dot(b)));
                        }
                    }
                }
                assert!(crease > 1e-3, "{case}: a crease of {crease} radians");
            }
        }
    }
}

/// Rails running on past the first and last profiles: the loft takes the
/// stretch between them, the frustum down their corners.
#[test]
fn a_rail_running_past_the_profiles_is_followed_between_them() {
    let (h, low, high) = (10.0, 6.0, 4.0);
    let want = h / 3.0 * (144.0 + 64.0 + 96.0);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let (first, last) = (
            square(&mut topo, pose, low, 0.0),
            square(&mut topo, pose, high, h),
        );
        let curves = CORNERS.map(|(x, y)| {
            let at = |s: f64| {
                let half = (high - low).mul_add(s, low);
                p(x * half, y * half, h * s)
            };
            bezier(pose, &[at(-0.5), at(0.5), at(1.5)])
        });
        let options = LoftOptions {
            rails: edge_rails(&mut topo, &curves),
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert_close(exact, want, 1e-9, name);
        assert_eq!(solid_faces(&topo, solid).unwrap().len(), 6, "{name}");
    }
}

/// Two squares 5e-6 apart at the far end of straight rails 100 long: the nearer
/// meets its rails within a knot's reach of their ends in parameter, but farther
/// from them than a vertex may lie, so it keeps its own place on them.
#[test]
fn profiles_just_short_of_a_rails_end_keep_their_own_places_on_it() {
    let (half, below, end) = (3.0, 99.999_995, 100.0);
    let want = 4.0 * half * half * (end - below);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let (first, last) = (
            square(&mut topo, pose, half, below),
            square(&mut topo, pose, half, end),
        );
        let rails = CORNERS
            .map(|(x, y)| {
                LoftRail::Edge(line_edge(
                    &mut topo,
                    pose,
                    p(x * half, y * half, 0.0),
                    p(x * half, y * half, end),
                ))
            })
            .to_vec();
        let options = LoftOptions {
            rails,
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert_close(exact, want, 1e-6, name);
    }
}

/// Four semicircular rails from one square round to another beside it in its
/// plane, a U-bend: the chord between them lies in both their planes, so only
/// the rails tell which way each square winds. The body is the square revolved
/// half a turn, so it holds Pappus's volume, whichever way the wires run.
#[test]
fn rails_round_a_u_bend_wind_the_far_profile_for_them() {
    let (near, far, half_y) = (5.0, 9.0, 3.0);
    let area = (far - near) * 2.0 * half_y;
    let want = area * std::f64::consts::PI * f64::midpoint(near, far);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let at = |x: f64, y: f64| p(x, y, 0.0);
        let first = polygon(
            &mut topo,
            pose,
            &[
                at(near, -half_y),
                at(far, -half_y),
                at(far, half_y),
                at(near, half_y),
            ],
        );
        let last = polygon(
            &mut topo,
            pose,
            &[
                at(-near, -half_y),
                at(-far, -half_y),
                at(-far, half_y),
                at(-near, half_y),
            ],
        );
        let mut rails = Vec::new();
        let mut samples = Vec::new();
        for (radius, y) in [
            (near, -half_y),
            (far, -half_y),
            (far, half_y),
            (near, half_y),
        ] {
            let arc = circle_through(
                pose,
                p(radius, y, 0.0),
                p(0.0, y, radius),
                p(-radius, y, 0.0),
            );
            let (a, b) = (
                pose.mul_point(p(radius, y, 0.0)),
                pose.mul_point(p(-radius, y, 0.0)),
            );
            let (va, vb) = (
                topo.add_vertex(Vertex::new(a, TOL)),
                topo.add_vertex(Vertex::new(b, TOL)),
            );
            let edge = topo.add_edge(Edge::new(va, vb, EdgeCurve::Circle(arc.clone())));
            rails.push(LoftRail::Edge(edge));
            let (t0, t1) = topo.edge(edge).unwrap().curve().domain_with_endpoints(a, b);
            samples
                .extend((0..=24).map(|i| arc.evaluate((t1 - t0).mul_add(f64::from(i) / 24.0, t0))));
        }
        let options = LoftOptions {
            rails,
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert_close(exact, want, 1e-7, name);
        assert_on_solid(&topo, solid, &samples, 2e-8, name);
        for (q, want_class) in [
            (p(0.0, 0.0, 7.0), PointClassification::Inside),
            (p(0.0, 0.0, 4.0), PointClassification::Outside),
            (p(0.0, 0.0, 10.0), PointClassification::Outside),
            (p(0.0, 0.0, -7.0), PointClassification::Outside),
        ] {
            assert_eq!(
                classify(&topo, solid, pose.mul_point(q)),
                want_class,
                "{name}: at {q:?}"
            );
        }

        // Every vertex of the curved sides lies in the plane of both caps, and the
        // sides are still what is left without them.
        let sides = loft_sides(&mut topo, solid, &[first, last]).unwrap();
        assert_eq!(topo.shell(sides).unwrap().faces().len(), 4, "{name}");
    }
}

/// Rails turning three quarters of the way round: the chord between the end
/// profiles then points against the way the rails run, which must not wind them.
#[test]
fn rails_round_three_quarters_of_a_turn_keep_the_solid_outward() {
    let (near, far, half_y) = (5.0, 9.0, 3.0);
    let area = (far - near) * 2.0 * half_y;
    let want = area * 1.5 * std::f64::consts::PI * f64::midpoint(near, far);
    let turn = 1.5 * std::f64::consts::PI;
    let (c, s) = (turn.cos(), turn.sin());
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let at = |rho: f64, y: f64, cos: f64, sin: f64| p(rho * cos, y, rho * sin);
        let first = polygon(
            &mut topo,
            pose,
            &[
                at(near, -half_y, 1.0, 0.0),
                at(far, -half_y, 1.0, 0.0),
                at(far, half_y, 1.0, 0.0),
                at(near, half_y, 1.0, 0.0),
            ],
        );
        let last = polygon(
            &mut topo,
            pose,
            &[
                at(near, -half_y, c, s),
                at(far, -half_y, c, s),
                at(far, half_y, c, s),
                at(near, half_y, c, s),
            ],
        );
        let mut rails = Vec::new();
        let middle = 0.5 * turn;
        for (radius, y) in [
            (near, -half_y),
            (far, -half_y),
            (far, half_y),
            (near, half_y),
        ] {
            let arc = circle_through(
                pose,
                at(radius, y, 1.0, 0.0),
                at(radius, y, middle.cos(), middle.sin()),
                at(radius, y, c, s),
            );
            let (a, b) = (
                pose.mul_point(at(radius, y, 1.0, 0.0)),
                pose.mul_point(at(radius, y, c, s)),
            );
            let (va, vb) = (
                topo.add_vertex(Vertex::new(a, TOL)),
                topo.add_vertex(Vertex::new(b, TOL)),
            );
            let edge = topo.add_edge(Edge::new(va, vb, EdgeCurve::Circle(arc)));
            rails.push(LoftRail::Edge(edge));
        }
        let options = LoftOptions {
            rails,
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert_close(exact, want, 1e-7, name);
    }
}

/// Rails kinking sideways half way up and back keep the square's section all the
/// way, and do not read as a surface turned inside out at the kink.
#[test]
fn rails_that_kink_sideways_and_back_are_a_sheared_prism() {
    let half = 3.0;
    let want = 4.0 * half * half * 10.0;
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let first = square(&mut topo, pose, half, 0.0);
        let last = square(&mut topo, pose, half, 10.0);
        let mut rails = Vec::new();
        for (x, y) in [(-half, -half), (half, -half), (half, half), (-half, half)] {
            let points: Vec<Point3> = [p(x, y, 0.0), p(x + 20.0, y, 5.0), p(x, y, 10.0)]
                .iter()
                .map(|&q| pose.mul_point(q))
                .collect();
            let (a, b) = (
                topo.add_vertex(Vertex::new(points[0], TOL)),
                topo.add_vertex(Vertex::new(points[2], TOL)),
            );
            let curve =
                NurbsCurve::new(1, vec![0.0, 0.0, 0.5, 1.0, 1.0], points, vec![1.0; 3]).unwrap();
            rails.push(LoftRail::Edge(topo.add_edge(Edge::new(
                a,
                b,
                EdgeCurve::NurbsCurve(curve),
            ))));
        }
        let options = LoftOptions {
            rails,
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert_close(exact, want, 1e-9, name);
    }
}

/// A prism 1000 long and 0.001 wide along four straight rails: its long sides
/// lie closer together than a millionth of the section's perimeter, and they do
/// not meet.
#[test]
fn a_thin_prism_along_straight_rails_does_not_meet_itself() {
    let (long, thin, h) = (500.0, 0.0005, 10.0);
    let want = 4.0 * long * thin * h;
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let at = |(x, y): (f64, f64), z: f64| p(x * long, y * thin, z);
        let first = polygon(&mut topo, pose, &CORNERS.map(|c| at(c, 0.0)));
        let last = polygon(&mut topo, pose, &CORNERS.map(|c| at(c, h)));
        let rails = CORNERS
            .map(|c| LoftRail::Edge(line_edge(&mut topo, pose, at(c, 0.0), at(c, h))))
            .to_vec();
        let options = LoftOptions {
            rails,
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert_close(exact, want, 1e-9, name);
    }
}

/// A profile whose vertices all lie on the face, but one of whose straight edges
/// crosses a hole in it.
#[test]
fn a_tangent_profile_across_a_hole_is_refused() {
    for (_, pose) in poses() {
        let mut topo = Topology::new();
        let first = square(&mut topo, pose, 6.0, 0.0);
        let last = square(&mut topo, pose, 4.0, 10.0);
        let ground = square(&mut topo, pose, 50.0, 0.0);
        let hole = polygon(
            &mut topo,
            pose,
            &[
                p(-1.0, -7.0, 0.0),
                p(-1.0, -5.0, 0.0),
                p(1.0, -5.0, 0.0),
                p(1.0, -7.0, 0.0),
            ],
        );
        let data = topo.face(ground).unwrap();
        let (outer, surface) = (data.outer_wire(), data.surface().clone());
        let hole_wire = topo.face(hole).unwrap().outer_wire();
        let holed = topo.add_face(Face::new(outer, vec![hole_wire], surface));
        let options = LoftOptions {
            start: LoftEnd::Tangent {
                face: holed,
                weight: 1.0,
            },
            ..LoftOptions::default()
        };
        assert_refused(
            loft_with_options(&mut topo, &[first, last], &options),
            "off the face",
        );
    }
}

/// A pillow at height `z`: four cubic edges bowing out by `bow` between the
/// corners of a square of half-size `h`, the third stored backward.
fn pillow(topo: &mut Topology, pose: Mat4, h: f64, z: f64, bow: f64) -> FaceId {
    let at = |(x, y): (f64, f64)| p(x * h, y * h, z);
    let outward = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)];
    let verts = CORNERS.map(|c| topo.add_vertex(Vertex::new(pose.mul_point(at(c)), TOL)));
    let edges = (0..4)
        .map(|i| {
            let (a, b) = (at(CORNERS[i]), at(CORNERS[(i + 1) % 4]));
            let third = (b - a) * (1.0 / 3.0);
            let push = Vec3::new(outward[i].0 * bow, outward[i].1 * bow, 0.0);
            let mut points = [a, a + third + push, b - third + push, b];
            let forward = i != 2;
            if !forward {
                points.reverse();
            }
            let curve = EdgeCurve::NurbsCurve(bezier(pose, &points));
            let (from, to) = (verts[i], verts[(i + 1) % 4]);
            let edge = if forward {
                Edge::new(from, to, curve)
            } else {
                Edge::new(to, from, curve)
            };
            OrientedEdge::new(topo.add_edge(edge), forward)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let normal = outward_of(pose);
    let d = normal.dot(pose.mul_point(p(0.0, 0.0, z)) - p(0.0, 0.0, 0.0));
    topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d }))
}

/// The normal a loop running anticlockwise about z has once `pose` carries it.
fn outward_of(pose: Mat4) -> Vec3 {
    let (o, x, y) = (
        pose.mul_point(p(0.0, 0.0, 0.0)),
        pose.mul_point(p(1.0, 0.0, 0.0)),
        pose.mul_point(p(0.0, 1.0, 0.0)),
    );
    (x - o).cross(y - o).normalize().unwrap()
}

/// All of a profile edge's own samples, for checking a solid passes through it.
fn profile_samples(topo: &Topology, face: FaceId) -> Vec<Point3> {
    let wire = topo.wire(topo.face(face).unwrap().outer_wire()).unwrap();
    wire.edges()
        .iter()
        .flat_map(|oe| {
            let edge = topo.edge(oe.edge()).unwrap();
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            (0..=24)
                .map(|k| {
                    edge.curve().evaluate_with_endpoints(
                        (t1 - t0).mul_add(f64::from(k) / 24.0, t0),
                        a,
                        b,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Spline-edged profiles, one edge stored backward, along rails and under a
/// normal end: the surface passes through their own curves.
#[test]
fn spline_profiles_stored_either_way_pass_through_their_curves() {
    let (h, low, high) = (10.0, 6.0, 4.0);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let first = pillow(&mut topo, pose, low, 0.0, 2.0);
        let last = pillow(&mut topo, pose, high, h, 1.0);
        let chord = (2.0 * (high - low) * (high - low) + h * h).sqrt();
        let curves = CORNERS.map(|(x, y)| {
            bezier(
                pose,
                &[
                    p(x * low, y * low, 0.0),
                    p(x * low, y * low, chord / 3.0),
                    p(x * (high + 1.0), y * (high + 1.0), h / 2.0),
                    p(x * high, y * high, h),
                ],
            )
        });
        let options = LoftOptions {
            rails: edge_rails(&mut topo, &curves),
            start: LoftEnd::Normal { weight: 1.0 },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        assert_solid(&topo, solid, name);
        for curve in &curves {
            assert_on_solid(&topo, solid, &curve_samples(curve, 40), 1e-9, name);
        }
        for face in [first, last] {
            assert_on_solid(&topo, solid, &profile_samples(&topo, face), 1e-9, name);
        }
        let up = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let off = worst_angle_off(
            &derivatives_along(
                &topo,
                solid,
                &on_plane(pose.mul_point(p(0.0, 0.0, 0.0)), up),
            ),
            |_, _| up,
        );
        assert!(off < 1e-6, "{name}: {off} radians off square");
    }
}

/// A square whose first side is a quadratic spline bowed out by a hair, 5e-6 at
/// its middle, lofted under a normal end: to the loft the spline is no line, and
/// the solid's edges pass through it.
#[test]
fn a_spline_side_bowed_by_a_hair_keeps_its_curve() {
    let (half, h) = (6.0, 10.0);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let corners = CORNERS.map(|(x, y)| p(x * half, y * half, 0.0));
        let verts = corners.map(|c| topo.add_vertex(Vertex::new(pose.mul_point(c), TOL)));
        let bow = bezier(pose, &[corners[0], p(0.0, -half - 1e-5, 0.0), corners[1]]);
        let mut edges = vec![OrientedEdge::new(
            topo.add_edge(Edge::new(
                verts[0],
                verts[1],
                EdgeCurve::NurbsCurve(bow.clone()),
            )),
            true,
        )];
        for i in 1..4 {
            let edge = topo.add_edge(Edge::new(verts[i], verts[(i + 1) % 4], EdgeCurve::Line));
            edges.push(OrientedEdge::new(edge, true));
        }
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let normal = outward_of(pose);
        let d = normal.dot(pose.mul_point(p(0.0, 0.0, 0.0)) - p(0.0, 0.0, 0.0));
        let first = topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d }));
        let last = square(&mut topo, pose, half, h);
        let options = LoftOptions {
            start: LoftEnd::Normal { weight: 1.0 },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        assert_solid(&topo, solid, name);
        let off = curve_samples(&bow, 16)
            .iter()
            .map(|&q| from_edges(&topo, solid, q))
            .fold(0.0, f64::max);
        assert!(
            off < 1e-9,
            "{name}: the spline lies {off} off the solid's edges"
        );
    }
}

/// A square side that is one degree-1 NURBS with a kink in the middle: the sides
/// end on the kink, so no crease lies inside a face for the mesh to chase.
#[test]
fn a_kink_in_a_spline_profile_edge_ends_the_face_there() {
    let (h, low) = (10.0, 6.0);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let corners = [
            p(-low, -low, 0.0),
            p(low, -low, 0.0),
            p(low, low, 0.0),
            p(-low, low, 0.0),
        ];
        let verts: Vec<_> = corners
            .iter()
            .map(|&c| topo.add_vertex(Vertex::new(pose.mul_point(c), TOL)))
            .collect();
        let bent = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.5, 1.0, 1.0],
            vec![
                pose.mul_point(corners[0]),
                pose.mul_point(p(0.0, -low - 3.0, 0.0)),
                pose.mul_point(corners[1]),
            ],
            vec![1.0; 3],
        )
        .unwrap();
        let mut edges = vec![OrientedEdge::new(
            topo.add_edge(Edge::new(verts[0], verts[1], EdgeCurve::NurbsCurve(bent))),
            true,
        )];
        for i in 1..4 {
            edges.push(OrientedEdge::new(
                topo.add_edge(Edge::new(verts[i], verts[(i + 1) % 4], EdgeCurve::Line)),
                true,
            ));
        }
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let normal = outward_of(pose);
        let d = normal.dot(pose.mul_point(corners[0]) - p(0.0, 0.0, 0.0));
        let first = topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d }));
        let last = square(&mut topo, pose, 4.0, h);
        let options = LoftOptions {
            end: LoftEnd::Normal { weight: 1.0 },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        assert_solid(&topo, solid, name);
        let mesh = tessellate_solid(&topo, solid, 0.05).unwrap();
        assert!(
            mesh.indices.len() / 3 < 5000,
            "{name}: {} triangles: the crease has no edge",
            mesh.indices.len() / 3
        );
    }
}

/// A square lofted to a circle under a normal end: a side from a line to an arc
/// has no one set of weights, so both are held as polynomial splines, within
/// their tolerance of the circle. The circle's seam moves to the square's first
/// corner, so the loft does not turn with the frame it happens to have.
#[test]
fn a_square_lofts_to_a_circle_under_a_normal_end() {
    let (h, low, r) = (10.0, 6.0, 5.0);
    let mut volumes = Vec::new();
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let first = square(&mut topo, pose, low, 0.0);
        let last = disc(&mut topo, pose, p(0.0, 0.0, h), r);
        let options = LoftOptions {
            start: LoftEnd::Normal { weight: 1.0 },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        volumes.push(assert_solid(&topo, solid, name));
        assert_on_solid(&topo, solid, &square_outline(pose, low, 0.0), 1e-9, name);
        assert_on_solid(
            &topo,
            solid,
            &circle_outline(pose, p(0.0, 0.0, h), r),
            2e-8,
            name,
        );
        assert_eq!(
            classify(&topo, solid, pose.mul_point(p(0.0, 0.0, 5.0))),
            PointClassification::Inside,
            "{name}"
        );
        let up = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let off = worst_angle_off(
            &derivatives_along(
                &topo,
                solid,
                &on_plane(pose.mul_point(p(0.0, 0.0, 0.0)), up),
            ),
            |_, _| up,
        );
        assert!(off < 1e-6, "{name}: {off} radians off square");
    }
    assert_close(volumes[1], volumes[0], 1e-9, "turned against upright");
    assert_close(volumes[2], volumes[0], 1e-9, "mirrored against upright");
}

/// A triangle lofted to a circle: three edges against the circle's four pieces,
/// each cut at the other's places, and the circle's arcs stay on it.
#[test]
fn a_triangle_lofts_to_a_circle_cut_where_the_counts_differ() {
    let (h, r) = (10.0, 5.0);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let first = polygon(
            &mut topo,
            pose,
            &[p(-6.0, -5.0, 0.0), p(7.0, -5.0, 0.0), p(0.0, 7.0, 0.0)],
        );
        let last = disc(&mut topo, pose, p(0.0, 0.0, h), r);
        let options = LoftOptions {
            end: LoftEnd::Normal { weight: 1.0 },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        assert_solid(&topo, solid, name);
        assert_on_solid(
            &topo,
            solid,
            &circle_outline(pose, p(0.0, 0.0, h), r),
            2e-8,
            name,
        );
        assert_on_solid(&topo, solid, &profile_samples(&topo, first), 1e-8, name);
        assert_eq!(
            classify(&topo, solid, pose.mul_point(p(0.0, 0.0, 5.0))),
            PointClassification::Inside,
            "{name}"
        );
    }
}

/// An ellipse and a closed spline: a closed edge is cut into four for its sides,
/// and it holds its curve. The ellipse's seam comes round to the spline's, which
/// is fixed: at whatever angle its frame puts it, the sections would otherwise
/// twist a quarter turn and hold half the volume, 395 against the untwisted 767.
#[test]
fn closed_curved_profile_edges_are_followed_round() {
    use brepkit_math::curves::Ellipse3D;
    let h = 10.0;
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let normal = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let centre = pose.mul_point(p(0.0, 0.0, 0.0));
        let ellipse = Ellipse3D::new(centre, normal, 6.0, 4.0).unwrap();
        let seam = topo.add_vertex(Vertex::new(ellipse.evaluate(0.0), TOL));
        let rim = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Ellipse(ellipse.clone())));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
        let first = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal,
                d: normal.dot(centre - p(0.0, 0.0, 0.0)),
            },
        ));
        // A closed cubic through four points in the plane z = h.
        let at = |x: f64, y: f64| pose.mul_point(p(x, y, h));
        let loop_points = vec![
            at(5.0, 0.0),
            at(5.0, 3.0),
            at(2.0, 5.0),
            at(-2.0, 5.0),
            at(-5.0, 3.0),
            at(-5.0, -3.0),
            at(-2.0, -5.0),
            at(2.0, -5.0),
            at(5.0, -3.0),
            at(5.0, 0.0),
        ];
        let spline = NurbsCurve::new(
            3,
            vec![
                0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0,
            ],
            loop_points,
            vec![1.0; 10],
        )
        .unwrap();
        let (d0, _) = spline.domain();
        let seam = topo.add_vertex(Vertex::new(spline.evaluate(d0), TOL));
        let rim = topo.add_edge(Edge::new(seam, seam, EdgeCurve::NurbsCurve(spline.clone())));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
        let plane_normal = outward_of(pose);
        let last = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: plane_normal,
                d: plane_normal.dot(at(0.0, 0.0) - p(0.0, 0.0, 0.0)),
            },
        ));
        let options = LoftOptions {
            start: LoftEnd::Normal { weight: 1.0 },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert!(exact > 700.0, "{name}: the sections have twisted: {exact}");
        let on_ellipse: Vec<Point3> = (0..48)
            .map(|k| ellipse.evaluate(f64::from(k) * std::f64::consts::TAU / 48.0))
            .collect();
        assert_on_solid(&topo, solid, &on_ellipse, 2e-8, name);
        assert_on_solid(&topo, solid, &curve_samples(&spline, 48), 2e-8, name);
        assert_eq!(
            classify(&topo, solid, pose.mul_point(p(0.0, 0.0, 5.0))),
            PointClassification::Inside,
            "{name}"
        );
    }
}

// ── End conditions ──────────────────────────────────────────────────────────

/// A cubic from `a` to `b` leaving and arriving at the slopes given.
fn hermite(a: f64, da: f64, b: f64, db: f64) -> (impl Fn(f64) -> f64, impl Fn(f64) -> f64) {
    let value = move |t: f64| {
        let (t2, t3) = (t * t, t * t * t);
        (2.0 * t3 - 3.0 * t2 + 1.0) * a
            + (t3 - 2.0 * t2 + t) * da
            + (3.0 * t2 - 2.0 * t3) * b
            + (t3 - t2) * db
    };
    let slope = move |t: f64| {
        let t2 = t * t;
        (6.0 * t2 - 6.0 * t) * a
            + (3.0 * t2 - 4.0 * t + 1.0) * da
            + (6.0 * t - 6.0 * t2) * b
            + (3.0 * t2 - 2.0 * t) * db
    };
    (value, slope)
}

/// Two squares lofted with both ends square to their planes at weight `w`: the
/// corners' chord per unit of parameter sized by `w` along z, no slope in the
/// half-size. The lines across the sides are straight, so each height is a
/// square.
fn normal_ends_volume(low: f64, high: f64, h: f64, w: f64) -> f64 {
    let chord = (2.0 * (high - low) * (high - low) + h * h).sqrt();
    let (a, _) = hermite(low, 0.0, high, 0.0);
    let (_, dz) = hermite(0.0, w * chord, h, w * chord);
    square_stack_volume(a, dz)
}

/// Each end leaves square to its plane all the way round, read off the surface,
/// and the solid holds the integral of its square sections. The same measure
/// reads well off square on the loft without the conditions.
#[test]
fn normal_ends_leave_square_to_their_planes() {
    let (h, low, high) = (10.0, 6.0, 4.0);
    let want = normal_ends_volume(low, high, h, 1.0);
    for (name, pose) in poses() {
        let up = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let (bottom, top) = (
            pose.mul_point(p(0.0, 0.0, 0.0)),
            pose.mul_point(p(0.0, 0.0, h)),
        );
        let mut topo = Topology::new();
        let first = square(&mut topo, pose, low, 0.0);
        let last = square(&mut topo, pose, high, h);
        let options = LoftOptions {
            start: LoftEnd::Normal { weight: 1.0 },
            end: LoftEnd::Normal { weight: 1.0 },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert_close(exact, want, 1e-9, name);
        for at in [bottom, top] {
            let off = worst_angle_off(
                &derivatives_along(&topo, solid, &on_plane(at, up)),
                |_, _| up,
            );
            assert!(off < 1e-6, "{name}: {off} radians off square");
        }
        assert_eq!(
            classify(&topo, solid, pose.mul_point(p(0.0, 0.0, 5.0))),
            PointClassification::Inside,
            "{name}"
        );
        assert_eq!(
            classify(&topo, solid, pose.mul_point(p(6.5, 6.5, 2.0))),
            PointClassification::Outside,
            "{name}"
        );

        // A loft through a third square square-free has sides that lean, and a
        // conditioned end still holds with a profile in between.
        let free = polygon(
            &mut topo,
            pose,
            &[
                p(-4.5, -4.5, 5.0),
                p(4.5, -4.5, 5.0),
                p(4.5, 4.5, 5.0),
                p(-4.5, 4.5, 5.0),
            ],
        );
        let leaning = LoftOptions {
            start: LoftEnd::Normal { weight: 1.0 },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, free, last], &leaning).unwrap();
        assert_solid(&topo, solid, name);
        let at_start = worst_angle_off(
            &derivatives_along(&topo, solid, &on_plane(bottom, up)),
            |_, _| up,
        );
        assert!(
            at_start < 1e-6,
            "{name}: {at_start} radians off square at the start"
        );
        let at_end = worst_angle_off(
            &derivatives_along(&topo, solid, &on_plane(top, up)),
            |_, _| up,
        );
        assert!(
            at_end > 0.01,
            "{name}: the free end reads {at_end} radians off square"
        );
        assert_on_solid(&topo, solid, &square_outline(pose, 4.5, 5.0), 1e-9, name);
    }
}

/// A square in a plane, lofted to a smaller square above it, leaves within the
/// plane heading in, mitred at its corners: each height is a square, and the
/// surface's normal along the whole profile is the plane's. With the next square
/// no nearer one way than the other it leaves outward, and a negative weight
/// turns it inward.
#[test]
fn tangent_to_a_plane_leaves_along_the_plane() {
    let (h, low) = (10.0, 10.0);
    let chord = |high: f64| (2.0 * (high - low) * (high - low) + h * h).sqrt();
    for (name, pose) in poses() {
        let up = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let bottom = pose.mul_point(p(0.0, 0.0, 0.0));
        for (high, weight, slope) in [
            (5.0, 1.0, -chord(5.0)),
            (15.0, 1.0, chord(15.0)),
            (10.0, 1.0, chord(10.0)),
            (10.0, -1.0, -chord(10.0)),
        ] {
            let case = format!("{name}, to {high} at {weight}");
            let mut topo = Topology::new();
            let ground = square(&mut topo, pose, 50.0, 0.0);
            let first = square(&mut topo, pose, low, 0.0);
            let last = square(&mut topo, pose, high, h);
            let options = LoftOptions {
                start: LoftEnd::Tangent {
                    face: ground,
                    weight,
                },
                ..LoftOptions::default()
            };
            let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
            let exact = assert_solid(&topo, solid, &case);
            let a = quadratic(low, low + slope / 2.0, high);
            let want = square_stack_volume(a, |v| 2.0 * v * h);
            assert_close(exact, want, 1e-9, &case);
            let pairs = derivatives_along(&topo, solid, &on_plane(bottom, up));
            let off = pairs
                .iter()
                .map(|&(_, t)| t.normalize().unwrap().dot(up).abs().asin())
                .fold(0.0, f64::max);
            assert!(off < 1e-6, "{case}: {off} radians off the plane");
            for (q, t) in pairs {
                let toward_axis = (pose.mul_point(p(0.0, 0.0, 0.0)) - q).dot(t);
                assert!(
                    toward_axis * slope < 0.0,
                    "{case}: leaves the wrong way at {q:?}"
                );
            }
        }
    }
}

/// The far end arrives along its plane the way the near end leaves its own: from
/// a smaller square it runs outward into the larger one, and from one the same
/// size it arrives from outside, so the body bulges as it does at a near end.
#[test]
fn tangent_at_the_far_end_arrives_along_the_plane() {
    let (h, high): (f64, f64) = (10.0, 10.0);
    for (name, pose) in poses() {
        let up = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let top = pose.mul_point(p(0.0, 0.0, h));
        for (low, slope_sign) in [(5.0_f64, 1.0_f64), (10.0, -1.0)] {
            let case = format!("{name}, from {low}");
            let chord = (2.0 * (high - low) * (high - low) + h * h).sqrt();
            let mut topo = Topology::new();
            let ceiling = square(&mut topo, pose, 50.0, h);
            let first = square(&mut topo, pose, low, 0.0);
            let last = square(&mut topo, pose, high, h);
            let options = LoftOptions {
                end: LoftEnd::Tangent {
                    face: ceiling,
                    weight: 1.0,
                },
                ..LoftOptions::default()
            };
            let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
            let exact = assert_solid(&topo, solid, &case);
            let a = quadratic(low, high - slope_sign * chord / 2.0, high);
            let want = square_stack_volume(a, |v| 2.0 * (1.0 - v) * h);
            assert_close(exact, want, 1e-9, &case);
            let pairs = derivatives_along(&topo, solid, &on_plane(top, up));
            let off = pairs
                .iter()
                .map(|&(_, t)| t.normalize().unwrap().dot(up).abs().asin())
                .fold(0.0, f64::max);
            assert!(off < 1e-6, "{case}: {off} radians off the plane");
        }
    }
}

/// A disc in a plane, lofted to a smaller coaxial disc above it, leaves within
/// the plane heading in, radially at every point of its rational rim: turning
/// about its axis, it holds the integral of its circles.
#[test]
fn tangent_to_a_plane_on_a_round_profile_leaves_radially() {
    let (h, low, high): (f64, f64, f64) = (10.0, 8.0, 3.0);
    let chord = ((high - low) * (high - low) + h * h).sqrt();
    let radius = quadratic(low, low - chord / 2.0, high);
    let want = integrate(
        |v| std::f64::consts::PI * radius(v) * radius(v) * 2.0 * v * h,
        0.0,
        1.0,
    );
    for (name, pose) in poses() {
        let up = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let bottom = pose.mul_point(p(0.0, 0.0, 0.0));
        let mut topo = Topology::new();
        let ground = square(&mut topo, pose, 50.0, 0.0);
        let first = disc(&mut topo, pose, p(0.0, 0.0, 0.0), low);
        let last = disc(&mut topo, pose, p(0.0, 0.0, h), high);
        let options = LoftOptions {
            start: LoftEnd::Tangent {
                face: ground,
                weight: 1.0,
            },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert_close(exact, want, 1e-9, name);
        let pairs = derivatives_along(&topo, solid, &on_plane(bottom, up));
        let off = worst_angle_off(&pairs, |q, _| (bottom - q).normalize().unwrap());
        assert!(off < 1e-6, "{name}: {off} radians off radially in");
    }
}

/// The lateral face of a cylinder.
fn cylinder_wall(topo: &Topology, solid: SolidId) -> FaceId {
    solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .expect("the cylinder's wall")
}

/// The cylinder of radius `r` and height `height` carried by `pose`, with its
/// wall.
fn posed_cylinder(topo: &mut Topology, pose: Mat4, r: f64, height: f64) -> (SolidId, FaceId) {
    let solid = brepkit_operations::primitives::make_cylinder(topo, r, height).unwrap();
    brepkit_operations::transform::transform_solid(topo, solid, &pose).unwrap();
    (solid, cylinder_wall(topo, solid))
}

/// A disc on a cylinder's rim lofted to a wider coaxial disc: it leaves along
/// the wall all round, and turning about the axis it holds the integral of its
/// circles, on a quadratic from the wall's direction to the wider disc. Moved
/// off the axis, the loft still leaves along the wall at every point.
#[test]
fn tangent_to_a_cylinder_leaves_along_its_wall() {
    let (r, wide, tall, h): (f64, f64, f64, f64) = (5.0, 8.0, 10.0, 10.0);
    let chord = ((wide - r) * (wide - r) + h * h).sqrt();
    let radius = quadratic(r, r, wide);
    let want = integrate(
        |v| {
            let dz = 2.0 * (1.0 - v) * (chord / 2.0) + 2.0 * v * (h - chord / 2.0);
            std::f64::consts::PI * radius(v) * radius(v) * dz
        },
        0.0,
        1.0,
    );
    for (name, pose) in poses() {
        let axis = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let (base, rim) = (
            pose.mul_point(p(0.0, 0.0, 0.0)),
            pose.mul_point(p(0.0, 0.0, tall)),
        );
        let radial = |q: Point3| {
            let off = q - base;
            (off - axis * off.dot(axis)).normalize().unwrap()
        };
        for offset in [0.0, 3.0] {
            let case = format!("{name}, {offset} off the axis");
            let mut topo = Topology::new();
            let (_, wall) = posed_cylinder(&mut topo, pose, r, tall);
            let first = disc(&mut topo, pose, p(0.0, 0.0, tall), r);
            let last = disc(&mut topo, pose, p(offset, 0.0, tall + h), wide);
            let options = LoftOptions {
                start: LoftEnd::Tangent {
                    face: wall,
                    weight: 1.0,
                },
                ..LoftOptions::default()
            };
            let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
            let exact = assert_solid(&topo, solid, &case);
            if offset == 0.0 {
                assert_close(exact, want, 1e-9, &case);
            }
            let pairs = derivatives_along(&topo, solid, &on_plane(rim, axis));
            let off_wall = pairs
                .iter()
                .map(|&(q, t)| t.normalize().unwrap().dot(radial(q)).abs().asin())
                .fold(0.0, f64::max);
            assert!(off_wall < 1e-6, "{case}: {off_wall} radians off the wall");
            let off_axis = worst_angle_off(&pairs, |_, _| axis);
            assert!(
                off_axis < 1e-6,
                "{case}: {off_axis} radians off square to the rim"
            );
            assert_on_solid(
                &topo,
                solid,
                &circle_outline(pose, p(0.0, 0.0, tall), r),
                1e-9,
                &case,
            );
            assert_on_solid(
                &topo,
                solid,
                &circle_outline(pose, p(offset, 0.0, tall + h), wide),
                1e-9,
                &case,
            );
        }
    }
}

/// A circle of latitude on a ball, lofted to a coaxial disc above it: it leaves
/// up the meridians all round, square to the ball's radius at every point, and
/// turning about its axis it holds the integral of its circles.
#[test]
fn tangent_to_a_sphere_leaves_up_its_meridians() {
    let (ball, z0, z1, top): (f64, f64, f64, f64) = (10.0, 6.0, 14.0, 3.0);
    let r0 = (ball * ball - z0 * z0).sqrt();
    let chord = ((top - r0) * (top - r0) + (z1 - z0) * (z1 - z0)).sqrt();
    let (dr, dz) = (-z0 / ball * chord, r0 / ball * chord);
    let radius = quadratic(r0, r0 + dr / 2.0, top);
    let height = quadratic(z0, z0 + dz / 2.0, z1);
    let rises = |v: f64| 2.0 * (1.0 - v) * (dz / 2.0) + 2.0 * v * (z1 - z0 - dz / 2.0);
    let want = integrate(
        |v| std::f64::consts::PI * radius(v) * radius(v) * rises(v),
        0.0,
        1.0,
    );
    assert!((height(1.0) - z1).abs() < 1e-12);
    for (name, pose) in poses() {
        let centre = pose.mul_point(p(0.0, 0.0, 0.0));
        let up = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let mut topo = Topology::new();
        let sphere = brepkit_operations::primitives::make_sphere(&mut topo, ball, 32).unwrap();
        brepkit_operations::transform::transform_solid(&mut topo, sphere, &pose).unwrap();
        let skin = solid_faces(&topo, sphere)
            .unwrap()
            .into_iter()
            .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
            .expect("the ball's face");
        let first = disc(&mut topo, pose, p(0.0, 0.0, z0), r0);
        let last = disc(&mut topo, pose, p(0.0, 0.0, z1), top);
        let options = LoftOptions {
            start: LoftEnd::Tangent {
                face: skin,
                weight: 1.0,
            },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let exact = assert_solid(&topo, solid, name);
        assert_close(exact, want, 1e-9, name);
        let on_rim = on_plane(pose.mul_point(p(0.0, 0.0, z0)), up);
        let pairs = derivatives_along(&topo, solid, &on_rim);
        let off = pairs
            .iter()
            .map(|&(q, t)| {
                t.normalize()
                    .unwrap()
                    .dot((q - centre).normalize().unwrap())
                    .abs()
                    .asin()
            })
            .fold(0.0, f64::max);
        assert!(off < 1e-6, "{name}: {off} radians off the ball");
    }
}

/// A patch on a cylinder's wall bounded by two arcs and two lines, lofted out to
/// a flat rectangle: its corners mitre in the wall's turning tangent plane, its
/// own surface caps it, and the sides leave within the plane at every point.
#[test]
fn tangent_to_a_curved_patch_follows_the_turning_tangent_plane() {
    let (r, half_angle, half_z) = (10.0, 0.5_f64, 4.0);
    for (name, pose) in poses() {
        let axis = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let base = pose.mul_point(p(0.0, 0.0, 0.0));
        let radial = |q: Point3| {
            let off = q - base;
            (off - axis * off.dot(axis)).normalize().unwrap()
        };
        let mut topo = Topology::new();
        let (_, wall) = posed_cylinder(&mut topo, pose, r, 20.0);
        let surface = topo.face(wall).unwrap().surface().clone();
        let at = |a: f64, z: f64| p(r * a.cos(), r * a.sin(), z);
        let corners = [
            at(-half_angle, 10.0 - half_z),
            at(half_angle, 10.0 - half_z),
            at(half_angle, 10.0 + half_z),
            at(-half_angle, 10.0 + half_z),
        ];
        let verts: Vec<_> = corners
            .iter()
            .map(|&c| topo.add_vertex(Vertex::new(pose.mul_point(c), TOL)))
            .collect();
        let circle = |z: f64| {
            brepkit_math::curves::Circle3D::new(pose.mul_point(p(0.0, 0.0, z)), axis, r).unwrap()
        };
        let mirrored = outward_of(pose).dot(axis) < 0.0;
        let arc = |topo: &mut Topology, a, b, z: f64| {
            if mirrored {
                OrientedEdge::new(
                    topo.add_edge(Edge::new(b, a, EdgeCurve::Circle(circle(z)))),
                    false,
                )
            } else {
                OrientedEdge::new(
                    topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle(z)))),
                    true,
                )
            }
        };
        let bottom = arc(&mut topo, verts[0], verts[1], 10.0 - half_z);
        let right = topo.add_edge(Edge::new(verts[1], verts[2], EdgeCurve::Line));
        let top = arc(&mut topo, verts[3], verts[2], 10.0 + half_z);
        let left = topo.add_edge(Edge::new(verts[3], verts[0], EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![
                    bottom,
                    OrientedEdge::new(right, true),
                    OrientedEdge::new(top.edge(), !top.is_forward()),
                    OrientedEdge::new(left, true),
                ],
                true,
            )
            .unwrap(),
        );
        let patch = topo.add_face(Face::new(wire, vec![], surface));
        let far = 22.0;
        let plate = polygon(
            &mut topo,
            pose,
            &[
                p(far, -7.0, 5.0),
                p(far, 7.0, 5.0),
                p(far, 7.0, 15.0),
                p(far, -7.0, 15.0),
            ],
        );
        let options = LoftOptions {
            start: LoftEnd::Tangent {
                face: wall,
                weight: 1.0,
            },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[patch, plate], &options).unwrap();
        assert_solid(&topo, solid, name);
        let kinds: Vec<_> = solid_faces(&topo, solid)
            .unwrap()
            .iter()
            .map(|&f| topo.face(f).unwrap().surface().type_tag())
            .collect();
        assert_eq!(
            kinds.iter().filter(|&&k| k == "cylinder").count(),
            1,
            "{name}: {kinds:?}"
        );
        let on_wall = |q: Point3| (q - base - axis * (q - base).dot(axis)).length() - r;
        let on_it = move |q: Point3| on_wall(q).abs() < 1e-6;
        let pairs = derivatives_along(&topo, solid, &on_it);
        let off = pairs
            .iter()
            .map(|&(q, t)| t.normalize().unwrap().dot(radial(q)).abs().asin())
            .fold(0.0, f64::max);
        assert!(off < 2e-6, "{name}: {off} radians off the wall");
    }
}

/// A patch on a dished NURBS sheet, bounded by four of the sheet's own iso-curves,
/// lofted up to a flat square: the sides leave along the sheet's tangent plane at
/// every point of the patch's boundary, and the sheet's own surface caps the
/// patch, trimmed to it.
#[test]
fn tangent_to_a_nurbs_face_follows_its_tangent_plane() {
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        // A biquadratic sheet over the square of half-size 12.
        let net: Vec<Vec<Point3>> = [-12.0, 0.0, 12.0]
            .iter()
            .map(|&x| {
                [-12.0, 0.0, 12.0]
                    .iter()
                    .map(|&y| {
                        // The control values of x^2 over [-12, 12].
                        let square_of = |c: f64| if c.abs() < 1.0 { -144.0 } else { 144.0 };
                        pose.mul_point(p(x, y, 0.06 * (square_of(x) + square_of(y))))
                    })
                    .collect()
            })
            .collect();
        let sheet = NurbsSurface::new(
            2,
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            net,
            vec![vec![1.0; 3]; 3],
        )
        .unwrap();
        let (u_of, at_uv) = (
            |c: f64| (c + 12.0) / 24.0,
            |u: f64, v: f64| sheet.evaluate(u, v),
        );
        let boundary_of = |topo: &mut Topology, half: f64| -> (FaceId, Vec<Vec<Point3>>) {
            let (lo, hi) = (u_of(-half), u_of(half));
            let corners = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)];
            let verts = corners.map(|(u, v)| topo.add_vertex(Vertex::new(at_uv(u, v), TOL)));
            let mut edges = Vec::new();
            let mut lines = Vec::new();
            for i in 0..4 {
                let (a, b) = (corners[i], corners[(i + 1) % 4]);
                let mid = (f64::midpoint(a.0, b.0), f64::midpoint(a.1, b.1));
                let (pa, pm, pb) = (at_uv(a.0, a.1), at_uv(mid.0, mid.1), at_uv(b.0, b.1));
                let control = pa + (pm - pa) * 1.5 + (pm - pb) * 0.5;
                let curve = NurbsCurve::new(
                    2,
                    vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    vec![pa, control, pb],
                    vec![1.0; 3],
                )
                .unwrap();
                lines.push(curve_samples(&curve, 8));
                let edge = topo.add_edge(Edge::new(
                    verts[i],
                    verts[(i + 1) % 4],
                    EdgeCurve::NurbsCurve(curve),
                ));
                edges.push(OrientedEdge::new(edge, true));
            }
            let wire = topo.add_wire(Wire::new(edges, true).unwrap());
            let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Nurbs(sheet.clone())));
            (face, lines)
        };
        let (whole, _) = boundary_of(&mut topo, 12.0);
        let (patch, lines) = boundary_of(&mut topo, 6.0);
        for line in &lines {
            for &q in line {
                let found = project_point_to_surface(&sheet, q, 1e-11).unwrap();
                assert!(
                    found.distance < 1e-9,
                    "{name}: the fixture's edge is off the sheet"
                );
            }
        }
        let plate = square(&mut topo, pose, 8.0, 15.0);
        let options = LoftOptions {
            start: LoftEnd::Tangent {
                face: whole,
                weight: 1.0,
            },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[patch, plate], &options).unwrap();
        assert_solid(&topo, solid, name);
        let on_sheet =
            |q: Point3| project_point_to_surface(&sheet, q, 1e-11).unwrap().distance < 1e-7;
        let pairs = derivatives_along(&topo, solid, &on_sheet);
        let off = pairs
            .iter()
            .map(|&(q, t)| {
                let found = project_point_to_surface(&sheet, q, 1e-11).unwrap();
                let normal = sheet.normal(found.u, found.v).unwrap();
                t.normalize().unwrap().dot(normal).abs().asin()
            })
            .fold(0.0, f64::max);
        assert!(off < 2e-6, "{name}: {off} radians off the sheet");
    }
}

/// A larger weight leaves the same way at twice the size, and the solid holds
/// each weight's own closed form, so the weight has moved the surface.
#[test]
fn a_weight_scales_the_derivative_and_keeps_its_direction() {
    let (h, low, high) = (10.0, 6.0, 4.0);
    for (name, pose) in poses() {
        let up = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let bottom = pose.mul_point(p(0.0, 0.0, 0.0));
        let mut sizes = Vec::new();
        let mut volumes = Vec::new();
        for weight in [0.5, 1.0, 2.0] {
            let case = format!("{name}, weight {weight}");
            let mut topo = Topology::new();
            let first = square(&mut topo, pose, low, 0.0);
            let last = square(&mut topo, pose, high, h);
            let options = LoftOptions {
                start: LoftEnd::Normal { weight },
                end: LoftEnd::Normal { weight },
                ..LoftOptions::default()
            };
            let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
            let exact = assert_solid(&topo, solid, &case);
            assert_close(exact, normal_ends_volume(low, high, h, weight), 1e-9, &case);
            let pairs = derivatives_along(&topo, solid, &on_plane(bottom, up));
            let off = worst_angle_off(&pairs, |_, _| up);
            assert!(off < 1e-6, "{case}: {off} radians off square");
            sizes.push(pairs[0].1.length());
            volumes.push(exact);
        }
        assert_close(sizes[1] / sizes[0], 2.0, 1e-6, name);
        assert_close(sizes[2] / sizes[1], 2.0, 1e-6, name);
        assert!(
            (volumes[1] - volumes[0]).abs() > 0.1 && (volumes[2] - volumes[1]).abs() > 0.1,
            "{name}: the weight moved nothing: {volumes:?}"
        );
    }
}

// ── Rails and conditions together ───────────────────────────────────────────

/// Four rails that leave and arrive square to their squares at the size the
/// condition asks: the conditions hold along the whole profile, the rails are
/// the sides' edges, and each height is a square of the rails' half-size.
#[test]
fn rails_and_normal_ends_together_hold_both() {
    let (h, low, high): (f64, f64, f64) = (10.0, 6.0, 4.0);
    let chord = (2.0 * (high - low) * (high - low) + h * h).sqrt();
    let z_controls = [0.0, chord / 3.0, h - chord / 3.0, h];
    let a_controls = [low, low, high, high];
    let cubic = |c: [f64; 4], v: f64| {
        let w = 1.0 - v;
        w * w * w * c[0] + 3.0 * w * w * v * c[1] + 3.0 * w * v * v * c[2] + v * v * v * c[3]
    };
    let dz = |v: f64| {
        let w = 1.0 - v;
        3.0 * (w * w * (z_controls[1] - z_controls[0])
            + 2.0 * w * v * (z_controls[2] - z_controls[1])
            + v * v * (z_controls[3] - z_controls[2]))
    };
    let want = square_stack_volume(|v| cubic(a_controls, v), dz);
    for (name, pose) in poses() {
        let up = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let (bottom, top) = (
            pose.mul_point(p(0.0, 0.0, 0.0)),
            pose.mul_point(p(0.0, 0.0, h)),
        );
        for step in [1, 2] {
            let case = format!("{name}, a rail on every {step} vertices");
            let mut topo = Topology::new();
            let first = square(&mut topo, pose, low, 0.0);
            let last = square(&mut topo, pose, high, h);
            let curves: Vec<NurbsCurve> = CORNERS
                .iter()
                .step_by(step)
                .map(|&(x, y)| {
                    let points: Vec<Point3> = (0..4)
                        .map(|k| p(x * a_controls[k], y * a_controls[k], z_controls[k]))
                        .collect();
                    bezier(pose, &points)
                })
                .collect();
            let options = LoftOptions {
                rails: edge_rails(&mut topo, &curves),
                start: LoftEnd::Normal { weight: 1.0 },
                end: LoftEnd::Normal { weight: 1.0 },
            };
            let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
            let exact = assert_solid(&topo, solid, &case);
            if step == 1 {
                assert_close(exact, want, 1e-9, &case);
            }
            for curve in &curves {
                assert_on_solid(&topo, solid, &curve_samples(curve, 40), 1e-9, &case);
            }
            assert_on_solid(&topo, solid, &square_outline(pose, low, 0.0), 1e-9, &case);
            assert_on_solid(&topo, solid, &square_outline(pose, high, h), 1e-9, &case);
            for at in [bottom, top] {
                let off = worst_angle_off(
                    &derivatives_along(&topo, solid, &on_plane(at, up)),
                    |_, _| up,
                );
                assert!(off < 1e-6, "{case}: {off} radians off square");
            }
        }
    }
}

/// Rails that leave a plane within it, one end under a tangent condition and two
/// rails carrying the other two vertices along: the side stays in the plane's
/// tangent all along the profile, whatever the rails' own sizes make of it, and
/// a weight moves the surface between the rails and leaves the rails alone.
#[test]
fn rails_and_a_tangent_end_together_hold_both() {
    let (h, low, high) = (10.0, 8.0, 4.0);
    for (name, pose) in poses() {
        let up = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let bottom = pose.mul_point(p(0.0, 0.0, 0.0));
        let mut volumes = Vec::new();
        for weight in [0.6, 1.0] {
            let case = format!("{name}, weight {weight}");
            let mut topo = Topology::new();
            let ground = square(&mut topo, pose, 50.0, 0.0);
            let first = square(&mut topo, pose, low, 0.0);
            let last = square(&mut topo, pose, high, h);
            let curves: Vec<NurbsCurve> = [CORNERS[0], CORNERS[2]]
                .iter()
                .map(|&(x, y)| {
                    bezier(
                        pose,
                        &[
                            p(x * low, y * low, 0.0),
                            p(x * (low - 3.0), y * (low - 3.0), 0.0),
                            p(x * (high + 1.0), y * (high + 1.0), h / 2.0),
                            p(x * high, y * high, h),
                        ],
                    )
                })
                .collect();
            let options = LoftOptions {
                rails: edge_rails(&mut topo, &curves),
                start: LoftEnd::Tangent {
                    face: ground,
                    weight,
                },
                ..LoftOptions::default()
            };
            let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
            volumes.push(assert_solid(&topo, solid, &case));
            for curve in &curves {
                assert_on_solid(&topo, solid, &curve_samples(curve, 40), 1e-9, &case);
            }
            assert_on_solid(&topo, solid, &square_outline(pose, low, 0.0), 1e-9, &case);
            let pairs = derivatives_along(&topo, solid, &on_plane(bottom, up));
            let off = pairs
                .iter()
                .map(|&(_, t)| t.normalize().unwrap().dot(up).abs().asin())
                .fold(0.0, f64::max);
            assert!(off < 1e-6, "{case}: {off} radians off the plane");
        }
        assert!(
            (volumes[0] - volumes[1]).abs() > 0.1,
            "{name}: the weight moved nothing: {volumes:?}"
        );
    }
}

/// Rails leaving a round profile square to its plane, its rim rational: the
/// deviation multiplies through the rim's weights, the rails stay on the sides,
/// and the sides still leave square at every point of the rim.
#[test]
fn rails_and_a_normal_end_together_on_a_round_profile() {
    let (h, low, high) = (10.0, 6.0, 4.0);
    for (name, pose) in poses() {
        let up = dir(pose, Vec3::new(0.0, 0.0, 1.0));
        let bottom = pose.mul_point(p(0.0, 0.0, 0.0));
        let mut topo = Topology::new();
        let first = quartered_disc(&mut topo, pose, low, 0.0);
        let last = quartered_disc(&mut topo, pose, high, h);
        let curves: Vec<NurbsCurve> = (0..4)
            .map(|k| {
                let angle = f64::from(k) * std::f64::consts::FRAC_PI_2;
                let (c, s) = (angle.cos(), angle.sin());
                bezier(
                    pose,
                    &[
                        p(c * low, s * low, 0.0),
                        p(c * low, s * low, 4.0),
                        p(c * 7.0, s * 7.0, 7.0),
                        p(c * high, s * high, h),
                    ],
                )
            })
            .collect();
        let options = LoftOptions {
            rails: edge_rails(&mut topo, &curves),
            start: LoftEnd::Normal { weight: 1.0 },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        assert_solid(&topo, solid, name);
        for curve in &curves {
            assert_on_solid(&topo, solid, &curve_samples(curve, 40), 1e-9, name);
        }
        assert_on_solid(
            &topo,
            solid,
            &circle_outline(pose, p(0.0, 0.0, 0.0), low),
            1e-9,
            name,
        );
        assert_on_solid(
            &topo,
            solid,
            &circle_outline(pose, p(0.0, 0.0, h), high),
            1e-9,
            name,
        );
        let pairs = derivatives_along(&topo, solid, &on_plane(bottom, up));
        let off = worst_angle_off(&pairs, |_, _| up);
        assert!(off < 1e-6, "{name}: {off} radians off square");
    }
}

// ── On to a boolean ─────────────────────────────────────────────────────────

/// The bowed solid's top half cut away by a box: the rest holds the integral of
/// its squares up to the cut, and a point above the cut has left it.
#[test]
fn a_boolean_takes_a_railed_loft_on() {
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::transform::transform_solid;
    let (h, low, mid, high) = (10.0, 6.0, 14.0, 4.0);
    let a = quadratic(low, mid, high);
    let want = integrate(|v| 4.0 * a(v) * a(v) * h, 0.0, 0.5);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let (first, last) = (
            square(&mut topo, pose, low, 0.0),
            square(&mut topo, pose, high, h),
        );
        let curves = CORNERS.map(|(x, y)| {
            bezier(
                pose,
                &[
                    p(x * low, y * low, 0.0),
                    p(x * mid, y * mid, h / 2.0),
                    p(x * high, y * high, h),
                ],
            )
        });
        let options = LoftOptions {
            rails: edge_rails(&mut topo, &curves),
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let block = make_box(&mut topo, 60.0, 60.0, 20.0).unwrap();
        transform_solid(
            &mut topo,
            block,
            &(pose * Mat4::translation(-30.0, -30.0, h / 2.0)),
        )
        .unwrap();
        let cut = boolean(&mut topo, BooleanOp::Cut, solid, block).unwrap();
        let exact = assert_solid(&topo, cut, name);
        assert_close(exact, want, 1e-6, name);
        for (q, want_class) in [
            (p(0.0, 0.0, 2.0), PointClassification::Inside),
            (p(0.0, 0.0, 7.0), PointClassification::Outside),
            (p(8.0, 8.0, 4.5), PointClassification::Inside),
            (p(8.0, 8.0, 5.5), PointClassification::Outside),
        ] {
            assert_eq!(
                classify(&topo, cut, pose.mul_point(q)),
                want_class,
                "{name}: at {q:?}"
            );
        }
    }
}

/// A round hole bored square through the bowed solid's two curved sides: it takes
/// out each height's width across the hole's chord, 2 a(z) long, over the disc
/// the bore cuts in the plane x = 0.
#[test]
fn a_bore_through_a_railed_lofts_curved_sides_takes_what_it_should() {
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::primitives::make_cylinder;
    use brepkit_operations::transform::transform_solid;
    let (h, low, mid, high): (f64, f64, f64, f64) = (10.0, 6.0, 14.0, 4.0);
    let a = quadratic(low, mid, high);
    let (z0, radius) = (5.0, 2.0);
    let before = square_stack_volume(&a, |_| h);
    let taken = integrate(
        |z| {
            let half_chord = (radius * radius - (z - z0) * (z - z0)).max(0.0).sqrt();
            2.0 * a(z / h) * 2.0 * half_chord
        },
        z0 - radius,
        z0 + radius,
    );
    let want = before - taken;
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let (first, last) = (
            square(&mut topo, pose, low, 0.0),
            square(&mut topo, pose, high, h),
        );
        let curves = CORNERS.map(|(x, y)| {
            bezier(
                pose,
                &[
                    p(x * low, y * low, 0.0),
                    p(x * mid, y * mid, h / 2.0),
                    p(x * high, y * high, h),
                ],
            )
        });
        let options = LoftOptions {
            rails: edge_rails(&mut topo, &curves),
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let bore = make_cylinder(&mut topo, radius, 60.0).unwrap();
        transform_solid(
            &mut topo,
            bore,
            &(pose
                * Mat4::translation(-30.0, 0.0, z0)
                * Mat4::rotation_y(std::f64::consts::FRAC_PI_2)),
        )
        .unwrap();
        let cut = boolean(&mut topo, BooleanOp::Cut, solid, bore).unwrap();
        let exact = assert_solid(&topo, cut, name);
        assert_close(exact, want, 1e-4, name);
        for (q, want_class) in [
            (p(0.0, 0.0, 5.0), PointClassification::Outside),
            (p(8.0, 0.0, 5.0), PointClassification::Outside),
            (p(0.0, 3.0, 5.0), PointClassification::Inside),
            (p(0.0, 0.0, 8.0), PointClassification::Inside),
        ] {
            assert_eq!(
                classify(&topo, cut, pose.mul_point(q)),
                want_class,
                "{name}: at {q:?}"
            );
        }
    }
}

/// A loft leaving a cylinder's rim tangent to its wall, fused onto the cylinder:
/// the two share the rim's disc and hold their two volumes.
#[test]
fn a_tangent_loft_fuses_onto_the_cylinder_it_leaves() {
    use brepkit_operations::boolean::{BooleanOp, boolean};
    let (r, wide, tall, h) = (5.0, 8.0, 10.0, 10.0);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let (cylinder, wall) = posed_cylinder(&mut topo, pose, r, tall);
        let first = disc(&mut topo, pose, p(0.0, 0.0, tall), r);
        let last = disc(&mut topo, pose, p(0.0, 0.0, tall + h), wide);
        let options = LoftOptions {
            start: LoftEnd::Tangent {
                face: wall,
                weight: 1.0,
            },
            ..LoftOptions::default()
        };
        let solid = loft_with_options(&mut topo, &[first, last], &options).unwrap();
        let loft_volume = exact_solid_volume(&topo, solid).unwrap().unwrap();
        let cylinder_volume = std::f64::consts::PI * r * r * tall;
        let fused = boolean(&mut topo, BooleanOp::Fuse, cylinder, solid).unwrap();
        let exact = assert_solid(&topo, fused, name);
        assert_close(exact, cylinder_volume + loft_volume, 1e-6, name);
        for (q, want_class) in [
            (p(0.0, 0.0, 5.0), PointClassification::Inside),
            (p(0.0, 0.0, 15.0), PointClassification::Inside),
            (p(7.0, 0.0, 5.0), PointClassification::Outside),
            (p(7.0, 0.0, 19.5), PointClassification::Inside),
        ] {
            assert_eq!(
                classify(&topo, fused, pose.mul_point(q)),
                want_class,
                "{name}: at {q:?}"
            );
        }
    }
}

// ── Refusals ────────────────────────────────────────────────────────────────

fn straight_rails(topo: &mut Topology, pose: Mat4, low: f64, high: f64, h: f64) -> Vec<LoftRail> {
    CORNERS
        .map(|(x, y)| {
            LoftRail::Edge(line_edge(
                topo,
                pose,
                p(x * low, y * low, 0.0),
                p(x * high, y * high, h),
            ))
        })
        .to_vec()
}

#[test]
fn a_rail_off_a_vertex_of_a_profile_is_refused() {
    for (_, pose) in poses() {
        a_rail_off_a_vertex_of_a_profile_is_refused_in(pose);
    }
}

fn a_rail_off_a_vertex_of_a_profile_is_refused_in(pose: Mat4) {
    for (off, refused) in [(5e-8, false), (1e-6, true), (0.1, true)] {
        let mut topo = Topology::new();
        let first = square(&mut topo, pose, 6.0, 0.0);
        let last = square(&mut topo, pose, 4.0, 10.0);
        let mut rails = straight_rails(&mut topo, pose, 6.0, 4.0, 10.0);
        rails[2] = LoftRail::Edge(line_edge(
            &mut topo,
            pose,
            p(6.0 + off, 6.0, 0.0),
            p(4.0, 4.0, 10.0),
        ));
        let options = LoftOptions {
            rails,
            ..LoftOptions::default()
        };
        let result = loft_with_options(&mut topo, &[first, last], &options);
        if refused {
            assert_refused(result, "does not pass through a vertex of profile 0");
        } else {
            let solid = result.unwrap();
            assert_solid(&topo, solid, "just within the rail's tolerance");
        }
    }
    let mut topo = Topology::new();
    let first = square(&mut topo, pose, 6.0, 0.0);
    let middle = square(&mut topo, pose, 7.0, 5.0);
    let last = square(&mut topo, pose, 4.0, 10.0);
    let rails = straight_rails(&mut topo, pose, 6.0, 4.0, 10.0);
    let options = LoftOptions {
        rails,
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[first, middle, last], &options),
        "does not pass through a vertex of profile 1",
    );
}

#[test]
fn rails_the_vertices_cannot_pair_with_are_refused() {
    for (_, pose) in poses() {
        rails_the_vertices_cannot_pair_with_are_refused_in(pose);
    }
}

fn rails_the_vertices_cannot_pair_with_are_refused_in(pose: Mat4) {
    let mut topo = Topology::new();
    let first = square(&mut topo, pose, 6.0, 0.0);
    let last = square(&mut topo, pose, 4.0, 10.0);

    let mut rails = straight_rails(&mut topo, pose, 6.0, 4.0, 10.0);
    rails.push(rails[0]);
    let options = LoftOptions {
        rails,
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[first, last], &options),
        "5 rails cannot pair with a profile of 4 vertices",
    );

    let rails = straight_rails(&mut topo, pose, 6.0, 4.0, 10.0);
    let options = LoftOptions {
        rails: vec![rails[0], rails[0]],
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[first, last], &options),
        "two rails run through one vertex",
    );

    // Three rails, the second and third changing places between the profiles.
    let at = |(x, y): (f64, f64), h: f64, z: f64| p(x * h, y * h, z);
    let crossing = vec![
        LoftRail::Edge(line_edge(
            &mut topo,
            pose,
            at(CORNERS[0], 6.0, 0.0),
            at(CORNERS[0], 4.0, 10.0),
        )),
        LoftRail::Edge(line_edge(
            &mut topo,
            pose,
            at(CORNERS[1], 6.0, 0.0),
            at(CORNERS[2], 4.0, 10.0),
        )),
        LoftRail::Edge(line_edge(
            &mut topo,
            pose,
            at(CORNERS[2], 6.0, 0.0),
            at(CORNERS[1], 4.0, 10.0),
        )),
    ];
    let options = LoftOptions {
        rails: crossing,
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[first, last], &options),
        "different orders",
    );

    // A rail that reaches the middle profile last.
    let middle = square(&mut topo, pose, 5.0, 5.0);
    let rails = vec![LoftRail::Edge(line_edge(
        &mut topo,
        pose,
        p(-6.0, -6.0, 0.0),
        p(-4.0, -4.0, 10.0),
    ))];
    let options = LoftOptions {
        rails,
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[first, last, middle], &options),
        "in order",
    );
}

#[test]
fn a_tangent_face_the_profile_is_not_on_is_refused() {
    for (_, pose) in poses() {
        a_tangent_face_the_profile_is_not_on_is_refused_in(pose);
    }
}

fn a_tangent_face_the_profile_is_not_on_is_refused_in(pose: Mat4) {
    let mut topo = Topology::new();
    let first = square(&mut topo, pose, 6.0, 0.0);
    let last = square(&mut topo, pose, 4.0, 10.0);
    let tangent = |face: FaceId| LoftOptions {
        start: LoftEnd::Tangent { face, weight: 1.0 },
        ..LoftOptions::default()
    };

    let above = square(&mut topo, pose, 50.0, 1.0);
    assert_refused(
        loft_with_options(&mut topo, &[first, last], &tangent(above)),
        "off the face",
    );
    let (_, wall) = posed_cylinder(&mut topo, pose, 5.0, 10.0);
    assert_refused(
        loft_with_options(&mut topo, &[first, last], &tangent(wall)),
        "off the face",
    );

    // In the face's plane, but past its boundary.
    let beside = polygon(
        &mut topo,
        pose,
        &[
            p(20.0, 20.0, 0.0),
            p(40.0, 20.0, 0.0),
            p(40.0, 40.0, 0.0),
            p(20.0, 40.0, 0.0),
        ],
    );
    assert_refused(
        loft_with_options(&mut topo, &[first, last], &tangent(beside)),
        "runs off the face",
    );

    // A face the square covers only part of.
    let part = square(&mut topo, pose, 3.0, 0.0);
    assert_refused(
        loft_with_options(&mut topo, &[first, last], &tangent(part)),
        "runs off the face",
    );
}

#[test]
fn other_inputs_a_loft_cannot_honour_are_refused() {
    for (_, pose) in poses() {
        other_inputs_a_loft_cannot_honour_are_refused_in(pose);
    }
}

fn other_inputs_a_loft_cannot_honour_are_refused_in(pose: Mat4) {
    let mut topo = Topology::new();
    let first = square(&mut topo, pose, 6.0, 0.0);
    let last = square(&mut topo, pose, 4.0, 10.0);

    let options = LoftOptions {
        start: LoftEnd::Normal { weight: 1.0 },
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[first], &options),
        "at least 2",
    );

    for weight in [0.0, f64::NAN, f64::INFINITY] {
        let options = LoftOptions {
            end: LoftEnd::Normal { weight },
            ..LoftOptions::default()
        };
        assert_refused(
            loft_with_options(&mut topo, &[first, last], &options),
            "finite and nonzero",
        );
    }

    // A profile out of any one plane cannot leave square to it.
    let bent = polygon(
        &mut topo,
        pose,
        &[
            p(-6.0, -6.0, 0.0),
            p(6.0, -6.0, 1.0),
            p(6.0, 6.0, 0.0),
            p(-6.0, 6.0, 1.0),
        ],
    );
    let options = LoftOptions {
        start: LoftEnd::Normal { weight: 1.0 },
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[bent, last], &options),
        "planar profile",
    );

    // Straight rails lean off square, so they cannot leave a normal end.
    let rails = straight_rails(&mut topo, pose, 6.0, 4.0, 10.0);
    let options = LoftOptions {
        rails,
        start: LoftEnd::Normal { weight: 1.0 },
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[first, last], &options),
        "off its end condition",
    );

    // Nor a plane's tangent, which they rise out of.
    let ground = square(&mut topo, pose, 50.0, 0.0);
    let rails = straight_rails(&mut topo, pose, 6.0, 4.0, 10.0);
    let options = LoftOptions {
        rails,
        start: LoftEnd::Tangent {
            face: ground,
            weight: 1.0,
        },
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[first, last], &options),
        "off its end condition",
    );

    // A spike whose tip's two sides nearly double back has no mitre to leave on.
    let spike = polygon(
        &mut topo,
        pose,
        &[p(-6.0, -0.5, 0.0), p(6.0, 0.0, 0.0), p(-6.0, 0.5, 0.0)],
    );
    let top = polygon(
        &mut topo,
        pose,
        &[p(-6.0, -0.5, 10.0), p(6.0, 0.0, 10.0), p(-6.0, 0.5, 10.0)],
    );
    let options = LoftOptions {
        start: LoftEnd::Tangent {
            face: ground,
            weight: 1.0,
        },
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[spike, top], &options),
        "doubles back",
    );

    // Two profiles in one plane leave no way to tell which way they wind.
    let beside = polygon(
        &mut topo,
        pose,
        &[
            p(20.0, -6.0, 0.0),
            p(32.0, -6.0, 0.0),
            p(32.0, 6.0, 0.0),
            p(20.0, 6.0, 0.0),
        ],
    );
    let options = LoftOptions {
        start: LoftEnd::Tangent {
            face: ground,
            weight: 1.0,
        },
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[first, beside], &options),
        "winding",
    );

    // A weight that carries every corner through the axis, so each section
    // shrinks through a point and comes out turned.
    let (wide, narrow) = (
        square(&mut topo, pose, 8.0, 0.0),
        square(&mut topo, pose, 4.0, 10.0),
    );
    let options = LoftOptions {
        start: LoftEnd::Tangent {
            face: ground,
            weight: 3.0,
        },
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[wide, narrow], &options),
        "runs back over itself",
    );

    // The same with rails holding two opposite corners back: the other two
    // pass each other and the loop round the sides crosses itself.
    let held: Vec<NurbsCurve> = [CORNERS[0], CORNERS[2]]
        .iter()
        .map(|&(x, y)| {
            bezier(
                pose,
                &[
                    p(x * 8.0, y * 8.0, 0.0),
                    p(x * 5.0, y * 5.0, 0.0),
                    p(x * 5.0, y * 5.0, 5.0),
                    p(x * 4.0, y * 4.0, 10.0),
                ],
            )
        })
        .collect();
    let options = LoftOptions {
        rails: edge_rails(&mut topo, &held),
        start: LoftEnd::Tangent {
            face: ground,
            weight: 3.0,
        },
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[wide, narrow], &options),
        "runs back over itself",
    );

    // A weight that throws the surface past where it comes back through.
    let options = LoftOptions {
        start: LoftEnd::Normal { weight: 40.0 },
        end: LoftEnd::Tangent {
            face: square(&mut topo, pose, 50.0, 10.0),
            weight: 40.0,
        },
        ..LoftOptions::default()
    };
    assert_refused(
        loft_with_options(&mut topo, &[first, last], &options),
        "runs back over itself",
    );
}

/// Without rails or conditions the options are `loft_smooth`, face for face.
#[test]
fn no_rails_and_free_ends_are_the_smooth_loft() {
    let pose = Mat4::identity();
    let mut topo = Topology::new();
    let a = square(&mut topo, pose, 6.0, 0.0);
    let b = square(&mut topo, pose, 4.0, 5.0);
    let c = square(&mut topo, pose, 7.0, 10.0);
    let with = loft_with_options(&mut topo, &[a, b, c], &LoftOptions::default()).unwrap();
    let plain = brepkit_operations::loft::loft_smooth(&mut topo, &[a, b, c]).unwrap();
    let net = |topo: &Topology, s: SolidId| -> Vec<(Vec<Vec<Point3>>, Vec<f64>)> {
        nurbs_sides(topo, s)
            .into_iter()
            .map(|side| (side.control_points().to_vec(), side.knots_v().to_vec()))
            .collect()
    };
    assert_eq!(
        solid_faces(&topo, with).unwrap().len(),
        solid_faces(&topo, plain).unwrap().len()
    );
    assert_eq!(net(&topo, with), net(&topo, plain));
}
