//! Open shells swept from open wires, checked against closed forms: each
//! shell's area, each face's own mesh, the edges its faces share, and the same
//! numbers under a rotation and a mirror.

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::print_stdout
    )]

    use std::collections::{HashMap, HashSet};
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    use brepkit_check::validate::checks::CheckId;
    use brepkit_check::validate::{ValidateOptions, validate_shell};
    use brepkit_math::curves::{Circle3D, Ellipse3D};
    use brepkit_math::mat::Mat4;
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::OperationsError;
    use brepkit_operations::measure::face_area;
    use brepkit_operations::sheet::{extrude_wire, loft_wires, revolve_wire, sweep_wire};
    use brepkit_operations::tessellate::{TriangleMesh, tessellate};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::face::{FaceId, FaceSurface};
    use brepkit_topology::shell::ShellId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

    const TOL: f64 = 1e-7;
    const DEFLECTION: f64 = 1e-3;

    #[derive(Clone, Copy)]
    struct Pose(Mat4);

    impl Pose {
        fn p(self, x: f64, y: f64, z: f64) -> Point3 {
            self.0.mul_point(Point3::new(x, y, z))
        }

        fn v(self, x: f64, y: f64, z: f64) -> Vec3 {
            self.p(x, y, z) - self.p(0.0, 0.0, 0.0)
        }
    }

    fn upright() -> Pose {
        Pose(Mat4::identity())
    }

    /// As drawn, turned and moved, and mirrored in a plane off the origin.
    fn poses() -> [(&'static str, Pose); 3] {
        let turned =
            Mat4::translation(3.0, -7.0, 11.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(1.1);
        let n = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
        let through = 1.0 * n[0] + 2.0 * n[1] + 3.0 * n[2];
        let mut mirror = [[0.0; 4]; 4];
        for (i, row) in mirror.iter_mut().enumerate().take(3) {
            for (j, cell) in row.iter_mut().enumerate().take(3) {
                *cell = f64::from(u8::from(i == j)) - 2.0 * n[i] * n[j];
            }
            row[3] = 2.0 * through * n[i];
        }
        mirror[3][3] = 1.0;
        [
            ("upright", upright()),
            ("turned", Pose(turned)),
            ("mirrored", Pose(Mat4(mirror))),
        ]
    }

    /// The circle through `a`, `through` and `b`, run from `a` to `b` by way
    /// of `through`.
    fn arc(a: Point3, through: Point3, b: Point3) -> EdgeCurve {
        let (u, v) = (through - a, b - a);
        let w = u.cross(v);
        let center = a + (v.cross(w) * u.dot(u) + w.cross(u) * v.dot(v)) * (1.0 / (2.0 * w.dot(w)));
        EdgeCurve::Circle(
            Circle3D::new(center, w.normalize().unwrap(), (a - center).length()).unwrap(),
        )
    }

    enum Run {
        Line(Point3),
        Arc(Point3, Point3),
        Curve(EdgeCurve, Point3),
    }

    fn open_wire(topo: &mut Topology, start: Point3, runs: Vec<Run>) -> WireId {
        let (mut from, mut at) = (topo.add_vertex(Vertex::new(start, TOL)), start);
        let mut edges = Vec::with_capacity(runs.len());
        for run in runs {
            let (curve, to) = match run {
                Run::Line(to) => (EdgeCurve::Line, to),
                Run::Arc(through, to) => (arc(at, through, to), to),
                Run::Curve(curve, to) => (curve, to),
            };
            let end = topo.add_vertex(Vertex::new(to, TOL));
            edges.push(OrientedEdge::new(
                topo.add_edge(Edge::new(from, end, curve)),
                true,
            ));
            (from, at) = (end, to);
        }
        topo.add_wire(Wire::new(edges, false).unwrap())
    }

    /// The wire from the issue: a line 10 long, then a half circle of radius 5.
    fn issue_wire(topo: &mut Topology, pose: Pose) -> WireId {
        open_wire(
            topo,
            pose.p(0.0, 0.0, 0.0),
            vec![
                Run::Line(pose.p(10.0, 0.0, 0.0)),
                Run::Arc(pose.p(15.0, 5.0, 0.0), pose.p(20.0, 0.0, 0.0)),
            ],
        )
    }

    fn faces(topo: &Topology, shell: ShellId) -> Vec<FaceId> {
        topo.shell(shell).unwrap().faces().to_vec()
    }

    fn count(topo: &Topology, shell: ShellId, is: fn(&FaceSurface) -> bool) -> usize {
        faces(topo, shell)
            .iter()
            .filter(|&&f| is(topo.face(f).unwrap().surface()))
            .count()
    }

    fn mesh_area(mesh: &TriangleMesh) -> f64 {
        mesh.indices
            .chunks_exact(3)
            .map(|t| {
                let [a, b, c] = [t[0], t[1], t[2]].map(|i| mesh.positions[i as usize]);
                0.5 * (b - a).cross(c - a).length()
            })
            .sum()
    }

    /// Each triangle of a face's mesh winds about the normals at its corners.
    fn assert_mesh_winds_with_its_normals(mesh: &TriangleMesh, face: FaceId, label: &str) {
        for t in mesh.indices.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| i as usize);
            let wound = (mesh.positions[b] - mesh.positions[a])
                .cross(mesh.positions[c] - mesh.positions[a]);
            if wound.length() < 1e-12 {
                continue;
            }
            let along = mesh.normals[a] + mesh.normals[b] + mesh.normals[c];
            assert!(
                wound.dot(along) > 0.0,
                "{label}: a triangle of face {} winds against its normals",
                face.index()
            );
        }
    }

    /// Whether one of a face's boundary edges passes through `at`.
    fn face_passes_through(topo: &Topology, face: FaceId, at: Point3) -> bool {
        let face = topo.face(face).unwrap();
        std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .any(|wid| {
                topo.wire(wid).unwrap().edges().iter().any(|oe| {
                    let edge = topo.edge(oe.edge()).unwrap();
                    let (s, e) = (
                        topo.vertex(edge.start()).unwrap().point(),
                        topo.vertex(edge.end()).unwrap().point(),
                    );
                    let (t0, t1) = edge.curve().domain_with_endpoints(s, e);
                    (0..=256).any(|k| {
                        let t = t0 + (t1 - t0) * f64::from(k) / 256.0;
                        let p = edge.curve().evaluate_with_endpoints(t, s, e);
                        (p - at).length() < 0.2
                    })
                })
            })
    }

    /// The faces through `at`, joined smoothly there, turn the same way: each
    /// face's mesh normal at its vertex nearest `at` agrees with the others'.
    fn assert_faces_turn_one_way_at(topo: &Topology, shell: ShellId, at: Point3, label: &str) {
        let mut normals = Vec::new();
        for f in faces(topo, shell) {
            if !face_passes_through(topo, f, at) {
                continue;
            }
            let mesh = tessellate(topo, f, DEFLECTION).unwrap();
            let nearest = (0..mesh.positions.len())
                .min_by(|&i, &j| {
                    (mesh.positions[i] - at)
                        .length()
                        .total_cmp(&(mesh.positions[j] - at).length())
                })
                .unwrap();
            normals.push(mesh.normals[nearest]);
        }
        assert!(normals.len() >= 2, "{label}: no two faces meet at {at:?}");
        for w in normals.windows(2) {
            assert!(
                w[0].dot(w[1]) > 0.8,
                "{label}: faces turn opposite ways at {at:?}: {normals:?}"
            );
        }
    }

    /// How a shell's faces use its edges. Every use of an edge runs against
    /// its other use, as two faces of one oriented surface must.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct Census {
        boundary: usize,
        interior: usize,
        seams: usize,
    }

    fn census(topo: &Topology, shell: ShellId) -> Census {
        let mut uses: HashMap<usize, Vec<(usize, bool)>> = HashMap::new();
        for f in faces(topo, shell) {
            let face = topo.face(f).unwrap();
            let wires =
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
            for wid in wires {
                for oe in topo.wire(wid).unwrap().edges() {
                    uses.entry(oe.edge().index())
                        .or_default()
                        .push((f.index(), oe.is_forward() != face.is_reversed()));
                }
            }
        }
        let mut out = Census::default();
        for (edge, used) in uses {
            match used.as_slice() {
                [_] => out.boundary += 1,
                [(a, run_a), (b, run_b)] => {
                    assert_ne!(run_a, run_b, "edge {edge} runs one way in both uses");
                    if a == b {
                        out.seams += 1;
                    } else {
                        out.interior += 1;
                    }
                }
                more => panic!("edge {edge} is used {} times", more.len()),
            }
        }
        out
    }

    /// Every edge of every face lies on that face's surface.
    fn assert_edges_on_surfaces(topo: &Topology, shell: ShellId, label: &str) {
        for f in faces(topo, shell) {
            let face = topo.face(f).unwrap();
            let wires =
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
            for wid in wires {
                for oe in topo.wire(wid).unwrap().edges() {
                    let edge = topo.edge(oe.edge()).unwrap();
                    let (s, e) = (
                        topo.vertex(edge.start()).unwrap().point(),
                        topo.vertex(edge.end()).unwrap().point(),
                    );
                    let (t0, t1) = edge.curve().domain_with_endpoints(s, e);
                    for k in 0..=8 {
                        let t = t0 + (t1 - t0) * f64::from(k) / 8.0;
                        let p = edge.curve().evaluate_with_endpoints(t, s, e);
                        let off = match face.surface() {
                            FaceSurface::Plane { normal, d } => {
                                (normal.dot(p - Point3::new(0.0, 0.0, 0.0)) - d).abs()
                            }
                            other => {
                                let (u, v) = other.project_point(p).unwrap();
                                (other.evaluate(u, v).unwrap() - p).length()
                            }
                        };
                        assert!(
                            off < 1e-5,
                            "{label}: edge {} is {off} off its face's surface at {p:?}",
                            oe.edge().index()
                        );
                    }
                }
            }
        }
    }

    /// The shell's area against its closed form, and each face's own mesh
    /// against its area within what `deflection` allows on a surface curving
    /// no tighter than `radius`.
    fn measure(topo: &Topology, shell: ShellId, exact: f64, radius: f64, label: &str) {
        let mut area = 0.0;
        let mut meshed = 0.0;
        for f in faces(topo, shell) {
            let a = face_area(topo, f, DEFLECTION).unwrap();
            let mesh = tessellate(topo, f, DEFLECTION).unwrap();
            assert_mesh_winds_with_its_normals(&mesh, f, label);
            let m = mesh_area(&mesh);
            assert!(
                (a - m).abs() <= 2.0 * a.max(1.0) * DEFLECTION / radius,
                "{label}: face {} has area {a} but its mesh has {m}",
                f.index()
            );
            area += a;
            meshed += m;
        }
        println!("{label}: area {area:.6}, mesh {meshed:.6}, closed form {exact:.6}");
        let allowed = 2.0 * exact * DEFLECTION / radius;
        assert!(
            (area - exact).abs() <= allowed,
            "{label}: area {area}, closed form {exact}"
        );
        assert!(
            (meshed - exact).abs() <= allowed,
            "{label}: mesh {meshed}, closed form {exact}"
        );
        assert_edges_on_surfaces(topo, shell, label);
    }

    /// [`measure`], and the shell passes validation, open as it is.
    fn check(topo: &Topology, shell: ShellId, exact: f64, radius: f64, label: &str) -> Census {
        measure(topo, shell, exact, radius, label);
        let edges = census(topo, shell);
        // The self-intersection check reads a seam's two uses as a crossing,
        // on a primitive cylinder's wall as well.
        let mut skipped = HashSet::from([CheckId::ShellClosed]);
        if edges.seams > 0 {
            skipped.insert(CheckId::WireSelfIntersection);
        }
        let options = ValidateOptions {
            disabled_checks: skipped,
            ..ValidateOptions::default()
        };
        let report = validate_shell(topo, shell, &options).unwrap();
        assert!(report.is_valid(), "{label}: {:?}", report.issues);
        edges
    }

    fn refused<T>(result: Result<T, OperationsError>, about: &str) {
        match result {
            Err(OperationsError::InvalidInput { reason }) => {
                assert!(reason.contains(about), "refused for: {reason}");
            }
            Err(other) => panic!("wrong error kind: {other}"),
            Ok(_) => panic!("expected a refusal about {about}"),
        }
    }

    fn close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a}, closed form {b}");
    }

    /// `f` integrated over `[a, b]` by Simpson's rule on `n` panels.
    fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: u32) -> f64 {
        let h = (b - a) / f64::from(2 * n);
        let mut sum = f(a) + f(b);
        for k in 1..2 * n {
            let w = if k % 2 == 1 { 4.0 } else { 2.0 };
            sum += w * f(a + h * f64::from(k));
        }
        sum * h / 3.0
    }

    fn polyline_length(points: &[Point3]) -> f64 {
        points.windows(2).map(|w| (w[1] - w[0]).length()).sum()
    }

    fn cubic_spline(points: [Point3; 4]) -> NurbsCurve {
        NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            points.to_vec(),
            vec![1.0; 4],
        )
        .unwrap()
    }

    fn is_plane(s: &FaceSurface) -> bool {
        matches!(s, FaceSurface::Plane { .. })
    }

    fn is_cylinder(s: &FaceSurface) -> bool {
        matches!(s, FaceSurface::Cylinder(_))
    }

    fn is_cone(s: &FaceSurface) -> bool {
        matches!(s, FaceSurface::Cone(_))
    }

    fn is_sphere(s: &FaceSurface) -> bool {
        matches!(s, FaceSurface::Sphere(_))
    }

    fn is_torus(s: &FaceSurface) -> bool {
        matches!(s, FaceSurface::Torus(_))
    }

    fn is_nurbs(s: &FaceSurface) -> bool {
        matches!(s, FaceSurface::Nurbs(_))
    }

    // Extrude.

    #[test]
    fn the_issue_wire_extrudes_to_its_length_times_the_distance() {
        let exact = 20.0 * (10.0 + 5.0 * PI);
        close(exact, 514.159, 1e-3, "the issue's own number");
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let wire = issue_wire(&mut topo, pose);
            let shell = extrude_wire(&mut topo, wire, pose.v(0.0, 0.0, 1.0), 20.0).unwrap();
            let edges = check(&topo, shell, exact, 5.0, name);
            assert_eq!(count(&topo, shell, is_plane), 1, "{name}");
            assert_eq!(count(&topo, shell, is_cylinder), 1, "{name}");
            assert_eq!(
                edges,
                Census {
                    boundary: 6,
                    interior: 1,
                    seams: 0
                },
                "{name}"
            );
        }
    }

    #[test]
    fn extrusion_takes_the_direction_times_the_distance_at_any_slant() {
        let mut topo = Topology::new();
        let wire = open_wire(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            vec![Run::Line(Point3::new(10.0, 0.0, 0.0))],
        );
        let shell = extrude_wire(&mut topo, wire, Vec3::new(3.0, 3.0, 4.0), 4.0).unwrap();
        check(&topo, shell, 10.0 * 20.0, 1.0, "slanted plane");
        assert_eq!(count(&topo, shell, is_plane), 1);
    }

    #[test]
    fn an_arc_past_half_a_turn_extrudes_its_own_span() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let wire = open_wire(
                &mut topo,
                pose.p(2.0, 0.0, 0.0),
                vec![Run::Arc(pose.p(-2.0, 0.0, 0.0), pose.p(0.0, -2.0, 0.0))],
            );
            let shell = extrude_wire(&mut topo, wire, pose.v(0.0, 0.0, 1.0), 3.0).unwrap();
            check(&topo, shell, 3.0 * 1.5 * PI * 2.0, 2.0, name);
            assert_eq!(count(&topo, shell, is_cylinder), 1, "{name}");
        }
    }

    #[test]
    fn an_arc_stored_against_the_wire_keeps_its_span() {
        for reversed_top in [false, true] {
            let mut topo = Topology::new();
            let a = Point3::new(2.0, 0.0, 0.0);
            let b = Point3::new(0.0, 2.0, 0.0);
            let c = Point3::new(-5.0, 2.0, 0.0);
            let (va, vb, vc) = (
                topo.add_vertex(Vertex::new(a, TOL)),
                topo.add_vertex(Vertex::new(b, TOL)),
                topo.add_vertex(Vertex::new(c, TOL)),
            );
            let through = Point3::new(2.0 * 0.5f64.sqrt(), 2.0 * 0.5f64.sqrt(), 0.0);
            let stored = arc(b, through, a);
            let arc_edge = topo.add_edge(Edge::new(vb, va, stored));
            let line = topo.add_edge(Edge::new(vb, vc, EdgeCurve::Line));
            let edges = vec![
                OrientedEdge::new(arc_edge, false),
                OrientedEdge::new(line, true),
            ];
            let wire = topo.add_wire(Wire::new(edges, false).unwrap());
            let offset = if reversed_top { -1.0 } else { 1.0 };
            let shell = extrude_wire(&mut topo, wire, Vec3::new(0.0, 0.0, offset), 4.0).unwrap();
            let edges = check(&topo, shell, 4.0 * (PI + 5.0), 2.0, "reversed arc");
            assert_eq!(edges.interior, 1);
        }
    }

    #[test]
    fn a_circle_across_the_direction_rules_an_oblique_wall_not_a_cylinder() {
        let tilt = 0.6f64;
        let r = 5.0;
        let (u, v) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, tilt.cos(), tilt.sin()),
        );
        let on = |a: f64| Point3::new(0.0, 0.0, 0.0) + u * (r * a.cos()) + v * (r * a.sin());
        let mut topo = Topology::new();
        let wire = open_wire(&mut topo, on(0.0), vec![Run::Arc(on(FRAC_PI_2), on(PI))]);
        let shell = extrude_wire(&mut topo, wire, Vec3::new(0.0, 0.0, 1.0), 6.0).unwrap();
        assert_eq!(count(&topo, shell, is_nurbs), 1);
        let shadow = simpson(
            |a| {
                let d = u * (-r * a.sin()) + v * (r * a.cos());
                (d.x() * d.x() + d.y() * d.y()).sqrt()
            },
            0.0,
            PI,
            2000,
        );
        check(&topo, shell, 6.0 * shadow, r, "oblique wall");
    }

    #[test]
    fn an_ellipse_and_a_spline_rule_their_own_spans() {
        let (a, b) = (6.0, 3.0);
        let mut topo = Topology::new();
        let ellipse = Ellipse3D::with_axes(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            a,
            b,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        let spline = cubic_spline([
            Point3::new(0.0, 3.0, 0.0),
            Point3::new(-2.0, 5.0, 0.0),
            Point3::new(-4.0, 1.0, 0.0),
            Point3::new(-7.0, 3.0, 0.0),
        ]);
        let wire = open_wire(
            &mut topo,
            Point3::new(a, 0.0, 0.0),
            vec![
                Run::Curve(EdgeCurve::Ellipse(ellipse), Point3::new(0.0, b, 0.0)),
                Run::Curve(
                    EdgeCurve::NurbsCurve(spline.clone()),
                    Point3::new(-7.0, 3.0, 0.0),
                ),
            ],
        );
        let shell = extrude_wire(&mut topo, wire, Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();
        assert_eq!(count(&topo, shell, is_nurbs), 2);
        let quarter = simpson(
            |t| ((a * t.sin()).powi(2) + (b * t.cos()).powi(2)).sqrt(),
            0.0,
            FRAC_PI_2,
            2000,
        );
        let samples: Vec<Point3> = (0..=20000)
            .map(|k| spline.evaluate(f64::from(k) / 20000.0))
            .collect();
        let exact = 5.0 * (quarter + polyline_length(&samples));
        let edges = check(&topo, shell, exact, 1.0, "ellipse and spline");
        assert_eq!(edges.interior, 1);
    }

    #[test]
    fn a_spline_drawn_arc_extrudes_to_a_cylinder() {
        let mut topo = Topology::new();
        let circle =
            Circle3D::new(Point3::new(15.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0), 5.0).unwrap();
        let (t0, t1) = EdgeCurve::Circle(circle.clone())
            .domain_with_endpoints(Point3::new(10.0, 0.0, 0.0), Point3::new(20.0, 0.0, 0.0));
        let nurbs = brepkit_geometry::convert::circle_to_nurbs(&circle, t0, t1).unwrap();
        let wire = open_wire(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            vec![
                Run::Line(Point3::new(10.0, 0.0, 0.0)),
                Run::Curve(EdgeCurve::NurbsCurve(nurbs), Point3::new(20.0, 0.0, 0.0)),
            ],
        );
        let shell = extrude_wire(&mut topo, wire, Vec3::new(0.0, 0.0, 1.0), 20.0).unwrap();
        check(&topo, shell, 20.0 * (10.0 + 5.0 * PI), 5.0, "spline arc");
        assert_eq!(count(&topo, shell, is_cylinder), 1);
    }

    #[test]
    fn extrusion_faces_of_a_tangent_chain_face_one_way() {
        let mut topo = Topology::new();
        let wire = open_wire(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            vec![
                Run::Line(Point3::new(10.0, 0.0, 0.0)),
                Run::Arc(
                    Point3::new(10.0 + 2.0 * 0.5f64.sqrt(), 2.0 - 2.0 * 0.5f64.sqrt(), 0.0),
                    Point3::new(12.0, 2.0, 0.0),
                ),
            ],
        );
        let shell = extrude_wire(&mut topo, wire, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        assert_faces_turn_one_way_at(&topo, shell, Point3::new(10.0, 0.0, 1.5), "tangent chain");
    }

    #[test]
    fn extrusion_refuses_what_sweeps_no_surface() {
        let mut topo = Topology::new();
        let wire = issue_wire(&mut topo, upright());
        refused(
            extrude_wire(&mut topo, wire, Vec3::new(0.0, 0.0, 0.0), 5.0),
            "zero",
        );
        refused(
            extrude_wire(&mut topo, wire, Vec3::new(0.0, 0.0, 1.0), 0.0),
            "zero",
        );
        refused(
            extrude_wire(&mut topo, wire, Vec3::new(1.0, 0.0, 0.0), 5.0),
            "sweeps no surface",
        );
    }

    // Revolve.

    #[test]
    fn the_issue_wire_revolved_a_full_turn_is_a_whole_sphere() {
        let exact = 100.0 * PI;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let wire = issue_wire(&mut topo, pose);
            let shell = revolve_wire(
                &mut topo,
                wire,
                pose.p(0.0, 0.0, 0.0),
                pose.v(1.0, 0.0, 0.0),
                TAU,
            )
            .unwrap();
            let edges = check(&topo, shell, exact, 5.0, name);
            assert_eq!(count(&topo, shell, is_sphere), 4, "{name}");
            assert_eq!(faces(&topo, shell).len(), 4, "{name}");
            assert_eq!(
                edges,
                Census {
                    boundary: 0,
                    interior: 4,
                    seams: 0
                },
                "{name}"
            );
        }
    }

    #[test]
    fn a_line_revolved_a_full_turn_closes_on_its_seam() {
        struct Case {
            from: (f64, f64),
            to: (f64, f64),
            exact: f64,
            is: fn(&FaceSurface) -> bool,
            edges: Census,
        }
        let rims = Census {
            boundary: 2,
            interior: 0,
            seams: 1,
        };
        let cases = [
            Case {
                from: (3.0, 0.0),
                to: (3.0, 4.0),
                exact: 24.0 * PI,
                is: is_cylinder,
                edges: rims,
            },
            Case {
                from: (3.0, 0.0),
                to: (5.0, 4.0),
                exact: 8.0 * PI * 20f64.sqrt(),
                is: is_cone,
                edges: rims,
            },
            Case {
                from: (0.0, 4.0),
                to: (3.0, 0.0),
                exact: 15.0 * PI,
                is: is_cone,
                edges: Census {
                    boundary: 1,
                    interior: 0,
                    seams: 1,
                },
            },
            Case {
                from: (3.0, 0.0),
                to: (5.0, 0.0),
                exact: 16.0 * PI,
                is: is_plane,
                edges: Census {
                    boundary: 2,
                    interior: 0,
                    seams: 0,
                },
            },
            Case {
                from: (5.0, 2.0),
                to: (0.0, 2.0),
                exact: 25.0 * PI,
                is: is_plane,
                edges: Census {
                    boundary: 1,
                    interior: 0,
                    seams: 0,
                },
            },
        ];
        for case in cases {
            for (name, pose) in poses() {
                let mut topo = Topology::new();
                let wire = open_wire(
                    &mut topo,
                    pose.p(case.from.0, 0.0, case.from.1),
                    vec![Run::Line(pose.p(case.to.0, 0.0, case.to.1))],
                );
                let shell = revolve_wire(
                    &mut topo,
                    wire,
                    pose.p(0.0, 0.0, 0.0),
                    pose.v(0.0, 0.0, 1.0),
                    TAU,
                )
                .unwrap();
                assert_eq!(faces(&topo, shell).len(), 1, "{name}");
                assert_eq!(count(&topo, shell, case.is), 1, "{name}");
                let edges = check(&topo, shell, case.exact, 3.0, name);
                assert_eq!(edges, case.edges, "{name}");
            }
        }
    }

    #[test]
    fn a_vertex_on_the_axis_stays_one_vertex() {
        let mut topo = Topology::new();
        let wire = open_wire(
            &mut topo,
            Point3::new(0.0, 0.0, 4.0),
            vec![Run::Line(Point3::new(3.0, 0.0, 0.0))],
        );
        let shell = revolve_wire(
            &mut topo,
            wire,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            PI,
        )
        .unwrap();
        let apex = topo
            .edge(topo.wire(wire).unwrap().edges()[0].edge())
            .unwrap()
            .start();
        for f in faces(&topo, shell) {
            let wid = topo.face(f).unwrap().outer_wire();
            let edges = topo.wire(wid).unwrap().edges();
            assert_eq!(edges.len(), 3, "a wedge closes at the apex");
            let at_apex = edges
                .iter()
                .filter(|oe| {
                    let e = topo.edge(oe.edge()).unwrap();
                    e.start() == apex || e.end() == apex
                })
                .count();
            assert_eq!(at_apex, 2);
        }
        check(&topo, shell, 7.5 * PI, 1.0, "half a pointed cone");
    }

    #[test]
    fn an_arc_revolved_a_full_turn_sweeps_a_torus_band_or_a_sphere_zone() {
        struct Case {
            from: (f64, f64),
            through: (f64, f64),
            to: (f64, f64),
            exact: f64,
            radius: f64,
            is: fn(&FaceSurface) -> bool,
            edges: Census,
        }
        let root = 0.5f64.sqrt();
        let cases = [
            Case {
                from: (10.0, -2.0),
                through: (12.0, 0.0),
                to: (10.0, 2.0),
                exact: 40.0 * PI * PI + 16.0 * PI,
                radius: 2.0,
                is: is_torus,
                edges: Census {
                    boundary: 2,
                    interior: 0,
                    seams: 1,
                },
            },
            Case {
                from: (10.0, 2.0),
                through: (8.0, 0.0),
                to: (10.0, -2.0),
                exact: 40.0 * PI * PI - 16.0 * PI,
                radius: 2.0,
                is: is_torus,
                edges: Census {
                    boundary: 2,
                    interior: 0,
                    seams: 1,
                },
            },
            Case {
                from: (5.0, 0.0),
                through: (
                    5.0 * (1.5f64.atan2(4.5)).cos(),
                    5.0 * (1.5f64.atan2(4.5)).sin(),
                ),
                to: (4.0, 3.0),
                exact: 30.0 * PI,
                radius: 5.0,
                is: is_sphere,
                edges: Census {
                    boundary: 2,
                    interior: 0,
                    seams: 0,
                },
            },
            Case {
                from: (0.0, 5.0),
                through: (5.0 * root.sin(), 5.0 * root.cos()),
                to: (4.0, 3.0),
                exact: 20.0 * PI,
                radius: 5.0,
                is: is_sphere,
                edges: Census {
                    boundary: 1,
                    interior: 0,
                    seams: 0,
                },
            },
        ];
        for case in cases {
            for (name, pose) in poses() {
                let mut topo = Topology::new();
                let wire = open_wire(
                    &mut topo,
                    pose.p(case.from.0, 0.0, case.from.1),
                    vec![Run::Arc(
                        pose.p(case.through.0, 0.0, case.through.1),
                        pose.p(case.to.0, 0.0, case.to.1),
                    )],
                );
                let shell = revolve_wire(
                    &mut topo,
                    wire,
                    pose.p(0.0, 0.0, 0.0),
                    pose.v(0.0, 0.0, 1.0),
                    TAU,
                )
                .unwrap();
                assert_eq!(faces(&topo, shell).len(), 1, "{name}");
                assert_eq!(count(&topo, shell, case.is), 1, "{name}");
                let edges = check(&topo, shell, case.exact, case.radius, name);
                assert_eq!(edges, case.edges, "{name}");
            }
        }
    }

    /// A wall, a fillet and a disc, swept about the z axis: their shared rims
    /// join them, a line on the axis at the far end sweeps nothing.
    fn wall_fillet_disc(topo: &mut Topology, pose: Pose) -> WireId {
        let root = 0.5f64.sqrt();
        open_wire(
            topo,
            pose.p(5.0, 0.0, 0.0),
            vec![
                Run::Line(pose.p(5.0, 0.0, 4.0)),
                Run::Arc(pose.p(4.0 + root, 0.0, 4.0 + root), pose.p(4.0, 0.0, 5.0)),
                Run::Line(pose.p(0.0, 0.0, 5.0)),
                Run::Line(pose.p(0.0, 0.0, 9.0)),
            ],
        )
    }

    #[test]
    fn a_chain_revolved_a_full_turn_shares_its_rims() {
        let exact = 58.0 * PI + 4.0 * PI * PI;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let wire = wall_fillet_disc(&mut topo, pose);
            let shell = revolve_wire(
                &mut topo,
                wire,
                pose.p(0.0, 0.0, 0.0),
                pose.v(0.0, 0.0, 1.0),
                TAU,
            )
            .unwrap();
            let edges = check(&topo, shell, exact, 1.0, name);
            assert_eq!(faces(&topo, shell).len(), 3, "{name}");
            assert_eq!(count(&topo, shell, is_cylinder), 1, "{name}");
            assert_eq!(count(&topo, shell, is_torus), 1, "{name}");
            assert_eq!(count(&topo, shell, is_plane), 1, "{name}");
            assert_eq!(
                edges,
                Census {
                    boundary: 1,
                    interior: 2,
                    seams: 2
                },
                "{name}"
            );
        }
    }

    #[test]
    fn a_partial_turn_sweeps_its_share_with_a_face_per_quarter() {
        let whole = 58.0 * PI + 4.0 * PI * PI;
        for (angle, per_edge) in [(FRAC_PI_2, 1), (PI, 2), (1.5 * PI, 3), (0.3, 1)] {
            for (name, pose) in poses() {
                let mut topo = Topology::new();
                let wire = wall_fillet_disc(&mut topo, pose);
                let shell = revolve_wire(
                    &mut topo,
                    wire,
                    pose.p(0.0, 0.0, 0.0),
                    pose.v(0.0, 0.0, 1.0),
                    angle,
                )
                .unwrap();
                assert_eq!(faces(&topo, shell).len(), 3 * per_edge, "{name}");
                check(&topo, shell, whole * angle / TAU, 1.0, name);
            }
        }
    }

    #[test]
    fn the_issue_wire_revolved_part_way_sweeps_its_share_of_the_sphere() {
        for (angle, lunes) in [(FRAC_PI_2, 1), (PI, 2), (1.5 * PI, 3)] {
            for (name, pose) in poses() {
                let mut topo = Topology::new();
                let wire = issue_wire(&mut topo, pose);
                let shell = revolve_wire(
                    &mut topo,
                    wire,
                    pose.p(0.0, 0.0, 0.0),
                    pose.v(1.0, 0.0, 0.0),
                    angle,
                )
                .unwrap();
                assert_eq!(count(&topo, shell, is_sphere), lunes, "{name}");
                check(&topo, shell, 100.0 * PI * angle / TAU, 5.0, name);
            }
        }
    }

    #[test]
    fn a_line_skew_to_the_axis_sweeps_a_hyperboloid_not_a_cone() {
        let exact = TAU * simpson(|z| (1.0 + 2.0 * z * z).sqrt(), 0.0, 1.0, 2000);
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let wire = open_wire(
                &mut topo,
                pose.p(1.0, 0.0, 0.0),
                vec![Run::Line(pose.p(1.0, 1.0, 1.0))],
            );
            let shell = revolve_wire(
                &mut topo,
                wire,
                pose.p(0.0, 0.0, 0.0),
                pose.v(0.0, 0.0, 1.0),
                TAU,
            )
            .unwrap();
            assert_eq!(count(&topo, shell, is_nurbs), 4, "{name}");
            let edges = check(&topo, shell, exact, 1.0, name);
            assert_eq!(
                edges,
                Census {
                    boundary: 8,
                    interior: 4,
                    seams: 0
                },
                "{name}"
            );
        }
    }

    #[test]
    fn an_arc_in_a_plane_beside_the_axis_sweeps_an_exact_rational_band() {
        // A half circle of radius 2 about (6, 2, 0) in a plane tilted across the
        // axis. Each point sweeps its own circle, so the area is the meridian's:
        // 2 pi times the integral of r ds over (r, z).
        let (center, u, v, r) = (
            Vec3::new(6.0, 2.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.6, 0.8),
            2.0,
        );
        let at = |t: f64| center + u * (r * t.cos()) + v * (r * t.sin());
        let radial = |p: Vec3| (p.x() * p.x() + p.y() * p.y()).sqrt();
        let element = |t: f64| {
            let h = 1e-5;
            let (a, b) = (at(t - h), at(t + h));
            let (dr, dz) = (radial(b) - radial(a), b.z() - a.z());
            radial(at(t)) * (dr * dr + dz * dz).sqrt() / (2.0 * h)
        };
        let exact = TAU * simpson(element, 0.0, PI, 4000);
        let point = |t: f64| {
            let p = at(t);
            Point3::new(p.x(), p.y(), p.z())
        };
        let mut topo = Topology::new();
        let wire = open_wire(
            &mut topo,
            point(0.0),
            vec![Run::Arc(point(FRAC_PI_2), point(PI))],
        );
        let shell = revolve_wire(
            &mut topo,
            wire,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            TAU,
        )
        .unwrap();
        assert_eq!(count(&topo, shell, is_nurbs), 4);
        check(&topo, shell, exact, 1.0, "arc beside the axis");
    }

    #[test]
    fn a_line_along_the_axis_between_two_discs_sweeps_nothing() {
        let mut topo = Topology::new();
        let wire = open_wire(
            &mut topo,
            Point3::new(3.0, 0.0, 0.0),
            vec![
                Run::Line(Point3::new(0.0, 0.0, 0.0)),
                Run::Line(Point3::new(0.0, 0.0, 4.0)),
                Run::Line(Point3::new(3.0, 0.0, 4.0)),
            ],
        );
        let shell = revolve_wire(
            &mut topo,
            wire,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            TAU,
        )
        .unwrap();
        assert_eq!(count(&topo, shell, is_plane), 2);
        let area: f64 = faces(&topo, shell)
            .iter()
            .map(|&f| face_area(&topo, f, DEFLECTION).unwrap())
            .sum();
        close(area, 18.0 * PI, 1e-9, "two discs");
    }

    #[test]
    fn an_edge_across_the_axis_is_refused_and_split_on_it_sweeps_both_nappes() {
        let origin = Point3::new(0.0, 0.0, 0.0);
        let z = Vec3::new(0.0, 0.0, 1.0);
        let (below, apex, above) = (
            Point3::new(-1.0, 0.0, -1.0),
            origin,
            Point3::new(2.0, 0.0, 2.0),
        );
        let mut topo = Topology::new();
        let across = open_wire(&mut topo, below, vec![Run::Line(above)]);
        refused(revolve_wire(&mut topo, across, origin, z, TAU), "crosses");
        refused(revolve_wire(&mut topo, across, origin, z, PI), "crosses");
        let over_the_pole = open_wire(
            &mut topo,
            Point3::new(3.0, 0.0, 4.0),
            vec![Run::Arc(
                Point3::new(0.0, 0.0, 5.0),
                Point3::new(-3.0, 0.0, 4.0),
            )],
        );
        refused(
            revolve_wire(&mut topo, over_the_pole, origin, z, TAU),
            "crosses",
        );

        let exact = PI * 2f64.sqrt() + 4.0 * PI * 2f64.sqrt();
        let split = open_wire(&mut topo, below, vec![Run::Line(apex), Run::Line(above)]);
        let shell = revolve_wire(&mut topo, split, origin, z, TAU).unwrap();
        assert_eq!(count(&topo, shell, is_cone), 2);
        measure(&topo, shell, exact, 1.0, "two nappes");
        assert_eq!(
            census(&topo, shell),
            Census {
                boundary: 2,
                interior: 0,
                seams: 2
            }
        );
        // The nappes share only their apex: no edge joins them.
        let options = ValidateOptions {
            disabled_checks: HashSet::from([
                CheckId::ShellClosed,
                CheckId::WireSelfIntersection,
                CheckId::ShellConnected,
            ]),
            ..ValidateOptions::default()
        };
        let report = validate_shell(&topo, shell, &options).unwrap();
        assert!(report.is_valid(), "{:?}", report.issues);
    }

    #[test]
    fn a_line_passing_near_the_axis_is_not_across_it() {
        // Skew by 0.1: its waist sweeps a narrow neck, and its ends lie on
        // opposite sides of the axis.
        let exact = TAU
            * simpson(
                |z| {
                    let t = z;
                    let r = (0.01 + t * t).sqrt();
                    r * (t * t / (r * r) + 1.0).sqrt()
                },
                -1.0,
                1.0,
                4000,
            );
        let mut topo = Topology::new();
        let wire = open_wire(
            &mut topo,
            Point3::new(-1.0, 0.1, -1.0),
            vec![Run::Line(Point3::new(1.0, 0.1, 1.0))],
        );
        let shell = revolve_wire(
            &mut topo,
            wire,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            TAU,
        )
        .unwrap();
        assert_eq!(count(&topo, shell, is_nurbs), 4);
        check(&topo, shell, exact, 0.1, "narrow neck");
    }

    #[test]
    fn a_chain_with_a_spline_revolves_in_quarters_all_along() {
        let spline = cubic_spline([
            Point3::new(3.0, 0.0, 4.0),
            Point3::new(4.0, 0.0, 5.0),
            Point3::new(5.0, 0.0, 6.0),
            Point3::new(4.0, 0.0, 8.0),
        ]);
        let mut topo = Topology::new();
        let wire = open_wire(
            &mut topo,
            Point3::new(3.0, 0.0, 0.0),
            vec![
                Run::Line(Point3::new(3.0, 0.0, 4.0)),
                Run::Curve(
                    EdgeCurve::NurbsCurve(spline.clone()),
                    Point3::new(4.0, 0.0, 8.0),
                ),
            ],
        );
        let shell = revolve_wire(
            &mut topo,
            wire,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            TAU,
        )
        .unwrap();
        assert_eq!(faces(&topo, shell).len(), 8);
        assert_eq!(count(&topo, shell, is_cylinder), 4);
        assert_eq!(count(&topo, shell, is_nurbs), 4);
        let band = simpson(
            |t| {
                let p = spline.evaluate(t);
                let d = spline.derivatives(t, 1)[1];
                TAU * p.x() * d.length()
            },
            0.0,
            1.0,
            4000,
        );
        let edges = check(&topo, shell, TAU * 3.0 * 4.0 + band, 1.0, "wall and spline");
        assert_eq!(
            edges,
            Census {
                boundary: 8,
                interior: 12,
                seams: 0
            }
        );
    }

    #[test]
    fn revolve_refuses_a_bad_turn_or_a_wire_on_its_axis() {
        let mut topo = Topology::new();
        let wire = issue_wire(&mut topo, upright());
        let origin = Point3::new(0.0, 0.0, 0.0);
        let x = Vec3::new(1.0, 0.0, 0.0);
        refused(
            revolve_wire(&mut topo, wire, origin, Vec3::new(0.0, 0.0, 0.0), PI),
            "zero",
        );
        refused(revolve_wire(&mut topo, wire, origin, x, 0.0), "angle");
        refused(revolve_wire(&mut topo, wire, origin, x, 7.0), "angle");
        let on_axis = open_wire(
            &mut topo,
            origin,
            vec![Run::Line(Point3::new(10.0, 0.0, 0.0))],
        );
        refused(
            revolve_wire(&mut topo, on_axis, origin, x, TAU),
            "lies along the axis",
        );
    }

    #[test]
    fn a_dome_keeps_one_orientation_whichever_way_its_wire_runs() {
        let root = 0.5f64.sqrt();
        let exact = 90.0 * PI;
        for upward in [true, false] {
            for (name, pose) in poses() {
                let mut topo = Topology::new();
                let wire = if upward {
                    open_wire(
                        &mut topo,
                        pose.p(5.0, 0.0, 0.0),
                        vec![
                            Run::Line(pose.p(5.0, 0.0, 4.0)),
                            Run::Arc(
                                pose.p(5.0 * root, 0.0, 4.0 + 5.0 * root),
                                pose.p(0.0, 0.0, 9.0),
                            ),
                        ],
                    )
                } else {
                    open_wire(
                        &mut topo,
                        pose.p(0.0, 0.0, 9.0),
                        vec![
                            Run::Arc(
                                pose.p(5.0 * root, 0.0, 4.0 + 5.0 * root),
                                pose.p(5.0, 0.0, 4.0),
                            ),
                            Run::Line(pose.p(5.0, 0.0, 0.0)),
                        ],
                    )
                };
                let shell = revolve_wire(
                    &mut topo,
                    wire,
                    pose.p(0.0, 0.0, 0.0),
                    pose.v(0.0, 0.0, 1.0),
                    TAU,
                )
                .unwrap();
                assert_eq!(count(&topo, shell, is_cylinder), 1, "{name}");
                assert_eq!(count(&topo, shell, is_sphere), 1, "{name}");
                let edges = check(&topo, shell, exact, 5.0, name);
                assert_eq!(
                    edges,
                    Census {
                        boundary: 1,
                        interior: 1,
                        seams: 1
                    },
                    "{name}"
                );
                assert_faces_turn_one_way_at(&topo, shell, pose.p(5.0, 0.0, 4.0), name);
            }
        }
    }

    #[test]
    fn a_fillet_chain_keeps_one_orientation_all_along() {
        for angle in [TAU, 0.75 * TAU] {
            for (name, pose) in poses() {
                let mut topo = Topology::new();
                let wire = wall_fillet_disc(&mut topo, pose);
                let shell = revolve_wire(
                    &mut topo,
                    wire,
                    pose.p(0.0, 0.0, 0.0),
                    pose.v(0.0, 0.0, 1.0),
                    angle,
                )
                .unwrap();
                assert_faces_turn_one_way_at(&topo, shell, pose.p(5.0, 0.0, 4.0), name);
                assert_faces_turn_one_way_at(&topo, shell, pose.p(4.0, 0.0, 5.0), name);
            }
        }
    }

    #[test]
    fn sphere_zones_and_caps_sweep_their_share_of_a_partial_turn() {
        let root = 0.5f64.sqrt();
        let through = (
            5.0 * (1.5f64.atan2(4.5)).cos(),
            5.0 * (1.5f64.atan2(4.5)).sin(),
        );
        let cases = [
            ((5.0, 0.0), through, (4.0, 3.0), 30.0 * PI),
            (
                (0.0, 5.0),
                (5.0 * root.sin(), 5.0 * root.cos()),
                (4.0, 3.0),
                20.0 * PI,
            ),
        ];
        for (from, through, to, whole) in cases {
            for angle in [0.4, FRAC_PI_2, PI, 1.75 * PI] {
                for (name, pose) in poses() {
                    let mut topo = Topology::new();
                    let wire = open_wire(
                        &mut topo,
                        pose.p(from.0, 0.0, from.1),
                        vec![Run::Arc(
                            pose.p(through.0, 0.0, through.1),
                            pose.p(to.0, 0.0, to.1),
                        )],
                    );
                    let shell = revolve_wire(
                        &mut topo,
                        wire,
                        pose.p(0.0, 0.0, 0.0),
                        pose.v(0.0, 0.0, 1.0),
                        angle,
                    )
                    .unwrap();
                    assert_eq!(count(&topo, shell, is_sphere), faces(&topo, shell).len());
                    check(&topo, shell, whole * angle / TAU, 5.0, name);
                }
            }
        }
    }

    #[test]
    fn an_arc_too_near_its_axis_for_a_torus_sweeps_an_exact_rational_band() {
        // Circle of radius 2 centred 1 from the axis, the part beyond that
        // centre: Pappus by slices gives 2 pi * 2 (pi + 4).
        let exact = 4.0 * PI * (PI + 4.0);
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let wire = open_wire(
                &mut topo,
                pose.p(1.0, 0.0, -2.0),
                vec![Run::Arc(pose.p(3.0, 0.0, 0.0), pose.p(1.0, 0.0, 2.0))],
            );
            let shell = revolve_wire(
                &mut topo,
                wire,
                pose.p(0.0, 0.0, 0.0),
                pose.v(0.0, 0.0, 1.0),
                TAU,
            )
            .unwrap();
            assert_eq!(count(&topo, shell, is_nurbs), 4, "{name}");
            check(&topo, shell, exact, 2.0, name);
        }
    }

    #[test]
    fn an_ellipse_arc_sweeps_half_a_spheroid() {
        let (a, b) = (6.0f64, 3.0f64);
        let e = (1.0 - b * b / (a * a)).sqrt();
        let exact = PI * a * a * (1.0 + (1.0 - e * e) / e * e.atanh());
        let mut topo = Topology::new();
        let ellipse = Ellipse3D::with_axes(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            a,
            b,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .unwrap();
        let wire = open_wire(
            &mut topo,
            Point3::new(a, 0.0, 0.0),
            vec![Run::Curve(
                EdgeCurve::Ellipse(ellipse),
                Point3::new(0.0, 0.0, b),
            )],
        );
        let shell = revolve_wire(
            &mut topo,
            wire,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            TAU,
        )
        .unwrap();
        assert_eq!(count(&topo, shell, is_nurbs), 4);
        check(&topo, shell, exact, 1.0, "half a spheroid");
    }

    /// The issue's wire with both edges stored against the way it runs.
    fn issue_wire_stored_backward(topo: &mut Topology, pose: Pose) -> WireId {
        let (a, b, c) = (
            pose.p(0.0, 0.0, 0.0),
            pose.p(10.0, 0.0, 0.0),
            pose.p(20.0, 0.0, 0.0),
        );
        let [va, vb, vc] = [a, b, c].map(|p| topo.add_vertex(Vertex::new(p, TOL)));
        let line = topo.add_edge(Edge::new(vb, va, EdgeCurve::Line));
        let bend = topo.add_edge(Edge::new(vc, vb, arc(c, pose.p(15.0, 5.0, 0.0), b)));
        topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(line, false),
                    OrientedEdge::new(bend, false),
                ],
                false,
            )
            .unwrap(),
        )
    }

    #[test]
    fn edges_stored_against_the_wire_sweep_alike() {
        let length = 10.0 + 5.0 * PI;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let wire = issue_wire_stored_backward(&mut topo, pose);
            let shell = extrude_wire(&mut topo, wire, pose.v(0.0, 0.0, 1.0), 20.0).unwrap();
            assert_eq!(check(&topo, shell, 20.0 * length, 5.0, name).interior, 1);

            let mut topo = Topology::new();
            let wire = issue_wire_stored_backward(&mut topo, pose);
            for (angle, share) in [(TAU, 1.0), (PI, 0.5)] {
                let shell = revolve_wire(
                    &mut topo,
                    wire,
                    pose.p(0.0, 0.0, 0.0),
                    pose.v(1.0, 0.0, 0.0),
                    angle,
                )
                .unwrap();
                check(&topo, shell, 100.0 * PI * share, 5.0, name);
            }

            let mut topo = Topology::new();
            let wire = issue_wire_stored_backward(&mut topo, pose);
            let other = issue_wire(&mut topo, Pose(pose.0 * Mat4::translation(0.0, 0.0, 7.0)));
            let shell = loft_wires(&mut topo, &[wire, other]).unwrap();
            assert_eq!(check(&topo, shell, 7.0 * length, 5.0, name).interior, 1);

            let mut topo = Topology::new();
            let wire = issue_wire_stored_backward(&mut topo, pose);
            let edges = run_then_turn(
                &mut topo,
                Pose(pose.0 * Mat4::rotation_x(FRAC_PI_2) * Mat4::rotation_y(-FRAC_PI_2)),
            );
            let shell = sweep_wire(&mut topo, wire, &edges).unwrap();
            assert_eq!(faces(&topo, shell).len(), 2, "{name}");
            assert_eq!(census(&topo, shell).interior, 1, "{name}");
            assert_edges_on_surfaces(&topo, shell, name);
        }
    }

    // Loft.

    #[test]
    fn the_issue_lines_loft_to_a_flat_rectangle() {
        let exact = 10.0 * 125f64.sqrt();
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let low = open_wire(
                &mut topo,
                pose.p(0.0, 0.0, 0.0),
                vec![Run::Line(pose.p(10.0, 0.0, 0.0))],
            );
            let high = open_wire(
                &mut topo,
                pose.p(0.0, 5.0, 10.0),
                vec![Run::Line(pose.p(10.0, 5.0, 10.0))],
            );
            let shell = loft_wires(&mut topo, &[low, high]).unwrap();
            assert_eq!(count(&topo, shell, is_plane), 1, "{name}");
            let edges = check(&topo, shell, exact, 1.0, name);
            assert_eq!(
                edges,
                Census {
                    boundary: 4,
                    interior: 0,
                    seams: 0
                },
                "{name}"
            );
        }
    }

    #[test]
    fn three_wires_of_two_edges_loft_two_faces_per_edge() {
        let exact = 2.0 + 2.0 * 4.0;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let mut wire_at = |y: f64, z: f64| {
                open_wire(
                    &mut topo,
                    pose.p(0.0, y, z),
                    vec![Run::Line(pose.p(1.0, y, z)), Run::Line(pose.p(2.0, y, z))],
                )
            };
            let wires = [wire_at(0.0, 0.0), wire_at(1.0, 0.0), wire_at(1.0, 4.0)];
            let shell = loft_wires(&mut topo, &wires).unwrap();
            assert_eq!(faces(&topo, shell).len(), 4, "{name}");
            let edges = check(&topo, shell, exact, 1.0, name);
            assert_eq!(
                edges,
                Census {
                    boundary: 8,
                    interior: 4,
                    seams: 0
                },
                "{name}"
            );
        }
    }

    #[test]
    fn a_twisted_pair_of_lines_lofts_a_bilinear_patch() {
        let mut topo = Topology::new();
        let low = open_wire(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            vec![Run::Line(Point3::new(1.0, 0.0, 0.0))],
        );
        let high = open_wire(
            &mut topo,
            Point3::new(0.0, 1.0, 0.0),
            vec![Run::Line(Point3::new(1.0, 1.0, 1.0))],
        );
        let shell = loft_wires(&mut topo, &[low, high]).unwrap();
        assert_eq!(count(&topo, shell, is_nurbs), 1);
        // P(u, v) = (u, v, u v): the area element is sqrt(1 + u² + v²).
        let exact = simpson(
            |u| simpson(|v| (1.0 + u * u + v * v).sqrt(), 0.0, 1.0, 200),
            0.0,
            1.0,
            200,
        );
        check(&topo, shell, exact, 1.0, "twisted lines");
    }

    #[test]
    fn arcs_loft_exact_rulings_between_them() {
        let root = 0.5f64.sqrt();
        for (name, pose) in poses() {
            // Equal arcs one above the other: a quarter of a cylinder wall.
            let mut topo = Topology::new();
            let mut quarter = |r: f64, z: f64| {
                open_wire(
                    &mut topo,
                    pose.p(r, 0.0, z),
                    vec![Run::Arc(pose.p(r * root, r * root, z), pose.p(0.0, r, z))],
                )
            };
            let (low, high) = (quarter(2.0, 0.0), quarter(2.0, 3.0));
            let shell = loft_wires(&mut topo, &[low, high]).unwrap();
            check(&topo, shell, 3.0 * PI, 2.0, name);

            // Coaxial arcs of radii 1 and 2 one apart: a quarter of a frustum.
            let mut topo = Topology::new();
            let mut quarter = |r: f64, z: f64| {
                open_wire(
                    &mut topo,
                    pose.p(r, 0.0, z),
                    vec![Run::Arc(pose.p(r * root, r * root, z), pose.p(0.0, r, z))],
                )
            };
            let (low, high) = (quarter(1.0, 0.0), quarter(2.0, 1.0));
            let shell = loft_wires(&mut topo, &[low, high]).unwrap();
            check(&topo, shell, 0.75 * PI * 2f64.sqrt(), 1.0, name);
        }
    }

    #[test]
    fn a_line_lofts_to_an_arc_past_a_quarter_turn() {
        let mut topo = Topology::new();
        let low = open_wire(
            &mut topo,
            Point3::new(-2.0, 0.0, 0.0),
            vec![Run::Line(Point3::new(2.0, 0.0, 0.0))],
        );
        let high = open_wire(
            &mut topo,
            Point3::new(-2.0, 0.0, 3.0),
            vec![Run::Arc(
                Point3::new(0.0, 2.0, 3.0),
                Point3::new(2.0, 0.0, 3.0),
            )],
        );
        let shell = loft_wires(&mut topo, &[low, high]).unwrap();
        assert_eq!(count(&topo, shell, is_nurbs), 1);
        let face = faces(&topo, shell)[0];
        let FaceSurface::Nurbs(surface) = topo.face(face).unwrap().surface() else {
            panic!("a ruled patch");
        };
        let ((u0, u1), (v0, v1)) = (surface.domain_u(), surface.domain_v());
        let exact = simpson(
            |u| {
                simpson(
                    |v| {
                        let d = surface.derivatives(u, v, 1);
                        d[1][0].cross(d[0][1]).length()
                    },
                    v0,
                    v1,
                    60,
                )
            },
            u0,
            u1,
            240,
        );
        check(&topo, shell, exact, 1.0, "line to arc");
    }

    #[test]
    fn a_tangent_chain_lofts_and_sweeps_with_one_orientation() {
        let root = 0.5f64.sqrt();
        for (name, pose) in poses() {
            let tangent_chain = |topo: &mut Topology, z: f64| {
                open_wire(
                    topo,
                    pose.p(0.0, 0.0, z),
                    vec![
                        Run::Line(pose.p(10.0, 0.0, z)),
                        Run::Arc(
                            pose.p(10.0 + 2.0 * root, 2.0 - 2.0 * root, z),
                            pose.p(12.0, 2.0, z),
                        ),
                    ],
                )
            };
            let mut topo = Topology::new();
            let (low, high) = (tangent_chain(&mut topo, 0.0), tangent_chain(&mut topo, 3.0));
            let shell = loft_wires(&mut topo, &[low, high]).unwrap();
            assert_faces_turn_one_way_at(&topo, shell, pose.p(10.0, 0.0, 1.5), name);
            check(&topo, shell, 3.0 * (10.0 + PI), 2.0, name);

            let mut topo = Topology::new();
            let wire = tangent_chain(&mut topo, 0.0);
            let root2 = 0.5f64.sqrt();
            let edges = path(
                &mut topo,
                vec![
                    (pose.p(0.0, 0.0, 0.0), None, pose.p(0.0, 0.0, 6.0), false),
                    (
                        pose.p(0.0, 0.0, 6.0),
                        Some(pose.p(0.0, 20.0 - 20.0 * root2, 6.0 + 20.0 * root2)),
                        pose.p(0.0, 20.0, 26.0),
                        false,
                    ),
                ],
            );
            let shell = sweep_wire(&mut topo, wire, &edges).unwrap();
            assert_eq!(census(&topo, shell).interior, 1, "{name}");
            assert_faces_turn_one_way_at(&topo, shell, pose.p(10.0, 0.0, 3.0), name);
        }
    }

    #[test]
    fn loft_refuses_what_it_cannot_pair() {
        let mut topo = Topology::new();
        let one = open_wire(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            vec![Run::Line(Point3::new(1.0, 0.0, 0.0))],
        );
        let two = open_wire(
            &mut topo,
            Point3::new(0.0, 0.0, 1.0),
            vec![
                Run::Line(Point3::new(1.0, 0.0, 1.0)),
                Run::Line(Point3::new(2.0, 0.0, 1.0)),
            ],
        );
        refused(loft_wires(&mut topo, &[]), "at least two");
        refused(loft_wires(&mut topo, &[one]), "at least two");
        refused(loft_wires(&mut topo, &[one, two]), "edges");

        let touching = open_wire(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            vec![Run::Line(Point3::new(0.0, 1.0, 0.0))],
        );
        refused(loft_wires(&mut topo, &[one, touching]), "touch");

        let backward = open_wire(
            &mut topo,
            Point3::new(1.0, 2.0, 0.0),
            vec![Run::Line(Point3::new(0.0, 2.0, 0.0))],
        );
        refused(loft_wires(&mut topo, &[one, backward]), "cross");

        // A reflex corner folds the ruled patch over itself, and a line ending
        // in the middle of the other doubles its boundary back.
        let reflex_low = open_wire(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            vec![Run::Line(Point3::new(2.0, 0.0, 0.0))],
        );
        let reflex_high = open_wire(
            &mut topo,
            Point3::new(0.0, 2.0, 0.0),
            vec![Run::Line(Point3::new(0.5, 0.5, 0.0))],
        );
        refused(loft_wires(&mut topo, &[reflex_low, reflex_high]), "cross");
        let across = open_wire(
            &mut topo,
            Point3::new(2.0, -1.0, 0.0),
            vec![Run::Line(Point3::new(2.0, 1.0, 0.0))],
        );
        refused(loft_wires(&mut topo, &[reflex_low, across]), "cross");
    }

    /// A curve that misses its edge's vertices would leave the shell's boundary
    /// apart from its topology.
    #[test]
    fn every_op_refuses_an_edge_off_its_vertices() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let mut topo = Topology::new();
        let a = topo.add_vertex(Vertex::new(Point3::new(5.0, 0.0, 0.0), TOL));
        let b = topo.add_vertex(Vertex::new(Point3::new(-9.0, 0.0, 0.0), TOL));
        let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), up, 5.0).unwrap();
        let edge = topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle)));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], false).unwrap());
        refused(extrude_wire(&mut topo, wire, up, 5.0), "reach");
    }

    // Sweep.

    #[test]
    fn the_issue_wire_sweeps_a_straight_path_to_its_length_times_the_path() {
        let exact = 30.0 * (10.0 + 5.0 * PI);
        close(exact, 771.239, 1e-3, "the issue's own number");
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let wire = issue_wire(&mut topo, pose);
            let (a, b) = (
                topo.add_vertex(Vertex::new(pose.p(0.0, 0.0, 0.0), TOL)),
                topo.add_vertex(Vertex::new(pose.p(0.0, 0.0, 30.0), TOL)),
            );
            let path = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
            let shell = sweep_wire(&mut topo, wire, &[path]).unwrap();
            let edges = check(&topo, shell, exact, 5.0, name);
            assert_eq!(count(&topo, shell, is_plane), 1, "{name}");
            assert_eq!(count(&topo, shell, is_cylinder), 1, "{name}");
            assert_eq!(edges.interior, 1, "{name}");
        }
    }

    /// The chain's edges, each stored as `flip` says.
    fn path(topo: &mut Topology, runs: Vec<(Point3, Option<Point3>, Point3, bool)>) -> Vec<EdgeId> {
        let mut edges = Vec::with_capacity(runs.len());
        let mut tail: Option<(Point3, brepkit_topology::vertex::VertexId)> = None;
        for (from, through, to, flip) in runs {
            let start = match tail {
                Some((at, v)) if (at - from).length() < TOL => v,
                _ => topo.add_vertex(Vertex::new(from, TOL)),
            };
            let end = topo.add_vertex(Vertex::new(to, TOL));
            let curve = through.map_or(EdgeCurve::Line, |m| {
                if flip {
                    arc(to, m, from)
                } else {
                    arc(from, m, to)
                }
            });
            edges.push(topo.add_edge(if flip {
                Edge::new(end, start, curve)
            } else {
                Edge::new(start, end, curve)
            }));
            tail = Some((to, end));
        }
        edges
    }

    #[test]
    fn collinear_path_edges_sweep_as_one_straight_run() {
        let mut topo = Topology::new();
        let wire = issue_wire(&mut topo, upright());
        let edges = path(
            &mut topo,
            vec![
                (
                    Point3::new(0.0, 0.0, 0.0),
                    None,
                    Point3::new(0.0, 0.0, 10.0),
                    true,
                ),
                (
                    Point3::new(0.0, 0.0, 10.0),
                    None,
                    Point3::new(0.0, 0.0, 30.0),
                    false,
                ),
            ],
        );
        let shell = sweep_wire(&mut topo, wire, &edges).unwrap();
        check(
            &topo,
            shell,
            30.0 * (10.0 + 5.0 * PI),
            5.0,
            "collinear path",
        );
        assert_eq!(count(&topo, shell, is_plane), 1);
    }

    #[test]
    fn a_single_arc_path_sweeps_exact_analytic_bands() {
        let root = 0.5f64.sqrt();
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let wire = open_wire(
                &mut topo,
                pose.p(5.0, 0.0, 0.0),
                vec![Run::Line(pose.p(5.0, 0.0, 3.0))],
            );
            let edges = path(
                &mut topo,
                vec![(
                    pose.p(5.0, 0.0, 0.0),
                    Some(pose.p(5.0 * root, 5.0 * root, 0.0)),
                    pose.p(0.0, 5.0, 0.0),
                    false,
                )],
            );
            let shell = sweep_wire(&mut topo, wire, &edges).unwrap();
            assert_eq!(
                count(&topo, shell, is_cylinder),
                faces(&topo, shell).len(),
                "{name}"
            );
            check(&topo, shell, 3.0 * 2.5 * PI, 5.0, name);
        }
    }

    #[test]
    fn a_spline_drawn_arc_path_sweeps_exact_analytic_bands() {
        let mut topo = Topology::new();
        let wire = open_wire(
            &mut topo,
            Point3::new(5.0, 0.0, 0.0),
            vec![Run::Line(Point3::new(5.0, 0.0, 3.0))],
        );
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();
        let (t0, t1) = EdgeCurve::Circle(circle.clone())
            .domain_with_endpoints(Point3::new(5.0, 0.0, 0.0), Point3::new(0.0, 5.0, 0.0));
        let nurbs = brepkit_geometry::convert::circle_to_nurbs(&circle, t0, t1).unwrap();
        let (a, b) = (
            topo.add_vertex(Vertex::new(Point3::new(5.0, 0.0, 0.0), TOL)),
            topo.add_vertex(Vertex::new(Point3::new(0.0, 5.0, 0.0), TOL)),
        );
        let edge = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(nurbs)));
        let shell = sweep_wire(&mut topo, wire, &[edge]).unwrap();
        assert_eq!(count(&topo, shell, is_cylinder), faces(&topo, shell).len());
        check(&topo, shell, 3.0 * 2.5 * PI, 5.0, "spline arc path");
    }

    #[test]
    fn a_full_circle_path_closes_up_on_its_seam() {
        let mut topo = Topology::new();
        let wire = open_wire(
            &mut topo,
            Point3::new(5.0, 0.0, 0.0),
            vec![Run::Line(Point3::new(5.0, 0.0, 3.0))],
        );
        let edge = brepkit_topology::builder::make_circle_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            TOL,
        )
        .unwrap();
        let shell = sweep_wire(&mut topo, wire, &[edge]).unwrap();
        let edges = check(&topo, shell, 3.0 * TAU * 5.0, 5.0, "full circle path");
        assert_eq!(
            edges,
            Census {
                boundary: 2,
                interior: 0,
                seams: 1
            }
        );
    }

    /// A straight run of 10 then a quarter circle of radius 5 turning left.
    fn run_then_turn(topo: &mut Topology, pose: Pose) -> Vec<EdgeId> {
        let root = 0.5f64.sqrt();
        path(
            topo,
            vec![
                (pose.p(0.0, 0.0, 0.0), None, pose.p(10.0, 0.0, 0.0), false),
                (
                    pose.p(10.0, 0.0, 0.0),
                    Some(pose.p(10.0 + 5.0 * root, 5.0 - 5.0 * root, 0.0)),
                    pose.p(15.0, 5.0, 0.0),
                    false,
                ),
            ],
        )
    }

    #[test]
    fn lines_sweep_a_run_then_a_turn_to_their_closed_forms() {
        // Up the z axis, the wire walls off the path: 3 tall all along it.
        // Across it, toward the turn's centre: a rectangle and a ring's quarter.
        let exact = 3.0 * (10.0 + 2.5 * PI) + (20.0 + 4.0 * PI);
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let wire = open_wire(
                &mut topo,
                pose.p(0.0, 2.0, 0.0),
                vec![
                    Run::Line(pose.p(0.0, 0.0, 0.0)),
                    Run::Line(pose.p(0.0, 0.0, 3.0)),
                ],
            );
            let edges = run_then_turn(&mut topo, pose);
            let shell = sweep_wire(&mut topo, wire, &edges).unwrap();
            assert_eq!(faces(&topo, shell).len(), 2, "{name}");
            let edges = check(&topo, shell, exact, 3.0, name);
            assert_eq!(
                edges,
                Census {
                    boundary: 6,
                    interior: 1,
                    seams: 0
                },
                "{name}"
            );
        }
    }

    #[test]
    fn path_edges_stored_against_the_chain_read_along_it() {
        let root = 0.5f64.sqrt();
        let exact = 3.0 * (10.0 + 2.5 * PI);
        let mut topo = Topology::new();
        let wire = open_wire(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            vec![Run::Line(Point3::new(0.0, 0.0, 3.0))],
        );
        let edges = path(
            &mut topo,
            vec![
                (
                    Point3::new(0.0, 0.0, 0.0),
                    None,
                    Point3::new(10.0, 0.0, 0.0),
                    true,
                ),
                (
                    Point3::new(10.0, 0.0, 0.0),
                    Some(Point3::new(10.0 + 5.0 * root, 5.0 - 5.0 * root, 0.0)),
                    Point3::new(15.0, 5.0, 0.0),
                    true,
                ),
            ],
        );
        let shell = sweep_wire(&mut topo, wire, &edges).unwrap();
        check(&topo, shell, exact, 3.0, "stored backward path");
    }

    #[test]
    fn an_arc_sweeps_a_run_then_a_turn_by_pappus() {
        // A half circle of radius 1 about the path, bulging toward the turn's
        // centre: its centroid runs 2/pi inside the path around the turn.
        let exact = 10.0 * PI + PI * FRAC_PI_2 * (5.0 - 2.0 / PI);
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let wire = open_wire(
                &mut topo,
                pose.p(0.0, 0.0, -1.0),
                vec![Run::Arc(pose.p(0.0, 1.0, 0.0), pose.p(0.0, 0.0, 1.0))],
            );
            let edges = run_then_turn(&mut topo, pose);
            let shell = sweep_wire(&mut topo, wire, &edges).unwrap();
            assert_eq!(count(&topo, shell, is_nurbs), 1, "{name}");
            check(&topo, shell, exact, 1.0, name);
        }
    }

    #[test]
    fn a_wire_along_the_binormal_sweeps_a_spline_path_to_height_times_length() {
        let spline = cubic_spline([
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(4.0, 6.0, 0.0),
            Point3::new(10.0, -6.0, 0.0),
            Point3::new(14.0, 0.0, 0.0),
        ]);
        let mut topo = Topology::new();
        let wire = open_wire(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            vec![Run::Line(Point3::new(0.0, 0.0, 2.0))],
        );
        let (a, b) = (
            topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL)),
            topo.add_vertex(Vertex::new(Point3::new(14.0, 0.0, 0.0), TOL)),
        );
        let edge = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(spline.clone())));
        let shell = sweep_wire(&mut topo, wire, &[edge]).unwrap();
        let samples: Vec<Point3> = (0..=20000)
            .map(|k| spline.evaluate(f64::from(k) / 20000.0))
            .collect();
        check(
            &topo,
            shell,
            2.0 * polyline_length(&samples),
            1.0,
            "spline path",
        );
    }

    #[test]
    fn sweep_refuses_a_kinked_or_closed_or_broken_path() {
        let mut topo = Topology::new();
        let wire = issue_wire(&mut topo, upright());
        refused(sweep_wire(&mut topo, wire, &[]), "no edges");
        let kinked = path(
            &mut topo,
            vec![
                (
                    Point3::new(0.0, 0.0, 0.0),
                    None,
                    Point3::new(0.0, 0.0, 10.0),
                    false,
                ),
                (
                    Point3::new(0.0, 0.0, 10.0),
                    None,
                    Point3::new(0.0, 10.0, 10.0),
                    false,
                ),
            ],
        );
        refused(sweep_wire(&mut topo, wire, &kinked), "kink");
        let broken = path(
            &mut topo,
            vec![
                (
                    Point3::new(0.0, 0.0, 0.0),
                    None,
                    Point3::new(0.0, 0.0, 10.0),
                    false,
                ),
                (
                    Point3::new(5.0, 0.0, 10.0),
                    None,
                    Point3::new(5.0, 0.0, 20.0),
                    false,
                ),
            ],
        );
        refused(sweep_wire(&mut topo, wire, &broken), "connect");
        let root = 0.5f64.sqrt();
        let closed = path(
            &mut topo,
            vec![
                (
                    Point3::new(0.0, 0.0, 0.0),
                    Some(Point3::new(root, 1.0 - root, 0.0)),
                    Point3::new(1.0, 1.0, 0.0),
                    false,
                ),
                (
                    Point3::new(1.0, 1.0, 0.0),
                    Some(Point3::new(1.0 - root, root, 0.0)),
                    Point3::new(0.0, 0.0, 0.0),
                    false,
                ),
            ],
        );
        refused(sweep_wire(&mut topo, wire, &closed), "closed");
        let (a, b) = (
            topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL)),
            topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL)),
        );
        let nowhere = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
        refused(sweep_wire(&mut topo, wire, &[nowhere]), "no length");
    }

    // Every operation, alike.

    fn every_op(topo: &mut Topology, wire: WireId) -> [Result<ShellId, OperationsError>; 4] {
        let other = open_wire(
            topo,
            Point3::new(100.0, 0.0, 0.0),
            vec![
                Run::Line(Point3::new(110.0, 0.0, 0.0)),
                Run::Line(Point3::new(120.0, 0.0, 0.0)),
            ],
        );
        let (a, b) = (
            topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL)),
            topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 30.0), TOL)),
        );
        let path = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
        [
            extrude_wire(topo, wire, Vec3::new(0.0, 0.0, 1.0), 5.0),
            revolve_wire(
                topo,
                wire,
                Point3::new(0.0, 0.0, -50.0),
                Vec3::new(1.0, 0.0, 0.0),
                PI,
            ),
            loft_wires(topo, &[wire, other]),
            sweep_wire(topo, wire, &[path]),
        ]
    }

    #[test]
    fn every_op_refuses_a_closed_wire_however_it_is_flagged() {
        for flag in [true, false] {
            let mut topo = Topology::new();
            let wire = {
                let points = [
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(4.0, 0.0, 0.0),
                    Point3::new(4.0, 4.0, 0.0),
                ];
                let vs = points.map(|p| topo.add_vertex(Vertex::new(p, TOL)));
                let edges = (0..3)
                    .map(|i| {
                        OrientedEdge::new(
                            topo.add_edge(Edge::new(vs[i], vs[(i + 1) % 3], EdgeCurve::Line)),
                            true,
                        )
                    })
                    .collect();
                topo.add_wire(Wire::new(edges, flag).unwrap())
            };
            for result in every_op(&mut topo, wire) {
                refused(result, "closed");
            }
        }
    }

    #[test]
    fn every_op_refuses_one_closed_edge() {
        let mut topo = Topology::new();
        let edge = brepkit_topology::builder::make_circle_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            TOL,
        )
        .unwrap();
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], false).unwrap());
        for result in every_op(&mut topo, wire) {
            refused(result, "closed");
        }
    }

    #[test]
    fn every_op_refuses_a_wire_that_meets_itself_or_breaks() {
        let mut topo = Topology::new();
        let touching = open_wire(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            vec![
                Run::Line(Point3::new(4.0, 0.0, 0.0)),
                Run::Line(Point3::new(4.0, 4.0, 0.0)),
                Run::Line(Point3::new(0.0, 0.0, 0.0)),
                Run::Line(Point3::new(0.0, -4.0, 0.0)),
            ],
        );
        for result in every_op(&mut topo, touching) {
            refused(result, "itself");
        }

        let (a, b, c, d) = (
            topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL)),
            topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), TOL)),
            topo.add_vertex(Vertex::new(Point3::new(2.0, 0.0, 0.0), TOL)),
            topo.add_vertex(Vertex::new(Point3::new(3.0, 0.0, 0.0), TOL)),
        );
        let first = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
        let second = topo.add_edge(Edge::new(c, d, EdgeCurve::Line));
        let broken = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(first, true),
                    OrientedEdge::new(second, true),
                ],
                false,
            )
            .unwrap(),
        );
        for result in every_op(&mut topo, broken) {
            refused(result, "connect");
        }
    }

    #[test]
    fn the_closed_flag_alone_does_not_close_a_wire() {
        let mut topo = Topology::new();
        let (a, b) = (
            topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL)),
            topo.add_vertex(Vertex::new(Point3::new(10.0, 0.0, 0.0), TOL)),
        );
        let edge = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
        let shell = extrude_wire(&mut topo, wire, Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();
        check(&topo, shell, 50.0, 1.0, "flagged closed, ends apart");
    }

    #[test]
    fn edges_ending_at_one_point_join_though_their_vertices_differ() {
        let mut topo = Topology::new();
        let (a, b, c, d) = (
            topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL)),
            topo.add_vertex(Vertex::new(Point3::new(10.0, 0.0, 0.0), TOL)),
            topo.add_vertex(Vertex::new(Point3::new(10.0, 0.0, 0.0), TOL)),
            topo.add_vertex(Vertex::new(Point3::new(20.0, 0.0, 0.0), TOL)),
        );
        let first = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
        let second = topo.add_edge(Edge::new(c, d, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(first, true),
                    OrientedEdge::new(second, true),
                ],
                false,
            )
            .unwrap(),
        );
        let shell = extrude_wire(&mut topo, wire, Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();
        let edges = check(&topo, shell, 100.0, 1.0, "unwelded chain");
        assert_eq!(
            edges,
            Census {
                boundary: 6,
                interior: 1,
                seams: 0
            }
        );
    }
}
