//! Sheets on every surface type in any pose, and continuity measured across a
//! seam with nothing shared with the code under test beyond the evaluators.

#![allow(dead_code)]

use std::f64::consts::FRAC_1_SQRT_2;

use brepkit_math::curves::Circle3D;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::fitting::interpolate;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::surfaces::{
    ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
};
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

/// A rigid motion, perhaps with a reflection.
#[derive(Debug, Clone, Copy)]
pub struct Pose {
    pub rows: [[f64; 3]; 3],
    pub shift: [f64; 3],
}

impl Pose {
    pub const fn upright() -> Self {
        Self {
            rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            shift: [0.0, 0.0, 0.0],
        }
    }

    /// A turn of `angle` about `axis`, then a shift.
    pub fn turn(axis: Vec3, angle: f64, shift: [f64; 3]) -> Self {
        let k = axis.normalize().unwrap();
        let (x, y, z) = (k.x(), k.y(), k.z());
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        Self {
            rows: [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ],
            shift,
        }
    }

    /// A pose off every axis.
    pub fn turned() -> Self {
        Self::turn(Vec3::new(1.0, 2.0, 3.0), 0.83, [3.0, -2.0, 5.0])
    }

    /// The turned pose seen in a mirror off every axis.
    pub fn mirrored() -> Self {
        Self::reflection(Point3::new(1.0, 0.5, -1.0), Vec3::new(0.6, 0.48, 0.64))
            .then(Self::turned())
    }

    /// Everything `factor` times the size.
    pub fn scaled(factor: f64) -> Self {
        Self {
            rows: [[factor, 0.0, 0.0], [0.0, factor, 0.0], [0.0, 0.0, factor]],
            shift: [0.0, 0.0, 0.0],
        }
    }

    /// The reflection in the plane through `at` with normal `normal`.
    pub fn reflection(at: Point3, normal: Vec3) -> Self {
        let n = normal.normalize().unwrap();
        let (x, y, z) = (n.x(), n.y(), n.z());
        let rows = [
            [1.0 - 2.0 * x * x, -2.0 * x * y, -2.0 * x * z],
            [-2.0 * y * x, 1.0 - 2.0 * y * y, -2.0 * y * z],
            [-2.0 * z * x, -2.0 * z * y, 1.0 - 2.0 * z * z],
        ];
        let d = 2.0 * (x * at.x() + y * at.y() + z * at.z());
        Self {
            rows,
            shift: [d * x, d * y, d * z],
        }
    }

    /// This pose, then `after`.
    pub fn then(self, after: Self) -> Self {
        let mut rows = [[0.0; 3]; 3];
        for (i, row) in rows.iter_mut().enumerate() {
            for (j, entry) in row.iter_mut().enumerate() {
                *entry = (0..3).map(|k| after.rows[i][k] * self.rows[k][j]).sum();
            }
        }
        let moved = after.point(Point3::new(self.shift[0], self.shift[1], self.shift[2]));
        Self {
            rows,
            shift: [moved.x(), moved.y(), moved.z()],
        }
    }

    pub fn point(&self, p: Point3) -> Point3 {
        let v = self.vector(Vec3::new(p.x(), p.y(), p.z()));
        Point3::new(
            v.x() + self.shift[0],
            v.y() + self.shift[1],
            v.z() + self.shift[2],
        )
    }

    pub fn vector(&self, v: Vec3) -> Vec3 {
        let r = self.rows;
        Vec3::new(
            r[0][0] * v.x() + r[0][1] * v.y() + r[0][2] * v.z(),
            r[1][0] * v.x() + r[1][1] * v.y() + r[1][2] * v.z(),
            r[2][0] * v.x() + r[2][1] * v.y() + r[2][2] * v.z(),
        )
    }

    pub fn flips(&self) -> bool {
        let r = self.rows;
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        det < 0.0
    }
}

/// Builds edges in a pose from points given before it.
pub struct Builder<'a> {
    pub topo: &'a mut Topology,
    pub pose: Pose,
}

impl Builder<'_> {
    pub fn vertex(&mut self, p: Point3) -> VertexId {
        self.topo.add_vertex(Vertex::new(self.pose.point(p), 1e-7))
    }

    pub fn line(&mut self, a: VertexId, b: VertexId) -> EdgeId {
        self.topo.add_edge(Edge::new(a, b, EdgeCurve::Line))
    }

    /// The arc from `a` to `b` that runs counter-clockwise about `normal`.
    pub fn arc(
        &mut self,
        center: Point3,
        normal: Vec3,
        radius: f64,
        a: VertexId,
        b: VertexId,
    ) -> EdgeId {
        let turned = self.pose.vector(normal);
        let posed = if self.pose.flips() { -turned } else { turned };
        let circle = Circle3D::new(self.pose.point(center), posed, radius).unwrap();
        self.topo
            .add_edge(Edge::new(a, b, EdgeCurve::Circle(circle)))
    }

    /// A cubic spline through `points` from `a` to `b`.
    pub fn spline(&mut self, points: &[Point3], a: VertexId, b: VertexId) -> EdgeId {
        let posed: Vec<Point3> = points.iter().map(|&p| self.pose.point(p)).collect();
        let curve = interpolate(&posed, 3).unwrap();
        self.topo
            .add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(curve)))
    }

    /// An edge along a curve already in its pose.
    pub fn curve(&mut self, curve: NurbsCurve, a: VertexId, b: VertexId) -> EdgeId {
        self.topo
            .add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(curve)))
    }
}

/// A face with a chain on its boundary and a point well inside it.
pub struct Sheet {
    pub face: FaceId,
    pub chain: Vec<EdgeId>,
    pub inside: Point3,
    /// Every edge in the wire, in order.
    pub boundary: Vec<EdgeId>,
}

pub const KINDS: [&str; 11] = [
    "plane",
    "cylinder across",
    "cylinder rim",
    "cone across",
    "cone rim",
    "sphere meridian",
    "sphere parallel",
    "torus ring",
    "torus tube",
    "spline",
    "rational spline",
];

/// The surface type a fixture kind tests.
pub fn surface_of(kind: &str) -> &'static str {
    match kind.split(' ').next().unwrap_or(kind) {
        "plane" => "plane",
        "cylinder" => "cylinder",
        "cone" => "cone",
        "sphere" => "sphere",
        "torus" => "torus",
        _ => "nurbs",
    }
}

/// The sheet of `kind` in `pose`, its chain in one direction off it, with
/// that direction (before the pose) and the chain's middle returned too.
pub fn sheet(topo: &mut Topology, kind: &str, pose: Pose) -> Sheet {
    let mut b = Builder { topo, pose };
    match kind {
        "plane" => plane(&mut b),
        "cylinder across" => cylinder(&mut b, true),
        "cylinder rim" => cylinder(&mut b, false),
        "cone across" => cone(&mut b, true),
        "cone rim" => cone(&mut b, false),
        "sphere meridian" => sphere(&mut b, true),
        "sphere parallel" => sphere(&mut b, false),
        "torus ring" => torus(&mut b, true),
        "torus tube" => torus(&mut b, false),
        "spline" => wavy_spline(&mut b),
        "rational spline" => rational_spline(&mut b),
        other => panic!("no fixture {other}"),
    }
}

/// Where the chain of `kind` is, and which way off the face it leaves, before
/// any pose: enough to set a second sheet across from the first.
pub fn chain_exit(kind: &str) -> (Point3, Vec3) {
    match kind {
        "plane" => (Point3::new(0.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
        "cylinder across" => {
            let a: f64 = 1.2;
            (
                Point3::new(-5.0 + 5.0 * a.cos(), 5.0, 5.0 * a.sin()),
                Vec3::new(a.sin(), 0.0, -a.cos()),
            )
        }
        "cylinder rim" => {
            let a: f64 = 0.8;
            (
                Point3::new(5.0 * a.cos(), 5.0 * a.sin(), 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            )
        }
        "cone across" => {
            let (a, u): (f64, f64) = (0.9, 1.3);
            (
                Point3::new(
                    6.0 * a.cos() * u.cos(),
                    6.0 * a.cos() * u.sin(),
                    6.0 * a.sin(),
                ),
                Vec3::new(-u.sin(), u.cos(), 0.0),
            )
        }
        "cone rim" => {
            let (a, u): (f64, f64) = (0.9, 0.85);
            (
                Point3::new(
                    9.0 * a.cos() * u.cos(),
                    9.0 * a.cos() * u.sin(),
                    9.0 * a.sin(),
                ),
                Vec3::new(a.cos() * u.cos(), a.cos() * u.sin(), a.sin()),
            )
        }
        "sphere meridian" => {
            let u: f64 = 1.2;
            (
                Point3::new(5.0 * u.cos(), 5.0 * u.sin(), 0.0),
                Vec3::new(-u.sin(), u.cos(), 0.0),
            )
        }
        "sphere parallel" => {
            let (u, v): (f64, f64) = (0.7, 0.6);
            (
                Point3::new(
                    5.0 * v.cos() * u.cos(),
                    5.0 * v.cos() * u.sin(),
                    5.0 * v.sin(),
                ),
                Vec3::new(-v.sin() * u.cos(), -v.sin() * u.sin(), v.cos()),
            )
        }
        "torus ring" => {
            let (u, v): (f64, f64) = (0.7, 1.3);
            let ring = 6.0 + 2.0 * v.cos();
            (
                Point3::new(ring * u.cos(), ring * u.sin(), 2.0 * v.sin()),
                Vec3::new(-v.sin() * u.cos(), -v.sin() * u.sin(), v.cos()),
            )
        }
        "torus tube" => {
            let (u, v): (f64, f64) = (1.2, 0.8);
            let ring = 6.0 + 2.0 * v.cos();
            (
                Point3::new(ring * u.cos(), ring * u.sin(), 2.0 * v.sin()),
                Vec3::new(-u.sin(), u.cos(), 0.0),
            )
        }
        "spline" => (Point3::new(0.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
        "rational spline" => (Point3::new(0.0, 5.0, 5.0), Vec3::new(-1.0, 0.0, 0.0)),
        other => panic!("no fixture {other}"),
    }
}

/// The pose of a second sheet of `kind`, across from the first, reflected
/// over and slightly turned so nothing about the two is parallel.
pub fn across_pose(kind: &str, pose: Pose) -> Pose {
    let (at, exit) = chain_exit(kind);
    let mirror = Pose::reflection(at + exit * 4.0, exit);
    let tilt = Pose::turn(Vec3::new(0.3, 0.2, 1.0), 0.12, [0.2, 0.1, -0.3]);
    mirror.then(tilt).then(pose)
}

fn wire_of(b: &mut Builder<'_>, edges: &[(EdgeId, bool)]) -> brepkit_topology::wire::WireId {
    let oriented = edges
        .iter()
        .map(|&(edge, forward)| OrientedEdge::new(edge, forward))
        .collect();
    b.topo.add_wire(Wire::new(oriented, true).unwrap())
}

fn finish(
    b: &mut Builder<'_>,
    surface: FaceSurface,
    edges: &[(EdgeId, bool)],
    chain: Vec<EdgeId>,
    inside: Point3,
) -> Sheet {
    let wire = wire_of(b, edges);
    let face = b.topo.add_face(Face::new(wire, Vec::new(), surface));
    Sheet {
        face,
        chain,
        inside: b.pose.point(inside),
        boundary: edges.iter().map(|e| e.0).collect(),
    }
}

fn plane(b: &mut Builder<'_>) -> Sheet {
    let corners = [
        Point3::new(-4.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.0, 10.0, 0.0),
        Point3::new(-4.0, 10.0, 0.0),
    ];
    let v: Vec<_> = corners.iter().map(|&p| b.vertex(p)).collect();
    let edges: Vec<_> = (0..4).map(|i| b.line(v[i], v[(i + 1) % 4])).collect();
    let normal = b.pose.vector(Vec3::new(0.0, 0.0, 1.0));
    let at = b.pose.point(Point3::new(0.0, 0.0, 0.0));
    let d = normal.dot(at - Point3::new(0.0, 0.0, 0.0));
    let oriented: Vec<_> = edges.iter().map(|&e| (e, true)).collect();
    finish(
        b,
        FaceSurface::Plane { normal, d },
        &oriented,
        vec![edges[1]],
        Point3::new(-2.0, 5.0, 0.0),
    )
}

fn cylinder(b: &mut Builder<'_>, across: bool) -> Sheet {
    if across {
        let at = |theta: f64, y: f64| Point3::new(-5.0 + 5.0 * theta.cos(), y, 5.0 * theta.sin());
        let (a, c) = (1.2, 2.1);
        let v = [
            b.vertex(at(a, 0.0)),
            b.vertex(at(c, 0.0)),
            b.vertex(at(c, 10.0)),
            b.vertex(at(a, 10.0)),
        ];
        let center_low = Point3::new(-5.0, 0.0, 0.0);
        let center_high = Point3::new(-5.0, 10.0, 0.0);
        let up = Vec3::new(0.0, 1.0, 0.0);
        let e0 = b.arc(center_low, -up, 5.0, v[0], v[1]);
        let e1 = b.line(v[1], v[2]);
        let e2 = b.arc(center_high, up, 5.0, v[2], v[3]);
        let e3 = b.line(v[3], v[0]);
        let surface =
            CylindricalSurface::new(b.pose.point(center_low), b.pose.vector(up), 5.0).unwrap();
        finish(
            b,
            FaceSurface::Cylinder(surface),
            &[(e0, true), (e1, true), (e2, true), (e3, true)],
            vec![e3],
            at(1.65, 5.0),
        )
    } else {
        let at = |theta: f64, z: f64| Point3::new(5.0 * theta.cos(), 5.0 * theta.sin(), z);
        let (a, c) = (0.3, 1.3);
        let v = [
            b.vertex(at(a, -5.0)),
            b.vertex(at(c, -5.0)),
            b.vertex(at(c, 0.0)),
            b.vertex(at(a, 0.0)),
        ];
        let up = Vec3::new(0.0, 0.0, 1.0);
        let e0 = b.arc(Point3::new(0.0, 0.0, -5.0), up, 5.0, v[0], v[1]);
        let e1 = b.line(v[1], v[2]);
        let e2 = b.arc(Point3::new(0.0, 0.0, 0.0), -up, 5.0, v[2], v[3]);
        let e3 = b.line(v[3], v[0]);
        let surface = CylindricalSurface::new(
            b.pose.point(Point3::new(0.0, 0.0, 0.0)),
            b.pose.vector(up),
            5.0,
        )
        .unwrap();
        finish(
            b,
            FaceSurface::Cylinder(surface),
            &[(e0, true), (e1, true), (e2, true), (e3, true)],
            vec![e2],
            at(0.8, -2.5),
        )
    }
}

fn cone(b: &mut Builder<'_>, across: bool) -> Sheet {
    let half: f64 = 0.9;
    let at = |u: f64, v: f64| {
        Point3::new(
            v * half.cos() * u.cos(),
            v * half.cos() * u.sin(),
            v * half.sin(),
        )
    };
    let (u0, u1, v0, v1) = (0.4, 1.3, 3.0, 9.0);
    let c = [
        b.vertex(at(u0, v0)),
        b.vertex(at(u1, v0)),
        b.vertex(at(u1, v1)),
        b.vertex(at(u0, v1)),
    ];
    let up = Vec3::new(0.0, 0.0, 1.0);
    let height = |v: f64| Point3::new(0.0, 0.0, v * half.sin());
    let e0 = b.arc(height(v0), up, v0 * half.cos(), c[0], c[1]);
    let e1 = b.line(c[1], c[2]);
    let e2 = b.arc(height(v1), -up, v1 * half.cos(), c[2], c[3]);
    let e3 = b.line(c[3], c[0]);
    let surface = ConicalSurface::new(
        b.pose.point(Point3::new(0.0, 0.0, 0.0)),
        b.pose.vector(up),
        half,
    )
    .unwrap();
    finish(
        b,
        FaceSurface::Cone(surface),
        &[(e0, true), (e1, true), (e2, true), (e3, true)],
        vec![if across { e1 } else { e2 }],
        at(0.85, 6.0),
    )
}

fn sphere(b: &mut Builder<'_>, meridian: bool) -> Sheet {
    let at = |u: f64, v: f64| {
        Point3::new(
            5.0 * v.cos() * u.cos(),
            5.0 * v.cos() * u.sin(),
            5.0 * v.sin(),
        )
    };
    let (u0, u1, v0, v1): (f64, f64, f64, f64) = (0.2, 1.2, -0.6, 0.6);
    let c = [
        b.vertex(at(u0, v0)),
        b.vertex(at(u1, v0)),
        b.vertex(at(u1, v1)),
        b.vertex(at(u0, v1)),
    ];
    let up = Vec3::new(0.0, 0.0, 1.0);
    let origin = Point3::new(0.0, 0.0, 0.0);
    let across = |u: f64| Vec3::new(u.sin(), -u.cos(), 0.0);
    let e0 = b.arc(
        Point3::new(0.0, 0.0, 5.0 * v0.sin()),
        up,
        5.0 * v0.cos(),
        c[0],
        c[1],
    );
    let e1 = b.arc(origin, across(u1), 5.0, c[1], c[2]);
    let e2 = b.arc(
        Point3::new(0.0, 0.0, 5.0 * v1.sin()),
        -up,
        5.0 * v1.cos(),
        c[2],
        c[3],
    );
    let e3 = b.arc(origin, -across(u0), 5.0, c[3], c[0]);
    let surface = SphericalSurface::with_axis(
        b.pose.point(origin),
        5.0,
        b.pose.vector(Vec3::new(0.2, 0.3, 1.0)),
    )
    .unwrap();
    finish(
        b,
        FaceSurface::Sphere(surface),
        &[(e0, true), (e1, true), (e2, true), (e3, true)],
        vec![if meridian { e1 } else { e2 }],
        at(0.7, 0.0),
    )
}

fn torus(b: &mut Builder<'_>, ring: bool) -> Sheet {
    let (major, minor) = (6.0, 2.0);
    let at = |u: f64, v: f64| {
        let r = minor * f64::cos(v) + major;
        Point3::new(r * f64::cos(u), r * f64::sin(u), minor * f64::sin(v))
    };
    let (u0, u1, v0, v1): (f64, f64, f64, f64) = (0.2, 1.2, 0.3, 1.3);
    let c = [
        b.vertex(at(u0, v0)),
        b.vertex(at(u1, v0)),
        b.vertex(at(u1, v1)),
        b.vertex(at(u0, v1)),
    ];
    let up = Vec3::new(0.0, 0.0, 1.0);
    let level = |v: f64| Point3::new(0.0, 0.0, minor * f64::sin(v));
    let middle = |u: f64| Point3::new(major * f64::cos(u), major * f64::sin(u), 0.0);
    let across = |u: f64| Vec3::new(u.sin(), -u.cos(), 0.0);
    let e0 = b.arc(level(v0), up, major + minor * v0.cos(), c[0], c[1]);
    let e1 = b.arc(middle(u1), across(u1), minor, c[1], c[2]);
    let e2 = b.arc(level(v1), -up, major + minor * v1.cos(), c[2], c[3]);
    let e3 = b.arc(middle(u0), -across(u0), minor, c[3], c[0]);
    let surface = ToroidalSurface::with_axis(
        b.pose.point(Point3::new(0.0, 0.0, 0.0)),
        major,
        minor,
        b.pose.vector(up),
    )
    .unwrap();
    finish(
        b,
        FaceSurface::Torus(surface),
        &[(e0, true), (e1, true), (e2, true), (e3, true)],
        vec![if ring { e2 } else { e1 }],
        at(0.7, 0.8),
    )
}

/// A control net `net[i][j]` in `pose`, as a sheet bounded by its four sides.
pub fn net_sheet(
    b: &mut Builder<'_>,
    degrees: (usize, usize),
    knots: (Vec<f64>, Vec<f64>),
    net: &[Vec<Point3>],
    weights: &[Vec<f64>],
) -> (FaceId, [EdgeId; 4]) {
    let posed: Vec<Vec<Point3>> = net
        .iter()
        .map(|row| row.iter().map(|&p| b.pose.point(p)).collect())
        .collect();
    let (n, m) = (posed.len(), posed[0].len());
    let surface = NurbsSurface::new(
        degrees.0,
        degrees.1,
        knots.0.clone(),
        knots.1.clone(),
        posed.clone(),
        weights.to_vec(),
    )
    .unwrap();
    let row = |j: usize| {
        NurbsCurve::new(
            degrees.0,
            knots.0.clone(),
            posed.iter().map(|r| r[j]).collect(),
            weights.iter().map(|r| r[j]).collect(),
        )
        .unwrap()
    };
    let column = |i: usize| {
        NurbsCurve::new(
            degrees.1,
            knots.1.clone(),
            posed[i].clone(),
            weights[i].clone(),
        )
        .unwrap()
    };
    let corner = |b: &mut Builder<'_>, p: Point3| b.topo.add_vertex(Vertex::new(p, 1e-7));
    let v = [
        corner(b, posed[0][0]),
        corner(b, posed[n - 1][0]),
        corner(b, posed[n - 1][m - 1]),
        corner(b, posed[0][m - 1]),
    ];
    let e0 = b.curve(row(0), v[0], v[1]);
    let e1 = b.curve(column(n - 1), v[1], v[2]);
    let e2 = b.curve(row(m - 1), v[3], v[2]);
    let e3 = b.curve(column(0), v[0], v[3]);
    let wire = wire_of(b, &[(e0, true), (e1, true), (e2, false), (e3, false)]);
    let face = b
        .topo
        .add_face(Face::new(wire, Vec::new(), FaceSurface::Nurbs(surface)));
    (face, [e0, e1, e2, e3])
}

fn cubic_knots(spans: usize) -> Vec<f64> {
    let mut knots = vec![0.0; 3];
    for i in 0..=spans {
        knots.push(i as f64 / spans as f64);
    }
    knots.extend([1.0; 3]);
    knots
}

fn wavy_spline(b: &mut Builder<'_>) -> Sheet {
    let n = 7;
    let net: Vec<Vec<Point3>> = (0..n)
        .map(|i| {
            let x = -6.0 + i as f64;
            (0..n)
                .map(|j| {
                    let y = 10.0 * j as f64 / 6.0;
                    let z = 0.5 * (x * 0.9).sin() * (y * 0.6).cos() + 0.05 * x * y;
                    Point3::new(x, y, z)
                })
                .collect()
        })
        .collect();
    let weights = vec![vec![1.0; n]; n];
    let (face, edges) = net_sheet(b, (3, 3), (cubic_knots(4), cubic_knots(4)), &net, &weights);
    Sheet {
        face,
        chain: vec![edges[1]],
        inside: b.pose.point(net[3][3]),
        boundary: edges.to_vec(),
    }
}

fn rational_spline(b: &mut Builder<'_>) -> Sheet {
    let r = 5.0;
    let columns = [
        (Point3::new(r, 0.0, 0.0), 1.0),
        (Point3::new(r, 0.0, r), FRAC_1_SQRT_2),
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
    let knots_u = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let knots_v = vec![0.0, 0.0, 1.0, 1.0];
    let (face, edges) = net_sheet(b, (2, 1), (knots_u, knots_v), &net, &weights);
    let inside = r * FRAC_1_SQRT_2;
    Sheet {
        face,
        chain: vec![edges[1]],
        inside: b.pose.point(Point3::new(inside, 5.0, inside)),
        boundary: edges.to_vec(),
    }
}

/// That each edge of a face's outer wire starts where the last one ended, and
/// lies on the face's surface.
pub fn assert_wire_sound(topo: &Topology, face: FaceId) {
    let data = topo.face(face).unwrap();
    let surface = data.surface();
    let wire = topo.wire(data.outer_wire()).unwrap();
    let ends: Vec<(VertexId, VertexId)> = wire
        .edges()
        .iter()
        .map(|oe| {
            let edge = topo.edge(oe.edge()).unwrap();
            if oe.is_forward() {
                (edge.start(), edge.end())
            } else {
                (edge.end(), edge.start())
            }
        })
        .collect();
    for (k, end) in ends.iter().enumerate() {
        let next = ends[(k + 1) % ends.len()];
        assert_eq!(end.1, next.0, "edge {k} does not run on to the next");
    }
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).unwrap();
        let start = topo.vertex(edge.start()).unwrap().point();
        let end = topo.vertex(edge.end()).unwrap().point();
        let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
        for k in 0..=12 {
            let t = t0 + (t1 - t0) * f64::from(k) / 12.0;
            let p = edge.curve().evaluate_with_endpoints(t, start, end);
            let (u, v) = foot_on(surface, p);
            let gap = (surface.evaluate(u, v).unwrap() - p).length();
            assert!(
                gap < 1e-7,
                "edge {} leaves its surface by {gap}",
                oe.edge().index()
            );
        }
    }
}

/// The worst figures measured along a seam.
#[derive(Debug, Clone, Copy, Default)]
pub struct Worst {
    /// Distance from the result's boundary to the neighbour's surface and to
    /// the chain.
    pub g0: f64,
    /// Angle between normals, radians.
    pub g1: f64,
    /// Difference in normal curvature in the direction square to the seam.
    pub g2: f64,
    /// The largest curvature it was compared against, for scale.
    pub curvature: f64,
}

impl Worst {
    pub fn join(&mut self, other: Self) {
        self.g0 = self.g0.max(other.g0);
        self.g1 = self.g1.max(other.g1);
        self.g2 = self.g2.max(other.g2);
        self.curvature = self.curvature.max(other.curvature);
    }
}

/// The nearest point on a surface, read by evaluators alone.
pub fn foot_on(surface: &FaceSurface, p: Point3) -> (f64, f64) {
    let Some(start) = surface.project_point(p) else {
        return (0.0, 0.0);
    };
    let (mut u, mut v) = start;
    let bounds = if let FaceSurface::Nurbs(n) = surface {
        Some((n.domain_u(), n.domain_v()))
    } else {
        None
    };
    for _ in 0..12 {
        let (su, sv) = (
            surface.partial_u(u, v).unwrap(),
            surface.partial_v(u, v).unwrap(),
        );
        let r = p - surface.evaluate(u, v).unwrap();
        let (e, f, g) = (su.dot(su), su.dot(sv), sv.dot(sv));
        let det = e * g - f * f;
        if det <= 1e-24 {
            break;
        }
        let (a, c) = (r.dot(su), r.dot(sv));
        u += (g * a - f * c) / det;
        v += (e * c - f * a) / det;
        if let Some((du, dv)) = bounds {
            u = u.clamp(du.0, du.1);
            v = v.clamp(dv.0, dv.1);
        }
    }
    (u, v)
}

/// The second partials by differencing the first, or by a spline's own
/// derivatives, whose continuity at a knot differencing would blur.
fn second_partials(surface: &FaceSurface, u: f64, v: f64) -> (Vec3, Vec3, Vec3) {
    if let FaceSurface::Nurbs(n) = surface {
        let d = n.derivatives(u, v, 2);
        return (d[2][0], d[1][1], d[0][2]);
    }
    let h = 1e-4;
    let su = |a: f64, b: f64| surface.partial_u(a, b).unwrap();
    let sv = |a: f64, b: f64| surface.partial_v(a, b).unwrap();
    (
        (su(u + h, v) - su(u - h, v)) * (0.5 / h),
        (su(u, v + h) - su(u, v - h)) * (0.5 / h),
        (sv(u, v + h) - sv(u, v - h)) * (0.5 / h),
    )
}

/// Normal curvature of `surface` at `(u, v)` along `d`, with the unit normal
/// it is signed against.
fn curvature_along(surface: &FaceSurface, u: f64, v: f64, d: Vec3) -> (f64, Vec3) {
    if let FaceSurface::Plane { normal, .. } = surface {
        return (0.0, normal.normalize().unwrap());
    }
    let (su, sv) = (
        surface.partial_u(u, v).unwrap(),
        surface.partial_v(u, v).unwrap(),
    );
    let n = su.cross(sv).normalize().unwrap();
    let (suu, suv, svv) = second_partials(surface, u, v);
    let (e, f, g) = (su.dot(su), su.dot(sv), sv.dot(sv));
    let det = e * g - f * f;
    let (a, c) = (d.dot(su), d.dot(sv));
    let du = (g * a - f * c) / det;
    let dv = (e * c - f * a) / det;
    let form = n.dot(suu) * du * du + 2.0 * n.dot(suv) * du * dv + n.dot(svv) * dv * dv;
    (form / d.dot(d), n)
}

/// The figures along the seam of `result` at `v = at_v` against `neighbour`.
///
/// `samples` points along `u`, the ends and any collapse excluded from the
/// normal and curvature checks when the surface has no normal there.
pub fn measure_seam(
    result: &NurbsSurface,
    at_v: f64,
    neighbour: &FaceSurface,
    samples: usize,
) -> Worst {
    let (u0, u1) = result.domain_u();
    let at: Vec<f64> = (0..=samples)
        .map(|i| u0 + (u1 - u0) * i as f64 / samples as f64)
        .collect();
    measure_seam_at(result, at_v, neighbour, &at)
}

/// As [`measure_seam`], at the given `u` only.
pub fn measure_seam_at(
    result: &NurbsSurface,
    at_v: f64,
    neighbour: &FaceSurface,
    at: &[f64],
) -> Worst {
    let mut worst = Worst::default();
    for &u in at {
        let d = result.derivatives(u, at_v, 2);
        let p = result.evaluate(u, at_v);
        let (fu, fv) = foot_on(neighbour, p);
        let off_surface = if let FaceSurface::Plane { normal, d } = neighbour {
            let unit = normal.normalize().unwrap();
            ((p - Point3::new(0.0, 0.0, 0.0)).dot(unit) - d / normal.length()).abs()
        } else {
            (neighbour.evaluate(fu, fv).unwrap() - p).length()
        };
        worst.g0 = worst.g0.max(off_surface);

        let (bu, bv) = (d[1][0], d[0][1]);
        let cross = bu.cross(bv);
        if cross.length() <= 1e-9 * bu.length().max(1.0) * bv.length().max(1e-300) {
            continue;
        }
        let n = cross.normalize().unwrap();
        let along = bu.normalize().unwrap();
        let (_, neighbour_normal) = curvature_along(
            neighbour,
            fu,
            fv,
            if matches!(neighbour, FaceSurface::Plane { .. }) {
                along
            } else {
                neighbour.partial_u(fu, fv).unwrap()
            },
        );
        worst.g1 = worst.g1.max(
            n.cross(neighbour_normal)
                .length()
                .atan2(n.dot(neighbour_normal).abs()),
        );

        let square = neighbour_normal.cross(along).normalize().unwrap();
        let (kappa, signed_by) = curvature_along(neighbour, fu, fv, square);
        let kappa = if signed_by.dot(n) < 0.0 {
            -kappa
        } else {
            kappa
        };

        let (e, f, g) = (bu.dot(bu), bu.dot(bv), bv.dot(bv));
        let det = e * g - f * f;
        let (a, c) = (square.dot(bu), square.dot(bv));
        let (du, dv) = ((g * a - f * c) / det, (e * c - f * a) / det);
        let own =
            (n.dot(d[2][0]) * du * du + 2.0 * n.dot(d[1][1]) * du * dv + n.dot(d[0][2]) * dv * dv)
                / square.dot(square);
        worst.g2 = worst.g2.max((own - kappa).abs());
        worst.curvature = worst.curvature.max(kappa.abs());
    }
    worst
}

/// The distance from `p` to `edge`, by dense sampling and a refined bracket.
pub fn distance_to_edge(topo: &Topology, edge: EdgeId, p: Point3) -> f64 {
    let data = topo.edge(edge).unwrap();
    let start = topo.vertex(data.start()).unwrap().point();
    let end = topo.vertex(data.end()).unwrap().point();
    let (t0, t1) = data.curve().domain_with_endpoints(start, end);
    let at = |t: f64| data.curve().evaluate_with_endpoints(t, start, end);
    let steps = 400;
    let mut best = (t0, f64::INFINITY);
    for i in 0..=steps {
        let t = t0 + (t1 - t0) * f64::from(i) / f64::from(steps);
        let gap = (at(t) - p).length();
        if gap < best.1 {
            best = (t, gap);
        }
    }
    let span = (t1 - t0).abs() / f64::from(steps);
    let (lo_limit, hi_limit) = (t0.min(t1), t0.max(t1));
    let (mut lo, mut hi) = ((best.0 - span).max(lo_limit), (best.0 + span).min(hi_limit));
    for _ in 0..80 {
        let a = lo + (hi - lo) / 3.0;
        let c = hi - (hi - lo) / 3.0;
        if (at(a) - p).length() < (at(c) - p).length() {
            hi = c;
        } else {
            lo = a;
        }
    }
    (at(0.5 * (lo + hi)) - p).length().min(best.1)
}

/// The farthest the boundary of `result` at `v = at_v` strays from a chain.
pub fn gap_to_chain(
    topo: &Topology,
    result: &NurbsSurface,
    at_v: f64,
    chain: &[EdgeId],
    samples: usize,
) -> f64 {
    let (u0, u1) = result.domain_u();
    let mut worst = 0.0_f64;
    for i in 0..=samples {
        let u = u0 + (u1 - u0) * i as f64 / samples as f64;
        let p = result.evaluate(u, at_v);
        let nearest = chain
            .iter()
            .map(|&edge| distance_to_edge(topo, edge, p))
            .fold(f64::INFINITY, f64::min);
        worst = worst.max(nearest);
    }
    worst
}

/// Whether the result leaves the seam at `v = at_v` away from `inside`, a
/// point well inside the neighbouring face.
pub fn leaves_the_face(result: &NurbsSurface, at_v: f64, inside: Point3) -> bool {
    let (u0, u1) = result.domain_u();
    let u = 0.5 * (u0 + u1);
    let d = result.derivatives(u, at_v, 1);
    let inward = if at_v < 0.5 { d[0][1] } else { -d[0][1] };
    inward.dot(inside - result.evaluate(u, at_v)) < 0.0
}
