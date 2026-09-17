//! Trimmed fillet stripes honour the requested mesh deflection.
//!
//! This exercises the display and export tessellation route on a 25.4 mm cube
//! eased on one, two or four top edges, from both construction sources and in
//! both poses. Every case asserts the deflection contract on each cylindrical
//! stripe: triangle vertices, edges and barycentric interiors sample the exact
//! cylinder within the requested tolerance, the UV triangles neither overlap
//! nor reverse, and each selected edge carries exactly one stripe face.
//!
//! A multi-edge selection meets at a toroidal corner and its stripes run the
//! full edge, so the oracles that read a single stripe's trimmed extent (mesh
//! closure across the corner, retained domain, analytic trimmed area, vertex
//! exactness) are asserted on the single-edge cases. Triangle counts are
//! diagnostic only and are never an acceptance oracle.

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unwrap_used
)]

use std::f64::consts::FRAC_PI_2;

use brepkit_math::chord::DEFAULT_ANGULAR_TOL;
use brepkit_math::mat::Mat4;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{
    boundary_edge_count, non_manifold_edge_count, tessellate_solid_grouped_with_tolerance,
};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;

const SIZE: f64 = 25.4;
const DEFLECTION: f64 = 0.25;
const NUMERIC_MARGIN: f64 = 1.0e-6;
const BARYCENTRIC_STEPS: usize = 8;

#[derive(Clone, Copy, Debug)]
enum Source {
    Box,
    Extrusion,
}

#[derive(Clone, Copy, Debug)]
enum Pose {
    Identity,
    Rigid,
}

#[derive(Clone, Copy, Debug)]
enum Selection {
    Single,
    Adjacent,
    Four,
}

impl Selection {
    const fn edge_count(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Adjacent => 2,
            Self::Four => 4,
        }
    }

    /// Whether every stripe of this selection runs the full edge, untrimmed by
    /// a neighbouring stripe.
    const fn stripes_are_untrimmed(self) -> bool {
        matches!(self, Self::Single)
    }
}

#[derive(Clone, Copy)]
struct UvTriangle {
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
}

fn extrusion(topo: &mut Topology) -> SolidId {
    let wire = make_polygon_wire(
        topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(SIZE, 0.0, 0.0),
            Point3::new(SIZE, SIZE, 0.0),
            Point3::new(0.0, SIZE, 0.0),
        ],
        Tolerance::new().linear,
    )
    .unwrap();
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), SIZE).unwrap()
}

fn edge_at(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let linear = Tolerance::new().linear;
    let matches: Vec<_> = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&id| {
            let edge = topo.edge(id).unwrap();
            let start = topo.vertex(edge.start()).unwrap().point();
            let end = topo.vertex(edge.end()).unwrap().point();
            ((start - a).length() < linear && (end - b).length() < linear)
                || ((start - b).length() < linear && (end - a).length() < linear)
        })
        .collect();
    assert_eq!(matches.len(), 1, "unique edge {a:?}--{b:?}");
    matches[0]
}

/// The eased top surface as a height above the base, `distance` away from the
/// filleted edge.
fn graph(distance: f64, radius: f64) -> f64 {
    let h = SIZE - radius;
    if distance >= radius {
        SIZE
    } else {
        h + (radius * radius - (radius - distance) * (radius - distance))
            .max(0.0)
            .sqrt()
    }
}

fn point_on_retained_domain(point_on_surface: Point3, inverse: &Mat4, radius: f64) -> bool {
    let local = inverse.mul_point(point_on_surface);
    let coordinate_margin = 4.0 * NUMERIC_MARGIN;
    local.x() >= -coordinate_margin
        && local.x() <= SIZE + coordinate_margin
        && local.y() >= -coordinate_margin
        && local.y() <= SIZE + coordinate_margin
        && local.z() >= SIZE - radius - coordinate_margin
        && local.z() <= SIZE + coordinate_margin
        && (local.z() - graph(local.y(), radius)).abs() <= coordinate_margin
}

fn triangle_samples(a: Point3, b: Point3, c: Point3) -> Vec<Point3> {
    let mut samples = Vec::with_capacity((BARYCENTRIC_STEPS + 1) * (BARYCENTRIC_STEPS + 2) / 2 + 1);
    for i in 0..=BARYCENTRIC_STEPS {
        for j in 0..=BARYCENTRIC_STEPS - i {
            let wb = i as f64 / BARYCENTRIC_STEPS as f64;
            let wc = j as f64 / BARYCENTRIC_STEPS as f64;
            let wa = 1.0 - wb - wc;
            samples.push(Point3::new(
                wa * a.x() + wb * b.x() + wc * c.x(),
                wa * a.y() + wb * b.y() + wc * c.y(),
                wa * a.z() + wb * b.z() + wc * c.z(),
            ));
        }
    }
    samples.push(Point3::new(
        (a.x() + b.x() + c.x()) / 3.0,
        (a.y() + b.y() + c.y()) / 3.0,
        (a.z() + b.z() + c.z()) / 3.0,
    ));
    samples
}

fn unwrap_near(value: f64, anchor: f64) -> f64 {
    anchor + (value - anchor + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
        - std::f64::consts::PI
}

fn twice_signed_area(triangle: UvTriangle) -> f64 {
    (triangle.b.0 - triangle.a.0) * (triangle.c.1 - triangle.a.1)
        - (triangle.b.1 - triangle.a.1) * (triangle.c.0 - triangle.a.0)
}

fn contains_uv(triangle: UvTriangle, point: (f64, f64)) -> bool {
    let cross = |a: (f64, f64), b: (f64, f64), p: (f64, f64)| {
        (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
    };
    let c0 = cross(triangle.a, triangle.b, point);
    let c1 = cross(triangle.b, triangle.c, point);
    let c2 = cross(triangle.c, triangle.a, point);
    let epsilon = 1.0e-10;
    (c0 >= -epsilon && c1 >= -epsilon && c2 >= -epsilon)
        || (c0 <= epsilon && c1 <= epsilon && c2 <= epsilon)
}

#[allow(clippy::too_many_lines)]
fn check_case(source: Source, pose: Pose, selection: Selection, radius: f64) -> Vec<String> {
    let mut topo = Topology::new();
    let solid = match source {
        Source::Box => make_box(&mut topo, SIZE, SIZE, SIZE).unwrap(),
        Source::Extrusion => extrusion(&mut topo),
    };
    let transform = match pose {
        Pose::Identity => Mat4::identity(),
        Pose::Rigid => {
            Mat4::translation(3.0 * SIZE, -2.0 * SIZE, 0.7 * SIZE)
                * Mat4::rotation_y(0.47)
                * Mat4::rotation_z(0.31)
        }
    };
    transform_solid(&mut topo, solid, &transform).unwrap();
    let inverse = transform.inverse().unwrap();
    let corners = [
        Point3::new(0.0, 0.0, SIZE),
        Point3::new(SIZE, 0.0, SIZE),
        Point3::new(SIZE, SIZE, SIZE),
        Point3::new(0.0, SIZE, SIZE),
    ];
    let selected: Vec<_> = (0..selection.edge_count())
        .map(|i| {
            edge_at(
                &topo,
                solid,
                transform.mul_point(corners[i]),
                transform.mul_point(corners[(i + 1) % 4]),
            )
        })
        .collect();
    let result = fillet_v2(&mut topo, solid, &selected, radius).unwrap();
    let (mesh, offsets) = tessellate_solid_grouped_with_tolerance(
        &topo,
        result.solid,
        DEFLECTION,
        DEFAULT_ANGULAR_TOL,
    )
    .unwrap();
    let faces = solid_faces(&topo, result.solid).unwrap();
    let label = format!("source={source:?} pose={pose:?} selection={selection:?} r={radius}");
    eprintln!(
        "{label}: faces={} vertices={} triangles={}",
        faces.len(),
        mesh.positions.len(),
        mesh.indices.len() / 3
    );

    let mut failures = Vec::new();
    if selection.stripes_are_untrimmed() {
        let boundary_edges = boundary_edge_count(&mesh);
        let non_manifold_edges = non_manifold_edge_count(&mesh);
        if boundary_edges != 0 || non_manifold_edges != 0 {
            failures.push(format!(
                "{label}: mesh must be watertight and manifold, got boundary={boundary_edges} non_manifold={non_manifold_edges}"
            ));
        }
    }
    let mut cylinder_faces = 0usize;
    let mut worst_deflection = 0.0_f64;
    let mut worst_vertex_deviation = 0.0_f64;

    for (face_index, face_id) in faces.iter().copied().enumerate() {
        let face = topo.face(face_id).unwrap();
        let FaceSurface::Cylinder(cylinder) = face.surface() else {
            continue;
        };
        cylinder_faces += 1;
        let begin = offsets[face_index] as usize;
        let end = offsets[face_index + 1] as usize;
        let mut raw_uv_triangles = Vec::new();
        let mut anchor_u = None;
        let mut face_worst = 0.0_f64;
        let mut domain_failure = None;

        for indices in mesh.indices[begin..end].chunks_exact(3) {
            let vertices = [
                mesh.positions[indices[0] as usize],
                mesh.positions[indices[1] as usize],
                mesh.positions[indices[2] as usize],
            ];
            let mut uv = [(0.0, 0.0); 3];
            for (slot, vertex) in vertices.iter().copied().enumerate() {
                let (u, v) = cylinder.project_point(vertex);
                let anchor = *anchor_u.get_or_insert(u);
                uv[slot] = (unwrap_near(u, anchor), v);
                let projected = cylinder.evaluate(u, v);
                worst_vertex_deviation = worst_vertex_deviation.max((projected - vertex).length());
            }
            raw_uv_triangles.push(UvTriangle {
                a: uv[0],
                b: uv[1],
                c: uv[2],
            });

            for sample in triangle_samples(vertices[0], vertices[1], vertices[2]) {
                let (u, v) = cylinder.project_point(sample);
                let projected = cylinder.evaluate(u, v);
                let deviation = (projected - sample).length();
                face_worst = face_worst.max(deviation);
                worst_deflection = worst_deflection.max(deviation);
                if selection.stripes_are_untrimmed()
                    && domain_failure.is_none()
                    && !point_on_retained_domain(projected, &inverse, radius)
                {
                    domain_failure = Some(format!(
                        "{label}: face {face_id:?} triangle sample projects outside retained stripe domain: local={:?}",
                        inverse.mul_point(projected)
                    ));
                }
            }
        }
        if let Some(failure) = domain_failure {
            failures.push(failure);
        }

        let anchor = anchor_u.unwrap();
        let mut triangles = Vec::with_capacity(raw_uv_triangles.len());
        for triangle in raw_uv_triangles {
            triangles.push(UvTriangle {
                a: (unwrap_near(triangle.a.0, anchor), triangle.a.1),
                b: (unwrap_near(triangle.b.0, anchor), triangle.b.1),
                c: (unwrap_near(triangle.c.0, anchor), triangle.c.1),
            });
        }
        let signed_area: f64 = triangles
            .iter()
            .copied()
            .map(twice_signed_area)
            .sum::<f64>()
            * 0.5;
        let absolute_area: f64 = triangles
            .iter()
            .copied()
            .map(|triangle| twice_signed_area(triangle).abs())
            .sum::<f64>()
            * 0.5;
        let winding_error = (absolute_area - signed_area.abs()).abs();
        if winding_error > NUMERIC_MARGIN * absolute_area.max(1.0) {
            failures.push(format!(
                "{label}: face {face_id:?} UV triangles overlap or reverse: absolute area={absolute_area}, signed area={signed_area}"
            ));
        }
        if selection.stripes_are_untrimmed() {
            let expected_area = SIZE * FRAC_PI_2;
            if (absolute_area - expected_area).abs() > NUMERIC_MARGIN * expected_area {
                failures.push(format!(
                    "{label}: face {face_id:?} does not cover analytic trimmed domain: mesh UV area={absolute_area}, expected={expected_area}"
                ));
            }

            let u_min = triangles
                .iter()
                .flat_map(|triangle| [triangle.a.0, triangle.b.0, triangle.c.0])
                .fold(f64::INFINITY, f64::min);
            let u_max = triangles
                .iter()
                .flat_map(|triangle| [triangle.a.0, triangle.b.0, triangle.c.0])
                .fold(f64::NEG_INFINITY, f64::max);
            let v_min = triangles
                .iter()
                .flat_map(|triangle| [triangle.a.1, triangle.b.1, triangle.c.1])
                .fold(f64::INFINITY, f64::min);
            let v_max = triangles
                .iter()
                .flat_map(|triangle| [triangle.a.1, triangle.b.1, triangle.c.1])
                .fold(f64::NEG_INFINITY, f64::max);
            let u_probes = ((u_max - u_min) * radius / DEFLECTION).ceil().max(8.0) as usize;
            let v_probes = ((v_max - v_min) / DEFLECTION).ceil().max(8.0) as usize;
            'coverage: for iu in 0..u_probes {
                for iv in 0..v_probes {
                    let u = u_min + (u_max - u_min) * (iu as f64 + 0.5) / u_probes as f64;
                    let v = v_min + (v_max - v_min) * (iv as f64 + 0.5) / v_probes as f64;
                    let on_surface = cylinder.evaluate(u, v);
                    if point_on_retained_domain(on_surface, &inverse, radius)
                        && !triangles
                            .iter()
                            .any(|&triangle| contains_uv(triangle, (u, v)))
                    {
                        failures.push(format!(
                            "{label}: face {face_id:?} leaves trimmed-domain probe uncovered at uv=({u},{v})"
                        ));
                        break 'coverage;
                    }
                }
            }
        }
        eprintln!(
            "  face {face_id:?}: triangles={} worst_deflection={face_worst:.6} mm UV_area={absolute_area:.6}",
            triangles.len()
        );
    }

    if cylinder_faces != selection.edge_count() {
        failures.push(format!(
            "{label}: expected {} cylindrical stripe faces, found {cylinder_faces}",
            selection.edge_count()
        ));
    }
    if selection.stripes_are_untrimmed() && worst_vertex_deviation > NUMERIC_MARGIN {
        failures.push(format!(
            "{label}: mesh vertex deviates from exact stripe surface by {worst_vertex_deviation} mm"
        ));
    }
    if worst_deflection > DEFLECTION + NUMERIC_MARGIN {
        failures.insert(
            0,
            format!(
                "{label}: triangle-to-surface deflection {worst_deflection} mm exceeds requested {DEFLECTION} mm + {NUMERIC_MARGIN} mm numeric margin"
            ),
        );
    }
    eprintln!("  {label}: worst_deflection={worst_deflection:.6} mm");
    failures
}

#[test]
fn fillet_stripe_mesh_honors_requested_deflection_and_trimmed_domain() {
    let mut failures = Vec::new();
    for source in [Source::Box, Source::Extrusion] {
        for pose in [Pose::Identity, Pose::Rigid] {
            for selection in [Selection::Single, Selection::Adjacent, Selection::Four] {
                for radius in [2.54, 12.6746] {
                    failures.extend(check_case(source, pose, selection, radius));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "fillet stripe tessellation violates deflection/domain contract:\n{}",
        failures.join("\n")
    );
}
