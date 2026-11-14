//! A round along an edge of a B-spline face whose boundary stands off its
//! surface. Faces read from a file often have edges a little off the
//! surface they bound (a wing's skin read from one file has its edges up to
//! 3e-3 off it); here a box's top is a flat B-spline 3e-4 above its edges.
//! The round's contact on the top ends on the top's boundary edges, so its
//! ends stand off the top's surface as those edges do; the trim mapped them
//! onto the surface only within 1e-4 and refused the round. It now allows
//! as far as the face's own boundary stands off, and the end faces' notch
//! and the trim's split read the contact's ends on their edges within that
//! too (an end face left whole was closed by runout patches, its mesh
//! open). The round takes off (1 - pi / 4) r^2 per unit length, up to the
//! offset's share.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::{all_blended, fillet_v2};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;
use brepkit_topology::wire::OrientedEdge;

const SIDE: f64 = 10.0;

/// How the top's edges are stored.
#[derive(Clone, Copy, Debug)]
enum Edges {
    Lines,
    /// Straight B-splines of `degree`, as a file can store them, each
    /// running its edge's way or, if `backward`, against it.
    Splines {
        degree: usize,
        backward: bool,
    },
}

/// A box whose top face is a flat B-spline `off` above its edges, which are
/// stored as `edges` says, turned into `pose`, and its top edge along x at
/// y = 0.
fn skinned_box(topo: &mut Topology, off: f64, pose: &Mat4, edges: Edges) -> (SolidId, EdgeId) {
    let solid = make_box(topo, SIDE, SIDE, SIDE).unwrap();
    let top = solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&f| {
            matches!(topo.face(f).unwrap().surface(),
                FaceSurface::Plane { normal, d } if normal.z() > 0.5 && (d - SIDE).abs() < 1e-9)
        })
        .unwrap();
    let z = SIDE + off;
    let (lo, hi) = (-1.0, SIDE + 1.0);
    let skin = NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(lo, lo, z), Point3::new(lo, hi, z)],
            vec![Point3::new(hi, lo, z), Point3::new(hi, hi, z)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap();
    topo.face_mut(top)
        .unwrap()
        .set_surface(FaceSurface::Nurbs(skin));
    if let Edges::Splines { degree, backward } = edges {
        let wire = topo.face(top).unwrap().outer_wire();
        let boundary: Vec<EdgeId> = topo
            .wire(wire)
            .unwrap()
            .edges()
            .iter()
            .map(OrientedEdge::edge)
            .collect();
        for edge_id in boundary {
            let edge = topo.edge(edge_id).unwrap();
            let mut p = topo.vertex(edge.start()).unwrap().point();
            let mut q = topo.vertex(edge.end()).unwrap().point();
            if backward {
                std::mem::swap(&mut p, &mut q);
            }
            let mut knots = vec![0.0; degree + 1];
            knots.extend(vec![1.0; degree + 1]);
            let points = (0..=degree)
                .map(|k| p + (q - p) * (k as f64 / degree as f64))
                .collect();
            let line = NurbsCurve::new(degree, knots, points, vec![1.0; degree + 1]).unwrap();
            topo.edge_mut(edge_id)
                .unwrap()
                .set_curve(EdgeCurve::NurbsCurve(line));
        }
    }
    let edge = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                p.y().abs() < 1e-9 && (p.z() - SIDE).abs() < 1e-9
            })
        })
        .unwrap();
    transform_solid(topo, solid, pose).unwrap();
    (solid, edge)
}

/// An edge as it is read, over the span of its curve that its vertices pick
/// out: its length along that span, and how far the span's ends fall from
/// the vertices.
fn read_edge(topo: &Topology, edge: EdgeId) -> (f64, f64) {
    const CHORDS: u32 = 64;
    let edge = topo.edge(edge).unwrap();
    let a = topo.vertex(edge.start()).unwrap().point();
    let b = topo.vertex(edge.end()).unwrap().point();
    let curve = edge.curve();
    let (t0, t1) = curve.domain_with_endpoints(a, b);
    let at = |k: u32| {
        curve.evaluate_with_endpoints(
            (t1 - t0).mul_add(f64::from(k) / f64::from(CHORDS), t0),
            a,
            b,
        )
    };
    let length = (0..CHORDS).map(|k| (at(k + 1) - at(k)).length()).sum();
    let miss = (at(0) - a).length().max((at(CHORDS) - b).length());
    (length, miss)
}

fn check(off: f64, r: f64, pose: &Mat4, edges: Edges) {
    let label = format!("off {off}, r {r}, {edges:?}");
    let mut topo = Topology::new();
    let (solid, edge) = skinned_box(&mut topo, off, pose, edges);
    let before = solid_volume(&topo, solid, 0.01).unwrap();
    assert!(
        is_watertight(&tessellate_solid(&topo, solid, 0.05).unwrap()),
        "{label}: input mesh not watertight"
    );
    let faces = solid_faces(&topo, solid).unwrap().len();
    let result = fillet_v2(&mut topo, solid, &[edge], r)
        .and_then(all_blended)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    // Each edge reads its own piece of its curve, from vertex to vertex. The
    // top keeps the contact and its back edge, `SIDE` long, and the far
    // pieces of its two side edges, `SIDE - r` each: a piece read as its
    // whole edge adds `r`.
    for edge in solid_edges(&topo, result).unwrap() {
        let (_, miss) = read_edge(&topo, edge);
        assert!(
            miss <= 1e-6,
            "{label}: an edge's ends fall {miss} from its vertices"
        );
    }
    let top = solid_faces(&topo, result)
        .unwrap()
        .into_iter()
        .find(|&f| {
            matches!(topo.face(f).unwrap().surface(),
                FaceSurface::Nurbs(skin) if skin.degree_u() == 1 && skin.degree_v() == 1)
        })
        .unwrap();
    let outer = topo.face(top).unwrap().outer_wire();
    let perimeter: f64 = topo
        .wire(outer)
        .unwrap()
        .edges()
        .iter()
        .map(|oriented| read_edge(&topo, oriented.edge()).0)
        .sum();
    let around = 2.0f64.mul_add(-r, 4.0 * SIDE);
    assert!(
        (perimeter - around).abs() <= off + 1e-6,
        "{label}: the top's boundary reads {perimeter} long, want {around}"
    );
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "{label}: mesh not watertight");
    assert_eq!(
        solid_faces(&topo, result).unwrap().len(),
        faces + 1,
        "{label}: faces"
    );
    let removed = before - solid_volume(&topo, result, 0.01).unwrap();
    let want = (1.0 - std::f64::consts::FRAC_PI_4) * r * r * SIDE;
    // The contact on the top runs on its surface, `off` above the edges the
    // end faces keep, and the skin's volume is read through its corners,
    // two of them now on that contact: the strip and the skin's reading
    // each move by about `off` times their area.
    assert!(
        (removed - want).abs() <= off * SIDE * (SIDE + 2.0 * r) + 1e-6,
        "{label}: removed {removed}, want {want}"
    );
}

#[test]
fn a_round_along_a_skin_whose_edges_stand_off_it() {
    for off in [0.0, 1e-5, 3e-4, 2e-3] {
        for r in [1.0, 2.5] {
            check(off, r, &Mat4::identity(), Edges::Lines);
        }
    }
}

#[test]
fn a_turned_skin_rounds_alike() {
    let pose =
        Mat4::translation(-37.2, 5.5, 140.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9);
    check(3e-4, 1.0, &pose, Edges::Lines);
}

/// The side edges a contact ends on are split where it ends, off them by as
/// far as they stand off the skin: each piece of a spline edge keeps its
/// own span of the curve, not the whole edge, and ends where the contact
/// does.
#[test]
fn a_skin_with_spline_edges_rounds_alike() {
    let along = Edges::Splines {
        degree: 1,
        backward: false,
    };
    for off in [0.0, 1e-5, 3e-4, 2e-3] {
        for r in [1.0, 2.5] {
            check(off, r, &Mat4::identity(), along);
        }
    }
    for off in [3e-4, 2e-3] {
        check(
            off,
            2.5,
            &Mat4::identity(),
            Edges::Splines {
                degree: 1,
                backward: true,
            },
        );
        for backward in [false, true] {
            check(
                off,
                1.0,
                &Mat4::identity(),
                Edges::Splines {
                    degree: 3,
                    backward,
                },
            );
        }
    }
    let pose =
        Mat4::translation(-37.2, 5.5, 140.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9);
    check(2e-3, 1.0, &pose, along);
}
