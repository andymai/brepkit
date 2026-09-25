//! A ball cut out of a box leaves its sphere faces reversed: the builder flips
//! the flag and keeps the wire, which runs about the sphere's outward normal.
//! A pocket bounded by one circle (a dimple) meshes over its own side of the
//! circle, a cavity's hemispheres each over their own half, a dimpled
//! box scaled unevenly keeps the dimple it had, and a hollowed ball's inner
//! wall meshes as the bowl it bounds.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{face_area, oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_sphere};
use brepkit_operations::shell_op::shell;
use brepkit_operations::tessellate::{is_watertight, tessellate, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// The box `[-5, 5]³` less a ball of radius 2 centred at `(0, 0, z)`.
fn box_less_ball(z: f64) -> (Topology, SolidId) {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(-5.0, -5.0, -5.0)).unwrap();
    let ball = make_sphere(&mut topo, 2.0, 32).unwrap();
    transform_solid(&mut topo, ball, &Mat4::translation(0.0, 0.0, z)).unwrap();
    let piece = boolean(&mut topo, BooleanOp::Cut, block, ball).unwrap();
    (topo, piece)
}

fn mesh_area(topo: &Topology, face: brepkit_topology::face::FaceId) -> f64 {
    let mesh = tessellate(topo, face, 0.005).unwrap();
    mesh.indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
            (b - a).cross(c - a).length() / 2.0
        })
        .sum()
}

/// A ball of radius 2 at `(0, 0, 5.5)` bites a cap `h = 1.5` deep out of the
/// box's top: the box loses `pi h² (3r - h) / 3`, and the dimple's face has
/// `2 pi r h` of the sphere, both in the exact measure and in the meshes.
#[test]
fn dimple_meshes_its_own_side() {
    let (r, h) = (2.0_f64, 1.5_f64);
    let truth = 1000.0 - PI * h * h * (3.0 * r - h) / 3.0;
    let (topo, piece) = box_less_ball(5.5);
    let volume = solid_volume(&topo, piece, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "volume {volume}, truth {truth}"
    );
    let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
    assert!(is_watertight(&mesh), "open or non-manifold mesh");
    let meshed = oriented_solid_volume(&topo, piece, 0.005).unwrap();
    assert!(
        (meshed - truth).abs() < 1e-3 * truth,
        "mesh volume {meshed}, truth {truth}"
    );
    let dimples: Vec<_> = solid_faces(&topo, piece)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
        .collect();
    assert_eq!(dimples.len(), 1, "one sphere face");
    assert!(
        topo.face(dimples[0]).unwrap().is_reversed(),
        "turned inward"
    );
    let cap = 2.0 * PI * r * h;
    for (label, area) in [
        ("exact", face_area(&topo, dimples[0], 0.005).unwrap()),
        ("mesh", mesh_area(&topo, dimples[0])),
    ] {
        assert!(
            (area - cap).abs() < 1e-2 * cap,
            "{label} area {area}, cap {cap}"
        );
    }
}

/// A ball fully inside the box leaves a cavity bounded by its two reversed
/// hemispheres; each hemisphere's own mesh covers its own half, the half its
/// wire leaves on its left about the sphere's outward normal.
#[test]
fn cavity_hemispheres_mesh_their_own_halves() {
    let (topo, piece) = box_less_ball(0.0);
    let volume = solid_volume(&topo, piece, 0.01).unwrap();
    let truth = 1000.0 - 4.0 / 3.0 * PI * 8.0;
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "volume {volume}, truth {truth}"
    );
    let mut halves = 0;
    for face in solid_faces(&topo, piece).unwrap() {
        let data = topo.face(face).unwrap();
        if !matches!(data.surface(), FaceSurface::Sphere(_)) {
            continue;
        }
        halves += 1;
        // The wire's turn about +z: counter-clockwise leaves the north half
        // on its left.
        let wire = topo.wire(data.outer_wire()).unwrap();
        let mut turn = 0.0;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            let (a, b) = if oe.is_forward() { (a, b) } else { (b, a) };
            turn += a.x().mul_add(b.y(), -(a.y() * b.x()));
        }
        let mesh = tessellate(&topo, face, 0.01).unwrap();
        #[allow(clippy::cast_precision_loss)]
        let mean_z =
            mesh.positions.iter().map(|p| p.z()).sum::<f64>() / mesh.positions.len() as f64;
        assert!(
            mean_z * turn > 0.0,
            "a hemisphere turning {turn} meshed around z {mean_z}"
        );
    }
    assert_eq!(halves, 2, "two hemispheres");
}

/// Scaling the dimpled box by 1.5 along x rebuilds the dimple as the scaled
/// sphere's NURBS image: the volume scales by the same 1.5.
#[test]
fn dimple_survives_an_uneven_scale() {
    let (r, h) = (2.0_f64, 1.5_f64);
    let truth = 1.5 * (1000.0 - PI * h * h * (3.0 * r - h) / 3.0);
    let (mut topo, piece) = box_less_ball(5.5);
    let mut stretch = Mat4::identity();
    stretch.0[0][0] = 1.5;
    transform_solid(&mut topo, piece, &stretch).unwrap();
    let volume = solid_volume(&topo, piece, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-3 * truth,
        "volume {volume}, truth {truth}"
    );
}

/// A ball of radius 10 hollowed to a 1-thick wall with its north hemisphere
/// open is a bowl: its inner wall is the reversed south hemisphere of radius
/// 9, whose own mesh stays below the rim, and the rim's hole winds as a hole.
#[test]
fn hollowed_ball_meshes_its_bowl() {
    let truth = 2.0 / 3.0 * PI * (1000.0 - 729.0);
    let mut topo = Topology::new();
    let ball = make_sphere(&mut topo, 10.0, 32).unwrap();
    let mean_z = |topo: &Topology, f| {
        let mesh = tessellate(topo, f, 0.05).unwrap();
        #[allow(clippy::cast_precision_loss)]
        let n = mesh.positions.len() as f64;
        mesh.positions.iter().map(|p| p.z()).sum::<f64>() / n
    };
    let north = solid_faces(&topo, ball)
        .unwrap()
        .into_iter()
        .max_by(|&a, &b| mean_z(&topo, a).total_cmp(&mean_z(&topo, b)))
        .unwrap();
    let bowl = shell(&mut topo, ball, 1.0, &[north]).unwrap();
    assert!(
        validate_solid(&topo, bowl).unwrap().is_valid(),
        "invalid bowl"
    );
    let volume = solid_volume(&topo, bowl, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "volume {volume}, truth {truth}"
    );
    let mesh = tessellate_solid(&topo, bowl, 0.01).unwrap();
    assert!(is_watertight(&mesh), "open or non-manifold mesh");
    let meshed = oriented_solid_volume(&topo, bowl, 0.01).unwrap();
    assert!(
        (meshed - truth).abs() < 1e-3 * truth,
        "mesh volume {meshed}, truth {truth}"
    );
    let inner: Vec<_> = solid_faces(&topo, bowl)
        .unwrap()
        .into_iter()
        .filter(|&f| topo.face(f).unwrap().is_reversed())
        .collect();
    assert_eq!(inner.len(), 1, "one inner wall");
    let wall = tessellate(&topo, inner[0], 0.05).unwrap();
    assert!(
        wall.positions.iter().all(|p| p.z() < 1e-6),
        "the inner wall meshed above its rim"
    );
}

/// The dimple and the cavity turned so the ball's axis leaves world z: each
/// reversed sphere face still meshes on its own side of its rim's plane, and
/// the solid mesh stays closed at the solid's volume.
#[test]
fn turned_dimple_and_cavity_mesh_their_own_sides() {
    let (r, h) = (2.0_f64, 1.5_f64);
    for (z, truth) in [
        (5.5, 1000.0 - PI * h * h * (3.0 * r - h) / 3.0),
        (0.0, 1000.0 - 4.0 / 3.0 * PI * 8.0),
    ] {
        for (label, turn) in [
            ("x90", Mat4::rotation_x(std::f64::consts::FRAC_PI_2)),
            ("x60", Mat4::rotation_x(1.0)),
            ("y90", Mat4::rotation_y(std::f64::consts::FRAC_PI_2)),
        ] {
            let dimple = z > 0.0;
            let (mut topo, piece) = box_less_ball(z);
            transform_solid(&mut topo, piece, &turn).unwrap();
            let origin = turn.mul_point(brepkit_math::vec::Point3::new(0.0, 0.0, 0.0));
            let up = turn.mul_point(brepkit_math::vec::Point3::new(0.0, 0.0, 1.0)) - origin;
            let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
            assert!(is_watertight(&mesh), "z {z} {label}: open mesh");
            let meshed = oriented_solid_volume(&topo, piece, 0.005).unwrap();
            assert!(
                (meshed - truth).abs() < 1e-3 * truth,
                "z {z} {label}: mesh volume {meshed}, truth {truth}"
            );
            let mut sides = Vec::new();
            for face in solid_faces(&topo, piece).unwrap() {
                if !matches!(topo.face(face).unwrap().surface(), FaceSurface::Sphere(_)) {
                    continue;
                }
                let heights: Vec<f64> = tessellate(&topo, face, 0.01)
                    .unwrap()
                    .positions
                    .iter()
                    .map(|p| (*p - origin).dot(up))
                    .collect();
                if dimple {
                    assert!(
                        heights.iter().all(|&t| t < 5.0 + 1e-6),
                        "z {z} {label}: the dimple meshed above the box's top"
                    );
                } else {
                    let below = heights.iter().all(|&t| t < 1e-6);
                    let above = heights.iter().all(|&t| t > -1e-6);
                    assert!(
                        below || above,
                        "z {z} {label}: a hemisphere crossed its rim"
                    );
                    sides.push(above);
                }
            }
            if !dimple {
                sides.sort_unstable();
                assert_eq!(
                    sides,
                    [false, true],
                    "z {z} {label}: one hemisphere each side"
                );
            }
        }
    }
}

/// Both point classifiers read a reversed sphere face's side from its wire:
/// points in the dimple or the cavity are outside the solid, points past the
/// ball are inside.
#[test]
fn reversed_caps_classify_by_their_wire() {
    use brepkit_check::classify::{ClassifyOptions, PointClassification as Check};
    use brepkit_math::vec::Point3;
    use brepkit_operations::classify::{PointClassification as Ops, classify_point};
    for (z, points) in [
        (
            5.5,
            [
                ((0.0, 0.0, 4.0), false),
                ((1.9, 0.0, 4.9), false),
                ((0.3, 0.2, 4.5), false),
                ((0.0, 0.0, 3.0), true),
                ((0.0, 0.0, -4.0), true),
            ],
        ),
        (
            0.0,
            [
                ((0.0, 0.0, 0.0), false),
                ((0.0, 0.0, 1.5), false),
                ((1.2, -0.9, -0.8), false),
                ((0.0, 0.0, 2.5), true),
                ((0.0, 0.0, -2.5), true),
            ],
        ),
    ] {
        let (topo, piece) = box_less_ball(z);
        for ((x, y, pz), inside) in points {
            let p = Point3::new(x, y, pz);
            let ops = classify_point(&topo, piece, p, 0.01, 1e-6).unwrap();
            assert_eq!(
                ops,
                if inside { Ops::Inside } else { Ops::Outside },
                "ball at z {z}: operations classifier at {p:?}"
            );
            let check = brepkit_check::classify::classify_point(
                &topo,
                piece,
                p,
                &ClassifyOptions::default(),
            )
            .unwrap();
            assert_eq!(
                check,
                if inside {
                    Check::Inside
                } else {
                    Check::Outside
                },
                "ball at z {z}: check classifier at {p:?}"
            );
        }
    }
}

/// A reversed face keeps its wire about its surface's normal, so validation
/// raises no orientation warning on a dimple, a cavity, a pocket's walls or a
/// bore's wall.
#[test]
fn reversed_faces_raise_no_orientation_warning() {
    use brepkit_check::validate::{ValidateOptions, validate_solid as check_solid};
    use brepkit_operations::primitives::make_cylinder;
    for tool in 0..4 {
        let mut topo = Topology::new();
        let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        transform_solid(&mut topo, block, &Mat4::translation(-5.0, -5.0, -5.0)).unwrap();
        let (label, cutter) = match tool {
            0 => {
                let b = make_sphere(&mut topo, 2.0, 32).unwrap();
                transform_solid(&mut topo, b, &Mat4::translation(0.0, 0.0, 5.5)).unwrap();
                ("dimple", b)
            }
            1 => ("cavity", make_sphere(&mut topo, 2.0, 32).unwrap()),
            2 => {
                let b = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
                transform_solid(&mut topo, b, &Mat4::translation(-1.0, -1.0, 4.0)).unwrap();
                ("pocket", b)
            }
            _ => {
                let c = make_cylinder(&mut topo, 1.0, 20.0).unwrap();
                transform_solid(&mut topo, c, &Mat4::translation(0.0, 0.0, -10.0)).unwrap();
                ("bore", c)
            }
        };
        let piece = boolean(&mut topo, BooleanOp::Cut, block, cutter).unwrap();
        let report = check_solid(&topo, piece, &ValidateOptions::default()).unwrap();
        let warned = report
            .issues
            .iter()
            .filter(|i| i.description.contains("face normal inconsistent"))
            .count();
        assert_eq!(warned, 0, "{label}: orientation warnings");
    }
}
