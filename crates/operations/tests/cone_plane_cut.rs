//! A cone cut by a plane that crosses its whole wall, level or tilted, keeping
//! either side. The plane meets the wall in a circle or an ellipse, and the
//! cone between the apex and the plane is a cone over that section: a third
//! of its area times the apex's distance to the plane. Seen along the axis the
//! section encloses `proj`, the wall between the apex and the plane covers
//! the same region, and the wall's normal makes a fixed angle with the axis.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::primitives::{make_box, make_cone};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

struct Case {
    label: String,
    top_radius: f64,
    slope: f64,
    turn: f64,
    keep_tip: bool,
    planes: usize,
    volume: f64,
    wall: f64,
    caps: f64,
}

/// Cones of base radius 3 and height 6 under the plane
/// `z = 3 + slope (x cos(turn) + y sin(turn))`.
fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for slope in [0.0_f64, 0.3, 0.6, 0.9] {
        for turn_deg in [0.0_f64, 60.0, 90.0, 200.0] {
            // Pointed: apex (0, 0, 6), 2 units of rise per unit of radius.
            let proj = 18.0 * PI / (4.0 - slope * slope).powf(1.5);
            let root5 = 5.0_f64.sqrt();
            // Frustum to radius 1.5: apex (0, 0, 12), 4 units of rise.
            let fproj = 324.0 * PI / (16.0 - slope * slope).powf(1.5);
            let root17 = 17.0_f64.sqrt();
            // The section leans off the base plane by the plane's tilt.
            let lean = slope.hypot(1.0);
            for (top_radius, keep_tip, planes, volume, wall, caps) in [
                (
                    0.0,
                    false,
                    2,
                    18.0 * PI - proj,
                    root5 * (9.0 * PI - proj),
                    9.0 * PI + lean * proj,
                ),
                (0.0, true, 1, proj, root5 * proj, lean * proj),
                (
                    1.5,
                    false,
                    2,
                    36.0 * PI - 3.0 * fproj,
                    root17 * (9.0 * PI - fproj),
                    9.0 * PI + lean * fproj,
                ),
                (
                    1.5,
                    true,
                    2,
                    3.0 * fproj - 4.5 * PI,
                    root17 * (fproj - 2.25 * PI),
                    2.25 * PI + lean * fproj,
                ),
            ] {
                let label = format!(
                    "{} {}, slope {slope}, turned {turn_deg} degrees",
                    if top_radius > 0.0 { "frustum" } else { "cone" },
                    if keep_tip { "tip" } else { "base" },
                );
                out.push(Case {
                    label,
                    top_radius,
                    slope,
                    turn: turn_deg.to_radians(),
                    keep_tip,
                    planes,
                    volume,
                    wall,
                    caps,
                });
            }
        }
    }
    out
}

#[test]
fn cone_cut_by_a_plane_across_its_wall() {
    for case in cases() {
        let label = &case.label;
        let mut topo = Topology::new();
        let cone = make_cone(&mut topo, 3.0, case.top_radius, 6.0).unwrap();
        let lid = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
        let place = Mat4::rotation_z(case.turn)
            * Mat4::translation(0.0, 0.0, 3.0)
            * Mat4::rotation_y(-case.slope.atan())
            * Mat4::translation(-10.0, -10.0, 0.0);
        transform_solid(&mut topo, lid, &place).unwrap();
        let op = if case.keep_tip {
            BooleanOp::Intersect
        } else {
            BooleanOp::Cut
        };
        let piece = boolean(&mut topo, op, cone, lid).unwrap();

        let report = validate_solid(&topo, piece).unwrap();
        assert!(report.is_valid(), "{label}: {:?}", report.issues);
        let faces = solid_faces(&topo, piece).unwrap();
        let planes = faces
            .iter()
            .filter(|&&f| topo.face(f).unwrap().surface().is_planar())
            .count();
        assert_eq!(
            (faces.len(), planes),
            (case.planes + 1, case.planes),
            "{label}: faces"
        );

        let volume = solid_volume(&topo, piece, 0.01).unwrap();
        assert!(
            (volume - case.volume).abs() < 1e-9 * case.volume,
            "{label}: volume {volume}, truth {}",
            case.volume
        );
        let wall = faces
            .iter()
            .copied()
            .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cone(_)))
            .expect("a cone wall");
        let area = face_area(&topo, wall, 0.01).unwrap();
        assert!(
            (area - case.wall).abs() < 1e-9 * case.wall,
            "{label}: wall area {area}, truth {}",
            case.wall
        );

        let caps: f64 = faces
            .iter()
            .filter(|&&f| topo.face(f).unwrap().surface().is_planar())
            .map(|&f| face_area(&topo, f, 0.01).unwrap())
            .sum();
        assert!(
            (caps - case.caps).abs() < 1e-9 * case.caps,
            "{label}: cap area {caps}, truth {}",
            case.caps
        );

        let (s, c) = case.turn.sin_cos();
        let plane_z = |x: f64, y: f64| 3.0 + case.slope * (x * c + y * s);
        let (below, above) = if case.keep_tip {
            (PointClassification::Outside, PointClassification::Inside)
        } else {
            (PointClassification::Inside, PointClassification::Outside)
        };
        for (x, y) in [
            (0.0, 0.0),
            (0.8 * c, 0.8 * s),
            (-0.8 * c, -0.8 * s),
            (-0.8 * s, 0.8 * c),
        ] {
            let z = plane_z(x, y);
            let at = |z: f64| {
                classify_point(
                    &topo,
                    piece,
                    Point3::new(x, y, z),
                    &ClassifyOptions::default(),
                )
                .unwrap()
            };
            assert_eq!(at(z - 0.05), below, "{label}: below at ({x}, {y})");
            assert_eq!(at(z + 0.05), above, "{label}: above at ({x}, {y})");
        }

        let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
        let meshed: f64 = mesh
            .indices
            .chunks_exact(3)
            .map(|t| {
                let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
                (a - Point3::new(0.0, 0.0, 0.0)).dot((b - a).cross(c - a)) / 6.0
            })
            .sum();
        // Inscribed: short of the solid by the rims' chords only (a level
        // cut's r = 1.5 rim takes 27 chords at this deflection).
        assert!(
            meshed <= case.volume && case.volume - meshed < 1.5e-2 * case.volume,
            "{label}: mesh volume {meshed}, truth {}",
            case.volume
        );
    }
}
