//! A rectangular window cut through a tube or cone wall leaves the wall face
//! with holes. Four layers each had to get it right: the face splitter winds the
//! hole against the outward normal, the CDT mesher constrains and removes it,
//! the per-face mesher (exports, areas) does the same, and the volume
//! integrator subtracts it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder};
use brepkit_operations::tessellate::{
    boundary_edge_count, tessellate, tessellate_solid, tessellate_with_uvs,
};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const R: f64 = 1.5;
const H: f64 = 4.0;

/// Frustum radii for the cone cases: `R0` at z = 0 down to `R1` at z = `H`.
const R0: f64 = 2.0;
const R1: f64 = 1.0;

/// `∫ 2 sqrt(r² - y²) dy` over `[y0, y1]`: the chord area of a strip of a
/// disc of radius `r`.
fn disc_strip(r: f64, y0: f64, y1: f64) -> f64 {
    let f = |y: f64| y * (r * r - y * y).sqrt() + r * r * (y / r).asin();
    f(y1) - f(y0)
}

fn strip_area(y0: f64, y1: f64) -> f64 {
    disc_strip(R, y0, y1)
}

/// The frustum's material in the slab `y0..y1`, `z0..z1`, by Simpson's rule
/// over z (the chord strip is smooth in z there).
fn frustum_slab(y0: f64, y1: f64, z0: f64, z1: f64) -> f64 {
    let n = 200;
    let h = (z1 - z0) / f64::from(n);
    (0..=n)
        .map(|i| {
            let z = z0 + h * f64::from(i);
            let w = if i == 0 || i == n {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            w * disc_strip(R0 + (R1 - R0) * z / H, y0, y1)
        })
        .sum::<f64>()
        * h
        / 3.0
}

fn tube(topo: &mut Topology) -> SolidId {
    make_cylinder(topo, R, H).unwrap()
}

fn frustum(topo: &mut Topology) -> SolidId {
    make_cone(topo, R0, R1, H).unwrap()
}

struct Case {
    name: &'static str,
    solid: fn(&mut Topology) -> SolidId,
    pose: Mat4,
    /// Cutter box: origin corner, then extents.
    cutter: [f64; 6],
    /// A point in the removed window (world coordinates, like the cutter)
    /// and one in untouched wall material (in the tube's own frame).
    window: Point3,
    material: Point3,
    exact_volume: Option<f64>,
}

fn cases() -> Vec<Case> {
    let cylinder = std::f64::consts::PI * R * R * H;
    vec![
        Case {
            solid: tube,
            name: "prism through both walls",
            pose: Mat4::identity(),
            cutter: [-5.0, 0.3, 1.0, 10.0, 0.5, 0.5],
            window: Point3::new(0.0, 0.55, 1.25),
            material: Point3::new(0.0, -1.2, 2.0),
            exact_volume: Some(cylinder - 0.5 * strip_area(0.3, 0.8)),
        },
        Case {
            solid: tube,
            name: "prism through the u origin",
            pose: Mat4::identity(),
            cutter: [-0.25, -5.0, 1.0, 0.5, 10.0, 0.5],
            window: Point3::new(0.0, 1.45, 1.25),
            material: Point3::new(1.2, 0.0, 2.0),
            exact_volume: Some(cylinder - 0.5 * strip_area(-0.25, 0.25)),
        },
        Case {
            solid: tube,
            name: "blind pocket through one wall",
            pose: Mat4::identity(),
            cutter: [0.0, 0.3, 1.0, 5.0, 0.5, 0.5],
            window: Point3::new(1.0, 0.55, 1.25),
            material: Point3::new(-1.0, 0.55, 1.25),
            exact_volume: Some(cylinder - 0.25 * strip_area(0.3, 0.8)),
        },
        Case {
            solid: tube,
            name: "tilted tube, cap to wall",
            pose: Mat4::rotation_x(0.7),
            cutter: [0.3, 0.3, -25.0, 0.5, 0.5, 50.0],
            window: Point3::new(0.55, 0.55, 0.5),
            material: Point3::new(-0.8, -0.6, 1.8),
            exact_volume: None,
        },
        Case {
            solid: tube,
            name: "shallow tilt",
            pose: Mat4::rotation_x(0.3),
            cutter: [0.3, 0.3, -25.0, 0.5, 0.5, 50.0],
            window: Point3::new(0.55, 0.55, 0.5),
            material: Point3::new(-0.8, -0.6, 2.0),
            exact_volume: None,
        },
        Case {
            solid: tube,
            name: "tilted tube, centred prism",
            pose: Mat4::rotation_x(0.7),
            cutter: [-0.25, -0.25, -25.0, 0.5, 0.5, 50.0],
            window: Point3::new(0.0, 0.0, 0.5),
            material: Point3::new(0.8, -0.6, 1.8),
            exact_volume: None,
        },
        Case {
            name: "frustum, prism through both walls",
            solid: frustum,
            pose: Mat4::identity(),
            cutter: [-5.0, 0.3, 1.0, 10.0, 0.5, 0.5],
            window: Point3::new(0.0, 0.55, 1.25),
            material: Point3::new(0.0, -1.2, 2.0),
            exact_volume: Some(
                std::f64::consts::PI * H / 3.0 * (R0 * R0 + R0 * R1 + R1 * R1)
                    - frustum_slab(0.3, 0.8, 1.0, 1.5),
            ),
        },
        Case {
            name: "tilted frustum, cap to wall",
            solid: frustum,
            pose: Mat4::rotation_x(0.7),
            cutter: [0.3, 0.3, -25.0, 0.5, 0.5, 50.0],
            window: Point3::new(0.55, 0.55, 0.5),
            material: Point3::new(-0.8, -0.6, 1.8),
            exact_volume: None,
        },
    ]
}

fn cut(topo: &mut Topology, case: &Case) -> SolidId {
    let tube = (case.solid)(topo);
    transform_solid(topo, tube, &case.pose).unwrap();
    let [x, y, z, dx, dy, dz] = case.cutter;
    let cutter = make_box(topo, dx, dy, dz).unwrap();
    transform_solid(topo, cutter, &Mat4::translation(x, y, z)).unwrap();
    boolean(topo, BooleanOp::Cut, tube, cutter).unwrap()
}

#[test]
fn window_cut_is_exact_and_valid() {
    for case in cases() {
        let mut topo = Topology::new();
        let result = cut(&mut topo, &case);
        let faces = solid_faces(&topo, result).unwrap();
        let walls = faces
            .iter()
            .filter(|&&f| {
                matches!(
                    topo.face(f).unwrap().surface(),
                    FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
                )
            })
            .count();
        assert_eq!(walls, 1, "{}: one curved wall", case.name);
        assert!(
            faces.len() <= 12,
            "{}: {} faces, a mesh fallback",
            case.name,
            faces.len()
        );
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{}: {:?}", case.name, report.issues);

        let classify = |p: Point3| {
            brepkit_check::classify::classify_point(
                &topo,
                result,
                p,
                &brepkit_check::classify::ClassifyOptions::default(),
            )
            .unwrap()
        };
        let (window, material) = (case.window, case.pose.mul_point(case.material));
        assert_eq!(
            classify(window),
            brepkit_check::classify::PointClassification::Outside,
            "{}: the window is empty",
            case.name
        );
        assert_eq!(
            classify(material),
            brepkit_check::classify::PointClassification::Inside,
            "{}: the wall is solid",
            case.name
        );
    }
}

#[test]
fn window_cut_meshes_watertight_at_the_right_volume() {
    for case in cases() {
        let mut topo = Topology::new();
        let result = cut(&mut topo, &case);
        for deflection in [0.01, 0.001] {
            let mesh = tessellate_solid(&topo, result, deflection).unwrap();
            assert_eq!(
                boundary_edge_count(&mesh),
                0,
                "{} at {deflection}: open mesh",
                case.name
            );
        }
        let fine = oriented_solid_volume(&topo, result, 0.0005).unwrap();
        let exact = solid_volume(&topo, result, 0.001).unwrap();
        if let Some(truth) = case.exact_volume {
            assert!(
                (exact - truth).abs() < 1e-9 * truth,
                "{}: solid_volume {exact}, closed form {truth}",
                case.name
            );
            assert!(
                (fine - truth).abs() < 1e-3 * truth,
                "{}: mesh volume {fine}, closed form {truth}",
                case.name
            );
        } else {
            assert!(
                (fine - exact).abs() < 1e-3 * exact,
                "{}: mesh volume {fine}, solid_volume {exact}",
                case.name
            );
        }
    }
}

/// The per-face mesher behind the OBJ, PLY and glTF writers and face areas
/// must carve the holes too.
#[test]
fn holed_wall_meshes_on_its_own() {
    let case = &cases()[0];
    let mut topo = Topology::new();
    let result = cut(&mut topo, case);
    let wall = solid_faces(&topo, result)
        .unwrap()
        .into_iter()
        .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .unwrap();
    let mesh = tessellate(&topo, wall, 0.001).unwrap();
    let area: f64 = mesh
        .indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| mesh.positions[i as usize]);
            0.5 * (b - a).cross(c - a).length()
        })
        .sum();
    // Lateral area less two windows, each 0.5 tall over the arc between
    // y = 0.3 and y = 0.8.
    let arc = R * ((0.8 / R).asin() - (0.3 / R).asin());
    let expected = 2.0 * std::f64::consts::PI * R * H - 2.0 * 0.5 * arc;
    assert!(
        (area - expected).abs() < 1e-3 * expected,
        "wall mesh area {area}, expected {expected}"
    );
}

/// Seam vertices are welded in the curved CDT; the per-face UV mesh must
/// still give every triangle continuous texture coordinates.
#[test]
fn holed_wall_uvs_do_not_jump_across_the_seam() {
    let case = &cases()[1];
    let mut topo = Topology::new();
    let result = cut(&mut topo, case);
    let wall = solid_faces(&topo, result)
        .unwrap()
        .into_iter()
        .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .unwrap();
    let mesh = tessellate_with_uvs(&topo, wall, 0.01).unwrap();
    assert!(!mesh.mesh.indices.is_empty());
    for tri in mesh.mesh.indices.chunks_exact(3) {
        let us = [tri[0], tri[1], tri[2]].map(|i| mesh.uvs[i as usize][0]);
        let span = us.iter().copied().fold(f64::MIN, f64::max)
            - us.iter().copied().fold(f64::MAX, f64::min);
        assert!(span < 1.0, "triangle spans {span} in u");
    }
}
