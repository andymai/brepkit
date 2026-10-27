//! A solid with many faces is ray cast through an index of which faces each
//! ray can meet. Every ray must read exactly what reading every face reads:
//! the same crossing parity, the same graze report, and the same vote.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_algo::classifier::{RayCastGeoms, ray_cast_inside_votes_cached, ray_parity_cached};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOptions, compound_cut};
use brepkit_operations::heal::convert_to_bspline;
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere, make_torus};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn grid_cut(
    t: &mut Topology,
    height: f64,
    tool: impl Fn(&mut Topology) -> SolidId,
    place: impl Fn(f64, f64) -> Mat4,
) -> SolidId {
    let plate = make_box(t, 12.0, 12.0, height).unwrap();
    let mut tools = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            let s = tool(t);
            let (x, y) = (
                2.0f64.mul_add(f64::from(i), 1.5),
                2.0f64.mul_add(f64::from(j), 1.5),
            );
            transform_solid(t, s, &place(x, y)).unwrap();
            tools.push(s);
        }
    }
    compound_cut(t, plate, &tools, BooleanOptions::default()).unwrap()
}

fn kinds(t: &Topology, s: SolidId) -> Vec<&'static str> {
    solid_faces(t, s)
        .unwrap()
        .iter()
        .map(|&f| t.face(f).unwrap().surface().type_tag())
        .collect()
}

fn directions() -> Vec<Vec3> {
    let mut dirs = vec![
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(
            0.447_213_595_499_957_9,
            0.547_722_557_505_166_1,
            std::f64::consts::FRAC_1_SQRT_2,
        ),
        Vec3::new(-0.5, 0.763_762_615_825_973_4, 0.408_248_290_463_863),
    ];
    dirs.push(Vec3::new(1.0, 1.0, 0.0).normalize().unwrap());
    dirs
}

/// `stride` thins the grid in x and y, and the random points with it, for
/// faces whose full scan is slow.
fn assert_index_matches(name: &str, t: &Topology, solid: SolidId, height: f64, stride: usize) {
    let indexed = RayCastGeoms::new(t, solid).unwrap();
    assert!(indexed.is_indexed(), "{name}: too few faces to index");
    let full = RayCastGeoms::new(t, solid).unwrap().without_index();

    let mut points = Vec::new();
    // On the plate's planes and between them, where rays run in faces and
    // graze edges.
    for ix in (-1..=13).step_by(stride) {
        for iy in (-1..=13).step_by(stride) {
            for iz in -2..=(2.0 * height) as i32 + 2 {
                points.push(Point3::new(
                    f64::from(ix),
                    f64::from(iy),
                    f64::from(iz) * 0.5,
                ));
            }
        }
    }
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..1000 / (stride * stride) {
        points.push(Point3::new(
            next().mul_add(14.0, -1.0),
            next().mul_add(14.0, -1.0),
            next().mul_add(height + 2.0, -1.0),
        ));
    }

    let dirs = directions();
    for p in points {
        for &d in &dirs {
            assert_eq!(
                ray_parity_cached(&indexed, p, d),
                ray_parity_cached(&full, p, d),
                "{name}: ray from {p:?} along {d:?}"
            );
        }
        assert_eq!(
            ray_cast_inside_votes_cached(&indexed, p).unwrap(),
            ray_cast_inside_votes_cached(&full, p).unwrap(),
            "{name}: votes at {p:?}"
        );
    }
}

#[test]
fn straight_and_tilted_cylinder_holes() {
    let mut t = Topology::default();
    let s = grid_cut(
        &mut t,
        2.0,
        |t| make_cylinder(t, 0.6, 6.0).unwrap(),
        |x, y| Mat4::translation(x, y, -2.0),
    );
    assert!(kinds(&t, s).contains(&"cylinder"));
    assert_index_matches("straight cylinders", &t, s, 2.0, 1);

    let mut t = Topology::default();
    let s = grid_cut(
        &mut t,
        2.0,
        |t| make_cylinder(t, 0.5, 8.0).unwrap(),
        // Rows pulled in so no leaning hole breaks through a side wall.
        |x, y| {
            Mat4::translation(x, 0.9f64.mul_add(y - 1.5, 1.4), 1.0)
                * Mat4::rotation_x(0.35)
                * Mat4::translation(0.0, 0.0, -4.0)
        },
    );
    assert!(kinds(&t, s).contains(&"cylinder"));
    assert_index_matches("tilted cylinders", &t, s, 2.0, 1);
}

#[test]
fn straight_and_tilted_cone_holes() {
    let mut t = Topology::default();
    let s = grid_cut(
        &mut t,
        2.0,
        |t| make_cone(t, 0.8, 0.3, 4.0).unwrap(),
        |x, y| Mat4::translation(x, y, -1.0),
    );
    assert!(kinds(&t, s).contains(&"cone"));
    assert_index_matches("straight cones", &t, s, 2.0, 1);

    let mut t = Topology::default();
    let s = grid_cut(
        &mut t,
        2.0,
        |t| make_cone(t, 0.7, 0.3, 6.0).unwrap(),
        |x, y| {
            Mat4::translation(x, y, 1.0) * Mat4::rotation_y(0.3) * Mat4::translation(0.0, 0.0, -3.0)
        },
    );
    assert!(kinds(&t, s).contains(&"cone"));
    assert_index_matches("tilted cones", &t, s, 2.0, 1);
}

#[test]
fn sphere_and_torus_pockets() {
    let mut t = Topology::default();
    let s = grid_cut(
        &mut t,
        3.0,
        |t| make_sphere(t, 0.7, 16).unwrap(),
        |x, y| Mat4::translation(x, y, 3.0),
    );
    assert!(kinds(&t, s).contains(&"sphere"));
    assert_index_matches("spheres", &t, s, 3.0, 1);

    let mut t = Topology::default();
    let s = grid_cut(
        &mut t,
        3.0,
        |t| make_torus(t, 0.6, 0.2, 16).unwrap(),
        |x, y| Mat4::translation(x, y, 3.0),
    );
    assert_index_matches("tori", &t, s, 3.0, 1);
}

/// Curved faces the classifier reads as a polygon against a fitted plane:
/// a hit counts where it lands inside the polygon seen along the plane's
/// steepest axis, which can lie off the polygon's own points.
#[test]
fn curved_faces_read_as_polygons() {
    for (name, tool) in [("nurbs spheres", 0), ("nurbs tori", 1)] {
        let mut t = Topology::default();
        let s = grid_cut(
            &mut t,
            3.0,
            |t| {
                if tool == 0 {
                    make_sphere(t, 0.7, 16).unwrap()
                } else {
                    make_torus(t, 0.6, 0.2, 16).unwrap()
                }
            },
            |x, y| Mat4::translation(x, y, 3.0),
        );
        convert_to_bspline(&mut t, s).unwrap();
        assert!(kinds(&t, s).iter().all(|k| *k == "nurbs"), "{name}");
        assert_index_matches(name, &t, s, 3.0, 2);
    }
}
