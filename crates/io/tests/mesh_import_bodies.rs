//! A mesh import gives one solid for each closed body and one shell for
//! each open one, with the triangles of a body joined at shared edges.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_2;

use brepkit_io::stl::writer::StlFormat;
use brepkit_io::stl::{import_mesh, import_mesh_parts, read_stl, write_stl};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::compound_ops::solid_pieces;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::TriangleMesh;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const CUBE: [[u32; 3]; 12] = [
    [0, 2, 1],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
    [3, 7, 6],
    [3, 6, 2],
    [0, 4, 7],
    [0, 7, 3],
    [1, 2, 6],
    [1, 6, 5],
];

/// Add a cube of edge `size` with its low corner at `at`, with corners of
/// its own. `inward` winds each triangle the other way.
fn add_cube(mesh: &mut TriangleMesh, at: [f64; 3], size: f64, inward: bool) {
    let base = u32::try_from(mesh.positions.len()).unwrap();
    for (x, y, z) in [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ] {
        mesh.positions.push(Point3::new(
            at[0] + x * size,
            at[1] + y * size,
            at[2] + z * size,
        ));
    }
    for [a, b, c] in CUBE {
        let tri = if inward { [a, c, b] } else { [a, b, c] };
        mesh.indices.extend(tri.map(|i| base + i));
    }
}

/// Add a unit square of two triangles at height `z`.
fn add_sheet(mesh: &mut TriangleMesh, z: f64) {
    let base = u32::try_from(mesh.positions.len()).unwrap();
    for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
        mesh.positions.push(Point3::new(x, y, z));
    }
    mesh.indices
        .extend([0, 1, 2, 0, 2, 3].map(|i: u32| base + i));
}

fn volume(topo: &Topology, solid: SolidId) -> f64 {
    solid_volume(topo, solid, 0.01).unwrap()
}

fn assert_valid(topo: &Topology, solid: SolidId) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
}

#[test]
fn a_closed_cube_is_one_solid_with_one_shell() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1, 0));
    let solid = parts.solids[0];
    assert!(topo.solid(solid).unwrap().inner_shells().is_empty());
    assert_eq!(solid_faces(&topo, solid).unwrap().len(), 12);
    // 12 cube edges and one diagonal on each of the 6 sides.
    assert_eq!(topo.edges().len(), 18);
    assert_eq!(topo.vertices().len(), 8);
    assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
    assert_valid(&topo, solid);
    assert_eq!(solid_pieces(&mut topo, solid).unwrap(), vec![solid]);
}

#[test]
fn an_stl_box_reads_back_as_one_piece() {
    let mut source = Topology::new();
    let b = make_box(&mut source, 10.0, 10.0, 10.0).unwrap();
    let bytes = write_stl(&source, &[b], 0.1, StlFormat::Binary).unwrap();
    let mesh = read_stl(&bytes).unwrap();

    let mut topo = Topology::new();
    let solid = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!(solid_pieces(&mut topo, solid).unwrap().len(), 1);
    assert!((volume(&topo, solid) - 1000.0).abs() < 1e-6);
    assert_valid(&topo, solid);
}

#[test]
fn two_separate_cubes_are_two_solids() {
    let mut source = Topology::new();
    let a = make_box(&mut source, 10.0, 10.0, 10.0).unwrap();
    let b = make_box(&mut source, 10.0, 10.0, 10.0).unwrap();
    transform_solid(&mut source, b, &Mat4::translation(20.0, 0.0, 0.0)).unwrap();
    let bytes = write_stl(&source, &[a, b], 0.1, StlFormat::Binary).unwrap();
    let mesh = read_stl(&bytes).unwrap();

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    for &solid in &parts.solids {
        assert!((volume(&topo, solid) - 1000.0).abs() < 1e-6);
        assert_valid(&topo, solid);
    }

    // As one solid, the two cubes are two pieces of it.
    let mut topo = Topology::new();
    let joined = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert!((volume(&topo, joined) - 2000.0).abs() < 1e-6);
    assert_eq!(solid_pieces(&mut topo, joined).unwrap().len(), 2);
}

#[test]
fn an_inward_cube_in_a_cube_is_its_cavity() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 20.0, false);
    add_cube(&mut mesh, [5.0; 3], 10.0, true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1, 0));
    let solid = parts.solids[0];
    assert_eq!(topo.solid(solid).unwrap().inner_shells().len(), 1);
    assert!((volume(&topo, solid) - 7000.0).abs() < 1e-9);
    assert_valid(&topo, solid);

    let mut topo = Topology::new();
    let one = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!(topo.solid(one).unwrap().inner_shells().len(), 1);
}

#[test]
fn an_outward_cube_in_a_cube_is_a_solid_of_its_own() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 20.0, false);
    add_cube(&mut mesh, [5.0; 3], 10.0, false);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    let mut volumes: Vec<f64> = parts.solids.iter().map(|&s| volume(&topo, s)).collect();
    volumes.sort_by(f64::total_cmp);
    assert!((volumes[0] - 1000.0).abs() < 1e-9);
    assert!((volumes[1] - 8000.0).abs() < 1e-9);
}

/// One solid cannot hold a body in the material of another: as pieces of
/// one solid, the inner body reads as a cavity.
#[test]
fn an_outward_cube_in_a_cube_is_not_one_solid() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 20.0, false);
    add_cube(&mut mesh, [5.0; 3], 10.0, false);

    let mut topo = Topology::new();
    assert!(import_mesh(&mut topo, &mesh, 1e-7).is_err());
    // A refused import adds nothing.
    assert_eq!(topo.vertices().len(), 0);
    assert_eq!(topo.faces().len(), 0);
    assert_eq!(topo.solids().len(), 0);
}

/// An inward body with no body round it is turned into a solid. An outward
/// body in it then lies in its material.
#[test]
fn an_outward_cube_in_an_inward_cube_is_not_one_solid() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 20.0, true);
    add_cube(&mut mesh, [5.0; 3], 10.0, false);

    let mut topo = Topology::new();
    assert!(import_mesh(&mut topo, &mesh, 1e-7).is_err());
    assert_eq!(topo.faces().len(), 0);

    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    let mut volumes: Vec<f64> = parts.solids.iter().map(|&s| volume(&topo, s)).collect();
    volumes.sort_by(f64::total_cmp);
    assert!((volumes[0] - 1000.0).abs() < 1e-9);
    assert!((volumes[1] - 8000.0).abs() < 1e-9);
}

/// An inward body directly in an inward body is in a void. It is an island
/// wound the wrong way: a solid, in the cavity of the outer body.
#[test]
fn an_inward_cube_in_a_cavity_is_turned_into_a_solid() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 30.0, false);
    add_cube(&mut mesh, [5.0; 3], 20.0, true);
    add_cube(&mut mesh, [10.0; 3], 10.0, true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    let mut volumes: Vec<f64> = parts.solids.iter().map(|&s| volume(&topo, s)).collect();
    volumes.sort_by(f64::total_cmp);
    assert!((volumes[0] - 1000.0).abs() < 1e-9);
    assert!((volumes[1] - 19000.0).abs() < 1e-9);

    // As one solid, its pieces are the same two bodies.
    let mut topo = Topology::new();
    let joined = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    let pieces = solid_pieces(&mut topo, joined).unwrap();
    let mut volumes: Vec<f64> = pieces.iter().map(|&s| volume(&topo, s)).collect();
    volumes.sort_by(f64::total_cmp);
    assert_eq!(volumes.len(), 2);
    assert!((volumes[0] - 1000.0).abs() < 1e-9);
    assert!((volumes[1] - 19000.0).abs() < 1e-9);
}

/// Two cubes that touch at one corner are two bodies with no vertex in
/// common, so a move of one leaves the other where it was.
#[test]
fn two_cubes_that_touch_at_a_corner_share_no_vertex() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    add_cube(&mut mesh, [10.0; 3], 10.0, false);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    assert_eq!(topo.vertices().len(), 16);

    transform_solid(
        &mut topo,
        parts.solids[1],
        &Mat4::translation(5.0, 0.0, 0.0),
    )
    .unwrap();
    for &solid in &parts.solids {
        assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
        assert_valid(&topo, solid);
    }
    let first = brepkit_topology::explorer::solid_vertices(&topo, parts.solids[0]).unwrap();
    assert!(
        first
            .iter()
            .all(|&v| topo.vertex(v).unwrap().point().x() <= 10.0)
    );
}

#[test]
fn an_island_in_a_cavity_is_a_solid_of_its_own() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 30.0, false);
    add_cube(&mut mesh, [5.0; 3], 20.0, true);
    add_cube(&mut mesh, [10.0; 3], 10.0, false);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    let mut volumes: Vec<f64> = parts.solids.iter().map(|&s| volume(&topo, s)).collect();
    volumes.sort_by(f64::total_cmp);
    assert!((volumes[0] - 1000.0).abs() < 1e-9);
    assert!((volumes[1] - 19000.0).abs() < 1e-9);
    for &solid in &parts.solids {
        assert_valid(&topo, solid);
    }
}

#[test]
fn an_inward_cube_alone_is_turned_into_a_solid() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, true);

    let mut topo = Topology::new();
    let solid = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
    assert_valid(&topo, solid);
}

#[test]
fn a_triangle_wound_against_its_neighbours_is_turned() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    mesh.indices.swap(4, 5);
    mesh.indices.swap(31, 32);

    let mut topo = Topology::new();
    let solid = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
    assert_valid(&topo, solid);
}

#[test]
fn an_open_sheet_is_a_shell() {
    let mut mesh = TriangleMesh::default();
    add_sheet(&mut mesh, 0.0);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (0, 1));
    assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 2);
    assert_eq!(topo.solids().len(), 0);

    let mut topo = Topology::new();
    assert!(import_mesh(&mut topo, &mesh, 1e-7).is_err());
}

#[test]
fn an_open_sheet_beside_a_cube_is_a_shell_beside_a_solid() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    add_sheet(&mut mesh, 50.0);
    // A loose triangle is a shell of its own.
    let base = u32::try_from(mesh.positions.len()).unwrap();
    mesh.positions.extend([
        Point3::new(0.0, 0.0, 80.0),
        Point3::new(1.0, 0.0, 80.0),
        Point3::new(1.0, 1.0, 80.0),
    ]);
    mesh.indices.extend([base, base + 1, base + 2]);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1, 2));
    assert!((volume(&topo, parts.solids[0]) - 1000.0).abs() < 1e-9);
    assert_eq!(solid_faces(&topo, parts.solids[0]).unwrap().len(), 12);
    assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 2);
    assert_eq!(topo.shell(parts.shells[1]).unwrap().faces().len(), 1);
}

#[test]
fn a_cube_with_a_side_missing_is_a_shell() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    mesh.indices.truncate(30);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (0, 1));
    assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 10);
}

#[test]
fn two_triangles_back_to_back_are_a_shell() {
    let mesh = TriangleMesh {
        positions: vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ],
        normals: Vec::new(),
        indices: vec![0, 1, 2, 0, 2, 1],
    };

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (0, 1));
}

#[test]
fn an_index_past_the_positions_is_an_error() {
    let mesh = TriangleMesh {
        positions: vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
        normals: Vec::new(),
        indices: vec![0, 1, 2],
    };
    let mut topo = Topology::new();
    assert!(import_mesh_parts(&mut topo, &mesh, 1e-7).is_err());
}

#[test]
fn positions_merge_within_the_tolerance_and_not_past_it() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    // Give each triangle corners of its own, a little off the shared ones.
    let mut split = TriangleMesh::default();
    for (n, &i) in mesh.indices.iter().enumerate() {
        let p = mesh.positions[i as usize];
        #[allow(clippy::cast_precision_loss)]
        let off = 1e-9 * (n % 3) as f64;
        split
            .positions
            .push(Point3::new(p.x() + off, p.y() - off, p.z() + off));
        split.indices.push(u32::try_from(n).unwrap());
    }

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &split, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1, 0));
    assert_eq!(topo.vertices().len(), 8);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &split, 0.0).unwrap();
    assert!(parts.solids.is_empty());
}

/// A reader with no normals in its file gives each vertex one direction.
/// Such normals say nothing, and the winding stands.
#[test]
fn normals_in_one_direction_leave_a_cavity_inward() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 20.0, false);
    add_cube(&mut mesh, [5.0; 3], 10.0, true);
    mesh.normals = vec![Vec3::new(0.0, 0.0, 1.0); mesh.positions.len()];

    let mut topo = Topology::new();
    let solid = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!(topo.solid(solid).unwrap().inner_shells().len(), 1);
    assert!((volume(&topo, solid) - 7000.0).abs() < 1e-9);
}

/// Normals that point against the winding of a whole body turn the body.
#[test]
fn normals_against_the_winding_turn_a_body() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 20.0, false);
    // Wound outward, with normals that point to its centre: a cavity.
    add_cube(&mut mesh, [5.0; 3], 10.0, false);
    let centre = Point3::new(10.0, 10.0, 10.0);
    mesh.normals = mesh.positions[..8]
        .iter()
        .map(|&p| p - centre)
        .chain(mesh.positions[8..].iter().map(|&p| centre - p))
        .collect();

    let mut topo = Topology::new();
    let solid = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!(topo.solid(solid).unwrap().inner_shells().len(), 1);
    assert!((volume(&topo, solid) - 7000.0).abs() < 1e-9);
    assert_valid(&topo, solid);
}

#[test]
fn a_long_thin_box_is_a_solid() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 1.0, false);
    for p in &mut mesh.positions {
        *p = Point3::new(p.x() * 1000.0, p.y() * 1e-4, p.z() * 1e-4);
    }

    let mut topo = Topology::new();
    let solid = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    let v = volume(&topo, solid);
    assert!((v - 1e-5).abs() < 1e-12, "volume {v}");
}

/// The first corner of the inner body is a corner of the outer one, where
/// the count of turns round the point has no meaning.
#[test]
fn a_body_that_touches_the_corner_of_its_container_is_in_it() {
    let tetra = |mesh: &mut TriangleMesh, inward: bool| {
        let base = u32::try_from(mesh.positions.len()).unwrap();
        mesh.positions.extend([
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.5, 0.2, 0.2),
            Point3::new(0.2, 0.5, 0.2),
            Point3::new(0.2, 0.2, 0.5),
        ]);
        for [a, b, c] in [[0u32, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]] {
            let tri = if inward { [a, c, b] } else { [a, b, c] };
            mesh.indices.extend(tri.map(|i| base + i));
        }
    };

    let mut mesh = TriangleMesh::default();
    tetra(&mut mesh, true);
    add_cube(&mut mesh, [0.0; 3], 2.0, false);
    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1, 0));
    let tetra_volume = 0.081 / 6.0;
    let v = volume(&topo, parts.solids[0]);
    assert!((v - (8.0 - tetra_volume)).abs() < 1e-9, "volume {v}");

    let mut mesh = TriangleMesh::default();
    tetra(&mut mesh, false);
    add_cube(&mut mesh, [0.0; 3], 2.0, false);
    let mut topo = Topology::new();
    assert!(import_mesh(&mut topo, &mesh, 1e-7).is_err());
}

#[test]
fn a_fan_of_triangles_on_one_edge_is_one_shell() {
    let mut mesh = TriangleMesh {
        positions: vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 1.0)],
        ..TriangleMesh::default()
    };
    for k in 0..2000_u32 {
        let angle = f64::from(k) * 0.003;
        mesh.positions
            .push(Point3::new(angle.cos(), angle.sin(), 0.5));
        mesh.indices.extend([0, 1, 2 + k]);
    }

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (0, 1));
    assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 2000);
}

/// The gap between the two boxes is far below the length of the outer one,
/// and far above the merge tolerance.
#[test]
fn a_cavity_close_to_the_wall_of_a_long_thin_box_is_its_cavity() {
    let scaled = |at: [f64; 3], size: [f64; 3], inward: bool, mesh: &mut TriangleMesh| {
        let first = mesh.positions.len();
        add_cube(mesh, [0.0; 3], 1.0, inward);
        for p in &mut mesh.positions[first..] {
            *p = Point3::new(
                at[0] + p.x() * size[0],
                at[1] + p.y() * size[1],
                at[2] + p.z() * size[2],
            );
        }
    };
    let mut mesh = TriangleMesh::default();
    scaled([0.0; 3], [1e6, 1e-4, 1e-4], false, &mut mesh);
    scaled([1.0, 2e-5, 2e-5], [10.0, 6e-5, 6e-5], true, &mut mesh);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1, 0));
    assert_eq!(topo.solid(parts.solids[0]).unwrap().inner_shells().len(), 1);
}

/// Add a prism of height `height` on `z = 0` over a footprint wound
/// counter-clockwise, with the triangles of its footprint.
fn add_prism(mesh: &mut TriangleMesh, footprint: &[(f64, f64)], caps: &[[u32; 3]], height: f64) {
    let base = u32::try_from(mesh.positions.len()).unwrap();
    let n = u32::try_from(footprint.len()).unwrap();
    for z in [0.0, height] {
        for &(x, y) in footprint {
            mesh.positions.push(Point3::new(x, y, z));
        }
    }
    for &[a, b, c] in caps {
        mesh.indices.extend([base + a, base + c, base + b]);
        mesh.indices
            .extend([base + n + a, base + n + b, base + n + c]);
    }
    for i in 0..n {
        let j = (i + 1) % n;
        mesh.indices.extend([base + i, base + j, base + n + j]);
        mesh.indices.extend([base + i, base + n + j, base + n + i]);
    }
}

fn sorted_volumes(topo: &Topology, solids: &[SolidId]) -> Vec<f64> {
    let mut volumes: Vec<f64> = solids.iter().map(|&s| volume(topo, s)).collect();
    volumes.sort_by(f64::total_cmp);
    volumes
}

#[test]
fn two_cubes_that_share_an_edge_are_two_solids() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    add_cube(&mut mesh, [10.0, 10.0, 0.0], 10.0, false);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    // Each cube has vertices and edges of its own.
    assert_eq!(topo.vertices().len(), 16);
    assert_eq!(topo.edges().len(), 36);
    for &solid in &parts.solids {
        assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
        assert_valid(&topo, solid);
        assert_eq!(solid_pieces(&mut topo, solid).unwrap(), vec![solid]);
    }

    // As one solid, the two cubes are two pieces of it.
    let mut topo = Topology::new();
    let joined = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert!((volume(&topo, joined) - 2000.0).abs() < 1e-9);
    assert_eq!(solid_pieces(&mut topo, joined).unwrap().len(), 2);
}

#[test]
fn two_stl_boxes_that_share_an_edge_are_two_solids() {
    let mut source = Topology::new();
    let a = make_box(&mut source, 10.0, 10.0, 10.0).unwrap();
    let b = make_box(&mut source, 10.0, 10.0, 10.0).unwrap();
    transform_solid(&mut source, b, &Mat4::translation(10.0, 10.0, 0.0)).unwrap();
    let bytes = write_stl(&source, &[a, b], 0.1, StlFormat::Binary).unwrap();
    let mesh = read_stl(&bytes).unwrap();

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    for &solid in &parts.solids {
        assert!((volume(&topo, solid) - 1000.0).abs() < 1e-6);
        assert_valid(&topo, solid);
    }
}

/// The two cubes share the four points and the two triangles of a side, so
/// each edge of the side has two triangles at one angle. Two cubes that
/// touch there and a wall of no thickness look the same: the edges do not
/// pair off, and the cubes are one shell.
#[test]
fn duplicated_triangles_on_a_shared_side_make_one_shell() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    add_cube(&mut mesh, [0.0, 0.0, 10.0], 10.0, false);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (0, 1));
    assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 24);
    assert!(import_mesh(&mut topo, &mesh, 1e-7).is_err());
}

/// Two triangles back to back on an edge of a cube: the edge has four
/// triangles, two of them at one angle.
#[test]
fn a_double_sided_fin_on_the_edge_of_a_cube_makes_a_shell() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    mesh.positions.push(Point3::new(15.0, 15.0, 5.0));
    // The edge from (10, 10, 0) to (10, 10, 10) is the edge from corner 2
    // to corner 6.
    mesh.indices.extend([2, 6, 8, 6, 2, 8]);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (0, 1));
    assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 14);
}

/// A triangle on the shared edge, wound against the rest of its cube, does
/// not change how the triangles pair off round the edge.
#[test]
fn a_turned_triangle_on_a_shared_edge_is_turned_back() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    add_cube(&mut mesh, [10.0, 10.0, 0.0], 10.0, false);
    // The first triangle of the side at x = 10 of the first cube is on the
    // shared edge.
    let at = 30;
    assert_eq!(mesh.indices[at..at + 3], [1, 2, 6]);
    mesh.indices.swap(at + 1, at + 2);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    for &solid in &parts.solids {
        assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
        assert_valid(&topo, solid);
    }
}

/// Round the shared edge, the material is the box round the two cubes, so
/// the two cubes are one cavity that pinches at the edge.
#[test]
fn two_inward_cubes_that_share_an_edge_are_a_cavity() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 30.0, false);
    add_cube(&mut mesh, [5.0, 5.0, 10.0], 10.0, true);
    add_cube(&mut mesh, [15.0, 15.0, 10.0], 10.0, true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1, 0));
    let solid = parts.solids[0];
    assert_eq!(topo.solid(solid).unwrap().inner_shells().len(), 1);
    assert!((volume(&topo, solid) - 25000.0).abs() < 1e-9);
    assert_valid(&topo, solid);
}

/// An edge with three triangles does not pair off: the cube and the fin
/// are one shell.
#[test]
fn a_fin_on_the_edge_of_a_cube_makes_a_shell() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    mesh.positions.push(Point3::new(15.0, 15.0, 5.0));
    // The edge from (10, 10, 0) to (10, 10, 10) is the edge from corner 2
    // to corner 6.
    mesh.indices.extend([2, 6, 8]);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (0, 1));
    assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 13);
}

/// The L-shaped prism and the inward cube cross. The first corner of the
/// cube lies in the prism, so one point read the cube as its cavity.
#[test]
fn an_inward_cube_that_crosses_a_prism_is_a_shell() {
    let mut mesh = TriangleMesh::default();
    add_prism(
        &mut mesh,
        &[
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 1.0),
            (1.0, 1.0),
            (1.0, 3.0),
            (0.0, 3.0),
        ],
        &[[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]],
        3.0,
    );
    add_cube(&mut mesh, [0.5; 3], 2.0, true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1, 1));
    let solid = parts.solids[0];
    assert!(topo.solid(solid).unwrap().inner_shells().is_empty());
    assert!((volume(&topo, solid) - 15.0).abs() < 1e-9);
    assert_valid(&topo, solid);
    assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 12);

    let mut topo = Topology::new();
    assert!(import_mesh(&mut topo, &mesh, 1e-7).is_err());
    assert_eq!(topo.faces().len(), 0);
}

/// Two outward cubes that cross are each a solid, and not one solid.
#[test]
fn two_cubes_that_cross_are_two_solids_and_not_one() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    add_cube(&mut mesh, [5.0; 3], 10.0, false);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    assert_eq!(sorted_volumes(&topo, &parts.solids), [1000.0, 1000.0]);

    let mut topo = Topology::new();
    assert!(import_mesh(&mut topo, &mesh, 1e-7).is_err());
    assert_eq!(topo.faces().len(), 0);
}

/// Two bars that cross like a plus sign: no corner of one is in the other.
#[test]
fn two_bars_that_cross_with_no_corner_in_the_other_are_not_one_solid() {
    let bar = |mesh: &mut TriangleMesh, size: [f64; 3], at: [f64; 3]| {
        let first = mesh.positions.len();
        add_cube(mesh, [0.0; 3], 1.0, false);
        for p in &mut mesh.positions[first..] {
            *p = Point3::new(
                at[0] + p.x() * size[0],
                at[1] + p.y() * size[1],
                at[2] + p.z() * size[2],
            );
        }
    };
    let mut mesh = TriangleMesh::default();
    bar(&mut mesh, [30.0, 10.0, 10.0], [0.0, 10.0, 10.0]);
    bar(&mut mesh, [10.0, 30.0, 20.0], [10.0, 0.0, 5.0]);

    let mut topo = Topology::new();
    assert!(import_mesh(&mut topo, &mesh, 1e-7).is_err());
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
}

/// Two cavities that cross bound no cavity of the solid round them.
#[test]
fn two_inward_cubes_that_cross_in_a_cube_are_shells() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 30.0, false);
    add_cube(&mut mesh, [5.0; 3], 10.0, true);
    add_cube(&mut mesh, [10.0; 3], 10.0, true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1, 2));
    let solid = parts.solids[0];
    assert!(topo.solid(solid).unwrap().inner_shells().is_empty());
    assert!((volume(&topo, solid) - 27000.0).abs() < 1e-9);
}

/// The cavity touches the wall of its body along a side: they touch, and do
/// not cross.
#[test]
fn a_cavity_that_touches_the_wall_of_its_body_is_its_cavity() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 30.0, false);
    add_cube(&mut mesh, [20.0, 5.0, 5.0], 10.0, true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1, 0));
    let solid = parts.solids[0];
    assert_eq!(topo.solid(solid).unwrap().inner_shells().len(), 1);
    assert!((volume(&topo, solid) - 26000.0).abs() < 1e-9);
}

/// Each corner of the octahedron is on a side of the cube round it, so the
/// centre of a triangle decides.
#[test]
fn an_inward_octahedron_with_its_corners_on_a_cube_is_its_cavity() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [-1.0; 3], 2.0, false);
    let base = u32::try_from(mesh.positions.len()).unwrap();
    mesh.positions.extend([
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(-1.0, 0.0, 0.0),
        Point3::new(0.0, 1.0, 0.0),
        Point3::new(0.0, -1.0, 0.0),
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(0.0, 0.0, -1.0),
    ]);
    // Wound inward: each triangle faces the centre.
    for [a, b, c] in [
        [0u32, 4, 2],
        [2, 4, 1],
        [1, 4, 3],
        [3, 4, 0],
        [0, 2, 5],
        [2, 1, 5],
        [1, 3, 5],
        [3, 0, 5],
    ] {
        mesh.indices.extend([base + a, base + b, base + c]);
    }

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1, 0));
    let solid = parts.solids[0];
    assert_eq!(topo.solid(solid).unwrap().inner_shells().len(), 1);
    assert!((volume(&topo, solid) - (8.0 - 4.0 / 3.0)).abs() < 1e-9);
}

#[test]
fn a_grid_of_separate_cubes_is_as_many_solids() {
    let mut mesh = TriangleMesh::default();
    for i in 0..10 {
        for j in 0..10 {
            for k in 0..10 {
                add_cube(
                    &mut mesh,
                    [f64::from(i) * 2.0, f64::from(j) * 2.0, f64::from(k) * 2.0],
                    1.0,
                    false,
                );
            }
        }
    }
    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1000, 0));
}

/// Shells nested in each other, outward and inward in turn: each outward
/// shell is a solid, with the inward shell in it as its cavity.
#[test]
fn nested_shells_are_solids_with_a_cavity_each() {
    let count = 200;
    let mut mesh = TriangleMesh::default();
    for s in 0..count {
        let at = f64::from(s);
        add_cube(&mut mesh, [at; 3], 2.0 * f64::from(count - s), s % 2 == 1);
    }
    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (100, 0));
    for &solid in &parts.solids {
        assert_eq!(topo.solid(solid).unwrap().inner_shells().len(), 1);
    }
}

/// Add a box with each side cut into `m` by `m` squares.
fn add_grid_box(mesh: &mut TriangleMesh, lo: [f64; 3], hi: [f64; 3], m: u32, inward: bool) {
    let at = |k: usize, i: u32| lo[k] + (hi[k] - lo[k]) * f64::from(i) / f64::from(m);
    for axis in 0..3 {
        for far in [false, true] {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let base = u32::try_from(mesh.positions.len()).unwrap();
            for i in 0..=m {
                for j in 0..=m {
                    let mut p = [0.0; 3];
                    p[axis] = if far { hi[axis] } else { lo[axis] };
                    p[u] = at(u, i);
                    p[v] = at(v, j);
                    mesh.positions.push(Point3::new(p[0], p[1], p[2]));
                }
            }
            let id = |i: u32, j: u32| base + i * (m + 1) + j;
            for i in 0..m {
                for j in 0..m {
                    let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
                    if far == inward {
                        mesh.indices.extend([a, c, b, a, d, c]);
                    } else {
                        mesh.indices.extend([a, b, c, a, c, d]);
                    }
                }
            }
        }
    }
}

#[test]
fn a_grid_box_is_a_solid() {
    let mut mesh = TriangleMesh::default();
    add_grid_box(&mut mesh, [0.0; 3], [10.0; 3], 2, false);
    let mut topo = Topology::new();
    let solid = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
    assert_valid(&topo, solid);
}

/// Wound inward, with normals that point out: the normals turn each cube
/// before its triangles pair off round the shared edge.
#[test]
fn normals_against_the_winding_turn_two_cubes_that_share_an_edge() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, true);
    add_cube(&mut mesh, [10.0, 10.0, 0.0], 10.0, true);
    let centres = [Point3::new(5.0, 5.0, 5.0), Point3::new(15.0, 15.0, 5.0)];
    mesh.normals = mesh
        .positions
        .iter()
        .enumerate()
        .map(|(i, &p)| p - centres[i / 8])
        .collect();

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    for &solid in &parts.solids {
        assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
        assert_valid(&topo, solid);
    }
}

/// With no tolerance, the margin of a test is the rounding of the
/// coordinates near it, and a body far away does not change it.
#[test]
fn a_distant_body_does_not_change_what_lies_near_the_origin() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 20.0, false);
    add_cube(&mut mesh, [5.0; 3], 10.0, true);
    add_cube(&mut mesh, [1e16; 3], 1024.0, false);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 0.0).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    let with_cavity: Vec<_> = parts
        .solids
        .iter()
        .filter(|&&s| topo.solid(s).unwrap().inner_shells().len() == 1)
        .collect();
    assert_eq!(with_cavity.len(), 1);
    assert!((volume(&topo, *with_cavity[0]) - 7000.0).abs() < 1e-9);

    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    add_cube(&mut mesh, [5.0; 3], 10.0, false);
    add_cube(&mut mesh, [1e16; 3], 1024.0, false);
    let mut topo = Topology::new();
    assert!(import_mesh(&mut topo, &mesh, 0.0).is_err());
}

/// The sides of the cube on `[5, 15]` cross the finer cube along the edges
/// of its triangles, where no triangle passes to both sides of the other.
#[test]
fn two_cubes_that_cross_along_edges_of_triangles_are_not_one_solid() {
    let mut mesh = TriangleMesh::default();
    add_grid_box(&mut mesh, [0.0; 3], [10.0; 3], 2, false);
    add_cube(&mut mesh, [5.0; 3], 10.0, false);

    let mut topo = Topology::new();
    assert!(import_mesh(&mut topo, &mesh, 1e-7).is_err());
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
}

/// The cavity touches the wall of the finer box along a side, across lines
/// of its triangles: they touch, and do not cross.
#[test]
fn a_cavity_that_touches_the_wall_of_a_finer_box_is_its_cavity() {
    let mut mesh = TriangleMesh::default();
    add_grid_box(&mut mesh, [0.0; 3], [20.0; 3], 4, false);
    add_cube(&mut mesh, [10.0, 5.0, 5.0], 10.0, true);

    let mut topo = Topology::new();
    let solid = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!(topo.solid(solid).unwrap().inner_shells().len(), 1);
    assert!((volume(&topo, solid) - 7000.0).abs() < 1e-9);
    // The cavity's face at x = 20 lies on the box's wall there, so the wall
    // between them has no thickness, and the validator reports the two faces.
    let report = validate_solid(&topo, solid).unwrap();
    assert!(
        !report.issues.is_empty()
            && report
                .issues
                .iter()
                .all(|issue| issue.description.contains("cover the same area")),
        "{:?}",
        report.issues
    );
}

/// Add a box from `lo` to `hi`, with corners of its own.
fn add_box(mesh: &mut TriangleMesh, lo: [f64; 3], hi: [f64; 3], inward: bool) {
    add_grid_box(mesh, lo, hi, 1, inward);
}

/// A reader with no normals in its file gives each vertex one direction.
/// Such normals say nothing, and the two cubes pair off at their shared
/// edge by their winding.
#[test]
fn normals_in_one_direction_leave_two_cubes_that_share_an_edge() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    add_cube(&mut mesh, [10.0, 10.0, 0.0], 10.0, false);
    mesh.normals = vec![Vec3::new(0.0, 0.0, 1.0); mesh.positions.len()];

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    for &solid in &parts.solids {
        assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
        assert_valid(&topo, solid);
    }
}

/// The two outward cubes cross, and a ray from the small cube leaves the
/// larger one first. The cavity goes to the smallest body round it.
#[test]
fn a_cavity_in_two_bodies_that_cross_goes_to_the_smaller() {
    let mut mesh = TriangleMesh::default();
    add_box(&mut mesh, [0.0; 3], [10.0; 3], false);
    add_box(&mut mesh, [-5.0, -1.0, -1.0], [7.0, 11.0, 11.0], false);
    add_box(&mut mesh, [1.0; 3], [2.0; 3], true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    let with_cavity: Vec<_> = parts
        .solids
        .iter()
        .filter(|&&s| topo.solid(s).unwrap().inner_shells().len() == 1)
        .collect();
    assert_eq!(with_cavity.len(), 1);
    assert!((volume(&topo, *with_cavity[0]) - 999.0).abs() < 1e-9);
}

/// The cavity touches the wall of the outer cube, so a ray from the island
/// can leave the cavity and the outer cube at one place.
#[test]
fn an_island_in_a_cavity_on_the_wall_of_its_body_is_a_solid_of_its_own() {
    let mut mesh = TriangleMesh::default();
    add_box(&mut mesh, [0.0; 3], [20.0; 3], false);
    add_box(&mut mesh, [10.0, 5.0, 5.0], [20.0, 15.0, 15.0], true);
    add_box(&mut mesh, [12.0, 8.0, 8.0], [14.0, 10.0, 10.0], false);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    assert_eq!(sorted_volumes(&topo, &parts.solids), [8.0, 7000.0]);

    let mut topo = Topology::new();
    let solid = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert!((volume(&topo, solid) - 7008.0).abs() < 1e-9);
}

/// The two cubes share material, and their surfaces meet only in the
/// planes of their sides.
#[test]
fn two_cubes_that_overlap_in_the_planes_of_their_sides_are_not_one_solid() {
    let mut mesh = TriangleMesh::default();
    add_box(&mut mesh, [0.0; 3], [10.0; 3], false);
    add_box(&mut mesh, [5.0, 0.0, 0.0], [15.0, 10.0, 10.0], false);

    let mut topo = Topology::new();
    let err = import_mesh(&mut topo, &mesh, 1e-7).unwrap_err();
    assert!(err.to_string().contains("cross"), "{err}");
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
}

/// The same two cubes far from the origin, where the coordinates are large
/// next to the overlap.
#[test]
fn two_cubes_far_away_that_overlap_in_the_planes_of_their_sides_are_not_one_solid() {
    let far = 2f64.powi(30);
    let mut mesh = TriangleMesh::default();
    add_box(&mut mesh, [far; 3], [far + 10.0; 3], false);
    add_box(
        &mut mesh,
        [far + 5.0, far, far],
        [far + 15.0, far + 10.0, far + 10.0],
        false,
    );

    let mut topo = Topology::new();
    let err = import_mesh(&mut topo, &mesh, 1e-7).unwrap_err();
    assert!(err.to_string().contains("cross"), "{err}");
}

/// Two cubes that touch along a side, from the two sides of it, are two
/// pieces of one solid.
#[test]
fn two_cubes_that_touch_along_a_side_are_one_solid() {
    let mut mesh = TriangleMesh::default();
    add_box(&mut mesh, [0.0; 3], [10.0; 3], false);
    add_box(&mut mesh, [10.0, 2.0, 2.0], [20.0, 8.0, 8.0], false);

    let mut topo = Topology::new();
    let solid = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert!((volume(&topo, solid) - 1360.0).abs() < 1e-9);
    assert_eq!(solid_pieces(&mut topo, solid).unwrap().len(), 2);
}

/// An outward cube on the wall inside another lies in its material: the
/// two do not cross.
#[test]
fn an_outward_cube_on_the_wall_inside_a_cube_lies_in_it() {
    let mut mesh = TriangleMesh::default();
    add_box(&mut mesh, [0.0; 3], [20.0; 3], false);
    add_box(&mut mesh, [10.0, 5.0, 5.0], [20.0, 15.0, 15.0], false);

    let mut topo = Topology::new();
    let err = import_mesh(&mut topo, &mesh, 1e-7).unwrap_err();
    assert!(err.to_string().contains("material"), "{err}");
}

/// Unit normals from `centre` out to each position from `first` on, or in
/// to `centre` when `inward`.
fn set_normals(mesh: &mut TriangleMesh, first: usize, centre: Point3, inward: bool) {
    mesh.normals
        .resize(mesh.positions.len(), Vec3::new(0.0, 0.0, 0.0));
    for i in first..mesh.positions.len() {
        let out = (mesh.positions[i] - centre).normalize().unwrap();
        mesh.normals[i] = if inward { -out } else { out };
    }
}

/// Both cubes are wound outward. The normals of the first point out and
/// those of the second point in, so over the two the normals neither agree
/// nor disagree with the winding: the shared edge does not pair off.
#[test]
fn normals_that_cancel_over_two_cubes_that_share_an_edge_make_a_shell() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    add_cube(&mut mesh, [10.0, 10.0, 0.0], 10.0, false);
    set_normals(&mut mesh, 0, Point3::new(5.0, 5.0, 5.0), false);
    set_normals(&mut mesh, 8, Point3::new(15.0, 15.0, 5.0), true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (0, 1));
    assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 24);
}

/// The normals of the large box agree with its winding and outweigh the
/// thin box, whose normals point against its winding. The thin box does
/// not keep the winding that its pairs at the shared edge took.
#[test]
fn a_body_wound_against_its_normals_at_a_shared_edge_is_a_shell() {
    let mut mesh = TriangleMesh::default();
    add_box(&mut mesh, [0.0; 3], [20.0; 3], false);
    set_normals(&mut mesh, 0, Point3::new(10.0, 10.0, 10.0), false);
    let first = mesh.positions.len();
    add_box(&mut mesh, [20.0, 20.0, 0.0], [25.0, 25.0, 20.0], false);
    set_normals(&mut mesh, first, Point3::new(22.5, 22.5, 10.0), true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (1, 1));
    assert!((volume(&topo, parts.solids[0]) - 8000.0).abs() < 1e-9);
    assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 12);
}

/// The two cubes that share an edge have normals in one direction, and a
/// cube apart from them has normals that point out. The one direction of the
/// first group says nothing, whatever the normals of the other group are.
#[test]
fn normals_in_one_direction_beside_true_normals_leave_two_cubes_that_share_an_edge() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, false);
    add_cube(&mut mesh, [10.0, 10.0, 0.0], 10.0, false);
    mesh.normals = vec![Vec3::new(0.0, 0.0, 1.0); mesh.positions.len()];
    add_cube(&mut mesh, [50.0; 3], 10.0, false);
    set_normals(&mut mesh, 16, Point3::new(55.0, 55.0, 55.0), false);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (3, 0));
    for &solid in &parts.solids {
        assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
        assert_valid(&topo, solid);
    }
}

/// Wound inside out, the two cubes put the material round them, and their
/// triangles pair off across the void round the shared edge. The import
/// turns the closed body that they make, and the pairs follow the turn.
#[test]
fn two_inward_cubes_that_share_an_edge_are_two_solids() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, true);
    add_cube(&mut mesh, [10.0, 10.0, 0.0], 10.0, true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    assert_eq!(topo.vertices().len(), 16);
    assert_eq!(topo.edges().len(), 36);
    for &solid in &parts.solids {
        assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
        assert_valid(&topo, solid);
        assert_eq!(solid_pieces(&mut topo, solid).unwrap(), vec![solid]);
    }

    let mut topo = Topology::new();
    let joined = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert!((volume(&topo, joined) - 2000.0).abs() < 1e-9);
    assert_eq!(solid_pieces(&mut topo, joined).unwrap().len(), 2);
}

/// Normals that point in agree with the winding: both put the material
/// round the cubes. The cubes lie in no body, so the import turns them.
#[test]
fn two_inward_cubes_with_inward_normals_that_share_an_edge_are_two_solids() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, true);
    add_cube(&mut mesh, [10.0, 10.0, 0.0], 10.0, true);
    set_normals(&mut mesh, 0, Point3::new(5.0, 5.0, 5.0), true);
    set_normals(&mut mesh, 8, Point3::new(15.0, 15.0, 5.0), true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    for &solid in &parts.solids {
        assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
        assert_valid(&topo, solid);
    }
}

/// Add a cube of edge `size` with one edge on the z axis from `z = lift`,
/// turned by `angle` round that axis.
fn add_turned_cube(mesh: &mut TriangleMesh, angle: f64, lift: f64, size: f64, inward: bool) {
    let first = mesh.positions.len();
    add_cube(mesh, [0.0, 0.0, lift], size, inward);
    let (s, c) = angle.sin_cos();
    for p in &mut mesh.positions[first..] {
        *p = Point3::new(c * p.x() - s * p.y(), s * p.x() + c * p.y(), p.z());
    }
}

/// Three cubes share the edge on the z axis, with a gap of 30 degrees
/// between each two.
#[test]
fn three_cubes_round_one_edge_are_three_solids() {
    for inward in [false, true] {
        let mut mesh = TriangleMesh::default();
        for k in 0..3 {
            let angle = f64::from(k) * std::f64::consts::TAU / 3.0;
            add_turned_cube(&mut mesh, angle, 0.0, 10.0, inward);
        }

        let mut topo = Topology::new();
        let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
        assert_eq!(
            (parts.solids.len(), parts.shells.len()),
            (3, 0),
            "inward {inward}"
        );
        assert_eq!(topo.vertices().len(), 24);
        for &solid in &parts.solids {
            assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
            assert_valid(&topo, solid);
        }

        let mut topo = Topology::new();
        let joined = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
        assert!((volume(&topo, joined) - 3000.0).abs() < 1e-9);
        assert_eq!(solid_pieces(&mut topo, joined).unwrap().len(), 3);
    }
}

/// A chain of three cubes, each two of them with an edge in common. Each
/// cube is a solid of its own, whichever way each cube is wound.
#[test]
fn a_chain_of_cubes_that_share_edges_is_a_solid_for_each_cube_in_any_winding() {
    for winding in 0..8_u32 {
        let inward = |k: u32| winding & (1 << k) != 0;
        let mut mesh = TriangleMesh::default();
        add_cube(&mut mesh, [0.0; 3], 10.0, inward(0));
        add_cube(&mut mesh, [10.0, 10.0, 0.0], 10.0, inward(1));
        add_cube(&mut mesh, [20.0, 0.0, 0.0], 10.0, inward(2));

        let mut topo = Topology::new();
        let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
        assert_eq!(
            (parts.solids.len(), parts.shells.len()),
            (3, 0),
            "winding {winding:03b}"
        );
        for &solid in &parts.solids {
            assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
            assert_valid(&topo, solid);
        }

        let mut topo = Topology::new();
        let joined = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
        assert!((volume(&topo, joined) - 3000.0).abs() < 1e-9);
        assert_eq!(solid_pieces(&mut topo, joined).unwrap().len(), 3);
    }
}

#[test]
fn two_inward_cubes_that_touch_at_a_corner_are_two_solids() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 10.0, true);
    add_cube(&mut mesh, [10.0; 3], 10.0, true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    for &solid in &parts.solids {
        assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
        assert_valid(&topo, solid);
    }
}

/// The inward cube round the two cubes lies in no body, and the two cubes
/// lie directly in it: the import turns all three. The two cubes then lie
/// in the material of the large one.
#[test]
fn two_inward_cubes_that_share_an_edge_in_an_inward_cube_are_solids_in_a_solid() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 30.0, true);
    add_cube(&mut mesh, [5.0, 5.0, 10.0], 10.0, true);
    add_cube(&mut mesh, [15.0, 15.0, 10.0], 10.0, true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (3, 0));
    assert_eq!(
        sorted_volumes(&topo, &parts.solids),
        [1000.0, 1000.0, 27000.0]
    );
    for &solid in &parts.solids {
        assert_valid(&topo, solid);
    }

    let mut topo = Topology::new();
    let err = import_mesh(&mut topo, &mesh, 1e-7).unwrap_err();
    assert!(err.to_string().contains("material"), "{err}");
}

/// The two inward cubes lie in the cavity of a cube: in a void, so the
/// import turns them into two islands.
#[test]
fn two_inward_cubes_that_share_an_edge_in_a_cavity_are_two_islands() {
    let mut mesh = TriangleMesh::default();
    add_cube(&mut mesh, [0.0; 3], 40.0, false);
    add_cube(&mut mesh, [5.0; 3], 30.0, true);
    add_cube(&mut mesh, [10.0, 10.0, 15.0], 10.0, true);
    add_cube(&mut mesh, [20.0, 20.0, 15.0], 10.0, true);

    let mut topo = Topology::new();
    let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
    assert_eq!((parts.solids.len(), parts.shells.len()), (3, 0));
    assert_eq!(
        sorted_volumes(&topo, &parts.solids),
        [1000.0, 1000.0, 37000.0]
    );
    for &solid in &parts.solids {
        assert_valid(&topo, solid);
    }

    let mut topo = Topology::new();
    let joined = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
    assert!((volume(&topo, joined) - 39000.0).abs() < 1e-9);
    assert_eq!(solid_pieces(&mut topo, joined).unwrap().len(), 3);
}

/// The second cube is turned by 120 degrees round the z axis, and moved up
/// by half its edge. The two touch along half of an edge, with no corner in
/// common there, and do not cross.
#[test]
fn two_cubes_that_touch_along_half_an_edge_are_one_solid() {
    for inward in [false, true] {
        let mut mesh = TriangleMesh::default();
        add_turned_cube(&mut mesh, 0.0, 0.0, 10.0, inward);
        add_turned_cube(&mut mesh, 120f64.to_radians(), 5.0, 10.0, inward);

        let mut topo = Topology::new();
        let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
        assert_eq!(
            (parts.solids.len(), parts.shells.len()),
            (2, 0),
            "inward {inward}"
        );
        for &solid in &parts.solids {
            assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9);
            assert_valid(&topo, solid);
        }

        let mut topo = Topology::new();
        let joined = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
        assert!((volume(&topo, joined) - 2000.0).abs() < 1e-9);
        assert_eq!(solid_pieces(&mut topo, joined).unwrap().len(), 2);
    }
}

/// The second cube is turned by 45 degrees: the cubes cross along half of
/// an edge.
#[test]
fn two_cubes_that_cross_along_half_an_edge_are_not_one_solid() {
    for inward in [false, true] {
        let mut mesh = TriangleMesh::default();
        add_turned_cube(&mut mesh, 0.0, 0.0, 10.0, inward);
        add_turned_cube(&mut mesh, 45f64.to_radians(), 5.0, 10.0, inward);

        let mut topo = Topology::new();
        let err = import_mesh(&mut topo, &mesh, 1e-7).unwrap_err();
        assert!(err.to_string().contains("cross"), "{err}");
        let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
        let expected = if inward { (0, 2) } else { (2, 0) };
        assert_eq!((parts.solids.len(), parts.shells.len()), expected);
    }
}

/// A mirror winds each triangle the other way. The two cubes that share an
/// edge are two solids in each pose: as they are, turned, mirrored, and
/// turned and mirrored.
#[test]
fn two_cubes_that_share_an_edge_are_two_solids_in_each_pose() {
    let turn = Mat4::rotation_z(0.7) * Mat4::rotation_y(0.4) * Mat4::rotation_x(1.1);
    let mirror = Mat4::scale(-1.0, 1.0, 1.0);
    for (name, pose) in [
        ("upright", Mat4::scale(1.0, 1.0, 1.0)),
        ("turned", turn),
        ("mirrored", mirror),
        ("turned and mirrored", mirror * turn),
    ] {
        let mut mesh = TriangleMesh::default();
        add_cube(&mut mesh, [0.0; 3], 10.0, false);
        add_cube(&mut mesh, [10.0, 10.0, 0.0], 10.0, false);
        for p in &mut mesh.positions {
            *p = pose.mul_point(*p);
        }

        let mut topo = Topology::new();
        let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
        assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0), "{name}");
        for &solid in &parts.solids {
            assert!((volume(&topo, solid) - 1000.0).abs() < 1e-9, "{name}");
            assert_valid(&topo, solid);
        }

        let mut topo = Topology::new();
        let joined = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
        assert!((volume(&topo, joined) - 2000.0).abs() < 1e-9, "{name}");
        assert_eq!(solid_pieces(&mut topo, joined).unwrap().len(), 2, "{name}");
    }
}

/// A hook round the corner `(10, 10)`: from that corner it runs along `+x`
/// and `+y`, and its other arm, `x` from 5 to 8, reaches down to `y = 5`.
const HOOK: [(f64, f64); 8] = [
    (10.0, 10.0),
    (20.0, 10.0),
    (20.0, 20.0),
    (5.0, 20.0),
    (5.0, 5.0),
    (8.0, 5.0),
    (8.0, 15.0),
    (10.0, 15.0),
];
const HOOK_CAPS: [[u32; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 7],
    [7, 2, 3],
    [7, 3, 6],
    [6, 3, 4],
    [6, 4, 5],
];

/// The cube and the hook-shaped prism share the cube's edge at `(10, 10)`,
/// and the arm of the hook passes through the cube.
#[test]
fn a_cube_and_a_prism_that_share_an_edge_and_cross_are_not_one_solid() {
    for inward in [false, true] {
        let mut mesh = TriangleMesh::default();
        add_cube(&mut mesh, [0.0; 3], 10.0, false);
        add_prism(&mut mesh, &HOOK, &HOOK_CAPS, 10.0);
        if inward {
            for tri in mesh.indices.chunks_exact_mut(3) {
                tri.swap(1, 2);
            }
        }

        let mut topo = Topology::new();
        let err = import_mesh(&mut topo, &mesh, 1e-7).unwrap_err();
        assert!(err.to_string().contains("cross"), "inward {inward}: {err}");
        let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
        let expected = if inward { (0, 2) } else { (2, 0) };
        assert_eq!(
            (parts.solids.len(), parts.shells.len()),
            expected,
            "inward {inward}"
        );
    }
}

/// The second cube is turned by a little less than 90 degrees, and its
/// corners move in `x` by `-2.5e-8` at `z = 5` to `-0.5e-8` at `z = 15`. It
/// lies against the side `x = 0` of the first cube within the tolerance,
/// and the two touch.
#[test]
fn two_cubes_that_touch_along_half_an_edge_within_the_tolerance_are_one_solid() {
    for inward in [false, true] {
        let mut mesh = TriangleMesh::default();
        add_turned_cube(&mut mesh, 0.0, 0.0, 10.0, inward);
        let first = mesh.positions.len();
        add_turned_cube(&mut mesh, FRAC_PI_2 - 1e-9, 5.0, 10.0, inward);
        for p in &mut mesh.positions[first..] {
            let shift = -2.5e-8 + (p.z() - 5.0) * 2e-9;
            *p = Point3::new(p.x() + shift, p.y(), p.z());
        }

        let mut topo = Topology::new();
        let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();
        assert_eq!(
            (parts.solids.len(), parts.shells.len()),
            (2, 0),
            "inward {inward}"
        );
        for &solid in &parts.solids {
            assert!((volume(&topo, solid) - 1000.0).abs() < 1e-6);
            assert_valid(&topo, solid);
        }

        let mut topo = Topology::new();
        let joined = import_mesh(&mut topo, &mesh, 1e-7).unwrap();
        assert!((volume(&topo, joined) - 2000.0).abs() < 1e-6);
        assert_eq!(solid_pieces(&mut topo, joined).unwrap().len(), 2);
    }
}
