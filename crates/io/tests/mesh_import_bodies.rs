//! A mesh import gives one solid for each closed body and one shell for
//! each open one, with the triangles of a body joined at shared edges.

#![allow(clippy::unwrap_used, clippy::expect_used)]

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
