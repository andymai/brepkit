//! Captured-operand repro for the gridfinity bin whose floor pattern meshed
//! hollow corners: a 2×2×5 bin with scoops on its back and left walls and a
//! round-hole floor pattern.
//!
//! `floorscoop_body.bin` is the bin body as the tool's boolean stage receives
//! it, and `floorscoop_floor_hole.bin` is one of the pattern's 46 hole cutters,
//! well clear of every corner. Any one of them reproduces it.
//!
//! The cut is exact and valid. At the `(-38, -38)` corner it merges the three
//! cylinder faces stacked along the axis at z = 27.65 and 30.25 into one face,
//! and keeps those two vertices on its side lines because the flat walls
//! beside it are still split there. The solid mesher's Delaunay over that
//! boundary in raw `(u, v)` fanned each side vertex to rim samples a quarter
//! turn away, so three corners carried flat chords through the wall and a
//! vertical ray read them as hollows up to about 14mm tall. Every later
//! boolean of the body inherited it, so the tool's preview and its exported
//! STL both showed the voids.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{self, BooleanOptions};
use brepkit_operations::tessellate::{TriangleMesh, tessellate_solid_grouped_with_tolerance};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn cut_body(topo: &mut Topology) -> SolidId {
    let body = load(topo, "floorscoop_body.bin");
    let hole = load(topo, "floorscoop_floor_hole.bin");
    boolean::compound_cut(topo, body, &[hole], BooleanOptions::default()).unwrap()
}

/// Heights where a vertical line through `(x, y)` crosses the mesh.
fn column(mesh: &TriangleMesh, x: f64, y: f64) -> Vec<f64> {
    let mut zs = Vec::new();
    for t in mesh.indices.chunks_exact(3) {
        let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
        let d = (b.y() - c.y()) * (a.x() - c.x()) + (c.x() - b.x()) * (a.y() - c.y());
        if d.abs() < 1e-14 {
            continue;
        }
        let l1 = ((b.y() - c.y()) * (x - c.x()) + (c.x() - b.x()) * (y - c.y())) / d;
        let l2 = ((c.y() - a.y()) * (x - c.x()) + (a.x() - c.x()) * (y - c.y())) / d;
        let l3 = 1.0 - l1 - l2;
        if l1 >= 0.0 && l2 >= 0.0 && l3 >= 0.0 {
            zs.push(l1 * a.z() + l2 * b.z() + l3 * c.z());
        }
    }
    zs.sort_by(f64::total_cmp);
    zs.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    zs
}

/// Inside each corner's arc, 0.37 in from both walls: wall material from the
/// floor up to the lip.
const CORNERS: [(f64, f64); 4] = [
    (-40.18, -40.14),
    (-40.18, 39.86),
    (39.82, -40.14),
    (39.82, 39.86),
];

const DEFLECTIONS: [f64; 2] = [0.01, 0.1];

#[test]
fn cut_body_meshes_its_corners_solid() {
    let mut topo = Topology::new();
    let result = cut_body(&mut topo);
    for deflection in DEFLECTIONS {
        let (mesh, _) = tessellate_solid_grouped_with_tolerance(
            &topo,
            result,
            deflection,
            brepkit_math::chord::DEFAULT_ANGULAR_TOL,
        )
        .unwrap();
        for (x, y) in CORNERS {
            let zs = column(&mesh, x, y);
            assert!(
                zs.len() == 2 && zs[1] > 33.0,
                "deflection {deflection}: column ({x}, {y}) crosses {zs:?}, not only the floor and the lip"
            );
        }
    }
}

#[test]
fn corner_walls_stay_on_their_cylinders() {
    let mut topo = Topology::new();
    let result = cut_body(&mut topo);
    let faces = solid_faces(&topo, result).unwrap();
    for deflection in DEFLECTIONS {
        let (mesh, offsets) = tessellate_solid_grouped_with_tolerance(
            &topo,
            result,
            deflection,
            brepkit_math::chord::DEFAULT_ANGULAR_TOL,
        )
        .unwrap();
        let mut walls = 0;
        for (i, &face) in faces.iter().enumerate() {
            let FaceSurface::Cylinder(cylinder) = topo.face(face).unwrap().surface() else {
                continue;
            };
            if cylinder.axis().z().abs() < 0.99 {
                continue;
            }
            walls += 1;
            let (r, o) = (cylinder.radius(), cylinder.origin());
            let tris = &mesh.indices[offsets[i] as usize..offsets[i + 1] as usize];
            for t in tris.chunks_exact(3) {
                let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
                let (cx, cy) = ((a.x() + b.x() + c.x()) / 3.0, (a.y() + b.y() + c.y()) / 3.0);
                let sag = r - (cx - o.x()).hypot(cy - o.y());
                assert!(
                    sag <= deflection * 1.01,
                    "deflection {deflection}: face {face:?} (r {r}) has a triangle {sag:.4} inside its cylinder"
                );
            }
        }
        assert!(
            walls >= 4,
            "expected the four outer corner cylinders, found {walls}"
        );
    }
}
