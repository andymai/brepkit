#![allow(
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr,
    missing_docs
)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives;
use brepkit_operations::tessellate;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use std::time::Instant;

fn main() {
    let mut topo = Topology::new();
    let mut result = primitives::make_box(&mut topo, 100.0, 100.0, 10.0).unwrap();
    for row in 0..8_u32 {
        for col in 0..8_u32 {
            let cyl = primitives::make_cylinder(&mut topo, 2.0, 20.0).unwrap();
            let x = 6.0 + f64::from(col) * 12.0;
            let y = 6.0 + f64::from(row) * 12.0;
            let mat = Mat4::translation(x, y, -5.0);
            transform_solid(&mut topo, cyl, &mat).unwrap();
            result = boolean(&mut topo, BooleanOp::Cut, result, cyl).unwrap();
        }
    }

    let faces = brepkit_topology::explorer::solid_faces(&topo, result).unwrap();
    eprintln!("Faces: {}", faces.len());

    let edge_map = brepkit_topology::explorer::edge_to_face_map(&topo, result).unwrap();
    eprintln!("Unique edges (from edge_to_face_map): {}", edge_map.len());

    // Count actual edges from faces
    let mut total_face_edges = 0;
    for &fid in &faces {
        let fedges = brepkit_topology::explorer::face_edges(&topo, fid).unwrap();
        total_face_edges += fedges.len();
        let face = topo.face(fid).unwrap();
        let inner = face.inner_wires().len();
        let surface_name = match face.surface() {
            brepkit_topology::face::FaceSurface::Plane { .. } => "Plane",
            brepkit_topology::face::FaceSurface::Cylinder { .. } => "Cylinder",
            brepkit_topology::face::FaceSurface::Cone { .. } => "Cone",
            brepkit_topology::face::FaceSurface::Sphere { .. } => "Sphere",
            brepkit_topology::face::FaceSurface::Torus { .. } => "Torus",
            brepkit_topology::face::FaceSurface::Nurbs(_) => "Nurbs",
        };
        eprintln!(
            "  {} face: {} edges, {} inner wires",
            surface_name,
            fedges.len(),
            inner
        );
    }
    eprintln!("Total face edges (with duplicates): {}", total_face_edges);

    // Arena sizes
    eprintln!(
        "Arena: vertices={} edges={} faces={}",
        topo.vertices().len(),
        topo.edges().len(),
        topo.faces().len()
    );

    // Time
    for _ in 0..3 {
        let t0 = Instant::now();
        let mesh = tessellate::tessellate_solid(&topo, result, 0.1).unwrap();
        let elapsed = t0.elapsed();
        eprintln!(
            "tessellate_solid: {:?} | Verts: {} | Tris: {}",
            elapsed,
            mesh.positions.len(),
            mesh.indices.len() / 3
        );
    }
}
