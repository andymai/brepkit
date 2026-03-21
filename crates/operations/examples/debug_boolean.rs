#![allow(
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::print_stderr,
    missing_docs
)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure;
use brepkit_operations::primitives;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;

fn main() {
    let mut topo = Topology::new();
    let bx = primitives::make_box(&mut topo, 50.0, 30.0, 10.0).unwrap();
    let cyl = primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();

    // Translate cylinder to center of box
    let mat = Mat4::translation(25.0, 15.0, -5.0);
    transform_solid(&mut topo, cyl, &mat).unwrap();

    let box_vol = measure::solid_volume(&topo, bx, 0.1).unwrap();
    eprintln!("Box volume: {box_vol}");

    let result = boolean(&mut topo, BooleanOp::Cut, bx, cyl).unwrap();

    let solid = topo.solid(result).unwrap();
    let shell = topo.shell(solid.outer_shell()).unwrap();
    eprintln!("Result has {} faces", shell.faces().len());

    for (i, &fid) in shell.faces().iter().enumerate() {
        let face = topo.face(fid).unwrap();
        let surface = face.surface();
        let reversed = face.is_reversed();
        let inner_count = face.inner_wires().len();

        // Compute face area
        let area = measure::face_area(&topo, fid, 0.1).unwrap_or(0.0);

        let surf_type = match surface {
            brepkit_topology::face::FaceSurface::Plane { normal, d } => {
                format!(
                    "Plane(n={:.2},{:.2},{:.2} d={:.2})",
                    normal.x(),
                    normal.y(),
                    normal.z(),
                    d
                )
            }
            brepkit_topology::face::FaceSurface::Nurbs(_) => "Nurbs".to_string(),
            brepkit_topology::face::FaceSurface::Cylinder(_) => "Cylinder".to_string(),
            brepkit_topology::face::FaceSurface::Cone(_) => "Cone".to_string(),
            brepkit_topology::face::FaceSurface::Sphere(_) => "Sphere".to_string(),
            brepkit_topology::face::FaceSurface::Torus(_) => "Torus".to_string(),
        };

        eprintln!(
            "  Face {i}: {surf_type}  reversed={reversed}  inner_wires={inner_count}  area={area:.2}"
        );
    }

    // Compute per-face signed volume contribution
    eprintln!("\nPer-face signed volume (divergence theorem):");
    let mut total_sv = 0.0;
    for (i, &fid) in shell.faces().iter().enumerate() {
        let mesh = brepkit_operations::tessellate::tessellate(&topo, fid, 0.1).unwrap();
        let mut face_vol = 0.0;
        for tri in mesh.indices.chunks_exact(3) {
            let v0 = mesh.positions[tri[0] as usize];
            let v1 = mesh.positions[tri[1] as usize];
            let v2 = mesh.positions[tri[2] as usize];
            // Signed tet volume: v0 · (v1 × v2) / 6
            let cross = brepkit_math::vec::Vec3::new(
                v1.y() * v2.z() - v1.z() * v2.y(),
                v1.z() * v2.x() - v1.x() * v2.z(),
                v1.x() * v2.y() - v1.y() * v2.x(),
            );
            let dot = v0.x() * cross.x() + v0.y() * cross.y() + v0.z() * cross.z();
            face_vol += dot / 6.0;
        }
        eprintln!(
            "  Face {i}: signed_vol = {face_vol:.2}  ({} tris)",
            mesh.indices.len() / 3
        );
        total_sv += face_vol;
    }
    eprintln!("\nTotal signed volume: {total_sv:.2}");

    let vol = measure::solid_volume(&topo, result, 0.1).unwrap();
    let expected = 50.0 * 30.0 * 10.0 - std::f64::consts::PI * 25.0 * 10.0;
    eprintln!("measure volume:  {vol:.2}");
    eprintln!("Expected volume: {expected:.2}");
    eprintln!("Box volume:      {box_vol:.2}");
}
