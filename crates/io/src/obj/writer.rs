//! OBJ file writer.

use std::fmt::Write;

use brepkit_operations::tessellate::{self, TriangleMesh};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// Write one or more solids to OBJ format as a UTF-8 string.
///
/// Tessellates each solid's faces and writes all triangles into a single
/// OBJ file. The `deflection` parameter controls tessellation quality.
///
/// # Errors
///
/// Returns an error if tessellation fails.
pub fn write_obj(
    topo: &Topology,
    solids: &[SolidId],
    deflection: f64,
) -> Result<String, crate::IoError> {
    let mut merged = TriangleMesh::default();

    for &solid_id in solids {
        let solid = topo.solid(solid_id)?;
        let shell = topo.shell(solid.outer_shell())?;

        for &face_id in shell.faces() {
            let mesh = tessellate::tessellate(topo, face_id, deflection)?;
            let offset = merged.positions.len();

            merged.positions.extend_from_slice(&mesh.positions);
            merged.normals.extend_from_slice(&mesh.normals);
            for &idx in &mesh.indices {
                #[allow(clippy::cast_possible_truncation)]
                merged.indices.push((idx as usize + offset) as u32);
            }
        }
    }

    let mut output = String::new();
    let _ = writeln!(output, "# brepkit OBJ export");
    let _ = writeln!(
        output,
        "# {} vertices, {} faces",
        merged.positions.len(),
        merged.indices.len() / 3,
    );

    // Vertices
    for pos in &merged.positions {
        let _ = writeln!(output, "v {:.6} {:.6} {:.6}", pos.x(), pos.y(), pos.z());
    }

    // Normals
    for normal in &merged.normals {
        let _ = writeln!(
            output,
            "vn {:.6} {:.6} {:.6}",
            normal.x(),
            normal.y(),
            normal.z()
        );
    }

    // Faces (OBJ is 1-indexed, format: f v1//n1 v2//n2 v3//n3)
    for tri in merged.indices.chunks_exact(3) {
        let i0 = tri[0] as usize + 1;
        let i1 = tri[1] as usize + 1;
        let i2 = tri[2] as usize + 1;
        let _ = writeln!(output, "f {i0}//{i0} {i1}//{i1} {i2}//{i2}");
    }

    Ok(output)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use brepkit_topology::Topology;

    use super::*;

    #[test]
    fn write_box_obj() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let obj = write_obj(&topo, &[solid], 0.1).unwrap();

        assert!(obj.starts_with("# brepkit OBJ export"));
        assert!(obj.contains("v "));
        assert!(obj.contains("vn "));
        assert!(obj.contains("f "));

        // Count vertex lines
        let v_count = obj.lines().filter(|l| l.starts_with("v ")).count();
        assert!(v_count > 0, "should have vertices");

        // Count face lines
        let f_count = obj.lines().filter(|l| l.starts_with("f ")).count();
        assert!(f_count > 0, "should have faces");
    }

    #[test]
    fn obj_is_valid_format() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let obj = write_obj(&topo, &[solid], 0.1).unwrap();

        // Verify all face indices are within range
        let v_count = obj.lines().filter(|l| l.starts_with("v ")).count();
        for line in obj.lines().filter(|l| l.starts_with("f ")) {
            for token in line.split_whitespace().skip(1) {
                let idx_str = token.split("//").next().unwrap();
                let idx: usize = idx_str.parse().unwrap();
                assert!(
                    idx >= 1 && idx <= v_count,
                    "face index {idx} out of range [1, {v_count}]"
                );
            }
        }
    }
}
