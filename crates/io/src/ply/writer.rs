//! PLY file writer (ASCII and binary little-endian).

use std::fmt::Write as FmtWrite;
use std::io::Write;

use brepkit_operations::tessellate;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

/// PLY output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlyFormat {
    /// ASCII PLY (human-readable).
    Ascii,
    /// Binary little-endian PLY (compact).
    BinaryLittleEndian,
}

/// Write one or more solids to PLY format.
///
/// # Errors
///
/// Returns an error if tessellation fails.
pub fn write_ply(
    topo: &Topology,
    solids: &[SolidId],
    deflection: f64,
    format: PlyFormat,
) -> Result<Vec<u8>, crate::IoError> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut indices: Vec<[u32; 3]> = Vec::new();
    let mut vertex_offset: u32 = 0;

    for &solid_id in solids {
        // Walk outer + inner (cavity) shells so hollow solids'
        // cavity surfaces are present in the PLY output.
        let face_ids = solid_faces(topo, solid_id)?;

        for &face_id in &face_ids {
            let mesh = tessellate::tessellate(topo, face_id, deflection)?;

            #[allow(clippy::cast_possible_truncation)]
            for (pos, norm) in mesh.positions.iter().zip(mesh.normals.iter()) {
                positions.push([pos.x() as f32, pos.y() as f32, pos.z() as f32]);
                normals.push([norm.x() as f32, norm.y() as f32, norm.z() as f32]);
            }

            for tri in mesh.indices.chunks_exact(3) {
                indices.push([
                    tri[0] + vertex_offset,
                    tri[1] + vertex_offset,
                    tri[2] + vertex_offset,
                ]);
            }

            #[allow(clippy::cast_possible_truncation)]
            {
                vertex_offset += mesh.positions.len() as u32;
            }
        }
    }

    match format {
        PlyFormat::Ascii => Ok(write_ply_ascii(&positions, &normals, &indices)),
        PlyFormat::BinaryLittleEndian => Ok(write_ply_binary(&positions, &normals, &indices)),
    }
}

fn write_ply_ascii(positions: &[[f32; 3]], normals: &[[f32; 3]], indices: &[[u32; 3]]) -> Vec<u8> {
    let mut out = String::new();

    out.push_str("ply\n");
    out.push_str("format ascii 1.0\n");
    let _ = writeln!(out, "element vertex {}", positions.len());
    out.push_str("property float x\n");
    out.push_str("property float y\n");
    out.push_str("property float z\n");
    out.push_str("property float nx\n");
    out.push_str("property float ny\n");
    out.push_str("property float nz\n");
    let _ = writeln!(out, "element face {}", indices.len());
    out.push_str("property list uchar int vertex_indices\n");
    out.push_str("end_header\n");

    for (pos, norm) in positions.iter().zip(normals.iter()) {
        let _ = writeln!(
            out,
            "{} {} {} {} {} {}",
            pos[0], pos[1], pos[2], norm[0], norm[1], norm[2]
        );
    }

    for tri in indices {
        let _ = writeln!(out, "3 {} {} {}", tri[0], tri[1], tri[2]);
    }

    out.into_bytes()
}

fn write_ply_binary(positions: &[[f32; 3]], normals: &[[f32; 3]], indices: &[[u32; 3]]) -> Vec<u8> {
    let header = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n",
        positions.len(),
        indices.len()
    );

    let mut buf = Vec::with_capacity(header.len() + positions.len() * 24 + indices.len() * 13);
    buf.extend_from_slice(header.as_bytes());

    for (pos, norm) in positions.iter().zip(normals.iter()) {
        for &val in pos {
            buf.write_all(&val.to_le_bytes()).ok();
        }
        for &val in norm {
            buf.write_all(&val.to_le_bytes()).ok();
        }
    }

    for tri in indices {
        buf.push(3u8); // vertex count per face
        for &idx in tri {
            buf.write_all(&idx.to_le_bytes()).ok();
        }
    }

    buf
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use brepkit_topology::Topology;

    use super::*;

    #[test]
    fn write_ascii_ply() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let ply = write_ply(&topo, &[solid], 0.1, PlyFormat::Ascii).unwrap();
        let text = std::str::from_utf8(&ply).unwrap();

        assert!(text.starts_with("ply\n"));
        assert!(text.contains("format ascii 1.0"));
        assert!(text.contains("element vertex"));
        assert!(text.contains("element face"));
        assert!(text.contains("end_header"));
    }

    #[test]
    fn write_binary_ply() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let ply = write_ply(&topo, &[solid], 0.1, PlyFormat::BinaryLittleEndian).unwrap();

        // Should start with "ply\n"
        assert!(ply.starts_with(b"ply\n"));
        // Should contain binary header
        let header_end = ply.windows(11).position(|w| w == b"end_header\n").unwrap();
        assert!(header_end > 0);

        // Binary data follows header
        assert!(ply.len() > header_end + 11);
    }
}
