//! PLY file reader (ASCII and binary little-endian).

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::tessellate::TriangleMesh;

/// Read a PLY file from bytes and return a triangle mesh.
///
/// Supports ASCII and binary little-endian formats.
/// Reads vertex positions, optional normals, and triangle faces.
///
/// # Errors
///
/// Returns an error if the file is malformed.
pub fn read_ply(data: &[u8]) -> Result<TriangleMesh, crate::IoError> {
    // Parse header to determine format
    let header_end = find_header_end(data)?;
    let header_text =
        std::str::from_utf8(&data[..header_end]).map_err(|_| crate::IoError::ParseError {
            reason: "PLY header is not valid UTF-8".into(),
        })?;

    let header = parse_header(header_text)?;
    let body = &data[header_end..];

    match header.format {
        PlyFormat::Ascii => parse_ascii_body(&header, body),
        PlyFormat::BinaryLittleEndian => parse_binary_body(&header, body),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
}

#[derive(Debug)]
struct PlyHeader {
    format: PlyFormat,
    vertex_count: usize,
    face_count: usize,
    has_normals: bool,
}

fn find_header_end(data: &[u8]) -> Result<usize, crate::IoError> {
    let marker = b"end_header\n";
    data.windows(marker.len())
        .position(|w| w == marker)
        .map(|pos| pos + marker.len())
        .ok_or_else(|| crate::IoError::ParseError {
            reason: "PLY header missing end_header".into(),
        })
}

fn parse_header(text: &str) -> Result<PlyHeader, crate::IoError> {
    let mut format = None;
    let mut vertex_count = 0;
    let mut face_count = 0;
    let mut has_normals = false;
    let mut in_vertex_element = false;

    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("format ascii") {
            format = Some(PlyFormat::Ascii);
        } else if line.starts_with("format binary_little_endian") {
            format = Some(PlyFormat::BinaryLittleEndian);
        } else if line.starts_with("element vertex") {
            vertex_count = parse_count(line)?;
            in_vertex_element = true;
        } else if line.starts_with("element face") {
            face_count = parse_count(line)?;
            in_vertex_element = false;
        } else if in_vertex_element && line.starts_with("property") && line.contains("nx") {
            has_normals = true;
        }
    }

    let format = format.ok_or_else(|| crate::IoError::ParseError {
        reason: "PLY format not specified".into(),
    })?;

    Ok(PlyHeader {
        format,
        vertex_count,
        face_count,
        has_normals,
    })
}

fn parse_count(line: &str) -> Result<usize, crate::IoError> {
    line.split_whitespace()
        .nth(2)
        .ok_or_else(|| crate::IoError::ParseError {
            reason: format!("missing count in: {line}"),
        })?
        .parse()
        .map_err(|_| crate::IoError::ParseError {
            reason: format!("invalid count in: {line}"),
        })
}

fn parse_ascii_body(header: &PlyHeader, body: &[u8]) -> Result<TriangleMesh, crate::IoError> {
    let text = std::str::from_utf8(body).map_err(|_| crate::IoError::ParseError {
        reason: "PLY body is not valid UTF-8".into(),
    })?;

    let mut lines = text.lines();
    let mut positions = Vec::with_capacity(header.vertex_count);
    let mut normals = Vec::with_capacity(header.vertex_count);
    let mut indices = Vec::new();

    // Parse vertices
    for _ in 0..header.vertex_count {
        let line = lines.next().ok_or_else(|| crate::IoError::ParseError {
            reason: "unexpected end of PLY vertex data".into(),
        })?;
        let vals: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if vals.len() < 3 {
            return Err(crate::IoError::ParseError {
                reason: format!("PLY vertex needs at least 3 values: {line}"),
            });
        }

        positions.push(Point3::new(vals[0], vals[1], vals[2]));

        if header.has_normals && vals.len() >= 6 {
            normals.push(Vec3::new(vals[3], vals[4], vals[5]));
        }
    }

    // Parse faces
    for _ in 0..header.face_count {
        let line = lines.next().ok_or_else(|| crate::IoError::ParseError {
            reason: "unexpected end of PLY face data".into(),
        })?;
        let vals: Vec<u32> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        if vals.is_empty() {
            continue;
        }

        let n_verts = vals[0] as usize;
        if vals.len() < n_verts + 1 {
            return Err(crate::IoError::ParseError {
                reason: format!("PLY face has fewer indices than declared: {line}"),
            });
        }

        // Fan triangulation for polygons: v0-v1-v2, v0-v2-v3, ...
        let v0 = vals[1];
        for i in 2..n_verts {
            indices.push(v0);
            indices.push(vals[i]);
            indices.push(vals[i + 1]);
        }
    }

    // Generate normals if not in file
    if normals.is_empty() {
        normals = compute_normals(&positions, &indices);
    }
    normals.resize(positions.len(), Vec3::new(0.0, 0.0, 1.0));

    Ok(TriangleMesh {
        positions,
        normals,
        indices,
    })
}

fn parse_binary_body(header: &PlyHeader, body: &[u8]) -> Result<TriangleMesh, crate::IoError> {
    let floats_per_vertex = if header.has_normals { 6 } else { 3 };
    let vertex_bytes = header.vertex_count * floats_per_vertex * 4;

    if body.len() < vertex_bytes {
        return Err(crate::IoError::ParseError {
            reason: "PLY binary body too short for vertices".into(),
        });
    }

    let mut positions = Vec::with_capacity(header.vertex_count);
    let mut normals = Vec::with_capacity(header.vertex_count);
    let mut offset = 0;

    for _ in 0..header.vertex_count {
        let x = f32::from_le_bytes([
            body[offset],
            body[offset + 1],
            body[offset + 2],
            body[offset + 3],
        ]);
        let y = f32::from_le_bytes([
            body[offset + 4],
            body[offset + 5],
            body[offset + 6],
            body[offset + 7],
        ]);
        let z = f32::from_le_bytes([
            body[offset + 8],
            body[offset + 9],
            body[offset + 10],
            body[offset + 11],
        ]);
        positions.push(Point3::new(f64::from(x), f64::from(y), f64::from(z)));
        offset += 12;

        if header.has_normals {
            let nx = f32::from_le_bytes([
                body[offset],
                body[offset + 1],
                body[offset + 2],
                body[offset + 3],
            ]);
            let ny = f32::from_le_bytes([
                body[offset + 4],
                body[offset + 5],
                body[offset + 6],
                body[offset + 7],
            ]);
            let nz = f32::from_le_bytes([
                body[offset + 8],
                body[offset + 9],
                body[offset + 10],
                body[offset + 11],
            ]);
            normals.push(Vec3::new(f64::from(nx), f64::from(ny), f64::from(nz)));
            offset += 12;
        }
    }

    // Parse faces
    let mut indices = Vec::new();
    for _ in 0..header.face_count {
        if offset >= body.len() {
            break;
        }
        let n_verts = body[offset] as usize;
        offset += 1;

        if offset + n_verts * 4 > body.len() {
            break;
        }

        let mut face_indices = Vec::with_capacity(n_verts);
        for _ in 0..n_verts {
            let idx = u32::from_le_bytes([
                body[offset],
                body[offset + 1],
                body[offset + 2],
                body[offset + 3],
            ]);
            face_indices.push(idx);
            offset += 4;
        }

        // Fan triangulation
        if face_indices.len() >= 3 {
            let v0 = face_indices[0];
            for i in 1..face_indices.len() - 1 {
                indices.push(v0);
                indices.push(face_indices[i]);
                indices.push(face_indices[i + 1]);
            }
        }
    }

    if normals.is_empty() {
        normals = compute_normals(&positions, &indices);
    }
    normals.resize(positions.len(), Vec3::new(0.0, 0.0, 1.0));

    Ok(TriangleMesh {
        positions,
        normals,
        indices,
    })
}

fn compute_normals(positions: &[Point3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::new(0.0, 0.0, 0.0); positions.len()];

    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }
        let e1 = positions[i1] - positions[i0];
        let e2 = positions[i2] - positions[i0];
        let n = e1.cross(e2);
        normals[i0] += n;
        normals[i1] += n;
        normals[i2] += n;
    }

    for n in &mut normals {
        let len = n.length();
        if len > 1e-15 {
            *n = Vec3::new(n.x() / len, n.y() / len, n.z() / len);
        } else {
            *n = Vec3::new(0.0, 0.0, 1.0);
        }
    }

    normals
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn read_ascii_triangle() {
        let ply = b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n";

        let mesh = read_ply(ply).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.indices.len(), 3);
    }

    #[test]
    fn roundtrip_ascii() {
        let mut topo = brepkit_topology::Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let ply_data =
            crate::ply::write_ply(&topo, &[solid], 0.1, crate::ply::writer::PlyFormat::Ascii)
                .unwrap();
        let mesh = read_ply(&ply_data).unwrap();

        assert!(!mesh.positions.is_empty());
        assert!(!mesh.indices.is_empty());
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn roundtrip_binary() {
        let mut topo = brepkit_topology::Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let ply_data = crate::ply::write_ply(
            &topo,
            &[solid],
            0.1,
            crate::ply::writer::PlyFormat::BinaryLittleEndian,
        )
        .unwrap();
        let mesh = read_ply(&ply_data).unwrap();

        assert!(!mesh.positions.is_empty());
        assert!(!mesh.indices.is_empty());
    }

    #[test]
    fn missing_header_error() {
        let data = b"not a ply file";
        assert!(read_ply(data).is_err());
    }
}
