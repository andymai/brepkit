//! OBJ file reader.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::tessellate::TriangleMesh;

/// Read an OBJ file from a string and return a triangle mesh.
///
/// Supports:
/// - Vertex positions (`v x y z`)
/// - Vertex normals (`vn x y z`)
/// - Triangle and polygon faces (`f v1 v2 v3 ...` or `f v1//n1 v2//n2 ...`)
///
/// Polygons with more than 3 vertices are triangulated using fan triangulation.
///
/// # Errors
///
/// Returns an error if the file is malformed.
pub fn read_obj(input: &str) -> Result<TriangleMesh, crate::IoError> {
    let mut positions: Vec<Point3> = Vec::new();
    let mut normals: Vec<Vec3> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("v") => {
                let coords = parse_3_floats(&mut parts, line)?;
                positions.push(Point3::new(coords[0], coords[1], coords[2]));
            }
            Some("vn") => {
                let coords = parse_3_floats(&mut parts, line)?;
                normals.push(Vec3::new(coords[0], coords[1], coords[2]));
            }
            Some("f") => {
                let face_indices = parse_face_indices(&mut parts, line)?;
                if face_indices.len() < 3 {
                    return Err(crate::IoError::ParseError {
                        reason: format!("face with fewer than 3 vertices: {line}"),
                    });
                }
                // Fan triangulation: v0-v1-v2, v0-v2-v3, v0-v3-v4, ...
                let v0 = face_indices[0];
                for i in 1..face_indices.len() - 1 {
                    indices.push(v0);
                    indices.push(face_indices[i]);
                    indices.push(face_indices[i + 1]);
                }
            }
            _ => {
                // Ignore unsupported lines (vt, g, mtllib, usemtl, s, etc.)
            }
        }
    }

    // If no normals were provided, generate per-face normals
    if normals.is_empty() {
        normals = compute_vertex_normals(&positions, &indices);
    }

    // Ensure normals vector matches positions length
    normals.resize(positions.len(), Vec3::new(0.0, 0.0, 1.0));

    Ok(TriangleMesh {
        positions,
        normals,
        indices,
    })
}

/// Parse a face index token like "1", "1/2", "1/2/3", or "1//3".
/// Returns the 0-based vertex index.
fn parse_face_index(token: &str, line: &str) -> Result<u32, crate::IoError> {
    let idx_str = token.split('/').next().unwrap_or(token);
    let idx: i64 = idx_str.parse().map_err(|_| crate::IoError::ParseError {
        reason: format!("invalid face index in: {line}"),
    })?;
    if idx <= 0 {
        return Err(crate::IoError::ParseError {
            reason: format!("negative or zero face index in: {line}"),
        });
    }
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    Ok((idx - 1) as u32) // OBJ is 1-indexed
}

fn parse_face_indices(
    parts: &mut std::str::SplitWhitespace<'_>,
    line: &str,
) -> Result<Vec<u32>, crate::IoError> {
    let mut result = Vec::new();
    for token in parts {
        result.push(parse_face_index(token, line)?);
    }
    Ok(result)
}

fn parse_3_floats(
    parts: &mut std::str::SplitWhitespace<'_>,
    line: &str,
) -> Result<[f64; 3], crate::IoError> {
    let mut coords = [0.0; 3];
    for coord in &mut coords {
        *coord = parts
            .next()
            .ok_or_else(|| crate::IoError::ParseError {
                reason: format!("expected 3 coordinates in: {line}"),
            })?
            .parse()
            .map_err(|_| crate::IoError::ParseError {
                reason: format!("invalid float in: {line}"),
            })?;
    }
    Ok(coords)
}

/// Compute per-vertex normals by averaging adjacent face normals.
fn compute_vertex_normals(positions: &[Point3], indices: &[u32]) -> Vec<Vec3> {
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
        let face_normal = e1.cross(e2);

        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }

    // Normalize
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
    fn read_simple_triangle() {
        let obj = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
";
        let mesh = read_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.indices.len(), 3);
        assert_eq!(mesh.indices, vec![0, 1, 2]);
    }

    #[test]
    fn read_quad_fan_triangulation() {
        let obj = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
";
        let mesh = read_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 4);
        // Quad should be split into 2 triangles
        assert_eq!(mesh.indices.len(), 6);
    }

    #[test]
    fn read_with_normals() {
        let obj = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vn 0.0 0.0 1.0
f 1//1 2//1 3//1
";
        let mesh = read_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.normals.len(), 3);
    }

    #[test]
    fn read_with_comments() {
        let obj = "\
# This is a comment
v 0.0 0.0 0.0
# Another comment
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
";
        let mesh = read_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 3);
    }

    #[test]
    fn roundtrip_write_read() {
        let mut topo = brepkit_topology::Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let obj_str = crate::obj::write_obj(&topo, &[solid], 0.1).unwrap();
        let mesh = read_obj(&obj_str).unwrap();

        assert!(!mesh.positions.is_empty(), "should have vertices");
        assert!(!mesh.indices.is_empty(), "should have faces");
        assert_eq!(mesh.indices.len() % 3, 0, "indices should be multiple of 3");
    }

    #[test]
    fn read_empty_error() {
        let obj = "";
        let mesh = read_obj(obj).unwrap();
        assert_eq!(mesh.positions.len(), 0);
        assert_eq!(mesh.indices.len(), 0);
    }

    #[test]
    fn computed_normals_are_unit() {
        let obj = "\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
";
        let mesh = read_obj(obj).unwrap();
        for n in &mesh.normals {
            let len = n.length();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "normal should be unit length, got {len}"
            );
        }
    }
}
