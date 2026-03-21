//! STL file reader: binary and ASCII formats.
//!
//! Parses STL files into [`TriangleMesh`] for further processing.
//! Automatically detects binary vs ASCII format.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::tessellate::TriangleMesh;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// Read an STL file (binary or ASCII) from raw bytes.
///
/// Automatically detects the format. Returns a [`TriangleMesh`] with
/// vertex positions, normals, and triangle indices.
///
/// # Errors
///
/// Returns [`IoError::ParseError`](crate::IoError::ParseError) if the
/// data is malformed or truncated.
pub fn read_stl(data: &[u8]) -> Result<TriangleMesh, crate::IoError> {
    if is_ascii_stl(data) {
        read_ascii_stl(data)
    } else {
        read_binary_stl(data)
    }
}

/// Detect whether the given bytes represent an ASCII STL file.
///
/// ASCII STL files start with "solid" followed by a name or whitespace.
/// However, some binary files also start with "solid" in their header,
/// so we also check whether the claimed triangle count is consistent
/// with the file length.
fn is_ascii_stl(data: &[u8]) -> bool {
    // Must start with "solid" (case-insensitive for robustness).
    if data.len() < 84 {
        // Too short for binary (80-byte header + 4-byte count), try ASCII.
        return data.len() >= 5 && data[..5].eq_ignore_ascii_case(b"solid");
    }

    if !data[..5].eq_ignore_ascii_case(b"solid") {
        return false;
    }

    // Check if the binary interpretation would be valid.
    // If the triangle count implies a file size that matches, it's binary.
    let tri_count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]);
    let expected_binary_len = 84 + u64::from(tri_count) * 50;

    #[allow(clippy::cast_possible_truncation)]
    if expected_binary_len == data.len() as u64 {
        // Consistent with binary format — treat as binary.
        return false;
    }

    // Not consistent with binary — treat as ASCII.
    true
}

/// Read a binary STL file.
///
/// Format: 80-byte header, 4-byte LE triangle count, then 50 bytes per
/// triangle (normal + 3 vertices + 2-byte attribute).
fn read_binary_stl(data: &[u8]) -> Result<TriangleMesh, crate::IoError> {
    if data.len() < 84 {
        return Err(crate::IoError::ParseError {
            reason: format!(
                "binary STL too short: {} bytes (need at least 84)",
                data.len()
            ),
        });
    }

    let tri_count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;

    let expected_len = 84 + tri_count * 50;
    if data.len() < expected_len {
        return Err(crate::IoError::ParseError {
            reason: format!(
                "binary STL truncated: got {} bytes, expected {} for {tri_count} triangles",
                data.len(),
                expected_len,
            ),
        });
    }

    let mut mesh = TriangleMesh {
        positions: Vec::with_capacity(tri_count * 3),
        normals: Vec::with_capacity(tri_count * 3),
        indices: Vec::with_capacity(tri_count * 3),
    };

    for t in 0..tri_count {
        let base = 84 + t * 50;

        // Normal (3 × f32 LE).
        let nx = read_f32_le(data, base);
        let ny = read_f32_le(data, base + 4);
        let nz = read_f32_le(data, base + 8);
        let normal = Vec3::new(nx, ny, nz);

        // Three vertices (each 3 × f32 LE).
        for v in 0..3 {
            let vbase = base + 12 + v * 12;
            let x = read_f32_le(data, vbase);
            let y = read_f32_le(data, vbase + 4);
            let z = read_f32_le(data, vbase + 8);

            mesh.positions.push(Point3::new(x, y, z));
            mesh.normals.push(normal);

            #[allow(clippy::cast_possible_truncation)]
            mesh.indices.push((t * 3 + v) as u32);
        }
    }

    Ok(mesh)
}

/// Read an ASCII STL file.
fn read_ascii_stl(data: &[u8]) -> Result<TriangleMesh, crate::IoError> {
    let text = std::str::from_utf8(data).map_err(|e| crate::IoError::ParseError {
        reason: format!("ASCII STL contains invalid UTF-8: {e}"),
    })?;

    let mut mesh = TriangleMesh::default();
    let mut current_normal = Vec3::new(0.0, 0.0, 1.0);
    let mut vertex_count: u32 = 0;

    for line in text.lines() {
        let trimmed = line.trim();

        if let Some(rest) = trimmed.strip_prefix("facet normal") {
            current_normal = parse_vec3(rest.trim())?;
        } else if let Some(rest) = trimmed.strip_prefix("vertex") {
            let pos = parse_point3(rest.trim())?;
            mesh.positions.push(pos);
            mesh.normals.push(current_normal);
            mesh.indices.push(vertex_count);
            vertex_count += 1;
        }
        // Skip: solid, outer loop, endloop, endfacet, endsolid.
    }

    // Validate: vertex count must be a multiple of 3.
    if mesh.positions.len() % 3 != 0 {
        return Err(crate::IoError::ParseError {
            reason: format!(
                "ASCII STL has {} vertices (not a multiple of 3)",
                mesh.positions.len(),
            ),
        });
    }

    Ok(mesh)
}

/// Read a little-endian f32 from a byte slice at the given offset.
fn read_f32_le(data: &[u8], offset: usize) -> f64 {
    let bytes = [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ];
    f64::from(f32::from_le_bytes(bytes))
}

/// Parse three space-separated floats into a `Vec3`.
fn parse_vec3(s: &str) -> Result<Vec3, crate::IoError> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(crate::IoError::ParseError {
            reason: format!("expected 3 floats, got: '{s}'"),
        });
    }
    let x = parse_f64(parts[0])?;
    let y = parse_f64(parts[1])?;
    let z = parse_f64(parts[2])?;
    Ok(Vec3::new(x, y, z))
}

/// Parse three space-separated floats into a `Point3`.
fn parse_point3(s: &str) -> Result<Point3, crate::IoError> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(crate::IoError::ParseError {
            reason: format!("expected 3 floats, got: '{s}'"),
        });
    }
    let x = parse_f64(parts[0])?;
    let y = parse_f64(parts[1])?;
    let z = parse_f64(parts[2])?;
    Ok(Point3::new(x, y, z))
}

/// Parse a single float string.
fn parse_f64(s: &str) -> Result<f64, crate::IoError> {
    s.parse::<f64>().map_err(|e| crate::IoError::ParseError {
        reason: format!("invalid float '{s}': {e}"),
    })
}

/// Read an STL file and import it as a solid with one planar face per triangle.
///
/// This is a convenience wrapper that calls [`read_stl`] followed by
/// [`import_mesh`](crate::stl::import::import_mesh). Vertices within
/// `tolerance` of each other are merged.
///
/// # Errors
///
/// Returns [`IoError`](crate::IoError) if the file is malformed or the mesh
/// cannot be converted to a valid solid.
pub fn read_stl_solid(
    topo: &mut Topology,
    data: &[u8],
    tolerance: f64,
) -> Result<SolidId, crate::IoError> {
    let mesh = read_stl(data)?;
    crate::stl::import::import_mesh(topo, &mesh, tolerance)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube;

    use super::*;
    use crate::stl::writer::{self, StlFormat};

    // ── Round-trip tests ────────────────────────────────────────────

    #[test]
    fn roundtrip_binary_stl_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        // Write → read round-trip.
        let bytes = writer::write_stl(&topo, &[solid], 0.1, StlFormat::Binary).unwrap();
        let mesh = read_stl(&bytes).unwrap();

        // Unit cube: 6 faces × 2 triangles = 12 triangles = 36 vertices.
        assert_eq!(mesh.positions.len(), 36);
        assert_eq!(mesh.normals.len(), 36);
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn roundtrip_ascii_stl_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let bytes = writer::write_stl(&topo, &[solid], 0.1, StlFormat::Ascii).unwrap();
        let mesh = read_stl(&bytes).unwrap();

        assert_eq!(mesh.positions.len(), 36);
        assert_eq!(mesh.normals.len(), 36);
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn roundtrip_binary_stl_box_primitive() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let bytes = writer::write_stl(&topo, &[solid], 0.1, StlFormat::Binary).unwrap();
        let mesh = read_stl(&bytes).unwrap();

        // 12 triangles × 3 vertices = 36 vertices.
        assert_eq!(mesh.positions.len(), 36);
    }

    // ── Format detection tests ──────────────────────────────────────

    #[test]
    fn detect_ascii_format() {
        let data = b"solid test\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid test\n";
        assert!(is_ascii_stl(data));
    }

    #[test]
    fn detect_binary_format() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);
        let bytes = writer::write_stl(&topo, &[solid], 0.1, StlFormat::Binary).unwrap();
        assert!(!is_ascii_stl(&bytes));
    }

    // ── Error handling tests ────────────────────────────────────────

    #[test]
    fn binary_stl_too_short() {
        let data = [0u8; 10];
        let result = read_stl(&data);
        assert!(result.is_err());
    }

    #[test]
    fn binary_stl_truncated() {
        // Header says 1 triangle but file is too short.
        let mut data = [0u8; 84];
        data[80] = 1; // tri_count = 1
        let result = read_binary_stl(&data);
        assert!(result.is_err());
    }

    #[test]
    fn ascii_stl_bad_vertex() {
        let data = b"solid test\n  facet normal 0 0 1\n    outer loop\n      vertex abc def ghi\n    endloop\n  endfacet\nendsolid test\n";
        let result = read_stl(data);
        assert!(result.is_err());
    }

    // ── Minimal valid files ─────────────────────────────────────────

    #[test]
    fn read_minimal_ascii_single_triangle() {
        let data = b"solid minimal\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid minimal\n";
        let mesh = read_stl(data).unwrap();

        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.indices.len(), 3);

        // Check vertex positions.
        assert!((mesh.positions[0].x() - 0.0).abs() < 1e-6);
        assert!((mesh.positions[1].x() - 1.0).abs() < 1e-6);
        assert!((mesh.positions[2].y() - 1.0).abs() < 1e-6);

        // Check normal was propagated.
        assert!((mesh.normals[0].z() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn read_minimal_binary_single_triangle() {
        // Build a minimal binary STL with 1 triangle.
        let mut data = Vec::new();

        // 80-byte header.
        data.extend_from_slice(&[0u8; 80]);

        // Triangle count = 1.
        data.extend_from_slice(&1u32.to_le_bytes());

        // Normal: (0, 0, 1).
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());

        // Vertex 0: (0, 0, 0).
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());

        // Vertex 1: (1, 0, 0).
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());

        // Vertex 2: (0, 1, 0).
        data.extend_from_slice(&0.0f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());

        // Attribute byte count.
        data.extend_from_slice(&[0u8, 0u8]);

        let mesh = read_stl(&data).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.indices.len(), 3);
    }

    // ── Vertex data integrity ───────────────────────────────────────

    #[test]
    fn binary_roundtrip_preserves_vertex_positions() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let bytes = writer::write_stl(&topo, &[solid], 0.1, StlFormat::Binary).unwrap();
        let mesh = read_stl(&bytes).unwrap();

        // All vertex coordinates should be within [0, 1] for unit cube.
        for pos in &mesh.positions {
            assert!(pos.x() >= -0.01 && pos.x() <= 1.01);
            assert!(pos.y() >= -0.01 && pos.y() <= 1.01);
            assert!(pos.z() >= -0.01 && pos.z() <= 1.01);
        }
    }

    // ── read_stl_solid smoke test ───────────────────────────────────

    #[test]
    fn read_stl_solid_returns_solid_id() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let bytes = writer::write_stl(&topo, &[solid], 0.1, StlFormat::Binary).unwrap();

        let mut import_topo = Topology::new();
        let result = read_stl_solid(&mut import_topo, &bytes, 1e-6);
        assert!(
            result.is_ok(),
            "read_stl_solid should return Ok: {result:?}"
        );
    }
}
