//! STL file writer: binary and ASCII formats.

use std::io::Write;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::tessellate::{self, TriangleMesh};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// STL output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StlFormat {
    /// Binary STL (compact, standard for 3D printing).
    Binary,
    /// ASCII STL (human-readable, larger files).
    Ascii,
}

/// Write one or more solids to STL format as bytes.
///
/// Tessellates each solid's faces and writes all triangles into a single
/// STL file. The `deflection` parameter controls tessellation quality.
///
/// # Errors
///
/// Returns an error if tessellation or serialization fails.
pub fn write_stl(
    topo: &Topology,
    solids: &[SolidId],
    deflection: f64,
    format: StlFormat,
) -> Result<Vec<u8>, crate::IoError> {
    // Tessellate all solids using shared-edge tessellation for watertight output,
    // then merge into a single mesh.
    let mut merged = TriangleMesh::default();

    for &solid_id in solids {
        let mesh = tessellate::tessellate_solid(topo, solid_id, deflection)?;

        #[allow(clippy::cast_possible_truncation)]
        let offset = merged.positions.len() as u32;

        merged.positions.extend_from_slice(&mesh.positions);
        merged.normals.extend_from_slice(&mesh.normals);
        merged
            .indices
            .extend(mesh.indices.iter().map(|i| i + offset));
    }

    match format {
        StlFormat::Binary => Ok(write_binary_stl(&merged)),
        StlFormat::Ascii => write_ascii_stl(&merged),
    }
}

/// Resolve the vertices and face normal for the `t`-th triangle in a mesh.
fn triangle_data(mesh: &TriangleMesh, t: usize) -> (Vec3, Point3, Point3, Point3) {
    let i0 = mesh.indices[t * 3] as usize;
    let i1 = mesh.indices[t * 3 + 1] as usize;
    let i2 = mesh.indices[t * 3 + 2] as usize;

    let v0 = mesh.positions[i0];
    let v1 = mesh.positions[i1];
    let v2 = mesh.positions[i2];

    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let normal = edge1
        .cross(edge2)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));

    (normal, v0, v1, v2)
}

/// Write a binary STL file.
///
/// Format:
/// - 80-byte header
/// - 4-byte little-endian triangle count
/// - Per triangle (50 bytes):
///   - 12 bytes: normal (3 × f32)
///   - 36 bytes: 3 vertices (3 × 3 × f32)
///   - 2 bytes: attribute byte count (0)
fn write_binary_stl(mesh: &TriangleMesh) -> Vec<u8> {
    let tri_count = mesh.indices.len() / 3;

    // 80-byte header + 4-byte count + 50 bytes per triangle.
    let mut buf = Vec::with_capacity(84 + tri_count * 50);

    // Header (80 bytes).
    let header = b"brepkit STL export";
    buf.extend_from_slice(header);
    buf.extend_from_slice(&[0u8; 80 - 18]); // Pad to 80 bytes.

    // Triangle count (little-endian u32).
    #[allow(clippy::cast_possible_truncation)]
    buf.extend_from_slice(&(tri_count as u32).to_le_bytes());

    for t in 0..tri_count {
        let (normal, v0, v1, v2) = triangle_data(mesh, t);

        write_f32_le(&mut buf, normal.x());
        write_f32_le(&mut buf, normal.y());
        write_f32_le(&mut buf, normal.z());

        for v in [v0, v1, v2] {
            write_f32_le(&mut buf, v.x());
            write_f32_le(&mut buf, v.y());
            write_f32_le(&mut buf, v.z());
        }

        // Attribute byte count (always 0).
        buf.extend_from_slice(&[0u8, 0u8]);
    }

    buf
}

/// Write an ASCII STL file.
fn write_ascii_stl(mesh: &TriangleMesh) -> Result<Vec<u8>, crate::IoError> {
    let tri_count = mesh.indices.len() / 3;
    let mut buf = Vec::new();

    writeln!(buf, "solid brepkit").map_err(crate::IoError::Io)?;

    for t in 0..tri_count {
        let (normal, v0, v1, v2) = triangle_data(mesh, t);

        writeln!(
            buf,
            "  facet normal {} {} {}",
            normal.x(),
            normal.y(),
            normal.z()
        )
        .map_err(crate::IoError::Io)?;
        writeln!(buf, "    outer loop").map_err(crate::IoError::Io)?;
        write_ascii_vertex(&mut buf, v0)?;
        write_ascii_vertex(&mut buf, v1)?;
        write_ascii_vertex(&mut buf, v2)?;
        writeln!(buf, "    endloop").map_err(crate::IoError::Io)?;
        writeln!(buf, "  endfacet").map_err(crate::IoError::Io)?;
    }

    writeln!(buf, "endsolid brepkit").map_err(crate::IoError::Io)?;

    Ok(buf)
}

/// Write a vertex line in ASCII STL format.
fn write_ascii_vertex(buf: &mut Vec<u8>, p: Point3) -> Result<(), crate::IoError> {
    writeln!(buf, "      vertex {} {} {}", p.x(), p.y(), p.z()).map_err(crate::IoError::Io)
}

/// Write an f64 as f32 little-endian bytes.
#[allow(clippy::cast_possible_truncation)]
fn write_f32_le(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&(v as f32).to_le_bytes());
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_non_manifold;

    use super::*;

    #[test]
    fn write_binary_stl_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let bytes = write_stl(&topo, &[solid], 0.1, StlFormat::Binary).unwrap();

        // Header: 80 bytes + count: 4 bytes = 84 bytes header.
        assert!(bytes.len() >= 84);

        // Parse triangle count from the binary.
        let tri_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;

        // Unit cube with 6 faces × 2 triangles each = 12 triangles.
        assert_eq!(tri_count, 12, "expected 12 triangles for unit cube");

        // Total size: 84 + 12 × 50 = 684 bytes.
        assert_eq!(bytes.len(), 84 + tri_count * 50);
    }

    #[test]
    fn write_ascii_stl_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let bytes = write_stl(&topo, &[solid], 0.1, StlFormat::Ascii).unwrap();
        let text = String::from_utf8(bytes).unwrap();

        assert!(text.starts_with("solid brepkit"));
        assert!(text.contains("facet normal"));
        assert!(text.contains("vertex"));
        assert!(text.trim().ends_with("endsolid brepkit"));

        // Count facets.
        let facet_count = text.matches("facet normal").count();
        assert_eq!(facet_count, 12, "expected 12 facets for unit cube");
    }

    #[test]
    fn write_stl_box_primitive() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let bytes = write_stl(&topo, &[solid], 0.1, StlFormat::Binary).unwrap();
        let tri_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;

        assert_eq!(tri_count, 12, "box should have 12 triangles");
    }

    #[test]
    fn write_stl_multiple_solids() {
        let mut topo = Topology::new();
        let s1 = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let bytes = write_stl(&topo, &[s1, s2], 0.1, StlFormat::Binary).unwrap();
        let tri_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;

        assert_eq!(tri_count, 24, "two boxes should have 24 triangles");
    }

    #[test]
    fn write_stl_watertight_box() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        // Tessellate the same way write_stl does internally (via tessellate_solid).
        let mesh = brepkit_operations::tessellate::tessellate_solid(&topo, solid, 0.1).unwrap();

        // A watertight mesh has 0 boundary edges: every half-edge (a,b) has a twin (b,a).
        let boundary = brepkit_operations::tessellate::boundary_edge_count(&mesh);
        assert_eq!(
            boundary, 0,
            "STL mesh should have 0 boundary edges (watertight)"
        );
    }

    #[test]
    fn write_stl_shared_vertices_box() {
        // With tessellate_solid, a box should have exactly 8 unique vertices
        // (one per corner), not 24 (4 per face × 6 faces) as with per-face tessellation.
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mesh = brepkit_operations::tessellate::tessellate_solid(&topo, solid, 0.1).unwrap();
        assert_eq!(
            mesh.positions.len(),
            8,
            "box should share vertices at corners"
        );
    }
}
