//! 3MF file reader.
//!
//! Parses [3D Manufacturing Format](https://3mf.io/specification/) (`.3mf`)
//! files into [`TriangleMesh`] objects. A `.3mf` file is a ZIP archive
//! containing an XML model description at `3D/3dmodel.model`.

use std::io::{Cursor, Read as _};

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::tessellate::TriangleMesh;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::IoError;

/// Read a 3MF file from raw bytes and return one [`TriangleMesh`] per object.
///
/// Each `<object>` in the model XML becomes a separate mesh.
///
/// # Errors
///
/// Returns [`IoError`] if:
/// - The data is not a valid ZIP archive
/// - The archive lacks a `3D/3dmodel.model` entry
/// - The XML is malformed or missing required elements
pub fn read_threemf(data: &[u8]) -> Result<Vec<TriangleMesh>, IoError> {
    let model_xml = extract_model_xml(data)?;
    parse_model_xml(&model_xml)
}

/// Extract the `3D/3dmodel.model` XML from the ZIP archive.
fn extract_model_xml(data: &[u8]) -> Result<String, IoError> {
    let cursor = Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor).map_err(IoError::Zip)?;

    let mut model_file = archive
        .by_name("3D/3dmodel.model")
        .map_err(|_| IoError::ParseError {
            reason: "3MF archive missing '3D/3dmodel.model' entry".to_string(),
        })?;

    let mut xml_str = String::new();
    model_file
        .read_to_string(&mut xml_str)
        .map_err(IoError::Io)?;

    Ok(xml_str)
}

/// Parse the model XML and extract meshes from each `<object>`.
fn parse_model_xml(xml: &str) -> Result<Vec<TriangleMesh>, IoError> {
    let mut reader = quick_xml::Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut meshes = Vec::new();
    let mut current_vertices: Vec<Point3> = Vec::new();
    let mut current_indices: Vec<u32> = Vec::new();
    let mut in_object = false;
    let mut in_vertices = false;
    let mut in_triangles = false;

    loop {
        match reader.read_event() {
            Ok(quick_xml::events::Event::Start(ref e)) => {
                let local = e.local_name();
                let name = std::str::from_utf8(local.as_ref()).unwrap_or("");

                match name {
                    "object" => {
                        in_object = true;
                        current_vertices.clear();
                        current_indices.clear();
                    }
                    "vertices" => in_vertices = true,
                    "triangles" => in_triangles = true,
                    _ => {}
                }
            }
            Ok(quick_xml::events::Event::Empty(ref e)) => {
                let local = e.local_name();
                let name = std::str::from_utf8(local.as_ref()).unwrap_or("");

                match name {
                    "vertex" if in_vertices => {
                        let pt = parse_vertex_attributes(e)?;
                        current_vertices.push(pt);
                    }
                    "triangle" if in_triangles => {
                        let (v1, v2, v3) = parse_triangle_attributes(e)?;
                        current_indices.push(v1);
                        current_indices.push(v2);
                        current_indices.push(v3);
                    }
                    _ => {}
                }
            }
            Ok(quick_xml::events::Event::End(ref e)) => {
                let local = e.local_name();
                let name = std::str::from_utf8(local.as_ref()).unwrap_or("");

                match name {
                    "object" if in_object => {
                        let mesh = build_mesh(&current_vertices, &current_indices)?;
                        meshes.push(mesh);
                        in_object = false;
                    }
                    "vertices" => in_vertices = false,
                    "triangles" => in_triangles = false,
                    _ => {}
                }
            }
            Ok(quick_xml::events::Event::Eof) => break,
            Err(e) => {
                return Err(IoError::ParseError {
                    reason: format!("XML parse error: {e}"),
                });
            }
            _ => {}
        }
    }

    if meshes.is_empty() {
        return Err(IoError::ParseError {
            reason: "no objects found in 3MF model".to_string(),
        });
    }

    Ok(meshes)
}

/// Parse `x`, `y`, `z` attributes from a `<vertex>` element.
fn parse_vertex_attributes(e: &quick_xml::events::BytesStart<'_>) -> Result<Point3, IoError> {
    let mut x = None;
    let mut y = None;
    let mut z = None;

    for attr in e.attributes().flatten() {
        let key = std::str::from_utf8(attr.key.as_ref()).unwrap_or("");
        let val = std::str::from_utf8(&attr.value).unwrap_or("");
        match key {
            "x" => x = Some(parse_f64_attr(val, "vertex x")?),
            "y" => y = Some(parse_f64_attr(val, "vertex y")?),
            "z" => z = Some(parse_f64_attr(val, "vertex z")?),
            _ => {}
        }
    }

    match (x, y, z) {
        (Some(x), Some(y), Some(z)) => Ok(Point3::new(x, y, z)),
        _ => Err(IoError::ParseError {
            reason: "vertex element missing x, y, or z attribute".to_string(),
        }),
    }
}

/// Parse `v1`, `v2`, `v3` attributes from a `<triangle>` element.
fn parse_triangle_attributes(
    e: &quick_xml::events::BytesStart<'_>,
) -> Result<(u32, u32, u32), IoError> {
    let mut v1 = None;
    let mut v2 = None;
    let mut v3 = None;

    for attr in e.attributes().flatten() {
        let key = std::str::from_utf8(attr.key.as_ref()).unwrap_or("");
        let val = std::str::from_utf8(&attr.value).unwrap_or("");
        match key {
            "v1" => v1 = Some(parse_u32_attr(val, "triangle v1")?),
            "v2" => v2 = Some(parse_u32_attr(val, "triangle v2")?),
            "v3" => v3 = Some(parse_u32_attr(val, "triangle v3")?),
            _ => {}
        }
    }

    match (v1, v2, v3) {
        (Some(a), Some(b), Some(c)) => Ok((a, b, c)),
        _ => Err(IoError::ParseError {
            reason: "triangle element missing v1, v2, or v3 attribute".to_string(),
        }),
    }
}

/// Build a [`TriangleMesh`] from indexed vertices, computing per-vertex normals.
fn build_mesh(vertices: &[Point3], indices: &[u32]) -> Result<TriangleMesh, IoError> {
    if indices.len() % 3 != 0 {
        return Err(IoError::ParseError {
            reason: format!(
                "triangle indices count {} is not a multiple of 3",
                indices.len()
            ),
        });
    }

    // Accumulate face normals into per-vertex normals.
    let mut normals = vec![Vec3::new(0.0, 0.0, 0.0); vertices.len()];

    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            return Err(IoError::ParseError {
                reason: format!(
                    "triangle index out of bounds: [{i0}, {i1}, {i2}] but only {} vertices",
                    vertices.len(),
                ),
            });
        }

        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let face_normal = edge1.cross(edge2);

        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }

    // Normalize accumulated normals.
    let up = Vec3::new(0.0, 0.0, 1.0);
    for n in &mut normals {
        *n = n.normalize().unwrap_or(up);
    }

    Ok(TriangleMesh {
        positions: vertices.to_vec(),
        normals,
        indices: indices.to_vec(),
    })
}

/// Parse a float attribute value.
fn parse_f64_attr(val: &str, context: &str) -> Result<f64, IoError> {
    val.parse::<f64>().map_err(|e| IoError::ParseError {
        reason: format!("invalid {context} value '{val}': {e}"),
    })
}

/// Parse a u32 attribute value.
fn parse_u32_attr(val: &str, context: &str) -> Result<u32, IoError> {
    val.parse::<u32>().map_err(|e| IoError::ParseError {
        reason: format!("invalid {context} value '{val}': {e}"),
    })
}

/// Read a 3MF file and import the first object as a solid.
///
/// This is a convenience wrapper that calls [`read_threemf`] followed by
/// [`import_mesh`](crate::stl::import::import_mesh) on the first mesh.
/// Vertices within `tolerance` of each other are merged.
///
/// **Note:** Only the first object is imported; additional objects in the
/// archive are silently ignored. Use [`read_threemf`] directly to access
/// all objects.
///
/// # Errors
///
/// Returns [`IoError`] if:
/// - The file is malformed
/// - The archive contains no objects
/// - The mesh cannot be converted to a valid solid
pub fn read_threemf_solid(
    topo: &mut Topology,
    data: &[u8],
    tolerance: f64,
) -> Result<SolidId, IoError> {
    let meshes = read_threemf(data)?;
    let mesh = meshes
        .into_iter()
        .next()
        .ok_or_else(|| IoError::ParseError {
            reason: "3MF file contains no objects".to_string(),
        })?;
    crate::stl::import::import_mesh(topo, &mesh, tolerance)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_non_manifold;

    use super::*;
    use crate::threemf::writer;

    // ── Round-trip tests ────────────────────────────────────────────

    #[test]
    fn roundtrip_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let bytes = writer::write_threemf(&topo, &[solid], 0.1).unwrap();
        let meshes = read_threemf(&bytes).unwrap();

        assert_eq!(meshes.len(), 1);
        let mesh = &meshes[0];

        // Unit cube: 8 corner vertices (shared across faces via watertight tessellation).
        assert_eq!(mesh.positions.len(), 8);
        // 12 triangles × 3 indices = 36.
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn roundtrip_box_primitive() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let bytes = writer::write_threemf(&topo, &[solid], 0.1).unwrap();
        let meshes = read_threemf(&bytes).unwrap();

        assert_eq!(meshes.len(), 1);
        assert_eq!(meshes[0].indices.len(), 36); // 12 triangles.
    }

    #[test]
    fn roundtrip_multiple_solids() {
        let mut topo = Topology::new();
        let s1 = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = make_unit_cube_non_manifold(&mut topo);

        let bytes = writer::write_threemf(&topo, &[s1, s2], 0.1).unwrap();
        let meshes = read_threemf(&bytes).unwrap();

        assert_eq!(meshes.len(), 2);
    }

    // ── Vertex data integrity ───────────────────────────────────────

    #[test]
    fn roundtrip_preserves_vertex_bounds() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let bytes = writer::write_threemf(&topo, &[solid], 0.1).unwrap();
        let meshes = read_threemf(&bytes).unwrap();

        for pos in &meshes[0].positions {
            assert!(pos.x() >= -0.01 && pos.x() <= 1.01);
            assert!(pos.y() >= -0.01 && pos.y() <= 1.01);
            assert!(pos.z() >= -0.01 && pos.z() <= 1.01);
        }
    }

    #[test]
    fn normals_are_unit_length() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let bytes = writer::write_threemf(&topo, &[solid], 0.1).unwrap();
        let meshes = read_threemf(&bytes).unwrap();

        for n in &meshes[0].normals {
            let len = n.length();
            assert!(
                (len - 1.0).abs() < 1e-6,
                "normal length should be ~1.0, got {len}"
            );
        }
    }

    // ── Error handling tests ────────────────────────────────────────

    #[test]
    fn invalid_zip_data() {
        let result = read_threemf(b"this is not a zip file");
        assert!(result.is_err());
    }

    #[test]
    fn missing_model_entry() {
        // Create a valid ZIP but without the required model file.
        let buf = std::io::Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(buf);
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("dummy.txt", options).unwrap();
        std::io::Write::write_all(&mut zip, b"hello").unwrap();
        let cursor = zip.finish().unwrap();

        let result = read_threemf(&cursor.into_inner());
        assert!(result.is_err());
    }

    // ── read_threemf_solid smoke test ───────────────────────────────

    #[test]
    fn read_threemf_solid_returns_solid_id() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let bytes = writer::write_threemf(&topo, &[solid], 0.1).unwrap();

        let mut import_topo = Topology::new();
        let result = read_threemf_solid(&mut import_topo, &bytes, 1e-6);
        assert!(
            result.is_ok(),
            "read_threemf_solid should return Ok: {result:?}"
        );
    }
}
