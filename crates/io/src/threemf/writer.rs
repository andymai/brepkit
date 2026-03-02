//! 3MF file writer.
//!
//! Exports one or more [`Solid`](brepkit_topology::solid::Solid)s to the
//! [3D Manufacturing Format](https://3mf.io/specification/) (`.3mf`).
//!
//! A `.3mf` file is a ZIP archive containing:
//! - `[Content_Types].xml` — MIME declarations
//! - `_rels/.rels` — root relationships
//! - `3D/3dmodel.model` — mesh XML (vertices + triangles per object)

use std::io::{Cursor, Write as _};

use brepkit_operations::tessellate::{self, TriangleMesh};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;
use quick_xml::Writer;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, Event};
use zip::ZipWriter;
use zip::write::SimpleFileOptions;

use crate::IoError;

/// The 3MF model namespace.
const NS_3MF: &str = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02";

/// Write one or more solids to a 3MF byte buffer.
///
/// Each solid is tessellated (all face meshes merged) and written as a
/// separate `<object>` in the model XML. The `deflection` parameter
/// controls tessellation density — smaller values produce finer meshes.
///
/// # Errors
///
/// Returns an error if:
/// - `solids` is empty
/// - `deflection` is not positive and finite
/// - Tessellation of any face fails
/// - ZIP or XML writing fails
pub fn write_threemf(
    topo: &Topology,
    solids: &[SolidId],
    deflection: f64,
) -> Result<Vec<u8>, IoError> {
    if solids.is_empty() {
        return Err(IoError::InvalidTopology {
            reason: "no solids to export".to_string(),
        });
    }
    if !deflection.is_finite() || deflection <= 0.0 {
        return Err(IoError::InvalidTopology {
            reason: format!("deflection must be positive and finite, got {deflection}"),
        });
    }

    let meshes: Vec<TriangleMesh> = solids
        .iter()
        .map(|&solid_id| tessellate_solid(topo, solid_id, deflection))
        .collect::<Result<_, _>>()?;

    // Validate every mesh has at least one triangle (3MF requires geometry).
    for (i, mesh) in meshes.iter().enumerate() {
        if mesh.indices.len() < 3 {
            return Err(IoError::InvalidTopology {
                reason: format!("solid {i} tessellated to zero triangles"),
            });
        }
        if mesh.indices.len() % 3 != 0 {
            return Err(IoError::InvalidTopology {
                reason: format!(
                    "solid {i} has {} indices (not a multiple of 3)",
                    mesh.indices.len()
                ),
            });
        }
    }

    let model_xml = write_model_xml(&meshes)?;
    write_zip(&model_xml)
}

/// Tessellate all faces of a solid into a single merged [`TriangleMesh`].
fn tessellate_solid(
    topo: &Topology,
    solid_id: SolidId,
    deflection: f64,
) -> Result<TriangleMesh, IoError> {
    let solid = topo.solid(solid_id).map_err(topo_err)?;
    let mut merged = TriangleMesh::default();

    for shell_id in std::iter::once(solid.outer_shell()).chain(solid.inner_shells().iter().copied())
    {
        let shell = topo.shell(shell_id).map_err(topo_err)?;
        for &face_id in shell.faces() {
            let face_mesh = tessellate::tessellate(topo, face_id, deflection)?;

            #[allow(clippy::cast_possible_truncation)]
            let offset = merged.positions.len() as u32;

            merged.positions.extend_from_slice(&face_mesh.positions);
            merged.normals.extend_from_slice(&face_mesh.normals);
            merged
                .indices
                .extend(face_mesh.indices.iter().map(|i| i + offset));
        }
    }

    Ok(merged)
}

/// Build the `3dmodel.model` XML document.
fn write_model_xml(meshes: &[TriangleMesh]) -> Result<Vec<u8>, IoError> {
    let mut buf = Vec::new();
    let mut writer = Writer::new_with_indent(&mut buf, b' ', 1);

    writer.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;

    let mut model = BytesStart::new("model");
    model.push_attribute(("xmlns", NS_3MF));
    model.push_attribute(("unit", "millimeter"));
    writer.write_event(Event::Start(model))?;

    writer.write_event(Event::Start(BytesStart::new("resources")))?;
    for (i, mesh) in meshes.iter().enumerate() {
        write_object(&mut writer, i, mesh)?;
    }
    writer.write_event(Event::End(BytesEnd::new("resources")))?;

    writer.write_event(Event::Start(BytesStart::new("build")))?;
    for i in 0..meshes.len() {
        let mut item = BytesStart::new("item");
        let id_str = (i + 1).to_string();
        item.push_attribute(("objectid", id_str.as_str()));
        writer.write_event(Event::Empty(item))?;
    }
    writer.write_event(Event::End(BytesEnd::new("build")))?;

    writer.write_event(Event::End(BytesEnd::new("model")))?;

    Ok(buf)
}

/// Write a single `<object>` element containing `<mesh>` data.
fn write_object(
    writer: &mut Writer<&mut Vec<u8>>,
    index: usize,
    mesh: &TriangleMesh,
) -> Result<(), IoError> {
    let id_str = (index + 1).to_string();

    let mut object = BytesStart::new("object");
    object.push_attribute(("id", id_str.as_str()));
    object.push_attribute(("type", "model"));
    writer.write_event(Event::Start(object))?;

    writer.write_event(Event::Start(BytesStart::new("mesh")))?;

    writer.write_event(Event::Start(BytesStart::new("vertices")))?;
    for pos in &mesh.positions {
        let mut vertex = BytesStart::new("vertex");
        vertex.push_attribute(("x", format_f64(pos.x()).as_str()));
        vertex.push_attribute(("y", format_f64(pos.y()).as_str()));
        vertex.push_attribute(("z", format_f64(pos.z()).as_str()));
        writer.write_event(Event::Empty(vertex))?;
    }
    writer.write_event(Event::End(BytesEnd::new("vertices")))?;

    writer.write_event(Event::Start(BytesStart::new("triangles")))?;
    for tri in mesh.indices.chunks_exact(3) {
        let mut triangle = BytesStart::new("triangle");
        triangle.push_attribute(("v1", tri[0].to_string().as_str()));
        triangle.push_attribute(("v2", tri[1].to_string().as_str()));
        triangle.push_attribute(("v3", tri[2].to_string().as_str()));
        writer.write_event(Event::Empty(triangle))?;
    }
    writer.write_event(Event::End(BytesEnd::new("triangles")))?;

    writer.write_event(Event::End(BytesEnd::new("mesh")))?;

    writer.write_event(Event::End(BytesEnd::new("object")))?;

    Ok(())
}

/// Format a float for XML output (enough precision, no trailing noise).
fn format_f64(v: f64) -> String {
    // Use enough digits for sub-micron precision in millimeters.
    format!("{v:.6}")
}

/// Convert a [`TopologyError`](brepkit_topology::TopologyError) into an [`IoError`]
/// by routing through [`OperationsError`](brepkit_operations::OperationsError).
fn topo_err(e: brepkit_topology::TopologyError) -> IoError {
    IoError::Operations(brepkit_operations::OperationsError::from(e))
}

/// Static content for `[Content_Types].xml`.
const CONTENT_TYPES_XML: &[u8] = br#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"#;

/// Static content for `_rels/.rels`.
const RELS_XML: &[u8] = br#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"#;

/// Pack the model XML and required metadata into a ZIP archive.
fn write_zip(model_xml: &[u8]) -> Result<Vec<u8>, IoError> {
    let buf = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(buf);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("[Content_Types].xml", options)?;
    zip.write_all(CONTENT_TYPES_XML)?;

    zip.start_file("_rels/.rels", options)?;
    zip.write_all(RELS_XML)?;

    zip.start_file("3D/3dmodel.model", options)?;
    zip.write_all(model_xml)?;

    let cursor = zip.finish()?;
    Ok(cursor.into_inner())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::revolve::revolve;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::{make_unit_cube, make_unit_square_face};

    use super::*;

    /// Helper: extrude a unit square along +Z to create a box solid.
    fn make_extruded_box(topo: &mut Topology) -> SolidId {
        let face = make_unit_square_face(topo);
        extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap()
    }

    #[test]
    fn export_extruded_box() {
        let mut topo = Topology::new();
        let solid = make_extruded_box(&mut topo);

        let bytes = write_threemf(&topo, &[solid], 0.1).unwrap();

        // Verify it's a valid ZIP with the expected entries.
        let reader = zip::ZipArchive::new(Cursor::new(&bytes)).unwrap();
        assert_eq!(reader.len(), 3);
        assert!(reader.file_names().any(|n| n == "[Content_Types].xml"));
        assert!(reader.file_names().any(|n| n == "_rels/.rels"));
        assert!(reader.file_names().any(|n| n == "3D/3dmodel.model"));
    }

    #[test]
    fn export_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let bytes = write_threemf(&topo, &[solid], 0.1).unwrap();

        // Verify the model XML contains expected elements.
        let mut archive = zip::ZipArchive::new(Cursor::new(&bytes)).unwrap();
        let mut model_file = archive.by_name("3D/3dmodel.model").unwrap();
        let mut xml_str = String::new();
        std::io::Read::read_to_string(&mut model_file, &mut xml_str).unwrap();

        assert!(xml_str.contains("<vertex"));
        assert!(xml_str.contains("<triangle"));
        assert!(xml_str.contains("<object"));
        assert!(xml_str.contains("<item"));
    }

    #[test]
    fn export_revolved_solid() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        // Revolve 90 degrees around Y axis at x=2 (offset so profile doesn't
        // intersect the axis).
        let solid = revolve(
            &mut topo,
            face,
            Point3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            std::f64::consts::FRAC_PI_2,
        )
        .unwrap();

        let bytes = write_threemf(&topo, &[solid], 0.25).unwrap();

        let reader = zip::ZipArchive::new(Cursor::new(&bytes)).unwrap();
        assert_eq!(reader.len(), 3);
    }

    #[test]
    fn export_multiple_solids() {
        let mut topo = Topology::new();
        let s1 = make_extruded_box(&mut topo);
        let s2 = make_unit_cube(&mut topo);

        let bytes = write_threemf(&topo, &[s1, s2], 0.1).unwrap();

        let mut archive = zip::ZipArchive::new(Cursor::new(&bytes)).unwrap();
        let mut model_file = archive.by_name("3D/3dmodel.model").unwrap();
        let mut xml_str = String::new();
        std::io::Read::read_to_string(&mut model_file, &mut xml_str).unwrap();

        // Two objects and two build items.
        assert_eq!(xml_str.matches("<object").count(), 2);
        assert_eq!(xml_str.matches("<item").count(), 2);
    }

    #[test]
    fn export_empty_solids_error() {
        let topo = Topology::new();
        let result = write_threemf(&topo, &[], 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn export_invalid_deflection_error() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        assert!(write_threemf(&topo, &[solid], 0.0).is_err());
        assert!(write_threemf(&topo, &[solid], -1.0).is_err());
        assert!(write_threemf(&topo, &[solid], f64::NAN).is_err());
        assert!(write_threemf(&topo, &[solid], f64::INFINITY).is_err());
    }

    /// Parse the model XML and count vertices/triangles for verification.
    fn count_elements(xml: &str, tag: &str) -> usize {
        // Count self-closing tags like `<vertex .../>` and open tags like `<vertex ...>`.
        xml.matches(&format!("<{tag} ")).count()
    }

    /// Extract the model XML from a 3MF byte buffer.
    fn extract_model_xml(bytes: &[u8]) -> String {
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
        let mut model_file = archive.by_name("3D/3dmodel.model").unwrap();
        let mut xml_str = String::new();
        std::io::Read::read_to_string(&mut model_file, &mut xml_str).unwrap();
        xml_str
    }

    #[test]
    fn unit_cube_vertex_and_triangle_counts() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let bytes = write_threemf(&topo, &[solid], 0.1).unwrap();
        let xml = extract_model_xml(&bytes);

        // Unit cube: 6 faces × 4 vertices = 24 vertices (not shared across faces).
        assert_eq!(count_elements(&xml, "vertex"), 24);
        // Unit cube: 6 faces × 2 triangles = 12 triangles.
        assert_eq!(count_elements(&xml, "triangle"), 12);
    }

    #[test]
    fn xml_has_correct_namespace() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let bytes = write_threemf(&topo, &[solid], 0.5).unwrap();
        let xml = extract_model_xml(&bytes);

        assert!(xml.contains(NS_3MF));
        assert!(xml.contains("unit=\"millimeter\""));
    }

    #[test]
    fn vertex_attributes_are_finite() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let bytes = write_threemf(&topo, &[solid], 0.5).unwrap();
        let xml = extract_model_xml(&bytes);

        // No NaN or Infinity should appear in vertex coordinates.
        assert!(!xml.contains("NaN"));
        assert!(!xml.contains("Infinity"));
        assert!(!xml.contains("inf"));
    }

    #[test]
    fn roundtrip_zip_contains_expected_entries() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let bytes = write_threemf(&topo, &[solid], 0.5).unwrap();

        let reader = zip::ZipArchive::new(Cursor::new(&bytes)).unwrap();
        let names: Vec<&str> = reader.file_names().collect();

        assert!(names.contains(&"[Content_Types].xml"));
        assert!(names.contains(&"_rels/.rels"));
        assert!(names.contains(&"3D/3dmodel.model"));
    }
}
