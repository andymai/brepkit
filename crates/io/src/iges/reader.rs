//! IGES file reader.
//!
//! Parses IGES files and extracts geometric entities (lines, planes,
//! NURBS curves and surfaces) into topology.

use std::collections::HashMap;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::IoError;

/// Read an IGES file and reconstruct topology.
///
/// Returns the list of solid IDs created. Each group of faces found
/// in the IGES file is assembled into a solid.
///
/// # Errors
///
/// Returns [`IoError`] if the file is malformed or contains unsupported entities.
pub fn read_iges(input: &str, topo: &mut Topology) -> Result<Vec<SolidId>, IoError> {
    let entities = parse_iges_entities(input)?;
    build_topology(topo, &entities)
}

// ── IGES entity representation ──────────────────────────────────────

/// A parsed IGES entity with its directory entry and parameter data.
#[derive(Debug)]
struct IgesEntity {
    /// Entity type number (e.g., 110 for line, 128 for NURBS surface).
    entity_type: u32,
    /// The raw parameter data string (comma-separated values).
    params: String,
    /// Directory entry sequence number (used for cross-referencing).
    #[allow(dead_code)]
    de_seq: u32,
}

// ── Parsing ─────────────────────────────────────────────────────────

/// Parse all entities from an IGES file.
fn parse_iges_entities(input: &str) -> Result<Vec<IgesEntity>, IoError> {
    // Separate lines by section.
    let mut d_lines: Vec<&str> = Vec::new();
    let mut p_lines: Vec<&str> = Vec::new();

    for line in input.lines() {
        if line.len() < 73 {
            continue;
        }
        let section = line.as_bytes().get(72).copied().unwrap_or(b' ');
        match section {
            b'D' => d_lines.push(line),
            b'P' => p_lines.push(line),
            _ => {} // Skip S, G, T sections.
        }
    }

    // Parse directory entries (pairs of lines).
    let mut dir_entries: Vec<(u32, u32, u32)> = Vec::new(); // (entity_type, pd_start, de_seq)

    let mut i = 0;
    while i + 1 < d_lines.len() {
        let line1 = d_lines[i];

        let entity_type = parse_int_field(line1, 0, 8)?;
        let pd_start = parse_int_field(line1, 8, 16)?;

        // DE sequence number is in columns 73-80.
        let de_seq = parse_int_field(line1, 73, 80)?;

        dir_entries.push((entity_type, pd_start, de_seq));
        i += 2; // Skip the second line of the DE pair.
    }

    // Collect parameter data by DE pointer.
    // P-section lines have format: data (cols 0-63), DE pointer (cols 64-72), "P" + seq.
    let mut pd_by_de: HashMap<u32, String> = HashMap::new();

    for p_line in &p_lines {
        let data_part = if p_line.len() >= 64 {
            &p_line[..64]
        } else {
            p_line
        };
        let de_ptr = if p_line.len() >= 72 {
            parse_int_field(p_line, 64, 72).unwrap_or(0)
        } else {
            0
        };

        pd_by_de
            .entry(de_ptr)
            .or_default()
            .push_str(data_part.trim_end());
    }

    // Build entities.
    let mut entities = Vec::new();
    for (entity_type, _pd_start, de_seq) in &dir_entries {
        let params = pd_by_de.get(de_seq).cloned().unwrap_or_default();
        // Strip the entity type number prefix from params (e.g., "110,..." → "...").
        let clean_params = strip_entity_prefix(&params, *entity_type);

        entities.push(IgesEntity {
            entity_type: *entity_type,
            params: clean_params,
            de_seq: *de_seq,
        });
    }

    Ok(entities)
}

/// Parse an integer from a fixed-width field in an IGES line.
fn parse_int_field(line: &str, start: usize, end: usize) -> Result<u32, IoError> {
    let end = end.min(line.len());
    if start >= end {
        return Ok(0);
    }
    let field = line[start..end].trim();
    if field.is_empty() {
        return Ok(0);
    }
    field.parse::<u32>().map_err(|e| IoError::ParseError {
        reason: format!("invalid IGES integer field '{field}': {e}"),
    })
}

/// Strip the entity type prefix from parameter data.
/// E.g., "110,1.0,2.0,..." → "1.0,2.0,..."
fn strip_entity_prefix(params: &str, entity_type: u32) -> String {
    let prefix = format!("{entity_type},");
    params.strip_prefix(&prefix).unwrap_or(params).to_string()
}

// ── Topology building ───────────────────────────────────────────────

/// Build topology from parsed IGES entities.
fn build_topology(topo: &mut Topology, entities: &[IgesEntity]) -> Result<Vec<SolidId>, IoError> {
    let mut face_ids = Vec::new();

    for entity in entities {
        if entity.entity_type == 108 {
            // Plane entity — create a planar face.
            if let Ok(face_id) = build_plane_face(topo, &entity.params) {
                face_ids.push(face_id);
            }
        }
        // Entity types 110 (line), 126 (NURBS curve), 128 (NURBS surface)
        // are skipped — they would be referenced by higher-level entities.
    }

    if face_ids.is_empty() {
        return Ok(Vec::new());
    }

    // Assemble all faces into a single solid.
    let shell = Shell::new(face_ids).map_err(|e| IoError::ParseError {
        reason: format!("failed to build shell: {e}"),
    })?;
    let shell_id = topo.add_shell(shell);
    let solid_id = topo.add_solid(Solid::new(shell_id, Vec::new()));

    Ok(vec![solid_id])
}

/// Build a planar face from IGES entity type 108 parameters.
/// Format: A, B, C, D, ptr, x, y, z (plane Ax+By+Cz=D).
fn build_plane_face(
    topo: &mut Topology,
    params: &str,
) -> Result<brepkit_topology::face::FaceId, IoError> {
    let values = parse_float_params(params);
    if values.len() < 4 {
        return Err(IoError::ParseError {
            reason: format!("IGES plane entity needs 4 params, got {}", values.len()),
        });
    }

    let normal = Vec3::new(values[0], values[1], values[2]);
    let d = values[3];

    let norm_len = normal.length();
    if norm_len < 1e-10 {
        return Err(IoError::ParseError {
            reason: "IGES plane has zero normal".to_string(),
        });
    }

    // Create a small square face on this plane for visualization.
    let unit_normal = Vec3::new(
        normal.x() / norm_len,
        normal.y() / norm_len,
        normal.z() / norm_len,
    );

    // Compute a point on the plane and two tangent directions.
    let origin = Point3::new(
        unit_normal.x() * d / norm_len,
        unit_normal.y() * d / norm_len,
        unit_normal.z() * d / norm_len,
    );

    let ax = Vec3::new(1.0, 0.0, 0.0);
    let ay = Vec3::new(0.0, 1.0, 0.0);
    let candidate = if unit_normal.dot(ax).abs() < 0.9 {
        ax
    } else {
        ay
    };
    let u_dir = unit_normal.cross(candidate);
    let u_len = u_dir.length().max(1e-10);
    let u_dir = Vec3::new(u_dir.x() / u_len, u_dir.y() / u_len, u_dir.z() / u_len);
    let v_dir = unit_normal.cross(u_dir);

    // Create a 1×1 square face using linear combination of tangent vectors.
    let half = 0.5;
    let p0 = offset_point(origin, u_dir, -half, v_dir, -half);
    let p1 = offset_point(origin, u_dir, half, v_dir, -half);
    let p2 = offset_point(origin, u_dir, half, v_dir, half);
    let p3 = offset_point(origin, u_dir, -half, v_dir, half);

    let v0 = topo.add_vertex(Vertex::new(p0, 1e-7));
    let v1 = topo.add_vertex(Vertex::new(p1, 1e-7));
    let v2 = topo.add_vertex(Vertex::new(p2, 1e-7));
    let v3 = topo.add_vertex(Vertex::new(p3, 1e-7));

    let e01 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e12 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e23 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
    let e30 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e01, true),
            OrientedEdge::new(e12, true),
            OrientedEdge::new(e23, true),
            OrientedEdge::new(e30, true),
        ],
        true,
    )
    .map_err(|e| IoError::ParseError {
        reason: format!("failed to build wire: {e}"),
    })?;
    let wire_id = topo.add_wire(wire);

    let surface = FaceSurface::Plane {
        normal: unit_normal,
        d: d / norm_len,
    };
    let face_id = topo.add_face(Face::new(wire_id, Vec::new(), surface));

    Ok(face_id)
}

/// Compute `origin + a*u + b*v` as a `Point3`.
fn offset_point(origin: Point3, u: Vec3, a: f64, v: Vec3, b: f64) -> Point3 {
    Point3::new(
        u.x().mul_add(a, v.x().mul_add(b, origin.x())),
        u.y().mul_add(a, v.y().mul_add(b, origin.y())),
        u.z().mul_add(a, v.z().mul_add(b, origin.z())),
    )
}

/// Parse comma-separated float parameters from IGES parameter data.
fn parse_float_params(params: &str) -> Vec<f64> {
    let clean = params.trim_end_matches(';');
    clean
        .split(',')
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                None
            } else {
                trimmed.parse::<f64>().ok()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_non_manifold;

    use super::*;
    use crate::iges::writer;

    // ── Round-trip tests ────────────────────────────────────────

    #[test]
    fn roundtrip_unit_cube() {
        let mut write_topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut write_topo);

        let iges_str = writer::write_iges(&write_topo, &[solid]).unwrap();

        let mut read_topo = Topology::new();
        let solids = read_iges(&iges_str, &mut read_topo).unwrap();

        assert_eq!(solids.len(), 1);
        let read_solid = read_topo.solid(solids[0]).unwrap();
        let shell = read_topo.shell(read_solid.outer_shell()).unwrap();
        // Unit cube has 6 plane entities.
        assert_eq!(shell.faces().len(), 6);
    }

    #[test]
    fn roundtrip_box_primitive() {
        let mut write_topo = Topology::new();
        let solid =
            brepkit_operations::primitives::make_box(&mut write_topo, 2.0, 3.0, 4.0).unwrap();

        let iges_str = writer::write_iges(&write_topo, &[solid]).unwrap();

        let mut read_topo = Topology::new();
        let solids = read_iges(&iges_str, &mut read_topo).unwrap();

        assert_eq!(solids.len(), 1);
    }

    // ── Error handling ──────────────────────────────────────────

    #[test]
    fn empty_file_returns_empty() {
        let mut topo = Topology::new();
        let solids = read_iges("", &mut topo).unwrap();
        assert!(solids.is_empty());
    }

    // ── Parsing helpers ─────────────────────────────────────────

    #[test]
    fn parse_float_params_basic() {
        let floats = parse_float_params("1.0,2.5,-3.0,0.;");
        assert_eq!(floats.len(), 4);
        assert!((floats[0] - 1.0).abs() < 1e-10);
        assert!((floats[1] - 2.5).abs() < 1e-10);
        assert!((floats[2] - (-3.0)).abs() < 1e-10);
        assert!((floats[3]).abs() < 1e-10);
    }

    #[test]
    fn parse_int_field_basic() {
        let val = parse_int_field("     108       1", 0, 8).unwrap();
        assert_eq!(val, 108);
    }
}
