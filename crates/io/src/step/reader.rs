//! STEP AP203 file reader.
//!
//! Parses ISO 10303-21 (STEP Part 21) files and reconstructs B-Rep
//! topology. Supports the entity types produced by our STEP writer:
//! `MANIFOLD_SOLID_BREP`, `CLOSED_SHELL`, `ADVANCED_FACE`, `PLANE`,
//! `EDGE_CURVE`, `LINE`, `CARTESIAN_POINT`, `DIRECTION`, etc.

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

/// Read a STEP file and reconstruct topology.
///
/// Returns the list of solid IDs created in the topology.
///
/// # Errors
///
/// Returns [`IoError`] if:
/// - The file is not valid STEP Part 21
/// - Required entities are missing or malformed
/// - Entity references cannot be resolved
pub fn read_step(input: &str, topo: &mut Topology) -> Result<Vec<SolidId>, IoError> {
    let entities = parse_step_entities(input)?;
    let mut builder = StepBuilder::new(topo, &entities);
    builder.build_all_solids()
}

// ── Parsing ─────────────────────────────────────────────────────────

/// A parsed STEP entity: `#id = TYPE(attrs)`.
#[derive(Debug)]
struct StepEntity {
    entity_type: String,
    attrs: String,
}

/// Parse all entity instances from the DATA section.
fn parse_step_entities(input: &str) -> Result<HashMap<u64, StepEntity>, IoError> {
    let mut entities = HashMap::new();

    let data_start = input.find("DATA;").ok_or_else(|| IoError::ParseError {
        reason: "no DATA section found".to_string(),
    })?;
    let data_end = input[data_start..]
        .find("ENDSEC;")
        .ok_or_else(|| IoError::ParseError {
            reason: "no ENDSEC after DATA".to_string(),
        })?;

    let data_section = &input[data_start + 5..data_start + data_end];
    let joined = data_section.replace(['\n', '\r'], " ");

    for statement in joined.split(';') {
        let stmt = statement.trim();
        if stmt.is_empty() {
            continue;
        }

        if let Some(eq_pos) = stmt.find('=') {
            let id_part = stmt[..eq_pos].trim();
            let rest = stmt[eq_pos + 1..].trim();

            if let Some(id) = parse_entity_id(id_part) {
                if let Some(paren_pos) = rest.find('(') {
                    let entity_type = rest[..paren_pos].trim().to_uppercase();
                    // Attrs = everything after the entity opening paren.
                    // E.g., for `TYPE('', (1.0, 2.0))`, attrs = `'', (1.0, 2.0))`
                    let attrs = rest[paren_pos + 1..].trim();

                    entities.insert(
                        id,
                        StepEntity {
                            entity_type,
                            attrs: attrs.to_string(),
                        },
                    );
                }
            }
        }
    }

    Ok(entities)
}

/// Parse `#123` into `123`.
fn parse_entity_id(s: &str) -> Option<u64> {
    let trimmed = s.trim();
    trimmed.strip_prefix('#')?.parse().ok()
}

// ── Building ────────────────────────────────────────────────────────

/// Reconstructs topology from parsed STEP entities.
struct StepBuilder<'a> {
    topo: &'a mut Topology,
    entities: &'a HashMap<u64, StepEntity>,
    vertex_cache: HashMap<u64, brepkit_topology::vertex::VertexId>,
    edge_cache: HashMap<u64, brepkit_topology::edge::EdgeId>,
}

impl<'a> StepBuilder<'a> {
    fn new(topo: &'a mut Topology, entities: &'a HashMap<u64, StepEntity>) -> Self {
        Self {
            topo,
            entities,
            vertex_cache: HashMap::new(),
            edge_cache: HashMap::new(),
        }
    }

    fn build_all_solids(&mut self) -> Result<Vec<SolidId>, IoError> {
        let brep_ids: Vec<u64> = self
            .entities
            .iter()
            .filter(|(_, e)| e.entity_type == "MANIFOLD_SOLID_BREP")
            .map(|(&id, _)| id)
            .collect();

        let mut solid_ids = Vec::new();
        for brep_id in brep_ids {
            let solid_id = self.build_solid(brep_id)?;
            solid_ids.push(solid_id);
        }
        Ok(solid_ids)
    }

    fn build_solid(&mut self, brep_id: u64) -> Result<SolidId, IoError> {
        let attrs = self.get_entity(brep_id)?.attrs.clone();
        let refs = parse_refs(&attrs);
        // MANIFOLD_SOLID_BREP('name', #shell) — shell is the only #ref.
        let shell_ref = refs.first().copied().ok_or_else(|| IoError::ParseError {
            reason: format!("MANIFOLD_SOLID_BREP #{brep_id} missing shell reference"),
        })?;

        let shell_id = self.build_shell(shell_ref)?;
        let solid_id = self.topo.solids.alloc(Solid::new(shell_id, Vec::new()));
        Ok(solid_id)
    }

    fn build_shell(&mut self, shell_ref: u64) -> Result<brepkit_topology::shell::ShellId, IoError> {
        let attrs = self.get_entity(shell_ref)?.attrs.clone();
        let face_refs = parse_list_refs(&attrs);

        let mut face_ids = Vec::new();
        for face_ref in face_refs {
            let face_id = self.build_face(face_ref)?;
            face_ids.push(face_id);
        }

        let shell = Shell::new(face_ids).map_err(|e| IoError::ParseError {
            reason: format!("failed to build shell from STEP: {e}"),
        })?;
        let shell_id = self.topo.shells.alloc(shell);
        Ok(shell_id)
    }

    #[allow(clippy::too_many_lines)]
    fn build_face(&mut self, face_ref: u64) -> Result<brepkit_topology::face::FaceId, IoError> {
        let attrs = self.get_entity(face_ref)?.attrs.clone();
        let all_refs = parse_refs(&attrs);
        let list_refs = parse_list_refs(&attrs);

        // Surface ref is the last #ref that's not in the bounds list.
        let list_set: std::collections::HashSet<u64> = list_refs.iter().copied().collect();
        let surface_ref = all_refs
            .iter()
            .rev()
            .find(|r| !list_set.contains(r))
            .copied()
            .ok_or_else(|| IoError::ParseError {
                reason: format!("ADVANCED_FACE #{face_ref} missing surface reference"),
            })?;

        let surface = self.build_surface(surface_ref)?;

        let mut outer_wire = None;
        let mut inner_wires = Vec::new();

        for &bound_ref in &list_refs {
            let bound_entity = self.get_entity(bound_ref)?;
            let is_outer = bound_entity.entity_type == "FACE_OUTER_BOUND";
            let bound_attrs = bound_entity.attrs.clone();
            let bound_refs = parse_refs(&bound_attrs);

            if let Some(&loop_ref) = bound_refs.first() {
                let wire_id = self.build_edge_loop(loop_ref)?;
                if is_outer && outer_wire.is_none() {
                    outer_wire = Some(wire_id);
                } else {
                    inner_wires.push(wire_id);
                }
            }
        }

        // If no FACE_OUTER_BOUND, use the first bound as outer.
        let outer = outer_wire.or_else(|| {
            if inner_wires.is_empty() {
                None
            } else {
                Some(inner_wires.remove(0))
            }
        });

        let outer = outer.ok_or_else(|| IoError::ParseError {
            reason: format!("ADVANCED_FACE #{face_ref} has no bounds"),
        })?;

        let face_id = self
            .topo
            .faces
            .alloc(Face::new(outer, inner_wires, surface));
        Ok(face_id)
    }

    fn build_surface(&self, surface_ref: u64) -> Result<FaceSurface, IoError> {
        let entity = self.get_entity(surface_ref)?;
        let entity_type = entity.entity_type.clone();
        let attrs = entity.attrs.clone();

        match entity_type.as_str() {
            "PLANE" => {
                let refs = parse_refs(&attrs);
                let axis_ref = refs.first().copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("PLANE #{surface_ref} missing axis reference"),
                })?;
                let (origin, normal, _ref_dir) = self.build_axis2_placement(axis_ref)?;
                let d = normal.dot(Vec3::new(origin.x(), origin.y(), origin.z()));
                Ok(FaceSurface::Plane { normal, d })
            }
            "CYLINDRICAL_SURFACE" => {
                let refs = parse_refs(&attrs);
                let floats = parse_floats(&attrs);
                let axis_ref = refs.first().copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("CYLINDRICAL_SURFACE #{surface_ref} missing axis"),
                })?;
                let radius = floats.first().copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("CYLINDRICAL_SURFACE #{surface_ref} missing radius"),
                })?;
                let (origin, axis, _ref_dir) = self.build_axis2_placement(axis_ref)?;
                let cyl = brepkit_math::surfaces::CylindricalSurface::new(origin, axis, radius)
                    .map_err(|e| IoError::ParseError {
                        reason: format!("CYLINDRICAL_SURFACE #{surface_ref}: {e}"),
                    })?;
                Ok(FaceSurface::Cylinder(cyl))
            }
            "CONICAL_SURFACE" => {
                let refs = parse_refs(&attrs);
                let floats = parse_floats(&attrs);
                let axis_ref = refs.first().copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("CONICAL_SURFACE #{surface_ref} missing axis"),
                })?;
                // STEP: CONICAL_SURFACE('', #axis, base_radius, half_angle)
                // half_angle is in radians in STEP AP203.
                let half_angle = floats.last().copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("CONICAL_SURFACE #{surface_ref} missing half_angle"),
                })?;
                let (apex, axis, _ref_dir) = self.build_axis2_placement(axis_ref)?;
                let cone = brepkit_math::surfaces::ConicalSurface::new(apex, axis, half_angle)
                    .map_err(|e| IoError::ParseError {
                        reason: format!("CONICAL_SURFACE #{surface_ref}: {e}"),
                    })?;
                Ok(FaceSurface::Cone(cone))
            }
            "SPHERICAL_SURFACE" => {
                let refs = parse_refs(&attrs);
                let floats = parse_floats(&attrs);
                let axis_ref = refs.first().copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("SPHERICAL_SURFACE #{surface_ref} missing axis"),
                })?;
                let radius = floats.first().copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("SPHERICAL_SURFACE #{surface_ref} missing radius"),
                })?;
                let (center, _axis, _ref_dir) = self.build_axis2_placement(axis_ref)?;
                let sphere = brepkit_math::surfaces::SphericalSurface::new(center, radius)
                    .map_err(|e| IoError::ParseError {
                        reason: format!("SPHERICAL_SURFACE #{surface_ref}: {e}"),
                    })?;
                Ok(FaceSurface::Sphere(sphere))
            }
            "TOROIDAL_SURFACE" => {
                let refs = parse_refs(&attrs);
                let floats = parse_floats(&attrs);
                let axis_ref = refs.first().copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("TOROIDAL_SURFACE #{surface_ref} missing axis"),
                })?;
                let major_r = floats.first().copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("TOROIDAL_SURFACE #{surface_ref} missing major_radius"),
                })?;
                let minor_r = floats.get(1).copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("TOROIDAL_SURFACE #{surface_ref} missing minor_radius"),
                })?;
                let (center, _axis, _ref_dir) = self.build_axis2_placement(axis_ref)?;
                let torus = brepkit_math::surfaces::ToroidalSurface::new(center, major_r, minor_r)
                    .map_err(|e| IoError::ParseError {
                        reason: format!("TOROIDAL_SURFACE #{surface_ref}: {e}"),
                    })?;
                Ok(FaceSurface::Torus(torus))
            }
            _ => Err(IoError::UnsupportedEntity {
                entity: entity_type,
            }),
        }
    }

    fn build_edge_loop(
        &mut self,
        loop_ref: u64,
    ) -> Result<brepkit_topology::wire::WireId, IoError> {
        let attrs = self.get_entity(loop_ref)?.attrs.clone();
        let oe_refs = parse_list_refs(&attrs);

        let mut oriented_edges = Vec::new();
        for oe_ref in oe_refs {
            let oe = self.build_oriented_edge(oe_ref)?;
            oriented_edges.push(oe);
        }

        let wire = Wire::new(oriented_edges, true).map_err(|e| IoError::ParseError {
            reason: format!("failed to create wire from edge loop #{loop_ref}: {e}"),
        })?;
        let wire_id = self.topo.wires.alloc(wire);
        Ok(wire_id)
    }

    fn build_oriented_edge(&mut self, oe_ref: u64) -> Result<OrientedEdge, IoError> {
        let attrs = self.get_entity(oe_ref)?.attrs.clone();
        let refs = parse_refs(&attrs);
        let forward = attrs.contains(".T.");

        let edge_curve_ref = refs.last().copied().ok_or_else(|| IoError::ParseError {
            reason: format!("ORIENTED_EDGE #{oe_ref} missing edge curve reference"),
        })?;

        let edge_id = self.build_edge_curve(edge_curve_ref)?;
        Ok(OrientedEdge::new(edge_id, forward))
    }

    fn build_edge_curve(&mut self, ec_ref: u64) -> Result<brepkit_topology::edge::EdgeId, IoError> {
        if let Some(&cached) = self.edge_cache.get(&ec_ref) {
            return Ok(cached);
        }

        let attrs = self.get_entity(ec_ref)?.attrs.clone();
        let refs = parse_refs(&attrs);
        if refs.len() < 3 {
            return Err(IoError::ParseError {
                reason: format!("EDGE_CURVE #{ec_ref} needs at least 3 references"),
            });
        }

        let start_vp = self.build_vertex_point(refs[0])?;
        let end_vp = self.build_vertex_point(refs[1])?;
        let edge_id = self
            .topo
            .edges
            .alloc(Edge::new(start_vp, end_vp, EdgeCurve::Line));

        self.edge_cache.insert(ec_ref, edge_id);
        Ok(edge_id)
    }

    fn build_vertex_point(
        &mut self,
        vp_ref: u64,
    ) -> Result<brepkit_topology::vertex::VertexId, IoError> {
        if let Some(&cached) = self.vertex_cache.get(&vp_ref) {
            return Ok(cached);
        }

        let attrs = self.get_entity(vp_ref)?.attrs.clone();
        let refs = parse_refs(&attrs);
        let cp_ref = refs.first().copied().ok_or_else(|| IoError::ParseError {
            reason: format!("VERTEX_POINT #{vp_ref} missing point reference"),
        })?;

        let point = self.build_cartesian_point(cp_ref)?;
        let vid = self.topo.vertices.alloc(Vertex::new(point, 1e-7));

        self.vertex_cache.insert(vp_ref, vid);
        Ok(vid)
    }

    fn build_cartesian_point(&self, cp_ref: u64) -> Result<Point3, IoError> {
        let attrs = &self.get_entity(cp_ref)?.attrs;
        let coords = parse_floats(attrs);
        if coords.len() < 3 {
            return Err(IoError::ParseError {
                reason: format!(
                    "CARTESIAN_POINT #{cp_ref} needs 3 coordinates, got {}",
                    coords.len()
                ),
            });
        }
        Ok(Point3::new(coords[0], coords[1], coords[2]))
    }

    fn build_direction(&self, dir_ref: u64) -> Result<Vec3, IoError> {
        let attrs = &self.get_entity(dir_ref)?.attrs;
        let coords = parse_floats(attrs);
        if coords.len() < 3 {
            return Err(IoError::ParseError {
                reason: format!(
                    "DIRECTION #{dir_ref} needs 3 components, got {}",
                    coords.len()
                ),
            });
        }
        Ok(Vec3::new(coords[0], coords[1], coords[2]))
    }

    fn build_axis2_placement(&self, axis_ref: u64) -> Result<(Point3, Vec3, Vec3), IoError> {
        let attrs = self.get_entity(axis_ref)?.attrs.clone();
        let refs = parse_refs(&attrs);
        if refs.len() < 3 {
            return Err(IoError::ParseError {
                reason: format!("AXIS2_PLACEMENT_3D #{axis_ref} needs 3 sub-references"),
            });
        }
        let origin = self.build_cartesian_point(refs[0])?;
        let axis = self.build_direction(refs[1])?;
        let ref_dir = self.build_direction(refs[2])?;
        Ok((origin, axis, ref_dir))
    }

    fn get_entity(&self, id: u64) -> Result<&StepEntity, IoError> {
        self.entities.get(&id).ok_or_else(|| IoError::ParseError {
            reason: format!("entity #{id} not found"),
        })
    }
}

// ── Attribute parsing helpers ───────────────────────────────────────

/// Extract all `#NNN` references from an attribute string.
fn parse_refs(attrs: &str) -> Vec<u64> {
    let mut refs = Vec::new();
    let mut i = 0;
    let bytes = attrs.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'#' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i > start {
                if let Ok(num) = attrs[start..i].parse::<u64>() {
                    refs.push(num);
                }
            }
        } else {
            i += 1;
        }
    }
    refs
}

/// Extract `#NNN` references from the first parenthesized list in attrs.
fn parse_list_refs(attrs: &str) -> Vec<u64> {
    if let Some(start) = attrs.find('(') {
        if let Some(end) = attrs[start..].find(')') {
            let inner = &attrs[start + 1..start + end];
            return parse_refs(inner);
        }
    }
    Vec::new()
}

/// Extract floating-point numbers from an attribute string.
///
/// Handles both nested `(1.0, 2.0)` and flat `'', #ref, 1.5E+00` formats.
fn parse_floats(attrs: &str) -> Vec<f64> {
    let mut result = Vec::new();
    // Try nested parentheses first.
    if let Some(start) = attrs.find('(') {
        if let Some(end) = attrs[start..].find(')') {
            let inner = &attrs[start + 1..start + end];
            for part in inner.split(',') {
                let trimmed = part.trim();
                if let Ok(v) = trimmed.parse::<f64>() {
                    result.push(v);
                }
            }
        }
    }
    // If no nested parens found, parse top-level comma-separated tokens.
    if result.is_empty() {
        for part in attrs.split(',') {
            let trimmed = part.trim().trim_matches('\'').trim_end_matches(')');
            if trimmed.starts_with('#') || trimmed.starts_with('.') || trimmed.is_empty() {
                continue;
            }
            if let Ok(v) = trimmed.parse::<f64>() {
                result.push(v);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube;

    use super::*;
    use crate::step::writer;

    #[test]
    fn roundtrip_unit_cube() {
        let mut write_topo = Topology::new();
        let solid = make_unit_cube(&mut write_topo);

        let step_str = writer::write_step(&write_topo, &[solid]).unwrap();

        let mut read_topo = Topology::new();
        let solids = read_step(&step_str, &mut read_topo).unwrap();

        assert_eq!(solids.len(), 1);

        let read_solid = read_topo.solid(solids[0]).unwrap();
        let shell = read_topo.shell(read_solid.outer_shell()).unwrap();
        assert_eq!(shell.faces().len(), 6);
    }

    #[test]
    fn roundtrip_box_primitive() {
        let mut write_topo = Topology::new();
        let solid =
            brepkit_operations::primitives::make_box(&mut write_topo, 2.0, 3.0, 4.0).unwrap();

        let step_str = writer::write_step(&write_topo, &[solid]).unwrap();

        let mut read_topo = Topology::new();
        let solids = read_step(&step_str, &mut read_topo).unwrap();

        assert_eq!(solids.len(), 1);
        let read_solid = read_topo.solid(solids[0]).unwrap();
        let shell = read_topo.shell(read_solid.outer_shell()).unwrap();
        assert_eq!(shell.faces().len(), 6);
    }

    #[test]
    fn roundtrip_multiple_solids() {
        let mut write_topo = Topology::new();
        let s1 = brepkit_operations::primitives::make_box(&mut write_topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = make_unit_cube(&mut write_topo);

        let step_str = writer::write_step(&write_topo, &[s1, s2]).unwrap();

        let mut read_topo = Topology::new();
        let solids = read_step(&step_str, &mut read_topo).unwrap();

        assert_eq!(solids.len(), 2);
    }

    #[test]
    fn roundtrip_faces_have_wires() {
        let mut write_topo = Topology::new();
        let solid = make_unit_cube(&mut write_topo);

        let step_str = writer::write_step(&write_topo, &[solid]).unwrap();

        let mut read_topo = Topology::new();
        let solids = read_step(&step_str, &mut read_topo).unwrap();

        let read_solid = read_topo.solid(solids[0]).unwrap();
        let shell = read_topo.shell(read_solid.outer_shell()).unwrap();

        for &face_id in shell.faces() {
            let face = read_topo.face(face_id).unwrap();
            let wire = read_topo.wire(face.outer_wire()).unwrap();
            assert_eq!(wire.edges().len(), 4, "cube face should have 4 edges");
        }
    }

    #[test]
    fn roundtrip_faces_are_planar() {
        let mut write_topo = Topology::new();
        let solid = make_unit_cube(&mut write_topo);

        let step_str = writer::write_step(&write_topo, &[solid]).unwrap();

        let mut read_topo = Topology::new();
        let solids = read_step(&step_str, &mut read_topo).unwrap();

        let read_solid = read_topo.solid(solids[0]).unwrap();
        let shell = read_topo.shell(read_solid.outer_shell()).unwrap();

        for &face_id in shell.faces() {
            let face = read_topo.face(face_id).unwrap();
            assert!(matches!(face.surface(), FaceSurface::Plane { .. }));
        }
    }

    #[test]
    fn empty_input_error() {
        let mut topo = Topology::new();
        let result = read_step("", &mut topo);
        assert!(result.is_err());
    }

    #[test]
    fn no_data_section_error() {
        let mut topo = Topology::new();
        let result = read_step("ISO-10303-21;\nHEADER;\nENDSEC;\n", &mut topo);
        assert!(result.is_err());
    }

    #[test]
    fn parse_refs_basic() {
        let refs = parse_refs("'', #10, #20, #30");
        assert_eq!(refs, vec![10, 20, 30]);
    }

    #[test]
    fn parse_list_refs_basic() {
        let refs = parse_list_refs("'name', (#1, #2, #3), #4");
        assert_eq!(refs, vec![1, 2, 3]);
    }

    #[test]
    fn parse_floats_basic() {
        let floats = parse_floats("'', (1.5, -2.3, 0.)");
        assert_eq!(floats.len(), 3);
        assert!((floats[0] - 1.5).abs() < 1e-10);
        assert!((floats[1] - (-2.3)).abs() < 1e-10);
        assert!((floats[2]).abs() < 1e-10);
    }

    #[test]
    fn parse_floats_scientific() {
        let floats = parse_floats("'', (1.000000000000000E+00, -5.000000000000000E-01, 0.)");
        assert_eq!(floats.len(), 3);
        assert!((floats[0] - 1.0).abs() < 1e-10);
        assert!((floats[1] - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn roundtrip_cylinder_preserves_surface() {
        let mut write_topo = Topology::new();
        let solid =
            brepkit_operations::primitives::make_cylinder(&mut write_topo, 1.5, 3.0).unwrap();

        let step_str = writer::write_step(&write_topo, &[solid]).unwrap();

        // Verify STEP contains CYLINDRICAL_SURFACE.
        assert!(step_str.contains("CYLINDRICAL_SURFACE"));

        let mut read_topo = Topology::new();
        let solids = read_step(&step_str, &mut read_topo).unwrap();
        assert!(!solids.is_empty(), "should import at least one solid");

        // Verify the imported solid has a cylindrical face.
        let read_solid = read_topo.solid(solids[0]).unwrap();
        let shell = read_topo.shell(read_solid.outer_shell()).unwrap();

        let has_cylinder = shell.faces().iter().any(|&fid| {
            matches!(
                read_topo.face(fid).unwrap().surface(),
                FaceSurface::Cylinder(_)
            )
        });
        assert!(
            has_cylinder,
            "imported cylinder should have a cylindrical face"
        );
    }
}
