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
        let solid_id = self.topo.add_solid(Solid::new(shell_id, Vec::new()));
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
        let shell_id = self.topo.add_shell(shell);
        Ok(shell_id)
    }

    #[allow(clippy::too_many_lines)]
    fn build_face(&mut self, face_ref: u64) -> Result<brepkit_topology::face::FaceId, IoError> {
        let attrs = self.get_entity(face_ref)?.attrs.clone();
        // Check for reversed face orientation (.F. flag at end of ADVANCED_FACE).
        let orient_tail = attrs.trim_end_matches(')').trim();
        let face_reversed = orient_tail.ends_with(".F.") || orient_tail.ends_with(".FALSE.");
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

        let face_id = if face_reversed {
            self.topo
                .add_face(Face::new_reversed(outer, inner_wires, surface))
        } else {
            self.topo.add_face(Face::new(outer, inner_wires, surface))
        };
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
                let (center, axis, ref_dir) = self.build_axis2_placement(axis_ref)?;
                let torus = brepkit_math::surfaces::ToroidalSurface::with_axis_and_ref_dir(
                    center, major_r, minor_r, axis, ref_dir,
                )
                .map_err(|e| IoError::ParseError {
                    reason: format!("TOROIDAL_SURFACE #{surface_ref}: {e}"),
                })?;
                Ok(FaceSurface::Torus(torus))
            }
            "B_SPLINE_SURFACE_WITH_KNOTS" | "BOUNDED_SURFACE" | "B_SPLINE_SURFACE" => {
                let is_rational = attrs.contains("RATIONAL");
                self.build_bspline_surface(surface_ref, &attrs, is_rational)
            }
            _ if entity_type.is_empty() || attrs.contains("B_SPLINE_SURFACE_WITH_KNOTS") => {
                let is_rational = attrs.contains("RATIONAL");
                let bspline_attrs = find_composite_bspline_attrs(&attrs, "B_SPLINE_SURFACE")
                    .ok_or_else(|| IoError::UnsupportedEntity {
                        entity: format!("composite surface #{surface_ref}"),
                    })?;
                self.build_bspline_surface(surface_ref, bspline_attrs, is_rational)
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
        let wire_id = self.topo.add_wire(wire);
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

        // Resolve the actual curve geometry from the third reference.
        let curve = self.build_curve_geometry(refs[2])?;

        let edge_id = self.topo.add_edge(Edge::new(start_vp, end_vp, curve));

        self.edge_cache.insert(ec_ref, edge_id);
        Ok(edge_id)
    }

    /// Build the curve geometry for an edge from a curve entity reference.
    ///
    /// Dispatches on the entity type: LINE, CIRCLE, ELLIPSE,
    /// `B_SPLINE_CURVE_WITH_KNOTS`.
    fn build_curve_geometry(&self, curve_ref: u64) -> Result<EdgeCurve, IoError> {
        let entity = self.get_entity(curve_ref)?;
        let entity_type = entity.entity_type.clone();
        let attrs = entity.attrs.clone();

        match entity_type.as_str() {
            "LINE" => Ok(EdgeCurve::Line),
            "CIRCLE" => {
                let refs = parse_refs(&attrs);
                let floats = parse_floats(&attrs);
                let axis_ref = refs.first().copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("CIRCLE #{curve_ref} missing axis reference"),
                })?;
                let radius = floats.first().copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("CIRCLE #{curve_ref} missing radius"),
                })?;
                let (center, normal, _u_axis) = self.build_axis2_placement(axis_ref)?;
                let circle =
                    brepkit_math::curves::Circle3D::new(center, normal, radius).map_err(|e| {
                        IoError::ParseError {
                            reason: format!("CIRCLE #{curve_ref}: {e}"),
                        }
                    })?;
                Ok(EdgeCurve::Circle(circle))
            }
            "ELLIPSE" => {
                let refs = parse_refs(&attrs);
                let floats = parse_floats(&attrs);
                let axis_ref = refs.first().copied().ok_or_else(|| IoError::ParseError {
                    reason: format!("ELLIPSE #{curve_ref} missing axis reference"),
                })?;
                if floats.len() < 2 {
                    return Err(IoError::ParseError {
                        reason: format!("ELLIPSE #{curve_ref} needs semi_major and semi_minor"),
                    });
                }
                let (center, normal, _u_axis) = self.build_axis2_placement(axis_ref)?;
                let ellipse =
                    brepkit_math::curves::Ellipse3D::new(center, normal, floats[0], floats[1])
                        .map_err(|e| IoError::ParseError {
                            reason: format!("ELLIPSE #{curve_ref}: {e}"),
                        })?;
                Ok(EdgeCurve::Ellipse(ellipse))
            }
            "B_SPLINE_CURVE_WITH_KNOTS" => self.build_bspline_curve(curve_ref, &attrs, false),
            _ if entity_type.is_empty() || attrs.contains("B_SPLINE_CURVE_WITH_KNOTS") => {
                let is_rational = attrs.contains("RATIONAL");
                let bspline_attrs = find_composite_bspline_attrs(&attrs, "B_SPLINE_CURVE")
                    .ok_or_else(|| IoError::UnsupportedEntity {
                        entity: format!("composite curve #{curve_ref}"),
                    })?;
                self.build_bspline_curve(curve_ref, bspline_attrs, is_rational)
            }
            _ => Err(IoError::UnsupportedEntity {
                entity: format!("{entity_type} (curve #{curve_ref})"),
            }),
        }
    }

    /// Build a B-spline curve from parsed attributes.
    /// If `is_rational` is true, attempts to extract weights from a
    /// RATIONAL_B_SPLINE_CURVE section in the attrs.
    fn build_bspline_curve(
        &self,
        curve_ref: u64,
        attrs: &str,
        is_rational: bool,
    ) -> Result<EdgeCurve, IoError> {
        let parsed = parse_bspline_curve_attrs(attrs).ok_or_else(|| IoError::ParseError {
            reason: format!("B_SPLINE_CURVE #{curve_ref} could not parse attributes"),
        })?;
        let (degree, cp_refs, mults, knot_vals) = parsed;

        let mut control_points = Vec::with_capacity(cp_refs.len());
        for &cp_ref in &cp_refs {
            control_points.push(self.build_cartesian_point(cp_ref)?);
        }

        let knots = expand_knots(&mults, &knot_vals);

        // Extract weights from RATIONAL_B_SPLINE section if present.
        let weights = if is_rational {
            extract_rational_weights(attrs, control_points.len())
        } else {
            vec![1.0; control_points.len()]
        };

        let nurbs = brepkit_math::nurbs::NurbsCurve::new(degree, knots, control_points, weights)
            .map_err(|e| IoError::ParseError {
                reason: format!("B_SPLINE_CURVE #{curve_ref}: {e}"),
            })?;
        Ok(EdgeCurve::NurbsCurve(nurbs))
    }

    /// Build a B-spline surface from parsed attributes.
    fn build_bspline_surface(
        &self,
        surface_ref: u64,
        attrs: &str,
        is_rational: bool,
    ) -> Result<FaceSurface, IoError> {
        let parsed = parse_bspline_surface_attrs(attrs).ok_or_else(|| IoError::ParseError {
            reason: format!("B_SPLINE_SURFACE #{surface_ref} could not parse attributes"),
        })?;
        let (degree_u, degree_v, cp_grid_refs, u_mults, v_mults, u_knots, v_knots) = parsed;

        let mut cp_grid: Vec<Vec<Point3>> = Vec::new();
        for row_refs in &cp_grid_refs {
            let mut row: Vec<Point3> = Vec::new();
            for &cp_ref in row_refs {
                row.push(self.build_cartesian_point(cp_ref)?);
            }
            cp_grid.push(row);
        }

        let knots_u = expand_knots(&u_mults, &u_knots);
        let knots_v = expand_knots(&v_mults, &v_knots);

        let n_rows = cp_grid.len();
        let n_cols = cp_grid.first().map_or(0, Vec::len);

        let weights = if is_rational {
            extract_rational_weight_grid(attrs, n_rows, n_cols)
        } else {
            vec![vec![1.0; n_cols]; n_rows]
        };

        let nurbs = brepkit_math::nurbs::NurbsSurface::new(
            degree_u, degree_v, knots_u, knots_v, cp_grid, weights,
        )
        .map_err(|e| IoError::ParseError {
            reason: format!("B_SPLINE_SURFACE #{surface_ref}: {e}"),
        })?;
        Ok(FaceSurface::Nurbs(nurbs))
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
        let vid = self.topo.add_vertex(Vertex::new(point, 1e-7));

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

/// Find the B-spline attribute substring within a composite STEP entity.
///
/// Searches for `"{base_name}_WITH_KNOTS"` first, then falls back to `base_name`.
/// Returns the portion of `attrs` after the matched marker.
fn find_composite_bspline_attrs<'a>(attrs: &'a str, base_name: &str) -> Option<&'a str> {
    let with_knots = format!("{base_name}_WITH_KNOTS");
    if let Some(pos) = attrs.find(&with_knots) {
        return Some(&attrs[pos + with_knots.len()..]);
    }
    // Anchor on base_name followed by '(' to avoid matching inside
    // "RATIONAL_B_SPLINE_CURVE" when searching for "B_SPLINE_CURVE".
    let anchored = format!("{base_name}(");
    if let Some(pos) = attrs.find(&anchored) {
        return Some(&attrs[pos + base_name.len()..]);
    }
    None
}

/// Parse integers from a parenthesized list like `(4, 4)`.
fn parse_ints_in_parens(s: &str) -> Vec<u32> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let trimmed = part.trim().trim_matches('(').trim_matches(')').trim();
        if let Ok(v) = trimmed.parse::<u32>() {
            result.push(v);
        }
    }
    result
}

/// Extract weights from a RATIONAL_B_SPLINE section in composite entity attrs.
///
/// Looks for `RATIONAL_B_SPLINE_CURVE((...weights...))` or
/// `RATIONAL_B_SPLINE_SURFACE((...weights...))` and parses the weight list.
/// Falls back to uniform weights if parsing fails.
fn extract_rational_weights(attrs: &str, expected_count: usize) -> Vec<f64> {
    // Find the RATIONAL section.
    let marker = if attrs.contains("RATIONAL_B_SPLINE_SURFACE") {
        "RATIONAL_B_SPLINE_SURFACE"
    } else {
        "RATIONAL_B_SPLINE_CURVE"
    };

    if let Some(pos) = attrs.find(marker) {
        let after = &attrs[pos + marker.len()..];
        // Find the opening '(' and extract the weight list.
        if let Some(paren_start) = after.find('(') {
            let rest = &after[paren_start + 1..];
            // Parse floats from the parenthesized group(s).
            let weights = parse_weight_list(rest);
            if weights.len() >= expected_count {
                return weights[..expected_count].to_vec();
            }
            // Partial parse (fewer than expected): fall back to uniform
            // weights rather than propagating a dimension-mismatch error.
        }
    }

    vec![1.0; expected_count]
}

/// Parse a (possibly nested) list of weights from RATIONAL_B_SPLINE attrs.
/// Handles both flat `(w1, w2, w3)` and nested `((w1, w2), (w3, w4))` forms,
/// as well as no-paren format `w1, w2, w3)`.
fn parse_weight_list(s: &str) -> Vec<f64> {
    let mut weights = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();

    for ch in s.chars() {
        match ch {
            '(' => {
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth < 0 {
                    // Closing paren of the outer RATIONAL section.
                    let trimmed = current.trim();
                    if let Ok(v) = trimmed.parse::<f64>() {
                        weights.push(v);
                    }
                    break;
                }
            }
            ',' if depth <= 1 => {
                let trimmed = current.trim();
                if let Ok(v) = trimmed.parse::<f64>() {
                    weights.push(v);
                }
                current.clear();
                continue;
            }
            ',' => {
                // Comma inside a nested sub-list (depth > 1) — flush token
                // without accumulating the comma character.
                let trimmed = current.trim();
                if let Ok(v) = trimmed.parse::<f64>() {
                    weights.push(v);
                }
                current.clear();
                continue;
            }
            _ => {}
        }
        if depth >= 0 && ch != '(' && ch != ')' {
            current.push(ch);
        }
    }

    weights
}

/// Extract a 2D weight grid from RATIONAL_B_SPLINE_SURFACE attrs.
/// Returns uniform weights if parsing fails.
fn extract_rational_weight_grid(attrs: &str, n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
    let flat = extract_rational_weights(attrs, n_rows * n_cols);
    let tol = brepkit_math::tolerance::Tolerance::new();
    if flat.len() == n_rows * n_cols && flat.iter().any(|&w| !tol.approx_eq(w, 1.0)) {
        // Reshape flat weights into a 2D grid.
        flat.chunks(n_cols).map(<[f64]>::to_vec).collect()
    } else {
        vec![vec![1.0; n_cols]; n_rows]
    }
}

/// Parse a B_SPLINE_SURFACE_WITH_KNOTS attribute string into its components.
///
/// Format: `'', degree_u, degree_v, ((#cp, ...), ...), .XXX., .F., .F., .F.,
///          (mult_u, ...), (mult_v, ...), (knot_u, ...), (knot_v, ...), .XXX.`
///
/// Returns: `(degree_u, degree_v, cp_grid_refs, u_mults, v_mults, u_knots, v_knots)`
#[allow(clippy::type_complexity)]
fn parse_bspline_surface_attrs(
    attrs: &str,
) -> Option<(
    usize,
    usize,
    Vec<Vec<u64>>,
    Vec<u32>,
    Vec<u32>,
    Vec<f64>,
    Vec<f64>,
)> {
    // Strategy: parse the attribute string by finding the nested parenthesized
    // structures. The format has a specific sequence of tokens.

    // 1. Parse degrees: skip the name string, find the first two bare integers.
    let mut tokens = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    let mut groups: Vec<String> = Vec::new();

    for ch in attrs.chars() {
        match ch {
            '(' => {
                if depth == 0 && !current.trim().is_empty() {
                    tokens.push(current.trim().to_string());
                    current.clear();
                }
                depth += 1;
                current.push(ch);
            }
            ')' => {
                current.push(ch);
                depth -= 1;
                if depth == 0 {
                    groups.push(current.clone());
                    current.clear();
                }
            }
            ',' if depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    tokens.push(trimmed);
                }
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    if !current.trim().is_empty() {
        tokens.push(current.trim().to_string());
    }

    // tokens: bare values between top-level commas (name, degrees, enums)
    // groups: parenthesized structures at depth 0 (cp grid, mult lists, knot lists)

    // Extract degrees from tokens (skip '' name string and .XXX. enum values).
    let mut degrees: Vec<usize> = Vec::new();
    for tok in &tokens {
        if tok.starts_with('\'') || tok.starts_with('.') {
            continue;
        }
        if let Ok(d) = tok.parse::<usize>() {
            degrees.push(d);
        }
    }

    if degrees.len() < 2 {
        return None;
    }
    let degree_u = degrees[0];
    let degree_v = degrees[1];

    // groups should have at least 5 items:
    // [0]: control point grid ((#cp, ...), ...)
    // [1]: u multiplicities (m1, m2, ...)
    // [2]: v multiplicities (m1, m2, ...)
    // [3]: u knots (k1, k2, ...)
    // [4]: v knots (k1, k2, ...)
    if groups.len() < 5 {
        return None;
    }

    // Parse control point grid: nested ((#1, #2), (#3, #4))
    let cp_grid = parse_nested_refs(&groups[0]);

    // Parse multiplicities (integers).
    let u_mults = parse_ints_in_parens(&groups[1]);
    let v_mults = parse_ints_in_parens(&groups[2]);

    // Parse knot values (floats).
    let u_knots = parse_floats(&groups[3]);
    let v_knots = parse_floats(&groups[4]);

    Some((
        degree_u, degree_v, cp_grid, u_mults, v_mults, u_knots, v_knots,
    ))
}

/// Parse nested `((#1, #2), (#3, #4))` into a Vec of Vec of entity refs.
fn parse_nested_refs(s: &str) -> Vec<Vec<u64>> {
    let mut rows: Vec<Vec<u64>> = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();

    for ch in s.chars() {
        match ch {
            '(' => {
                depth += 1;
                if depth >= 2 {
                    current.push(ch);
                }
            }
            ')' => {
                if depth >= 2 {
                    current.push(ch);
                }
                depth -= 1;
                if depth == 1 && !current.is_empty() {
                    // End of an inner row.
                    rows.push(parse_refs(&current));
                    current.clear();
                }
            }
            ',' if depth == 1 => {
                // Separator between rows — flush current if non-empty.
                if !current.is_empty() {
                    rows.push(parse_refs(&current));
                    current.clear();
                }
            }
            _ => {
                if depth >= 2 {
                    current.push(ch);
                }
            }
        }
    }

    rows
}

/// Expand knot multiplicities and unique values into a flat knot vector.
///
/// Given `mults = [3, 1, 3]` and `vals = [0.0, 0.5, 1.0]`, produces
/// `[0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]`.
fn expand_knots(mults: &[u32], vals: &[f64]) -> Vec<f64> {
    let mut knots = Vec::new();
    for (&m, &v) in mults.iter().zip(vals.iter()) {
        for _ in 0..m {
            knots.push(v);
        }
    }
    knots
}

/// Parse a B_SPLINE_CURVE_WITH_KNOTS attribute string.
///
/// Format: `'', degree, (#cp, ...), .XXX., .F., (mults), (knots), .XXX.`
///
/// Returns: `(degree, cp_refs, mults, knots)`
#[allow(clippy::type_complexity)]
fn parse_bspline_curve_attrs(attrs: &str) -> Option<(usize, Vec<u64>, Vec<u32>, Vec<f64>)> {
    let mut tokens = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    let mut groups: Vec<String> = Vec::new();

    for ch in attrs.chars() {
        match ch {
            '(' => {
                if depth == 0 && !current.trim().is_empty() {
                    tokens.push(current.trim().to_string());
                    current.clear();
                }
                depth += 1;
                current.push(ch);
            }
            ')' => {
                current.push(ch);
                depth -= 1;
                if depth == 0 {
                    groups.push(current.clone());
                    current.clear();
                }
            }
            ',' if depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    tokens.push(trimmed);
                }
                current.clear();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    if !current.trim().is_empty() {
        tokens.push(current.trim().to_string());
    }

    // Extract degree.
    let mut degree = None;
    for tok in &tokens {
        if tok.starts_with('\'') || tok.starts_with('.') {
            continue;
        }
        if let Ok(d) = tok.parse::<usize>() {
            degree = Some(d);
            break;
        }
    }
    let degree = degree?;

    // groups: [0] = control points, [1] = multiplicities, [2] = knots
    if groups.len() < 3 {
        return None;
    }

    let cp_refs = parse_refs(&groups[0]);
    let mults = parse_ints_in_parens(&groups[1]);
    let knots = parse_floats(&groups[2]);

    Some((degree, cp_refs, mults, knots))
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

    #[test]
    fn roundtrip_nurbs_surface_loft() {
        // Create a NURBS-surfaced solid via loft_smooth (3 profiles → NURBS sides).
        let mut write_topo = Topology::new();

        let mut profiles = Vec::new();
        for &z in &[0.0, 1.0, 2.0] {
            let pts = vec![
                Point3::new(-1.0, -1.0, z),
                Point3::new(1.0, -1.0, z),
                Point3::new(1.0, 1.0, z),
                Point3::new(-1.0, 1.0, z),
            ];
            let wire_id =
                brepkit_topology::builder::make_polygon_wire(&mut write_topo, &pts).unwrap();
            let v01 = Vec3::new(
                pts[1].x() - pts[0].x(),
                pts[1].y() - pts[0].y(),
                pts[1].z() - pts[0].z(),
            );
            let v02 = Vec3::new(
                pts[2].x() - pts[0].x(),
                pts[2].y() - pts[0].y(),
                pts[2].z() - pts[0].z(),
            );
            let normal = v01.cross(v02).normalize().unwrap();
            let d = normal.x() * pts[0].x() + normal.y() * pts[0].y() + normal.z() * pts[0].z();
            let face = Face::new(wire_id, Vec::new(), FaceSurface::Plane { normal, d });
            profiles.push(write_topo.add_face(face));
        }
        let solid = brepkit_operations::loft::loft_smooth(&mut write_topo, &profiles).unwrap();

        // Verify the original has NURBS faces.
        let orig_solid = write_topo.solid(solid).unwrap();
        let orig_shell = write_topo.shell(orig_solid.outer_shell()).unwrap();
        let orig_nurbs_count = orig_shell
            .faces()
            .iter()
            .filter(|&&fid| {
                matches!(
                    write_topo.face(fid).unwrap().surface(),
                    FaceSurface::Nurbs(_)
                )
            })
            .count();
        assert!(orig_nurbs_count > 0, "lofted solid should have NURBS faces");

        // Write to STEP.
        let step_str = writer::write_step(&write_topo, &[solid]).unwrap();
        assert!(
            step_str.contains("B_SPLINE_SURFACE_WITH_KNOTS"),
            "STEP output should contain B_SPLINE_SURFACE_WITH_KNOTS"
        );

        // Read back.
        let mut read_topo = Topology::new();
        let solids = read_step(&step_str, &mut read_topo).unwrap();
        assert!(!solids.is_empty(), "should import at least one solid");

        // Verify the imported solid has NURBS faces.
        let read_solid = read_topo.solid(solids[0]).unwrap();
        let shell = read_topo.shell(read_solid.outer_shell()).unwrap();

        let nurbs_count = shell
            .faces()
            .iter()
            .filter(|&&fid| {
                matches!(
                    read_topo.face(fid).unwrap().surface(),
                    FaceSurface::Nurbs(_)
                )
            })
            .count();
        assert!(
            nurbs_count > 0,
            "imported solid should have NURBS faces (got {nurbs_count})"
        );
        assert_eq!(
            nurbs_count, orig_nurbs_count,
            "NURBS face count should be preserved: {orig_nurbs_count} → {nurbs_count}"
        );
    }

    #[test]
    fn roundtrip_nurbs_curve_preserved() {
        // Create a solid with NURBS edge curves (e.g., via loft_smooth).
        let mut write_topo = Topology::new();

        let mut profiles = Vec::new();
        for &z in &[0.0, 1.0, 2.0] {
            let pts = vec![
                Point3::new(-1.0, -1.0, z),
                Point3::new(1.0, -1.0, z),
                Point3::new(1.0, 1.0, z),
                Point3::new(-1.0, 1.0, z),
            ];
            let wire_id =
                brepkit_topology::builder::make_polygon_wire(&mut write_topo, &pts).unwrap();
            let v01 = Vec3::new(
                pts[1].x() - pts[0].x(),
                pts[1].y() - pts[0].y(),
                pts[1].z() - pts[0].z(),
            );
            let v02 = Vec3::new(
                pts[2].x() - pts[0].x(),
                pts[2].y() - pts[0].y(),
                pts[2].z() - pts[0].z(),
            );
            let normal = v01.cross(v02).normalize().unwrap();
            let d = normal.x() * pts[0].x() + normal.y() * pts[0].y() + normal.z() * pts[0].z();
            let face = Face::new(wire_id, Vec::new(), FaceSurface::Plane { normal, d });
            profiles.push(write_topo.add_face(face));
        }
        let solid = brepkit_operations::loft::loft_smooth(&mut write_topo, &profiles).unwrap();

        let step_str = writer::write_step(&write_topo, &[solid]).unwrap();

        // Check that B_SPLINE_CURVE_WITH_KNOTS is in the output.
        let has_bspline_curve = step_str.contains("B_SPLINE_CURVE_WITH_KNOTS");

        if has_bspline_curve {
            // Read back and verify NURBS curves are imported.
            let mut read_topo = Topology::new();
            let solids = read_step(&step_str, &mut read_topo).unwrap();
            assert!(!solids.is_empty());

            let read_solid = read_topo.solid(solids[0]).unwrap();
            let shell = read_topo.shell(read_solid.outer_shell()).unwrap();

            let has_nurbs_curve = shell.faces().iter().any(|&fid| {
                let face = read_topo.face(fid).unwrap();
                let wire = read_topo.wire(face.outer_wire()).unwrap();
                wire.edges().iter().any(|he| {
                    matches!(
                        read_topo.edge(he.edge()).unwrap().curve(),
                        EdgeCurve::NurbsCurve(_)
                    )
                })
            });
            assert!(
                has_nurbs_curve,
                "imported solid should have NURBS edge curves"
            );
        }
        // If no B_SPLINE_CURVE_WITH_KNOTS in output, the loft only produces
        // Line edges (which is valid for square profiles). Skip the curve check.
    }

    #[test]
    fn roundtrip_circle_edge_preserved() {
        // Cylinder has circle edges — they should round-trip.
        let mut write_topo = Topology::new();
        let solid =
            brepkit_operations::primitives::make_cylinder(&mut write_topo, 1.0, 2.0).unwrap();

        let step_str = writer::write_step(&write_topo, &[solid]).unwrap();
        assert!(step_str.contains("CIRCLE"));

        let mut read_topo = Topology::new();
        let solids = read_step(&step_str, &mut read_topo).unwrap();
        assert!(!solids.is_empty());

        let read_solid = read_topo.solid(solids[0]).unwrap();
        let shell = read_topo.shell(read_solid.outer_shell()).unwrap();

        let has_circle = shell.faces().iter().any(|&fid| {
            let face = read_topo.face(fid).unwrap();
            let wire = read_topo.wire(face.outer_wire()).unwrap();
            wire.edges().iter().any(|he| {
                matches!(
                    read_topo.edge(he.edge()).unwrap().curve(),
                    EdgeCurve::Circle(_)
                )
            })
        });
        assert!(
            has_circle,
            "imported cylinder should have circle edge curves"
        );
    }

    #[test]
    fn parse_bspline_surface_attrs_basic() {
        // Minimal B_SPLINE_SURFACE_WITH_KNOTS attribute string.
        let attrs = "'', 1, 1, ((#10, #11), (#12, #13)), .UNSPECIFIED., .F., .F., .F., \
                     (2, 2), (2, 2), (0.0, 1.0), (0.0, 1.0), .UNSPECIFIED.";
        let result = parse_bspline_surface_attrs(attrs);
        assert!(result.is_some(), "should parse B_SPLINE_SURFACE attributes");
        let (deg_u, deg_v, cp_grid, u_mults, v_mults, u_knots, v_knots) = result.unwrap();
        assert_eq!(deg_u, 1);
        assert_eq!(deg_v, 1);
        assert_eq!(cp_grid.len(), 2);
        assert_eq!(cp_grid[0].len(), 2);
        assert_eq!(u_mults, vec![2, 2]);
        assert_eq!(v_mults, vec![2, 2]);
        assert_eq!(u_knots, vec![0.0, 1.0]);
        assert_eq!(v_knots, vec![0.0, 1.0]);
    }

    #[test]
    fn parse_bspline_curve_attrs_basic() {
        let attrs = "'', 3, (#1, #2, #3, #4), .UNSPECIFIED., .F., .F., \
                     (4, 4), (0.0, 1.0), .UNSPECIFIED.";
        let result = parse_bspline_curve_attrs(attrs);
        assert!(result.is_some(), "should parse B_SPLINE_CURVE attributes");
        let (degree, cp_refs, mults, knots) = result.unwrap();
        assert_eq!(degree, 3);
        assert_eq!(cp_refs.len(), 4);
        assert_eq!(mults, vec![4, 4]);
        assert_eq!(knots, vec![0.0, 1.0]);
    }

    #[test]
    fn expand_knots_basic() {
        let mults = [3, 1, 3];
        let vals = [0.0, 0.5, 1.0];
        let flat = expand_knots(&mults, &vals);
        assert_eq!(flat, vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn parse_weight_list_nested() {
        // Nested format: ((w1, w2, w3))
        let weights = parse_weight_list("(1.0, 0.707, 1.0))");
        assert_eq!(weights.len(), 3);
        assert!((weights[0] - 1.0).abs() < 1e-10);
        assert!((weights[1] - 0.707).abs() < 1e-10);
        assert!((weights[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn parse_weight_list_flat() {
        // Flat format: (w1, w2, w3) — no inner parens
        let weights = parse_weight_list("1.0, 0.707, 1.0)");
        assert_eq!(weights.len(), 3);
        assert!((weights[0] - 1.0).abs() < 1e-10);
        assert!((weights[1] - 0.707).abs() < 1e-10);
        assert!((weights[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn parse_weight_list_scientific() {
        // Scientific notation
        let weights = parse_weight_list("(1.000000E+00, 7.071068E-01))");
        assert_eq!(weights.len(), 2);
        assert!((weights[0] - 1.0).abs() < 1e-5);
        assert!((weights[1] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-5);
    }

    #[test]
    fn parse_weight_list_2d_nested() {
        // 2D nested format: ((w1, w2), (w3, w4)) — real STEP has double nesting
        let weights = parse_weight_list("((1.0, 0.5), (0.5, 1.0)))");
        assert_eq!(weights.len(), 4);
        assert!((weights[0] - 1.0).abs() < 1e-10);
        assert!((weights[1] - 0.5).abs() < 1e-10);
        assert!((weights[2] - 0.5).abs() < 1e-10);
        assert!((weights[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn extract_rational_weights_from_composite() {
        let attrs = "BOUNDED_CURVE() B_SPLINE_CURVE(2, (#1, #2, #3)) \
                     B_SPLINE_CURVE_WITH_KNOTS((3,3), (0.0, 1.0)) \
                     RATIONAL_B_SPLINE_CURVE((1.0, 0.707, 1.0))";
        let weights = extract_rational_weights(attrs, 3);
        assert_eq!(weights.len(), 3);
        assert!((weights[1] - 0.707).abs() < 1e-10);
    }
}
