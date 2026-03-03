//! STEP AP203 file writer.
//!
//! Exports B-Rep solids to ISO 10303-21 (STEP Part 21) format.
//! Supports planar faces with line edges and NURBS curves/surfaces.

use std::collections::HashMap;
use std::fmt::Write as _;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::WireId;

use crate::IoError;

/// Write one or more solids to STEP AP203 format.
///
/// Returns the STEP file as a UTF-8 string.
///
/// # Errors
///
/// Returns an error if:
/// - `solids` is empty
/// - Topology lookups fail
/// - An unsupported geometry type is encountered
#[allow(clippy::too_many_lines)]
pub fn write_step(topo: &Topology, solids: &[SolidId]) -> Result<String, IoError> {
    if solids.is_empty() {
        return Err(IoError::InvalidTopology {
            reason: "no solids to export".to_string(),
        });
    }

    let mut ctx = StepWriteContext::new();

    // Write product structure and geometric context.
    let repr_context_id = ctx.write_geometric_context();
    let product_ids = ctx.write_product_structure();

    // Write each solid.
    let mut brep_ids = Vec::new();
    for &solid_id in solids {
        let brep_id = ctx.write_solid(topo, solid_id)?;
        brep_ids.push(brep_id);
    }

    // Write shape representation containing all solids.
    let items: Vec<String> = brep_ids.iter().map(|id| format!("#{id}")).collect();
    let shape_repr_id = ctx.next_id();
    ctx.write_entity(
        shape_repr_id,
        "ADVANCED_BREP_SHAPE_REPRESENTATION",
        &format!(
            "'brepkit export', ({},), #{})",
            items.join(", "),
            repr_context_id
        ),
    );

    // Link shape representation to product definition.
    let prod_def_shape_id = ctx.next_id();
    ctx.write_entity(
        prod_def_shape_id,
        "PRODUCT_DEFINITION_SHAPE",
        &format!("'','',#{})", product_ids.definition),
    );

    let shape_def_repr_id = ctx.next_id();
    ctx.write_entity(
        shape_def_repr_id,
        "SHAPE_DEFINITION_REPRESENTATION",
        &format!("#{prod_def_shape_id}, #{shape_repr_id})"),
    );

    Ok(ctx.finish())
}

/// Incremental STEP entity ID counter and output buffer.
struct StepWriteContext {
    next: u64,
    entities: String,
    /// Vertex index to STEP entity ID.
    vertex_map: HashMap<u64, u64>,
    /// Edge index to STEP entity ID.
    edge_map: HashMap<u64, u64>,
}

/// Product structure entity IDs.
struct ProductIds {
    definition: u64,
}

impl StepWriteContext {
    fn new() -> Self {
        Self {
            next: 1,
            entities: String::new(),
            vertex_map: HashMap::new(),
            edge_map: HashMap::new(),
        }
    }

    const fn next_id(&mut self) -> u64 {
        let id = self.next;
        self.next += 1;
        id
    }

    fn write_entity(&mut self, id: u64, entity: &str, attrs: &str) {
        let _ = writeln!(self.entities, "#{id} = {entity}({attrs};");
    }

    fn write_point(&mut self, p: Point3) -> u64 {
        let id = self.next_id();
        self.write_entity(
            id,
            "CARTESIAN_POINT",
            &format!(
                "'', ({}, {}, {}))",
                fmt_f64(p.x()),
                fmt_f64(p.y()),
                fmt_f64(p.z())
            ),
        );
        id
    }

    fn write_direction(&mut self, d: Vec3) -> u64 {
        let id = self.next_id();
        self.write_entity(
            id,
            "DIRECTION",
            &format!(
                "'', ({}, {}, {}))",
                fmt_f64(d.x()),
                fmt_f64(d.y()),
                fmt_f64(d.z())
            ),
        );
        id
    }

    fn write_axis2_placement(&mut self, origin: Point3, axis: Vec3, ref_dir: Vec3) -> u64 {
        let origin_id = self.write_point(origin);
        let axis_id = self.write_direction(axis);
        let ref_id = self.write_direction(ref_dir);
        let id = self.next_id();
        self.write_entity(
            id,
            "AXIS2_PLACEMENT_3D",
            &format!("'', #{origin_id}, #{axis_id}, #{ref_id})"),
        );
        id
    }

    /// Write geometric context (units, representation context).
    fn write_geometric_context(&mut self) -> u64 {
        let len_unit = self.next_id();
        let _ = writeln!(
            self.entities,
            "#{len_unit} = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );"
        );

        let angle_unit = self.next_id();
        let _ = writeln!(
            self.entities,
            "#{angle_unit} = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );"
        );

        let solid_angle_unit = self.next_id();
        let _ = writeln!(
            self.entities,
            "#{solid_angle_unit} = ( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() );"
        );

        let uncertainty = self.next_id();
        self.write_entity(
            uncertainty,
            "UNCERTAINTY_MEASURE_WITH_UNIT",
            &format!(
                "LENGTH_MEASURE(1.E-07), #{len_unit}, 'distance_accuracy_value', \
                 'confusion accuracy')"
            ),
        );

        let ctx = self.next_id();
        let _ = writeln!(
            self.entities,
            "#{ctx} = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) \
             GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncertainty})) \
             GLOBAL_UNIT_ASSIGNED_CONTEXT((#{len_unit},#{angle_unit},#{solid_angle_unit})) \
             REPRESENTATION_CONTEXT('Context3D','3D Context with UNIT and UNCERTAINTY') );"
        );

        ctx
    }

    /// Write product structure entities.
    #[allow(clippy::similar_names)]
    fn write_product_structure(&mut self) -> ProductIds {
        let app_context = self.next_id();
        self.write_entity(
            app_context,
            "APPLICATION_CONTEXT",
            "'configuration controlled 3D design of mechanical parts and assemblies')",
        );

        let mech_context = self.next_id();
        self.write_entity(
            mech_context,
            "MECHANICAL_CONTEXT",
            &format!("'', #{app_context}, 'mechanical')"),
        );

        let protocol_def = self.next_id();
        self.write_entity(
            protocol_def,
            "APPLICATION_PROTOCOL_DEFINITION",
            &format!("'international standard', 'config_control_design', 1994, #{app_context})"),
        );

        let product = self.next_id();
        self.write_entity(
            product,
            "PRODUCT",
            &format!("'brepkit_solid', 'brepkit_solid', '', (#{mech_context}))"),
        );

        let formation = self.next_id();
        self.write_entity(
            formation,
            "PRODUCT_DEFINITION_FORMATION",
            &format!("'', '', #{product})"),
        );

        let def_context = self.next_id();
        self.write_entity(
            def_context,
            "PRODUCT_DEFINITION_CONTEXT",
            &format!("'part definition', #{app_context}, 'design')"),
        );

        let definition = self.next_id();
        self.write_entity(
            definition,
            "PRODUCT_DEFINITION",
            &format!("'design', '', #{formation}, #{def_context})"),
        );

        ProductIds { definition }
    }

    fn write_vertex(&mut self, topo: &Topology, vid: VertexId) -> Result<u64, IoError> {
        let key = vid.index() as u64;
        if let Some(&cached) = self.vertex_map.get(&key) {
            return Ok(cached);
        }

        let vertex = topo.vertex(vid).map_err(topo_err)?;
        let pt_id = self.write_point(vertex.point());
        let vp_id = self.next_id();
        self.write_entity(vp_id, "VERTEX_POINT", &format!("'', #{pt_id})"));

        self.vertex_map.insert(key, vp_id);
        Ok(vp_id)
    }

    fn write_edge_curve(&mut self, topo: &Topology, eid: EdgeId) -> Result<u64, IoError> {
        let key = eid.index() as u64;
        if let Some(&cached) = self.edge_map.get(&key) {
            return Ok(cached);
        }

        let edge = topo.edge(eid).map_err(topo_err)?;
        let start_vp = self.write_vertex(topo, edge.start())?;
        let end_vp = self.write_vertex(topo, edge.end())?;

        let curve_id = match edge.curve() {
            EdgeCurve::Line => {
                let start_pt = topo.vertex(edge.start()).map_err(topo_err)?.point();
                let end_pt = topo.vertex(edge.end()).map_err(topo_err)?.point();
                let dir = (end_pt - start_pt)
                    .normalize()
                    .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                let length = (end_pt - start_pt).length();

                let line_origin = self.write_point(start_pt);
                let dir_id = self.write_direction(dir);

                let vector = self.next_id();
                self.write_entity(
                    vector,
                    "VECTOR",
                    &format!("'', #{dir_id}, {})", fmt_f64(length)),
                );

                let line = self.next_id();
                self.write_entity(line, "LINE", &format!("'', #{line_origin}, #{vector})"));
                line
            }
            EdgeCurve::NurbsCurve(nurbs) => self.write_nurbs_curve(nurbs),
            EdgeCurve::Circle(circle) => {
                let placement =
                    self.write_axis2_placement(circle.center(), circle.normal(), circle.u_axis());
                let cid = self.next_id();
                self.write_entity(
                    cid,
                    "CIRCLE",
                    &format!("'', #{placement}, {}", fmt_f64(circle.radius())),
                );
                cid
            }
            EdgeCurve::Ellipse(ellipse) => {
                let placement = self.write_axis2_placement(
                    ellipse.center(),
                    ellipse.normal(),
                    ellipse.u_axis(),
                );
                let eid = self.next_id();
                self.write_entity(
                    eid,
                    "ELLIPSE",
                    &format!(
                        "'', #{placement}, {}, {}",
                        fmt_f64(ellipse.semi_major()),
                        fmt_f64(ellipse.semi_minor())
                    ),
                );
                eid
            }
        };

        let edge_curve = self.next_id();
        self.write_entity(
            edge_curve,
            "EDGE_CURVE",
            &format!("'', #{start_vp}, #{end_vp}, #{curve_id}, .T.)"),
        );

        self.edge_map.insert(key, edge_curve);
        Ok(edge_curve)
    }

    fn write_nurbs_curve(&mut self, nurbs: &brepkit_math::nurbs::NurbsCurve) -> u64 {
        let cp_ids: Vec<u64> = nurbs
            .control_points()
            .iter()
            .map(|p| self.write_point(*p))
            .collect();

        let cp_refs: Vec<String> = cp_ids.iter().map(|id| format!("#{id}")).collect();

        let knots = nurbs.knots();
        let (knot_mults, knot_vals) = compute_knot_multiplicities(knots);

        let mults_str: Vec<String> = knot_mults.iter().map(ToString::to_string).collect();
        let vals_str: Vec<String> = knot_vals.iter().map(|v| fmt_f64(*v)).collect();

        let id = self.next_id();
        let _ = writeln!(
            self.entities,
            "#{id} = B_SPLINE_CURVE_WITH_KNOTS('', {}, ({}), \
             .UNSPECIFIED., .F., .F., ({}), ({}), .UNSPECIFIED.);",
            nurbs.degree(),
            cp_refs.join(", "),
            mults_str.join(", "),
            vals_str.join(", "),
        );

        id
    }

    fn write_edge_loop(&mut self, topo: &Topology, wire_id: WireId) -> Result<u64, IoError> {
        let wire = topo.wire(wire_id).map_err(topo_err)?;
        let mut oriented_edge_ids = Vec::new();

        for oriented in wire.edges() {
            let edge_curve = self.write_edge_curve(topo, oriented.edge())?;
            let oriented_edge = self.next_id();
            let orient = if oriented.is_forward() { ".T." } else { ".F." };
            self.write_entity(
                oriented_edge,
                "ORIENTED_EDGE",
                &format!("'', *, *, #{edge_curve}, {orient})"),
            );
            oriented_edge_ids.push(oriented_edge);
        }

        let refs: Vec<String> = oriented_edge_ids
            .iter()
            .map(|id| format!("#{id}"))
            .collect();
        let loop_id = self.next_id();
        self.write_entity(loop_id, "EDGE_LOOP", &format!("'', ({}))", refs.join(", ")));

        Ok(loop_id)
    }

    #[allow(clippy::too_many_lines)]
    fn write_face(&mut self, topo: &Topology, face_id: FaceId) -> Result<u64, IoError> {
        let face = topo.face(face_id).map_err(topo_err)?;

        let mut bound_ids = Vec::new();

        let outer_loop = self.write_edge_loop(topo, face.outer_wire())?;
        let outer_bound = self.next_id();
        self.write_entity(
            outer_bound,
            "FACE_OUTER_BOUND",
            &format!("'', #{outer_loop}, .T.)"),
        );
        bound_ids.push(outer_bound);

        for &inner_wire in face.inner_wires() {
            let inner_loop = self.write_edge_loop(topo, inner_wire)?;
            let inner_bound = self.next_id();
            self.write_entity(
                inner_bound,
                "FACE_BOUND",
                &format!("'', #{inner_loop}, .T.)"),
            );
            bound_ids.push(inner_bound);
        }

        let surface_id = match face.surface() {
            FaceSurface::Plane { normal, d } => {
                let origin = Point3::new(normal.x() * d, normal.y() * d, normal.z() * d);
                let ref_dir = compute_ref_direction(*normal);
                let axis = self.write_axis2_placement(origin, *normal, ref_dir);
                let plane = self.next_id();
                self.write_entity(plane, "PLANE", &format!("'', #{axis})"));
                plane
            }
            FaceSurface::Nurbs(nurbs) => self.write_nurbs_surface(nurbs)?,
            FaceSurface::Cylinder(cyl) => {
                let ref_dir = compute_ref_direction(cyl.axis());
                let axis = self.write_axis2_placement(cyl.origin(), cyl.axis(), ref_dir);
                let id = self.next_id();
                self.write_entity(
                    id,
                    "CYLINDRICAL_SURFACE",
                    &format!("'', #{axis}, {:.15E})", cyl.radius()),
                );
                id
            }
            FaceSurface::Cone(cone) => {
                let ref_dir = compute_ref_direction(cone.axis());
                let axis = self.write_axis2_placement(cone.apex(), cone.axis(), ref_dir);
                let id = self.next_id();
                self.write_entity(
                    id,
                    "CONICAL_SURFACE",
                    &format!("'', #{axis}, 0.0E0, {:.15E})", cone.half_angle()),
                );
                id
            }
            FaceSurface::Sphere(sphere) => {
                let z = Vec3::new(0.0, 0.0, 1.0);
                let ref_dir = compute_ref_direction(z);
                let axis = self.write_axis2_placement(sphere.center(), z, ref_dir);
                let id = self.next_id();
                self.write_entity(
                    id,
                    "SPHERICAL_SURFACE",
                    &format!("'', #{axis}, {:.15E})", sphere.radius()),
                );
                id
            }
            FaceSurface::Torus(torus) => {
                let ref_dir = compute_ref_direction(torus.z_axis());
                let axis = self.write_axis2_placement(torus.center(), torus.z_axis(), ref_dir);
                let id = self.next_id();
                self.write_entity(
                    id,
                    "TOROIDAL_SURFACE",
                    &format!(
                        "'', #{axis}, {:.15E}, {:.15E})",
                        torus.major_radius(),
                        torus.minor_radius()
                    ),
                );
                id
            }
        };

        let bound_refs: Vec<String> = bound_ids.iter().map(|id| format!("#{id}")).collect();
        let advanced_face = self.next_id();
        self.write_entity(
            advanced_face,
            "ADVANCED_FACE",
            &format!("'', ({}), #{surface_id}, .T.)", bound_refs.join(", ")),
        );

        Ok(advanced_face)
    }

    fn write_nurbs_surface(
        &mut self,
        nurbs: &brepkit_math::nurbs::NurbsSurface,
    ) -> Result<u64, IoError> {
        let cps = nurbs.control_points();
        if cps.is_empty() {
            return Err(IoError::InvalidTopology {
                reason: "NURBS surface has no control points".to_string(),
            });
        }

        let mut cp_grid_refs = Vec::new();
        for row in cps {
            let row_ids: Vec<u64> = row.iter().map(|p| self.write_point(*p)).collect();
            let row_refs: Vec<String> = row_ids.iter().map(|id| format!("#{id}")).collect();
            cp_grid_refs.push(format!("({})", row_refs.join(", ")));
        }

        let (u_mults, u_vals) = compute_knot_multiplicities(nurbs.knots_u());
        let (v_mults, v_vals) = compute_knot_multiplicities(nurbs.knots_v());

        let u_mults_str: Vec<String> = u_mults.iter().map(ToString::to_string).collect();
        let u_vals_str: Vec<String> = u_vals.iter().map(|v| fmt_f64(*v)).collect();
        let v_mults_str: Vec<String> = v_mults.iter().map(ToString::to_string).collect();
        let v_vals_str: Vec<String> = v_vals.iter().map(|v| fmt_f64(*v)).collect();

        let id = self.next_id();
        let _ = writeln!(
            self.entities,
            "#{id} = B_SPLINE_SURFACE_WITH_KNOTS('', {}, {}, ({}), \
             .UNSPECIFIED., .F., .F., .F., ({}), ({}), ({}), ({}), .UNSPECIFIED.);",
            nurbs.degree_u(),
            nurbs.degree_v(),
            cp_grid_refs.join(", "),
            u_mults_str.join(", "),
            v_mults_str.join(", "),
            u_vals_str.join(", "),
            v_vals_str.join(", "),
        );

        Ok(id)
    }

    fn write_solid(&mut self, topo: &Topology, solid_id: SolidId) -> Result<u64, IoError> {
        let solid = topo.solid(solid_id).map_err(topo_err)?;
        let shell = self.write_shell(topo, solid.outer_shell())?;

        let brep = self.next_id();
        self.write_entity(brep, "MANIFOLD_SOLID_BREP", &format!("'', #{shell})"));
        Ok(brep)
    }

    fn write_shell(
        &mut self,
        topo: &Topology,
        shell_id: brepkit_topology::shell::ShellId,
    ) -> Result<u64, IoError> {
        let shell = topo.shell(shell_id).map_err(topo_err)?;
        let mut face_step_ids = Vec::new();

        for &face_id in shell.faces() {
            let step_face = self.write_face(topo, face_id)?;
            face_step_ids.push(step_face);
        }

        let refs: Vec<String> = face_step_ids.iter().map(|id| format!("#{id}")).collect();
        let closed_shell = self.next_id();
        self.write_entity(
            closed_shell,
            "CLOSED_SHELL",
            &format!("'', ({}))", refs.join(", ")),
        );

        Ok(closed_shell)
    }

    fn finish(self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "ISO-10303-21;");
        let _ = writeln!(out, "HEADER;");
        let _ = writeln!(out, "FILE_DESCRIPTION(('brepkit STEP export'), '2;1');");
        let _ = writeln!(
            out,
            "FILE_NAME('output.stp', '2024-01-01T00:00:00', (''), (''), \
             'brepkit', 'brepkit', '');"
        );
        let _ = writeln!(out, "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));");
        let _ = writeln!(out, "ENDSEC;");
        let _ = writeln!(out, "DATA;");
        out.push_str(&self.entities);
        let _ = writeln!(out, "ENDSEC;");
        let _ = writeln!(out, "END-ISO-10303-21;");
        out
    }
}

/// Format a float for STEP output with sufficient precision.
fn fmt_f64(v: f64) -> String {
    if v.abs() < 1e-15 {
        "0.".to_string()
    } else {
        format!("{v:.15E}")
    }
}

/// Compute a reference direction perpendicular to the given normal.
fn compute_ref_direction(normal: Vec3) -> Vec3 {
    let ax = Vec3::new(1.0, 0.0, 0.0);
    let ay = Vec3::new(0.0, 1.0, 0.0);

    let candidate = if normal.dot(ax).abs() < 0.9 { ax } else { ay };
    let ref_dir = normal.cross(candidate);
    ref_dir.normalize().unwrap_or(ax)
}

/// Compute knot multiplicities and unique knot values from a flat knot vector.
fn compute_knot_multiplicities(knots: &[f64]) -> (Vec<u32>, Vec<f64>) {
    if knots.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let mut mults = Vec::new();
    let mut vals = Vec::new();

    let mut current = knots[0];
    let mut count = 1u32;

    for &k in &knots[1..] {
        if (k - current).abs() < 1e-10 {
            count += 1;
        } else {
            mults.push(count);
            vals.push(current);
            current = k;
            count = 1;
        }
    }
    mults.push(count);
    vals.push(current);

    (mults, vals)
}

/// Convert a [`TopologyError`](brepkit_topology::TopologyError) into an [`IoError`].
fn topo_err(e: brepkit_topology::TopologyError) -> IoError {
    IoError::Operations(brepkit_operations::OperationsError::from(e))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube;

    use super::*;

    #[test]
    fn write_step_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let step_str = write_step(&topo, &[solid]).unwrap();

        assert!(step_str.contains("ISO-10303-21;"));
        assert!(step_str.contains("HEADER;"));
        assert!(step_str.contains("DATA;"));
        assert!(step_str.contains("END-ISO-10303-21;"));
    }

    #[test]
    fn step_contains_required_entities() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let step_str = write_step(&topo, &[solid]).unwrap();

        assert!(step_str.contains("MANIFOLD_SOLID_BREP"));
        assert!(step_str.contains("CLOSED_SHELL"));
        assert!(step_str.contains("ADVANCED_FACE"));
        assert!(step_str.contains("FACE_OUTER_BOUND"));
        assert!(step_str.contains("EDGE_LOOP"));
        assert!(step_str.contains("ORIENTED_EDGE"));
        assert!(step_str.contains("EDGE_CURVE"));
        assert!(step_str.contains("VERTEX_POINT"));
        assert!(step_str.contains("CARTESIAN_POINT"));
        assert!(step_str.contains("PLANE"));
    }

    #[test]
    fn step_contains_product_structure() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let step_str = write_step(&topo, &[solid]).unwrap();

        assert!(step_str.contains("PRODUCT("));
        assert!(step_str.contains("PRODUCT_DEFINITION("));
        assert!(step_str.contains("SHAPE_DEFINITION_REPRESENTATION"));
        assert!(step_str.contains("ADVANCED_BREP_SHAPE_REPRESENTATION"));
    }

    #[test]
    fn step_contains_geometric_context() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let step_str = write_step(&topo, &[solid]).unwrap();

        assert!(step_str.contains("LENGTH_UNIT"));
        assert!(step_str.contains("PLANE_ANGLE_UNIT"));
        assert!(step_str.contains("SI_UNIT"));
        assert!(step_str.contains("GEOMETRIC_REPRESENTATION_CONTEXT"));
    }

    #[test]
    fn step_unit_cube_has_six_faces() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let step_str = write_step(&topo, &[solid]).unwrap();

        let face_count = step_str.matches("ADVANCED_FACE(").count();
        assert_eq!(
            face_count, 6,
            "unit cube should have 6 ADVANCED_FACE entities"
        );
    }

    #[test]
    fn step_unit_cube_has_edges() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let step_str = write_step(&topo, &[solid]).unwrap();

        let edge_count = step_str.matches("EDGE_CURVE(").count();
        // Edges may or may not be shared depending on topology construction.
        assert!(edge_count >= 12, "unit cube should have at least 12 edges");
        assert!(edge_count <= 24, "unit cube should have at most 24 edges");
    }

    #[test]
    fn step_unit_cube_has_eight_vertices() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let step_str = write_step(&topo, &[solid]).unwrap();

        let vertex_count = step_str.matches("VERTEX_POINT(").count();
        assert_eq!(
            vertex_count, 8,
            "unit cube should have 8 VERTEX_POINT entities"
        );
    }

    #[test]
    fn step_box_primitive() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let step_str = write_step(&topo, &[solid]).unwrap();

        assert!(step_str.contains("MANIFOLD_SOLID_BREP"));
        let face_count = step_str.matches("ADVANCED_FACE(").count();
        assert_eq!(face_count, 6);
    }

    #[test]
    fn step_multiple_solids() {
        let mut topo = Topology::new();
        let s1 = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = make_unit_cube(&mut topo);

        let step_str = write_step(&topo, &[s1, s2]).unwrap();

        let brep_count = step_str.matches("MANIFOLD_SOLID_BREP(").count();
        assert_eq!(brep_count, 2);
    }

    #[test]
    fn step_empty_solids_error() {
        let topo = Topology::new();
        let result = write_step(&topo, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn step_entity_ids_are_sequential() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let step_str = write_step(&topo, &[solid]).unwrap();
        assert!(step_str.contains("#1 = "));
    }

    #[test]
    fn fmt_f64_output() {
        assert_eq!(fmt_f64(0.0), "0.");
        assert_eq!(fmt_f64(1e-20), "0.");

        let result = fmt_f64(1.5);
        assert!(result.contains("1.5"));
    }

    #[test]
    fn knot_multiplicities_basic() {
        let knots = vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        let (mults, vals) = compute_knot_multiplicities(&knots);

        assert_eq!(mults, vec![3, 1, 3]);
        assert_eq!(vals.len(), 3);
        assert!((vals[0]).abs() < 1e-10);
        assert!((vals[1] - 0.5).abs() < 1e-10);
        assert!((vals[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn step_exports_cylinder() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

        let step_str = write_step(&topo, &[solid]).unwrap();

        // Verify CYLINDRICAL_SURFACE entity is present.
        assert!(
            step_str.contains("CYLINDRICAL_SURFACE"),
            "STEP export should contain CYLINDRICAL_SURFACE entity"
        );
        // Verify the file is structurally valid.
        assert!(step_str.contains("MANIFOLD_SOLID_BREP"));
    }

    #[test]
    fn step_exports_sphere() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_sphere(&mut topo, 1.5, 16).unwrap();

        let step_str = write_step(&topo, &[solid]).unwrap();

        assert!(
            step_str.contains("SPHERICAL_SURFACE"),
            "STEP export should contain SPHERICAL_SURFACE entity"
        );
    }

    #[test]
    fn step_exports_cone() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_cone(&mut topo, 1.0, 0.0, 2.0).unwrap();

        let step_str = write_step(&topo, &[solid]).unwrap();

        assert!(
            step_str.contains("CONICAL_SURFACE"),
            "STEP export should contain CONICAL_SURFACE entity"
        );
    }
}
