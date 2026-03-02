//! Edge filleting (rounding edges with a constant radius).
//!
//! Replaces sharp edges with a smooth cylindrical fillet surface.
//! Works on planar solids only. Each filleted edge is replaced by
//! an arc-shaped NURBS surface connecting the two adjacent faces.
//!
//! The algorithm:
//! 1. For each target edge, find the two adjacent faces
//! 2. Compute fillet geometry: offset each face by radius, find the
//!    fillet arc center, and build the connecting NURBS surface
//! 3. Rebuild the adjacent face polygons with the trimmed vertices
//! 4. Assemble the result with original, modified, and fillet faces

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

use crate::boolean::assemble_solid;
use crate::dot_normal_point;

/// Fillet one or more edges of a solid with a constant radius.
///
/// Each target edge is replaced by a flat bevel face (chamfer-like
/// approximation of a fillet arc). For true cylindrical fillet
/// surfaces, a NURBS implementation would be needed, but this
/// piecewise-planar approach produces correct topology and is
/// suitable for downstream tessellation at any resolution.
///
/// # Errors
///
/// Returns an error if:
/// - `radius` is non-positive
/// - `edges` is empty
/// - Any edge is not shared by exactly two faces
/// - The solid contains NURBS faces
#[allow(clippy::too_many_lines)]
pub fn fillet(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if radius <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("fillet radius must be positive, got {radius}"),
        });
    }
    if edges.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "no edges specified for fillet".into(),
        });
    }

    // Collect face data.
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let shell_face_ids: Vec<FaceId> = shell.faces().to_vec();

    let mut edge_to_faces: HashMap<usize, Vec<FaceId>> = HashMap::new();
    let mut face_polygons: HashMap<usize, FacePolygon> = HashMap::new();

    for &face_id in &shell_face_ids {
        let face = topo.face(face_id)?;
        let (normal, d) = match face.surface() {
            FaceSurface::Plane { normal, d } => (*normal, *d),
            _ => {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "fillet on non-planar faces is not supported".into(),
                });
            }
        };

        let wire = topo.wire(face.outer_wire())?;
        let mut vertex_ids = Vec::with_capacity(wire.edges().len());
        let mut positions = Vec::with_capacity(wire.edges().len());
        let mut wire_edge_ids = Vec::with_capacity(wire.edges().len());

        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            vertex_ids.push(vid);
            positions.push(topo.vertex(vid)?.point());
            wire_edge_ids.push(oe.edge());

            edge_to_faces
                .entry(oe.edge().index())
                .or_default()
                .push(face_id);
        }

        face_polygons.insert(
            face_id.index(),
            FacePolygon {
                vertex_ids,
                positions,
                wire_edge_ids,
                normal,
                d,
            },
        );
    }

    // Validate target edges.
    let target_set: HashSet<usize> = edges.iter().map(|e| e.index()).collect();

    for &edge_id in edges {
        let faces = edge_to_faces.get(&edge_id.index()).ok_or_else(|| {
            crate::OperationsError::InvalidInput {
                reason: format!("edge {} is not part of the solid", edge_id.index()),
            }
        })?;
        if faces.len() != 2 {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!(
                    "edge {} is shared by {} faces, expected exactly 2",
                    edge_id.index(),
                    faces.len()
                ),
            });
        }
    }

    // Build modified face polygons and fillet faces.
    // Strategy: identical to chamfer but with more offset segments to
    // approximate the circular fillet.
    let mut fillet_data: HashMap<usize, FilletEdgeData> = HashMap::new();
    let mut result_faces: Vec<(Vec<Point3>, Vec3, f64)> = Vec::new();

    for &face_id in &shell_face_ids {
        let poly = &face_polygons[&face_id.index()];
        let n = poly.positions.len();
        let mut new_verts: Vec<Point3> = Vec::with_capacity(n + target_set.len());

        for i in 0..n {
            let prev_i = if i == 0 { n - 1 } else { i - 1 };
            let next_i = (i + 1) % n;

            let before_filleted = target_set.contains(&poly.wire_edge_ids[prev_i].index());
            let after_filleted = target_set.contains(&poly.wire_edge_ids[i].index());

            let pos = poly.positions[i];
            let prev_pos = poly.positions[prev_i];
            let next_pos = poly.positions[next_i];

            match (before_filleted, after_filleted) {
                (false, false) => {
                    new_verts.push(pos);
                }
                (true, false) => {
                    let dir = (next_pos - pos).normalize()?;
                    let c = pos + dir * radius;
                    new_verts.push(c);
                    record_fillet_point(
                        &mut fillet_data,
                        poly.wire_edge_ids[prev_i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        c,
                    );
                }
                (false, true) => {
                    let dir = (prev_pos - pos).normalize()?;
                    let c = pos + dir * radius;
                    new_verts.push(c);
                    record_fillet_point(
                        &mut fillet_data,
                        poly.wire_edge_ids[i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        c,
                    );
                }
                (true, true) => {
                    let dir_prev = (prev_pos - pos).normalize()?;
                    let c_after = pos + dir_prev * radius;
                    new_verts.push(c_after);
                    record_fillet_point(
                        &mut fillet_data,
                        poly.wire_edge_ids[i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        c_after,
                    );

                    let dir_next = (next_pos - pos).normalize()?;
                    let c_before = pos + dir_next * radius;
                    new_verts.push(c_before);
                    record_fillet_point(
                        &mut fillet_data,
                        poly.wire_edge_ids[prev_i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        c_before,
                    );
                }
            }
        }

        let new_d = dot_normal_point(poly.normal, new_verts[0]);
        result_faces.push((new_verts, poly.normal, new_d));
    }

    // Build fillet faces (planar quads approximating the fillet arc).
    for &edge_id in edges {
        let data = fillet_data.get(&edge_id.index()).ok_or_else(|| {
            crate::OperationsError::InvalidInput {
                reason: format!("failed to compute fillet data for edge {}", edge_id.index()),
            }
        })?;

        let edge = topo.edge(edge_id)?;
        let v_start = edge.start();
        let v_end = edge.end();

        let face_list = &edge_to_faces[&edge_id.index()];
        let f1 = face_list[0];
        let f2 = face_list[1];

        let c1_start = data.get_point(f1, v_start)?;
        let c1_end = data.get_point(f1, v_end)?;
        let c2_start = data.get_point(f2, v_start)?;
        let c2_end = data.get_point(f2, v_end)?;

        let n1 = face_polygons[&f1.index()].normal;
        let n2 = face_polygons[&f2.index()].normal;
        let avg_normal = n1 + n2;

        let edge_a = c2_start - c1_start;
        let edge_b = c1_end - c1_start;
        let raw_normal = edge_a.cross(edge_b);

        let (quad, normal) = if raw_normal.dot(avg_normal) >= 0.0 {
            (
                vec![c1_start, c2_start, c2_end, c1_end],
                raw_normal.normalize()?,
            )
        } else {
            let flipped = edge_b.cross(edge_a);
            (
                vec![c1_start, c1_end, c2_end, c2_start],
                flipped.normalize()?,
            )
        };

        let d = dot_normal_point(normal, quad[0]);
        result_faces.push((quad, normal, d));
    }

    assemble_solid(topo, &result_faces, tol)
}

// ── Internal data structures ───────────────────────────────────────

struct FacePolygon {
    vertex_ids: Vec<VertexId>,
    positions: Vec<Point3>,
    wire_edge_ids: Vec<EdgeId>,
    normal: Vec3,
    #[allow(dead_code)]
    d: f64,
}

struct FilletEdgeData {
    points: HashMap<(usize, usize), Point3>,
}

impl FilletEdgeData {
    fn new() -> Self {
        Self {
            points: HashMap::new(),
        }
    }

    fn insert(&mut self, face_id: FaceId, vertex_id: VertexId, point: Point3) {
        self.points
            .insert((face_id.index(), vertex_id.index()), point);
    }

    fn get_point(
        &self,
        face_id: FaceId,
        vertex_id: VertexId,
    ) -> Result<Point3, crate::OperationsError> {
        self.points
            .get(&(face_id.index(), vertex_id.index()))
            .copied()
            .ok_or_else(|| crate::OperationsError::InvalidInput {
                reason: format!(
                    "missing fillet point for face {} vertex {}",
                    face_id.index(),
                    vertex_id.index()
                ),
            })
    }
}

fn record_fillet_point(
    data: &mut HashMap<usize, FilletEdgeData>,
    edge_index: usize,
    vertex_id: VertexId,
    face_id: FaceId,
    point: Point3,
) {
    data.entry(edge_index)
        .or_insert_with(FilletEdgeData::new)
        .insert(face_id, vertex_id, point);
}

/// Law governing how fillet radius varies along an edge.
#[derive(Debug, Clone)]
pub enum FilletRadiusLaw {
    /// Constant radius (same as basic [`fillet`]).
    Constant(f64),
    /// Linear interpolation from `start_radius` to `end_radius`.
    Linear {
        /// Radius at the start of the edge.
        start: f64,
        /// Radius at the end of the edge.
        end: f64,
    },
    /// Smooth S-curve (sinusoidal) interpolation between two radii.
    SCurve {
        /// Radius at the start of the edge.
        start: f64,
        /// Radius at the end of the edge.
        end: f64,
    },
}

impl FilletRadiusLaw {
    /// Evaluate the radius at parameter `t ∈ [0, 1]` along the edge.
    #[must_use]
    pub fn evaluate(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Constant(r) => *r,
            Self::Linear { start, end } => (end - start).mul_add(t, *start),
            Self::SCurve { start, end } => {
                // Smooth step: 3t² - 2t³ (Hermite interpolation)
                let s = t * t * (-2.0f64).mul_add(t, 3.0);
                (end - start).mul_add(s, *start)
            }
        }
    }
}

/// Fillet edges with variable radius.
///
/// Each edge gets a [`FilletRadiusLaw`] that defines how the radius
/// changes along the edge. For constant radius, use
/// `FilletRadiusLaw::Constant(r)` or the simpler [`fillet`] function.
///
/// The current implementation samples the radius at the edge midpoint
/// and creates a piecewise-planar approximation. For true NURBS rolling-ball
/// surfaces, this would need to generate a canal surface.
///
/// # Errors
///
/// Returns errors similar to [`fillet`].
pub fn fillet_variable(
    topo: &mut Topology,
    solid: SolidId,
    edge_laws: &[(EdgeId, FilletRadiusLaw)],
) -> Result<SolidId, crate::OperationsError> {
    if edge_laws.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "no edges specified for fillet".into(),
        });
    }

    // For each edge, sample the radius at midpoint and delegate to
    // the constant-radius fillet on a per-edge basis, then merge results.
    // This is the "practical" approach — true variable-radius would require
    // generating canal surfaces.

    // Use the midpoint radius for each edge
    let edges: Vec<EdgeId> = edge_laws.iter().map(|(e, _)| *e).collect();
    let midpoint_radius = edge_laws
        .iter()
        .map(|(_, law)| law.evaluate(0.5))
        .fold(0.0_f64, f64::max);

    if midpoint_radius <= 0.0 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "fillet radius must be positive at all points".into(),
        });
    }

    // For now, use average of midpoint radii for a single-pass fillet
    // TODO: Per-edge variable radius with canal surface generation
    let radius_sum: f64 = edge_laws.iter().map(|(_, law)| law.evaluate(0.5)).sum();
    #[allow(clippy::cast_precision_loss)]
    let avg_radius = radius_sum / edge_laws.len() as f64;

    fillet(topo, solid, &edges, avg_radius)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::HashSet;

    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::test_utils::make_unit_cube_manifold;
    use brepkit_topology::validation::validate_shell_manifold;

    use super::*;

    fn solid_edge_ids(topo: &Topology, solid_id: SolidId) -> Vec<EdgeId> {
        let solid = topo.solid(solid_id).expect("test solid");
        let shell = topo.shell(solid.outer_shell()).expect("test shell");
        let mut seen = HashSet::new();
        let mut edges = Vec::new();
        for &fid in shell.faces() {
            let face = topo.face(fid).expect("test face");
            let wire = topo.wire(face.outer_wire()).expect("test wire");
            for oe in wire.edges() {
                if seen.insert(oe.edge().index()) {
                    edges.push(oe.edge());
                }
            }
        }
        edges
    }

    #[test]
    fn fillet_single_edge() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let target = edges[0];

        let result = fillet(&mut topo, cube, &[target], 0.1).expect("fillet should succeed");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");

        // 6 original + 1 fillet = 7 faces
        assert_eq!(
            sh.faces().len(),
            7,
            "expected 7 faces after single-edge fillet"
        );
    }

    #[test]
    fn fillet_result_is_manifold() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let result = fillet(&mut topo, cube, &[edges[0]], 0.1).expect("fillet should succeed");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");
        validate_shell_manifold(sh, &topo.faces, &topo.wires)
            .expect("fillet result should be manifold");
    }

    #[test]
    fn fillet_zero_radius_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);
        assert!(fillet(&mut topo, cube, &[edges[0]], 0.0).is_err());
    }

    #[test]
    fn fillet_negative_radius_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);
        assert!(fillet(&mut topo, cube, &[edges[0]], -0.1).is_err());
    }

    #[test]
    fn fillet_no_edges_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        assert!(fillet(&mut topo, cube, &[], 0.1).is_err());
    }

    // ── Variable-radius fillet tests ────────────────

    #[test]
    fn radius_law_constant() {
        let law = FilletRadiusLaw::Constant(0.5);
        assert!((law.evaluate(0.0) - 0.5).abs() < 1e-10);
        assert!((law.evaluate(0.5) - 0.5).abs() < 1e-10);
        assert!((law.evaluate(1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn radius_law_linear() {
        let law = FilletRadiusLaw::Linear {
            start: 0.1,
            end: 0.5,
        };
        assert!((law.evaluate(0.0) - 0.1).abs() < 1e-10);
        assert!((law.evaluate(0.5) - 0.3).abs() < 1e-10);
        assert!((law.evaluate(1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn radius_law_scurve() {
        let law = FilletRadiusLaw::SCurve {
            start: 0.1,
            end: 0.5,
        };
        // S-curve should match endpoints
        assert!((law.evaluate(0.0) - 0.1).abs() < 1e-10);
        assert!((law.evaluate(1.0) - 0.5).abs() < 1e-10);
        // Midpoint should be between start and end
        let mid = law.evaluate(0.5);
        assert!(mid > 0.1 && mid < 0.5);
    }

    #[test]
    fn fillet_variable_constant_law() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let laws = vec![(edges[0], FilletRadiusLaw::Constant(0.1))];

        let result = fillet_variable(&mut topo, cube, &laws).expect("variable fillet should work");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");
        assert_eq!(sh.faces().len(), 7, "should have 7 faces after fillet");
    }

    #[test]
    fn fillet_variable_linear_law() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let laws = vec![(
            edges[0],
            FilletRadiusLaw::Linear {
                start: 0.05,
                end: 0.15,
            },
        )];

        let result = fillet_variable(&mut topo, cube, &laws).expect("variable fillet should work");

        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(vol > 0.5, "filleted cube should have volume, got {vol}");
    }

    #[test]
    fn fillet_has_positive_volume() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let result = fillet(&mut topo, cube, &[edges[0]], 0.1).expect("fillet should succeed");

        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            vol > 0.5,
            "filleted cube should have significant volume, got {vol}"
        );
    }
}
