//! Comprehensive solid validation.
//!
//! Equivalent to `BRepCheck_Analyzer` in `OpenCascade`. Performs
//! structural and geometric validation on solids.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::TopologyError;
use brepkit_topology::explorer;
use brepkit_topology::solid::SolidId;

/// A validation issue found in a solid.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity of the issue.
    pub severity: Severity,
    /// Human-readable description.
    pub description: String,
}

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// The solid is invalid and may cause downstream failures.
    Error,
    /// The solid has a potential problem but may still be usable.
    Warning,
}

/// Result of validating a solid.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// All issues found.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Whether the solid passed all validation checks (no errors).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.issues.iter().any(|i| i.severity == Severity::Error)
    }

    /// Count of error-severity issues.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .count()
    }

    /// Count of warning-severity issues.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Warning)
            .count()
    }
}

/// Validate a solid, returning a report of all issues found.
///
/// Checks performed:
/// Returns `true` if every edge in the face is a straight line.
fn face_all_edges_straight(
    topo: &Topology,
    face: &brepkit_topology::face::Face,
) -> Result<bool, TopologyError> {
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let wire = topo.wire(wire_id)?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            if !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// 1. **Euler-Poincaré**: V - E + F = 2(1 - g) for genus-g closed solid
/// 2. **Manifold edges**: each edge shared by exactly 2 faces
/// 3. **Boundary edges**: no edge shared by only 1 face (open shell)
/// 4. **Degenerate faces**: each face has at least 3 vertices
/// 5. **Face normal consistency**: normals should be non-zero
/// 6. **Wire closure**: every wire forms a closed loop
/// 7. **Degenerate face area**: near-zero polygon area warning for planar faces
/// 8. **Zero-length edges**: edges with coincident start/end vertices
/// 9. **Empty wires**: wires with no edges
/// 10. **Shell connectivity**: all faces reachable from any face
/// 11. **Redundant faces**: same face ID appearing twice in shell
/// 12. **Edge vertex consistency**: edge vertices belong to the solid
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn validate_solid(
    topo: &Topology,
    solid: SolidId,
) -> Result<ValidationReport, crate::OperationsError> {
    let mut issues = Vec::new();
    let tol = Tolerance::new();

    // 1. Entity counts and Euler characteristic.
    let (f, e, v) = explorer::solid_entity_counts(topo, solid)?;

    // Euler-Poincaré formula for a cell complex with inner loops:
    //   V - E + F = 2(1 - g) + L
    // where g is the genus and L is the total number of inner wire loops
    // across all faces. For a genus-0 solid with no holes: V-E+F = 2.
    // With L inner wires: V-E+F = 2 + L.
    //
    // Count total inner wires across all faces.
    let mut total_inner_loops: i64 = 0;
    let faces = explorer::solid_faces(topo, solid)?;
    for fid in &faces {
        let face = topo.face(*fid)?;
        #[allow(clippy::cast_possible_wrap)]
        {
            total_inner_loops += face.inner_wires().len() as i64;
        }
    }

    #[allow(clippy::cast_possible_wrap)]
    let euler = (v as i64) - (e as i64) + (f as i64);
    // Adjusted Euler: subtract inner loops to get the standard characteristic.
    let adjusted_euler = euler - total_inner_loops;
    let genus_times_2 = 2 - adjusted_euler;
    if genus_times_2 < 0 || genus_times_2 % 2 != 0 {
        issues.push(ValidationIssue {
            severity: Severity::Error,
            description: format!(
                "Euler characteristic V-E+F = {euler} is invalid \
                 (expected V-E+F = 2+L with L={total_inner_loops} inner loops, \
                 got V={v}, E={e}, F={f})"
            ),
        });
    }

    // 2 & 3. Edge sharing: each edge should be in exactly 2 faces.
    let edge_map = explorer::edge_to_face_map(topo, solid)?;
    let mut boundary_edges = 0;
    let mut non_manifold_edges = 0;

    for (&edge_idx, faces) in &edge_map {
        match faces.len() {
            0 => {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!("edge {edge_idx} is not referenced by any face"),
                });
            }
            1 => {
                boundary_edges += 1;
            }
            2 => {} // correct
            n => {
                non_manifold_edges += 1;
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!(
                        "edge {edge_idx} is shared by {n} faces (non-manifold, expected 2)"
                    ),
                });
            }
        }
    }

    if boundary_edges > 0 {
        issues.push(ValidationIssue {
            severity: Severity::Error,
            description: format!("{boundary_edges} boundary edge(s) found (shell is not closed)"),
        });
    }

    if non_manifold_edges > 0 {
        issues.push(ValidationIssue {
            severity: Severity::Error,
            description: format!("{non_manifold_edges} non-manifold edge(s) found"),
        });
    }

    // 4. Degenerate faces.
    //
    // Only faces on a planar surface bounded entirely by straight edges
    // require ≥3 unique vertices. Faces with curved edges (Circle,
    // Ellipse, NURBS) or non-planar surfaces (Cylinder, Sphere, Torus,
    // etc.) can validly have fewer vertices because the surface/edge
    // geometry defines the boundary shape.
    let faces = explorer::solid_faces(topo, solid)?;
    for fid in &faces {
        let face_data = topo.face(*fid)?;
        let is_planar = matches!(
            face_data.surface(),
            brepkit_topology::face::FaceSurface::Plane { .. }
        );

        if is_planar && face_all_edges_straight(topo, face_data)? {
            let face_verts = explorer::face_vertices(topo, *fid)?;
            if face_verts.len() < 3 {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!(
                        "face {} has only {} vertices (need at least 3)",
                        fid.index(),
                        face_verts.len()
                    ),
                });
            }
        }
    }

    // 5. Face normal consistency.
    for fid in &faces {
        let face = topo.face(*fid)?;
        if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() {
            let len = normal.length();
            if !tol.approx_eq(len, 1.0) {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    description: format!(
                        "face {} has non-unit normal (length = {len})",
                        fid.index()
                    ),
                });
            }
        }
    }

    // 6. Wire closure: every wire must form a closed loop.
    for fid in &faces {
        let face = topo.face(*fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wire_id in wire_ids {
            let wire = topo.wire(wire_id)?;
            if let Err(_e) = brepkit_topology::validation::validate_wire_closed(wire, topo.edges())
            {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!(
                        "wire {} on face {} is not closed",
                        wire_id.index(),
                        fid.index()
                    ),
                });
            }
        }
    }

    // 7. Degenerate face area: faces with near-zero polygon area are likely slivers.
    //
    // Only meaningful for faces bounded entirely by straight edges.
    // The polygon area formula uses vertex positions, which is
    // meaningless when edges are curved (e.g. a cylinder cap has
    // 1 vertex → zero polygon area despite being a valid disc).
    let area_tol_sq = tol.linear * tol.linear;
    for fid in &faces {
        let face = topo.face(*fid)?;

        // Skip non-planar faces and faces with curved edges — the polygon
        // area formula is only meaningful for planar faces with straight edges.
        if !matches!(
            face.surface(),
            brepkit_topology::face::FaceSurface::Plane { .. }
        ) {
            continue;
        }
        if !face_all_edges_straight(topo, face)? {
            continue;
        }

        let wire = topo.wire(face.outer_wire())?;

        // Collect wire vertex positions.
        let mut positions = Vec::new();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            positions.push(topo.vertex(vid)?.point());
        }

        // Compute face area using the shoelace formula generalized to 3D
        // (sum of cross products of consecutive edge vectors from centroid).
        if positions.len() >= 3 {
            let area = polygon_area_3d(&positions);
            if area < area_tol_sq {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    description: format!(
                        "face {} has near-zero area ({area:.2e} < {area_tol_sq:.2e})",
                        fid.index()
                    ),
                });
            }
        }
    }

    // 8. Zero-length edges: edges with coincident start/end vertices
    // (but not intentionally closed edges like circles).
    let all_edges = explorer::solid_edges(topo, solid)?;
    for eid in &all_edges {
        let edge = topo.edge(*eid)?;
        if !edge.is_closed() {
            let p_start = topo.vertex(edge.start())?.point();
            let p_end = topo.vertex(edge.end())?.point();
            let dx = p_start.x() - p_end.x();
            let dy = p_start.y() - p_end.y();
            let dz = p_start.z() - p_end.z();
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < tol.linear {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!(
                        "edge {} has near-zero length ({dist:.2e} < {:.2e})",
                        eid.index(),
                        tol.linear
                    ),
                });
            }
        }
    }

    // 9. Empty wires.
    for fid in &faces {
        let face = topo.face(*fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wire_id in wire_ids {
            let wire = topo.wire(wire_id)?;
            if wire.edges().is_empty() {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!(
                        "wire {} on face {} has no edges",
                        wire_id.index(),
                        fid.index()
                    ),
                });
            }
        }
    }

    // 10. Shell connectivity: all faces should be reachable from any face.
    // For genus-0 solids (sphere-like), all faces must be in one connected
    // component. Higher-genus solids (e.g. hollow revolves creating a torus)
    // can legitimately have multiple face-connected components (inner/outer
    // shells sharing no edges), so we skip this check for genus > 0.
    if !faces.is_empty() && genus_times_2 == 0 {
        let face_set: std::collections::HashSet<usize> = faces.iter().map(|f| f.index()).collect();
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        visited.insert(faces[0].index());
        queue.push_back(faces[0]);

        while let Some(current) = queue.pop_front() {
            // Find all faces that share an edge with current
            for adj_faces in edge_map.values() {
                if adj_faces.iter().any(|f| f.index() == current.index()) {
                    for neighbor in adj_faces {
                        if face_set.contains(&neighbor.index()) && visited.insert(neighbor.index())
                        {
                            queue.push_back(*neighbor);
                        }
                    }
                }
            }
        }

        let unreachable = face_set.len() - visited.len();
        if unreachable > 0 {
            issues.push(ValidationIssue {
                severity: Severity::Error,
                description: format!(
                    "shell is disconnected: {unreachable} face(s) not reachable from first face"
                ),
            });
        }
    }

    // 11. Redundant faces in shell.
    {
        let mut face_counts = std::collections::HashMap::new();
        for fid in &faces {
            *face_counts.entry(fid.index()).or_insert(0usize) += 1;
        }
        for (&idx, &count) in &face_counts {
            if count > 1 {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!("face {idx} appears {count} times in shell (redundant)"),
                });
            }
        }
    }

    // 12. Edge vertex consistency: edge vertices must belong to the solid.
    let vertex_set: std::collections::HashSet<usize> = {
        let verts = explorer::solid_vertices(topo, solid)?;
        verts.iter().map(|v| v.index()).collect()
    };
    for eid in &all_edges {
        let edge = topo.edge(*eid)?;
        if !vertex_set.contains(&edge.start().index()) {
            issues.push(ValidationIssue {
                severity: Severity::Error,
                description: format!(
                    "edge {} start vertex {} not found in solid",
                    eid.index(),
                    edge.start().index()
                ),
            });
        }
        if !vertex_set.contains(&edge.end().index()) {
            issues.push(ValidationIssue {
                severity: Severity::Error,
                description: format!(
                    "edge {} end vertex {} not found in solid",
                    eid.index(),
                    edge.end().index()
                ),
            });
        }
    }

    Ok(ValidationReport { issues })
}

/// Validate a solid with relaxed checks suitable for assembled geometry.
///
/// Operations like boolean, fillet, and shell produce solids where faces
/// may not share edges (each face has its own wire/edge topology). These
/// shapes are geometrically correct (volumes, tessellation, I/O all work)
/// but fail strict manifold checks.
///
/// Relaxed mode checks:
/// - Wire closure (every wire forms a closed loop)
/// - Degenerate faces (planar faces with < 3 vertices)
/// - Empty wires
/// - Zero-length edges
/// - Redundant faces
/// - Edge vertex consistency
///
/// Skipped in relaxed mode:
/// - Euler-Poincaré characteristic (assembled shells may have multiple components)
/// - Boundary edges (faces from different operations may not share edges)
/// - Non-manifold edges (edge duplication is expected in assembled geometry)
/// - Shell connectivity (multiple disconnected face groups are valid)
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn validate_solid_relaxed(
    topo: &Topology,
    solid: SolidId,
) -> Result<ValidationReport, crate::OperationsError> {
    let mut issues = Vec::new();
    let tol = Tolerance::new();

    let faces = explorer::solid_faces(topo, solid)?;

    // Degenerate faces (planar + straight edges only).
    for fid in &faces {
        let face_data = topo.face(*fid)?;
        let is_planar = matches!(
            face_data.surface(),
            brepkit_topology::face::FaceSurface::Plane { .. }
        );

        if is_planar && face_all_edges_straight(topo, face_data)? {
            let face_verts = explorer::face_vertices(topo, *fid)?;
            if face_verts.len() < 3 {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!(
                        "face {} has only {} vertices (need at least 3)",
                        fid.index(),
                        face_verts.len()
                    ),
                });
            }
        }
    }

    // Face normal consistency (planar faces).
    for fid in &faces {
        let face = topo.face(*fid)?;
        if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() {
            let len = normal.length();
            if !tol.approx_eq(len, 1.0) {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    description: format!(
                        "face {} has non-unit normal (length = {len})",
                        fid.index()
                    ),
                });
            }
        }
    }

    // Wire closure.
    for fid in &faces {
        let face = topo.face(*fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wire_id in wire_ids {
            let wire = topo.wire(wire_id)?;
            if let Err(_e) = brepkit_topology::validation::validate_wire_closed(wire, topo.edges())
            {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!(
                        "wire {} on face {} is not closed",
                        wire_id.index(),
                        fid.index()
                    ),
                });
            }
        }
    }

    // Degenerate face area (planar + straight edges only).
    let area_tol_sq = tol.linear * tol.linear;
    for fid in &faces {
        let face = topo.face(*fid)?;

        if !matches!(
            face.surface(),
            brepkit_topology::face::FaceSurface::Plane { .. }
        ) {
            continue;
        }
        if !face_all_edges_straight(topo, face)? {
            continue;
        }

        let wire = topo.wire(face.outer_wire())?;
        let mut positions = Vec::new();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            positions.push(topo.vertex(vid)?.point());
        }

        if positions.len() >= 3 {
            let area = polygon_area_3d(&positions);
            if area < area_tol_sq {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    description: format!(
                        "face {} has near-zero area ({area:.2e} < {area_tol_sq:.2e})",
                        fid.index()
                    ),
                });
            }
        }
    }

    // Zero-length edges.
    let all_edges = explorer::solid_edges(topo, solid)?;
    for eid in &all_edges {
        let edge = topo.edge(*eid)?;
        if !edge.is_closed() {
            let p_start = topo.vertex(edge.start())?.point();
            let p_end = topo.vertex(edge.end())?.point();
            let dx = p_start.x() - p_end.x();
            let dy = p_start.y() - p_end.y();
            let dz = p_start.z() - p_end.z();
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < tol.linear {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!(
                        "edge {} has near-zero length ({dist:.2e} < {:.2e})",
                        eid.index(),
                        tol.linear
                    ),
                });
            }
        }
    }

    // Empty wires.
    for fid in &faces {
        let face = topo.face(*fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wire_id in wire_ids {
            let wire = topo.wire(wire_id)?;
            if wire.edges().is_empty() {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!(
                        "wire {} on face {} has no edges",
                        wire_id.index(),
                        fid.index()
                    ),
                });
            }
        }
    }

    // Redundant faces.
    {
        let mut face_counts = std::collections::HashMap::new();
        for fid in &faces {
            *face_counts.entry(fid.index()).or_insert(0usize) += 1;
        }
        for (&idx, &count) in &face_counts {
            if count > 1 {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    description: format!("face {idx} appears {count} times in shell (redundant)"),
                });
            }
        }
    }

    // Edge vertex consistency.
    let vertex_set: std::collections::HashSet<usize> = {
        let verts = explorer::solid_vertices(topo, solid)?;
        verts.iter().map(|v| v.index()).collect()
    };
    for eid in &all_edges {
        let edge = topo.edge(*eid)?;
        if !vertex_set.contains(&edge.start().index()) {
            issues.push(ValidationIssue {
                severity: Severity::Error,
                description: format!(
                    "edge {} start vertex {} not found in solid",
                    eid.index(),
                    edge.start().index()
                ),
            });
        }
        if !vertex_set.contains(&edge.end().index()) {
            issues.push(ValidationIssue {
                severity: Severity::Error,
                description: format!(
                    "edge {} end vertex {} not found in solid",
                    eid.index(),
                    edge.end().index()
                ),
            });
        }
    }

    Ok(ValidationReport { issues })
}

/// Compute the area of a 3D polygon using the cross-product method.
///
/// For a planar polygon with vertices `p0, p1, ..., pN`, the area is
/// half the magnitude of the sum of cross products `(p[i] - p[0]) × (p[i+1] - p[0])`.
fn polygon_area_3d(positions: &[brepkit_math::vec::Point3]) -> f64 {
    use brepkit_math::vec::Vec3;

    if positions.len() < 3 {
        return 0.0;
    }

    let p0 = positions[0];
    let mut sum = Vec3::new(0.0, 0.0, 0.0);

    for i in 1..positions.len() - 1 {
        let a = positions[i] - p0;
        let b = positions[i + 1] - p0;
        sum += a.cross(b);
    }

    sum.length() * 0.5
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    use super::*;

    #[test]
    fn valid_cube() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let report = validate_solid(&topo, cube).unwrap();
        assert!(
            report.is_valid(),
            "manifold cube should be valid, got {} error(s): {:?}",
            report.error_count(),
            report.issues
        );
    }

    #[test]
    fn valid_box_primitive() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        assert!(
            report.is_valid(),
            "box should be valid: {:?}",
            report.issues
        );
    }

    #[test]
    fn extruded_solid_is_valid() {
        let mut topo = Topology::new();
        let face = brepkit_topology::test_utils::make_unit_square_face(&mut topo);
        let solid = crate::extrude::extrude(
            &mut topo,
            face,
            brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0),
            1.0,
        )
        .unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        assert!(
            report.is_valid(),
            "extruded solid should be valid: {:?}",
            report.issues
        );
    }

    #[test]
    fn report_counts_work() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let report = validate_solid(&topo, cube).unwrap();
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.warning_count(), 0);
    }

    #[test]
    fn open_shell_has_boundary_edges() {
        // A solid with an open shell (missing a face) should report boundary edges
        let mut topo = Topology::new();
        // Make a box but with 5 faces instead of 6 — one face missing
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Remove one face from the shell to make it open
        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let shell = topo.shell(shell_id).unwrap();
        let mut faces: Vec<_> = shell.faces().to_vec();
        faces.pop(); // Remove last face

        let open_shell = brepkit_topology::shell::Shell::new(faces).unwrap();
        *topo.shell_mut(shell_id).unwrap() = open_shell;

        let report = validate_solid(&topo, solid).unwrap();
        assert!(!report.is_valid(), "open shell should not be valid");
        assert!(
            report.error_count() > 0,
            "should have errors for boundary/euler"
        );
        // Check description mentions boundary
        let has_boundary_msg = report
            .issues
            .iter()
            .any(|i| i.description.contains("boundary") || i.description.contains("Euler"));
        assert!(has_boundary_msg, "should mention boundary edges or Euler");
    }

    #[test]
    fn report_is_valid_when_only_warnings() {
        let report = ValidationReport {
            issues: vec![ValidationIssue {
                severity: Severity::Warning,
                description: "minor issue".into(),
            }],
        };
        assert!(report.is_valid(), "warnings alone should not make invalid");
        assert_eq!(report.error_count(), 0);
        assert_eq!(report.warning_count(), 1);
    }

    #[test]
    fn report_not_valid_with_errors() {
        let report = ValidationReport {
            issues: vec![
                ValidationIssue {
                    severity: Severity::Error,
                    description: "critical issue".into(),
                },
                ValidationIssue {
                    severity: Severity::Warning,
                    description: "minor issue".into(),
                },
            ],
        };
        assert!(!report.is_valid());
        assert_eq!(report.error_count(), 1);
        assert_eq!(report.warning_count(), 1);
    }

    #[test]
    fn cylinder_solid_validates() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        assert!(
            report.is_valid(),
            "cylinder should be valid: {:?}",
            report.issues
        );
    }

    #[test]
    fn sphere_solid_validates() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_sphere(&mut topo, 2.0, 32).unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        assert!(
            report.is_valid(),
            "sphere should be valid: {:?}",
            report.issues
        );
    }

    #[test]
    fn cone_solid_validates() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_cone(&mut topo, 2.0, 0.0, 3.0).unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        assert!(
            report.is_valid(),
            "cone should be valid: {:?}",
            report.issues
        );
    }

    #[test]
    fn torus_solid_validates() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_torus(&mut topo, 5.0, 1.0, 32).unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        assert!(
            report.is_valid(),
            "torus should be valid: {:?}",
            report.issues
        );
    }

    #[test]
    fn hollow_revolve_is_valid() {
        use brepkit_math::vec::{Point3, Vec3};
        use brepkit_topology::face::{Face, FaceSurface};

        let mut topo = Topology::new();

        // Outer: 2×1 rectangle at x=1..3, y=0..1.
        let outer_pts = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(3.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];
        let outer_wire =
            brepkit_topology::builder::make_polygon_wire(&mut topo, &outer_pts).unwrap();

        // Inner: 0.5×0.5 hole.
        let inner_pts = vec![
            Point3::new(1.5, 0.25, 0.0),
            Point3::new(1.5, 0.75, 0.0),
            Point3::new(2.5, 0.75, 0.0),
            Point3::new(2.5, 0.25, 0.0),
        ];
        let inner_wire =
            brepkit_topology::builder::make_polygon_wire(&mut topo, &inner_pts).unwrap();

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let face = Face::new(
            outer_wire,
            vec![inner_wire],
            FaceSurface::Plane { normal, d: 0.0 },
        );
        let face_id = topo.add_face(face);

        // Full revolution → genus-1 torus topology.
        let solid = crate::revolve::revolve(
            &mut topo,
            face_id,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            2.0 * std::f64::consts::PI,
        )
        .unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        assert!(
            report.is_valid(),
            "genus-1 hollow revolve should be valid, got {} error(s): {:?}",
            report.error_count(),
            report.issues
        );
    }

    #[test]
    fn extruded_hollow_box_is_valid() {
        use brepkit_math::vec::{Point3, Vec3};
        use brepkit_topology::face::{Face, FaceSurface};

        let mut topo = Topology::new();

        // Outer: 2×2 square.
        let outer_pts = vec![
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
        ];
        let outer_wire =
            brepkit_topology::builder::make_polygon_wire(&mut topo, &outer_pts).unwrap();

        // Inner: 0.5×0.5 hole.
        let inner_pts = vec![
            Point3::new(-0.25, -0.25, 0.0),
            Point3::new(-0.25, 0.25, 0.0),
            Point3::new(0.25, 0.25, 0.0),
            Point3::new(0.25, -0.25, 0.0),
        ];
        let inner_wire =
            brepkit_topology::builder::make_polygon_wire(&mut topo, &inner_pts).unwrap();

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let face = Face::new(
            outer_wire,
            vec![inner_wire],
            FaceSurface::Plane { normal, d: 0.0 },
        );
        let face_id = topo.add_face(face);

        // Extrude → genus-0 (V-E+F=2) with inner walls.
        let solid =
            crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        assert!(
            report.is_valid(),
            "extruded hollow box should be valid, got {} error(s): {:?}",
            report.error_count(),
            report.issues
        );
    }

    // ── Wire closure validation ──────────────────────

    #[test]
    fn wire_closure_check_on_valid_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        // No wire closure errors on a properly-built box.
        let wire_issues: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.description.contains("wire"))
            .collect();
        assert!(
            wire_issues.is_empty(),
            "valid box should have no wire issues: {wire_issues:?}"
        );
    }

    // ── Degenerate face area ─────────────────────────

    #[test]
    fn polygon_area_unit_square() {
        use brepkit_math::vec::Point3;
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let area = polygon_area_3d(&pts);
        assert!(
            (area - 1.0).abs() < 1e-10,
            "unit square area should be 1.0, got {area}"
        );
    }

    #[test]
    fn polygon_area_triangle() {
        use brepkit_math::vec::Point3;
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(0.0, 3.0, 0.0),
        ];
        let area = polygon_area_3d(&pts);
        assert!(
            (area - 3.0).abs() < 1e-10,
            "triangle area should be 3.0, got {area}"
        );
    }

    #[test]
    fn polygon_area_degenerate() {
        use brepkit_math::vec::Point3;
        // Collinear points → area 0.
        let pts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ];
        let area = polygon_area_3d(&pts);
        assert!(
            area < 1e-15,
            "collinear points should have zero area, got {area}"
        );
    }

    #[test]
    fn validate_detects_non_manifold_edge() {
        // 3 faces sharing one edge is non-manifold
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Duplicate one face to create a non-manifold edge
        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let shell = topo.shell(shell_id).unwrap();
        let mut faces: Vec<_> = shell.faces().to_vec();
        let extra_face = faces[0]; // duplicate first face
        faces.push(extra_face);

        let new_shell = brepkit_topology::shell::Shell::new(faces).unwrap();
        *topo.shell_mut(shell_id).unwrap() = new_shell;

        let report = validate_solid(&topo, solid).unwrap();
        assert!(!report.is_valid(), "non-manifold edge should be invalid");
        let has_nm = report
            .issues
            .iter()
            .any(|i| i.description.contains("non-manifold") || i.description.contains("shared by"));
        assert!(has_nm, "should mention non-manifold: {:?}", report.issues);
    }

    #[test]
    fn validate_detects_zero_length_normal() {
        // A face with a zero-length normal should produce a warning.
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Corrupt a face normal to zero-length.
        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let face_id = topo.shell(shell_id).unwrap().faces()[0];
        let face = topo.face_mut(face_id).unwrap();
        *face = brepkit_topology::face::Face::new(
            face.outer_wire(),
            face.inner_wires().to_vec(),
            brepkit_topology::face::FaceSurface::Plane {
                normal: brepkit_math::vec::Vec3::new(0.0, 0.0, 0.0),
                d: 0.0,
            },
        );

        let report = validate_solid(&topo, solid).unwrap();
        assert!(
            report.warning_count() > 0,
            "zero-length normal should produce a warning: {:?}",
            report.issues
        );
    }

    #[test]
    fn no_area_warnings_on_valid_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        let area_warnings: Vec<_> = report
            .issues
            .iter()
            .filter(|i| i.description.contains("area"))
            .collect();
        assert!(
            area_warnings.is_empty(),
            "valid box should have no area warnings: {area_warnings:?}"
        );
    }

    #[test]
    fn validate_detects_open_wire() {
        // Construct a solid with a wire that doesn't close
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Get a wire and break its closure by removing the closed flag
        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let face_id = topo.shell(shell_id).unwrap().faces()[0];
        let wire_id = topo.face(face_id).unwrap().outer_wire();
        let wire = topo.wire(wire_id).unwrap();
        let edges = wire.edges().to_vec();

        // Create an open wire (not closed) with same edges
        if edges.len() > 1 {
            use brepkit_topology::wire::Wire;
            let open_wire = Wire::new(edges[..edges.len() - 1].to_vec(), false);
            if let Ok(w) = open_wire {
                *topo.wire_mut(wire_id).unwrap() = w;

                let report = validate_solid(&topo, solid).unwrap();
                assert!(
                    !report.is_valid(),
                    "open wire should be invalid: {:?}",
                    report.issues
                );
            }
        }
    }

    // ── repair_solid ─────────────────────────────────

    // ── Zero-length edge detection ─────────────────────

    #[test]
    fn validate_detects_zero_length_edge() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Find an edge and make its vertices coincident
        let edges = explorer::solid_edges(&topo, solid).unwrap();
        let edge = topo.edge(edges[0]).unwrap();
        let end_vid = edge.end();
        let start_pos = topo.vertex(edge.start()).unwrap().point();

        // Move end vertex to same position as start
        topo.vertex_mut(end_vid).unwrap().set_point(start_pos);

        let report = validate_solid(&topo, solid).unwrap();
        let has_zero_length = report.issues.iter().any(|i| {
            i.description.contains("zero length") || i.description.contains("near-zero length")
        });
        assert!(
            has_zero_length,
            "should detect zero-length edge: {:?}",
            report.issues
        );
    }

    // ── Disconnected shell detection ───────────────────

    #[test]
    fn validate_connected_shell_passes() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        let has_disconnect = report
            .issues
            .iter()
            .any(|i| i.description.contains("disconnected"));
        assert!(
            !has_disconnect,
            "valid box should not be disconnected: {:?}",
            report.issues
        );
    }

    // ── Redundant face detection ───────────────────────

    #[test]
    fn validate_detects_redundant_face() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Duplicate a face in the shell
        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let shell = topo.shell(shell_id).unwrap();
        let mut faces: Vec<_> = shell.faces().to_vec();
        let dup = faces[0];
        faces.push(dup);

        let new_shell = brepkit_topology::shell::Shell::new(faces).unwrap();
        *topo.shell_mut(shell_id).unwrap() = new_shell;

        let report = validate_solid(&topo, solid).unwrap();
        let has_redundant = report
            .issues
            .iter()
            .any(|i| i.description.contains("redundant") || i.description.contains("appears"));
        assert!(
            has_redundant,
            "should detect redundant face: {:?}",
            report.issues
        );
    }

    // ── Boolean result validation ──────────────────────

    #[test]
    fn boolean_fuse_result_validates() {
        let mut topo = Topology::new();
        let a = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

        let result =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, a, b).unwrap();

        let report = validate_solid(&topo, result).unwrap();
        assert!(
            report.is_valid(),
            "boolean fuse should produce a valid solid: {:?}",
            report.issues
        );
    }

    #[test]
    fn boolean_cut_result_validates() {
        let mut topo = Topology::new();
        let a = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

        let result =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, a, b).unwrap();

        let report = validate_solid(&topo, result).unwrap();
        assert!(
            report.is_valid(),
            "boolean cut should produce a valid solid: {:?}",
            report.issues
        );
    }

    #[test]
    fn boolean_intersect_result_validates() {
        let mut topo = Topology::new();
        let a = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

        let result =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Intersect, a, b).unwrap();

        let report = validate_solid(&topo, result).unwrap();
        assert!(
            report.is_valid(),
            "boolean intersect should produce a valid solid: {:?}",
            report.issues
        );
    }

    #[test]
    #[allow(deprecated)]
    fn fillet_result_validates() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Find edges for fillet
        let s = topo.solid(cube).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        let mut edges = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &fid in sh.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                if seen.insert(oe.edge().index()) {
                    edges.push(oe.edge());
                }
            }
        }

        let result = crate::fillet::fillet(&mut topo, cube, &[edges[0]], 0.1).unwrap();

        let report = validate_solid(&topo, result).unwrap();
        assert!(
            report.is_valid(),
            "fillet should produce a valid solid: {:?}",
            report.issues
        );
    }

    #[test]
    fn extrude_result_validates() {
        let mut topo = Topology::new();
        let face = brepkit_topology::test_utils::make_unit_square_face(&mut topo);
        let solid = crate::extrude::extrude(
            &mut topo,
            face,
            brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0),
            2.0,
        )
        .unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        assert!(
            report.is_valid(),
            "extrude result should validate: {:?}",
            report.issues
        );
    }

    #[test]
    fn revolve_result_validates() {
        use brepkit_math::vec::{Point3, Vec3};

        let mut topo = Topology::new();
        let face = brepkit_topology::test_utils::make_unit_square_face(&mut topo);

        // Move face away from axis to avoid degenerate geometry
        for vid in explorer::face_vertices(&topo, face).unwrap() {
            let v = topo.vertex_mut(vid).unwrap();
            v.set_point(Point3::new(
                v.point().x() + 2.0,
                v.point().y(),
                v.point().z(),
            ));
        }

        let solid = crate::revolve::revolve(
            &mut topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            std::f64::consts::PI,
        )
        .unwrap();

        let report = validate_solid(&topo, solid).unwrap();
        assert!(
            report.is_valid(),
            "revolve result should validate: {:?}",
            report.issues
        );
    }

    // ── repair_solid ─────────────────────────────────

    #[test]
    fn repair_clean_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let report = crate::heal::repair_solid(&mut topo, solid, 1e-7).unwrap();
        assert!(
            report.is_valid_after(),
            "clean box should be valid after repair"
        );
        assert_eq!(report.total_repairs(), 0, "clean box needs no repairs");
    }

    #[test]
    fn repair_preserves_volume() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let vol_before = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let _report = crate::heal::repair_solid(&mut topo, solid, 1e-7).unwrap();
        let vol_after = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

        assert!(
            (vol_before - vol_after).abs() < 0.01,
            "repair should preserve volume: {vol_before} vs {vol_after}"
        );
    }

    #[test]
    fn repair_cylinder_no_crash() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

        let report = crate::heal::repair_solid(&mut topo, solid, 1e-7).unwrap();
        // Should not crash; may or may not be valid depending on cylinder topology
        let _ = report.is_valid_after();
    }

    // ── Relaxed validation ────────────────────────────

    #[test]
    fn relaxed_valid_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let report = validate_solid_relaxed(&topo, solid).unwrap();
        assert!(
            report.is_valid(),
            "box should pass relaxed validation: {:?}",
            report.issues
        );
    }

    #[test]
    fn relaxed_fillet_passes() {
        let mut topo = Topology::new();
        let cube = crate::primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();

        let s = topo.solid(cube).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        let mut edges = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &fid in sh.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                if seen.insert(oe.edge().index()) {
                    edges.push(oe.edge());
                }
            }
        }

        let result = crate::fillet::fillet_rolling_ball(&mut topo, cube, &[edges[0]], 2.0).unwrap();

        // Strict validation fails for fillet results (boundary edges + disconnected)
        let strict = validate_solid(&topo, result).unwrap();
        assert!(!strict.is_valid(), "fillet should fail strict validation");

        // Relaxed validation should pass
        let relaxed = validate_solid_relaxed(&topo, result).unwrap();
        assert!(
            relaxed.is_valid(),
            "fillet should pass relaxed validation: {:?}",
            relaxed.issues
        );
    }

    #[test]
    fn relaxed_shell_passes() {
        let mut topo = Topology::new();
        let cube = crate::primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();

        let s = topo.solid(cube).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        let open_face = sh.faces()[0];

        let result = crate::shell_op::shell(&mut topo, cube, 1.0, &[open_face]).unwrap();

        // Both strict and relaxed validation should pass for properly
        // constructed shells (plane-intersected inner vertex positions
        // ensure watertight geometry).
        let strict = validate_solid(&topo, result).unwrap();
        assert!(
            strict.is_valid(),
            "shell should pass strict validation: {:?}",
            strict.issues
        );

        let relaxed = validate_solid_relaxed(&topo, result).unwrap();
        assert!(
            relaxed.is_valid(),
            "shell should pass relaxed validation: {:?}",
            relaxed.issues
        );
    }

    #[test]
    fn relaxed_boolean_cut_passes() {
        let mut topo = Topology::new();
        let a = crate::primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
        let b = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

        let mat = brepkit_math::mat::Mat4::translation(5.0, 5.0, 5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let result =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, a, b).unwrap();

        let relaxed = validate_solid_relaxed(&topo, result).unwrap();
        assert!(
            relaxed.is_valid(),
            "boolean cut should pass relaxed validation: {:?}",
            relaxed.issues
        );
    }

    #[test]
    fn relaxed_detects_open_wire() {
        use brepkit_topology::wire::Wire;

        // Open wire should still be caught by relaxed validation
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let face_id = topo.shell(shell_id).unwrap().faces()[0];
        let wire_id = topo.face(face_id).unwrap().outer_wire();
        let wire = topo.wire(wire_id).unwrap();
        let edges = wire.edges().to_vec();

        assert!(edges.len() > 1, "box face should have > 1 edge");
        let open_wire = Wire::new(edges[..edges.len() - 1].to_vec(), false).unwrap();
        *topo.wire_mut(wire_id).unwrap() = open_wire;

        let report = validate_solid_relaxed(&topo, solid).unwrap();
        assert!(
            !report.is_valid(),
            "open wire should fail even relaxed validation: {:?}",
            report.issues
        );
    }
}
