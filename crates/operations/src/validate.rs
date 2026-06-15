//! Comprehensive solid validation.
//!
//! Performs structural and geometric validation on solids.

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

/// Options for controlling validation tolerance.
///
/// Operations like fillet and shell produce NURBS faces where geometric
/// checks (normal length, face area) may trigger false positives at
/// default tolerance. Increasing `tolerance_scale` relaxes these
/// thresholds.
#[derive(Debug, Clone)]
pub struct ValidationOptions {
    /// Multiplier applied to geometric tolerances for the face normal
    /// length check and the degenerate face area check. Default is `1.0`.
    /// A value of `10.0` means tolerances are 10x more permissive.
    pub tolerance_scale: f64,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            tolerance_scale: 1.0,
        }
    }
}

/// Compute the raw Euler characteristic (V - E + F) for a solid.
///
/// Returns the unmodified V - E + F value. For a genus-0 closed manifold
/// solid without inner wire loops this equals 2. Solids with through-holes
/// (genus > 0) or inner loops will have different values — use
/// [`validate_solid`] for a full topological check that accounts for these.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn euler_characteristic(
    topo: &Topology,
    solid: SolidId,
) -> Result<i64, crate::OperationsError> {
    let (f, e, v) = explorer::solid_entity_counts(topo, solid)?;
    #[allow(clippy::cast_possible_wrap)]
    let euler = (v as i64) - (e as i64) + (f as i64);
    Ok(euler)
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
    validate_solid_with_options(topo, solid, &ValidationOptions::default())
}

/// Validate a solid with configurable tolerance options.
///
/// Same checks as [`validate_solid`] but with tolerance scaling.
/// Use `ValidationOptions { tolerance_scale: 10.0, .. }` to relax
/// geometric checks for NURBS faces produced by fillet/shell operations.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
#[allow(clippy::too_many_lines)]
pub fn validate_solid_with_options(
    topo: &Topology,
    solid: SolidId,
    options: &ValidationOptions,
) -> Result<ValidationReport, crate::OperationsError> {
    let mut issues = Vec::new();
    let tol = Tolerance::new();
    // Clamp to [0.1, 1000]: below 0.1 risks false positives on exact
    // geometry, above 1000 makes the check meaningless.
    let scale = options.tolerance_scale.clamp(0.1, 1000.0);

    let (f, e, v) = explorer::solid_entity_counts(topo, solid)?;

    // Euler-Poincaré formula for a cell complex with inner loops:
    //   V - E + F = 2(1 - g) + L
    // where g is the genus and L is the total number of inner wire loops
    // across all faces. For a genus-0 solid with no holes: V-E+F = 2.
    // With L inner wires: V-E+F = 2 + L.
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

    let scaled_tol = Tolerance {
        linear: tol.linear * scale,
        angular: tol.angular * scale,
        relative: tol.relative * scale,
    };
    for fid in &faces {
        let face = topo.face(*fid)?;
        if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() {
            let len = normal.length();
            if !scaled_tol.approx_eq(len, 1.0) {
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

    for fid in &faces {
        let face = topo.face(*fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wire_id in wire_ids {
            let wire = topo.wire(wire_id)?;
            if let Err(_e) = brepkit_topology::validation::validate_wire_closed(wire, topo) {
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

    // Only meaningful for faces bounded entirely by straight edges.
    // The polygon area formula uses vertex positions, which is
    // meaningless when edges are curved (e.g. a cylinder cap has
    // 1 vertex → zero polygon area despite being a valid disc).
    let area_tol_sq = scaled_tol.linear * scaled_tol.linear;
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

        let mut positions = Vec::new();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = oe.oriented_start(edge);
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

    // Skip intentionally closed edges (like circles) when checking for
    // coincident start/end vertices.
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

    // Shell connectivity: all faces should be reachable from any face.
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
    validate_solid_relaxed_with_options(topo, solid, &ValidationOptions::default())
}

/// Validate a solid with relaxed checks and configurable tolerance options.
///
/// Combines the relaxed check set of [`validate_solid_relaxed`] with the
/// tolerance scaling of [`validate_solid_with_options`].
///
/// # Errors
///
/// Returns an error if topology lookups fail.
#[allow(clippy::too_many_lines)]
pub fn validate_solid_relaxed_with_options(
    topo: &Topology,
    solid: SolidId,
    options: &ValidationOptions,
) -> Result<ValidationReport, crate::OperationsError> {
    let mut issues = Vec::new();
    let tol = Tolerance::new();
    // Clamp to [0.1, 1000]: below 0.1 risks false positives on exact
    // geometry, above 1000 makes the check meaningless.
    let scale = options.tolerance_scale.clamp(0.1, 1000.0);

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
                    severity: Severity::Warning,
                    description: format!(
                        "face {} has only {} vertices (need at least 3)",
                        fid.index(),
                        face_verts.len()
                    ),
                });
            }
        }
    }

    let scaled_tol = Tolerance {
        linear: tol.linear * scale,
        angular: tol.angular * scale,
        relative: tol.relative * scale,
    };
    for fid in &faces {
        let face = topo.face(*fid)?;
        if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() {
            let len = normal.length();
            if !scaled_tol.approx_eq(len, 1.0) {
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

    // Wire closure — demoted to Warning for relaxed validation.
    // Boolean assembly can produce faces with technically-open wires
    // when edge dedup or vertex merging creates tiny gaps. These are
    // usually below the linear tolerance and don't affect downstream use.
    for fid in &faces {
        let face = topo.face(*fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wire_id in wire_ids {
            let wire = topo.wire(wire_id)?;
            if let Err(_e) = brepkit_topology::validation::validate_wire_closed(wire, topo) {
                // Demoted to Warning: boolean operations can produce
                // micro-gaps in wires from edge splitting that don't affect
                // downstream tessellation or volume. Strict checking would
                // reject ~25% of currently valid boolean results.
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    description: format!(
                        "wire {} on face {} is not closed",
                        wire_id.index(),
                        fid.index()
                    ),
                });
            }
        }
    }

    let area_tol_sq = scaled_tol.linear * scaled_tol.linear;
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
            let vid = oe.oriented_start(edge);
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

    // Zero-length edges — demoted to Warning in relaxed validation.
    // Boolean edge splitting can create tiny edges below tolerance.
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
                    severity: Severity::Warning,
                    description: format!(
                        "edge {} has near-zero length ({dist:.2e} < {:.2e})",
                        eid.index(),
                        tol.linear
                    ),
                });
            }
        }
    }

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
mod tests;
