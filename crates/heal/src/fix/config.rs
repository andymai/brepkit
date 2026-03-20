//! Fix configuration with tri-state modes.
//!
//! Each fix type has a [`FixMode`] that controls whether the fix is
//! applied: `Off` (never), `Auto` (when analysis detects the issue),
//! or `On` (always attempt).

/// Tri-state control for a fix operation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum FixMode {
    /// Never apply this fix.
    Off,
    /// Apply only if analysis detects the issue (default).
    #[default]
    Auto,
    /// Always attempt this fix.
    On,
}

impl FixMode {
    /// Whether this mode allows the fix when the issue is detected.
    #[must_use]
    pub fn should_fix(self, issue_detected: bool) -> bool {
        match self {
            Self::Off => false,
            Self::Auto => issue_detected,
            Self::On => true,
        }
    }
}

/// Configuration for all fix operations.
///
/// Each field controls a specific fix type.  The default configuration
/// sets all modes to `Auto`.
#[derive(Debug, Clone)]
#[allow(clippy::struct_field_names)]
pub struct FixConfig {
    // ── Wire fixes ──────────────────────────────────────────────
    /// Reorder edges in wires to form a connected chain.
    pub fix_reorder: FixMode,
    /// Close gaps between consecutive edges by adjusting vertices.
    pub fix_connectivity: FixMode,
    /// Ensure wires are topologically closed.
    pub fix_closure: FixMode,
    /// Remove edges shorter than tolerance.
    pub fix_small_edges: FixMode,
    /// Resolve self-intersecting wire edges.
    pub fix_self_intersection: FixMode,
    /// Remove degenerate edges (zero-length curves).
    pub fix_degenerate_edges: FixMode,
    /// Close 2D parameter-space gaps between edges.
    pub fix_gaps_2d: FixMode,
    /// Close 3D gaps between edges.
    pub fix_gaps_3d: FixMode,
    /// Fix edges where PCurve and 3D curve diverge.
    pub fix_lacking: FixMode,
    /// Remove notched (cusp) edges at vertices.
    pub fix_notched: FixMode,
    /// Remove tail edges (short dangling edges).
    pub fix_tail: FixMode,
    /// Fix intersecting adjacent edges.
    pub fix_intersecting_edges: FixMode,

    // ── Face fixes ──────────────────────────────────────────────
    /// Fix wire orientation relative to face normal.
    pub fix_wire_orientation: FixMode,
    /// Add natural boundary wire to faces on closed surfaces.
    pub fix_add_natural_bound: FixMode,
    /// Insert missing seam edges for periodic surfaces.
    pub fix_missing_seam: FixMode,
    /// Remove faces with area below tolerance.
    pub fix_small_area: FixMode,
    /// Remove duplicate faces (identical vertex sets).
    pub fix_duplicate_faces: FixMode,
    /// Fix intersecting wires within a face.
    pub fix_intersecting_wires: FixMode,

    // ── Shell fixes ─────────────────────────────────────────────
    /// Fix shell orientation (outward-facing normals).
    pub fix_orientation: FixMode,

    // ── Edge fixes ──────────────────────────────────────────────
    /// Fix SameParameter: ensure PCurve and 3D curve agree.
    pub fix_same_parameter: FixMode,
    /// Adjust vertex tolerances to match edge geometry.
    pub fix_vertex_tolerance: FixMode,
    /// Rebuild PCurves to match 3D curves.
    pub fix_pcurve: FixMode,

    // ── Solid fixes ─────────────────────────────────────────────
    /// Merge coincident vertices across the solid.
    pub fix_coincident_vertices: FixMode,

    // ── Global fixes ────────────────────────────────────────────
    /// Fix wireframe: repair missing or misaligned edges in shells.
    pub fix_wireframe: FixMode,
    /// Split vertices shared by too many non-adjacent edges.
    pub fix_split_common_vertex: FixMode,
    /// Remove or merge small faces.
    pub fix_small_faces: FixMode,
}

impl Default for FixConfig {
    fn default() -> Self {
        Self {
            fix_reorder: FixMode::Auto,
            fix_connectivity: FixMode::Auto,
            fix_closure: FixMode::Auto,
            fix_small_edges: FixMode::Auto,
            fix_self_intersection: FixMode::Auto,
            fix_degenerate_edges: FixMode::Auto,
            fix_gaps_2d: FixMode::Auto,
            fix_gaps_3d: FixMode::Auto,
            fix_lacking: FixMode::Auto,
            fix_notched: FixMode::Auto,
            fix_tail: FixMode::Auto,
            fix_intersecting_edges: FixMode::Auto,
            fix_wire_orientation: FixMode::Auto,
            fix_add_natural_bound: FixMode::Auto,
            fix_missing_seam: FixMode::Auto,
            fix_small_area: FixMode::Auto,
            fix_duplicate_faces: FixMode::Auto,
            fix_intersecting_wires: FixMode::Auto,
            fix_orientation: FixMode::Auto,
            fix_same_parameter: FixMode::Auto,
            fix_vertex_tolerance: FixMode::Auto,
            fix_pcurve: FixMode::Auto,
            fix_coincident_vertices: FixMode::Auto,
            fix_wireframe: FixMode::Auto,
            fix_split_common_vertex: FixMode::Auto,
            fix_small_faces: FixMode::Auto,
        }
    }
}
