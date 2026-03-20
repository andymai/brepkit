//! Check identifiers, severity levels, and validation issue types.

use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::shell::ShellId;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::WireId;

/// Identifies a specific validation check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckId {
    // Vertex checks
    /// Vertex not on edge 3D curve within tolerance.
    VertexOnCurve,
    /// Vertex not on face surface within tolerance.
    VertexOnSurface,
    // Edge checks
    /// Edge has no 3D curve representation.
    EdgeNoCurve3D,
    /// 3D curve deviates from PCurve(surface) beyond tolerance.
    EdgeSameParameter,
    /// Edge parameter range is invalid.
    EdgeRangeValid,
    /// Edge is degenerate (zero length).
    EdgeDegenerate,
    // Wire checks
    /// Wire contains no edges.
    WireEmpty,
    /// Consecutive edges not connected at shared vertices.
    WireNotConnected,
    /// Wire is not topologically closed (3D).
    WireClosure3D,
    /// Edge appears 3+ times in the same wire.
    WireRedundantEdge,
    // Face checks
    /// Face has no surface.
    FaceNoSurface,
    /// Face orientation inconsistent with wire winding.
    FaceOrientationConsistency,
    // Shell checks
    /// Shell contains no faces.
    ShellEmpty,
    /// Shell faces not all connected via shared edges.
    ShellConnected,
    /// Shell has free edges (not shared by exactly 2 faces).
    ShellClosed,
    /// Adjacent faces use shared edge in same direction (orientation inconsistent).
    ShellOrientationConsistent,
    // Solid checks
    /// Euler characteristic V-E+F != 2 for genus-0 solid.
    SolidEulerCharacteristic,
    /// Same face ID appears in multiple shells.
    SolidDuplicateFaces,
}

/// Issue severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational observation.
    Info,
    /// Potential problem.
    Warning,
    /// Invalid topology.
    Error,
}

/// Reference to a topological entity.
#[derive(Debug, Clone, Copy)]
pub enum EntityRef {
    /// A vertex.
    Vertex(VertexId),
    /// An edge.
    Edge(EdgeId),
    /// A wire.
    Wire(WireId),
    /// A face.
    Face(FaceId),
    /// A shell.
    Shell(ShellId),
    /// A solid.
    Solid(SolidId),
}

/// A single validation issue.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Which check detected this.
    pub check: CheckId,
    /// How severe.
    pub severity: Severity,
    /// Which entity.
    pub entity: EntityRef,
    /// Human-readable description.
    pub description: String,
    /// Measured deviation (for geometric checks).
    pub deviation: Option<f64>,
}

/// Result of validating a shape.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// All issues found.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Whether the shape passed all checks (no errors).
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
