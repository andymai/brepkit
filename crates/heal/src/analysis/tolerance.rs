//! Tolerance analysis — tolerance distribution and statistics.

use brepkit_topology::Topology;
use brepkit_topology::shell::ShellId;

use crate::HealError;

/// Statistical summary of vertex tolerances in a shell.
#[derive(Debug, Clone)]
pub struct ToleranceAnalysis {
    /// Number of vertices examined.
    pub vertex_count: usize,
    /// Minimum vertex tolerance found.
    pub min_tolerance: f64,
    /// Maximum vertex tolerance found.
    pub max_tolerance: f64,
    /// Arithmetic mean of all vertex tolerances.
    pub average_tolerance: f64,
}

/// Compute tolerance statistics for all vertices in a shell.
///
/// Walks every face in the shell, collects unique vertices (by index),
/// and computes min/max/average of their tolerance values.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn analyze_tolerances(
    topo: &Topology,
    shell_id: ShellId,
) -> Result<ToleranceAnalysis, HealError> {
    let shell = topo.shell(shell_id)?;

    let mut seen = std::collections::HashSet::new();
    let mut min_tol = f64::INFINITY;
    let mut max_tol = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut count = 0usize;

    for &face_id in shell.faces() {
        let face = topo.face(face_id)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                for vid in [edge.start(), edge.end()] {
                    if seen.insert(vid.index()) {
                        let tol = topo.vertex(vid)?.tolerance();
                        if tol < min_tol {
                            min_tol = tol;
                        }
                        if tol > max_tol {
                            max_tol = tol;
                        }
                        sum += tol;
                        count += 1;
                    }
                }
            }
        }
    }

    if count == 0 {
        return Ok(ToleranceAnalysis {
            vertex_count: 0,
            min_tolerance: 0.0,
            max_tolerance: 0.0,
            average_tolerance: 0.0,
        });
    }

    Ok(ToleranceAnalysis {
        vertex_count: count,
        min_tolerance: min_tol,
        max_tolerance: max_tol,
        average_tolerance: sum / count as f64,
    })
}
