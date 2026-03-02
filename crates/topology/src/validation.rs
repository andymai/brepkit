//! Topology validation utilities.
//!
//! These functions check structural invariants of topological entities
//! such as wire closure and shell manifoldness.

use crate::TopologyError;
use crate::arena::Arena;
use crate::edge::Edge;
use crate::shell::Shell;
use crate::wire::Wire;

/// Validates that a wire forms a closed loop.
///
/// A closed wire requires that for each consecutive pair of oriented edges
/// the end vertex of the first equals the start vertex of the second, and
/// that the last edge connects back to the first.
///
/// # Errors
///
/// Returns [`TopologyError::WireNotClosed`] if the wire is not closed.
/// Returns [`TopologyError::EdgeNotFound`] if any edge id is invalid.
pub fn validate_wire_closed(wire: &Wire, edges: &Arena<Edge>) -> Result<(), TopologyError> {
    if !wire.is_closed() {
        return Err(TopologyError::WireNotClosed);
    }

    let oriented = wire.edges();
    for window in oriented.windows(2) {
        let current = &window[0];
        let next = &window[1];

        let current_edge = edges
            .get(current.edge())
            .ok_or_else(|| TopologyError::EdgeNotFound(current.edge()))?;
        let next_edge = edges
            .get(next.edge())
            .ok_or_else(|| TopologyError::EdgeNotFound(next.edge()))?;

        let current_end = if current.is_forward() {
            current_edge.end()
        } else {
            current_edge.start()
        };
        let next_start = if next.is_forward() {
            next_edge.start()
        } else {
            next_edge.end()
        };

        if current_end != next_start {
            return Err(TopologyError::WireNotClosed);
        }
    }

    // Check last -> first closure.
    if let (Some(last), Some(first)) = (oriented.last(), oriented.first()) {
        let last_edge = edges
            .get(last.edge())
            .ok_or_else(|| TopologyError::EdgeNotFound(last.edge()))?;
        let first_edge = edges
            .get(first.edge())
            .ok_or_else(|| TopologyError::EdgeNotFound(first.edge()))?;

        let last_end = if last.is_forward() {
            last_edge.end()
        } else {
            last_edge.start()
        };
        let first_start = if first.is_forward() {
            first_edge.start()
        } else {
            first_edge.end()
        };

        if last_end != first_start {
            return Err(TopologyError::WireNotClosed);
        }
    }

    Ok(())
}

/// Validates that a shell is manifold.
///
/// A manifold shell requires that every edge is shared by exactly two faces
/// (for a closed shell) or at most two faces (for an open shell).
///
/// # Errors
///
/// Returns [`TopologyError::NonManifold`] if the shell violates the
/// manifold condition.
#[allow(clippy::unnecessary_wraps, clippy::missing_const_for_fn)]
pub fn validate_shell_manifold(_shell: &Shell) -> Result<(), TopologyError> {
    // TODO: walk the shell faces, collect edge usage counts, and verify
    // each edge is shared by at most two faces.
    Ok(())
}
