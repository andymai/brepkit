//! Operations on compound entities.
//!
//! Provides utilities for working with compounds of solids:
//! extracting individual solids, fusing all solids in a compound,
//! and computing compound-level measurements.

use brepkit_topology::Topology;
use brepkit_topology::compound::CompoundId;
use brepkit_topology::solid::SolidId;

use crate::boolean::{BooleanOp, boolean};

/// Extract all solid IDs from a compound.
///
/// # Errors
///
/// Returns an error if the compound ID is invalid.
pub fn explode(
    topo: &Topology,
    compound: CompoundId,
) -> Result<Vec<SolidId>, crate::OperationsError> {
    let comp = topo.compound(compound)?;
    Ok(comp.solids().to_vec())
}

/// Fuse (union) all solids in a compound into a single solid.
///
/// Performs iterative boolean union on all solids. Requires at least
/// one solid in the compound.
///
/// # Errors
///
/// Returns an error if the compound is empty or a boolean operation fails.
pub fn fuse_all(
    topo: &mut Topology,
    compound: CompoundId,
) -> Result<SolidId, crate::OperationsError> {
    let solids = {
        let comp = topo.compound(compound)?;
        comp.solids().to_vec()
    };

    if solids.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "compound has no solids to fuse".into(),
        });
    }

    let mut result = solids[0];
    for &sid in &solids[1..] {
        result = boolean(topo, BooleanOp::Fuse, result, sid)?;
    }

    Ok(result)
}

/// Count the total number of solids in a compound.
///
/// # Errors
///
/// Returns an error if the compound ID is invalid.
pub fn solid_count(topo: &Topology, compound: CompoundId) -> Result<usize, crate::OperationsError> {
    let comp = topo.compound(compound)?;
    Ok(comp.solids().len())
}

/// Compute the combined bounding box of all solids in a compound.
///
/// # Errors
///
/// Returns an error if the compound is empty or measurement fails.
pub fn compound_bounding_box(
    topo: &Topology,
    compound: CompoundId,
) -> Result<brepkit_math::aabb::Aabb3, crate::OperationsError> {
    let comp = topo.compound(compound)?;
    let solids = comp.solids();

    if solids.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "compound is empty".into(),
        });
    }

    let mut combined = crate::measure::solid_bounding_box(topo, solids[0])?;
    for &sid in &solids[1..] {
        let bb = crate::measure::solid_bounding_box(topo, sid)?;
        combined = combined.union(bb);
    }

    Ok(combined)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::compound::Compound;

    use super::*;

    #[test]
    fn explode_returns_solids() {
        let mut topo = Topology::new();
        let s1 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let cid = topo.compounds.alloc(Compound::new(vec![s1, s2]));

        let solids = explode(&topo, cid).unwrap();
        assert_eq!(solids.len(), 2);
    }

    #[test]
    fn solid_count_works() {
        let mut topo = Topology::new();
        let s1 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let cid = topo.compounds.alloc(Compound::new(vec![s1]));

        assert_eq!(solid_count(&topo, cid).unwrap(), 1);
    }

    #[test]
    fn compound_bbox() {
        let mut topo = Topology::new();
        let s1 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Move s2 to (5, 0, 0).
        crate::transform::transform_solid(
            &mut topo,
            s2,
            &brepkit_math::mat::Mat4::translation(5.0, 0.0, 0.0),
        )
        .unwrap();

        let cid = topo.compounds.alloc(Compound::new(vec![s1, s2]));
        let bb = compound_bounding_box(&topo, cid).unwrap();

        let tol = Tolerance::loose();
        assert!(tol.approx_eq(bb.min.x(), -0.5));
        assert!(tol.approx_eq(bb.max.x(), 5.5));
    }

    #[test]
    fn fuse_all_two_overlapping_boxes() {
        let mut topo = Topology::new();
        let s1 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Offset s2 slightly — overlapping boxes.
        crate::transform::transform_solid(
            &mut topo,
            s2,
            &brepkit_math::mat::Mat4::translation(0.5, 0.0, 0.0),
        )
        .unwrap();

        let cid = topo.compounds.alloc(Compound::new(vec![s1, s2]));
        let fused = fuse_all(&mut topo, cid).unwrap();

        let vol = crate::measure::solid_volume(&topo, fused, 0.1).unwrap();
        // Two overlapping unit cubes: total should be less than 2.0.
        assert!(
            vol > 1.0 && vol < 2.0,
            "fused volume should be between 1 and 2, got {vol}"
        );
    }
}
