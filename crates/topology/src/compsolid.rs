//! `CompSolid` — a set of solids sharing faces.
//!
//! A `CompSolid` represents multiple solids that share boundary faces
//! (e.g., two volumes separated by a common wall). This is the 8th
//! topology type in OCCT, between Solid and Compound.

use crate::arena;
use crate::face::FaceId;
use crate::solid::SolidId;

/// Typed handle for a [`CompSolid`] stored in an [`Arena`](crate::Arena).
pub type CompSolidId = arena::Id<CompSolid>;

/// A topological `CompSolid`: a set of solids sharing faces.
///
/// Unlike a [`Compound`](crate::compound::Compound), which is a loose
/// collection, a `CompSolid` implies topological connectivity through
/// shared boundary faces.
#[derive(Debug, Clone)]
pub struct CompSolid {
    /// The solids in this comp-solid.
    solids: Vec<SolidId>,
    /// Faces shared between adjacent solids.
    shared_faces: Vec<FaceId>,
}

impl CompSolid {
    /// Creates a new comp-solid from the given solids and shared faces.
    #[must_use]
    pub const fn new(solids: Vec<SolidId>, shared_faces: Vec<FaceId>) -> Self {
        Self {
            solids,
            shared_faces,
        }
    }

    /// Returns the solids in this comp-solid.
    #[must_use]
    pub fn solids(&self) -> &[SolidId] {
        &self.solids
    }

    /// Returns the faces shared between solids.
    #[must_use]
    pub fn shared_faces(&self) -> &[FaceId] {
        &self.shared_faces
    }

    /// Returns the number of solids.
    #[must_use]
    pub fn num_solids(&self) -> usize {
        self.solids().len()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use brepkit_math::vec::Point3;

    use crate::topology::Topology;
    use crate::vertex::Vertex;

    use super::*;

    #[test]
    fn create_empty_compsolid() {
        let cs = CompSolid::new(vec![], vec![]);
        assert_eq!(cs.num_solids(), 0);
        assert!(cs.shared_faces().is_empty());
    }

    #[test]
    fn compsolid_in_arena() {
        let mut topo = Topology::new();

        // Create two minimal solids (just for ID purposes)
        let _v = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));

        let cs = CompSolid::new(vec![], vec![]);
        let cs_id = topo.add_compsolid(cs);

        let retrieved = topo.compsolid(cs_id).unwrap();
        assert_eq!(retrieved.num_solids(), 0);
    }
}
