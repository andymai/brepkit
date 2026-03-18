//! GFA data structures — transient state for boolean intersection.
//!
//! These types are created during PaveFiller execution and consumed
//! by the Builder. They do NOT modify `Topology` — they reference
//! topology entities by ID and own their own arena for pave blocks.

mod arena;
mod curve;
mod face_info;
mod interference;
mod pave;
mod shape_index;

pub use arena::GfaArena;
pub use curve::IntersectionCurveDS;
pub use face_info::FaceInfo;
pub use interference::{Interference, InterferenceTable};
pub use pave::{CommonBlock, CommonBlockId, Pave, PaveBlock, PaveBlockId};
pub use shape_index::{Rank, ShapeRef};
