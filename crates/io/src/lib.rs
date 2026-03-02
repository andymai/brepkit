//! # brepkit-io
//!
//! Data exchange for brepkit: STEP (ISO 10303) and 3MF import/export.
//!
//! This is layer L2, depending on `brepkit-math` and `brepkit-topology`.

pub mod iges;
pub mod obj;
pub mod step;
pub mod stl;
pub mod threemf;

/// Errors from data exchange operations.
#[derive(Debug, thiserror::Error)]
pub enum IoError {
    /// The input file format is invalid or malformed.
    #[error("parse error: {reason}")]
    ParseError {
        /// Description of the parse failure.
        reason: String,
    },

    /// An unsupported STEP entity was encountered.
    #[error("unsupported STEP entity: {entity}")]
    UnsupportedEntity {
        /// The entity type name.
        entity: String,
    },

    /// The topology is incomplete or inconsistent for export.
    #[error("invalid topology for export: {reason}")]
    InvalidTopology {
        /// Description of the topology issue.
        reason: String,
    },

    /// A topology lookup failed.
    #[error(transparent)]
    Topology(#[from] brepkit_topology::TopologyError),

    /// An I/O error occurred.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// An error from a modeling operation (e.g. tessellation).
    #[error(transparent)]
    Operations(#[from] brepkit_operations::OperationsError),

    /// An error writing the ZIP archive.
    #[error(transparent)]
    Zip(#[from] zip::result::ZipError),
}
