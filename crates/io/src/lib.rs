//! # brepkit-io
//!
//! Data exchange: STEP, IGES, 3MF, STL, OBJ, PLY, and glTF import and export.
//! Layer L3, depending on `brepkit-math`, `brepkit-topology`, and
//! `brepkit-operations`.
//!
//! # Format support
//!
//! | Format | Type | Import | Export | Module |
//! |--------|------|--------|--------|--------|
//! | STEP | B-Rep | yes | yes | [`step`] |
//! | STL | Mesh | yes | yes | [`stl`] |
//! | 3MF | Mesh | yes | yes | [`threemf`] |
//! | OBJ | Mesh | yes | yes | [`obj`] |
//! | PLY | Mesh | yes | yes | [`ply`] |
//! | glTF (`.glb`) | Mesh | yes | yes | [`gltf`] |
//! | IGES | B-Rep | preview | lossy | [`iges`] |
//!
//! # STEP preserves exact geometry
//!
//! Analytic surfaces (plane, cylinder, cone, sphere, torus) are written as
//! native STEP surface entities rather than tessellated, and read back as the
//! same surface types. NURBS surfaces are preserved, as are line, circle,
//! ellipse, and NURBS edges. No surface is tessellated on the way out.
//!
//! Exact refers to the geometry, not to the bits. The writer emits 15
//! significant digits, flushes magnitudes below `1e-15` to zero, and merges
//! knots that agree to within `1e-10`. A round-tripped solid is the same
//! shape, not the same floating-point values.
//!
//! One limitation bounds that guarantee: the writer serializes a solid's
//! outer shell only. A solid carrying inner shells, which are the cavity
//! walls left by a hollowing operation or a boolean cut that opened a void,
//! loses those voids on export and reads back solid. Check
//! `Solid::inner_shells` before treating a round trip as lossless.
//!
//! Mesh formats export tessellated triangles, which is a one-way trip for
//! exact geometry. The `read_*_solid` helpers (`read_stl_solid`,
//! `read_obj_solid`, `read_ply_solid`, `read_threemf_solid`,
//! `read_glb_solid`) do rebuild a B-Rep solid from the triangles, but every
//! face comes back planar. A cylinder exported to STL returns as a fan of
//! flat facets, not as a cylinder.
//!
//! IGES is experimental. Export skips analytic surfaces and approximates
//! circular and elliptical edges as polylines; import reconstructs planar
//! placeholder faces only. Use STEP for B-Rep exchange.
//!
//! # Signatures
//!
//! Readers take the input first and the topology second, and B-Rep readers
//! return every solid in the file. Writers take the topology and the solids to
//! write; text formats return a `String`, binary formats a `Vec<u8>`.
//!
//! # Example
//!
//! ```
//! use brepkit_io::step::{read_step, write_step};
//! use brepkit_operations::primitives::make_cylinder;
//! use brepkit_topology::Topology;
//!
//! let mut topo = Topology::new();
//! let cylinder = make_cylinder(&mut topo, 5.0, 20.0)?;
//!
//! let step = write_step(&topo, &[cylinder])?;
//!
//! let mut reloaded = Topology::new();
//! let solids = read_step(&step, &mut reloaded)?;
//! assert_eq!(solids.len(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Importing real files
//!
//! A STEP file that another system wrote is not guaranteed to be a model this
//! one can operate on. They routinely arrive with wires out of order, gaps
//! between adjacent faces, edges whose 3D curve disagrees with their vertices,
//! and inconsistently oriented shells. Every one of those imports without
//! error and then fails a boolean three operations later, far from the cause.
//!
//! Heal on the way in, not when something breaks:
//!
//! ```
//! use brepkit_io::step::{read_step, write_step};
//! use brepkit_operations::heal::heal_solid;
//! use brepkit_operations::primitives::make_box;
//! use brepkit_operations::validate::validate_solid;
//! use brepkit_topology::Topology;
//!
//! # let mut source = Topology::new();
//! # let part = make_box(&mut source, 10.0, 10.0, 10.0)?;
//! # let step = write_step(&source, &[part])?;
//! let mut topo = Topology::new();
//! let solids = read_step(&step, &mut topo)?;
//!
//! for solid in solids {
//!     heal_solid(&mut topo, solid, 1e-6)?;
//!     assert!(validate_solid(&topo, solid)?.is_valid());
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The healing tolerance is the gap width to close, not the kernel's linear
//! tolerance. Pass something near the precision the file was written at:
//! `1e-6` suits a STEP file from a CAD system, while a mesh format converted
//! to B-Rep may need `1e-3` or looser. Too tight leaves the gaps; too loose
//! merges features that were meant to be distinct.
//!
//! # Exporting
//!
//! Mesh formats take a deflection, the maximum distance a triangle may sit
//! from the true surface. It is a direct trade against file size, and the
//! right value depends on the destination: screen rendering tolerates far
//! coarser output than 3D printing.
//!
//! STEP takes no deflection, because nothing is approximated.
//!
//! # See also
//!
//! - [`brepkit_operations`](https://docs.rs/brepkit-operations): modeling,
//!   healing, tessellation, and validation.
//! - [brepjs.dev](https://brepjs.dev/tasks/import-export): the same formats
//!   from the TypeScript side.

pub mod arena_io;
pub mod gltf;
pub mod iges;
pub mod obj;
pub mod ply;
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
