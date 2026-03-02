//! Face — a bounded region of a surface.

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::Vec3;

use crate::arena;
use crate::wire::WireId;

/// Typed handle for a [`Face`] stored in an [`Arena`](crate::Arena).
pub type FaceId = arena::Id<Face>;

/// The geometric surface associated with a face.
#[derive(Debug, Clone)]
pub enum FaceSurface {
    /// An infinite plane defined by a normal vector and signed distance from
    /// the origin.
    Plane {
        /// Outward-pointing normal of the plane.
        normal: Vec3,
        /// Signed distance from the origin along the normal.
        d: f64,
    },
    /// A NURBS surface.
    Nurbs(NurbsSurface),
}

/// A topological face: a bounded region of a surface.
///
/// A face has exactly one outer wire (boundary) and zero or more inner
/// wires (holes/voids).
#[derive(Debug, Clone)]
pub struct Face {
    /// The outer boundary wire of this face.
    outer_wire: WireId,
    /// Inner boundary wires representing holes in this face.
    inner_wires: Vec<WireId>,
    /// The geometric surface underlying this face.
    surface: FaceSurface,
}

impl Face {
    /// Creates a new face with the given outer wire, inner wires, and surface.
    #[must_use]
    pub const fn new(outer_wire: WireId, inner_wires: Vec<WireId>, surface: FaceSurface) -> Self {
        Self {
            outer_wire,
            inner_wires,
            surface,
        }
    }

    /// Returns the outer boundary wire of this face.
    #[must_use]
    pub const fn outer_wire(&self) -> WireId {
        self.outer_wire
    }

    /// Returns the inner boundary wires (holes) of this face.
    #[must_use]
    pub fn inner_wires(&self) -> &[WireId] {
        &self.inner_wires
    }

    /// Returns a reference to the surface geometry of this face.
    #[must_use]
    pub const fn surface(&self) -> &FaceSurface {
        &self.surface
    }

    /// Sets the surface geometry of this face.
    pub fn set_surface(&mut self, surface: FaceSurface) {
        self.surface = surface;
    }
}
