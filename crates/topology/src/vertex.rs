//! Vertex — a point in 3D space with an associated tolerance.

use brepkit_math::vec::Point3;

use crate::arena;

/// Typed handle for a [`Vertex`] stored in an [`Arena`](crate::Arena).
pub type VertexId = arena::Id<Vertex>;

/// A topological vertex: a point location with a tolerance ball.
///
/// Two geometric points that fall within `tolerance` of each other
/// are considered the same vertex.
#[derive(Debug, Clone)]
pub struct Vertex {
    /// Position of the vertex in model space.
    point: Point3,
    /// Radius of the tolerance ball around the vertex point.
    tolerance: f64,
}

impl Vertex {
    /// Creates a new vertex at the given point with the specified tolerance.
    #[must_use]
    pub const fn new(point: Point3, tolerance: f64) -> Self {
        Self { point, tolerance }
    }

    /// Returns the position of this vertex.
    #[must_use]
    pub const fn point(&self) -> Point3 {
        self.point
    }

    /// Returns the tolerance of this vertex.
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Sets the position of this vertex.
    pub const fn set_point(&mut self, point: Point3) {
        self.point = point;
    }
}
