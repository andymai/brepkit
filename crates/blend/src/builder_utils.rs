//! Shared utilities for fillet and chamfer builders.
//!
//! Functions used by both [`FilletBuilder`](crate::fillet_builder::FilletBuilder)
//! and [`ChamferBuilder`](crate::chamfer_builder::ChamferBuilder) for creating
//! blend faces and sampling contact curves.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::BlendError;
use crate::stripe::Stripe;

/// Sample the start and end points of a NURBS curve.
#[must_use]
pub fn sample_nurbs_endpoints(curve: &NurbsCurve) -> Vec<Point3> {
    let (t0, t1) = curve.domain();
    vec![curve.evaluate(t0), curve.evaluate(t1)]
}

/// Create a blend face from a stripe's surface and contact curves.
///
/// Builds a minimal quadrilateral wire from the four contact-curve endpoints
/// and associates the blend surface with it.
///
/// # Errors
///
/// Returns [`BlendError`] if wire or face construction fails.
pub fn create_blend_face(topo: &mut Topology, stripe: &Stripe) -> Result<FaceId, BlendError> {
    let (t0_1, t1_1) = stripe.contact1.domain();
    let (t0_2, t1_2) = stripe.contact2.domain();

    // Four corner points of the blend quad.
    let p1_start = stripe.contact1.evaluate(t0_1);
    let p1_end = stripe.contact1.evaluate(t1_1);
    let p2_start = stripe.contact2.evaluate(t0_2);
    let p2_end = stripe.contact2.evaluate(t1_2);

    // Create vertices (snapshot then allocate).
    let v1s = topo.add_vertex(Vertex::new(p1_start, 1e-7));
    let v1e = topo.add_vertex(Vertex::new(p1_end, 1e-7));
    let v2s = topo.add_vertex(Vertex::new(p2_start, 1e-7));
    let v2e = topo.add_vertex(Vertex::new(p2_end, 1e-7));

    // Build quad: p1_start -> p1_end -> p2_end -> p2_start -> p1_start.
    // Use actual contact curves for e0 and e2 (the longitudinal edges along
    // the spine direction). Cross edges e1 and e3 are straight lines connecting
    // the two contact curves at the spine endpoints.
    let e0 = topo.add_edge(Edge::new(
        v1s,
        v1e,
        EdgeCurve::NurbsCurve(stripe.contact1.clone()),
    ));
    let e1 = topo.add_edge(Edge::new(v1e, v2e, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(
        v2e,
        v2s,
        EdgeCurve::NurbsCurve(stripe.contact2.clone()),
    ));
    let e3 = topo.add_edge(Edge::new(v2s, v1s, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e0, true),
            OrientedEdge::new(e1, true),
            OrientedEdge::new(e2, true),
            OrientedEdge::new(e3, true),
        ],
        true,
    )?;
    let wire_id = topo.add_wire(wire);

    let face = Face::new(wire_id, Vec::new(), stripe.surface.clone());
    let face_id = topo.add_face(face);

    Ok(face_id)
}

/// Adapter that provides [`ParametricSurface`] for a `FaceSurface::Plane`.
///
/// Planes store only a normal and signed distance `d`, with no parametric
/// frame.  This adapter builds an orthonormal UV frame from the normal so
/// that the walking engine can evaluate, project, and differentiate the
/// plane surface uniformly.
pub struct PlaneAdapter {
    /// Origin point on the plane (the point closest to the world origin).
    pub origin: Point3,
    /// U-direction tangent (unit vector in the plane).
    pub u_dir: Vec3,
    /// V-direction tangent (unit vector in the plane, orthogonal to `u_dir`).
    pub v_dir: Vec3,
    /// Outward-facing unit normal.
    pub norm: Vec3,
}

impl PlaneAdapter {
    /// Build a `PlaneAdapter` from a plane normal and signed distance.
    ///
    /// The UV frame is constructed by choosing a non-parallel reference vector
    /// and computing the cross products.
    #[must_use]
    pub fn from_normal_and_d(normal: Vec3, d: f64) -> Self {
        let origin = Point3::new(normal.x() * d, normal.y() * d, normal.z() * d);

        // Pick a reference vector that is not parallel to the normal.
        let ref_vec = if normal.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };

        let u_dir = normal
            .cross(ref_vec)
            .normalize()
            .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
        let v_dir = normal
            .cross(u_dir)
            .normalize()
            .unwrap_or(Vec3::new(0.0, 1.0, 0.0));

        Self {
            origin,
            u_dir,
            v_dir,
            norm: normal,
        }
    }
}

impl ParametricSurface for PlaneAdapter {
    fn evaluate(&self, u: f64, v: f64) -> Point3 {
        self.origin + self.u_dir * u + self.v_dir * v
    }

    fn normal(&self, _u: f64, _v: f64) -> Vec3 {
        self.norm
    }

    fn project_point(&self, point: Point3) -> (f64, f64) {
        let d = point - self.origin;
        (d.dot(self.u_dir), d.dot(self.v_dir))
    }

    fn partial_u(&self, _u: f64, _v: f64) -> Vec3 {
        self.u_dir
    }

    fn partial_v(&self, _u: f64, _v: f64) -> Vec3 {
        self.v_dir
    }
}

/// Extract a `&dyn ParametricSurface` from a `FaceSurface`, or build a
/// `PlaneAdapter` for plane faces.
///
/// Returns `Ok(adapter)` for planes and `Err(face_id)` for unsupported types.
/// For analytic and NURBS surfaces that already implement `ParametricSurface`,
/// the reference is extracted directly and the adapter is unused.
///
/// # Usage pattern
///
/// ```ignore
/// let mut adapter = None;
/// let surf: &dyn ParametricSurface = surface_ref_or_adapter(&face_surface, &mut adapter);
/// ```
#[must_use]
pub fn surface_ref_or_adapter<'a>(
    surface: &'a FaceSurface,
    adapter_slot: &'a mut Option<PlaneAdapter>,
) -> &'a dyn ParametricSurface {
    // For Plane faces, we need to populate the adapter_slot first,
    // then return a reference to it. For all other variants, we can
    // return a reference directly to the surface inside FaceSurface.
    if let FaceSurface::Plane { normal, d } = surface {
        let adapter = adapter_slot.insert(PlaneAdapter::from_normal_and_d(*normal, *d));
        return adapter as &dyn ParametricSurface;
    }
    match surface {
        FaceSurface::Plane { .. } => {
            // Already handled above; this arm is unreachable.
            adapter_slot.insert(PlaneAdapter::from_normal_and_d(
                Vec3::new(0.0, 0.0, 1.0),
                0.0,
            )) as &dyn ParametricSurface
        }
        FaceSurface::Cylinder(c) => c as &dyn ParametricSurface,
        FaceSurface::Cone(c) => c as &dyn ParametricSurface,
        FaceSurface::Sphere(s) => s as &dyn ParametricSurface,
        FaceSurface::Torus(t) => t as &dyn ParametricSurface,
        FaceSurface::Nurbs(n) => n as &dyn ParametricSurface,
    }
}
