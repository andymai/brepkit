//! Tolerances of B-spline sections, proved off both of their faces.
//!
//! A marched section is a fit: it lies off the two surfaces it is a section
//! of by up to its fit error, along its whole length. Each B-spline section
//! edge whose bound against both faces' surfaces is proved and exceeds the
//! linear tolerance takes that bound as its tolerance, which
//! [`super::link_existing`] reads when it proves the section one curve with
//! a boundary edge. A section with no proved bound keeps no tolerance.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;

use crate::ds::GfaArena;
use crate::error::AlgoError;

/// Gives each B-spline section edge of `arena` the greater of its proved
/// bounds off its two faces' surfaces as its tolerance, where both are
/// proved and the greater exceeds the linear tolerance.
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails.
pub fn perform(topo: &mut Topology, tol: Tolerance, arena: &GfaArena) -> Result<(), AlgoError> {
    let mut proved = Vec::new();
    for section in &arena.curves {
        let EdgeCurve::NurbsCurve(curve) = &section.curve else {
            continue;
        };
        let Some(pb) = section
            .pave_blocks
            .first()
            .and_then(|&id| arena.pave_blocks.get(id))
        else {
            continue;
        };
        let (t0, t1) = section.t_range;
        let span = (t0.min(t1), t0.max(t1));
        let off = |face| -> Result<Option<f64>, AlgoError> {
            Ok(crate::surface_reach::spline_off_face(
                curve,
                span,
                topo.face(face)?.surface(),
                tol.linear,
            ))
        };
        if let (Some(a), Some(b)) = (off(section.face_a)?, off(section.face_b)?) {
            let bound = a.max(b);
            if bound > tol.linear {
                proved.push((pb.original_edge, bound));
            }
        }
    }
    for (edge, bound) in proved {
        let edge = topo.edge_mut(edge)?;
        if bound > edge.tolerance().unwrap_or(0.0) {
            edge.set_tolerance(Some(bound));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::aabb::Aabb3;
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use crate::ds::{GfaArena, IntersectionCurveDS, Pave, PaveBlock};

    /// A cubic along a ruling of a cylinder of radius 2, `off` outside it,
    /// in the plane through the axis that holds it, as a section of the two
    /// faces: its edge takes a tolerance of at least `off` and at most twice
    /// it, and one on the cylinder takes none.
    #[test]
    fn a_b_spline_section_takes_its_proved_distance_from_its_faces() {
        for off in [0.0, 1e-4] {
            let curve = NurbsCurve::new(
                3,
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                (0..7)
                    .map(|k| Point3::new(2.0 + off, 0.0, f64::from(k)))
                    .collect(),
                vec![1.0; 7],
            )
            .unwrap();
            let mut topo = Topology::default();
            let a = topo.add_vertex(Vertex::new(curve.evaluate(0.0), 1e-7));
            let b = topo.add_vertex(Vertex::new(curve.evaluate(4.0), 1e-7));
            let section = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(curve.clone())));
            let wire =
                topo.add_wire(Wire::new(vec![OrientedEdge::new(section, true)], false).unwrap());
            let cylinder =
                CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0)
                    .unwrap();
            let face_a = topo.add_face(Face::new(wire, vec![], FaceSurface::Cylinder(cylinder)));
            let face_b = topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    d: 0.0,
                },
            ));
            let mut arena = GfaArena::new();
            let pb = arena.pave_blocks.alloc(PaveBlock::new(
                section,
                Pave::new(a, 0.0),
                Pave::new(b, 4.0),
            ));
            arena.curves.push(IntersectionCurveDS {
                curve: EdgeCurve::NurbsCurve(curve.clone()),
                face_a,
                face_b,
                bbox: Aabb3::from_points(curve.control_points().iter().copied()),
                pave_blocks: vec![pb],
                t_range: (0.0, 4.0),
            });

            super::perform(&mut topo, Tolerance::default(), &arena).unwrap();

            let held = topo.edge(section).unwrap().tolerance();
            if off > 0.0 {
                assert!(
                    held.is_some_and(|t| t >= off && t <= 2.0 * off + 1e-12),
                    "off {off}: {held:?}"
                );
            } else {
                assert!(held.is_none(), "on the cylinder: {held:?}");
            }
        }
    }
}
