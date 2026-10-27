//! A solid's cavities kept through an operation that rebuilds only its
//! outer shell.

use brepkit_math::aabb::Aabb3;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// A box holding everything within `reach` of `edge`: the edge's own box (a
/// line's ends, the cube about a circle or ellipse, a NURBS curve's control
/// points, which hold it while every weight is positive) grown by `reach`.
pub fn edge_reach(topo: &Topology, edge: EdgeId, reach: f64) -> Result<Aabb3, OperationsError> {
    let data = topo.edge(edge)?;
    let cube = |center: Point3, half: f64| {
        let d = Vec3::new(half, half, half);
        vec![center - d, center + d]
    };
    let points = match data.curve() {
        EdgeCurve::Line => vec![
            topo.vertex(data.start())?.point(),
            topo.vertex(data.end())?.point(),
        ],
        EdgeCurve::Circle(c) => cube(c.center(), c.radius()),
        EdgeCurve::Ellipse(e) => cube(e.center(), e.semi_major().max(e.semi_minor())),
        EdgeCurve::NurbsCurve(n) if n.weights().iter().all(|&w| w > 0.0) => {
            n.control_points().to_vec()
        }
        EdgeCurve::NurbsCurve(_) => cube(Point3::new(0.0, 0.0, 0.0), f64::INFINITY),
    };
    Ok(Aabb3::from_points(points).expanded(reach))
}

/// `result`, rebuilt from `source`'s outer shell alone, given `source`'s
/// cavities, which carry over unchanged. A cavity whose box meets one of
/// `removed` (boxes holding the material the operation took off) could be
/// cut into, and refuses the operation, named `what` in the error.
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if a cavity comes within
/// `removed`, or a topology error if a lookup fails.
pub fn keep_cavities(
    topo: &mut Topology,
    source: SolidId,
    result: SolidId,
    removed: &[Aabb3],
    what: &str,
) -> Result<SolidId, OperationsError> {
    let cavities = topo.solid(source)?.inner_shells().to_vec();
    for &cavity in &cavities {
        let faces = topo.shell(cavity)?.faces().to_vec();
        let bounds = crate::measure::face_set_bounding_box(topo, &faces)?;
        if removed.iter().any(|r| r.intersects(bounds)) {
            return Err(OperationsError::InvalidInput {
                reason: format!("the {what} would cut into a cavity of the solid"),
            });
        }
    }
    let solid = topo.solid_mut(result)?;
    for cavity in cavities {
        solid.add_inner_shell(cavity);
    }
    Ok(result)
}
