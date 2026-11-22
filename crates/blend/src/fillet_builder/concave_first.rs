//! Rounds that add material and rounds that take it off, meeting at a corner.
//!
//! A concave round (it adds material) and a convex round (it takes material
//! off) that meet at a corner are not closed by a patch over the corner: at
//! a square step they touch each other at one point, with no crease between
//! them. So the concave rounds are built first. Each convex round then runs
//! along what is left of its edge. Where the end of a concave round joins
//! two convex rounds of one radius, tangent to both, the convex round runs
//! on round that end too: a ball rolling along one convex edge rolls round
//! the concave round's end on to the other.
//!
//! Where a concave round takes a convex round's whole edge, or where either
//! build fails or leaves an edge unrounded, the rounds are built in one
//! pass, as with no such corner.

use std::collections::{HashMap, HashSet};

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use super::FilletBuilder;
use crate::builder_utils::into_face_from_edge;
use crate::fillet_plan::{CornerClassification, FilletPlan};
use crate::radius_law::RadiusLaw;
use crate::{BlendError, BlendResult};

/// Below this, the sum of the cosines between each face's way in from the
/// edge and the other face's outward normal reads as a flat edge.
const FLAT: f64 = 1e-6;

/// The picked edges split into the convex rounds, built on what the others
/// leave, and the rest (the concave rounds), built first: each set with its
/// radius.
pub(super) struct Stages {
    concave: Vec<(Vec<EdgeId>, RadiusLaw)>,
    convex: Vec<(Vec<EdgeId>, f64)>,
}

/// A face's outward normal at its surface's point nearest `point`.
fn outward_normal(
    topo: &Topology,
    face: FaceId,
    point: Point3,
) -> Result<Option<Vec3>, BlendError> {
    let face = topo.face(face)?;
    let surface = face.surface();
    let normal = match (surface, surface.project_point(point)) {
        (FaceSurface::Plane { normal, .. }, _) => *normal,
        (_, Some((u, v))) => surface.normal(u, v),
        (_, None) => return Ok(None),
    };
    let Ok(normal) = normal.normalize() else {
        return Ok(None);
    };
    Ok(Some(if face.is_reversed() { -normal } else { normal }))
}

/// Whether the round of `edge` between `a` and `b` adds material: each face
/// leaves the edge toward the other's outward side. `None` where the two
/// faces meet nearly flat, or where a side cannot be read.
fn adds_material(
    topo: &Topology,
    edge: EdgeId,
    a: FaceId,
    b: FaceId,
) -> Result<Option<bool>, BlendError> {
    let (Some(into_a), Some(into_b)) = (
        into_face_from_edge(topo, a, edge)?,
        into_face_from_edge(topo, b, edge)?,
    ) else {
        return Ok(None);
    };
    let data = topo.edge(edge)?;
    let (start, end) = (
        topo.vertex(data.start())?.point(),
        topo.vertex(data.end())?.point(),
    );
    let (t0, t1) = data.curve().domain_with_endpoints(start, end);
    let middle = data
        .curve()
        .evaluate_with_endpoints(f64::midpoint(t0, t1), start, end);
    let (Some(out_a), Some(out_b)) = (
        outward_normal(topo, a, middle)?,
        outward_normal(topo, b, middle)?,
    ) else {
        return Ok(None);
    };
    let turn = into_a.dot(out_b) + into_b.dot(out_a);
    Ok(if turn > FLAT {
        Some(true)
    } else if turn < -FLAT {
        Some(false)
    } else {
        None
    })
}

/// A copy of a radius law that has no closure.
fn copy_law(law: &RadiusLaw) -> Option<RadiusLaw> {
    match *law {
        RadiusLaw::Constant(radius) => Some(RadiusLaw::Constant(radius)),
        RadiusLaw::Linear { start, end } => Some(RadiusLaw::Linear { start, end }),
        RadiusLaw::SCurve { start, end } => Some(RadiusLaw::SCurve { start, end }),
        RadiusLaw::Custom(_) => None,
    }
}

/// Split the picked edges into [`Stages`] where a corner joins a concave
/// round to a convex one. `None` where no corner does, where such a corner
/// has a round that is neither (its edges turn both ways, or read flat),
/// where a convex round varies in radius (its edge is shorter once the
/// concave rounds are built), or where a concave round's law has a closure.
///
/// # Errors
/// A topology lookup failure.
pub(super) fn stages(
    topo: &Topology,
    solid: SolidId,
    edge_sets: &[(Vec<EdgeId>, RadiusLaw)],
) -> Result<Option<Stages>, BlendError> {
    let Ok(plan) = FilletPlan::build(topo, solid, edge_sets) else {
        return Ok(None);
    };
    let adjacency = topo.build_adjacency(solid)?;
    let mut turns: Vec<Option<bool>> = Vec::with_capacity(plan.contours.len());
    for contour in &plan.contours {
        let mut turn = None;
        for (k, &edge) in contour.edges.iter().enumerate() {
            let this = match adjacency.faces_for_edge(edge) {
                &[a, b] => adds_material(topo, edge, a, b)?,
                _ => None,
            };
            if k > 0 && this != turn {
                turn = None;
                break;
            }
            turn = this;
        }
        turns.push(turn);
    }
    let mut mixed = false;
    for junction in &plan.junctions {
        if junction.classification != CornerClassification::Junction {
            continue;
        }
        let at: Vec<Option<bool>> = junction
            .incident_contours
            .iter()
            .map(|&contour| turns[contour])
            .collect();
        if at.contains(&Some(true)) && at.contains(&Some(false)) {
            if at.contains(&None) {
                return Ok(None);
            }
            mixed = true;
        }
    }
    if !mixed {
        return Ok(None);
    }
    let convex: HashSet<EdgeId> = plan
        .contours
        .iter()
        .zip(&turns)
        .filter(|(_, turn)| **turn == Some(false))
        .flat_map(|(contour, _)| contour.edges.iter().copied())
        .collect();
    let mut stages = Stages {
        concave: Vec::new(),
        convex: Vec::new(),
    };
    for (edges, law) in edge_sets {
        let (second, first): (Vec<EdgeId>, Vec<EdgeId>) = edges
            .iter()
            .copied()
            .partition(|edge| convex.contains(edge));
        if !second.is_empty() {
            let RadiusLaw::Constant(radius) = *law else {
                return Ok(None);
            };
            stages.convex.push((second, radius));
        }
        if !first.is_empty() {
            let Some(law) = copy_law(law) else {
                return Ok(None);
            };
            stages.concave.push((first, law));
        }
    }
    Ok(Some(stages))
}

/// Whether `point` lies on `edge` within `tol`, between its ends.
fn on_edge(topo: &Topology, edge: EdgeId, point: Point3, tol: f64) -> Result<bool, BlendError> {
    let data = topo.edge(edge)?;
    let (start, end) = (
        topo.vertex(data.start())?.point(),
        topo.vertex(data.end())?.point(),
    );
    let (t0, t1) = data.curve().domain_with_endpoints(start, end);
    let (low, high) = (t0.min(t1), t0.max(t1));
    let within = |t: f64, period: Option<f64>| {
        let slack = 1e-9 * (1.0 + high.abs());
        match period {
            Some(period) => {
                let t = low + (t - low).rem_euclid(period);
                t <= high + slack || t >= low + period - slack
            }
            None => t >= low - slack && t <= high + slack,
        }
    };
    let period = std::f64::consts::TAU;
    Ok(match data.curve() {
        EdgeCurve::Line => {
            let run = end - start;
            let t = (point - start).dot(run) / run.length_squared().max(1e-300);
            within(t, None) && (start + run * t - point).length() <= tol
        }
        EdgeCurve::Circle(circle) => {
            let t = circle.project(point);
            let on = brepkit_math::traits::ParametricCurve::evaluate(circle, t);
            (on - point).length() <= tol && within(t, Some(period))
        }
        EdgeCurve::Ellipse(ellipse) => {
            let t = ellipse.project(point);
            let on = brepkit_math::traits::ParametricCurve::evaluate(ellipse, t);
            (on - point).length() <= tol && within(t, Some(period))
        }
        EdgeCurve::NurbsCurve(curve) => {
            brepkit_math::nurbs::projection::project_point_to_curve(curve, point, tol)
                .is_ok_and(|found| found.distance <= tol && within(found.parameter, None))
        }
    })
}

/// An edge's two ends, each with its unit tangent there pointing along it.
type Ends = [(Point3, Vec3); 2];

/// An edge's two ends and its unit tangents there, both pointing along it.
fn ends(topo: &Topology, edge: EdgeId) -> Result<Option<Ends>, BlendError> {
    let data = topo.edge(edge)?;
    let (start, end) = (
        topo.vertex(data.start())?.point(),
        topo.vertex(data.end())?.point(),
    );
    let (t0, t1) = data.curve().domain_with_endpoints(start, end);
    let along = |t: f64| {
        let tangent = data.curve().tangent_with_endpoints(t, start, end);
        let tangent = if t1 < t0 { -tangent } else { tangent };
        tangent.normalize().ok()
    };
    let (Some(a), Some(b)) = (along(t0), along(t1)) else {
        return Ok(None);
    };
    Ok(Some([(start, a), (end, b)]))
}

/// Whether `edge` lies along `source` from end to end.
fn image_of(topo: &Topology, edge: EdgeId, source: EdgeId, tol: f64) -> Result<bool, BlendError> {
    let data = topo.edge(edge)?;
    let (start, end) = (
        topo.vertex(data.start())?.point(),
        topo.vertex(data.end())?.point(),
    );
    let (t0, t1) = data.curve().domain_with_endpoints(start, end);
    let middle = data
        .curve()
        .evaluate_with_endpoints(f64::midpoint(t0, t1), start, end);
    for point in [start, middle, end] {
        if !on_edge(topo, source, point, tol)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Build the concave rounds, then the convex rounds on what they leave (see
/// the module notes): `None` where a concave round takes a convex round's
/// whole edge, or where either build fails or leaves an edge unrounded.
///
/// # Errors
/// A topology lookup failure.
#[allow(clippy::too_many_lines)]
pub(super) fn build(
    topo: &mut Topology,
    solid: SolidId,
    stages: Stages,
) -> Result<Option<BlendResult>, BlendError> {
    let Stages { concave, convex } = stages;
    let mut first = FilletBuilder::new(topo, solid);
    for (edges, law) in concave {
        first.add_edges_with_law(&edges, law);
    }
    let Ok(made) = first.build() else {
        return Ok(None);
    };
    if made.solid == solid || !made.failed.is_empty() {
        return Ok(None);
    }

    let before: HashSet<EdgeId> = solid_edges(topo, solid)?.into_iter().collect();
    let after = solid_edges(topo, made.solid)?;
    let kept: HashSet<EdgeId> = after.iter().copied().collect();
    let fresh: Vec<EdgeId> = after
        .iter()
        .copied()
        .filter(|edge| !before.contains(edge))
        .collect();
    let mut scale = 1.0_f64;
    for &edge in &after {
        let data = topo.edge(edge)?;
        for vertex in [data.start(), data.end()] {
            let p = topo.vertex(vertex)?.point();
            scale = scale.max(p.x().abs()).max(p.y().abs()).max(p.z().abs());
        }
    }
    let tol = 1e-7 * scale;

    // Each convex edge's pieces left by the concave rounds.
    let mut source_of: HashMap<EdgeId, EdgeId> = HashMap::new();
    let mut sets: Vec<(Vec<EdgeId>, f64)> = Vec::with_capacity(convex.len());
    for (edges, radius) in &convex {
        let mut images = Vec::new();
        for &edge in edges {
            let found: Vec<EdgeId> = if kept.contains(&edge) {
                vec![edge]
            } else {
                let mut found = Vec::new();
                for &candidate in &fresh {
                    if !source_of.contains_key(&candidate) && image_of(topo, candidate, edge, tol)?
                    {
                        found.push(candidate);
                    }
                }
                found
            };
            if found.is_empty() {
                return Ok(None);
            }
            for image in found {
                source_of.insert(image, edge);
                images.push(image);
            }
        }
        sets.push((images, *radius));
    }

    // A new edge joining two convex rounds of one radius, tangent to both,
    // is the end of a concave round they run on round.
    let mut ends_of: Vec<(EdgeId, usize, Ends)> = Vec::new();
    for (set, (images, _)) in sets.iter().enumerate() {
        for &image in images {
            if let Some(image_ends) = ends(topo, image)? {
                ends_of.push((image, set, image_ends));
            }
        }
    }
    let tangent_to = |point: Point3, along: Vec3| {
        ends_of
            .iter()
            .filter_map(move |(image, set, image_ends)| {
                image_ends
                    .iter()
                    .any(|&(at, tangent)| {
                        (at - point).length() <= tol && along.cross(tangent).length() <= 1e-6
                    })
                    .then_some((*image, *set))
            })
            .collect::<Vec<_>>()
    };
    let mut bridges: Vec<(EdgeId, usize)> = Vec::new();
    for &candidate in &fresh {
        if source_of.contains_key(&candidate) {
            continue;
        }
        let Some([(a, ta), (b, tb)]) = ends(topo, candidate)? else {
            continue;
        };
        let (at_a, at_b) = (tangent_to(a, ta), tangent_to(b, tb));
        let joined = at_a.iter().find_map(|&(image_a, set_a)| {
            at_b.iter().find_map(|&(image_b, set_b)| {
                let (ra, rb) = (sets[set_a].1, sets[set_b].1);
                (image_a != image_b && (ra - rb).abs() <= 1e-9 * ra.max(rb)).then_some(set_a)
            })
        });
        if let Some(set) = joined {
            bridges.push((candidate, set));
        }
    }
    for (bridge, set) in bridges {
        sets[set].0.push(bridge);
    }

    let mut second = FilletBuilder::new(topo, made.solid);
    for (images, radius) in &sets {
        second.add_edges(images, *radius);
    }
    let Ok(rounded) = second.build() else {
        return Ok(None);
    };
    if !rounded.failed.is_empty() || rounded.solid == made.solid {
        return Ok(None);
    }
    let mut succeeded = made.succeeded;
    succeeded.extend(convex.iter().flat_map(|(edges, _)| edges.iter().copied()));
    Ok(Some(BlendResult {
        solid: rounded.solid,
        succeeded,
        failed: Vec::new(),
        is_partial: false,
    }))
}
