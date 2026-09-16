//! Reconstruct and jointly rebuild an isolated, accepted cylindrical fillet.
//!
//! The ordinary polygon trimmer cannot trim a cylindrical face touched only
//! at a vertex: passing that face through leaves a missing transition. For a
//! convex quarter-cylinder between perpendicular planar supports, bounded by
//! two perpendicular planar caps, the sharp topology is uniquely recoverable.
//! This module recognizes that geometry (not operation history), extends only
//! collinear straight edges, and rebuilds the inherited and requested blends
//! together. It never sews unrelated boundaries or changes tolerances/radii.
//!
//! N340 additionally recognizes the exact N339 two-strip setback surface when
//! its retained sharp edge is subsequently selected at a smaller radius. That
//! edge starts at the setback surface's documented singular endpoint. The old
//! corner remains immutable while an exact, monotone rational runout grows from
//! that point into the new constant-radius strip. Unrelated cylinders, arbitrary
//! NURBS corners, inner shells, and non-isolated strips do not enter either route;
//! no partial reconstruction is a successful result.

use std::collections::HashMap;

use brepkit_math::det_hash::DetHashMap;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::adjacency::AdjacencyIndex;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::OperationsError;

struct Strip {
    face: FaceId,
    axial: [EdgeId; 2],
    ends: [([VertexId; 2], Point3); 2],
    /// The strip's own (already-built) radius. N341: this need not equal the
    /// newly requested radius any more -- `recognize` only checks internal
    /// consistency (cylinder radius == both end-arc radii), not equality to
    /// the caller's `radius`. The corner patch is then built from this exact
    /// radius on the inherited side and the caller's `radius` on the requested
    /// side, never a rescale of one template.
    radius: f64,
}

/// Return `None` unless exactly one isolated equal-radius strip adjoins the
/// single requested sharp edge. Allocation starts only after recognition.
pub(super) fn try_adjoining_strip(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
) -> Result<Option<SolidId>, OperationsError> {
    let [target] = edges else { return Ok(None) };
    if !radius.is_finite() || !topo.solid(solid)?.inner_shells().is_empty() {
        return Ok(None);
    }
    let adjacency = topo.build_adjacency(solid)?;
    if !adjacency.is_manifold() || adjacency.faces_for_edge(*target).len() != 2 {
        return Ok(None);
    }

    // The retained edge of N339's exact two-strip patch begins at that patch's
    // singular endpoint. A smaller third radius needs a runout there; treating
    // it as an ordinary full-radius strip leaves five unmatched boundaries.
    if let Some(runout) = MixedRadiusRunout::recognize(topo, solid, &adjacency, *target, radius)? {
        let result =
            super::rolling_ball::build(topo, solid, &[*target], radius, None, None, Some(&runout))?;
        qualify_adjoining_result(topo, result, "N340")?;
        return Ok(Some(result));
    }

    let faces = topo
        .shell(topo.solid(solid)?.outer_shell())?
        .faces()
        .to_vec();
    let mut strips = Vec::new();
    for &face in &faces {
        if let Some(strip) = recognize(topo, &adjacency, face, *target, radius)? {
            strips.push(strip);
        }
    }
    let [strip] = strips.as_slice() else {
        return Ok(None);
    };
    if !can_extend(topo, solid, strip)? {
        return Ok(None);
    }
    let inherited_radius = strip.radius;
    let (recovered, inherited, requested) = reconstruct(topo, solid, strip, *target)?;
    if !crate::validate::validate_solid(topo, recovered)?.is_valid() {
        return Err(OperationsError::NonManifoldResult);
    }
    // Rebuild the inherited strip with a G1-constrained rational setback
    // corner. A positional Coons patch closes the shell but leaves creases.
    //
    // N341: `inherited_radius` (the strip's own, already-accepted radius) and
    // `radius` (the newly requested radius on the adjacent edge) need not be
    // equal any more. `SetbackCorner` builds the exact two-radius patch
    // (`setback_patch::surface_mixed`) rather than a rescale of one template,
    // and `rolling_ball::build` is given each edge's own radius via
    // `edge_radii` rather than one radius applied to both.
    let corner = SetbackCorner::new(
        topo,
        recovered,
        inherited,
        requested,
        inherited_radius,
        radius,
    )?;
    let mut edge_radii = DetHashMap::default();
    edge_radii.insert(inherited.index(), inherited_radius);
    edge_radii.insert(requested.index(), radius);
    let result = super::rolling_ball::build(
        topo,
        recovered,
        &[inherited, requested],
        radius,
        Some(&edge_radii),
        Some(&corner),
        None,
    )?;
    qualify_adjoining_result(topo, result, "N339/N341")?;
    Ok(Some(result))
}

fn qualify_adjoining_result(
    topo: &Topology,
    result: SolidId,
    task: &str,
) -> Result<(), OperationsError> {
    let validation = crate::validate::validate_solid(topo, result)?;
    if !validation.is_valid() {
        let adjacency = topo.build_adjacency(result)?;
        for &id in adjacency.boundary_edges() {
            let edge = topo.edge(id)?;
            log::debug!(
                "{task} free edge {id:?}: {:?} -> {:?}, curve={:?}, faces={:?}",
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
                edge.curve(),
                adjacency.faces_for_edge(id)
            );
        }
        return Err(OperationsError::Blend(
            brepkit_blend::BlendError::PlanningFailure {
                reason: format!(
                    "adjoining fillet output failed validation: {:?}",
                    validation.issues
                ),
            },
        ));
    }
    let volume = crate::measure::solid_volume(topo, result, 0.01)?;
    if !volume.is_finite() || volume <= 0.0 {
        return Err(OperationsError::NonManifoldResult);
    }
    Ok(())
}

#[derive(Clone)]
struct CarriedCurve {
    start: Point3,
    end: Point3,
    curve: EdgeCurve,
}

/// Exact endpoint runout for a smaller third radius at N339's singular point.
///
/// This object is created only after matching N339's complete rational patch,
/// its two equal-radius cylinders, both planar supports, and the retained sharp
/// edge. The runout grows over two requested radii and is constant-radius after
/// that station. It never infers support from operation history or entity ids.
pub(super) struct MixedRadiusRunout {
    pub vertex: usize,
    pub target: EdgeId,
    origin: Point3,
    along: Vec3,
    first: Vec3,
    second: Vec3,
    supports: [FaceId; 2],
    radius: f64,
    carried_curves: Vec<CarriedCurve>,
}

impl MixedRadiusRunout {
    fn recognize(
        topo: &Topology,
        solid: SolidId,
        adjacency: &AdjacencyIndex,
        target: EdgeId,
        radius: f64,
    ) -> Result<Option<Self>, OperationsError> {
        let tol = Tolerance::new();
        let target_edge = topo.edge(target)?;
        if !matches!(target_edge.curve(), EdgeCurve::Line) || radius <= tol.linear {
            return Ok(None);
        }
        let target_faces = adjacency.faces_for_edge(target);
        let [support_a, support_b] = target_faces else {
            return Ok(None);
        };
        let Some(normal_a) = topo.face(*support_a)?.effective_plane_normal() else {
            return Ok(None);
        };
        let Some(normal_b) = topo.face(*support_b)?.effective_plane_normal() else {
            return Ok(None);
        };
        if !topo.face(*support_a)?.inner_wires().is_empty()
            || !topo.face(*support_b)?.inner_wires().is_empty()
            || normal_a.dot(normal_b).abs() > tol.angular
        {
            return Ok(None);
        }

        for vertex in [target_edge.start(), target_edge.end()] {
            let far = if vertex == target_edge.start() {
                target_edge.end()
            } else {
                target_edge.start()
            };
            let origin = topo.vertex(vertex)?.point();
            let span = topo.vertex(far)?.point() - origin;
            let Ok(along) = span.normalize() else {
                continue;
            };
            if along.dot(normal_a).abs() > tol.angular || along.dot(normal_b).abs() > tol.angular {
                continue;
            }

            let incident: Vec<_> = solid_edges(topo, solid)?
                .into_iter()
                .filter(|&id| {
                    id != target
                        && topo
                            .edge(id)
                            .is_ok_and(|edge| edge.start() == vertex || edge.end() == vertex)
                })
                .collect();
            let [boundary_a, boundary_b] = incident.as_slice() else {
                continue;
            };
            if !matches!(topo.edge(*boundary_a)?.curve(), EdgeCurve::NurbsCurve(_))
                || !matches!(topo.edge(*boundary_b)?.curve(), EdgeCurve::NurbsCurve(_))
            {
                continue;
            }

            let common: Vec<_> = adjacency
                .faces_for_edge(*boundary_a)
                .iter()
                .filter(|face| adjacency.faces_for_edge(*boundary_b).contains(face))
                .copied()
                .collect();
            let [patch_id] = common.as_slice() else {
                continue;
            };
            if target_faces.contains(patch_id) {
                continue;
            }
            let patch = topo.face(*patch_id)?;
            let FaceSurface::Nurbs(patch_surface) = patch.surface() else {
                continue;
            };
            if !patch.inner_wires().is_empty()
                || patch_surface.degree_u() != 5
                || patch_surface.degree_v() != 5
                || patch_surface.control_points().len() != 6
                || patch_surface
                    .control_points()
                    .iter()
                    .any(|row| row.len() != 6)
            {
                continue;
            }
            let patch_edges = topo.wire(patch.outer_wire())?.edges();
            if patch_edges.len() != 4 {
                continue;
            }
            let circles: Vec<_> = patch_edges
                .iter()
                .map(brepkit_topology::wire::OrientedEdge::edge)
                .filter(|id| matches!(topo.edge(*id).map(Edge::curve), Ok(EdgeCurve::Circle(_))))
                .collect();
            let [circle_a, circle_b] = circles.as_slice() else {
                continue;
            };
            let (EdgeCurve::Circle(circle_geometry_a), EdgeCurve::Circle(circle_geometry_b)) =
                (topo.edge(*circle_a)?.curve(), topo.edge(*circle_b)?.curve())
            else {
                continue;
            };
            let inherited_radius = circle_geometry_a.radius();
            if (circle_geometry_b.radius() - inherited_radius).abs() > tol.linear
                || radius >= inherited_radius - tol.linear
            {
                continue;
            }

            let mut cylinder_faces = Vec::new();
            for circle in [*circle_a, *circle_b] {
                let Some(face_id) = other_face(adjacency, circle, *patch_id) else {
                    cylinder_faces.clear();
                    break;
                };
                let FaceSurface::Cylinder(cylinder) = topo.face(face_id)?.surface() else {
                    cylinder_faces.clear();
                    break;
                };
                if (cylinder.radius() - inherited_radius).abs() > tol.linear {
                    cylinder_faces.clear();
                    break;
                }
                cylinder_faces.push((circle, face_id, cylinder.axis()));
            }
            if cylinder_faces.len() != 2 || cylinder_faces[0].1 == cylinder_faces[1].1 {
                continue;
            }

            let mut boundaries = Vec::new();
            for boundary in [*boundary_a, *boundary_b] {
                let faces = adjacency.faces_for_edge(boundary);
                let Some(&support) = faces.iter().find(|face| target_faces.contains(face)) else {
                    boundaries.clear();
                    break;
                };
                if other_face(adjacency, boundary, support) != Some(*patch_id) {
                    boundaries.clear();
                    break;
                }
                let edge = topo.edge(boundary)?;
                let end_vertex = if edge.start() == vertex {
                    edge.end()
                } else if edge.end() == vertex {
                    edge.start()
                } else {
                    boundaries.clear();
                    break;
                };
                let end = topo.vertex(end_vertex)?.point();
                let sharp = origin - along * (2.0 * inherited_radius);
                let Ok(axis) = ((end - sharp) * inherited_radius.recip() - along).normalize()
                else {
                    boundaries.clear();
                    break;
                };
                let Some(support_normal) = topo.face(support)?.effective_plane_normal() else {
                    boundaries.clear();
                    break;
                };
                let other_support = if support == *support_a {
                    *support_b
                } else {
                    *support_a
                };
                let Some(other_normal) = topo.face(other_support)?.effective_plane_normal() else {
                    boundaries.clear();
                    break;
                };
                if axis.dot(along).abs() > tol.angular
                    || axis.dot(support_normal).abs() > tol.angular
                    || axis.dot(-other_normal) < 1.0 - tol.angular
                    || ((end - origin) - (axis - along) * inherited_radius).length()
                        > tol.linear * 100.0
                {
                    boundaries.clear();
                    break;
                }
                boundaries.push((boundary, support, end, axis));
            }
            if boundaries.len() != 2 || boundaries[0].1 == boundaries[1].1 {
                continue;
            }
            if boundaries[0].3.dot(boundaries[1].3).abs() > tol.angular {
                continue;
            }

            // Each planar boundary's far point lies on the corresponding
            // inherited cylinder end. This ties the support frame to both exact
            // radius strips, rather than accepting an arbitrary degree-5 patch.
            let mut axes_match_cylinders = true;
            for (_, _, end, axis) in &boundaries {
                let mut matching = 0;
                for (circle, _, cylinder_axis) in &cylinder_faces {
                    let edge = topo.edge(*circle)?;
                    let p = topo.vertex(edge.start())?.point();
                    let q = topo.vertex(edge.end())?.point();
                    if ((*end - p).length() < tol.linear || (*end - q).length() < tol.linear)
                        && cylinder_axis.cross(*axis).length() < tol.angular
                    {
                        matching += 1;
                    }
                }
                if matching != 1 {
                    axes_match_cylinders = false;
                    break;
                }
            }
            if !axes_match_cylinders {
                continue;
            }

            let sharp = origin - along * (2.0 * inherited_radius);
            if !matches_n339_patch(
                patch_surface,
                sharp,
                boundaries[0].3,
                along,
                boundaries[1].3,
                inherited_radius,
            )? {
                continue;
            }
            if span.length() <= 2.0 * radius + tol.linear {
                return Err(OperationsError::InvalidInput {
                    reason: "mixed-radius adjoining fillet runout requires more than twice the requested radius of target-edge clearance".into(),
                });
            }

            let mut carried_curves = Vec::with_capacity(2);
            for (edge_id, _, end, _) in &boundaries {
                carried_curves.push(CarriedCurve {
                    start: origin,
                    end: *end,
                    curve: topo.edge(*edge_id)?.curve().clone(),
                });
            }
            return Ok(Some(Self {
                vertex: vertex.index(),
                target,
                origin,
                along,
                first: boundaries[0].3,
                second: boundaries[1].3,
                supports: [boundaries[0].1, boundaries[1].1],
                radius,
                carried_curves,
            }));
        }
        Ok(None)
    }

    pub(super) fn setback(&self) -> f64 {
        2.0 * self.radius
    }

    pub(super) fn preserved(&self) -> Point3 {
        self.origin
    }

    fn point(&self, local: Point3) -> Point3 {
        self.origin
            + (self.first * local.x() + self.along * local.y() + self.second * local.z())
                * self.radius
    }

    pub(super) fn contacts(&self) -> [(FaceId, Point3); 2] {
        [
            (self.supports[0], self.point(Point3::new(1.0, 2.0, 0.0))),
            (self.supports[1], self.point(Point3::new(0.0, 2.0, 1.0))),
        ]
    }

    pub(super) fn face_spec(&self) -> Result<crate::boolean::FaceSpec, OperationsError> {
        let local = super::setback_patch::mixed_runout_surface()?;
        let points = local
            .control_points()
            .iter()
            .map(|row| row.iter().map(|&point| self.point(point)).collect())
            .collect();
        let surface = brepkit_math::nurbs::surface::NurbsSurface::new(
            local.degree_u(),
            local.degree_v(),
            local.knots_u().to_vec(),
            local.knots_v().to_vec(),
            points,
            local.weights().to_vec(),
        )?;
        // Natural boundary order for `du x dv`: singular point, side contact,
        // bottom contact. Reverse both topology and surface sense if the local
        // frame is left-handed.
        let mut vertices = vec![
            self.origin,
            self.point(Point3::new(0.0, 2.0, 1.0)),
            self.point(Point3::new(1.0, 2.0, 0.0)),
        ];
        let reversed = self.first.cross(self.along).dot(self.second) < 0.0;
        if reversed {
            vertices.reverse();
        }
        Ok(crate::boolean::FaceSpec::Surface {
            vertices,
            surface: FaceSurface::Nurbs(surface),
            reversed,
            inner_wires: vec![],
        })
    }

    /// Restore both retained N339 cubic boundaries and install the exact two
    /// cubic runout/support contacts after mixed-surface assembly.
    pub(super) fn install_contact_curves(
        &self,
        topo: &mut Topology,
        solid: SolidId,
    ) -> Result<(), OperationsError> {
        let mut curves = self.carried_curves.clone();
        for controls in super::setback_patch::mixed_runout_planar_controls() {
            let points = controls.map(|point| self.point(point));
            let knots = [0.0; 4].into_iter().chain([1.0; 4]).collect();
            let curve = NurbsCurve::new(3, knots, points.to_vec(), vec![1.0; 4])?;
            curves.push(CarriedCurve {
                start: points[0],
                end: points[3],
                curve: EdgeCurve::NurbsCurve(curve),
            });
        }
        let tolerance = Tolerance::new().linear;
        for carried in curves {
            let matches: Vec<_> = solid_edges(topo, solid)?
                .into_iter()
                .filter(|&id| {
                    let Ok(edge) = topo.edge(id) else {
                        return false;
                    };
                    let Ok(start) = topo.vertex(edge.start()).map(Vertex::point) else {
                        return false;
                    };
                    let Ok(end) = topo.vertex(edge.end()).map(Vertex::point) else {
                        return false;
                    };
                    ((start - carried.start).length() < tolerance
                        && (end - carried.end).length() < tolerance)
                        || ((start - carried.end).length() < tolerance
                            && (end - carried.start).length() < tolerance)
                })
                .collect();
            let [edge] = matches.as_slice() else {
                return Err(OperationsError::NonManifoldResult);
            };
            topo.edge_mut(*edge)?.set_curve(carried.curve);
        }
        Ok(())
    }
}

fn matches_n339_patch(
    actual: &brepkit_math::nurbs::surface::NurbsSurface,
    sharp: Point3,
    first: Vec3,
    along: Vec3,
    second: Vec3,
    radius: f64,
) -> Result<bool, OperationsError> {
    let expected = super::setback_patch::surface()?;
    if actual.knots_u() != expected.knots_u()
        || actual.knots_v() != expected.knots_v()
        || actual.weights().len() != expected.weights().len()
    {
        return Ok(false);
    }
    let tolerance = Tolerance::new().linear * 100.0;
    let direct = actual
        .control_points()
        .iter()
        .zip(expected.control_points())
        .all(|(actual_row, expected_row)| {
            actual_row.iter().zip(expected_row).all(|(actual, local)| {
                let mapped =
                    sharp + (first * local.x() + along * local.y() + second * local.z()) * radius;
                (*actual - mapped).length() < tolerance
            })
        });
    if !direct {
        return Ok(false);
    }
    Ok(actual
        .weights()
        .iter()
        .flatten()
        .zip(expected.weights().iter().flatten())
        .all(|(actual, expected)| (actual - expected).abs() < f64::EPSILON * 100.0))
}

/// G1 rational transition for the recognized convex right-angle corner.
///
/// N341: `first`'s own radius (`radius_first`, the inherited edge) and
/// `second`'s own radius (`radius_second`, the newly requested edge) need not
/// be equal any more. The corner patch is `setback_patch::surface_mixed`,
/// the exact two-radius generalization of the original equal-radius
/// `setback_patch::surface` (which is `surface_mixed`'s own `r1 == r2` case,
/// checked numerically in `setback_patch`'s own tests) -- never a rescale of
/// one radius's template by the other's ratio.
pub(super) struct SetbackCorner {
    pub vertex: usize,
    origin: Point3,
    first: brepkit_math::vec::Vec3,
    second: brepkit_math::vec::Vec3,
    inward: brepkit_math::vec::Vec3,
    radius_first: f64,
    radius_second: f64,
}

impl SetbackCorner {
    fn new(
        topo: &Topology,
        solid: SolidId,
        first: EdgeId,
        second: EdgeId,
        radius_first: f64,
        radius_second: f64,
    ) -> Result<Self, OperationsError> {
        let fail = || OperationsError::InvalidInput {
            reason:
                "unsupported adjoining fillet corner: expected convex perpendicular planar supports"
                    .into(),
        };
        let a = topo.edge(first)?;
        let b = topo.edge(second)?;
        let shared: Vec<_> = [a.start(), a.end()]
            .into_iter()
            .filter(|&v| v == b.start() || v == b.end())
            .collect();
        let [vertex] = shared.as_slice() else {
            return Err(fail());
        };
        let origin = topo.vertex(*vertex)?.point();
        let adjacency = topo.build_adjacency(solid)?;
        let common: Vec<_> = adjacency
            .faces_for_edge(first)
            .iter()
            .filter(|id| adjacency.faces_for_edge(second).contains(id))
            .copied()
            .collect();
        let [common] = common.as_slice() else {
            return Err(fail());
        };
        let normal = topo
            .face(*common)?
            .effective_plane_normal()
            .ok_or_else(fail)?;
        // The transition stays inside the three rectangular support faces.
        // Curved boundaries, notches and holes need a different corner solver.
        for &face_id in adjacency
            .faces_for_edge(first)
            .iter()
            .chain(adjacency.faces_for_edge(second))
        {
            let face = topo.face(face_id)?;
            let wire = topo.wire(face.outer_wire())?;
            if !face.inner_wires().is_empty()
                || wire.edges().len() != 4
                || wire.edges().iter().any(|oe| {
                    !topo
                        .edge(oe.edge())
                        .is_ok_and(|edge| matches!(edge.curve(), EdgeCurve::Line))
                })
            {
                return Err(fail());
            }
            let mut sides = Vec::new();
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                sides.push(
                    (topo.vertex(edge.end())?.point() - topo.vertex(edge.start())?.point())
                        .normalize()?,
                );
            }
            for i in 0..4 {
                if sides[i].dot(sides[(i + 1) % 4]).abs() > Tolerance::new().angular {
                    return Err(fail());
                }
            }
        }
        let mut directions = Vec::new();
        // N341: the corner patch's extent along an edge's own axis is set by
        // the OTHER edge's radius, not its own -- e.g. `first`'s own edge
        // (local axis `first`) is consumed out to `radius_second` before the
        // patch's shared corner point, the same cross relationship
        // `setback_patch::surface_mixed`'s corner positions are built from
        // (see that module's doc: `A = (r2, r1, 0)`, `B = (0, r2, r1)`).
        // This clearance check is therefore conservative (uses the sum of
        // both radii, `>= max(2*radius_second, 2*radius_first)` when the two
        // are unequal) rather than picking the exact cross radius per edge,
        // to avoid asserting a tighter bound than has been independently
        // re-derived for the asymmetric case.
        let clearance = 2.0 * radius_first.max(radius_second);
        for (eid, edge) in [(first, a), (second, b)] {
            let far = if edge.start() == *vertex {
                edge.end()
            } else {
                edge.start()
            };
            let direction = (topo.vertex(far)?.point() - origin).normalize()?;
            if (topo.vertex(far)?.point() - origin).length() <= clearance {
                return Err(OperationsError::InvalidInput { reason: "adjoining fillet setback requires both sharp edges longer than twice the radius".into() });
            }
            let other = other_face(&adjacency, eid, *common).ok_or_else(fail)?;
            let side_normal = topo
                .face(other)?
                .effective_plane_normal()
                .ok_or_else(fail)?;
            if side_normal.dot(normal).abs() > Tolerance::new().angular
                || direction.dot(normal).abs() > Tolerance::new().angular
            {
                return Err(fail());
            }
            directions.push((direction, side_normal));
        }
        if directions[0].0.dot(directions[1].0).abs() > Tolerance::new().angular
            || directions[0].0.dot(directions[1].1) > -1.0 + Tolerance::new().angular
            || directions[1].0.dot(directions[0].1) > -1.0 + Tolerance::new().angular
        {
            return Err(fail());
        }
        let mut retained = Vec::new();
        for eid in solid_edges(topo, solid)? {
            let edge = topo.edge(eid)?;
            if eid != first && eid != second && (edge.start() == *vertex || edge.end() == *vertex) {
                retained.push(eid);
            }
        }
        let [retained] = retained.as_slice() else {
            return Err(fail());
        };
        let edge = topo.edge(*retained)?;
        let far = if edge.start() == *vertex {
            edge.end()
        } else {
            edge.start()
        };
        let span = topo.vertex(far)?.point() - origin;
        // The retained edge's own setback height is exactly `h = r1 + r2`
        // (`setback_patch::surface_mixed`'s derived, oracle-checked corner
        // height; `2 * radius` at `radius_first == radius_second`).
        if !matches!(edge.curve(), EdgeCurve::Line)
            || span.cross(normal).length() > Tolerance::new().linear
            || span.dot(-normal) <= radius_first + radius_second
        {
            return Err(OperationsError::InvalidInput { reason: "adjoining fillet setback requires more than twice the radius of clearance along the retained sharp edge".into() });
        }
        Ok(Self {
            vertex: vertex.index(),
            origin,
            first: directions[0].0,
            second: directions[1].0,
            inward: -normal,
            radius_first,
            radius_second,
        })
    }

    /// Map a point already expressed in physical (radius-scaled) local
    /// `(first, inward, second)` coordinates into world space. N341: unlike
    /// the old single-scalar `point()`, `local` is not a unit-template
    /// coordinate needing a further `* radius` -- `setback_patch::surface_mixed`
    /// and `mixed_planar_controls` already bake `radius_first`/`radius_second`
    /// into every coordinate they emit (this is what makes the two-radius
    /// construction possible at all: see that module's doc for why no single
    /// linear rescale of one radius's template can stand in for this).
    fn point(&self, local: Point3) -> Point3 {
        self.origin + self.first * local.x() + self.inward * local.y() + self.second * local.z()
    }

    pub(super) fn preserved(&self) -> Point3 {
        let h = self.radius_first + self.radius_second;
        self.point(Point3::new(0.0, h, 0.0))
    }

    pub(super) fn face_spec(&self) -> Result<crate::boolean::FaceSpec, OperationsError> {
        let local = super::setback_patch::surface_mixed(self.radius_first, self.radius_second)?;
        let points = local
            .control_points()
            .iter()
            .map(|row| row.iter().map(|&p| self.point(p)).collect())
            .collect();
        let surface = brepkit_math::nurbs::surface::NurbsSurface::new(
            5,
            5,
            local.knots_u().to_vec(),
            local.knots_v().to_vec(),
            points,
            local.weights().to_vec(),
        )?;
        let h = self.radius_first + self.radius_second;
        let (r1, r2) = (self.radius_first, self.radius_second);
        let mut vertices = [
            Point3::new(0.0, h, 0.0),
            Point3::new(r2, r1, 0.0),
            Point3::new(r2, 0.0, r1),
            Point3::new(0.0, r2, r1),
        ]
        .map(|p| self.point(p))
        .to_vec();
        // The local (first, inward, second) frame may be left-handed.
        let reversed = self.first.cross(self.inward).dot(self.second) < 0.0;
        if reversed {
            vertices.reverse();
        }
        Ok(crate::boolean::FaceSpec::Surface {
            vertices,
            surface: FaceSurface::Nurbs(surface),
            reversed,
            inner_wires: vec![],
        })
    }

    /// Replace the two planar-support chords by the exact cubic boundaries
    /// of the corner patch. Each shared edge is replaced once, for both faces.
    pub(super) fn install_contact_curves(
        &self,
        topo: &mut Topology,
        solid: SolidId,
    ) -> Result<(), OperationsError> {
        for control in
            super::setback_patch::mixed_planar_controls(self.radius_first, self.radius_second)
        {
            let points = control.map(|p| self.point(p));
            let [a, _, _, b] = points;
            let mut matches = Vec::new();
            for id in solid_edges(topo, solid)? {
                let edge = topo.edge(id)?;
                let p = topo.vertex(edge.start())?.point();
                let q = topo.vertex(edge.end())?.point();
                let tol = Tolerance::new().linear;
                if ((p - a).length() < tol && (q - b).length() < tol)
                    || ((p - b).length() < tol && (q - a).length() < tol)
                {
                    matches.push((id, (p - a).length() < tol));
                }
            }
            let [(id, forward)] = matches.as_slice() else {
                return Err(OperationsError::NonManifoldResult);
            };
            let mut controls = points.to_vec();
            if !forward {
                controls.reverse();
            }
            let knots = [0.0; 4].into_iter().chain([1.0; 4]).collect();
            let curve =
                brepkit_math::nurbs::curve::NurbsCurve::new(3, knots, controls, vec![1.0; 4])?;
            topo.edge_mut(*id)?.set_curve(EdgeCurve::NurbsCurve(curve));
        }
        Ok(())
    }
}

/// Recognize geometry before allocating anything. In particular, a circle's
/// endpoints alone do not prove the selected arc is the quarter-circle rather
/// than its three-quarter complement.
///
/// N341: the newly requested radius is no longer compared against the
/// recognized strip's own radius here -- see `Strip::radius` and the doc on
/// `strip_radius` below. The parameter is kept (renamed) so the call site
/// reads clearly and future checks that genuinely need the requested radius
/// (as opposed to the strip's own) have an obvious place to add them.
#[allow(clippy::too_many_lines)]
fn recognize(
    topo: &Topology,
    adjacency: &AdjacencyIndex,
    face_id: FaceId,
    target: EdgeId,
    _requested_radius: f64,
) -> Result<Option<Strip>, OperationsError> {
    let tol = Tolerance::new();
    let face = topo.face(face_id)?;
    let FaceSurface::Cylinder(cylinder) = face.surface() else {
        return Ok(None);
    };
    // N341: the inherited strip's own radius need not equal the newly
    // requested `radius` any more. `strip_radius` is that strip's own,
    // already-built radius; every check below that used to compare against
    // the caller's `radius` now checks internal self-consistency of the
    // recognized strip instead (its cylinder and both end arcs must agree
    // with each other), and the corner patch is built from `strip_radius`
    // (inherited side) and `radius` (requested side) independently.
    let strip_radius = cylinder.radius();
    if face.is_reversed() || !face.inner_wires().is_empty() || strip_radius <= tol.linear {
        return Ok(None);
    }
    let wire = topo.wire(face.outer_wire())?;
    if wire.edges().len() != 4 || wire.edges().iter().any(|oe| oe.edge() == target) {
        return Ok(None);
    }
    let selected = topo.edge(target)?;
    if !matches!(selected.curve(), EdgeCurve::Line) {
        return Ok(None);
    }
    let mut axial = Vec::new();
    let mut arcs = Vec::new();
    for oe in wire.edges() {
        match topo.edge(oe.edge())?.curve() {
            EdgeCurve::Line => axial.push(oe.edge()),
            EdgeCurve::Circle(_) => arcs.push(oe.edge()),
            _ => return Ok(None),
        }
    }
    let ([a0, a1], [c0, c1]) = (axial.as_slice(), arcs.as_slice()) else {
        return Ok(None);
    };
    let mut supports = Vec::new();
    for &edge_id in &[*a0, *a1] {
        let Some(other) = other_face(adjacency, edge_id, face_id) else {
            return Ok(None);
        };
        let support = topo.face(other)?;
        let Some(normal) = support.effective_plane_normal() else {
            return Ok(None);
        };
        if !support.inner_wires().is_empty() || normal.dot(cylinder.axis()).abs() > tol.angular {
            return Ok(None);
        }
        let edge = topo.edge(edge_id)?;
        let p = topo.vertex(edge.start())?.point();
        let q = topo.vertex(edge.end())?.point();
        if (q - p).cross(cylinder.axis()).length() > tol.linear
            || (q - p).length() <= 2.0 * strip_radius
        {
            return Ok(None);
        }
        for point in [p, q] {
            let center = cylinder.origin()
                + cylinder.axis() * (point - cylinder.origin()).dot(cylinder.axis());
            if (point - center - normal * strip_radius).length() > tol.linear {
                return Ok(None);
            }
        }
        supports.push((other, normal));
    }
    if supports[0].0 == supports[1].0 || supports[0].1.dot(supports[1].1).abs() > tol.angular {
        return Ok(None);
    }
    let mut ends = Vec::new();
    let mut caps = Vec::new();
    let mut touches = 0;
    for &arc_id in &[*c0, *c1] {
        let Some(cap_id) = other_face(adjacency, arc_id, face_id) else {
            return Ok(None);
        };
        let cap = topo.face(cap_id)?;
        let Some(normal) = cap.effective_plane_normal() else {
            return Ok(None);
        };
        if !cap.inner_wires().is_empty()
            || normal.cross(cylinder.axis()).length() > tol.angular
            || supports.iter().any(|&(id, _)| id == cap_id)
        {
            return Ok(None);
        }
        let edge = topo.edge(arc_id)?;
        let EdgeCurve::Circle(circle) = edge.curve() else {
            return Ok(None);
        };
        let (va, vb) = (edge.start(), edge.end());
        let (pa, pb) = (topo.vertex(va)?.point(), topo.vertex(vb)?.point());
        let center =
            cylinder.origin() + cylinder.axis() * (pa - cylinder.origin()).dot(cylinder.axis());
        let (t0, t1) = edge.curve().domain_with_endpoints(pa, pb);
        if (circle.center() - center).length() > tol.linear
            || (circle.radius() - strip_radius).abs() > tol.linear
            || circle.normal().cross(cylinder.axis()).length() > tol.angular
            || ((pb - center).length() - strip_radius).abs() > tol.linear
            || (pa - pb).dot(cylinder.axis()).abs() > tol.linear
            || (t1 - t0 - std::f64::consts::FRAC_PI_2).abs() > tol.angular
        {
            return Ok(None);
        }
        // Each circular end must connect one vertex of each axial contact.
        for &axial_id in &[*a0, *a1] {
            let line = topo.edge(axial_id)?;
            if [va, vb]
                .iter()
                .filter(|&&v| v == line.start() || v == line.end())
                .count()
                != 1
            {
                return Ok(None);
            }
        }
        let sharp = pa + (pb - center);
        for &(id, _) in &supports {
            let FaceSurface::Plane { normal, d } = topo.face(id)?.surface() else {
                return Ok(None);
            };
            if (crate::dot_normal_point(*normal, sharp) - d).abs() > tol.linear {
                return Ok(None);
            }
        }
        if [va, vb]
            .iter()
            .any(|&v| v == selected.start() || v == selected.end())
        {
            let target_faces = adjacency.faces_for_edge(target);
            if !target_faces.contains(&cap_id)
                || !supports.iter().any(|&(id, _)| target_faces.contains(&id))
            {
                return Ok(None);
            }
            touches += 1;
        }
        ends.push(([va, vb], sharp));
        caps.push(cap_id);
    }
    if caps[0] == caps[1] || touches != 1 {
        return Ok(None);
    }
    Ok(Some(Strip {
        face: face_id,
        axial: [*a0, *a1],
        ends: [ends[0], ends[1]],
        radius: strip_radius,
    }))
}

fn other_face(adjacency: &AdjacencyIndex, edge: EdgeId, face: FaceId) -> Option<FaceId> {
    match adjacency.faces_for_edge(edge) {
        [a, b] if *a == face && *b != face => Some(*b),
        [a, b] if *b == face && *a != face => Some(*a),
        _ => None,
    }
}

fn mapped_point(strip: &Strip, vertex: VertexId) -> Option<Point3> {
    strip
        .ends
        .iter()
        .find_map(|(vertices, point)| vertices.contains(&vertex).then_some(*point))
}

/// All moved edges outside the strip must merely extend their original line.
/// A curved or additional face at any affected vertex is not an isolated strip.
fn can_extend(topo: &Topology, solid: SolidId, strip: &Strip) -> Result<bool, OperationsError> {
    let tolerance = Tolerance::new().linear;
    for edge_id in solid_edges(topo, solid)? {
        let edge = topo.edge(edge_id)?;
        let (a, b) = (
            mapped_point(strip, edge.start()),
            mapped_point(strip, edge.end()),
        );
        let ((Some(sharp), None) | (None, Some(sharp))) = (a, b) else {
            continue;
        };
        if !matches!(edge.curve(), EdgeCurve::Line) {
            return Ok(false);
        }
        let p = topo.vertex(edge.start())?.point();
        let q = topo.vertex(edge.end())?.point();
        if (q - p).length() <= tolerance
            || (sharp - p).cross(q - p).length() > tolerance * (q - p).length()
        {
            return Ok(false);
        }
        // Extending, not collapsing/reversing a neighbor through its far end.
        let (contact, far) = if a.is_some() { (p, q) } else { (q, p) };
        if (sharp - contact).dot(far - contact) >= 0.0 {
            return Ok(false);
        }
    }
    for &fid in topo.shell(topo.solid(solid)?.outer_shell())?.faces() {
        if fid == strip.face {
            continue;
        }
        let face = topo.face(fid)?;
        let mut touched = false;
        for &wid in std::iter::once(&face.outer_wire()).chain(face.inner_wires()) {
            for oe in topo.wire(wid)?.edges() {
                let edge = topo.edge(oe.edge())?;
                if mapped_point(strip, edge.start()).is_some()
                    || mapped_point(strip, edge.end()).is_some()
                {
                    touched = true;
                    if wid != face.outer_wire() {
                        return Ok(false);
                    }
                }
            }
        }
        if touched && !face.surface().is_planar() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn reconstruct(
    topo: &mut Topology,
    solid: SolidId,
    strip: &Strip,
    target: EdgeId,
) -> Result<(SolidId, EdgeId, EdgeId), OperationsError> {
    let mut vertices = HashMap::new();
    let mut sharp = Vec::new();
    for (old, point) in strip.ends {
        let new = topo.add_vertex(Vertex::new(point, Tolerance::new().linear));
        sharp.push(new);
        for vertex in old {
            vertices.insert(vertex, new);
        }
    }
    let inherited = topo.add_edge(Edge::new(sharp[0], sharp[1], EdgeCurve::Line));
    let mut replacements = HashMap::new();
    for edge_id in solid_edges(topo, solid)? {
        let edge = topo.edge(edge_id)?;
        let start = vertices
            .get(&edge.start())
            .copied()
            .unwrap_or_else(|| edge.start());
        let end = vertices
            .get(&edge.end())
            .copied()
            .unwrap_or_else(|| edge.end());
        let replacement = if start == end {
            None
        } else if strip.axial.contains(&edge_id) {
            Some((inherited, start == sharp[0]))
        } else if start != edge.start() || end != edge.end() {
            Some((
                topo.add_edge(Edge::new(start, end, edge.curve().clone())),
                true,
            ))
        } else {
            Some((edge_id, true))
        };
        replacements.insert(edge_id, replacement);
    }
    let requested = replacements
        .get(&target)
        .and_then(|value| *value)
        .ok_or(OperationsError::NonManifoldResult)?
        .0;
    let mut faces = Vec::new();
    let original_faces = topo
        .shell(topo.solid(solid)?.outer_shell())?
        .faces()
        .to_vec();
    for face_id in original_faces {
        if face_id == strip.face {
            continue;
        }
        let mut face = topo.face(face_id)?.clone();
        let wire = topo.wire(face.outer_wire())?;
        let mut changed = false;
        let mut edges = Vec::new();
        for oe in wire.edges() {
            match replacements[&oe.edge()] {
                Some((edge, same_direction)) => {
                    changed |= edge != oe.edge();
                    edges.push(OrientedEdge::new(edge, oe.is_forward() == same_direction));
                }
                None => changed = true,
            }
        }
        if changed {
            let wire = topo.add_wire(Wire::new(edges, true)?);
            face.set_outer_wire(wire);
            faces.push(topo.add_face(face));
        } else {
            faces.push(face_id);
        }
    }
    let shell = topo.add_shell(Shell::new(faces)?);
    Ok((
        topo.add_solid(Solid::new(shell, vec![])),
        inherited,
        requested,
    ))
}
