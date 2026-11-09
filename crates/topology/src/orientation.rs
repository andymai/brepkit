//! Face-orientation propagation and normal alignment.

use std::collections::VecDeque;

use brepkit_math::det_hash::{DetHashMap, DetHashSet};
use brepkit_math::vec::Point3;

use crate::Topology;
use crate::edge::EdgeId;
use crate::face::{FaceId, FaceSurface};

/// Propagate effective edge-sense consistency across a set of faces.
///
/// `seeds` are faces whose existing orientation is authoritative. Every
/// reachable non-seed face is flipped when it traverses a shared manifold edge
/// in the same effective direction as its visited neighbor. If no seed is
/// present, the highest-index face is used as a deterministic anchor.
///
/// Returns the number of faces whose reversal flag changed.
///
/// # Errors
///
/// Returns an error if topology lookup or mutation fails.
pub fn propagate_orientation(
    topo: &mut Topology,
    faces: &[FaceId],
    seeds: &[FaceId],
) -> Result<usize, crate::TopologyError> {
    let face_set: DetHashSet<FaceId> = faces.iter().copied().collect();
    let mut raw_senses: DetHashMap<FaceId, Vec<(EdgeId, bool)>> = DetHashMap::default();
    let mut reversals: DetHashMap<FaceId, bool> = DetHashMap::default();

    for &face_id in faces {
        let face = topo.face(face_id)?;
        reversals.insert(face_id, face.is_reversed());
        let mut senses = Vec::new();
        for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
        {
            for oriented in topo.wire(wire_id)?.edges() {
                senses.push((oriented.edge(), oriented.is_forward()));
            }
        }
        raw_senses.insert(face_id, senses);
    }

    let mut edge_users: DetHashMap<EdgeId, Vec<FaceId>> = DetHashMap::default();
    for (&face_id, senses) in &raw_senses {
        for &(edge_id, _) in senses {
            edge_users.entry(edge_id).or_default().push(face_id);
        }
    }

    let mut visited = DetHashSet::default();
    let mut queue = VecDeque::new();
    for &seed in seeds {
        if face_set.contains(&seed) && visited.insert(seed) {
            queue.push_back(seed);
        }
    }
    let mut flipped = 0usize;

    while visited.len() < face_set.len() || !queue.is_empty() {
        if queue.is_empty()
            && let Some(face_id) = faces
                .iter()
                .copied()
                .filter(|face| !visited.contains(face))
                .max()
        {
            visited.insert(face_id);
            queue.push_back(face_id);
        }

        let Some(face_id) = queue.pop_front() else {
            break;
        };
        let reversed = reversals.get(&face_id).copied().unwrap_or(false);
        let Some(senses) = raw_senses.get(&face_id) else {
            continue;
        };
        for &(edge_id, raw_sense) in senses {
            let effective_sense = raw_sense ^ reversed;
            let Some(users) = edge_users.get(&edge_id) else {
                continue;
            };
            if users.len() != 2 {
                continue;
            }
            for &neighbor in users {
                if neighbor == face_id || visited.contains(&neighbor) {
                    continue;
                }
                let neighbor_reversed = reversals.get(&neighbor).copied().unwrap_or(false);
                let Some(&(_, neighbor_raw_sense)) =
                    raw_senses.get(&neighbor).and_then(|neighbor_senses| {
                        neighbor_senses
                            .iter()
                            .find(|(candidate, _)| *candidate == edge_id)
                    })
                else {
                    continue;
                };
                if neighbor_raw_sense ^ neighbor_reversed == effective_sense {
                    topo.face_mut(neighbor)?.set_reversed(!neighbor_reversed);
                    reversals.insert(neighbor, !neighbor_reversed);
                    flipped += 1;
                }
                visited.insert(neighbor);
                queue.push_back(neighbor);
            }
        }
    }

    Ok(flipped)
}

/// The winding of a cylinder, cone or torus face's boundary samples about
/// the surface's natural normal: the signed area of their loop in `(u, v)`,
/// positive when it runs counterclockwise about `∂u × ∂v` taken along the
/// normal. A sample on the axis (a cone's apex, a horn torus's pinch) has no
/// `u` of its own and takes its neighbour's. A B-spline face is read in its
/// own parameters ([`nurbs_uv_winding`]). `None` for other surfaces, and
/// for a loop that does not close once unwrapped (a band around the axis).
fn uv_winding(surface: &FaceSurface, points: &[Point3]) -> Option<f64> {
    use std::f64::consts::{PI, TAU};
    let v_periodic = match surface {
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => false,
        FaceSurface::Torus(_) => true,
        FaceSurface::Nurbs(nurbs) => return nurbs_uv_winding(surface, nurbs, points),
        FaceSurface::Plane { .. } | FaceSurface::Sphere(_) => return None,
    };
    let wrap = |d: f64| d - TAU * (d / TAU).round();
    let h = 1e-6;
    // `∂u` along the surface at a parameter pair; it vanishes on the axis.
    let du = |(u, v): (f64, f64)| Some(surface.evaluate(u + h, v)? - surface.evaluate(u - h, v)?);
    let raw: Vec<(f64, f64)> = points
        .iter()
        .map(|&p| surface.project_point(p))
        .collect::<Option<_>>()?;
    let speeds: Vec<f64> = raw
        .iter()
        .map(|&s| du(s).map(brepkit_math::vec::Vector::length))
        .collect::<Option<_>>()?;
    let scale = speeds.iter().copied().fold(0.0, f64::max);
    if scale <= 0.0 {
        return None;
    }
    let on_axis: Vec<bool> = speeds.iter().map(|&s| s <= 1e-6 * scale).collect();
    let anchor = on_axis.iter().position(|&a| !a)?;
    let mut loop_uv: Vec<(f64, f64)> = Vec::with_capacity(raw.len());
    for k in 0..raw.len() {
        let index = (anchor + k) % raw.len();
        let (u, v) = raw[index];
        let Some(&(pu, pv)) = loop_uv.last() else {
            loop_uv.push((u, v));
            continue;
        };
        let u = if on_axis[index] {
            pu
        } else {
            pu + wrap(u - pu)
        };
        let v = if v_periodic { pv + wrap(v - pv) } else { v };
        loop_uv.push((u, v));
    }
    let (first, last) = (loop_uv[0], loop_uv[loop_uv.len() - 1]);
    let closes = |a: f64, b: f64, periodic: bool| !periodic || (a - b).abs() < PI;
    if !closes(first.0, last.0, true) || !closes(first.1, last.1, v_periodic) {
        return None;
    }
    let area = 0.5
        * (0..loop_uv.len())
            .map(|k| {
                let (a, b) = (loop_uv[k], loop_uv[(k + 1) % loop_uv.len()]);
                a.0.mul_add(b.1, -(b.0 * a.1))
            })
            .sum::<f64>();
    let at = loop_uv[0];
    let dv = surface.evaluate(at.0, at.1 + h)? - surface.evaluate(at.0, at.1 - h)?;
    let sense = du(at)?.cross(dv).dot(surface.normal(at.0, at.1));
    (sense != 0.0).then(|| area * sense.signum())
}

/// [`uv_winding`] on a B-spline surface: the signed area of the samples'
/// loop in `(u, v)`, positive when it runs counterclockwise about
/// `∂u × ∂v` taken along the normal. A long curved band (a round along a
/// spline rim) bends its loop round a centroid that falls outside it, which
/// the boundary integral misreads; its parameters do not bend. `None` where
/// a sample's projection fails or lands where `∂u` or `∂v` vanishes (a
/// patch collapsed to a pole, whose `u` there is anyone's), and for a loop
/// that winds round a periodic direction.
fn nurbs_uv_winding(
    surface: &FaceSurface,
    nurbs: &brepkit_math::nurbs::surface::NurbsSurface,
    points: &[Point3],
) -> Option<f64> {
    let (u0, u1) = nurbs.domain_u();
    let (v0, v1) = nurbs.domain_v();
    let periods = (
        nurbs.is_periodic_u().then_some(u1 - u0),
        nurbs.is_periodic_v().then_some(v1 - v0),
    );
    let wrap = |d: f64, period: Option<f64>| period.map_or(d, |p| d - p * (d / p).round());
    let tol = brepkit_math::tolerance::Tolerance::new().linear;
    let raw: Vec<(f64, f64)> = points
        .iter()
        .map(|&p| {
            brepkit_math::nurbs::projection::project_point_to_surface(nurbs, p, tol)
                .ok()
                .map(|found| (found.u, found.v))
        })
        .collect::<Option<_>>()?;
    let speeds: Vec<(f64, f64)> = raw
        .iter()
        .map(|&(u, v)| {
            let d = nurbs.derivatives(u, v, 1);
            (d[1][0].length(), d[0][1].length())
        })
        .collect();
    let scale = speeds
        .iter()
        .fold(0.0_f64, |m, &(su, sv)| m.max(su).max(sv));
    if scale <= 0.0
        || speeds
            .iter()
            .any(|&(su, sv)| su <= 1e-6 * scale || sv <= 1e-6 * scale)
    {
        return None;
    }
    let mut loop_uv: Vec<(f64, f64)> = Vec::with_capacity(raw.len());
    for &(u, v) in &raw {
        let Some(&(pu, pv)) = loop_uv.last() else {
            loop_uv.push((u, v));
            continue;
        };
        loop_uv.push((pu + wrap(u - pu, periods.0), pv + wrap(v - pv, periods.1)));
    }
    let (first, last) = (loop_uv[0], loop_uv[loop_uv.len() - 1]);
    let closes = |a: f64, b: f64, period: Option<f64>| {
        period.is_none_or(|p| (b + wrap(a - b, Some(p)) - a).abs() < 0.5 * p)
    };
    if !closes(first.0, last.0, periods.0) || !closes(first.1, last.1, periods.1) {
        return None;
    }
    let area = 0.5
        * (0..loop_uv.len())
            .map(|k| {
                let (a, b) = (loop_uv[k], loop_uv[(k + 1) % loop_uv.len()]);
                a.0.mul_add(b.1, -(b.0 * a.1))
            })
            .sum::<f64>();
    let at = raw[0];
    let d = nurbs.derivatives(at.0, at.1, 1);
    let sense = d[1][0].cross(d[0][1]).dot(surface.normal(at.0, at.1));
    (sense != 0.0).then(|| area * sense.signum())
}

/// Align effective surface normals with the boundary-walk convention of seed faces.
///
/// Orientation propagation can make edge senses consistent while leaving a
/// reconstructed face's effective surface normal backwards. This pass
/// calibrates the solid's boundary-walk convention from authoritative input
/// `seeds` (or, when no seed reads one, from the majority of `faces`), then
/// triple-flips disagreeing result faces: reverse each wire,
/// invert its raw edge senses, and toggle the face reversal flag. The triple
/// flip changes the effective normal without disturbing shared-edge senses.
///
/// Faces with holes or repeated boundary edges are skipped because their
/// boundary integral is not a reliable disk-like orientation test.
///
/// Returns the number of faces triple-flipped.
///
/// # Errors
///
/// Returns an error if topology lookup or mutation fails.
#[allow(clippy::too_many_lines)]
pub fn normalize_face_normals(
    topo: &mut Topology,
    faces: &[FaceId],
    seeds: &[FaceId],
) -> Result<usize, crate::TopologyError> {
    let seed_set: DetHashSet<FaceId> = seeds.iter().copied().collect();
    let mut convention = 0.0;
    let mut seed_read = false;
    let mut readings: Vec<(FaceId, bool)> = Vec::new();

    for &face_id in seeds.iter().chain(faces.iter()) {
        let face = topo.face(face_id)?;
        let reversed = face.is_reversed();
        let surface = face.surface().clone();
        let wire_id = face.outer_wire();
        if !face.inner_wires().is_empty() {
            continue;
        }
        let wire = topo.wire(wire_id)?;
        {
            let mut seen = DetHashSet::default();
            if wire
                .edges()
                .iter()
                .any(|oriented| !seen.insert(oriented.edge()))
            {
                continue;
            }
        }

        let oriented_edges = if reversed {
            wire.edges().iter().rev().copied().collect::<Vec<_>>()
        } else {
            wire.edges().to_vec()
        };
        let mut points = Vec::new();
        for oriented in &oriented_edges {
            let edge = topo.edge(oriented.edge())?;
            let start = topo.vertex(edge.start())?.point();
            let end = topo.vertex(edge.end())?.point();
            let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
            let effective_forward = oriented.is_forward() ^ reversed;
            for sample in 0..8 {
                #[allow(clippy::cast_precision_loss)]
                let fraction = sample as f64 / 8.0;
                let parameter = if effective_forward {
                    t0 + (t1 - t0) * fraction
                } else {
                    t1 - (t1 - t0) * fraction
                };
                points.push(edge.curve().evaluate_with_endpoints(parameter, start, end));
            }
        }
        if points.len() < 3 {
            continue;
        }

        #[allow(clippy::cast_precision_loss)]
        let inverse_count = 1.0 / points.len() as f64;
        let mut center_x = 0.0;
        let mut center_y = 0.0;
        let mut center_z = 0.0;
        for point in &points {
            center_x += point.x();
            center_y += point.y();
            center_z += point.z();
        }
        let center = Point3::new(
            center_x * inverse_count,
            center_y * inverse_count,
            center_z * inverse_count,
        );

        let normal_at = |point: Point3| {
            if let FaceSurface::Plane { normal, .. } = &surface {
                Some(*normal)
            } else {
                let (u, v) = surface.project_point(point)?;
                Some(surface.normal(u, v))
            }
        };
        let mut integral = 0.0;
        let mut total_length = 0.0;
        for (index, &start) in points.iter().enumerate() {
            let end = points[(index + 1) % points.len()];
            let segment = end - start;
            let length = segment.length();
            if length < 1e-12 {
                continue;
            }
            let midpoint = Point3::new(
                f64::midpoint(start.x(), end.x()),
                f64::midpoint(start.y(), end.y()),
                f64::midpoint(start.z(), end.z()),
            );
            let Some(mut normal) = normal_at(midpoint) else {
                continue;
            };
            if reversed {
                normal = -normal;
            }
            integral += normal.cross(segment).dot(center - midpoint);
            total_length += length;
        }
        if total_length < 1e-12 || integral.abs() < 1e-9 * total_length * total_length {
            continue;
        }
        // Read against a reference point, the integral is a winding only on a
        // plane: on a curved band spanning more than half a turn the samples'
        // centroid falls outside the band and the sign can flip. A loop that
        // closes in `(u, v)` reads its winding there instead.
        let integral = match uv_winding(&surface, &points) {
            Some(area) if reversed => -area,
            Some(area) => area,
            None => integral,
        };
        // A zero reading (exactly; small ones were skipped above) says
        // nothing either way.
        if integral == 0.0 {
            continue;
        }
        if seed_set.contains(&face_id) {
            convention += integral.signum();
            seed_read = true;
        } else {
            readings.push((face_id, integral > 0.0));
        }
    }
    // With no seed to read the convention from (every input face around the
    // result was rebuilt), the result's own faces vote: a closed shell walks
    // every face's boundary the same way about its effective normal, so the
    // few that disagree are the ones built backwards. Seeds that read but
    // disagree among themselves still decide nothing.
    if !seed_read {
        convention = readings
            .iter()
            .map(|&(_, positive)| if positive { 1.0 } else { -1.0 })
            .sum();
    }
    let flips: Vec<FaceId> = if convention == 0.0 {
        Vec::new()
    } else {
        readings
            .iter()
            .filter(|&&(_, positive)| positive != (convention > 0.0))
            .map(|&(face_id, _)| face_id)
            .collect()
    };

    for face_id in &flips {
        let face = topo.face(*face_id)?;
        let reversed = face.is_reversed();
        let mut wire_ids = vec![face.outer_wire()];
        wire_ids.extend_from_slice(face.inner_wires());
        for wire_id in wire_ids {
            let wire = topo.wire_mut(wire_id)?;
            let mut oriented_edges = wire.edges().to_vec();
            oriented_edges.reverse();
            for oriented in &mut oriented_edges {
                *oriented = crate::wire::OrientedEdge::new(oriented.edge(), !oriented.is_forward());
            }
            for (slot, oriented) in wire.edges_mut().iter_mut().zip(oriented_edges) {
                *slot = oriented;
            }
        }
        topo.face_mut(*face_id)?.set_reversed(!reversed);
    }

    Ok(flips.len())
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4, PI, TAU};

    use brepkit_math::curves::Circle3D;
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::{Point3, Vec3};

    use super::{normalize_face_normals, uv_winding};
    use crate::Topology;
    use crate::edge::{Edge, EdgeCurve};
    use crate::explorer::solid_faces;
    use crate::face::{Face, FaceId, FaceSurface};
    use crate::test_utils::make_unit_cube_manifold;
    use crate::vertex::Vertex;
    use crate::wire::{OrientedEdge, Wire};

    const RADIUS: f64 = 2.0;
    const HEIGHT: f64 = 3.0;

    /// The rows of a full circle of `RADIUS` about `+z` as a rational
    /// quadratic, from `+x` counterclockwise, with the weight of each row.
    fn circle_rows() -> Vec<(f64, f64, f64)> {
        (0..9)
            .map(|k| {
                let angle = f64::from(k) * FRAC_PI_4;
                let (reach, weight) = if k % 2 == 0 {
                    (RADIUS, 1.0)
                } else {
                    (RADIUS / FRAC_1_SQRT_2, FRAC_1_SQRT_2)
                };
                (reach * angle.cos(), reach * angle.sin(), weight)
            })
            .collect()
    }

    fn circle_knots() -> Vec<f64> {
        vec![
            0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
        ]
    }

    /// A B-spline cylinder of `RADIUS` from `z = 0` to `HEIGHT`: `u` runs
    /// counterclockwise about `+z` from `+x`, `v` up, so `∂u × ∂v` points
    /// out.
    fn nurbs_cylinder() -> NurbsSurface {
        let rows = circle_rows();
        NurbsSurface::new(
            2,
            1,
            circle_knots(),
            vec![0.0, 0.0, 1.0, 1.0],
            rows.iter()
                .map(|&(x, y, _)| vec![Point3::new(x, y, 0.0), Point3::new(x, y, HEIGHT)])
                .collect(),
            rows.iter().map(|&(_, _, w)| vec![w, w]).collect(),
        )
        .unwrap()
    }

    /// The cylinder's patch from `-half` to `half` radians about `+z`, its
    /// wire counterclockwise about the outward normal.
    fn seam_patch(topo: &mut Topology, half: f64, reversed: bool) -> FaceId {
        let at = |angle: f64, z: f64| Point3::new(RADIUS * angle.cos(), RADIUS * angle.sin(), z);
        let corners = [
            at(-half, 0.0),
            at(half, 0.0),
            at(half, HEIGHT),
            at(-half, HEIGHT),
        ];
        let [v0, v1, v2, v3] = corners.map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
        let rim = |z: f64| {
            EdgeCurve::Circle(
                Circle3D::new_with_ref(
                    Point3::new(0.0, 0.0, z),
                    Vec3::new(0.0, 0.0, 1.0),
                    RADIUS,
                    Vec3::new(1.0, 0.0, 0.0),
                )
                .unwrap(),
            )
        };
        let bottom = topo.add_edge(Edge::new(v0, v1, rim(0.0)));
        let right = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let top = topo.add_edge(Edge::new(v3, v2, rim(HEIGHT)));
        let left = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(bottom, true),
                    OrientedEdge::new(right, true),
                    OrientedEdge::new(top, false),
                    OrientedEdge::new(left, true),
                ],
                true,
            )
            .unwrap(),
        );
        let mut face = Face::new(wire, vec![], FaceSurface::Nurbs(nurbs_cylinder()));
        face.set_reversed(reversed);
        topo.add_face(face)
    }

    /// Turn a face's effective normal round without changing the senses its
    /// edges are used in: the state of a face built with its wire backwards
    /// and then flagged to agree with its neighbours.
    fn turn_inside_out(topo: &mut Topology, face_id: FaceId) {
        let wire_id = topo.face(face_id).unwrap().outer_wire();
        let wire = topo.wire_mut(wire_id).unwrap();
        let turned: Vec<OrientedEdge> = wire
            .edges()
            .iter()
            .rev()
            .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
            .collect();
        for (slot, oe) in wire.edges_mut().iter_mut().zip(turned) {
            *slot = oe;
        }
        let face = topo.face_mut(face_id).unwrap();
        face.set_reversed(!face.is_reversed());
    }

    #[test]
    fn without_seeds_the_faces_vote_on_the_convention() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let faces = solid_faces(&topo, cube).unwrap();
        assert_eq!(normalize_face_normals(&mut topo, &faces, &[]).unwrap(), 0);

        let flags = |topo: &Topology| -> Vec<bool> {
            faces
                .iter()
                .map(|&f| topo.face(f).unwrap().is_reversed())
                .collect()
        };
        let before = flags(&topo);
        turn_inside_out(&mut topo, faces[2]);
        assert_eq!(normalize_face_normals(&mut topo, &faces, &[]).unwrap(), 1);
        assert_eq!(flags(&topo), before);
    }

    #[test]
    fn seeds_that_disagree_decide_nothing() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let faces = solid_faces(&topo, cube).unwrap();
        // Two seeds reading opposite ways, and a result face turned: the
        // seeds read, so the result faces do not vote, and nothing turns.
        turn_inside_out(&mut topo, faces[0]);
        turn_inside_out(&mut topo, faces[2]);
        let seeds = [faces[0], faces[1]];
        assert_eq!(
            normalize_face_normals(&mut topo, &faces[2..], &seeds).unwrap(),
            0
        );
    }

    #[test]
    fn seeds_still_set_the_convention() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let faces = solid_faces(&topo, cube).unwrap();
        // Turn five of the six: the one seed outvotes them.
        for &face_id in &faces[1..] {
            turn_inside_out(&mut topo, face_id);
        }
        let flipped = normalize_face_normals(&mut topo, &faces[1..], &faces[..1]).unwrap();
        assert_eq!(flipped, 5);
    }

    /// A patch three quarters of a turn round a B-spline cylinder, across its
    /// seam, read against a cube's faces: as built it agrees with them, and
    /// turned inside out it is the one face turned back, plain and reversed
    /// alike.
    #[test]
    fn a_bspline_patch_across_the_seam_reads_its_winding() {
        for reversed in [false, true] {
            let mut topo = Topology::new();
            let cube = make_unit_cube_manifold(&mut topo);
            let seeds = solid_faces(&topo, cube).unwrap();
            let patch = seam_patch(&mut topo, 0.75 * PI, reversed);
            assert_eq!(
                normalize_face_normals(&mut topo, &[patch], &seeds).unwrap(),
                0,
                "reversed: {reversed}"
            );

            turn_inside_out(&mut topo, patch);
            assert_eq!(
                normalize_face_normals(&mut topo, &[patch], &seeds).unwrap(),
                1,
                "reversed: {reversed}"
            );
            assert_eq!(topo.face(patch).unwrap().is_reversed(), reversed);
        }
    }

    /// The patch's samples, counterclockwise about the outward normal, wind
    /// positive in `(u, v)` across the seam.
    #[test]
    fn a_bspline_patch_across_the_seam_winds_positive() {
        let surface = FaceSurface::Nurbs(nurbs_cylinder());
        let half = 0.75 * PI;
        let mut points = Vec::new();
        for k in 0..16 {
            let angle = (f64::from(k) / 16.0).mul_add(2.0 * half, -half);
            points.push(Point3::new(RADIUS * angle.cos(), RADIUS * angle.sin(), 0.0));
        }
        for k in 0..16 {
            let angle = (f64::from(k) / 16.0).mul_add(-2.0 * half, half);
            points.push(Point3::new(
                RADIUS * angle.cos(),
                RADIUS * angle.sin(),
                HEIGHT,
            ));
        }
        let area = uv_winding(&surface, &points).unwrap();
        assert!(area > 0.0, "area {area}");
    }

    /// A loop once round the cylinder does not close in `(u, v)`: the winding
    /// is left to the boundary integral.
    #[test]
    fn a_loop_once_round_a_bspline_cylinder_has_no_uv_winding() {
        let surface = FaceSurface::Nurbs(nurbs_cylinder());
        let points: Vec<Point3> = (0..32)
            .map(|k| {
                let angle = f64::from(k) / 32.0 * TAU;
                let z = (0.25 * HEIGHT).mul_add(angle.sin(), 0.5 * HEIGHT);
                Point3::new(RADIUS * angle.cos(), RADIUS * angle.sin(), z)
            })
            .collect();
        assert_eq!(uv_winding(&surface, &points), None);
    }

    /// A double cone, its apex at the middle of the domain: the patch's sense
    /// is read where its loop is, not at the apex, where `∂u` vanishes.
    #[test]
    fn a_bspline_patch_reads_its_sense_off_a_degenerate_middle() {
        let rows = circle_rows();
        let cone = NurbsSurface::new(
            2,
            1,
            circle_knots(),
            vec![0.0, 0.0, 1.0, 1.0],
            rows.iter()
                .map(|&(x, y, _)| vec![Point3::new(-x, -y, -RADIUS), Point3::new(x, y, RADIUS)])
                .collect(),
            rows.iter().map(|&(_, _, w)| vec![w, w]).collect(),
        )
        .unwrap();
        let corners = [(0.1, 0.6), (0.3, 0.6), (0.3, 0.9), (0.1, 0.9)];
        let mut points = Vec::new();
        for (k, &(u0, v0)) in corners.iter().enumerate() {
            let (u1, v1) = corners[(k + 1) % corners.len()];
            for step in 0..8 {
                let t = f64::from(step) / 8.0;
                points.push(cone.evaluate(t.mul_add(u1 - u0, u0), t.mul_add(v1 - v0, v0)));
            }
        }
        let area = uv_winding(&FaceSurface::Nurbs(cone), &points).unwrap();
        assert!(area > 0.0, "area {area}");
    }
}
