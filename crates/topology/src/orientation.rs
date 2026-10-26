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
/// `u` of its own and takes its neighbour's. `None` for other surfaces, and
/// for a loop that does not close once unwrapped (a band around the axis).
fn uv_winding(surface: &FaceSurface, points: &[Point3]) -> Option<f64> {
    use std::f64::consts::{PI, TAU};
    let v_periodic = match surface {
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => false,
        FaceSurface::Torus(_) => true,
        FaceSurface::Plane { .. } | FaceSurface::Sphere(_) | FaceSurface::Nurbs(_) => return None,
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

/// Align effective surface normals with the boundary-walk convention of seed faces.
///
/// Orientation propagation can make edge senses consistent while leaving a
/// reconstructed face's effective surface normal backwards. This pass
/// calibrates the solid's boundary-walk convention from authoritative input
/// `seeds`, then triple-flips disagreeing result faces: reverse each wire,
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
    let mut flips = Vec::new();

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
        if seed_set.contains(&face_id) {
            convention += integral.signum();
        } else if convention != 0.0 && integral.signum() != convention.signum() {
            flips.push(face_id);
        }
    }

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
