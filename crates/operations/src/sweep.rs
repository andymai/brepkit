//! Path sweep: sweep a profile along a NURBS curve.
//!
//! Creates a solid by moving a planar profile along an arbitrary NURBS curve
//! path, keeping the profile perpendicular to the path tangent at each sample
//! point. Uses rotation-minimizing frames (double-reflection method) to avoid
//! Frenet-frame singularities on straight segments and inflection points.

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface_fitting::interpolate_surface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::dot_normal_point;

/// A coordinate frame at a point along the path.
struct Frame {
    origin: Point3,
    tangent: Vec3,
    up: Vec3,
    right: Vec3,
}

/// Compute rotation-minimizing frames along a NURBS path.
///
/// Samples the path at evenly-spaced parameter values and propagates the
/// initial up-vector using the double-reflection method to produce smooth,
/// twist-free frames. For open paths, produces `num_segments + 1` frames
/// (t=0 through t=1). For closed paths, produces `num_segments` frames
/// (t=0 through t=(N-1)/N), omitting t=1 since it duplicates t=0.
fn compute_frames(
    path: &NurbsCurve,
    num_segments: usize,
    initial_up: Vec3,
    is_closed: bool,
) -> Result<Vec<Frame>, crate::OperationsError> {
    let frame_count = if is_closed {
        num_segments
    } else {
        num_segments + 1
    };
    let mut frames = Vec::with_capacity(frame_count);

    let t0 = path.tangent(0.0)?;
    let up0 = orthogonalize(initial_up, t0);
    let right0 = t0.cross(up0);
    frames.push(Frame {
        origin: path.evaluate(0.0),
        tangent: t0,
        up: up0,
        right: right0,
    });

    // Propagate frames using the double-reflection method (Wang et al. 2008).
    //
    // Two reflections per step:
    //   1. Reflect across the plane bisecting consecutive origins (position change).
    //   2. Reflect across the plane bisecting the reflected tangent and new tangent.
    let last_k = if is_closed {
        num_segments - 1
    } else {
        num_segments
    };
    for k in 1..=last_k {
        #[allow(clippy::cast_precision_loss)]
        let t_param = (k as f64) / (num_segments as f64);

        let origin = path.evaluate(t_param);
        let tangent = path.tangent(t_param)?;

        let prev = &frames[k - 1];

        // Reflection 1: across the plane bisecting the two consecutive origins.
        let v1 = origin - prev.origin;
        let c1 = v1.dot(v1);
        let (up_l, tangent_l) = if c1 < 1e-30 {
            (prev.up, prev.tangent)
        } else {
            let up_r = prev.up - v1 * (2.0 * v1.dot(prev.up) / c1);
            let t_r = prev.tangent - v1 * (2.0 * v1.dot(prev.tangent) / c1);
            (up_r, t_r)
        };

        // Reflection 2: across the plane bisecting the reflected tangent
        // and the actual tangent at the new sample.
        let v2 = tangent - tangent_l;
        let c2 = v2.dot(v2);
        let up = if c2 < 1e-30 {
            orthogonalize(up_l, tangent)
        } else {
            let reflected = up_l - v2 * (2.0 * v2.dot(up_l) / c2);
            orthogonalize(reflected, tangent)
        };

        let right = tangent.cross(up);
        frames.push(Frame {
            origin,
            tangent,
            up,
            right,
        });
    }

    Ok(frames)
}

/// The profile's own orthonormal basis `(right, up, tangent)` for mapping its 2D
/// shape onto the path's perpendicular plane.
///
/// `tangent` is the profile normal, so a planar profile has ~zero tangent
/// component and is mapped entirely into `right`/`up`; the profile is then swept
/// perpendicular to the path regardless of how its plane was oriented relative
/// to the path (an edge-on profile no longer collapses to a flat ribbon). For a
/// profile already perpendicular to the path this equals the path frame's basis,
/// leaving such sweeps unchanged.
fn profile_basis(input_normal: Vec3) -> (Vec3, Vec3, Vec3) {
    let tangent = input_normal
        .normalize()
        .unwrap_or_else(|_| Vec3::new(0.0, 0.0, 1.0));
    let up = orthogonalize(pick_reference_axis(tangent), tangent);
    let right = tangent.cross(up);
    (right, up, tangent)
}

/// Project `v` to be perpendicular to `tangent`, then normalize.
///
/// Falls back to a world-axis-based vector if the projection is degenerate.
fn orthogonalize(v: Vec3, tangent: Vec3) -> Vec3 {
    let projected = v - tangent * tangent.dot(v);
    projected.normalize().unwrap_or_else(|_| {
        // Fallback: pick a world axis that isn't parallel to the tangent.
        let candidate = if tangent.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let proj2 = candidate - tangent * tangent.dot(candidate);
        // This should always succeed since candidate is chosen to not be
        // parallel to tangent.
        proj2.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0))
    })
}

/// Transform a profile vertex from its original position to a frame location.
///
/// The vertex's offset from the profile centroid is decomposed into the
/// initial frame's coordinate system (right, up, tangent), then
/// reconstructed in the target frame. Including the tangent component
/// ensures correct geometry even when the profile plane is not
/// perpendicular to the initial path tangent.
fn transform_point(
    point: Point3,
    centroid: Point3,
    initial_right: Vec3,
    initial_up: Vec3,
    initial_tangent: Vec3,
    frame: &Frame,
) -> Point3 {
    let offset = point - centroid;
    let local_r = initial_right.dot(offset);
    let local_u = initial_up.dot(offset);
    let local_t = initial_tangent.dot(offset);
    frame.origin + frame.right * local_r + frame.up * local_u + frame.tangent * local_t
}

/// Data from sweeping a single wire through frames.
struct SweptWireData {
    ring_verts: Vec<Vec<VertexId>>,
    ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>>,
    path_edges: Vec<Vec<brepkit_topology::edge::EdgeId>>,
    n: usize,
}

/// Sweep a wire's vertices through the given frames, creating ring vertices,
/// ring edges, and path edges.
///
/// `centroid`, `initial_right/up/tangent` define the local coordinate system
/// from which profile offsets are measured.
#[allow(clippy::too_many_arguments)]
fn sweep_wire_through_frames(
    topo: &mut Topology,
    wire_id: brepkit_topology::wire::WireId,
    centroid: Point3,
    initial_right: Vec3,
    initial_up: Vec3,
    initial_tangent: Vec3,
    frames: &[Frame],
    num_segments: usize,
    is_closed: bool,
) -> Result<SweptWireData, crate::OperationsError> {
    let tol = Tolerance::new();

    let wire = topo.wire(wire_id)?;
    let oriented: Vec<_> = wire.edges().to_vec();
    let n = oriented.len();

    let mut verts: Vec<VertexId> = Vec::with_capacity(n);
    for oe in &oriented {
        let edge = topo.edge(oe.edge())?;
        let vid = oe.oriented_start(edge);
        verts.push(vid);
    }

    let positions: Vec<Point3> = verts
        .iter()
        .map(|&vid| {
            topo.vertex(vid)
                .map(brepkit_topology::vertex::Vertex::point)
        })
        .collect::<Result<_, _>>()?;

    let mut ring_verts: Vec<Vec<VertexId>> = Vec::with_capacity(num_segments + 1);
    for frame in frames {
        let ring: Vec<VertexId> = positions
            .iter()
            .map(|&pos| {
                let transformed = transform_point(
                    pos,
                    centroid,
                    initial_right,
                    initial_up,
                    initial_tangent,
                    frame,
                );
                topo.add_vertex(Vertex::new(transformed, tol.linear))
            })
            .collect();
        ring_verts.push(ring);
    }

    // For closed paths, alias first ring as last so indexing works unchanged.
    if is_closed {
        ring_verts.push(ring_verts[0].clone());
    }

    let real_ring_count = if is_closed {
        ring_verts.len() - 1
    } else {
        ring_verts.len()
    };
    let mut ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> =
        Vec::with_capacity(num_segments + 1);
    for ring in &ring_verts[..real_ring_count] {
        let edges: Vec<_> = (0..n)
            .map(|i| {
                let next = (i + 1) % n;
                topo.add_edge(Edge::new(ring[i], ring[next], EdgeCurve::Line))
            })
            .collect();
        ring_edges.push(edges);
    }
    if is_closed {
        ring_edges.push(ring_edges[0].clone());
    }

    let mut path_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = Vec::with_capacity(num_segments);
    for seg in 0..num_segments {
        let edges: Vec<_> = (0..n)
            .map(|i| {
                topo.add_edge(Edge::new(
                    ring_verts[seg][i],
                    ring_verts[seg + 1][i],
                    EdgeCurve::Line,
                ))
            })
            .collect();
        path_edges.push(edges);
    }

    Ok(SweptWireData {
        ring_verts,
        ring_edges,
        path_edges,
        n,
    })
}

/// Build inward-facing side faces for an inner wire swept through frames.
fn build_inner_side_faces(
    topo: &mut Topology,
    iwd: &SweptWireData,
    num_segments: usize,
) -> Result<Vec<FaceId>, crate::OperationsError> {
    let mut faces = Vec::new();

    for seg in 0..num_segments {
        for i in 0..iwd.n {
            let next_i = (i + 1) % iwd.n;

            let p0 = topo.vertex(iwd.ring_verts[seg][i])?.point();
            let p1 = topo.vertex(iwd.ring_verts[seg][next_i])?.point();
            let p_next = topo.vertex(iwd.ring_verts[seg + 1][i])?.point();
            let edge_dir = p1 - p0;
            let path_dir = p_next - p0;
            // Reversed normal (inward-facing).
            let side_normal = path_dir
                .cross(edge_dir)
                .normalize()
                .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
            let side_d = dot_normal_point(side_normal, p0);

            // Reversed winding compared to outer side faces.
            let side_wire = Wire::new(
                vec![
                    OrientedEdge::new(iwd.path_edges[seg][i], true),
                    OrientedEdge::new(iwd.ring_edges[seg + 1][i], true),
                    OrientedEdge::new(iwd.path_edges[seg][next_i], false),
                    OrientedEdge::new(iwd.ring_edges[seg][i], false),
                ],
                true,
            )
            .map_err(crate::OperationsError::Topology)?;

            let side_wire_id = topo.add_wire(side_wire);
            let fid = topo.add_face(Face::new(
                side_wire_id,
                vec![],
                FaceSurface::Plane {
                    normal: side_normal,
                    d: side_d,
                },
            ));
            faces.push(fid);
        }
    }

    Ok(faces)
}

/// Build inner wire loops for a cap face at the given ring index.
fn build_inner_cap_wires(
    topo: &mut Topology,
    inner_data: &[SweptWireData],
    ring_idx: usize,
    reversed: bool,
) -> Result<Vec<brepkit_topology::wire::WireId>, crate::OperationsError> {
    let mut wires = Vec::new();
    for iwd in inner_data {
        let edges: Vec<OrientedEdge> = if reversed {
            (0..iwd.n)
                .rev()
                .map(|i| OrientedEdge::new(iwd.ring_edges[ring_idx][i], false))
                .collect()
        } else {
            (0..iwd.n)
                .map(|i| OrientedEdge::new(iwd.ring_edges[ring_idx][i], true))
                .collect()
        };
        let wire = Wire::new(edges, true).map_err(crate::OperationsError::Topology)?;
        wires.push(topo.add_wire(wire));
    }
    Ok(wires)
}

/// Insert interior points into a polyline so no gap exceeds a small multiple
/// of the typical (median) sample spacing.
///
/// A single global interpolating NURBS fit through unevenly-spaced points
/// overshoots wildly where a long, sparsely-sampled span (e.g. a long straight
/// spine edge sampled only at its endpoints) sits between densely-sampled
/// high-curvature corners — chord-length parameterization does not prevent this
/// when one span is orders of magnitude longer than its neighbours. Bounding
/// the max gap keeps the fit well-conditioned. The threshold is derived from
/// the median gap, so it is scale-invariant; interior insertion per segment is
/// capped to avoid blow-up on pathological inputs.
#[must_use]
pub fn densify_path_points(points: &[Point3]) -> Vec<Point3> {
    const GAP_RATIO: f64 = 4.0;
    const MAX_INSERT_PER_SEGMENT: usize = 256;

    if points.len() < 3 {
        return points.to_vec();
    }

    let mut gaps: Vec<f64> = points
        .windows(2)
        .map(|w| (w[1] - w[0]).length())
        .filter(|d| *d > 1e-12)
        .collect();
    if gaps.is_empty() {
        return points.to_vec();
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = gaps[gaps.len() / 2];
    let max_gap = median * GAP_RATIO;
    if max_gap <= 0.0 {
        return points.to_vec();
    }

    let mut out: Vec<Point3> = Vec::with_capacity(points.len());
    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        out.push(a);
        let seg = (b - a).length();
        if seg > max_gap {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let steps = ((seg / max_gap).ceil() as usize).min(MAX_INSERT_PER_SEGMENT);
            for s in 1..steps {
                #[allow(clippy::cast_precision_loss)]
                let f = (s as f64) / (steps as f64);
                out.push(a + (b - a) * f);
            }
        }
    }
    if let Some(&last) = points.last() {
        out.push(last);
    }
    out
}

/// Interior samples used to confirm a path is a straight segment.
const STRAIGHT_SAMPLES: usize = 8;

/// Approximate the centroid of a planar profile's outer boundary by sampling
/// its edges.
///
/// Sampling (rather than averaging stored vertices) is needed for a single
/// closed circle, whose start and end vertex coincide at the seam — the seam
/// point is not the center.
fn profile_outer_centroid(
    topo: &Topology,
    profile: FaceId,
) -> Result<Point3, crate::OperationsError> {
    let face = topo.face(profile)?;
    let wire = topo.wire(face.outer_wire())?;
    let (mut sx, mut sy, mut sz) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut count = 0_usize;
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let start = topo.vertex(edge.start())?.point();
        let end = topo.vertex(edge.end())?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
        // Four samples per edge span a closed circle's full sweep.
        for k in 0..4 {
            let t = t0 + (t1 - t0) * (f64::from(k) / 4.0);
            let p = edge.curve().evaluate_with_endpoints(t, start, end);
            sx += p.x();
            sy += p.y();
            sz += p.z();
            count += 1;
        }
    }
    if count == 0 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep profile has no outer-wire edges".into(),
        });
    }
    #[allow(clippy::cast_precision_loss)]
    let n = count as f64;
    Ok(Point3::new(sx / n, sy / n, sz / n))
}

/// Fast exact path for a straight, perpendicular sweep: it is a prism, so
/// delegate to [`crate::extrude::extrude`], which builds exact analytic side
/// faces (a circular profile becomes a true cylinder matching π·r²·L). The
/// general sweep inscribes a curved profile as a polygon and undercounts its
/// volume by ~2% (gh #965).
///
/// Returns `Ok(None)` — fall back to the general sweep — when the path is not
/// straight or the profile plane is not perpendicular to it (an oblique sweep
/// is not a prism and has different semantics).
fn try_straight_extrude(
    topo: &mut Topology,
    profile: FaceId,
    path: &NurbsCurve,
) -> Result<Option<SolidId>, crate::OperationsError> {
    let tol = Tolerance::new();

    let normal = match topo.face(profile)?.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => return Ok(None),
    };

    let start = path.evaluate(0.0);
    let end = path.evaluate(1.0);
    let chord = end - start;
    let length = chord.length();
    if length < tol.linear {
        return Ok(None); // closed or degenerate path
    }
    let Ok(dir) = chord.normalize() else {
        return Ok(None);
    };

    // Confirm straightness: every interior sample stays on the start→end line
    // and advances monotonically along it.
    for k in 1..STRAIGHT_SAMPLES {
        #[allow(clippy::cast_precision_loss)]
        let t = k as f64 / STRAIGHT_SAMPLES as f64;
        let v = path.evaluate(t) - start;
        let along = v.dot(dir);
        let perp = (v - dir * along).length();
        if perp > tol.linear * 100.0 || along < -tol.linear || along > length + tol.linear {
            return Ok(None);
        }
    }

    // The profile must be perpendicular to the path (normal parallel to the
    // direction); an oblique profile is not a prism.
    if normal.dot(dir).abs() < 1.0 - 1e-6 {
        return Ok(None);
    }

    // Position a copy of the profile so its centroid lands on the path start —
    // matching the general sweep's frame[0] placement — then extrude.
    let centroid = profile_outer_centroid(topo, profile)?;
    let moved = crate::copy::copy_face(topo, profile)?;
    let shift = start - centroid;
    crate::transform::transform_face(
        topo,
        moved,
        &Mat4::translation(shift.x(), shift.y(), shift.z()),
    )?;

    Ok(Some(crate::extrude::extrude(topo, moved, dir, length)?))
}

/// Sweep a face along a path curve to produce a solid.
///
/// Creates a solid by moving a profile along a NURBS curve, with the profile
/// oriented perpendicular to the path tangent at each sample point. Side faces
/// are planar quads connecting consecutive profile rings. The profile surface
/// may be planar or curved — only its boundary is used; the end caps are filled
/// from that boundary (a planar ring gets a `Plane` cap, a non-planar 4-sided
/// ring a bilinear patch).
///
/// # Errors
///
/// Returns an error if the path has fewer than 2 control points, a degenerate
/// tangent is encountered, or a section boundary is non-planar with more than
/// four edges or with holes (unsupported cap).
#[allow(clippy::too_many_lines)]
pub fn sweep(
    topo: &mut Topology,
    profile: FaceId,
    path: &NurbsCurve,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if path.control_points().len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep path must have at least 2 control points".into(),
        });
    }

    // A straight perpendicular sweep is a prism — build it exactly via extrude.
    if let Some(solid) = try_straight_extrude(topo, profile, path)? {
        return Ok(solid);
    }

    let face_data = topo.face(profile)?;
    let input_wire_id = face_data.outer_wire();
    let inner_wire_ids: Vec<brepkit_topology::wire::WireId> = face_data.inner_wires().to_vec();

    let start_end_coincide = tol.approx_eq(
        (path.evaluate(1.0) - path.evaluate(0.0)).length_squared(),
        0.0,
    );
    let is_closed = if start_end_coincide {
        // Distinguish closed loop (midpoint differs from start) from degenerate
        // (all control points coincident, truly zero arc length).
        let mid = path.evaluate(0.5);
        let mid_dist_sq = (mid - path.evaluate(0.0)).length_squared();
        if tol.approx_eq(mid_dist_sq, 0.0) {
            return Err(crate::OperationsError::InvalidInput {
                reason: "sweep path has zero length (start and end coincide)".into(),
            });
        }
        true
    } else {
        false
    };

    let input_wire = topo.wire(input_wire_id)?;
    let original_oriented: Vec<_> = input_wire.edges().to_vec();

    if original_oriented.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep profile has no edges".into(),
        });
    }

    // Split closed edges (e.g. full circles) into multiple line segments
    // so that the sweep can create proper side faces.
    let input_oriented = crate::extrude::maybe_split_closed_wire(
        topo,
        &original_oriented,
        tol.linear,
        crate::extrude::DEFAULT_DEFLECTION,
    )?;
    let n = input_oriented.len();

    let mut input_verts: Vec<VertexId> = Vec::with_capacity(n);
    for oe in &input_oriented {
        let edge = topo.edge(oe.edge())?;
        let vid = oe.oriented_start(edge);
        input_verts.push(vid);
    }

    let mut input_positions: Vec<Point3> = input_verts
        .iter()
        .map(|&vid| {
            topo.vertex(vid)
                .map(brepkit_topology::vertex::Vertex::point)
        })
        .collect::<Result<_, _>>()?;

    // Up-hint for the rotation-minimizing frame: the section boundary's own
    // normal (Newell), which works for planar and non-planar profiles alike —
    // a profile's stored surface is no longer required to be a plane.
    let mut input_normal = crate::winding::newell_normal(&input_positions)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));

    // Ensure CCW winding relative to the path direction at t=0.
    // CW-wound profiles (e.g. from brepjs) make `edge_dir.cross(path_dir)` point
    // inward instead of outward, producing inside-out side faces.
    let path_tangent_0 = path.tangent(0.0)?;
    if crate::winding::ensure_ccw_positions(&mut input_positions, path_tangent_0) {
        // Positions were reversed → boundary normal flips with them.
        input_normal = -input_normal;
    }

    let centroid = crate::winding::polygon_centroid(&input_positions);

    let num_segments = (path.control_points().len() * 2).max(4);

    // Seed the first frame's up-vector from the profile normal, projected
    // perpendicular to the path tangent at t=0.
    let up_hint = orthogonalize(input_normal, path_tangent_0);

    let frames = compute_frames(path, num_segments, up_hint, is_closed)?;

    // The first frame's basis vectors define the local coordinate system
    // in which profile vertex offsets are expressed.
    // Map the profile's 2D shape onto the path's perpendicular plane using the
    // profile's own basis, so a profile whose plane is not perpendicular to the
    // path is still swept perpendicular (rather than collapsing to a ribbon).
    let (initial_right, initial_up, initial_tangent) = profile_basis(input_normal);

    // ring_verts[k][i] = vertex at path sample k, profile vertex i.
    // For closed paths, frames has num_segments entries; we append a copy
    // of the first ring so indexing ring_verts[num_segments] works unchanged.
    let mut ring_verts: Vec<Vec<VertexId>> = Vec::with_capacity(num_segments + 1);

    for frame in &frames {
        let ring: Vec<VertexId> = input_positions
            .iter()
            .map(|&pos| {
                let transformed = transform_point(
                    pos,
                    centroid,
                    initial_right,
                    initial_up,
                    initial_tangent,
                    frame,
                );
                topo.add_vertex(Vertex::new(transformed, tol.linear))
            })
            .collect();
        ring_verts.push(ring);
    }

    // For closed paths, alias the first ring as the "last" so that
    // ring_verts[num_segments] == ring_verts[0] by vertex ID.
    if is_closed {
        ring_verts.push(ring_verts[0].clone());
    }

    // ring_edges[k][i] = edge from ring_verts[k][i] to ring_verts[k][(i+1)%n].
    let real_ring_count = if is_closed {
        num_segments
    } else {
        num_segments + 1
    };
    let mut ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> =
        Vec::with_capacity(num_segments + 1);
    for ring in &ring_verts[..real_ring_count] {
        let edges: Vec<_> = (0..n)
            .map(|i| {
                let next = (i + 1) % n;
                topo.add_edge(Edge::new(ring[i], ring[next], EdgeCurve::Line))
            })
            .collect();
        ring_edges.push(edges);
    }
    // For closed paths, alias the first ring's edges as the last ring's edges.
    if is_closed {
        ring_edges.push(ring_edges[0].clone());
    }

    // path_edges[seg][i] = edge from ring_verts[seg][i] to ring_verts[seg+1][i].
    let mut path_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = Vec::with_capacity(num_segments);
    for seg in 0..num_segments {
        let edges: Vec<_> = (0..n)
            .map(|i| {
                topo.add_edge(Edge::new(
                    ring_verts[seg][i],
                    ring_verts[seg + 1][i],
                    EdgeCurve::Line,
                ))
            })
            .collect();
        path_edges.push(edges);
    }

    let mut inner_swept: Vec<SweptWireData> = Vec::new();
    for &iw_id in &inner_wire_ids {
        inner_swept.push(sweep_wire_through_frames(
            topo,
            iw_id,
            centroid,
            initial_right,
            initial_up,
            initial_tangent,
            &frames,
            num_segments,
            is_closed,
        )?);
    }

    let mut all_faces = Vec::with_capacity(num_segments * n + if is_closed { 0 } else { 2 });

    // Start cap (open paths only — closed paths have no caps).
    if !is_closed {
        let start_inner_wires = build_inner_cap_wires(topo, &inner_swept, 0, true)?;
        let start_verts = crate::cap::ring_point_positions(topo, &ring_verts[0])?;
        let outward = crate::cap::outward_normal(&start_verts, -frames[0].tangent)?;
        all_faces.push(crate::cap::build_cap_face(
            topo,
            &ring_edges[0],
            start_inner_wires,
            &start_verts,
            outward,
            true,
        )?);
    }

    // Side faces: one quad per profile-edge × path-segment.
    // Winding: ring_edge[seg][i](fwd) → path_edge[seg][next_i](fwd) →
    //          ring_edge[seg+1][i](rev) → path_edge[seg][i](rev).
    for seg in 0..num_segments {
        for i in 0..n {
            let next_i = (i + 1) % n;

            let p0 = topo.vertex(ring_verts[seg][i])?.point();
            let p1 = topo.vertex(ring_verts[seg][next_i])?.point();
            let p_next = topo.vertex(ring_verts[seg + 1][i])?.point();
            let edge_dir = p1 - p0;
            let path_dir = p_next - p0;
            let side_normal = edge_dir
                .cross(path_dir)
                .normalize()
                .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
            let side_d = dot_normal_point(side_normal, p0);

            let side_wire = Wire::new(
                vec![
                    OrientedEdge::new(ring_edges[seg][i], true),
                    OrientedEdge::new(path_edges[seg][next_i], true),
                    OrientedEdge::new(ring_edges[seg + 1][i], false),
                    OrientedEdge::new(path_edges[seg][i], false),
                ],
                true,
            )
            .map_err(crate::OperationsError::Topology)?;

            let side_wire_id = topo.add_wire(side_wire);
            let side_face = topo.add_face(Face::new(
                side_wire_id,
                vec![],
                FaceSurface::Plane {
                    normal: side_normal,
                    d: side_d,
                },
            ));
            all_faces.push(side_face);
        }
    }

    for iwd in &inner_swept {
        let inner_faces = build_inner_side_faces(topo, iwd, num_segments)?;
        all_faces.extend(inner_faces);
    }

    // End cap (open paths only).
    if !is_closed {
        let end_inner_wires = build_inner_cap_wires(topo, &inner_swept, num_segments, false)?;
        let end_verts = crate::cap::ring_point_positions(topo, &ring_verts[num_segments])?;
        let outward = crate::cap::outward_normal(&end_verts, frames[num_segments].tangent)?;
        all_faces.push(crate::cap::build_cap_face(
            topo,
            &ring_edges[num_segments],
            end_inner_wires,
            &end_verts,
            outward,
            false,
        )?);
    }

    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    let solid = topo.add_solid(Solid::new(shell_id, vec![]));

    Ok(solid)
}

/// Sweep a face along a path with smooth NURBS side surfaces.
///
/// Like [`sweep`], but produces a single NURBS surface per edge strip
/// instead of `N` flat quads. The side surfaces interpolate through all
/// ring positions using tensor-product surface fitting, giving smooth
/// geometry that tessellates to arbitrary quality.
///
/// This produces `n + 2` faces (n NURBS sides + 2 caps) instead of
/// `num_segments × n + 2` flat faces, making the topology significantly
/// more compact while improving geometric quality.
///
/// The profile surface may be planar or curved — only its boundary is used; the
/// end caps are filled from that boundary (planar ring → `Plane`, non-planar
/// 4-edge ring → bilinear patch).
///
/// # Errors
///
/// Returns an error if the path has fewer than 2 control points, surface
/// fitting fails, or a section boundary is non-planar with more than four edges
/// or with holes (unsupported cap).
#[allow(clippy::too_many_lines)]
pub fn sweep_smooth(
    topo: &mut Topology,
    profile: FaceId,
    path: &NurbsCurve,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if path.control_points().len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep path must have at least 2 control points".into(),
        });
    }

    let face_data = topo.face(profile)?;
    let input_wire_id = face_data.outer_wire();
    let inner_wire_ids_smooth: Vec<brepkit_topology::wire::WireId> =
        face_data.inner_wires().to_vec();

    // Detect closed vs degenerate paths.
    let start_end_coincide_smooth = tol.approx_eq(
        (path.evaluate(1.0) - path.evaluate(0.0)).length_squared(),
        0.0,
    );
    let is_closed = if start_end_coincide_smooth {
        let mid = path.evaluate(0.5);
        let mid_dist_sq = (mid - path.evaluate(0.0)).length_squared();
        if tol.approx_eq(mid_dist_sq, 0.0) {
            return Err(crate::OperationsError::InvalidInput {
                reason: "sweep path has zero length".into(),
            });
        }
        true
    } else {
        false
    };

    let input_wire = topo.wire(input_wire_id)?;
    let original_oriented: Vec<_> = input_wire.edges().to_vec();

    if original_oriented.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep profile has no edges".into(),
        });
    }

    let input_oriented = crate::extrude::maybe_split_closed_wire(
        topo,
        &original_oriented,
        tol.linear,
        crate::extrude::DEFAULT_DEFLECTION,
    )?;
    let n = input_oriented.len();

    let mut input_verts: Vec<VertexId> = Vec::with_capacity(n);
    for oe in &input_oriented {
        let edge = topo.edge(oe.edge())?;
        let vid = oe.oriented_start(edge);
        input_verts.push(vid);
    }

    let mut input_positions: Vec<Point3> = input_verts
        .iter()
        .map(|&vid| {
            topo.vertex(vid)
                .map(brepkit_topology::vertex::Vertex::point)
        })
        .collect::<Result<_, _>>()?;

    // Profile normal from the section boundary (planar or non-planar alike); the
    // gate that required a planar surface is gone.
    let mut input_normal = crate::winding::newell_normal(&input_positions)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));

    // Ensure CCW winding relative to path direction (same fix as sweep()).
    let path_tangent_0 = path.tangent(0.0)?;
    if crate::winding::ensure_ccw_positions(&mut input_positions, path_tangent_0) {
        input_normal = -input_normal;
    }

    let centroid = crate::winding::polygon_centroid(&input_positions);

    let num_segments = (path.control_points().len() * 2).max(4);
    let up_hint = orthogonalize(input_normal, path_tangent_0);
    let frames = compute_frames(path, num_segments, up_hint, is_closed)?;

    // Map the profile's 2D shape onto the path's perpendicular plane using the
    // profile's own basis, so a profile whose plane is not perpendicular to the
    // path is still swept perpendicular (rather than collapsing to a ribbon).
    let (initial_right, initial_up, initial_tangent) = profile_basis(input_normal);

    // Compute all ring positions (without allocating vertices yet).
    let num_rings = frames.len();
    let ring_positions: Vec<Vec<Point3>> = frames
        .iter()
        .map(|frame| {
            input_positions
                .iter()
                .map(|&pos| {
                    transform_point(
                        pos,
                        centroid,
                        initial_right,
                        initial_up,
                        initial_tangent,
                        frame,
                    )
                })
                .collect()
        })
        .collect();

    // For closed paths, delegate to non-smooth sweep which already handles
    // closed topology properly. Smooth closed sweep (periodic NURBS surfaces)
    // is more complex and can be added later.
    if is_closed {
        return sweep(topo, profile, path);
    }

    // Create vertices for first and last rings only (for edge topology).
    let first_ring: Vec<VertexId> = ring_positions[0]
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, tol.linear)))
        .collect();
    let last_ring: Vec<VertexId> = ring_positions[num_rings - 1]
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, tol.linear)))
        .collect();

    let first_ring_edges: Vec<_> = (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            topo.add_edge(Edge::new(first_ring[i], first_ring[next], EdgeCurve::Line))
        })
        .collect();
    let last_ring_edges: Vec<_> = (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            topo.add_edge(Edge::new(last_ring[i], last_ring[next], EdgeCurve::Line))
        })
        .collect();

    let mut inner_swept_smooth: Vec<SweptWireData> = Vec::new();
    for &iw_id in &inner_wire_ids_smooth {
        inner_swept_smooth.push(sweep_wire_through_frames(
            topo,
            iw_id,
            centroid,
            initial_right,
            initial_up,
            initial_tangent,
            &frames,
            num_segments,
            is_closed,
        )?);
    }

    let mut all_faces = Vec::with_capacity(n + 2);

    let start_inner_wires_smooth = build_inner_cap_wires(topo, &inner_swept_smooth, 0, true)?;
    let start_outward = crate::cap::outward_normal(&ring_positions[0], -frames[0].tangent)?;
    all_faces.push(crate::cap::build_cap_face(
        topo,
        &first_ring_edges,
        start_inner_wires_smooth,
        &ring_positions[0],
        start_outward,
        true,
    )?);

    // NURBS side faces: one surface per edge index spanning all rings.
    let degree_u = (num_rings - 1).min(3);
    let degree_v = 1;

    // Rail edges, one per profile vertex (the swept-vertex path from the first
    // to the last ring), shared between the two adjacent side faces so the shell
    // is manifold.
    let rail_edges: Vec<_> = (0..n)
        .map(|i| topo.add_edge(Edge::new(first_ring[i], last_ring[i], EdgeCurve::Line)))
        .collect();

    for i in 0..n {
        let next_i = (i + 1) % n;

        // Build interpolation grid: rings × 2 (edge endpoints).
        let grid: Vec<Vec<Point3>> = (0..num_rings)
            .map(|k| vec![ring_positions[k][i], ring_positions[k][next_i]])
            .collect();

        let surface =
            interpolate_surface(&grid, degree_u, degree_v).map_err(crate::OperationsError::Math)?;

        let side_wire = Wire::new(
            vec![
                OrientedEdge::new(first_ring_edges[i], true),
                OrientedEdge::new(rail_edges[next_i], true),
                OrientedEdge::new(last_ring_edges[i], false),
                OrientedEdge::new(rail_edges[i], false),
            ],
            true,
        )
        .map_err(crate::OperationsError::Topology)?;

        let side_wire_id = topo.add_wire(side_wire);
        // The surface normal ∂u×∂v = path×edge points *into* the swept body for
        // a CCW profile; reverse the face when it opposes the geometric outward
        // so the solid's faces are consistently outward. Probe everything at the
        // start ring (u=0), mid-edge (v=0.5): the edge tangent crossed with the
        // local path direction *at the edge midpoint*. If they are parallel (the
        // edge runs along the path) the cross product vanishes, so fall back to
        // the radial direction from the ring centroid.
        let mid0 = ring_positions[0][i] + (ring_positions[0][next_i] - ring_positions[0][i]) * 0.5;
        let mid1 = ring_positions[1][i] + (ring_positions[1][next_i] - ring_positions[1][i]) * 0.5;
        let edge_dir = ring_positions[0][next_i] - ring_positions[0][i];
        let mut outward = edge_dir.cross(mid1 - mid0);
        if outward.length() < tol.linear {
            outward = mid0 - crate::winding::polygon_centroid(&ring_positions[0]);
        }
        let reversed = surface
            .normal(0.0, 0.5)
            .map(|nrm| nrm.dot(outward) < 0.0)
            .unwrap_or(false);
        let mut face = Face::new(side_wire_id, vec![], FaceSurface::Nurbs(surface));
        if reversed {
            face.set_reversed(true);
        }
        all_faces.push(topo.add_face(face));
    }

    for iwd in &inner_swept_smooth {
        let inner_faces = build_inner_side_faces(topo, iwd, num_segments)?;
        all_faces.extend(inner_faces);
    }

    let end_inner_wires_smooth =
        build_inner_cap_wires(topo, &inner_swept_smooth, num_segments, false)?;
    let end_outward =
        crate::cap::outward_normal(&ring_positions[num_rings - 1], frames[num_segments].tangent)?;
    all_faces.push(crate::cap::build_cap_face(
        topo,
        &last_ring_edges,
        end_inner_wires_smooth,
        &ring_positions[num_rings - 1],
        end_outward,
        false,
    )?);

    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

/// Contact mode for advanced sweep operations.
///
/// Determines how the profile is oriented as it moves along the path.
#[derive(Debug, Clone, Copy, Default)]
pub enum SweepContactMode {
    /// Rotation-minimizing frames (default, twist-free).
    #[default]
    RotationMinimizing,
    /// Fixed orientation: profile does not rotate along the path.
    Fixed,
    /// Profile normal stays aligned to a given direction.
    ConstantNormal(Vec3),
}

/// Corner handling mode for sweep operations.
///
/// When a path has sharp corners (tangent discontinuities), this controls
/// how the swept solid handles the transition between path segments.
#[derive(Debug, Clone, Copy, Default)]
pub enum SweepCornerMode {
    /// Smooth interpolation through corners (default behavior).
    /// May produce self-intersections at sharp corners.
    #[default]
    Smooth,
    /// Miter joints at sharp corners.
    /// Each path segment is swept independently, and adjacent segments
    /// are joined by miter faces on the bisector plane of the two
    /// tangent directions. Produces clean geometry at sharp turns.
    Miter,
    /// At each kink, insert a smooth fillet blend by rotating the profile
    /// through the turn angle in small angular steps.
    Round,
}

/// Options for advanced sweep operations.
#[derive(Default)]
pub struct SweepOptions {
    /// Contact mode for profile orientation.
    pub contact_mode: SweepContactMode,
    /// Corner handling mode for path kinks.
    pub corner_mode: SweepCornerMode,
    /// Scale function: maps path parameter `t ∈ [0, 1]` to a scale factor.
    /// `None` means uniform scale (1.0 everywhere).
    pub scale_law: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
    /// Number of path segments (0 = auto from control point count).
    pub segments: usize,
    /// Auxiliary spine (guide curve). When set, the profile is oriented so its
    /// up-vector points toward this curve at each path parameter — a guided
    /// (two-rail) sweep — overriding `contact_mode`. Sampled at the same
    /// parameter as the main path.
    pub aux_spine: Option<NurbsCurve>,
}

impl std::fmt::Debug for SweepOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SweepOptions")
            .field("contact_mode", &self.contact_mode)
            .field("corner_mode", &self.corner_mode)
            .field(
                "scale_law",
                &self.scale_law.as_ref().map(|_| "fn(f64)->f64"),
            )
            .field("segments", &self.segments)
            .field("aux_spine", &self.aux_spine.as_ref().map(|_| "NurbsCurve"))
            .finish()
    }
}

/// Sweep a face along a path with advanced options.
///
/// Supports scaling laws (tapered sweep) and multiple contact modes.
///
/// # Errors
///
/// Returns errors for invalid input (see [`sweep`]).
#[allow(clippy::too_many_lines)]
pub fn sweep_with_options(
    topo: &mut Topology,
    profile: FaceId,
    path: &NurbsCurve,
    options: &SweepOptions,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if path.control_points().len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep path must have at least 2 control points".into(),
        });
    }

    // Dispatch to miter sweep for Miter corners, but only if the path
    // actually has kinks. This avoids entering sweep_miter just to fall
    // back to smooth sweep, which would drop the caller's scale_law
    // (Box<dyn Fn> is not Clone).
    // Round is not yet implemented — fall through to smooth sweep as the safe default.
    // TODO: implement proper fillet-blend round corners.
    if matches!(options.corner_mode, SweepCornerMode::Miter) && !detect_kinks(path).is_empty() {
        return sweep_miter(topo, profile, path, options);
    }

    let face_data = topo.face(profile)?;
    let input_wire_id = face_data.outer_wire();
    let inner_wire_ids_opts: Vec<brepkit_topology::wire::WireId> = face_data.inner_wires().to_vec();

    // Detect closed paths — delegate to basic sweep for now since advanced
    // options (scale laws, contact modes) with closed paths needs more work.
    let start_end_coincide_opts = tol.approx_eq(
        (path.evaluate(1.0) - path.evaluate(0.0)).length_squared(),
        0.0,
    );
    if start_end_coincide_opts {
        let mid = path.evaluate(0.5);
        let mid_dist_sq = (mid - path.evaluate(0.0)).length_squared();
        if tol.approx_eq(mid_dist_sq, 0.0) {
            return Err(crate::OperationsError::InvalidInput {
                reason: "sweep path has zero length (start and end coincide)".into(),
            });
        }
        return sweep(topo, profile, path);
    }

    // A straight perpendicular sweep is a prism — build it exactly via extrude.
    // Only when no scale law or guide spine applies, since either makes the
    // result non-prismatic.
    if options.scale_law.is_none()
        && options.aux_spine.is_none()
        && let Some(solid) = try_straight_extrude(topo, profile, path)?
    {
        return Ok(solid);
    }

    let input_wire = topo.wire(input_wire_id)?;
    let original_oriented: Vec<_> = input_wire.edges().to_vec();

    if original_oriented.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep profile has no edges".into(),
        });
    }

    let input_oriented = crate::extrude::maybe_split_closed_wire(
        topo,
        &original_oriented,
        tol.linear,
        crate::extrude::DEFAULT_DEFLECTION,
    )?;
    let n = input_oriented.len();

    let mut input_verts: Vec<VertexId> = Vec::with_capacity(n);
    for oe in &input_oriented {
        let edge = topo.edge(oe.edge())?;
        let vid = oe.oriented_start(edge);
        input_verts.push(vid);
    }

    let mut input_positions: Vec<Point3> = input_verts
        .iter()
        .map(|&vid| {
            topo.vertex(vid)
                .map(brepkit_topology::vertex::Vertex::point)
        })
        .collect::<Result<_, _>>()?;

    // Profile normal from the section boundary (planar or non-planar alike); the
    // gate that required a planar surface is gone.
    let mut input_normal = crate::winding::newell_normal(&input_positions)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));

    // Ensure CCW winding relative to path direction (same fix as sweep()).
    let path_tangent_0 = path.tangent(0.0)?;
    if crate::winding::ensure_ccw_positions(&mut input_positions, path_tangent_0) {
        input_normal = -input_normal;
    }

    let centroid = crate::winding::polygon_centroid(&input_positions);

    let num_segments = if options.segments > 0 {
        options.segments
    } else {
        (path.control_points().len() * 2).max(4)
    };

    // Compute frames based on contact mode (open paths only at this point).
    let frames: Vec<Frame> = if let Some(aux) = options.aux_spine.as_ref() {
        // Guided (two-rail) sweep: orient the profile so its up-vector points
        // toward the auxiliary spine at each path parameter. The tangent still
        // follows the main path, so this overrides `contact_mode`.
        let mut out = Vec::with_capacity(num_segments + 1);
        let mut prev_up: Option<Vec3> = None;
        for k in 0..=num_segments {
            #[allow(clippy::cast_precision_loss)]
            let t = k as f64 / num_segments as f64;
            let origin = path.evaluate(t);
            let tangent = path.tangent(t).unwrap_or(path_tangent_0);
            let guide_dir = aux.evaluate(t) - origin;
            // Where the guide momentarily coincides with the spine the up-vector
            // is undefined; carry the previous frame's up (re-orthogonalized) so
            // orientation stays continuous instead of snapping to a world axis.
            let up = if guide_dir.length() < 1e-9 {
                let seed = prev_up.unwrap_or_else(|| pick_reference_axis(tangent));
                orthogonalize(seed, tangent)
            } else {
                orthogonalize(guide_dir, tangent)
            };
            prev_up = Some(up);
            out.push(Frame {
                origin,
                tangent,
                up,
                right: tangent.cross(up),
            });
        }
        out
    } else {
        match options.contact_mode {
            SweepContactMode::RotationMinimizing => {
                let up_hint = orthogonalize(input_normal, path_tangent_0);
                compute_frames(path, num_segments, up_hint, false)?
            }
            SweepContactMode::Fixed => {
                // Fixed: use the same orientation at every point
                let tangent0 = path_tangent_0;
                let up = orthogonalize(input_normal, tangent0);
                let right = tangent0.cross(up);

                (0..=num_segments)
                    .map(|k| {
                        #[allow(clippy::cast_precision_loss)]
                        let t = k as f64 / num_segments as f64;
                        Frame {
                            origin: path.evaluate(t),
                            tangent: path.tangent(t).unwrap_or(tangent0),
                            up,
                            right,
                        }
                    })
                    .collect()
            }
            SweepContactMode::ConstantNormal(normal_dir) => {
                // Constant normal: up vector stays aligned to normal_dir
                (0..=num_segments)
                    .map(|k| {
                        #[allow(clippy::cast_precision_loss)]
                        let t = k as f64 / num_segments as f64;
                        let tangent = path.tangent(t).unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                        let up = orthogonalize(normal_dir, tangent);
                        let right = tangent.cross(up);
                        Frame {
                            origin: path.evaluate(t),
                            tangent,
                            up,
                            right,
                        }
                    })
                    .collect()
            }
        }
    };

    // Map the profile's 2D shape onto the path's perpendicular plane using the
    // profile's own basis, so a profile whose plane is not perpendicular to the
    // path is still swept perpendicular (rather than collapsing to a ribbon).
    let (initial_right, initial_up, initial_tangent) = profile_basis(input_normal);

    let mut ring_verts: Vec<Vec<VertexId>> = Vec::with_capacity(num_segments + 1);

    for (k, frame) in frames.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let t = k as f64 / num_segments as f64;
        let scale = options.scale_law.as_ref().map_or(1.0, |law| law(t));

        let ring: Vec<VertexId> = input_positions
            .iter()
            .map(|&pos| {
                let mut transformed = transform_point(
                    pos,
                    centroid,
                    initial_right,
                    initial_up,
                    initial_tangent,
                    frame,
                );
                // Apply scaling relative to frame origin
                if (scale - 1.0).abs() > tol.linear {
                    let offset = transformed - frame.origin;
                    transformed = frame.origin
                        + Vec3::new(offset.x() * scale, offset.y() * scale, offset.z() * scale);
                }
                topo.add_vertex(Vertex::new(transformed, tol.linear))
            })
            .collect();
        ring_verts.push(ring);
    }

    // Build edges, faces, and assemble (same as basic sweep)
    let mut ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> =
        Vec::with_capacity(num_segments + 1);
    for ring in &ring_verts {
        let edges: Vec<_> = (0..n)
            .map(|i| {
                let next = (i + 1) % n;
                topo.add_edge(Edge::new(ring[i], ring[next], EdgeCurve::Line))
            })
            .collect();
        ring_edges.push(edges);
    }

    let mut path_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = Vec::with_capacity(num_segments);
    for seg in 0..num_segments {
        let edges: Vec<_> = (0..n)
            .map(|i| {
                topo.add_edge(Edge::new(
                    ring_verts[seg][i],
                    ring_verts[seg + 1][i],
                    EdgeCurve::Line,
                ))
            })
            .collect();
        path_edges.push(edges);
    }

    // Closed paths are delegated to sweep() earlier, so is_closed is always false here.
    let mut inner_swept_opts: Vec<SweptWireData> = Vec::new();
    for &iw_id in &inner_wire_ids_opts {
        inner_swept_opts.push(sweep_wire_through_frames(
            topo,
            iw_id,
            centroid,
            initial_right,
            initial_up,
            initial_tangent,
            &frames,
            num_segments,
            false,
        )?);
    }

    let mut all_faces = Vec::with_capacity(num_segments * n + 2);

    let start_inner_wires_opts = build_inner_cap_wires(topo, &inner_swept_opts, 0, true)?;
    let start_verts = crate::cap::ring_point_positions(topo, &ring_verts[0])?;
    let start_outward = crate::cap::outward_normal(&start_verts, -frames[0].tangent)?;
    all_faces.push(crate::cap::build_cap_face(
        topo,
        &ring_edges[0],
        start_inner_wires_opts,
        &start_verts,
        start_outward,
        true,
    )?);

    for seg in 0..num_segments {
        for i in 0..n {
            let next_i = (i + 1) % n;
            let p0 = topo.vertex(ring_verts[seg][i])?.point();
            let p1 = topo.vertex(ring_verts[seg][next_i])?.point();
            let p_next = topo.vertex(ring_verts[seg + 1][i])?.point();
            let edge_dir = p1 - p0;
            let path_dir = p_next - p0;
            let side_normal = edge_dir
                .cross(path_dir)
                .normalize()
                .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
            let side_d = dot_normal_point(side_normal, p0);

            let side_wire = Wire::new(
                vec![
                    OrientedEdge::new(ring_edges[seg][i], true),
                    OrientedEdge::new(path_edges[seg][next_i], true),
                    OrientedEdge::new(ring_edges[seg + 1][i], false),
                    OrientedEdge::new(path_edges[seg][i], false),
                ],
                true,
            )
            .map_err(crate::OperationsError::Topology)?;

            let side_wire_id = topo.add_wire(side_wire);
            let side_face = topo.add_face(Face::new(
                side_wire_id,
                vec![],
                FaceSurface::Plane {
                    normal: side_normal,
                    d: side_d,
                },
            ));
            all_faces.push(side_face);
        }
    }

    for iwd in &inner_swept_opts {
        let inner_faces = build_inner_side_faces(topo, iwd, num_segments)?;
        all_faces.extend(inner_faces);
    }

    let end_inner_wires_opts = build_inner_cap_wires(topo, &inner_swept_opts, num_segments, false)?;
    let end_verts = crate::cap::ring_point_positions(topo, &ring_verts[num_segments])?;
    let end_outward = crate::cap::outward_normal(&end_verts, frames[num_segments].tangent)?;
    all_faces.push(crate::cap::build_cap_face(
        topo,
        &ring_edges[num_segments],
        end_inner_wires_opts,
        &end_verts,
        end_outward,
        false,
    )?);

    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

/// Detect kink parameters in a NURBS path.
///
/// A kink is an internal knot where the tangent direction changes
/// discontinuously (C0 but not C1 continuity). For a degree-`p` curve,
/// this happens at knots with multiplicity >= `p`. For degree-1 (polyline)
/// paths, every internal knot is a kink where the line direction changes.
///
/// Returns the kink parameter values (not including the path endpoints).
fn detect_kinks(path: &NurbsCurve) -> Vec<f64> {
    /// Small epsilon for comparing knot parameter values (dimensionless).
    const KNOT_EPS: f64 = 1e-10;
    /// Angular threshold for tangent discontinuity detection (~1 degree).
    const KINK_ANGLE_RAD: f64 = 0.0175;

    let p = path.degree();
    let knots = path.knots();
    let (u_min, u_max) = path.domain();

    let mut kinks = Vec::new();
    let mut i = 0;
    while i < knots.len() {
        let u = knots[i];

        if u <= u_min + KNOT_EPS || u >= u_max - KNOT_EPS {
            i += 1;
            continue;
        }

        let mut mult = 1;
        while i + mult < knots.len() && (knots[i + mult] - u).abs() < KNOT_EPS {
            mult += 1;
        }

        // For a degree-p curve, a knot with multiplicity m gives C^(p-m)
        // continuity. C0 (position only) occurs when m >= p. For degree 1
        // (polyline), every internal knot has multiplicity 1 == degree,
        // so every junction is a kink.
        if mult >= p {
            // Verify there's actually a tangent discontinuity by checking
            // the tangent just before and just after the knot.
            let eps = 1e-8;
            if let (Ok(t_before), Ok(t_after)) = (path.tangent(u - eps), path.tangent(u + eps)) {
                let dot = t_before.dot(t_after).clamp(-1.0, 1.0);
                // Compare using angular threshold: if the angle between
                // tangents exceeds ~1 degree, it's a kink.
                let angle = dot.acos();
                if angle > KINK_ANGLE_RAD {
                    kinks.push(u);
                }
            }
        }

        i += mult;
    }

    kinks
}

/// Sweep a face along a path with miter joints at sharp corners.
///
/// Detects kinks (tangent discontinuities) in the path, sweeps each
/// smooth segment independently, and joins them with miter faces on
/// the bisector plane between adjacent tangent directions.
///
/// # Errors
///
/// Returns an error if the profile is invalid, path has fewer than 2
/// control points, or the path has no kinks (falls back to smooth sweep).
#[allow(clippy::too_many_lines)]
fn sweep_miter(
    topo: &mut Topology,
    profile: FaceId,
    path: &NurbsCurve,
    options: &SweepOptions,
) -> Result<SolidId, crate::OperationsError> {
    use brepkit_math::nurbs::knot_ops::curve_split;

    let tol = Tolerance::new();

    // Detect kinks in the path. The caller (sweep_with_options) already
    // checks for empty kinks before dispatching here.
    let kinks = detect_kinks(path);
    debug_assert!(
        !kinks.is_empty(),
        "sweep_miter should only be called when the path has kinks"
    );

    let face_data = topo.face(profile)?;
    let mut input_normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "sweep of non-planar faces is not supported".into(),
            });
        }
    };
    let input_wire_id = face_data.outer_wire();
    let inner_wire_ids: Vec<brepkit_topology::wire::WireId> = face_data.inner_wires().to_vec();

    let input_wire = topo.wire(input_wire_id)?;
    let original_oriented: Vec<_> = input_wire.edges().to_vec();
    if original_oriented.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep profile has no edges".into(),
        });
    }
    let input_oriented = crate::extrude::maybe_split_closed_wire(
        topo,
        &original_oriented,
        tol.linear,
        crate::extrude::DEFAULT_DEFLECTION,
    )?;
    let n = input_oriented.len();

    let mut input_verts: Vec<VertexId> = Vec::with_capacity(n);
    for oe in &input_oriented {
        let edge = topo.edge(oe.edge())?;
        let vid = oe.oriented_start(edge);
        input_verts.push(vid);
    }
    let mut input_positions: Vec<Point3> = input_verts
        .iter()
        .map(|&vid| {
            topo.vertex(vid)
                .map(brepkit_topology::vertex::Vertex::point)
        })
        .collect::<Result<_, _>>()?;

    // Ensure CCW winding relative to the path direction at domain start.
    let (domain_start, _domain_end) = path.domain();
    let path_tangent_0 = path.tangent(domain_start)?;
    if crate::winding::ensure_ccw_positions(&mut input_positions, path_tangent_0) {
        input_normal = -input_normal;
    }

    let centroid = crate::winding::polygon_centroid(&input_positions);

    // Split the path at each kink to get smooth sub-curves.
    let mut sub_paths: Vec<NurbsCurve> = Vec::with_capacity(kinks.len() + 1);
    let mut remaining = path.clone();
    let mut offset = domain_start;

    for &kink_u in &kinks {
        // The kink parameter is in the original domain. After splitting,
        // the remaining curve's domain starts at the split point.
        // curve_split takes a parameter in the current curve's domain.
        let split_u = kink_u - offset + remaining.domain().0;
        let (left, right) = curve_split(&remaining, split_u)?;
        sub_paths.push(left);
        offset = kink_u;
        remaining = right;
    }
    sub_paths.push(remaining);

    let mut all_faces: Vec<FaceId> = Vec::new();

    // Track the ring vertices at the end of each segment / start of the next
    // so we can connect them via miter faces.
    let mut prev_end_ring: Option<Vec<VertexId>> = None;
    let mut prev_end_ring_edges: Option<Vec<brepkit_topology::edge::EdgeId>> = None;

    for (seg_idx, sub_path) in sub_paths.iter().enumerate() {
        let is_first = seg_idx == 0;
        let is_last = seg_idx == sub_paths.len() - 1;

        let sub_tangent_0 = sub_path.tangent(sub_path.domain().0)?;
        let up_hint = orthogonalize(input_normal, sub_tangent_0);

        let num_segments = if options.segments > 0 {
            options.segments
        } else {
            (sub_path.control_points().len() * 2).max(4)
        };

        let sub_frames = match options.contact_mode {
            SweepContactMode::RotationMinimizing => {
                compute_frames(sub_path, num_segments, up_hint, false)?
            }
            SweepContactMode::Fixed => {
                let up = orthogonalize(input_normal, sub_tangent_0);
                let right = sub_tangent_0.cross(up);
                (0..=num_segments)
                    .map(|k| {
                        let (u0, u1) = sub_path.domain();
                        #[allow(clippy::cast_precision_loss)]
                        let t = u0 + (u1 - u0) * (k as f64 / num_segments as f64);
                        Frame {
                            origin: sub_path.evaluate(t),
                            tangent: sub_path.tangent(t).unwrap_or(sub_tangent_0),
                            up,
                            right,
                        }
                    })
                    .collect()
            }
            SweepContactMode::ConstantNormal(normal_dir) => (0..=num_segments)
                .map(|k| {
                    let (u0, u1) = sub_path.domain();
                    #[allow(clippy::cast_precision_loss)]
                    let t = u0 + (u1 - u0) * (k as f64 / num_segments as f64);
                    let tangent = sub_path.tangent(t).unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                    let up = orthogonalize(normal_dir, tangent);
                    let right = tangent.cross(up);
                    Frame {
                        origin: sub_path.evaluate(t),
                        tangent,
                        up,
                        right,
                    }
                })
                .collect(),
        };

        let initial_right = sub_frames[0].right;
        let initial_up = sub_frames[0].up;
        let initial_tangent = sub_frames[0].tangent;

        // Create ring vertices for this segment.
        let mut ring_verts: Vec<Vec<VertexId>> = Vec::with_capacity(num_segments + 1);
        for frame in &sub_frames {
            let ring: Vec<VertexId> = input_positions
                .iter()
                .map(|&pos| {
                    let transformed = transform_point(
                        pos,
                        centroid,
                        initial_right,
                        initial_up,
                        initial_tangent,
                        frame,
                    );
                    topo.add_vertex(Vertex::new(transformed, tol.linear))
                })
                .collect();
            ring_verts.push(ring);
        }

        // If we have a previous segment's end ring, replace this segment's
        // start ring with the miter ring (computed from bisector plane).
        #[allow(clippy::useless_let_if_seq)]
        let mut miter_ring_edges_for_reuse: Option<Vec<brepkit_topology::edge::EdgeId>> = None;
        if let Some(ref prev_ring) = prev_end_ring {
            // The kink point is where the previous segment ended / this one starts.
            let kink_idx = seg_idx - 1;
            let kink_u = kinks[kink_idx];
            let eps = 1e-8;

            // Get tangents on either side of the kink.
            let t_before = path.tangent(kink_u - eps)?;
            let t_after = path.tangent(kink_u + eps)?;

            // Bisector direction: average of the two tangent directions.
            let bisector = (t_before + t_after).normalize().unwrap_or(t_before);

            // Miter plane: passes through the kink point with normal = bisector.
            let kink_point = path.evaluate(kink_u);

            // Project the profile ring onto the miter plane.
            // For each profile vertex, find where the line from the previous
            // segment's end position to the current segment's start position
            // intersects the bisector plane.
            let miter_ring: Vec<VertexId> = (0..n)
                .map(|i| {
                    let prev_pos = topo
                        .vertex(prev_ring[i])
                        .map(brepkit_topology::vertex::Vertex::point)
                        .unwrap_or(kink_point);
                    let curr_pos = topo
                        .vertex(ring_verts[0][i])
                        .map(brepkit_topology::vertex::Vertex::point)
                        .unwrap_or(kink_point);

                    // Ray-plane intersection: find t where
                    // prev_pos + t*(curr_pos - prev_pos) lies on the bisector plane.
                    let ray_dir = curr_pos - prev_pos;
                    let denom = bisector.dot(ray_dir);
                    let miter_pos = if denom.abs() > tol.linear {
                        let d = bisector.dot(Vec3::new(
                            kink_point.x() - prev_pos.x(),
                            kink_point.y() - prev_pos.y(),
                            kink_point.z() - prev_pos.z(),
                        ));
                        let t_intersect = d / denom;
                        prev_pos + ray_dir * t_intersect
                    } else {
                        // Ray parallel to plane — use midpoint.
                        Point3::new(
                            (prev_pos.x() + curr_pos.x()) * 0.5,
                            (prev_pos.y() + curr_pos.y()) * 0.5,
                            (prev_pos.z() + curr_pos.z()) * 0.5,
                        )
                    };
                    topo.add_vertex(Vertex::new(miter_pos, tol.linear))
                })
                .collect();

            let miter_ring_edges: Vec<brepkit_topology::edge::EdgeId> = (0..n)
                .map(|i| {
                    let next = (i + 1) % n;
                    topo.add_edge(Edge::new(miter_ring[i], miter_ring[next], EdgeCurve::Line))
                })
                .collect();

            // Build miter face connecting the previous segment's end to
            // the miter ring. The miter face is on the bisector plane.
            let prev_ring_edges = prev_end_ring_edges.as_ref().ok_or_else(|| {
                crate::OperationsError::InvalidInput {
                    reason: "internal error: missing previous ring edges".into(),
                }
            })?;

            let prev_to_miter_path_edges: Vec<brepkit_topology::edge::EdgeId> = (0..n)
                .map(|i| topo.add_edge(Edge::new(prev_ring[i], miter_ring[i], EdgeCurve::Line)))
                .collect();

            for i in 0..n {
                let next_i = (i + 1) % n;

                let p0 = topo.vertex(prev_ring[i])?.point();
                let p1 = topo.vertex(prev_ring[next_i])?.point();
                let p_next = topo.vertex(miter_ring[i])?.point();
                let edge_dir = p1 - p0;
                let path_dir = p_next - p0;
                let side_normal = edge_dir
                    .cross(path_dir)
                    .normalize()
                    .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                let side_d = dot_normal_point(side_normal, p0);

                let side_wire = Wire::new(
                    vec![
                        OrientedEdge::new(prev_ring_edges[i], true),
                        OrientedEdge::new(prev_to_miter_path_edges[next_i], true),
                        OrientedEdge::new(miter_ring_edges[i], false),
                        OrientedEdge::new(prev_to_miter_path_edges[i], false),
                    ],
                    true,
                )
                .map_err(crate::OperationsError::Topology)?;

                let side_wire_id = topo.add_wire(side_wire);
                all_faces.push(topo.add_face(Face::new(
                    side_wire_id,
                    vec![],
                    FaceSurface::Plane {
                        normal: side_normal,
                        d: side_d,
                    },
                )));
            }

            // Replace this segment's start ring with the miter ring so the
            // next segment's side faces connect miter→ring[1]. No separate
            // miter cap faces are needed — the transition quad faces already
            // connect prev_end_ring→miter_ring.
            ring_verts[0] = miter_ring;
            miter_ring_edges_for_reuse = Some(miter_ring_edges);
        }

        // Create ring edges. If the start ring was replaced by a miter ring,
        // reuse the miter_ring_edges so both the miter transition faces and
        // this segment's side faces reference the same edge entities.
        let mut ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> =
            Vec::with_capacity(num_segments + 1);
        for (ring_idx, ring) in ring_verts.iter().enumerate() {
            if ring_idx == 0 {
                if let Some(ref reused) = miter_ring_edges_for_reuse {
                    ring_edges.push(reused.clone());
                } else {
                    let edges: Vec<_> = (0..n)
                        .map(|i| {
                            let next = (i + 1) % n;
                            topo.add_edge(Edge::new(ring[i], ring[next], EdgeCurve::Line))
                        })
                        .collect();
                    ring_edges.push(edges);
                }
            } else {
                let edges: Vec<_> = (0..n)
                    .map(|i| {
                        let next = (i + 1) % n;
                        topo.add_edge(Edge::new(ring[i], ring[next], EdgeCurve::Line))
                    })
                    .collect();
                ring_edges.push(edges);
            }
        }

        let mut path_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> =
            Vec::with_capacity(num_segments);
        for seg in 0..num_segments {
            let edges: Vec<_> = (0..n)
                .map(|i| {
                    topo.add_edge(Edge::new(
                        ring_verts[seg][i],
                        ring_verts[seg + 1][i],
                        EdgeCurve::Line,
                    ))
                })
                .collect();
            path_edges.push(edges);
        }

        let mut inner_swept: Vec<SweptWireData> = Vec::new();
        for &iw_id in &inner_wire_ids {
            inner_swept.push(sweep_wire_through_frames(
                topo,
                iw_id,
                centroid,
                initial_right,
                initial_up,
                initial_tangent,
                &sub_frames,
                num_segments,
                false,
            )?);
        }

        // Start cap (only for the first segment).
        if is_first {
            let start_reversed_edges: Vec<OrientedEdge> = (0..n)
                .rev()
                .map(|i| OrientedEdge::new(ring_edges[0][i], false))
                .collect();
            let start_wire =
                Wire::new(start_reversed_edges, true).map_err(crate::OperationsError::Topology)?;
            let start_wire_id = topo.add_wire(start_wire);
            let start_inner_wires = build_inner_cap_wires(topo, &inner_swept, 0, true)?;

            let start_normal = -sub_frames[0].tangent;
            let start_d = dot_normal_point(start_normal, topo.vertex(ring_verts[0][0])?.point());
            all_faces.push(topo.add_face(Face::new(
                start_wire_id,
                start_inner_wires,
                FaceSurface::Plane {
                    normal: start_normal,
                    d: start_d,
                },
            )));
        }

        for seg in 0..num_segments {
            for i in 0..n {
                let next_i = (i + 1) % n;
                let p0 = topo.vertex(ring_verts[seg][i])?.point();
                let p1 = topo.vertex(ring_verts[seg][next_i])?.point();
                let p_next = topo.vertex(ring_verts[seg + 1][i])?.point();
                let edge_dir = p1 - p0;
                let path_dir = p_next - p0;
                let side_normal = edge_dir
                    .cross(path_dir)
                    .normalize()
                    .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
                let side_d = dot_normal_point(side_normal, p0);

                let side_wire = Wire::new(
                    vec![
                        OrientedEdge::new(ring_edges[seg][i], true),
                        OrientedEdge::new(path_edges[seg][next_i], true),
                        OrientedEdge::new(ring_edges[seg + 1][i], false),
                        OrientedEdge::new(path_edges[seg][i], false),
                    ],
                    true,
                )
                .map_err(crate::OperationsError::Topology)?;

                let side_wire_id = topo.add_wire(side_wire);
                all_faces.push(topo.add_face(Face::new(
                    side_wire_id,
                    vec![],
                    FaceSurface::Plane {
                        normal: side_normal,
                        d: side_d,
                    },
                )));
            }
        }

        for iwd in &inner_swept {
            let inner_faces = build_inner_side_faces(topo, iwd, num_segments)?;
            all_faces.extend(inner_faces);
        }

        // End cap (only for the last segment).
        if is_last {
            let end_edges: Vec<OrientedEdge> = (0..n)
                .map(|i| OrientedEdge::new(ring_edges[num_segments][i], true))
                .collect();
            let end_wire = Wire::new(end_edges, true).map_err(crate::OperationsError::Topology)?;
            let end_wire_id = topo.add_wire(end_wire);
            let end_inner_wires = build_inner_cap_wires(topo, &inner_swept, num_segments, false)?;

            let end_normal = sub_frames[num_segments].tangent;
            let end_d = dot_normal_point(
                end_normal,
                topo.vertex(ring_verts[num_segments][0])?.point(),
            );
            all_faces.push(topo.add_face(Face::new(
                end_wire_id,
                end_inner_wires,
                FaceSurface::Plane {
                    normal: end_normal,
                    d: end_d,
                },
            )));
        }

        // Save the end ring for the next segment's miter connection.
        prev_end_ring = Some(ring_verts[num_segments].clone());
        prev_end_ring_edges = Some(ring_edges[num_segments].clone());
    }

    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

/// Sweep through multiple section profiles along a `spine`, lofting the
/// positioned profiles.
///
/// Each planar profile is placed rigidly at its parameter along the spine: its
/// centroid maps to the spine point, its normal to the spine tangent, and its
/// plane to the frame's right/up plane. Orientation uses a rotation-minimizing
/// frame so profiles stay twist-free on curved spines (unlike a per-section
/// swing rotation). The placed profiles are then lofted — ruled (planar bands)
/// or smooth (NURBS).
///
/// `sections` pairs each profile face with its spine parameter in `[0, 1]`.
///
/// # Errors
///
/// Returns [`crate::OperationsError::InvalidInput`] for fewer than two
/// sections, a parameter outside `[0, 1]`, or a spine with fewer than two
/// control points; propagates loft errors otherwise. Sections may be planar or
/// non-planar (they are joined by [`crate::loft::loft`], which supports both).
pub fn multi_section_sweep(
    topo: &mut Topology,
    spine: &NurbsCurve,
    sections: &[(FaceId, f64)],
    ruled: bool,
) -> Result<SolidId, crate::OperationsError> {
    if sections.len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!(
                "multi-section sweep requires at least 2 sections, got {}",
                sections.len()
            ),
        });
    }
    if spine.control_points().len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "multi-section sweep spine must have at least 2 control points".into(),
        });
    }
    for &(_, p) in sections {
        if !(0.0..=1.0).contains(&p) {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!("section parameter {p} is outside [0, 1]"),
            });
        }
    }

    // Dense rotation-minimizing frame for twist-free orientation; the exact
    // origin/tangent per section is taken directly from the spine.
    let dense_segments: usize = 64;
    let t_start = spine.tangent(0.0)?;
    let initial_up = orthogonalize(pick_reference_axis(t_start), t_start);
    let dense = compute_frames(spine, dense_segments, initial_up, false)?;

    // The loft joins profiles in order along the spine, so place them by
    // ascending parameter regardless of the caller's ordering.
    let mut ordered: Vec<(FaceId, f64)> = sections.to_vec();
    ordered.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut placed: Vec<FaceId> = Vec::with_capacity(ordered.len());
    for (face_id, p) in ordered {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let idx = ((p * dense_segments as f64).round() as usize).min(dense_segments);
        let tangent = spine.tangent(p)?;
        let up = orthogonalize(dense[idx].up, tangent);
        let frame = Frame {
            origin: spine.evaluate(p),
            tangent,
            up,
            right: tangent.cross(up),
        };
        let mat = profile_to_frame_matrix(topo, face_id, &frame)?;
        let placed_face = crate::copy::copy_face(topo, face_id)?;
        crate::transform::transform_face(topo, placed_face, &mat)?;
        placed.push(placed_face);
    }

    if ruled {
        crate::loft::loft(topo, &placed)
    } else {
        crate::loft::loft_smooth(topo, &placed)
    }
}

/// Pick a world axis not nearly parallel to `dir`, for seeding a frame.
fn pick_reference_axis(dir: Vec3) -> Vec3 {
    if dir.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    }
}

/// Rigid transform placing a profile face into `frame`: centroid →
/// `frame.origin`, normal → tangent, in-plane axes → right/up. A planar profile
/// uses its stored plane normal; a non-planar section uses its boundary's
/// Newell normal.
fn profile_to_frame_matrix(
    topo: &Topology,
    face_id: FaceId,
    frame: &Frame,
) -> Result<Mat4, crate::OperationsError> {
    // Centroid of the sampled boundary — robust for a circle profile whose
    // outer wire is a single closed edge (where averaging start vertices would
    // collapse to the seam point).
    let boundary = crate::boolean::face_polygon(topo, face_id)?;
    if boundary.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "multi-section sweep profile has no boundary".into(),
        });
    }
    // A planar profile uses its stored plane normal; a non-planar section's
    // orientation normal is its boundary's Newell normal (the planar-only gate
    // is gone — the placed sections are joined by `loft`, which handles
    // non-planar profiles).
    let normal = match topo.face(face_id)?.surface() {
        FaceSurface::Plane { normal, .. } => normal.normalize().unwrap_or(*normal),
        _ => crate::winding::newell_normal(&boundary)
            .normalize()
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0)),
    };
    let centroid = crate::winding::polygon_centroid(&boundary);

    // Profile in-plane basis from a consistent world reference, so all profiles
    // share an orientation and the RMF alone controls twist along the spine.
    // `p_y = p_x × normal` makes (p_x, p_y, normal) left-handed to match the
    // target frame (right, up, tangent) — so R is a proper rotation (det +1)
    // and asymmetric profiles are not mirrored.
    let p_x = orthogonalize(pick_reference_axis(normal), normal);
    let p_y = p_x.cross(normal);

    // R maps the profile basis (p_x, p_y, normal) onto (right, up, tangent):
    // R = right⊗p_x + up⊗p_y + tangent⊗normal.
    let t_cols = [
        [frame.right.x(), frame.up.x(), frame.tangent.x()],
        [frame.right.y(), frame.up.y(), frame.tangent.y()],
        [frame.right.z(), frame.up.z(), frame.tangent.z()],
    ];
    let l_cols = [
        [p_x.x(), p_y.x(), normal.x()],
        [p_x.y(), p_y.y(), normal.y()],
        [p_x.z(), p_y.z(), normal.z()],
    ];
    let mut rot = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rot[i][j] = (0..3).map(|k| t_cols[i][k] * l_cols[j][k]).sum();
        }
    }

    // translation = origin − R·centroid
    let c = [centroid.x(), centroid.y(), centroid.z()];
    let o = [frame.origin.x(), frame.origin.y(), frame.origin.z()];
    let trans: [f64; 3] =
        std::array::from_fn(|i| o[i] - (0..3).map(|k| rot[i][k] * c[k]).sum::<f64>());

    Ok(Mat4([
        [rot[0][0], rot[0][1], rot[0][2], trans[0]],
        [rot[1][0], rot[1][1], rot[1][2], trans[1]],
        [rot[2][0], rot[2][1], rot[2][2], trans[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]))
}

/// Guided (two-rail) sweep of `profile` along `spine`, oriented by `aux`.
///
/// At each path parameter the profile's up-vector points toward the auxiliary
/// spine `aux`, so the profile rolls to track the guide curve rather than
/// holding a fixed or rotation-minimizing orientation. Thin wrapper over
/// [`sweep_with_options`] with `aux_spine` set.
///
/// # Errors
///
/// Propagates [`sweep_with_options`] errors (e.g. a degenerate path or an
/// unsupported non-planar cap).
pub fn sweep_guided(
    topo: &mut Topology,
    profile: FaceId,
    spine: &NurbsCurve,
    aux: NurbsCurve,
) -> Result<SolidId, crate::OperationsError> {
    sweep_with_options(
        topo,
        profile,
        spine,
        &SweepOptions {
            aux_spine: Some(aux),
            ..Default::default()
        },
    )
}

#[cfg(test)]
mod tests;
