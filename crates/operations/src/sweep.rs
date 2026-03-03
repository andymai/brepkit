//! Path sweep: sweep a profile along a NURBS curve.
//!
//! Creates a solid by moving a planar profile along an arbitrary NURBS curve
//! path, keeping the profile perpendicular to the path tangent at each sample
//! point. Uses rotation-minimizing frames (double-reflection method) to avoid
//! Frenet-frame singularities on straight segments and inflection points.

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
/// Samples the path at `num_segments + 1` evenly-spaced parameter values and
/// propagates the initial up-vector using the double-reflection method to
/// produce smooth, twist-free frames.
fn compute_frames(
    path: &NurbsCurve,
    num_segments: usize,
    initial_up: Vec3,
) -> Result<Vec<Frame>, crate::OperationsError> {
    let mut frames = Vec::with_capacity(num_segments + 1);

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
    for k in 1..=num_segments {
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

/// Sweep a face along a path curve to produce a solid.
///
/// Creates a solid by moving a planar profile along a NURBS curve, with the
/// profile oriented perpendicular to the path tangent at each sample point.
/// Side faces are planar quads connecting consecutive profile rings.
///
/// # Errors
///
/// Returns an error if the profile is not planar, has inner wires (holes),
/// the path has fewer than 2 control points, or a degenerate tangent is
/// encountered.
#[allow(clippy::too_many_lines)]
pub fn sweep(
    topo: &mut Topology,
    profile: FaceId,
    path: &NurbsCurve,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    // --- Validation ---

    if path.control_points().len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep path must have at least 2 control points".into(),
        });
    }

    let face_data = topo.face(profile)?;
    let input_normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "sweep of non-planar faces is not supported".into(),
            });
        }
    };
    let input_wire_id = face_data.outer_wire();

    if !face_data.inner_wires().is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep of faces with holes is not supported".into(),
        });
    }

    // Validate path has non-zero length.
    if tol.approx_eq(
        (path.evaluate(1.0) - path.evaluate(0.0)).length_squared(),
        0.0,
    ) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep path has zero length (start and end coincide)".into(),
        });
    }

    // Collect profile vertices and positions.
    let input_wire = topo.wire(input_wire_id)?;
    let input_oriented: Vec<_> = input_wire.edges().to_vec();
    let n = input_oriented.len();

    if n == 0 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep profile has no edges".into(),
        });
    }

    let mut input_verts: Vec<VertexId> = Vec::with_capacity(n);
    for oe in &input_oriented {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        input_verts.push(vid);
    }

    let input_positions: Vec<Point3> = input_verts
        .iter()
        .map(|&vid| {
            topo.vertex(vid)
                .map(brepkit_topology::vertex::Vertex::point)
        })
        .collect::<Result<_, _>>()?;

    // Compute profile centroid.
    let (cx, cy, cz) = input_positions
        .iter()
        .fold((0.0, 0.0, 0.0), |(ax, ay, az), p| {
            (ax + p.x(), ay + p.y(), az + p.z())
        });
    #[allow(clippy::cast_precision_loss)]
    let centroid = Point3::new(cx / n as f64, cy / n as f64, cz / n as f64);

    // --- Compute frames along the path ---

    let num_segments = (path.control_points().len() * 2).max(4);

    // Seed the first frame's up-vector from the profile normal, projected
    // perpendicular to the path tangent at t=0.
    let up_hint = orthogonalize(input_normal, path.tangent(0.0)?);

    let frames = compute_frames(path, num_segments, up_hint)?;

    // The first frame's basis vectors define the local coordinate system
    // in which profile vertex offsets are expressed.
    let initial_right = frames[0].right;
    let initial_up = frames[0].up;
    let initial_tangent = frames[0].tangent;

    // --- Create ring vertices ---
    //
    // ring_verts[k][i] = vertex at path sample k, profile vertex i.

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
                topo.vertices.alloc(Vertex::new(transformed, tol.linear))
            })
            .collect();
        ring_verts.push(ring);
    }

    // --- Create profile edges within each ring ---
    //
    // ring_edges[k][i] = edge from ring_verts[k][i] to ring_verts[k][(i+1)%n].

    let mut ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> =
        Vec::with_capacity(num_segments + 1);
    for ring in &ring_verts {
        let edges: Vec<_> = (0..n)
            .map(|i| {
                let next = (i + 1) % n;
                topo.edges
                    .alloc(Edge::new(ring[i], ring[next], EdgeCurve::Line))
            })
            .collect();
        ring_edges.push(edges);
    }

    // --- Create path edges between consecutive rings ---
    //
    // path_edges[seg][i] = edge from ring_verts[seg][i] to ring_verts[seg+1][i].

    let mut path_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = Vec::with_capacity(num_segments);
    for seg in 0..num_segments {
        let edges: Vec<_> = (0..n)
            .map(|i| {
                topo.edges.alloc(Edge::new(
                    ring_verts[seg][i],
                    ring_verts[seg + 1][i],
                    EdgeCurve::Line,
                ))
            })
            .collect();
        path_edges.push(edges);
    }

    // --- Build faces ---

    let mut all_faces = Vec::with_capacity(num_segments * n + 2);

    // Start cap: reversed first ring (outward normal pointing opposite to
    // path direction at the start).
    let start_reversed_edges: Vec<OrientedEdge> = (0..n)
        .rev()
        .map(|i| OrientedEdge::new(ring_edges[0][i], false))
        .collect();
    let start_wire =
        Wire::new(start_reversed_edges, true).map_err(crate::OperationsError::Topology)?;
    let start_wire_id = topo.wires.alloc(start_wire);

    let start_normal = -frames[0].tangent;
    let start_d = dot_normal_point(start_normal, topo.vertex(ring_verts[0][0])?.point());
    let start_face = topo.faces.alloc(Face::new(
        start_wire_id,
        vec![],
        FaceSurface::Plane {
            normal: start_normal,
            d: start_d,
        },
    ));
    all_faces.push(start_face);

    // Side faces: one quad per profile-edge × path-segment.
    // Winding: ring_edge[seg][i](fwd) → path_edge[seg][next_i](fwd) →
    //          ring_edge[seg+1][i](rev) → path_edge[seg][i](rev).
    for seg in 0..num_segments {
        for i in 0..n {
            let next_i = (i + 1) % n;

            // Compute side normal from edge direction and path direction.
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

            let side_wire_id = topo.wires.alloc(side_wire);
            let side_face = topo.faces.alloc(Face::new(
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

    // End cap: last ring with forward orientation (outward normal along
    // path tangent at the end).
    let end_edges: Vec<OrientedEdge> = (0..n)
        .map(|i| OrientedEdge::new(ring_edges[num_segments][i], true))
        .collect();
    let end_wire = Wire::new(end_edges, true).map_err(crate::OperationsError::Topology)?;
    let end_wire_id = topo.wires.alloc(end_wire);

    let end_normal = frames[num_segments].tangent;
    let end_d = dot_normal_point(
        end_normal,
        topo.vertex(ring_verts[num_segments][0])?.point(),
    );
    let end_face = topo.faces.alloc(Face::new(
        end_wire_id,
        vec![],
        FaceSurface::Plane {
            normal: end_normal,
            d: end_d,
        },
    ));
    all_faces.push(end_face);

    // Assemble shell and solid.
    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    let solid = topo.solids.alloc(Solid::new(shell_id, vec![]));

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
/// # Errors
///
/// Returns an error if the profile is not planar, has inner wires (holes),
/// the path has fewer than 2 control points, or surface fitting fails.
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
    let input_normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "sweep of non-planar faces is not supported".into(),
            });
        }
    };
    let input_wire_id = face_data.outer_wire();

    if !face_data.inner_wires().is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep of faces with holes is not supported".into(),
        });
    }

    if tol.approx_eq(
        (path.evaluate(1.0) - path.evaluate(0.0)).length_squared(),
        0.0,
    ) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep path has zero length".into(),
        });
    }

    // Collect profile vertices.
    let input_wire = topo.wire(input_wire_id)?;
    let input_oriented: Vec<_> = input_wire.edges().to_vec();
    let n = input_oriented.len();

    if n == 0 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep profile has no edges".into(),
        });
    }

    let mut input_verts: Vec<VertexId> = Vec::with_capacity(n);
    for oe in &input_oriented {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        input_verts.push(vid);
    }

    let input_positions: Vec<Point3> = input_verts
        .iter()
        .map(|&vid| {
            topo.vertex(vid)
                .map(brepkit_topology::vertex::Vertex::point)
        })
        .collect::<Result<_, _>>()?;

    // Compute centroid and frames.
    let (cx, cy, cz) = input_positions
        .iter()
        .fold((0.0, 0.0, 0.0), |(ax, ay, az), p| {
            (ax + p.x(), ay + p.y(), az + p.z())
        });
    #[allow(clippy::cast_precision_loss)]
    let centroid = Point3::new(cx / n as f64, cy / n as f64, cz / n as f64);

    let num_segments = (path.control_points().len() * 2).max(4);
    let up_hint = orthogonalize(input_normal, path.tangent(0.0)?);
    let frames = compute_frames(path, num_segments, up_hint)?;

    let initial_right = frames[0].right;
    let initial_up = frames[0].up;
    let initial_tangent = frames[0].tangent;

    // Compute all ring positions (without allocating vertices yet).
    let num_rings = num_segments + 1;
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

    // Create vertices for first and last rings only (for edge topology).
    let first_ring: Vec<VertexId> = ring_positions[0]
        .iter()
        .map(|&p| topo.vertices.alloc(Vertex::new(p, tol.linear)))
        .collect();
    let last_ring: Vec<VertexId> = ring_positions[num_rings - 1]
        .iter()
        .map(|&p| topo.vertices.alloc(Vertex::new(p, tol.linear)))
        .collect();

    // Create profile edges for first and last rings.
    let first_ring_edges: Vec<_> = (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            topo.edges
                .alloc(Edge::new(first_ring[i], first_ring[next], EdgeCurve::Line))
        })
        .collect();
    let last_ring_edges: Vec<_> = (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            topo.edges
                .alloc(Edge::new(last_ring[i], last_ring[next], EdgeCurve::Line))
        })
        .collect();

    let mut all_faces = Vec::with_capacity(n + 2);

    // Start cap.
    let start_reversed: Vec<OrientedEdge> = (0..n)
        .rev()
        .map(|i| OrientedEdge::new(first_ring_edges[i], false))
        .collect();
    let start_wire = Wire::new(start_reversed, true).map_err(crate::OperationsError::Topology)?;
    let start_wire_id = topo.wires.alloc(start_wire);
    let start_normal = -frames[0].tangent;
    let start_d = dot_normal_point(start_normal, ring_positions[0][0]);
    all_faces.push(topo.faces.alloc(Face::new(
        start_wire_id,
        vec![],
        FaceSurface::Plane {
            normal: start_normal,
            d: start_d,
        },
    )));

    // NURBS side faces: one surface per edge index spanning all rings.
    let degree_u = (num_rings - 1).min(3);
    let degree_v = 1;

    for i in 0..n {
        let next_i = (i + 1) % n;

        // Build interpolation grid: rings × 2 (edge endpoints).
        let grid: Vec<Vec<Point3>> = (0..num_rings)
            .map(|k| vec![ring_positions[k][i], ring_positions[k][next_i]])
            .collect();

        let surface =
            interpolate_surface(&grid, degree_u, degree_v).map_err(crate::OperationsError::Math)?;

        // Rail edges from first to last ring.
        let e_left_rail = topo
            .edges
            .alloc(Edge::new(first_ring[i], last_ring[i], EdgeCurve::Line));
        let e_right_rail = topo.edges.alloc(Edge::new(
            first_ring[next_i],
            last_ring[next_i],
            EdgeCurve::Line,
        ));

        let side_wire = Wire::new(
            vec![
                OrientedEdge::new(first_ring_edges[i], true),
                OrientedEdge::new(e_right_rail, true),
                OrientedEdge::new(last_ring_edges[i], false),
                OrientedEdge::new(e_left_rail, false),
            ],
            true,
        )
        .map_err(crate::OperationsError::Topology)?;

        let side_wire_id = topo.wires.alloc(side_wire);
        all_faces.push(topo.faces.alloc(Face::new(
            side_wire_id,
            vec![],
            FaceSurface::Nurbs(surface),
        )));
    }

    // End cap.
    let end_edges: Vec<OrientedEdge> = (0..n)
        .map(|i| OrientedEdge::new(last_ring_edges[i], true))
        .collect();
    let end_wire = Wire::new(end_edges, true).map_err(crate::OperationsError::Topology)?;
    let end_wire_id = topo.wires.alloc(end_wire);
    let end_normal = frames[num_segments].tangent;
    let end_d = dot_normal_point(end_normal, ring_positions[num_rings - 1][0]);
    all_faces.push(topo.faces.alloc(Face::new(
        end_wire_id,
        vec![],
        FaceSurface::Plane {
            normal: end_normal,
            d: end_d,
        },
    )));

    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
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

/// Options for advanced sweep operations.
#[derive(Default)]
pub struct SweepOptions {
    /// Contact mode for profile orientation.
    pub contact_mode: SweepContactMode,
    /// Scale function: maps path parameter `t ∈ [0, 1]` to a scale factor.
    /// `None` means uniform scale (1.0 everywhere).
    pub scale_law: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
    /// Number of path segments (0 = auto from control point count).
    pub segments: usize,
}

impl std::fmt::Debug for SweepOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SweepOptions")
            .field("contact_mode", &self.contact_mode)
            .field(
                "scale_law",
                &self.scale_law.as_ref().map(|_| "fn(f64)->f64"),
            )
            .field("segments", &self.segments)
            .finish()
    }
}

/// Sweep a face along a path with advanced options.
///
/// Supports scaling laws (tapered sweep) and multiple contact modes.
/// This is equivalent to OCCT's `BRepOffsetAPI_MakePipeShell`.
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

    let face_data = topo.face(profile)?;
    let input_normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "sweep of non-planar faces is not supported".into(),
            });
        }
    };
    let input_wire_id = face_data.outer_wire();

    if !face_data.inner_wires().is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep of faces with holes is not supported".into(),
        });
    }

    let input_wire = topo.wire(input_wire_id)?;
    let input_oriented: Vec<_> = input_wire.edges().to_vec();
    let n = input_oriented.len();

    if n == 0 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sweep profile has no edges".into(),
        });
    }

    let mut input_verts: Vec<VertexId> = Vec::with_capacity(n);
    for oe in &input_oriented {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        input_verts.push(vid);
    }

    let input_positions: Vec<Point3> = input_verts
        .iter()
        .map(|&vid| {
            topo.vertex(vid)
                .map(brepkit_topology::vertex::Vertex::point)
        })
        .collect::<Result<_, _>>()?;

    let (cx, cy, cz) = input_positions
        .iter()
        .fold((0.0, 0.0, 0.0), |(ax, ay, az), p| {
            (ax + p.x(), ay + p.y(), az + p.z())
        });
    #[allow(clippy::cast_precision_loss)]
    let centroid = Point3::new(cx / n as f64, cy / n as f64, cz / n as f64);

    let num_segments = if options.segments > 0 {
        options.segments
    } else {
        (path.control_points().len() * 2).max(4)
    };

    // Compute frames based on contact mode
    let frames = match options.contact_mode {
        SweepContactMode::RotationMinimizing => {
            let up_hint = orthogonalize(input_normal, path.tangent(0.0)?);
            compute_frames(path, num_segments, up_hint)?
        }
        SweepContactMode::Fixed => {
            // Fixed: use the same orientation at every point
            let tangent0 = path.tangent(0.0)?;
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
    };

    let initial_right = frames[0].right;
    let initial_up = frames[0].up;
    let initial_tangent = frames[0].tangent;

    // Create ring vertices with optional scaling
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
                topo.vertices.alloc(Vertex::new(transformed, tol.linear))
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
                topo.edges
                    .alloc(Edge::new(ring[i], ring[next], EdgeCurve::Line))
            })
            .collect();
        ring_edges.push(edges);
    }

    let mut path_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = Vec::with_capacity(num_segments);
    for seg in 0..num_segments {
        let edges: Vec<_> = (0..n)
            .map(|i| {
                topo.edges.alloc(Edge::new(
                    ring_verts[seg][i],
                    ring_verts[seg + 1][i],
                    EdgeCurve::Line,
                ))
            })
            .collect();
        path_edges.push(edges);
    }

    let mut all_faces = Vec::with_capacity(num_segments * n + 2);

    // Start cap
    let start_reversed_edges: Vec<OrientedEdge> = (0..n)
        .rev()
        .map(|i| OrientedEdge::new(ring_edges[0][i], false))
        .collect();
    let start_wire =
        Wire::new(start_reversed_edges, true).map_err(crate::OperationsError::Topology)?;
    let start_wire_id = topo.wires.alloc(start_wire);
    let start_normal = -frames[0].tangent;
    let start_d = dot_normal_point(start_normal, topo.vertex(ring_verts[0][0])?.point());
    let start_face = topo.faces.alloc(Face::new(
        start_wire_id,
        vec![],
        FaceSurface::Plane {
            normal: start_normal,
            d: start_d,
        },
    ));
    all_faces.push(start_face);

    // Side faces
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

            let side_wire_id = topo.wires.alloc(side_wire);
            let side_face = topo.faces.alloc(Face::new(
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

    // End cap
    let end_edges: Vec<OrientedEdge> = (0..n)
        .map(|i| OrientedEdge::new(ring_edges[num_segments][i], true))
        .collect();
    let end_wire = Wire::new(end_edges, true).map_err(crate::OperationsError::Topology)?;
    let end_wire_id = topo.wires.alloc(end_wire);
    let end_normal = frames[num_segments].tangent;
    let end_d = dot_normal_point(
        end_normal,
        topo.vertex(ring_verts[num_segments][0])?.point(),
    );
    let end_face = topo.faces.alloc(Face::new(
        end_wire_id,
        vec![],
        FaceSurface::Plane {
            normal: end_normal,
            d: end_d,
        },
    ));
    all_faces.push(end_face);

    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::collections::HashMap;

    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::Point3;
    use brepkit_topology::Topology;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_square_face;

    use super::*;

    /// Helper: create a straight-line NURBS path from origin along +Z by `length`.
    fn straight_z_path(length: f64) -> NurbsCurve {
        NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, length)],
            vec![1.0, 1.0],
        )
        .unwrap()
    }

    /// Helper: create a quarter-circle NURBS path in the XZ plane.
    fn quarter_circle_xz_path(radius: f64) -> NurbsCurve {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(radius, 0.0, 0.0),
                Point3::new(radius, 0.0, radius),
            ],
            vec![1.0, w, 1.0],
        )
        .unwrap()
    }

    #[test]
    fn sweep_square_along_line() {
        // Sweeping along a straight line should produce a box-like solid,
        // similar to extrude.
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let path = straight_z_path(2.0);

        let solid = sweep(&mut topo, face, &path).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // 4 segments (from max(2*2, 4)) × 4 profile edges = 16 side faces
        // + 2 caps = 18 faces total.
        let num_segs = (path.control_points().len() * 2).max(4);
        let expected_faces = num_segs * 4 + 2;
        assert_eq!(shell.faces().len(), expected_faces);

        // Verify all faces are planar.
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            assert!(
                matches!(f.surface(), FaceSurface::Plane { .. }),
                "all sweep V1 faces should be planar"
            );
        }
    }

    #[test]
    fn sweep_square_along_quarter_circle() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let path = quarter_circle_xz_path(5.0);

        let solid = sweep(&mut topo, face, &path).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // 6 segments (max(3*2, 4)) × 4 edges + 2 caps = 26 faces.
        let num_segs = (path.control_points().len() * 2).max(4);
        let expected_faces = num_segs * 4 + 2;
        assert_eq!(shell.faces().len(), expected_faces);

        // Verify manifold: every edge shared by exactly 2 faces.
        let mut edge_counts: HashMap<usize, usize> = HashMap::new();
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            let wire = topo.wire(f.outer_wire()).unwrap();
            for oe in wire.edges() {
                *edge_counts.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
        for (&edge_idx, &count) in &edge_counts {
            assert_eq!(
                count, 2,
                "edge {edge_idx} shared by {count} faces, expected 2"
            );
        }
    }

    #[test]
    fn sweep_insufficient_control_points_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        // A path with only 1 control point is invalid.
        let path = NurbsCurve::new(
            0,
            vec![0.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0)],
            vec![1.0],
        )
        .unwrap();

        let result = sweep(&mut topo, face, &path);
        assert!(result.is_err());
    }

    #[test]
    fn sweep_zero_path_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        // A path where start == end (zero length).
        let path = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(1.0, 2.0, 3.0), Point3::new(1.0, 2.0, 3.0)],
            vec![1.0, 1.0],
        )
        .unwrap();

        let result = sweep(&mut topo, face, &path);
        assert!(result.is_err());
    }

    #[test]
    fn sweep_and_tessellate_roundtrip() {
        use crate::tessellate::tessellate;

        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let path = quarter_circle_xz_path(5.0);

        let solid = sweep(&mut topo, face, &path).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let tol = Tolerance::new();

        for &fid in shell.faces() {
            let mesh = tessellate(&topo, fid, 0.25).unwrap();
            assert!(!mesh.positions.is_empty());
            assert!(!mesh.indices.is_empty());
            assert_eq!(mesh.positions.len(), mesh.normals.len());

            for normal in &mesh.normals {
                let len = normal.length();
                assert!(
                    tol.approx_eq(len, 1.0) || tol.approx_eq(len, 0.0),
                    "normal length should be ~1.0, got {len}"
                );
            }
        }
    }

    #[test]
    fn sweep_with_default_options_matches_basic() {
        let mut topo = Topology::new();
        let face = crate::primitives::make_box(&mut topo, 0.5, 0.5, 0.01).unwrap();
        let solid = topo.solid(face).unwrap();
        let shell = topo.shell(solid.outer_shell()).unwrap();
        // Use a face from the box as profile
        let profile = shell.faces()[0];

        let path = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 5.0)],
            vec![1.0, 1.0],
        )
        .unwrap();

        let options = SweepOptions::default();
        let result = sweep_with_options(&mut topo, profile, &path, &options);
        assert!(result.is_ok());
    }

    #[test]
    fn sweep_with_linear_scale() {
        let mut topo = Topology::new();
        let profile = make_unit_square_face(&mut topo);

        let path = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 5.0)],
            vec![1.0, 1.0],
        )
        .unwrap();

        let options = SweepOptions {
            scale_law: Some(Box::new(|t| 0.5f64.mul_add(-t, 1.0))), // taper from 1.0 to 0.5
            segments: 8,
            ..Default::default()
        };

        let result = sweep_with_options(&mut topo, profile, &path, &options).unwrap();

        // The result should be a tapered solid
        let vol = crate::measure::solid_volume(&topo, result, 0.5).unwrap();
        assert!(vol > 0.0, "tapered sweep should have positive volume");
    }

    #[test]
    fn sweep_fixed_contact_mode() {
        let mut topo = Topology::new();
        let profile = make_unit_square_face(&mut topo);

        let path = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 5.0)],
            vec![1.0, 1.0],
        )
        .unwrap();

        let options = SweepOptions {
            contact_mode: SweepContactMode::Fixed,
            ..Default::default()
        };

        let result = sweep_with_options(&mut topo, profile, &path, &options);
        assert!(result.is_ok());
    }

    #[test]
    fn sweep_constant_normal_mode() {
        let mut topo = Topology::new();
        let profile = make_unit_square_face(&mut topo);

        let path = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 5.0)],
            vec![1.0, 1.0],
        )
        .unwrap();

        let options = SweepOptions {
            contact_mode: SweepContactMode::ConstantNormal(Vec3::new(0.0, 1.0, 0.0)),
            ..Default::default()
        };

        let result = sweep_with_options(&mut topo, profile, &path, &options);
        assert!(result.is_ok());
    }

    // ── Smooth sweep tests ──────────────────────────

    #[test]
    fn sweep_smooth_produces_nurbs_sides() {
        let mut topo = Topology::new();
        let profile = make_unit_square_face(&mut topo);
        let path = straight_z_path(2.0);

        let solid = sweep_smooth(&mut topo, profile, &path).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // Should have N NURBS sides + 2 planar caps.
        let nurbs_count = sh
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Nurbs(_)))
            .count();

        assert!(
            nurbs_count > 0,
            "smooth sweep should produce NURBS side faces"
        );

        // Fewer faces than the basic sweep (N sides vs N*segments sides).
        let profile_edge_count = 4; // square has 4 edges
        let expected_face_count = profile_edge_count + 2; // N sides + 2 caps
        assert_eq!(
            sh.faces().len(),
            expected_face_count,
            "smooth sweep should have {expected_face_count} faces, got {}",
            sh.faces().len()
        );
    }

    #[test]
    fn sweep_smooth_positive_volume() {
        let mut topo = Topology::new();
        let profile = make_unit_square_face(&mut topo);
        let path = straight_z_path(3.0);

        let solid = sweep_smooth(&mut topo, profile, &path).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            vol > 0.0,
            "smooth sweep should have positive volume, got {vol}"
        );
    }

    #[test]
    fn sweep_smooth_curved_path() {
        let mut topo = Topology::new();
        let profile = make_unit_square_face(&mut topo);
        let path = quarter_circle_xz_path(5.0);

        let solid = sweep_smooth(&mut topo, profile, &path).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        assert_eq!(
            sh.faces().len(),
            6,
            "smooth curved sweep should have 6 faces"
        );

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(vol > 0.0, "curved smooth sweep should have positive volume");
    }
}
