//! Open shells ("sheets") swept from an open wire.
//!
//! The solid operations own closed profiles. These build a surface from a wire
//! with two free ends: one or more faces per wire edge, adjacent faces sharing
//! their edge. Every face runs the way its wire edge does, its effective normal
//! the edge's direction crossed with the sweep's, so one orientation holds
//! across the whole shell.

use brepkit_math::madd::Madd;
use std::f64::consts::TAU;

use brepkit_math::curves::Circle3D;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::decompose::curve_degree_elevate;
use brepkit_math::nurbs::knot_ops::curve_knot_insert;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::surfaces::SphericalSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::{Shell, ShellId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::OperationsError;
use crate::dot_normal_point;
use crate::extrude::{
    normalize_edge_curve, ruled_nurbs_surface, side_face_surface, translate_edge_curve,
};
use crate::revolve::{
    RevEdge, arc_segmentation, classify_profile_edge, curve_revolution_surface,
    make_revolution_surface, next_ring_index, profile_arc_center_radius, profile_band_curve,
    radial_axial, revolution_band_surface, revolution_wall_surface, rotate_point, rotation_about,
};
use crate::sweep::{clamped_edge_span, detect_kinks, path_trajectories};
use crate::transform::curve_image;

/// Steps along an edge when reading where it runs.
const EDGE_STEPS: u32 = 16;

/// How far from parallel a circle's normal may lean off a direction and still
/// count as along it (the sine of the angle between them).
const PARALLEL_TOL: f64 = 1e-9;

/// Two path edges meet within this many linear tolerances, as the sweep along
/// a chain of edges reads them.
const PATH_JOIN_FACTOR: f64 = 100.0;

/// Knots of two ruled curves within this of each other, on `[0, 1]`, are one.
const KNOT_EPS: f64 = 1e-9;
/// How far an edge's vertex may sit off its curve: a fitted curve's own error.
const ON_CURVE: f64 = 1e-5;

fn refuse(reason: impl Into<String>) -> OperationsError {
    OperationsError::InvalidInput {
        reason: reason.into(),
    }
}

/// One edge of a chain, in the order the chain runs it.
struct EdgeRun {
    edge: OrientedEdge,
    curve: EdgeCurve,
    from: Point3,
    to: Point3,
}

impl EdgeRun {
    const fn is_forward(&self) -> bool {
        self.edge.is_forward()
    }

    /// The point `fraction` of the way along the edge's own domain, from its
    /// stored start.
    fn point_at(&self, fraction: f64) -> Point3 {
        let (start, end) = if self.is_forward() {
            (self.from, self.to)
        } else {
            (self.to, self.from)
        };
        let (t0, t1) = self.curve.domain_with_endpoints(start, end);
        self.curve
            .evaluate_with_endpoints((t1 - t0).madd(fraction, t0), start, end)
    }

    fn samples(&self) -> Vec<Point3> {
        (0..=EDGE_STEPS)
            .map(|k| self.point_at(f64::from(k) / f64::from(EDGE_STEPS)))
            .collect()
    }

    /// The edge's own curve from `from` to `to` as a clamped NURBS.
    fn span(&self) -> Result<NurbsCurve, OperationsError> {
        clamped_edge_span(&self.curve, self.is_forward(), (self.from, self.to))?
            .ok_or_else(|| refuse("a NURBS edge on an unclamped knot vector cannot be swept"))
    }
}

/// An open wire read end to end: `n` runs through `n + 1` distinct vertices.
struct OpenChain {
    runs: Vec<EdgeRun>,
    vertices: Vec<VertexId>,
    points: Vec<Point3>,
}

/// Read `wire` as an open chain, its spline-encoded edges recognized back to
/// lines and circles the way an extrusion's profile edges are.
///
/// A wire's closed flag is not consulted: callers set it without looking at
/// the ends, so the ends themselves decide.
fn open_chain(topo: &mut Topology, wire: WireId) -> Result<OpenChain, OperationsError> {
    let tol = Tolerance::new().linear;
    let oriented = topo.wire(wire)?.edges().to_vec();
    let mut edges = Vec::with_capacity(oriented.len());
    let mut vertices: Vec<VertexId> = Vec::with_capacity(oriented.len() + 1);
    for oe in oriented {
        let edge = topo.edge(oe.edge())?;
        let (start, end) = (oe.oriented_start(edge), oe.oriented_end(edge));
        let tail = *vertices.last().unwrap_or(&start);
        if tail == start {
            edges.push(oe);
        } else {
            if (topo.vertex(tail)?.point() - topo.vertex(start)?.point()).length() > tol {
                return Err(refuse("the wire's edges do not connect end to end"));
            }
            // Ends at one point on two vertices: the shared rail needs one.
            let curve = edge.curve().clone();
            let (s, e) = if oe.is_forward() {
                (tail, end)
            } else {
                (end, tail)
            };
            let welded = topo.add_edge(Edge::new(s, e, curve));
            edges.push(OrientedEdge::new(welded, oe.is_forward()));
        }
        if vertices.is_empty() {
            vertices.push(start);
        }
        vertices.push(end);
    }
    let points = vertices
        .iter()
        .map(|&v| Ok(topo.vertex(v)?.point()))
        .collect::<Result<Vec<_>, OperationsError>>()?;
    if (points[0] - points[points.len() - 1]).length() <= tol {
        return Err(refuse(
            "the wire is closed: a closed profile belongs to the solid operations",
        ));
    }
    for (i, &p) in points.iter().enumerate() {
        if points[i + 1..].iter().any(|&q| (q - p).length() <= tol) {
            return Err(refuse("the wire meets itself at a vertex"));
        }
    }
    let mut runs = Vec::with_capacity(edges.len());
    for (i, oe) in edges.into_iter().enumerate() {
        let edge = topo.edge(oe.edge())?;
        let (start, end) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
        let misses = |t: f64, p: Point3| {
            (edge.curve().evaluate_with_endpoints(t, start, end) - p).length() > ON_CURVE
        };
        if misses(t0, start) || misses(t1, end) {
            return Err(refuse("an edge's curve does not reach its vertices"));
        }
        normalize_edge_curve(topo, oe.edge(), tol)?;
        runs.push(EdgeRun {
            edge: oe,
            curve: topo.edge(oe.edge())?.curve().clone(),
            from: points[i],
            to: points[i + 1],
        });
    }
    Ok(OpenChain {
        runs,
        vertices,
        points,
    })
}

const fn flipped(oe: OrientedEdge) -> OrientedEdge {
    OrientedEdge::new(oe.edge(), !oe.is_forward())
}

/// The face between `near` and `far`, one edge at two stages of a sweep, both
/// running the way the chain does, joined by the rails its ends sweep. A rail
/// is `None` where the sweep leaves that end in place. `near` and `far` are one
/// edge on a band closing on its own seam, and the rails are its rims.
fn band_face(
    topo: &mut Topology,
    (near, far): (OrientedEdge, OrientedEdge),
    (start_rail, end_rail): (Option<EdgeId>, Option<EdgeId>),
    surface: FaceSurface,
    reversed: bool,
) -> Result<FaceId, OperationsError> {
    let rail = |edge: Option<EdgeId>, forward: bool| edge.map(|e| OrientedEdge::new(e, forward));
    let edges: Vec<OrientedEdge> = if reversed {
        [
            rail(start_rail, true),
            Some(far),
            rail(end_rail, false),
            Some(flipped(near)),
        ]
    } else {
        [
            Some(near),
            rail(end_rail, true),
            Some(flipped(far)),
            rail(start_rail, false),
        ]
    }
    .into_iter()
    .flatten()
    .collect();
    let wire = topo.add_wire(Wire::new(edges, true)?);
    Ok(topo.add_face(if reversed {
        Face::new_reversed(wire, vec![], surface)
    } else {
        Face::new(wire, vec![], surface)
    }))
}

fn finish(topo: &mut Topology, faces: Vec<FaceId>) -> Result<ShellId, OperationsError> {
    Ok(topo.add_shell(Shell::new(faces)?))
}

/// Extrude an open wire along `direction * distance` into an open shell.
///
/// One face per edge: a line sweeps a plane and a circle whose normal lies along
/// the direction a cylinder, and any other edge a ruled NURBS on its own curve.
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if the offset has no length, an
/// edge runs along the direction and so sweeps no surface, or the wire is
/// closed, meets itself or breaks.
pub fn extrude_wire(
    topo: &mut Topology,
    wire: WireId,
    direction: Vec3,
    distance: f64,
) -> Result<ShellId, OperationsError> {
    let tol = Tolerance::new().linear;
    let offset = direction * distance;
    let reach = offset.length();
    if !reach.is_finite() || reach <= tol {
        return Err(refuse(format!(
            "the extrusion has zero length: direction {direction:?} by {distance}"
        )));
    }
    let along = offset.normalize()?;
    let chain = open_chain(topo, wire)?;
    for run in &chain.runs {
        if run
            .samples()
            .iter()
            .all(|&p| (p - run.from).cross(along).length() <= tol)
        {
            return Err(refuse(
                "an edge along the extrusion direction sweeps no surface",
            ));
        }
    }

    let top: Vec<VertexId> = chain
        .points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p + offset, tol)))
        .collect();
    let rails: Vec<EdgeId> = chain
        .vertices
        .iter()
        .zip(&top)
        .map(|(&low, &high)| topo.add_edge(Edge::new(low, high, EdgeCurve::Line)))
        .collect();
    let mut faces = Vec::with_capacity(chain.runs.len());
    for (i, run) in chain.runs.iter().enumerate() {
        let (surface, reversed) = extruded_surface(run, offset)?;
        let (s, e) = if run.is_forward() {
            (top[i], top[i + 1])
        } else {
            (top[i + 1], top[i])
        };
        let far = topo.add_edge(Edge::new(s, e, translate_edge_curve(&run.curve, offset)?));
        faces.push(band_face(
            topo,
            (run.edge, OrientedEdge::new(far, run.is_forward())),
            (Some(rails[i]), Some(rails[i + 1])),
            surface,
            reversed,
        )?);
    }
    finish(topo, faces)
}

fn extruded_surface(run: &EdgeRun, offset: Vec3) -> Result<(FaceSurface, bool), OperationsError> {
    let along = offset.normalize()?;
    let swept_exactly = match &run.curve {
        EdgeCurve::Line => true,
        EdgeCurve::Circle(c) => c.normal().cross(along).length() <= PARALLEL_TOL,
        EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => false,
    };
    if swept_exactly {
        let (start, end) = if run.is_forward() {
            (run.from, run.to)
        } else {
            (run.to, run.from)
        };
        return side_face_surface(&run.curve, run.from, run.to, start, end, offset, false);
    }
    // The ruled surface's rows climb the offset, so its normal is the offset
    // crossed with the curve: against the edge's direction crossed with it.
    Ok((
        FaceSurface::Nurbs(ruled_nurbs_surface(&run.span()?, offset)?),
        true,
    ))
}

/// What one edge sweeps in a full turn, as a single face closing on itself.
enum FullBand {
    /// The edge lies along the axis.
    Nothing,
    Disc,
    Wall(RevEdge),
    Sphere {
        center: Point3,
        radius: f64,
    },
}

/// The offset from the axis to `p`, square to the axis.
fn radial_offset(p: Point3, axis_origin: Point3, axis: Vec3) -> Vec3 {
    let v = p - axis_origin;
    v - axis * v.dot(axis)
}

/// Whether an edge lies in one half plane bounded by the axis, the meridian a
/// surface of revolution is drawn from, and so is not skew to the axis.
fn is_meridian(run: &EdgeRun, axis_origin: Point3, axis: Vec3, tol: f64) -> bool {
    let samples = run.samples();
    let farthest = samples
        .iter()
        .map(|&p| radial_offset(p, axis_origin, axis))
        .max_by(|a, b| a.length().total_cmp(&b.length()));
    let Some(Ok(side)) = farthest.map(Vec3::normalize) else {
        return false;
    };
    samples.iter().all(|&p| {
        let r = radial_offset(p, axis_origin, axis);
        let along = r.dot(side);
        along >= -tol && (r - side * along).length() <= tol
    })
}

/// Whether an edge passes through the axis between its ends, stepping to the
/// far side of it along a chord that meets it: not merely passing near, as a
/// line skew to the axis does.
fn crosses_axis(run: &EdgeRun, axis_origin: Point3, axis: Vec3, tol: f64) -> bool {
    let offsets: Vec<Vec3> = run
        .samples()
        .iter()
        .map(|&p| radial_offset(p, axis_origin, axis))
        .filter(|r| r.length() > tol)
        .collect();
    offsets
        .windows(2)
        .any(|w| w[0].dot(w[1]) < 0.0 && w[0].cross(w[1]).length() <= tol * (w[1] - w[0]).length())
}

/// The single closed face a full turn of `run` sweeps, or `None` when it has
/// none and the turn goes in quarters. An arc with both ends on the axis is
/// such a case: it sweeps a whole sphere, which no rim bounds, and a face needs
/// a boundary to be measured and meshed by.
fn full_band(run: &EdgeRun, axis_origin: Point3, axis: Vec3, tol: f64) -> Option<FullBand> {
    if !is_meridian(run, axis_origin, axis, tol) {
        return None;
    }
    match classify_profile_edge(&run.curve, run.from, run.to, axis_origin, axis, tol) {
        Some(RevEdge::OnAxis) => Some(FullBand::Nothing),
        Some(RevEdge::Plane) => Some(FullBand::Disc),
        Some(wall) => Some(FullBand::Wall(wall)),
        None => {
            let (center, radius) =
                profile_arc_center_radius(&run.curve, run.from, run.to, axis_origin, axis)?;
            let to_center = center - axis_origin;
            let centred = (to_center - axis * to_center.dot(axis)).length() <= tol;
            let bounded = radial_axial(run.from, axis_origin, axis).0 > tol
                || radial_axial(run.to, axis_origin, axis).0 > tol;
            (centred && bounded).then_some(FullBand::Sphere { center, radius })
        }
    }
}

/// Revolve an open wire by `angle_radians` about the axis into an open shell.
///
/// A line in a plane holding the axis sweeps a cylinder, cone or plane, and a
/// circular arc in such a plane a torus or sphere band; any other edge sweeps
/// an exact rational NURBS band. An edge along the axis sweeps no face, and a
/// vertex on the axis stays one vertex. A full turn of edges that all have an
/// analytic band closes each on its own seam, one face per edge; otherwise each
/// edge sweeps one face per quarter turn or part of one, a full turn wrapping
/// round to the wire itself.
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if the axis has no length, the
/// angle is outside `(0, 2π]`, the whole wire lies along the axis, an edge
/// crosses the axis between its ends, or the wire is closed, meets itself or
/// breaks.
pub fn revolve_wire(
    topo: &mut Topology,
    wire: WireId,
    axis_origin: Point3,
    axis_direction: Vec3,
    angle_radians: f64,
) -> Result<ShellId, OperationsError> {
    let tol = Tolerance::new();
    if tol.approx_eq(axis_direction.length_squared(), 0.0) {
        return Err(refuse("the revolve axis has zero length"));
    }
    let axis = axis_direction.normalize()?;
    let in_range = angle_radians > 0.0 && angle_radians <= TAU + tol.angular;
    if !in_range {
        return Err(refuse(format!(
            "the revolve angle must be in (0, 2π], got {angle_radians}"
        )));
    }
    let is_full = angle_radians >= TAU - tol.angular;
    let angle = if is_full { TAU } else { angle_radians };

    let chain = open_chain(topo, wire)?;
    let along_axis: Vec<bool> = chain
        .runs
        .iter()
        .map(|run| {
            run.samples()
                .iter()
                .all(|&p| radial_axial(p, axis_origin, axis).0 <= tol.linear)
        })
        .collect();
    if along_axis.iter().all(|&along| along) {
        return Err(refuse(
            "the wire lies along the axis, so its revolution has no surface",
        ));
    }
    if chain
        .runs
        .iter()
        .any(|run| crosses_axis(run, axis_origin, axis, tol.linear))
    {
        return Err(refuse(
            "an edge crosses the revolve axis and would sweep through itself: split it on the axis",
        ));
    }

    let closed_bands: Option<Vec<FullBand>> = if is_full {
        chain
            .runs
            .iter()
            .zip(&along_axis)
            .map(|(run, &along)| {
                if along {
                    Some(FullBand::Nothing)
                } else {
                    full_band(run, axis_origin, axis, tol.linear)
                }
            })
            .collect()
    } else {
        None
    };
    let faces = if let Some(bands) = closed_bands {
        full_turn_faces(topo, &chain, &bands, axis_origin, axis)?
    } else {
        let turn = Turn {
            axis_origin,
            axis,
            angle,
            is_full,
        };
        quarter_turn_faces(topo, &chain, &along_axis, &turn)?
    };
    finish(topo, faces)
}

/// One face per edge, each closing on itself: a wall on its seam (the profile
/// edge, used both ways) between the rims its ends sweep, a disc, annulus,
/// sphere cap or zone inside its rims. The rims are shared with the next edge's
/// face.
fn full_turn_faces(
    topo: &mut Topology,
    chain: &OpenChain,
    bands: &[FullBand],
    axis_origin: Point3,
    axis: Vec3,
) -> Result<Vec<FaceId>, OperationsError> {
    let tol = Tolerance::new().linear;
    let mut rims: Vec<Option<EdgeId>> = Vec::with_capacity(chain.points.len());
    for (&p, &vertex) in chain.points.iter().zip(&chain.vertices) {
        let (r, z) = radial_axial(p, axis_origin, axis);
        if r <= tol {
            rims.push(None);
            continue;
        }
        let center = axis_origin + axis * z;
        let u = (p - center).normalize()?;
        let circle = Circle3D::with_axes(center, axis, r, u, axis.cross(u))?;
        rims.push(Some(topo.add_edge(Edge::new(
            vertex,
            vertex,
            EdgeCurve::Circle(circle),
        ))));
    }

    let mut faces = Vec::with_capacity(bands.len());
    for (i, (run, band)) in chain.runs.iter().zip(bands).enumerate() {
        let (rim_from, rim_to) = (rims[i], rims[i + 1]);
        let mid = run.point_at(0.5);
        // Where a circular arc's middle lies, its tangent runs along its chord.
        let facing = (run.to - run.from).cross(axis.cross(mid - axis_origin));
        match band {
            FullBand::Nothing => {}
            FullBand::Disc => {
                let outward = radial_axial(run.to, axis_origin, axis).0
                    > radial_axial(run.from, axis_origin, axis).0;
                let normal = if outward { axis } else { -axis };
                let surface = FaceSurface::Plane {
                    normal,
                    d: dot_normal_point(normal, run.from),
                };
                let (outer, hole) = if outward {
                    ((rim_to, true), (rim_from, false))
                } else {
                    ((rim_from, false), (rim_to, true))
                };
                faces.push(rim_face(topo, outer, hole, surface, false)?);
            }
            FullBand::Wall(class) => {
                let (surface, reversed) =
                    revolution_wall_surface(class, axis_origin, axis, mid, facing)?;
                faces.push(band_face(
                    topo,
                    (run.edge, run.edge),
                    (rim_from, rim_to),
                    surface,
                    reversed,
                )?);
            }
            FullBand::Sphere { center, radius } => {
                let surface = SphericalSurface::with_axis(*center, *radius, axis)?;
                let reversed = (mid - *center).dot(facing) < 0.0;
                // A sphere face takes the rim nearer a pole as its hole, the
                // cap past it held inside the region its outer rim bounds.
                let rise = |p: Point3| (p - *center).dot(axis).abs();
                let (from, to) = ((rim_from, reversed), (rim_to, !reversed));
                let (outer, hole) = if rise(run.from) >= rise(run.to) {
                    (to, from)
                } else {
                    (from, to)
                };
                faces.push(rim_face(
                    topo,
                    outer,
                    hole,
                    FaceSurface::Sphere(surface),
                    reversed,
                )?);
            }
        }
    }
    Ok(faces)
}

/// A face bounded by whole rims, the edges at its two ends each sweeping one
/// (none where an end is on the axis). In its effective sense the rim at an
/// edge's end runs with the sweep and the one at its start against it, as on a
/// wall.
fn rim_face(
    topo: &mut Topology,
    outer: (Option<EdgeId>, bool),
    hole: (Option<EdgeId>, bool),
    surface: FaceSurface,
    reversed: bool,
) -> Result<FaceId, OperationsError> {
    let mut rim_wire = |(rim, forward): (Option<EdgeId>, bool)| -> Result<_, OperationsError> {
        rim.map(|e| Ok(topo.add_wire(Wire::new(vec![OrientedEdge::new(e, forward)], true)?)))
            .transpose()
    };
    let outer = rim_wire(outer)?.ok_or_else(|| refuse("a closed band has no outer rim"))?;
    let holes = rim_wire(hole)?.into_iter().collect();
    Ok(topo.add_face(if reversed {
        Face::new_reversed(outer, holes, surface)
    } else {
        Face::new(outer, holes, surface)
    }))
}

struct Turn {
    axis_origin: Point3,
    axis: Vec3,
    angle: f64,
    is_full: bool,
}

/// One face per edge per quarter turn or part of one, each bounded by the edge
/// turned to both its ends and the arcs its ends sweep. A vertex on the axis
/// sweeps no arc, and an edge along it no face.
fn quarter_turn_faces(
    topo: &mut Topology,
    chain: &OpenChain,
    along_axis: &[bool],
    turn: &Turn,
) -> Result<Vec<FaceId>, OperationsError> {
    let Turn {
        axis_origin,
        axis,
        angle,
        is_full,
    } = *turn;
    let tol = Tolerance::new().linear;
    let (num_segs, seg_angle) = arc_segmentation(angle);
    let num_rings = if is_full { num_segs } else { num_segs + 1 };
    #[allow(clippy::cast_precision_loss)]
    let turned_by = |k: usize| seg_angle * k as f64;
    let on_axis: Vec<bool> = chain
        .points
        .iter()
        .map(|&p| radial_axial(p, axis_origin, axis).0 <= tol)
        .collect();

    let ring_points: Vec<Vec<Point3>> = (0..num_rings)
        .map(|k| {
            chain
                .points
                .iter()
                .map(|&p| rotate_point(p, axis_origin, axis, turned_by(k)))
                .collect()
        })
        .collect();
    let mut ring_vertices: Vec<Vec<VertexId>> = vec![chain.vertices.clone()];
    for ring in &ring_points[1..] {
        let vertices = ring
            .iter()
            .zip(&chain.vertices)
            .zip(&on_axis)
            .map(|((&p, &vertex), &stays)| {
                if stays {
                    vertex
                } else {
                    topo.add_vertex(Vertex::new(p, tol))
                }
            })
            .collect();
        ring_vertices.push(vertices);
    }

    let mut arcs: Vec<Vec<Option<EdgeId>>> = Vec::with_capacity(num_segs);
    for seg in 0..num_segs {
        let next = next_ring_index(seg, num_segs, is_full);
        let mut row = Vec::with_capacity(chain.points.len());
        for j in 0..chain.points.len() {
            if on_axis[j] {
                row.push(None);
                continue;
            }
            let (radius, along) = radial_axial(ring_points[seg][j], axis_origin, axis);
            let center = axis_origin + axis * along;
            let u = (ring_points[seg][j] - center).normalize()?;
            let circle = Circle3D::with_axes(center, axis, radius, u, axis.cross(u))?;
            row.push(Some(topo.add_edge(Edge::new(
                ring_vertices[seg][j],
                ring_vertices[next][j],
                EdgeCurve::Circle(circle),
            ))));
        }
        arcs.push(row);
    }

    let mut ring_edges: Vec<Vec<OrientedEdge>> =
        vec![chain.runs.iter().map(|run| run.edge).collect()];
    for (k, ring) in ring_vertices.iter().enumerate().skip(1) {
        let matrix = rotation_about(axis_origin, axis, turned_by(k));
        let mut row = Vec::with_capacity(chain.runs.len());
        for (i, run) in chain.runs.iter().enumerate() {
            if along_axis[i] {
                row.push(run.edge);
                continue;
            }
            let (curve, _) = curve_image(&run.curve, &matrix)?;
            let (s, e) = if run.is_forward() {
                (ring[i], ring[i + 1])
            } else {
                (ring[i + 1], ring[i])
            };
            row.push(OrientedEdge::new(
                topo.add_edge(Edge::new(s, e, curve)),
                run.is_forward(),
            ));
        }
        ring_edges.push(row);
    }

    let meridian: Vec<bool> = chain
        .runs
        .iter()
        .map(|run| is_meridian(run, axis_origin, axis, tol))
        .collect();
    let mut faces = Vec::with_capacity(num_segs * chain.runs.len());
    for seg in 0..num_segs {
        let next = next_ring_index(seg, num_segs, is_full);
        for (i, run) in chain.runs.iter().enumerate() {
            if along_axis[i] {
                continue;
            }
            let (p0_start, p0_end, p1_start, p1_end) = (
                ring_points[seg][i],
                ring_points[next][i],
                ring_points[seg][i + 1],
                ring_points[next][i + 1],
            );
            let seg_start = turned_by(seg);
            let (surface, reversed) = if meridian[i] {
                revolution_band_surface(
                    &run.curve,
                    run.is_forward(),
                    p0_start,
                    p0_end,
                    p1_start,
                    p1_end,
                    axis_origin,
                    axis,
                    seg_angle,
                    seg_start,
                )?
            } else {
                // The analytic bands read a line's radii at its ends, which
                // cannot tell a line skew to the axis.
                let surface = match profile_band_curve(
                    &run.curve,
                    run.is_forward(),
                    (p0_start, p1_start),
                    axis_origin,
                    axis,
                    seg_start,
                )? {
                    Some(profile) => {
                        curve_revolution_surface(&profile, axis_origin, axis, seg_angle)?
                    }
                    None => make_revolution_surface(
                        p0_start,
                        p0_end,
                        p1_start,
                        p1_end,
                        axis_origin,
                        axis,
                        seg_angle,
                    )?,
                };
                (FaceSurface::Nurbs(surface), false)
            };
            faces.push(band_face(
                topo,
                (ring_edges[seg][i], ring_edges[next][i]),
                (arcs[seg][i], arcs[seg][i + 1]),
                surface,
                reversed,
            )?);
        }
    }
    Ok(faces)
}

/// Loft ruled faces between two or more open wires of equal edge counts.
///
/// One face per edge per gap, each wire's start paired with the next wire's
/// start: a plane between two lines in one, and otherwise a ruled NURBS on the
/// edges' own curves.
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if there are fewer than two wires,
/// their edge counts differ, two touch where they would be joined, two lines
/// paired across a plane cross each other, or a wire is closed, meets itself
/// or breaks.
pub fn loft_wires(topo: &mut Topology, wires: &[WireId]) -> Result<ShellId, OperationsError> {
    if wires.len() < 2 {
        return Err(refuse(format!(
            "a loft needs at least two wires, got {}",
            wires.len()
        )));
    }
    let tol = Tolerance::new().linear;
    let chains = wires
        .iter()
        .map(|&wire| open_chain(topo, wire))
        .collect::<Result<Vec<_>, _>>()?;
    let count = chains[0].runs.len();
    if let Some(other) = chains.iter().find(|chain| chain.runs.len() != count) {
        return Err(refuse(format!(
            "lofted wires need equal edge counts, got {count} edges and {} edges",
            other.runs.len()
        )));
    }

    let mut faces = Vec::with_capacity((chains.len() - 1) * count);
    for pair in chains.windows(2) {
        let (low, high) = (&pair[0], &pair[1]);
        if low
            .points
            .iter()
            .zip(&high.points)
            .any(|(&a, &b)| (a - b).length() <= tol)
        {
            return Err(refuse("two lofted wires touch where they would be joined"));
        }
        let rails: Vec<EdgeId> = low
            .vertices
            .iter()
            .zip(&high.vertices)
            .map(|(&a, &b)| topo.add_edge(Edge::new(a, b, EdgeCurve::Line)))
            .collect();
        for i in 0..count {
            let (near, far) = (&low.runs[i], &high.runs[i]);
            let surface = match (&near.curve, &far.curve) {
                (EdgeCurve::Line, EdgeCurve::Line) => {
                    ruled_lines(near.from, near.to, far.from, far.to)?
                }
                (
                    EdgeCurve::Line
                    | EdgeCurve::Circle(_)
                    | EdgeCurve::Ellipse(_)
                    | EdgeCurve::NurbsCurve(_),
                    EdgeCurve::Line
                    | EdgeCurve::Circle(_)
                    | EdgeCurve::Ellipse(_)
                    | EdgeCurve::NurbsCurve(_),
                ) => FaceSurface::Nurbs(ruled_spans(&near.span()?, &far.span()?)?),
            };
            faces.push(band_face(
                topo,
                (near.edge, far.edge),
                (Some(rails[i]), Some(rails[i + 1])),
                surface,
                false,
            )?);
        }
    }
    finish(topo, faces)
}

/// The ruled surface between the segments `a0 → a1` and `b0 → b1`: their plane
/// when all four corners lie in one, and otherwise the bilinear patch, its
/// normal running as the loop `a0, a1, b1, b0` winds either way.
fn ruled_lines(
    a0: Point3,
    a1: Point3,
    b0: Point3,
    b1: Point3,
) -> Result<FaceSurface, OperationsError> {
    let tol = Tolerance::new().linear;
    let corners = [a0, a1, b1, b0];
    let crossing = || refuse("two lofted lines cross each other: run the wires the same way");
    let normal = crate::winding::newell_normal(&corners)
        .normalize()
        .map_err(|_| crossing())?;
    if corners.iter().all(|&c| (c - a0).dot(normal).abs() <= tol) {
        // The ruled patch between two lines only fills a convex quad: at a
        // reflex corner it folds over itself, and one of its sides doubling
        // back along the next retraces the boundary.
        let folds = (0..4).any(|k| {
            let (p, q, r) = (corners[k], corners[(k + 1) % 4], corners[(k + 2) % 4]);
            let (along, next) = (q - p, r - q);
            let turn = along.cross(next).dot(normal);
            turn < -tol * tol || (turn.abs() <= tol * tol && along.dot(next) < 0.0)
        });
        if folds {
            return Err(crossing());
        }
        return Ok(FaceSurface::Plane {
            normal,
            d: dot_normal_point(normal, a0),
        });
    }
    let knots = vec![0.0, 0.0, 1.0, 1.0];
    Ok(FaceSurface::Nurbs(NurbsSurface::new(
        1,
        1,
        knots.clone(),
        knots,
        vec![vec![a0, b0], vec![a1, b1]],
        vec![vec![1.0; 2]; 2],
    )?))
}

/// `curve` on the parameter range `[0, 1]`.
pub(crate) fn on_unit_domain(curve: &NurbsCurve) -> Result<NurbsCurve, OperationsError> {
    let (d0, d1) = curve.domain();
    Ok(NurbsCurve::new(
        curve.degree(),
        curve.knots().iter().map(|k| (k - d0) / (d1 - d0)).collect(),
        curve.control_points().to_vec(),
        curve.weights().to_vec(),
    )?)
}

/// `curve` with every knot of `other` inserted, on a shared `[0, 1]`.
fn with_knots_of(curve: &NurbsCurve, other: &NurbsCurve) -> Result<NurbsCurve, OperationsError> {
    let mut out = curve.clone();
    let knots = other.knots();
    let mut i = 0;
    while i < knots.len() {
        let knot = knots[i];
        let want = knots[i..]
            .iter()
            .take_while(|&&k| (k - knot).abs() <= KNOT_EPS)
            .count();
        let have = out
            .knots()
            .iter()
            .filter(|&&k| (k - knot).abs() <= KNOT_EPS)
            .count();
        if want > have && knot > KNOT_EPS && knot < 1.0 - KNOT_EPS {
            out = curve_knot_insert(&out, knot, want - have)?;
        }
        i += want;
    }
    Ok(out)
}

/// The ruled NURBS between two curves running the same way: `u` along them at
/// a common degree and knot vector, and `v` from one to the other.
fn ruled_spans(a: &NurbsCurve, b: &NurbsCurve) -> Result<NurbsSurface, OperationsError> {
    let degree = a.degree().max(b.degree());
    let raise = |c: &NurbsCurve| -> Result<NurbsCurve, OperationsError> {
        on_unit_domain(&curve_degree_elevate(c, degree - c.degree())?)
    };
    let (a, b) = (raise(a)?, raise(b)?);
    let (a, b) = (with_knots_of(&a, &b)?, with_knots_of(&b, &a)?);
    if a.control_points().len() != b.control_points().len() {
        return Err(refuse(
            "two lofted curves did not reach one knot vector to rule between",
        ));
    }
    let pair = |(&p, &q): (&Point3, &Point3)| vec![p, q];
    let weights = |(&p, &q): (&f64, &f64)| vec![p, q];
    Ok(NurbsSurface::new(
        degree,
        1,
        a.knots().to_vec(),
        vec![0.0, 0.0, 1.0, 1.0],
        a.control_points()
            .iter()
            .zip(b.control_points())
            .map(pair)
            .collect(),
        a.weights().iter().zip(b.weights()).map(weights).collect(),
    )?)
}

/// Sweep an open wire along a chain of path edges into an open shell, the wire
/// carried from where it lies without twist on rotation-minimizing frames.
///
/// A straight path is [`extrude_wire`], and a path of one circle or arc is
/// [`revolve_wire`] about its axis, each with the exact surfaces those build.
/// Any other open, kink-free chain sweeps one NURBS face per wire edge, keeping
/// the edge's own curve across the sweep; each control point's trajectory along
/// the path is fitted to within the sweep's tolerance, a fraction `1e-7` of its
/// extent, not held exact.
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if the chain has no edges or no
/// length, breaks, closes (other than one full circle) or has a kink, or if
/// the wire is closed, meets itself or breaks.
pub fn sweep_wire(
    topo: &mut Topology,
    wire: WireId,
    path_edges: &[EdgeId],
) -> Result<ShellId, OperationsError> {
    let tol = Tolerance::new().linear;
    let path = path_runs(topo, path_edges)?;
    let (first, last) = match path.as_slice() {
        [] => return Err(refuse("the sweep path has no edges")),
        [only] => (only, only),
        [first, .., last] => (first, last),
    };
    let (start, end) = (first.from, last.to);

    if path.iter().all(|run| matches!(run.curve, EdgeCurve::Line)) {
        let walked: f64 = path.iter().map(|run| (run.to - run.from).length()).sum();
        if walked <= tol {
            return Err(refuse("the sweep path has no length"));
        }
        let reach = (end - start).length();
        if walked - reach <= tol {
            return extrude_wire(topo, wire, (end - start).normalize()?, reach);
        }
    }
    if let ([only], EdgeCurve::Circle(circle)) = (path.as_slice(), &first.curve) {
        let (t0, t1) = only.curve.domain_with_endpoints(start, end);
        return revolve_wire(topo, wire, circle.center(), circle.normal(), t1 - t0);
    }
    if (end - start).length() <= tol * PATH_JOIN_FACTOR {
        return Err(refuse(
            "the sweep path is closed: an open wire sweeps only an open path or one full circle",
        ));
    }

    let spans = path
        .iter()
        .map(EdgeRun::span)
        .collect::<Result<Vec<_>, _>>()?;
    let joined = joined_path(&spans)?;
    if !detect_kinks(&joined).is_empty() {
        return Err(refuse(
            "the sweep path has a kink: an open wire sweeps only a tangent-continuous path",
        ));
    }
    let chain = open_chain(topo, wire)?;
    let faces = faces_along_curve(topo, &chain, &joined)?;
    finish(topo, faces)
}

/// The path's edges in chain order, each read along the way the chain runs,
/// recognized back to lines and circles where they are spline-encoded.
fn path_runs(topo: &mut Topology, edges: &[EdgeId]) -> Result<Vec<EdgeRun>, OperationsError> {
    let tol = Tolerance::new().linear;
    let reach = tol * PATH_JOIN_FACTOR;
    for &edge in edges {
        normalize_edge_curve(topo, edge, tol)?;
    }
    let ends = |topo: &Topology, edge: EdgeId| -> Result<(Point3, Point3), OperationsError> {
        let data = topo.edge(edge)?;
        Ok((
            topo.vertex(data.start())?.point(),
            topo.vertex(data.end())?.point(),
        ))
    };
    let mut runs: Vec<EdgeRun> = Vec::with_capacity(edges.len());
    for (i, &edge) in edges.iter().enumerate() {
        let (start, stop) = ends(topo, edge)?;
        let forward = if let Some(previous) = runs.last() {
            if (start - previous.to).length() <= reach {
                true
            } else if (stop - previous.to).length() <= reach {
                false
            } else {
                return Err(refuse("the sweep path's edges do not connect end to end"));
            }
        } else if let Some(&second) = edges.get(i + 1) {
            let (c, d) = ends(topo, second)?;
            let gap = |p: Point3| (p - c).length().min((p - d).length());
            gap(stop) <= gap(start)
        } else {
            true
        };
        let (from, to) = if forward {
            (start, stop)
        } else {
            (stop, start)
        };
        runs.push(EdgeRun {
            edge: OrientedEdge::new(edge, forward),
            curve: topo.edge(edge)?.curve().clone(),
            from,
            to,
        });
    }
    Ok(runs)
}

/// Clamped curves, each starting where the one before ends, as one curve of
/// their highest degree, each given parameter in proportion to its length.
fn joined_path(spans: &[NurbsCurve]) -> Result<NurbsCurve, OperationsError> {
    if let [only] = spans {
        return Ok(only.clone());
    }
    let degree = spans.iter().map(NurbsCurve::degree).max().unwrap_or(1);
    let mut knots: Vec<f64> = Vec::new();
    let mut points: Vec<Point3> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    let mut offset = 0.0;
    for span in spans {
        let span = curve_degree_elevate(span, degree - span.degree())?;
        let cps = span.control_points();
        let width: f64 = cps.windows(2).map(|w| (w[1] - w[0]).length()).sum();
        let (d0, d1) = span.domain();
        let mapped: Vec<f64> = span
            .knots()
            .iter()
            .map(|k| offset + (k - d0) / (d1 - d0) * width)
            .collect();
        if let Some(&joint) = weights.last() {
            // Scaling a rational curve's weights keeps the curve, and the two
            // pieces must agree on the weight of the point they share.
            let scale = joint / span.weights()[0];
            knots.extend_from_slice(&mapped[degree + 1..mapped.len() - 1]);
            points.extend_from_slice(&cps[1..]);
            weights.extend(span.weights()[1..].iter().map(|w| w * scale));
        } else {
            knots.extend_from_slice(&mapped[..mapped.len() - 1]);
            points.extend_from_slice(cps);
            weights.extend_from_slice(span.weights());
        }
        offset += width;
    }
    knots.push(offset);
    Ok(NurbsCurve::new(degree, knots, points, weights)?)
}

/// One face per wire edge along a curved path: each control point of the edge
/// runs its own fitted trajectory, and the edge's own curve is every section.
fn faces_along_curve(
    topo: &mut Topology,
    chain: &OpenChain,
    path: &NurbsCurve,
) -> Result<Vec<FaceId>, OperationsError> {
    let tol = Tolerance::new().linear;
    let spans = chain
        .runs
        .iter()
        .map(EdgeRun::span)
        .collect::<Result<Vec<_>, _>>()?;
    let mut first_point = Vec::with_capacity(spans.len());
    let mut points = Vec::new();
    for span in &spans {
        first_point.push(points.len());
        points.extend_from_slice(span.control_points());
    }
    let (trajectories, end_map) = path_trajectories(path, &points)?;

    let far: Vec<VertexId> = chain
        .points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(end_map.mul_point(p), tol)))
        .collect();
    let mut rails = Vec::with_capacity(chain.vertices.len());
    for (j, (&near, &end)) in chain.vertices.iter().zip(&far).enumerate() {
        let trajectory = first_point
            .get(j)
            .map_or(&trajectories[points.len() - 1], |&k| &trajectories[k]);
        let curve = if trajectory.degree() == 1 {
            EdgeCurve::Line
        } else {
            EdgeCurve::NurbsCurve(trajectory.clone())
        };
        rails.push(topo.add_edge(Edge::new(near, end, curve)));
    }

    let mut faces = Vec::with_capacity(spans.len());
    for (i, (run, span)) in chain.runs.iter().zip(&spans).enumerate() {
        let rows: Vec<Vec<Point3>> = trajectories[first_point[i]..]
            .iter()
            .take(span.control_points().len())
            .map(|t| t.control_points().to_vec())
            .collect();
        let columns = rows[0].len();
        let weights: Vec<Vec<f64>> = span.weights().iter().map(|&w| vec![w; columns]).collect();
        let surface = NurbsSurface::new(
            span.degree(),
            trajectories[0].degree(),
            span.knots().to_vec(),
            trajectories[0].knots().to_vec(),
            rows,
            weights,
        )?;
        let (s, e) = if run.is_forward() {
            (far[i], far[i + 1])
        } else {
            (far[i + 1], far[i])
        };
        let (curve, _) = curve_image(&run.curve, &end_map)?;
        let far_edge = topo.add_edge(Edge::new(s, e, curve));
        faces.push(band_face(
            topo,
            (run.edge, OrientedEdge::new(far_edge, run.is_forward())),
            (Some(rails[i]), Some(rails[i + 1])),
            FaceSurface::Nurbs(surface),
            false,
        )?);
    }
    Ok(faces)
}
