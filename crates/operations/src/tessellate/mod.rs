//! Tessellation: convert B-Rep faces to triangle meshes.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::manual_let_else,
    clippy::tuple_array_conversions,
    clippy::imprecise_flops,
    clippy::too_many_lines,
    clippy::option_if_let_else,
    clippy::bool_to_int_with_if,
    clippy::if_same_then_else,
    clippy::used_underscore_binding,
    clippy::map_unwrap_or
)]

mod edge_sampling;
mod face;
mod mesh_ops;
mod nonplanar;
mod nurbs;
mod planar;
mod solid;
#[cfg(test)]
mod tests;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;

// Re-export all public items.
pub(crate) use face::sphere_face_is_off_grid;
pub use face::{tessellate_with_uvs, tessellate_with_uvs_a};
pub(crate) use mesh_ops::COINCIDENT_DEDUPE_GRID;
pub(crate) use mesh_ops::drawn_edge_samples;
pub use mesh_ops::{
    EdgeLines, boundary_edge_count, is_watertight, non_manifold_edge_count, sample_solid_edges,
    sample_solid_edges_filtered,
};
pub(crate) use nonplanar::sphere_loop_reads_right;
pub use solid::{
    tessellate_solid, tessellate_solid_faceted_with_tolerance, tessellate_solid_for_boolean,
    tessellate_solid_grouped_with_tolerance, tessellate_solid_with_tolerance,
};

/// Merge-grid cell size for tolerance-based vertex deduplication.
///
/// Vertices within this distance are quantized to the same grid cell and share
/// a single global vertex index. This catches near-identical vertices produced
/// when boolean operations create separate edge entities for the same curve.
const MERGE_GRID: f64 = 1e-7;

/// The global vertex at each [`point_merge_key`] of a solid's mesh. No
/// result may depend on the order of its entries.
pub(super) type PointPool = brepkit_math::det_hash::WordHashMap<(i64, i64, i64), u32>;

/// Quantize a 3D point to a spatial grid cell for tolerance-based deduplication.
///
/// Two points whose coordinates differ by less than `grid` will (usually) map to
/// the same `(i64, i64, i64)` key. Grid-boundary splits are possible but rare,
/// and the subsequent CDT / weld phases handle any remaining gaps.
pub(super) fn point_merge_key(pt: Point3, grid: f64) -> (i64, i64, i64) {
    #[allow(clippy::cast_possible_truncation)]
    (
        (pt.x() / grid).round() as i64,
        (pt.y() / grid).round() as i64,
        (pt.z() / grid).round() as i64,
    )
}

/// A triangle mesh produced by tessellation.
#[derive(Debug, Clone, Default)]
pub struct TriangleMesh {
    /// Vertex positions.
    pub positions: Vec<Point3>,
    /// Per-vertex normals.
    pub normals: Vec<Vec3>,
    /// Triangle indices (groups of 3).
    pub indices: Vec<u32>,
}

/// A triangle mesh with per-vertex UV coordinates.
#[derive(Debug, Clone, Default)]
pub struct TriangleMeshUV {
    /// The base mesh (positions, normals, indices).
    pub mesh: TriangleMesh,
    /// Per-vertex UV coordinates (same length as `mesh.positions`).
    pub uvs: Vec<[f64; 2]>,
}

/// Kind of special handling needed for analytic surface tessellation.
pub(super) enum AnalyticKind {
    /// Standard quad grid with no degenerate handling.
    General,
    /// Triangle fan at v extremes (sphere poles at v_min and v_max).
    SpherePole,
    /// Triangle fan at v_min (cone apex at v = 0).
    ConeApex,
    /// Triangle fan at v_max only (sphere north pole for a hemisphere face).
    VMaxPole,
}

/// How far out a mesh point may lie, in each coordinate. The mesher welds
/// points on integer grids as fine as [`MERGE_GRID`]. A point this far out
/// lies 1e18 cells out, and a cell index, its neighbours and their
/// differences still fit in an `i64`, whose largest value is 9.2e18. Past
/// it, a cell index overflows.
const MESH_REACH: f64 = 1e11;

/// The largest magnitude among `values`, or infinity if one is NaN.
fn largest(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(0.0, |most: f64, value| {
        let size = value.abs();
        if size <= most {
            most
        } else if size.is_nan() {
            f64::INFINITY
        } else {
            size
        }
    })
}

/// The largest coordinate magnitude of each of `points`.
fn points_reach<'a>(points: impl IntoIterator<Item = &'a Point3>) -> f64 {
    largest(points.into_iter().flat_map(|p| [p.x(), p.y(), p.z()]))
}

/// A bound on the coordinate magnitudes of the points of `curve`. A NURBS
/// curve lies in the hull of its control points.
fn curve_reach(curve: &brepkit_topology::edge::EdgeCurve) -> f64 {
    use brepkit_topology::edge::EdgeCurve;
    let corners = |b: brepkit_math::aabb::Aabb3| points_reach(&[b.min, b.max]);
    match curve {
        EdgeCurve::Line => 0.0,
        EdgeCurve::Circle(c) => corners(c.aabb()),
        EdgeCurve::Ellipse(e) => corners(e.aabb()),
        EdgeCurve::NurbsCurve(n) => points_reach(n.control_points()),
    }
}

/// A bound on `|dir · (p - from)|` over the points `p` of `curve`, beyond
/// its ends, for a unit `dir`. A NURBS curve lies in the hull of its
/// control points.
fn curve_extent_along(curve: &brepkit_topology::edge::EdgeCurve, from: Point3, dir: Vec3) -> f64 {
    use brepkit_topology::edge::EdgeCurve;
    match curve {
        EdgeCurve::Line => 0.0,
        EdgeCurve::Circle(c) => {
            let swing = c.radius() * dir.dot(c.u_axis()).hypot(dir.dot(c.v_axis()));
            dir.dot(c.center() - from).abs() + swing
        }
        EdgeCurve::Ellipse(e) => {
            let swing =
                (e.semi_major() * dir.dot(e.u_axis())).hypot(e.semi_minor() * dir.dot(e.v_axis()));
            dir.dot(e.center() - from).abs() + swing
        }
        EdgeCurve::NurbsCurve(n) => largest(n.control_points().iter().map(|&q| dir.dot(q - from))),
    }
}

/// Refuses a face whose mesh points could lie past [`MESH_REACH`].
///
/// The bound holds the face's vertices and edge curves, and the points the
/// mesher reads from the surface's parameters. A sphere's or torus's points
/// lie within its radii of its centre at any parameters. A NURBS surface's
/// points lie in its control hull, since a projection onto it stays in its
/// domain. A cylinder's and a cone's mesher reads the height along the axis
/// at which each wire point projects: a cone reads it over the sine of its
/// half angle, as a distance along a generator, or below a sine of 1e-15 the
/// distance from the axis over the cosine, as `ConicalSurface::project_point`
/// does. The wires need not lie on the surface, so a cone of small angle can
/// read a short wire as a generator far longer than the wire.
pub(super) fn check_reach(topo: &Topology, face: FaceId) -> Result<(), crate::OperationsError> {
    use brepkit_topology::face::FaceSurface;
    let face_data = topo.face(face)?;
    let mut edges = Vec::new();
    for wire in
        std::iter::once(face_data.outer_wire()).chain(face_data.inner_wires().iter().copied())
    {
        for oe in topo.wire(wire)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let ends = [
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            ];
            edges.push((edge.curve(), ends));
        }
    }
    let boundary = largest(
        edges
            .iter()
            .flat_map(|(curve, ends)| [points_reach(ends), curve_reach(curve)]),
    );
    // The largest `|dir · (p - from)|` over the wires' points `p`.
    let along = |from: Point3, dir: Vec3| {
        largest(edges.iter().flat_map(|(curve, ends)| {
            [
                dir.dot(ends[0] - from),
                dir.dot(ends[1] - from),
                curve_extent_along(curve, from, dir),
            ]
        }))
    };
    let surface_reach = match face_data.surface() {
        FaceSurface::Plane { .. } => 0.0,
        FaceSurface::Cylinder(c) => {
            points_reach(&[c.origin()]) + c.radius() + along(c.origin(), c.axis())
        }
        FaceSurface::Cone(c) => {
            let (sin_a, cos_a) = c.half_angle().sin_cos();
            let run = if sin_a.abs() > 1e-15 {
                along(c.apex(), c.axis()) / sin_a.abs()
            } else {
                3.0_f64.sqrt() * (boundary + points_reach(&[c.apex()])) / cos_a.abs()
            };
            points_reach(&[c.apex()]) + run
        }
        FaceSurface::Sphere(s) => points_reach(&[s.center()]) + s.radius(),
        FaceSurface::Torus(t) => points_reach(&[t.center()]) + t.major_radius() + t.minor_radius(),
        FaceSurface::Nurbs(n) => points_reach(n.control_points().iter().flatten()),
    };
    let reach = largest([boundary, surface_reach]);
    if reach <= MESH_REACH {
        Ok(())
    } else {
        Err(crate::OperationsError::InvalidInput {
            reason: format!(
                "face {} reaches {reach:e} from the origin, too far out to mesh: \
                 the mesher's integer grids hold points to {MESH_REACH:e}",
                face.index()
            ),
        })
    }
}

/// The most samples the mesher takes in one grid across a face.
///
/// A sample costs about a hundred bytes of mesh and working space, so a
/// grid at the limit holds about a gigabyte, near what a 32-bit wasm
/// instance can address. Over the operations and io tests, the largest face
/// mesh holds 2.2e5 samples and the largest solid mesh 2.0e5: the limit is
/// 45 times the largest face.
const MAX_SAMPLES: f64 = 1e7;

/// The most samples the mesher takes along one edge, or along one curved
/// direction across a face over a whole turn.
///
/// A face's boundary samples are triangulated together before its interior
/// grid is sized, so this bounds that work too. Over the operations and io
/// tests no edge asks for more than 6.6e3 samples: the limit is 150 times
/// that.
const MAX_EDGE_SAMPLES: f64 = 1e6;

/// The words that mark a refusal for too many samples, so that a mesher's
/// fallback passes it on instead of trying a mesher that samples the same
/// grid.
const SAMPLE_LIMIT: &str = "mesh samples, past the mesher's limit of";

/// The refusal of `what`, which would take `samples` samples at
/// `deflection`, past `limit`.
fn too_many_samples(
    what: &str,
    samples: f64,
    limit: f64,
    deflection: f64,
) -> crate::OperationsError {
    crate::OperationsError::InvalidInput {
        reason: format!(
            "{what} would take about {samples:.1e} {SAMPLE_LIMIT} {limit:e} at deflection \
             {deflection:e}"
        ),
    }
}

/// Whether `error` refuses a mesh for too many samples.
pub(super) fn is_sample_limit(error: &crate::OperationsError) -> bool {
    matches!(error, crate::OperationsError::InvalidInput { reason } if reason.contains(SAMPLE_LIMIT))
}

/// Refuses a grid of `columns` by `rows` samples past [`MAX_SAMPLES`], before
/// a mesher allocates it.
pub(super) fn check_grid(
    columns: usize,
    rows: usize,
    deflection: f64,
) -> Result<(), crate::OperationsError> {
    let samples = columns as f64 * rows as f64;
    if samples <= MAX_SAMPLES {
        Ok(())
    } else {
        Err(too_many_samples(
            "a face's grid",
            samples,
            MAX_SAMPLES,
            deflection,
        ))
    }
}

/// The segments the chord formula gives an arc of `radius` over `span`
/// with the curvature floor: the most any mesher takes along such an arc.
fn arc_segments(radius: f64, span: f64, deflection: f64, angular_tol: f64) -> f64 {
    edge_sampling::segments_for_chord_deviation_a(radius, span, deflection, angular_tol, true)
        as f64
}

/// The samples the mesher takes along an edge on `curve`, at most. A NURBS
/// edge's samples are capped at 4096 spans where they are chosen.
fn edge_samples(
    curve: &brepkit_topology::edge::EdgeCurve,
    deflection: f64,
    angular_tol: f64,
) -> f64 {
    use brepkit_topology::edge::EdgeCurve;
    use std::f64::consts::TAU;
    match curve {
        EdgeCurve::Line => 2.0,
        EdgeCurve::Circle(c) => arc_segments(c.radius(), TAU, deflection, angular_tol) + 1.0,
        // The meshers read an ellipse at its largest radius of curvature.
        EdgeCurve::Ellipse(e) => {
            let radius = e.semi_major() * e.semi_major() / e.semi_minor();
            arc_segments(radius, TAU, deflection, angular_tol) + 1.0
        }
        EdgeCurve::NurbsCurve(_) => 4097.0,
    }
}

/// The samples the mesher takes along each curved direction of a face on
/// `surface`, at most: the chord formula's segments over a whole turn. A
/// cone's columns are counted at the radius its meshers count them at,
/// from the face's wires ([`nurbs::cone_column_radius`]). A plane has no
/// curved direction, and a NURBS face's grid is capped where it is chosen.
fn direction_samples(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    angular_tol: f64,
) -> [f64; 2] {
    use brepkit_topology::face::FaceSurface;
    use std::f64::consts::TAU;
    let arc = |radius: f64| arc_segments(radius, TAU, deflection, angular_tol);
    match face_data.surface() {
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => [0.0, 0.0],
        FaceSurface::Cylinder(c) => [arc(c.radius()), 0.0],
        FaceSurface::Cone(c) => {
            let v_range = nurbs::compute_v_param_range(topo, face_data, |p| c.project_point(p).1);
            [arc(nurbs::cone_column_radius(c, v_range)), 0.0]
        }
        FaceSurface::Sphere(s) => [arc(s.radius()), 0.0],
        FaceSurface::Torus(t) => [
            arc(t.major_radius() + t.minor_radius()),
            arc(t.minor_radius()),
        ],
    }
}

/// Refuses a face, before any of its samples is allocated, when one of its
/// edges or one direction across it would take more than
/// [`MAX_EDGE_SAMPLES`] samples at `deflection`. A grid across both
/// directions of a sphere or a torus is checked by the mesher that sizes it
/// from the part of the surface the face covers ([`check_grid`]).
pub(super) fn check_samples(
    topo: &Topology,
    face: FaceId,
    deflection: f64,
    angular_tol: f64,
) -> Result<(), crate::OperationsError> {
    let face_data = topo.face(face)?;
    for wire in
        std::iter::once(face_data.outer_wire()).chain(face_data.inner_wires().iter().copied())
    {
        for oe in topo.wire(wire)?.edges() {
            check_edge_samples(topo.edge(oe.edge())?.curve(), deflection, angular_tol)?;
        }
    }
    let directions = direction_samples(topo, face_data, deflection, angular_tol);
    match directions.into_iter().find(|&n| n > MAX_EDGE_SAMPLES) {
        None => Ok(()),
        Some(samples) => Err(too_many_samples(
            &format!("face {}", face.index()),
            samples,
            MAX_EDGE_SAMPLES,
            deflection,
        )),
    }
}

/// Refuses an edge on `curve` whose mesh would take more than
/// [`MAX_EDGE_SAMPLES`] samples at `deflection`.
pub(super) fn check_edge_samples(
    curve: &brepkit_topology::edge::EdgeCurve,
    deflection: f64,
    angular_tol: f64,
) -> Result<(), crate::OperationsError> {
    let samples = edge_samples(curve, deflection, angular_tol);
    if samples <= MAX_EDGE_SAMPLES {
        Ok(())
    } else {
        Err(too_many_samples(
            "an edge",
            samples,
            MAX_EDGE_SAMPLES,
            deflection,
        ))
    }
}

/// A deflection the meshers can refine down to: a zero or negative one
/// subdivides without end.
pub(super) fn check_deflection(deflection: f64) -> Result<(), crate::OperationsError> {
    if deflection > 0.0 && deflection.is_finite() {
        Ok(())
    } else {
        Err(crate::OperationsError::InvalidInput {
            reason: format!(
                "tessellation deflection must be positive and finite, got {deflection}"
            ),
        })
    }
}

/// Tessellate a face into a triangle mesh.
///
/// For planar faces, this performs fan triangulation from the first vertex,
/// which produces correct results for convex polygons.
///
/// For NURBS faces, the surface is sampled on a uniform (u, v) grid whose
/// density is derived from `deflection` — smaller values produce finer meshes.
///
/// # Errors
///
/// Returns an error if the face geometry cannot be tessellated.
pub fn tessellate(
    topo: &Topology,
    face: FaceId,
    deflection: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    tessellate_with_uvs(topo, face, deflection).map(|uv| uv.mesh)
}

/// Tessellate a face with explicit linear and angular tolerances.
///
/// `angular_tol` (radians) caps the per-segment tangent turn; pass `0.0` to
/// disable the angular criterion (linear-only path).
///
/// # Errors
///
/// Returns an error if the face geometry cannot be tessellated.
pub fn tessellate_with_tolerance(
    topo: &Topology,
    face: FaceId,
    deflection: f64,
    angular_tol: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    face::tessellate_with_uvs_a(topo, face, deflection, angular_tol).map(|uv| uv.mesh)
}

/// Check if a mesh is watertight (every edge shared by exactly 2 triangles).
///
/// Returns `true` if the mesh is a closed 2-manifold: every half-edge
/// `(a, b)` in the mesh has a corresponding reverse half-edge `(b, a)`.
///
/// This is useful for validating that `tessellate_solid` produces
/// gap-free meshes.
#[cfg(test)]
#[must_use]
pub(super) fn position_based_boundary_count(mesh: &TriangleMesh) -> usize {
    /// 1um grid -- intentionally coarser than `MERGE_GRID` (1e-7) to catch
    /// gaps the production pipeline should have closed.
    const DIAGNOSTIC_GRID: f64 = 1e-6;

    use brepkit_math::det_hash::{DetHashMap, DetHashSet};

    // Build canonical vertex ID from snapped position.
    let mut pos_to_canonical: DetHashMap<(i64, i64, i64), u32> = DetHashMap::default();
    let mut canonical_ids: Vec<u32> = Vec::with_capacity(mesh.positions.len());
    let mut next_id: u32 = 0;

    for pos in &mesh.positions {
        let key = point_merge_key(*pos, DIAGNOSTIC_GRID);
        let id = *pos_to_canonical.entry(key).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        canonical_ids.push(id);
    }

    // Build half-edge set using canonical IDs.
    let mut half_edges: DetHashSet<(u32, u32)> = DetHashSet::default();
    for tri in mesh.indices.chunks_exact(3) {
        let i0 = canonical_ids[tri[0] as usize];
        let i1 = canonical_ids[tri[1] as usize];
        let i2 = canonical_ids[tri[2] as usize];
        half_edges.insert((i0, i1));
        half_edges.insert((i1, i2));
        half_edges.insert((i2, i0));
    }

    half_edges
        .iter()
        .filter(|&&(a, b)| !half_edges.contains(&(b, a)))
        .count()
}
