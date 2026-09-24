//! Bounding box computation for B-rep solids.

use std::collections::HashSet;

use brepkit_math::aabb::Aabb3;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use super::helpers::collect_solid_vertex_points;

/// Compute the axis-aligned bounding box of a solid.
///
/// Uses vertex positions as the base AABB, then expands for non-planar
/// surfaces by sampling edge midpoints on the surface. This captures
/// curvature without over-expanding (unlike projecting the surface's
/// full theoretical extent).
///
/// # Errors
///
/// Returns an error if the solid has no vertices or a topology lookup fails.
pub fn solid_bounding_box(
    topo: &Topology,
    solid: SolidId,
) -> Result<Aabb3, crate::OperationsError> {
    placed_solid_bounding_box(topo, solid, Placement(None))
}

/// Compute the axis-aligned bounding box of a solid as it would sit after
/// applying the affine `transform`, without modifying the topology.
///
/// The box is as tight as [`solid_bounding_box`] of a transformed copy: conic
/// edge extremes, NURBS hulls, and whole-sphere and whole-torus extents are
/// all evaluated in the placed frame, so a rotated cylinder is not bounded by
/// the rotated corners of its local box.
///
/// # Errors
///
/// Returns an error if the solid has no vertices or a topology lookup fails.
pub fn solid_bounding_box_transformed(
    topo: &Topology,
    solid: SolidId,
    transform: &Mat4,
) -> Result<Aabb3, crate::OperationsError> {
    placed_solid_bounding_box(topo, solid, Placement(Some(transform)))
}

/// An optional affine map applied to geometry before it enters a box.
#[derive(Clone, Copy)]
struct Placement<'a>(Option<&'a Mat4>);

impl Placement<'_> {
    fn point(self, p: Point3) -> Point3 {
        self.0.map_or(p, |m| m.mul_point(p))
    }

    fn vector(self, v: Vec3) -> Vec3 {
        if self.0.is_some() {
            Vec3::new(self.row(0).dot(v), self.row(1).dot(v), self.row(2).dot(v))
        } else {
            v
        }
    }

    /// Row `k` of the linear part: the direction whose dot product with a
    /// source-frame offset gives the placed offset's `k`-th coordinate.
    fn row(self, k: usize) -> Vec3 {
        self.0.map_or_else(
            || {
                let mut e = [0.0; 3];
                e[k] = 1.0;
                Vec3::new(e[0], e[1], e[2])
            },
            |m| Vec3::new(m.0[k][0], m.0[k][1], m.0[k][2]),
        )
    }
}

fn placed_solid_bounding_box(
    topo: &Topology,
    solid: SolidId,
    placement: Placement<'_>,
) -> Result<Aabb3, crate::OperationsError> {
    let points = collect_solid_vertex_points(topo, solid)?;
    let mut aabb =
        Aabb3::try_from_points(points.iter().map(|&p| placement.point(p))).ok_or_else(|| {
            crate::OperationsError::InvalidInput {
                reason: "solid has no vertices".into(),
            }
        })?;

    // Expand AABB for non-planar faces by sampling edge midpoints on the
    // actual surface. This captures curvature (e.g., the arc midpoint of a
    // fillet cylinder) without over-expanding to the surface's full extent.
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    for &fid in shell.faces() {
        if let Ok(face) = topo.face(fid) {
            expand_aabb_for_face(topo, &mut aabb, fid, face.surface(), placement);
        }
    }

    Ok(aabb)
}

/// Compute a conservative axis-aligned bounding box over an arbitrary set of
/// faces (e.g. one connected component of a multi-region solid).
///
/// Like [`solid_bounding_box`], the box starts from the faces' vertex
/// positions and is then expanded for surface curvature, so the returned box
/// is a conservative *outer* bound of every face in the set. Used by the
/// disjoint-fuse fast path to test whether two operands' components are
/// spatially separated.
///
/// # Errors
///
/// Returns an error if the face set is empty (no vertices) or a topology
/// lookup fails.
pub fn face_set_bounding_box(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<Aabb3, crate::OperationsError> {
    let mut vertex_ids = HashSet::new();
    for &fid in faces {
        let face = topo.face(fid)?;
        for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
        {
            let wire = topo.wire(wire_id)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                vertex_ids.insert(edge.start());
                vertex_ids.insert(edge.end());
            }
        }
    }

    let mut points = Vec::with_capacity(vertex_ids.len());
    for vid in vertex_ids {
        points.push(topo.vertex(vid)?.point());
    }
    let mut aabb = Aabb3::try_from_points(points.iter().copied()).ok_or_else(|| {
        crate::OperationsError::InvalidInput {
            reason: "face set has no vertices".into(),
        }
    })?;

    for &fid in faces {
        if let Ok(face) = topo.face(fid) {
            expand_aabb_for_face(topo, &mut aabb, fid, face.surface(), Placement(None));
        }
    }

    Ok(aabb)
}

/// Expand an AABB to include a point.
fn aabb_include(aabb: &mut Aabb3, p: Point3) {
    *aabb = aabb.union(Aabb3 { min: p, max: p });
}

/// Expand an AABB for a face, accounting for surface curvature.
///
/// Uses different strategies based on surface type:
/// - **Sphere**: analytic full-surface expansion
/// - **Torus**: analytic expansion only for the doubly-periodic seam topology;
///   trimmed patches use their sampled wire bounds
/// - **Cylinder/Cone**: wire-bounded expansion (sample edge midpoints
///   to avoid over-expanding for partial arcs like fillets)
/// - **NURBS**: sparse interior grid sampling
/// - **Plane**: no expansion needed
#[allow(clippy::too_many_lines)]
fn expand_aabb_for_face(
    topo: &Topology,
    aabb: &mut Aabb3,
    face_id: brepkit_topology::face::FaceId,
    surface: &FaceSurface,
    placement: Placement<'_>,
) {
    // Always sample wire midpoints — captures curvature of curved boundary
    // edges (Circle, Ellipse, NurbsCurve) regardless of surface type.
    // Critical for: cone base discs (Plane face with circle edge), partial
    // arcs whose extremes lie between vertices, and any curved edge on a
    // planar face.
    sample_face_wire_midpoints(topo, aabb, face_id, placement);
    expand_boundary_edges_exact(topo, aabb, face_id, placement);

    match surface {
        FaceSurface::Plane { .. } => {}

        // Spheres are represented as full or near-full surfaces. A torus can
        // also represent a trimmed fillet patch, where expanding to the full
        // analytic torus shifts the public solid bounds by the minor radius.
        // The placed half-extent along world axis k is the support of the
        // source surface in direction d = row k of the linear part: r·|d| for
        // a sphere, R·|d ⊥ axis| + r·|d| for a torus.
        FaceSurface::Sphere(s) => {
            let c = placement.point(s.center());
            let r = s.radius();
            let h = [0, 1, 2].map(|k| r * placement.row(k).length());
            aabb_include(aabb, Point3::new(c.x() - h[0], c.y() - h[1], c.z() - h[2]));
            aabb_include(aabb, Point3::new(c.x() + h[0], c.y() + h[1], c.z() + h[2]));
        }
        FaceSurface::Torus(t) => {
            if torus_face_is_full_periodic(topo, face_id) {
                let c = placement.point(t.center());
                let r_major = t.major_radius();
                let r_minor = t.minor_radius();
                let axis = t.z_axis();
                let h = [0, 1, 2].map(|k| {
                    let d = placement.row(k);
                    let along = d.dot(axis);
                    let perp = along.mul_add(-along, d.dot(d)).max(0.0).sqrt();
                    r_major.mul_add(perp, r_minor * d.length())
                });
                aabb_include(aabb, Point3::new(c.x() - h[0], c.y() - h[1], c.z() - h[2]));
                aabb_include(aabb, Point3::new(c.x() + h[0], c.y() + h[1], c.z() + h[2]));
            }
        }

        // A ruled quadric's extreme in any direction is attained along a whole
        // ruling, whose ends lie on boundary edges: the exact edge bounds
        // already cover the face.
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => {}

        // NURBS: sample the surface at a sparse interior grid.
        FaceSurface::Nurbs(nurbs) => {
            let (u_min, u_max) = nurbs.domain_u();
            let (v_min, v_max) = nurbs.domain_v();
            let n_samples = 4;
            #[allow(clippy::cast_precision_loss)]
            for iu in 1..n_samples {
                let u = u_min + (u_max - u_min) * (iu as f64) / (n_samples as f64);
                for iv in 1..n_samples {
                    let v = v_min + (v_max - v_min) * (iv as f64) / (n_samples as f64);
                    aabb_include(aabb, placement.point(nurbs.evaluate(u, v)));
                }
            }
        }
    }
}

/// Return whether a torus face is the minimal doubly-periodic whole-torus
/// topology (`a, b, a⁻¹, b⁻¹`). Trimmed torus patches have ordinary boundary
/// edges and must not expand to the analytic surface's full extent.
fn torus_face_is_full_periodic(topo: &Topology, face_id: FaceId) -> bool {
    let Ok(face) = topo.face(face_id) else {
        return false;
    };
    let Ok(wire) = topo.wire(face.outer_wire()) else {
        return false;
    };
    let oriented = wire.edges();
    if oriented.len() != 4
        || oriented[0].edge() != oriented[2].edge()
        || oriented[1].edge() != oriented[3].edge()
        || oriented[0].edge() == oriented[1].edge()
        || oriented[0].is_forward() == oriented[2].is_forward()
        || oriented[1].is_forward() == oriented[3].is_forward()
    {
        return false;
    }
    [oriented[0].edge(), oriented[1].edge()]
        .into_iter()
        .all(|edge_id| {
            topo.edge(edge_id)
                .is_ok_and(|edge| edge.start() == edge.end())
        })
}

/// Sample edge midpoints along a face's outer wire to expand the AABB.
///
/// Returns `true` if any curved (non-Line) edges were found. For curved
/// edges (Circle, Ellipse, NurbsCurve), sampling at 0.25, 0.5, 0.75
/// captures the curvature.
fn sample_face_wire_midpoints(
    topo: &Topology,
    aabb: &mut Aabb3,
    face_id: brepkit_topology::face::FaceId,
    placement: Placement<'_>,
) -> bool {
    let Ok(face) = topo.face(face_id) else {
        return false;
    };
    let Ok(wire) = topo.wire(face.outer_wire()) else {
        return false;
    };
    let mut has_curved = false;
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        if !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
            has_curved = true;
        }
        let Ok(sv) = topo.vertex(edge.start()) else {
            continue;
        };
        let Ok(ev) = topo.vertex(edge.end()) else {
            continue;
        };
        let p_start = sv.point();
        let p_end = ev.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(p_start, p_end);
        for &frac in &[0.25, 0.5, 0.75] {
            let t = t0 + (t1 - t0) * frac;
            let pt = edge.curve().evaluate_with_endpoints(t, p_start, p_end);
            aabb_include(aabb, placement.point(pt));
        }
    }
    has_curved
}

/// Include the extremes of every boundary edge over its own span.
///
/// Circle and ellipse edges reach their axis-aligned extremes at analytic
/// parameters; only those inside the edge's span count, so a quarter arc
/// contributes nothing beyond its endpoints. NURBS edges contribute the
/// control points of their subdivided Bezier segments, which never
/// under-shoot the curve: the intersect early-out and the compound-cut
/// contact gate both rely on that.
fn expand_boundary_edges_exact(
    topo: &Topology,
    aabb: &mut Aabb3,
    face_id: brepkit_topology::face::FaceId,
    placement: Placement<'_>,
) {
    use brepkit_topology::edge::EdgeCurve;

    let Ok(face) = topo.face(face_id) else {
        return;
    };
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let Ok(wire) = topo.wire(wire_id) else {
            continue;
        };
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
                continue;
            };
            let (p_start, p_end) = (sv.point(), ev.point());
            let (t0, t1) = edge.curve().domain_with_endpoints(p_start, p_end);
            match edge.curve() {
                EdgeCurve::Line => {}
                EdgeCurve::Circle(c) => {
                    let r = c.radius();
                    include_conic_extremes(
                        aabb,
                        placement.vector(c.u_axis()),
                        placement.vector(c.v_axis()),
                        r,
                        r,
                        t0,
                        t1,
                        |t| placement.point(c.evaluate(t)),
                    );
                }
                EdgeCurve::Ellipse(e) => include_conic_extremes(
                    aabb,
                    placement.vector(e.u_axis()),
                    placement.vector(e.v_axis()),
                    e.semi_major(),
                    e.semi_minor(),
                    t0,
                    t1,
                    |t| placement.point(e.evaluate(t)),
                ),
                EdgeCurve::NurbsCurve(n) => {
                    include_nurbs_edge_hull(aabb, n, t0, t1, placement);
                }
            }
        }
    }
}

/// Conservative bound of a NURBS edge over `[t0, t1]`.
///
/// The curve is split to the edge's span, decomposed into Bezier segments,
/// and each segment is subdivided by de Casteljau in homogeneous space. A
/// rational Bezier segment with positive weights lies inside the convex hull
/// of its control points, so including every subdivided control point can
/// never under-shoot; three subdivisions keep a quarter-circle span within
/// half a percent of its radius.
fn include_nurbs_edge_hull(
    aabb: &mut Aabb3,
    curve: &brepkit_math::nurbs::curve::NurbsCurve,
    t0: f64,
    t1: f64,
    placement: Placement<'_>,
) {
    use brepkit_math::nurbs::decompose::curve_to_bezier_segments;
    use brepkit_math::nurbs::knot_ops::curve_split;
    const SUBDIVISIONS: usize = 3;

    let (d0, d1) = curve.domain();
    let eps = 1e-9 * (d1 - d0).abs();
    let (lo, hi) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
    let mut span = curve.clone();
    if lo > d0 + eps
        && lo < d1 - eps
        && let Ok((_, right)) = curve_split(&span, lo)
    {
        span = right;
    }
    if hi > d0 + eps
        && hi < d1 - eps
        && let Ok((left, _)) = curve_split(&span, hi)
    {
        span = left;
    }
    let segments = match curve_to_bezier_segments(&span) {
        Ok(segments) => segments,
        Err(_) => vec![span],
    };
    for segment in &segments {
        let poles: Vec<[f64; 4]> = segment
            .control_points()
            .iter()
            .zip(segment.weights())
            .map(|(&p, &w)| {
                let p = placement.point(p);
                [p.x() * w, p.y() * w, p.z() * w, w]
            })
            .collect();
        include_bezier_hull(aabb, &poles, SUBDIVISIONS);
    }
}

/// Include the projected control points of a homogeneous Bezier segment after
/// `depth` midpoint subdivisions.
fn include_bezier_hull(aabb: &mut Aabb3, poles: &[[f64; 4]], depth: usize) {
    if depth == 0 || poles.len() < 2 {
        for p in poles {
            if p[3] > 0.0 {
                aabb_include(aabb, Point3::new(p[0] / p[3], p[1] / p[3], p[2] / p[3]));
            }
        }
        return;
    }
    let n = poles.len();
    let mut current = poles.to_vec();
    let mut left = Vec::with_capacity(n);
    let mut right = vec![[0.0; 4]; n];
    left.push(current[0]);
    right[n - 1] = current[n - 1];
    for level in 1..n {
        for i in 0..n - level {
            for k in 0..4 {
                current[i][k] = 0.5 * (current[i][k] + current[i + 1][k]);
            }
        }
        left.push(current[0]);
        right[n - 1 - level] = current[n - 1 - level];
    }
    include_bezier_hull(aabb, &left, depth - 1);
    include_bezier_hull(aabb, &right, depth - 1);
}

/// Include the points where `c + a·cos(t)·u + b·sin(t)·v` is extreme along
/// each world axis, for the parameters that fall inside `[t0, t1]` taken
/// modulo the full turn.
#[allow(clippy::too_many_arguments)]
fn include_conic_extremes(
    aabb: &mut Aabb3,
    u_axis: brepkit_math::vec::Vec3,
    v_axis: brepkit_math::vec::Vec3,
    a: f64,
    b: f64,
    t0: f64,
    t1: f64,
    evaluate: impl Fn(f64) -> Point3,
) {
    let span = t1 - t0;
    let u = [u_axis.x(), u_axis.y(), u_axis.z()];
    let v = [v_axis.x(), v_axis.y(), v_axis.z()];
    for k in 0..3 {
        let (au, bv) = (a * u[k], b * v[k]);
        if au.hypot(bv) <= 0.0 {
            continue;
        }
        let phi = bv.atan2(au);
        for extreme in [phi, phi + std::f64::consts::PI] {
            let rel = (extreme - t0).rem_euclid(std::f64::consts::TAU);
            if rel <= span + 1e-12 {
                aabb_include(aabb, evaluate(t0 + rel));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::boolean::{self, BooleanOp};
    use crate::extrude::extrude;
    use crate::primitives::make_box;
    use crate::transform::transform_solid;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Vec3;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    fn vertex(topo: &mut Topology, x: f64, y: f64) -> brepkit_topology::vertex::VertexId {
        topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7))
    }

    fn arc(
        topo: &mut Topology,
        a: brepkit_topology::vertex::VertexId,
        b: brepkit_topology::vertex::VertexId,
        center: Point3,
        radius: f64,
    ) -> brepkit_topology::edge::EdgeId {
        let circle = Circle3D::new(center, Vec3::new(0.0, 0.0, 1.0), radius).unwrap();
        topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle)))
    }

    fn extrude_profile(topo: &mut Topology, edges: Vec<OrientedEdge>, depth: f64) -> SolidId {
        let wire = Wire::new(edges, true).unwrap();
        let wid = topo.add_wire(wire);
        let fid = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        extrude(topo, fid, Vec3::new(0.0, 0.0, 1.0), depth).unwrap()
    }

    /// 40 x 33 profile whose two bottom corners are r=5 quarter arcs.
    fn u_profile(topo: &mut Topology) -> SolidId {
        let (hw, r, h) = (20.0, 5.0, 33.0);
        let v0 = vertex(topo, -hw, h);
        let v1 = vertex(topo, -hw, r);
        let v2 = vertex(topo, -(hw - r), 0.0);
        let v3 = vertex(topo, hw - r, 0.0);
        let v4 = vertex(topo, hw, r);
        let v5 = vertex(topo, hw, h);
        let edges = [
            topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line)),
            arc(topo, v1, v2, Point3::new(-(hw - r), r, 0.0), r),
            topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line)),
            arc(topo, v3, v4, Point3::new(hw - r, r, 0.0), r),
            topo.add_edge(Edge::new(v4, v5, EdgeCurve::Line)),
            topo.add_edge(Edge::new(v5, v0, EdgeCurve::Line)),
        ];
        extrude_profile(
            topo,
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            8.0,
        )
    }

    #[test]
    fn concave_arc_notch_does_not_inflate_the_box() {
        // 40 x 20 rectangle with a semicircular r=5 bite out of the top edge.
        // The bite's circle reaches y=25 but the solid never leaves y<=20.
        let mut topo = Topology::new();
        let v0 = vertex(&mut topo, -20.0, 0.0);
        let v1 = vertex(&mut topo, 20.0, 0.0);
        let v2 = vertex(&mut topo, 20.0, 20.0);
        let v3 = vertex(&mut topo, 5.0, 20.0);
        let v4 = vertex(&mut topo, -5.0, 20.0);
        let v5 = vertex(&mut topo, -20.0, 20.0);
        // The CCW span from v4 to v3 runs through the bite's bottom (0, 15);
        // the wire traverses it reversed.
        let bite = arc(&mut topo, v4, v3, Point3::new(0.0, 20.0, 0.0), 5.0);
        let edges = vec![
            OrientedEdge::new(topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line)), true),
            OrientedEdge::new(topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line)), true),
            OrientedEdge::new(topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line)), true),
            OrientedEdge::new(bite, false),
            OrientedEdge::new(topo.add_edge(Edge::new(v4, v5, EdgeCurve::Line)), true),
            OrientedEdge::new(topo.add_edge(Edge::new(v5, v0, EdgeCurve::Line)), true),
        ];
        let solid = extrude_profile(&mut topo, edges, 8.0);

        let aabb = solid_bounding_box(&topo, solid).unwrap();
        assert!((aabb.min.x() + 20.0).abs() < 1e-9, "min.x={}", aabb.min.x());
        assert!((aabb.max.x() - 20.0).abs() < 1e-9, "max.x={}", aabb.max.x());
        assert!(aabb.min.y().abs() < 1e-9, "min.y={}", aabb.min.y());
        assert!((aabb.max.y() - 20.0).abs() < 1e-9, "max.y={}", aabb.max.y());
        assert!((aabb.max.z() - 8.0).abs() < 1e-9, "max.z={}", aabb.max.z());
    }

    #[test]
    fn thin_slab_through_arc_corners_has_tight_bounds() {
        // The gridfinity tool measures cutout widths by intersecting with a
        // 0.001 mm slab and reading the bounds. Across the r=5 corner arcs the
        // sliver is 2 * (20 - (5 - sqrt(25 - (5 - y)^2))) wide, not the full
        // circle's 40.
        let mut topo = Topology::new();
        let body = u_profile(&mut topo);
        let (y0, thickness) = (2.0, 0.001);
        let slab = make_box(&mut topo, 160.0, thickness, 40.0).unwrap();
        transform_solid(&mut topo, slab, &Mat4::translation(-80.0, y0, -20.0)).unwrap();

        let sliver = boolean::boolean(&mut topo, BooleanOp::Intersect, body, slab).unwrap();
        let aabb = solid_bounding_box(&topo, sliver).unwrap();

        let half_width_at = |y: f64| 20.0 - (5.0 - (25.0f64 - (5.0 - y).powi(2)).sqrt());
        let expected = half_width_at(y0 + thickness);
        assert!(
            (aabb.max.x() - expected).abs() < 1e-6,
            "max.x={} expected {expected}",
            aabb.max.x()
        );
        assert!(
            (aabb.min.x() + expected).abs() < 1e-6,
            "min.x={} expected -{expected}",
            aabb.min.x()
        );
        assert!((aabb.min.y() - y0).abs() < 1e-9, "min.y={}", aabb.min.y());
        assert!(
            (aabb.max.y() - (y0 + thickness)).abs() < 1e-9,
            "max.y={}",
            aabb.max.y()
        );
        assert!(aabb.min.z().abs() < 1e-9 && (aabb.max.z() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn nurbs_arc_hull_never_under_shoots_and_stays_tight() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        // A rational quadratic quarter circle of radius 10 from 45 to 135
        // degrees: its middle control point sits at (0, 10·√2), well above the
        // arc's true top at y = 10.
        let r = 10.0_f64;
        let (c45, s45) = (
            std::f64::consts::FRAC_PI_4.cos(),
            std::f64::consts::FRAC_PI_4.sin(),
        );
        let curve = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(r * c45, r * s45, 0.0),
                Point3::new(0.0, r * std::f64::consts::SQRT_2, 0.0),
                Point3::new(-r * c45, r * s45, 0.0),
            ],
            vec![1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0],
        )
        .unwrap();
        let seed = Point3::new(r * c45, r * s45, 0.0);
        let mut aabb = Aabb3 {
            min: seed,
            max: seed,
        };
        include_nurbs_edge_hull(&mut aabb, &curve, 0.0, 1.0, Placement(None));
        assert!(
            aabb.max.y() >= r - 1e-9,
            "under-shoots the top: {}",
            aabb.max.y()
        );
        assert!(aabb.max.y() <= r * 1.005, "too loose: {}", aabb.max.y());
        assert!(aabb.min.x() >= -r * c45 - 1e-9 && aabb.max.x() <= r * c45 + 1e-9);

        // Half the span: the split keeps the bound on the kept half only.
        let mut half = Aabb3 {
            min: seed,
            max: seed,
        };
        include_nurbs_edge_hull(&mut half, &curve, 0.0, 0.5, Placement(None));
        assert!(
            half.min.x() >= -1e-6,
            "left half leaked in: {}",
            half.min.x()
        );
        assert!(half.max.y() >= r - 1e-9 && half.max.y() <= r * 1.005);
    }

    #[test]
    fn elliptical_arc_bounds_follow_its_own_span() {
        use brepkit_math::curves::Ellipse3D;
        // Half of a tilted ellipse (a=10, b=4, major axis at 30 degrees) closed
        // by a chord. The ellipse's global extremes on the far half must stay
        // out of the box, and the near half's analytic extremes must be in it.
        let (a, b) = (10.0_f64, 4.0_f64);
        let tilt = std::f64::consts::FRAC_PI_6;
        let u = Vec3::new(tilt.cos(), tilt.sin(), 0.0);
        let v = Vec3::new(-tilt.sin(), tilt.cos(), 0.0);
        let center = Point3::new(3.0, 2.0, 0.0);
        let ellipse = Ellipse3D::with_axes(center, Vec3::new(0.0, 0.0, 1.0), a, b, u, v).unwrap();
        let start = ellipse.evaluate(0.0);
        let end = ellipse.evaluate(std::f64::consts::PI);
        let mut topo = Topology::new();
        let v0 = topo.add_vertex(Vertex::new(start, 1e-7));
        let v1 = topo.add_vertex(Vertex::new(end, 1e-7));
        let arc = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Ellipse(ellipse.clone())));
        let chord = topo.add_edge(Edge::new(v1, v0, EdgeCurve::Line));
        let wire = Wire::new(
            vec![OrientedEdge::new(arc, true), OrientedEdge::new(chord, true)],
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        let fid = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        let aabb = face_set_bounding_box(&topo, &[fid]).unwrap();

        let mut expected = Aabb3 {
            min: start,
            max: start,
        };
        for i in 0..=100_000 {
            let t = std::f64::consts::PI * f64::from(i) / 100_000.0;
            aabb_include(&mut expected, ellipse.evaluate(t));
        }
        for (got, want) in [
            (aabb.min.x(), expected.min.x()),
            (aabb.min.y(), expected.min.y()),
            (aabb.max.x(), expected.max.x()),
            (aabb.max.y(), expected.max.y()),
        ] {
            assert!((got - want).abs() < 1e-6, "got {got} want {want}");
        }
    }

    fn assert_boxes_match(got: Aabb3, want: Aabb3, tol: f64, what: &str) {
        for (g, w) in [
            (got.min.x(), want.min.x()),
            (got.min.y(), want.min.y()),
            (got.min.z(), want.min.z()),
            (got.max.x(), want.max.x()),
            (got.max.y(), want.max.y()),
            (got.max.z(), want.max.z()),
        ] {
            assert!((g - w).abs() < tol, "{what}: got {got:?} want {want:?}");
        }
    }

    #[test]
    fn transformed_box_matches_the_box_of_a_transformed_copy() {
        use crate::copy::copy_solid;
        use crate::primitives::{make_cylinder, make_sphere, make_torus};
        use crate::transform::transform_solid;

        let placements = [
            Mat4::translation(3.0, -2.0, 7.5),
            Mat4::translation(1.0, 2.0, 3.0) * Mat4::rotation_z(0.7),
            Mat4::rotation_x(0.9) * Mat4::rotation_y(-0.4) * Mat4::rotation_z(1.3),
            Mat4::translation(-4.0, 0.5, 2.0) * Mat4::rotation_y(2.1) * Mat4::scale(1.5, 1.5, 1.5),
        ];
        let mut topo = Topology::new();
        let shapes = [
            ("box", make_box(&mut topo, 2.0, 3.0, 4.0).unwrap()),
            ("cylinder", make_cylinder(&mut topo, 1.5, 4.0).unwrap()),
            ("sphere", make_sphere(&mut topo, 2.0, 16).unwrap()),
            ("torus", make_torus(&mut topo, 6.0, 1.5, 16).unwrap()),
            ("u-profile", u_profile(&mut topo)),
        ];
        for (name, solid) in shapes {
            for (i, m) in placements.iter().enumerate() {
                let placed = solid_bounding_box_transformed(&topo, solid, m).unwrap();
                let copy = copy_solid(&mut topo, solid).unwrap();
                transform_solid(&mut topo, copy, m).unwrap();
                let moved = solid_bounding_box(&topo, copy).unwrap();
                assert_boxes_match(placed, moved, 1e-9, &format!("{name} placement {i}"));
            }
        }
    }

    #[test]
    fn cylinder_spun_about_its_own_axis_keeps_its_box() {
        use crate::primitives::make_cylinder;

        let mut topo = Topology::new();
        let cyl = make_cylinder(&mut topo, 1.0, 4.0).unwrap();
        let spun = Mat4::rotation_z(std::f64::consts::FRAC_PI_4);
        let got = solid_bounding_box_transformed(&topo, cyl, &spun).unwrap();
        let want = Aabb3 {
            min: Point3::new(-1.0, -1.0, 0.0),
            max: Point3::new(1.0, 1.0, 4.0),
        };
        assert_boxes_match(got, want, 1e-9, "spun cylinder");
    }

    #[test]
    fn tilted_torus_box_is_its_support_not_its_rotated_local_box() {
        use crate::primitives::make_torus;

        let mut topo = Topology::new();
        let torus = make_torus(&mut topo, 10.0, 3.0, 16).unwrap();
        let tilt = Mat4::rotation_x(std::f64::consts::FRAC_PI_4);
        let got = solid_bounding_box_transformed(&topo, torus, &tilt).unwrap();
        let h = 10.0 * std::f64::consts::FRAC_1_SQRT_2 + 3.0;
        let want = Aabb3 {
            min: Point3::new(-13.0, -h, -h),
            max: Point3::new(13.0, h, h),
        };
        assert_boxes_match(got, want, 1e-9, "tilted torus");
    }
}
