//! Boolean preflight diagnostics.
//!
//! Lightweight checks that callers can run BEFORE invoking a boolean
//! operation, to detect input configurations that are likely to
//! exercise the same-domain detector or known boolean robustness
//! gaps. The diagnostic does NOT run the full GFA pipeline — it only
//! compares the underlying face surfaces and AABBs.

use brepkit_math::aabb::Aabb3;
use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::builder::same_domain::surfaces_same_domain;

/// One detected pair of same-domain faces between two solids.
///
/// "Same-domain" here is the surface-level relationship: the two
/// faces lie on the same underlying analytic surface (e.g., the same
/// plane, or two cylinders with matching axis and radius). It does
/// NOT guarantee that the faces overlap geometrically — see
/// `aabb_overlap` for a coarse geometric filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CoincidentFacePair {
    /// Face from solid A.
    pub face_a: FaceId,
    /// Face from solid B.
    pub face_b: FaceId,
    /// `true` if the surface normals point the same direction at
    /// corresponding parametric points; `false` if they point opposite.
    pub same_orientation: bool,
    /// `true` if the face AABBs overlap (geometric overlap candidate).
    /// Pairs with `aabb_overlap = false` are same-domain on the surface
    /// but geometrically disjoint (will not interact in a boolean).
    pub aabb_overlap: bool,
}

/// Detect surface-level same-domain face pairs between two solids.
///
/// Walks each face of `solid_a` against each face of `solid_b` and
/// reports pairs whose underlying surfaces are same-domain (per the
/// rules in `same_domain.rs`). For each pair we also compute the
/// AABB overlap so callers can filter pairs that won't actually
/// interact during a boolean.
///
/// This is O(|A.faces| × |B.faces|) — fine for typical CAD shapes
/// (tens to a few hundred faces) but should not be called in tight
/// rendering loops on large assemblies.
///
/// # Errors
///
/// Returns an error if `solid_a` or `solid_b` cannot be found in
/// `topo`, or if face data is missing.
pub fn detect_coincident_faces(
    topo: &Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
) -> Result<Vec<CoincidentFacePair>, crate::error::AlgoError> {
    use brepkit_topology::explorer;

    let faces_a = explorer::solid_faces(topo, solid_a)?;
    let faces_b = explorer::solid_faces(topo, solid_b)?;

    let mut pairs = Vec::new();
    for &fa in &faces_a {
        let face_a = topo.face(fa)?;
        let aabb_a = face_aabb(topo, fa)?;
        for &fb in &faces_b {
            let face_b = topo.face(fb)?;
            if let Some(same_orientation) =
                surfaces_same_domain(face_a.surface(), face_b.surface(), tol)
            {
                let aabb_b = face_aabb(topo, fb)?;
                pairs.push(CoincidentFacePair {
                    face_a: fa,
                    face_b: fb,
                    same_orientation,
                    aabb_overlap: aabbs_overlap(&aabb_a, &aabb_b, tol.linear),
                });
            }
        }
    }
    Ok(pairs)
}

/// Number of interior parametric samples used per edge when computing
/// the face AABB. Endpoints are always included; this controls how
/// many midpoints are sampled along curved edges (arcs, NURBS, etc.)
/// to capture curve bulge that would otherwise be missed by a chord-only
/// AABB.
///
/// **Why 11:** for a full-circle edge with domain `[0, 2π]`, 11 interior
/// samples land at fractions `{1/12, 2/12, ..., 11/12}` — i.e. 30° steps.
/// In the world frame, the worst-case angular offset from any cardinal
/// extremum is half a step (15°), so each AABB axis recovers at least
/// `cos(15°) ≈ 0.966` of the true extent — a ~3% per-axis underestimate
/// in the worst case, vs. ~13% with 5 samples. For axis-aligned circles
/// the cardinals are hit exactly. This is intentionally a coarse but
/// O(1) approximation; callers needing exact bounds for large arcs
/// should use a curve-specific extrema routine on `Circle3D` / `Arc3D`.
const EDGE_INTERIOR_SAMPLES: usize = 11;

/// Compute a curve-aware face AABB.
///
/// For each boundary edge we include the two endpoint vertices AND
/// `EDGE_INTERIOR_SAMPLES` interior samples along the edge curve, so
/// that bulge from arcs, full circles, and NURBS curves contributes to
/// the bounding box. A vertex-only AABB would underestimate non-planar
/// face extents (a full-circle edge whose endpoints coincide collapses
/// to a single point), causing `aabb_overlap` to silently miss real
/// overlaps for curved coincident faces.
fn face_aabb(topo: &Topology, fid: FaceId) -> Result<Aabb3, crate::error::AlgoError> {
    use brepkit_math::vec::Point3;
    let face = topo.face(fid)?;
    let outer = topo.wire(face.outer_wire())?;

    let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut any = false;

    let mut include = |p: Point3, any: &mut bool| {
        min = Point3::new(min.x().min(p.x()), min.y().min(p.y()), min.z().min(p.z()));
        max = Point3::new(max.x().max(p.x()), max.y().max(p.y()), max.z().max(p.z()));
        *any = true;
    };

    for oe in outer.edges() {
        let edge = topo.edge(oe.edge())?;
        let start = topo.vertex(edge.start())?.point();
        let end = topo.vertex(edge.end())?.point();
        include(start, &mut any);
        include(end, &mut any);

        let curve = edge.curve();
        let (t0, t1) = curve.domain_with_endpoints(start, end);
        for i in 1..=EDGE_INTERIOR_SAMPLES {
            #[allow(clippy::cast_precision_loss)]
            let frac = i as f64 / (EDGE_INTERIOR_SAMPLES as f64 + 1.0);
            let t = t0 + (t1 - t0) * frac;
            let p = curve.evaluate_with_endpoints(t, start, end);
            include(p, &mut any);
        }
    }
    if !any {
        return Err(crate::error::AlgoError::AssemblyFailed(format!(
            "face {fid:?} has no boundary vertices",
        )));
    }
    Ok(Aabb3 { min, max })
}

fn aabbs_overlap(a: &Aabb3, b: &Aabb3, slack: f64) -> bool {
    a.min.x() <= b.max.x() + slack
        && a.max.x() + slack >= b.min.x()
        && a.min.y() <= b.max.y() + slack
        && a.max.y() + slack >= b.min.y()
        && a.min.z() <= b.max.z() + slack
        && a.max.z() + slack >= b.min.z()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use brepkit_math::vec::Point3;
    use brepkit_math::vec::Vec3;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    fn make_box(topo: &mut Topology, min: [f64; 3], max: [f64; 3]) -> SolidId {
        let [x0, y0, z0] = min;
        let [x1, y1, z1] = max;
        let v = [
            topo.add_vertex(Vertex::new(Point3::new(x0, y0, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y0, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y1, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x0, y1, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x0, y0, z1), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y0, z1), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y1, z1), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x0, y1, z1), 1e-7)),
        ];
        let mut edge = |a: usize, b: usize| topo.add_edge(Edge::new(v[a], v[b], EdgeCurve::Line));
        let e01 = edge(0, 1);
        let e12 = edge(1, 2);
        let e23 = edge(2, 3);
        let e30 = edge(3, 0);
        let e45 = edge(4, 5);
        let e56 = edge(5, 6);
        let e67 = edge(6, 7);
        let e74 = edge(7, 4);
        let e04 = edge(0, 4);
        let e15 = edge(1, 5);
        let e26 = edge(2, 6);
        let e37 = edge(3, 7);
        let fwd = |eid| OrientedEdge::new(eid, true);
        let w_bot =
            topo.add_wire(Wire::new(vec![fwd(e01), fwd(e12), fwd(e23), fwd(e30)], true).unwrap());
        let w_top =
            topo.add_wire(Wire::new(vec![fwd(e45), fwd(e56), fwd(e67), fwd(e74)], true).unwrap());
        let w_front =
            topo.add_wire(Wire::new(vec![fwd(e01), fwd(e15), fwd(e45), fwd(e04)], true).unwrap());
        let w_back =
            topo.add_wire(Wire::new(vec![fwd(e23), fwd(e37), fwd(e67), fwd(e26)], true).unwrap());
        let w_left =
            topo.add_wire(Wire::new(vec![fwd(e30), fwd(e04), fwd(e74), fwd(e37)], true).unwrap());
        let w_right =
            topo.add_wire(Wire::new(vec![fwd(e12), fwd(e26), fwd(e56), fwd(e15)], true).unwrap());
        let f_bot = topo.add_face(Face::new(
            w_bot,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: -z0,
            },
        ));
        let f_top = topo.add_face(Face::new(
            w_top,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: z1,
            },
        ));
        let f_front = topo.add_face(Face::new(
            w_front,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, -1.0, 0.0),
                d: -y0,
            },
        ));
        let f_back = topo.add_face(Face::new(
            w_back,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: y1,
            },
        ));
        let f_left = topo.add_face(Face::new(
            w_left,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(-1.0, 0.0, 0.0),
                d: -x0,
            },
        ));
        let f_right = topo.add_face(Face::new(
            w_right,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(1.0, 0.0, 0.0),
                d: x1,
            },
        ));
        let shell = topo
            .add_shell(Shell::new(vec![f_bot, f_top, f_front, f_back, f_left, f_right]).unwrap());
        topo.add_solid(Solid::new(shell, vec![]))
    }

    #[test]
    fn face_stack_detects_expected_pairs() {
        // Two unit cubes face-stacked on z=1.
        // Same-domain (surface-level) pairs that ALSO have AABB overlap:
        //   1× (z=1) cap pair, opposite normals
        //   4× side-plane pairs (front/back/left/right), same normals
        // Total = 5.
        let mut topo = Topology::default();
        let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = make_box(&mut topo, [0.0, 0.0, 1.0], [1.0, 1.0, 2.0]);
        let pairs = detect_coincident_faces(&topo, a, b, Tolerance::default()).unwrap();
        let overlapping_count = pairs.iter().filter(|p| p.aabb_overlap).count();
        assert_eq!(
            overlapping_count, 5,
            "should detect 5 SD pairs with AABB overlap, got {overlapping_count}",
        );
        let opposite_count = pairs
            .iter()
            .filter(|p| p.aabb_overlap && !p.same_orientation)
            .count();
        assert_eq!(
            opposite_count, 1,
            "exactly the cap pair has opposite normals"
        );
    }

    #[test]
    fn curve_edge_face_aabb_includes_bulge() {
        // Regression for the chord-only AABB bug: a face bounded by a
        // single full-circle edge has start == end, so a vertex-only
        // AABB collapses to a point. The curve-aware AABB must instead
        // span the circle's diameter in x and y.
        use brepkit_math::curves::Circle3D;
        let mut topo = Topology::default();
        // Single vertex serving as start/end of the full-circle edge.
        let v0 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let edge = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Circle(circle)));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let bbox = face_aabb(&topo, face).unwrap();
        // True AABB of the unit circle in z=0: x∈[-1,1], y∈[-1,1].
        // With 11 interior samples at 30° steps the worst-case axis
        // recovery is cos(15°) ≈ 0.966 — Circle3D's auto-generated
        // local frame is not guaranteed to align with the world axes,
        // so we assert ≥0.85 on each axis (well above 0 and well above
        // the chord-only AABB which would collapse to a point).
        let dx = bbox.max.x() - bbox.min.x();
        let dy = bbox.max.y() - bbox.min.y();
        assert!(
            dx > 1.7,
            "circle edge AABB x-span = {dx}, expected close to 2 (chord-only would be 0)",
        );
        assert!(
            dy > 1.7,
            "circle edge AABB y-span = {dy}, expected close to 2 (chord-only would be 0)",
        );
    }

    #[test]
    fn disjoint_boxes_no_coincident_overlap() {
        // Boxes far apart — surfaces may be same-domain (parallel planes
        // at the same offset) by coincidence, but AABBs do not overlap.
        let mut topo = Topology::default();
        let a = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = make_box(&mut topo, [10.0, 10.0, 10.0], [11.0, 11.0, 11.0]);
        let pairs = detect_coincident_faces(&topo, a, b, Tolerance::default()).unwrap();
        let overlapping = pairs.iter().any(|p| p.aabb_overlap);
        assert!(
            !overlapping,
            "disjoint boxes should have no overlapping coincident pairs"
        );
    }
}
