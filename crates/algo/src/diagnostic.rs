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
#[derive(Debug, Clone, Copy, PartialEq)]
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

    let faces_a = explorer::solid_faces(topo, solid_a)
        .map_err(|_| crate::error::AlgoError::AssemblyFailed("solid_a not found".into()))?;
    let faces_b = explorer::solid_faces(topo, solid_b)
        .map_err(|_| crate::error::AlgoError::AssemblyFailed("solid_b not found".into()))?;

    let mut pairs = Vec::new();
    for &fa in &faces_a {
        let face_a = topo
            .face(fa)
            .map_err(|_| crate::error::AlgoError::AssemblyFailed("face_a not found".into()))?;
        let aabb_a = face_aabb(topo, fa)?;
        for &fb in &faces_b {
            let face_b = topo
                .face(fb)
                .map_err(|_| crate::error::AlgoError::AssemblyFailed("face_b not found".into()))?;
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

/// Compute a face AABB from its boundary vertex positions.
fn face_aabb(topo: &Topology, fid: FaceId) -> Result<Aabb3, crate::error::AlgoError> {
    use brepkit_math::vec::Point3;
    let face = topo
        .face(fid)
        .map_err(|_| crate::error::AlgoError::AssemblyFailed("face not found".into()))?;
    let outer = topo
        .wire(face.outer_wire())
        .map_err(|_| crate::error::AlgoError::AssemblyFailed("wire not found".into()))?;

    let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut any = false;

    for oe in outer.edges() {
        let edge = topo
            .edge(oe.edge())
            .map_err(|_| crate::error::AlgoError::AssemblyFailed("edge not found".into()))?;
        for vid in [edge.start(), edge.end()] {
            let v = topo
                .vertex(vid)
                .map_err(|_| crate::error::AlgoError::AssemblyFailed("vertex not found".into()))?;
            let p = v.point();
            min = Point3::new(min.x().min(p.x()), min.y().min(p.y()), min.z().min(p.z()));
            max = Point3::new(max.x().max(p.x()), max.y().max(p.y()), max.z().max(p.z()));
            any = true;
        }
    }
    if !any {
        // Empty wire — give a degenerate AABB at origin.
        min = Point3::new(0.0, 0.0, 0.0);
        max = Point3::new(0.0, 0.0, 0.0);
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
