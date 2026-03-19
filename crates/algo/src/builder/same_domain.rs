//! Same-domain face detection and merging.
//!
//! When two faces from opposing solids share the same underlying surface
//! (coplanar planes, coincident cylinders, etc.), they are "same-domain"
//! faces. These need special handling:
//!
//! - **CoplanarSame**: normals point the same way — kept in fuse/intersect.
//! - **CoplanarOpposite**: normals point opposite — kept in cut (for B faces).

use std::collections::HashMap;
use std::hash::BuildHasher;

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::{FaceClass, SubFace};
use crate::ds::{GfaArena, Rank};

/// Detect same-domain face pairs and update their classification.
///
/// Scans all sub-faces looking for pairs where both faces lie on the
/// same geometric surface. Updates classification to `CoplanarSame`
/// or `CoplanarOpposite` based on normal alignment.
#[allow(clippy::too_many_lines)]
pub fn detect_same_domain<S: BuildHasher>(
    topo: &Topology,
    _arena: &GfaArena,
    sub_faces: &mut [SubFace],
    _face_ranks: &HashMap<FaceId, Rank, S>,
    tol: Tolerance,
) {
    let n = sub_faces.len();
    let mut same_domain_count = 0u32;

    // Collect face surfaces to avoid repeated lookups.
    let surfaces: Vec<Option<&FaceSurface>> = sub_faces
        .iter()
        .map(|sf| {
            topo.face(sf.face_id)
                .ok()
                .map(brepkit_topology::face::Face::surface)
        })
        .collect();

    // Check every pair of sub-faces from opposing solids.
    for i in 0..n {
        if sub_faces[i].classification != FaceClass::Unknown {
            continue;
        }
        let Some(surf_i) = surfaces[i] else {
            continue;
        };
        let rank_i = sub_faces[i].rank;

        for j in (i + 1)..n {
            if sub_faces[j].classification != FaceClass::Unknown {
                continue;
            }
            // Only compare faces from opposing solids
            if sub_faces[j].rank == rank_i {
                continue;
            }
            let Some(surf_j) = surfaces[j] else {
                continue;
            };

            // Check overlap between face boundary AABBs
            let overlaps =
                face_bboxes_overlap(topo, sub_faces[i].face_id, sub_faces[j].face_id, tol);
            if !overlaps {
                continue;
            }

            if let Some(same_dir) = surfaces_same_domain(surf_i, surf_j, tol) {
                if same_dir {
                    sub_faces[i].classification = FaceClass::CoplanarSame;
                    sub_faces[j].classification = FaceClass::CoplanarSame;
                } else {
                    sub_faces[i].classification = FaceClass::CoplanarOpposite;
                    sub_faces[j].classification = FaceClass::CoplanarOpposite;
                }
                same_domain_count += 1;
                break; // Each face can only be same-domain with one opposing face
            }
        }
    }

    log::debug!("detect_same_domain: {same_domain_count} same-domain pairs found");
}

/// Check if two face bounding boxes overlap (with tolerance expansion).
fn face_bboxes_overlap(topo: &Topology, a: FaceId, b: FaceId, tol: Tolerance) -> bool {
    let bbox = |fid: FaceId| -> Option<brepkit_math::aabb::Aabb3> {
        let face = topo.face(fid).ok()?;
        let wire = topo.wire(face.outer_wire()).ok()?;
        let mut points = Vec::new();
        for oe in wire.edges() {
            let e = topo.edge(oe.edge()).ok()?;
            let sp = topo.vertex(e.start()).ok()?.point();
            let ep = topo.vertex(e.end()).ok()?.point();
            points.push(sp);
            points.push(ep);
            // Curved edges can bulge beyond their endpoints
            if !matches!(e.curve(), brepkit_topology::edge::EdgeCurve::Line) {
                let (t0, t1) = e.curve().domain_with_endpoints(sp, ep);
                let t_mid = 0.5_f64.mul_add(t1 - t0, t0);
                points.push(e.curve().evaluate_with_endpoints(t_mid, sp, ep));
            }
        }
        brepkit_math::aabb::Aabb3::try_from_points(points)
    };
    let Some(ba) = bbox(a) else { return false };
    let Some(bb) = bbox(b) else { return false };
    ba.expanded(tol.linear).intersects(bb.expanded(tol.linear))
}

/// Check if two surfaces represent the same geometric domain.
///
/// Returns `Some(true)` for same-direction normals (CoplanarSame),
/// `Some(false)` for opposite normals (CoplanarOpposite), or
/// `None` if not the same domain.
fn surfaces_same_domain(a: &FaceSurface, b: &FaceSurface, tol: Tolerance) -> Option<bool> {
    match (a, b) {
        (FaceSurface::Plane { normal: na, d: da }, FaceSurface::Plane { normal: nb, d: db }) => {
            let dot = na.dot(*nb);
            if dot > 1.0 - tol.angular {
                // Same direction — check distance
                if (da - db).abs() < tol.linear {
                    return Some(true);
                }
            } else if dot < -1.0 + tol.angular {
                // Opposite direction — check distance
                if (da + db).abs() < tol.linear {
                    return Some(false);
                }
            }
            None
        }
        (FaceSurface::Cylinder(ca), FaceSurface::Cylinder(cb)) => {
            // Same cylinder: same origin, same axis, same radius
            if (ca.radius() - cb.radius()).abs() > tol.linear {
                return None;
            }
            let axis_dot = ca.axis().dot(cb.axis());
            if axis_dot.abs() < 1.0 - tol.angular {
                return None;
            }
            // Check if origins lie on the same axis line
            let diff = cb.origin() - ca.origin();
            let along_axis = diff.dot(ca.axis());
            let perp_dist = (diff - ca.axis() * along_axis).length();
            if perp_dist > tol.linear {
                return None;
            }
            Some(axis_dot > 0.0)
        }
        (FaceSurface::Sphere(sa), FaceSurface::Sphere(sb)) => {
            if (sa.radius() - sb.radius()).abs() > tol.linear {
                return None;
            }
            let dist = (sa.center() - sb.center()).length();
            if dist > tol.linear {
                return None;
            }
            // Spheres with same center/radius are always same-domain-same
            Some(true)
        }
        (FaceSurface::Cone(ca), FaceSurface::Cone(cb)) => {
            if (ca.half_angle() - cb.half_angle()).abs() > tol.angular {
                return None;
            }
            let axis_dot = ca.axis().dot(cb.axis());
            if axis_dot.abs() < 1.0 - tol.angular {
                return None;
            }
            let dist = (ca.apex() - cb.apex()).length();
            if dist > tol.linear {
                return None;
            }
            Some(axis_dot > 0.0)
        }
        _ => None,
    }
}
