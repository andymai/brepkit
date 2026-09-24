//! Draft angle operation for injection molding applications.
//!
//! Tapers selected planar faces of a solid relative to a pull direction by
//! turning each face's plane about its neutral line and re-solving the
//! vertices it moves, so the solid keeps its topology.

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

use crate::dot_normal_point;

/// Apply a draft angle to selected planar faces of a solid.
///
/// Each drafted face turns about its neutral line (where it crosses the
/// neutral plane through `neutral_point` normal to `pull_direction`) until
/// its outward normal `n` satisfies `n · pull = sin(angle)`: with a positive
/// angle the face leans in toward the pull direction above the neutral
/// plane and out below it, so the part releases along the pull. Every
/// vertex of a drafted face moves to the meeting point of the planes around
/// it; the rest of the solid, its faces and its topology are unchanged. The
/// input solid is left as it was and the drafted copy returned.
///
/// # Errors
///
/// Returns an error if:
/// - `angle_radians` is zero, not below a right angle in magnitude, or not
///   finite, or `pull_direction` is zero-length
/// - a drafted face is not a planar face of the solid, or lies parallel to
///   the neutral plane (it has no direction to lean in)
/// - a vertex of a drafted face meets a non-planar face, or more planes
///   than three that no longer share a point (the draft would split it)
/// - an edge of a moved vertex would turn back on itself (the angle is too
///   steep for the face)
#[allow(clippy::too_many_lines)]
pub fn draft(
    topo: &mut Topology,
    solid: SolidId,
    draft_faces: &[FaceId],
    pull_direction: Vec3,
    neutral_point: Point3,
    angle_radians: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();
    let invalid = |reason: &str| crate::OperationsError::InvalidInput {
        reason: reason.into(),
    };

    if !angle_radians.is_finite()
        || angle_radians.abs() <= tol.angular
        || angle_radians.abs() >= std::f64::consts::FRAC_PI_2
    {
        return Err(invalid(
            "draft angle must be non-zero and smaller than a right angle",
        ));
    }
    let pull = pull_direction.normalize()?;
    let neutral_d = dot_normal_point(pull, neutral_point);

    let faces = solid_faces(topo, solid)?;
    let draft_set: HashSet<FaceId> = draft_faces.iter().copied().collect();
    if draft_set.iter().any(|f| !faces.contains(f)) {
        return Err(invalid("draft faces must belong to the solid"));
    }

    // Outward unit plane `(n, d)` with `n · p = d` for every planar face;
    // drafted faces get their turned planes.
    let (sin_a, cos_a) = angle_radians.sin_cos();
    let mut planes: HashMap<FaceId, (Vec3, f64)> = HashMap::new();
    for &fid in &faces {
        let face = topo.face(fid)?;
        let FaceSurface::Plane { normal, d } = face.surface() else {
            if draft_set.contains(&fid) {
                return Err(invalid("draft target faces must be planar"));
            }
            continue;
        };
        let len = normal.length();
        let sign = if face.is_reversed() { -1.0 } else { 1.0 };
        let (n, off) = (*normal * (sign / len), d * sign / len);
        if !draft_set.contains(&fid) {
            planes.insert(fid, (n, off));
            continue;
        }
        let across = n - pull * n.dot(pull);
        let Ok(across) = across.normalize() else {
            return Err(invalid(
                "a face parallel to the neutral plane cannot be drafted",
            ));
        };
        // A point of the neutral line: on the face's plane and the neutral
        // plane, in the span of their normals.
        let c = n.dot(pull);
        let det = c.mul_add(-c, 1.0);
        let (a, b) = (
            c.mul_add(-neutral_d, off) / det,
            c.mul_add(-off, neutral_d) / det,
        );
        let on_line = n * a + pull * b;
        let turned = across * cos_a + pull * sin_a;
        planes.insert(fid, (turned, turned.dot(on_line)));
    }

    // Faces around each vertex and each edge.
    let mut around: HashMap<VertexId, Vec<FaceId>> = HashMap::new();
    let mut edge_faces: HashMap<EdgeId, Vec<FaceId>> = HashMap::new();
    for &fid in &faces {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid)?.edges() {
                edge_faces.entry(oe.edge()).or_default().push(fid);
                let edge = topo.edge(oe.edge())?;
                for vid in [edge.start(), edge.end()] {
                    let list = around.entry(vid).or_default();
                    if !list.contains(&fid) {
                        list.push(fid);
                    }
                }
            }
        }
    }

    // Re-solve every vertex of a drafted face as its planes' meeting point.
    let mut moved: HashMap<VertexId, Point3> = HashMap::new();
    for &fid in &draft_set {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid)?.edges() {
                let edge = topo.edge(oe.edge())?;
                for vid in [edge.start(), edge.end()] {
                    if moved.contains_key(&vid) {
                        continue;
                    }
                    let mut vertex_planes = Vec::new();
                    for f in &around[&vid] {
                        let Some(&plane) = planes.get(f) else {
                            return Err(invalid("a drafted face's vertex meets a non-planar face"));
                        };
                        vertex_planes.push(plane);
                    }
                    let old = topo.vertex(vid)?.point();
                    let point =
                        meeting_point(&vertex_planes, old, tol.linear).ok_or_else(|| {
                            invalid("the draft would split a vertex whose planes no longer meet")
                        })?;
                    moved.insert(vid, point);
                }
            }
        }
    }

    // An edge between two moved vertices must keep its direction.
    for &vid in moved.keys() {
        for &fid in &around[&vid] {
            let face = topo.face(fid)?;
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                for oe in topo.wire(wid)?.edges() {
                    let edge = topo.edge(oe.edge())?;
                    if edge.start() != vid && edge.end() != vid {
                        continue;
                    }
                    let at = |v: VertexId| -> Result<(Point3, Point3), crate::OperationsError> {
                        let old = topo.vertex(v)?.point();
                        Ok((old, moved.get(&v).copied().unwrap_or(old)))
                    };
                    let ((s0, s1), (e0, e1)) = (at(edge.start())?, at(edge.end())?);
                    if (e1 - s1).dot(e0 - s0) <= 0.0 {
                        return Err(invalid("the draft angle is too steep for this face"));
                    }
                }
            }
        }
    }

    // Every edge at a moved vertex runs between two of its planes, so it is
    // straight; a curve stored for it (a boolean's straight NURBS) would keep
    // the old span, and becomes a line.
    let mut straighten = HashSet::new();
    for &vid in moved.keys() {
        for (&eid, owners) in &edge_faces {
            let edge = topo.edge(eid)?;
            if edge.start() != vid && edge.end() != vid {
                continue;
            }
            if matches!(edge.curve(), EdgeCurve::Line) {
                continue;
            }
            let distinct = owners.len() == 2
                && owners
                    .iter()
                    .map(|f| planes.get(f))
                    .collect::<Option<Vec<_>>>()
                    .is_some_and(|p| p[0].0.cross(p[1].0).length() > 1e-9);
            if !distinct {
                return Err(invalid("a drafted face's vertex meets a curved edge"));
            }
            straighten.insert(eid);
        }
    }

    // Apply to a copy, matched to the source by traversal order.
    let copy = crate::copy::copy_solid(topo, solid)?;
    let copy_faces = solid_faces(topo, copy)?;
    let mut vertex_map: HashMap<VertexId, VertexId> = HashMap::new();
    let mut edge_map: HashMap<(EdgeId, FaceId), (EdgeId, FaceId)> = HashMap::new();
    let mut face_map: HashMap<FaceId, FaceId> = HashMap::new();
    for (&src, &dst) in faces.iter().zip(&copy_faces) {
        face_map.insert(src, dst);
        let (sf, df) = (topo.face(src)?, topo.face(dst)?);
        let src_wires = std::iter::once(sf.outer_wire()).chain(sf.inner_wires().iter().copied());
        let dst_wires: Vec<_> = std::iter::once(df.outer_wire())
            .chain(df.inner_wires().iter().copied())
            .collect();
        for (sw, dw) in src_wires.zip(dst_wires) {
            let (sw, dw) = (topo.wire(sw)?, topo.wire(dw)?);
            for (se, de) in sw.edges().iter().zip(dw.edges()) {
                edge_map.insert((se.edge(), src), (de.edge(), dst));
                let (se, de) = (topo.edge(se.edge())?, topo.edge(de.edge())?);
                vertex_map.insert(se.start(), de.start());
                vertex_map.insert(se.end(), de.end());
            }
        }
    }
    for (vid, point) in &moved {
        let target = vertex_map
            .get(vid)
            .copied()
            .ok_or_else(|| invalid("draft lost track of a vertex"))?;
        topo.vertex_mut(target)?.set_point(*point);
    }
    let mut stale = HashSet::new();
    for &fid in &draft_set {
        let dst = face_map[&fid];
        let (n, off) = planes[&fid];
        let face = topo.face_mut(dst)?;
        let sign = if face.is_reversed() { -1.0 } else { 1.0 };
        face.set_surface(FaceSurface::Plane {
            normal: n * sign,
            d: off * sign,
        });
        stale.insert(dst);
    }
    for &vid in moved.keys() {
        for f in &around[&vid] {
            stale.insert(face_map[f]);
        }
    }
    for (&(src_edge, src_face), &(dst_edge, dst_face)) in &edge_map {
        if straighten.contains(&src_edge) {
            topo.edge_mut(dst_edge)?.set_curve(EdgeCurve::Line);
        }
        // `copy_solid` carries no pcurves; untouched faces keep theirs.
        if !stale.contains(&dst_face)
            && let Some(pcurve) = topo.pcurves().get(src_edge, src_face).cloned()
        {
            topo.pcurves_mut().set(dst_edge, dst_face, pcurve);
        }
    }
    Ok(copy)
}

/// The point where planes `(n, d)` (`n · p = d`, `n` unit) meet near `near`:
/// the best-conditioned triple fixes it, and every other plane must pass
/// within `linear` of it. The solve runs about `near`, so a solid's distance
/// from the origin never multiplies into it; only the offsets' own storage
/// precision at that distance widens the bound.
fn meeting_point(planes: &[(Vec3, f64)], near: Point3, linear: f64) -> Option<Point3> {
    let local: Vec<(Vec3, f64)> = planes
        .iter()
        .map(|&(n, d)| (n, d - dot_normal_point(n, near)))
        .collect();
    let mut best: Option<(f64, Vec3)> = None;
    for i in 0..local.len() {
        for j in i + 1..local.len() {
            for k in j + 1..local.len() {
                let ((n1, d1), (n2, d2), (n3, d3)) = (local[i], local[j], local[k]);
                let det = n1.dot(n2.cross(n3));
                if best.is_some_and(|(b, _)| det.abs() <= b) || det.abs() < 1e-14 {
                    continue;
                }
                let v = (n2.cross(n3) * d1 + n3.cross(n1) * d2 + n1.cross(n2) * d3) * (1.0 / det);
                best = Some((det.abs(), v));
            }
        }
    }
    let (_, v) = best?;
    let reach = local
        .iter()
        .map(|&(_, d)| d.abs())
        .fold(v.length(), f64::max);
    let far = near.x().abs().max(near.y().abs()).max(near.z().abs());
    let miss = 64.0f64.mul_add(f64::EPSILON * (far + reach), 10.0 * linear);
    local
        .iter()
        .all(|&(n, d)| (n.dot(v) - d).abs() <= miss)
        .then(|| near + v)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    use super::*;

    /// Helper: find faces whose normal is approximately equal to `target`.
    fn find_faces(topo: &Topology, solid: SolidId, target: Vec3) -> Vec<FaceId> {
        let tol = Tolerance::loose();
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        sh.faces()
            .iter()
            .filter(|&&fid| {
                let f = topo.face(fid).unwrap();
                if let FaceSurface::Plane { normal, .. } = f.surface() {
                    tol.approx_eq(normal.x(), target.x())
                        && tol.approx_eq(normal.y(), target.y())
                        && tol.approx_eq(normal.z(), target.z())
                } else {
                    false
                }
            })
            .copied()
            .collect()
    }

    /// Three coordinate planes and a slanted fourth through the point `at`,
    /// the fourth shifted off it by `gap`.
    fn corner(at: Point3, gap: f64) -> Vec<(Vec3, f64)> {
        let slant = Vec3::new(1.0, 1.0, 1.0).normalize().unwrap();
        [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            slant,
        ]
        .into_iter()
        .map(|n| (n, dot_normal_point(n, at)))
        .enumerate()
        .map(|(i, (n, d))| (n, if i == 3 { d + gap } else { d }))
        .collect()
    }

    #[test]
    fn corner_planes_meet_wherever_the_corner_is() {
        for at in [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0e6, -2.0e6, 3.0e6),
            Point3::new(1.0e12, -2.0e12, 3.0e12),
        ] {
            let point =
                meeting_point(&corner(at, 0.0), at + Vec3::new(0.5, 0.5, 0.5), 1e-7).unwrap();
            assert!(
                (point - at).length()
                    <= 1e-3 * (at - Point3::new(0.0, 0.0, 0.0)).length().max(1e-6)
            );
        }
    }

    #[test]
    fn a_plane_off_the_corner_is_rejected_near_and_far() {
        for at in [Point3::new(0.0, 0.0, 0.0), Point3::new(1.0e9, 0.0, 0.0)] {
            assert!(
                meeting_point(&corner(at, 1e-3), at, 1e-7).is_none(),
                "corner at {at:?} accepted a plane 1e-3 off"
            );
        }
    }

    #[test]
    fn draft_single_face() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let right_faces = find_faces(&topo, cube, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(right_faces.len(), 1);

        let result = draft(
            &mut topo,
            cube,
            &right_faces,
            Vec3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            5.0_f64.to_radians(),
        )
        .unwrap();

        let s = topo.solid(result).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(
            sh.faces().len(),
            6,
            "drafted solid should still have 6 faces"
        );

        // Volume should decrease slightly (draft tapers the face inward).
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            vol > 0.5,
            "drafted solid should have significant volume, got {vol}"
        );
    }

    #[test]
    fn draft_preserves_non_draft_faces() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let right_faces = find_faces(&topo, cube, Vec3::new(1.0, 0.0, 0.0));
        let result = draft(
            &mut topo,
            cube,
            &right_faces,
            Vec3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            5.0_f64.to_radians(),
        )
        .unwrap();

        // The top and bottom faces should still be planar with ±Z normals.
        let top = find_faces(&topo, result, Vec3::new(0.0, 0.0, 1.0));
        let bottom = find_faces(&topo, result, Vec3::new(0.0, 0.0, -1.0));
        assert_eq!(top.len(), 1, "should still have top face");
        assert_eq!(bottom.len(), 1, "should still have bottom face");
    }

    fn assert_drafted(topo: &Topology, solid: SolidId, truth: f64, what: &str) {
        let report = crate::validate::validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        let mesh = crate::tessellate::tessellate_solid(topo, solid, 0.01).unwrap();
        assert_eq!(
            crate::tessellate::boundary_edge_count(&mesh),
            0,
            "{what}: open mesh"
        );
        let volume = crate::measure::solid_volume(topo, solid, 0.001).unwrap();
        assert!(
            (volume - truth).abs() < 1e-9 * truth,
            "{what}: volume {volume}, expected {truth}"
        );
    }

    /// One side of a unit cube leans in by `tan(a)` at the top: the lost
    /// wedge is `tan(a) / 2`, and the face's normal gains `sin(a)` along
    /// the pull.
    #[test]
    fn drafted_side_is_an_exact_wedge() {
        let a = 5.0_f64.to_radians();
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let right = find_faces(&topo, cube, Vec3::new(1.0, 0.0, 0.0));
        let z = Vec3::new(0.0, 0.0, 1.0);
        let result = draft(&mut topo, cube, &right, z, Point3::new(0.0, 0.0, 0.0), a).unwrap();
        assert_drafted(&topo, result, 1.0 - a.tan() / 2.0, "one side");
        let turned = solid_faces(&topo, result)
            .unwrap()
            .into_iter()
            .filter_map(|f| match topo.face(f).unwrap().surface() {
                FaceSurface::Plane { normal, .. } if normal.x() > 0.5 => Some(*normal),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(turned.len(), 1);
        assert!((turned[0].dot(z) - a.sin()).abs() < 1e-12);

        // About a neutral plane at mid height the wedges above and below
        // cancel.
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let right = find_faces(&topo, cube, Vec3::new(1.0, 0.0, 0.0));
        let result = draft(&mut topo, cube, &right, z, Point3::new(0.0, 0.0, 0.5), a).unwrap();
        assert_drafted(&topo, result, 1.0, "mid-height neutral");
    }

    /// All four sides of a box drafted from its base, as built and mirrored:
    /// each section at height z is `(w - 2 z t) x (d - 2 z t)`.
    #[test]
    fn drafted_box_is_an_exact_frustum() {
        let t = 3.0_f64.to_radians().tan();
        let truth = 24.0 - 28.0 * t + 32.0 / 3.0 * t * t;
        for mirrored in [false, true] {
            let mut topo = Topology::new();
            let block = crate::primitives::make_box(&mut topo, 4.0, 3.0, 2.0).unwrap();
            if mirrored {
                crate::transform::transform_solid(
                    &mut topo,
                    block,
                    &brepkit_math::mat::Mat4::scale(-1.0, 1.0, 1.0),
                )
                .unwrap();
            }
            let sides: Vec<FaceId> = solid_faces(&topo, block)
                .unwrap()
                .into_iter()
                .filter(|&f| match topo.face(f).unwrap().surface() {
                    FaceSurface::Plane { normal, .. } => normal.z().abs() < 0.5,
                    _ => false,
                })
                .collect();
            assert_eq!(sides.len(), 4);
            let result = draft(
                &mut topo,
                block,
                &sides,
                Vec3::new(0.0, 0.0, 1.0),
                Point3::new(0.0, 0.0, 0.0),
                3.0_f64.to_radians(),
            )
            .unwrap();
            assert_drafted(&topo, result, truth, &format!("mirrored={mirrored}"));
        }
    }

    #[test]
    fn face_parallel_to_the_neutral_plane_cannot_be_drafted() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let top = find_faces(&topo, cube, Vec3::new(0.0, 0.0, 1.0));
        assert!(
            draft(
                &mut topo,
                cube,
                &top,
                Vec3::new(0.0, 0.0, 1.0),
                Point3::new(0.0, 0.0, 0.0),
                5.0_f64.to_radians(),
            )
            .is_err()
        );
    }

    /// A drafted face whose corner meets a rounded edge would need that
    /// cylinder re-intersected; the draft declines instead of bending it.
    #[test]
    fn draft_next_to_a_curved_face_errors() {
        let mut topo = Topology::new();
        let block = crate::primitives::make_box(&mut topo, 4.0, 3.0, 2.0).unwrap();
        let faces = solid_faces(&topo, block).unwrap();
        let right = faces
            .iter()
            .copied()
            .find(|&f| {
                matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal, .. } if normal.x() > 0.5)
            })
            .unwrap();
        let vertical = topo
            .wire(topo.face(right).unwrap().outer_wire())
            .unwrap()
            .edges()
            .iter()
            .map(brepkit_topology::wire::OrientedEdge::edge)
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                let (a, b) = (
                    topo.vertex(edge.start()).unwrap().point(),
                    topo.vertex(edge.end()).unwrap().point(),
                );
                (a.z() - b.z()).abs() > 1.0
            })
            .unwrap();
        let rounded = crate::blend_ops::fillet_v2(&mut topo, block, &[vertical], 0.5)
            .unwrap()
            .solid;
        let right = solid_faces(&topo, rounded)
            .unwrap()
            .into_iter()
            .find(|&f| {
                matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal, .. } if normal.x() > 0.5)
            })
            .unwrap();
        assert!(
            draft(
                &mut topo,
                rounded,
                &[right],
                Vec3::new(0.0, 0.0, 1.0),
                Point3::new(0.0, 0.0, 0.0),
                5.0_f64.to_radians(),
            )
            .is_err()
        );
    }

    #[test]
    fn right_angle_drafts_are_refused() {
        use std::f64::consts::{FRAC_PI_2, PI};
        for angle in [FRAC_PI_2, PI, -FRAC_PI_2, f64::NAN] {
            let mut topo = Topology::new();
            let cube = make_unit_cube_manifold(&mut topo);
            let right = find_faces(&topo, cube, Vec3::new(1.0, 0.0, 0.0));
            assert!(
                draft(
                    &mut topo,
                    cube,
                    &right,
                    Vec3::new(0.0, 0.0, 1.0),
                    Point3::new(0.0, 0.0, 0.0),
                    angle,
                )
                .is_err(),
                "angle {angle}"
            );
        }
    }

    /// The vertex check's tolerance follows the solid's size, so a box far
    /// from the origin drafts like one at it: the drafted side's top edge
    /// sits `2 tan(a)` in from its base.
    #[test]
    fn draft_far_from_the_origin_matches() {
        let far = 1.0e6;
        let a = 5.0_f64.to_radians();
        let mut topo = Topology::new();
        let block = crate::primitives::make_box(&mut topo, 4.0, 3.0, 2.0).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            block,
            &brepkit_math::mat::Mat4::translation(far, far, far),
        )
        .unwrap();
        let right = find_faces(&topo, block, Vec3::new(1.0, 0.0, 0.0));
        let result = draft(
            &mut topo,
            block,
            &right,
            Vec3::new(0.0, 0.0, 1.0),
            Point3::new(far, far, far),
            a,
        )
        .unwrap();
        let report = crate::validate::validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{:?}", report.issues);
        let mut xs: Vec<(f64, f64)> = Vec::new();
        for f in solid_faces(&topo, result).unwrap() {
            for oe in topo
                .wire(topo.face(f).unwrap().outer_wire())
                .unwrap()
                .edges()
            {
                let p = topo
                    .vertex(topo.edge(oe.edge()).unwrap().start())
                    .unwrap()
                    .point();
                xs.push((p.x() - far, p.z() - far));
            }
        }
        let top_right = xs
            .iter()
            .filter(|(x, z)| *z > 1.0 && *x > 1.0)
            .map(|(x, _)| *x)
            .collect::<Vec<_>>();
        assert!(!top_right.is_empty());
        for x in top_right {
            assert!((x - (4.0 - 2.0 * a.tan())).abs() < 1e-6, "top edge at {x}");
        }
    }

    /// A straight edge a boolean stored as a NURBS line follows its moved
    /// vertices as a line; a face the draft never touches keeps its pcurves.
    #[test]
    fn draft_straightens_nurbs_edges_and_keeps_untouched_pcurves() {
        use brepkit_math::curves2d::{Curve2D, Line2D};
        use brepkit_math::vec::{Point2, Vec2};
        use brepkit_topology::edge::EdgeCurve;
        use brepkit_topology::pcurve::PCurve;

        let a = 5.0_f64.to_radians();
        let mut topo = Topology::new();
        let block = crate::primitives::make_box(&mut topo, 4.0, 3.0, 2.0).unwrap();
        let right = find_faces(&topo, block, Vec3::new(1.0, 0.0, 0.0));
        let left = find_faces(&topo, block, Vec3::new(-1.0, 0.0, 0.0))[0];

        let bent = topo
            .wire(topo.face(right[0]).unwrap().outer_wire())
            .unwrap()
            .edges()[0]
            .edge();
        let (p, q) = {
            let edge = topo.edge(bent).unwrap();
            (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            )
        };
        let line = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![p, q],
            vec![1.0, 1.0],
        )
        .unwrap();
        topo.edge_mut(bent)
            .unwrap()
            .set_curve(EdgeCurve::NurbsCurve(line));

        let kept = topo
            .wire(topo.face(left).unwrap().outer_wire())
            .unwrap()
            .edges()[0]
            .edge();
        let pcurve = PCurve::new(
            Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap()),
            0.0,
            1.0,
        );
        topo.pcurves_mut().set(kept, left, pcurve);

        let result = draft(
            &mut topo,
            block,
            &right,
            Vec3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            a,
        )
        .unwrap();
        let volume = crate::measure::solid_volume(&topo, result, 0.001).unwrap();
        let truth = 24.0 - 6.0 * a.tan();
        assert!(
            (volume - truth).abs() < 1e-9 * truth,
            "volume {volume}, expected {truth}"
        );

        let faces = solid_faces(&topo, result).unwrap();
        for &f in &faces {
            for oe in topo
                .wire(topo.face(f).unwrap().outer_wire())
                .unwrap()
                .edges()
            {
                assert!(matches!(
                    topo.edge(oe.edge()).unwrap().curve(),
                    EdgeCurve::Line
                ));
            }
        }
        let new_left = find_faces(&topo, result, Vec3::new(-1.0, 0.0, 0.0))[0];
        let new_kept = topo
            .wire(topo.face(new_left).unwrap().outer_wire())
            .unwrap()
            .edges()[0]
            .edge();
        assert!(topo.pcurves().get(new_kept, new_left).is_some());
    }

    #[test]
    fn draft_zero_angle_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let right = find_faces(&topo, cube, Vec3::new(1.0, 0.0, 0.0));

        assert!(
            draft(
                &mut topo,
                cube,
                &right,
                Vec3::new(0.0, 0.0, 1.0),
                Point3::new(0.0, 0.0, 0.0),
                0.0,
            )
            .is_err()
        );
    }

    #[test]
    fn draft_zero_pull_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let right = find_faces(&topo, cube, Vec3::new(1.0, 0.0, 0.0));

        assert!(
            draft(
                &mut topo,
                cube,
                &right,
                Vec3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 0.0),
                5.0_f64.to_radians(),
            )
            .is_err()
        );
    }
}
