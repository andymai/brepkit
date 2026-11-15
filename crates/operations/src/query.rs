//! Shape query utilities.

use std::collections::{HashMap, HashSet};

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// Filter edges to only those shared by two planar faces in a solid.
///
/// Given a solid and a set of edge IDs, returns only the edges
/// where both adjacent faces have a planar surface.
///
/// # Errors
///
/// Returns `OperationsError::Topology` if any entity ID is invalid.
pub fn filter_planar_edges(
    topo: &Topology,
    solid_id: SolidId,
    edge_ids: &[EdgeId],
) -> Result<Vec<EdgeId>, OperationsError> {
    let solid_data = topo.solid(solid_id)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut edge_faces: HashMap<usize, Vec<FaceId>> = HashMap::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            edge_faces.entry(oe.edge().index()).or_default().push(fid);
        }
    }

    let mut result = Vec::new();
    for &eid in edge_ids {
        if let Some(adj_faces) = edge_faces.get(&eid.index()) {
            let all_planar = adj_faces.iter().all(|&fid| {
                topo.face(fid)
                    .map(|f| matches!(f.surface(), FaceSurface::Plane { .. }))
                    .unwrap_or(false)
            });
            if all_planar {
                result.push(eid);
            }
        }
    }
    Ok(result)
}

/// Filter edges to only those the blend engine can fillet: manifold edges
/// (shared by exactly two distinct faces) that meet at a real (non-tangent)
/// angle.
///
/// Edges bordering a curved neighbour — including a previous fillet's NURBS
/// blend face — ARE filletable: the corrected [`brepkit_blend::fillet_builder::FilletBuilder`]
/// solves the true ball-tangent contacts against analytic and NURBS surfaces.
/// The cases that genuinely have no fillet are **tangent / G1** edges (the two
/// faces meet smoothly, e.g. a fillet face's contact line with its planar
/// neighbour) and degenerate folds; those are excluded here so callers never
/// feed them to the engine.
///
/// Production dispatch additionally validates complete edge accounting and
/// the assembled solid before publication.
///
/// # Errors
///
/// Returns `OperationsError::Topology` if any entity ID is invalid.
pub fn filter_filletable_edges(
    topo: &Topology,
    solid_id: SolidId,
    edge_ids: &[EdgeId],
) -> Result<Vec<EdgeId>, OperationsError> {
    let solid_data = topo.solid(solid_id)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    // Map each edge to its set of *distinct* adjacent faces, walking both outer
    // and inner (hole-boundary) wires — the same adjacency the fillet engine
    // sees. The set dedups a seam edge that a single face's wire lists twice.
    let mut edge_faces: HashMap<usize, HashSet<FaceId>> = HashMap::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let mut wires = vec![face.outer_wire()];
        wires.extend(face.inner_wires().iter().copied());
        for wid in wires {
            for oe in topo.wire(wid)?.edges() {
                edge_faces.entry(oe.edge().index()).or_default().insert(fid);
            }
        }
    }

    let mut result = Vec::new();
    for &eid in edge_ids {
        let Some(adj_faces) = edge_faces.get(&eid.index()) else {
            continue;
        };
        if adj_faces.len() != 2 {
            continue;
        }
        if edge_is_tangent(topo, eid, adj_faces)? {
            continue;
        }
        result.push(eid);
    }
    Ok(result)
}

/// Whether the two faces of `eid` meet tangentially (G1) — their effective
/// outward normals are (anti)parallel at the edge midpoint, so there is no
/// real dihedral to round. Returns `true` for the degenerate cases the fillet
/// engine cannot blend.
fn edge_is_tangent(
    topo: &Topology,
    eid: EdgeId,
    faces: &HashSet<FaceId>,
) -> Result<bool, OperationsError> {
    let mut it = faces.iter().copied();
    let (Some(f1), Some(f2)) = (it.next(), it.next()) else {
        return Ok(true);
    };
    let edge = topo.edge(eid)?;
    let a = topo.vertex(edge.start())?.point();
    let b = topo.vertex(edge.end())?.point();
    let mid = a + (b - a) * 0.5;

    let normal = |fid: FaceId| outward_normal_near(topo, fid, mid);

    match (normal(f1), normal(f2)) {
        (Some(n1), Some(n2)) => {
            let cos = n1.dot(n2).clamp(-1.0, 1.0);
            // Tangent within ~5°: |angle| < 5° (cos > 0.9962) or > 175°.
            Ok(cos.abs() > 0.9962)
        }
        // Can't determine a normal — don't exclude; the engine + manifold guard
        // will decide.
        _ => Ok(false),
    }
}

/// The outward normal of a face at the point of its surface nearest `near`.
fn outward_normal_near(
    topo: &Topology,
    fid: FaceId,
    near: brepkit_math::vec::Point3,
) -> Option<brepkit_math::vec::Vec3> {
    let face = topo.face(fid).ok()?;
    let n = match face.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        other => {
            let (u, v) = other.project_point(near)?;
            other.normal(u, v)
        }
    };
    let n = if face.is_reversed() { -n } else { n };
    n.normalize().ok()
}

/// The edges of `edge_ids` that lie inside one surface of the solid, so that
/// they have no angle to round or bevel.
///
/// Such an edge is the seam of one face: the face uses it one time in each
/// direction. Or it is the edge between two faces that are parts of one
/// surface with the material on one side, such as the two halves of a
/// cylinder wall that a boolean or an imported file keeps as two faces.
///
/// The solid's faces are read one time for the whole of `edge_ids`.
///
/// # Errors
///
/// Returns `OperationsError::Topology` if any entity ID is invalid.
pub fn edges_in_one_surface(
    topo: &Topology,
    solid_id: SolidId,
    edge_ids: &[EdgeId],
) -> Result<HashSet<usize>, OperationsError> {
    let wanted: HashSet<usize> = edge_ids.iter().map(|e| e.index()).collect();
    let mut uses: HashMap<usize, Vec<(FaceId, bool)>> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid_id)? {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid)?.edges() {
                if wanted.contains(&oe.edge().index()) {
                    uses.entry(oe.edge().index())
                        .or_default()
                        .push((fid, oe.is_forward()));
                }
            }
        }
    }

    let mut inside = HashSet::new();
    for &edge_id in edge_ids {
        let Some(&[(f1, forward1), (f2, forward2)]) = uses.get(&edge_id.index()).map(Vec::as_slice)
        else {
            continue;
        };
        let one_surface = if f1 == f2 {
            forward1 != forward2
        } else if same_surface(topo.face(f1)?.surface(), topo.face(f2)?.surface()) {
            // The middle of the edge on its curve: the middle of an arc's
            // chord can be on the axis of its surface, which has no normal.
            let edge = topo.edge(edge_id)?;
            let a = topo.vertex(edge.start())?.point();
            let b = topo.vertex(edge.end())?.point();
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            let mid = edge.curve().evaluate_with_endpoints(0.5 * (t0 + t1), a, b);
            match (
                outward_normal_near(topo, f1, mid),
                outward_normal_near(topo, f2, mid),
            ) {
                (Some(n1), Some(n2)) => n1.dot(n2) > 0.9962,
                _ => false,
            }
        } else {
            false
        };
        if one_surface {
            inside.insert(edge_id.index());
        }
    }
    Ok(inside)
}

/// Whether one edge lies inside one surface of the solid: see
/// [`edges_in_one_surface`], which reads the solid one time for many edges.
///
/// # Errors
///
/// Returns `OperationsError::Topology` if any entity ID is invalid.
pub fn edge_lies_in_one_surface(
    topo: &Topology,
    solid_id: SolidId,
    edge_id: EdgeId,
) -> Result<bool, OperationsError> {
    Ok(!edges_in_one_surface(topo, solid_id, &[edge_id])?.is_empty())
}

/// Whether two faces' surfaces are one surface: equal analytic surfaces
/// within the standard tolerances (an angle is read as an angle), or B-spline surfaces with the same
/// degrees, knots, control points and weights.
fn same_surface(a: &FaceSurface, b: &FaceSurface) -> bool {
    let tol = brepkit_math::tolerance::Tolerance::new();
    // The healer reads two planes as one within a wide tolerance, which a
    // shallow crease also passes.
    if let (FaceSurface::Plane { normal: na, d: da }, FaceSurface::Plane { normal: nb, d: db }) =
        (a, b)
    {
        // The cross product reads the angle itself: `1 - |dot|` reads half
        // its square, which a crease of 1e-6 rad passes.
        let dot = na.dot(*nb);
        return na.cross(*nb).length() <= tol.angular
            && (da - db * dot.signum()).abs() < tol.linear;
    }
    // The healer reads an angle between two axes from `1 - |dot|` too, so
    // each analytic pair has its compare here.
    let along =
        |p: brepkit_math::vec::Vec3, q: brepkit_math::vec::Vec3| p.cross(q).length() <= tol.angular;
    let (a, b) = match (a, b) {
        (FaceSurface::Nurbs(a), FaceSurface::Nurbs(b)) => (a, b),
        (FaceSurface::Cylinder(a), FaceSurface::Cylinder(b)) => {
            return (a.radius() - b.radius()).abs() < tol.linear
                && along(a.axis(), b.axis())
                && (b.origin() - a.origin()).cross(a.axis()).length() < tol.linear;
        }
        (FaceSurface::Cone(a), FaceSurface::Cone(b)) => {
            return (a.half_angle() - b.half_angle()).abs() <= tol.angular
                && along(a.axis(), b.axis())
                && a.axis().dot(b.axis()) > 0.0
                && (b.apex() - a.apex()).length() < tol.linear;
        }
        (FaceSurface::Sphere(a), FaceSurface::Sphere(b)) => {
            return (a.radius() - b.radius()).abs() < tol.linear
                && (b.center() - a.center()).length() < tol.linear;
        }
        (FaceSurface::Torus(a), FaceSurface::Torus(b)) => {
            return (a.major_radius() - b.major_radius()).abs() < tol.linear
                && (a.minor_radius() - b.minor_radius()).abs() < tol.linear
                && along(a.z_axis(), b.z_axis())
                && (b.center() - a.center()).length() < tol.linear;
        }
        (
            FaceSurface::Plane { .. }
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
            | FaceSurface::Nurbs(_),
            _,
        ) => return false,
    };
    // Parts of one surface hold copies of it. The compare is relative, since
    // a weight has no length scale.
    let same_values = |x: &[f64], y: &[f64]| {
        x.len() == y.len()
            && x.iter()
                .zip(y)
                .all(|(&p, &q)| (p - q).abs() <= 1e-12 * p.abs().max(q.abs()))
    };
    let same_rows = |x: &[Vec<f64>], y: &[Vec<f64>]| {
        x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_values(p, q))
    };
    let same_points = a.control_points().len() == b.control_points().len()
        && a.control_points()
            .iter()
            .zip(b.control_points())
            .all(|(p, q)| {
                p.len() == q.len()
                    && p.iter()
                        .zip(q)
                        .all(|(&s, &t)| (s - t).length() <= tol.linear)
            });
    // Parts of one surface hold copies of its knots, bit for bit. A knot
    // has no scale to compare against: a small move of one changes the
    // surface.
    let same_knots = |x: &[f64], y: &[f64]| {
        x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
    };
    a.degree_u() == b.degree_u()
        && a.degree_v() == b.degree_v()
        && same_knots(a.knots_u(), b.knots_u())
        && same_knots(a.knots_v(), b.knots_v())
        && same_points
        && same_rows(a.weights(), b.weights())
        && !can_crease(a.knots_u(), a.degree_u())
        && !can_crease(a.knots_v(), a.degree_v())
}

/// Whether a B-spline of this degree can have a corner at an inner knot: a
/// knot there occurs as many times as the degree, or more. Two faces on the
/// two sides of such a knot can meet at a real angle on one surface.
fn can_crease(knots: &[f64], degree: usize) -> bool {
    let (Some(first), Some(last)) = (knots.first(), knots.last()) else {
        return false;
    };
    let mut run = 0;
    let mut previous = f64::NAN;
    for knot in knots {
        if knot.to_bits() == previous.to_bits() {
            run += 1;
        } else {
            run = 1;
            previous = *knot;
        }
        let inner = knot.to_bits() != first.to_bits() && knot.to_bits() != last.to_bits();
        if inner && run >= degree.max(1) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, deprecated)]

    use brepkit_topology::explorer::solid_edges;

    use super::*;

    #[test]
    fn a_box_edge_and_a_cylinders_rim_lie_in_two_surfaces() {
        let mut topo = Topology::new();
        let cube = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        for edge in solid_edges(&topo, cube).unwrap() {
            assert!(!edge_lies_in_one_surface(&topo, cube, edge).unwrap());
        }
        let rod = crate::primitives::make_cylinder(&mut topo, 3.0, 10.0).unwrap();
        let inside: Vec<bool> = solid_edges(&topo, rod)
            .unwrap()
            .into_iter()
            .map(|edge| edge_lies_in_one_surface(&topo, rod, edge).unwrap())
            .collect();
        assert_eq!(
            inside.iter().filter(|&&seam| seam).count(),
            1,
            "the wall's seam is the one edge inside a surface: {inside:?}"
        );
        assert_eq!(inside.len(), 3);
    }

    #[test]
    fn the_line_between_two_halves_of_a_wall_lies_in_one_surface() {
        use crate::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
        use brepkit_math::mat::Mat4;

        // A rod cut by a box that takes the side of the wall away from the
        // seam: without a merge of faces, the wall stays in two faces.
        let mut topo = Topology::new();
        let rod = crate::primitives::make_cylinder(&mut topo, 3.0, 10.0).unwrap();
        let tool = crate::primitives::make_box(&mut topo, 4.0, 2.0, 12.0).unwrap();
        crate::transform::transform_solid(&mut topo, tool, &Mat4::translation(-5.0, -1.0, -1.0))
            .unwrap();
        let opts = BooleanOptions {
            mesh_fallback: false,
            unify_faces: false,
            ..BooleanOptions::default()
        };
        let cut = boolean_with_options(&mut topo, BooleanOp::Cut, rod, tool, opts).unwrap();

        let walls = brepkit_topology::explorer::solid_faces(&topo, cut)
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
            .count();
        assert_eq!(walls, 2, "the wall is in two faces");
        let inside: Vec<EdgeId> = solid_edges(&topo, cut)
            .unwrap()
            .into_iter()
            .filter(|&edge| edge_lies_in_one_surface(&topo, cut, edge).unwrap())
            .collect();
        assert_eq!(inside.len(), 1, "one line between the two halves");
        let line = topo.edge(inside[0]).unwrap();
        for end in [line.start(), line.end()] {
            let p = topo.vertex(end).unwrap().point();
            assert!((p.x() - 3.0).abs() < 1e-9 && p.y().abs() < 1e-9, "{p:?}");
        }
        assert!(
            filter_filletable_edges(&topo, cut, &inside)
                .unwrap()
                .is_empty(),
            "the line has no angle to round"
        );
    }

    #[test]
    fn a_rounds_contact_line_lies_in_two_surfaces() {
        let mut topo = Topology::new();
        let cube = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let edges = solid_edges(&topo, cube).unwrap();
        let rounded = crate::blend_ops::fillet_v2(&mut topo, cube, &[edges[0]], 1.0)
            .unwrap()
            .solid;
        let all = solid_edges(&topo, rounded).unwrap();
        let sharp: HashSet<usize> = filter_filletable_edges(&topo, rounded, &all)
            .unwrap()
            .iter()
            .map(|e| e.index())
            .collect();
        let tangent: Vec<EdgeId> = all
            .iter()
            .copied()
            .filter(|e| !sharp.contains(&e.index()))
            .collect();
        assert_eq!(tangent.len(), 2, "the round's two contact lines");
        for edge in tangent {
            assert!(!edge_lies_in_one_surface(&topo, rounded, edge).unwrap());
        }
    }

    #[test]
    fn two_planes_at_a_shallow_crease_are_two_surfaces() {
        use brepkit_math::vec::Vec3;

        let flat = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 2.0,
        };
        let tilted = FaceSurface::Plane {
            normal: Vec3::new(0.01_f64.sin(), 0.0, 0.01_f64.cos()),
            d: 2.0,
        };
        let turned_over = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: -2.0,
        };
        let shifted = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 2.0 + 1e-4,
        };
        assert!(same_surface(&flat, &flat));
        assert!(same_surface(&flat, &turned_over));
        assert!(!same_surface(&flat, &tilted));
        let barely_tilted = FaceSurface::Plane {
            normal: Vec3::new(1e-6_f64.sin(), 0.0, 1e-6_f64.cos()),
            d: 2.0,
        };
        assert!(!same_surface(&flat, &barely_tilted));
        assert!(!same_surface(&flat, &shifted));
    }

    #[test]
    fn an_edge_that_one_face_uses_one_time_is_no_seam() {
        // A cylinder's wall alone: each rim is used one time, by one face.
        let mut topo = Topology::new();
        let rod = crate::primitives::make_cylinder(&mut topo, 3.0, 10.0).unwrap();
        let wall = brepkit_topology::explorer::solid_faces(&topo, rod)
            .unwrap()
            .into_iter()
            .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
            .unwrap();
        let shell = topo.add_shell(brepkit_topology::shell::Shell::new(vec![wall]).unwrap());
        let open = topo.add_solid(brepkit_topology::solid::Solid::new(shell, vec![]));
        let edges = brepkit_topology::explorer::face_edges(&topo, wall).unwrap();
        let inside = edges_in_one_surface(&topo, open, &edges).unwrap();
        assert_eq!(inside.len(), 1, "the seam only, and no rim: {inside:?}");
    }

    #[test]
    fn a_balls_equator_lies_in_one_surface() {
        let mut topo = Topology::new();
        let ball = crate::primitives::make_sphere(&mut topo, 3.0, 16).unwrap();
        let all = solid_edges(&topo, ball).unwrap();
        let inside = edges_in_one_surface(&topo, ball, &all).unwrap();
        assert_eq!(all.len(), 16);
        assert_eq!(inside.len(), 16, "each edge is between the two halves");
    }

    #[test]
    fn two_cylinders_at_a_shallow_tilt_are_two_surfaces() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::{Point3, Vec3};

        let wall = |axis: Vec3, radius: f64| {
            FaceSurface::Cylinder(
                CylindricalSurface::new(Point3::new(1.0, 2.0, 3.0), axis, radius).unwrap(),
            )
        };
        let upright = wall(Vec3::new(0.0, 0.0, 1.0), 3.0);
        assert!(same_surface(&upright, &wall(Vec3::new(0.0, 0.0, 1.0), 3.0)));
        assert!(same_surface(
            &upright,
            &wall(Vec3::new(0.0, 0.0, -1.0), 3.0)
        ));
        assert!(!same_surface(
            &upright,
            &wall(Vec3::new(1e-6_f64.sin(), 0.0, 1e-6_f64.cos()), 3.0)
        ));
        assert!(!same_surface(
            &upright,
            &wall(Vec3::new(0.0, 0.0, 1.0), 3.0 + 1e-4)
        ));
    }

    #[test]
    fn two_spline_surfaces_with_other_weights_are_two_surfaces() {
        use brepkit_math::nurbs::surface::NurbsSurface;
        use brepkit_math::vec::Point3;

        let patch = |w: f64| {
            let points = vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
            ];
            let weights = vec![vec![1e-8, 1e-8], vec![1e-8, w]];
            let knots = vec![0.0, 0.0, 1.0, 1.0];
            FaceSurface::Nurbs(
                NurbsSurface::new(1, 1, knots.clone(), knots, points, weights).unwrap(),
            )
        };
        assert!(same_surface(&patch(1e-8), &patch(1e-8)));
        assert!(!same_surface(&patch(1e-8), &patch(3e-8)));
    }

    #[test]
    fn two_spline_surfaces_with_other_knots_or_a_corner_knot_are_two_surfaces() {
        use brepkit_math::nurbs::surface::NurbsSurface;
        use brepkit_math::vec::Point3;

        // Degree 2 by 1, four rows of control points, one inner knot in u.
        let strip = |inner: f64, offset: f64| {
            let points = (0..4)
                .map(|i| {
                    let x = f64::from(i);
                    vec![Point3::new(x, 0.0, x * x), Point3::new(x, 1.0, x * x)]
                })
                .collect();
            let knots_u: Vec<f64> = [0.0, 0.0, 0.0, inner, 2.0, 2.0, 2.0]
                .iter()
                .map(|k| k + offset)
                .collect();
            FaceSurface::Nurbs(
                NurbsSurface::new(
                    2,
                    1,
                    knots_u,
                    vec![0.0, 0.0, 1.0, 1.0],
                    points,
                    vec![vec![1.0; 2]; 4],
                )
                .unwrap(),
            )
        };
        assert!(same_surface(&strip(1.0, 0.0), &strip(1.0, 0.0)));
        assert!(!same_surface(&strip(1.0, 1e12), &strip(0.5, 1e12)));

        assert!(!can_crease(&[0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0], 2));
        assert!(can_crease(&[0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0], 2));
        assert!(can_crease(&[0.0, 0.0, 1.0, 2.0, 2.0], 1));
    }

    #[test]
    fn filletable_edges_all_planar_box() {
        let mut topo = Topology::new();
        let cube = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let edges = solid_edges(&topo, cube).unwrap();
        let filletable = filter_filletable_edges(&topo, cube, &edges).unwrap();
        assert_eq!(
            filletable.len(),
            edges.len(),
            "every box edge is plane↔plane and filletable"
        );
        assert_eq!(edges.len(), 12);
    }

    #[test]
    fn filletable_edges_keep_nontangent_blend_edges_drop_tangent() {
        // A single rolling-ball fillet makes a watertight solid with a NURBS
        // blend face. Its NURBS-blend-border edges split into tangent/G1 contact
        // lines (degenerate → excluded) and real-angle end-caps (→ kept).
        let mut topo = Topology::new();
        let cube = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let edges = solid_edges(&topo, cube).unwrap();
        let filleted =
            crate::fillet::fillet_rolling_ball(&mut topo, cube, &[edges[0]], 1.0).unwrap();
        let r_edges = solid_edges(&topo, filleted).unwrap();
        let filletable: HashSet<usize> = filter_filletable_edges(&topo, filleted, &r_edges)
            .unwrap()
            .iter()
            .map(|e| e.index())
            .collect();

        let sh = topo
            .shell(topo.solid(filleted).unwrap().outer_shell())
            .unwrap();
        // The blend face, whatever surface type it carries. A straight box
        // edge blends to an exact cylinder; only curved neighbours give NURBS.
        let blend_faces: HashSet<usize> = sh
            .faces()
            .iter()
            .filter(|&&f| !topo.face(f).unwrap().surface().is_planar())
            .map(|f| f.index())
            .collect();
        assert!(
            !blend_faces.is_empty(),
            "first fillet must create a blend face"
        );

        let mut ef: HashMap<usize, HashSet<FaceId>> = HashMap::new();
        for &fid in sh.faces() {
            for oe in topo
                .wire(topo.face(fid).unwrap().outer_wire())
                .unwrap()
                .edges()
            {
                ef.entry(oe.edge().index()).or_default().insert(fid);
            }
        }

        let (mut saw_kept, mut saw_dropped_tangent) = (false, false);
        for &e in &r_edges {
            let Some(fs) = ef.get(&e.index()) else {
                continue;
            };
            if fs.len() != 2 || !fs.iter().any(|f| blend_faces.contains(&f.index())) {
                continue;
            }
            if edge_is_tangent(&topo, e, fs).unwrap() {
                assert!(
                    !filletable.contains(&e.index()),
                    "tangent blend-contact edge {} must be excluded",
                    e.index()
                );
                saw_dropped_tangent = true;
            } else {
                assert!(
                    filletable.contains(&e.index()),
                    "non-tangent blend-adjacent edge {} must stay filletable",
                    e.index()
                );
                saw_kept = true;
            }
        }
        assert!(saw_kept, "expected a kept non-tangent NURBS-blend edge");
        assert!(
            saw_dropped_tangent,
            "expected an excluded tangent contact edge"
        );
    }
}
