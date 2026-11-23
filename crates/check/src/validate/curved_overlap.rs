//! Whether two faces of one solid on one sphere, cylinder, cone or torus
//! cover the same area.
//!
//! As for plane faces, each edge of such a solid can have two uses, so the
//! edge checks pass, while the faces cover the surface twice there. Each face
//! is read beside its edges: at two places along each edge, on each side, at
//! two distances from it. Where both reads lie inside the face, they are read
//! on the other face, and both inside each face by more than the tolerance
//! show that the two share an area. Two faces that share only a part
//! narrower than the reads are not reported.

use brepkit_math::aabb::Aabb3;
use brepkit_math::madd::Madd;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use super::checks::{CheckId, EntityRef, Severity, ValidationIssue};
use crate::CheckError;
use crate::classify::boundary::{Holes, SphereRegion, face_contains_each};

/// The places along each edge a face is read beside, as fractions of it.
const ALONG: [f64; 2] = [0.25, 0.75];

/// The distances from an edge a face is read at, as fractions of the size
/// of the face's outline.
const BESIDE: [f64; 2] = [0.125, 0.0625];

/// A face on a sphere, a cylinder, a cone or a torus.
struct Curved {
    face: FaceId,
    surface: FaceSurface,
    bounds: Aabb3,
    /// Which kind of surface, and its size that two faces on one surface
    /// share: they are sorted by these.
    kind: u8,
    key: f64,
}

impl Curved {
    fn size(&self) -> f64 {
        (self.bounds.max - self.bounds.min).length()
    }
}

/// Whether two surfaces are one within `tol`, over a part `size` across.
fn one_surface(a: &FaceSurface, b: &FaceSurface, tol: f64, size: f64) -> bool {
    let parallel = |u: Vec3, v: Vec3| u.cross(v).length() * size <= tol;
    let off_line = |p: Point3, o: Point3, d: Vec3| {
        let w = p - o;
        (w - d * w.dot(d)).length()
    };
    match (a, b) {
        (FaceSurface::Sphere(s), FaceSurface::Sphere(t)) => {
            (s.center() - t.center()).length() <= tol && (s.radius() - t.radius()).abs() <= tol
        }
        (FaceSurface::Cylinder(s), FaceSurface::Cylinder(t)) => {
            (s.radius() - t.radius()).abs() <= tol
                && parallel(s.axis(), t.axis())
                && off_line(t.origin(), s.origin(), s.axis()) <= tol
        }
        (FaceSurface::Cone(s), FaceSurface::Cone(t)) => {
            (s.apex() - t.apex()).length() <= tol
                && s.axis().dot(t.axis()) > 0.0
                && parallel(s.axis(), t.axis())
                && (s.half_angle() - t.half_angle()).abs() * size <= tol
        }
        (FaceSurface::Torus(s), FaceSurface::Torus(t)) => {
            (s.center() - t.center()).length() <= tol
                && parallel(s.z_axis(), t.z_axis())
                && (s.major_radius() - t.major_radius()).abs() <= tol
                && (s.minor_radius() - t.minor_radius()).abs() <= tol
        }
        _ => false,
    }
}

/// Points beside each edge of `face`, inside it or out: for each place
/// along each edge and each side of it, a pair of points at the distances
/// of [`BESIDE`], each brought onto the surface.
fn probes(topo: &Topology, face: &Curved) -> Result<Vec<[Point3; 2]>, CheckError> {
    let size = outline_size(topo, face.face)?;
    let surface = &face.surface;
    let onto = |p: Point3| {
        surface
            .project_point(p)
            .and_then(|(u, v)| surface.evaluate(u, v))
    };
    let data = topo.face(face.face)?;
    let mut runs = Vec::new();
    for wid in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
        for oe in topo.wire(wid)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let curve = edge.curve();
            let (t0, t1) = curve.domain_with_endpoints(a, b);
            for f in ALONG {
                let t = (t1 - t0).madd(f, t0);
                let m = curve.evaluate_with_endpoints(t, a, b);
                let along = curve.tangent_with_endpoints(t, a, b);
                let Some((u, v)) = surface.project_point(m) else {
                    continue;
                };
                let Ok(across) = surface.normal(u, v).cross(along).normalize() else {
                    continue;
                };
                for sign in [1.0, -1.0] {
                    let run = BESIDE.map(|k| onto(m + across * (sign * k * size)));
                    if let [Some(p0), Some(p1)] = run {
                        runs.push([p0, p1]);
                    }
                }
            }
        }
    }
    Ok(runs)
}

/// The diagonal of the box of points along a face's outline. A face's own
/// box can be far larger: a cap's box is its whole sphere's.
fn outline_size(topo: &Topology, face_id: FaceId) -> Result<f64, CheckError> {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for oe in topo.wire(topo.face(face_id)?.outer_wire())?.edges() {
        let edge = topo.edge(oe.edge())?;
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let curve = edge.curve();
        let (t0, t1) = curve.domain_with_endpoints(a, b);
        for k in 0..=8 {
            let p = curve.evaluate_with_endpoints((t1 - t0).madd(f64::from(k) / 8.0, t0), a, b);
            for (c, x) in [p.x(), p.y(), p.z()].into_iter().enumerate() {
                lo[c] = lo[c].min(x);
                hi[c] = hi[c].max(x);
            }
        }
    }
    Ok((0..3)
        .map(|c| (hi[c] - lo[c]).max(0.0).powi(2))
        .sum::<f64>()
        .sqrt())
}

/// Whether `p`, read inside `face`, stands farther inside than the
/// tolerance: farther than it from each edge of the face, the edges read
/// along chords between samples less half the longest chord, and on a
/// sphere, inside beyond the slack of the tests against a wire's plane. A
/// read near the boundary shows no shared area, and a face too small to
/// hold such a read shows none.
fn strictly_inside(
    topo: &Topology,
    face: &Curved,
    p: Point3,
    tolerance: f64,
) -> Result<bool, CheckError> {
    const SAMPLES: u32 = 64;
    let data = topo.face(face.face)?;
    for wid in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
        for oe in topo.wire(wid)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let curve = edge.curve();
            let (t0, t1) = curve.domain_with_endpoints(a, b);
            let at = |k: u32| {
                curve.evaluate_with_endpoints(
                    (t1 - t0).madd(f64::from(k) / f64::from(SAMPLES), t0),
                    a,
                    b,
                )
            };
            let (mut near, mut longest) = (f64::INFINITY, 0.0_f64);
            for k in 0..SAMPLES {
                let (q, r) = (at(k), at(k + 1));
                let d = r - q;
                let d2 = d.dot(d);
                let s = if d2 > 0.0 {
                    ((p - q).dot(d) / d2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                near = near.min((p - (q + d * s)).length());
                longest = longest.max(d2.sqrt());
            }
            if near - 0.5 * longest <= tolerance {
                return Ok(false);
            }
        }
    }
    if !matches!(face.surface, FaceSurface::Sphere(_)) {
        return Ok(true);
    }
    let Some(region) = SphereRegion::of(topo, face.face)? else {
        return Ok(false);
    };
    region.contains_strictly(&Holes::new(), topo, face.face, p)
}

/// Which of `points` lie on `face`, or `None` for a sphere face whose
/// outline does not read.
fn on_face(
    topo: &Topology,
    face: &Curved,
    points: &[Point3],
) -> Result<Option<Vec<bool>>, CheckError> {
    if !matches!(face.surface, FaceSurface::Sphere(_)) {
        return face_contains_each(topo, face.face, points).map(Some);
    }
    let Some(region) = SphereRegion::of(topo, face.face)? else {
        return Ok(None);
    };
    let holes = Holes::new();
    points
        .iter()
        .map(|&p| region.contains_with(&holes, topo, face.face, p))
        .collect::<Result<_, _>>()
        .map(Some)
}

/// A face's points beside its edges and which of them lie on it (`None` for
/// a face that does not read).
type Read = (Vec<Point3>, Option<Vec<bool>>);

/// Check that no two faces of a solid on one sphere, cylinder, cone or
/// torus cover the same area.
///
/// Reports each such pair that shares an area. Faces that meet only along
/// an edge or at a point are not reported. Only faces whose surfaces agree
/// within `tolerance` and whose boxes meet are read.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn check_curved_faces_overlap(
    topo: &Topology,
    solid_id: SolidId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let mut curved = Vec::new();
    for face_id in solid_faces(topo, solid_id)? {
        let surface = topo.face(face_id)?.surface();
        let (kind, key) = match surface {
            FaceSurface::Sphere(s) => (0, s.radius()),
            FaceSurface::Cylinder(c) => (1, c.radius()),
            FaceSurface::Cone(c) => (2, c.half_angle()),
            FaceSurface::Torus(t) => (3, t.major_radius()),
            FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => continue,
        };
        curved.push(Curved {
            face: face_id,
            surface: surface.clone(),
            bounds: crate::util::face_aabb(topo, face_id)?,
            kind,
            key,
        });
    }
    curved.sort_by(|a, b| a.kind.cmp(&b.kind).then(a.key.total_cmp(&b.key)));
    // The pairs on one surface whose boxes meet.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (i, a) in curved.iter().enumerate() {
        // A cone's key is an angle, which parts across the faces: by no more
        // than the tolerance over the larger face, and so over this one.
        let window = if a.kind == 2 {
            tolerance / a.size().max(tolerance)
        } else {
            tolerance
        };
        for (j, b) in curved.iter().enumerate().skip(i + 1) {
            if b.kind != a.kind || b.key - a.key > window {
                break;
            }
            let size = a.size().max(b.size());
            if a.bounds.expanded(tolerance).intersects(b.bounds)
                && one_surface(&a.surface, &b.surface, tolerance, size)
            {
                pairs.push((i, j));
            }
        }
    }
    if pairs.is_empty() {
        return Ok(Vec::new());
    }
    // Each face in a pair is read once at its own points. A pair of points
    // is read on the other face only where both lie inside their own.
    let mut own: Vec<Option<Read>> = vec![None; curved.len()];
    let mut issues = Vec::new();
    for &(i, j) in &pairs {
        for k in [i, j] {
            if own[k].is_none() {
                let points: Vec<Point3> = probes(topo, &curved[k])?.into_iter().flatten().collect();
                let inside = on_face(topo, &curved[k], &points)?;
                own[k] = Some((points, inside));
            }
        }
        let mut shared = false;
        for (k, other) in [(i, j), (j, i)] {
            let Some((points, Some(inside))) = &own[k] else {
                continue;
            };
            // Each pair's first point.
            let candidates: Vec<usize> = (0..points.len())
                .step_by(BESIDE.len())
                .filter(|&at| inside[at] && inside[at + 1])
                .collect();
            if candidates.is_empty() {
                continue;
            }
            let read: Vec<Point3> = candidates
                .iter()
                .flat_map(|&at| [points[at], points[at + 1]])
                .collect();
            let Some(on_other) = on_face(topo, &curved[other], &read)? else {
                continue;
            };
            for pair in read.chunks_exact(2).zip(on_other.chunks_exact(2)) {
                let (points, [true, true]) = pair else {
                    continue;
                };
                let mut deep = true;
                for &p in points {
                    for face in [&curved[k], &curved[other]] {
                        deep = deep && strictly_inside(topo, face, p, tolerance)?;
                    }
                }
                if deep {
                    shared = true;
                    break;
                }
            }
            if shared {
                break;
            }
        }
        if shared {
            let (a, b) = (&curved[i], &curved[j]);
            issues.push(ValidationIssue {
                check: CheckId::SolidFacesOverlap,
                severity: Severity::Error,
                entity: EntityRef::Face(a.face),
                description: format!(
                    "faces {} and {} lie on one {} and cover the same area",
                    a.face.index(),
                    b.face.index(),
                    a.surface.type_tag()
                ),
                deviation: None,
            });
        }
    }
    Ok(issues)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, PI};

    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    const R: f64 = 10.0;

    fn sphere() -> FaceSurface {
        FaceSurface::Sphere(SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), R).unwrap())
    }

    /// A circle on the sphere, square to `axis`, `h` along it from the
    /// centre, as a closed edge.
    fn rim(topo: &mut Topology, axis: Vec3, h: f64) -> brepkit_topology::edge::EdgeId {
        let centre = Point3::new(0.0, 0.0, 0.0) + axis * h;
        let circle = Circle3D::new(centre, axis, (R * R - h * h).sqrt()).unwrap();
        let v = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
        topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)))
    }

    /// The face of the sphere on the side of `rim` its run faces: `forward`
    /// runs it about the axis it was made square to, which bounds the cap
    /// past it.
    fn on_sphere(
        topo: &mut Topology,
        rim: brepkit_topology::edge::EdgeId,
        forward: bool,
    ) -> FaceId {
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, forward)], true).unwrap());
        topo.add_face(Face::new(wire, vec![], sphere()))
    }

    fn solid(topo: &mut Topology, faces: Vec<FaceId>) -> SolidId {
        let shell = topo.add_shell(Shell::new(faces).unwrap());
        topo.add_solid(Solid::new(shell, vec![]))
    }

    fn found(topo: &Topology, solid: SolidId) -> Vec<String> {
        check_curved_faces_overlap(topo, solid, 1e-6)
            .unwrap()
            .into_iter()
            .map(|i| i.description)
            .collect()
    }

    fn z() -> Vec3 {
        Vec3::new(0.0, 0.0, 1.0)
    }

    /// Two faces of the cap past `z = 5`, one each way round, which close
    /// a shell of no volume along their one rim: reported, and by
    /// `validate_solid`.
    #[test]
    fn a_cap_and_its_copy_are_reported() {
        let mut topo = Topology::new();
        let edge = rim(&mut topo, z(), 5.0);
        let (a, b) = (
            on_sphere(&mut topo, edge, true),
            on_sphere(&mut topo, edge, true),
        );
        let lens = solid(&mut topo, vec![a, b]);
        let got = found(&topo, lens);
        assert_eq!(got.len(), 1, "{got:?}");
        assert!(got[0].contains("lie on one sphere and cover the same area"));
        let report = crate::validate::validate_solid(
            &topo,
            lens,
            &crate::validate::ValidateOptions::default(),
        )
        .unwrap();
        assert!(
            report
                .issues
                .iter()
                .any(|i| i.check == CheckId::SolidFacesOverlap),
            "{:?}",
            report.issues
        );
    }

    /// Two faces of the small cap past `z = √99`, 2 across on a sphere whose
    /// box is 20 across: read at distances from the cap's own rim, not its
    /// box's, they are reported.
    #[test]
    fn a_small_cap_and_its_copy_are_reported() {
        let mut topo = Topology::new();
        let edge = rim(&mut topo, z(), 99f64.sqrt());
        let (a, b) = (
            on_sphere(&mut topo, edge, true),
            on_sphere(&mut topo, edge, true),
        );
        let lens = solid(&mut topo, vec![a, b]);
        let got = found(&topo, lens);
        assert_eq!(got.len(), 1, "{got:?}");
    }

    /// On a ball of radius 1, the cap past `z = √(1 - 1e-10)`, 2e-5 across,
    /// and the rest of the sphere share only their rim: points beside the
    /// rim stand within the side tests' slack of the rim's plane on both, so
    /// they show nothing, and the pair passes.
    #[test]
    fn a_flat_small_cap_and_the_rest_of_its_sphere_pass() {
        let mut topo = Topology::new();
        let h = (1.0 - 1e-10_f64).sqrt();
        let circle = Circle3D::new(Point3::new(0.0, 0.0, h), z(), 1e-5).unwrap();
        let v = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
        let edge = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let ball =
            FaceSurface::Sphere(SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap());
        let faces: Vec<FaceId> = [true, false]
            .into_iter()
            .map(|forward| {
                let wire =
                    topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, forward)], true).unwrap());
                topo.add_face(Face::new(wire, vec![], ball.clone()))
            })
            .collect();
        let pair = solid(&mut topo, faces);
        let got = found(&topo, pair);
        assert!(got.is_empty(), "{got:?}");
    }

    /// The caps past `z = 5` and past `x = 5` share the part of the sphere
    /// round `(1, 0, 1)`; the caps past `z = 5` and past `z = -5` share
    /// nothing, nor do the cap past `z = 5` and the rest of the sphere.
    #[test]
    fn caps_that_share_an_area_are_reported_and_others_pass() {
        let mut topo = Topology::new();
        let top = rim(&mut topo, z(), 5.0);
        let side = rim(&mut topo, Vec3::new(1.0, 0.0, 0.0), 5.0);
        let bottom = rim(&mut topo, -z(), 5.0);
        let cap = on_sphere(&mut topo, top, true);
        let beside = on_sphere(&mut topo, side, true);
        let opposite = on_sphere(&mut topo, bottom, true);
        let rest = on_sphere(&mut topo, top, false);
        for (other, overlaps) in [(beside, true), (opposite, false), (rest, false)] {
            let pair = solid(&mut topo, vec![cap, other]);
            assert_eq!(
                found(&topo, pair).len(),
                usize::from(overlaps),
                "face {}",
                other.index()
            );
        }
    }

    /// A face of the cylinder of radius 5 about `z` between the angles
    /// `from` and `to` (anticlockwise) and the heights 0 and 10.
    fn wall(topo: &mut Topology, from: f64, to: f64) -> FaceId {
        let at = |t: f64, h: f64| Point3::new(5.0 * t.cos(), 5.0 * t.sin(), h);
        let arc = |topo: &mut Topology, h: f64| {
            let circle = Circle3D::new_with_ref(
                Point3::new(0.0, 0.0, h),
                z(),
                5.0,
                Vec3::new(1.0, 0.0, 0.0),
            )
            .unwrap();
            let a = topo.add_vertex(Vertex::new(at(from, h), 1e-7));
            let b = topo.add_vertex(Vertex::new(at(to, h), 1e-7));
            (
                a,
                b,
                topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle))),
            )
        };
        let (a0, b0, low) = arc(topo, 0.0);
        let (a1, b1, high) = arc(topo, 10.0);
        let up = topo.add_edge(Edge::new(b0, b1, EdgeCurve::Line));
        let down = topo.add_edge(Edge::new(a1, a0, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(low, true),
                    OrientedEdge::new(up, true),
                    OrientedEdge::new(high, false),
                    OrientedEdge::new(down, true),
                ],
                true,
            )
            .unwrap(),
        );
        let cylinder = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), z(), 5.0).unwrap();
        topo.add_face(Face::new(wire, vec![], FaceSurface::Cylinder(cylinder)))
    }

    /// Two halves of a wall share an edge; a quarter across the seam between
    /// them shares an area with each.
    #[test]
    fn walls_that_share_an_area_are_reported_and_halves_pass() {
        let mut topo = Topology::new();
        let first = wall(&mut topo, 0.0, PI);
        let second = wall(&mut topo, PI, 2.0 * PI);
        let across = wall(&mut topo, PI - FRAC_PI_2 * 0.5, PI + FRAC_PI_2 * 0.5);
        let halves = solid(&mut topo, vec![first, second]);
        assert!(found(&topo, halves).is_empty());
        for half in [first, second] {
            let pair = solid(&mut topo, vec![half, across]);
            assert_eq!(found(&topo, pair).len(), 1, "half {}", half.index());
        }
    }
}
