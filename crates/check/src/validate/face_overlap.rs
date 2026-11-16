//! Whether two flat faces of one solid cover the same area.
//!
//! A face can lie on a part of another face, either way round. Each edge of
//! such a solid can have two uses, so the edge checks pass, while its mesh
//! is open.
//!
//! The two faces are read in their common plane, each with its own lines,
//! arcs and B-splines. A point just inside one face, beside one of its edges,
//! that lies inside the other face shows that the two faces share an area.
//! Where two faces share an area, the edge of one of them runs inside the
//! other or along the other's edge on the same side, so such a point is
//! found beside it.

use brepkit_math::aabb::Aabb3;
use brepkit_math::frame::Frame3;
use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::planar::{PlanarRegion, face_region_2d};
use brepkit_topology::solid::SolidId;

use super::checks::{CheckId, EntityRef, Severity, ValidationIssue};
use crate::CheckError;

/// The fractions of an edge beside which a face's points are read.
const FRACTIONS: [f64; 7] = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875];

/// A direction square to no common plane, which turns each plane's normal to
/// one side, so that two faces on one plane read the same normal and offset
/// whichever way round they face.
const SIDE: Vec3 = Vec3::new(0.577_215_664_9, 0.618_033_988_7, 0.533_102_917_2);

/// A plane face, its plane turned to [`SIDE`].
struct Flat {
    face: FaceId,
    normal: Vec3,
    offset: f64,
    origin: Point3,
    bounds: Aabb3,
}

impl Flat {
    fn of(topo: &Topology, face_id: FaceId) -> Result<Option<Self>, CheckError> {
        let face = topo.face(face_id)?;
        let FaceSurface::Plane { normal, .. } = face.surface() else {
            return Ok(None);
        };
        let Ok(mut normal) = normal.normalize() else {
            return Ok(None);
        };
        if normal.dot(SIDE) < 0.0 {
            normal = -normal;
        }
        let Some(first) = topo.wire(face.outer_wire())?.edges().first() else {
            return Ok(None);
        };
        let origin = topo.vertex(topo.edge(first.edge())?.start())?.point();
        let bounds = crate::util::face_aabb(topo, face_id)?;
        Ok(Some(Self {
            face: face_id,
            normal,
            offset: normal.dot(Vec3::new(origin.x(), origin.y(), origin.z())),
            origin,
            bounds,
        }))
    }

    fn size(&self) -> f64 {
        (self.bounds.max - self.bounds.min).length()
    }
}

/// Whether the two faces lie on one plane within `tol` and their boxes meet.
fn on_one_plane(a: &Flat, b: &Flat, tol: f64) -> bool {
    let reach = a.size().max(b.size());
    (a.offset - b.offset).abs() <= tol
        && (a.normal - b.normal).length() * reach <= tol
        && a.bounds.expanded(tol).intersects(b.bounds)
}

/// Points of `face` read in the plane of `frame`: for each edge, at
/// [`FRACTIONS`], the point a little way to each side of the edge, the
/// nearest that lies inside the face. Each is tested against `other`, and
/// the first inside it returns `true`.
fn shares_area(
    topo: &Topology,
    face_id: FaceId,
    region: &PlanarRegion,
    other: &PlanarRegion,
    frame: &Frame3,
    size: f64,
    tol: f64,
) -> Result<bool, CheckError> {
    let flat = |p: Point3| {
        let d = p - frame.origin;
        Point2::new(d.dot(frame.x), d.dot(frame.y))
    };
    let face = topo.face(face_id)?;
    let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
    for wid in wires {
        for oe in topo.wire(wid)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let curve = edge.curve();
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            if matches!(curve, EdgeCurve::Line) && (b - a).length() <= tol {
                continue;
            }
            let (t0, t1) = curve.domain_with_endpoints(a, b);
            for f in FRACTIONS {
                let t = t0 + (t1 - t0) * f;
                let p = flat(curve.evaluate_with_endpoints(t, a, b));
                let along = curve.tangent_with_endpoints(t, a, b);
                let along = Vec2::new(along.dot(frame.x), along.dot(frame.y));
                let Ok(across) = Vec2::new(-along.y(), along.x()).normalize() else {
                    continue;
                };
                // The nearest point inside the face, of three steps each a
                // sixteenth of the last, the smallest well clear of `tol`.
                let mut step = 1e-3 * size;
                for _ in 0..3 {
                    if step <= 16.0 * tol {
                        break;
                    }
                    let inside: Vec<Point2> = [step, -step]
                        .into_iter()
                        .map(|s| p + across * s)
                        .filter(|&q| region.side(q, tol) == Some(true))
                        .collect();
                    if inside.iter().any(|&q| other.side(q, tol) == Some(true)) {
                        return Ok(true);
                    }
                    if !inside.is_empty() {
                        break;
                    }
                    step /= 16.0;
                }
            }
        }
    }
    Ok(false)
}

/// Check that no two plane faces of a solid cover the same area.
///
/// Reports each pair of plane faces that lie on one plane, either way round,
/// and share an area. Faces that meet only along an edge or at a point are
/// not reported. Only faces whose planes agree within `tolerance` and whose
/// boxes meet are read along their edges.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn check_faces_overlap(
    topo: &Topology,
    solid_id: SolidId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let mut flats = Vec::new();
    for face_id in solid_faces(topo, solid_id)? {
        if let Some(flat) = Flat::of(topo, face_id)? {
            flats.push(flat);
        }
    }
    flats.sort_by(|a, b| a.offset.total_cmp(&b.offset));
    let mut issues = Vec::new();
    for (i, a) in flats.iter().enumerate() {
        for b in &flats[i + 1..] {
            if b.offset - a.offset > tolerance {
                break;
            }
            if !on_one_plane(a, b, tolerance) {
                continue;
            }
            let Ok(frame) = Frame3::from_normal(a.origin, a.normal) else {
                continue;
            };
            let region =
                |face| face_region_2d(topo, face, frame.origin, frame.x, frame.y, tolerance);
            let (Some(ra), Some(rb)) = (region(a.face)?, region(b.face)?) else {
                continue;
            };
            if shares_area(topo, a.face, &ra, &rb, &frame, a.size(), tolerance)?
                || shares_area(topo, b.face, &rb, &ra, &frame, b.size(), tolerance)?
            {
                issues.push(ValidationIssue {
                    check: CheckId::SolidFacesOverlap,
                    severity: Severity::Error,
                    entity: EntityRef::Face(a.face),
                    description: format!(
                        "faces {} and {} lie on one plane and cover the same area",
                        a.face.index(),
                        b.face.index()
                    ),
                    deviation: None,
                });
            }
        }
    }
    Ok(issues)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::HashMap;

    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_topology::edge::{Edge, EdgeId};
    use brepkit_topology::face::Face;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::{Vertex, VertexId};
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// Plane faces of lines through shared vertices and edges.
    #[derive(Default)]
    struct Builder {
        topo: Topology,
        vertices: Vec<(Point3, VertexId)>,
        edges: HashMap<(usize, usize), EdgeId>,
        faces: Vec<FaceId>,
    }

    impl Builder {
        fn vertex(&mut self, p: [f64; 3]) -> VertexId {
            let p = Point3::new(p[0], p[1], p[2]);
            if let Some(&(_, v)) = self.vertices.iter().find(|(q, _)| (*q - p).length() < 1e-9) {
                return v;
            }
            let v = self.topo.add_vertex(Vertex::new(p, 1e-7));
            self.vertices.push((p, v));
            v
        }

        fn edge(&mut self, a: VertexId, b: VertexId) -> OrientedEdge {
            let key = (a.index().min(b.index()), a.index().max(b.index()));
            let id = *self
                .edges
                .entry(key)
                .or_insert_with(|| self.topo.add_edge(Edge::new(a, b, EdgeCurve::Line)));
            OrientedEdge::new(id, self.topo.edge(id).unwrap().start() == a)
        }

        /// A face through `corners`, run counter-clockwise about `normal`.
        fn face(&mut self, corners: &[[f64; 3]], normal: [f64; 3]) -> FaceId {
            let normal = Vec3::new(normal[0], normal[1], normal[2])
                .normalize()
                .unwrap();
            let points: Vec<Point3> = corners
                .iter()
                .map(|c| Point3::new(c[0], c[1], c[2]))
                .collect();
            let mut turn = Vec3::new(0.0, 0.0, 0.0);
            for k in 0..points.len() {
                let (a, b) = (points[k], points[(k + 1) % points.len()]);
                turn += Vec3::new(a.x(), a.y(), a.z()).cross(Vec3::new(b.x(), b.y(), b.z()));
            }
            let mut ids: Vec<VertexId> = corners.iter().map(|&c| self.vertex(c)).collect();
            if turn.dot(normal) < 0.0 {
                ids.reverse();
            }
            let edges = (0..ids.len())
                .map(|k| self.edge(ids[k], ids[(k + 1) % ids.len()]))
                .collect();
            let wire = self.topo.add_wire(Wire::new(edges, true).unwrap());
            let origin = Vec3::new(corners[0][0], corners[0][1], corners[0][2]);
            let face = self.topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane {
                    normal,
                    d: normal.dot(origin),
                },
            ));
            self.faces.push(face);
            face
        }

        fn solid(&mut self) -> SolidId {
            let shell = self.topo.add_shell(Shell::new(self.faces.clone()).unwrap());
            self.topo.add_solid(Solid::new(shell, vec![]))
        }
    }

    /// A 10 cube less a chamfer of 1 along its edge on y = 0 and z = 10.
    /// With `patched`, each end wall keeps its whole square, and a triangle
    /// on it, facing into the cube, closes the chamfer's end.
    fn chamfered_cube(patched: bool) -> (Topology, SolidId) {
        let mut b = Builder::default();
        b.face(
            &[[0., 0., 0.], [10., 0., 0.], [10., 10., 0.], [0., 10., 0.]],
            [0., 0., -1.],
        );
        b.face(
            &[
                [0., 10., 0.],
                [10., 10., 0.],
                [10., 10., 10.],
                [0., 10., 10.],
            ],
            [0., 1., 0.],
        );
        b.face(
            &[
                [0., 1., 10.],
                [10., 1., 10.],
                [10., 10., 10.],
                [0., 10., 10.],
            ],
            [0., 0., 1.],
        );
        b.face(
            &[[0., 0., 0.], [10., 0., 0.], [10., 0., 9.], [0., 0., 9.]],
            [0., -1., 0.],
        );
        b.face(
            &[[0., 0., 9.], [10., 0., 9.], [10., 1., 10.], [0., 1., 10.]],
            [0., -1., 1.],
        );
        for (x, out) in [(0.0, -1.0), (10.0, 1.0)] {
            let mut wall = vec![[x, 0., 0.], [x, 10., 0.], [x, 10., 10.], [x, 1., 10.]];
            if patched {
                wall.push([x, 0., 10.]);
                b.face(&[[x, 0., 9.], [x, 1., 10.], [x, 0., 10.]], [-out, 0., 0.]);
            }
            wall.push([x, 0., 9.]);
            b.face(&wall, [out, 0., 0.]);
        }
        let solid = b.solid();
        (b.topo, solid)
    }

    fn found(topo: &Topology, solid: SolidId) -> Vec<String> {
        check_faces_overlap(topo, solid, 1e-6)
            .unwrap()
            .into_iter()
            .map(|i| i.description)
            .collect()
    }

    #[test]
    fn a_patch_on_its_end_wall_is_reported() {
        let (topo, solid) = chamfered_cube(true);
        let got = found(&topo, solid);
        assert_eq!(got.len(), 2, "{got:?}");
        assert!(got.iter().all(|d| d.contains("cover the same area")));
    }

    #[test]
    fn a_chamfered_cube_passes() {
        let (topo, solid) = chamfered_cube(false);
        assert!(found(&topo, solid).is_empty());
    }

    /// A box's top split in two along x = 4, each half its own face.
    #[test]
    fn two_halves_of_a_face_pass() {
        let mut b = Builder::default();
        b.face(
            &[[0., 0., 10.], [4., 0., 10.], [4., 10., 10.], [0., 10., 10.]],
            [0., 0., 1.],
        );
        b.face(
            &[
                [4., 0., 10.],
                [10., 0., 10.],
                [10., 10., 10.],
                [4., 10., 10.],
            ],
            [0., 0., 1.],
        );
        let solid = b.solid();
        assert!(found(&b.topo, solid).is_empty());
    }

    /// Two faces on one plane, facing one way, that overlap in a strip at
    /// the side of each, and a face that lies inside another.
    #[test]
    fn faces_that_overlap_facing_one_way_are_reported() {
        let mut b = Builder::default();
        b.face(
            &[[0., 0., 0.], [6., 0., 0.], [6., 10., 0.], [0., 10., 0.]],
            [0., 0., 1.],
        );
        b.face(
            &[[5., 0., 0.], [10., 0., 0.], [10., 10., 0.], [5., 10., 0.]],
            [0., 0., 1.],
        );
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);

        let mut b = Builder::default();
        b.face(
            &[[0., 0., 0.], [10., 0., 0.], [10., 10., 0.], [0., 10., 0.]],
            [0., 0., 1.],
        );
        b.face(
            &[[2., 2., 0.], [3., 2., 0.], [3., 3., 0.], [2., 3., 0.]],
            [0., 0., -1.],
        );
        let solid = b.solid();
        assert_eq!(found(&b.topo, solid).len(), 1);
    }

    /// A square face with a round hole and the disc that fills it, on one
    /// plane, share the circle and nothing more. A disc a little off the
    /// hole's centre reaches across the hole's rim.
    #[test]
    fn a_disc_in_a_rings_hole_passes_and_one_across_its_rim_is_reported() {
        for (centre, overlaps) in [(5.0, false), (5.5, true)] {
            let mut b = Builder::default();
            let ring = b.face(
                &[[0., 0., 0.], [10., 0., 0.], [10., 10., 0.], [0., 10., 0.]],
                [0., 0., 1.],
            );
            let z = Vec3::new(0.0, 0.0, 1.0);
            let circle = |b: &mut Builder, c: f64| {
                let circle = Circle3D::new(Point3::new(c, 5.0, 0.0), z, 2.0).unwrap();
                let v = b.vertex([c + 2.0, 5.0, 0.0]);
                let e = b.topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
                (
                    e,
                    b.topo
                        .add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap()),
                )
            };
            let (_, hole) = circle(&mut b, 5.0);
            let (rim, _) = circle(&mut b, centre);
            b.topo.face_mut(ring).unwrap().inner_wires_mut().push(hole);
            let disc = b
                .topo
                .add_wire(Wire::new(vec![OrientedEdge::new(rim, false)], true).unwrap());
            let face = b.topo.add_face(Face::new(
                disc,
                vec![],
                FaceSurface::Plane { normal: z, d: 0.0 },
            ));
            b.faces.push(face);
            let solid = b.solid();
            assert_eq!(
                !found(&b.topo, solid).is_empty(),
                overlaps,
                "centre {centre}"
            );
        }
    }
}
