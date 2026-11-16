//! `validate_solid` reports two flat faces of one solid that cover the same
//! area. A round that ends on an end wall it does not notch leaves the wall
//! whole and closes its end with a flat patch on the wall, facing into the
//! solid. Each edge of such a solid has two uses and the faces agree across
//! each edge, so only the faces' areas show the fault. The solid here is a
//! cube chamfered along one edge, its end walls built that way.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
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
    fn face(&mut self, corners: &[[f64; 3]], normal: [f64; 3]) {
        let normal = Vec3::new(normal[0], normal[1], normal[2])
            .normalize()
            .unwrap();
        let at = |c: [f64; 3]| Vec3::new(c[0], c[1], c[2]);
        let mut turn = Vec3::new(0.0, 0.0, 0.0);
        for k in 0..corners.len() {
            turn += at(corners[k]).cross(at(corners[(k + 1) % corners.len()]));
        }
        let mut ids: Vec<VertexId> = corners.iter().map(|&c| self.vertex(c)).collect();
        if turn.dot(normal) < 0.0 {
            ids.reverse();
        }
        let edges = (0..ids.len())
            .map(|k| self.edge(ids[k], ids[(k + 1) % ids.len()]))
            .collect();
        let wire = self.topo.add_wire(Wire::new(edges, true).unwrap());
        let face = self.topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal,
                d: normal.dot(at(corners[0])),
            },
        ));
        self.faces.push(face);
    }
}

/// A 10 cube less a chamfer of 1 along its edge on y = 0 and z = 10. With
/// `patched`, each end wall keeps its whole square, and a triangle on it,
/// facing into the cube, closes the chamfer's end.
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
    let shell = b.topo.add_shell(Shell::new(b.faces.clone()).unwrap());
    let solid = b.topo.add_solid(Solid::new(shell, vec![]));
    (b.topo, solid)
}

/// The patched cube passes every other check: its only errors are the two
/// pairs of faces that cover the same area. It holds the chamfered cube's
/// volume, as each patch takes its triangle off its wall again.
#[test]
fn a_patch_on_its_end_wall_is_reported() {
    let (topo, solid) = chamfered_cube(true);
    let report = validate_solid(&topo, solid).unwrap();
    let errors: Vec<&str> = report
        .issues
        .iter()
        .filter(|i| i.severity == brepkit_operations::validate::Severity::Error)
        .map(|i| i.description.as_str())
        .collect();
    assert_eq!(errors.len(), 2, "{errors:?}");
    assert!(
        errors
            .iter()
            .all(|d| d.contains("lie on one plane and cover the same area")),
        "{errors:?}"
    );
    let volume = solid_volume(&topo, solid, 0.01).unwrap();
    assert!((volume - 995.0).abs() < 1e-6, "{volume}");
}

#[test]
fn the_chamfered_cube_is_valid() {
    let (topo, solid) = chamfered_cube(false);
    let report = validate_solid(&topo, solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
}
