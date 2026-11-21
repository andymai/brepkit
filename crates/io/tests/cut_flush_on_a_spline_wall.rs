//! A part extruded from an outline with a B-spline side, cut by a tool
//! whose own side runs along a stretch of that spline: the tool's side lies
//! on the part's B-spline wall, and the cut opens a hole in the wall.
//!
//! The two walls are different B-spline surfaces lying on each other, so
//! they have no section curve: marched, they gave scraps wherever the solve
//! happened to land, and the wall was never split along the tool's outline.
//! The cut fell back to the mesh, or kept the wall whole and the tool's
//! side with it, the two facing each other (a valid solid whose own volume
//! was right, but whose wall covered the opening). The planes meeting the
//! wall along the tool's outline marched sections stopping short of its
//! corners or running past them. The tool's side may also come in pieces
//! (a spline written in two), whose shared edge then runs across the hole.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::knot_ops::curve_split;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);

/// The part's spline side, in the plane `z`: a cubic from (0, 0) dipping
/// to y = -22.5 and back up to (60, 0).
fn spline(z: f64) -> NurbsCurve {
    NurbsCurve::new(
        3,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [(0.0, 0.0), (10.0, -30.0), (50.0, -30.0), (60.0, 0.0)]
            .iter()
            .map(|&(x, y)| Point3::new(x, y, z))
            .collect(),
        vec![1.0; 4],
    )
    .unwrap()
}

/// A prism of the profile bounded by `pieces` (B-spline edges end to end)
/// and the line back from the last one's end to the first one's start,
/// `height` up from the plane `z`.
fn prism(topo: &mut Topology, pieces: &[NurbsCurve], z: f64, height: f64) -> SolidId {
    let ends: Vec<Point3> = std::iter::once(pieces[0].evaluate(pieces[0].domain().0))
        .chain(pieces.iter().map(|c| c.evaluate(c.domain().1)))
        .collect();
    let vertices: Vec<_> = ends
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let mut edges: Vec<OrientedEdge> = pieces
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let e = topo.add_edge(Edge::new(
                vertices[i],
                vertices[i + 1],
                EdgeCurve::NurbsCurve(c.clone()),
            ));
            OrientedEdge::new(e, true)
        })
        .collect();
    let back = topo.add_edge(Edge::new(
        vertices[pieces.len()],
        vertices[0],
        EdgeCurve::Line,
    ));
    edges.push(OrientedEdge::new(back, true));
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    // The spline dips below its chord, so the loop runs clockwise seen from
    // above: the face looks down.
    let face = topo.add_face(Face::new(
        wire,
        Vec::new(),
        FaceSurface::Plane { normal: -UP, d: -z },
    ));
    extrude(topo, face, UP, height).unwrap()
}

/// The spline's stretch from `a` to `b` (fractions of its domain), in
/// `pieces` pieces.
fn stretch(z: f64, a: f64, b: f64, pieces: usize) -> Vec<NurbsCurve> {
    let (_, rest) = curve_split(&spline(z), a).unwrap();
    let (mut rest, _) = curve_split(&rest, b).unwrap();
    let mut out = Vec::new();
    for k in (1..pieces).rev() {
        let (d0, d1) = rest.domain();
        #[allow(clippy::cast_precision_loss)]
        let t = d0 + (d1 - d0) * (1.0 / (k + 1) as f64);
        let (first, second) = curve_split(&rest, t).unwrap();
        out.push(first);
        rest = second;
    }
    out.push(rest);
    out
}

#[test]
fn a_tool_flush_on_a_spline_wall_opens_it() {
    for (turn, pieces) in [(0.0, 1), (0.7, 1), (0.0, 2), (0.7, 2), (2.5, 3)] {
        let tag = format!("turn {turn}, the tool's side in {pieces}");
        let mut topo = Topology::new();
        let part = prism(&mut topo, &[spline(0.0)], 0.0, 40.0);
        let tool = prism(&mut topo, &stretch(10.0, 0.3, 0.7, pieces), 10.0, 20.0);
        let turned = Mat4::rotation_z(turn);
        transform_solid(&mut topo, part, &turned).unwrap();
        transform_solid(&mut topo, tool, &turned).unwrap();
        let want =
            solid_volume(&topo, part, 1e-3).unwrap() - solid_volume(&topo, tool, 1e-3).unwrap();
        let before = mesh_fallback_count();
        let cut = boolean(&mut topo, BooleanOp::Cut, part, tool).unwrap();
        assert_eq!(mesh_fallback_count(), before, "{tag}: mesh fallback");
        let report = validate_solid(&topo, cut).unwrap();
        assert!(report.is_valid(), "{tag}: {:?}", report.issues);
        let volume = solid_volume(&topo, cut, 1e-3).unwrap();
        assert!(
            (volume - want).abs() < 1e-6 * want,
            "{tag}: volume {volume}, want {want}"
        );
        // The part's four faces, the wall holed, and the tool's chord face
        // and two caps.
        let faces = solid_faces(&topo, cut).unwrap();
        assert_eq!(faces.len(), 7, "{tag}: faces");
        let holed = faces
            .iter()
            .filter(|&&f| {
                let face = topo.face(f).unwrap();
                matches!(face.surface(), FaceSurface::Nurbs(_)) && face.inner_wires().len() == 1
            })
            .count();
        assert_eq!(holed, 1, "{tag}: the wall holed");
        let mesh = tessellate_solid(&topo, cut, 0.05).unwrap();
        assert!(is_watertight(&mesh), "{tag}: open mesh");
    }
}
