//! Topology sewing: merge loose faces into connected shells.
//!
//! Takes a set of independent faces and merges coincident edges/vertices
//! to create a topologically connected shell.

use std::collections::HashMap;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

/// Sew a set of loose faces into a solid.
///
/// Finds geometrically coincident edges between faces (within
/// `tolerance`) and merges them, creating shared topology. Then
/// assembles the sewn faces into a shell and solid.
///
/// # Algorithm
///
/// 1. Merge coincident vertices using spatial hashing
/// 2. Merge coincident edges: the same two vertices, and each edge's points
///    on the other, so the two halves of a circle, or two closed edges
///    through one vertex, stay apart
/// 3. Rebuild each face, holes included, from the merged edges, each keeping
///    its own curve
/// 4. Assemble into a shell and solid
///
/// # Errors
///
/// Returns an error if fewer than 2 faces are provided or a topology lookup
/// fails.
pub fn sew_faces(
    topo: &mut Topology,
    faces: &[FaceId],
    tolerance: f64,
) -> Result<SolidId, crate::OperationsError> {
    if faces.len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "sewing requires at least 2 faces".into(),
        });
    }

    let tol = if tolerance > 0.0 {
        tolerance
    } else {
        Tolerance::new().linear
    };

    let mut face_snapshots: Vec<FaceSnapshot> = Vec::with_capacity(faces.len());
    for &fid in faces {
        let face = topo.face(fid)?;
        let mut wires = Vec::with_capacity(1 + face.inner_wires().len());
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let mut edges = Vec::new();
            for oe in topo.wire(wid)?.edges() {
                let edge = topo.edge(oe.edge())?;
                let start = topo.vertex(edge.start())?.point();
                let end = topo.vertex(edge.end())?.point();
                edges.push(EdgeSnapshot {
                    span: Span::new(edge.curve().clone(), start, end),
                    forward: oe.is_forward(),
                    source: oe.edge().index(),
                });
            }
            wires.push(edges);
        }
        face_snapshots.push(FaceSnapshot {
            surface: face.surface().clone(),
            reversed: face.is_reversed(),
            wires,
        });
    }

    let resolution = 1.0 / tol;
    let mut vertex_map: HashMap<(i64, i64, i64), VertexId> = HashMap::new();
    let mut vertex_at = |topo: &mut Topology, p: Point3| {
        #[allow(clippy::cast_possible_truncation)]
        let key = (
            (p.x() * resolution).round() as i64,
            (p.y() * resolution).round() as i64,
            (p.z() * resolution).round() as i64,
        );
        *vertex_map
            .entry(key)
            .or_insert_with(|| topo.add_vertex(Vertex::new(p, tol)))
    };

    // The edges made so far between each pair of vertices, and the edge
    // each input edge became, with whether it runs the same way.
    let mut edge_map: HashMap<(usize, usize), Vec<MadeEdge>> = HashMap::new();
    let mut by_source: HashMap<usize, (EdgeId, bool)> = HashMap::new();
    let mut new_face_ids: Vec<FaceId> = Vec::with_capacity(face_snapshots.len());

    for snap in &face_snapshots {
        let mut wire_ids = Vec::with_capacity(snap.wires.len());
        for edges in &snap.wires {
            let mut oriented_edges: Vec<OrientedEdge> = Vec::with_capacity(edges.len());
            for e in edges {
                let v_start = vertex_at(topo, e.span.start);
                let v_end = vertex_at(topo, e.span.end);
                let key = if v_start.index() <= v_end.index() {
                    (v_start.index(), v_end.index())
                } else {
                    (v_end.index(), v_start.index())
                };
                let made = edge_map.entry(key).or_default();
                let (edge_id, along) = if let Some(&shared) = by_source.get(&e.source) {
                    shared
                } else if let Some(m) = made
                    .iter()
                    .find(|m| e.span.lies_on(&m.span, tol) && m.span.lies_on(&e.span, tol))
                {
                    let along = if v_start == v_end {
                        e.span.runs_with(&m.span)
                    } else {
                        m.start == v_start
                    };
                    (m.edge, along)
                } else {
                    let edge = topo.add_edge(Edge::new(v_start, v_end, e.span.curve.clone()));
                    made.push(MadeEdge {
                        span: e.span.clone(),
                        edge,
                        start: v_start,
                    });
                    (edge, true)
                };
                by_source.insert(e.source, (edge_id, along));
                // An edge met again the other way along runs against its
                // first face's traversal in this one.
                let forward = e.forward == along;
                oriented_edges.push(OrientedEdge::new(edge_id, forward));
            }
            let wire = Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
            wire_ids.push(topo.add_wire(wire));
        }
        let (outer, holes) = wire_ids
            .split_first()
            .map(|(o, h)| (*o, h.to_vec()))
            .ok_or_else(|| crate::OperationsError::InvalidInput {
                reason: "a face to sew has no boundary".into(),
            })?;
        let mut face = Face::new(outer, holes, snap.surface.clone());
        if snap.reversed {
            face.set_reversed(true);
        }
        new_face_ids.push(topo.add_face(face));
    }

    let shell = Shell::new(new_face_ids).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

/// An edge sewing has made: its curve, and the vertex it starts at.
struct MadeEdge {
    span: Span,
    edge: EdgeId,
    start: VertexId,
}

/// One edge of a face to sew, as its wire runs it.
struct EdgeSnapshot {
    span: Span,
    forward: bool,
    /// The input edge's index.
    source: usize,
}

/// An edge's curve from its start to its end, read by the fraction of the
/// way along its parameter range, so a range of any scale reads alike.
#[derive(Clone)]
struct Span {
    curve: EdgeCurve,
    start: Point3,
    end: Point3,
    t0: f64,
    t1: f64,
    /// The fractions where the curve's pieces meet, 0 and 1 among them: a
    /// NURBS curve's knots, eighths of a turn round a circle or an ellipse.
    breaks: Vec<f64>,
    /// How many points inside each piece decide whether it lies on another
    /// curve.
    per_piece: usize,
}

impl Span {
    fn new(curve: EdgeCurve, start: Point3, end: Point3) -> Self {
        let (t0, t1) = curve.domain_with_endpoints(start, end);
        let (mut breaks, per_piece) = match &curve {
            EdgeCurve::Line => (vec![0.0, 1.0], 3),
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let pieces = ((t1 - t0).abs() / std::f64::consts::FRAC_PI_4)
                    .ceil()
                    .max(1.0) as u32;
                (
                    (0..=pieces)
                        .map(|k| f64::from(k) / f64::from(pieces))
                        .collect(),
                    4,
                )
            }
            EdgeCurve::NurbsCurve(nurbs) => (
                nurbs
                    .knots()
                    .iter()
                    .map(|&k| (k - t0) / (t1 - t0))
                    .filter(|f| *f > 0.0 && *f < 1.0)
                    .chain([0.0, 1.0])
                    .collect(),
                nurbs.degree() + 3,
            ),
        };
        breaks.sort_by(f64::total_cmp);
        breaks.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);
        Self {
            curve,
            start,
            end,
            t0,
            t1,
            breaks,
            per_piece,
        }
    }

    /// The point a fraction `f` of the way along the parameter range.
    fn at(&self, f: f64) -> Point3 {
        self.curve.evaluate_with_endpoints(
            f.mul_add(self.t1 - self.t0, self.t0),
            self.start,
            self.end,
        )
    }

    /// The fractions inside each piece that decide whether it lies on
    /// another curve.
    fn samples(&self) -> impl Iterator<Item = f64> + '_ {
        #[allow(clippy::cast_precision_loss)]
        let n = (self.per_piece + 1) as f64;
        self.breaks.windows(2).flat_map(move |w| {
            #[allow(clippy::cast_precision_loss)]
            (1..=self.per_piece).map(move |k| (w[1] - w[0]).mul_add(k as f64 / n, w[0]))
        })
    }

    /// The fraction along this span nearest `p`, and its distance. Each
    /// piece is read at eight points, and every least distance among them
    /// is closed in on by golden section, so a curve folding back near
    /// itself is not read on the wrong fold.
    fn nearest(&self, p: Point3) -> (f64, f64) {
        let dist = |f: f64| (self.at(f) - p).length();
        let mut fs: Vec<f64> = self
            .breaks
            .windows(2)
            .flat_map(|w| (0..8).map(move |k| (w[1] - w[0]).mul_add(f64::from(k) / 8.0, w[0])))
            .collect();
        fs.push(1.0);
        let d: Vec<f64> = fs.iter().map(|&f| dist(f)).collect();
        let golden = (5.0_f64.sqrt() - 1.0) / 2.0;
        let mut best = (0.0, f64::INFINITY);
        for i in 0..fs.len() {
            let left = if i > 0 { d[i - 1] } else { f64::INFINITY };
            let right = d.get(i + 1).copied().unwrap_or(f64::INFINITY);
            if d[i] > left || d[i] > right {
                continue;
            }
            let (mut a, mut b) = (fs[i.saturating_sub(1)], fs[(i + 1).min(fs.len() - 1)]);
            let (mut c, mut e) = (b - golden * (b - a), golden.mul_add(b - a, a));
            let (mut dc, mut de) = (dist(c), dist(e));
            for _ in 0..80 {
                if dc <= de {
                    (b, e, de) = (e, c, dc);
                    c = b - golden * (b - a);
                    dc = dist(c);
                } else {
                    (a, c, dc) = (c, e, de);
                    e = golden.mul_add(b - a, a);
                    de = dist(e);
                }
            }
            for found in [(fs[i], d[i]), (c, dc), (e, de)] {
                if found.1 < best.1 {
                    best = found;
                }
            }
        }
        best
    }

    /// Whether points through every piece of this span lie on `other`,
    /// however either is parameterized.
    fn lies_on(&self, other: &Self, tol: f64) -> bool {
        self.samples().all(|f| other.nearest(self.at(f)).1 <= tol)
    }

    /// The way the span runs at fraction `f`.
    fn heading(&self, f: f64) -> Vec3 {
        let h = 1e-6;
        self.at((f + h).min(1.0)) - self.at((f - h).max(0.0))
    }

    /// Whether this span runs the same way along as `other`, which it lies
    /// on. A closed edge's ends cannot tell.
    fn runs_with(&self, other: &Self) -> bool {
        let there = other.nearest(self.at(0.5)).0;
        self.heading(0.5).dot(other.heading(there)) > 0.0
    }
}

/// Snapshot of a face's geometry for the sewing algorithm.
struct FaceSnapshot {
    surface: FaceSurface,
    reversed: bool,
    /// The outer wire first, then the holes.
    wires: Vec<Vec<EdgeSnapshot>>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    /// Helper: create a standalone quad face with independent vertices/edges.
    fn make_loose_quad(
        topo: &mut Topology,
        p0: Point3,
        p1: Point3,
        p2: Point3,
        p3: Point3,
        normal: Vec3,
        d: f64,
    ) -> FaceId {
        let tol_val = 1e-7;
        let v0 = topo.add_vertex(Vertex::new(p0, tol_val));
        let v1 = topo.add_vertex(Vertex::new(p1, tol_val));
        let v2 = topo.add_vertex(Vertex::new(p2, tol_val));
        let v3 = topo.add_vertex(Vertex::new(p3, tol_val));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
                OrientedEdge::new(e3, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        topo.add_face(Face::new(wid, vec![], FaceSurface::Plane { normal, d }))
    }

    #[test]
    fn sew_two_adjacent_quads() {
        let mut topo = Topology::new();

        // Two quads sharing an edge along x=1.
        let f0 = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );
        let f1 = make_loose_quad(
            &mut topo,
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );

        let solid = sew_faces(&mut topo, &[f0, f1], 1e-6).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(sh.faces().len(), 2, "sewn result should have 2 faces");
    }

    #[test]
    fn sew_shares_coincident_edges() {
        let mut topo = Topology::new();

        let f0 = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );
        let f1 = make_loose_quad(
            &mut topo,
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );

        let solid = sew_faces(&mut topo, &[f0, f1], 1e-6).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        let mut edge_set = std::collections::HashSet::new();
        for &fid in sh.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                edge_set.insert(oe.edge().index());
            }
        }

        // 2 quads with 4 edges each, sharing 1 edge = 7 unique edges.
        assert_eq!(
            edge_set.len(),
            7,
            "two adjacent quads should share 1 edge (7 unique), got {}",
            edge_set.len()
        );
    }

    #[test]
    fn sew_six_cube_faces() {
        let mut topo = Topology::new();

        let bottom = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            0.0,
        );
        let top = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
        );
        let front = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, -1.0, 0.0),
            0.0,
        );
        let back = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            1.0,
        );
        let left = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(0.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 0.0),
            0.0,
        );
        let right = make_loose_quad(
            &mut topo,
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
        );

        let solid = sew_faces(&mut topo, &[bottom, top, front, back, left, right], 1e-6).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(sh.faces().len(), 6, "sewn cube should have 6 faces");

        // Count unique edges — a cube has 12.
        let mut edge_set = std::collections::HashSet::new();
        for &fid in sh.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                edge_set.insert(oe.edge().index());
            }
        }
        assert_eq!(
            edge_set.len(),
            12,
            "cube should have 12 edges, got {}",
            edge_set.len()
        );
    }

    #[test]
    fn sew_single_face_error() {
        let mut topo = Topology::new();
        let f = make_loose_quad(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );
        assert!(sew_faces(&mut topo, &[f], 1e-6).is_err());
    }

    /// The kinds of curve a solid's edges carry, in face order.
    fn curve_kinds(topo: &Topology, solid: SolidId) -> Vec<Vec<&'static str>> {
        brepkit_topology::explorer::solid_faces(topo, solid)
            .unwrap()
            .iter()
            .map(|&f| {
                let face = topo.face(f).unwrap();
                std::iter::once(face.outer_wire())
                    .chain(face.inner_wires().iter().copied())
                    .flat_map(|w| topo.wire(w).unwrap().edges().to_vec())
                    .map(|oe| topo.edge(oe.edge()).unwrap().curve().type_tag())
                    .collect()
            })
            .collect()
    }

    /// A cylinder's three faces sew back into the cylinder: its rims stay
    /// circles and it holds π r² h.
    #[test]
    fn sew_a_cylinders_faces_keeps_its_circles() {
        let mut topo = Topology::new();
        let cyl = crate::primitives::make_cylinder(&mut topo, 10.0, 20.0).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&topo, cyl).unwrap();
        let sewn = sew_faces(&mut topo, &faces, 1e-7).unwrap();
        assert_eq!(curve_kinds(&topo, sewn), curve_kinds(&topo, cyl));
        assert!(
            crate::validate::validate_solid(&topo, sewn)
                .unwrap()
                .is_valid()
        );
        let volume = crate::measure::solid_volume(&topo, sewn, 0.01).unwrap();
        let exact = std::f64::consts::PI * 100.0 * 20.0;
        assert!((volume - exact).abs() < 1e-3 * exact, "{volume}");
    }

    /// A quarter-circle sector sewn beside a far triangle keeps its arc and
    /// its area of 25π/4.
    #[test]
    fn sew_keeps_a_sectors_arc() {
        use brepkit_math::curves::Circle3D;

        let mut topo = Topology::new();
        let v = |topo: &mut Topology, x: f64, y: f64| {
            topo.add_vertex(Vertex::new(Point3::new(x, y, 20.0), 1e-7))
        };
        let (c, a, b) = (
            v(&mut topo, 20.0, 15.0),
            v(&mut topo, 20.0, 20.0),
            v(&mut topo, 25.0, 15.0),
        );
        let circle = Circle3D::new(
            Point3::new(20.0, 15.0, 20.0),
            Vec3::new(0.0, 0.0, -1.0),
            5.0,
        )
        .unwrap();
        let edges = [
            topo.add_edge(Edge::new(c, a, EdgeCurve::Line)),
            topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle))),
            topo.add_edge(Edge::new(b, c, EdgeCurve::Line)),
        ];
        let wire = topo.add_wire(
            Wire::new(
                edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
                true,
            )
            .unwrap(),
        );
        let sector = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: -20.0,
            },
        ));
        let far = make_loose_quad(
            &mut topo,
            Point3::new(100.0, 0.0, 0.0),
            Point3::new(110.0, 0.0, 0.0),
            Point3::new(110.0, 10.0, 0.0),
            Point3::new(100.0, 10.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        );
        let sewn = sew_faces(&mut topo, &[sector, far], 1e-7).unwrap();
        let first = brepkit_topology::explorer::solid_faces(&topo, sewn).unwrap()[0];
        let area = crate::measure::face_area(&topo, first, 0.001).unwrap();
        let exact = 25.0 * std::f64::consts::FRAC_PI_4;
        assert!((area - exact).abs() < 1e-3 * exact, "{area}");
        assert!(curve_kinds(&topo, sewn)[0].contains(&"circle"));
    }

    /// A disc bounded by two half circles between the same two points keeps
    /// both: its area stays 25π.
    #[test]
    fn sew_keeps_two_arcs_between_one_pair_of_points() {
        use brepkit_math::curves::Circle3D;

        let mut topo = Topology::new();
        let a = topo.add_vertex(Vertex::new(Point3::new(5.0, 0.0, 0.0), 1e-7));
        let b = topo.add_vertex(Vertex::new(Point3::new(-5.0, 0.0, 0.0), 1e-7));
        let up = Vec3::new(0.0, 0.0, 1.0);
        let circle =
            || EdgeCurve::Circle(Circle3D::new(Point3::new(0.0, 0.0, 0.0), up, 5.0).unwrap());
        let edges = [
            topo.add_edge(Edge::new(a, b, circle())),
            topo.add_edge(Edge::new(b, a, circle())),
        ];
        let wire = topo.add_wire(
            Wire::new(
                edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
                true,
            )
            .unwrap(),
        );
        let disc = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane { normal: up, d: 0.0 },
        ));
        let far = make_loose_quad(
            &mut topo,
            Point3::new(100.0, 0.0, 0.0),
            Point3::new(110.0, 0.0, 0.0),
            Point3::new(110.0, 10.0, 0.0),
            Point3::new(100.0, 10.0, 0.0),
            up,
            0.0,
        );
        let sewn = sew_faces(&mut topo, &[disc, far], 1e-7).unwrap();
        let first = brepkit_topology::explorer::solid_faces(&topo, sewn).unwrap()[0];
        let area = crate::measure::face_area(&topo, first, 0.001).unwrap();
        let exact = 25.0 * std::f64::consts::PI;
        assert!((area - exact).abs() < 1e-3 * exact, "{area}");
    }

    /// A span of a degree-1 curve through `points` in the plane z = 0 over
    /// `knots`, from its first point to its last.
    fn polyline_span(knots: Vec<f64>, points: &[(f64, f64)]) -> Span {
        use brepkit_math::nurbs::curve::NurbsCurve;
        let pts: Vec<Point3> = points
            .iter()
            .map(|&(x, y)| Point3::new(x, y, 0.0))
            .collect();
        let (start, end) = (pts[0], pts[pts.len() - 1]);
        let curve = NurbsCurve::new(1, knots, pts.clone(), vec![1.0; pts.len()]).unwrap();
        Span::new(EdgeCurve::NurbsCurve(curve), start, end)
    }

    /// A unit segment whose knots run to 1e9 lies on its copy.
    #[test]
    fn a_span_lies_on_its_copy_however_far_its_knots_run() {
        let span = polyline_span(vec![0.0, 0.0, 1e9, 1e9], &[(0.0, 0.0), (1.0, 0.0)]);
        assert!(span.lies_on(&span.clone(), 1e-7));
    }

    /// A curve folding back 0.001 above itself lies on its copy: a point on
    /// its second leg is read on that leg, not on the first.
    #[test]
    fn a_folded_span_lies_on_its_copy() {
        let span = polyline_span(
            vec![0.0, 0.0, 0.4, 0.5, 1.0, 1.0],
            &[(0.0, 0.0), (1.0, 0.0), (1.0, 0.001), (0.0, 0.001)],
        );
        assert!(span.lies_on(&span.clone(), 1e-7));
    }

    /// A curve with a bump 0.1 high inside its first knot span, from 0 to
    /// 0.1, does not lie on the unit segment. The segment's points all lie
    /// on the curve's long straight span, so the bump is what keeps the two
    /// apart, merging needing each on the other.
    #[test]
    fn a_bump_in_a_short_knot_span_keeps_curves_apart() {
        let bumped = polyline_span(
            vec![0.0, 0.0, 0.05, 0.1, 1.0, 1.0],
            &[(0.0, 0.0), (0.05, 0.1), (0.1, 0.0), (1.0, 0.0)],
        );
        let straight = Span::new(
            EdgeCurve::Line,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        );
        assert!(!bumped.lies_on(&straight, 1e-7));
        assert!(straight.lies_on(&bumped, 1e-7));
    }

    /// The cylinder's faces with one cap's rim swapped for `curve`, used
    /// forward or not.
    fn cylinder_with_rim(
        topo: &mut Topology,
        top: bool,
        curve: brepkit_math::curves::Circle3D,
        forward: bool,
    ) -> Vec<FaceId> {
        let cyl = crate::primitives::make_cylinder(topo, 10.0, 20.0).unwrap();
        let mut faces = brepkit_topology::explorer::solid_faces(topo, cyl).unwrap();
        let z = if top { 20.0 } else { 0.0 };
        let at = faces
            .iter()
            .position(|&f| {
                let face = topo.face(f).unwrap();
                face.surface().is_planar() && {
                    let wire = topo.wire(face.outer_wire()).unwrap();
                    let edge = topo.edge(wire.edges()[0].edge()).unwrap();
                    (topo.vertex(edge.start()).unwrap().point().z() - z).abs() < 1e-9
                }
            })
            .unwrap();
        let (surface, reversed, vertex) = {
            let face = topo.face(faces[at]).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            let edge = topo.edge(wire.edges()[0].edge()).unwrap();
            (face.surface().clone(), face.is_reversed(), edge.start())
        };
        let edge = topo.add_edge(Edge::new(vertex, vertex, EdgeCurve::Circle(curve)));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, forward)], true).unwrap());
        let mut cap = Face::new(wire, vec![], surface);
        cap.set_reversed(reversed);
        faces[at] = topo.add_face(cap);
        faces
    }

    fn assert_whole_cylinder(topo: &Topology, sewn: SolidId) {
        let edges: std::collections::HashSet<usize> =
            brepkit_topology::explorer::solid_faces(topo, sewn)
                .unwrap()
                .iter()
                .flat_map(|&f| {
                    let face = topo.face(f).unwrap();
                    topo.wire(face.outer_wire()).unwrap().edges().to_vec()
                })
                .map(|oe| oe.edge().index())
                .collect();
        assert_eq!(edges.len(), 3);
        assert!(
            crate::validate::validate_solid(topo, sewn)
                .unwrap()
                .is_valid()
        );
        let volume = crate::measure::solid_volume(topo, sewn, 0.01).unwrap();
        let exact = std::f64::consts::PI * 100.0 * 20.0;
        assert!((volume - exact).abs() < 1e-3 * exact, "{volume}");
    }

    /// A bottom cap whose rim is the circle about -z, run forward, meets
    /// the wall's circle about +z, from the same start, the other way along.
    #[test]
    fn sew_turns_a_closed_rim_running_the_other_way() {
        use brepkit_math::curves::Circle3D;

        let mut topo = Topology::new();
        let rim = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            10.0,
            Vec3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        let faces = cylinder_with_rim(&mut topo, false, rim, true);
        let sewn = sew_faces(&mut topo, &faces, 1e-7).unwrap();
        assert_whole_cylinder(&topo, sewn);
    }

    /// A top cap whose rim circle starts a quarter turn round from the
    /// wall's still shares the wall's edge.
    #[test]
    fn sew_shares_a_closed_rim_whose_circle_starts_elsewhere() {
        use brepkit_math::curves::Circle3D;

        let mut topo = Topology::new();
        let rim = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 20.0),
            Vec3::new(0.0, 0.0, 1.0),
            10.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let faces = cylinder_with_rim(&mut topo, true, rim, true);
        let sewn = sew_faces(&mut topo, &faces, 1e-7).unwrap();
        assert_whole_cylinder(&topo, sewn);
    }

    /// Two discs on perpendicular planes, both closed at (1, 0, 0) with the
    /// same middle point, keep their own circles.
    #[test]
    fn sew_keeps_perpendicular_circles_through_one_point_apart() {
        use brepkit_math::curves::Circle3D;

        let mut topo = Topology::new();
        let x = Vec3::new(1.0, 0.0, 0.0);
        let disc = |topo: &mut Topology, normal: Vec3| {
            let v = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
            let circle =
                Circle3D::new_with_ref(Point3::new(0.0, 0.0, 0.0), normal, 1.0, x).unwrap();
            let edge = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
            let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
            topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane { normal, d: 0.0 },
            ))
        };
        let flat = disc(&mut topo, Vec3::new(0.0, 0.0, 1.0));
        let upright = disc(&mut topo, Vec3::new(0.0, 1.0, 0.0));
        let sewn = sew_faces(&mut topo, &[flat, upright], 1e-7).unwrap();
        let rims: Vec<_> = brepkit_topology::explorer::solid_faces(&topo, sewn)
            .unwrap()
            .iter()
            .map(|&f| {
                let wire = topo.wire(topo.face(f).unwrap().outer_wire()).unwrap();
                wire.edges()[0].edge()
            })
            .collect();
        assert_ne!(rims[0], rims[1]);
        let EdgeCurve::Circle(upright_rim) = topo.edge(rims[1]).unwrap().curve() else {
            unreachable!("the rim is a circle")
        };
        assert!(upright_rim.evaluate(1.0).y().abs() < 1e-12);
    }

    /// A cube whose bottom and front meet along one segment carried by two
    /// differently parameterized curves sews into the closed cube.
    #[test]
    fn sew_shares_a_segment_however_its_curves_run() {
        use brepkit_math::nurbs::curve::NurbsCurve;

        let mut topo = Topology::new();
        let p = Point3::new;
        let segment = |xs: [f64; 3]| {
            EdgeCurve::NurbsCurve(
                NurbsCurve::new(
                    2,
                    vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    xs.iter().map(|&x| p(x, 0.0, 0.0)).collect(),
                    vec![1.0; 3],
                )
                .unwrap(),
            )
        };
        let quad = |topo: &mut Topology,
                    corners: [Point3; 4],
                    normal: Vec3,
                    d: f64,
                    first: Option<(EdgeCurve, bool)>| {
            let v: Vec<_> = corners
                .iter()
                .map(|&c| topo.add_vertex(Vertex::new(c, 1e-7)))
                .collect();
            let mut edges = Vec::new();
            for i in 0..4 {
                let (a, b) = (v[i], v[(i + 1) % 4]);
                let oriented = match (&first, i) {
                    (Some((curve, true)), 0) => {
                        OrientedEdge::new(topo.add_edge(Edge::new(a, b, curve.clone())), true)
                    }
                    (Some((curve, false)), 0) => {
                        OrientedEdge::new(topo.add_edge(Edge::new(b, a, curve.clone())), false)
                    }
                    _ => OrientedEdge::new(topo.add_edge(Edge::new(a, b, EdgeCurve::Line)), true),
                };
                edges.push(oriented);
            }
            let wire = topo.add_wire(Wire::new(edges, true).unwrap());
            topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d }))
        };
        let faces = [
            quad(
                &mut topo,
                [
                    p(0.0, 0.0, 0.0),
                    p(1.0, 0.0, 0.0),
                    p(1.0, 1.0, 0.0),
                    p(0.0, 1.0, 0.0),
                ],
                Vec3::new(0.0, 0.0, -1.0),
                0.0,
                Some((segment([0.0, 0.2, 1.0]), true)),
            ),
            quad(
                &mut topo,
                [
                    p(0.0, 0.0, 1.0),
                    p(1.0, 0.0, 1.0),
                    p(1.0, 1.0, 1.0),
                    p(0.0, 1.0, 1.0),
                ],
                Vec3::new(0.0, 0.0, 1.0),
                1.0,
                None,
            ),
            quad(
                &mut topo,
                [
                    p(0.0, 0.0, 0.0),
                    p(1.0, 0.0, 0.0),
                    p(1.0, 0.0, 1.0),
                    p(0.0, 0.0, 1.0),
                ],
                Vec3::new(0.0, -1.0, 0.0),
                0.0,
                Some((segment([1.0, 0.8, 0.0]), false)),
            ),
            quad(
                &mut topo,
                [
                    p(0.0, 1.0, 0.0),
                    p(1.0, 1.0, 0.0),
                    p(1.0, 1.0, 1.0),
                    p(0.0, 1.0, 1.0),
                ],
                Vec3::new(0.0, 1.0, 0.0),
                1.0,
                None,
            ),
            quad(
                &mut topo,
                [
                    p(0.0, 0.0, 0.0),
                    p(0.0, 1.0, 0.0),
                    p(0.0, 1.0, 1.0),
                    p(0.0, 0.0, 1.0),
                ],
                Vec3::new(-1.0, 0.0, 0.0),
                0.0,
                None,
            ),
            quad(
                &mut topo,
                [
                    p(1.0, 0.0, 0.0),
                    p(1.0, 1.0, 0.0),
                    p(1.0, 1.0, 1.0),
                    p(1.0, 0.0, 1.0),
                ],
                Vec3::new(1.0, 0.0, 0.0),
                1.0,
                None,
            ),
        ];
        let sewn = sew_faces(&mut topo, &faces, 1e-7).unwrap();
        let edges: std::collections::HashSet<usize> =
            brepkit_topology::explorer::solid_faces(&topo, sewn)
                .unwrap()
                .iter()
                .flat_map(|&f| {
                    let face = topo.face(f).unwrap();
                    topo.wire(face.outer_wire()).unwrap().edges().to_vec()
                })
                .map(|oe| oe.edge().index())
                .collect();
        assert_eq!(edges.len(), 12);
    }

    /// A block with a round hole through it, its faces sewn again, keeps
    /// the hole: the caps keep their inner circles and the solid its volume.
    #[test]
    fn sew_keeps_a_faces_holes() {
        use crate::boolean::{BooleanOp, boolean};

        let mut topo = Topology::new();
        let block = crate::primitives::make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
        let rod = crate::primitives::make_cylinder(&mut topo, 4.0, 30.0).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            rod,
            &brepkit_math::mat::Mat4::translation(10.0, 10.0, -10.0),
        )
        .unwrap();
        let holed = boolean(&mut topo, BooleanOp::Cut, block, rod).unwrap();
        let before = crate::measure::solid_volume(&topo, holed, 0.01).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&topo, holed).unwrap();
        let sewn = sew_faces(&mut topo, &faces, 1e-7).unwrap();
        let holes: usize = brepkit_topology::explorer::solid_faces(&topo, sewn)
            .unwrap()
            .iter()
            .map(|&f| topo.face(f).unwrap().inner_wires().len())
            .sum();
        assert_eq!(holes, 2);
        assert!(
            crate::validate::validate_solid(&topo, sewn)
                .unwrap()
                .is_valid()
        );
        let after = crate::measure::solid_volume(&topo, sewn, 0.01).unwrap();
        assert!(
            (after - before).abs() < 1e-6 * before,
            "{after} vs {before}"
        );
    }
}
