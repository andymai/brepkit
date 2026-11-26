//! Tolerances of B-spline sections, proved off both of their faces.
//!
//! A marched section is a fit: it lies off the two surfaces it is a section
//! of by up to its fit error, along its whole length. Each B-spline section
//! edge whose bound against both faces' surfaces is proved and exceeds the
//! linear tolerance takes that bound as its tolerance, which
//! [`super::link_existing`] reads when it proves the section one curve with
//! a boundary edge. A section with no proved bound keeps no tolerance.
//!
//! A B-spline section then ends on, and is cut at, the vertices of its faces
//! it passes within their tolerance plus its own: a vertex an earlier boolean
//! fitted can stand off the section's true point by its fit error.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;

use crate::ds::GfaArena;
use crate::error::AlgoError;

/// Gives each B-spline section edge of `arena` the greater of its proved
/// bounds off its two faces' surfaces as its tolerance, where both are
/// proved and the greater exceeds the linear tolerance.
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails.
pub fn perform(topo: &mut Topology, tol: Tolerance, arena: &mut GfaArena) -> Result<(), AlgoError> {
    let mut proved = Vec::new();
    for section in &arena.curves {
        let EdgeCurve::NurbsCurve(curve) = &section.curve else {
            continue;
        };
        let Some(pb) = section
            .pave_blocks
            .first()
            .and_then(|&id| arena.pave_blocks.get(id))
        else {
            continue;
        };
        let (t0, t1) = section.t_range;
        let span = (t0.min(t1), t0.max(t1));
        let off = |face| -> Result<Option<f64>, AlgoError> {
            Ok(crate::surface_reach::spline_off_face(
                curve,
                span,
                topo.face(face)?.surface(),
                tol.linear,
            ))
        };
        if let (Some(a), Some(b)) = (off(section.face_a)?, off(section.face_b)?) {
            let bound = a.max(b);
            if bound > tol.linear {
                proved.push((pb.original_edge, bound));
            }
        }
    }
    for (edge, bound) in proved {
        let edge = topo.edge_mut(edge)?;
        if bound > edge.tolerance().unwrap_or(0.0) {
            edge.set_tolerance(Some(bound));
        }
    }
    settle_on_vertices(topo, tol, arena)
}

/// Moves each end at parameter `from` of the blocks of section `index` from
/// `root` down to the parameter `to`, and the section's range with them.
/// Nothing moves unless `to` lies inside the section's range, where its
/// bound is proved, and every block with an end at `from` keeps its other
/// end strictly on the same side of `to`. Returns whether the ends moved.
fn move_end(
    arena: &mut GfaArena,
    index: usize,
    root: crate::ds::PaveBlockId,
    from: f64,
    to: f64,
) -> bool {
    let Some(section) = arena.curves.get(index) else {
        return false;
    };
    let (lo, hi) = (
        section.t_range.0.min(section.t_range.1),
        section.t_range.0.max(section.t_range.1),
    );
    if to.is_nan() || to < lo || to > hi {
        return false;
    }
    let keeps_side = |other: f64| {
        let (was, now) = (other - from, other - to);
        now != 0.0 && was.is_sign_positive() == now.is_sign_positive()
    };
    let mut moves: Vec<(crate::ds::PaveBlockId, bool)> = Vec::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        let Some(pb) = arena.pave_blocks.get(id) else {
            continue;
        };
        for (at_start, end, other) in [
            (true, pb.start.parameter, pb.end.parameter),
            (false, pb.end.parameter, pb.start.parameter),
        ] {
            if end.to_bits() == from.to_bits() {
                if !keeps_side(other) {
                    return false;
                }
                moves.push((id, at_start));
            }
        }
        stack.extend(pb.children.iter().copied());
    }
    if moves.is_empty() {
        return false;
    }
    for (id, at_start) in moves {
        if let Some(pb) = arena.pave_blocks.get_mut(id) {
            if at_start {
                pb.start.parameter = to;
            } else {
                pb.end.parameter = to;
            }
        }
    }
    if let Some(section) = arena.curves.get_mut(index) {
        if section.t_range.0.to_bits() == from.to_bits() {
            section.t_range.0 = to;
        }
        if section.t_range.1.to_bits() == from.to_bits() {
            section.t_range.1 = to;
        }
    }
    true
}

/// Ends each B-spline section on, and cuts it at, the vertices of its two
/// faces it passes within their tolerance plus its own.
///
/// A section's end that is no vertex of its faces, and no vertex the pave
/// filler put on a boundary edge, joins the nearest vertex of its faces
/// within those tolerances of the section's end point. Each end then moves
/// to the point where the section passes its end vertex, read by the
/// vertex's projection on the section
/// ([`crate::surface_reach::point_off_spline`], an upper bound of its
/// distance), where the end point stands off the vertex by more than the
/// linear tolerance and that point is nearer: past it, a marched end
/// overshoots its vertex. Each other vertex of the faces the section passes
/// within those tolerances, read the same way, cuts each block of the
/// section that is in no common block where that point falls inside it. A
/// vertex so joined takes its distance as its tolerance where that is
/// larger.
fn settle_on_vertices(
    topo: &mut Topology,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    use brepkit_topology::vertex::VertexId;
    if !arena
        .curves
        .iter()
        .any(|c| matches!(c.curve, EdgeCurve::NurbsCurve(_)))
    {
        return Ok(());
    }
    let mut raised: Vec<(VertexId, f64)> = Vec::new();
    // Vertices the pave filler put on boundary edges: a section that ends on
    // one ends where it crosses that edge, and keeps it.
    let mut on_edges: std::collections::HashSet<VertexId> = std::collections::HashSet::new();
    let roots: Vec<_> = arena.edge_pave_blocks.values().flatten().copied().collect();
    for leaf in arena.collect_leaf_pave_blocks(&roots) {
        if let Some(pb) = arena.pave_blocks.get(leaf) {
            on_edges.insert(arena.resolve_vertex(pb.start.vertex));
            on_edges.insert(arena.resolve_vertex(pb.end.vertex));
        }
    }
    for index in 0..arena.curves.len() {
        let (curve, faces, roots) = {
            let section = &arena.curves[index];
            let EdgeCurve::NurbsCurve(curve) = &section.curve else {
                continue;
            };
            (
                curve.clone(),
                [section.face_a, section.face_b],
                section.pave_blocks.clone(),
            )
        };
        let mut own: Vec<VertexId> = Vec::new();
        for face in faces {
            let face = topo.face(face)?;
            for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                for oe in topo.wire(wire)?.edges() {
                    let edge = topo.edge(oe.edge())?;
                    for v in [edge.start(), edge.end()] {
                        let v = arena.resolve_vertex(v);
                        if !own.contains(&v) {
                            own.push(v);
                        }
                    }
                }
            }
        }
        let Some(hull) =
            brepkit_math::aabb::Aabb3::try_from_points(curve.control_points().iter().copied())
        else {
            continue;
        };
        for root in roots {
            let Some(pb) = arena.pave_blocks.get(root) else {
                continue;
            };
            let (start, end) = (pb.start, pb.end);
            let held = topo
                .edge(pb.original_edge)?
                .tolerance()
                .unwrap_or(0.0)
                .max(tol.linear);
            let reach = |v: VertexId| -> Result<(f64, brepkit_math::vec::Point3), AlgoError> {
                let vertex = topo.vertex(v)?;
                Ok((vertex.tolerance() + held, vertex.point()))
            };
            // The ends.
            for pave in [start, end] {
                let mut at = arena.resolve_vertex(pave.vertex);
                let point = curve.evaluate(pave.parameter);
                if !own.contains(&at) && !on_edges.contains(&at) {
                    let size = point.x().abs().max(point.y().abs()).max(point.z().abs());
                    let mut best: Option<(f64, VertexId)> = None;
                    for &v in &own {
                        let (within, p) = reach(v)?;
                        let gap = (p - point).length()
                            + 64.0
                                * f64::EPSILON
                                * size.max(p.x().abs().max(p.y().abs()).max(p.z().abs()));
                        if gap <= within && best.is_none_or(|(g, _)| gap < g) {
                            best = Some((gap, v));
                        }
                    }
                    if let Some((gap, v)) = best
                        && arena.resolve_vertex(v) != at
                    {
                        arena.same_domain_vertices.insert(at, v);
                        raised.push((v, gap));
                        at = v;
                    }
                }
                // The section ends where it passes its end vertex: past it,
                // its end would stand off the vertex by the overshoot.
                let (within, p) = reach(at)?;
                let off_end = (p - point).length();
                if off_end > tol.linear
                    && let Some((pass, t)) = crate::surface_reach::point_off_spline(p, &curve)
                    && pass <= within
                    && pass < off_end
                    && move_end(arena, index, root, pave.parameter, t)
                {
                    raised.push((at, pass));
                }
            }
            // The vertices the section passes inside its span, as its ends
            // now stand.
            let Some((start, end)) = arena.pave_blocks.get(root).map(|pb| (pb.start, pb.end))
            else {
                continue;
            };
            let (lo, hi) = (
                start.parameter.min(end.parameter),
                start.parameter.max(end.parameter),
            );
            let margin = 1e-9 * (hi - lo);
            let ends = [
                arena.resolve_vertex(start.vertex),
                arena.resolve_vertex(end.vertex),
            ];
            let leaves = arena.collect_leaf_pave_blocks(&[root]);
            let mut taken: Vec<VertexId> = ends.to_vec();
            for &leaf in &leaves {
                if let Some(l) = arena.pave_blocks.get(leaf) {
                    taken.push(arena.resolve_vertex(l.start.vertex));
                    taken.push(arena.resolve_vertex(l.end.vertex));
                }
            }
            let mut cuts: Vec<crate::ds::Pave> = Vec::new();
            for &v in &own {
                if taken.contains(&v) {
                    continue;
                }
                let (within, p) = reach(v)?;
                if !hull.expanded(within).contains_point(p) {
                    continue;
                }
                let Some((gap, t)) = crate::surface_reach::point_off_spline(p, &curve) else {
                    continue;
                };
                if gap <= within && t > lo + margin && t < hi - margin {
                    cuts.push(crate::ds::Pave::new(v, t));
                    raised.push((v, gap));
                }
            }
            for leaf in leaves {
                if arena.pb_to_cb.contains_key(&leaf) {
                    continue;
                }
                let Some(l) = arena.pave_blocks.get(leaf) else {
                    continue;
                };
                let (from, to, edge) = (l.start, l.end, l.original_edge);
                let rising = from.parameter <= to.parameter;
                let mut inside: Vec<crate::ds::Pave> = cuts
                    .iter()
                    .copied()
                    .filter(|c| {
                        c.parameter > from.parameter.min(to.parameter) + margin
                            && c.parameter < from.parameter.max(to.parameter) - margin
                    })
                    .collect();
                if inside.is_empty() {
                    continue;
                }
                inside.sort_by(|a, b| {
                    if rising {
                        a.parameter.total_cmp(&b.parameter)
                    } else {
                        b.parameter.total_cmp(&a.parameter)
                    }
                });
                let mut children = Vec::with_capacity(inside.len() + 1);
                let mut at = from;
                for pave in inside.into_iter().chain(std::iter::once(to)) {
                    children.push(
                        arena
                            .pave_blocks
                            .alloc(crate::ds::PaveBlock::new(edge, at, pave)),
                    );
                    at = pave;
                }
                if let Some(l) = arena.pave_blocks.get_mut(leaf) {
                    l.children = children;
                }
            }
        }
    }
    for (v, gap) in raised {
        let vertex = topo.vertex_mut(v)?;
        if gap > vertex.tolerance() {
            vertex.set_tolerance(gap);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::aabb::Aabb3;
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use crate::ds::{GfaArena, IntersectionCurveDS, Pave, PaveBlock};

    /// A pose: its name, where it takes the origin, and how it turns a
    /// direction.
    type Pose = (&'static str, Point3, Box<dyn Fn(Vec3) -> Vec3>);

    /// A unit cube whose top face is a B-spline patch on its plane, and whose
    /// corner over (1, 1) stands 3e-6 above it with a tolerance of 1e-7, and
    /// the section of that face and the side x = 1, from (1, 0, 1) to the
    /// corner's true point. The section's end joins the corner only once the
    /// corner's tolerance is raised to its proved distance from the patch.
    /// Upright, turned and mirrored.
    #[test]
    fn a_section_end_joins_a_vertex_off_it_only_once_its_tolerance_is_raised() {
        use brepkit_math::nurbs::surface::NurbsSurface;
        use brepkit_topology::test_utils::make_unit_cube_manifold_at;
        let axis = Vec3::new(1.0, 2.0, 3.0).normalize().unwrap();
        let (sin, cos) = 0.7f64.sin_cos();
        let turn =
            move |e: Vec3| e * cos + axis.cross(e) * sin + axis * (axis.dot(e) * (1.0 - cos));
        let mirror = |e: Vec3| Vec3::new(-e.x(), e.y(), e.z());
        let poses: [Pose; 3] = [
            ("upright", Point3::new(0.0, 0.0, 0.0), Box::new(|e| e)),
            ("turned", Point3::new(3.0, -2.0, 5.0), Box::new(turn)),
            (
                "mirrored",
                Point3::new(-3.0, -2.0, 5.0),
                Box::new(move |e| mirror(turn(e))),
            ),
        ];
        for (name, origin, along) in &poses {
            let at = |p: Point3| *origin + along(Vec3::new(p.x(), p.y(), p.z()));
            for raise in [false, true] {
                let mut topo = Topology::default();
                let solid = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
                let near =
                    |p: Point3, x: f64, y: f64, z: f64| (p - Point3::new(x, y, z)).length() < 1e-12;
                let mut corner = None;
                let mut low = None;
                let mut moved = std::collections::HashSet::new();
                for e in brepkit_topology::explorer::solid_edges(&topo, solid).unwrap() {
                    let edge = topo.edge(e).unwrap();
                    for v in [edge.start(), edge.end()] {
                        if !moved.insert(v) {
                            continue;
                        }
                        let p = topo.vertex(v).unwrap().point();
                        if near(p, 1.0, 1.0, 1.0) {
                            corner = Some(v);
                        }
                        if near(p, 1.0, 0.0, 1.0) {
                            low = Some(v);
                        }
                        topo.vertex_mut(v).unwrap().set_point(at(p));
                    }
                }
                let (corner, low) = (corner.unwrap(), low.unwrap());
                let lift = along(Vec3::new(0.0, 0.0, 1.0)) * 3e-6;
                topo.vertex_mut(corner)
                    .unwrap()
                    .set_point(at(Point3::new(1.0, 1.0, 1.0)) + lift);
                let mut top = None;
                let mut side = None;
                for f in brepkit_topology::explorer::solid_faces(&topo, solid).unwrap() {
                    let FaceSurface::Plane { normal, d } = topo.face(f).unwrap().surface().clone()
                    else {
                        unreachable!()
                    };
                    let foot = Point3::new(0.0, 0.0, 0.0) + normal * d;
                    let n = along(normal);
                    let posed = FaceSurface::Plane {
                        normal: n,
                        d: n.dot(Vec3::new(at(foot).x(), at(foot).y(), at(foot).z())),
                    };
                    if (normal - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-12 {
                        top = Some(f);
                        let patch = NurbsSurface::new(
                            1,
                            1,
                            vec![0.0, 0.0, 1.0, 1.0],
                            vec![0.0, 0.0, 1.0, 1.0],
                            vec![
                                vec![
                                    at(Point3::new(0.0, 0.0, 1.0)),
                                    at(Point3::new(0.0, 1.0, 1.0)),
                                ],
                                vec![
                                    at(Point3::new(1.0, 0.0, 1.0)),
                                    at(Point3::new(1.0, 1.0, 1.0)),
                                ],
                            ],
                            vec![vec![1.0; 2]; 2],
                        )
                        .unwrap();
                        topo.face_mut(f)
                            .unwrap()
                            .set_surface(FaceSurface::Nurbs(patch));
                    } else {
                        if (normal - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-12 {
                            side = Some(f);
                        }
                        topo.face_mut(f).unwrap().set_surface(posed);
                    }
                }
                if raise {
                    crate::ds::shape_store::prove_source_pcurves(
                        &mut topo,
                        &[solid],
                        &[],
                        Tolerance::default(),
                    )
                    .unwrap();
                }
                let curve = NurbsCurve::new(
                    1,
                    vec![0.0, 0.0, 1.0, 1.0],
                    vec![
                        at(Point3::new(1.0, 0.0, 1.0)),
                        at(Point3::new(1.0, 1.0, 1.0)),
                    ],
                    vec![1.0; 2],
                )
                .unwrap();
                let end = topo.add_vertex(Vertex::new(curve.evaluate(1.0), 1e-7));
                let section =
                    topo.add_edge(Edge::new(low, end, EdgeCurve::NurbsCurve(curve.clone())));
                let mut arena = GfaArena::new();
                let pb = arena.pave_blocks.alloc(PaveBlock::new(
                    section,
                    Pave::new(low, 0.0),
                    Pave::new(end, 1.0),
                ));
                arena.curves.push(IntersectionCurveDS {
                    curve: EdgeCurve::NurbsCurve(curve.clone()),
                    face_a: top.unwrap(),
                    face_b: side.unwrap(),
                    bbox: Aabb3::from_points(curve.control_points().iter().copied()),
                    pave_blocks: vec![pb],
                    t_range: (0.0, 1.0),
                });

                super::perform(&mut topo, Tolerance::default(), &mut arena).unwrap();

                assert_eq!(
                    arena.resolve_vertex(end) == corner,
                    raise,
                    "{name}: raised {raise}, corner tolerance {:e}",
                    topo.vertex(corner).unwrap().tolerance()
                );
            }
        }
    }

    /// A section over [2, 8] of a straight spline on [0, 10] that ends on a
    /// vertex of its faces lying on the spline at 8.5, past the section: the
    /// vertex's nearest point on the spline is outside the section, so the
    /// end stays where it is, and so does the section's range.
    #[test]
    fn a_section_end_never_moves_past_its_own_range() {
        let curve = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 10.0, 10.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)],
            vec![1.0; 2],
        )
        .unwrap();
        let mut topo = Topology::default();
        let start = topo.add_vertex(Vertex::new(curve.evaluate(2.0), 1e-7));
        let end = topo.add_vertex(Vertex::new(curve.evaluate(8.5), 1e-7));
        let far = topo.add_vertex(Vertex::new(Point3::new(8.5, 5.0, 5.0), 1e-7));
        let side = topo.add_edge(Edge::new(end, far, EdgeCurve::Line));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(side, true)], false).unwrap());
        let plane = |normal: Vec3| FaceSurface::Plane { normal, d: 0.0 };
        let face_a = topo.add_face(Face::new(wire, vec![], plane(Vec3::new(0.0, 0.0, 1.0))));
        let face_b = topo.add_face(Face::new(wire, vec![], plane(Vec3::new(0.0, 1.0, 0.0))));
        let section = topo.add_edge(Edge::new(start, end, EdgeCurve::NurbsCurve(curve.clone())));
        let mut arena = GfaArena::new();
        let pb = arena.pave_blocks.alloc(PaveBlock::new(
            section,
            Pave::new(start, 2.0),
            Pave::new(end, 8.0),
        ));
        arena.curves.push(IntersectionCurveDS {
            curve: EdgeCurve::NurbsCurve(curve.clone()),
            face_a,
            face_b,
            bbox: Aabb3::from_points(curve.control_points().iter().copied()),
            pave_blocks: vec![pb],
            t_range: (2.0, 8.0),
        });

        super::perform(&mut topo, Tolerance::default(), &mut arena).unwrap();

        let block = arena.pave_blocks.get(pb).unwrap();
        assert_eq!(
            (block.start.parameter, block.end.parameter),
            (2.0, 8.0),
            "the block's ends"
        );
        assert_eq!(arena.curves[0].t_range, (2.0, 8.0), "the section's range");
    }

    /// A cubic along a ruling of a cylinder of radius 2, `off` outside it,
    /// in the plane through the axis that holds it, as a section of the two
    /// faces: its edge takes a tolerance of at least `off` and at most twice
    /// it, and one on the cylinder takes none.
    #[test]
    fn a_b_spline_section_takes_its_proved_distance_from_its_faces() {
        for off in [0.0, 1e-4] {
            let curve = NurbsCurve::new(
                3,
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                (0..7)
                    .map(|k| Point3::new(2.0 + off, 0.0, f64::from(k)))
                    .collect(),
                vec![1.0; 7],
            )
            .unwrap();
            let mut topo = Topology::default();
            let a = topo.add_vertex(Vertex::new(curve.evaluate(0.0), 1e-7));
            let b = topo.add_vertex(Vertex::new(curve.evaluate(4.0), 1e-7));
            let section = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(curve.clone())));
            let wire =
                topo.add_wire(Wire::new(vec![OrientedEdge::new(section, true)], false).unwrap());
            let cylinder =
                CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0)
                    .unwrap();
            let face_a = topo.add_face(Face::new(wire, vec![], FaceSurface::Cylinder(cylinder)));
            let face_b = topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    d: 0.0,
                },
            ));
            let mut arena = GfaArena::new();
            let pb = arena.pave_blocks.alloc(PaveBlock::new(
                section,
                Pave::new(a, 0.0),
                Pave::new(b, 4.0),
            ));
            arena.curves.push(IntersectionCurveDS {
                curve: EdgeCurve::NurbsCurve(curve.clone()),
                face_a,
                face_b,
                bbox: Aabb3::from_points(curve.control_points().iter().copied()),
                pave_blocks: vec![pb],
                t_range: (0.0, 4.0),
            });

            super::perform(&mut topo, Tolerance::default(), &mut arena).unwrap();

            let held = topo.edge(section).unwrap().tolerance();
            if off > 0.0 {
                assert!(
                    held.is_some_and(|t| t >= off && t <= 2.0 * off + 1e-12),
                    "off {off}: {held:?}"
                );
            } else {
                assert!(held.is_none(), "on the cylinder: {held:?}");
            }
        }
    }
}
