//! N411: equal-radius TwoEdge corners must have incident wires and closed supports.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    clippy::panic,
    deprecated
)]

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;

fn source(topo: &mut Topology, size: f64) -> SolidId {
    let wire = make_polygon_wire(
        topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(size, 0.0, 0.0),
            Point3::new(size, size, 0.0),
            Point3::new(0.0, size, 0.0),
        ],
        Tolerance::new().linear,
    )
    .unwrap();
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), size).unwrap()
}

fn edge_at(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let tolerance = Tolerance::new().linear;
    let matches: Vec<_> = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&id| {
            let edge = topo.edge(id).unwrap();
            let start = topo.vertex(edge.start()).unwrap().point();
            let end = topo.vertex(edge.end()).unwrap().point();
            ((start - a).length() < tolerance && (end - b).length() < tolerance)
                || ((start - b).length() < tolerance && (end - a).length() < tolerance)
        })
        .collect();
    assert_eq!(matches.len(), 1, "unique target {a:?}–{b:?}");
    matches[0]
}

/// A planar support outer wire must not cross itself in edge interiors.
/// These contact curves are straight degree-one NURBS, so solving their
/// endpoint segments is exact and independent of tessellation.
#[test]
fn n413_native_contact_matrix() {
    let mut failures = Vec::new();
    for boxed in [false, true] {
        for transformed in [false, true] {
            for reverse in [false, true] {
                for fraction in [0.05, 0.1] {
                    for size in [1.0, 254.0] {
                        for count in [1, 2, 4] {
                            let mut topo = Topology::new();
                            let input = if boxed {
                                brepkit_operations::primitives::make_box(
                                    &mut topo, size, size, size,
                                )
                                .unwrap()
                            } else {
                                source(&mut topo, size)
                            };
                            let matrix = if transformed {
                                brepkit_math::mat::Mat4::translation(
                                    3.0 * size,
                                    -2.0 * size,
                                    0.7 * size,
                                ) * brepkit_math::mat::Mat4::rotation_y(0.47)
                                    * brepkit_math::mat::Mat4::rotation_z(0.31)
                            } else {
                                brepkit_math::mat::Mat4::identity()
                            };
                            brepkit_operations::transform::transform_solid(
                                &mut topo, input, &matrix,
                            )
                            .unwrap();
                            let corners = [
                                Point3::new(0., 0., size),
                                Point3::new(size, 0., size),
                                Point3::new(size, size, size),
                                Point3::new(0., size, size),
                            ];
                            let mut selected: Vec<_> = (0..count)
                                .map(|i| {
                                    edge_at(
                                        &topo,
                                        input,
                                        matrix.mul_point(corners[i]),
                                        matrix.mul_point(corners[(i + 1) % 4]),
                                    )
                                })
                                .collect();
                            if reverse {
                                selected.reverse();
                            }
                            let before = source_snapshot(&topo, input);
                            let result_solid = if count == 1 {
                                brepkit_operations::fillet::fillet_rolling_ball(
                                    &mut topo,
                                    input,
                                    &selected,
                                    size * fraction,
                                )
                                .unwrap()
                            } else {
                                fillet_v2(&mut topo, input, &selected, size * fraction)
                                    .unwrap()
                                    .solid
                            };
                            assert_eq!(before, source_snapshot(&topo, input));
                            let inverse = matrix.inverse().unwrap();
                            if count != 1 {
                                exact_product_oracle(
                                    &topo,
                                    result_solid,
                                    size,
                                    fraction,
                                    count,
                                    &inverse,
                                );
                            }
                            let begin = failures.len();
                            let label = format!(
                                "box={boxed} moved={transformed} reverse={reverse} S={size} q={fraction} count={count}"
                            );
                            for fid in solid_faces(&topo, result_solid).unwrap() {
                                let face = topo.face(fid).unwrap();
                                let FaceSurface::Plane { .. } = face.surface() else {
                                    continue;
                                };
                                let wire = topo.wire(face.outer_wire()).unwrap();
                                let contacts: Vec<_> = wire
                                    .edges()
                                    .iter()
                                    .filter_map(|oe| {
                                        let edge = topo.edge(oe.edge()).unwrap();
                                        match edge.curve() {
                                            EdgeCurve::NurbsCurve(curve)
                                                if curve.degree() == 1
                                                    && curve.control_points().len() == 2 =>
                                            {
                                                Some((
                                                    oe.edge(),
                                                    inverse.mul_point(
                                                        topo.vertex(edge.start()).unwrap().point(),
                                                    ),
                                                    inverse.mul_point(
                                                        topo.vertex(edge.end()).unwrap().point(),
                                                    ),
                                                ))
                                            }
                                            _ => None,
                                        }
                                    })
                                    .collect();
                                for (i, &(eid, a, b)) in contacts.iter().enumerate() {
                                    for &(other, c, d) in &contacts[i + 1..] {
                                        if [a, b, c, d]
                                            .iter()
                                            .any(|p| (p.z() - size).abs() > size * 1e-7)
                                        {
                                            continue;
                                        }
                                        let (u, v, w) = (b - a, d - c, c - a);
                                        let cross =
                                            |a: Vec3, b: Vec3| a.x() * b.y() - a.y() * b.x();
                                        let denominator = cross(u, v);
                                        if denominator.abs() < 1e-12 {
                                            continue;
                                        }
                                        let t = cross(w, v) / denominator;
                                        let s = cross(w, u) / denominator;
                                        if t > 1e-7 && t < 1. - 1e-7 && s > 1e-7 && s < 1. - 1e-7 {
                                            failures.push(format!("size={size} count={count} face={fid:?} edges={eid:?}/{other:?} crossing={:?}",a+u*t));
                                        }
                                    }
                                }
                            }
                            eprintln!("ROW {label} crossings={}", failures.len() - begin);
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "native support-wire self-intersections: {failures:#?}"
    );
}

fn source_snapshot(topo: &Topology, solid: SolidId) -> Vec<String> {
    let mut rows = vec![format!("{:?}", topo.solid(solid).unwrap())];
    let solid_data = topo.solid(solid).unwrap();
    for shell in
        std::iter::once(solid_data.outer_shell()).chain(solid_data.inner_shells().iter().copied())
    {
        rows.push(format!("{shell:?}:{:?}", topo.shell(shell).unwrap()));
    }
    for face in solid_faces(topo, solid).unwrap() {
        let f = topo.face(face).unwrap();
        rows.push(format!("{face:?}:{f:?}"));
        for wire in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
            rows.push(format!("{wire:?}:{:?}", topo.wire(wire).unwrap()));
            for oe in topo.wire(wire).unwrap().edges() {
                rows.push(format!(
                    "{:?}:{face:?}:{:?}",
                    oe.edge(),
                    topo.pcurves().get(oe.edge(), face)
                ));
            }
        }
    }
    for edge in solid_edges(topo, solid).unwrap() {
        let e = topo.edge(edge).unwrap();
        rows.push(format!("{edge:?}:{e:?}"));
        for vertex in [e.start(), e.end()] {
            rows.push(format!("{vertex:?}:{:?}", topo.vertex(vertex).unwrap()));
        }
    }
    rows.sort();
    rows
}

/// Independent analytic graph oracle, in the fixture frame rather than the
/// builder's chosen UV frame. No projection onto the tested NURBS is used.
fn exact_product_oracle(
    topo: &Topology,
    solid: SolidId,
    size: f64,
    fraction: f64,
    count: usize,
    inverse: &brepkit_math::mat::Mat4,
) {
    use brepkit_topology::explorer::solid_vertices;
    let r = size * fraction;
    let h = size - r;
    let tol = size * 1e-7;
    let vertices = solid_vertices(topo, solid).unwrap();
    let edges = solid_edges(topo, solid).unwrap();
    let faces = solid_faces(topo, solid).unwrap();
    // N413/N414 amendment: the sharp mitered n=2 corner. No corner face,
    // no torus, no NURBS surface at all — every blend surface is an exact
    // radius-r cylinder, and the two cylinders at each corner meet along
    // one exact `Ellipse3D` crease edge, with no top-contact crossing
    // (checked independently by this test's own outer loop) and no
    // corner-fill arc left over.
    assert_eq!(
        (vertices.len(), edges.len(), faces.len()),
        if count == 2 {
            (11, 17, 8)
        } else {
            (12, 20, 10)
        },
        "V/E/F must match the amendment's exact counts"
    );
    for (i, &v) in vertices.iter().enumerate() {
        for &w in &vertices[i + 1..] {
            assert!(
                (topo.vertex(v).unwrap().point() - topo.vertex(w).unwrap().point()).length() > tol,
                "distinct coincident vertex identities {v:?}/{w:?}"
            );
        }
    }
    let mut uses = std::collections::HashMap::<_, Vec<_>>::new();
    let mut cylinders = Vec::new();
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        let wire = topo.wire(face.outer_wire()).unwrap();
        for (i, oe) in wire.edges().iter().enumerate() {
            let edge = topo.edge(oe.edge()).unwrap();
            let next = wire.edges()[(i + 1) % wire.edges().len()];
            assert_eq!(
                oe.oriented_end(edge),
                next.oriented_start(topo.edge(next.edge()).unwrap()),
                "face {fid:?} wire must close"
            );
            uses.entry(oe.edge())
                .or_default()
                .push(oe.is_forward() ^ face.is_reversed());
        }
        assert!(
            !matches!(face.surface(), FaceSurface::Torus(_)),
            "no torus: the horn-torus construction is the defect this replaces"
        );
        assert!(
            !matches!(face.surface(), FaceSurface::Nurbs(_)),
            "no NURBS corner face: the amendment has no corner face at all"
        );
        if let FaceSurface::Cylinder(c) = face.surface() {
            assert!(
                (c.radius() - r).abs() < tol,
                "every stripe cylinder must have exact requested radius"
            );
            cylinders.push((fid, c.clone()));
        }
    }
    assert_eq!(
        cylinders.len(),
        if count == 2 { 2 } else { 4 },
        "exact stripe cylinder count"
    );
    for (edge, senses) in &uses {
        assert_eq!(senses.len(), 2, "edge {edge:?} must have exactly two uses");
        assert_ne!(
            senses[0], senses[1],
            "edge {edge:?} uses must be opposite-sense"
        );
    }

    // The crease: exactly one `Ellipse3D` edge per corner, shared by
    // exactly the two stripe cylinders meeting there. Sample it and check
    // it lies at exact perpendicular distance r from *both* cylinder axes
    // independently (not merely that it was constructed from them), and
    // that the two cylinders' outward normals there have a signed dot
    // product in [0,1) — convex and never fully tangent except at the
    // isolated top-contact meeting vertex — proving the crease is the
    // true, convex, non-degenerate intersection of both cylinders, not a
    // coincidental sampling match.
    let mut crease_count = 0;
    for &edge in &edges {
        let e = topo.edge(edge).unwrap();
        let EdgeCurve::Ellipse(_) = e.curve() else {
            continue;
        };
        crease_count += 1;
        let owners: Vec<_> = faces
            .iter()
            .copied()
            .filter(|&f| {
                topo.wire(topo.face(f).unwrap().outer_wire())
                    .unwrap()
                    .edges()
                    .iter()
                    .any(|oe| oe.edge() == edge)
            })
            .collect();
        assert_eq!(
            owners.len(),
            2,
            "crease {edge:?} must have exactly two owning faces"
        );
        let cyls: Vec<_> = owners
            .iter()
            .filter_map(|&f| match topo.face(f).unwrap().surface() {
                FaceSurface::Cylinder(c) => Some(c.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            cyls.len(),
            2,
            "crease {edge:?} must be owned by exactly two cylinders"
        );
        assert!(
            (cyls[0].axis().dot(cyls[1].axis())).abs() < 1e-9,
            "the two stripe cylinders at a corner must be exactly perpendicular"
        );
        let a = topo.vertex(e.start()).unwrap().point();
        let b = topo.vertex(e.end()).unwrap().point();
        let (lo, hi) = e.curve().domain_with_endpoints(a, b);
        let dist_to_axis = |p: Point3, c: &brepkit_math::surfaces::CylindricalSurface| {
            let delta = p - c.origin();
            let along = delta.dot(c.axis());
            (delta - c.axis() * along).length()
        };
        let normal_at = |p: Point3, c: &brepkit_math::surfaces::CylindricalSurface| {
            let delta = p - c.origin();
            let along = delta.dot(c.axis());
            (delta - c.axis() * along).normalize().unwrap()
        };
        for i in 0..=32 {
            let t = lo + (hi - lo) * f64::from(i) / 32.;
            let p = e.curve().evaluate_with_endpoints(t, a, b);
            let d0 = dist_to_axis(p, &cyls[0]);
            let d1 = dist_to_axis(p, &cyls[1]);
            assert!(
                (d0 - r).abs() < tol,
                "crease point off cylinder 0 axis by {}",
                d0 - r
            );
            assert!(
                (d1 - r).abs() < tol,
                "crease point off cylinder 1 axis by {}",
                d1 - r
            );
            let owner_faces: Vec<_> = owners
                .iter()
                .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
                .copied()
                .collect();
            let n0 = normal_at(p, &cyls[0])
                * if topo.face(owner_faces[0]).unwrap().is_reversed() {
                    -1.0
                } else {
                    1.0
                };
            let n1 = normal_at(p, &cyls[1])
                * if topo.face(owner_faces[1]).unwrap().is_reversed() {
                    -1.0
                } else {
                    1.0
                };
            let dot = n0.dot(n1);
            assert!(
                (-1e-9..1.0 + 1e-9).contains(&dot),
                "signed crease normal dot out of [0,1): {dot}"
            );
        }
    }
    assert_eq!(
        crease_count,
        if count == 2 { 1 } else { 4 },
        "exact crease count"
    );
    let _ = (inverse, h);

    assert!(
        brepkit_operations::validate::validate_solid(topo, solid)
            .unwrap()
            .is_valid()
    );
}

#[test]
fn n414_orders_repeats_and_refusal_preserve_sources() {
    let size = 254.;
    let points = [
        Point3::new(0., 0., size),
        Point3::new(size, 0., size),
        Point3::new(size, size, size),
        Point3::new(0., size, size),
    ];
    for count in [2, 4] {
        let mut reference = None;
        for shift in 0..count {
            for reverse in [false, true] {
                for _ in 0..2 {
                    let mut topo = Topology::new();
                    let source = source(&mut topo, size);
                    let mut edges: Vec<_> = (0..count)
                        .map(|i| edge_at(&topo, source, points[i], points[(i + 1) % 4]))
                        .collect();
                    edges.rotate_left(shift);
                    if reverse {
                        edges.reverse();
                    }
                    let before = source_snapshot(&topo, source);
                    let result = fillet_v2(&mut topo, source, &edges, 25.4).unwrap();
                    assert_eq!(result.succeeded.len(), count);
                    assert!(result.failed.is_empty());
                    assert_eq!(before, source_snapshot(&topo, source));
                    exact_product_oracle(
                        &topo,
                        result.solid,
                        size,
                        0.1,
                        count,
                        &brepkit_math::mat::Mat4::identity(),
                    );
                    let signature = geometry_signature(&topo, result.solid);
                    if let Some(expected) = &reference {
                        assert_eq!(&signature, expected, "order {shift}/{reverse}");
                    } else {
                        reference = Some(signature);
                    }
                    let refused = fillet_v2(&mut topo, source, &edges, 2. * size);
                    assert!(refused.is_err() || refused.unwrap().succeeded.is_empty());
                    assert_eq!(before, source_snapshot(&topo, source));
                }
            }
        }
    }
}

/// N414: the equal-radius planar TwoEdge corner's admissibility bound is
/// derived from real per-stripe geometry, not guessed. Case A (`count == 2`,
/// two adjacent top edges: only the shared vertex is a qualifying trihedral
/// corner, the far ends are plain terminals) is bounded by the stripe's own
/// length. Case B (`count == 4`, all four top edges: every vertex is a
/// qualifying corner, so every edge is consumed from both ends) is bounded
/// by half that length. See
/// docs/N414-evidence/radius-admissibility-derivation.md, sections 2-5.
///
/// N432: case A's exact boundary is now admissible — the limit construction
/// closes the corner around the vanished third edge — and must build on the
/// closed form; case B's still refuses, routed to N433.
#[test]
fn n414_radius_admissibility_bound_matches_derivation() {
    for size in [1.0, 254.0] {
        let points = [
            Point3::new(0., 0., size),
            Point3::new(size, 0., size),
            Point3::new(size, size, size),
            Point3::new(0., size, size),
        ];
        for (count, max_radius) in [(2usize, size), (4usize, size / 2.0)] {
            for shift in 0..count {
                for reverse in [false, true] {
                    let mut topo = Topology::new();
                    let source_solid = source(&mut topo, size);
                    let mut edges: Vec<_> = (0..count)
                        .map(|i| edge_at(&topo, source_solid, points[i], points[(i + 1) % 4]))
                        .collect();
                    edges.rotate_left(shift);
                    if reverse {
                        edges.reverse();
                    }
                    let before = source_snapshot(&topo, source_solid);
                    let label =
                        format!("size={size} count={count} shift={shift} reverse={reverse}");

                    // Comfortably admissible: a real, unambiguous success.
                    let comfortable = max_radius * 0.5;
                    let result = fillet_v2(&mut topo, source_solid, &edges, comfortable)
                        .unwrap_or_else(|e| {
                            panic!("{label}: comfortable radius must succeed: {e:?}")
                        });
                    assert_eq!(result.succeeded.len(), count);
                    assert_eq!(before, source_snapshot(&topo, source_solid));
                    exact_product_oracle(
                        &topo,
                        result.solid,
                        size,
                        comfortable / size,
                        count,
                        &brepkit_math::mat::Mat4::identity(),
                    );
                    assert!(
                        brepkit_operations::validate::validate_solid(&topo, result.solid)
                            .unwrap()
                            .is_valid(),
                        "{label}: comfortable radius must remain valid/closed/oriented"
                    );

                    // Just below the derived maximum: the residual stripe
                    // domain is small (a 1e-4 fraction of the edge length)
                    // but strictly positive and non-degenerate -- the
                    // construction must still succeed and produce the same
                    // well-formed topology/identity/tangency the comfortable
                    // case does.
                    let just_below = max_radius * (1.0 - 1e-4);
                    let result = fillet_v2(&mut topo, source_solid, &edges, just_below)
                        .unwrap_or_else(|e| {
                            panic!("{label}: just-below-max radius must succeed: {e:?}")
                        });
                    assert_eq!(result.succeeded.len(), count);
                    assert_eq!(before, source_snapshot(&topo, source_solid));
                    exact_product_oracle(
                        &topo,
                        result.solid,
                        size,
                        just_below / size,
                        count,
                        &brepkit_math::mat::Mat4::identity(),
                    );
                    assert!(
                        brepkit_operations::validate::validate_solid(&topo, result.solid)
                            .unwrap()
                            .is_valid(),
                        "{label}: just-below-max radius must remain valid/closed/oriented"
                    );

                    // Exactly at the derived maximum: the residual domain is
                    // exactly zero. N432 admits it for the two-edge corner —
                    // the limit construction closes around the vanished
                    // material (`close_collapsed_miter_corner`) — so it must
                    // build one valid, closed, oriented solid on the closed
                    // form `(2/3) size^3`. The four-edge pillow's exact limit
                    // is still routed to N433 and must refuse.
                    let at_max = fillet_v2(&mut topo, source_solid, &edges, max_radius);
                    if count == 2 {
                        let result = at_max.unwrap_or_else(|e| {
                            panic!("{label}: the two-edge exact limit must now build: {e:?}")
                        });
                        assert_eq!(result.succeeded.len(), count);
                        let volume = brepkit_operations::measure::solid_volume(
                            &topo,
                            result.solid,
                            1e-3 * size,
                        )
                        .unwrap();
                        let exact = 2.0 / 3.0 * size * size * size;
                        assert!(
                            (volume - exact).abs() <= exact * 5e-4,
                            "{label}: limit volume {volume} is not the closed form {exact}"
                        );
                        assert!(
                            brepkit_operations::validate::validate_solid(&topo, result.solid)
                                .unwrap()
                                .is_valid(),
                            "{label}: the exact limit must remain valid/closed/oriented"
                        );
                    } else {
                        assert!(
                            at_max.is_err() || at_max.unwrap().succeeded.is_empty(),
                            "{label}: exact boundary radius must refuse"
                        );
                    }
                    assert_eq!(before, source_snapshot(&topo, source_solid));

                    // Just above the derived maximum: refused.
                    let just_above = max_radius * (1.0 + 1e-4);
                    let refused = fillet_v2(&mut topo, source_solid, &edges, just_above);
                    assert!(
                        refused.is_err() || refused.unwrap().succeeded.is_empty(),
                        "{label}: just-above-max radius must refuse"
                    );
                    assert_eq!(before, source_snapshot(&topo, source_solid));
                }
            }
        }
    }
}

/// N414: case A and case B have genuinely different admissible radii on the
/// identical edge, because B must additionally fit the same-radius setback
/// from *both* ends without the two corners' consumption overlapping. A
/// radius strictly between `length / 2` (B's bound) and `length` (A's
/// bound) must therefore succeed for the two-adjacent-edge selection and
/// refuse for the four-edge selection on the same box.
#[test]
fn n414_case_a_and_case_b_have_distinct_bounds_on_the_same_edge() {
    let size = 254.0;
    let points = [
        Point3::new(0., 0., size),
        Point3::new(size, 0., size),
        Point3::new(size, size, size),
        Point3::new(0., size, size),
    ];
    let radius = 0.51 * size; // > size/2 (B's bound), < size (A's bound)

    let mut topo_a = Topology::new();
    let source_a = source(&mut topo_a, size);
    let edges_a: Vec<_> = (0..2)
        .map(|i| edge_at(&topo_a, source_a, points[i], points[(i + 1) % 4]))
        .collect();
    let result_a = fillet_v2(&mut topo_a, source_a, &edges_a, radius)
        .expect("case A must accept a radius between length/2 and length");
    assert_eq!(result_a.succeeded.len(), 2);

    let mut topo_b = Topology::new();
    let source_b = source(&mut topo_b, size);
    let edges_b: Vec<_> = (0..4)
        .map(|i| edge_at(&topo_b, source_b, points[i], points[(i + 1) % 4]))
        .collect();
    let before_b = source_snapshot(&topo_b, source_b);
    let refused_b = fillet_v2(&mut topo_b, source_b, &edges_b, radius);
    assert!(
        refused_b.is_err() || refused_b.unwrap().succeeded.is_empty(),
        "case B must refuse the same radius: both ends of each shared edge would overlap"
    );
    assert_eq!(before_b, source_snapshot(&topo_b, source_b));
}

/// N414: the admissibility bound is a function of real 3D edge length and
/// vertex proximity, so a rigid transform of the whole solid must not
/// change whether a given radius is accepted or refused.
#[test]
fn n414_admissibility_is_rigid_transform_invariant() {
    let size = 254.0;
    let points = [
        Point3::new(0., 0., size),
        Point3::new(size, 0., size),
        Point3::new(size, size, size),
        Point3::new(0., size, size),
    ];
    let matrix = brepkit_math::mat::Mat4::translation(11.0 * size, -4.0 * size, 2.3 * size)
        * brepkit_math::mat::Mat4::rotation_y(0.83)
        * brepkit_math::mat::Mat4::rotation_z(1.19);

    for count in [2usize, 4usize] {
        let max_radius = if count == 2 { size } else { size / 2.0 };
        for (label, radius, must_succeed) in [
            ("below", max_radius * (1.0 - 1e-4), true),
            ("above", max_radius * (1.0 + 1e-4), false),
        ] {
            let mut topo = Topology::new();
            let input = source(&mut topo, size);
            brepkit_operations::transform::transform_solid(&mut topo, input, &matrix).unwrap();
            let edges: Vec<_> = (0..count)
                .map(|i| {
                    edge_at(
                        &topo,
                        input,
                        matrix.mul_point(points[i]),
                        matrix.mul_point(points[(i + 1) % 4]),
                    )
                })
                .collect();
            let before = source_snapshot(&topo, input);
            let result = fillet_v2(&mut topo, input, &edges, radius);
            if must_succeed {
                let result = result.unwrap_or_else(|e| {
                    panic!("count={count} {label} must succeed after rigid transform: {e:?}")
                });
                assert_eq!(result.succeeded.len(), count);
            } else {
                assert!(
                    result.is_err() || result.unwrap().succeeded.is_empty(),
                    "count={count} {label} must refuse after rigid transform"
                );
            }
            assert_eq!(before, source_snapshot(&topo, input));
        }
    }
}

fn geometry_signature(topo: &Topology, solid: SolidId) -> Vec<String> {
    let point = |p: Point3| format!("{:.6},{:.6},{:.6}", p.x() + 0., p.y() + 0., p.z() + 0.);
    let edge_signature = |id| {
        let edge = topo.edge(id).unwrap();
        let a = topo.vertex(edge.start()).unwrap().point();
        let b = topo.vertex(edge.end()).unwrap().point();
        let (lo, hi) = edge.curve().domain_with_endpoints(a, b);
        let mut samples: Vec<_> = (0..=8)
            .map(|i| {
                point(edge.curve().evaluate_with_endpoints(
                    lo + (hi - lo) * f64::from(i) / 8.,
                    a,
                    b,
                ))
            })
            .collect();
        samples.sort();
        samples.join(";")
    };
    let mut rows = Vec::new();
    for id in solid_faces(topo, solid).unwrap() {
        let f = topo.face(id).unwrap();
        let mut edges: Vec<_> = topo
            .wire(f.outer_wire())
            .unwrap()
            .edges()
            .iter()
            .map(|oe| edge_signature(oe.edge()))
            .collect();
        edges.sort();
        let kind = match f.surface() {
            FaceSurface::Plane { .. } => "plane",
            FaceSurface::Cylinder(_) => "cylinder",
            FaceSurface::Nurbs(_) => "product",
            _ => panic!("unexpected surface"),
        };
        rows.push(format!("{kind}:{edges:?}"));
    }
    rows.sort();
    rows
}
