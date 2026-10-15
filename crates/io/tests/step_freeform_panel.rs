//! The freeform panel of the step-conformance suite (CC0): a tray whose two
//! walls are cubic B-splines closed round, trimmed at z = 0 and z = 44 by
//! rims written as fitted curves, with no pcurves. Near its foot the outer
//! wall folds over, so its bottom rim curls through itself: the rim winds
//! once round the base and twice round a 2 mm² curl inside it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_io::step::reader::read_step;
use brepkit_math::chord::DEFAULT_ANGULAR_TOL;
use brepkit_math::nurbs::projection::project_point_to_surface;
use brepkit_math::vec::Point3;
use brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

/// The area the bottom rim winds round once: its signed area, 86,259.7 mm²,
/// less twice the curl it winds round twice.
const BASE_AREA: f64 = 86_255.6;

/// A point inside the curl, which is no part of the base.
const IN_CURL: (f64, f64) = (-120.376, -100.652);

/// The tip of the fold, on the outer wall 0.86 mm above its foot.
const FOLD_TIP: (f64, f64, f64) = (-120.446, -99.105, 0.859);

fn point_triangle(p: Point3, [a, b, c]: [Point3; 3]) -> f64 {
    let n = (b - a).cross(c - a);
    let side = |s: Point3, e: Point3| {
        let d = e - s;
        let t = ((p - s).dot(d) / d.dot(d).max(1e-300)).clamp(0.0, 1.0);
        (p - (s + d * t)).length()
    };
    if let Ok(n) = n.normalize() {
        let h = (p - a).dot(n);
        let q = p - n * h;
        let inside = [(a, b), (b, c), (c, a)]
            .iter()
            .all(|&(s, e)| (e - s).cross(q - s).dot(n) >= 0.0);
        if inside {
            return h.abs();
        }
    }
    side(a, b).min(side(b, c)).min(side(c, a))
}

/// Within four deflections is the step-conformance pass mark. The walls are
/// checked about the fold, where a wall can leave its surface.
#[test]
fn the_folded_freeform_panel_meshes_within_four_deflections_of_its_faces() {
    let mut topo = Topology::new();
    let solids = read_step(include_str!("data/freeform_panel.step"), &mut topo).unwrap();
    assert_eq!(solids.len(), 1);
    let faces = solid_faces(&topo, solids[0]).unwrap();
    let tip = Point3::new(FOLD_TIP.0, FOLD_TIP.1, FOLD_TIP.2);
    for deflection in [0.5, 0.1] {
        let reach = 4.0 * deflection;
        let (mesh, offsets) = tessellate_solid_grouped_with_tolerance(
            &topo,
            solids[0],
            deflection,
            DEFAULT_ANGULAR_TOL,
        )
        .unwrap();
        // Every wall is trimmed between its rims, not meshed whole.
        for p in &mesh.positions {
            assert!(
                (-1e-6..=44.0 + 1e-6).contains(&p.z()),
                "{deflection}: a vertex at z = {} lies off the part",
                p.z()
            );
        }
        let mut tip_gap = f64::INFINITY;
        for (fi, span) in offsets.windows(2).enumerate() {
            let triangles: Vec<[Point3; 3]> = mesh.indices[span[0] as usize..span[1] as usize]
                .chunks_exact(3)
                .map(|t| [t[0], t[1], t[2]].map(|i| mesh.positions[i as usize]))
                .collect();
            for &t in &triangles {
                tip_gap = tip_gap.min(point_triangle(tip, t));
            }
            match topo.face(faces[fi]).unwrap().surface() {
                FaceSurface::Nurbs(surface) => {
                    for [a, b, c] in &triangles {
                        let centre = Point3::new(
                            (a.x() + b.x() + c.x()) / 3.0,
                            (a.y() + b.y() + c.y()) / 3.0,
                            (a.z() + b.z() + c.z()) / 3.0,
                        );
                        if (centre - tip).length() > 10.0 {
                            continue;
                        }
                        let off = project_point_to_surface(surface, centre, 1e-9)
                            .unwrap()
                            .distance;
                        assert!(
                            off <= reach,
                            "{deflection}: face {fi} has a triangle {off} off its surface at {centre:?}"
                        );
                    }
                }
                FaceSurface::Plane { .. }
                    if triangles.iter().flatten().all(|p| p.z().abs() < 1e-9) =>
                {
                    let area: f64 = triangles
                        .iter()
                        .map(|&[a, b, c]| 0.5 * (b - a).cross(c - a).length())
                        .sum();
                    assert!(
                        (area - BASE_AREA).abs() < 1e-3 * BASE_AREA,
                        "{deflection}: the base covers {area} of {BASE_AREA}"
                    );
                    let (x, y) = IN_CURL;
                    let covers = |&[a, b, c]: &[Point3; 3]| {
                        let s = |p: Point3, q: Point3| {
                            (q.x() - p.x()) * (y - p.y()) - (q.y() - p.y()) * (x - p.x())
                        };
                        let (d1, d2, d3) = (s(a, b), s(b, c), s(c, a));
                        (d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0)
                            || (d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0)
                    };
                    assert!(
                        !triangles.iter().any(covers),
                        "{deflection}: the base covers the curl its rim winds round twice"
                    );
                }
                _ => {}
            }
        }
        assert!(
            tip_gap <= reach,
            "{deflection}: the fold's tip lies {tip_gap} from the mesh"
        );
    }
}
