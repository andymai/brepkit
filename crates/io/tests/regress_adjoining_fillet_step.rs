//! N339: STEP geometry and mass-property qualification of the reduced sequence.
#![allow(clippy::unwrap_used, clippy::print_stderr, deprecated)]

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::fillet::fillet_rolling_ball;
use brepkit_operations::measure::solid_volume;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces, solid_vertices};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;

fn source(topo: &mut Topology) -> SolidId {
    let wire = make_polygon_wire(
        topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(40.0, 0.0, 0.0),
            Point3::new(40.0, 25.0, 0.0),
            Point3::new(0.0, 25.0, 0.0),
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
    brepkit_operations::extrude::extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), 30.0).unwrap()
}

fn edge_at(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let matches: Vec<_> = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&id| {
            let e = topo.edge(id).unwrap();
            let p = topo.vertex(e.start()).unwrap().point();
            let q = topo.vertex(e.end()).unwrap().point();
            let t = Tolerance::new().linear;
            ((p - a).length() < t && (q - b).length() < t)
                || ((p - b).length() < t && (q - a).length() < t)
        })
        .collect();
    assert_eq!(matches.len(), 1);
    matches[0]
}

#[test]
fn reduced_adjoining_step_geometry_and_volume() {
    let mut topo = Topology::new();
    let source = source(&mut topo);
    let edge = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let first = fillet_rolling_ball(&mut topo, source, &[edge], 1.0).unwrap();
    let edge = edge_at(
        &topo,
        first,
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(0.0, 0.0, 30.0),
    );
    let second = fillet_rolling_ball(&mut topo, first, &[edge], 1.0).unwrap();
    let step = brepkit_io::step::write_step(&topo, &[second]).unwrap();
    let mut imported = Topology::new();
    let imported_solid = brepkit_io::step::read_step(&step, &mut imported).unwrap()[0];
    assert!(
        brepkit_operations::validate::validate_solid(&imported, imported_solid)
            .unwrap()
            .is_valid()
    );
    let patch = |topo: &Topology, solid| {
        solid_faces(topo, solid)
            .unwrap()
            .into_iter()
            .filter_map(|fid| {
                let face = topo.face(fid).unwrap();
                if let FaceSurface::Nurbs(s) = face.surface() {
                    Some((s.clone(), face.is_reversed()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    };
    let source_patches = patch(&topo, second);
    let imported_patches = patch(&imported, imported_solid);
    assert_eq!(source_patches.len(), 1);
    assert_eq!(imported_patches.len(), 1);
    let (a, ar) = &source_patches[0];
    let (b, br) = &imported_patches[0];
    assert_eq!(ar, br);
    let mut deviation = 0.0_f64;
    for i in 0..=100 {
        for j in 0..=100 {
            let (u, v) = (f64::from(i) / 100.0, f64::from(j) / 100.0);
            deviation = deviation.max((a.evaluate(u, v) - b.evaluate(u, v)).length());
        }
    }
    assert_eq!(a.degree_u(), b.degree_u());
    assert_eq!(a.degree_v(), b.degree_v());
    assert_eq!(a.knots_u(), b.knots_u());
    assert_eq!(a.knots_v(), b.knots_v());
    assert_eq!(a.weights(), b.weights());
    let mut curve_deviation = 0.0_f64;
    for edge in solid_edges(&topo, second).unwrap() {
        let e = topo.edge(edge).unwrap();
        if !matches!(e.curve(), EdgeCurve::NurbsCurve(_)) {
            continue;
        }
        let p = topo.vertex(e.start()).unwrap().point();
        let q = topo.vertex(e.end()).unwrap().point();
        let other = imported
            .edge(edge_at(&imported, imported_solid, p, q))
            .unwrap();
        let op = imported.vertex(other.start()).unwrap().point();
        let oq = imported.vertex(other.end()).unwrap().point();
        for i in 0..=100 {
            let t = f64::from(i) / 100.0;
            let ot = if (p - op).length() < Tolerance::new().linear {
                t
            } else {
                1.0 - t
            };
            curve_deviation = curve_deviation.max(
                (e.curve().evaluate_with_endpoints(t, p, q)
                    - other.curve().evaluate_with_endpoints(ot, op, oq))
                .length(),
            );
        }
    }
    assert!(curve_deviation < Tolerance::new().linear);
    eprintln!("N339 STEP surface deviation={deviation:e}, curve deviation={curve_deviation:e}");
    for deflection in [0.01, 0.001, 0.0001] {
        let before = solid_volume(&topo, second, deflection).unwrap();
        let after = solid_volume(&imported, imported_solid, deflection).unwrap();
        eprintln!(
            "N339 STEP deflection={deflection}: {before} -> {after}, delta={}",
            (before - after).abs()
        );
    }
    assert!(deviation < Tolerance::new().linear);
    // OttoCAD's unchanged canonical bound: 1e-6 mm × bounding-box surface area.
    assert!(
        (solid_volume(&topo, second, 0.01).unwrap()
            - solid_volume(&imported, imported_solid, 0.01).unwrap())
        .abs()
            <= 1e-6 * 5900.0
    );
}

#[test]
fn third_mixed_radius_step_preserves_both_rational_patches_and_actual_radii() {
    let mut topo = Topology::new();
    let source = source(&mut topo);
    let first_edge = edge_at(
        &topo,
        source,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(40.0, 0.0, 0.0),
    );
    let first = fillet_rolling_ball(&mut topo, source, &[first_edge], 2.54).unwrap();
    let second_edge = edge_at(
        &topo,
        first,
        Point3::new(0.0, 0.0, 2.54),
        Point3::new(0.0, 0.0, 30.0),
    );
    let second = fillet_rolling_ball(&mut topo, first, &[second_edge], 2.54).unwrap();
    let third_edge = edge_at(
        &topo,
        second,
        Point3::new(0.0, 5.08, 0.0),
        Point3::new(0.0, 25.0, 0.0),
    );
    let third = fillet_rolling_ball(&mut topo, second, &[third_edge], 1.27).unwrap();
    assert!(
        brepkit_operations::validate::validate_solid(&topo, third)
            .unwrap()
            .is_valid()
    );

    let step = brepkit_io::step::write_step(&topo, &[third]).unwrap();
    let mut imported = Topology::new();
    let imported_third = brepkit_io::step::read_step(&step, &mut imported).unwrap()[0];
    let imported_validation =
        brepkit_operations::validate::validate_solid(&imported, imported_third).unwrap();
    assert!(
        imported_validation.is_valid(),
        "{:?}",
        imported_validation.issues
    );
    assert_eq!(
        (
            solid_vertices(&topo, third).unwrap().len(),
            solid_edges(&topo, third).unwrap().len(),
            solid_faces(&topo, third).unwrap().len(),
        ),
        (16, 25, 11)
    );
    assert_eq!(
        (
            solid_vertices(&imported, imported_third).unwrap().len(),
            solid_edges(&imported, imported_third).unwrap().len(),
            solid_faces(&imported, imported_third).unwrap().len(),
        ),
        (16, 25, 11)
    );

    let patches = |topology: &Topology, solid| {
        let mut patches = solid_faces(topology, solid)
            .unwrap()
            .into_iter()
            .filter_map(|face_id| {
                let face = topology.face(face_id).unwrap();
                if let FaceSurface::Nurbs(surface) = face.surface() {
                    Some((
                        (surface.degree_u(), surface.degree_v()),
                        surface.clone(),
                        face.is_reversed(),
                    ))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        patches.sort_by_key(|(degree, _, _)| *degree);
        patches
    };
    let source_patches = patches(&topo, third);
    let imported_patches = patches(&imported, imported_third);
    assert_eq!(source_patches.len(), 2);
    assert_eq!(imported_patches.len(), 2);
    assert_eq!(
        source_patches.iter().map(|item| item.0).collect::<Vec<_>>(),
        [(2, 3), (5, 5)]
    );

    let mut maximum_surface_deviation = 0.0_f64;
    for ((degree, before, before_reversed), (imported_degree, after, after_reversed)) in
        source_patches.iter().zip(&imported_patches)
    {
        assert_eq!(degree, imported_degree);
        assert_eq!(before_reversed, after_reversed);
        assert_eq!(before.knots_u(), after.knots_u());
        assert_eq!(before.knots_v(), after.knots_v());
        assert_eq!(before.weights(), after.weights());
        assert!(
            before
                .weights()
                .iter()
                .flatten()
                .any(|weight| (*weight - 1.0).abs() > f64::EPSILON)
        );
        for i in 0..=100 {
            for j in 0..=100 {
                let (u, v) = (f64::from(i) / 100.0, f64::from(j) / 100.0);
                maximum_surface_deviation = maximum_surface_deviation
                    .max((before.evaluate(u, v) - after.evaluate(u, v)).length());
            }
        }
    }

    let mut maximum_curve_deviation = 0.0_f64;
    let mut rational_boundaries = 0;
    for edge_id in solid_edges(&topo, third).unwrap() {
        let edge = topo.edge(edge_id).unwrap();
        if !matches!(edge.curve(), EdgeCurve::NurbsCurve(_)) {
            continue;
        }
        rational_boundaries += 1;
        let start = topo.vertex(edge.start()).unwrap().point();
        let end = topo.vertex(edge.end()).unwrap().point();
        let imported_edge_id = edge_at(&imported, imported_third, start, end);
        let imported_edge = imported.edge(imported_edge_id).unwrap();
        assert!(matches!(imported_edge.curve(), EdgeCurve::NurbsCurve(_)));
        let imported_start = imported.vertex(imported_edge.start()).unwrap().point();
        let imported_end = imported.vertex(imported_edge.end()).unwrap().point();
        let same_direction = (start - imported_start).length() < Tolerance::new().linear;
        let (source_lo, source_hi) = edge.curve().domain_with_endpoints(start, end);
        let (imported_lo, imported_hi) = imported_edge
            .curve()
            .domain_with_endpoints(imported_start, imported_end);
        for i in 0..=100 {
            let t = f64::from(i) / 100.0;
            let source_t = source_lo + (source_hi - source_lo) * t;
            let imported_fraction = if same_direction { t } else { 1.0 - t };
            let imported_t = imported_lo + (imported_hi - imported_lo) * imported_fraction;
            maximum_curve_deviation = maximum_curve_deviation.max(
                (edge.curve().evaluate_with_endpoints(source_t, start, end)
                    - imported_edge.curve().evaluate_with_endpoints(
                        imported_t,
                        imported_start,
                        imported_end,
                    ))
                .length(),
            );
        }
    }
    assert_eq!(rational_boundaries, 4);

    let mut source_radii: Vec<_> = solid_faces(&topo, third)
        .unwrap()
        .into_iter()
        .filter_map(|face_id| match topo.face(face_id).unwrap().surface() {
            FaceSurface::Cylinder(cylinder) => Some(cylinder.radius()),
            _ => None,
        })
        .collect();
    let mut imported_radii: Vec<_> = solid_faces(&imported, imported_third)
        .unwrap()
        .into_iter()
        .filter_map(|face_id| match imported.face(face_id).unwrap().surface() {
            FaceSurface::Cylinder(cylinder) => Some(cylinder.radius()),
            _ => None,
        })
        .collect();
    source_radii.sort_by(f64::total_cmp);
    imported_radii.sort_by(f64::total_cmp);
    assert_eq!(source_radii, [1.27, 2.54, 2.54]);
    assert_eq!(source_radii, imported_radii);

    for deflection in [0.01, 0.001, 0.0001] {
        let before = solid_volume(&topo, third, deflection).unwrap();
        let after = solid_volume(&imported, imported_third, deflection).unwrap();
        eprintln!(
            "N340 STEP deflection={deflection}: {before:.15} -> {after:.15}, delta={:.3e}",
            (before - after).abs()
        );
        assert!((before - after).abs() < 1.0e-9);
    }
    eprintln!(
        "N340 STEP patches=2 rational_boundaries={rational_boundaries} surface_deviation={maximum_surface_deviation:e} curve_deviation={maximum_curve_deviation:e}"
    );
    assert!(maximum_surface_deviation < Tolerance::new().linear);
    assert!(maximum_curve_deviation < Tolerance::new().linear);
}
