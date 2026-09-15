//! N339: STEP geometry and mass-property qualification of the reduced sequence.
#![allow(clippy::unwrap_used, clippy::print_stderr, deprecated)]

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::fillet::fillet_rolling_ball;
use brepkit_operations::measure::solid_volume;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;

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
    let wire = make_polygon_wire(
        &mut topo,
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
    let source =
        brepkit_operations::extrude::extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 30.0)
            .unwrap();
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
