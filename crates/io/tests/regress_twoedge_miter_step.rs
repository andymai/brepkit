//! Complete STEP roundtrip of the sharp-mitered two-edge corner, on both the
//! two-edge and the four-edge selection. The exported and re-imported solid
//! must keep the construction's exact V/E/F counts, every stripe cylinder's
//! exact radius, every crease's exact position on both adjacent cylinders,
//! and full solid validity, rather than merely round-tripping an isolated
//! curve.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces, solid_vertices};
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
    brepkit_operations::extrude::extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), size).unwrap()
}

fn edge_at(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let tol = Tolerance::new().linear;
    let matches: Vec<_> = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&id| {
            let e = topo.edge(id).unwrap();
            let p = topo.vertex(e.start()).unwrap().point();
            let q = topo.vertex(e.end()).unwrap().point();
            ((p - a).length() < tol && (q - b).length() < tol)
                || ((p - b).length() < tol && (q - a).length() < tol)
        })
        .collect();
    assert_eq!(matches.len(), 1, "unique target {a:?}-{b:?}");
    matches[0]
}

/// Roundtrip one two-edge (`count=2`) or four-edge (`count=4`) construction
/// through STEP and check it fully, on the *re-imported* solid: exact V/E/F,
/// exact stripe cylinder radius, and every crease edge still lying at exact
/// perpendicular distance r from both its owning cylinders' axes (the same
/// independent geometric check `regress_fillet_miter_corner_matrix.rs`'s
/// oracle runs on the native in-memory solid). That is what proves the STEP
/// writer and reader preserve the crease as the same physical curve, not
/// merely a solid that happens to still validate topologically.
fn roundtrip_and_check(size: f64, fraction: f64, count: usize) {
    let mut topo = Topology::new();
    let input = source(&mut topo, size);
    let corners = [
        Point3::new(0., 0., size),
        Point3::new(size, 0., size),
        Point3::new(size, size, size),
        Point3::new(0., size, size),
    ];
    let selected: Vec<_> = (0..count)
        .map(|i| edge_at(&topo, input, corners[i], corners[(i + 1) % 4]))
        .collect();
    let result = fillet_v2(&mut topo, input, &selected, size * fraction).unwrap();
    assert_eq!(result.succeeded.len(), count);
    assert!(result.failed.is_empty());

    let step = brepkit_io::step::write_step(&topo, &[result.solid]).unwrap();
    let mut imported = Topology::new();
    let imported_solid = brepkit_io::step::read_step(&step, &mut imported).unwrap()[0];

    assert!(
        brepkit_operations::validate::validate_solid(&imported, imported_solid)
            .unwrap()
            .is_valid(),
        "size={size} fraction={fraction} count={count}: imported solid must be valid"
    );

    let r = size * fraction;
    let tol = size * 1e-6; // STEP text round-trips through a fixed-precision writer.
    let vertices = solid_vertices(&imported, imported_solid).unwrap();
    let edges = solid_edges(&imported, imported_solid).unwrap();
    let faces = solid_faces(&imported, imported_solid).unwrap();
    assert_eq!(
        (vertices.len(), edges.len(), faces.len()),
        if count == 2 {
            (11, 17, 8)
        } else {
            (12, 20, 10)
        },
        "size={size} fraction={fraction} count={count}: imported V/E/F must be preserved"
    );

    let mut cylinders = Vec::new();
    for &fid in &faces {
        let face = imported.face(fid).unwrap();
        assert!(!matches!(face.surface(), FaceSurface::Torus(_)));
        assert!(!matches!(face.surface(), FaceSurface::Nurbs(_)));
        if let FaceSurface::Cylinder(c) = face.surface() {
            assert!(
                (c.radius() - r).abs() < tol,
                "imported stripe cylinder radius drifted: {} vs {r}",
                c.radius()
            );
            cylinders.push((fid, c.clone()));
        }
    }
    assert_eq!(cylinders.len(), if count == 2 { 2 } else { 4 });

    let mut crease_count = 0;
    for &edge in &edges {
        let e = imported.edge(edge).unwrap();
        let EdgeCurve::Ellipse(_) = e.curve() else {
            continue;
        };
        crease_count += 1;
        let owners: Vec<_> = faces
            .iter()
            .copied()
            .filter(|&f| {
                imported
                    .wire(imported.face(f).unwrap().outer_wire())
                    .unwrap()
                    .edges()
                    .iter()
                    .any(|oe| oe.edge() == edge)
            })
            .collect();
        assert_eq!(
            owners.len(),
            2,
            "crease {edge:?} must have two owning faces after import"
        );
        let cyls: Vec<_> = owners
            .iter()
            .filter_map(|&f| match imported.face(f).unwrap().surface() {
                FaceSurface::Cylinder(c) => Some(c.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            cyls.len(),
            2,
            "crease {edge:?} must be owned by two cylinders after import"
        );
        let a = imported.vertex(e.start()).unwrap().point();
        let b = imported.vertex(e.end()).unwrap().point();
        let (lo, hi) = e.curve().domain_with_endpoints(a, b);
        let dist_to_axis = |p: Point3, c: &brepkit_math::surfaces::CylindricalSurface| {
            let delta = p - c.origin();
            let along = delta.dot(c.axis());
            (delta - c.axis() * along).length()
        };
        for i in 0..=16 {
            let t = lo + (hi - lo) * f64::from(i) / 16.;
            let p = e.curve().evaluate_with_endpoints(t, a, b);
            let d0 = dist_to_axis(p, &cyls[0]);
            let d1 = dist_to_axis(p, &cyls[1]);
            assert!(
                (d0 - r).abs() < tol,
                "size={size} fraction={fraction} count={count}: imported crease off cylinder 0 axis by {}",
                d0 - r
            );
            assert!(
                (d1 - r).abs() < tol,
                "size={size} fraction={fraction} count={count}: imported crease off cylinder 1 axis by {}",
                d1 - r
            );
        }
    }
    assert_eq!(
        crease_count,
        if count == 2 { 1 } else { 4 },
        "size={size} fraction={fraction} count={count}: imported crease count"
    );
}

#[test]
fn two_edge_miter_step_roundtrip_1mm() {
    roundtrip_and_check(1.0, 0.05, 2);
    roundtrip_and_check(1.0, 0.1, 2);
}

#[test]
fn four_edge_miter_step_roundtrip_1mm() {
    roundtrip_and_check(1.0, 0.05, 4);
    roundtrip_and_check(1.0, 0.1, 4);
}

#[test]
fn two_edge_miter_step_roundtrip_254mm() {
    roundtrip_and_check(254.0, 12.7 / 254.0, 2);
    roundtrip_and_check(254.0, 25.4 / 254.0, 2);
}

#[test]
fn four_edge_miter_step_roundtrip_254mm() {
    roundtrip_and_check(254.0, 12.7 / 254.0, 4);
    roundtrip_and_check(254.0, 25.4 / 254.0, 4);
}
