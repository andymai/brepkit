//! A nacelle cut open by a wedge, from operands captured from brepjs's
//! `turbofan-engine` playground example.
//!
//! The nacelle is the revolve of a face with one closed spline edge: sixteen
//! NURBS faces, four to a quarter turn, whose first meridian lies in the plane
//! `y = 0`. The wedge is the revolve of a rectangle through 130°. With a flat
//! face of the wedge in that plane, the plane meets the nacelle along the
//! meridian's own edges. The face-face phase drops such a section as being
//! the edge, and the edge-face phase skipped an edge lying in a face's plane
//! as a coplanar pair's to split, so the wedge's face went unsplit and the cut
//! lost its cap there: an open shell, and a mesh fallback of 37,792 facets
//! that took 55 s.
//!
//! Data: `revolve_seam_nacelle.bin`, `revolve_seam_wedge_at_seam.bin` (the
//! wedge from 0° to 130°) and `revolve_seam_wedge_off_seam.bin` (from 10° to
//! 140°).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn census(topo: &Topology, solid: SolidId) -> BTreeMap<&'static str, usize> {
    let mut m = BTreeMap::new();
    for f in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        *m.entry(topo.face(f).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    m
}

/// A point in the nacelle's wall at `degrees` round the axis: at radius 116
/// and z = 60, between the inner wall near 102 and the outer near 130.
fn in_the_wall(degrees: f64) -> Point3 {
    let a = degrees.to_radians();
    Point3::new(116.0 * a.cos(), 116.0 * a.sin(), 60.0)
}

/// The nacelle less the wedge from `from` to `from + 130` degrees: exact,
/// valid, with `nurbs` NURBS faces and a flat cap at each end of the gap.
fn cut_open(wedge: &str, from: f64, nurbs: usize) {
    let mut topo = Topology::new();
    let nacelle = load(&mut topo, "revolve_seam_nacelle.bin");
    let wedge = load(&mut topo, wedge);

    let before = boolean::mesh_fallback_count();
    let result = boolean::boolean(&mut topo, BooleanOp::Cut, nacelle, wedge).unwrap();
    assert_eq!(boolean::mesh_fallback_count(), before, "mesh fallback");
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(
        census(&topo, result),
        BTreeMap::from([("nurbs", nurbs), ("plane", 2)])
    );

    let class =
        |degrees: f64| classify_point(&topo, result, in_the_wall(degrees), 0.01, 1e-7).unwrap();
    for inside_gap in [5.0, 40.0, 65.0, 90.0, 125.0] {
        assert_eq!(
            class(from + inside_gap),
            PointClassification::Outside,
            "{inside_gap} degrees into the gap"
        );
    }
    for past_gap in [135.0, 180.0, 270.0, 355.0] {
        assert_eq!(
            class(from + past_gap),
            PointClassification::Inside,
            "{past_gap} degrees from the gap's start"
        );
    }
}

#[test]
fn a_wedge_off_the_seam_cuts_the_nacelle_open() {
    // Two quarters cut, two whole: 4 + 4 + 8 NURBS faces.
    cut_open("revolve_seam_wedge_off_seam.bin", 10.0, 16);
}

#[test]
fn a_wedge_with_a_face_in_the_seams_plane_cuts_the_nacelle_open() {
    // The first quarter gone, the second cut, two whole: 4 + 8 NURBS faces.
    cut_open("revolve_seam_wedge_at_seam.bin", 0.0, 12);
}
