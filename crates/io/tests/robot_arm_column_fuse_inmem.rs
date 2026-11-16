//! Captured operands of the `robot-arm` playground example's `fuseAll`
//! (#2372): a base plate with rounded corners, a turret standing on it, and
//! a column with rounded corners that stands on the plate and runs into the
//! turret.
//!
//! With the base and the turret fused first, the plate's top has the
//! turret's foot as an opening. The column's foot crosses that opening's rim
//! on its two corner arcs, so two arc sections end on the rim. The rim was
//! split only where a line section met it: the arcs were dropped as loose
//! ends, the top stayed whole under the column, and the fuse fell back to a
//! mesh. With the column fused onto the plate first, no section ends on a
//! rim, and that order was exact: it gives the volume to compare with.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::compound_ops::fuse_all;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::compound::Compound;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn load(name: &str, topo: &mut Topology) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn operands(topo: &mut Topology) -> [SolidId; 3] {
    ["base", "turret", "column"].map(|part| load(&format!("robot_arm_{part}.bin"), topo))
}

fn fuse(topo: &mut Topology, a: SolidId, b: SolidId) -> SolidId {
    let exact = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    boolean_with_options(topo, BooleanOp::Fuse, a, b, exact).unwrap()
}

/// The valid, analytic solid's volume and its face count for each surface.
fn read(topo: &Topology, solid: SolidId, label: &str) -> (f64, [usize; 3]) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    let mut counts = [0; 3];
    for face in solid_faces(topo, solid).unwrap() {
        let slot = match topo.face(face).unwrap().surface().type_tag() {
            "plane" => 0,
            "cylinder" => 1,
            "cone" => 2,
            other => panic!("{label}: a {other} face"),
        };
        counts[slot] += 1;
    }
    (solid_volume(topo, solid, 0.001).unwrap(), counts)
}

#[test]
fn the_column_fuses_onto_the_base_and_turret() {
    let mut topo = Topology::new();
    let [base, turret, column] = operands(&mut topo);
    let column_first = fuse(&mut topo, base, column);
    let reference = fuse(&mut topo, column_first, turret);
    let (want, want_faces) = read(&topo, reference, "column first");

    let mut topo = Topology::new();
    let [base, turret, column] = operands(&mut topo);
    let base_and_turret = fuse(&mut topo, base, turret);
    let whole = fuse(&mut topo, base_and_turret, column);
    let (volume, faces) = read(&topo, whole, "column last");
    assert_eq!(faces, want_faces, "faces as plane, cylinder, cone");
    assert!(
        (volume - want).abs() <= 1e-9 * want,
        "volume {volume}, want {want}"
    );
}

#[test]
fn the_examples_own_call_fuses_the_three() {
    let mut topo = Topology::new();
    let [base, turret, column] = operands(&mut topo);
    let column_first = fuse(&mut topo, base, column);
    let reference = fuse(&mut topo, column_first, turret);
    let (want, want_faces) = read(&topo, reference, "column first");

    let mut topo = Topology::new();
    let parts = operands(&mut topo);
    let compound = topo.add_compound(Compound::new(parts.to_vec()));
    let whole = fuse_all(&mut topo, compound).unwrap();
    // The one-pass fuse leaves the turret's faces in the pieces its own
    // seams cut, so it has more faces of the same kinds.
    let (volume, faces) = read(&topo, whole, "fuse_all");
    assert!(
        faces.iter().sum::<usize>() <= 2 * want_faces.iter().sum::<usize>(),
        "faces {faces:?}, of {want_faces:?}"
    );
    assert!(
        (volume - want).abs() <= 1e-9 * want,
        "volume {volume}, want {want}"
    );
}
