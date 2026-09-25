//! A solid with cavities goes to STEP as a `BREP_WITH_VOIDS`, each cavity an
//! `ORIENTED_CLOSED_SHELL` flagged false, and reads back with its cavities.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_io::step::reader::read_step;
use brepkit_io::step::writer::write_step;
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// The box `[-5, 5]³` less balls of radius 1.5 at `centres`.
fn box_with_cavities(topo: &mut Topology, centres: &[(f64, f64, f64)]) -> SolidId {
    let mut piece = make_box(topo, 10.0, 10.0, 10.0).unwrap();
    transform_solid(topo, piece, &Mat4::translation(-5.0, -5.0, -5.0)).unwrap();
    for &(x, y, z) in centres {
        let ball = make_sphere(topo, 1.5, 32).unwrap();
        transform_solid(topo, ball, &Mat4::translation(x, y, z)).unwrap();
        piece = boolean(topo, BooleanOp::Cut, piece, ball).unwrap();
    }
    piece
}

#[test]
fn cavities_survive_a_round_trip() {
    for centres in [
        vec![(0.0, 0.0, 0.0)],
        vec![(-2.0, 0.0, 0.0), (2.0, 0.5, -1.0)],
    ] {
        let label = format!("{} cavities", centres.len());
        #[allow(clippy::cast_precision_loss)]
        let truth = 4.0f64.mul_add(-PI * 1.125 * centres.len() as f64, 1000.0);
        let mut topo = Topology::new();
        let piece = box_with_cavities(&mut topo, &centres);
        assert_eq!(
            topo.solid(piece).unwrap().inner_shells().len(),
            centres.len(),
            "{label}: cavity shells"
        );
        let mut step = write_step(&topo, &[piece]).unwrap();
        assert!(
            step.contains("BREP_WITH_VOIDS("),
            "{label}: no BREP_WITH_VOIDS"
        );
        assert_eq!(
            step.lines()
                .filter(|l| l.contains("ORIENTED_CLOSED_SHELL(") && l.ends_with(".F.);"))
                .count(),
            centres.len(),
            "{label}: voids flagged false"
        );
        // Twice: the read-back solid writes the same structure again.
        for pass in 0..2 {
            let mut back = Topology::new();
            let solids = read_step(&step, &mut back).unwrap();
            assert_eq!(solids.len(), 1, "{label} pass {pass}: one solid");
            let solid = solids[0];
            assert_eq!(
                back.solid(solid).unwrap().inner_shells().len(),
                centres.len(),
                "{label} pass {pass}: cavities read back"
            );
            assert!(
                validate_solid(&back, solid).unwrap().is_valid(),
                "{label} pass {pass}: invalid"
            );
            let volume = solid_volume(&back, solid, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-9 * truth,
                "{label} pass {pass}: volume {volume}, truth {truth}"
            );
            let mesh = tessellate_solid(&back, solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label} pass {pass}: open mesh");
            step = write_step(&back, &[solid]).unwrap();
        }
    }
}

#[test]
fn a_solid_without_cavities_stays_a_manifold_brep() {
    let mut topo = Topology::new();
    let piece = box_with_cavities(&mut topo, &[]);
    let step = write_step(&topo, &[piece]).unwrap();
    assert!(step.contains("MANIFOLD_SOLID_BREP("));
    assert!(!step.contains("BREP_WITH_VOIDS("));
    assert!(
        !step.contains(",), #"),
        "a representation's item list ends without a comma"
    );
}
