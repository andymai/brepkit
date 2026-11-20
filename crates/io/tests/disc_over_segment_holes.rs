//! A plate with two holes, each a circular segment (an arc and its chord),
//! cut by a disc flush with the plate's underside whose rim runs along both
//! holes' arcs. On the underside the disc's rim is the two holes' arcs and
//! two short arcs between them, so the plate's face keeps its outline and
//! takes that circle as its hole.
//!
//! The rim's arcs end at the holes' vertices: read as a loop inside the face
//! it left the holes attached under it, edges used three times.
//!
//! The operands go through a STEP file, as a part read from one.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use std::f64::consts::PI;

const R: f64 = 20.0;
const CHORD: f64 = 15.0;
const W: f64 = 100.0;
const H: f64 = 20.0;
const DEPTH: f64 = 5.0;

/// The circle's part beyond a chord `CHORD` from its centre.
fn segment_area() -> f64 {
    (CHORD / R)
        .acos()
        .mul_add(R * R, -CHORD * (R * R - CHORD * CHORD).sqrt())
}

/// The plate, a `W` square, less the holes, cut by the disc.
fn cut(turn: f64, seam: f64, slots: bool) -> (u64, Topology, brepkit_topology::solid::SolidId) {
    let mut topo = Topology::new();
    let before = mesh_fallback_count();
    let mut plate = make_box(&mut topo, W, W, H).unwrap();
    let place = Mat4::translation(-W / 2.0, -W / 2.0, 0.0);
    transform_solid(&mut topo, plate, &place).unwrap();
    for side in [1.0, -1.0] {
        let drum = make_cylinder(&mut topo, R, H + 2.0).unwrap();
        let at = Mat4::translation(0.0, 0.0, -1.0) * Mat4::rotation_z(seam);
        transform_solid(&mut topo, drum, &at).unwrap();
        let half = make_box(&mut topo, 2.0 * R + 2.0, R, H + 4.0).unwrap();
        let y0 = if side > 0.0 { CHORD } else { -CHORD - R };
        transform_solid(&mut topo, half, &Mat4::translation(-R - 1.0, y0, -2.0)).unwrap();
        let hole = boolean(&mut topo, BooleanOp::Intersect, drum, half).unwrap();
        plate = boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap();
    }
    let mut disc = make_cylinder(&mut topo, R, DEPTH).unwrap();
    transform_solid(&mut topo, disc, &Mat4::rotation_z(seam)).unwrap();
    if slots {
        // The disc's own slots, segments of a circle a unit smaller beyond
        // the same chords: inside the plate's holes, they take nothing more.
        for side in [1.0, -1.0] {
            let drum = make_cylinder(&mut topo, R - 1.0, DEPTH + 2.0).unwrap();
            let at = Mat4::translation(0.0, 0.0, -1.0) * Mat4::rotation_z(seam + 0.5);
            transform_solid(&mut topo, drum, &at).unwrap();
            let half = make_box(&mut topo, 2.0 * R + 2.0, R, DEPTH + 4.0).unwrap();
            let y0 = if side > 0.0 { CHORD } else { -CHORD - R };
            transform_solid(&mut topo, half, &Mat4::translation(-R - 1.0, y0, -2.0)).unwrap();
            let slot = boolean(&mut topo, BooleanOp::Intersect, drum, half).unwrap();
            disc = boolean(&mut topo, BooleanOp::Cut, disc, slot).unwrap();
        }
    }
    let turned = Mat4::rotation_z(turn);
    transform_solid(&mut topo, plate, &turned).unwrap();
    transform_solid(&mut topo, disc, &turned).unwrap();
    // Through a file, as the operands of a part read from one.
    let text = brepkit_io::step::write_step(&topo, &[plate, disc]).unwrap();
    let mut topo = Topology::new();
    let read = brepkit_io::step::read_step(&text, &mut topo).unwrap();
    let (plate, disc) = (read[0], read[1]);
    assert_eq!(
        mesh_fallback_count(),
        before,
        "turn {turn}, seam {seam}, slots {slots}: the operands fell back"
    );
    let result = boolean(&mut topo, BooleanOp::Cut, plate, disc).unwrap();
    (mesh_fallback_count() - before, topo, result)
}

#[test]
fn a_disc_whose_rim_runs_along_segment_holes_cuts_its_circle() {
    let seg = segment_area();
    for (turn, seam, slots) in
        (0..12).map(|k| ([0.0, 0.7][k % 2], [0.0, 1.0, 3.0][k / 2 % 3], k >= 6))
    {
        let plate = W * W;
        let want = plate.mul_add(H, -2.0 * seg * H) - PI.mul_add(R * R, -2.0 * seg) * DEPTH;
        // The underside: the plate less the disc.
        let floor = PI.mul_add(-R * R, plate);
        {
            let tag = format!("turn {turn}, seam {seam}, slots {slots}");
            let (fallbacks, topo, result) = cut(turn, seam, slots);
            assert_eq!(fallbacks, 0, "{tag}: mesh fallback");
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{tag}: {:?}", report.issues);
            let volume = solid_volume(&topo, result, 0.001).unwrap();
            assert!(
                (volume - want).abs() < 1e-6 * want,
                "{tag}: volume {volume}, want {want}"
            );
            let faces = solid_faces(&topo, result).unwrap();
            let underside: f64 = faces
                .iter()
                .filter(|&&f| {
                    let face = topo.face(f).unwrap();
                    matches!(face.surface(), brepkit_topology::face::FaceSurface::Plane { normal, d }
                        if (normal.z().abs() - 1.0).abs() < 1e-9 && d.abs() < 1e-9)
                })
                .map(|&f| face_area(&topo, f, 0.001).unwrap())
                .sum();
            assert!(
                (underside - floor).abs() < 1e-4 * floor,
                "{tag}: underside area {underside}, want {floor}"
            );
            let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
            assert!(is_watertight(&mesh), "{tag}: open mesh");
        }
    }
}
