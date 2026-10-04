//! STEP faces on a surface of linear extrusion (a curve swept along a
//! vector): an export from the reference kernel whose three curved walls are
//! cubic B-spline curves swept straight down, and a cylinder whose wall is
//! written as its rim circle swept along its axis and against it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_io::step::reader::read_step;
use brepkit_io::step::writer::write_step;
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

/// Reads the one solid in `step`, checks that it validates, meshes
/// watertight and encloses `volume` with its faces turned outward, and
/// returns the type tags of its faces' surfaces.
fn read_checked(step: &str, volume: f64, label: &str) -> Vec<&'static str> {
    let mut topo = Topology::new();
    let solids = read_step(step, &mut topo).unwrap();
    assert_eq!(solids.len(), 1, "{label}");
    let solid = solids[0];
    let report = validate_solid(&topo, solid).unwrap();
    assert!(report.is_valid(), "{label}: {report:?}");
    let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{label}: open mesh");
    // The oriented volume integrates the mesh, so it agrees only to the
    // mesh's accuracy; a face turned inward would miss by far more.
    for (measured, within) in [
        (solid_volume(&topo, solid, 0.001).unwrap(), 1e-9),
        (oriented_solid_volume(&topo, solid, 0.001).unwrap(), 1e-3),
    ] {
        assert!(
            (measured - volume).abs() < within * volume,
            "{label}: {measured} vs {volume}"
        );
    }
    solid_faces(&topo, solid)
        .unwrap()
        .iter()
        .map(|&f| topo.face(f).unwrap().surface().type_tag())
        .collect()
}

#[test]
fn swept_spline_walls_from_the_reference_kernel_read_at_its_volume() {
    // The reference kernel measures this solid at 3596.3787629512553.
    let tags = read_checked(
        include_str!("data/swept_wall_cut.step"),
        3_596.378_762_951_255,
        "swept walls",
    );
    assert_eq!(tags.len(), 16);
    assert_eq!(
        tags.iter().filter(|&&t| t == "nurbs").count(),
        3,
        "{tags:?}"
    );
}

/// A cylinder of radius 2 and height 5, written with its wall as its rim
/// circle swept along the axis, or swept against it with the wall's face
/// flag turned so that the face still points out.
fn cylinder_as_swept_circle(against: bool) -> String {
    let mut topo = Topology::new();
    let cylinder = make_cylinder(&mut topo, 2.0, 5.0).unwrap();
    let step = write_step(&topo, &[cylinder]).unwrap();
    let record = |id: &str| {
        step.lines()
            .find(|l| l.starts_with(&format!("{id} =")))
            .unwrap()
    };
    let refs = |line: &str| -> Vec<String> {
        line.split('#')
            .skip(2)
            .map(|p| {
                format!(
                    "#{}",
                    p.split(|c: char| !c.is_ascii_digit()).next().unwrap()
                )
            })
            .collect()
    };
    let next = step
        .lines()
        .filter_map(|l| l.strip_prefix('#')?.split_once(' ')?.0.parse::<u64>().ok())
        .max()
        .unwrap()
        + 1;
    let wall = step
        .lines()
        .find(|l| l.contains("= CYLINDRICAL_SURFACE("))
        .unwrap();
    let id = wall.split(' ').next().unwrap();
    let axis = refs(wall).remove(0);
    let radius = wall[wall.rfind(',').unwrap() + 1..wall.rfind(')').unwrap()].trim();
    let direction = record(&refs(record(&axis))[1]).to_string();
    let inner = &direction[direction.rfind('(').unwrap() + 1..direction.find(')').unwrap()];
    let sign = if against { -1.0 } else { 1.0 };
    let along: Vec<String> = inner
        .split(',')
        .map(|x| format!("{:?}", sign * x.trim().parse::<f64>().unwrap()))
        .collect();
    let (circle, dir, vector) = (next, next + 1, next + 2);
    let mut out = step.replace(
        wall,
        &format!("{id} = SURFACE_OF_LINEAR_EXTRUSION('', #{circle}, #{vector});"),
    );
    if against {
        let face = step
            .lines()
            .find(|l| l.contains("ADVANCED_FACE(") && l.contains(&format!("{id}, .T.")))
            .unwrap();
        out = out.replace(face, &face.replace(".T.);", ".F.);"));
    }
    out.replacen(
        "ENDSEC;\nEND-ISO",
        &format!(
            "#{circle} = CIRCLE('', {axis}, {radius});\n\
             #{dir} = DIRECTION('', ({}));\n\
             #{vector} = VECTOR('', #{dir}, 1.);\nENDSEC;\nEND-ISO",
            along.join(", ")
        ),
        1,
    )
}

#[test]
fn a_circle_swept_along_its_axis_or_against_it_reads_as_the_cylinder() {
    for against in [false, true] {
        let step = cylinder_as_swept_circle(against);
        assert!(step.contains("SURFACE_OF_LINEAR_EXTRUSION") && !step.contains("CYLINDRICAL"));
        let tags = read_checked(&step, PI * 4.0 * 5.0, &format!("against {against}"));
        assert_eq!(tags.len(), 3, "{tags:?}");
        assert_eq!(
            tags.iter().filter(|&&t| t == "cylinder").count(),
            1,
            "{tags:?}"
        );
    }
}
