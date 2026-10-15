//! Cylinder and cone faces written without a seam edge, as many systems
//! write them: a full band bounded by its two rims alone, a pointed cone by
//! its base rim alone or with a vertex loop at its apex. The reader gives each
//! the seam brepkit's own primitives carry, so the solid is valid, holds its
//! volume, and meshes over its whole surface.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "support/step_text.rs"]
mod step_text;

use std::collections::HashSet;
use std::f64::consts::{FRAC_1_SQRT_2, PI};

use brepkit_io::step::reader::read_step;
use brepkit_math::chord::DEFAULT_ANGULAR_TOL;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{
    is_watertight, tessellate_solid, tessellate_solid_grouped_with_tolerance,
};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use step_text::StepText;

const R: f64 = 10.0;
const H: f64 = 30.0;

/// How a rim of radius `r` at height `z` is written: one closed circle with
/// its vertex at an angle, arcs between vertices at several angles, or one
/// closed rational B-spline circle starting at an angle.
#[derive(Clone)]
enum Rim {
    Circle(f64),
    Arcs(Vec<f64>),
    Spline(f64),
}

fn at(r: f64, z: f64, angle: f64) -> [f64; 3] {
    [r * angle.cos(), r * angle.sin(), z]
}

/// The edges of a rim, running counterclockwise about +Z.
fn rim_edges(s: &mut StepText, rim: &Rim, r: f64, z: f64) -> Vec<u64> {
    match rim {
        Rim::Circle(angle) => {
            let v = s.vertex(at(r, z, *angle));
            let c = s.circle([0.0, 0.0, z], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], r);
            vec![s.edge(v, v, c)]
        }
        Rim::Arcs(angles) => {
            let vs: Vec<u64> = angles.iter().map(|&a| s.vertex(at(r, z, a))).collect();
            let c = s.circle([0.0, 0.0, z], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], r);
            (0..vs.len())
                .map(|k| s.edge(vs[k], vs[(k + 1) % vs.len()], c))
                .collect()
        }
        Rim::Spline(angle) => {
            let v = s.vertex(at(r, z, *angle));
            let points: Vec<String> = (0..9)
                .map(|k| {
                    let reach = if k % 2 == 0 { r } else { r * 2f64.sqrt() };
                    let p = s.point(at(reach, z, angle + f64::from(k) * PI / 4.0));
                    format!("#{p}")
                })
                .collect();
            let w = FRAC_1_SQRT_2;
            let curve = s.add(format!(
                "( BOUNDED_CURVE() B_SPLINE_CURVE(2,({}),.UNSPECIFIED.,.T.,.F.) \
                 B_SPLINE_CURVE_WITH_KNOTS((3,2,2,2,3),(0.,0.25,0.5,0.75,1.),.UNSPECIFIED.) \
                 CURVE() GEOMETRIC_REPRESENTATION_ITEM() \
                 RATIONAL_B_SPLINE_CURVE((1.,{w:?},1.,{w:?},1.,{w:?},1.,{w:?},1.)) \
                 REPRESENTATION_ITEM('') )",
                points.join(",")
            ));
            vec![s.edge(v, v, curve)]
        }
    }
}

fn forward(edges: &[u64]) -> Vec<(u64, bool)> {
    edges.iter().map(|&e| (e, true)).collect()
}

fn backward(edges: &[u64]) -> Vec<(u64, bool)> {
    edges.iter().rev().map(|&e| (e, false)).collect()
}

/// A solid of revolution about +Z: a flat bottom of radius `r0` at height 0,
/// a lateral face on `surface`, and a flat top of radius `r1` at `H`, or an
/// apex there when `r1` is zero (with a vertex loop if `apex_loop`). The
/// lateral face lists its rims as `bottom, top` or `top, bottom`.
#[allow(clippy::too_many_arguments)]
fn revolved_solid(
    s: &mut StepText,
    surface: u64,
    r0: f64,
    bottom: &Rim,
    r1: f64,
    top: &Rim,
    apex_loop: bool,
    top_first: bool,
) -> u64 {
    let low = rim_edges(s, bottom, r0, 0.0);
    let floor_loop = s.edge_loop(&backward(&low));
    let floor_plane = s.plane([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let floor = s.face(&[floor_loop], floor_plane, false);
    let low_loop = s.edge_loop(&forward(&low));
    if r1 == 0.0 {
        let mut loops = vec![low_loop];
        if apex_loop {
            let apex = s.vertex([0.0, 0.0, H]);
            loops.push(s.vertex_loop(apex));
        }
        let wall = s.face(&loops, surface, true);
        return s.solid(&[floor, wall]);
    }
    let high = rim_edges(s, top, r1, H);
    let high_loop = s.edge_loop(&backward(&high));
    let loops = if top_first {
        [high_loop, low_loop]
    } else {
        [low_loop, high_loop]
    };
    let wall = s.face(&loops, surface, true);
    let roof_loop = s.edge_loop(&forward(&high));
    let roof_plane = s.plane([0.0, 0.0, H], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let roof = s.face(&[roof_loop], roof_plane, true);
    s.solid(&[floor, wall, roof])
}

fn cylinder_step(bottom: &Rim, top: &Rim, top_first: bool) -> String {
    let mut s = StepText::new();
    let a = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let surface = s.add(format!("CYLINDRICAL_SURFACE('',#{a},{R:?})"));
    let solid = revolved_solid(&mut s, surface, R, bottom, R, top, false, top_first);
    s.representation("ADVANCED_BREP_SHAPE_REPRESENTATION", &[solid]);
    s.finish()
}

/// A cone of radius `R` at its base, narrowing to `r1` at `H`.
fn cone_step(r1: f64, apex_loop: bool) -> String {
    let mut s = StepText::new();
    let a = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let semi_angle = ((R - r1) / H).atan();
    let surface = s.add(format!("CONICAL_SURFACE('',#{a},{R:?},{:?})", -semi_angle));
    let solid = revolved_solid(
        &mut s,
        surface,
        R,
        &Rim::Circle(0.0),
        r1,
        &Rim::Circle(0.0),
        apex_loop,
        false,
    );
    s.representation("ADVANCED_BREP_SHAPE_REPRESENTATION", &[solid]);
    s.finish()
}

/// Read `step`'s one solid and check it against the solid of revolution
/// with bottom radius `R`, top radius `r1` and height `H`.
fn assert_revolved(step: &str, r1: f64, label: &str) {
    let mut topo = Topology::new();
    let solids = read_step(step, &mut topo).unwrap();
    assert_eq!(solids.len(), 1, "{label}: one solid");
    let solid = solids[0];

    let faces = solid_faces(&topo, solid).unwrap();
    let wall = faces
        .iter()
        .copied()
        .find(|&f| {
            matches!(
                topo.face(f).unwrap().surface(),
                FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
            )
        })
        .unwrap();
    let wall_face = topo.face(wall).unwrap();
    assert!(
        wall_face.inner_wires().is_empty(),
        "{label}: the rims make one loop"
    );
    let mut seen = HashSet::new();
    let seamed = topo
        .wire(wall_face.outer_wire())
        .unwrap()
        .edges()
        .iter()
        .any(|oe| !seen.insert(oe.edge()));
    assert!(seamed, "{label}: the wall runs up a seam and back");

    let report = validate_solid(&topo, solid).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");

    let truth = PI * H / 3.0 * r1.mul_add(R + r1, R * R);
    let volume = solid_volume(&topo, solid, 0.001).unwrap();
    assert!(
        (volume - truth).abs() < 1e-4 * truth,
        "{label}: volume {volume}, truth {truth}"
    );

    let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{label}: open mesh");

    // The wall's mesh lies on its surface and covers all of it.
    let deflection = 0.01;
    let (mesh, offsets) =
        tessellate_solid_grouped_with_tolerance(&topo, solid, deflection, DEFAULT_ANGULAR_TOL)
            .unwrap();
    let k = faces.iter().position(|&f| f == wall).unwrap();
    let mut area = 0.0;
    for t in mesh.indices[offsets[k] as usize..offsets[k + 1] as usize].chunks_exact(3) {
        let [a, b, c] = [0, 1, 2].map(|i| mesh.positions[t[i] as usize]);
        area += 0.5 * (b - a).cross(c - a).length();
        for p in [a, b, c] {
            let radial = p.x().hypot(p.y());
            let expected = (r1 - R).mul_add(p.z() / H, R);
            assert!(
                (radial - expected).abs() < 1e-6 && (-1e-9..=H + 1e-9).contains(&p.z()),
                "{label}: mesh vertex {p:?} off the wall"
            );
        }
    }
    let slant = (R - r1).hypot(H);
    let truth = PI * (R + r1) * slant;
    assert!(
        (area - truth).abs() < 2e-3 * truth,
        "{label}: wall mesh area {area}, truth {truth}"
    );
}

#[test]
fn a_band_bounded_by_its_rims_gets_a_seam() {
    for top_first in [false, true] {
        let step = cylinder_step(&Rim::Circle(0.0), &Rim::Circle(0.0), top_first);
        assert_revolved(&step, R, &format!("cylinder, top first {top_first}"));
    }
}

#[test]
fn rims_whose_vertices_miss_each_other_still_meet_on_a_seam() {
    let cases = [
        ("closed circles", Rim::Circle(0.0), Rim::Circle(PI / 2.0)),
        (
            "circle and arcs",
            Rim::Circle(0.3),
            Rim::Arcs(vec![PI / 4.0, 1.25 * PI]),
        ),
        (
            "arcs and arcs",
            Rim::Arcs(vec![0.0, PI]),
            Rim::Arcs(vec![PI / 4.0, 1.25 * PI]),
        ),
        (
            "arcs and a spline",
            Rim::Arcs(vec![0.0, PI]),
            Rim::Spline(PI / 2.0),
        ),
    ];
    for (label, bottom, top) in cases {
        assert_revolved(&cylinder_step(&bottom, &top, false), R, label);
    }
}

#[test]
fn a_cone_band_bounded_by_its_rims_gets_a_seam() {
    assert_revolved(&cone_step(4.0, false), 4.0, "frustum");
}

#[test]
fn a_pointed_cone_gets_a_seam_to_its_apex() {
    assert_revolved(&cone_step(0.0, false), 0.0, "pointed cone");
    assert_revolved(
        &cone_step(0.0, true),
        0.0,
        "pointed cone with a vertex loop",
    );
}

#[test]
fn a_seamed_band_reads_as_written() {
    let mut s = StepText::new();
    let a = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let surface = s.add(format!("CYLINDRICAL_SURFACE('',#{a},{R:?})"));
    let (v0, v1) = (s.vertex([R, 0.0, 0.0]), s.vertex([R, 0.0, H]));
    let c0 = s.circle([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], R);
    let c1 = s.circle([0.0, 0.0, H], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], R);
    let (e0, e1) = (s.edge(v0, v0, c0), s.edge(v1, v1, c1));
    let up = s.line([R, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let seam = s.edge(v0, v1, up);
    let floor_loop = s.edge_loop(&[(e0, false)]);
    let floor_plane = s.plane([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let floor = s.face(&[floor_loop], floor_plane, false);
    let wall_loop = s.edge_loop(&[(e0, true), (seam, true), (e1, false), (seam, false)]);
    let wall = s.face(&[wall_loop], surface, true);
    let roof_loop = s.edge_loop(&[(e1, true)]);
    let roof_plane = s.plane([0.0, 0.0, H], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let roof = s.face(&[roof_loop], roof_plane, true);
    let solid = s.solid(&[floor, wall, roof]);
    s.representation("ADVANCED_BREP_SHAPE_REPRESENTATION", &[solid]);
    let step = s.finish();

    assert_revolved(&step, R, "seamed cylinder");
    let mut topo = Topology::new();
    let solid = read_step(&step, &mut topo).unwrap()[0];
    let wall = solid_faces(&topo, solid).unwrap()[1];
    let edges = topo
        .wire(topo.face(wall).unwrap().outer_wire())
        .unwrap()
        .edges()
        .len();
    assert_eq!(edges, 4, "no second seam");
}
