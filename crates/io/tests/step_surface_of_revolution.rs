//! A `SURFACE_OF_REVOLUTION` reads as the exact surface its profile sweeps:
//! a cylinder or cone from a line, a sphere or torus from a circle in a plane
//! through the axis, and a rational B-spline surface from a B-spline profile.
//! Either way round the profile runs, the face keeps its side: each solid
//! below holds its volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "support/step_text.rs"]
mod step_text;

use std::f64::consts::PI;

use brepkit_io::step::reader::read_step;
use brepkit_io::step::writer::write_step;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::measure::oriented_solid_volume;
use brepkit_operations::primitives::{make_cone, make_cylinder, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;
use step_text::StepText;

/// `step` with its `kind` surface records swapped for a surface of
/// revolution about +Z of the profile curve `profile` writes, whose records
/// it numbers from the id it is given and returns with the curve's id.
/// `against` says the profile runs so that the revolution's normal opposes
/// the swapped surface's: each face on it then turns its sense over, so it
/// keeps facing out of the solid.
fn revolve_instead(
    step: &str,
    kind: &str,
    against: bool,
    profile: impl Fn(u64) -> (Vec<String>, u64),
) -> String {
    let next = step
        .lines()
        .filter_map(|l| l.strip_prefix('#')?.split(' ').next()?.parse::<u64>().ok())
        .max()
        .unwrap()
        + 1;
    let (mut records, curve) = profile(next);
    let base = next + records.len() as u64;
    records.push(format!("#{base} = CARTESIAN_POINT('',(0.,0.,0.));"));
    records.push(format!("#{} = DIRECTION('',(0.,0.,1.));", base + 1));
    records.push(format!(
        "#{} = AXIS1_PLACEMENT('',#{base},#{});",
        base + 2,
        base + 1
    ));
    let axis = base + 2;
    let mut swapped = Vec::new();
    let mut out: Vec<String> = step
        .lines()
        .map(|l| {
            if l.contains(&format!(" = {kind}(")) {
                let id = l.split(' ').next().unwrap();
                swapped.push(id.to_string());
                format!("{id} = SURFACE_OF_REVOLUTION('',#{curve},#{axis});")
            } else {
                l.to_string()
            }
        })
        .collect();
    assert!(!swapped.is_empty(), "no {kind} to swap");
    if against {
        for l in &mut out {
            for id in &swapped {
                for (from, to) in [(".T.);", ".F.);"), (".F.);", ".T.);")] {
                    let tail = format!("{id}, {from}");
                    if l.contains("ADVANCED_FACE(") && l.ends_with(&tail) {
                        let keep = l.len() - from.len();
                        *l = format!("{}{to}", &l[..keep]);
                        break;
                    }
                }
            }
        }
    }
    let end = out.iter().rposition(|l| l.trim() == "ENDSEC;").unwrap();
    out.splice(end..end, records);
    out.join("\n")
}

/// A line through `p` along `d`, numbered from `next`.
fn line(next: u64, p: [f64; 3], d: [f64; 3]) -> (Vec<String>, u64) {
    (
        vec![
            format!(
                "#{next} = CARTESIAN_POINT('',({:?},{:?},{:?}));",
                p[0], p[1], p[2]
            ),
            format!(
                "#{} = DIRECTION('',({:?},{:?},{:?}));",
                next + 1,
                d[0],
                d[1],
                d[2]
            ),
            format!("#{} = VECTOR('',#{},1.);", next + 2, next + 1),
            format!("#{} = LINE('',#{next},#{});", next + 3, next + 2),
        ],
        next + 3,
    )
}

/// A circle in the XZ plane about `center`, turning about `±Y`.
fn circle(next: u64, center: [f64; 3], r: f64, about_plus_y: bool) -> (Vec<String>, u64) {
    let y = if about_plus_y { 1.0 } else { -1.0 };
    (
        vec![
            format!(
                "#{next} = CARTESIAN_POINT('',({:?},{:?},{:?}));",
                center[0], center[1], center[2]
            ),
            format!("#{} = DIRECTION('',(0.,{y:?},0.));", next + 1),
            format!("#{} = DIRECTION('',(1.,0.,0.));", next + 2),
            format!(
                "#{} = AXIS2_PLACEMENT_3D('',#{next},#{},#{});",
                next + 3,
                next + 1,
                next + 2
            ),
            format!("#{} = CIRCLE('',#{},{r:?});", next + 4, next + 3),
        ],
        next + 4,
    )
}

fn export(build: impl Fn(&mut Topology) -> SolidId) -> String {
    let mut topo = Topology::new();
    let solid = build(&mut topo);
    write_step(&topo, &[solid]).unwrap()
}

/// Read `step`'s one solid, check its revolved face came in as `expected`,
/// and that it holds `volume` and, if `closed`, meshes closed.
fn assert_solid(step: &str, expected: &str, volume: f64, label: &str, closed: bool) {
    let mut topo = Topology::new();
    let solids = read_step(step, &mut topo).unwrap();
    assert_eq!(solids.len(), 1, "{label}: one solid");
    let solid = solids[0];
    let kinds: Vec<&str> = solid_faces(&topo, solid)
        .unwrap()
        .into_iter()
        .map(|f| topo.face(f).unwrap().surface().type_tag())
        .collect();
    assert!(kinds.contains(&expected), "{label}: faces {kinds:?}");
    let got = oriented_solid_volume(&topo, solid, 0.001).unwrap();
    assert!(
        (got - volume).abs() < 2e-3 * volume,
        "{label}: volume {got}, truth {volume}"
    );
    if closed {
        assert!(
            is_watertight(&tessellate_solid(&topo, solid, 0.01).unwrap()),
            "{label}: open mesh"
        );
    }
}

#[test]
fn a_line_along_the_axis_sweeps_a_cylinder() {
    let step = export(|t| make_cylinder(t, 4.0, 10.0).unwrap());
    // Up the wall, the profile crossed with the turn points into the axis.
    for d in [1.0, -1.0] {
        let step = revolve_instead(&step, "CYLINDRICAL_SURFACE", d > 0.0, |n| {
            line(n, [4.0, 0.0, 3.0], [0.0, 0.0, d])
        });
        assert_solid(
            &step,
            "cylinder",
            PI * 160.0,
            &format!("cylinder, line along {d}z"),
            true,
        );
    }
}

#[test]
fn a_line_through_the_axis_sweeps_a_cone() {
    let step = export(|t| make_cone(t, 4.0, 0.0, 10.0).unwrap());
    for d in [1.0, -1.0] {
        let step = revolve_instead(&step, "CONICAL_SURFACE", d > 0.0, |n| {
            line(n, [4.0, 0.0, 0.0], [-0.4 * d, 0.0, d])
        });
        assert_solid(
            &step,
            "cone",
            PI * 160.0 / 3.0,
            &format!("cone, line sense {d}"),
            true,
        );
    }
}

#[test]
fn a_line_through_the_axis_sweeps_the_cone_its_face_is_on() {
    // The line's own point lies on the far side of the apex, on the cone
    // the face does not use.
    let step = export(|t| make_cone(t, 4.0, 0.0, 10.0).unwrap());
    for d in [1.0, -1.0] {
        let step = revolve_instead(&step, "CONICAL_SURFACE", d > 0.0, |n| {
            line(n, [-2.0, 0.0, 15.0], [-0.4 * d, 0.0, d])
        });
        assert_solid(
            &step,
            "cone",
            PI * 160.0 / 3.0,
            &format!("far cone, sense {d}"),
            true,
        );
    }
}

/// A dome of radius 5 on a flat floor, its wall swept by a circle about the
/// origin in the XZ plane turning about `±Y`, with the seam from the
/// equator to the pole running along that circle itself.
fn dome_step(about_plus_y: bool) -> String {
    let mut s = StepText::new();
    let (equator_point, pole) = (s.vertex([5.0, 0.0, 0.0]), s.vertex([0.0, 0.0, 5.0]));
    let equator_circle = s.circle([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 5.0);
    let equator = s.edge(equator_point, equator_point, equator_circle);
    let y = if about_plus_y { 1.0 } else { -1.0 };
    let profile = s.circle([0.0; 3], [0.0, y, 0.0], [1.0, 0.0, 0.0], 5.0);
    // About -Y the circle turns from +X up to +Z; about +Y it turns down.
    let seam = s.add(format!(
        "EDGE_CURVE('',#{equator_point},#{pole},#{profile},{})",
        if about_plus_y { ".F." } else { ".T." }
    ));
    let (origin, up) = (s.point([0.0; 3]), s.direction([0.0, 0.0, 1.0]));
    let axis = s.add(format!("AXIS1_PLACEMENT('',#{origin},#{up})"));
    let surface = s.add(format!("SURFACE_OF_REVOLUTION('',#{profile},#{axis})"));
    let wall_loop = s.edge_loop(&[(equator, true), (seam, true), (seam, false)]);
    // Turning down about +Y the circle's tangent crossed with the turn
    // points out of the dome; turning up, into it.
    let wall = s.face(&[wall_loop], surface, about_plus_y);
    let floor_loop = s.edge_loop(&[(equator, false)]);
    let floor_plane = s.plane([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let floor = s.face(&[floor_loop], floor_plane, false);
    let solid = s.solid(&[floor, wall]);
    s.representation("ADVANCED_BREP_SHAPE_REPRESENTATION", &[solid]);
    s.finish()
}

/// Which way face `face` of `solid` faces at `point`.
fn facing(topo: &Topology, face: brepkit_topology::face::FaceId, point: Point3) -> Vec3 {
    let face = topo.face(face).unwrap();
    let normal = match face.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        surface => {
            let (u, v) = surface.project_point(point).unwrap();
            surface.normal(u, v)
        }
    };
    if face.is_reversed() { -normal } else { normal }
}

#[test]
fn a_circle_about_a_point_of_the_axis_sweeps_a_sphere() {
    // The circle sweeps the sphere twice, from either side of the axis, with
    // opposite normals; the seam along the circle says which side the dome
    // is on.
    for about_plus_y in [true, false] {
        let label = format!("sphere, circle about +y {about_plus_y}");
        let mut topo = Topology::new();
        let solids = read_step(&dome_step(about_plus_y), &mut topo).unwrap();
        let faces = solid_faces(&topo, solids[0]).unwrap();
        let wall = faces[1];
        assert_eq!(
            topo.face(wall).unwrap().surface().type_tag(),
            "sphere",
            "{label}"
        );
        let up = Vec3::new(0.0, 0.0, 1.0);
        assert!(
            facing(&topo, wall, Point3::new(0.0, 0.0, 5.0)).dot(up) > 0.99,
            "{label}: the dome faces in at its pole"
        );
        assert!(
            facing(&topo, faces[0], Point3::new(0.0, 0.0, 0.0)).dot(up) < -0.99,
            "{label}: the floor faces up"
        );
    }
}

#[test]
fn a_circle_off_the_axis_sweeps_a_torus() {
    let step = export(|t| make_torus(t, 10.0, 3.0, 32).unwrap());
    for about_plus_y in [true, false] {
        let step = revolve_instead(&step, "TOROIDAL_SURFACE", !about_plus_y, |n| {
            circle(n, [10.0, 0.0, 0.0], 3.0, about_plus_y)
        });
        assert_solid(
            &step,
            "torus",
            2.0 * PI * PI * 10.0 * 9.0,
            &format!("torus, circle about +y {about_plus_y}"),
            true,
        );
    }
}

/// A cylinder of radius 4 and height 10 whose wall is swept by a straight
/// B-spline from height `a` to `b`, bounded by its two rims with no seam.
fn spline_band_step(a: f64, b: f64) -> String {
    let mut s = StepText::new();
    let (p0, p1) = (s.point([4.0, 0.0, a]), s.point([4.0, 0.0, b]));
    let profile = s.add(format!(
        "B_SPLINE_CURVE_WITH_KNOTS('',1,(#{p0},#{p1}),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),\
         .UNSPECIFIED.)"
    ));
    let (origin, up) = (s.point([0.0; 3]), s.direction([0.0, 0.0, 1.0]));
    let axis = s.add(format!("AXIS1_PLACEMENT('',#{origin},#{up})"));
    let surface = s.add(format!("SURFACE_OF_REVOLUTION('',#{profile},#{axis})"));
    let (v0, v1) = (s.vertex([4.0, 0.0, 0.0]), s.vertex([4.0, 0.0, 10.0]));
    let c0 = s.circle([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 4.0);
    let c1 = s.circle([0.0, 0.0, 10.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 4.0);
    let (e0, e1) = (s.edge(v0, v0, c0), s.edge(v1, v1, c1));
    let floor_loop = s.edge_loop(&[(e0, false)]);
    let floor_plane = s.plane([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let floor = s.face(&[floor_loop], floor_plane, false);
    let (low, high) = (s.edge_loop(&[(e0, true)]), s.edge_loop(&[(e1, false)]));
    // Up the wall, the profile crossed with the turn points into the axis.
    let wall = s.face(&[low, high], surface, a > b);
    let roof_loop = s.edge_loop(&[(e1, true)]);
    let roof_plane = s.plane([0.0, 0.0, 10.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let roof = s.face(&[roof_loop], roof_plane, true);
    let solid = s.solid(&[floor, wall, roof]);
    s.representation("ADVANCED_BREP_SHAPE_REPRESENTATION", &[solid]);
    s.finish()
}

#[test]
fn a_spline_profile_sweeps_a_spline_surface() {
    for (a, b) in [(0.0, 10.0), (10.0, 0.0)] {
        let step = spline_band_step(a, b);
        let mut topo = Topology::new();
        let solid = read_step(&step, &mut topo).unwrap()[0];
        let wall = solid_faces(&topo, solid)
            .unwrap()
            .into_iter()
            .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_)))
            .expect("the wall is a spline surface");
        let FaceSurface::Nurbs(surface) = topo.face(wall).unwrap().surface() else {
            unreachable!()
        };
        // Every point of the surface lies on the cylinder it sweeps.
        for i in 0..=8 {
            for j in 0..=16 {
                let p = surface.evaluate(f64::from(i) / 8.0, f64::from(j) / 16.0);
                assert!(
                    (p.x().hypot(p.y()) - 4.0).abs() < 1e-12,
                    "profile {a}..{b}: {p:?} off the cylinder"
                );
            }
        }
        assert_solid(
            &step,
            "nurbs",
            PI * 160.0,
            &format!("spline, profile {a}..{b}"),
            false,
        );
    }
}
