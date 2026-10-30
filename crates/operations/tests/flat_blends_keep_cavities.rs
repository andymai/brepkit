//! The planar chamfer, the flat-bevel fillet and the variable-radius fillet
//! rebuild a solid's outer shell alone. A 10 cube with a 3 cube cavity keeps
//! its cavity through each, and each refuses a cut that would reach a cavity
//! 0.05 behind the front face.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::fillet::{FilletRadiusLaw, fillet_variable};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

/// A 10 cube less a 3 cube cavity at `corner`, and its top edge along x at
/// y = 0.
fn hollow_block(topo: &mut Topology, corner: (f64, f64, f64)) -> (SolidId, EdgeId) {
    let block = make_box(topo, 10.0, 10.0, 10.0).unwrap();
    let cube = make_box(topo, 3.0, 3.0, 3.0).unwrap();
    transform_solid(topo, cube, &Mat4::translation(corner.0, corner.1, corner.2)).unwrap();
    let hollow = boolean(topo, BooleanOp::Cut, block, cube).unwrap();
    assert_eq!(topo.solid(hollow).unwrap().inner_shells().len(), 1);
    let edge = solid_edges(topo, hollow)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                p.y().abs() < 1e-9 && (p.z() - 10.0).abs() < 1e-9
            })
        })
        .unwrap();
    (hollow, edge)
}

const CLEAR: (f64, f64, f64) = (3.5, 3.5, 3.5);
const NEAR: (f64, f64, f64) = (3.5, 0.05, 6.5);

/// The result keeps one valid cavity, and its volume.
fn kept(topo: &Topology, result: SolidId) -> f64 {
    assert_eq!(topo.solid(result).unwrap().inner_shells().len(), 1);
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    solid_volume(topo, result, 0.001).unwrap()
}

#[test]
fn a_chamfer_keeps_a_cavity() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, CLEAR);
    let result = chamfer(&mut topo, hollow, &[edge], 1.0).unwrap();
    let volume = kept(&topo, result);
    assert!((volume - (1000.0 - 27.0 - 5.0)).abs() < 1e-6, "{volume}");
}

#[test]
fn a_chamfer_that_would_reach_a_cavity_is_refused() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, NEAR);
    assert!(chamfer(&mut topo, hollow, &[edge], 1.0).is_err());
}

/// The hollow block loses what a plain block does to the same fillet.
#[test]
fn a_variable_fillet_keeps_a_cavity() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, CLEAR);
    let result =
        fillet_variable(&mut topo, hollow, &[(edge, FilletRadiusLaw::Constant(1.0))]).unwrap();
    let volume = kept(&topo, result);

    let mut plain_topo = Topology::new();
    let plain = make_box(&mut plain_topo, 10.0, 10.0, 10.0).unwrap();
    let plain_edge = solid_edges(&plain_topo, plain)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = plain_topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = plain_topo.vertex(v).unwrap().point();
                p.y().abs() < 1e-9 && (p.z() - 10.0).abs() < 1e-9
            })
        })
        .unwrap();
    let plain_result = fillet_variable(
        &mut plain_topo,
        plain,
        &[(plain_edge, FilletRadiusLaw::Constant(1.0))],
    )
    .unwrap();
    let plain_volume = solid_volume(&plain_topo, plain_result, 0.001).unwrap();
    assert!(plain_volume < 1000.0, "{plain_volume}");
    assert!(
        (volume + 27.0 - plain_volume).abs() < 1e-6,
        "{volume} vs {plain_volume}"
    );
}

#[test]
fn a_variable_fillet_that_would_reach_a_cavity_is_refused() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, NEAR);
    let law = FilletRadiusLaw::Linear {
        start: 0.5,
        end: 1.0,
    };
    assert!(fillet_variable(&mut topo, hollow, &[(edge, law)]).is_err());
}

/// The flat bevel takes off less than the chamfer's triangle across the
/// edge and more than nothing.
#[test]
#[allow(deprecated)]
fn a_bevel_fillet_keeps_a_cavity() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, CLEAR);
    let result = brepkit_operations::fillet::fillet(&mut topo, hollow, &[edge], 1.0).unwrap();
    let volume = kept(&topo, result);
    assert!((968.0..973.0).contains(&volume), "{volume}");
}

#[test]
#[allow(deprecated)]
fn a_bevel_fillet_that_would_reach_a_cavity_is_refused() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, NEAR);
    assert!(brepkit_operations::fillet::fillet(&mut topo, hollow, &[edge], 1.0).is_err());
}

/// The result's cavity is its own: moving the result in place leaves the
/// input's cavity where it was.
#[test]
fn a_kept_cavity_is_the_results_own() {
    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, CLEAR);
    let result = chamfer(&mut topo, hollow, &[edge], 1.0).unwrap();
    transform_solid(&mut topo, result, &Mat4::translation(0.0, 0.0, 100.0)).unwrap();
    let cavity = topo.solid(hollow).unwrap().inner_shells()[0];
    for &face in topo.shell(cavity).unwrap().faces() {
        for v in brepkit_topology::explorer::face_vertices(&topo, face).unwrap() {
            let z = topo.vertex(v).unwrap().point().z();
            assert!(
                (3.5 - 1e-9..=6.5 + 1e-9).contains(&z),
                "input cavity moved to z {z}"
            );
        }
    }
    let report = validate_solid(&topo, hollow).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
}

/// A box cavity's corner and size.
type Cavity = ((f64, f64, f64), (f64, f64, f64));

/// A 20 x 20 x 10 block, z from 0 to 10, with a 30 degree notch cut in
/// from its +x side to its centre, less a box cavity at `corner` of `size`;
/// and the notch's two top edges, which meet at a reflex corner over the
/// origin.
fn notched_block(topo: &mut Topology, cavity: Option<Cavity>) -> (SolidId, [EdgeId; 2]) {
    notched(topo, 15.0, -10.0, cavity)
}

/// A block from x = `left` to 10, y from -10 to 10 and z from 0 to 10, with
/// a notch of `half_angle` degrees either side of the x axis cut in from
/// its +x side to the origin, less an optional box cavity; and the notch's
/// two top edges.
fn notched(
    topo: &mut Topology,
    half_angle: f64,
    left: f64,
    cavity: Option<Cavity>,
) -> (SolidId, [EdgeId; 2]) {
    use brepkit_math::vec::Vec3;
    let t = 10.0 * half_angle.to_radians().tan();
    let outline: Vec<Point3> = [
        (left, -10.0),
        (10.0, -10.0),
        (10.0, -t),
        (0.0, 0.0),
        (10.0, t),
        (10.0, 10.0),
        (left, 10.0),
    ]
    .iter()
    .map(|&(x, y)| Point3::new(x, y, 0.0))
    .collect();
    let wire = brepkit_topology::builder::make_polygon_wire(topo, &outline, 1e-7).unwrap();
    let face = brepkit_topology::builder::make_face_from_wire(topo, wire).unwrap();
    let mut block =
        brepkit_operations::extrude::extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), 10.0).unwrap();
    if let Some((corner, size)) = cavity {
        let cube = make_box(topo, size.0, size.1, size.2).unwrap();
        transform_solid(topo, cube, &Mat4::translation(corner.0, corner.1, corner.2)).unwrap();
        block = boolean(topo, BooleanOp::Cut, block, cube).unwrap();
        assert_eq!(topo.solid(block).unwrap().inner_shells().len(), 1);
    }
    let along = |end: f64| {
        solid_edges(topo, block)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                let ends = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
                ends.iter()
                    .any(|p| (*p - Point3::new(0.0, 0.0, 10.0)).length() < 1e-9)
                    && ends
                        .iter()
                        .any(|p| (*p - Point3::new(10.0, end, 10.0)).length() < 1e-9)
            })
            .unwrap()
    };
    let edges = [along(-t), along(t)];
    (block, edges)
}

/// Chamfered at 1, the notch's two top edges trim the top face back to
/// x = -1 / sin 15 degrees, about -3.86, well past either edge's reach.
#[test]
fn a_chamfer_into_a_reflex_corner_is_bounded_by_its_trim() {
    let mut topo = Topology::new();
    let (plain, edges) = notched_block(&mut topo, None);
    let cut = chamfer(&mut topo, plain, &edges, 1.0).unwrap();
    let report = validate_solid(&topo, cut).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);

    // In the trimmed corner, beyond either edge's reach: refused.
    let mut topo = Topology::new();
    let (near, edges) = notched_block(&mut topo, Some(((-2.5, -0.1, 9.75), (0.3, 0.2, 0.15))));
    let refused = chamfer(&mut topo, near, &edges, 1.0).unwrap_err();
    assert!(refused.to_string().contains("cavity"), "{refused}");

    // Well clear of the corner: kept.
    let mut topo = Topology::new();
    let (clear, edges) = notched_block(&mut topo, Some(((-8.0, -0.5, 4.0), (1.0, 1.0, 1.0))));
    let cut = chamfer(&mut topo, clear, &edges, 1.0).unwrap();
    assert_eq!(topo.solid(cut).unwrap().inner_shells().len(), 1);
}

/// A 2 degree notch trims its top face back to x = -1 / sin 1 degree, about
/// -57.3, so near another chamfered edge at x = -60: a cavity at x = -20 in
/// the notch's long wedge refuses the chamfer.
#[test]
fn a_long_reflex_trim_is_bounded_beside_another_cut() {
    let left_edge = |topo: &Topology, solid: SolidId| {
        solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                [edge.start(), edge.end()].iter().all(|&v| {
                    let p = topo.vertex(v).unwrap().point();
                    (p.x() + 60.0).abs() < 1e-9 && (p.z() - 10.0).abs() < 1e-9
                })
            })
            .unwrap()
    };

    let mut topo = Topology::new();
    let (plain, [a, b]) = notched(&mut topo, 1.0, -60.0, None);
    let left = left_edge(&topo, plain);
    let cut = chamfer(&mut topo, plain, &[a, b, left], 1.0).unwrap();
    let report = validate_solid(&topo, cut).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);

    let mut topo = Topology::new();
    let (near, [a, b]) = notched(
        &mut topo,
        1.0,
        -60.0,
        Some(((-20.5, -0.05, 9.45), (1.0, 0.1, 0.25))),
    );
    let left = left_edge(&topo, near);
    let refused = chamfer(&mut topo, near, &[a, b, left], 1.0).unwrap_err();
    assert!(refused.to_string().contains("cavity"), "{refused}");
}

/// A block whose back rises into a tower with a sloped face on the plane
/// z = y + 9, extruded 10 along x, less an optional box cavity; and its top
/// front edge, whose 1 x 1 chamfer lies on that same plane at the front.
fn block_with_a_sloped_tower(topo: &mut Topology, cavity: Option<Cavity>) -> (SolidId, EdgeId) {
    use brepkit_math::vec::Vec3;
    let profile: Vec<Point3> = [
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 23.0),
        (14.0, 23.0),
        (12.0, 21.0),
        (12.0, 10.0),
        (0.0, 10.0),
    ]
    .iter()
    .map(|&(y, z)| Point3::new(0.0, y, z))
    .collect();
    let wire = brepkit_topology::builder::make_polygon_wire(topo, &profile, 1e-7).unwrap();
    let face = brepkit_topology::builder::make_face_from_wire(topo, wire).unwrap();
    let mut block =
        brepkit_operations::extrude::extrude(topo, face, Vec3::new(1.0, 0.0, 0.0), 10.0).unwrap();
    if let Some((corner, size)) = cavity {
        let cube = make_box(topo, size.0, size.1, size.2).unwrap();
        transform_solid(topo, cube, &Mat4::translation(corner.0, corner.1, corner.2)).unwrap();
        block = boolean(topo, BooleanOp::Cut, block, cube).unwrap();
        assert_eq!(topo.solid(block).unwrap().inner_shells().len(), 1);
    }
    let edge = solid_edges(topo, block)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                p.y().abs() < 1e-9 && (p.z() - 10.0).abs() < 1e-9
            })
        })
        .unwrap();
    (block, edge)
}

/// The chamfer face shares its plane with the tower's sloped face far
/// behind it, but is no part of it: a cavity across the chamfer is refused.
#[test]
fn a_cut_on_the_plane_of_a_remote_face_still_bounds_its_cavity() {
    let mut topo = Topology::new();
    let (plain, edge) = block_with_a_sloped_tower(&mut topo, None);
    let cut = chamfer(&mut topo, plain, &[edge], 1.0).unwrap();
    let report = validate_solid(&topo, cut).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);

    let mut topo = Topology::new();
    let (near, edge) =
        block_with_a_sloped_tower(&mut topo, Some(((4.0, 0.2, 9.0), (2.0, 0.6, 0.6))));
    let refused = chamfer(&mut topo, near, &[edge], 1.0).unwrap_err();
    assert!(refused.to_string().contains("cavity"), "{refused}");
}

/// A cavity face's registered pcurve (the UV trim a curved face is
/// meshed by) is kept, on the cavity's copy, through a chamfer elsewhere.
#[test]
fn a_kept_cavity_keeps_its_pcurves() {
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::{Point2, Vec2};
    use brepkit_topology::pcurve::PCurve;

    let mut topo = Topology::new();
    let (hollow, edge) = hollow_block(&mut topo, CLEAR);
    let cavity = topo.solid(hollow).unwrap().inner_shells()[0];
    let face = topo.shell(cavity).unwrap().faces()[0];
    let rim = brepkit_topology::explorer::face_edges(&topo, face).unwrap()[0];
    let line = Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap();
    topo.pcurves_mut()
        .set(rim, face, PCurve::new(Curve2D::Line(line), 0.0, 1.0));

    let cut = chamfer(&mut topo, hollow, &[edge], 1.0).unwrap();
    let kept = topo.solid(cut).unwrap().inner_shells()[0];
    assert_ne!(kept, cavity);
    let copied_face = topo.shell(kept).unwrap().faces()[0];
    assert_eq!(topo.pcurves().pcurves_for_face(copied_face).len(), 1);
}

/// The walking engine's chamfer and fillet keep a cavity of their own and
/// refuse to open one.
#[test]
fn walking_blends_keep_their_own_cavity() {
    use brepkit_operations::blend_ops::{chamfer_v2, fillet_v2};

    for chamfer in [true, false] {
        let mut topo = Topology::new();
        let (hollow, edge) = hollow_block(&mut topo, CLEAR);
        let blend = if chamfer {
            chamfer_v2(&mut topo, hollow, &[edge], 1.0, 1.0)
        } else {
            fillet_v2(&mut topo, hollow, &[edge], 1.0)
        };
        let result = brepkit_operations::blend_ops::all_blended(blend.unwrap()).unwrap();
        let volume = kept(&topo, result);
        let removed = if chamfer {
            0.5
        } else {
            1.0 - std::f64::consts::FRAC_PI_4
        };
        assert!(
            (volume - (1000.0 - 27.0 - 10.0 * removed)).abs() < 1e-2,
            "chamfer {chamfer}: {volume}"
        );
        let cavity = topo.solid(hollow).unwrap().inner_shells()[0];
        assert_ne!(topo.solid(result).unwrap().inner_shells(), &[cavity]);
        transform_solid(&mut topo, result, &Mat4::translation(0.0, 0.0, 100.0)).unwrap();
        for &face in topo.shell(cavity).unwrap().faces() {
            for v in brepkit_topology::explorer::face_vertices(&topo, face).unwrap() {
                let z = topo.vertex(v).unwrap().point().z();
                assert!(
                    (3.5 - 1e-9..=6.5 + 1e-9).contains(&z),
                    "input cavity at z {z}"
                );
            }
        }

        let mut topo = Topology::new();
        let (near, edge) = hollow_block(&mut topo, NEAR);
        let refused = if chamfer {
            chamfer_v2(&mut topo, near, &[edge], 1.0, 1.0)
        } else {
            fillet_v2(&mut topo, near, &[edge], 1.0)
        };
        assert!(
            refused
                .and_then(brepkit_operations::blend_ops::all_blended)
                .is_err(),
            "chamfer {chamfer}"
        );
    }
}

/// A cylinder of radius 10 and height 100 holding a 3 cube cavity near its
/// bottom, turned 45° about y: a fillet round its top rim keeps the cavity.
/// The untouched bottom cap's box meets the cavity's, but the cap is no
/// face the fillet made. The outer shell loses what a plain cylinder's
/// does.
#[test]
fn a_walking_fillet_far_from_a_cavity_keeps_it_on_a_turned_solid() {
    use brepkit_operations::blend_ops::{all_blended, fillet_v2};
    use brepkit_operations::primitives::make_cylinder;
    use brepkit_topology::solid::Solid;

    let build = |topo: &mut Topology, hollow: bool| {
        let mut solid = make_cylinder(topo, 10.0, 100.0).unwrap();
        if hollow {
            let cube = make_box(topo, 3.0, 3.0, 3.0).unwrap();
            transform_solid(topo, cube, &Mat4::translation(-1.5, -1.5, 3.5)).unwrap();
            solid = boolean(topo, BooleanOp::Cut, solid, cube).unwrap();
        }
        transform_solid(topo, solid, &Mat4::rotation_y(std::f64::consts::FRAC_PI_4)).unwrap();
        let rim = solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(c)
                    if c.center().x() > 50.0)
            })
            .unwrap();
        all_blended(fillet_v2(topo, solid, &[rim], 1.0).unwrap()).unwrap()
    };
    let shell_volume = |topo: &mut Topology, shell| {
        let alone = topo.add_solid(Solid::new(shell, Vec::new()));
        solid_volume(topo, alone, 0.001).unwrap()
    };
    let mut topo = Topology::new();
    let hollow = build(&mut topo, true);
    let cavities = topo.solid(hollow).unwrap().inner_shells().to_vec();
    assert_eq!(cavities.len(), 1);
    let outer = topo.solid(hollow).unwrap().outer_shell();
    let hollow_outer = shell_volume(&mut topo, outer);
    let cavity = shell_volume(&mut topo, cavities[0]);
    let mut plain_topo = Topology::new();
    let plain = build(&mut plain_topo, false);
    let plain = solid_volume(&plain_topo, plain, 0.001).unwrap();
    assert!(
        (hollow_outer - plain).abs() < 1e-6 * plain,
        "{hollow_outer} vs {plain}"
    );
    assert!((cavity - 27.0).abs() < 1e-9, "{cavity}");
}

/// A walking chamfer's result shares nothing with its input: moving it
/// leaves every vertex of the input box where it was.
#[test]
fn a_walking_chamfers_result_is_its_own() {
    use brepkit_operations::blend_ops::{all_blended, chamfer_v2};

    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let edge = solid_edges(&topo, block)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                p.y().abs() < 1e-9 && (p.z() - 10.0).abs() < 1e-9
            })
        })
        .unwrap();
    let corners = |topo: &Topology| -> Vec<Point3> {
        brepkit_topology::explorer::solid_vertices(topo, block)
            .unwrap()
            .iter()
            .map(|&v| topo.vertex(v).unwrap().point())
            .collect()
    };
    let before = corners(&topo);
    let cut = all_blended(chamfer_v2(&mut topo, block, &[edge], 1.0, 1.0).unwrap()).unwrap();
    transform_solid(&mut topo, cut, &Mat4::translation(0.0, 0.0, 100.0)).unwrap();
    assert_eq!(corners(&topo), before);
    let report = validate_solid(&topo, block).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
}

/// A cylinder of radius 5 and height 10 less a 0.4 cube cavity whose least
/// corner is `corner`, and its top rim.
fn hollow_cylinder(topo: &mut Topology, corner: (f64, f64, f64)) -> (SolidId, EdgeId) {
    let cyl = brepkit_operations::primitives::make_cylinder(topo, 5.0, 10.0).unwrap();
    let cube = make_box(topo, 0.4, 0.4, 0.4).unwrap();
    transform_solid(topo, cube, &Mat4::translation(corner.0, corner.1, corner.2)).unwrap();
    let hollow = boolean(topo, BooleanOp::Cut, cyl, cube).unwrap();
    assert_eq!(topo.solid(hollow).unwrap().inner_shells().len(), 1);
    let rim = solid_edges(topo, hollow)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(c)
                if c.center().z() > 9.0 && (c.radius() - 5.0).abs() < 1e-9)
        })
        .unwrap();
    (hollow, rim)
}

/// A 1 chamfer round a cylinder's top rim takes off the ring where
/// `r + z > 14`: it keeps a cavity at the cylinder's middle, and is refused
/// where a cavity just under the rim reaches into that ring.
#[test]
fn a_rim_chamfer_keeps_a_cavity_clear_of_it_and_refuses_one_it_reaches() {
    let mut topo = Topology::new();
    let (hollow, rim) = hollow_cylinder(&mut topo, (-0.2, -0.2, 4.8));
    let result = chamfer(&mut topo, hollow, &[rim], 1.0).unwrap();
    let volume = kept(&topo, result);
    // The ring is a unit right triangle swept about the axis, its centroid
    // at radius 14/3: 14π/3 by Pappus.
    let expected = std::f64::consts::PI * (250.0 - 14.0 / 3.0) - 0.064;
    assert!(
        (volume - expected).abs() < 1e-6 * expected,
        "{volume} vs {expected}"
    );

    let mut topo = Topology::new();
    let (near, rim) = hollow_cylinder(&mut topo, (4.2, -0.2, 9.2));
    assert!(chamfer(&mut topo, near, &[rim], 1.0).is_err());
}
