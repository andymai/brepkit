//! A round along an edge between a flat wall and a vault: a box whose top
//! is part of a cylinder along x, bowed up or dished down. The wall along
//! the vault meets it on a straight edge, whose round ends on the two end
//! walls; each end wall's edge on the vault is an arc. The end wall's notch
//! took a corner with straight or B-spline edges only, so with an arc there
//! the end wall stayed whole and a flat patch closed the round's end,
//! lying on the end wall and facing into the solid: a solid the validator
//! accepted, with an open mesh. The notch now takes an arc corner edge.
//!
//! The arc bows past the straight line from the corner to the contact's
//! end, so the notch reaches past it too: a bore through the end walls in
//! that sliver is in the round's way, and the round is refused. So it is
//! where each end wall's edge on the vault is a B-spline that traces the
//! arc.
//!
//! The round along an end wall's arc is a torus band about the vault's
//! axis; it ends on the axial planes through the arc's ends.
//!
//! The round along the straight edge is a cylinder of the round's radius:
//! the ball's centre runs along a line parallel to the vault's axis. A wall
//! that rises above the vault meets it on a concave edge, whose round fills
//! the corner.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::{FRAC_PI_4, PI, TAU};

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_geometry::convert::circle_to_nurbs;
use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::surfaces::CylindricalSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::{all_blended, fillet_v2};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const SIDE: f64 = 10.0;
const HALF: f64 = SIDE / 2.0;

/// The top of the box: part of a cylinder of radius `radius` along x through
/// the top's two edges along x, bowed up or, if `dished`, down.
#[derive(Clone, Copy, Debug)]
struct Vault {
    radius: f64,
    dished: bool,
}

impl Vault {
    /// The height of the cylinder's axis.
    fn axis_z(self) -> f64 {
        let rise = (self.radius * self.radius - HALF * HALF).sqrt();
        if self.dished {
            SIDE + rise
        } else {
            SIDE - rise
        }
    }
}

/// The picked edge of the vault.
#[derive(Clone, Copy, Debug)]
enum Pick {
    /// The straight edge on the wall y = 0 or y = `SIDE`.
    Along { far: bool },
    /// The arc on the end wall x = 0 or x = `SIDE`.
    Across { far: bool },
}

fn vaulted_box(topo: &mut Topology, vault: Vault, pick: Pick) -> (SolidId, EdgeId) {
    let solid = make_box(topo, SIDE, SIDE, SIDE).unwrap();
    let axis_z = vault.axis_z();
    let top = solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&f| {
            matches!(topo.face(f).unwrap().surface(),
                FaceSurface::Plane { normal, d } if normal.z() > 0.5 && (d - SIDE).abs() < 1e-9)
        })
        .unwrap();
    let surface = CylindricalSurface::new(
        Point3::new(0.0, HALF, axis_z),
        Vec3::new(1.0, 0.0, 0.0),
        vault.radius,
    )
    .unwrap();
    topo.face_mut(top)
        .unwrap()
        .set_surface(FaceSurface::Cylinder(surface));
    if vault.dished {
        // A dish's material lies away from the axis: the face is reversed,
        // and its wire winds about the cylinder's normal, the other way.
        let outer = topo.face(top).unwrap().outer_wire();
        let backward: Vec<OrientedEdge> = topo
            .wire(outer)
            .unwrap()
            .edges()
            .iter()
            .rev()
            .map(|oriented| OrientedEdge::new(oriented.edge(), !oriented.is_forward()))
            .collect();
        let wire = topo.add_wire(Wire::new(backward, true).unwrap());
        let face = topo.face_mut(top).unwrap();
        face.set_outer_wire(wire);
        face.set_reversed(true);
    }
    let on_top = |p: Point3, q: Point3| (p.z() - SIDE).abs() < 1e-9 && (q.z() - SIDE).abs() < 1e-9;
    let ends = |topo: &Topology, edge: EdgeId| {
        let edge = topo.edge(edge).unwrap();
        (
            topo.vertex(edge.start()).unwrap().point(),
            topo.vertex(edge.end()).unwrap().point(),
        )
    };
    let edges = solid_edges(topo, solid).unwrap();
    for &edge_id in &edges {
        let (p, q) = ends(topo, edge_id);
        if on_top(p, q) && (p.x() - q.x()).abs() < 1e-9 {
            // An arc runs counter-clockwise about its normal: over the top
            // from y = SIDE to y = 0 about +x, and under it the other way.
            let over = if p.y() > q.y() { 1.0 } else { -1.0 };
            let normal = if vault.dished { -over } else { over };
            let arc = Circle3D::new(
                Point3::new(p.x(), HALF, axis_z),
                Vec3::new(normal, 0.0, 0.0),
                vault.radius,
            )
            .unwrap();
            topo.edge_mut(edge_id)
                .unwrap()
                .set_curve(EdgeCurve::Circle(arc));
        }
    }
    let edge = edges
        .into_iter()
        .find(|&e| {
            let (p, q) = ends(topo, e);
            let at = |far: bool| if far { SIDE } else { 0.0 };
            on_top(p, q)
                && match pick {
                    Pick::Along { far } => {
                        [p.y(), q.y()].iter().all(|y| (y - at(far)).abs() < 1e-9)
                    }
                    Pick::Across { far } => {
                        [p.x(), q.x()].iter().all(|x| (x - at(far)).abs() < 1e-9)
                    }
                }
        })
        .unwrap();
    (solid, edge)
}

/// Twice the area swept from the origin along a counter-clockwise or
/// clockwise arc about `centre`, the short way from `from` to `to`.
fn arc_sweep(centre: (f64, f64), from: (f64, f64), to: (f64, f64)) -> f64 {
    let radius = (from.0 - centre.0).hypot(from.1 - centre.1);
    let a0 = (from.1 - centre.1).atan2(from.0 - centre.0);
    let a1 = (to.1 - centre.1).atan2(to.0 - centre.0);
    let turn = (a1 - a0 + PI).rem_euclid(TAU) - PI;
    let a1 = a0 + turn;
    radius * radius * turn
        + radius * (centre.0 * (a1.sin() - a0.sin()) - centre.1 * (a1.cos() - a0.cos()))
}

/// The centre of a round of radius `r` in the section across the straight
/// edge on the wall y = 0: a radius off the wall, on the side the vault runs
/// to, and a radius off the vault, on the side the wall runs to. A wall that
/// rises above the vault (`concave`) runs to the other side of the vault
/// from a wall below it.
fn round_centre(vault: Vault, r: f64, concave: bool) -> (f64, f64) {
    let axis = (HALF, vault.axis_z());
    let reach = if vault.dished == concave {
        vault.radius - r
    } else {
        vault.radius + r
    };
    let up = (reach * reach - (r - HALF) * (r - HALF)).sqrt();
    (
        r,
        if vault.dished {
            axis.1 - up
        } else {
            axis.1 + up
        },
    )
}

/// The area in the section across the straight edge on the wall y = 0
/// between the wall, the round and the vault: the corner, along the wall
/// to the round's contact, along the round to its contact on the vault,
/// and back along the vault.
fn corner_area(vault: Vault, r: f64, concave: bool) -> f64 {
    let axis = (HALF, vault.axis_z());
    let centre = round_centre(vault, r, concave);
    let scale = vault.radius / (centre.0 - axis.0).hypot(centre.1 - axis.1);
    let on_vault = (
        axis.0 + (centre.0 - axis.0) * scale,
        axis.1 + (centre.1 - axis.1) * scale,
    );
    let (corner, on_wall) = ((0.0, SIDE), (0.0, centre.1));
    let twice = corner.0 * on_wall.1 - corner.1 * on_wall.0
        + arc_sweep(centre, on_wall, on_vault)
        + arc_sweep(axis, on_vault, corner);
    0.5 * twice.abs()
}

/// The volume a round of radius `r` takes off the vaulted box.
fn removed(vault: Vault, pick: Pick, r: f64) -> f64 {
    let big = vault.radius;
    match pick {
        Pick::Along { .. } => corner_area(vault, r, false) * SIDE,
        // A square corner turned about the vault's axis between the axial
        // planes through the arc's ends: the square less its quarter disc,
        // each at its own distance from the axis.
        Pick::Across { .. } => {
            let turn = 2.0 * (HALF / big).asin();
            let sign = if vault.dished { 1.0 } else { -1.0 };
            let square = r * r * (big + sign * r / 2.0);
            let disc = FRAC_PI_4 * r * r * (big + sign * (r - 4.0 * r / (3.0 * PI)));
            turn * (square - disc)
        }
    }
}

fn check(vault: Vault, pick: Pick, r: f64, pose: &Mat4) {
    let label = format!("{vault:?}, {pick:?}, r {r}");
    let mut topo = Topology::new();
    let (solid, edge) = vaulted_box(&mut topo, vault, pick);
    transform_solid(&mut topo, solid, pose).unwrap();
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let faces = solid_faces(&topo, solid).unwrap().len();
    let result = fillet_v2(&mut topo, solid, &[edge], r)
        .and_then(all_blended)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    for deflection in [0.05, 0.5] {
        let mesh = tessellate_solid(&topo, result, deflection).unwrap();
        assert!(
            is_watertight(&mesh),
            "{label}: mesh not watertight at {deflection}"
        );
    }
    let after = solid_faces(&topo, result).unwrap();
    assert_no_spline_faces(&topo, &after, &label);
    if matches!(pick, Pick::Along { .. }) {
        // The round, and each end wall notched: no patch over an end.
        assert_eq!(after.len(), faces + 1, "{label}: faces");
        assert_round_is_a_cylinder(&topo, &after, r, pose, &label);
        let walls_with_five_edges = after
            .iter()
            .filter(|&&f| {
                let face = topo.face(f).unwrap();
                face.surface().is_planar()
                    && topo.wire(face.outer_wire()).unwrap().edges().len() == 5
            })
            .count();
        assert_eq!(walls_with_five_edges, 2, "{label}: notched end walls");
    }
    let took = before - solid_volume(&topo, result, 1e-3).unwrap();
    let want = removed(vault, pick, r);
    // The round along the straight edge is exact; the torus band along an
    // arc is measured to its integrator's rounding.
    let within = if matches!(pick, Pick::Along { .. }) {
        1e-8
    } else {
        1e-4
    };
    assert!(
        (took - want).abs() <= within,
        "{label}: removed {took}, want {want}"
    );
    // A point in the removed corner is outside, and one under the round is
    // inside.
    if let Pick::Along { far } = pick {
        let y = |inset: f64| if far { SIDE - inset } else { inset };
        let at = |inset: f64, below: f64| pose.mul_point(Point3::new(HALF, y(inset), SIDE - below));
        let class =
            |p: Point3| classify_point(&topo, result, p, &ClassifyOptions::default()).unwrap();
        assert_eq!(
            class(at(0.02 * r, 0.02 * r)),
            PointClassification::Outside,
            "{label}: corner"
        );
        assert_eq!(
            class(at(r, 1.5 * r)),
            PointClassification::Inside,
            "{label}: under the round"
        );
    }
}

fn poses() -> [Mat4; 3] {
    [
        Mat4::identity(),
        Mat4::translation(3.0, -7.0, 2.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-1.1),
        Mat4::translation(-4.0, 2.0, 5.0) * Mat4::rotation_y(0.9) * Mat4::scale(1.0, -1.0, 1.0),
    ]
}

fn assert_no_spline_faces(topo: &Topology, faces: &[FaceId], label: &str) {
    for &face in faces {
        assert!(
            !matches!(topo.face(face).unwrap().surface(), FaceSurface::Nurbs(_)),
            "{label}: a B-spline face"
        );
    }
}

/// One face is a cylinder of radius `r` along the vault's axis, moved by
/// `pose`.
fn assert_round_is_a_cylinder(topo: &Topology, faces: &[FaceId], r: f64, pose: &Mat4, label: &str) {
    let along =
        pose.mul_point(Point3::new(1.0, 0.0, 0.0)) - pose.mul_point(Point3::new(0.0, 0.0, 0.0));
    let rounds = faces
        .iter()
        .filter(|&&face| {
            matches!(topo.face(face).unwrap().surface(),
                FaceSurface::Cylinder(cylinder) if (cylinder.radius() - r).abs() <= 1e-9
                    && cylinder.axis().cross(along).length() <= 1e-9)
        })
        .count();
    assert_eq!(
        rounds, 1,
        "{label}: a cylinder of radius {r} along the vault"
    );
}

const BOWED: Vault = Vault {
    radius: 13.0,
    dished: false,
};
const DISHED: Vault = Vault {
    radius: 13.0,
    dished: true,
};

#[test]
fn a_round_along_a_wall_under_a_vault() {
    for pose in poses() {
        for far in [false, true] {
            for r in [1.0, 2.5] {
                check(BOWED, Pick::Along { far }, r, &pose);
            }
        }
    }
}

#[test]
fn a_round_along_a_wall_under_a_steep_vault() {
    let steep = Vault {
        radius: 6.0,
        dished: false,
    };
    for pose in poses() {
        check(steep, Pick::Along { far: false }, 1.0, &pose);
    }
}

#[test]
fn a_round_along_a_wall_under_a_dish() {
    for pose in poses() {
        for far in [false, true] {
            for r in [1.0, 2.5] {
                check(DISHED, Pick::Along { far }, r, &pose);
            }
        }
    }
}

#[test]
fn a_round_along_an_end_walls_arc() {
    for pose in poses() {
        for far in [false, true] {
            for r in [1.0, 2.5] {
                check(BOWED, Pick::Across { far }, r, &pose);
            }
        }
    }
}

#[test]
fn a_round_along_a_dishs_end_arc() {
    for pose in poses() {
        for far in [false, true] {
            check(DISHED, Pick::Across { far }, 1.0, &pose);
        }
    }
}

/// How far the wall rises above the vault, and how thick it is.
const RISE: f64 = 4.0;
const THICK: f64 = 3.0;

/// The section across x of the vaulted box with a wall on its side y = 0
/// that rises `RISE` above the vault, swept `SIDE` along x, and the concave
/// edge where the wall meets the vault.
fn walled_vault(topo: &mut Topology, vault: Vault) -> (SolidId, EdgeId) {
    let mut at = |y: f64, z: f64| {
        let point = Point3::new(0.0, y, z);
        (point, topo.add_vertex(Vertex::new(point, 1e-7)))
    };
    // Counter-clockwise about +x: along the floor, up the far side, back
    // over the vault, up the wall and down its outside.
    let corners = [
        at(-THICK, 0.0),
        at(SIDE, 0.0),
        at(SIDE, SIDE),
        at(0.0, SIDE),
        at(0.0, SIDE + RISE),
        at(-THICK, SIDE + RISE),
    ];
    let edges: Vec<OrientedEdge> = (0..corners.len())
        .map(|i| {
            let ((p, start), (_, end)) = (corners[i], corners[(i + 1) % corners.len()]);
            let curve = if i == 2 {
                // Over the top about +x when bowed, under it when dished.
                let normal = if vault.dished { -1.0 } else { 1.0 };
                EdgeCurve::Circle(
                    Circle3D::new(
                        Point3::new(p.x(), HALF, vault.axis_z()),
                        Vec3::new(normal, 0.0, 0.0),
                        vault.radius,
                    )
                    .unwrap(),
                )
            } else {
                EdgeCurve::Line
            };
            OrientedEdge::new(topo.add_edge(Edge::new(start, end, curve)), true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let along = Vec3::new(1.0, 0.0, 0.0);
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: along,
            d: 0.0,
        },
    ));
    let solid = extrude(topo, face, along, SIDE).unwrap();
    let edge = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                p.y().abs() < 1e-9 && (p.z() - SIDE).abs() < 1e-9
            })
        })
        .unwrap();
    (solid, edge)
}

fn check_concave(vault: Vault, r: f64, pose: &Mat4) {
    let label = format!("{vault:?}, concave, r {r}");
    let mut topo = Topology::new();
    let (solid, edge) = walled_vault(&mut topo, vault);
    transform_solid(&mut topo, solid, pose).unwrap();
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let faces = solid_faces(&topo, solid).unwrap().len();
    let result = fillet_v2(&mut topo, solid, &[edge], r)
        .and_then(all_blended)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    for deflection in [0.05, 0.5] {
        let mesh = tessellate_solid(&topo, result, deflection).unwrap();
        assert!(
            is_watertight(&mesh),
            "{label}: mesh not watertight at {deflection}"
        );
    }
    let after = solid_faces(&topo, result).unwrap();
    assert_eq!(after.len(), faces + 1, "{label}: faces");
    assert_no_spline_faces(&topo, &after, &label);
    assert_round_is_a_cylinder(&topo, &after, r, pose, &label);
    let added = solid_volume(&topo, result, 1e-3).unwrap() - before;
    let want = corner_area(vault, r, true) * SIDE;
    assert!(
        (added - want).abs() <= 1e-8,
        "{label}: added {added}, want {want}"
    );
    // A point in the filled corner is inside, and the round's centre is
    // outside.
    let at = |y: f64, z: f64| pose.mul_point(Point3::new(HALF, y, z));
    let class = |p: Point3| classify_point(&topo, result, p, &ClassifyOptions::default()).unwrap();
    assert_eq!(
        class(at(0.02 * r, SIDE + 0.02 * r)),
        PointClassification::Inside,
        "{label}: corner"
    );
    let centre = round_centre(vault, r, true);
    assert_eq!(
        class(at(centre.0, centre.1)),
        PointClassification::Outside,
        "{label}: the round's centre"
    );
}

#[test]
fn a_round_where_a_wall_rises_from_a_vault() {
    for pose in poses() {
        for r in [1.0, 2.5] {
            check_concave(BOWED, r, &pose);
        }
    }
}

#[test]
fn a_round_where_a_wall_rises_from_a_dish() {
    for pose in poses() {
        for r in [1.0, 2.5] {
            check_concave(DISHED, r, &pose);
        }
    }
}

/// Gives each arc of the solid as the B-spline that traces it.
fn arcs_as_splines(topo: &mut Topology, solid: SolidId) {
    for edge_id in solid_edges(topo, solid).unwrap() {
        let edge = topo.edge(edge_id).unwrap();
        let EdgeCurve::Circle(circle) = edge.curve().clone() else {
            continue;
        };
        let (p, q) = (
            topo.vertex(edge.start()).unwrap().point(),
            topo.vertex(edge.end()).unwrap().point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(p, q);
        let spline = circle_to_nurbs(&circle, t0, t1).unwrap();
        topo.edge_mut(edge_id)
            .unwrap()
            .set_curve(EdgeCurve::NurbsCurve(spline));
    }
}

/// The bowed box with a bore of radius `bore` along x through both end
/// walls at (`y`, `z`), and its straight top edge on the wall y = 0. With
/// `splines`, each end wall's edge on the vault is a B-spline.
fn bored_vault(topo: &mut Topology, y: f64, z: f64, bore: f64, splines: bool) -> (SolidId, EdgeId) {
    let (solid, _) = vaulted_box(topo, BOWED, Pick::Along { far: false });
    if splines {
        arcs_as_splines(topo, solid);
    }
    let drill = make_cylinder(topo, bore, SIDE + 2.0).unwrap();
    let lay = Mat4::translation(-1.0, y, z) * Mat4::rotation_y(std::f64::consts::FRAC_PI_2);
    transform_solid(topo, drill, &lay).unwrap();
    let solid = boolean(topo, BooleanOp::Cut, solid, drill).unwrap();
    assert!(validate_solid(topo, solid).unwrap().is_valid());
    assert_eq!(solid_faces(topo, solid).unwrap().len(), 7, "bored box");
    let edge = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                p.y().abs() < 1e-9 && (p.z() - SIDE).abs() < 1e-9
            })
        })
        .unwrap();
    (solid, edge)
}

#[test]
fn a_bore_between_the_arc_and_its_chord_is_in_the_rounds_way() {
    bore_between_the_arc_and_its_chord(false);
}

#[test]
fn a_bore_between_a_spline_arc_and_its_chord_is_in_the_rounds_way() {
    bore_between_the_arc_and_its_chord(true);
}

fn bore_between_the_arc_and_its_chord(splines: bool) {
    let r = 2.5;
    // The contact's end on the vault, as in `removed`, and the middle of
    // the arc from the corner to it: the bore lies between that and the
    // middle of the chord.
    let axis = (HALF, BOWED.axis_z());
    let reach = BOWED.radius - r;
    let up = (reach * reach - (r - HALF) * (r - HALF)).sqrt();
    let scale = BOWED.radius / reach;
    let on_vault = (axis.0 + (r - axis.0) * scale, axis.1 + up * scale);
    let chord = (on_vault.0 / 2.0, f64::midpoint(SIDE, on_vault.1));
    let out = (chord.0 - axis.0).hypot(chord.1 - axis.1);
    let sag = BOWED.radius - out;
    let middle = 1.0 + 0.5 * sag / out;
    let (y, z) = (
        axis.0 + (chord.0 - axis.0) * middle,
        axis.1 + (chord.1 - axis.1) * middle,
    );
    let mut topo = Topology::new();
    let (solid, edge) = bored_vault(&mut topo, y, z, sag / 4.0, splines);
    let result = fillet_v2(&mut topo, solid, &[edge], r).and_then(all_blended);
    let refusal = result.expect_err("a round over a bore in its end's way");
    assert!(
        refusal.to_string().contains("cross another boundary"),
        "{refusal}"
    );
}

#[test]
fn a_bore_clear_of_the_rounds_end_stays() {
    bore_clear_of_the_rounds_end(false);
}

#[test]
fn a_bore_clear_of_the_rounds_end_stays_by_spline_arcs() {
    bore_clear_of_the_rounds_end(true);
}

fn bore_clear_of_the_rounds_end(splines: bool) {
    let r = 2.5;
    let mut topo = Topology::new();
    let (solid, edge) = bored_vault(&mut topo, HALF, HALF, 1.0, splines);
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let result = fillet_v2(&mut topo, solid, &[edge], r)
        .and_then(all_blended)
        .unwrap();
    assert!(validate_solid(&topo, result).unwrap().is_valid());
    for deflection in [0.05, 0.5] {
        let mesh = tessellate_solid(&topo, result, deflection).unwrap();
        assert!(is_watertight(&mesh), "mesh not watertight at {deflection}");
    }
    assert_eq!(solid_faces(&topo, result).unwrap().len(), 8);
    let took = before - solid_volume(&topo, result, 1e-3).unwrap();
    let want = removed(BOWED, Pick::Along { far: false }, r);
    assert!((took - want).abs() <= 1e-4, "removed {took}, want {want}");
}
