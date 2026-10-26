//! Fillets of every floor edge of a prism standing on its outline, all of one
//! radius `r`. At height `z < r` the result's section is the outline eroded
//! by `w(z) = r - sqrt(r^2 - (r - z)^2)`: sharp where the outline turns
//! convex (the stripes meet in a crease) and rounded where it turns reflex
//! (a horn torus about the corner's vertical edge). A point therefore lies
//! inside exactly when it lies over the outline at least `w(z)` from its
//! boundary.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::extrude::extrude;
use brepkit_operations::fillet::{FilletRadiusLaw, fillet_variable};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const R: f64 = 2.0;
const HEIGHT: f64 = 5.0;

/// The prism of `HEIGHT` on the outline through `corners`, each side a line
/// but those in `arcs` as `(side, center)`: an arc counter-clockwise about
/// `+z` from that corner to the next.
fn prism(topo: &mut Topology, corners: &[(f64, f64)], arcs: &[(usize, (f64, f64))]) -> SolidId {
    let ids: Vec<_> = corners
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
        .collect();
    let mut edges = Vec::new();
    for i in 0..ids.len() {
        let (a, b) = (ids[i], ids[(i + 1) % ids.len()]);
        let curve = match arcs.iter().find(|s| s.0 == i) {
            Some(&(_, (cx, cy))) => {
                let center = Point3::new(cx, cy, 0.0);
                let radius = (topo.vertex(a).unwrap().point() - center).length();
                EdgeCurve::Circle(Circle3D::new(center, Vec3::new(0.0, 0.0, 1.0), radius).unwrap())
            }
            None => EdgeCurve::Line,
        };
        edges.push(OrientedEdge::new(
            topo.add_edge(Edge::new(a, b, curve)),
            true,
        ));
    }
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let plane = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 0.0,
    };
    let face = topo.add_face(Face::new(wire, Vec::new(), plane));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), HEIGHT).unwrap()
}

/// Fillets every edge of `solid` in `z = 0` at radius `R`.
fn fillet_floor(topo: &mut Topology, solid: SolidId) -> SolidId {
    let laws: Vec<_> = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()]
                .iter()
                .all(|&v| topo.vertex(v).unwrap().point().z().abs() < 1e-9)
        })
        .map(|e| (e, FilletRadiusLaw::Constant(R)))
        .collect();
    fillet_variable(topo, solid, &laws).unwrap()
}

/// The fillet of `solid`'s floor is valid, watertight, and holds exactly the
/// points of a grid over `lo..hi` that lie over the outline at least `w(z)`
/// from its boundary, `distance` giving that clearance (negative off the
/// outline). Points within 0.02 of either boundary are skipped.
fn assert_floor_rounds_off(
    topo: &mut Topology,
    solid: SolidId,
    lo: (f64, f64),
    hi: (f64, f64),
    distance: impl Fn(f64, f64) -> f64,
) {
    let result = fillet_floor(topo, solid);
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mesh = tessellate_solid(topo, result, 0.01).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    let w = |z: f64| R - (R * R - (R - z) * (R - z)).max(0.0).sqrt();
    let (mut wrong, mut read) = (Vec::new(), 0);
    for i in 0..30 {
        for j in 0..30 {
            for k in 0..8 {
                let f = |n: i32, of: f64| (f64::from(n) + 0.37) / of;
                let x = (hi.0 - lo.0).mul_add(f(i, 30.0), lo.0);
                let y = (hi.1 - lo.1).mul_add(f(j, 30.0), lo.1);
                let z = 2.5 * f(k, 8.0);
                let d = distance(x, y);
                if d.abs() < 0.02 || (z < R && (d - w(z)).abs() < 0.02) {
                    continue;
                }
                let truth = d > 0.0 && (z >= R || d >= w(z));
                let got = match classify_point(topo, result, Point3::new(x, y, z), 0.01, 1e-7) {
                    Ok(PointClassification::Inside) => true,
                    Ok(PointClassification::Outside) => false,
                    _ => continue,
                };
                read += 1;
                if got != truth {
                    wrong.push((x, y, z));
                }
            }
        }
    }
    assert!(read > 5000, "too few points read: {read}");
    assert!(
        wrong.is_empty(),
        "{} of {read} misread: {:?}",
        wrong.len(),
        &wrong[..wrong.len().min(5)]
    );
}

fn segment_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let t = ((p.0 - a.0).mul_add(dx, (p.1 - a.1) * dy) / dx.mul_add(dx, dy * dy)).clamp(0.0, 1.0);
    (p.0 - t.mul_add(dx, a.0)).hypot(p.1 - t.mul_add(dy, a.1))
}

/// A box over `x` in `-20..-2`, `|y| <= 7.5`, its right side replaced by the
/// arc of radius 8.5 about `(2, 0)` that bulges past its top and bottom: the
/// arc's wall meets the box's long sides at two reflex corners, and the floor
/// reaches far past the arc's circle. The arc is split at `(10.5, 0)`.
#[test]
fn a_floor_reaching_past_its_arc_wall_rounds_off_inside_the_arc() {
    let mut topo = Topology::new();
    let corners = [
        (-20.0, -7.5),
        (-2.0, -7.5),
        (10.5, 0.0),
        (-2.0, 7.5),
        (-20.0, 7.5),
    ];
    let solid = prism(&mut topo, &corners, &[(1, (2.0, 0.0)), (2, (2.0, 0.0))]);
    assert_floor_rounds_off(&mut topo, solid, (-21.0, -9.5), (11.5, 9.5), |x, y| {
        let rc = (x - 2.0).hypot(y);
        let inside = ((-20.0..=-2.0).contains(&x) && y.abs() <= 7.5) || (x >= -2.0 && rc <= 8.5);
        // The arc runs over the angles about its center within that of its
        // ends, (-4, 7.5) from the center.
        let on_arc = y.atan2(x - 2.0).abs() <= 7.5_f64.atan2(-4.0);
        let arc = if on_arc { (rc - 8.5).abs() } else { f64::MAX };
        let d = [
            segment_distance((x, y), (-20.0, -7.5), (-2.0, -7.5)),
            segment_distance((x, y), (-2.0, 7.5), (-20.0, 7.5)),
            segment_distance((x, y), (-20.0, 7.5), (-20.0, -7.5)),
            (x + 2.0).hypot(y - 7.5),
            (x + 2.0).hypot(y + 7.5),
            arc,
        ]
        .into_iter()
        .fold(f64::MAX, f64::min);
        if inside { d } else { -d }
    });
}
