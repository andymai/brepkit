//! A round on the whole rim of a slab's top whose radius is more than the
//! slab's thickness: the ball touches the top and rolls over the bottom's
//! rim, and the side wall goes.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// The edges of the top face's outer wire (the plane facing +z highest up).
fn top_rim(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
    let mut best = None;
    for f in solid_faces(topo, solid).unwrap() {
        let face = topo.face(f).unwrap();
        if let FaceSurface::Plane { normal, d } = face.surface() {
            let up = if face.is_reversed() {
                -normal.z()
            } else {
                normal.z()
            };
            let height = d * normal.z();
            if up > 0.5 && best.is_none_or(|(h, _)| height > h) {
                best = Some((height, f));
            }
        }
    }
    let face = topo.face(best.unwrap().1).unwrap();
    topo.wire(face.outer_wire())
        .unwrap()
        .edges()
        .iter()
        .map(brepkit_topology::wire::OrientedEdge::edge)
        .collect()
}

/// The volume a ball of radius `r` takes off a slab `h` thick rolling along
/// a straight rim, per unit length: the part of the `r` by `h` corner
/// outside its circle, the circle's centre `r` below the top and
/// `s = sqrt(r² - (r - h)²)` in from the side.
fn section_area(r: f64, h: f64) -> f64 {
    let s = (r * r - (r - h).powi(2)).sqrt();
    // Area of the circle's cap past the bottom within the strip from its
    // centre out to the side: integrate sqrt(r² - y²) for y from r-h to r.
    let under = |y: f64| 0.5 * (y * (r * r - y * y).sqrt() + r * r * (y / r).asin());
    let inside = under(r) - under(r - h);
    s * h - inside
}

fn check(topo: &Topology, before: f64, result: SolidId, removed: f64, what: &str) {
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "{what}: open mesh");
    let faces = solid_faces(topo, result).unwrap();
    assert!(faces.len() < 20, "{what}: {} faces", faces.len());
    let after = solid_volume(topo, result, 1e-3).unwrap();
    assert!(
        ((before - after) - removed).abs() < 1e-4 * before,
        "{what}: took off {}, the round takes {removed}",
        before - after
    );
}

#[test]
fn a_disc_rounded_past_its_thickness() {
    let (rad, h, r) = (60.0, 25.0, 27.5);
    let mut topo = Topology::new();
    let disc = make_cylinder(&mut topo, rad, h).unwrap();
    let before = solid_volume(&topo, disc, 1e-3).unwrap();
    let rim = top_rim(&topo, disc);
    let result = fillet_v2(&mut topo, disc, &rim, r).unwrap();
    // Pappus: the section turned about the axis at its centroid.
    let s = (r * r - (r - h).powi(2)).sqrt();
    let n = 20_000;
    let (mut area, mut moment) = (0.0, 0.0);
    for i in 0..n {
        for j in 0..400 {
            let z = h * (f64::from(i) + 0.5) / f64::from(n);
            let x = s * (f64::from(j) + 0.5) / 400.0;
            let (dz, dx) = (h / f64::from(n), s / 400.0);
            // Out from the centre's line, x from 0 at s in from the side.
            if x * x + (h - r - z).powi(2) > r * r {
                area += dz * dx;
                moment += dz * dx * (rad - s + x);
            }
        }
    }
    let _ = section_area(r, h);
    assert!(area > 0.0);
    check(
        &topo,
        before,
        result.solid,
        2.0 * std::f64::consts::PI * moment,
        "disc",
    );
}

#[test]
fn a_plate_rounded_past_its_thickness() {
    let (w, d, h, r) = (120.0, 80.0, 6.0, 8.0);
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, w, d, h).unwrap();
    let before = solid_volume(&topo, plate, 1e-3).unwrap();
    let rim = top_rim(&topo, plate);
    let result = fillet_v2(&mut topo, plate, &rim, r).unwrap();
    // Four straight runs, and the corners, where the side edges stay sharp
    // and the two runs' rounds meet in a mitre: a point there stays when
    // both runs keep it.
    let s = (r * r - (r - h).powi(2)).sqrt();
    let a = section_area(r, h);
    let straight = a * 2.0 * ((w - 2.0 * s) + (d - 2.0 * s));
    // Each corner: the s by s by h block less what both rounds keep of it.
    let n = 200;
    let mut kept = 0.0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let (x, y, z) = (
                    s * (f64::from(i) + 0.5) / f64::from(n),
                    s * (f64::from(j) + 0.5) / f64::from(n),
                    h * (f64::from(k) + 0.5) / f64::from(n),
                );
                // x, y out from the centres' corner, z up from the bottom.
                let dz = (h - r - z).powi(2);
                if x * x + dz <= r * r && y * y + dz <= r * r {
                    kept += 1.0;
                }
            }
        }
    }
    let cell = s * s * h / f64::from(n).powi(3);
    let corner = s * s * h - kept * cell;
    check(
        &topo,
        before,
        result.solid,
        straight + 4.0 * corner,
        "plate",
    );
}
