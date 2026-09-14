//! Per-pair marcher diagnostic for the kumiko corner-wrap cut: every cylinder
//! face of the band against every NURBS wall of one strut, through the same
//! NURBS-by-NURBS marcher `phase_ff` uses, reporting wall time and how far the
//! returned curves sit from both surfaces.
//!
//! Run: `cargo run --release -p brepkit-io --example kumiko_pair_probe`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::print_stdout,
    clippy::too_many_lines,
    clippy::type_complexity
)]

use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::aabb::Aabb3;
use brepkit_math::nurbs::intersection::intersect_nurbs_nurbs;
use brepkit_math::nurbs::projection::project_point_to_surface;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{FaceId, FaceSurface};

fn face_points(topo: &Topology, fid: FaceId) -> Vec<Point3> {
    let face = topo.face(fid).unwrap();
    let mut pts = Vec::new();
    let mut wires = vec![face.outer_wire()];
    wires.extend_from_slice(face.inner_wires());
    for wid in wires {
        for oe in topo.wire(wid).unwrap().edges() {
            let e = topo.edge(oe.edge()).unwrap();
            let (a, b) = (
                topo.vertex(e.start()).unwrap().point(),
                topo.vertex(e.end()).unwrap().point(),
            );
            let (t0, t1) = e.curve().domain_with_endpoints(a, b);
            for k in 0..=8 {
                let t = t0 + (t1 - t0) * k as f64 / 8.0;
                pts.push(e.curve().evaluate_with_endpoints(t, a, b));
            }
        }
    }
    pts
}

fn main() {
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut topo = Topology::new();
    let band = deserialize_solid(
        &std::fs::read(data.join("kumiko_wrap_band.bin")).unwrap(),
        &mut topo,
    )
    .unwrap();
    let strut = deserialize_solid(
        &std::fs::read(data.join("kumiko_wrap_strut.bin")).unwrap(),
        &mut topo,
    )
    .unwrap();

    let mut cylinders: Vec<(
        FaceId,
        NurbsSurface,
        brepkit_math::surfaces::CylindricalSurface,
        Aabb3,
    )> = Vec::new();
    for fid in solid_faces(&topo, band).unwrap() {
        let surface = topo.face(fid).unwrap().surface().clone();
        if let FaceSurface::Cylinder(c) = &surface {
            let pts = face_points(&topo, fid);
            let (mut v0, mut v1) = (f64::MAX, f64::MIN);
            for p in &pts {
                if let Some((_, v)) = surface.project_point(*p) {
                    v0 = v0.min(v);
                    v1 = v1.max(v);
                }
            }
            let nurbs = c.to_nurbs(v0, v1).unwrap();
            cylinders.push((fid, nurbs, c.clone(), Aabb3::from_points(pts).expanded(0.5)));
        }
    }
    let mut walls: Vec<(FaceId, NurbsSurface, Aabb3)> = Vec::new();
    for fid in solid_faces(&topo, strut).unwrap() {
        if let FaceSurface::Nurbs(n) = topo.face(fid).unwrap().surface() {
            let pts = face_points(&topo, fid);
            walls.push((fid, n.clone(), Aabb3::from_points(pts).expanded(0.5)));
        }
    }
    println!(
        "band cylinders={} strut nurbs walls={}",
        cylinders.len(),
        walls.len()
    );

    let mut rows = Vec::new();
    let total = std::time::Instant::now();
    for (cf, cn, cyl, cbb) in &cylinders {
        for (wf, wn, wbb) in &walls {
            if !cbb.intersects(*wbb) {
                continue;
            }
            let t = std::time::Instant::now();
            let curves = intersect_nurbs_nurbs(cn, wn, 32, 0.01);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let mut n_curves = 0;
            let mut n_points = 0;
            let mut max_dev_cyl = 0.0_f64;
            let mut max_dev_wall = 0.0_f64;
            let mut max_dev_curve = 0.0_f64;
            if let Ok(curves) = &curves {
                n_curves = curves.len();
                for ic in curves {
                    n_points += ic.points.len();
                    for p in &ic.points {
                        let rel = p.point - cyl.origin();
                        let axial = cyl.axis().dot(rel);
                        let radial = (rel - cyl.axis() * axial).length();
                        max_dev_cyl = max_dev_cyl.max((radial - cyl.radius()).abs());
                        if let Ok(pr) = project_point_to_surface(wn, p.point, 1e-9) {
                            max_dev_wall = max_dev_wall.max(pr.distance);
                        }
                    }
                    let (d0, d1) = ic.curve.domain();
                    for k in 0..=16 {
                        let q = ic.curve.evaluate(d0 + (d1 - d0) * k as f64 / 16.0);
                        let rel = q - cyl.origin();
                        let axial = cyl.axis().dot(rel);
                        let radial = (rel - cyl.axis() * axial).length();
                        max_dev_curve = max_dev_curve.max((radial - cyl.radius()).abs());
                    }
                }
            }
            rows.push((
                ms,
                *cf,
                *wf,
                n_curves,
                n_points,
                max_dev_cyl,
                max_dev_wall,
                max_dev_curve,
                curves.is_err(),
            ));
        }
    }
    let total_ms = total.elapsed().as_secs_f64() * 1e3;
    println!(
        "pairs={} total={total_ms:.0}ms with_curves={}",
        rows.len(),
        rows.iter().filter(|r| r.3 > 0).count()
    );
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("-- slowest pairs (ms, cyl, wall, curves, pts, dev_cyl, dev_wall, dev_curve, err)");
    for r in rows.iter().take(8) {
        println!(
            "  {:8.1}ms {:?} x {:?} curves={} pts={} dev_cyl={:.2e} dev_wall={:.2e} dev_curve={:.2e} err={}",
            r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8
        );
    }
    rows.sort_by(|a, b| b.5.max(b.6).total_cmp(&a.5.max(a.6)));
    println!("-- worst point deviation from the surfaces");
    for r in rows.iter().take(8) {
        println!(
            "  {:8.1}ms {:?} x {:?} curves={} pts={} dev_cyl={:.2e} dev_wall={:.2e} dev_curve={:.2e}",
            r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7
        );
    }
    rows.sort_by(|a, b| b.7.total_cmp(&a.7));
    println!("-- worst fitted-curve deviation from the cylinder");
    for r in rows.iter().take(5) {
        println!(
            "  {:8.1}ms {:?} x {:?} curves={} pts={} dev_cyl={:.2e} dev_wall={:.2e} dev_curve={:.2e}",
            r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7
        );
    }
}
