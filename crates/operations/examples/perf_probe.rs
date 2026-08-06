//! Native timing probe for the two thinnest head-to-head leads:
//! cut(box, corner cylinder) and fine sphere tessellation.
#![allow(
    clippy::print_stdout,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::significant_drop_tightening
)]

use std::time::Instant;

use brepkit_math::vec::Point3;
use brepkit_operations::{boolean, primitives, tessellate};
use brepkit_topology::Topology;

struct StampLogger {
    clock: std::sync::Mutex<(Option<Instant>, Option<Instant>)>,
}

impl log::Log for StampLogger {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        let now = Instant::now();
        let (total_us, delta_us) = {
            let mut clock = self.clock.lock().unwrap();
            let start = *clock.0.get_or_insert(now);
            let delta = clock.1.map_or(0, |l| now.duration_since(l).as_micros());
            clock.1 = Some(now);
            (now.duration_since(start).as_micros(), delta)
        };
        if delta_us > 20 {
            let msg = format!("{}", record.args());
            let msg = &msg[..msg.len().min(110)];
            println!("[{total_us:>7}us +{delta_us:>6}us] {msg}");
        }
    }
    fn flush(&self) {}
}

fn main() {
    if std::env::var("CUT_TRACE").is_ok() {
        let logger = Box::leak(Box::new(StampLogger {
            clock: std::sync::Mutex::new((None, None)),
        }));
        log::set_logger(logger).unwrap();
        log::set_max_level(log::LevelFilter::Debug);
        let mut topo = Topology::new();
        let bx = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let cyl = primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();
        let t = Instant::now();
        let _ = boolean::boolean(&mut topo, boolean::BooleanOp::Cut, bx, cyl).unwrap();
        println!("single cut total: {:?}", t.elapsed());
        return;
    }
    // Warm-up + timed loop mirroring the JS bench (10 cuts per sample).
    let cut_rounds: usize = std::env::var("CUT_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    for round in 0..cut_rounds {
        let t = Instant::now();
        for _ in 0..10 {
            let mut topo = Topology::new();
            let bx = primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            let cyl = primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();
            let _ = boolean::boolean(&mut topo, boolean::BooleanOp::Cut, bx, cyl).unwrap();
        }
        if round % 10 == 0 || cut_rounds <= 5 {
            println!("cut(box,cyl) x10 round {round}: {:?}", t.elapsed());
        }
    }
    if std::env::var("CUT_ONLY").is_ok() {
        return;
    }

    let mut topo = Topology::new();
    let sph = primitives::make_sphere(&mut topo, 10.0, 32).unwrap();
    for round in 0..8 {
        let t = Instant::now();
        let mesh = tessellate::tessellate_solid_with_tolerance(&topo, sph, 0.01, 0.1).unwrap();
        println!(
            "mesh sphere tol=0.01 ang=0.1rad round {round}: {:?} ({} tris)",
            t.elapsed(),
            mesh.indices.len() / 3
        );
    }
    for round in 0..8 {
        let t = Instant::now();
        let (mesh, _offsets) =
            tessellate::tessellate_solid_grouped_with_tolerance(&topo, sph, 0.01, 0.1).unwrap();
        println!(
            "grouped mesh sphere round {round}: {:?} ({} tris)",
            t.elapsed(),
            mesh.indices.len() / 3
        );
    }

    let (mesh, _offsets) =
        tessellate::tessellate_solid_grouped_with_tolerance(&topo, sph, 0.01, 0.1).unwrap();
    let mut max_sag: f64 = 0.0;
    let mut area = 0.0;
    for t in mesh.indices.chunks(3) {
        let p: Vec<_> = t.iter().map(|&i| mesh.positions[i as usize]).collect();
        let c = Point3::new(
            (p[0].x() + p[1].x() + p[2].x()) / 3.0,
            (p[0].y() + p[1].y() + p[2].y()) / 3.0,
            (p[0].z() + p[1].z() + p[2].z()) / 3.0,
        );
        let d = (c.x() * c.x() + c.y() * c.y() + c.z() * c.z()).sqrt();
        max_sag = max_sag.max((10.0 - d).abs());
        let e1 = p[1] - p[0];
        let e2 = p[2] - p[0];
        area += e1.cross(e2).length() / 2.0;
    }
    println!(
        "sphere mesh: max centroid sag {max_sag:.5} (deflection budget 0.01), area {area:.3} vs analytic {:.3}",
        4.0 * std::f64::consts::PI * 100.0
    );
}
