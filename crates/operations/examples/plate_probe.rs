//! Native probe for issue #1488: plain rectangular baseplate pocket cuts.
//!
//! Mirrors the gridfinity baseplate export path: a slab cut by one tapered
//! pocket loft per cell via `compound_cut`. Pocket walls at the top band are
//! exactly coplanar with their neighbors (INSET_TOP = 0) and with the slab
//! perimeter, which is the suspected super-linear coincident-face path.
#![allow(
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::significant_drop_tightening
)]

use std::time::Instant;

use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::{boolean, loft, measure, primitives};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const CELL: f64 = 42.0;
const SOCKET_HEIGHT: f64 = 5.0;
const COPLANAR_MARGIN: f64 = 1.0;

fn rounded_rect_face(topo: &mut Topology, cx: f64, cy: f64, w: f64, r: f64, z: f64) -> FaceId {
    let tol_val = 1e-7;
    let c = w / 2.0 - r;
    let corners = [
        (cx + c, cy + c, 0.0_f64),
        (cx - c, cy + c, 90.0),
        (cx - c, cy - c, 180.0),
        (cx + c, cy - c, 270.0),
    ];
    let pt = |kx: f64, ky: f64, deg: f64| {
        let a = deg.to_radians();
        Point3::new(kx + r * a.cos(), ky + r * a.sin(), z)
    };
    let mut vids = Vec::new();
    for &(kx, ky, a0) in &corners {
        vids.push(topo.add_vertex(Vertex::new(pt(kx, ky, a0), tol_val)));
        vids.push(topo.add_vertex(Vertex::new(pt(kx, ky, a0 + 90.0), tol_val)));
    }
    let mut oes = Vec::new();
    for (k, &(kx, ky, _)) in corners.iter().enumerate() {
        let circle = Circle3D::new(Point3::new(kx, ky, z), Vec3::new(0.0, 0.0, 1.0), r).unwrap();
        let arc = topo.add_edge(Edge::new(
            vids[2 * k],
            vids[2 * k + 1],
            EdgeCurve::Circle(circle),
        ));
        oes.push(OrientedEdge::new(arc, true));
        let line = topo.add_edge(Edge::new(
            vids[2 * k + 1],
            vids[(2 * k + 2) % 8],
            EdgeCurve::Line,
        ));
        oes.push(OrientedEdge::new(line, true));
    }
    let wid = topo.add_wire(Wire::new(oes, true).unwrap());
    topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z,
        },
    ))
}

fn pocket(topo: &mut Topology, cx: f64, cy: f64, z_top: f64) -> SolidId {
    // (z offset from slab top, inset): the export 5-section socket profile
    // plus the +1 coplanar-margin cap and the -1 through-cut extension.
    let sections = [
        (COPLANAR_MARGIN, 0.0),
        (0.0, 0.0),
        (-0.25, 0.0),
        (-2.4, 2.15),
        (-4.2, 2.15),
        (-SOCKET_HEIGHT, 2.95),
        (-SOCKET_HEIGHT - COPLANAR_MARGIN, 2.95),
    ];
    let profs: Vec<FaceId> = sections
        .iter()
        .map(|&(dz, inset)| {
            let w = CELL - 2.0 * inset;
            let r = (4.0_f64 - inset).max(0.1);
            rounded_rect_face(topo, cx, cy, w, r, z_top + dz)
        })
        .collect();
    loft::loft(topo, &profs).unwrap()
}

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
        if delta_us > 5000 {
            let msg = format!("{}", record.args());
            let cut = msg.char_indices().nth(110).map_or(msg.len(), |(i, _)| i);
            let msg = &msg[..cut];
            println!("[{total_us:>8}us +{delta_us:>7}us] {msg}");
        }
    }
    fn flush(&self) {}
}

fn main() {
    if std::env::var("PLATE_TRACE").is_ok() {
        let logger = Box::leak(Box::new(StampLogger {
            clock: std::sync::Mutex::new((None, None)),
        }));
        log::set_logger(logger).unwrap();
        log::set_max_level(log::LevelFilter::Trace);
    }
    let grids: Vec<(usize, usize)> = std::env::args().nth(1).map_or_else(
        || vec![(1, 1), (2, 2), (3, 3), (4, 4), (6, 4)],
        |s| {
            let (a, b) = s.split_once('x').unwrap();
            vec![(a.parse().unwrap(), b.parse().unwrap())]
        },
    );

    for (n, m) in grids {
        let mut topo = Topology::new();
        let slab = primitives::make_box(&mut topo, CELL * n as f64, CELL * m as f64, SOCKET_HEIGHT)
            .unwrap();
        let mut pockets = Vec::new();
        for i in 0..n {
            for j in 0..m {
                let cx = CELL / 2.0 + CELL * i as f64;
                let cy = CELL / 2.0 + CELL * j as f64;
                pockets.push(pocket(&mut topo, cx, cy, SOCKET_HEIGHT));
            }
        }
        let split_stages = std::env::var("PLATE_STAGES").is_ok();
        if split_stages {
            let t0 = Instant::now();
            let fused = brepkit_algo::gfa::fuse_n(&mut topo, &pockets).unwrap();
            let fuse_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let fused_faces =
                brepkit_topology::explorer::solid_faces(&topo, fused).map_or(0, |f| f.len());
            let t1 = Instant::now();
            let result = boolean::boolean(&mut topo, boolean::BooleanOp::Cut, slab, fused);
            let cut_ms = t1.elapsed().as_secs_f64() * 1000.0;
            match result {
                Ok(solid) => {
                    let faces = brepkit_topology::explorer::solid_faces(&topo, solid)
                        .map_or(0, |f| f.len());
                    println!(
                        "{n}x{m}: fuse={fuse_ms:>8.1}ms (tool faces={fused_faces})  cut={cut_ms:>8.1}ms  faces={faces}"
                    );
                }
                Err(e) => println!("{n}x{m}: fuse={fuse_ms:>8.1}ms  cut ERROR {e}"),
            }
            continue;
        }
        let t = Instant::now();
        let result = boolean::compound_cut(
            &mut topo,
            slab,
            &pockets,
            boolean::BooleanOptions::default(),
        );
        let elapsed = t.elapsed();
        match result {
            Ok(solid) => {
                let faces =
                    brepkit_topology::explorer::solid_faces(&topo, solid).map_or(0, |f| f.len());
                let vol = measure::solid_volume(&topo, solid, 0.1).unwrap_or(f64::NAN);
                println!(
                    "{n}x{m}: {:>8.1}ms  faces={faces}  volume={vol:.1}",
                    elapsed.as_secs_f64() * 1000.0
                );
            }
            Err(e) => println!(
                "{n}x{m}: {:>8.1}ms  ERROR {e}",
                elapsed.as_secs_f64() * 1000.0
            ),
        }
    }
}
