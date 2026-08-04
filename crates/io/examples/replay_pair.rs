//! Replay one captured boolean operand pair natively.
//!
//! The tool-side capture writes `<op>-<n>.bin` per operand; this loads two of
//! them and runs the op both through `operations::boolean` (which may fall back
//! to mesh) and through raw GFA (which reports the analytic failure directly).
//!
//! ```sh
//! A=.../op1-fuseWithEvolution-0.bin B=.../op1-fuseWithEvolution-1.bin \
//!   OP=fuse cargo run --release -p brepkit-io --example replay_pair
//! ```
#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn describe(topo: &Topology, sid: SolidId, label: &str) {
    let Ok(faces) = solid_faces(topo, sid) else {
        println!("  {label}: <no faces>");
        return;
    };
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    let mut mix: HashMap<&'static str, usize> = HashMap::new();
    for &fid in &faces {
        let Ok(face) = topo.face(fid) else { continue };
        *mix.entry(face.surface().type_tag()).or_default() += 1;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let Ok(w) = topo.wire(wid) else { continue };
            for oe in w.edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    let free = uses.values().filter(|&&c| c == 1).count();
    let over = uses.values().filter(|&&c| c > 2).count();
    if std::env::var("FREE_EDGES").is_ok() {
        for (eid, n) in &uses {
            if *n != 2
                && let Ok(e) = topo.edge(*eid)
                && let (Ok(a), Ok(b)) = (topo.vertex(e.start()), topo.vertex(e.end()))
            {
                let (a, b) = (a.point(), b.point());
                println!(
                    "  {} edge {eid:?} {} ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3})",
                    if *n == 1 { "FREE" } else { "OVER" },
                    e.curve().type_tag(),
                    a.x(),
                    a.y(),
                    a.z(),
                    b.x(),
                    b.y(),
                    b.z()
                );
            }
        }
    }
    let mut mix: Vec<_> = mix.into_iter().collect();
    mix.sort_unstable();
    let vol =
        brepkit_operations::measure::oriented_solid_volume(topo, sid, 0.05).unwrap_or(f64::NAN);
    println!(
        "  {label}: F={} mix={mix:?} free={free} over={over} vol={vol:.3}",
        faces.len()
    );
}

struct Tap;
impl log::Log for Tap {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.target().starts_with("brepkit_") && m.level() <= log::Level::Debug
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            println!("    [log] {}", r.args());
        }
    }
    fn flush(&self) {}
}
static TAP: Tap = Tap;

fn main() {
    let _ = log::set_logger(&TAP);
    log::set_max_level(log::LevelFilter::Debug);

    let a_path = PathBuf::from(std::env::var_os("A").expect("A=<path>"));
    let b_path = PathBuf::from(std::env::var_os("B").expect("B=<path>"));
    let op = std::env::var("OP").unwrap_or_else(|_| "fuse".to_string());

    let mut topo = Topology::new();
    let a = deserialize_solid(&std::fs::read(&a_path).unwrap(), &mut topo).unwrap();
    let b = deserialize_solid(&std::fs::read(&b_path).unwrap(), &mut topo).unwrap();
    describe(&topo, a, "A");
    describe(&topo, b, "B");

    let bop = match op.as_str() {
        "cut" => brepkit_algo::bop::BooleanOp::Cut,
        "intersect" => brepkit_algo::bop::BooleanOp::Intersect,
        _ => brepkit_algo::bop::BooleanOp::Fuse,
    };

    // POINT_IN=x,y,z classifies a point against BOTH operands with the
    // independent operations-level oracle. For a Fuse, a face is needed
    // wherever one side of a surface is inside the union and the other is not.
    if let Ok(spec) = std::env::var("POINT_IN") {
        // Semicolon-separated points, so a batch is classified in one process
        // instead of one process per point.
        // Deflection matters: classify_point tessellates, and this lattice has
        // 0.05mm features that a coarse deflection cannot represent, which
        // makes the verdict itself an artifact of the setting.
        let defl: f64 = std::env::var("POINT_DEFL")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0.01);
        for (i, one) in spec.split(';').filter(|t| !t.trim().is_empty()).enumerate() {
            let c: Vec<f64> = one
                .split(',')
                .filter_map(|t| t.trim().parse().ok())
                .collect();
            if c.len() != 3 {
                continue;
            }
            let p = brepkit_math::vec::Point3::new(c[0], c[1], c[2]);
            let mut row = format!("  POINT_IN[{i}] ({:.3},{:.3},{:.3})", c[0], c[1], c[2]);
            for (label, sid) in [("A", a), ("B", b)] {
                match brepkit_operations::classify::classify_point(&topo, sid, p, defl, 1e-7) {
                    Ok(v) => {
                        let _ = write!(row, "  {label}={v:?}");
                    }
                    Err(_) => {
                        let _ = write!(row, "  {label}=ERR");
                    }
                }
            }
            println!("{row}");
        }
        return;
    }

    // TOOLS=<comma-separated paths> replays a compound_cut, which is how the
    // kumiko wrap chain is actually built; a pairwise replay cannot reach it.
    if let Ok(list) = std::env::var("TOOLS") {
        let tools: Vec<_> = list
            .split(',')
            .filter(|t| !t.trim().is_empty())
            .map(|t| deserialize_solid(&std::fs::read(t.trim()).unwrap(), &mut topo).unwrap())
            .collect();
        println!("-- compound_cut with {} tools --", tools.len());
        let t = std::time::Instant::now();
        match brepkit_operations::boolean::compound_cut(
            &mut topo,
            a,
            &tools,
            brepkit_operations::boolean::BooleanOptions::default(),
        ) {
            Ok(sid) => describe(
                &topo,
                sid,
                &format!("compound_cut {}ms", t.elapsed().as_millis()),
            ),
            Err(e) => println!(
                "  compound_cut FAILED in {}ms: {e}",
                t.elapsed().as_millis()
            ),
        }
        return;
    }

    println!("-- raw GFA {op} --");
    let t = std::time::Instant::now();
    match brepkit_algo::gfa::boolean(&mut topo, bop, a, b) {
        Ok(sid) => describe(
            &topo,
            sid,
            &format!("RAW {op} {}ms", t.elapsed().as_millis()),
        ),
        Err(e) => println!("  RAW {op} FAILED in {}ms: {e}", t.elapsed().as_millis()),
    }
}
