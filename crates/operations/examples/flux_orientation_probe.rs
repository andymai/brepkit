#![allow(clippy::print_stdout, clippy::expect_used, missing_docs)]
//! Discriminating experiment: does `shell_is_outward_oriented` read a
//! KNOWN-GOOD primitive as outward?
//!
//! The kumiko corner-wedge cut fails because its result shell's flux integral
//! comes out at -852.42 while 3V is +854.6 — right magnitude, wrong sign. Two
//! explanations remain and they need opposite fixes: either the flux
//! convention is inverted, or the captured wedge operand is genuinely inward
//! and the defect is upstream. Every measurement available in `brepkit-io` is
//! orientation-blind (`solid_volume` returns `.abs()`, ray-parity
//! classification ignores normals), so this runs the same code path on a box
//! built right here, whose orientation is not in question.
//!
//! `Cut(box, far-away disjoint box)` keeps all of A's faces and none of B's,
//! so the shell under test is exactly the box.
//!
//! Outward for the box but inward for the wedge → the wedge is inverted.
//! Inward for BOTH → the flux convention itself is wrong.
//!
//! Run with `BK_AREAS=1 BK_FLUX=1`.

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives::make_box;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;

struct StdoutLogger;
impl log::Log for StdoutLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.target().starts_with("brepkit_algo") && m.level() <= log::Level::Debug
    }
    fn log(&self, r: &log::Record) {
        if !self.enabled(r.metadata()) {
            return;
        }
        let msg = format!("{}", r.args());
        if msg.contains("FLUX") || msg.contains("AREAS") || msg.contains("growth shell") {
            println!("    [algo] {msg}");
        }
    }
    fn flush(&self) {}
}
static LOGGER: StdoutLogger = StdoutLogger;

fn main() {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Debug);

    let mut topo = Topology::new();
    let a = make_box(&mut topo, 10.0, 10.0, 10.0).expect("box a");
    // Far enough away that no face pair can overlap, so the cut is a no-op and
    // the result shell is A itself.
    let b = make_box(&mut topo, 5.0, 5.0, 5.0).expect("box b");
    transform_solid(&mut topo, b, &Mat4::translation(500.0, 500.0, 500.0)).expect("move b");

    println!("Cut(box, far-away disjoint box) — result shell should be the box:");
    match boolean(&mut topo, BooleanOp::Cut, a, b) {
        Ok(sid) => {
            let faces = brepkit_topology::explorer::solid_faces(&topo, sid).expect("faces");
            println!("  ok: F={}", faces.len());
        }
        Err(e) => println!("  ERR {e}"),
    }
}
