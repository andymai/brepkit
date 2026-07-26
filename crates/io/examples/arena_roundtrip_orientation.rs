#![allow(clippy::print_stdout, clippy::expect_used, missing_docs)]
//! Does the arena `.bin` round-trip preserve solid orientation?
//!
//! The kumiko corner-wedge operands, captured via `serializeSolid` and replayed
//! through `deserialize_solid`, are genuinely inward-oriented: both of GFA's
//! orientation tests report inward for them, and the same tests are exact and
//! correctly signed on a natively-built cube (`signed_vol=1000.000000`, flux
//! `3000.0 = 3V`). So something upstream inverts the wedge. The round-trip is
//! the cheapest candidate, and it would explain EVERY captured operand at once.
//!
//! Method: build a box natively, run `Cut(box, far-away disjoint box)` on it —
//! which keeps all of A's faces, so the shell under test is the box — then do
//! the same on a serialize/deserialize copy of that box. If the copy reports
//! `0 growth shells, 1 hole shells` while the original reports the reverse, the
//! round-trip inverts orientation.
//!
//! Run with `BK_AREAS=1` to see each shell's signed volume.

use brepkit_io::arena_io::{deserialize_solid, serialize_solid};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives::make_box;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

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
        if msg.contains("AREAS") || msg.contains("growth shell") {
            println!("    [algo] {msg}");
        }
    }
    fn flush(&self) {}
}
static LOGGER: StdoutLogger = StdoutLogger;

/// `Cut(solid, far-away box)` — a no-op whose result shell is `solid` itself.
fn cut_by_far_box(topo: &mut Topology, solid: SolidId, label: &str) {
    let far = make_box(topo, 5.0, 5.0, 5.0).expect("far box");
    transform_solid(topo, far, &Mat4::translation(500.0, 500.0, 500.0)).expect("move far box");
    match boolean(topo, BooleanOp::Cut, solid, far) {
        Ok(sid) => {
            let n = brepkit_topology::explorer::solid_faces(topo, sid)
                .map(|f| f.len())
                .unwrap_or(0);
            println!("  {label}: ok F={n}");
        }
        Err(e) => println!("  {label}: ERR {e}"),
    }
}

/// A rectangle in the XZ plane (which CONTAINS the Z axis), spanning the
/// kumiko wedge's own extents: radius 1.55..4.75, height 2.7..20.8.
///
/// The profile plane must contain the revolve axis. A unit square in XY is
/// perpendicular to Z and revolves degenerately — it returns six planar faces
/// that look like a result and are not one.
fn wedge_profile(topo: &mut Topology, ccw: bool, ny: f64) -> brepkit_topology::face::FaceId {
    let tol = brepkit_math::tolerance::Tolerance::new().linear;
    let mut pts = [
        Point3::new(1.55, 0.0, 2.7),
        Point3::new(4.75, 0.0, 2.7),
        Point3::new(4.75, 0.0, 20.8),
        Point3::new(1.55, 0.0, 20.8),
    ];
    if !ccw {
        pts.reverse();
    }
    let vs: Vec<_> = pts
        .iter()
        .map(|p| topo.add_vertex(Vertex::new(*p, tol)))
        .collect();
    let mut oes = Vec::new();
    for i in 0..4 {
        let e = topo.add_edge(Edge::new(vs[i], vs[(i + 1) % 4], EdgeCurve::Line));
        oes.push(OrientedEdge::new(e, true));
    }
    let wid = topo.add_wire(Wire::new(oes, true).expect("wire"));
    topo.add_face(Face::new(
        wid,
        vec![],
        // XZ plane: normal along -Y so the profile winds consistently.
        FaceSurface::Plane {
            normal: Vec3::new(0.0, ny, 0.0),
            d: 0.0,
        },
    ))
}

fn main() {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Debug);

    let mut topo = Topology::new();
    let original = make_box(&mut topo, 10.0, 10.0, 10.0).expect("box");

    let bytes = serialize_solid(&topo, original).expect("serialize");
    let restored = deserialize_solid(&bytes, &mut topo).expect("deserialize");
    println!("serialized {} bytes", bytes.len());

    println!("native box:");
    cut_by_far_box(&mut topo, original, "native");

    println!("round-tripped box:");
    cut_by_far_box(&mut topo, restored, "round-trip");

    // Candidate (1): does `revolve` emit an inward solid for a wedge profile?
    // All four combinations of profile winding and plane normal: if any one
    // yields an outward wedge, `revolve` is fine and the input convention is
    // what matters. If all four are inward, `revolve` itself inverts.
    for (ccw, ny) in [(true, -1.0), (true, 1.0), (false, -1.0), (false, 1.0)] {
        println!("PARTIAL revolve 45deg (ccw={ccw} normal_y={ny}):");
        let profile = wedge_profile(&mut topo, ccw, ny);
        match brepkit_operations::revolve::revolve(
            &mut topo,
            profile,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        ) {
            Ok(wedge) => cut_by_far_box(&mut topo, wedge, "  wedge"),
            Err(e) => println!("  revolve ERR {e}"),
        }
    }

    // The analytic FULL-revolution path documents itself as exact for both
    // windings. If full is clean where partial is not, the gap is specific to
    // the segmented path.
    for ccw in [true, false] {
        println!("FULL revolve 360deg (ccw={ccw}):");
        let profile = wedge_profile(&mut topo, ccw, -1.0);
        match brepkit_operations::revolve::revolve(
            &mut topo,
            profile,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::TAU,
        ) {
            Ok(full) => cut_by_far_box(&mut topo, full, "  full"),
            Err(e) => println!("  revolve ERR {e}"),
        }
    }
}
