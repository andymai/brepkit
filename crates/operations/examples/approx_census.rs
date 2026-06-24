//! Approximation-path census.
//!
//! Reports, per operation, whether it produced an exact analytic B-Rep or
//! degraded to an approximation — and which one: the boolean mesh (co-refinement)
//! fallback, the fillet Newton-Raphson walker, the sampled-NURBS surface offset,
//! the grid-sampling offset trim, or the rolling-ball planar corner patch.
//!
//! It installs an in-process logger that captures the `brepkit_approx` debug
//! probes, so each row shows exactly which fallback (if any) fired during that
//! single operation, alongside wall-clock and result face count.
//!
//! Run:
//!   cargo run --release --example approx_census -p brepkit-operations
//!
//! The boolean matrix uses overlapping primitives; offset/fillet/chamfer run on
//! every analytic primitive to show they stay exact (no probe fires). The
//! NURBS-faced and non-analytic-pair fallbacks (sampled-NURBS offset, fillet
//! walker, chamfer UnsupportedSurface) are exercised by the `brepkit-offset` and
//! `brepkit-blend` test suites, not reproducible from primitives alone (no
//! primitive has a NURBS face).

#![allow(clippy::print_stdout, deprecated, missing_docs)]

use std::error::Error;
use std::sync::Mutex;
use std::time::Instant;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::OperationsError;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::fillet::fillet_rolling_ball;
use brepkit_operations::loft::loft_smooth;
use brepkit_operations::offset_v2::{offset_solid_v2, shell_v2};
use brepkit_operations::primitives;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

static EVENTS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Captures only `brepkit_approx` probe records into `EVENTS`; everything else is
/// cheaply ignored (filtered in `enabled` and again in `log`) so unrelated engine
/// logging does not skew the per-op timings this example measures.
struct CaptureLogger;
impl log::Log for CaptureLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.target() == "brepkit_approx"
    }
    fn log(&self, record: &log::Record) {
        if record.target() == "brepkit_approx"
            && let Ok(mut ev) = EVENTS.lock()
        {
            ev.push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

static LOGGER: CaptureLogger = CaptureLogger;

/// Take and clear the captured probe messages (no clone; poisoned lock → empty).
fn drain() -> Vec<String> {
    EVENTS
        .lock()
        .map(|mut ev| std::mem::take(&mut *ev))
        .unwrap_or_default()
}

type PrimBuild = fn(&mut Topology) -> Result<SolidId, OperationsError>;

fn face_count(topo: &Topology, s: SolidId) -> usize {
    solid_faces(topo, s).map(|f| f.len()).unwrap_or(0)
}

fn report(family: &str, case: &str, ms: f64, faces: usize, events: &[String]) {
    let path = if events.is_empty() {
        "exact analytic".to_string()
    } else {
        // Show every distinct probe (an op can hit more than one path), with the
        // raw count so per-corner repeats stay visible.
        let mut uniq: Vec<&str> = Vec::new();
        for e in events {
            if !uniq.contains(&e.as_str()) {
                uniq.push(e.as_str());
            }
        }
        format!("FALLBACK x{}: {}", events.len(), uniq.join(" | "))
    };
    println!("  {family:<9} {case:<30} {ms:>8.2}ms  faces={faces:<4}  {path}");
}

fn box_at(topo: &mut Topology, d: f64, x: f64, y: f64, z: f64) -> Result<SolidId, OperationsError> {
    let s = primitives::make_box(topo, d, d, d)?;
    transform_solid(topo, s, &Mat4::translation(x, y, z))?;
    Ok(s)
}

fn bool_case(
    name: &str,
    op: BooleanOp,
    build: impl FnOnce(&mut Topology) -> Result<(SolidId, SolidId), OperationsError>,
) {
    let mut topo = Topology::new();
    let (a, b) = match build(&mut topo) {
        Ok(v) => v,
        Err(e) => {
            report(
                "boolean",
                &format!("{name} [build ERR]"),
                0.0,
                0,
                &[format!("err: {e}")],
            );
            return;
        }
    };
    let _ = drain();
    let t = Instant::now();
    let res = boolean(&mut topo, op, a, b);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut ev = drain();
    match res {
        Ok(s) => report("boolean", name, ms, face_count(&topo, s), &ev),
        Err(e) => {
            ev.push(format!("err: {e}"));
            report("boolean", &format!("{name} [ERR]"), ms, 0, &ev);
        }
    }
}

fn boolean_matrix() {
    println!("BOOLEAN (overlapping primitives):");
    bool_case("box ∪ box (overlap)", BooleanOp::Fuse, |t| {
        Ok((
            box_at(t, 10.0, 0.0, 0.0, 0.0)?,
            box_at(t, 10.0, 5.0, 5.0, 5.0)?,
        ))
    });
    bool_case("box ∪ box (flush coplanar)", BooleanOp::Fuse, |t| {
        Ok((
            box_at(t, 10.0, 0.0, 0.0, 0.0)?,
            box_at(t, 10.0, 10.0, 0.0, 0.0)?,
        ))
    });
    bool_case("box − cyl (through hole)", BooleanOp::Cut, |t| {
        let b = box_at(t, 10.0, 0.0, 0.0, 0.0)?;
        let c = primitives::make_cylinder(t, 3.0, 20.0)?;
        transform_solid(t, c, &Mat4::translation(5.0, 5.0, -5.0))?;
        Ok((b, c))
    });
    bool_case("box ∩ sphere", BooleanOp::Intersect, |t| {
        let b = box_at(t, 10.0, 0.0, 0.0, 0.0)?;
        let s = primitives::make_sphere(t, 6.0, 24)?;
        transform_solid(t, s, &Mat4::translation(5.0, 5.0, 5.0))?;
        Ok((b, s))
    });
    bool_case("sphere − cyl (3 pieces)", BooleanOp::Cut, |t| {
        let s = primitives::make_sphere(t, 6.0, 24)?;
        let c = primitives::make_cylinder(t, 3.0, 30.0)?;
        transform_solid(t, c, &Mat4::translation(0.0, 0.0, -15.0))?;
        Ok((s, c))
    });
    bool_case("cyl ∪ cyl (perp cross)", BooleanOp::Fuse, |t| {
        let c1 = primitives::make_cylinder(t, 3.0, 20.0)?;
        transform_solid(t, c1, &Mat4::translation(0.0, 0.0, -10.0))?;
        let c2 = primitives::make_cylinder(t, 3.0, 20.0)?;
        transform_solid(t, c2, &Mat4::rotation_y(std::f64::consts::FRAC_PI_2))?;
        transform_solid(t, c2, &Mat4::translation(-10.0, 0.0, 0.0))?;
        Ok((c1, c2))
    });
    bool_case("cyl ∩ cyl (coaxial)", BooleanOp::Intersect, |t| {
        let c1 = primitives::make_cylinder(t, 5.0, 20.0)?;
        let c2 = primitives::make_cylinder(t, 5.0, 20.0)?;
        transform_solid(t, c2, &Mat4::translation(0.0, 0.0, 10.0))?;
        Ok((c1, c2))
    });
    bool_case("cone ∪ box", BooleanOp::Fuse, |t| {
        let c = primitives::make_cone(t, 6.0, 2.0, 12.0)?;
        let b = box_at(t, 8.0, -4.0, -4.0, 6.0)?;
        Ok((c, b))
    });
    bool_case("torus − box", BooleanOp::Cut, |t| {
        let tor = primitives::make_torus(t, 10.0, 3.0, 32)?;
        let b = box_at(t, 8.0, 6.0, -4.0, -4.0)?;
        Ok((tor, b))
    });
    bool_case("cyl − box (slot)", BooleanOp::Cut, |t| {
        let c = primitives::make_cylinder(t, 6.0, 20.0)?;
        let b = box_at(t, 4.0, -2.0, -8.0, 5.0)?;
        Ok((c, b))
    });
}

fn offset_matrix() -> Result<(), Box<dyn Error>> {
    println!("\nOFFSET / SHELL (each analytic primitive):");
    let cases: [(&str, PrimBuild); 5] = [
        ("box", |t| primitives::make_box(t, 10.0, 10.0, 10.0)),
        ("cylinder", |t| primitives::make_cylinder(t, 5.0, 12.0)),
        ("cone", |t| primitives::make_cone(t, 6.0, 2.0, 12.0)),
        ("sphere", |t| primitives::make_sphere(t, 6.0, 24)),
        ("torus", |t| primitives::make_torus(t, 10.0, 3.0, 32)),
    ];
    for (name, build) in cases {
        let mut topo = Topology::new();
        let s = match build(&mut topo) {
            Ok(s) => s,
            Err(e) => {
                report(
                    "offset",
                    &format!("{name} [build ERR]"),
                    0.0,
                    0,
                    &[format!("err: {e}")],
                );
                continue;
            }
        };
        let _ = drain();
        let t = Instant::now();
        let res = offset_solid_v2(&mut topo, s, 1.0);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut ev = drain();
        match res {
            Ok(r) => report("offset", name, ms, face_count(&topo, r), &ev),
            Err(e) => {
                ev.push(format!("err: {e}"));
                report("offset", &format!("{name} [ERR]"), ms, 0, &ev);
            }
        }
    }
    // Shell (hollow) — excludes the first face.
    let mut topo = Topology::new();
    let s = primitives::make_box(&mut topo, 10.0, 10.0, 10.0)?;
    let exclude = solid_faces(&topo, s)?
        .first()
        .copied()
        .into_iter()
        .collect::<Vec<_>>();
    let _ = drain();
    let t = Instant::now();
    let res = shell_v2(&mut topo, s, 1.0, &exclude);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut ev = drain();
    match res {
        Ok(r) => report("shell", "box (1 face open)", ms, face_count(&topo, r), &ev),
        Err(e) => {
            ev.push(format!("err: {e}"));
            report("shell", "box [ERR]", ms, 0, &ev);
        }
    }
    Ok(())
}

fn make_square_at(topo: &mut Topology, size: f64, z: f64) -> Result<FaceId, Box<dyn Error>> {
    let hs = size / 2.0;
    let tol = 1e-7;
    let v0 = topo.add_vertex(Vertex::new(Point3::new(-hs, -hs, z), tol));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(hs, -hs, z), tol));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(hs, hs, z), tol));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(-hs, hs, z), tol));
    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
    let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));
    let wire = Wire::new(
        vec![
            OrientedEdge::new(e0, true),
            OrientedEdge::new(e1, true),
            OrientedEdge::new(e2, true),
            OrientedEdge::new(e3, true),
        ],
        true,
    )?;
    let wid = topo.add_wire(wire);
    Ok(topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z,
        },
    )))
}

fn nurbs_section() -> Result<(), Box<dyn Error>> {
    println!("\nNURBS-FACED solid (3-profile loft_smooth) — offset must sample+refit:");
    let mut topo = Topology::new();
    let p0 = make_square_at(&mut topo, 6.0, 0.0)?;
    let p1 = make_square_at(&mut topo, 3.0, 5.0)?;
    let p2 = make_square_at(&mut topo, 6.0, 10.0)?;
    let solid = match loft_smooth(&mut topo, &[p0, p1, p2]) {
        Ok(s) => s,
        Err(e) => {
            println!("  loft_smooth construction failed: {e}");
            return Ok(());
        }
    };
    let _ = drain();
    let t = Instant::now();
    let res = offset_solid_v2(&mut topo, solid, 0.5);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut ev = drain();
    match res {
        Ok(r) => report("offset", "nurbs-loft solid", ms, face_count(&topo, r), &ev),
        Err(e) => {
            ev.push(format!("err: {e}"));
            report("offset", "nurbs-loft [ERR]", ms, 0, &ev);
        }
    }
    Ok(())
}

fn blend_matrix() -> Result<(), Box<dyn Error>> {
    println!("\nFILLET / CHAMFER (box, all edges):");
    // rolling-ball fillet
    {
        let mut topo = Topology::new();
        let s = primitives::make_box(&mut topo, 10.0, 10.0, 10.0)?;
        let edges = solid_edges(&topo, s)?;
        let _ = drain();
        let t = Instant::now();
        let res = fillet_rolling_ball(&mut topo, s, &edges, 1.0);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut ev = drain();
        match res {
            Ok(r) => report("fillet-rb", "box all edges", ms, face_count(&topo, r), &ev),
            Err(e) => {
                ev.push(format!("err: {e}"));
                report("fillet-rb", "box [ERR]", ms, 0, &ev);
            }
        }
    }
    // blend-v2 fillet
    {
        let mut topo = Topology::new();
        let s = primitives::make_box(&mut topo, 10.0, 10.0, 10.0)?;
        let edges = solid_edges(&topo, s)?;
        let _ = drain();
        let t = Instant::now();
        let res = fillet_v2(&mut topo, s, &edges, 1.0);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut ev = drain();
        match res {
            Ok(r) => report(
                "fillet-v2",
                "box all edges",
                ms,
                face_count(&topo, r.solid),
                &ev,
            ),
            Err(e) => {
                ev.push(format!("err: {e}"));
                report("fillet-v2", "box [ERR]", ms, 0, &ev);
            }
        }
    }
    // chamfer
    {
        let mut topo = Topology::new();
        let s = primitives::make_box(&mut topo, 10.0, 10.0, 10.0)?;
        let edges = solid_edges(&topo, s)?;
        let _ = drain();
        let t = Instant::now();
        let res = chamfer(&mut topo, s, &edges, 1.0);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut ev = drain();
        match res {
            Ok(r) => report("chamfer", "box all edges", ms, face_count(&topo, r), &ev),
            Err(e) => {
                ev.push(format!("err: {e}"));
                report("chamfer", "box [ERR]", ms, 0, &ev);
            }
        }
    }
    // fillet_v2 on a torus — analytic fast-path declines Torus pairs → walker.
    {
        let mut topo = Topology::new();
        let s = primitives::make_torus(&mut topo, 10.0, 3.0, 32)?;
        let edges = solid_edges(&topo, s)?;
        let _ = drain();
        let t = Instant::now();
        let res = fillet_v2(&mut topo, s, &edges, 0.5);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut ev = drain();
        match res {
            Ok(r) => report(
                "fillet-v2",
                "torus (walker)",
                ms,
                face_count(&topo, r.solid),
                &ev,
            ),
            Err(e) => {
                ev.push(format!("err: {e}"));
                report("fillet-v2", "torus (walker) [ERR]", ms, 0, &ev);
            }
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    log::set_logger(&LOGGER)?;
    log::set_max_level(log::LevelFilter::Debug);

    println!("=== brepkit approximation-path census ===");
    println!("(a probe firing = that op degraded from exact analytic B-Rep)\n");

    boolean_matrix();
    offset_matrix()?;
    nurbs_section()?;
    blend_matrix()?;

    println!("\nLegend: 'exact analytic' = no degradation; 'FALLBACK' = an");
    println!("approximation path fired (see the brepkit_approx probe text).");
    Ok(())
}
