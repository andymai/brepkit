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
//! every analytic primitive to show they stay exact (no probe fires). A final
//! "remaining paths" section then constructs the inputs the primitive matrix
//! cannot reach — a NURBS-faced loft, a torus, and a 4-valence pyramid apex — so
//! that all seven approximation paths fire at least once.

#![allow(clippy::print_stdout, deprecated, missing_docs)]

use std::error::Error;
use std::sync::Mutex;
use std::time::Instant;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::OperationsError;
use brepkit_operations::blend_ops::{chamfer_v2, fillet_v2};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::fillet::fillet_rolling_ball;
use brepkit_operations::loft::loft_smooth;
use brepkit_operations::offset_face::offset_face;
use brepkit_operations::offset_v2::{offset_solid_v2, shell_v2};
use brepkit_operations::primitives;
use brepkit_operations::revolve::revolve;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
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

/// Build a closed planar profile in the XZ plane (Y-up normal) from
/// `(radial, axial)` points, for revolving about the Z axis.
fn rz_profile(topo: &mut Topology, pts: &[(f64, f64)]) -> Result<FaceId, Box<dyn Error>> {
    let tol = 1e-7;
    let v: Vec<_> = pts
        .iter()
        .map(|(r, z)| topo.add_vertex(Vertex::new(Point3::new(*r, 0.0, *z), tol)))
        .collect();
    let n = v.len();
    let e: Vec<_> = (0..n)
        .map(|i| topo.add_edge(Edge::new(v[i], v[(i + 1) % n], EdgeCurve::Line)))
        .collect();
    let wire = Wire::new(
        (0..n).map(|i| OrientedEdge::new(e[i], true)).collect(),
        true,
    )?;
    let wid = topo.add_wire(wire);
    Ok(topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 1.0, 0.0),
            d: 0.0,
        },
    )))
}

/// Count faces by analytic surface type for the revolve survey.
fn surf_tags(topo: &Topology, s: SolidId) -> String {
    let (mut pl, mut cy, mut co, mut sp, mut to, mut nu) = (0, 0, 0, 0, 0, 0);
    for f in solid_faces(topo, s).unwrap_or_default() {
        // Exhaustive over `FaceSurface` so a new variant is compiler-flagged here
        // (the project forbids a `_ =>` wildcard on `FaceSurface`).
        match topo.face(f).map(|fc| fc.surface().clone()) {
            Ok(FaceSurface::Plane { .. }) => pl += 1,
            Ok(FaceSurface::Cylinder(_)) => cy += 1,
            Ok(FaceSurface::Cone(_)) => co += 1,
            Ok(FaceSurface::Sphere(_)) => sp += 1,
            Ok(FaceSurface::Torus(_)) => to += 1,
            Ok(FaceSurface::Nurbs(_)) => nu += 1,
            Err(_) => {}
        }
    }
    format!("plane={pl} cyl={cy} cone={co} sphere={sp} torus={to} NURBS={nu}")
}

/// Revolve survey: each profile-edge type revolves into its exact analytic
/// surface of revolution — axis-parallel line → `Cylinder`, oblique line →
/// `Cone`, perpendicular line → `Plane`, circular arc → `Torus`. A fully-analytic
/// full revolution with disc caps builds ONE periodic face per profile edge
/// (frustum/cylinder → 3 faces, matching the primitives); profiles with a
/// pointed-cone apex or an annulus cap keep the segmented (analytic but
/// over-segmented) bands — still NURBS-free for the walls.
fn revolve_matrix() {
    println!("\nREVOLVE (profile edges → analytic surfaces of revolution):");
    let z = Point3::new(0.0, 0.0, 0.0);
    let zdir = Vec3::new(0.0, 0.0, 1.0);

    let cases: [(&str, &[(f64, f64)]); 3] = [
        // Oblique outer wall → Cone, perpendicular caps → Plane (solid frustum,
        // caps reach the axis).
        (
            "frustum (oblique→Cone, caps→Plane)",
            &[(6.0, 0.0), (2.0, 12.0), (0.0, 12.0), (0.0, 0.0)],
        ),
        // Axis-parallel outer wall → Cylinder, perpendicular caps → Plane.
        (
            "cylinder (parallel→Cyl, caps→Plane)",
            &[(5.0, 0.0), (5.0, 10.0), (0.0, 10.0), (0.0, 0.0)],
        ),
        // Pointed cone apex on the axis (exercises the apex-band volume guard).
        (
            "pointed cone (apex on axis)",
            &[(5.0, 0.0), (0.0, 12.0), (0.0, 0.0)],
        ),
    ];
    for (name, pts) in cases {
        let mut topo = Topology::new();
        let face = match rz_profile(&mut topo, pts) {
            Ok(f) => f,
            Err(e) => {
                report(
                    "revolve",
                    &format!("{name} [build ERR]"),
                    0.0,
                    0,
                    &[format!("err: {e}")],
                );
                continue;
            }
        };
        report_revolve(&mut topo, name, face, z, zdir);
    }

    // Circular-arc profile edge → Torus band. A half-disc (semicircle arc + its
    // diameter on an axis-parallel line) makes the arc bands `Torus`.
    let mut topo = Topology::new();
    match build_half_disc_profile(&mut topo) {
        Ok(face) => report_revolve(&mut topo, "half-disc (arc→Torus)", face, z, zdir),
        Err(e) => report(
            "revolve",
            "half-disc (arc→Torus) [build ERR]",
            0.0,
            0,
            &[format!("err: {e}")],
        ),
    }
}

/// Build the half-disc profile (a semicircle arc bulging away from the axis,
/// closed by its diameter on an axis-parallel line) for the revolve survey's
/// torus case. Surfaces any construction error to the caller instead of
/// swallowing it.
fn build_half_disc_profile(topo: &mut Topology) -> Result<FaceId, Box<dyn Error>> {
    use brepkit_math::curves::Circle3D;
    use brepkit_topology::edge::Edge;
    use brepkit_topology::vertex::Vertex;
    let circ = Circle3D::new(Point3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 3.0)?;
    let pa = circ.evaluate(-std::f64::consts::FRAC_PI_2);
    let pb = circ.evaluate(std::f64::consts::FRAC_PI_2);
    let va = topo.add_vertex(Vertex::new(pa, 1e-7));
    let vb = topo.add_vertex(Vertex::new(pb, 1e-7));
    let e_arc = topo.add_edge(Edge::new(va, vb, EdgeCurve::Circle(circ)));
    let e_dia = topo.add_edge(Edge::new(vb, va, EdgeCurve::Line));
    let wire = Wire::new(
        vec![
            OrientedEdge::new(e_arc, true),
            OrientedEdge::new(e_dia, true),
        ],
        true,
    )?;
    let wid = topo.add_wire(wire);
    Ok(topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 1.0, 0.0),
            d: 0.0,
        },
    )))
}

/// Revolve a profile a full turn and report via the shared capture/report path
/// (so any `brepkit_approx` probe that fires is surfaced), with the analytic
/// surface-type breakdown appended.
fn report_revolve(topo: &mut Topology, name: &str, face: FaceId, z: Point3, zdir: Vec3) {
    let _ = drain();
    let t = Instant::now();
    let res = revolve(topo, face, z, zdir, std::f64::consts::TAU);
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut ev = drain();
    match res {
        Ok(s) => {
            report("revolve", name, ms, face_count(topo, s), &ev);
            println!("            {}", surf_tags(topo, s));
        }
        Err(e) => {
            ev.push(format!("err: {e}"));
            report("revolve", &format!("{name} [ERR]"), ms, 0, &ev);
        }
    }
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

fn first_nurbs_face(topo: &Topology, solid: SolidId) -> Option<FaceId> {
    solid_faces(topo, solid).ok()?.into_iter().find(|&f| {
        topo.face(f)
            .is_ok_and(|fc| matches!(fc.surface(), FaceSurface::Nurbs(_)))
    })
}

/// Build a triangular planar side face from three pre-made oriented edges; the
/// outward normal is `(b-a)×(c-a)` for the wire `a→b→c`.
fn tri_side(
    topo: &mut Topology,
    oriented: [OrientedEdge; 3],
    a: Point3,
    b: Point3,
    c: Point3,
) -> Result<FaceId, Box<dyn Error>> {
    let normal = (b - a).cross(c - a).normalize()?;
    let d = normal.x() * a.x() + normal.y() * a.y() + normal.z() * a.z();
    let wire = topo.add_wire(Wire::new(oriented.to_vec(), true)?);
    Ok(topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d })))
}

/// Square pyramid: a base square at z=0 and an apex at (0,0,h). The apex is a
/// 4-valence vertex (four edges meet), which is what drives the rolling-ball
/// fillet's non-triangular corner → planar-blend fallback.
fn make_pyramid(topo: &mut Topology, s: f64, h: f64) -> Result<SolidId, Box<dyn Error>> {
    let tol = 1e-7;
    let p0 = Point3::new(-s, -s, 0.0);
    let p1 = Point3::new(s, -s, 0.0);
    let p2 = Point3::new(s, s, 0.0);
    let p3 = Point3::new(-s, s, 0.0);
    let pa = Point3::new(0.0, 0.0, h);
    let v0 = topo.add_vertex(Vertex::new(p0, tol));
    let v1 = topo.add_vertex(Vertex::new(p1, tol));
    let v2 = topo.add_vertex(Vertex::new(p2, tol));
    let v3 = topo.add_vertex(Vertex::new(p3, tol));
    let va = topo.add_vertex(Vertex::new(pa, tol));
    let e01 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e12 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e23 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
    let e30 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));
    let a0 = topo.add_edge(Edge::new(v0, va, EdgeCurve::Line));
    let a1 = topo.add_edge(Edge::new(v1, va, EdgeCurve::Line));
    let a2 = topo.add_edge(Edge::new(v2, va, EdgeCurve::Line));
    let a3 = topo.add_edge(Edge::new(v3, va, EdgeCurve::Line));
    // Base, outward normal -z: wire v0→v3→v2→v1.
    let base_wire = topo.add_wire(Wire::new(
        vec![
            OrientedEdge::new(e30, false),
            OrientedEdge::new(e23, false),
            OrientedEdge::new(e12, false),
            OrientedEdge::new(e01, false),
        ],
        true,
    )?);
    let base = topo.add_face(Face::new(
        base_wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ));
    let s01 = tri_side(
        topo,
        [
            OrientedEdge::new(e01, true),
            OrientedEdge::new(a1, true),
            OrientedEdge::new(a0, false),
        ],
        p0,
        p1,
        pa,
    )?;
    let s12 = tri_side(
        topo,
        [
            OrientedEdge::new(e12, true),
            OrientedEdge::new(a2, true),
            OrientedEdge::new(a1, false),
        ],
        p1,
        p2,
        pa,
    )?;
    let s23 = tri_side(
        topo,
        [
            OrientedEdge::new(e23, true),
            OrientedEdge::new(a3, true),
            OrientedEdge::new(a2, false),
        ],
        p2,
        p3,
        pa,
    )?;
    let s30 = tri_side(
        topo,
        [
            OrientedEdge::new(e30, true),
            OrientedEdge::new(a0, true),
            OrientedEdge::new(a3, false),
        ],
        p3,
        p0,
        pa,
    )?;
    let shell = topo.add_shell(Shell::new(vec![base, s01, s12, s23, s30])?);
    Ok(topo.add_solid(Solid::new(shell, vec![])))
}

/// Trigger the four fallbacks that the primitive matrix does not reach: chamfer
/// has no walker fallback (errors), offset-trim grid-sampling and offset-face
/// raw-surface need a NURBS face, and the rolling-ball planar corner needs a
/// 4-valence vertex.
fn remaining_paths() -> Result<(), Box<dyn Error>> {
    println!("\nREMAINING paths (targeted triggers):");

    // chamfer v2 on a torus: analytic chamfer declines Torus pairs and v1 has
    // no walker → UnsupportedSurface (probe fires, op errors).
    {
        let mut topo = Topology::new();
        let s = primitives::make_torus(&mut topo, 10.0, 3.0, 32)?;
        let edges = solid_edges(&topo, s)?;
        let _ = drain();
        let t = Instant::now();
        let res = chamfer_v2(&mut topo, s, &edges, 0.5, 0.5);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut ev = drain();
        match res {
            Ok(r) => report("chamfer-v2", "torus", ms, face_count(&topo, r.solid), &ev),
            Err(e) => {
                ev.push(format!("err: {e}"));
                report("chamfer-v2", "torus [ERR]", ms, 0, &ev);
            }
        }
    }

    // offset_face on a NURBS face: a gentle offset has no self-intersection so
    // SSI detection finds nothing → grid-sampling trim; a large offset self-
    // intersects past the limit → trim errors → raw offset surface.
    {
        // A sharply waisted loft (8→1→8) folds under a large inward offset.
        let mut topo = Topology::new();
        let p0 = make_square_at(&mut topo, 8.0, 0.0)?;
        let p1 = make_square_at(&mut topo, 1.0, 1.5)?;
        let p2 = make_square_at(&mut topo, 8.0, 3.0)?;
        match loft_smooth(&mut topo, &[p0, p1, p2]) {
            Ok(solid) => match first_nurbs_face(&topo, solid) {
                Some(nf) => {
                    for (label, dist) in [("gentle +0.3", 0.3_f64), ("inward -3.0", -3.0)] {
                        let _ = drain();
                        let t = Instant::now();
                        let res = offset_face(&mut topo, nf, dist, 16);
                        let ms = t.elapsed().as_secs_f64() * 1000.0;
                        let mut ev = drain();
                        match res {
                            // offset_face returns a single face (not a solid).
                            Ok(_) => report("offset_face", label, ms, 1, &ev),
                            Err(e) => {
                                ev.push(format!("err: {e}"));
                                report("offset_face", &format!("{label} [ERR]"), ms, 0, &ev);
                            }
                        }
                    }
                }
                None => println!("  (no NURBS face found on loft solid)"),
            },
            Err(e) => println!("  loft_smooth construction failed: {e}"),
        }
    }

    // rolling-ball fillet on a square pyramid: the 4-valence apex yields a
    // non-triangular corner → flat planar-blend fallback.
    {
        let mut topo = Topology::new();
        let pyr = make_pyramid(&mut topo, 5.0, 8.0)?;
        let edges = solid_edges(&topo, pyr)?;
        let _ = drain();
        let t = Instant::now();
        let res = fillet_rolling_ball(&mut topo, pyr, &edges, 0.8);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let mut ev = drain();
        match res {
            Ok(r) => report(
                "fillet-rb",
                "pyramid (4-valence apex)",
                ms,
                face_count(&topo, r),
                &ev,
            ),
            Err(e) => {
                ev.push(format!("err: {e}"));
                report("fillet-rb", "pyramid [ERR]", ms, 0, &ev);
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
    revolve_matrix();
    blend_matrix()?;
    remaining_paths()?;

    println!("\nLegend: 'exact analytic' = no degradation; 'FALLBACK' = an");
    println!("approximation path fired (see the brepkit_approx probe text).");
    Ok(())
}
