//! Native replay of the gridfinity tool's assembly parts
//! (`assemblyPartTemplate.ts`): a bar eased on its vertical corners and top
//! rim, then slotted into a comb. Runs the same `try_fillet` chain the tool's
//! `fillet` binding takes, with the tool's radius reductions, and reports
//! face census, edge uses, mesh watertightness and volume after every stage.
//!
//! `cargo test -p brepkit-wasm --lib assembly_probe -- --ignored --nocapture`
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::cast_precision_loss
)]

use std::collections::{HashMap, HashSet};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{self, BooleanOptions};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{
    boundary_edge_count, non_manifold_edge_count, tessellate_solid_with_tolerance,
};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::solid::SolidId;

use crate::helpers::try_fillet;

/// The tool's `COPLANAR_OVERLAP`: parts sink this far into the base.
const SINK: f64 = 0.01;

/// Prints the blend engine's debug log (`LOG=1`).
struct Tap;

impl log::Log for Tap {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        eprintln!("    [{}] {}", record.level(), record.args());
    }
    fn flush(&self) {}
}

static TAP: Tap = Tap;

fn install_log_tap() {
    if std::env::var("LOG").is_ok() {
        let _ = log::set_logger(&TAP);
        log::set_max_level(log::LevelFilter::Debug);
    }
}
const CORNER_FILLET_MM: f64 = 2.5;
const TOP_EASE_MM: f64 = 1.0;
const FALLBACK_FACTORS: [f64; 4] = [1.0, 0.75, 0.5, 0.25];
const MIN_FILLET_RADIUS: f64 = 0.1;

fn box_at(topo: &mut Topology, size: [f64; 3], min: [f64; 3]) -> SolidId {
    let solid = make_box(topo, size[0], size[1], size[2]).unwrap();
    transform_solid(topo, solid, &Mat4::translation(min[0], min[1], min[2])).unwrap();
    solid
}

fn edges_where(
    topo: &Topology,
    solid: SolidId,
    pred: impl Fn(Point3, Point3) -> bool,
) -> Vec<EdgeId> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let eid = oe.edge();
                if !seen.insert(eid) {
                    continue;
                }
                let e = topo.edge(eid).unwrap();
                let a = topo.vertex(e.start()).unwrap().point();
                let b = topo.vertex(e.end()).unwrap().point();
                if pred(a, b) {
                    out.push(eid);
                }
            }
        }
    }
    out.sort_by_key(|e| e.index());
    out
}

/// The tool's `verticalEdges`: zero XY extent, more than 0.5 tall.
fn vertical_edges(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
    edges_where(topo, solid, |a, b| {
        (a.x() - b.x()).abs() < 1e-3 && (a.y() - b.y()).abs() < 1e-3 && (a.z() - b.z()).abs() > 0.5
    })
}

/// The tool's `edgesNearPlane`: both ends within 0.4 of the plane.
fn edges_near_plane(topo: &Topology, solid: SolidId, z: f64) -> Vec<EdgeId> {
    edges_where(topo, solid, |a, b| {
        (a.z() - z).abs() <= 0.4 && (b.z() - z).abs() <= 0.4
    })
}

fn report(topo: &Topology, solid: SolidId, label: &str) {
    let mut types: HashMap<&'static str, usize> = HashMap::new();
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        *types.entry(face.surface().type_tag()).or_insert(0) += 1;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    let free = uses.values().filter(|&&c| c == 1).count();
    let over = uses.values().filter(|&&c| c > 2).count();
    let mut mix: Vec<_> = types.into_iter().collect();
    mix.sort_unstable();
    let (bnd, nm) = tessellate_solid_with_tolerance(topo, solid, 0.01, 5.0_f64.to_radians())
        .map_or((usize::MAX, usize::MAX), |mesh| {
            (boundary_edge_count(&mesh), non_manifold_edge_count(&mesh))
        });
    let volume = solid_volume(topo, solid, 0.01).unwrap_or(f64::NAN);
    let oriented = oriented_solid_volume(topo, solid, 0.01).unwrap_or(f64::NAN);
    eprintln!(
        "  {label}: mix={mix:?} free={free} over={over} tess_bnd={bnd} tess_nm={nm} volume={volume:.3} oriented={oriented:.3}"
    );
}

/// Every edge with an endpoint within `radius` of `target`, with its curve
/// type, endpoints, and owning face types.
fn dump_near(topo: &Topology, solid: SolidId, target: Point3, radius: f64) {
    let mut rows = Vec::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        let tag = face.surface().type_tag();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = topo.vertex(e.start()).unwrap().point();
                let b = topo.vertex(e.end()).unwrap().point();
                if (a - target).length() <= radius || (b - target).length() <= radius {
                    rows.push(format!(
                        "    e{} {} ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3}) on {fid:?}:{tag} rev={} fwd={}",
                        oe.edge().index(),
                        e.curve().type_tag(),
                        a.x(),
                        a.y(),
                        a.z(),
                        b.x(),
                        b.y(),
                        b.z(),
                        face.is_reversed(),
                        oe.is_forward()
                    ));
                }
            }
        }
    }
    rows.sort();
    for row in rows {
        eprintln!("{row}");
    }
}

/// Every face's wires: edge id, curve type and oriented endpoints.
fn dump_wires(topo: &Topology, solid: SolidId) {
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        eprintln!(
            "    {fid:?} {} rev={}",
            face.surface().type_tag(),
            face.is_reversed()
        );
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = topo.vertex(oe.oriented_start(e)).unwrap().point();
                let b = topo.vertex(oe.oriented_end(e)).unwrap().point();
                eprintln!(
                    "      e{} {} ({:.2},{:.2},{:.2})->({:.2},{:.2},{:.2})",
                    oe.edge().index(),
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
}

/// Signed volume contribution of every face (tetrahedra from the origin):
/// a face whose sign disagrees with its neighbours is inverted.
fn dump_face_flux(topo: &Topology, solid: SolidId) {
    let faces = brepkit_topology::explorer::solid_faces(topo, solid).unwrap();
    let (mesh, offsets) = brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance(
        topo,
        solid,
        0.01,
        5.0_f64.to_radians(),
    )
    .unwrap();
    let mut center = brepkit_math::vec::Vec3::new(0.0, 0.0, 0.0);
    for p in &mesh.positions {
        center += brepkit_math::vec::Vec3::new(p.x(), p.y(), p.z());
    }
    let center = center * (1.0 / mesh.positions.len().max(1) as f64);
    for (i, fid) in faces.iter().enumerate() {
        let (from, to) = (offsets[i] as usize, offsets[i + 1] as usize);
        let mut flux = 0.0;
        for tri in mesh.indices[from..to].chunks_exact(3) {
            let a = mesh.positions[tri[0] as usize];
            let b = mesh.positions[tri[1] as usize];
            let c = mesh.positions[tri[2] as usize];
            let (a, b, c) = (
                brepkit_math::vec::Vec3::new(a.x(), a.y(), a.z()) - center,
                brepkit_math::vec::Vec3::new(b.x(), b.y(), b.z()) - center,
                brepkit_math::vec::Vec3::new(c.x(), c.y(), c.z()) - center,
            );
            flux += a.dot(b.cross(c)) / 6.0;
        }
        let face = topo.face(*fid).unwrap();
        eprintln!(
            "    {fid:?} {} rev={} tris={} flux={flux:.1}",
            face.surface().type_tag(),
            face.is_reversed(),
            (to - from) / 3
        );
    }
}

/// The tool's `applyFilletWithFallback`: the wasm chain, then the radius
/// reductions, else the input unchanged.
fn fillet_step(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
    label: &str,
) -> SolidId {
    match brepkit_operations::blend_ops::fillet_v2(topo, solid, edges, radius) {
        Ok(result) => {
            eprintln!(
                "  {label}: v2 at r={radius}: succeeded={} failed={} partial={}",
                result.succeeded.len(),
                result.failed.len(),
                result.is_partial
            );
            for (edge, reason) in result.failed.iter().take(2) {
                eprintln!("    v2 failed {edge:?}: {reason}");
            }
            if result.failed.is_empty() {
                report(topo, result.solid, "    v2 result");
            }
        }
        Err(reason) => eprintln!("  {label}: v2 at r={radius} errored: {reason}"),
    }
    for factor in FALLBACK_FACTORS {
        let r = radius * factor;
        if r < MIN_FILLET_RADIUS {
            break;
        }
        match try_fillet(topo, solid, edges, r) {
            Ok(result) => {
                eprintln!("  {label}: {} edges filleted at r={r}", edges.len());
                return result;
            }
            Err(error) => {
                eprintln!("  {label}: r={r} failed: {error}");
                match brepkit_operations::blend_ops::fillet_v2(topo, solid, edges, r) {
                    Ok(result) => {
                        eprintln!(
                            "    v2: succeeded={} failed={} partial={}",
                            result.succeeded.len(),
                            result.failed.len(),
                            result.is_partial
                        );
                        for (edge, reason) in result.failed.iter().take(2) {
                            eprintln!("    v2 failed {edge:?}: {reason}");
                        }
                        if result.failed.is_empty() {
                            report(topo, result.solid, "    v2 result");
                        }
                    }
                    Err(reason) => eprintln!("    v2 error: {reason}"),
                }
            }
        }
    }
    eprintln!("  {label}: every radius failed, part kept unfilleted");
    solid
}

/// A prism's four vertical corner fillets must come out watertight: each
/// stripe ends on an untouched cap that takes the notch.
#[test]
fn prism_vertical_corner_fillets_are_watertight() {
    let mut topo = Topology::new();
    let (width, depth, height) = (70.0_f64, 14.0_f64, 35.0_f64);
    let body = box_at(
        &mut topo,
        [width, depth, height + SINK],
        [-width / 2.0, -depth / 2.0, -SINK],
    );
    let verticals = vertical_edges(&topo, body);
    assert_eq!(verticals.len(), 4);
    let eased = try_fillet(&mut topo, body, &verticals, CORNER_FILLET_MM).unwrap();
    let mesh = tessellate_solid_with_tolerance(&topo, eased, 0.01, 5.0_f64.to_radians()).unwrap();
    assert_eq!(
        boundary_edge_count(&mesh),
        0,
        "corner fillets leave open mesh edges"
    );
    let volume = solid_volume(&topo, eased, 0.01).unwrap();
    let oriented = oriented_solid_volume(&topo, eased, 0.01).unwrap();
    // N422: the 1.0 mm^3 agreement this assertion used to require held only while
    // BOTH estimators were coarse. `solid_volume` takes the exact analytic
    // per-face path for this solid (planes + quadrics), while
    // `oriented_solid_volume` integrates an *inscribed* mesh at the requested
    // deflection, which under-counts every curved face by construction. N415's
    // stripe-deflection repair (carried here by N422) refines those stripes from
    // 252 to 8450 triangles, which removes ~207 mm^3 of inscribed-mesh
    // under-count (33912.67 -> 34119.87 against 34121.26 exact) and leaves the
    // two estimators 1.385 mm^3 apart -- about 4e-5 of the volume, and the
    // expected deflection-level residue of the mesh route rather than the
    // inverted face this check exists to catch (an inversion moves a whole face's
    // flux, orders of magnitude more). The tolerance is therefore widened to a
    // value that still catches an inverted face but admits the inscribed-mesh
    // under-count at this deflection.
    assert!(
        (volume - oriented).abs() < 5.0,
        "volume {volume} vs oriented {oriented}: a face is inverted"
    );
}

/// The assembly base's floor plate (`assemblyGenerator.ts`): a 2x1 deck,
/// 83.5 x 41.5 x 2.01 with r=4 corners, whose top rim the junction pass
/// eases at r=1.5. Reports the mesh at the export settings and at the
/// wasm gate's coarser settings.
#[test]
#[ignore = "diagnostic — native replay of the assembly base's floor plate rim ease"]
fn floor_plate_probe() {
    install_log_tap();
    let mut topo = Topology::new();
    let (width, depth, thickness) = (83.5_f64, 41.5_f64, 2.01_f64);
    let plate = box_at(
        &mut topo,
        [width, depth, thickness],
        [-width / 2.0, -depth / 2.0, 0.0],
    );
    let corners = vertical_edges(&topo, plate);
    let plate = fillet_step(&mut topo, plate, &corners, 4.0, "plate corners");
    report(&topo, plate, "after corners");
    let rim = edges_near_plane(&topo, plate, thickness);
    eprintln!("  rim edges: {}", rim.len());
    let eased = fillet_step(&mut topo, plate, &rim, 1.5, "rim ease");
    report(&topo, eased, "after rim ease");
    for (deflection, angular) in [(0.01, 5.0_f64), (0.1, 10.0)] {
        let mesh = tessellate_solid_with_tolerance(&topo, eased, deflection, angular.to_radians())
            .unwrap();
        eprintln!(
            "  mesh at deflection {deflection} angular {angular}: boundary={} nonmanifold={} tris={}",
            boundary_edge_count(&mesh),
            non_manifold_edge_count(&mesh),
            mesh.indices.len() / 3
        );
    }
}

/// Every edge of a 10 mm cube at r=1 (the `try_fillet_all_box_edges` case):
/// per-face flux and the winding log.
#[test]
#[ignore = "diagnostic — all twelve cube edges through the v2 engine with per-face flux"]
fn box_all_edges_probe() {
    install_log_tap();
    let mut topo = Topology::new();
    let cube = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let edges = edges_where(&topo, cube, |_, _| true);
    assert_eq!(edges.len(), 12);
    match brepkit_operations::blend_ops::fillet_v2(&mut topo, cube, &edges, 1.0) {
        Ok(result) => {
            eprintln!(
                "  v2: succeeded={} failed={} partial={}",
                result.succeeded.len(),
                result.failed.len(),
                result.is_partial
            );
            for (edge, reason) in result.failed.iter().take(2) {
                eprintln!("    v2 failed {edge:?}: {reason}");
            }
            report(&topo, result.solid, "all edges r=1");
            dump_face_flux(&topo, result.solid);
        }
        Err(reason) => eprintln!("  v2 error: {reason}"),
    }
}

/// `assemblyPartTemplate.ts` `case 'comb'` with the combriser test's bar
/// (70 x 14 x 35, four 9 mm slots 25 deep).
#[test]
#[ignore = "diagnostic — native replay of the tool's comb part (corner and rim fillets, then slot cuts)"]
fn comb_part_probe() {
    install_log_tap();
    let mut topo = Topology::new();
    let (width, depth, height) = (70.0_f64, 14.0_f64, 35.0_f64);
    let total = height + SINK;
    let body = box_at(
        &mut topo,
        [width, depth, total],
        [-width / 2.0, -depth / 2.0, -SINK],
    );
    report(&topo, body, "bar");
    let corner = CORNER_FILLET_MM.min(width / 8.0).min(depth / 4.0);
    let verticals = vertical_edges(&topo, body);
    let body = fillet_step(&mut topo, body, &verticals, corner, "vertical corners");
    report(&topo, body, "after corners");
    if std::env::var("STAGE").is_ok_and(|stage| stage == "corners") {
        if std::env::var("DUMP_WIRES").is_ok() {
            dump_wires(&topo, body);
        } else if std::env::var("FLUX").is_ok() {
            dump_face_flux(&topo, body);
        } else {
            dump_near(
                &topo,
                body,
                Point3::new(width / 2.0, depth / 2.0, height),
                3.0,
            );
        }
        return;
    }
    let rim = edges_near_plane(&topo, body, height);
    let body = fillet_step(
        &mut topo,
        body,
        &rim,
        TOP_EASE_MM.min(depth / 5.0),
        "top rim",
    );
    report(&topo, body, "after rim");
    if std::env::var("STAGE").is_ok_and(|stage| stage == "rim") {
        if std::env::var("DUMP_WIRES").is_ok() {
            dump_wires(&topo, body);
        } else if std::env::var("FLUX").is_ok() {
            dump_face_flux(&topo, body);
        } else {
            dump_near(
                &topo,
                body,
                Point3::new(width / 2.0, depth / 2.0, height),
                3.0,
            );
        }
        return;
    }

    let slot_count = 4;
    let slot_width = 9.0_f64.min(width / slot_count as f64 - 2.0);
    let slot_depth = 25.0_f64.min(height - 1.5);
    let pitch = width / slot_count as f64;
    let mut cutters = Vec::new();
    for i in 0..slot_count {
        let cx = -width / 2.0 + pitch * (i as f64 + 0.5);
        let size = [slot_width, depth + 2.0, slot_depth + 1.0];
        let center_z = height - f64::midpoint(slot_depth, 1.0) + 1.0;
        cutters.push(box_at(
            &mut topo,
            size,
            [cx - size[0] / 2.0, -size[1] / 2.0, center_z - size[2] / 2.0],
        ));
    }
    let before = boolean::mesh_fallback_count();
    let carved =
        boolean::compound_cut(&mut topo, body, &cutters, BooleanOptions::default()).unwrap();
    eprintln!(
        "  slots: {} cutters, fallbacks={}",
        cutters.len(),
        boolean::mesh_fallback_count() - before
    );
    report(&topo, carved, "after slots");
}
