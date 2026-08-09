//! Replay a captured `filletVariable` call: deserialize the input solid,
//! match the captured per-edge radii onto the fresh arena by endpoint pairs
//! (midpoint parameters are not portable across arenas), run
//! `fillet_variable`, and report the result's health.
//!
//! Usage: `F=<input.bin> SPEC=<spec.json> cargo run --release -p brepkit-io \
//!   --example replay_fillet_variable`
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::collections::HashMap;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::fillet::{FilletRadiusLaw, fillet_variable};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn census(topo: &Topology, solid: SolidId, label: &str) {
    let mut types: HashMap<&'static str, usize> = HashMap::new();
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        *types.entry(face.surface().type_tag()).or_insert(0) += 1;
        let mut wires = vec![face.outer_wire()];
        wires.extend_from_slice(face.inner_wires());
        for wid in wires {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    let free = uses.values().filter(|&&c| c == 1).count();
    let over = uses.values().filter(|&&c| c > 2).count();
    if let Ok(spec) = std::env::var("DUMP_NEAR") {
        let v: Vec<f64> = spec.split(',').map(|x| x.parse().unwrap()).collect();
        let target = Point3::new(v[0], v[1], v[2]);
        let r = v[3];
        for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
            let face = topo.face(fid).unwrap();
            let tag = face.surface().type_tag();
            let mut wires = vec![face.outer_wire()];
            wires.extend_from_slice(face.inner_wires());
            for wid in wires {
                for oe in topo.wire(wid).unwrap().edges() {
                    let e = topo.edge(oe.edge()).unwrap();
                    let (a, b) = (
                        topo.vertex(e.start()).unwrap().point(),
                        topo.vertex(e.end()).unwrap().point(),
                    );
                    if (a - target).length() < r || (b - target).length() < r {
                        println!(
                            "  NEAR e{} {} ({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4}) on {fid:?}:{tag}",
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
    }
    if std::env::var("ORIENT").is_ok() {
        let mut senses: HashMap<usize, Vec<(String, bool)>> = HashMap::new();
        for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
            let face = topo.face(fid).unwrap();
            let rev = face.is_reversed();
            let tag = format!("{fid:?}:{}", face.surface().type_tag());
            let mut wires = vec![face.outer_wire()];
            wires.extend_from_slice(face.inner_wires());
            for wid in wires {
                for oe in topo.wire(wid).unwrap().edges() {
                    let eff = oe.is_forward() ^ rev;
                    senses
                        .entry(oe.edge().index())
                        .or_default()
                        .push((tag.clone(), eff));
                }
            }
        }
        let mut bad = 0;
        for (eidx, users) in &senses {
            if users.len() == 2 && users[0].1 == users[1].1 {
                bad += 1;
                let e = topo.edge(topo.edge_id_from_index(*eidx).unwrap()).unwrap();
                let (a, b) = (
                    topo.vertex(e.start()).unwrap().point(),
                    topo.vertex(e.end()).unwrap().point(),
                );
                println!(
                    "  SAMESENSE-BREP e{eidx} {} ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3}) users={:?}",
                    e.curve().type_tag(),
                    a.x(),
                    a.y(),
                    a.z(),
                    b.x(),
                    b.y(),
                    b.z(),
                    users
                );
            }
        }
        println!("  {label} brep_same_sense={bad}");
    }
    if std::env::var("TESS_BND").is_ok() {
        match brepkit_operations::tessellate::tessellate_solid_with_tolerance(
            topo,
            solid,
            0.01,
            5.0_f64.to_radians(),
        ) {
            Ok(mesh) => {
                println!(
                    "  {label} tess_bnd={} tess_nm={}",
                    brepkit_operations::tessellate::boundary_edge_count(&mesh),
                    brepkit_operations::tessellate::non_manifold_edge_count(&mesh)
                );
                if std::env::var("TESS_BND").is_ok_and(|v| v == "2") {
                    let mut edge_uses: HashMap<(usize, usize), usize> = HashMap::new();
                    let mut edge_tri: HashMap<(usize, usize), usize> = HashMap::new();
                    for (ti, t) in mesh.indices.chunks_exact(3).enumerate() {
                        let t = [t[0] as usize, t[1] as usize, t[2] as usize];
                        for k in 0..3 {
                            let (a, b) = (t[k], t[(k + 1) % 3]);
                            *edge_uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                            edge_tri.insert((a.min(b), a.max(b)), ti);
                        }
                    }
                    let pos = |i: usize| {
                        let p = mesh.positions[i];
                        (p.x(), p.y(), p.z())
                    };
                    let mut rows: Vec<String> = edge_uses
                        .iter()
                        .filter(|&(_, &c)| c == 1)
                        .map(|(&(a, b), _)| {
                            let (pa, pb) = (pos(a), pos(b));
                            format!(
                                "  BND [{a},{b}] tri={} ({:.6},{:.6},{:.6})->({:.6},{:.6},{:.6})",
                                edge_tri.get(&(a.min(b), a.max(b))).copied().unwrap_or(0),
                                pa.0,
                                pa.1,
                                pa.2,
                                pb.0,
                                pb.1,
                                pb.2
                            )
                        })
                        .collect();
                    for (&(a, b), &c) in &edge_uses {
                        if c >= 3 {
                            let (pa, pb) = (pos(a), pos(b));
                            println!(
                                "  NM x{c} [{a},{b}] ({:.4},{:.4},{:.4})->({:.4},{:.4},{:.4})",
                                pa.0, pa.1, pa.2, pb.0, pb.1, pb.2
                            );
                        }
                    }
                    let mut directed: HashMap<(usize, usize), usize> = HashMap::new();
                    for t in mesh.indices.chunks_exact(3) {
                        let t = [t[0] as usize, t[1] as usize, t[2] as usize];
                        for k in 0..3 {
                            *directed.entry((t[k], t[(k + 1) % 3])).or_insert(0) += 1;
                        }
                    }
                    for &(a, b) in directed.keys() {
                        if a < b && directed.contains_key(&(b, a)) {
                            continue;
                        }
                        if a > b && directed.contains_key(&(b, a)) {
                            continue;
                        }
                        let und = edge_uses.get(&(a.min(b), a.max(b))).copied().unwrap_or(0);
                        if und >= 2 {
                            let (pa, pb) = (pos(a), pos(b));
                            rows.push(format!(
                                "  SAMESENSE x{und} ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3})",
                                pa.0, pa.1, pa.2, pb.0, pb.1, pb.2
                            ));
                        }
                    }
                    rows.sort();
                    for r in &rows {
                        println!("{r}");
                    }
                }
            }
            Err(e) => println!("  {label} tess_err={e}"),
        }
    }
    let mut rows: Vec<_> = types.into_iter().collect();
    rows.sort_unstable();
    println!("  {label}: mix={rows:?} free={free} over={over}");
}

struct Tap;
impl log::Log for Tap {
    fn enabled(&self, _m: &log::Metadata) -> bool {
        true
    }
    fn log(&self, r: &log::Record) {
        println!("    [log {}] {}", r.level(), r.args());
    }
    fn flush(&self) {}
}
static TAP: Tap = Tap;

fn main() {
    let _ = log::set_logger(&TAP);
    log::set_max_level(log::LevelFilter::Debug);
    let input = std::env::var_os("F").expect("F=<input.bin>");
    let spec_path = std::env::var_os("SPEC").expect("SPEC=<spec.json>");

    let mut topo = Topology::new();
    let solid = deserialize_solid(&std::fs::read(input).unwrap(), &mut topo).unwrap();
    census(&topo, solid, "INPUT");

    let specs: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(spec_path).unwrap()).unwrap();

    // Endpoint pairs are the portable identity: the tool-side midpoint eval
    // uses the stored curve's own parameterization, which need not put the
    // segment midpoint at t=0.5.
    let mut seen: std::collections::HashSet<brepkit_topology::edge::EdgeId> =
        std::collections::HashSet::new();
    let mut edge_ends: Vec<(brepkit_topology::edge::EdgeId, Point3, Point3)> = Vec::new();
    for fid in brepkit_topology::explorer::solid_faces(&topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        let mut wires = vec![face.outer_wire()];
        wires.extend_from_slice(face.inner_wires());
        for wid in wires {
            for oe in topo.wire(wid).unwrap().edges() {
                let eid = oe.edge();
                if !seen.insert(eid) {
                    continue;
                }
                let e = topo.edge(eid).unwrap();
                edge_ends.push((
                    eid,
                    topo.vertex(e.start()).unwrap().point(),
                    topo.vertex(e.end()).unwrap().point(),
                ));
            }
        }
    }
    let mut edge_laws: Vec<(brepkit_topology::edge::EdgeId, FilletRadiusLaw)> = Vec::new();
    for spec in &specs {
        let v = spec["verts"].as_array().unwrap();
        let f = |i: usize| v[i].as_f64().unwrap();
        let (pa, pb) = (Point3::new(f(0), f(1), f(2)), Point3::new(f(3), f(4), f(5)));
        let r = spec["startRadius"].as_f64().unwrap();
        let (best, dist) = edge_ends
            .iter()
            .map(|(eid, a, b)| {
                let d = ((*a - pa).length() + (*b - pb).length())
                    .min((*a - pb).length() + (*b - pa).length());
                (*eid, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        println!("  match edge {best:?} dist={dist:.6} r={r}");
        assert!(dist < 1e-3, "edge endpoint match too far: {dist}");
        edge_laws.push((best, FilletRadiusLaw::Constant(r)));
    }

    let t = std::time::Instant::now();
    match fillet_variable(&mut topo, solid, &edge_laws) {
        Ok(result) => {
            println!("-- fillet_variable {}ms --", t.elapsed().as_millis());
            census(&topo, result, "RESULT");
            if std::env::var("FREE_DETAIL").is_ok() {
                let mut users: HashMap<
                    usize,
                    (Vec<String>, Point3, Point3, brepkit_topology::edge::EdgeId),
                > = HashMap::new();
                for fid in brepkit_topology::explorer::solid_faces(&topo, result).unwrap() {
                    let face = topo.face(fid).unwrap();
                    let tag = face.surface().type_tag();
                    let mut wires = vec![face.outer_wire()];
                    wires.extend_from_slice(face.inner_wires());
                    for wid in wires {
                        for oe in topo.wire(wid).unwrap().edges() {
                            let e = topo.edge(oe.edge()).unwrap();
                            let (a, b) = (
                                topo.vertex(e.start()).unwrap().point(),
                                topo.vertex(e.end()).unwrap().point(),
                            );
                            users
                                .entry(oe.edge().index())
                                .or_insert_with(|| (Vec::new(), a, b, oe.edge()))
                                .0
                                .push(format!("{fid:?}:{tag}"));
                        }
                    }
                }
                if std::env::var("FREE_DETAIL").is_ok_and(|v| v == "2") {
                    let free_owner_faces: std::collections::HashSet<String> = users
                        .values()
                        .filter(|(o, ..)| o.len() == 1)
                        .map(|(o, ..)| o[0].clone())
                        .collect();
                    for fid in brepkit_topology::explorer::solid_faces(&topo, result).unwrap() {
                        let face = topo.face(fid).unwrap();
                        let tag = face.surface().type_tag();
                        if !free_owner_faces.contains(&format!("{fid:?}:{tag}")) {
                            continue;
                        }
                        println!("  owner {fid:?} {tag} rev={}", face.is_reversed());
                        for oe in topo.wire(face.outer_wire()).unwrap().edges() {
                            let e = topo.edge(oe.edge()).unwrap();
                            let (a, b) = (
                                topo.vertex(e.start()).unwrap().point(),
                                topo.vertex(e.end()).unwrap().point(),
                            );
                            println!(
                                "    e{} {} fwd={} ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3})",
                                oe.edge().index(),
                                e.curve().type_tag(),
                                oe.is_forward(),
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
                for (eidx, (owners, a, b, eid)) in &users {
                    if owners.len() != 1 {
                        continue;
                    }
                    let desc = topo.edge(*eid).map_or_else(
                        |_| "?".to_string(),
                        |e| match e.curve() {
                            brepkit_topology::edge::EdgeCurve::Line => "line".to_string(),
                            brepkit_topology::edge::EdgeCurve::Circle(c) => format!(
                                "circle c=({:.4},{:.4},{:.4}) r={:.5} n=({:.3},{:.3},{:.3})",
                                c.center().x(),
                                c.center().y(),
                                c.center().z(),
                                c.radius(),
                                c.normal().x(),
                                c.normal().y(),
                                c.normal().z()
                            ),
                            brepkit_topology::edge::EdgeCurve::Ellipse(_) => "ellipse".to_string(),
                            brepkit_topology::edge::EdgeCurve::NurbsCurve(n) => {
                                let (t0, t1) = n.domain();
                                let m = n.evaluate(f64::midpoint(t0, t1));
                                format!(
                                    "nurbs cp={} mid=({:.4},{:.4},{:.4})",
                                    n.control_points().len(),
                                    m.x(),
                                    m.y(),
                                    m.z()
                                )
                            }
                        },
                    );
                    println!(
                        "  FREE e{eidx} {desc} ({:.2},{:.2},{:.2})->({:.2},{:.2},{:.2}) owner={:?}",
                        a.x(),
                        a.y(),
                        a.z(),
                        b.x(),
                        b.y(),
                        b.z(),
                        owners
                    );
                }
            }
        }
        Err(e) => println!("-- fillet_variable FAILED: {e}"),
    }
}
