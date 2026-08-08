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
