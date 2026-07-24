#![allow(clippy::print_stdout, clippy::expect_used, missing_docs)]
//! Replay a captured `cut(base, tools...)` natively from arena `.bin` files.
//!
//! Generic over captures laid out as `<prefix>-base.bin` + `<prefix>-tool<i>.bin`.
//! Built for the goma pattern-application call, which the tool-side probes
//! narrowed to a single `cutAll` of EIGHT tools taking **203.5 s** (telemetry:
//! one batch attempt, one success, no fallbacks — honest N-way work, not retry
//! churn). Capture:
//! `~/.cache/brepkit-parity-captures/2026-07-24/goma-bisect/`
//!
//! Usage:
//!   CAPTURE_DIR=<dir> PREFIX=gomabisect \
//!     cargo run --release --example replay_cut_capture -p brepkit-io [N]
//!
//! `N` limits the tool count, which is how you get the cost-vs-tool-count
//! curve without waiting for the full run.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOptions, compound_cut};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;

fn describe(topo: &Topology, sid: brepkit_topology::solid::SolidId, label: &str) {
    let faces = solid_faces(topo, sid).expect("faces");
    let mut mix: HashMap<&str, usize> = HashMap::new();
    for &fid in &faces {
        *mix.entry(topo.face(fid).expect("face").surface().type_tag())
            .or_default() += 1;
    }
    let mut mix: Vec<_> = mix.into_iter().collect();
    mix.sort_unstable();
    println!("  {label}: F={} mix={mix:?}", faces.len());
}

fn main() {
    let dir = PathBuf::from(std::env::var_os("CAPTURE_DIR").unwrap_or_default());
    let prefix = std::env::var("PREFIX").unwrap_or_else(|_| "gomabisect".to_string());
    let limit: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(usize::MAX);

    let mut topo = Topology::new();
    let base_path = dir.join(format!("{prefix}-base.bin"));
    let base = deserialize_solid(
        &std::fs::read(&base_path).expect("base .bin — set CAPTURE_DIR"),
        &mut topo,
    )
    .expect("base parse");

    let mut tools = Vec::new();
    for i in 0.. {
        let p = dir.join(format!("{prefix}-tool{i}.bin"));
        if !p.exists() || tools.len() >= limit {
            break;
        }
        tools.push(deserialize_solid(&std::fs::read(&p).expect("tool"), &mut topo).expect("parse"));
    }
    if tools.is_empty() {
        println!("no tools found for prefix '{prefix}' in {}", dir.display());
        return;
    }

    // XSCAN=<v>: list X-normal planes near v in each operand, to tell whether a
    // thin slab is pre-existing in the inputs or produced by the boolean.
    if let Ok(v) = std::env::var("XSCAN") {
        let target: f64 = v.parse().expect("XSCAN");
        let report = |label: &str, sid: brepkit_topology::solid::SolidId| {
            let mut xs: Vec<f64> = Vec::new();
            for fid in solid_faces(&topo, sid).expect("faces") {
                if let brepkit_topology::face::FaceSurface::Plane { normal, d } =
                    topo.face(fid).expect("face").surface()
                    && normal.x().abs() > 0.99
                {
                    let x = d / normal.x();
                    if (x - target).abs() < 1.0 {
                        xs.push(x);
                    }
                }
            }
            xs.sort_by(|a, b| a.partial_cmp(b).expect("cmp"));
            xs.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
            println!("  {label}: X-normal planes near {target}: {xs:?}");
        };
        report("base", base);
        for (i, &t) in tools.iter().enumerate() {
            report(&format!("tool{i}"), t);
        }
        return;
    }

    println!("loaded base + {} tools", tools.len());
    describe(&topo, base, "base");
    for (i, &t) in tools.iter().enumerate() {
        describe(&topo, t, &format!("tool{i}"));
    }

    // RAW=1: call the analytic GFA directly, bypassing the ops-level gate and
    // its mesh fallback, to see whether GFA itself produces a usable result.
    if std::env::var("RAW").is_ok() {
        let mut acc = base;
        for (i, &tool) in tools.iter().enumerate() {
            let t = Instant::now();
            match brepkit_algo::gfa::boolean(
                &mut topo,
                brepkit_algo::bop::BooleanOp::Cut,
                acc,
                tool,
            ) {
                Ok(next) => {
                    let faces = solid_faces(&topo, next).expect("faces");
                    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
                    let mut mix: HashMap<&str, usize> = HashMap::new();
                    for &fid in &faces {
                        let f = topo.face(fid).expect("face");
                        *mix.entry(f.surface().type_tag()).or_default() += 1;
                        for wid in
                            std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied())
                        {
                            for oe in topo.wire(wid).expect("wire").edges() {
                                *uses.entry(oe.edge()).or_default() += 1;
                            }
                        }
                    }
                    let free = uses.values().filter(|&&c| c == 1).count();
                    let over = uses.values().filter(|&&c| c > 2).count();
                    let mut mix: Vec<_> = mix.into_iter().collect();
                    mix.sort_unstable();
                    println!(
                        "  RAW cut {i}: {}ms F={} mix={mix:?} free={free} over={over}",
                        t.elapsed().as_millis(),
                        faces.len()
                    );
                    if free > 0 && std::env::var("DUMP_FREE").is_ok() {
                        for (eid, _) in uses.iter().filter(|&(_, &c)| c == 1) {
                            let e = topo.edge(*eid).expect("edge");
                            let a = topo.vertex(e.start()).expect("v").point();
                            let b = topo.vertex(e.end()).expect("v").point();
                            // Which face owns it, and what surface is that face?
                            let owner = faces.iter().find(|&&fid| {
                                let f = topo.face(fid).expect("face");
                                std::iter::once(f.outer_wire())
                                    .chain(f.inner_wires().iter().copied())
                                    .any(|w| {
                                        topo.wire(w)
                                            .expect("wire")
                                            .edges()
                                            .iter()
                                            .any(|oe| oe.edge() == *eid)
                                    })
                            });
                            let tag = owner.map_or("?", |&fid| {
                                topo.face(fid).expect("face").surface().type_tag()
                            });
                            println!(
                                "    free {} on {tag} ({:.2},{:.2},{:.2})->({:.2},{:.2},{:.2})",
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
                    acc = next;
                }
                Err(e) => {
                    println!("  RAW cut {i}: {}ms ERR {e}", t.elapsed().as_millis());
                    return;
                }
            }
        }
        return;
    }

    let t0 = Instant::now();
    let result = compound_cut(&mut topo, base, &tools, BooleanOptions::default());
    let ms = t0.elapsed().as_millis();

    match result {
        Ok(sid) => {
            let faces = solid_faces(&topo, sid).expect("faces");
            let mut uses: HashMap<EdgeId, usize> = HashMap::new();
            for &fid in &faces {
                let f = topo.face(fid).expect("face");
                for wid in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
                    for oe in topo.wire(wid).expect("wire").edges() {
                        *uses.entry(oe.edge()).or_default() += 1;
                    }
                }
            }
            let free = uses.values().filter(|&&c| c == 1).count();
            let over = uses.values().filter(|&&c| c > 2).count();
            let mut mix: HashMap<&str, usize> = HashMap::new();
            for &fid in &faces {
                *mix.entry(topo.face(fid).expect("face").surface().type_tag())
                    .or_default() += 1;
            }
            let mut mix: Vec<_> = mix.into_iter().collect();
            mix.sort_unstable();
            println!(
                "ok in {ms}ms: F={} mix={mix:?} free={free} over={over}",
                faces.len()
            );
        }
        Err(e) => println!("ERR in {ms}ms: {e}"),
    }
}
