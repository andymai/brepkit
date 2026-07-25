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
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for &fid in &faces {
        let f = topo.face(fid).expect("face");
        *mix.entry(f.surface().type_tag()).or_default() += 1;
        for wid in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).expect("wire").edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    let mut mix: Vec<_> = mix.into_iter().collect();
    mix.sort_unstable();
    // Ray-cast parity is only meaningful against a CLOSED operand: a point
    // outside a watertight solid must cross it an even number of times. An
    // operand with free edges silently poisons every classification.
    let free = uses.values().filter(|&&c| c == 1).count();
    let over = uses.values().filter(|&&c| c > 2).count();
    println!(
        "  {label}: F={} mix={mix:?} free={free} over={over}",
        faces.len()
    );
}

struct DropLogger;
impl log::Log for DropLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        // Only the assembler's shell chatter — an unconditional `true` here
        // would route every record in the workspace through `log()`.
        m.target().starts_with("brepkit_algo") && m.level() <= log::Level::Debug
    }
    fn log(&self, r: &log::Record) {
        if !self.enabled(r.metadata()) {
            return;
        }
        let msg = format!("{}", r.args());
        if msg.contains("growth sliver")
            || msg.contains("growth shell")
            || msg.contains("FF_TRACE")
            || msg.contains("SUBFACE")
            || msg.contains("RAYTRACE")
        {
            println!("    [algo] {msg}");
        }
    }
    fn flush(&self) {}
}
static DROP_LOGGER: DropLogger = DropLogger;

fn main() {
    if std::env::var("SHELL_LOG").is_ok() {
        let _ = log::set_logger(&DROP_LOGGER);
        log::set_max_level(log::LevelFilter::Debug);
    }

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

    // POINT_IN=x,y,z: classify that point against the base and every tool.
    // Answers "is this splitter interior point genuinely inside the cutter?",
    // which separates an incomplete face split from a classifier misjudgement.
    if let Ok(spec) = std::env::var("POINT_IN") {
        // Parse exactly three floats. Discarding unparseable tokens would let
        // POINT_IN=1,2,foo,3 run silently as (1,2,3) — a probe answering about
        // a different point than asked is worse than no probe.
        let tokens: Vec<&str> = spec.split(',').map(str::trim).collect();
        let v: Vec<f64> = tokens
            .iter()
            .map(|t| {
                t.parse::<f64>()
                    .expect("POINT_IN component must be a float")
            })
            .collect();
        assert!(
            v.len() == 3,
            "POINT_IN needs exactly x,y,z — got {} component(s)",
            v.len()
        );
        let p = brepkit_math::vec::Point3::new(v[0], v[1], v[2]);
        let labelled = std::iter::once(("base".to_string(), base)).chain(
            tools
                .iter()
                .enumerate()
                .map(|(i, &t)| (format!("tool{i}"), t)),
        );
        for (label, sid) in labelled {
            match brepkit_operations::classify::classify_point(&topo, sid, p, 0.01, 1e-7) {
                Ok(c) => println!("  POINT_IN {label} {sid:?}: {c:?}"),
                Err(e) => println!("  POINT_IN {label} {sid:?}: ERR {e}"),
            }
        }
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

    // BASE_FACES_NEAR_X=<v>: same slab scan, but on the INPUT operands, to tell
    // whether a trim boundary pre-exists or is produced by the boolean.
    if let Ok(v) = std::env::var("BASE_FACES_NEAR_X") {
        let target: f64 = v.parse().expect("BASE_FACES_NEAR_X");
        let scan = |label: &str, sid: brepkit_topology::solid::SolidId| {
            for fid in solid_faces(&topo, sid).expect("faces") {
                let f = topo.face(fid).expect("face");
                let (mut lo, mut hi) = (f64::MAX, f64::MIN);
                for wid in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
                    for oe in topo.wire(wid).expect("wire").edges() {
                        let e = topo.edge(oe.edge()).expect("edge");
                        for vid in [e.start(), e.end()] {
                            let x = topo.vertex(vid).expect("vertex").point().x();
                            lo = lo.min(x);
                            hi = hi.max(x);
                        }
                    }
                }
                if lo <= hi && lo < target + 0.1 && hi > target - 0.1 {
                    println!(
                        "    {label} {fid:?} {} x[{lo:.3},{hi:.3}]",
                        f.surface().type_tag()
                    );
                }
            }
        };
        scan("base", base);
        if let Some(&t0) = tools.first() {
            scan("tool0", t0);
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
        // TOOL=<i>: cut the base by that ONE tool, instead of chaining from 0.
        let single = std::env::var("TOOL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let selected: Vec<(usize, brepkit_topology::solid::SolidId)> = match single {
            Some(i) if i < tools.len() => vec![(i, tools[i])],
            _ => tools.iter().copied().enumerate().collect(),
        };
        let mut acc = base;
        for (i, tool) in selected {
            let t = Instant::now();
            // OP=cut|fuse|intersect — if the faces survive under a different op,
            // the splitter created them and classification is dropping them; if
            // every op loses them, the splitter never made them.
            let op = match std::env::var("OP")
                .unwrap_or_else(|_| "cut".into())
                .as_str()
            {
                "fuse" => brepkit_algo::bop::BooleanOp::Fuse,
                "intersect" => brepkit_algo::bop::BooleanOp::Intersect,
                _ => brepkit_algo::bop::BooleanOp::Cut,
            };
            match brepkit_algo::gfa::boolean(&mut topo, op, acc, tool) {
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
                    if let Ok(v) = std::env::var("FACES_NEAR_X") {
                        // Which faces actually exist in the sliver slab? If the
                        // splitter never made the missing patches, nothing here
                        // will span the gap.
                        let target: f64 = v.parse().expect("FACES_NEAR_X");
                        for &fid in &faces {
                            let f = topo.face(fid).expect("face");
                            let mut lo = f64::MAX;
                            let mut hi = f64::MIN;
                            let mut n = 0;
                            for wid in std::iter::once(f.outer_wire())
                                .chain(f.inner_wires().iter().copied())
                            {
                                for oe in topo.wire(wid).expect("wire").edges() {
                                    let e = topo.edge(oe.edge()).expect("edge");
                                    for vid in [e.start(), e.end()] {
                                        let x = topo.vertex(vid).expect("vertex").point().x();
                                        lo = lo.min(x);
                                        hi = hi.max(x);
                                        n += 1;
                                    }
                                }
                            }
                            if n > 0 && lo < target + 0.1 && hi > target - 0.1 {
                                println!(
                                    "    face {fid:?} {} x[{lo:.3},{hi:.3}] edges={}",
                                    f.surface().type_tag(),
                                    n / 2
                                );
                            }
                        }
                    }
                    if free > 0 && std::env::var("FREE_OWNERS").is_ok() {
                        // Each free edge is used by exactly one face. Those
                        // faces are the rim around the hole, so their surfaces
                        // say which input face should have owned the missing
                        // patch.
                        let mut rows: Vec<(usize, String, usize, [f64; 3])> = Vec::new();
                        for &fid in &faces {
                            let f = topo.face(fid).expect("face");
                            let n = std::iter::once(f.outer_wire())
                                .chain(f.inner_wires().iter().copied())
                                .flat_map(|wid| topo.wire(wid).expect("wire").edges())
                                .filter(|oe| uses.get(&oe.edge()).copied() == Some(1))
                                .count();
                            if n == 0 {
                                continue;
                            }
                            let w = topo.wire(f.outer_wire()).expect("w");
                            let p = topo
                                .vertex(topo.edge(w.edges()[0].edge()).expect("e").start())
                                .expect("v")
                                .point();
                            rows.push((
                                fid.index(),
                                f.surface().type_tag().to_string(),
                                n,
                                [p.x(), p.y(), p.z()],
                            ));
                        }
                        rows.sort_unstable_by_key(|r| r.0);
                        println!("    free-edge rim faces: {}", rows.len());
                        for (fid, tag, n, p) in rows {
                            println!(
                                "      face Id({fid}) {tag} free={n} at ({:.3},{:.3},{:.3})",
                                p[0], p[1], p[2]
                            );
                        }
                    }
                    if std::env::var("FACE_WIRES").is_ok() {
                        // Sections reach the two corner cylinders but no sliver
                        // sub-faces come out. Either the splitter declined the
                        // split (one plain outer wire) or it mis-wired the face
                        // (an inner wire duplicating the outer). Print the wire
                        // structure of every curved face carrying more edges
                        // than an untouched quarter-cylinder's four.
                        for &fid in &faces {
                            let f = topo.face(fid).expect("face");
                            if f.surface().is_planar() {
                                continue;
                            }
                            let wires: Vec<_> = std::iter::once(f.outer_wire())
                                .chain(f.inner_wires().iter().copied())
                                .collect();
                            let total: usize = wires
                                .iter()
                                .map(|&w| topo.wire(w).expect("wire").edges().len())
                                .sum();
                            if total <= 4 {
                                continue;
                            }
                            println!(
                                "    face {fid:?} {} wires={} edges={total}",
                                f.surface().type_tag(),
                                wires.len()
                            );
                            for (k, &wid) in wires.iter().enumerate() {
                                let w = topo.wire(wid).expect("wire");
                                let kind = if k == 0 { "outer" } else { "inner" };
                                println!("      {kind} wire {wid:?} n={}", w.edges().len());
                                for oe in w.edges() {
                                    let e = topo.edge(oe.edge()).expect("edge");
                                    let a = topo.vertex(e.start()).expect("v").point();
                                    let b = topo.vertex(e.end()).expect("v").point();
                                    let f = if uses.get(&oe.edge()).copied() == Some(1) {
                                        " FREE"
                                    } else {
                                        ""
                                    };
                                    println!(
                                        "        e{} {} ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3}){f}",
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
                    if free > 0 && std::env::var("FREE_LOOPS").is_ok() {
                        // Free edges bound the hole(s) left by dropped faces.
                        // Chain them by shared vertex: each closed chain is one
                        // missing face's outline.
                        let mut segs: Vec<(usize, usize)> = Vec::new();
                        for (eid, _) in uses.iter().filter(|&(_, &c)| c == 1) {
                            let e = topo.edge(*eid).expect("edge");
                            segs.push((e.start().index(), e.end().index()));
                        }
                        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
                        for &(a, b) in &segs {
                            adj.entry(a).or_default().push(b);
                            adj.entry(b).or_default().push(a);
                        }
                        let mut seen: std::collections::HashSet<usize> =
                            std::collections::HashSet::new();
                        let mut loops = 0;
                        for &(a, _) in &segs {
                            if !seen.insert(a) {
                                continue;
                            }
                            let mut stack = vec![a];
                            let mut n = 1;
                            while let Some(v) = stack.pop() {
                                for &w in adj.get(&v).into_iter().flatten() {
                                    if seen.insert(w) {
                                        n += 1;
                                        stack.push(w);
                                    }
                                }
                            }
                            loops += 1;
                            println!("    free component {loops}: {n} vertices");
                        }
                        // A simple closed outline needs EVERY vertex at degree
                        // exactly 2. Even-degree alone is not enough: a degree-4
                        // junction (figure-eight) is even but is not one loop.
                        let mut deg: HashMap<usize, usize> = HashMap::new();
                        for (v, ns) in &adj {
                            *deg.entry(ns.len()).or_default() += 1;
                            let _ = v;
                        }
                        let mut deg: Vec<_> = deg.into_iter().collect();
                        deg.sort_unstable();
                        if std::env::var("LOOP_GEOM").is_ok() {
                            // Print each component's edges so the missing face's
                            // surface can be read off the loop it bounds.
                            let mut comp: HashMap<usize, usize> = HashMap::new();
                            let mut cid = 0;
                            for &(a, _) in &segs {
                                if comp.contains_key(&a) {
                                    continue;
                                }
                                cid += 1;
                                let mut stack = vec![a];
                                while let Some(v) = stack.pop() {
                                    if comp.insert(v, cid).is_some() {
                                        continue;
                                    }
                                    for &w in adj.get(&v).into_iter().flatten() {
                                        if !comp.contains_key(&w) {
                                            stack.push(w);
                                        }
                                    }
                                }
                            }
                            for want in 1..=cid {
                                println!("    --- component {want} ---");
                                for (eid, _) in uses.iter().filter(|&(_, &c)| c == 1) {
                                    let e = topo.edge(*eid).expect("edge");
                                    if comp.get(&e.start().index()) != Some(&want) {
                                        continue;
                                    }
                                    let a = topo.vertex(e.start()).expect("v").point();
                                    let b = topo.vertex(e.end()).expect("v").point();
                                    println!(
                                        "      {} ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3})",
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
                        let all_two = adj.values().all(|ns| ns.len() == 2);
                        println!(
                            "    free components={loops} degree histogram(deg:count)={deg:?} all_degree_2={all_two}"
                        );
                    }
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
