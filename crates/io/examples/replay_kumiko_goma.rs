#![allow(clippy::print_stdout, clippy::expect_used, missing_docs)]
//! Replay the captured kumiko `goma` compound_cut operands natively.
//!
//! The tool's `kumiko › goma carves a 1×1×6 bin` export runs ~48s and then
//! throws "recursive use of an object detected which would lead to unsafe
//! aliasing in rust" — wasm borrow poisoning, i.e. a panic inside the kernel.
//! Native replay turns that into a real panic with a backtrace.
//!
//! REFUTED 2026-07-24: this captured `compound_cut` is NOT where the poisoning
//! happens. Full 180-tool replay completes in 11.8s with F=1146, free=0,
//! over=0 — a clean result. The goma export failure is somewhere else in that
//! scenario's chain, so do not replay this capture expecting a repro.
//!
//! Capture: `~/.cache/brepkit-parity-captures/2026-07-23/kumiko-goma/`
//! Usage: `cargo run --release --example replay_kumiko_goma -p brepkit-io -- [N]`

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::compound_cut;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;

fn main() {
    let dir = std::env::var_os("CAPTURE_DIR").map_or_else(
        || {
            let mut p = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
            p.push(".cache");
            p.push("brepkit-parity-captures");
            p.push("2026-07-23");
            p.push("kumiko-goma");
            p
        },
        PathBuf::from,
    );
    let limit: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(usize::MAX);

    let mut topo = Topology::new();
    let region = deserialize_solid(
        &std::fs::read(dir.join("cut1-region.bin")).expect("region"),
        &mut topo,
    )
    .expect("region parse");

    let mut tools = Vec::new();
    for i in 0.. {
        let p = dir.join(format!("cut1-tool{i}.bin"));
        if !p.exists() || tools.len() >= limit {
            break;
        }
        tools.push(
            deserialize_solid(&std::fs::read(&p).expect("tool"), &mut topo).expect("tool parse"),
        );
    }
    println!("loaded region + {} tools", tools.len());

    if let Some(chunks) = std::env::var("CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        // A zero CHUNK (or no tools) would make the batch size zero and panic;
        // this harness exists to produce numbers, so say so and stop.
        if chunks == 0 || tools.is_empty() {
            println!("CHUNK={chunks} with {} tools: nothing to do", tools.len());
            return;
        }
        let per = tools.len().div_ceil(chunks);
        let mut acc = region;
        for (i, batch) in tools.chunks(per).enumerate() {
            let t = Instant::now();
            match compound_cut(
                &mut topo,
                acc,
                batch,
                brepkit_operations::boolean::BooleanOptions::default(),
            ) {
                Ok(next) => {
                    // Never report F=0 on an enumeration error — a silent zero
                    // would corrupt the complexity/timing series this prints.
                    let f = match solid_faces(&topo, next) {
                        Ok(v) => v.len(),
                        Err(e) => {
                            println!("  batch {i}: face enumeration failed: {e}");
                            return;
                        }
                    };
                    println!(
                        "  batch {i}: {} tools {}ms -> F={f}",
                        batch.len(),
                        t.elapsed().as_millis()
                    );
                    acc = next;
                }
                Err(e) => {
                    println!("  batch {i}: {} tools ERR {e}", batch.len());
                    return;
                }
            }
        }
        return;
    }

    let t0 = Instant::now();
    let result = compound_cut(
        &mut topo,
        region,
        &tools,
        brepkit_operations::boolean::BooleanOptions::default(),
    );
    let ms = t0.elapsed().as_millis();

    match result {
        Ok(sid) => {
            let faces = solid_faces(&topo, sid).expect("faces");
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
            let free = uses.values().filter(|&&c| c == 1).count();
            let over = uses.values().filter(|&&c| c > 2).count();
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
