//! Diagnostic harness for the 64-cut perf nondeterminism.
//!
//! Background: HashMap iteration nondeterminism in the GFA boolean
//! pipeline drives wide per-iter variance on `bench_boolean_64_holes`
//! (criterion suite "hangs" because some iterations randomly hit a
//! slow path — typically the mesh-boolean fallback at
//! `boolean/mod.rs:516` for ~6 min per iter). The two source-side
//! fixes in this PR (sorted `vv_vertex_seed` + sorted SD pairs) narrow
//! the variance but don't fully close it; more HashMap iteration sites
//! likely remain. See `memory/project_64cut-perf-bisect.md` for the
//! full diagnosis and history.
//!
//! This test is `#[ignore]`d — it's an explicit-only diagnostic for
//! the next investigator to localize which cut introduces divergence
//! between two runs.
//!
//! Run with:
//!   cargo test --release -p brepkit-operations \
//!       --test perf_64cut_determinism -- --ignored --nocapture

#![allow(clippy::unwrap_used, clippy::print_stdout)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer;
use brepkit_topology::solid::SolidId;

/// Run a 64-cut sequence, snapshotting (face, edge, vertex) counts after
/// each cut. Returns a Vec of 64 snapshots.
fn run_64_cut_snapshot() -> Vec<(usize, usize, usize)> {
    let mut topo = Topology::new();
    let mut result: SolidId = primitives::make_box(&mut topo, 100.0, 100.0, 10.0).unwrap();
    let mut snapshots = Vec::with_capacity(64);
    for row in 0..8 {
        for col in 0..8 {
            let cyl = primitives::make_cylinder(&mut topo, 2.0, 20.0).unwrap();
            let mat = Mat4::translation(
                6.0 + f64::from(col) * 12.0,
                6.0 + f64::from(row) * 12.0,
                -5.0,
            );
            transform_solid(&mut topo, cyl, &mat).unwrap();
            result = boolean(&mut topo, BooleanOp::Cut, result, cyl).unwrap();
            let (f, e, v) = explorer::solid_entity_counts(&topo, result).unwrap();
            snapshots.push((f, e, v));
        }
    }
    snapshots
}

/// Run the 64-cut sequence twice in one process and report the first
/// cut where the (face, edge, vertex) count snapshots diverge between
/// runs. Successive HashMap creations in the same thread draw different
/// random seeds from the thread-local RNG, so cross-run divergence in a
/// single process is the same kind of nondeterminism that drives the
/// bench's slow-path occurrence.
///
/// After the two source fixes in this PR (sd_pairs sort,
/// vv_vertex_seed sorted Vec) the divergence still occurs at cut 1;
/// follow this harness to localize the next site.
#[test]
#[ignore = "diagnostic — explicit-only nondeterminism bisect tool; fails until full determinism is achieved"]
fn diverge_first_cut() {
    let a = run_64_cut_snapshot();
    let b = run_64_cut_snapshot();
    let mut divergence: Option<usize> = None;
    for (i, (sa, sb)) in a.iter().zip(b.iter()).enumerate() {
        if sa != sb {
            println!(
                "DIVERGE at cut {i}: A=(f={},e={},v={}) vs B=(f={},e={},v={})",
                sa.0, sa.1, sa.2, sb.0, sb.1, sb.2
            );
            for j in i.saturating_sub(3)..=i {
                println!(
                    "  cut {j}: A=(f={},e={},v={}) B=(f={},e={},v={})",
                    a[j].0, a[j].1, a[j].2, b[j].0, b[j].1, b[j].2
                );
            }
            divergence = Some(i);
            break;
        }
    }
    // Fail loudly so explicit `--ignored` runs (and any script wrapping
    // this) get an unambiguous non-zero exit when nondeterminism is
    // present. Today this assertion fails (the two source-side fixes in
    // this PR narrow but don't close the variance); when the next round
    // of HashMap iteration fixes lands, this assertion will start
    // passing and the test becomes a real determinism gate.
    assert!(
        divergence.is_none(),
        "64-cut sequence is nondeterministic — first divergence at cut {} (full trace above)",
        divergence.unwrap_or(0)
    );
    println!("No divergence in 64 cuts — runs are deterministic");
}
