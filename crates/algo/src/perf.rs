//! Deterministic work counters for complexity-regression guards.
//!
//! These count the *inner work* of the boolean hot paths that were once
//! O(N²) — the pave-vertex coincidence lookup and the same-domain polygon
//! clip — so a test can assert the work grows sub-quadratically with input
//! size. Counting work (not wall-clock) makes the guard deterministic: a
//! reintroduced per-item full scan turns a linear count into a quadratic one,
//! which trips the bound with no timing flakiness.
//!
//! The counters are gated behind the `perf-counters` feature. With the feature
//! off (every normal and release build) the `bump_*` calls are empty `#[inline]`
//! functions that compile to nothing, so the instrumented hot loops pay zero
//! cost. The scaling guard enables the feature only for its own test build.

#[cfg(feature = "perf-counters")]
use core::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "perf-counters")]
static PAVE_VERTEX_PROBES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "perf-counters")]
static SD_POLY_CLIPS: AtomicU64 = AtomicU64::new(0);

/// Count one pave-vertex distance comparison (per candidate examined while
/// snapping an intersection endpoint to a coincident vertex).
#[inline]
pub fn bump_pave_vertex_probe() {
    #[cfg(feature = "perf-counters")]
    PAVE_VERTEX_PROBES.fetch_add(1, Ordering::Relaxed);
}

/// Count one same-domain polygon-intersection clip (the expensive narrow-phase
/// in `planar_faces_overlap`).
#[inline]
pub fn bump_sd_poly_clip() {
    #[cfg(feature = "perf-counters")]
    SD_POLY_CLIPS.fetch_add(1, Ordering::Relaxed);
}

/// Reset all counters to zero. Only available with `perf-counters`.
#[cfg(feature = "perf-counters")]
pub fn reset() {
    PAVE_VERTEX_PROBES.store(0, Ordering::Relaxed);
    SD_POLY_CLIPS.store(0, Ordering::Relaxed);
}

/// `(pave_vertex_probes, sd_poly_clips)` since the last [`reset`]. Only
/// available with `perf-counters`.
#[cfg(feature = "perf-counters")]
#[must_use]
pub fn snapshot() -> (u64, u64) {
    (
        PAVE_VERTEX_PROBES.load(Ordering::Relaxed),
        SD_POLY_CLIPS.load(Ordering::Relaxed),
    )
}
