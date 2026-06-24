//! Deterministic work counters for complexity-regression guards.
//!
//! These count the *inner work* of the boolean hot paths that issue #987 found
//! to be O(N²) — so a test can assert the work grows sub-quadratically with
//! input size. Counting work (not wall-clock) makes the guard deterministic: a
//! reintroduced per-item full scan turns a linear count into a quadratic one,
//! which trips the bound with no timing flakiness.
//!
//! Five hot paths were fixed in #990; each has a counter here:
//!
//! | Counter | Hot path | Bounded shape with the fix in |
//! |---|---|---|
//! | `pave_vertex_probes` | PaveFiller endpoint→vertex snap | spatial hash → near-constant per query |
//! | `sd_poly_clips` | `detect_same_domain` polygon clip | bbox gate → ~0 clips |
//! | `ray_geom_builds` | classify sub-faces | ray-cast geometry built once per solid, not per sub-face |
//! | `face_split_probes` | face-splitter section/loop scans | grid index → near-constant candidates per query |
//! | `local_vertex_inserts` | `build_topology_face` vertex pool | layered lookup → only genuinely-new vertices materialized |
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
#[cfg(feature = "perf-counters")]
static RAY_GEOM_BUILDS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "perf-counters")]
static FACE_SPLIT_PROBES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "perf-counters")]
static LOCAL_VERTEX_INSERTS: AtomicU64 = AtomicU64::new(0);

/// Count one pave-vertex distance comparison (per candidate examined while
/// snapping an intersection endpoint to a coincident vertex). Crate-internal:
/// only `reset`/`snapshot` cross the crate boundary (for the scaling guard).
#[inline]
pub(crate) fn bump_pave_vertex_probe() {
    #[cfg(feature = "perf-counters")]
    PAVE_VERTEX_PROBES.fetch_add(1, Ordering::Relaxed);
}

/// Count one same-domain polygon-intersection clip (the expensive narrow-phase
/// in `planar_faces_overlap`). Crate-internal, like `bump_pave_vertex_probe`.
#[inline]
pub(crate) fn bump_sd_poly_clip() {
    #[cfg(feature = "perf-counters")]
    SD_POLY_CLIPS.fetch_add(1, Ordering::Relaxed);
}

/// Count one ray-cast geometry collection for a solid (`collect_face_geoms`).
/// This is the O(faces) build the classify loop now does *once* per argument
/// solid; rebuilding it per sub-face was the quadratic. A regression that
/// classifies via the per-call (uncached) path inside the sub-face loop bumps
/// this once per sub-face, so the count grows with the result's face count.
#[inline]
pub(crate) fn bump_ray_geom_build() {
    #[cfg(feature = "perf-counters")]
    RAY_GEOM_BUILDS.fetch_add(1, Ordering::Relaxed);
}

/// Count one candidate endpoint examined by a face-splitter grid query (the
/// per-section / per-loop "is there a point near this edge" scan). The grid
/// returns only nearby points, so this is near-constant per query; a reverted
/// full scan returns every endpoint per query → O(sections²). Crate-internal.
#[inline]
pub(crate) fn bump_face_split_probe() {
    #[cfg(feature = "perf-counters")]
    FACE_SPLIT_PROBES.fetch_add(1, Ordering::Relaxed);
}

/// Count one vertex materialized into a sub-face's local vertex map during
/// `build_topology_face`. The layered lookup resolves existing vertices by
/// reference from the shared seed/rank pools, so only genuinely-new vertices
/// land here — O(new vertices), linear in the result. Re-seeding the per-sub-face
/// map from the shared pools (the former clone) re-materializes pool-sized state
/// per sub-face → O(pool · sub-faces), quadratic. Crate-internal.
#[inline]
pub(crate) fn bump_local_vertex_insert() {
    #[cfg(feature = "perf-counters")]
    LOCAL_VERTEX_INSERTS.fetch_add(1, Ordering::Relaxed);
}

/// A snapshot of every work counter since the last [`reset`]. Only available
/// with `perf-counters`.
#[cfg(feature = "perf-counters")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerfSnapshot {
    /// Pave-vertex coincidence-lookup candidate comparisons.
    pub pave_vertex_probes: u64,
    /// Same-domain polygon-intersection clips (the expensive narrow-phase).
    pub sd_poly_clips: u64,
    /// Ray-cast geometry collections (`collect_face_geoms` calls).
    pub ray_geom_builds: u64,
    /// Face-splitter grid-query candidate endpoints examined.
    pub face_split_probes: u64,
    /// Sub-face-local vertex materializations in `build_topology_face`.
    pub local_vertex_inserts: u64,
}

/// Reset all counters to zero. Only available with `perf-counters`.
#[cfg(feature = "perf-counters")]
pub fn reset() {
    PAVE_VERTEX_PROBES.store(0, Ordering::Relaxed);
    SD_POLY_CLIPS.store(0, Ordering::Relaxed);
    RAY_GEOM_BUILDS.store(0, Ordering::Relaxed);
    FACE_SPLIT_PROBES.store(0, Ordering::Relaxed);
    LOCAL_VERTEX_INSERTS.store(0, Ordering::Relaxed);
}

/// Every work counter since the last [`reset`]. Only available with
/// `perf-counters`.
#[cfg(feature = "perf-counters")]
#[must_use]
pub fn snapshot() -> PerfSnapshot {
    PerfSnapshot {
        pave_vertex_probes: PAVE_VERTEX_PROBES.load(Ordering::Relaxed),
        sd_poly_clips: SD_POLY_CLIPS.load(Ordering::Relaxed),
        ray_geom_builds: RAY_GEOM_BUILDS.load(Ordering::Relaxed),
        face_split_probes: FACE_SPLIT_PROBES.load(Ordering::Relaxed),
        local_vertex_inserts: LOCAL_VERTEX_INSERTS.load(Ordering::Relaxed),
    }
}
