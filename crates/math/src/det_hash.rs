//! Deterministic hashing primitives.
//!
//! The standard-library `HashMap`/`HashSet` seed their hasher from a
//! process-global source, so iteration order — and therefore any algorithm
//! whose output depends on the order it visits map entries — varies between
//! runs (and even between successive map instances within one process, since
//! the seed counter advances per instance). For geometry kernels this surfaces
//! as nondeterministic mesh welding and triangulation: the same input can
//! produce different vertex merges and face counts run-to-run.
//!
//! [`DetHashMap`] / [`DetHashSet`] use a fixed-seed FNV-1a hasher so iteration
//! order is fully reproducible for a given set of keys. Use them anywhere map
//! iteration order feeds geometry construction (vertex welds, triangulation,
//! shell assembly).

use std::hash::{BuildHasherDefault, Hasher};

/// Fixed-seed FNV-1a hasher. Deterministic across processes and instances.
#[derive(Default, Clone)]
pub struct DetHasher(u64);

impl Hasher for DetHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = if self.0 == 0 { FNV_OFFSET } else { self.0 };
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
        self.0 = h;
    }
}

/// `BuildHasher` for [`DetHasher`].
pub type DetState = BuildHasherDefault<DetHasher>;

/// A `HashMap` with deterministic, reproducible iteration order.
pub type DetHashMap<K, V> = std::collections::HashMap<K, V, DetState>;

/// A `HashSet` with deterministic, reproducible iteration order.
pub type DetHashSet<K> = std::collections::HashSet<K, DetState>;
