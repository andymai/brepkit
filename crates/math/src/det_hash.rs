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
//!
//! # Security
//!
//! This is a deterministic, fixed-seed FNV-1a hasher. It is **not**
//! DoS-resistant: an adversary who controls the keys can trivially force
//! collisions. Use it only for internal geometry processing where keys are
//! derived from trusted topology, never for attacker-controlled input.

use std::hash::{BuildHasher, Hasher};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fixed-seed FNV-1a hasher. Deterministic across processes and instances.
///
/// Not DoS-resistant — see the [module docs](self#security). For internal
/// geometry processing only.
#[derive(Clone)]
pub struct DetHasher(u64);

impl DetHasher {
    /// Create a hasher seeded with the FNV-1a offset basis.
    #[must_use]
    pub const fn new() -> Self {
        Self(FNV_OFFSET)
    }
}

impl Default for DetHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for DetHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(FNV_PRIME);
        }
        self.0 = h;
    }
}

/// `BuildHasher` for [`DetHasher`], seeding each hasher at the FNV-1a offset.
#[derive(Default, Clone)]
pub struct DetState;

impl BuildHasher for DetState {
    type Hasher = DetHasher;

    fn build_hasher(&self) -> Self::Hasher {
        DetHasher::new()
    }
}

/// A `HashMap` with deterministic, reproducible iteration order.
pub type DetHashMap<K, V> = std::collections::HashMap<K, V, DetState>;

/// A `HashSet` with deterministic, reproducible iteration order.
pub type DetHashSet<K> = std::collections::HashSet<K, DetState>;

const WORD_MULTIPLIER: u64 = 0xf135_7aea_2e62_a9c5;

/// Fixed-seed hasher that folds in a whole word at each step.
///
/// It takes one add and one multiply for each integer, against eight
/// multiplies for each `u64` in [`DetHasher`]. It is deterministic across
/// processes and instances, like [`DetHasher`], but its values (and so the
/// iteration order of its maps) differ from those of [`DetHasher`].
///
/// Not DoS-resistant — see the [module docs](self#security). For keys made
/// of integers, such as grid cells and ids.
#[derive(Clone, Default)]
pub struct WordHasher(u64);

impl WordHasher {
    #[inline]
    const fn add(&mut self, word: u64) {
        self.0 = self.0.wrapping_add(word).wrapping_mul(WORD_MULTIPLIER);
    }
}

impl Hasher for WordHasher {
    #[inline]
    fn finish(&self) -> u64 {
        // The multiply leaves its best mixed bits at the top, and a hash
        // table indexes with the bottom ones.
        self.0.rotate_left(26)
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0u8; 8];
            word.copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(word) ^ ((rest.len() as u64) << 59));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn write_u128(&mut self, i: u128) {
        self.add(i as u64);
        self.add((i >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn write_i8(&mut self, i: i8) {
        self.add(i as u64);
    }

    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn write_i16(&mut self, i: i16) {
        self.add(i as u64);
    }

    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn write_i32(&mut self, i: i32) {
        self.add(i as u64);
    }

    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn write_i64(&mut self, i: i64) {
        self.add(i as u64);
    }

    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn write_i128(&mut self, i: i128) {
        self.write_u128(i as u128);
    }

    #[inline]
    #[allow(clippy::cast_sign_loss)]
    fn write_isize(&mut self, i: isize) {
        self.add(i as u64);
    }
}

/// `BuildHasher` for [`WordHasher`].
#[derive(Default, Clone)]
pub struct WordState;

impl BuildHasher for WordState {
    type Hasher = WordHasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        WordHasher::default()
    }
}

/// A `HashMap` over [`WordHasher`]: deterministic, and quick on integer keys.
pub type WordHashMap<K, V> = std::collections::HashMap<K, V, WordState>;

/// A `HashSet` over [`WordHasher`]: deterministic, and quick on integer keys.
pub type WordHashSet<K> = std::collections::HashSet<K, WordState>;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn hash_writes(writes: &[&[u8]]) -> u64 {
        let mut h = DetHasher::new();
        for w in writes {
            h.write(w);
        }
        h.finish()
    }

    #[test]
    fn empty_hash_is_offset_basis() {
        assert_eq!(DetHasher::new().finish(), FNV_OFFSET);
    }

    #[test]
    fn distinct_sequences_differ() {
        assert_ne!(hash_writes(&[b"abc"]), hash_writes(&[b"abd"]));
    }

    #[test]
    fn write_is_order_sensitive() {
        assert_ne!(hash_writes(&[b"ab", b"cd"]), hash_writes(&[b"cd", b"ab"]),);
    }

    #[test]
    fn zero_state_does_not_restart() {
        // A running state that reaches exactly zero must keep folding in
        // subsequent bytes rather than silently reverting to the offset
        // basis. Find a single byte that drives the state to 0, then verify
        // that appending more bytes keeps the two distinct streams distinct.
        let target = (0u16..=u16::from(u8::MAX)).map(|b| b as u8).find(|&b| {
            let mut h = DetHasher::new();
            h.write(&[b]);
            h.finish() == 0
        });
        if let Some(zeroing) = target {
            let mut a = DetHasher::new();
            a.write(&[zeroing]);
            a.write(&[0x01]);

            // A hasher that began life at zero would, under the old buggy
            // logic, hash [0x01] identically. Build that comparison stream.
            let mut b = DetHasher::new();
            b.write(&[0x01]);
            assert_ne!(a.finish(), b.finish());
        }

        // Regardless of whether a zeroing byte exists for this FNV constant,
        // assert the invariant that distinct streams stay distinct even when
        // one of them is the empty (offset-basis) stream.
        assert_ne!(hash_writes(&[]), hash_writes(&[b"x"]));
    }

    fn word_hash<T: std::hash::Hash>(value: &T) -> u64 {
        WordState.hash_one(value)
    }

    #[test]
    fn word_hash_is_fixed_and_tells_keys_apart() {
        assert_eq!(
            word_hash(&(1_i64, -2_i64, 3_i64)),
            word_hash(&(1_i64, -2_i64, 3_i64))
        );
        assert_ne!(
            word_hash(&(1_i64, 2_i64, 3_i64)),
            word_hash(&(3_i64, 2_i64, 1_i64))
        );
        assert_ne!(word_hash(&(0_u32, 1_u32)), word_hash(&(1_u32, 0_u32)));
        let mut short = WordHasher::default();
        short.write(&[0]);
        let mut empty = WordHasher::default();
        empty.write(&[]);
        assert_ne!(short.finish(), empty.finish());
    }

    /// Neighbouring grid cells spread over the buckets of a small table: the
    /// bottom bits of their hashes, which index the table, all differ.
    #[test]
    fn neighbouring_cells_take_distinct_low_bits() {
        let mut low = std::collections::HashSet::new();
        for x in -4_i64..4 {
            for y in -4_i64..4 {
                for z in -2_i64..2 {
                    low.insert(word_hash(&(x, y, z)) & 0xffff);
                }
            }
        }
        assert!(low.len() > 250, "{}", low.len());
    }
}
