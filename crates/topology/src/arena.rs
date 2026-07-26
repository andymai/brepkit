//! A typed arena allocator for topological entities.
//!
//! Entities are stored in a `Vec` and referenced by typed index handles.
//! This provides O(1) access and avoids reference counting.

use std::marker::PhantomData;

/// A typed index handle into an [`Arena`].
///
/// The type parameter `T` ensures that an `Id<Vertex>` cannot be used
/// to look up an `Edge`, for example.
pub struct Id<T> {
    index: usize,
    _marker: PhantomData<fn() -> T>,
}

// Manual impls to avoid requiring T: Debug/Clone/etc.

impl<T> std::fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Id").field(&self.index).finish()
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Id<T> {}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index.cmp(&other.index)
    }
}

impl<T> std::hash::Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<T> Id<T> {
    /// Returns the raw index of this handle.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }
}

/// A typed arena allocator.
///
/// Stores values of type `T` in a contiguous `Vec` and hands out
/// [`Id<T>`] handles for O(1) lookup.
#[derive(Debug, Clone)]
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Arena<T> {
    /// Creates a new, empty arena.
    #[must_use]
    pub const fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Creates a new arena with the given capacity pre-allocated.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
        }
    }

    /// Reserves capacity for at least `additional` more entries, growing by at
    /// most an eighth of the arena.
    ///
    /// Neither `Vec` primitive is usable here at CAD scale, for opposite
    /// reasons. `reserve` rounds up to `max(2 * capacity, len + additional)`, so
    /// a bulk-insert hint on an arena already holding millions of entries
    /// requests a full DOUBLING whose reallocation holds the old and new buffers
    /// at once — invisible on a 64-bit host, fatal on wasm32 (linear memory caps
    /// at 4GB, and the failed allocation reaches `handle_alloc_error` →
    /// `abort()`, trapping the instance with no panic message). `reserve_exact`
    /// removes the spike but leaves `capacity == len`, so the NEXT hint copies
    /// the whole arena again — O(n²) across the thousands of booleans a large
    /// export runs.
    ///
    /// Growing by `len / 8` keeps growth geometric (so total copying stays
    /// amortized O(n)) while capping the transient overshoot at 12.5% instead of
    /// 100%. `Vec::push` past the hint still amortizes on its own.
    pub fn reserve(&mut self, additional: usize) {
        let len = self.items.len();
        if self.items.capacity() - len >= additional {
            return;
        }
        self.items.reserve_exact(additional.max(len / 8));
    }

    /// Allocates a new entry in the arena and returns its typed handle.
    pub fn alloc(&mut self, value: T) -> Id<T> {
        let index = self.items.len();
        self.items.push(value);
        Id {
            index,
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the value at `id`, or `None` if the id
    /// is out of bounds.
    #[must_use]
    pub fn get(&self, id: Id<T>) -> Option<&T> {
        self.items.get(id.index)
    }

    /// Returns a mutable reference to the value at `id`, or `None` if
    /// the id is out of bounds.
    #[must_use]
    pub fn get_mut(&mut self, id: Id<T>) -> Option<&mut T> {
        self.items.get_mut(id.index)
    }

    /// Returns the number of entries in the arena.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the arena contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns an iterator over all `(Id<T>, &T)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (Id<T>, &T)> {
        self.items.iter().enumerate().map(|(i, v)| {
            (
                Id {
                    index: i,
                    _marker: PhantomData,
                },
                v,
            )
        })
    }

    /// Reconstructs a typed [`Id`] from a raw index, returning `None`
    /// if the index is out of bounds.
    ///
    /// This is intended for FFI boundaries (e.g. WASM) where handles
    /// are passed as plain integers.
    #[must_use]
    pub fn id_from_index(&self, index: usize) -> Option<Id<T>> {
        self.items.get(index).map(|_| Id {
            index,
            _marker: PhantomData,
        })
    }

    /// Returns a mutable iterator over all `(Id<T>, &mut T)` pairs.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Id<T>, &mut T)> {
        self.items.iter_mut().enumerate().map(|(i, v)| {
            (
                Id {
                    index: i,
                    _marker: PhantomData,
                },
                v,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// A bulk-insert hint on a large arena must not double it.
    ///
    /// `Vec::reserve` would round up to `2 * capacity`; on wasm32 that
    /// reallocation holds both buffers and aborts the instance once the arena
    /// reaches millions of entries (the goma export died exactly here, on an
    /// arena of 5.78M edges). The overshoot must stay bounded — but capacity
    /// must still grow geometrically, or the next hint re-copies everything.
    #[test]
    fn reserve_hint_grows_a_large_arena_by_a_bounded_fraction() {
        let mut arena: Arena<u32> = Arena::new();
        for i in 0..10_000 {
            arena.alloc(i);
        }
        arena.items.shrink_to_fit();
        let before = arena.items.capacity();

        arena.reserve(10);
        let after = arena.items.capacity();

        assert!(after >= before + 10, "hint must satisfy the request");
        assert!(
            after < before * 2,
            "must not double a large arena: {before} -> {after}"
        );
        // Geometric, so repeated hints amortize instead of re-copying.
        assert!(
            after >= before + before / 8,
            "growth must stay geometric: {before} -> {after}"
        );

        // Already-sufficient capacity must not reallocate at all.
        let settled = arena.items.capacity();
        arena.reserve(1);
        assert_eq!(
            arena.items.capacity(),
            settled,
            "no-op when capacity suffices"
        );
    }

    #[test]
    fn id_from_index_valid() {
        let mut arena: Arena<String> = Arena::new();
        let id0 = arena.alloc("hello".into());
        let id1 = arena.alloc("world".into());

        let reconstructed = arena.id_from_index(0).unwrap();
        assert_eq!(reconstructed, id0);

        let reconstructed = arena.id_from_index(1).unwrap();
        assert_eq!(reconstructed, id1);
    }

    #[test]
    fn id_from_index_out_of_bounds() {
        let arena: Arena<String> = Arena::new();
        assert!(arena.id_from_index(0).is_none());

        let mut arena: Arena<String> = Arena::new();
        arena.alloc("one".into());
        assert!(arena.id_from_index(1).is_none());
        assert!(arena.id_from_index(100).is_none());
    }

    #[test]
    fn reserve_does_not_change_len() {
        let mut arena: Arena<String> = Arena::new();
        arena.alloc("first".into());
        assert_eq!(arena.len(), 1);

        arena.reserve(100);
        assert_eq!(arena.len(), 1);

        let id = arena.alloc("second".into());
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.get(id).unwrap(), "second");
    }
}
