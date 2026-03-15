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

    /// Reserves capacity for at least `additional` more entries.
    pub fn reserve(&mut self, additional: usize) {
        self.items.reserve(additional);
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
