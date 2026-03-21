//! Generational arena and geometric entity types for the GCS.

use std::marker::PhantomData;

// ── Generational Arena ──────────────────────────────────────────────

/// A typed handle into a [`GenArena`].
///
/// Stores an index and a generation counter. If the generation doesn't
/// match the slot's current generation, the handle is stale (the entity
/// was removed and the slot may have been reused).
pub struct Handle<T> {
    pub(crate) index: u32,
    pub(crate) generation: u32,
    pub(crate) _marker: PhantomData<fn() -> T>,
}

// Manual impls to avoid requiring T: Debug/Clone/etc.
impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handle")
            .field("index", &self.index)
            .field("gen", &self.generation)
            .finish()
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}
impl<T> Eq for Handle<T> {}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> Handle<T> {
    /// Raw index into the arena's slot vector.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Generation counter for stale-handle detection.
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }
}

/// Slot in the generational arena — either occupied or free.
enum Entry<T> {
    Occupied {
        value: T,
        generation: u32,
    },
    Free {
        next_free: Option<u32>,
        generation: u32,
    },
}

/// A generational arena that supports O(1) insert, get, and remove.
///
/// Removed slots are recycled via a free list. Each slot has a generation
/// counter that is bumped on removal, so stale handles are detected.
pub struct GenArena<T> {
    entries: Vec<Entry<T>>,
    free_head: Option<u32>,
    len: usize,
}

impl<T: Clone> Clone for GenArena<T> {
    fn clone(&self) -> Self {
        let entries = self
            .entries
            .iter()
            .map(|e| match e {
                Entry::Occupied { value, generation } => Entry::Occupied {
                    value: value.clone(),
                    generation: *generation,
                },
                Entry::Free {
                    next_free,
                    generation,
                } => Entry::Free {
                    next_free: *next_free,
                    generation: *generation,
                },
            })
            .collect();
        Self {
            entries,
            free_head: self.free_head,
            len: self.len,
        }
    }
}

impl<T> Default for GenArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GenArena<T> {
    /// Creates an empty arena.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
            free_head: None,
            len: 0,
        }
    }

    /// Number of live entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the arena is empty.
    #[must_use]
    #[allow(dead_code)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert a value and return its handle.
    pub fn insert(&mut self, value: T) -> Handle<T> {
        self.len += 1;
        if let Some(free_idx) = self.free_head {
            let idx = free_idx as usize;
            let generation = match &self.entries[idx] {
                Entry::Free {
                    next_free,
                    generation,
                } => {
                    self.free_head = *next_free;
                    *generation
                }
                Entry::Occupied { .. } => {
                    // Should never happen — free_head pointed to an occupied slot.
                    // Defensive: just append instead.
                    self.free_head = None;
                    return self.push_new(value);
                }
            };
            self.entries[idx] = Entry::Occupied { value, generation };
            Handle {
                index: free_idx,
                generation,
                _marker: PhantomData,
            }
        } else {
            self.push_new(value)
        }
    }

    /// Append a new entry at the end (no free slot available).
    fn push_new(&mut self, value: T) -> Handle<T> {
        let index = self.entries.len() as u32;
        self.entries.push(Entry::Occupied {
            value,
            generation: 0,
        });
        Handle {
            index,
            generation: 0,
            _marker: PhantomData,
        }
    }

    /// Get a reference to the value at `handle`, or `None` if stale/invalid.
    #[must_use]
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        let entry = self.entries.get(handle.index as usize)?;
        match entry {
            Entry::Occupied { value, generation } if *generation == handle.generation => {
                Some(value)
            }
            _ => None,
        }
    }

    /// Get a mutable reference to the value at `handle`.
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        let entry = self.entries.get_mut(handle.index as usize)?;
        match entry {
            Entry::Occupied { value, generation } if *generation == handle.generation => {
                Some(value)
            }
            _ => None,
        }
    }

    /// Remove the value at `handle`. Returns the removed value, or `None` if stale.
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let idx = handle.index as usize;
        let entry = self.entries.get(idx)?;
        let cur_gen = match entry {
            Entry::Occupied { generation, .. } if *generation == handle.generation => *generation,
            _ => return None,
        };
        // Replace with a free entry, bumping the generation.
        let old = std::mem::replace(
            &mut self.entries[idx],
            Entry::Free {
                next_free: self.free_head,
                generation: cur_gen + 1,
            },
        );
        self.free_head = Some(handle.index);
        self.len -= 1;
        match old {
            Entry::Occupied { value, .. } => Some(value),
            Entry::Free { .. } => None, // unreachable
        }
    }

    /// Iterate over all live `(Handle<T>, &T)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, entry)| match entry {
                Entry::Occupied { value, generation } => Some((
                    Handle {
                        index: i as u32,
                        generation: *generation,
                        _marker: PhantomData,
                    },
                    value,
                )),
                Entry::Free { .. } => None,
            })
    }

    /// Iterate over all live `(Handle<T>, &mut T)` pairs.
    #[allow(dead_code)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Handle<T>, &mut T)> {
        self.entries
            .iter_mut()
            .enumerate()
            .filter_map(|(i, entry)| match entry {
                Entry::Occupied { value, generation } => Some((
                    Handle {
                        index: i as u32,
                        generation: *generation,
                        _marker: PhantomData,
                    },
                    value,
                )),
                Entry::Free { .. } => None,
            })
    }

    /// Check if a handle is still valid (points to a live entry).
    #[must_use]
    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }
}

// ── Entity Types ────────────────────────────────────────────────────

/// A handle to a point in the GCS.
pub type PointId = Handle<PointData>;
/// A handle to a line in the GCS.
pub type LineId = Handle<LineData>;
/// A handle to a circle in the GCS.
pub type CircleId = Handle<CircleData>;

/// A 2D point in the constraint system.
#[derive(Debug, Clone, Copy)]
pub struct PointData {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Whether this point is fixed (not adjusted by the solver).
    pub fixed: bool,
}

/// A line defined by two points.
#[derive(Debug, Clone, Copy)]
pub struct LineData {
    /// First endpoint.
    pub p1: PointId,
    /// Second endpoint.
    pub p2: PointId,
}

/// A circle defined by a center point and radius.
#[derive(Debug, Clone, Copy)]
pub struct CircleData {
    /// Center point.
    pub center: PointId,
    /// Radius (a solver parameter if not fixed).
    pub radius: f64,
}

/// A reference to a solver parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamRef {
    /// X coordinate of a point.
    PointX(PointId),
    /// Y coordinate of a point.
    PointY(PointId),
    /// Radius of a circle.
    CircleRadius(CircleId),
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove() {
        let mut arena = GenArena::<i32>::new();
        let h1 = arena.insert(10);
        let h2 = arena.insert(20);

        assert_eq!(*arena.get(h1).unwrap(), 10);
        assert_eq!(*arena.get(h2).unwrap(), 20);
        assert_eq!(arena.len(), 2);

        let removed = arena.remove(h1).unwrap();
        assert_eq!(removed, 10);
        assert!(arena.get(h1).is_none()); // stale
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn stale_handle_after_remove() {
        let mut arena = GenArena::<i32>::new();
        let h = arena.insert(42);
        arena.remove(h);

        // Reuse the slot
        let h2 = arena.insert(99);
        assert_eq!(h2.index(), h.index()); // same slot
        assert_ne!(h2.generation(), h.generation()); // different gen

        // Old handle is stale
        assert!(arena.get(h).is_none());
        assert_eq!(*arena.get(h2).unwrap(), 99);
    }

    #[test]
    fn free_list_reuse() {
        let mut arena = GenArena::<i32>::new();
        let h0 = arena.insert(0);
        let h1 = arena.insert(1);
        let h2 = arena.insert(2);

        arena.remove(h1);
        arena.remove(h0);

        // Next inserts should reuse freed slots (LIFO)
        let h3 = arena.insert(30);
        assert_eq!(h3.index(), h0.index());
        let h4 = arena.insert(40);
        assert_eq!(h4.index(), h1.index());

        assert_eq!(*arena.get(h3).unwrap(), 30);
        assert_eq!(*arena.get(h4).unwrap(), 40);
        assert_eq!(*arena.get(h2).unwrap(), 2);
    }

    #[test]
    fn iteration() {
        let mut arena = GenArena::<i32>::new();
        let _h0 = arena.insert(10);
        let h1 = arena.insert(20);
        let _h2 = arena.insert(30);

        arena.remove(h1);

        let values: Vec<i32> = arena.iter().map(|(_, v)| *v).collect();
        assert_eq!(values.len(), 2);
        assert!(values.contains(&10));
        assert!(values.contains(&30));
    }

    #[test]
    fn empty_arena() {
        let arena = GenArena::<i32>::new();
        assert!(arena.is_empty());
        assert_eq!(arena.len(), 0);
        assert_eq!(arena.iter().count(), 0);
    }

    #[test]
    fn double_remove() {
        let mut arena = GenArena::<i32>::new();
        let h = arena.insert(42);
        assert!(arena.remove(h).is_some());
        assert!(arena.remove(h).is_none()); // already removed
    }

    #[test]
    fn get_mut_works() {
        let mut arena = GenArena::<i32>::new();
        let h = arena.insert(10);
        *arena.get_mut(h).unwrap() = 20;
        assert_eq!(*arena.get(h).unwrap(), 20);
    }
}
