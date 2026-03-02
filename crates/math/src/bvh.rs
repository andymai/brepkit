//! Flat-array AABB tree for broad-phase spatial queries.
//!
//! Uses SAH (Surface Area Heuristic) for tree construction, consistent
//! with the arena-based patterns used in the topology crate.

use crate::aabb::Aabb3;
use crate::vec::Point3;

/// A node in the flat-array BVH.
#[derive(Debug, Clone, Copy)]
struct BvhNode {
    /// Bounding box of this node.
    aabb: Aabb3,
    /// For leaf nodes: the index of the primitive. For internal nodes: `usize::MAX`.
    primitive: usize,
    /// Index of the left child. `usize::MAX` for leaf nodes.
    left: usize,
    /// Index of the right child. `usize::MAX` for leaf nodes.
    right: usize,
}

/// A flat-array AABB tree for spatial queries.
#[derive(Debug, Clone)]
pub struct Bvh {
    nodes: Vec<BvhNode>,
}

impl Bvh {
    /// Build a BVH from a set of `(primitive_id, aabb)` pairs.
    ///
    /// Uses SAH-based construction for good query performance. If `aabbs`
    /// is empty, returns an empty BVH.
    #[must_use]
    pub fn build(aabbs: &[(usize, Aabb3)]) -> Self {
        if aabbs.is_empty() {
            return Self { nodes: Vec::new() };
        }

        let mut indices: Vec<usize> = (0..aabbs.len()).collect();
        let mut nodes = Vec::with_capacity(2 * aabbs.len());
        build_recursive(aabbs, &mut indices, &mut nodes);
        Self { nodes }
    }

    /// Query all primitives whose AABB overlaps `test`.
    #[must_use]
    pub fn query_overlap(&self, test: &Aabb3) -> Vec<usize> {
        let mut results = Vec::new();
        if self.nodes.is_empty() {
            return results;
        }
        let mut stack = vec![0usize];
        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];
            if !node.aabb.intersects(*test) {
                continue;
            }
            if node.primitive == usize::MAX {
                stack.push(node.left);
                stack.push(node.right);
            } else {
                results.push(node.primitive);
            }
        }
        results
    }

    /// Find the primitive whose AABB is closest to `point`.
    ///
    /// Returns `None` if the BVH is empty.
    #[must_use]
    pub fn query_closest(&self, point: Point3) -> Option<usize> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut best_id = None;
        let mut best_dist = f64::INFINITY;
        let mut stack = vec![0usize];

        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];
            let node_dist = node.aabb.distance_squared_to_point(point);
            if node_dist >= best_dist {
                continue;
            }
            if node.primitive == usize::MAX {
                // Visit closer child first for better pruning.
                let dl = self.nodes[node.left].aabb.distance_squared_to_point(point);
                let dr = self.nodes[node.right].aabb.distance_squared_to_point(point);
                if dl < dr {
                    stack.push(node.right);
                    stack.push(node.left);
                } else {
                    stack.push(node.left);
                    stack.push(node.right);
                }
            } else {
                best_id = Some(node.primitive);
                best_dist = node_dist;
            }
        }

        best_id
    }
}

/// Recursively build the BVH, appending nodes to `nodes`.
#[allow(clippy::expect_used, clippy::cast_precision_loss)]
fn build_recursive(
    aabbs: &[(usize, Aabb3)],
    indices: &mut [usize],
    nodes: &mut Vec<BvhNode>,
) -> usize {
    let node_idx = nodes.len();

    if indices.len() == 1 {
        let i = indices[0];
        nodes.push(BvhNode {
            aabb: aabbs[i].1,
            primitive: aabbs[i].0,
            left: usize::MAX,
            right: usize::MAX,
        });
        return node_idx;
    }

    // Compute bounding box of all primitives in this subset.
    let combined = indices
        .iter()
        .map(|&i| aabbs[i].1)
        .reduce(super::aabb::Aabb3::union)
        .expect("non-empty");

    if indices.len() == 2 {
        // Direct leaf pair.
        nodes.push(BvhNode {
            aabb: combined,
            primitive: usize::MAX,
            left: 0,
            right: 0,
        });
        let left = build_recursive(aabbs, &mut indices[..1], nodes);
        let right = build_recursive(aabbs, &mut indices[1..], nodes);
        nodes[node_idx].left = left;
        nodes[node_idx].right = right;
        return node_idx;
    }

    // SAH split: try each axis and find the best split.
    let parent_area = combined.surface_area();
    let mut best_cost = f64::INFINITY;
    let mut best_axis = 0;
    let mut best_split = indices.len() / 2;

    for axis in 0..3 {
        indices.sort_by(|&a, &b| {
            let ca = centroid_axis(aabbs[a].1, axis);
            let cb = centroid_axis(aabbs[b].1, axis);
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let n = indices.len();

        // Precompute suffix unions so each split candidate is O(1).
        let mut suffix = vec![aabbs[indices[n - 1]].1; n];
        for k in (0..n - 1).rev() {
            suffix[k] = suffix[k + 1].union(aabbs[indices[k]].1);
        }

        let mut left_aabb = aabbs[indices[0]].1;
        for split in 1..n {
            left_aabb = left_aabb.union(aabbs[indices[split - 1]].1);
            let right_aabb = suffix[split];

            let cost = (split as f64).mul_add(
                left_aabb.surface_area(),
                (n - split) as f64 * right_aabb.surface_area(),
            ) / parent_area;

            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_split = split;
            }
        }
    }

    // Re-sort by the best axis.
    indices.sort_by(|&a, &b| {
        let ca = centroid_axis(aabbs[a].1, best_axis);
        let cb = centroid_axis(aabbs[b].1, best_axis);
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Allocate internal node.
    nodes.push(BvhNode {
        aabb: combined,
        primitive: usize::MAX,
        left: 0,
        right: 0,
    });

    let (left_indices, right_indices) = indices.split_at_mut(best_split);
    let left = build_recursive(aabbs, left_indices, nodes);
    let right = build_recursive(aabbs, right_indices, nodes);
    nodes[node_idx].left = left;
    nodes[node_idx].right = right;

    node_idx
}

/// Get the centroid coordinate along a specific axis.
fn centroid_axis(aabb: Aabb3, axis: usize) -> f64 {
    match axis {
        0 => aabb.center().x(),
        1 => aabb.center().y(),
        _ => aabb.center().z(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_box(id: usize, x: f64, y: f64, z: f64, size: f64) -> (usize, Aabb3) {
        (
            id,
            Aabb3::from_points([
                Point3::new(x, y, z),
                Point3::new(x + size, y + size, z + size),
            ]),
        )
    }

    #[test]
    fn bvh_empty() {
        let bvh = Bvh::build(&[]);
        assert!(
            bvh.query_overlap(&Aabb3::from_points([Point3::new(0.0, 0.0, 0.0)]))
                .is_empty()
        );
        assert!(bvh.query_closest(Point3::new(0.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn bvh_single() {
        let aabbs = vec![make_box(42, 0.0, 0.0, 0.0, 1.0)];
        let bvh = Bvh::build(&aabbs);

        let hits = bvh.query_overlap(&Aabb3::from_points([
            Point3::new(0.5, 0.5, 0.5),
            Point3::new(0.6, 0.6, 0.6),
        ]));
        assert_eq!(hits, vec![42]);

        let miss = bvh.query_overlap(&Aabb3::from_points([
            Point3::new(5.0, 5.0, 5.0),
            Point3::new(6.0, 6.0, 6.0),
        ]));
        assert!(miss.is_empty());
    }

    #[test]
    fn bvh_multiple_overlap() {
        let aabbs = vec![
            make_box(0, 0.0, 0.0, 0.0, 1.0),
            make_box(1, 2.0, 0.0, 0.0, 1.0),
            make_box(2, 4.0, 0.0, 0.0, 1.0),
            make_box(3, 0.0, 2.0, 0.0, 1.0),
        ];
        let bvh = Bvh::build(&aabbs);

        // Query that overlaps box 0 and 1
        let test = Aabb3::from_points([Point3::new(0.5, 0.5, 0.5), Point3::new(2.5, 0.5, 0.5)]);
        let mut hits = bvh.query_overlap(&test);
        hits.sort_unstable();
        assert_eq!(hits, vec![0, 1]);
    }

    #[test]
    fn bvh_closest() {
        let aabbs = vec![
            make_box(0, 0.0, 0.0, 0.0, 1.0),
            make_box(1, 10.0, 10.0, 10.0, 1.0),
            make_box(2, 5.0, 5.0, 5.0, 1.0),
        ];
        let bvh = Bvh::build(&aabbs);

        assert_eq!(bvh.query_closest(Point3::new(0.5, 0.5, 0.5)), Some(0));
        assert_eq!(bvh.query_closest(Point3::new(10.5, 10.5, 10.5)), Some(1));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn bvh_many_primitives() {
        let aabbs: Vec<(usize, Aabb3)> = (0..100)
            .map(|i| make_box(i, i as f64 * 2.0, 0.0, 0.0, 1.0))
            .collect();
        let bvh = Bvh::build(&aabbs);

        // Query around primitive 50
        let test = Aabb3::from_points([Point3::new(99.5, 0.0, 0.0), Point3::new(100.5, 1.0, 1.0)]);
        let hits = bvh.query_overlap(&test);
        assert!(
            hits.contains(&50),
            "expected primitive 50 in hits: {hits:?}"
        );
    }
}
