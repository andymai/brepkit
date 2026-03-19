//! Assembly management: hierarchical product structure with positioned components.
//!
//! Provides a tree-based assembly model where each component is a solid
//! placed at a specific location via a transform matrix. Components can
//! be instances of the same shape (instance sharing).
//!
//! Provides a product structure for managing multi-component assemblies.

use std::collections::HashMap;

use brepkit_math::aabb::Aabb3;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// A unique identifier for a component in an assembly.
pub type ComponentId = usize;

/// A positioned component in an assembly.
#[derive(Debug, Clone)]
pub struct Component {
    /// Human-readable name.
    pub name: String,
    /// The solid shape this component represents.
    pub solid: SolidId,
    /// Transform placing this component in the assembly's coordinate system.
    pub transform: Mat4,
    /// Parent component (None for root-level components).
    pub parent: Option<ComponentId>,
    /// Child component IDs.
    pub children: Vec<ComponentId>,
}

/// A hierarchical assembly of positioned components.
///
/// The assembly tree supports:
/// - Adding components with transforms
/// - Parent-child hierarchy
/// - Instance sharing (same solid, different transforms)
/// - Bounding box computation for the entire assembly
/// - Flattening to a list of positioned solids
#[derive(Debug, Default, Clone)]
pub struct Assembly {
    /// All components, indexed by their ID.
    components: HashMap<ComponentId, Component>,
    /// Root-level component IDs (no parent).
    roots: Vec<ComponentId>,
    /// Next available component ID.
    next_id: ComponentId,
    /// Assembly name.
    name: String,
}

impl Assembly {
    /// Creates a new empty assembly.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Returns the assembly name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Adds a root-level component (no parent).
    pub fn add_root_component(
        &mut self,
        name: impl Into<String>,
        solid: SolidId,
        transform: Mat4,
    ) -> ComponentId {
        let id = self.next_id;
        self.next_id += 1;

        self.components.insert(
            id,
            Component {
                name: name.into(),
                solid,
                transform,
                parent: None,
                children: Vec::new(),
            },
        );
        self.roots.push(id);
        id
    }

    /// Adds a child component under an existing parent.
    ///
    /// # Errors
    /// Returns an error if the parent ID doesn't exist.
    pub fn add_child_component(
        &mut self,
        parent: ComponentId,
        name: impl Into<String>,
        solid: SolidId,
        transform: Mat4,
    ) -> Result<ComponentId, OperationsError> {
        if !self.components.contains_key(&parent) {
            return Err(OperationsError::InvalidInput {
                reason: format!("parent component {parent} not found"),
            });
        }

        let id = self.next_id;
        self.next_id += 1;

        self.components.insert(
            id,
            Component {
                name: name.into(),
                solid,
                transform,
                parent: Some(parent),
                children: Vec::new(),
            },
        );

        // Update parent's children list
        if let Some(parent_comp) = self.components.get_mut(&parent) {
            parent_comp.children.push(id);
        }

        Ok(id)
    }

    /// Returns a component by ID.
    #[must_use]
    pub fn component(&self, id: ComponentId) -> Option<&Component> {
        self.components.get(&id)
    }

    /// Returns root-level component IDs.
    #[must_use]
    pub fn roots(&self) -> &[ComponentId] {
        &self.roots
    }

    /// Returns the total number of components.
    #[must_use]
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Computes the world transform for a component by multiplying
    /// all parent transforms in the hierarchy.
    #[must_use]
    pub fn world_transform(&self, id: ComponentId) -> Option<Mat4> {
        let comp = self.components.get(&id)?;
        let mut result = comp.transform;

        let mut current_parent = comp.parent;
        while let Some(pid) = current_parent {
            let parent = self.components.get(&pid)?;
            result = parent.transform * result;
            current_parent = parent.parent;
        }

        Some(result)
    }

    /// Flattens the assembly to a list of `(solid, world_transform)` pairs.
    ///
    /// This resolves the full hierarchy, computing the accumulated
    /// transform for each leaf component.
    #[must_use]
    pub fn flatten(&self) -> Vec<(SolidId, Mat4)> {
        let mut result = Vec::new();
        for &root_id in &self.roots {
            self.flatten_recursive(root_id, Mat4::identity(), &mut result);
        }
        result
    }

    fn flatten_recursive(
        &self,
        id: ComponentId,
        parent_transform: Mat4,
        result: &mut Vec<(SolidId, Mat4)>,
    ) {
        let Some(comp) = self.components.get(&id) else {
            return;
        };

        let world = parent_transform * comp.transform;

        if comp.children.is_empty() {
            // Leaf node — include in output
            result.push((comp.solid, world));
        } else {
            // Non-leaf — recurse into children
            for &child_id in &comp.children {
                self.flatten_recursive(child_id, world, result);
            }
        }
    }

    /// Computes the bounding box of the entire assembly.
    ///
    /// # Errors
    /// Returns an error if any solid's bounding box computation fails.
    pub fn bounding_box(&self, topo: &Topology) -> Result<Aabb3, OperationsError> {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut min_z = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        let mut max_z = f64::MIN;

        for (solid_id, transform) in self.flatten() {
            let bbox = crate::measure::solid_bounding_box(topo, solid_id)?;
            // Transform the 8 corners of the bounding box
            let lo = bbox.min;
            let hi = bbox.max;
            let corners = [
                Point3::new(lo.x(), lo.y(), lo.z()),
                Point3::new(hi.x(), lo.y(), lo.z()),
                Point3::new(lo.x(), hi.y(), lo.z()),
                Point3::new(hi.x(), hi.y(), lo.z()),
                Point3::new(lo.x(), lo.y(), hi.z()),
                Point3::new(hi.x(), lo.y(), hi.z()),
                Point3::new(lo.x(), hi.y(), hi.z()),
                Point3::new(hi.x(), hi.y(), hi.z()),
            ];

            for corner in &corners {
                let transformed = transform.mul_point(*corner);
                min_x = min_x.min(transformed.x());
                min_y = min_y.min(transformed.y());
                min_z = min_z.min(transformed.z());
                max_x = max_x.max(transformed.x());
                max_y = max_y.max(transformed.y());
                max_z = max_z.max(transformed.z());
            }
        }

        Ok(Aabb3 {
            min: Point3::new(min_x, min_y, min_z),
            max: Point3::new(max_x, max_y, max_z),
        })
    }

    /// Generate a bill of materials: list of unique solids and their instance count.
    #[must_use]
    pub fn bill_of_materials(&self) -> Vec<BomEntry> {
        let mut solid_counts: HashMap<usize, (String, usize)> = HashMap::new();

        for comp in self.components.values() {
            let entry = solid_counts
                .entry(comp.solid.index())
                .or_insert_with(|| (comp.name.clone(), 0));
            entry.1 += 1;
        }

        solid_counts
            .into_iter()
            .map(|(solid_idx, (name, count))| BomEntry {
                name,
                solid_index: solid_idx,
                instance_count: count,
            })
            .collect()
    }
}

/// An entry in the bill of materials.
#[derive(Debug, Clone)]
pub struct BomEntry {
    /// Component name.
    pub name: String,
    /// Arena index of the solid shape.
    pub solid_index: usize,
    /// Number of instances of this shape in the assembly.
    pub instance_count: usize,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::primitives::make_box;

    #[test]
    fn empty_assembly() {
        let asm = Assembly::new("test");
        assert_eq!(asm.component_count(), 0);
        assert!(asm.roots().is_empty());
        assert!(asm.flatten().is_empty());
    }

    #[test]
    fn add_root_component() {
        let mut topo = Topology::new();
        let box1 = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mut asm = Assembly::new("test");
        let id = asm.add_root_component("box1", box1, Mat4::identity());

        assert_eq!(asm.component_count(), 1);
        assert_eq!(asm.roots().len(), 1);

        let comp = asm.component(id).unwrap();
        assert_eq!(comp.name, "box1");
        assert!(comp.parent.is_none());
    }

    #[test]
    fn add_child_component() {
        let mut topo = Topology::new();
        let box1 = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let box2 = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();

        let mut asm = Assembly::new("test");
        let parent = asm.add_root_component("parent", box1, Mat4::identity());
        let child = asm
            .add_child_component(parent, "child", box2, Mat4::translation(2.0, 0.0, 0.0))
            .unwrap();

        assert_eq!(asm.component_count(), 2);
        assert_eq!(asm.roots().len(), 1);

        let parent_comp = asm.component(parent).unwrap();
        assert_eq!(parent_comp.children.len(), 1);

        let child_comp = asm.component(child).unwrap();
        assert_eq!(child_comp.parent, Some(parent));
    }

    #[test]
    fn flatten_assembly() {
        let mut topo = Topology::new();
        let box1 = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mut asm = Assembly::new("test");
        asm.add_root_component("box_a", box1, Mat4::identity());
        asm.add_root_component("box_b", box1, Mat4::translation(3.0, 0.0, 0.0));

        let flat = asm.flatten();
        assert_eq!(flat.len(), 2, "two root components = two instances");
    }

    #[test]
    fn world_transform_chain() {
        let mut topo = Topology::new();
        let box1 = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mut asm = Assembly::new("test");
        let parent = asm.add_root_component("parent", box1, Mat4::translation(1.0, 0.0, 0.0));
        let child = asm
            .add_child_component(parent, "child", box1, Mat4::translation(0.0, 2.0, 0.0))
            .unwrap();

        let world = asm.world_transform(child).unwrap();
        let origin = world.mul_point(Point3::new(0.0, 0.0, 0.0));

        // Parent translates (1,0,0), child translates (0,2,0)
        // World should be (1,2,0)
        assert!((origin.x() - 1.0).abs() < 1e-10);
        assert!((origin.y() - 2.0).abs() < 1e-10);
        assert!((origin.z() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn bill_of_materials() {
        let mut topo = Topology::new();
        let box1 = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mut asm = Assembly::new("test");
        asm.add_root_component("wheel_1", box1, Mat4::identity());
        asm.add_root_component("wheel_2", box1, Mat4::translation(1.0, 0.0, 0.0));
        asm.add_root_component("wheel_3", box1, Mat4::translation(2.0, 0.0, 0.0));

        let bom = asm.bill_of_materials();
        // All 3 use the same solid — should have 1 BOM entry with count 3
        assert_eq!(bom.len(), 1);
        assert_eq!(bom[0].instance_count, 3);
    }

    #[test]
    fn invalid_parent_error() {
        let mut topo = Topology::new();
        let box1 = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mut asm = Assembly::new("test");
        let result = asm.add_child_component(999, "child", box1, Mat4::identity());
        assert!(result.is_err());
    }

    #[test]
    fn assembly_bounding_box() {
        let mut topo = Topology::new();
        let box1 = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let mut asm = Assembly::new("test");
        asm.add_root_component("box_a", box1, Mat4::identity());
        asm.add_root_component("box_b", box1, Mat4::translation(10.0, 0.0, 0.0));

        let bbox = asm.bounding_box(&topo).unwrap();

        // box_a at origin: [0, 2]³, box_b translated by 10: [10, 12] in x
        assert!(bbox.min.x() < 0.5);
        assert!(bbox.max.x() > 11.5);
    }
}
