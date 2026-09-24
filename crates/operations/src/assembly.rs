//! Assembly management: hierarchical product structure with positioned components.
//!
//! An [`Assembly`] is a tree of components. Every component is one placed
//! instance of a solid: its transform is relative to its parent, so a
//! component's world placement is the product of the transforms on the path
//! from its root. Several components may share one solid (instance sharing),
//! which is what the bill of materials counts.

use std::collections::HashMap;

use brepkit_math::aabb::Aabb3;
use brepkit_math::mat::Mat4;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// A unique identifier for a component in an assembly.
///
/// Ids are assigned sequentially from zero in insertion order.
pub type ComponentId = usize;

/// A positioned component in an assembly.
#[derive(Debug, Clone)]
pub struct Component {
    /// Human-readable name.
    pub name: String,
    /// The solid shape this component is an instance of.
    pub solid: SolidId,
    /// Transform placing this component in its parent's coordinate system
    /// (the assembly's, for a root component).
    pub transform: Mat4,
    /// Parent component (None for root-level components).
    pub parent: Option<ComponentId>,
    /// Child component IDs, in insertion order.
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
///
/// Every query walks the tree in a fixed order (roots in insertion order,
/// each component before its children, children in insertion order), so
/// results are deterministic.
#[derive(Debug, Default, Clone)]
pub struct Assembly {
    /// All components, indexed by their ID.
    components: Vec<Component>,
    /// Root-level component IDs (no parent).
    roots: Vec<ComponentId>,
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
        let id = self.components.len();
        self.components.push(Component {
            name: name.into(),
            solid,
            transform,
            parent: None,
            children: Vec::new(),
        });
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
        let id = self.components.len();
        let parent_comp =
            self.components
                .get_mut(parent)
                .ok_or_else(|| OperationsError::InvalidInput {
                    reason: format!("parent component {parent} not found"),
                })?;
        parent_comp.children.push(id);
        self.components.push(Component {
            name: name.into(),
            solid,
            transform,
            parent: Some(parent),
            children: Vec::new(),
        });
        Ok(id)
    }

    /// Returns a component by ID.
    #[must_use]
    pub fn component(&self, id: ComponentId) -> Option<&Component> {
        self.components.get(id)
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
    ///
    /// Returns `None` if the ID doesn't exist.
    #[must_use]
    pub fn world_transform(&self, id: ComponentId) -> Option<Mat4> {
        let comp = self.components.get(id)?;
        let mut result = comp.transform;

        let mut current_parent = comp.parent;
        while let Some(pid) = current_parent {
            let parent = self.components.get(pid)?;
            result = parent.transform * result;
            current_parent = parent.parent;
        }

        Some(result)
    }

    /// Flattens the assembly to a list of `(solid, world_transform)` pairs,
    /// one per component.
    ///
    /// Every component contributes its own solid, including components that
    /// have children: a parent's solid is geometry placed in the assembly
    /// like any other, and its children are positioned relative to it.
    #[must_use]
    pub fn flatten(&self) -> Vec<(SolidId, Mat4)> {
        self.placed_components()
            .into_iter()
            .map(|(id, world)| (self.components[id].solid, world))
            .collect()
    }

    /// Every component with its world transform, in tree order.
    fn placed_components(&self) -> Vec<(ComponentId, Mat4)> {
        let mut out = Vec::with_capacity(self.components.len());
        let mut stack: Vec<(ComponentId, Mat4)> = self
            .roots
            .iter()
            .rev()
            .map(|&id| (id, Mat4::identity()))
            .collect();
        while let Some((id, parent_world)) = stack.pop() {
            let Some(comp) = self.components.get(id) else {
                continue;
            };
            let world = parent_world * comp.transform;
            out.push((id, world));
            stack.extend(comp.children.iter().rev().map(|&child| (child, world)));
        }
        out
    }

    /// Computes the axis-aligned bounding box of the entire assembly in its
    /// own coordinate system.
    ///
    /// Each instance is bounded in its placed frame (see
    /// [`crate::measure::solid_bounding_box_transformed`]), so a rotated
    /// instance contributes its own tight box rather than the rotated
    /// corners of its local one.
    ///
    /// # Errors
    /// Returns an error if the assembly has no components, or if any
    /// solid's bounding box computation fails.
    pub fn bounding_box(&self, topo: &Topology) -> Result<Aabb3, OperationsError> {
        let mut total: Option<Aabb3> = None;
        for (solid_id, transform) in self.flatten() {
            let bbox = crate::measure::solid_bounding_box_transformed(topo, solid_id, &transform)?;
            total = Some(total.map_or(bbox, |t| t.union(bbox)));
        }
        total.ok_or_else(|| OperationsError::InvalidInput {
            reason: format!("assembly '{}' has no components", self.name),
        })
    }

    /// Generate a bill of materials: one entry per distinct solid with the
    /// number of components that instance it.
    ///
    /// Entries are ordered by each solid's first component in tree order,
    /// and each entry is named after that first component.
    #[must_use]
    pub fn bill_of_materials(&self) -> Vec<BomEntry> {
        let mut entries: Vec<BomEntry> = Vec::new();
        let mut entry_of_solid: HashMap<usize, usize> = HashMap::new();
        for (id, _) in self.placed_components() {
            let comp = &self.components[id];
            let slot = *entry_of_solid.entry(comp.solid.index()).or_insert_with(|| {
                entries.push(BomEntry {
                    name: comp.name.clone(),
                    solid_index: comp.solid.index(),
                    instance_count: 0,
                });
                entries.len() - 1
            });
            entries[slot].instance_count += 1;
        }
        entries
    }
}

/// An entry in the bill of materials.
#[derive(Debug, Clone)]
pub struct BomEntry {
    /// Name of the first component (in tree order) that instances this solid.
    pub name: String,
    /// Arena index of the solid shape.
    pub solid_index: usize,
    /// Number of instances of this shape in the assembly.
    pub instance_count: usize,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use brepkit_math::vec::Point3;

    use super::*;
    use crate::primitives::{make_box, make_cylinder};

    fn assert_point(got: Point3, want: Point3) {
        assert!((got - want).length() < 1e-10, "got {got:?} want {want:?}");
    }

    #[test]
    fn empty_assembly() {
        let asm = Assembly::new("test");
        assert_eq!(asm.component_count(), 0);
        assert!(asm.roots().is_empty());
        assert!(asm.flatten().is_empty());
        assert!(asm.bill_of_materials().is_empty());
    }

    #[test]
    fn empty_assembly_has_no_bounding_box() {
        let topo = Topology::new();
        assert!(Assembly::new("empty").bounding_box(&topo).is_err());
    }

    #[test]
    fn add_root_component() {
        let mut topo = Topology::new();
        let box1 = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mut asm = Assembly::new("test");
        let id = asm.add_root_component("box1", box1, Mat4::identity());

        assert_eq!(asm.component_count(), 1);
        assert_eq!(asm.roots(), &[id]);

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
        assert_eq!(asm.roots(), &[parent]);
        assert_eq!(asm.component(parent).unwrap().children, vec![child]);
        assert_eq!(asm.component(child).unwrap().parent, Some(parent));
    }

    #[test]
    fn invalid_parent_error() {
        let mut topo = Topology::new();
        let box1 = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mut asm = Assembly::new("test");
        assert!(
            asm.add_child_component(999, "child", box1, Mat4::identity())
                .is_err()
        );
        assert_eq!(asm.component_count(), 0, "a failed add leaves no trace");
        assert!(asm.world_transform(0).is_none());
    }

    #[test]
    fn world_transform_applies_the_parent_after_the_child() {
        let mut topo = Topology::new();
        let box1 = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mut asm = Assembly::new("test");
        let parent = asm.add_root_component(
            "parent",
            box1,
            Mat4::translation(1.0, 0.0, 0.0) * Mat4::rotation_z(std::f64::consts::FRAC_PI_2),
        );
        let child = asm
            .add_child_component(parent, "child", box1, Mat4::translation(2.0, 0.0, 0.0))
            .unwrap();
        let grandchild = asm
            .add_child_component(child, "grandchild", box1, Mat4::translation(0.0, 0.0, 3.0))
            .unwrap();

        // The child's (2,0,0) offset is expressed in the parent's rotated
        // frame, so it lands along +y before the parent's translation.
        let child_origin = asm
            .world_transform(child)
            .unwrap()
            .mul_point(Point3::new(0.0, 0.0, 0.0));
        assert_point(child_origin, Point3::new(1.0, 2.0, 0.0));

        let grandchild_origin = asm
            .world_transform(grandchild)
            .unwrap()
            .mul_point(Point3::new(0.0, 0.0, 0.0));
        assert_point(grandchild_origin, Point3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn flatten_places_every_component_in_tree_order() {
        let mut topo = Topology::new();
        let chassis = make_box(&mut topo, 4.0, 2.0, 1.0).unwrap();
        let wheel = make_cylinder(&mut topo, 0.5, 0.2).unwrap();
        let lamp = make_box(&mut topo, 0.1, 0.1, 0.1).unwrap();

        let mut asm = Assembly::new("cart");
        let body = asm.add_root_component("chassis", chassis, Mat4::translation(0.0, 0.0, 1.0));
        asm.add_child_component(body, "wheel_front", wheel, Mat4::translation(3.0, 0.0, 0.0))
            .unwrap();
        asm.add_child_component(body, "wheel_rear", wheel, Mat4::translation(1.0, 0.0, 0.0))
            .unwrap();
        asm.add_root_component("lamp", lamp, Mat4::translation(0.0, 0.0, 5.0));

        let flat = asm.flatten();
        let solids: Vec<SolidId> = flat.iter().map(|&(s, _)| s).collect();
        assert_eq!(
            solids,
            vec![chassis, wheel, wheel, lamp],
            "a parent's own solid is placed, ahead of its children"
        );
        let origins: Vec<Point3> = flat
            .iter()
            .map(|(_, m)| m.mul_point(Point3::new(0.0, 0.0, 0.0)))
            .collect();
        assert_point(origins[0], Point3::new(0.0, 0.0, 1.0));
        assert_point(origins[1], Point3::new(3.0, 0.0, 1.0));
        assert_point(origins[2], Point3::new(1.0, 0.0, 1.0));
        assert_point(origins[3], Point3::new(0.0, 0.0, 5.0));
    }

    #[test]
    fn flatten_matches_world_transform_for_every_component() {
        let mut topo = Topology::new();
        let part = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let mut asm = Assembly::new("chain");
        let mut last = asm.add_root_component("link0", part, Mat4::rotation_x(0.3));
        for i in 1..6 {
            last = asm
                .add_child_component(
                    last,
                    format!("link{i}"),
                    part,
                    Mat4::translation(1.0, 0.0, 0.0) * Mat4::rotation_z(0.2),
                )
                .unwrap();
        }
        let flat = asm.flatten();
        assert_eq!(flat.len(), 6);
        for (id, (_, world)) in flat.iter().enumerate() {
            let p = Point3::new(0.3, -0.7, 1.1);
            assert_point(
                world.mul_point(p),
                asm.world_transform(id).unwrap().mul_point(p),
            );
        }
    }

    #[test]
    fn bill_of_materials_counts_instances_in_tree_order() {
        let mut topo = Topology::new();
        let chassis = make_box(&mut topo, 4.0, 2.0, 1.0).unwrap();
        let wheel = make_cylinder(&mut topo, 0.5, 0.2).unwrap();

        let mut asm = Assembly::new("cart");
        let body = asm.add_root_component("chassis", chassis, Mat4::identity());
        for i in 0..4 {
            asm.add_child_component(body, format!("wheel_{i}"), wheel, Mat4::identity())
                .unwrap();
        }
        asm.add_root_component("spare_chassis", chassis, Mat4::identity());

        let bom = asm.bill_of_materials();
        let rows: Vec<(&str, usize, usize)> = bom
            .iter()
            .map(|e| (e.name.as_str(), e.solid_index, e.instance_count))
            .collect();
        assert_eq!(
            rows,
            vec![
                ("chassis", chassis.index(), 2),
                ("wheel_0", wheel.index(), 4),
            ]
        );
    }

    #[test]
    fn assembly_bounding_box_covers_every_instance() {
        let mut topo = Topology::new();
        let box1 = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let mut asm = Assembly::new("test");
        let a = asm.add_root_component("box_a", box1, Mat4::identity());
        asm.add_child_component(a, "box_b", box1, Mat4::translation(10.0, 0.0, 0.0))
            .unwrap();

        let bbox = asm.bounding_box(&topo).unwrap();
        assert_point(bbox.min, Point3::new(0.0, 0.0, 0.0));
        assert_point(bbox.max, Point3::new(12.0, 2.0, 2.0));
    }

    #[test]
    fn assembly_bounding_box_is_tight_for_rotated_instances() {
        let mut topo = Topology::new();
        let pin = make_cylinder(&mut topo, 1.0, 4.0).unwrap();

        // Spinning a cylinder about its own axis moves none of its surface;
        // the rotated corners of its local box would reach sqrt(2).
        let mut asm = Assembly::new("pins");
        asm.add_root_component(
            "pin",
            pin,
            Mat4::translation(5.0, 0.0, 0.0) * Mat4::rotation_z(std::f64::consts::FRAC_PI_4),
        );
        let bbox = asm.bounding_box(&topo).unwrap();
        assert_point(bbox.min, Point3::new(4.0, -1.0, 0.0));
        assert_point(bbox.max, Point3::new(6.0, 1.0, 4.0));
    }
}
