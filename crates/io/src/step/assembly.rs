//! Where a STEP file puts its shapes: the length unit each representation
//! is written in, and the assembly placements that set one representation's
//! shapes inside another's.
//!
//! brepkit works in millimetres. A shape is read in its representation's own
//! unit and coordinates, then scaled to millimetres and carried up through
//! every chain of placements to a representation nothing places, once per
//! chain: a part an assembly uses twice comes out twice.

use std::collections::{HashMap, HashSet};

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};

use crate::IoError;

use super::reader::{
    StepEntity, composite_components, parse_floats, parse_refs, refs_after_name, take_parenthesized,
};

/// Records that are representations in name but not in shape:
/// `(definition, representation)` pairs, not `(name, items, context)`.
const NOT_REPRESENTATIONS: [&str; 3] = [
    "SHAPE_DEFINITION_REPRESENTATION",
    "PROPERTY_DEFINITION_REPRESENTATION",
    "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
];

/// How deep a chain of units or placements may run before it is taken for
/// a reference cycle.
const MAX_DEPTH: usize = 64;

/// How many placed shapes one file may produce. Each level of an assembly
/// that uses a subassembly twice doubles its count, so a short file can
/// name more placements than any machine holds; past this the file is
/// refused rather than built.
pub(super) const MAX_INSTANCES: usize = 100_000;

/// How many faces the placed copies of a file's shapes may hold between
/// them: a large part placed many times is the same blow-up.
pub(super) const MAX_PLACED_FACES: usize = 10_000_000;

/// Spend `n` of `allowance`, or fail when it runs short.
pub(super) fn spend(allowance: &mut usize, n: usize) -> Result<(), IoError> {
    *allowance = allowance
        .checked_sub(n)
        .ok_or_else(|| IoError::ParseError {
            reason: "STEP assembly places more shapes than brepkit reads from one file".into(),
        })?;
    Ok(())
}

/// The placements of every shape item in a file.
pub(super) struct Placements {
    /// The representation each shape item belongs to.
    owner: HashMap<u64, u64>,
    /// Millimetres per unit of each representation.
    scale: HashMap<u64, f64>,
    /// The frame each representation shares with others, by union-find.
    frame: HashMap<u64, u64>,
    /// Each frame's placements in parent frames, as child-to-parent
    /// transforms in millimetres.
    parents: HashMap<u64, Vec<(u64, Mat4)>>,
    /// Millimetres per unit for a shape no representation holds.
    default_scale: f64,
}

impl Placements {
    pub(super) fn new(entities: &HashMap<u64, StepEntity>) -> Self {
        let file = File { entities };
        let mut ids: Vec<u64> = entities.keys().copied().collect();
        ids.sort_unstable();

        let default_scale = ids
            .iter()
            .find_map(|&id| file.context_scale(id))
            .unwrap_or(1.0);
        let mut owner = HashMap::new();
        let mut scale = HashMap::new();
        let mut holders: HashMap<u64, Vec<u64>> = HashMap::new();
        for &id in &ids {
            let Some((items, context)) = file.representation(id) else {
                continue;
            };
            for &item in &items {
                owner.entry(item).or_insert(id);
                holders.entry(item).or_default().push(id);
            }
            let unit = context.and_then(|c| file.context_scale(c));
            scale.insert(id, unit.unwrap_or(default_scale));
        }

        let mut frames = UnionFind::default();
        let mut links = Vec::new();
        for &id in &ids {
            match file.relationship(id) {
                Some(Relationship::Same(a, b)) => frames.union(a, b),
                Some(Relationship::Placed {
                    rep_1,
                    rep_2,
                    item_1,
                    item_2,
                }) => links.push((id, rep_1, rep_2, item_1, item_2)),
                None => {}
            }
        }
        let children = file.assembly_children(&mut frames);

        let mut placements = Self {
            owner,
            scale,
            frame: HashMap::new(),
            parents: HashMap::new(),
            default_scale,
        };
        for &(relation, rep_1, rep_2, item_1, item_2) in &links {
            // `rep_1` is the component by convention; the assembly usage the
            // relationship serves says so when the two disagree.
            let child_is_2 = children.get(&relation).is_some_and(|child| {
                frames.find(rep_2) == frames.find(*child)
                    && frames.find(rep_1) != frames.find(*child)
            });
            let ((child, child_item), (parent, parent_item)) = if child_is_2 {
                ((rep_2, item_2), (rep_1, item_1))
            } else {
                ((rep_1, item_1), (rep_2, item_2))
            };
            placements.link(&file, &mut frames, child, child_item, parent, parent_item);
        }
        for &id in &ids {
            let Some((map, target)) = file.mapped_item(id) else {
                continue;
            };
            let Some((origin, mapped)) = file.representation_map(map) else {
                continue;
            };
            for &parent in holders.get(&id).map_or(&[][..], Vec::as_slice) {
                placements.link(&file, &mut frames, mapped, origin, parent, target);
            }
        }
        for &id in &ids {
            if placements.scale.contains_key(&id) {
                placements.frame.insert(id, frames.find(id));
            }
        }
        placements
    }

    /// Add the placement of `child`'s frame in `parent`'s: `child_item` in
    /// the child's coordinates lands on `parent_item` in the parent's.
    fn link(
        &mut self,
        file: &File<'_>,
        frames: &mut UnionFind,
        child: u64,
        child_item: u64,
        parent: u64,
        parent_item: u64,
    ) {
        let child_scale = self
            .scale
            .get(&child)
            .copied()
            .unwrap_or(self.default_scale);
        let parent_scale = self
            .scale
            .get(&parent)
            .copied()
            .unwrap_or(self.default_scale);
        let (Some(from), Some(to)) = (
            file.placement(child_item, child_scale),
            file.placement(parent_item, parent_scale),
        ) else {
            log::warn!("STEP placement #{child_item} or #{parent_item} unreadable; skipped");
            return;
        };
        let Ok(from_inverse) = from.inverse() else {
            return;
        };
        self.parents
            .entry(frames.find(child))
            .or_default()
            .push((frames.find(parent), to * from_inverse));
    }

    /// Every placement of shape item `item`, scaled to millimetres: one per
    /// chain of placements from its representation up to one nothing places.
    /// Each placement spends one of `allowance`.
    ///
    /// # Errors
    ///
    /// Returns an error if the placements outrun `allowance`.
    pub(super) fn instances(&self, item: u64, allowance: &mut usize) -> Result<Vec<Mat4>, IoError> {
        let Some(&rep) = self.owner.get(&item) else {
            spend(allowance, 1)?;
            return Ok(vec![uniform(self.default_scale)]);
        };
        let unit = uniform(self.scale.get(&rep).copied().unwrap_or(self.default_scale));
        let frame = self.frame.get(&rep).copied().unwrap_or(rep);
        let mut out = Vec::new();
        self.chains(frame, unit, &mut HashSet::new(), 0, allowance, &mut out)?;
        Ok(out)
    }

    /// Push `below` carried up every chain of placements from `frame` onto
    /// `out`. Each chain ends in one push, which spends one of `allowance`,
    /// and runs at most `MAX_DEPTH` links, so the walk does at most that much
    /// work per unit spent whatever the placements' shape.
    fn chains(
        &self,
        frame: u64,
        below: Mat4,
        path: &mut HashSet<u64>,
        depth: usize,
        allowance: &mut usize,
        out: &mut Vec<Mat4>,
    ) -> Result<(), IoError> {
        let parents = self.parents.get(&frame).map_or(&[][..], Vec::as_slice);
        if parents.is_empty() || depth > MAX_DEPTH || path.contains(&frame) {
            spend(allowance, 1)?;
            out.push(below);
            return Ok(());
        }
        path.insert(frame);
        for &(parent, to_parent) in parents {
            self.chains(parent, to_parent * below, path, depth + 1, allowance, out)?;
        }
        path.remove(&frame);
        Ok(())
    }
}

fn uniform(s: f64) -> Mat4 {
    Mat4::scale(s, s, s)
}

/// Whether `m` moves nothing.
pub(super) fn is_identity(m: &Mat4) -> bool {
    let id = Mat4::identity();
    (0..4).all(|i| (0..4).all(|j| (m.0[i][j] - id.0[i][j]).abs() < 1e-12))
}

enum Relationship {
    /// Two representations of one shape in one coordinate frame.
    Same(u64, u64),
    /// A representation placed in another: `item_1` of `rep_1` lands on
    /// `item_2` of `rep_2`.
    Placed {
        rep_1: u64,
        rep_2: u64,
        item_1: u64,
        item_2: u64,
    },
}

#[derive(Default)]
struct UnionFind(HashMap<u64, u64>);

impl UnionFind {
    fn find(&mut self, x: u64) -> u64 {
        let mut root = x;
        while let Some(&up) = self.0.get(&root) {
            if up == root {
                break;
            }
            root = up;
        }
        let mut at = x;
        while at != root {
            let up = self.0.insert(at, root).unwrap_or(root);
            at = up;
        }
        root
    }

    fn union(&mut self, a: u64, b: u64) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.0.insert(ra.max(rb), ra.min(rb));
        }
    }
}

struct File<'a> {
    entities: &'a HashMap<u64, StepEntity>,
}

impl File<'_> {
    /// Whether a record could hold `word`: a cheap screen before taking a
    /// complex record apart.
    fn mentions(&self, id: u64, word: &str) -> bool {
        self.entities
            .get(&id)
            .is_some_and(|e| e.entity_type.contains(word) || e.attrs.contains(word))
    }

    /// The components of a record: itself, or each part of a complex one.
    fn components(&self, id: u64) -> Vec<(&str, &str)> {
        let Some(e) = self.entities.get(&id) else {
            return Vec::new();
        };
        if e.entity_type.is_empty() {
            composite_components(&e.attrs).unwrap_or_default()
        } else {
            vec![(e.entity_type.as_str(), e.attrs.as_str())]
        }
    }

    fn component(&self, id: u64, name: &str) -> Option<&str> {
        self.components(id)
            .into_iter()
            .find(|(kind, _)| *kind == name)
            .map(|(_, body)| body)
    }

    /// A representation's items and context.
    fn representation(&self, id: u64) -> Option<(Vec<u64>, Option<u64>)> {
        let e = self.entities.get(&id)?;
        let body = if e.entity_type.is_empty() {
            // A B-spline's `REPRESENTATION_ITEM(` is no representation.
            if !e.attrs.contains("REPRESENTATION(") {
                return None;
            }
            self.component(id, "REPRESENTATION")?
        } else if e.entity_type.ends_with("REPRESENTATION")
            && !NOT_REPRESENTATIONS.contains(&e.entity_type.as_str())
        {
            e.attrs.as_str()
        } else {
            return None;
        };
        let (items, tail) = take_parenthesized(after_name(body).trim_start())?;
        Some((parse_refs(items), parse_refs(tail).first().copied()))
    }

    /// Millimetres per length unit of a representation context.
    fn context_scale(&self, context: u64) -> Option<f64> {
        if !self.mentions(context, "GLOBAL_UNIT_ASSIGNED_CONTEXT") {
            return None;
        }
        let units = self.component(context, "GLOBAL_UNIT_ASSIGNED_CONTEXT")?;
        parse_refs(units)
            .into_iter()
            .find_map(|unit| self.length_scale(unit, 0))
    }

    /// Millimetres per unit of a length unit.
    fn length_scale(&self, unit: u64, depth: usize) -> Option<f64> {
        if depth > MAX_DEPTH {
            return None;
        }
        let components = self.components(unit);
        let named = |name: &str| {
            components
                .iter()
                .find(|(kind, _)| *kind == name)
                .map(|(_, body)| *body)
        };
        if let Some(si) = named("SI_UNIT") {
            if !si.contains(".METRE.") {
                return None;
            }
            let prefix = si
                .split(',')
                .map(|t| t.trim().trim_end_matches(')'))
                .find(|t| t.starts_with('.') && *t != ".METRE.");
            return Some(1000.0 * prefix.map_or(Some(1.0), si_prefix)?);
        }
        named("LENGTH_UNIT")?;
        let conversion = named("CONVERSION_BASED_UNIT")?;
        let measure = *refs_after_name(conversion).first()?;
        let (value, base) = self.measure(measure)?;
        Some(value * self.length_scale(base, depth + 1)?)
    }

    /// A measure with unit: its value and its unit.
    fn measure(&self, id: u64) -> Option<(f64, u64)> {
        let body = self
            .components(id)
            .into_iter()
            .map(|(_, body)| body)
            .find(|body| !parse_refs(body).is_empty())?;
        let value = *parse_floats(body).first()?;
        Some((value, *parse_refs(body).last()?))
    }

    /// A shape representation relationship, with its placement if it has one.
    fn relationship(&self, id: u64) -> Option<Relationship> {
        if !self.mentions(id, "REPRESENTATION_RELATIONSHIP") {
            return None;
        }
        let components = self.components(id);
        let body = components.iter().find_map(|(kind, body)| {
            matches!(
                *kind,
                "REPRESENTATION_RELATIONSHIP"
                    | "SHAPE_REPRESENTATION_RELATIONSHIP"
                    | "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION"
            )
            .then_some(*body)
            .filter(|body| unquoted(body).matches('#').count() >= 2)
        })?;
        let refs = parse_refs(&unquoted(body));
        let (rep_1, rep_2) = (*refs.first()?, *refs.get(1)?);
        let transformation = components
            .iter()
            .find(|(kind, _)| *kind == "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION")
            .and_then(|(_, body)| parse_refs(&unquoted(body)).last().copied())
            .filter(|&t| t != rep_1 && t != rep_2);
        let Some(transformation) = transformation else {
            return Some(Relationship::Same(rep_1, rep_2));
        };
        let transform = self.entities.get(&transformation)?;
        if transform.entity_type != "ITEM_DEFINED_TRANSFORMATION" {
            log::warn!(
                "STEP placement #{transformation} is a {}; read in place",
                transform.entity_type
            );
            return Some(Relationship::Same(rep_1, rep_2));
        }
        let items = parse_refs(&unquoted(&transform.attrs));
        Some(Relationship::Placed {
            rep_1,
            rep_2,
            item_1: *items.first()?,
            item_2: *items.get(1)?,
        })
    }

    /// For each placement relationship an assembly usage names, a
    /// representation of the component it places.
    fn assembly_children(&self, frames: &mut UnionFind) -> HashMap<u64, u64> {
        // A product definition's representations, through its shape.
        let mut reps_of_definition: HashMap<u64, u64> = HashMap::new();
        for e in self.entities.values() {
            if e.entity_type != "SHAPE_DEFINITION_REPRESENTATION" {
                continue;
            }
            let refs = parse_refs(&unquoted(&e.attrs));
            let (Some(&shape), Some(&rep)) = (refs.first(), refs.get(1)) else {
                continue;
            };
            if let Some(definition) = self.shape_definition(shape) {
                let rep = frames.find(rep);
                reps_of_definition
                    .entry(definition)
                    .and_modify(|r| *r = (*r).min(rep))
                    .or_insert(rep);
            }
        }
        let mut children = HashMap::new();
        for e in self.entities.values() {
            if e.entity_type != "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION" {
                continue;
            }
            let refs = parse_refs(&e.attrs);
            let (Some(&relation), Some(&shape)) = (refs.first(), refs.get(1)) else {
                continue;
            };
            let Some(usage) = self.shape_definition(shape) else {
                continue;
            };
            let Some(usage) = self.entities.get(&usage) else {
                continue;
            };
            if !usage.entity_type.ends_with("ASSEMBLY_USAGE_OCCURRENCE") {
                continue;
            }
            // (id, name, description, relating, related, designator): the
            // related definition is the component.
            let refs = parse_refs(&unquoted(&usage.attrs));
            if let Some(child) = refs.get(1).and_then(|d| reps_of_definition.get(d)) {
                children.insert(relation, *child);
            }
        }
        children
    }

    /// What a product definition shape describes.
    fn shape_definition(&self, shape: u64) -> Option<u64> {
        let e = self.entities.get(&shape)?;
        if e.entity_type != "PRODUCT_DEFINITION_SHAPE" {
            return None;
        }
        parse_refs(&unquoted(&e.attrs)).first().copied()
    }

    /// `MAPPED_ITEM('', mapping_source, mapping_target)`.
    fn mapped_item(&self, id: u64) -> Option<(u64, u64)> {
        if !self.mentions(id, "MAPPED_ITEM") {
            return None;
        }
        let body = self.component(id, "MAPPED_ITEM")?;
        let refs = parse_refs(&unquoted(body));
        Some((*refs.first()?, *refs.get(1)?))
    }

    /// `REPRESENTATION_MAP(mapping_origin, mapped_representation)`.
    fn representation_map(&self, id: u64) -> Option<(u64, u64)> {
        let body = self.component(id, "REPRESENTATION_MAP")?;
        let refs = parse_refs(body);
        Some((*refs.first()?, *refs.get(1)?))
    }

    /// An `AXIS2_PLACEMENT_3D` as the transform from its frame to its
    /// representation's, with its origin scaled to millimetres.
    fn placement(&self, id: u64, scale: f64) -> Option<Mat4> {
        let e = self.entities.get(&id)?;
        if e.entity_type != "AXIS2_PLACEMENT_3D" {
            return None;
        }
        // ('', location, axis, ref_direction): either direction may be `$`.
        let fields: Vec<&str> = after_name(&e.attrs)
            .trim_end()
            .trim_end_matches(')')
            .split(',')
            .map(str::trim)
            .collect();
        let point_of = |field: Option<&&str>| -> Option<Vec<f64>> {
            let r = parse_refs(field?).first().copied()?;
            Some(parse_floats(&self.entities.get(&r)?.attrs))
        };
        let location = point_of(fields.first()).filter(|c| c.len() >= 3)?;
        let direction = |field: Option<&&str>| {
            point_of(field)
                .filter(|c| c.len() >= 3)
                .map(|c| Vec3::new(c[0], c[1], c[2]))
        };
        let z = direction(fields.get(1))
            .unwrap_or_else(|| Vec3::new(0.0, 0.0, 1.0))
            .normalize()
            .ok()?;
        let reference = direction(fields.get(2))
            .filter(|r| r.cross(z).length() > 1e-12)
            .unwrap_or_else(|| {
                if Vec3::new(1.0, 0.0, 0.0).cross(z).length() > 1e-12 {
                    Vec3::new(1.0, 0.0, 0.0)
                } else {
                    Vec3::new(0.0, 0.0, 1.0)
                }
            });
        let x = (reference - z * reference.dot(z)).normalize().ok()?;
        let y = z.cross(x);
        let o = Point3::new(location[0], location[1], location[2]);
        Some(Mat4([
            [x.x(), y.x(), z.x(), o.x() * scale],
            [x.y(), y.y(), z.y(), o.y() * scale],
            [x.z(), y.z(), z.z(), o.z() * scale],
            [0.0, 0.0, 0.0, 1.0],
        ]))
    }
}

/// Factors of the SI prefixes.
fn si_prefix(prefix: &str) -> Option<f64> {
    Some(match prefix {
        ".EXA." => 1e18,
        ".PETA." => 1e15,
        ".TERA." => 1e12,
        ".GIGA." => 1e9,
        ".MEGA." => 1e6,
        ".KILO." => 1e3,
        ".HECTO." => 1e2,
        ".DECA." => 1e1,
        ".DECI." => 1e-1,
        ".CENTI." => 1e-2,
        ".MILLI." => 1e-3,
        ".MICRO." => 1e-6,
        ".NANO." => 1e-9,
        ".PICO." => 1e-12,
        ".FEMTO." => 1e-15,
        ".ATTO." => 1e-18,
        _ => return None,
    })
}

/// A record's attributes past its leading quoted name.
fn after_name(attrs: &str) -> &str {
    let mut quoted = false;
    for (i, ch) in attrs.char_indices() {
        if ch == '\'' {
            quoted = !quoted;
        }
        if !quoted && ch == ',' {
            return &attrs[i + 1..];
        }
    }
    ""
}

/// `attrs` with its quoted strings emptied, so a `#` in a name is no
/// reference.
fn unquoted(attrs: &str) -> String {
    let mut out = String::with_capacity(attrs.len());
    let mut quoted = false;
    for ch in attrs.chars() {
        if ch == '\'' {
            quoted = !quoted;
            out.push(ch);
        } else if !quoted {
            out.push(ch);
        }
    }
    out
}
