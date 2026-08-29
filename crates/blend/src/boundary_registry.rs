#![allow(clippy::missing_errors_doc)]

//! Canonical planned-boundary ownership for fillet topology construction.
//!
//! A boundary is identified by the planner's logical key, never by endpoint
//! coordinates. Registering a key twice therefore returns the original entry;
//! materializing it allocates one [`EdgeId`] for all of its owners.

use std::collections::HashMap;
use std::fmt;

use brepkit_math::vec::Point3;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::FaceId;
use brepkit_topology::pcurve::PCurve;
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::OrientedEdge;
use brepkit_topology::{Topology, TopologyError};

/// The geometric/topological role of a planned boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BoundaryKind {
    /// Contact between a rebuilt support face and a blend face.
    Contact,
    /// Cross-section boundary between neighbouring stripe/corner faces.
    CrossSection,
    /// Open-contour runout boundary.
    Runout,
    /// Boundary emitted by a corner patch.
    Corner,
}

/// Stable logical identity assigned by the fillet plan.
///
/// These are planner identities, not arena IDs or rounded endpoint values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BoundaryKey {
    /// Boundary role.
    pub kind: BoundaryKind,
    /// Stable contour or junction identity.
    pub contour: usize,
    /// Stable segment identity within the contour.
    pub segment: usize,
    /// Planner-defined side (normally 0 or 1).
    pub side: u8,
}

/// Alias emphasizing that keys are planner identities, not geometry hashes.
pub type LogicalBoundary = BoundaryKey;

impl BoundaryKey {
    /// Construct a contact-boundary key.
    #[must_use]
    pub const fn contact(contour: usize, segment: usize, side: u8) -> Self {
        Self {
            kind: BoundaryKind::Contact,
            contour,
            segment,
            side,
        }
    }

    /// Construct a cross-section-boundary key.
    #[must_use]
    pub const fn cross_section(contour: usize, segment: usize, side: u8) -> Self {
        Self {
            kind: BoundaryKind::CrossSection,
            contour,
            segment,
            side,
        }
    }

    /// Construct a runout-boundary key.
    #[must_use]
    pub const fn runout(contour: usize, segment: usize, side: u8) -> Self {
        Self {
            kind: BoundaryKind::Runout,
            contour,
            segment,
            side,
        }
    }

    /// Construct a corner-boundary key.
    #[must_use]
    pub const fn corner(contour: usize, segment: usize, side: u8) -> Self {
        Self {
            kind: BoundaryKind::Corner,
            contour,
            segment,
            side,
        }
    }
}

/// A planned vertex reference, including periodic identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlannedVertex {
    /// Topology vertex allocated for this planned identity.
    pub vertex: VertexId,
    /// Identity of a periodic seam occurrence, when applicable.
    pub periodic_identity: Option<u64>,
}

impl PlannedVertex {
    /// Construct a non-periodic planned vertex.
    #[must_use]
    pub const fn new(vertex: VertexId) -> Self {
        Self {
            vertex,
            periodic_identity: None,
        }
    }

    /// Construct a planned vertex with an explicit periodic occurrence.
    #[must_use]
    pub const fn periodic(vertex: VertexId, periodic_identity: u64) -> Self {
        Self {
            vertex,
            periodic_identity: Some(periodic_identity),
        }
    }
}

/// One expected face owner of a planned boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryOwner {
    /// Face that should consume the boundary, when already materialized.
    pub face: Option<FaceId>,
    /// Actionable planner label.
    pub label: String,
    /// Orientation in the owner's wire traversal.
    pub forward: bool,
}

impl BoundaryOwner {
    /// Construct an owner tied to an existing face.
    pub fn new(face: FaceId, label: impl Into<String>, forward: bool) -> Self {
        Self {
            face: Some(face),
            label: label.into(),
            forward,
        }
    }

    /// Construct a planned owner whose face will be allocated later.
    pub fn planned(label: impl Into<String>, forward: bool) -> Self {
        Self {
            face: None,
            label: label.into(),
            forward,
        }
    }
}

/// A boundary's complete planned geometric and ownership record.
#[derive(Debug, Clone)]
pub struct BoundaryEntry {
    /// Logical planner identity.
    pub key: BoundaryKey,
    /// Start planned vertex (including periodic occurrence).
    pub start: PlannedVertex,
    /// End planned vertex (including periodic occurrence).
    pub end: PlannedVertex,
    /// Exact 3D curve geometry.
    pub curve: EdgeCurve,
    /// Parameter interval on `curve`.
    pub parameter_range: (f64, f64),
    /// Optional pcurve for each expected owner, in owner order.
    pub pcurves: [Option<PCurve>; 2],
    /// Exactly two expected owners.
    pub owners: [BoundaryOwner; 2],
    /// Materialized shared topology edge.
    pub edge: Option<EdgeId>,
    uses: Vec<BoundaryUse>,
}

impl BoundaryEntry {
    /// Number of planned owner uses recorded for this entry.
    #[must_use]
    pub fn planned_uses(&self) -> usize {
        self.uses.len()
    }

    /// Materialized edge, if allocation has happened.
    #[must_use]
    pub const fn edge_id(&self) -> Option<EdgeId> {
        self.edge
    }

    /// Planned owner-use records.
    #[must_use]
    pub fn uses(&self) -> &[BoundaryUse] {
        &self.uses
    }
}

/// One planned use recorded through [`BoundaryRegistry::oriented_edge`].
#[derive(Debug, Clone, Copy)]
pub struct BoundaryUse {
    /// Owner index in the expected-owner array.
    pub owner: usize,
    /// Actual orientation requested by the consumer.
    pub forward: bool,
}

/// A handle into a [`BoundaryRegistry`].
pub type BoundaryHandle = usize;

/// One incidence count from an audit.
#[derive(Debug, Clone)]
pub struct BoundaryIncidence {
    /// Registry key when this is a registered edge, otherwise `None`.
    pub key: Option<BoundaryKey>,
    /// Edge being audited.
    pub edge: EdgeId,
    /// Number of wire uses.
    pub uses: usize,
    /// Expected owner labels (empty for non-registry result edges).
    pub expected_owners: Vec<String>,
    /// Actual owner labels observed in result wires.
    pub actual_owners: Vec<String>,
}

/// Actionable preassembly/postassembly incidence failure.
#[derive(Debug, Clone)]
pub struct BoundaryAuditError {
    phase: &'static str,
    issues: Vec<String>,
}

impl BoundaryAuditError {
    fn new(phase: &'static str, issues: Vec<String>) -> Self {
        Self { phase, issues }
    }

    /// Audit phase (`preassembly` or `postassembly`).
    #[must_use]
    pub const fn phase(&self) -> &'static str {
        self.phase
    }

    /// Individual owner-labelled diagnostics.
    #[must_use]
    pub fn issues(&self) -> &[String] {
        &self.issues
    }
}

impl fmt::Display for BoundaryAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} boundary incidence audit failed", self.phase)?;
        for issue in &self.issues {
            write!(f, "; {issue}")?;
        }
        Ok(())
    }
}

impl std::error::Error for BoundaryAuditError {}

/// Canonical registry for all fillet topology boundaries.
#[derive(Debug, Default, Clone)]
pub struct BoundaryRegistry {
    entries: Vec<BoundaryEntry>,
    by_key: HashMap<BoundaryKey, BoundaryHandle>,
    by_edge: HashMap<EdgeId, BoundaryHandle>,
}

impl BoundaryRegistry {
    /// Construct an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of planned boundaries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no boundary has been planned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over entries in deterministic planner insertion order.
    pub fn entries(&self) -> impl Iterator<Item = &BoundaryEntry> {
        self.entries.iter()
    }

    /// Look up a boundary by its logical planner key.
    #[must_use]
    pub fn lookup(&self, key: BoundaryKey) -> Option<BoundaryHandle> {
        self.by_key.get(&key).copied()
    }

    /// Alias for [`Self::lookup`] used by topology builders.
    #[must_use]
    pub fn lookup_by_logical_key(&self, key: &LogicalBoundary) -> Option<BoundaryHandle> {
        self.lookup(*key)
    }

    /// Access an entry by handle.
    #[must_use]
    pub fn entry(&self, handle: BoundaryHandle) -> Option<&BoundaryEntry> {
        self.entries.get(handle)
    }
    /// Alias for [`Self::entry`] for callers that use map-like terminology.
    #[must_use]
    pub fn get(&self, handle: BoundaryHandle) -> Option<&BoundaryEntry> {
        self.entry(handle)
    }

    /// Plan a boundary, returning the existing handle for a repeated key.
    ///
    /// The first registration owns the exact curve, parameter range, vertices,
    /// and owner labels. Re-registering a key must describe that same plan;
    /// endpoint proximity is never used to identify or merge boundaries.
    pub fn register(
        &mut self,
        key: BoundaryKey,
        start: PlannedVertex,
        end: PlannedVertex,
        curve: EdgeCurve,
        parameter_range: (f64, f64),
        owners: [BoundaryOwner; 2],
    ) -> Result<BoundaryHandle, BoundaryAuditError> {
        if let Some(&handle) = self.by_key.get(&key) {
            let existing = &self.entries[handle];
            let same_shape = existing.start == start
                && existing.end == end
                && existing.parameter_range == parameter_range
                && existing.curve.type_tag() == curve.type_tag();
            let same_owners = existing.owners.iter().zip(owners.iter()).all(|(old, new)| {
                old.face == new.face && old.label == new.label && old.forward == new.forward
            });
            if !same_shape || !same_owners {
                return Err(BoundaryAuditError::new(
                    "preassembly",
                    vec![format!(
                        "logical boundary {key:?} was independently planned with different geometry or owners (expected {} / {}, got {} / {})",
                        existing.owners[0].label,
                        existing.owners[1].label,
                        owners[0].label,
                        owners[1].label
                    )],
                ));
            }
            return Ok(handle);
        }
        let handle = self.entries.len();
        self.entries.push(BoundaryEntry {
            key,
            start,
            end,
            curve,
            parameter_range,
            pcurves: [None, None],
            owners,
            edge: None,
            uses: Vec::new(),
        });
        self.by_key.insert(key, handle);
        Ok(handle)
    }

    /// Alias emphasizing that registration is the planner allocation step.
    pub fn allocate(
        &mut self,
        key: BoundaryKey,
        start: PlannedVertex,
        end: PlannedVertex,
        curve: EdgeCurve,
        parameter_range: (f64, f64),
        owners: [BoundaryOwner; 2],
    ) -> Result<BoundaryHandle, BoundaryAuditError> {
        self.register(key, start, end, curve, parameter_range, owners)
    }

    /// Set the optional pcurve for one expected owner.
    pub fn set_pcurve(
        &mut self,
        handle: BoundaryHandle,
        owner: usize,
        pcurve: PCurve,
    ) -> Result<(), BoundaryAuditError> {
        let Some(entry) = self.entries.get_mut(handle) else {
            return Err(BoundaryAuditError::new(
                "preassembly",
                vec![format!("unknown boundary handle {handle}")],
            ));
        };
        let key = entry.key;
        let Some(slot) = entry.pcurves.get_mut(owner) else {
            return Err(BoundaryAuditError::new(
                "preassembly",
                vec![format!("boundary {key:?} has invalid owner index {owner}")],
            ));
        };
        *slot = Some(pcurve);
        Ok(())
    }
    /// Attach an allocated result face to a planned owner.
    ///
    /// This is used when a face is constructed after its boundaries have
    /// been planned. Replacing a known, different face is rejected so an
    /// owner cannot silently change during assembly.
    pub fn set_owner_face(
        &mut self,
        handle: BoundaryHandle,
        owner: usize,
        face: FaceId,
    ) -> Result<(), BoundaryAuditError> {
        let Some(entry) = self.entries.get_mut(handle) else {
            return Err(BoundaryAuditError::new(
                "preassembly",
                vec![format!("unknown boundary handle {handle}")],
            ));
        };
        let key = entry.key;
        let Some(expected) = entry.owners.get_mut(owner) else {
            return Err(BoundaryAuditError::new(
                "preassembly",
                vec![format!("boundary {key:?} has invalid owner index {owner}")],
            ));
        };
        if let Some(existing) = expected.face {
            if existing != face {
                return Err(BoundaryAuditError::new(
                    "preassembly",
                    vec![format!(
                        "boundary {key:?} owner `{}` already belongs to face {existing:?}, not {face:?}",
                        expected.label
                    )],
                ));
            }
        } else {
            expected.face = Some(face);
        }
        Ok(())
    }

    /// Materialize the one shared edge for a planned boundary.
    pub fn materialize(
        &mut self,
        topo: &mut Topology,
        handle: BoundaryHandle,
    ) -> Result<EdgeId, BoundaryAuditError> {
        let Some(entry) = self.entries.get_mut(handle) else {
            return Err(BoundaryAuditError::new(
                "preassembly",
                vec![format!("unknown boundary handle {handle}")],
            ));
        };
        if let Some(edge) = entry.edge {
            return Ok(edge);
        }
        if topo.vertex(entry.start.vertex).is_err() || topo.vertex(entry.end.vertex).is_err() {
            return Err(BoundaryAuditError::new(
                "preassembly",
                vec![format!(
                    "boundary {:?} references missing planned vertices {:?}->{:?}",
                    entry.key, entry.start.vertex, entry.end.vertex
                )],
            ));
        }
        let edge = topo.add_edge(Edge::new(
            entry.start.vertex,
            entry.end.vertex,
            entry.curve.clone(),
        ));
        entry.edge = Some(edge);
        self.by_edge.insert(edge, handle);
        Ok(edge)
    }
    /// Bind a topology edge allocated by a copy-on-write face rebuild.
    ///
    /// Batch support-face reconstruction allocates its split/contact edges
    /// while constructing the result wire. Binding that edge here transfers
    /// ownership to the canonical registry without allocating a duplicate.
    pub fn bind_existing_edge(
        &mut self,
        topo: &Topology,
        handle: BoundaryHandle,
        edge: EdgeId,
    ) -> Result<(), BoundaryAuditError> {
        let Some(entry) = self.entries.get_mut(handle) else {
            return Err(BoundaryAuditError::new(
                "preassembly",
                vec![format!("unknown boundary handle {handle}")],
            ));
        };
        let actual = topo.edge(edge).map_err(topology_audit_error)?;
        if actual.start() != entry.start.vertex || actual.end() != entry.end.vertex {
            return Err(BoundaryAuditError::new(
                "preassembly",
                vec![format!(
                    "boundary {:?} edge {edge:?} has vertices {:?}->{:?}, expected {:?}->{:?}",
                    entry.key,
                    actual.start(),
                    actual.end(),
                    entry.start.vertex,
                    entry.end.vertex
                )],
            ));
        }
        if let Some(existing) = entry.edge {
            if existing != edge {
                return Err(BoundaryAuditError::new(
                    "preassembly",
                    vec![format!(
                        "boundary {:?} already bound to edge {existing:?}, not {edge:?}",
                        entry.key
                    )],
                ));
            }
            return Ok(());
        }
        entry.edge = Some(edge);
        self.by_edge.insert(edge, handle);
        Ok(())
    }

    /// Return an owner-oriented edge and record its planned use.
    pub fn oriented_edge(
        &mut self,
        topo: &mut Topology,
        handle: BoundaryHandle,
        owner: usize,
    ) -> Result<OrientedEdge, BoundaryAuditError> {
        let edge = self.materialize(topo, handle)?;
        let Some(entry) = self.entries.get_mut(handle) else {
            unreachable!("materialize validated boundary handle")
        };
        let Some(expected) = entry.owners.get(owner) else {
            return Err(BoundaryAuditError::new(
                "preassembly",
                vec![format!(
                    "boundary {:?} has invalid owner index {owner}",
                    entry.key
                )],
            ));
        };
        entry.uses.push(BoundaryUse {
            owner,
            forward: expected.forward,
        });
        Ok(OrientedEdge::new(edge, expected.forward))
    }
    /// Install registered owner pcurves into the topology's relational
    /// pcurve registry after the edge and owner faces have been materialized.
    pub fn install_pcurves(
        &self,
        topo: &mut Topology,
        handle: BoundaryHandle,
    ) -> Result<(), BoundaryAuditError> {
        let entry = self.entry(handle).ok_or_else(|| {
            BoundaryAuditError::new(
                "preassembly",
                vec![format!("unknown boundary handle {handle}")],
            )
        })?;
        let edge = entry.edge_id().ok_or_else(|| {
            BoundaryAuditError::new(
                "preassembly",
                vec![format!(
                    "boundary {:?} has no materialized EdgeId",
                    entry.key
                )],
            )
        })?;
        let pcurves: Vec<(FaceId, PCurve)> = entry
            .owners
            .iter()
            .zip(entry.pcurves.iter())
            .filter_map(|(owner, pcurve)| owner.face.zip(pcurve.clone()))
            .collect();
        for (face, pcurve) in pcurves {
            if topo.face(face).is_err() {
                return Err(BoundaryAuditError::new(
                    "preassembly",
                    vec![format!(
                        "boundary {:?} references missing owner face {face:?}",
                        entry.key
                    )],
                ));
            }
            topo.pcurves_mut().set(edge, face, pcurve);
        }
        Ok(())
    }

    /// Look up the registry entry owning a materialized edge.
    #[must_use]
    pub fn handle_for_edge(&self, edge: EdgeId) -> Option<BoundaryHandle> {
        self.by_edge.get(&edge).copied()
    }

    /// Check that every planned boundary has precisely its two expected uses.
    pub fn preassembly_audit(&self) -> Result<(), BoundaryAuditError> {
        let mut issues = Vec::new();
        for entry in &self.entries {
            if entry.edge.is_none() {
                issues.push(format!(
                    "boundary {:?} (owners: {}, {}) has no materialized EdgeId",
                    entry.key, entry.owners[0].label, entry.owners[1].label
                ));
            }
            let uses = entry.uses.len();
            if uses != 2 {
                issues.push(format!(
                    "boundary {:?} has {uses} planned uses; expected 2 from owners `{}` and `{}`",
                    entry.key, entry.owners[0].label, entry.owners[1].label
                ));
            }
            for owner in 0..2 {
                let count = entry.uses.iter().filter(|use_| use_.owner == owner).count();
                if count != 1 {
                    issues.push(format!(
                        "boundary {:?} owner `{}` has {count} planned uses; expected exactly one",
                        entry.key, entry.owners[owner].label
                    ));
                }
            }
        }
        if issues.is_empty() {
            Ok(())
        } else {
            Err(BoundaryAuditError::new("preassembly", issues))
        }
    }

    /// Audit all result-wire incidences, including non-registry edges.
    pub fn postassembly_audit(
        &self,
        topo: &Topology,
        result_faces: &[FaceId],
    ) -> Result<Vec<BoundaryIncidence>, BoundaryAuditError> {
        let mut incidences: HashMap<EdgeId, Vec<(FaceId, bool)>> = HashMap::new();
        for &face_id in result_faces {
            let face = topo.face(face_id).map_err(topology_audit_error)?;
            let mut wires = vec![face.outer_wire()];
            wires.extend_from_slice(face.inner_wires());
            for wire_id in wires {
                let wire = topo.wire(wire_id).map_err(topology_audit_error)?;
                for oriented in wire.edges() {
                    incidences
                        .entry(oriented.edge())
                        .or_default()
                        .push((face_id, oriented.is_forward()));
                }
            }
        }

        let mut reports = Vec::new();
        let mut issues = Vec::new();
        for (edge, uses) in &incidences {
            let handle = self.by_edge.get(edge).copied();
            let expected_owners = handle
                .map(|h| {
                    self.entries[h]
                        .owners
                        .iter()
                        .map(|owner| owner.label.clone())
                        .collect()
                })
                .unwrap_or_default();
            let actual_owners: Vec<String> = uses
                .iter()
                .map(|(face, forward)| {
                    let label = handle.and_then(|h| {
                        self.entries[h]
                            .owners
                            .iter()
                            .find(|owner| owner.face == Some(*face))
                            .map(|owner| owner.label.as_str())
                    });
                    match label {
                        Some(label) => {
                            format!("{label} ({})", if *forward { "forward" } else { "reverse" })
                        }
                        None => format!(
                            "face {face:?} ({})",
                            if *forward { "forward" } else { "reverse" }
                        ),
                    }
                })
                .collect();
            reports.push(BoundaryIncidence {
                key: handle.map(|h| self.entries[h].key),
                edge: *edge,
                uses: uses.len(),
                expected_owners,
                actual_owners: actual_owners.clone(),
            });
            if uses.len() != 2 {
                let owner_labels = handle
                    .map(|h| {
                        format!(
                            " expected owners `{}` and `{}`",
                            self.entries[h].owners[0].label, self.entries[h].owners[1].label
                        )
                    })
                    .unwrap_or_default();
                issues.push(format!(
                    "result edge {edge:?} has {} uses ({}){}",
                    uses.len(),
                    actual_owners.join(", "),
                    owner_labels
                ));
            }
            if let Some(handle) = handle {
                let entry = &self.entries[handle];
                for (owner_index, owner) in entry.owners.iter().enumerate() {
                    if let Some(face) = owner.face {
                        let matches = uses.iter().any(|(actual_face, forward)| {
                            *actual_face == face && *forward == owner.forward
                        });
                        if !matches {
                            issues.push(format!(
                                "result edge {edge:?} is missing the oriented use for owner {} `{}`",
                                owner_index, owner.label
                            ));
                        }
                    }
                }
            }
        }
        for entry in &self.entries {
            let Some(edge) = entry.edge else { continue };
            if !incidences.contains_key(&edge) {
                issues.push(format!(
                    "planned boundary {:?} edge {edge:?} has use-0; expected owners `{}` and `{}`",
                    entry.key, entry.owners[0].label, entry.owners[1].label
                ));
            }
        }
        if issues.is_empty() {
            Ok(reports)
        } else {
            Err(BoundaryAuditError::new("postassembly", issues))
        }
    }

    /// Allocate one planned vertex identity exactly once for generated endpoints.
    #[allow(clippy::needless_pass_by_ref_mut, clippy::unused_self)]
    pub fn allocate_vertex(
        &mut self,
        topo: &mut Topology,
        vertices: &mut HashMap<(u64, Option<u64>), PlannedVertex>,
        identity: u64,
        periodic_identity: Option<u64>,
        point: Point3,
        tolerance: f64,
    ) -> PlannedVertex {
        if let Some(vertex) = vertices.get(&(identity, periodic_identity)) {
            return *vertex;
        }
        let vertex = PlannedVertex {
            vertex: topo.add_vertex(Vertex::new(point, tolerance)),
            periodic_identity,
        };
        vertices.insert((identity, periodic_identity), vertex);
        vertex
    }
}

fn topology_audit_error(error: TopologyError) -> BoundaryAuditError {
    BoundaryAuditError::new("postassembly", vec![error.to_string()])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::wire::Wire;

    fn fixture() -> (Topology, VertexId, VertexId, FaceId, FaceId) {
        let mut topo = Topology::new();
        let a = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let b = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let existing = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(existing, true)], false).unwrap());
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(existing, false)], false).unwrap());
        let f1 = topo.add_face(Face::new(
            w1,
            Vec::new(),
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let f2 = topo.add_face(Face::new(
            w2,
            Vec::new(),
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));
        (topo, a, b, f1, f2)
    }

    #[test]
    fn logical_key_reuses_exact_edge_and_two_owner_audits_pass() {
        let (mut topo, a, b, _f1, _f2) = fixture();
        let owners = [
            BoundaryOwner::planned("support face", true),
            BoundaryOwner::planned("blend face", false),
        ];
        let key = BoundaryKey::contact(4, 2, 0);
        let mut registry = BoundaryRegistry::new();
        let first = registry
            .register(
                key,
                PlannedVertex::new(a),
                PlannedVertex::new(b),
                EdgeCurve::Line,
                (0.0, 1.0),
                owners.clone(),
            )
            .unwrap();
        let second = registry
            .register(
                key,
                PlannedVertex::new(a),
                PlannedVertex::new(b),
                EdgeCurve::Line,
                (0.0, 1.0),
                owners,
            )
            .unwrap();
        assert_eq!(first, second);
        let edge = registry.materialize(&mut topo, first).unwrap();
        assert_eq!(registry.materialize(&mut topo, second).unwrap(), edge);
        assert_eq!(
            registry.oriented_edge(&mut topo, first, 0).unwrap().edge(),
            edge
        );
        assert_eq!(
            registry.oriented_edge(&mut topo, second, 1).unwrap().edge(),
            edge
        );
        registry.preassembly_audit().unwrap();
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], false).unwrap());
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, false)], false).unwrap());
        let rf1 = topo.add_face(Face::new(
            w1,
            Vec::new(),
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let rf2 = topo.add_face(Face::new(
            w2,
            Vec::new(),
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));
        registry.set_owner_face(first, 0, rf1).unwrap();
        registry.set_owner_face(first, 1, rf2).unwrap();
        let report = registry.postassembly_audit(&topo, &[rf1, rf2]).unwrap();
        assert_eq!(report.iter().find(|row| row.edge == edge).unwrap().uses, 2);
    }

    #[test]
    fn coincident_endpoints_do_not_merge_different_logical_boundaries() {
        let (mut topo, a, b, f1, f2) = fixture();
        let owners = [
            BoundaryOwner::new(f1, "support", true),
            BoundaryOwner::new(f2, "blend", false),
        ];
        let mut registry = BoundaryRegistry::new();
        let first = registry
            .allocate(
                BoundaryKey::contact(0, 0, 0),
                PlannedVertex::new(a),
                PlannedVertex::new(b),
                EdgeCurve::Line,
                (0.0, 1.0),
                owners.clone(),
            )
            .unwrap();
        let second = registry
            .allocate(
                BoundaryKey::contact(0, 1, 0),
                PlannedVertex::new(a),
                PlannedVertex::new(b),
                EdgeCurve::Line,
                (0.0, 1.0),
                owners,
            )
            .unwrap();
        assert_ne!(first, second);
        assert_ne!(
            registry.materialize(&mut topo, first).unwrap(),
            registry.materialize(&mut topo, second).unwrap()
        );
    }

    #[test]
    fn preassembly_diagnostic_names_both_owners() {
        let (mut topo, a, b, f1, f2) = fixture();
        let mut registry = BoundaryRegistry::new();
        let handle = registry
            .register(
                BoundaryKey::runout(0, 0, 0),
                PlannedVertex::new(a),
                PlannedVertex::new(b),
                EdgeCurve::Line,
                (0.0, 1.0),
                [
                    BoundaryOwner::new(f1, "left support", true),
                    BoundaryOwner::new(f2, "right blend", false),
                ],
            )
            .unwrap();
        registry.materialize(&mut topo, handle).unwrap();
        let error = registry.preassembly_audit().unwrap_err().to_string();
        assert!(error.contains("left support"));
        assert!(error.contains("right blend"));
    }
}
