#![allow(missing_docs)]

//! Immutable Stage 1 fillet planning.
//!
//! Planning snapshots all topology and selection decisions before any blend
//! geometry is generated.  The resulting value is deterministic and contains
//! only source topology/geometry; later stages must not rediscover selection.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::FaceId;
use brepkit_topology::pcurve::PCurve;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::WireId;

use crate::BlendError;
use crate::g1_chain::group_g1_contours;
use crate::radius_law::RadiusLaw;
use crate::spine::Spine;

/// Whether a source edge belongs to the outer or an inner face wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WireKind {
    Outer,
    Inner,
}

/// Source wire membership of a face restriction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireMembership {
    pub wire: WireId,
    pub kind: WireKind,
}

/// Which side of a source face remains after offsetting for a fillet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeptSide {
    FaceInterior,
}

/// Classification used by Stage 3 corner construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CornerClassification {
    Terminal,
    G1Continuation,
    Junction,
    Periodic,
}

/// A cloneable representation of a radius law.
///
/// Custom closures cannot be cloned, so the immutable plan stores their
/// endpoint samples.  Built-in laws retain their exact evaluation behavior.
#[derive(Debug, Clone, PartialEq)]
pub enum RadiusLawPlan {
    Constant(f64),
    Linear { start: f64, end: f64 },
    SCurve { start: f64, end: f64 },
    Sampled { start: f64, end: f64 },
}

impl RadiusLawPlan {
    #[must_use]
    pub fn evaluate(&self, t: f64) -> f64 {
        match self {
            Self::Constant(radius) => *radius,
            Self::Linear { start, end } | Self::Sampled { start, end } => start + (end - start) * t,
            Self::SCurve { start, end } => {
                let s = t * t * (3.0 - 2.0 * t);
                start + (end - start) * s
            }
        }
    }
}

fn plan_law(law: &RadiusLaw) -> RadiusLawPlan {
    match law {
        RadiusLaw::Constant(radius) => RadiusLawPlan::Constant(*radius),
        RadiusLaw::Linear { start, end } => RadiusLawPlan::Linear {
            start: *start,
            end: *end,
        },
        RadiusLaw::SCurve { start, end } => RadiusLawPlan::SCurve {
            start: *start,
            end: *end,
        },
        RadiusLaw::Custom(_) => RadiusLawPlan::Sampled {
            start: law.evaluate(0.0),
            end: law.evaluate(1.0),
        },
    }
}

/// One ordered ridgeline contour and its source spine.
#[derive(Debug, Clone)]
pub struct FilletContour {
    pub edges: Vec<EdgeId>,
    pub spine: Spine,
    pub side1: FaceId,
    pub side2: FaceId,
    pub radius_law: RadiusLawPlan,
    pub periodic: bool,
    pub terminal_junctions: Vec<VertexId>,
}

/// One source-face restriction to be consumed by trimming.
#[derive(Debug, Clone)]
pub struct FaceRestriction {
    pub contour: usize,
    pub edge: EdgeId,
    pub face: FaceId,
    pub curve: EdgeCurve,
    pub pcurve: Option<PCurve>,
    pub kept_side: KeptSide,
    pub wire: WireMembership,
}

/// Source topology at a selected-edge vertex.
#[derive(Debug, Clone)]
pub struct VertexJunction {
    pub vertex: VertexId,
    pub incident_contours: Vec<usize>,
    pub unselected_sharp_edges: Vec<EdgeId>,
    pub face_fan: Vec<FaceId>,
    pub classification: CornerClassification,
}

/// Complete immutable Stage 1 result.
#[derive(Debug, Clone)]
pub struct FilletPlan {
    pub contours: Vec<FilletContour>,
    pub restrictions: Vec<FaceRestriction>,
    pub junctions: Vec<VertexJunction>,
    pub selected_edges: Vec<EdgeId>,
}

impl FilletPlan {
    /// Build a deterministic plan from a source solid and edge/law requests.
    ///
    /// # Errors
    ///
    /// Returns [`BlendError`] when a selected edge is duplicated, missing,
    /// non-manifold, or cannot be converted into a source spine.
    pub fn build(
        topo: &Topology,
        solid: SolidId,
        edge_sets: &[(Vec<EdgeId>, RadiusLaw)],
    ) -> Result<Self, BlendError> {
        let mut selected = Vec::new();
        let mut laws = HashMap::<usize, usize>::new();
        for (law_index, (edges, _)) in edge_sets.iter().enumerate() {
            for &edge in edges {
                if laws.insert(edge.index(), law_index).is_some() {
                    return Err(BlendError::PlanningFailure {
                        reason: format!("edge {edge:?} requested more than once"),
                    });
                }
                selected.push(edge);
            }
        }
        selected.sort_unstable_by_key(|edge| edge.index());
        if selected.is_empty() {
            return Err(brepkit_topology::TopologyError::Empty {
                entity: "fillet edge set",
            }
            .into());
        }

        let contour_edges = group_g1_contours(topo, solid, &selected, Tolerance::default())?;
        let adjacency = topo.build_adjacency(solid)?;
        let mut contours = Vec::with_capacity(contour_edges.len());
        let mut edge_contour = HashMap::<usize, usize>::new();

        for (contour_index, edges) in contour_edges.into_iter().enumerate() {
            if edges.is_empty() {
                return Err(BlendError::PlanningFailure {
                    reason: "selected edge is not present in the source shell".to_owned(),
                });
            }
            let mut sides = adjacency.faces_for_edge(edges[0]).to_vec();
            sides.sort_unstable_by_key(|face| face.index());
            if sides.len() != 2 {
                return Err(BlendError::PlanningFailure {
                    reason: format!("selected edge {:?} is not manifold", edges[0]),
                });
            }
            let spine = Spine::from_chain(topo, edges.clone())?;
            let law_index = laws[&edges[0].index()];
            if edges.iter().any(|edge| laws[&edge.index()] != law_index) {
                return Err(BlendError::PlanningFailure {
                    reason: "one contour requested multiple radius laws".to_owned(),
                });
            }
            let periodic = spine.is_closed();
            let terminal_junctions = if periodic {
                Vec::new()
            } else {
                let mut vertex_counts = HashMap::<usize, (VertexId, usize)>::new();
                for &edge_id in &edges {
                    let edge = topo.edge(edge_id)?;
                    for vertex in [edge.start(), edge.end()] {
                        vertex_counts
                            .entry(vertex.index())
                            .and_modify(|entry| entry.1 += 1)
                            .or_insert((vertex, 1));
                    }
                }
                let mut terminals: Vec<_> = vertex_counts
                    .into_values()
                    .filter_map(|(vertex, count)| (count == 1).then_some(vertex))
                    .collect();
                terminals.sort_unstable_by_key(|vertex| vertex.index());
                terminals
            };
            for &edge in &edges {
                edge_contour.insert(edge.index(), contour_index);
            }
            contours.push(FilletContour {
                edges,
                spine,
                side1: sides[0],
                side2: sides[1],
                radius_law: plan_law(&edge_sets[law_index].1),
                periodic,
                terminal_junctions,
            });
        }

        let mut restrictions = Vec::with_capacity(selected.len() * 2);
        for contour in &contours {
            for &edge_id in &contour.edges {
                let edge = topo.edge(edge_id)?.clone();
                for &face_id in &[contour.side1, contour.side2] {
                    let face = topo.face(face_id)?;
                    let mut membership = None;
                    let wires = std::iter::once((face.outer_wire(), WireKind::Outer)).chain(
                        face.inner_wires()
                            .iter()
                            .copied()
                            .map(|wire| (wire, WireKind::Inner)),
                    );
                    for (wire_id, kind) in wires {
                        if topo
                            .wire(wire_id)?
                            .edges()
                            .iter()
                            .any(|oriented| oriented.edge() == edge_id)
                        {
                            membership = Some(WireMembership {
                                wire: wire_id,
                                kind,
                            });
                            break;
                        }
                    }
                    let wire = membership.ok_or_else(|| BlendError::PlanningFailure {
                        reason: format!("edge {edge_id:?} missing from face {face_id:?} wire"),
                    })?;
                    restrictions.push(FaceRestriction {
                        contour: edge_contour[&edge_id.index()],
                        edge: edge_id,
                        face: face_id,
                        curve: edge.curve().clone(),
                        pcurve: topo.pcurves().get(edge_id, face_id).cloned(),
                        kept_side: KeptSide::FaceInterior,
                        wire,
                    });
                }
            }
        }

        let (vertex_edges, vertex_faces) = source_vertex_maps(topo, solid)?;
        let selected_set: HashSet<usize> = selected.iter().map(|edge| edge.index()).collect();
        let mut vertices = HashSet::<usize>::new();
        for contour in &contours {
            for &edge_id in &contour.edges {
                let edge = topo.edge(edge_id)?;
                vertices.insert(edge.start().index());
                vertices.insert(edge.end().index());
            }
        }
        let mut junctions = Vec::with_capacity(vertices.len());
        let mut vertex_ids: Vec<_> = vertices.into_iter().collect();
        vertex_ids.sort_unstable();
        for vertex_index in vertex_ids {
            let vertex = vertex_edges[&vertex_index].0;
            let incident_contours: Vec<_> = contours
                .iter()
                .enumerate()
                .filter_map(|(index, contour)| {
                    contour_touches_vertex(topo, contour, vertex).then_some(index)
                })
                .collect();
            let mut sharp_edges = vertex_edges[&vertex_index].1.clone();
            sharp_edges.retain(|edge| !selected_set.contains(&edge.index()));
            let classification = if incident_contours
                .iter()
                .any(|&index| contours[index].periodic)
            {
                CornerClassification::Periodic
            } else if incident_contours.len() == 1 {
                CornerClassification::Terminal
            } else if incident_contours.len() == 2 && sharp_edges.is_empty() {
                CornerClassification::G1Continuation
            } else {
                CornerClassification::Junction
            };
            let mut face_fan = vertex_faces[&vertex_index].clone();
            face_fan.sort_unstable_by_key(|face| face.index());
            face_fan.dedup();
            junctions.push(VertexJunction {
                vertex,
                incident_contours,
                unselected_sharp_edges: sharp_edges,
                face_fan,
                classification,
            });
        }

        Ok(Self {
            contours,
            restrictions,
            junctions,
            selected_edges: selected,
        })
    }

    /// Stable, topology-index-based representation for regression tests.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let mut result = String::new();
        for contour in &self.contours {
            let _ = write!(
                result,
                "C:{}:{}:{}:{}:[{}];",
                contour.side1.index(),
                contour.side2.index(),
                contour.periodic,
                contour.radius_law.evaluate(0.0),
                contour
                    .edges
                    .iter()
                    .map(|edge| edge.index().to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
        }
        for restriction in &self.restrictions {
            let _ = write!(
                result,
                "R:{}:{}:{}:{:?};",
                restriction.contour,
                restriction.edge.index(),
                restriction.face.index(),
                restriction.wire.kind
            );
        }
        for junction in &self.junctions {
            let _ = write!(
                result,
                "J:{}:{:?}:{:?}:{:?};",
                junction.vertex.index(),
                junction.incident_contours,
                junction
                    .unselected_sharp_edges
                    .iter()
                    .map(|edge| edge.index())
                    .collect::<Vec<_>>(),
                junction.classification
            );
        }
        result
    }

    /// Byte form suitable for exact deterministic comparisons.
    #[must_use]
    pub fn canonical_fingerprint(&self) -> Vec<u8> {
        self.fingerprint().into_bytes()
    }
}

type VertexMaps = (
    HashMap<usize, (VertexId, Vec<EdgeId>)>,
    HashMap<usize, Vec<FaceId>>,
);

fn source_vertex_maps(topo: &Topology, solid: SolidId) -> Result<VertexMaps, BlendError> {
    let shell_id = topo.solid(solid)?.outer_shell();
    let shell = topo.shell(shell_id)?;
    let mut edges = HashMap::<usize, (VertexId, Vec<EdgeId>)>::new();
    let mut faces = HashMap::<usize, Vec<FaceId>>::new();
    for &face_id in shell.faces() {
        let face = topo.face(face_id)?;
        let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
        for wire_id in wires {
            for oriented in topo.wire(wire_id)?.edges() {
                let edge_id = oriented.edge();
                let edge = topo.edge(edge_id)?;
                for vertex in [edge.start(), edge.end()] {
                    let entry = edges
                        .entry(vertex.index())
                        .or_insert_with(|| (vertex, Vec::new()));
                    entry.1.push(edge_id);
                    faces.entry(vertex.index()).or_default().push(face_id);
                }
            }
        }
    }
    for (_, edge_ids) in edges.values_mut() {
        edge_ids.sort_unstable_by_key(|edge| edge.index());
        edge_ids.dedup_by_key(|edge| edge.index());
    }
    for face_ids in faces.values_mut() {
        face_ids.sort_unstable_by_key(|face| face.index());
        face_ids.dedup_by_key(|face| face.index());
    }
    Ok((edges, faces))
}

fn contour_touches_vertex(topo: &Topology, contour: &FilletContour, vertex: VertexId) -> bool {
    contour.edges.iter().any(|&edge_id| {
        topo.edge(edge_id)
            .map(|edge| edge.start() == vertex || edge.end() == vertex)
            .unwrap_or(false)
    })
}
