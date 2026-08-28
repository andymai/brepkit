//! Stage 0 characterization for the roughly-180-edge cross fillet.
//!
//! Fixture provenance: generated through BrepJS/Brepkit from a 90 x 60 x
//! 10 mm base, one extruded cross rib, stacked cylindrical bosses with a
//! coaxial opening, and one row of eleven rectangular vent cuts. The source
//! was captured through the arena serializer, not regenerated or substituted:
//! `/tmp/brepkit-cross-one-row.bin` (37,181 bytes,
//! SHA-256 `7384e289f907982017826fe06bb6da3ce010cf208579c63c715966d65fb92ba5`).
//! The corresponding external OCCT oracle is `/tmp/brepkit-cross-one-row.step`
//! (112,039 bytes, SHA-256
//! `18430a7fae03283415ad59facfb74e0c6d8f83425edc19f977db0d96c62c3077`).
//!
//! The source baseline is F/E/V = 68/189/126, with zero free or non-manifold
//! edges, one face-adjacency component, oriented volume 64,989.530446 at
//! 0.05 mm deflection, and 186 selected fillet edges.

#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

use brepkit_io::arena_io::{deserialize_solid, serialize_solid};
use brepkit_operations::measure::oriented_solid_volume;
use brepkit_operations::query::filter_filletable_edges;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_entity_counts, solid_faces};
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

#[path = "support/shell_audit.rs"]
mod shell_audit;

const RADIUS: f64 = 0.5;
const DEFLECTION: f64 = 0.05;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("cross_one_row_fillet_source.bin")
}

fn load(topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture()).unwrap(), topo).unwrap()
}

fn components(topo: &Topology, solid: SolidId) -> usize {
    let faces = solid_faces(topo, solid).unwrap();
    let mut edge_faces: HashMap<EdgeId, Vec<FaceId>> = HashMap::new();
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                edge_faces.entry(oe.edge()).or_default().push(fid);
            }
        }
    }

    let mut adjacent: HashMap<FaceId, Vec<FaceId>> = HashMap::new();
    for owners in edge_faces.values() {
        for (i, &a) in owners.iter().enumerate() {
            for &b in &owners[i + 1..] {
                adjacent.entry(a).or_default().push(b);
                adjacent.entry(b).or_default().push(a);
            }
        }
    }
    let face_set: HashSet<FaceId> = faces.iter().copied().collect();
    let mut seen = HashSet::new();
    let mut count = 0;
    for &root in &faces {
        if !seen.insert(root) {
            continue;
        }
        count += 1;
        let mut stack = vec![root];
        while let Some(fid) = stack.pop() {
            for &next in adjacent.get(&fid).into_iter().flatten() {
                if face_set.contains(&next) && seen.insert(next) {
                    stack.push(next);
                }
            }
        }
    }
    count
}

fn source_fingerprint(topo: &Topology, solid: SolidId) -> Vec<u8> {
    serialize_solid(topo, solid).unwrap()
}

fn all_edges(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
    solid_edges(topo, solid).unwrap()
}

#[derive(Debug)]
struct EdgeInventory {
    curve_types: BTreeMap<String, usize>,
    surface_pairs: BTreeMap<String, usize>,
    contours: BTreeMap<usize, usize>,
    periodic_open: BTreeMap<&'static str, usize>,
    junction_valences: BTreeMap<(usize, usize), usize>,
}

fn selected_face_map(topo: &Topology, solid: SolidId) -> HashMap<EdgeId, Vec<FaceId>> {
    let mut map: HashMap<EdgeId, Vec<FaceId>> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                map.entry(oe.edge()).or_default().push(fid);
            }
        }
    }
    map
}

fn inventory(topo: &Topology, solid: SolidId, selected: &[EdgeId]) -> EdgeInventory {
    let face_map = selected_face_map(topo, solid);
    let mut curve_types = BTreeMap::new();
    let mut surface_pairs = BTreeMap::new();
    let mut periodic_open = BTreeMap::new();
    let mut vertices: HashMap<_, Vec<EdgeId>> = HashMap::new();

    for &eid in selected {
        let edge = topo.edge(eid).unwrap();
        *curve_types
            .entry(edge.curve().type_tag().to_owned())
            .or_insert(0) += 1;
        let mut surfaces = face_map
            .get(&eid)
            .unwrap()
            .iter()
            .map(|fid| topo.face(*fid).unwrap().surface().type_tag())
            .collect::<Vec<_>>();
        surfaces.sort_unstable();
        *surface_pairs.entry(surfaces.join("+")).or_insert(0) += 1;
        *periodic_open
            .entry(if edge.is_closed() { "periodic" } else { "open" })
            .or_insert(0) += 1;
        vertices.entry(edge.start()).or_default().push(eid);
        vertices.entry(edge.end()).or_default().push(eid);
    }

    let selected_set: HashSet<EdgeId> = selected.iter().copied().collect();
    let mut contour_sizes = Vec::new();
    let mut seen = HashSet::new();
    for &root in selected {
        if !seen.insert(root) {
            continue;
        }
        let mut stack = vec![root];
        let mut size = 0;
        while let Some(eid) = stack.pop() {
            size += 1;
            let edge = topo.edge(eid).unwrap();
            for vertex in [edge.start(), edge.end()] {
                for &next in vertices.get(&vertex).into_iter().flatten() {
                    if selected_set.contains(&next) && seen.insert(next) {
                        stack.push(next);
                    }
                }
            }
        }
        contour_sizes.push(size);
    }
    let mut contours = BTreeMap::new();
    for size in contour_sizes {
        *contours.entry(size).or_insert(0) += 1;
    }

    let mut junction_valences = BTreeMap::new();
    for &eid in selected {
        let edge = topo.edge(eid).unwrap();
        let a = vertices.get(&edge.start()).map_or(0, Vec::len);
        let b = vertices.get(&edge.end()).map_or(0, Vec::len);
        *junction_valences
            .entry(if a <= b { (a, b) } else { (b, a) })
            .or_insert(0) += 1;
    }

    EdgeInventory {
        curve_types,
        surface_pairs,
        contours,
        periodic_open,
        junction_valences,
    }
}

/// Select by durable geometry and adjacent surface types, never by arena ID.
fn geometric_selector(topo: &Topology, solid: SolidId, selected: &[EdgeId]) -> Vec<EdgeId> {
    let face_map = selected_face_map(topo, solid);
    let mut candidates = selected
        .iter()
        .copied()
        .filter(|&eid| {
            let edge = topo.edge(eid).unwrap();
            if edge.curve().type_tag() != "line" || edge.is_closed() {
                return false;
            }
            let mut surfaces = face_map
                .get(&eid)
                .unwrap()
                .iter()
                .map(|fid| topo.face(*fid).unwrap().surface().type_tag())
                .collect::<Vec<_>>();
            surfaces.sort_unstable();
            surfaces == ["plane", "plane"]
        })
        .collect::<Vec<_>>();

    // Canonicalize by endpoint coordinates before reducing the set. This
    // remains stable when later operations rewrite arena IDs or reverse an
    // edge's orientation.
    let endpoint_key = |eid: EdgeId| {
        let edge = topo.edge(eid).unwrap();
        let start = topo.vertex(edge.start()).unwrap().point();
        let end = topo.vertex(edge.end()).unwrap().point();
        let (low, high) = if (start.x(), start.y(), start.z()) <= (end.x(), end.y(), end.z()) {
            (start, end)
        } else {
            (end, start)
        };
        [low.x(), low.y(), low.z(), high.x(), high.y(), high.z()]
    };
    candidates.sort_by(|&a, &b| {
        endpoint_key(a)
            .into_iter()
            .zip(endpoint_key(b))
            .map(|(left, right)| left.total_cmp(&right))
            .find(|ordering| *ordering != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    });
    candidates.truncate(2);
    candidates
}

#[test]
fn cross_source_baseline_and_selection() {
    let mut topo = Topology::new();
    let solid = load(&mut topo);
    assert_eq!(solid_entity_counts(&topo, solid).unwrap(), (68, 189, 126));
    assert_eq!(shell_audit::shell_health(&topo, solid), (0, 0));
    assert_eq!(components(&topo, solid), 1);
    let volume = oriented_solid_volume(&topo, solid, DEFLECTION).unwrap();
    assert!(
        (volume - 64_989.530_446).abs() < 1e-3,
        "source volume {volume}"
    );

    let selected = filter_filletable_edges(&topo, solid, &all_edges(&topo, solid)).unwrap();
    assert_eq!(selected.len(), 186);
    let inventory = inventory(&topo, solid, &selected);
    assert_eq!(
        inventory.curve_types.values().sum::<usize>(),
        selected.len()
    );
    assert_eq!(
        inventory.surface_pairs.values().sum::<usize>(),
        selected.len()
    );
    assert_eq!(
        inventory.periodic_open.values().sum::<usize>(),
        selected.len()
    );
    assert_eq!(
        inventory
            .contours
            .iter()
            .map(|(size, count)| size * count)
            .sum::<usize>(),
        selected.len()
    );
    assert_eq!(
        inventory.junction_valences.values().sum::<usize>(),
        selected.len()
    );
    assert!(inventory.curve_types.contains_key("line"));
    assert!(inventory.surface_pairs.contains_key("plane+plane"));
    assert!(inventory.periodic_open.contains_key("open"));
    assert!(!inventory.contours.is_empty());
    assert!(!inventory.junction_valences.is_empty());
    assert!(!geometric_selector(&topo, solid, &selected).is_empty());
}

#[test]
fn v2_failure_is_measurable_and_audited() {
    let mut topo = Topology::new();
    let solid = load(&mut topo);
    let source = source_fingerprint(&topo, solid);
    let edges = filter_filletable_edges(&topo, solid, &all_edges(&topo, solid)).unwrap();
    let result = brepkit_operations::blend_ops::fillet_v2(&mut topo, solid, &edges, RADIUS)
        .expect("v2 reports its malformed result as a BlendResult");
    let counts = solid_entity_counts(&topo, result.solid).unwrap();
    let health = shell_audit::shell_health(&topo, result.solid);
    assert_eq!(counts, (520, 1_167, 1_114));
    assert_eq!(result.succeeded.len(), 186);
    assert!(result.failed.is_empty());
    assert!(!result.is_partial);
    assert_eq!(health, (272, 8));
    assert_eq!(components(&topo, result.solid), 32);
    let volume = oriented_solid_volume(&topo, result.solid, DEFLECTION).unwrap();
    assert!((volume - 4_994.913_637).abs() < 1e-3, "v2 volume {volume}");
    let audit = shell_audit::audit_shell(&topo, result.solid);
    assert_eq!(audit.len(), 280);
    assert!(audit.iter().all(|entry| !entry.category.is_empty()));
    assert!(audit.iter().all(|entry| !entry.owners.is_empty()));
    assert_eq!(solid_entity_counts(&topo, solid).unwrap(), (68, 189, 126));
    assert_eq!(source_fingerprint(&topo, solid), source);
}

#[test]
#[allow(deprecated)]
fn rolling_production_rejects_degenerate_face_without_source_mutation() {
    let mut topo = Topology::new();
    let solid = load(&mut topo);
    let source = source_fingerprint(&topo, solid);
    let edges = filter_filletable_edges(&topo, solid, &all_edges(&topo, solid)).unwrap();
    let error = brepkit_operations::fillet::fillet_rolling_ball(&mut topo, solid, &edges, RADIUS)
        .expect_err("rolling-ball production path must reject the degenerate face");
    let message = error.to_string();
    assert!(
        message.contains("degenerate face") && message.contains("closed circular edges"),
        "unexpected rolling-ball rejection: {message}"
    );
    assert_eq!(source_fingerprint(&topo, solid), source);
}

#[test]
#[allow(deprecated)]
fn rolling_diagnostic_bypass_result_is_measurable() {
    let mut topo = Topology::new();
    let solid = load(&mut topo);
    let source = source_fingerprint(&topo, solid);
    let edges = filter_filletable_edges(&topo, solid, &all_edges(&topo, solid)).unwrap();
    let solids_before = topo.num_solids();
    let error = brepkit_operations::fillet::fillet_rolling_ball(&mut topo, solid, &edges, RADIUS)
        .expect_err("production guard should reject after assembling the diagnostic result");
    assert!(error.to_string().contains("degenerate face"));
    let diagnostic = topo.solid_id_from_index(solids_before).unwrap();
    assert_eq!(
        solid_entity_counts(&topo, diagnostic).unwrap(),
        (372, 853, 412)
    );
    assert_eq!(shell_audit::shell_health(&topo, diagnostic), (177, 0));
    assert_eq!(components(&topo, diagnostic), 10);
    let volume = oriented_solid_volume(&topo, diagnostic, DEFLECTION).unwrap();
    assert!(
        (volume - 27_435.763_162).abs() < 1e-3,
        "rolling volume {volume}"
    );
    let audit = shell_audit::audit_shell(&topo, diagnostic);
    assert_eq!(audit.len(), 177);
    assert!(audit.iter().all(|entry| !entry.owners.is_empty()));
    assert!(
        audit
            .iter()
            .all(|entry| entry.category == "missing producer/consumer")
    );
    assert_eq!(source_fingerprint(&topo, solid), source);
}
