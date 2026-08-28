//! Test-only shell incidence diagnostics shared by the cross-fillet regressions.

use std::collections::HashMap;

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

/// A free or over-shared edge, including the face-side ownership labels that
/// the batch fillet planner is expected to make explicit.
#[derive(Debug)]
pub struct EdgeAudit {
    pub edge: EdgeId,
    pub uses: usize,
    pub owners: Vec<(FaceId, String)>,
    pub category: &'static str,
}

/// Count edge incidence and describe every edge that is not owned exactly
/// twice. The category names are intentionally stable: they distinguish a
/// missing producer (free edge) from a duplicate producer (over-shared edge),
/// while `owners` retains the concrete surface types for diagnosis.
pub fn audit_shell(topo: &Topology, solid: SolidId) -> Vec<EdgeAudit> {
    let mut uses: HashMap<EdgeId, Vec<FaceId>> = HashMap::new();
    for fid in solid_faces(topo, solid).expect("solid faces should be traversable") {
        let face = topo.face(fid).expect("face should be present");
        let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
        for wid in wires {
            for oe in topo.wire(wid).expect("wire should be present").edges() {
                uses.entry(oe.edge()).or_default().push(fid);
            }
        }
    }

    let mut audit = uses
        .into_iter()
        .filter(|(_, owners)| owners.len() != 2)
        .map(|(edge, owners)| {
            let owner_labels: Vec<(FaceId, String)> = owners
                .into_iter()
                .map(|fid| {
                    let tag = topo
                        .face(fid)
                        .expect("owner face should be present")
                        .surface()
                        .type_tag()
                        .to_owned();
                    (fid, tag)
                })
                .collect();
            let category = if owner_labels.len() == 1 {
                "missing producer/consumer"
            } else {
                "duplicate producer/consumer"
            };
            EdgeAudit {
                edge,
                uses: owner_labels.len(),
                owners: owner_labels,
                category,
            }
        })
        .collect::<Vec<_>>();
    audit.sort_unstable_by_key(|entry| entry.edge.index());
    audit
}

/// Return the shell's free-edge and over-shared-edge counts.
pub fn shell_health(topo: &Topology, solid: SolidId) -> (usize, usize) {
    let audit = audit_shell(topo, solid);
    (
        audit.iter().filter(|entry| entry.uses == 1).count(),
        audit.iter().filter(|entry| entry.uses > 2).count(),
    )
}
