//! Operations on compound entities.
//!
//! Provides utilities for working with compounds of solids:
//! extracting individual solids, fusing all solids in a compound,
//! and computing compound-level measurements.

use brepkit_math::aabb::Aabb3;
use brepkit_topology::Topology;
use brepkit_topology::compound::CompoundId;
use brepkit_topology::solid::SolidId;

use crate::boolean::{BooleanOp, boolean};

/// Extract all solid IDs from a compound.
///
/// # Errors
///
/// Returns an error if the compound ID is invalid.
pub fn explode(
    topo: &Topology,
    compound: CompoundId,
) -> Result<Vec<SolidId>, crate::OperationsError> {
    let comp = topo.compound(compound)?;
    Ok(comp.solids().to_vec())
}

/// Fuse (union) all solids in a compound into a single solid.
///
/// Performs iterative boolean union on all solids. Requires at least
/// one solid in the compound.
///
/// # Errors
///
/// Returns an error if the compound is empty or a boolean operation fails.
pub fn fuse_all(
    topo: &mut Topology,
    compound: CompoundId,
) -> Result<SolidId, crate::OperationsError> {
    let solids = {
        let comp = topo.compound(compound)?;
        comp.solids().to_vec()
    };

    if solids.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "compound has no solids to fuse".into(),
        });
    }

    // Partition solids into overlapping groups. Disjoint solids can be merged
    // directly (no boolean needed), while overlapping groups use boolean fuse.
    let bboxes: Vec<Aabb3> = solids
        .iter()
        .map(|&sid| crate::measure::solid_bounding_box(topo, sid))
        .collect::<Result<_, _>>()?;

    // Per-solid polyhedral bounds (plane normals + vertices), or `None` for any
    // solid with a curved face. Lets `partition_touching` prove that two solids
    // whose loose AABBs overlap are actually disjoint (e.g. honeycomb hex prisms
    // packed tighter than their corner-to-corner AABB extent), keeping them off
    // the expensive boolean path.
    let margin = brepkit_math::tolerance::Tolerance::new().linear;
    let poly_bounds: Vec<Option<PolyhedralBounds>> =
        solids.iter().map(|&s| polyhedral_bounds(topo, s)).collect();

    let groups = partition_touching(&bboxes, &poly_bounds, margin);

    let mut group_results: Vec<SolidId> = Vec::new();
    for group in &groups {
        let group_solids: Vec<SolidId> = group.iter().map(|&i| solids[i]).collect();
        if group_solids.len() == 1 {
            group_results.push(group_solids[0]);
            continue;
        }
        // Pairwise balanced reduction within the overlapping group.
        let mut current = group_solids;
        while current.len() > 1 {
            let mut next = Vec::with_capacity(current.len().div_ceil(2));
            let mut i = 0;
            while i + 1 < current.len() {
                next.push(boolean(topo, BooleanOp::Fuse, current[i], current[i + 1])?);
                i += 2;
            }
            if i < current.len() {
                next.push(current[i]);
            }
            current = next;
        }
        group_results.push(current[0]);
    }

    if group_results.len() == 1 {
        return Ok(group_results[0]);
    }

    merge_disjoint_solids(topo, &group_results)
}

/// Count the total number of solids in a compound.
///
/// # Errors
///
/// Returns an error if the compound ID is invalid.
pub fn solid_count(topo: &Topology, compound: CompoundId) -> Result<usize, crate::OperationsError> {
    let comp = topo.compound(compound)?;
    Ok(comp.solids().len())
}

/// Compute the combined bounding box of all solids in a compound.
///
/// # Errors
///
/// Returns an error if the compound is empty or measurement fails.
pub fn compound_bounding_box(
    topo: &Topology,
    compound: CompoundId,
) -> Result<brepkit_math::aabb::Aabb3, crate::OperationsError> {
    let comp = topo.compound(compound)?;
    let solids = comp.solids();

    if solids.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "compound is empty".into(),
        });
    }

    let mut combined = crate::measure::solid_bounding_box(topo, solids[0])?;
    for &sid in &solids[1..] {
        let bb = crate::measure::solid_bounding_box(topo, sid)?;
        combined = combined.union(bb);
    }

    Ok(combined)
}

/// Union-find path-compressed lookup.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Plane normals (candidate separating axes) and boundary vertices of a
/// solid, used to prove disjointness via the separating-axis theorem.
struct PolyhedralBounds {
    normals: Vec<brepkit_math::vec::Vec3>,
    verts: Vec<brepkit_math::vec::Point3>,
}

/// Collect a solid's plane normals and vertices — but only if *every* outer-shell
/// face is planar. A flat-faced solid is contained in the convex hull of its
/// vertices, which makes the vertex-projection separation test (below) sound.
/// A single curved face can bulge past that hull, so any non-`Plane` face makes
/// this return `None` (the caller then falls back to the conservative AABB test).
fn polyhedral_bounds(topo: &Topology, sid: SolidId) -> Option<PolyhedralBounds> {
    use brepkit_topology::face::FaceSurface;

    let solid = topo.solid(sid).ok()?;
    let shell = topo.shell(solid.outer_shell()).ok()?;

    let mut normals = Vec::new();
    let mut vert_ids = std::collections::HashSet::new();
    for &fid in shell.faces() {
        let face = topo.face(fid).ok()?;
        match face.surface() {
            // Normalize: stored plane normals aren't guaranteed unit length (e.g.
            // raw STEP `DIRECTION` data), and `polyhedral_separated` compares
            // projection gaps against a world-space margin, which is only valid
            // for unit axes. Bail the whole solid to the AABB path on a
            // degenerate normal.
            FaceSurface::Plane { normal, .. } => normals.push(normal.normalize().ok()?),
            _ => return None,
        }
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).ok()?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge()).ok()?;
                vert_ids.insert(edge.start());
                vert_ids.insert(edge.end());
            }
        }
    }

    let mut verts = Vec::with_capacity(vert_ids.len());
    for vid in vert_ids {
        verts.push(topo.vertex(vid).ok()?.point());
    }
    if verts.is_empty() {
        return None;
    }
    Some(PolyhedralBounds { normals, verts })
}

/// Whether two flat-faced solids are provably disjoint: `true` iff some face
/// normal of either separates their vertex projections by a clear `margin`.
///
/// Soundness: each solid lies within the convex hull of its vertices (all faces
/// planar), so a gap between the vertex projections on any axis is a real gap
/// between the solids. Only face-normal axes are tried (not edge-edge cross
/// products), so the test is sound but not complete — an undetected separation
/// just falls through to the boolean, never a false "disjoint" for touching
/// inputs.
fn polyhedral_separated(a: &PolyhedralBounds, b: &PolyhedralBounds, margin: f64) -> bool {
    let project = |verts: &[brepkit_math::vec::Point3], axis: &brepkit_math::vec::Vec3| {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for p in verts {
            let d = p.x() * axis.x() + p.y() * axis.y() + p.z() * axis.z();
            lo = lo.min(d);
            hi = hi.max(d);
        }
        (lo, hi)
    };
    a.normals.iter().chain(b.normals.iter()).any(|axis| {
        let (a_lo, a_hi) = project(&a.verts, axis);
        let (b_lo, b_hi) = project(&b.verts, axis);
        b_lo - a_hi > margin || a_lo - b_hi > margin
    })
}

/// Partition indices into groups that may actually touch (union-find).
///
/// Two solids share a group when their AABBs overlap *unless* both are flat-faced
/// and a separating axis proves a real gap between them. This keeps geometrically
/// disjoint pieces whose loose AABBs overlap (honeycomb hex prisms, tightly
/// packed feet) in separate groups, so `fuse_all` merges them via the cheap
/// disjoint-shell path instead of an O(n) chain of boolean unions.
fn partition_touching(
    bboxes: &[Aabb3],
    poly_bounds: &[Option<PolyhedralBounds>],
    margin: f64,
) -> Vec<Vec<usize>> {
    let n = bboxes.len();
    let mut parent: Vec<usize> = (0..n).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            if !bboxes[i].intersects(bboxes[j]) {
                continue;
            }
            // AABBs overlap. Only keep them apart if we can *prove* a gap.
            if let (Some(pi), Some(pj)) = (&poly_bounds[i], &poly_bounds[j])
                && polyhedral_separated(pi, pj, margin)
            {
                continue;
            }
            let ri = uf_find(&mut parent, i);
            let rj = uf_find(&mut parent, j);
            if ri != rj {
                parent[ri] = rj;
            }
        }
    }

    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        groups.entry(uf_find(&mut parent, i)).or_default().push(i);
    }
    groups.into_values().collect()
}

/// Merge disjoint solids into a single solid by combining all faces.
///
/// Note: the resulting outer shell contains disconnected face groups,
/// which technically violates the connected-shell invariant. This is
/// acceptable for volume measurement and tessellation (which iterate
/// faces independently), but algorithms that assume shell connectivity
/// should be aware. A future improvement would return a `Compound`.
///
/// The result references the input solids' existing faces (no deep copy),
/// so callers that need an independent result must pass copies.
pub(crate) fn merge_disjoint_solids(
    topo: &mut Topology,
    solids: &[SolidId],
) -> Result<SolidId, crate::OperationsError> {
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;

    let mut all_faces = Vec::new();
    let mut inner_shell_ids = Vec::new();

    // Snapshot phase: collect all face IDs and inner shell face sets.
    let mut inner_face_sets: Vec<Vec<brepkit_topology::face::FaceId>> = Vec::new();
    for &sid in solids {
        let solid_data = topo.solid(sid)?;
        let outer_shell = topo.shell(solid_data.outer_shell())?;
        all_faces.extend_from_slice(outer_shell.faces());

        let inner_ids: Vec<_> = solid_data.inner_shells().to_vec();
        for inner_id in inner_ids {
            let inner_shell = topo.shell(inner_id)?;
            inner_face_sets.push(inner_shell.faces().to_vec());
        }
    }

    // Allocate phase: create inner shells.
    for faces in inner_face_sets {
        let inner = Shell::new(faces).map_err(crate::OperationsError::Topology)?;
        inner_shell_ids.push(topo.add_shell(inner));
    }

    let outer = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let outer_id = topo.add_shell(outer);
    Ok(topo.add_solid(Solid::new(outer_id, inner_shell_ids)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::compound::Compound;

    use super::*;

    #[test]
    fn explode_returns_solids() {
        let mut topo = Topology::new();
        let s1 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let cid = topo.add_compound(Compound::new(vec![s1, s2]));

        let solids = explode(&topo, cid).unwrap();
        assert_eq!(solids.len(), 2);
    }

    #[test]
    fn solid_count_works() {
        let mut topo = Topology::new();
        let s1 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let cid = topo.add_compound(Compound::new(vec![s1]));

        assert_eq!(solid_count(&topo, cid).unwrap(), 1);
    }

    #[test]
    fn compound_bbox() {
        let mut topo = Topology::new();
        let s1 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        crate::transform::transform_solid(
            &mut topo,
            s2,
            &brepkit_math::mat::Mat4::translation(5.0, 0.0, 0.0),
        )
        .unwrap();

        let cid = topo.add_compound(Compound::new(vec![s1, s2]));
        let bb = compound_bounding_box(&topo, cid).unwrap();

        let tol = Tolerance::loose();
        // s1 is [0,1], s2 translated by 5 is [5,6]
        assert!(tol.approx_eq(bb.min.x(), 0.0));
        assert!(tol.approx_eq(bb.max.x(), 6.0));
    }

    #[test]
    fn fuse_all_two_overlapping_boxes() {
        let mut topo = Topology::new();
        let s1 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Offset s2 slightly — overlapping boxes.
        crate::transform::transform_solid(
            &mut topo,
            s2,
            &brepkit_math::mat::Mat4::translation(0.5, 0.0, 0.0),
        )
        .unwrap();

        let cid = topo.add_compound(Compound::new(vec![s1, s2]));
        let fused = fuse_all(&mut topo, cid).unwrap();

        let vol = crate::measure::solid_volume(&topo, fused, 0.1).unwrap();
        // Two overlapping unit cubes: total should be less than 2.0.
        assert!(
            vol > 1.0 && vol < 2.0,
            "fused volume should be between 1 and 2, got {vol}"
        );
    }

    /// Build a hexagonal prism (flat top at z=0..h) centred at the origin via
    /// convex hull — a polyhedral stand-in for a honeycomb pocket.
    fn make_hex_prism(topo: &mut Topology, circumradius: f64, height: f64) -> SolidId {
        use brepkit_math::vec::Point3;
        let mut pts = Vec::with_capacity(12);
        for k in 0..6 {
            let a = std::f64::consts::PI / 3.0 * k as f64;
            let (x, y) = (circumradius * a.cos(), circumradius * a.sin());
            pts.push(Point3::new(x, y, 0.0));
            pts.push(Point3::new(x, y, height));
        }
        crate::primitives::make_convex_hull(topo, &pts).unwrap()
    }

    /// Honeycomb-packed hex prisms with a real gap between every pair, but
    /// corner-to-corner AABBs that overlap. The AABB-only partition collapsed
    /// these into one giant group and unioned them with an O(n) boolean chain;
    /// `partition_touching` proves the gaps with the separating-axis test and
    /// keeps each prism in its own group, so `fuse_all` takes the cheap
    /// disjoint-shell merge.
    #[test]
    fn fuse_all_honeycomb_stays_disjoint() {
        let r = 1.0_f64; // circumradius; across-corners = 2r = 2.0
        let pitch = 2.3_f64; // clear gap on every neighbour, AABBs still overlap
        let height = 4.0_f64;
        let nx = 6;
        let ny = 6;

        let mut topo = Topology::new();
        let mut bboxes = Vec::new();
        let mut solids = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let s = make_hex_prism(&mut topo, r, height);
                let x = i as f64 * pitch + (j % 2) as f64 * pitch / 2.0;
                let y = j as f64 * pitch * 0.9;
                crate::transform::transform_solid(
                    &mut topo,
                    s,
                    &brepkit_math::mat::Mat4::translation(x, y, 0.0),
                )
                .unwrap();
                bboxes.push(crate::measure::solid_bounding_box(&topo, s).unwrap());
                solids.push(s);
            }
        }
        let n = solids.len();

        // Every prism is provably disjoint from the others -> one group each.
        let margin = brepkit_math::tolerance::Tolerance::new().linear;
        let pb: Vec<Option<PolyhedralBounds>> = solids
            .iter()
            .map(|&s| polyhedral_bounds(&topo, s))
            .collect();
        let groups = partition_touching(&bboxes, &pb, margin);
        assert_eq!(
            groups.len(),
            n,
            "disjoint hex prisms should each be their own group, got {} groups",
            groups.len()
        );

        // Geometry is still the full disjoint union: volume == n * hex-prism volume.
        let cid = topo.add_compound(Compound::new(solids));
        let fused = fuse_all(&mut topo, cid).unwrap();
        let vol = crate::measure::solid_volume(&topo, fused, 0.05).unwrap();
        let hex_area = 3.0_f64.sqrt() * 1.5 * r * r; // (3*sqrt(3)/2) r^2
        let expected = n as f64 * hex_area * height;
        assert!(
            (vol - expected).abs() < expected * 0.02,
            "fused volume {vol:.2} should match {expected:.2} (n disjoint prisms)"
        );
    }
}
