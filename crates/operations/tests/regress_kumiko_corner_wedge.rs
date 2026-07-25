//! Regression: cutting a kumiko corner wedge by a coaxial strut must stay
//! analytic — the root of the whole kumiko export-integrity family.
//!
//! The tool carves each lattice band rather than fusing it: it starts from a
//! `wedge` (a `revolve`, so it carries cylindrical corner faces) and runs
//! `cutter = cutAll(cutter, family)` per strut family. Both operands are small
//! coaxial revolve wedges — six analytic faces each, two cylinders each. The cut
//! used to come back all-planar (the mesh-fallback signature) because `revolve`
//! emitted INWARD-oriented solids for one profile winding: GFA built one shell,
//! classified it a hole, and aborted with "no outer shell found". Every corner
//! cut in every band took that path, so every band was mesh-derived, and four of
//! eight goma bands arrived at the export non-watertight.
//!
//! This fixture is built natively rather than from captured operands: the
//! captures in `crates/io/tests/data/kumiko_corner_*.bin` predate the fix, so
//! they hold already-inverted wedges and can never pass (see
//! `crates/io/tests/kumiko_corner_wedge_inmem.rs`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::revolve::revolve;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;

/// A wedge: rectangle `r0..r1 × z0..z1` in the XZ plane (which contains the Z
/// axis), revolved `angle` about Z. Wound the way the tool's own profiles are —
/// CCW in the (radial, axial) chart, the winding that used to invert.
fn wedge(
    topo: &mut Topology,
    r0: f64,
    r1: f64,
    z0: f64,
    z1: f64,
    angle: f64,
) -> brepkit_topology::solid::SolidId {
    let pts = vec![
        Point3::new(r0, 0.0, z0),
        Point3::new(r1, 0.0, z0),
        Point3::new(r1, 0.0, z1),
        Point3::new(r0, 0.0, z1),
    ];
    let wire = brepkit_topology::builder::make_polygon_wire(topo, &pts, 1e-7).unwrap();
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, -1.0, 0.0),
            d: 0.0,
        },
    ));
    revolve(
        topo,
        face,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        angle,
    )
    .expect("revolve wedge")
}

fn surface_mix(topo: &Topology, sid: SolidId) -> HashMap<&'static str, usize> {
    let mut mix: HashMap<&'static str, usize> = HashMap::new();
    for fid in solid_faces(topo, sid).unwrap() {
        *mix.entry(topo.face(fid).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    mix
}

/// Mesh edges not shared by exactly two triangles, after welding coincident
/// vertices by quantized position — geometric watertightness, not index sharing.
fn mesh_boundary_edges(positions: &[Point3], indices: &[u32]) -> usize {
    let q = 1e6;
    let mut canon: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut remap = vec![0u32; positions.len()];
    for (i, p) in positions.iter().enumerate() {
        let key = (
            (p.x() * q).round() as i64,
            (p.y() * q).round() as i64,
            (p.z() * q).round() as i64,
        );
        let next = u32::try_from(canon.len()).unwrap();
        remap[i] = *canon.entry(key).or_insert(next);
    }
    let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in indices.chunks_exact(3) {
        let v = [
            remap[tri[0] as usize],
            remap[tri[1] as usize],
            remap[tri[2] as usize],
        ];
        for &(a, b) in &[(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            *edges
                .entry(if a < b { (a, b) } else { (b, a) })
                .or_insert(0) += 1;
        }
    }
    edges.values().filter(|&&c| c != 2).count()
}

/// `(free, over)` edge uses — a watertight manifold solid reports `(0, 0)`.
fn edge_uses(topo: &Topology, sid: SolidId) -> (usize, usize) {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for fid in solid_faces(topo, sid).unwrap() {
        let f = topo.face(fid).unwrap();
        for wid in std::iter::once(f.outer_wire()).chain(f.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    (
        uses.values().filter(|&&c| c == 1).count(),
        uses.values().filter(|&&c| c > 2).count(),
    )
}

#[test]
fn corner_wedge_operands_are_outward_analytic() {
    // Guard the guard: an unvalidated operand already cost this campaign several
    // passes. Both wedges must be outward-oriented (positive signed volume) and
    // carry their two cylindrical corner walls.
    let mut topo = Topology::new();
    for (label, sid) in [
        (
            "base",
            wedge(&mut topo, 1.55, 4.75, 2.7, 20.8, 45.0_f64.to_radians()),
        ),
        (
            "strut",
            wedge(&mut topo, 3.0, 6.0, 8.0, 9.0, 25.0_f64.to_radians()),
        ),
    ] {
        let mix = surface_mix(&topo, sid);
        assert_eq!(
            mix.get("cylinder").copied(),
            Some(2),
            "{label} should carry 2 cylindrical corner faces, got {mix:?}"
        );
        assert_eq!(
            edge_uses(&topo, sid),
            (0, 0),
            "{label} operand must be watertight and manifold"
        );
        let signed = brepkit_operations::measure::oriented_solid_volume(&topo, sid, 0.05).unwrap();
        assert!(
            signed > 0.0,
            "{label} operand must be outward-oriented, got signed volume {signed:.3}"
        );
    }
}

#[test]
fn corner_wedge_cut_stays_analytic() {
    let mut topo = Topology::new();
    let base = wedge(&mut topo, 1.55, 4.75, 2.7, 20.8, 45.0_f64.to_radians());
    let strut = wedge(&mut topo, 3.0, 6.0, 8.0, 9.0, 25.0_f64.to_radians());

    let result = boolean(&mut topo, BooleanOp::Cut, base, strut).expect("coaxial wedge cut");

    let mix = surface_mix(&topo, result);
    let faces: usize = mix.values().sum();

    // The tell: both operands carry cylinders, so an analytic result keeps some.
    // All-planar means the mesh fallback ran.
    assert!(
        mix.get("cylinder").copied().unwrap_or(0) > 0,
        "cut must stay analytic and keep cylindrical corner faces, got {faces} faces {mix:?}"
    );
    assert_eq!(
        edge_uses(&topo, result),
        (0, 0),
        "cut result must be watertight and manifold, got {faces} faces {mix:?}"
    );

    // The by-edge-id gate above is blind to POSITION-duplicate free edges, so
    // confirm watertightness on the welded mesh too.
    let mesh = brepkit_operations::tessellate::tessellate_solid(&topo, result, 0.01).unwrap();
    assert_eq!(
        mesh_boundary_edges(&mesh.positions, &mesh.indices),
        0,
        "cut result's mesh must have no boundary edges"
    );

    // Ground truth by Pappus, since a doubled or dropped face can still pass
    // every topological gate: the removed lens is the strut's overlap with the
    // base, r 3.0..4.75 × z 8..9 over the strut's 25°.
    let base_vol = 45.0_f64.to_radians() * (0.5 * (1.55 + 4.75)) * ((4.75 - 1.55) * (20.8 - 2.7));
    let cut_vol = 25.0_f64.to_radians() * (0.5 * (3.0 + 4.75)) * ((4.75 - 3.0) * (9.0 - 8.0));
    let expected = base_vol - cut_vol;
    let signed = brepkit_operations::measure::oriented_solid_volume(&topo, result, 0.01).unwrap();
    let rel_err = (signed - expected).abs() / expected;
    assert!(
        rel_err < 0.01,
        "cut result volume should be ~{expected:.3}, got {signed:.3} (rel_err={rel_err:.2e})"
    );
}

#[test]
fn corner_wedge_cut_by_disjoint_strut_keeps_the_base() {
    // Four of the eight goma bands cut with a strut that does not touch the base
    // at all (tool0 spans z[-8.3, 2.2] against a base at z[2.7, 20.8]). GFA keeps
    // all of A's faces, so the result is the base wedge unmodified — which only
    // holds if the base is outward-oriented in the first place.
    let mut topo = Topology::new();
    let base = wedge(&mut topo, 1.55, 4.75, 2.7, 20.8, 45.0_f64.to_radians());
    let away = wedge(&mut topo, 1.55, 4.75, -8.3, 2.2, 45.0_f64.to_radians());

    let result = boolean(&mut topo, BooleanOp::Cut, base, away).expect("disjoint wedge cut");

    let mix = surface_mix(&topo, result);
    assert_eq!(
        mix.get("cylinder").copied(),
        Some(2),
        "a disjoint cut must return the base wedge's own faces, got {mix:?}"
    );
    assert_eq!(mix.values().sum::<usize>(), 6, "got {mix:?}");
}
