//! Two rods of radius 0.9 that end at the centre of a ball of their radius,
//! nearly opposite, fused exactly: the poses of `two_rods_at_one_ball` in
//! `crates/operations/tests/rod_ending_in_a_ball.rs` at roll 5 and yaw 3.1,
//! captured as exact solids.
//!
//! The rims of the two rods are two circles on the ball that cross at its
//! top, 2.4 degrees apart, where the walls of the rods also meet. The vertex
//! there lies 7.3e-7 past the crossing, along both circles and within its
//! tolerance of each. So one sphere face runs out along one circle past the
//! crossing to the vertex, and back along the other. Its mesh must still
//! close with the faces on the far side of each circle.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::path::{Path, PathBuf};

    use brepkit_io::arena_io::deserialize_solid;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::tessellate::{
        TriangleMesh, boundary_edge_count, non_manifold_edge_count, tessellate_solid,
    };
    use brepkit_topology::Topology;
    use brepkit_topology::solid::SolidId;

    const POSES: [&str; 6] = [
        "rods_at_a_ball_swap_10_0.bin",
        "rods_at_a_ball_swap_10_1.bin",
        "rods_at_a_ball_swap_10_3.bin",
        "rods_at_a_ball_kept_11_0.bin",
        "rods_at_a_ball_kept_11_1.bin",
        "rods_at_a_ball_kept_10_0.bin",
    ];

    fn load(name: &str, topo: &mut Topology) -> SolidId {
        let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name);
        deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
    }

    /// The volume the mesh's triangles enclose, by the divergence theorem.
    fn mesh_volume(mesh: &TriangleMesh) -> f64 {
        mesh.indices
            .chunks(3)
            .map(|t| {
                let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
                let (a, b, c) = (
                    brepkit_math::vec::Vec3::new(a.x(), a.y(), a.z()),
                    brepkit_math::vec::Vec3::new(b.x(), b.y(), b.z()),
                    brepkit_math::vec::Vec3::new(c.x(), c.y(), c.z()),
                );
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    #[test]
    fn rods_crossing_at_a_balls_top_mesh_closed() {
        for name in POSES {
            let mut topo = Topology::new();
            let solid = load(name, &mut topo);
            let volume = solid_volume(&topo, solid, 0.0005).unwrap();
            for deflection in [0.005, 0.01, 0.05] {
                let mesh = tessellate_solid(&topo, solid, deflection).unwrap();
                assert_eq!(
                    boundary_edge_count(&mesh),
                    0,
                    "{name} at {deflection}: open mesh"
                );
                assert_eq!(
                    non_manifold_edge_count(&mesh),
                    0,
                    "{name} at {deflection}: non-manifold mesh"
                );
                // The triangles keep their faces' sense: an inverted run would
                // take twice its volume off.
                let enclosed = mesh_volume(&mesh);
                assert!(
                    (enclosed - volume).abs() < 0.02 * volume,
                    "{name} at {deflection}: the mesh holds {enclosed} of {volume}"
                );
            }
        }
    }
}
