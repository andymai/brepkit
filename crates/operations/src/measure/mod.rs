//! Measurement operations for B-rep geometry: bounding boxes, areas,
//! volumes, and centers of mass.

mod area;
mod bounding_box;
mod edge_length;
pub(crate) mod helpers;
mod volume;

pub use area::{face_area, solid_surface_area};
pub use bounding_box::solid_bounding_box;
pub use edge_length::{edge_length, face_perimeter, wire_length};
pub use volume::{solid_center_of_mass, solid_volume, solid_volume_from_faces};

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_cube_non_manifold;

    use super::*;

    // ── Helper: assert relative error within tolerance ────────────
    //
    // For analytic primitives (box, cylinder, sphere, cone, torus) we
    // expect 1e-8 relative error. For NURBS-involving operations
    // (fillet, boolean, tessellation-based) we accept 1e-4.

    fn assert_rel(actual: f64, expected: f64, rel_tol: f64, label: &str) {
        let rel_err = if expected.abs() < 1e-15 {
            actual.abs()
        } else {
            (actual - expected).abs() / expected.abs()
        };
        assert!(
            rel_err < rel_tol,
            "{label}: expected {expected:.8}, got {actual:.8}, \
             rel_err={rel_err:.2e} (tolerance={rel_tol:.0e})"
        );
    }

    /// Assert that `aabb` contains the box `[min_x..max_x, min_y..max_y, min_z..max_z]`
    /// within a 1e-6 slack on each bound.
    fn assert_aabb_contains(
        aabb: &brepkit_math::aabb::Aabb3,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
    ) {
        let slack = 1e-6;
        assert!(
            aabb.min.x() <= min_x + slack,
            "min.x={} > {min_x}",
            aabb.min.x()
        );
        assert!(
            aabb.min.y() <= min_y + slack,
            "min.y={} > {min_y}",
            aabb.min.y()
        );
        assert!(
            aabb.min.z() <= min_z + slack,
            "min.z={} > {min_z}",
            aabb.min.z()
        );
        assert!(
            aabb.max.x() >= max_x - slack,
            "max.x={} < {max_x}",
            aabb.max.x()
        );
        assert!(
            aabb.max.y() >= max_y - slack,
            "max.y={} < {max_y}",
            aabb.max.y()
        );
        assert!(
            aabb.max.z() >= max_z - slack,
            "max.z={} < {max_z}",
            aabb.max.z()
        );
    }

    // ── Bounding box ─────────────────────────────────────────────

    #[test]
    fn unit_cube_bounding_box() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let aabb = solid_bounding_box(&topo, solid).unwrap();
        let tol = Tolerance::new();

        assert!(tol.approx_eq(aabb.min.x(), 0.0));
        assert!(tol.approx_eq(aabb.min.y(), 0.0));
        assert!(tol.approx_eq(aabb.min.z(), 0.0));
        assert!(tol.approx_eq(aabb.max.x(), 1.0));
        assert!(tol.approx_eq(aabb.max.y(), 1.0));
        assert!(tol.approx_eq(aabb.max.z(), 1.0));
    }

    /// AABB for a sphere must include the full radius extent in all axes.
    /// Sphere at origin with r=5: AABB should be [-5,-5,-5] to [5,5,5].
    #[test]
    fn sphere_bounding_box() {
        use crate::primitives::make_sphere;

        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 5.0, 8).unwrap();

        let aabb = solid_bounding_box(&topo, solid).unwrap();
        // Sphere AABB is expanded by expand_aabb_for_surface to include +/-r.
        assert_aabb_contains(&aabb, -5.0, -5.0, -5.0, 5.0, 5.0, 5.0);
    }

    /// AABB for a cylinder at origin, r=3, h=10 (z=0..10).
    /// Must include the full circular extent: x,y in [-3,3].
    #[test]
    fn cylinder_bounding_box() {
        use crate::primitives::make_cylinder;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 3.0, 10.0).unwrap();

        let aabb = solid_bounding_box(&topo, solid).unwrap();
        assert_aabb_contains(&aabb, -3.0, -3.0, 0.0, 3.0, 3.0, 10.0);
    }

    /// AABB for a torus at origin, R=10, r=3, axis along Z.
    /// Radial extent: +/-(R+r) = +/-13 in x,y.
    /// Axial extent: +/-r = +/-3 in z.
    #[test]
    fn torus_bounding_box() {
        use crate::primitives::make_torus;

        let mut topo = Topology::new();
        let solid = make_torus(&mut topo, 10.0, 3.0, 16).unwrap();

        let aabb = solid_bounding_box(&topo, solid).unwrap();
        // Radial: +/-(R+r) = +/-13
        assert!(aabb.min.x() <= -13.0 + 1e-6, "min.x={}", aabb.min.x());
        assert!(aabb.min.y() <= -13.0 + 1e-6, "min.y={}", aabb.min.y());
        assert!(aabb.max.x() >= 13.0 - 1e-6, "max.x={}", aabb.max.x());
        assert!(aabb.max.y() >= 13.0 - 1e-6, "max.y={}", aabb.max.y());
        // Axial: +/-r = +/-3
        assert!(aabb.min.z() <= -3.0 + 1e-6, "min.z={}", aabb.min.z());
        assert!(aabb.max.z() >= 3.0 - 1e-6, "max.z={}", aabb.max.z());
    }

    // ── Volume: analytic primitives (1e-8 tolerance) ─────────────

    #[test]
    fn unit_cube_volume() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        // Unit cube is all-planar -- volume is computed via polygon method,
        // which should be exact to floating-point precision.
        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        assert_rel(vol, 1.0, 1e-8, "unit cube volume");
    }

    /// make_box(10,10,10) -> volume = 1000.0 exactly.
    /// Previously tested with 5% tolerance (|vol-1000| < 50) -- absurdly loose
    /// for a pure-planar box that uses no tessellation.
    #[test]
    fn box_volume_exact() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        // All-planar solid -> exact polygon-based volume.
        assert_rel(vol, 1000.0, 1e-8, "10x10x10 box volume");
    }

    /// Non-cube rectangular box: volume = dx * dy * dz.
    #[test]
    fn rectangular_box_volume_exact() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.5, 7.3, 4.1).unwrap();
        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        // 2.5 * 7.3 * 4.1 = 74.825
        assert_rel(vol, 2.5 * 7.3 * 4.1, 1e-8, "2.5x7.3x4.1 box volume");
    }

    #[test]
    fn sphere_volume_analytic_exact() {
        use crate::primitives::make_sphere;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        // Low segment count -- analytic path must not depend on tessellation quality.
        let solid = make_sphere(&mut topo, 3.0, 4).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        // V = (4/3)*pi*r^3 = (4/3)*pi*(27) ~ 113.097
        let expected = 4.0 / 3.0 * PI * 27.0;
        assert_rel(vol, expected, 1e-10, "sphere r=3 volume");
    }

    #[test]
    fn cylinder_volume_analytic_exact() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 5.0, 20.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        // V = pi*r^2*h = pi*(25)*(20) ~ 1570.796
        let expected = PI * 25.0 * 20.0;
        assert_rel(vol, expected, 1e-10, "cylinder r=5 h=20 volume");
    }

    #[test]
    fn cone_pointed_volume_analytic_exact() {
        use crate::primitives::make_cone;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 5.0, 0.0, 15.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        // V = (pi/3)*r^2*h = (pi/3)*(25)*(15) ~ 392.699
        let expected = PI / 3.0 * 25.0 * 15.0;
        assert_rel(vol, expected, 1e-10, "pointed cone r=5 h=15 volume");
    }

    #[test]
    fn cone_frustum_volume_analytic_exact() {
        use crate::primitives::make_cone;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 2.0, 1.0, 3.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        // V = (pi*h/3)*(r1^2 + r1*r2 + r2^2) = (pi*3/3)*(4+2+1) ~ 21.991
        let expected = PI / 3.0 * 3.0 * (4.0 + 2.0 + 1.0);
        assert_rel(vol, expected, 1e-10, "frustum r1=2 r2=1 h=3 volume");
    }

    #[test]
    fn torus_volume_analytic_exact() {
        use crate::primitives::make_torus;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_torus(&mut topo, 10.0, 3.0, 32).unwrap();

        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        // V = 2*pi^2*R*r^2 = 2*pi^2*(10)*(9) ~ 1776.529
        let expected = 2.0 * PI * PI * 10.0 * 9.0;
        assert_rel(vol, expected, 1e-10, "torus R=10 r=3 volume");
    }

    // ── Volume: tessellation-based (1e-4 tolerance) ──────────────

    /// Ellipsoid via non-uniform scale of a unit sphere.
    /// V = (4/3)*pi*a*b*c where a,b,c are the semi-axes.
    ///
    /// Non-uniform scale defeats the analytic sphere detector (vertex
    /// distances no longer match stored radius), so this goes through
    /// tessellation. A fine deflection (0.01) is needed because the NURBS
    /// ellipsoid has high curvature variation across its surface.
    #[test]
    fn ellipsoid_volume() {
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = crate::primitives::make_sphere(&mut topo, 1.0, 16).unwrap();
        let mat = brepkit_math::mat::Mat4::scale(5.0, 3.0, 2.0);
        crate::transform::transform_solid(&mut topo, solid, &mat).unwrap();
        // Use fine deflection (0.01) for NURBS ellipsoid -- the adaptive
        // tessellator needs small chord tolerance to refine the high-curvature
        // regions near the minor axis (z semi-axis = 2).
        let vol = solid_volume(&topo, solid, 0.01).unwrap();
        // V = (4/3)*pi*5*3*2 = 40*pi ~ 125.664
        let expected = 4.0 / 3.0 * PI * 5.0 * 3.0 * 2.0;
        assert_rel(vol, expected, 0.01, "ellipsoid 5x3x2 volume");
    }

    /// Extruded 2x3 rectangle by 4 -> volume = 2*3*4 = 24.0 exactly.
    #[test]
    fn extruded_box_volume() {
        use brepkit_math::vec::{Point3, Vec3 as V};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();

        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-10));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(2.0, 0.0, 0.0), 1e-10));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(2.0, 3.0, 0.0), 1e-10));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 3.0, 0.0), 1e-10));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
                OrientedEdge::new(e3, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        let face_id = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: V::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        let solid =
            crate::extrude::extrude(&mut topo, face_id, V::new(0.0, 0.0, 1.0), 4.0).unwrap();

        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        // All-planar extrusion: 2 * 3 * 4 = 24.0 exactly.
        assert_rel(vol, 24.0, 1e-8, "extruded 2x3x4 box volume");
    }

    // ── Volume: operations that involve NURBS (1e-4 tolerance) ───

    /// Fillet on one edge of a 20^3 box.
    ///
    /// A rolling-ball fillet of radius r on one edge of length L removes
    /// a prismatic quarter-cylinder notch:
    ///   V_removed = (1 - pi/4) * r^2 * L
    ///
    /// For r=2, L=20:
    ///   V_removed = (1 - pi/4) * 4 * 20 = (1 - 0.7854) * 80 ~ 17.168
    ///   V_expected = 8000 - 17.168 ~ 7982.83
    ///
    /// Previously: bounds were 7000 < vol < 8000 (12.5% tolerance window).
    /// Now: 1% tolerance around the derived value.
    #[test]
    fn fillet_single_edge_volume() {
        use brepkit_topology::explorer;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
        let edges: Vec<_> = explorer::solid_edges(&topo, solid).unwrap();
        #[allow(deprecated)]
        let filleted =
            crate::fillet::fillet_rolling_ball(&mut topo, solid, &[edges[0]], 2.0).unwrap();
        let vol = solid_volume(&topo, filleted, 0.01).unwrap();
        // V = 20^3 - (1 - pi/4) * r^2 * L = 8000 - (1-pi/4)*4*20 ~ 7982.83
        let expected = 8000.0 - (1.0 - PI / 4.0) * 4.0 * 20.0;
        assert_rel(vol, expected, 0.01, "fillet r=2 on 20^3 box, one edge");
    }

    // ── Surface area ─────────────────────────────────────────────

    #[test]
    fn unit_cube_surface_area() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let area = solid_surface_area(&topo, solid, 0.1).unwrap();
        // 6 faces * 1.0 each = 6.0 exactly for all-planar solid.
        assert_rel(area, 6.0, 1e-8, "unit cube surface area");
    }

    #[test]
    fn unit_cube_face_area() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let area = face_area(&topo, fid, 0.1).unwrap();
            assert_rel(area, 1.0, 1e-8, "unit cube face area");
        }
    }

    /// Box surface area = 2(ab + bc + ac).
    /// 10x10x10 -> SA = 6 * 100 = 600.
    #[test]
    fn box_surface_area_exact() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let area = solid_surface_area(&topo, solid, 0.1).unwrap();
        assert_rel(area, 600.0, 1e-8, "10^3 box surface area");
    }

    /// Cylinder total surface area = 2*pi*r^2 + 2*pi*r*h (two caps + lateral).
    /// r=3, h=10 -> SA = 2*pi*(9) + 2*pi*(3)*(10) = 18*pi + 60*pi = 78*pi ~ 245.04.
    #[test]
    fn cylinder_surface_area() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
        let area = solid_surface_area(&topo, solid, 0.01).unwrap();
        // SA = 2*pi*r^2 + 2*pi*r*h = 2*pi*(9 + 30) = 78*pi ~ 245.04
        let expected = 2.0 * PI * (9.0 + 30.0);
        assert_rel(area, expected, 1e-4, "cylinder r=3 h=10 surface area");
    }

    /// Sphere surface area = 4*pi*r^2. r=5 -> SA = 100*pi ~ 314.16.
    #[test]
    fn sphere_surface_area() {
        use crate::primitives::make_sphere;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 5.0, 16).unwrap();
        let area = solid_surface_area(&topo, solid, 0.01).unwrap();
        // SA = 4*pi*r^2 = 4*pi*(25) = 100*pi ~ 314.159
        let expected = 4.0 * PI * 25.0;
        assert_rel(area, expected, 1e-4, "sphere r=5 surface area");
    }

    // ── Center of mass ───────────────────────────────────────────

    #[test]
    fn unit_cube_center_of_mass() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let com = solid_center_of_mass(&topo, solid, 0.1).unwrap();
        // Symmetric solid centered at (0.5, 0.5, 0.5).
        assert_rel(com.x(), 0.5, 1e-8, "cube CoM x");
        assert_rel(com.y(), 0.5, 1e-8, "cube CoM y");
        assert_rel(com.z(), 0.5, 1e-8, "cube CoM z");
    }

    /// Cylinder r=3, h=10, base at z=0. CoM is at (0, 0, h/2) = (0, 0, 5).
    #[test]
    fn cylinder_center_of_mass() {
        use crate::primitives::make_cylinder;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
        let com = solid_center_of_mass(&topo, solid, 0.01).unwrap();
        // By symmetry: x=0, y=0. By uniform density: z = h/2 = 5.
        assert_rel(com.x().abs(), 0.0, 1e-4, "cylinder CoM x");
        assert_rel(com.y().abs(), 0.0, 1e-4, "cylinder CoM y");
        assert_rel(com.z(), 5.0, 1e-4, "cylinder CoM z");
    }

    /// Pointed cone, r_bottom=4, h=12, base at z=0.
    /// CoM_z = h/4 = 3.0 (standard formula for solid cone).
    #[test]
    fn cone_center_of_mass() {
        use crate::primitives::make_cone;

        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 4.0, 0.0, 12.0).unwrap();
        let com = solid_center_of_mass(&topo, solid, 0.01).unwrap();
        // Cone CoM is at h/4 from the base.
        assert_rel(com.x().abs(), 0.0, 1e-4, "cone CoM x");
        assert_rel(com.y().abs(), 0.0, 1e-4, "cone CoM y");
        assert_rel(com.z(), 3.0, 1e-4, "cone CoM z = h/4 = 3");
    }

    /// Non-symmetric box: 2x3x4, origin at (0,0,0).
    /// CoM = (1, 1.5, 2) -- midpoint of each dimension.
    #[test]
    fn rectangular_box_center_of_mass() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();
        let com = solid_center_of_mass(&topo, solid, 0.1).unwrap();
        assert_rel(com.x(), 1.0, 1e-8, "rect box CoM x");
        assert_rel(com.y(), 1.5, 1e-8, "rect box CoM y");
        assert_rel(com.z(), 2.0, 1e-8, "rect box CoM z");
    }

    // ── Edge & wire lengths ──────────────────────────────────────

    #[test]
    fn edge_length_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let tol = Tolerance::new();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                let len = edge_length(&topo, oe.edge()).unwrap();
                assert!(
                    tol.approx_eq(len, 1.0),
                    "unit cube edge should have length ~1.0, got {len}"
                );
            }
        }
    }

    /// Cylinder circumference edge: 2*pi*r.
    /// r=3 -> 6*pi ~ 18.8496.
    #[test]
    fn edge_length_circle() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 3.0, 10.0).unwrap();

        // Find a circular edge (cap boundary).
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let mut found_circle = false;
        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            if matches!(face.surface(), FaceSurface::Plane { .. }) {
                let wire = topo.wire(face.outer_wire()).unwrap();
                for oe in wire.edges() {
                    let edge = topo.edge(oe.edge()).unwrap();
                    if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_)) {
                        let len = edge_length(&topo, oe.edge()).unwrap();
                        // Circumference = 2*pi*r = 6*pi ~ 18.8496
                        assert_rel(len, 2.0 * PI * 3.0, 1e-8, "circle edge length");
                        found_circle = true;
                    }
                }
            }
        }
        assert!(found_circle, "should have found at least one circular edge");
    }

    #[test]
    fn face_perimeter_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let perim = face_perimeter(&topo, fid).unwrap();
            assert_rel(perim, 4.0, 1e-8, "unit cube face perimeter");
        }
    }

    #[test]
    fn wire_length_rectangle() {
        use brepkit_topology::builder::make_rectangle_face;

        let mut topo = Topology::new();
        let fid = make_rectangle_face(&mut topo, 3.0, 5.0, 1e-7).unwrap();

        let face = topo.face(fid).unwrap();
        let len = wire_length(&topo, face.outer_wire()).unwrap();
        // Perimeter = 2(3+5) = 16.0 exactly.
        assert_rel(len, 16.0, 1e-8, "3x5 rectangle perimeter");
    }

    // ── Boolean volume ───────────────────────────────────────────

    /// Boolean cut must reduce volume: cut(box, cylinder) < box volume.
    ///
    /// Regression test for the cylinder band classification bug.
    #[test]
    #[ignore = "GFA pipeline limitation -- old boolean pipeline removed"]
    fn cut_box_cylinder_volume_decreases() {
        use crate::boolean::{BooleanOp, boolean};
        use crate::primitives::{make_box, make_cylinder};
        use crate::transform::transform_solid;
        use brepkit_math::mat::Mat4;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let bx = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let cyl = make_cylinder(&mut topo, 3.0, 20.0).unwrap();
        transform_solid(&mut topo, cyl, &Mat4::translation(5.0, 5.0, 0.0)).unwrap();
        let cut = boolean(&mut topo, BooleanOp::Cut, bx, cyl).unwrap();

        let box_vol = solid_volume(&topo, bx, 0.01).unwrap();
        let cut_vol = solid_volume(&topo, cut, 0.01).unwrap();
        // V = 10^3 - pi*r^2*h = 1000 - pi*(9)*(10) = 1000 - 90*pi ~ 717.35
        let expected = 1000.0 - PI * 9.0 * 10.0;

        let s = topo.solid(cut).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(
            sh.faces().len(),
            7,
            "expected 7 faces (6 plane + 1 cylinder bore)"
        );

        assert!(
            cut_vol < box_vol,
            "cut volume ({cut_vol:.2}) must be less than box volume ({box_vol:.2})"
        );
        assert_rel(cut_vol, expected, 0.02, "cut(box, cylinder) volume");
    }

    // ── Edge case: degenerate and boundary inputs ────────────────

    /// Volume and AABB of a very thin box (one dimension near-zero).
    /// 10 * 10 * 0.001 -> V = 0.1, SA = 2(100 + 0.01 + 0.01) = 200.04.
    #[test]
    fn thin_box_volume_and_area() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 0.001).unwrap();
        let vol = solid_volume(&topo, solid, 0.001).unwrap();
        let area = solid_surface_area(&topo, solid, 0.001).unwrap();
        // V = 10 * 10 * 0.001 = 0.1
        assert_rel(vol, 0.1, 1e-8, "thin box volume");
        // SA = 2(10*10 + 10*0.001 + 10*0.001) = 2(100+0.01+0.01) = 200.04
        assert_rel(area, 200.04, 1e-8, "thin box surface area");
    }

    /// Very large box to check numerical stability at scale.
    /// 1000 * 1000 * 1000 -> V = 1e9.
    #[test]
    fn large_box_volume() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1000.0, 1000.0, 1000.0).unwrap();
        let vol = solid_volume(&topo, solid, 1.0).unwrap();
        assert_rel(vol, 1e9, 1e-8, "1000^3 box volume");
    }

    /// Very small box to check numerical stability at micro-scale.
    /// 0.001 * 0.001 * 0.001 -> V = 1e-9.
    #[test]
    fn tiny_box_volume() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 0.001, 0.001, 0.001).unwrap();
        let vol = solid_volume(&topo, solid, 0.0001).unwrap();
        assert_rel(vol, 1e-9, 1e-8, "0.001^3 box volume");
    }

    /// Cylinder with very small radius: r=0.01, h=10.
    /// V = pi*(0.0001)*(10) = 0.001*pi ~ 0.003142.
    #[test]
    fn thin_cylinder_volume() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 0.01, 10.0).unwrap();
        let vol = solid_volume(&topo, solid, 0.01).unwrap();
        let expected = PI * 0.0001 * 10.0;
        assert_rel(vol, expected, 1e-8, "thin cylinder r=0.01 h=10 volume");
    }

    /// Flat cylinder (disc-like): r=10, h=0.01.
    /// V = pi*(100)*(0.01) = pi ~ 3.1416.
    #[test]
    fn flat_cylinder_volume() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 10.0, 0.01).unwrap();
        let vol = solid_volume(&topo, solid, 0.01).unwrap();
        let expected = PI * 100.0 * 0.01;
        assert_rel(vol, expected, 1e-8, "flat cylinder r=10 h=0.01 volume");
    }

    /// Nearly-pointed frustum: r_bottom=5, r_top=0.001, h=10.
    /// V = (pi*h/3)*(r1^2 + r1*r2 + r2^2) = (10*pi/3)*(25 + 0.005 + 0.000001) ~ 261.80.
    #[test]
    fn near_pointed_frustum_volume() {
        use crate::primitives::make_cone;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 5.0, 0.001, 10.0).unwrap();
        let vol = solid_volume(&topo, solid, 0.5).unwrap();
        let r1 = 5.0_f64;
        let r2 = 0.001_f64;
        let expected = PI * 10.0 / 3.0 * (r1 * r1 + r1 * r2 + r2 * r2);
        assert_rel(vol, expected, 1e-8, "near-pointed frustum volume");
    }

    // ── Composition: measure after operations ────────────────────

    /// Volume after transform: uniform scale by 2 triples each dimension.
    /// Unit cube -> 2x2x2 cube -> V = 8.
    #[test]
    fn volume_after_uniform_scale() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let mat = brepkit_math::mat::Mat4::scale(2.0, 2.0, 2.0);
        crate::transform::transform_solid(&mut topo, solid, &mat).unwrap();
        let vol = solid_volume(&topo, solid, 0.1).unwrap();
        assert_rel(vol, 8.0, 1e-8, "unit cube scaled x2 volume");
    }

    /// CoM shifts correctly under translation.
    /// Box (0,0,0)-(1,1,1) translated by (10,20,30) -> CoM = (10.5, 20.5, 30.5).
    #[test]
    fn center_of_mass_after_translation() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let mat = brepkit_math::mat::Mat4::translation(10.0, 20.0, 30.0);
        crate::transform::transform_solid(&mut topo, solid, &mat).unwrap();
        let com = solid_center_of_mass(&topo, solid, 0.1).unwrap();
        assert_rel(com.x(), 10.5, 1e-8, "translated CoM x");
        assert_rel(com.y(), 20.5, 1e-8, "translated CoM y");
        assert_rel(com.z(), 30.5, 1e-8, "translated CoM z");
    }

    /// Volume is invariant under rotation.
    #[test]
    fn volume_invariant_under_rotation() {
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 3.0, 4.0, 5.0).unwrap();
        let vol_before = solid_volume(&topo, solid, 0.1).unwrap();

        // Rotate 45 deg around Z axis.
        let mat = brepkit_math::mat::Mat4::rotation_z(PI / 4.0);
        crate::transform::transform_solid(&mut topo, solid, &mat).unwrap();
        let vol_after = solid_volume(&topo, solid, 0.1).unwrap();

        // V = 3*4*5 = 60, should be unchanged.
        assert_rel(vol_before, 60.0, 1e-8, "box volume before rotation");
        assert_rel(vol_after, 60.0, 1e-8, "box volume after rotation");
    }

    // ── Error path validation ────────────────────────────────────

    /// Cone total surface area = pi*r*l + pi*r^2 (lateral + base cap).
    /// r=1, h=1: slant l=sqrt(2), lateral = pi*1*sqrt(2) = pi*sqrt(2) ~ 4.4429.
    /// Total = pi*sqrt(2) + pi ~ 7.584.
    #[test]
    fn cone_face_area_analytic() {
        use crate::primitives::make_cone;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 1.0, 0.0, 1.0).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        let mut lateral_area = 0.0;
        let mut cap_area = 0.0;
        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let area = face_area(&topo, fid, 0.01).unwrap();
            match face.surface() {
                FaceSurface::Cone(_) => lateral_area += area,
                FaceSurface::Plane { .. } => cap_area += area,
                _ => panic!("unexpected surface type in cone"),
            }
        }

        // Lateral area = pi*r*l = pi*1*sqrt(2) ~ 4.4429
        let slant = 2.0_f64.sqrt();
        assert_rel(lateral_area, PI * slant, 1e-8, "cone r=1 h=1 lateral area");
        // Cap (base disk) = pi*r^2 = pi ~ 3.1416
        assert_rel(cap_area, PI, 2e-4, "cone r=1 h=1 base cap area");
    }

    /// Torus total surface area = 4*pi^2*R*r.
    /// R=2, r=0.5: area = 4*pi^2*2*0.5 = 4*pi^2 ~ 39.478.
    #[test]
    fn torus_face_area_analytic() {
        use crate::primitives::make_torus;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        // Use 32 segments for a decent tessellation of boundary
        let solid = make_torus(&mut topo, 2.0, 0.5, 32).unwrap();

        let area = solid_surface_area(&topo, solid, 0.01).unwrap();
        // SA = 4*pi^2*R*r = 4*pi^2*2*0.5 = 4*pi^2 ~ 39.478
        let expected = 4.0 * PI * PI * 2.0 * 0.5;
        assert_rel(area, expected, 1e-8, "torus R=2 r=0.5 surface area");
    }

    /// face_area with a non-planar face and zero deflection should still
    /// return a reasonable result (tessellation at maximum detail).
    #[test]
    fn cylinder_face_area_analytic() {
        use crate::primitives::make_cylinder;
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 3.0, 10.0).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        let mut lateral_area = 0.0;
        let mut cap_area = 0.0;
        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let area = face_area(&topo, fid, 0.01).unwrap();
            match face.surface() {
                FaceSurface::Cylinder(_) => lateral_area += area,
                FaceSurface::Plane { .. } => cap_area += area,
                _ => panic!("unexpected surface type in cylinder"),
            }
        }

        // Lateral area = 2*pi*r*h = 2*pi*(3)*(10) = 60*pi ~ 188.496
        assert_rel(
            lateral_area,
            2.0 * PI * 3.0 * 10.0,
            1e-4,
            "cylinder lateral area",
        );
        // Two caps = 2*pi*r^2 = 2*pi*(9) = 18*pi ~ 56.549
        // Cap area uses Newell's method on 256-sample polygon of circle edge,
        // so discretization error is O(1/n^2) ~ 2e-5 per cap.
        assert_rel(cap_area, 2.0 * PI * 9.0, 2e-4, "cylinder cap area");
    }
}
