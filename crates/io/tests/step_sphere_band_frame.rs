//! A sphere band written as one face, its seam running from rim to rim,
//! reads in a frame whose latitudes are its rims, whatever frame its file
//! gives the sphere. In a frame about an axis across the band, the frame's
//! poles lie on the band and its seam runs through one: the mesh opened.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_io::step::read_step;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;

    const HEADER: &str = "ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('sphere band'), '2;1');
FILE_NAME('band.stp', '2024-01-01T00:00:00', (''), (''), '', '', '');
FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#1 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );
#2 = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );
#3 = ( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() );
#4 = UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07), #1, 'distance_accuracy_value', 'confusion accuracy');
#5 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#4)) GLOBAL_UNIT_ASSIGNED_CONTEXT((#1,#2,#3)) REPRESENTATION_CONTEXT('Context3D','3D Context with UNIT and UNCERTAINTY') );
#6 = APPLICATION_CONTEXT('configuration controlled 3D design of mechanical parts and assemblies');
#7 = MECHANICAL_CONTEXT('', #6, 'mechanical');
#8 = APPLICATION_PROTOCOL_DEFINITION('international standard', 'config_control_design', 1994, #6);
#9 = PRODUCT('band', 'band', '', (#7));
#10 = PRODUCT_DEFINITION_FORMATION('', '', #9);
#11 = PRODUCT_DEFINITION_CONTEXT('part definition', #6, 'design');
#12 = PRODUCT_DEFINITION('design', '', #10, #11);
";

    /// The band of a sphere of radius 10 between `z = -5` and `z = 5`, with
    /// a disc at each end, turned `turn` about `y`. The sphere's frame is
    /// about `axis` with reference direction `ref_dir`, both given before
    /// the turn.
    fn zone_step(axis: [f64; 3], ref_dir: [f64; 3], turn: f64) -> String {
        let a = 75f64.sqrt();
        let (c, s) = (turn.cos(), turn.sin());
        let p = |x: f64, y: f64, z: f64| {
            let (x, z) = (c * x + s * z, -s * x + c * z);
            format!("({x:.15E}, {y:.15E}, {z:.15E})")
        };
        let body = format!(
            "#13 = CARTESIAN_POINT('', {v1});
#14 = VERTEX_POINT('', #13);
#15 = CARTESIAN_POINT('', {v2});
#16 = VERTEX_POINT('', #15);
#17 = CARTESIAN_POINT('', {top});
#18 = DIRECTION('', {dz});
#19 = DIRECTION('', {dx});
#20 = AXIS2_PLACEMENT_3D('', #17, #18, #19);
#21 = CIRCLE('', #20, {a:.15E});
#22 = EDGE_CURVE('', #14, #14, #21, .T.);
#23 = CARTESIAN_POINT('', {bot});
#24 = AXIS2_PLACEMENT_3D('', #23, #18, #19);
#25 = CIRCLE('', #24, {a:.15E});
#26 = EDGE_CURVE('', #16, #16, #25, .T.);
#27 = CARTESIAN_POINT('', (0., 0., 0.));
#28 = DIRECTION('', {dy});
#29 = AXIS2_PLACEMENT_3D('', #27, #28, #19);
#30 = CIRCLE('', #29, 10.);
#31 = EDGE_CURVE('', #14, #16, #30, .T.);
#32 = ORIENTED_EDGE('', *, *, #26, .T.);
#33 = ORIENTED_EDGE('', *, *, #31, .F.);
#34 = ORIENTED_EDGE('', *, *, #22, .F.);
#35 = ORIENTED_EDGE('', *, *, #31, .T.);
#36 = EDGE_LOOP('', (#32, #33, #34, #35));
#37 = FACE_OUTER_BOUND('', #36, .T.);
#38 = DIRECTION('', {axis});
#39 = DIRECTION('', {refd});
#40 = AXIS2_PLACEMENT_3D('', #27, #38, #39);
#41 = SPHERICAL_SURFACE('', #40, 10.);
#42 = ADVANCED_FACE('', (#37), #41, .T.);
#43 = ORIENTED_EDGE('', *, *, #22, .T.);
#44 = EDGE_LOOP('', (#43));
#45 = FACE_OUTER_BOUND('', #44, .T.);
#46 = PLANE('', #20);
#47 = ADVANCED_FACE('', (#45), #46, .T.);
#48 = ORIENTED_EDGE('', *, *, #26, .F.);
#49 = EDGE_LOOP('', (#48));
#50 = FACE_OUTER_BOUND('', #49, .T.);
#51 = DIRECTION('', {dmz});
#52 = AXIS2_PLACEMENT_3D('', #23, #51, #19);
#53 = PLANE('', #52);
#54 = ADVANCED_FACE('', (#50), #53, .T.);
#55 = CLOSED_SHELL('', (#42, #47, #54));
#56 = MANIFOLD_SOLID_BREP('', #55);
#57 = ADVANCED_BREP_SHAPE_REPRESENTATION('band', (#56), #5);
#58 = PRODUCT_DEFINITION_SHAPE('','',#12);
#59 = SHAPE_DEFINITION_REPRESENTATION(#58, #57);
ENDSEC;
END-ISO-10303-21;
",
            v1 = p(a, 0.0, 5.0),
            v2 = p(a, 0.0, -5.0),
            top = p(0.0, 0.0, 5.0),
            bot = p(0.0, 0.0, -5.0),
            dz = p(0.0, 0.0, 1.0),
            dx = p(1.0, 0.0, 0.0),
            dy = p(0.0, 1.0, 0.0),
            dmz = p(0.0, 0.0, -1.0),
            axis = p(axis[0], axis[1], axis[2]),
            refd = p(ref_dir[0], ref_dir[1], ref_dir[2]),
        );
        format!("{HEADER}{body}")
    }

    #[test]
    fn a_sphere_band_reads_about_its_rims_in_any_frame_of_its_file() {
        let exact = PI * (2.0 * 100.0 * 5.0 - 2.0 * 125.0 / 3.0);
        let mut mesh_volumes = Vec::new();
        for (label, axis, ref_dir, turn) in [
            ("about z", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 0.0),
            ("about x", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0),
            ("about x, ref y", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0),
            ("tilted", [0.6, 0.0, 0.8], [0.8, 0.0, -0.6], 0.0),
            (
                "turned, about its axis",
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                0.5,
            ),
            ("turned, about x", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.5),
            (
                "turned further, about x",
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                1.2,
            ),
        ] {
            let mut topo = Topology::new();
            let ids = read_step(&zone_step(axis, ref_dir, turn), &mut topo).unwrap();
            assert_eq!(ids.len(), 1, "{label}");
            let solid = ids[0];
            let band_axis = Vec3::new(turn.sin(), 0.0, turn.cos());
            for face in solid_faces(&topo, solid).unwrap() {
                if let FaceSurface::Sphere(sphere) = topo.face(face).unwrap().surface() {
                    assert!(
                        sphere.z_axis().cross(band_axis).length() < 1e-9,
                        "{label}: the band reads about {:?}",
                        sphere.z_axis()
                    );
                    assert!((sphere.center() - Point3::new(0.0, 0.0, 0.0)).length() < 1e-12);
                }
            }
            let report = validate_solid(&topo, solid).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            let volume = solid_volume(&topo, solid, 0.01).unwrap();
            assert!(
                (volume - exact).abs() < 1e-9 * exact,
                "{label}: {volume} against {exact}"
            );
            let mesh = tessellate_solid(&topo, solid, 0.05).unwrap();
            assert!(is_watertight(&mesh), "{label}: the mesh is open");
            let origin = Point3::new(0.0, 0.0, 0.0);
            let mesh_volume: f64 = mesh
                .indices
                .chunks_exact(3)
                .map(|t| {
                    let [a, b, c] = [t[0], t[1], t[2]].map(|i| mesh.positions[i as usize] - origin);
                    a.dot(b.cross(c)) / 6.0
                })
                .sum();
            assert!(
                (mesh_volume - exact).abs() < 3e-3 * exact,
                "{label}: mesh volume {mesh_volume} against {exact}"
            );
            mesh_volumes.push(mesh_volume);
        }
        let first = mesh_volumes[0];
        assert!(
            mesh_volumes
                .iter()
                .all(|v| (v - first).abs() < 1e-6 * first),
            "{mesh_volumes:?}"
        );
    }
}
