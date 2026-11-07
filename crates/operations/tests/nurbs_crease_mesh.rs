//! A NURBS face creased along a knot line meshes along the crease, rather than
//! dividing to the depth limit on either side of it.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::mat::Mat4;
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::Point3;
    use brepkit_operations::tessellate::tessellate;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::make_nurbs_face;

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6),
            ),
            (
                "far away",
                Mat4::translation(370.0, 4830.0, -91.0) * Mat4::rotation_y(1.1),
            ),
        ]
    }

    /// A sheet 10 wide, 4.3 long then 5.7 on, bent by `angle` at `knot`.
    fn bent_sheet(pose: Mat4, angle: f64, knot: f64) -> NurbsSurface {
        let (first, second) = (10.0 * knot, 10.0 * (1.0 - knot));
        let rows = [0.0, 10.0]
            .iter()
            .map(|&x| {
                [
                    Point3::new(x, 0.0, 0.0),
                    Point3::new(x, first, 0.0),
                    Point3::new(x, first + second * angle.cos(), second * angle.sin()),
                ]
                .iter()
                .map(|&p| pose.mul_point(p))
                .collect()
            })
            .collect();
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, knot, 1.0, 1.0],
            rows,
            vec![vec![1.0; 3]; 2],
        )
        .unwrap()
    }

    #[test]
    fn a_bent_sheet_meshes_as_two_flats() {
        for (name, pose) in poses() {
            for (angle, knot) in [
                (0.5, 0.43),
                (0.01, 0.43),
                (1.5, 0.3),
                (0.5, 0.25),
                (0.5, 0.26),
            ] {
                let case = format!("{name} {angle} at {knot}");
                let mut topo = Topology::new();
                let face = make_nurbs_face(&mut topo, bent_sheet(pose, angle, knot), 1e-7).unwrap();
                let mesh = tessellate(&topo, face, 0.05).unwrap();
                let triangles = mesh.indices.len() / 3;
                assert!(triangles <= 64, "{case}: {triangles} triangles");

                // Every triangle lies on the sheet: its middle is on one flat or
                // the other, not cutting across the bend.
                let (a, b, c) = (
                    pose.mul_point(Point3::new(0.0, 0.0, 0.0)),
                    pose.mul_point(Point3::new(1.0, 0.0, 0.0)),
                    pose.mul_point(Point3::new(0.0, 1.0, 0.0)),
                );
                let width = (b - a).normalize().unwrap();
                let along = (c - a).normalize().unwrap();
                let up = width.cross(along);
                let first = 10.0 * knot;
                for t in mesh.indices.chunks_exact(3) {
                    let p = |i: u32| mesh.positions[i as usize] - a;
                    let middle = (p(t[0]) + p(t[1]) + p(t[2])) * (1.0 / 3.0);
                    let (y, z) = (middle.dot(along), middle.dot(up));
                    let off = if y <= first {
                        z.abs()
                    } else {
                        ((y - first) * angle.sin() - z * angle.cos()).abs()
                    };
                    assert!(off < 1e-6, "{case}: a triangle {off} off the sheet");
                }
            }
        }
    }

    /// More creased knots than the root grid takes keep the even grid, and a
    /// sheet creased both ways stays within a budget.
    #[test]
    fn a_faceted_sheet_stays_bounded() {
        let facets = 60_usize;
        let knots = |count: usize| {
            let mut knots = vec![0.0, 0.0];
            for k in 1..count {
                #[allow(clippy::cast_precision_loss)]
                knots.push(k as f64 / count as f64);
            }
            knots.extend([1.0, 1.0]);
            knots
        };
        #[allow(clippy::cast_precision_loss)]
        let rows: Vec<Vec<Point3>> = (0..=facets)
            .map(|i| {
                (0..=facets)
                    .map(|j| {
                        let (x, y) = (i as f64, j as f64);
                        let z = f64::from(i32::from(i % 2 == 1)) + f64::from(i32::from(j % 2 == 1));
                        Point3::new(x, y, z)
                    })
                    .collect()
            })
            .collect();
        let weights = vec![vec![1.0; facets + 1]; facets + 1];
        let sheet = NurbsSurface::new(1, 1, knots(facets), knots(facets), rows, weights).unwrap();
        let mut topo = Topology::new();
        let face = make_nurbs_face(&mut topo, sheet, 1e-7).unwrap();
        let started = std::time::Instant::now();
        let mesh = tessellate(&topo, face, 0.05).unwrap();
        assert!(!mesh.indices.is_empty());
        assert!(
            mesh.indices.len() / 3 <= 500_000,
            "{}",
            mesh.indices.len() / 3
        );
        assert!(started.elapsed().as_secs() < 60, "{:?}", started.elapsed());
    }

    /// A repeated knot on a smooth surface is no crease, and adds no cells.
    #[test]
    fn a_repeated_knot_on_a_flat_sheet_meshes_as_a_flat_sheet() {
        let mut topo = Topology::new();
        let flat = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 10.0, 0.0)],
                vec![Point3::new(10.0, 0.0, 0.0), Point3::new(10.0, 10.0, 0.0)],
            ],
            vec![vec![1.0; 2]; 2],
        )
        .unwrap();
        let plain = make_nurbs_face(&mut topo, flat, 1e-7).unwrap();
        let with_knot =
            make_nurbs_face(&mut topo, bent_sheet(Mat4::identity(), 0.0, 0.43), 1e-7).unwrap();
        let (a, b) = (
            tessellate(&topo, plain, 0.05).unwrap(),
            tessellate(&topo, with_knot, 0.05).unwrap(),
        );
        assert_eq!(a.indices.len(), b.indices.len());
    }
}
