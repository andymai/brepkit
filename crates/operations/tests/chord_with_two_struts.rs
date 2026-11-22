//! A chord of radius 25 takes two struts of radius 8 one at a time, as in
//! the brepjs playground example `cube-truss-frame`. Strut 5 runs square to
//! the chord and strut 7 at 53 degrees to it. Each strut ends on the chord's
//! axis, and the two struts meet at one point away from the chord.
//!
//! Each strut leaves the chord in a closed curve that bounds the strut's
//! wall. The cases turn the seams of the chord and the struts to where the
//! fuses meet them: the chord's seam under both struts, and a strut's seam
//! on the side of the other strut.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_cylinder;
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::solid::SolidId;

    const CHORD: ([f64; 3], [f64; 3]) = ([55.0, -120.0, 120.0], [445.0, -120.0, 120.0]);
    const STRUT_5: ([f64; 3], [f64; 3]) = ([430.0, 120.0, 120.0], [430.0, -120.0, 120.0]);
    const STRUT_7: ([f64; 3], [f64; 3]) = ([250.0, -120.0, 120.0], [430.0, 120.0, 120.0]);

    /// A finite cylinder: its start, unit axis, radius and length.
    #[derive(Clone, Copy)]
    struct Rod {
        start: [f64; 3],
        axis: [f64; 3],
        radius: f64,
        length: f64,
    }

    fn dot(p: [f64; 3], q: [f64; 3]) -> f64 {
        p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
    }

    fn sub(p: [f64; 3], q: [f64; 3]) -> [f64; 3] {
        [p[0] - q[0], p[1] - q[1], p[2] - q[2]]
    }

    fn cross(p: [f64; 3], q: [f64; 3]) -> [f64; 3] {
        [
            p[1] * q[2] - p[2] * q[1],
            p[2] * q[0] - p[0] * q[2],
            p[0] * q[1] - p[1] * q[0],
        ]
    }

    fn unit(p: [f64; 3]) -> [f64; 3] {
        let n = dot(p, p).sqrt();
        [p[0] / n, p[1] / n, p[2] / n]
    }

    fn posed(pose: &Mat4, p: [f64; 3]) -> [f64; 3] {
        let q = pose.mul_point(Point3::new(p[0], p[1], p[2]));
        [q.x(), q.y(), q.z()]
    }

    /// A cylinder from `from` to `to` with its seam on the side `seam`
    /// points to, then posed.
    fn rod(
        topo: &mut Topology,
        pose: &Mat4,
        radius: f64,
        (from, to): ([f64; 3], [f64; 3]),
        seam: [f64; 3],
    ) -> (SolidId, Rod) {
        let d = sub(to, from);
        let length = dot(d, d).sqrt();
        let a = unit(d);
        let x = unit(sub(
            seam,
            [
                a[0] * dot(seam, a),
                a[1] * dot(seam, a),
                a[2] * dot(seam, a),
            ],
        ));
        let y = cross(a, x);
        let place = Mat4([
            [x[0], y[0], a[0], from[0]],
            [x[1], y[1], a[1], from[1]],
            [x[2], y[2], a[2], from[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let solid = make_cylinder(topo, radius, length).unwrap();
        transform_solid(topo, solid, &(*pose * place)).unwrap();
        let (start, end) = (posed(pose, from), posed(pose, to));
        let rod = Rod {
            start,
            axis: unit(sub(end, start)),
            radius,
            length,
        };
        (solid, rod)
    }

    /// The parameters of the line `p + t d` inside the rod.
    fn line_in_rod(rod: &Rod, p: [f64; 3], d: [f64; 3]) -> Option<(f64, f64)> {
        let a = rod.axis;
        let w = sub(p, rod.start);
        let (wa, da) = (dot(w, a), dot(d, a));
        let wp = [w[0] - a[0] * wa, w[1] - a[1] * wa, w[2] - a[2] * wa];
        let dp = [d[0] - a[0] * da, d[1] - a[1] * da, d[2] - a[2] * da];
        let (qa, qb, qc) = (
            dot(dp, dp),
            2.0 * dot(wp, dp),
            dot(wp, wp) - rod.radius * rod.radius,
        );
        let (mut lo, mut hi) = if qa < 1e-14 {
            if qc > 0.0 {
                return None;
            }
            (f64::NEG_INFINITY, f64::INFINITY)
        } else {
            let disc = qb * qb - 4.0 * qa * qc;
            if disc <= 0.0 {
                return None;
            }
            (
                (-qb - disc.sqrt()) / (2.0 * qa),
                (-qb + disc.sqrt()) / (2.0 * qa),
            )
        };
        if da.abs() < 1e-14 {
            if wa < 0.0 || wa > rod.length {
                return None;
            }
        } else {
            let (t0, t1) = (-wa / da, (rod.length - wa) / da);
            lo = lo.max(t0.min(t1));
            hi = hi.min(t0.max(t1));
        }
        (lo < hi).then_some((lo, hi))
    }

    /// The volume of the union of the rods: each rod less the rods before
    /// it, integrated over its cross-section along lines parallel to its
    /// axis, each line cut exactly by the other rods.
    fn union_volume(rods: &[Rod]) -> f64 {
        const RINGS: usize = 8;
        const SPOKES: usize = 512;
        let gauss = brepkit_math::quadrature::gauss_legendre_points(20);
        let mut total = 0.0;
        for (i, rod) in rods.iter().enumerate() {
            let a = rod.axis;
            let helper = if a[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let e1 = unit(cross(cross(a, helper), a));
            let e2 = cross(a, e1);
            for ring in 0..RINGS {
                let r0 = rod.radius * ring as f64 / RINGS as f64;
                let r1 = rod.radius * (ring + 1) as f64 / RINGS as f64;
                for g in gauss {
                    let r = 0.5 * (r0 + r1) + 0.5 * (r1 - r0) * g.x;
                    let mut free = 0.0;
                    for spoke in 0..SPOKES {
                        let phi = std::f64::consts::TAU * (spoke as f64 + 0.5) / SPOKES as f64;
                        let (c, s) = (phi.cos(), phi.sin());
                        let p = [
                            rod.start[0] + r * (c * e1[0] + s * e2[0]),
                            rod.start[1] + r * (c * e1[1] + s * e2[1]),
                            rod.start[2] + r * (c * e1[2] + s * e2[2]),
                        ];
                        let mut cuts: Vec<(f64, f64)> = rods[..i]
                            .iter()
                            .filter_map(|other| line_in_rod(other, p, a))
                            .map(|(lo, hi)| (lo.max(0.0), hi.min(rod.length)))
                            .filter(|(lo, hi)| lo < hi)
                            .collect();
                        cuts.sort_by(|x, y| x.0.total_cmp(&y.0));
                        let (mut covered, mut end) = (0.0, f64::NEG_INFINITY);
                        for (lo, hi) in cuts {
                            let lo = lo.max(end);
                            if hi > lo {
                                covered += hi - lo;
                                end = hi;
                            }
                        }
                        free += rod.length - covered;
                    }
                    total +=
                        0.5 * (r1 - r0) * g.w * r * free * std::f64::consts::TAU / SPOKES as f64;
                }
            }
        }
        total
    }

    fn census(topo: &Topology, solid: SolidId) -> BTreeMap<&'static str, usize> {
        let mut m = BTreeMap::new();
        for f in solid_faces(topo, solid).unwrap() {
            *m.entry(topo.face(f).unwrap().surface().type_tag())
                .or_default() += 1;
        }
        m
    }

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(-310.0, 75.0, 940.0)
                    * Mat4::rotation_x(0.41)
                    * Mat4::rotation_z(1.13),
            ),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
        ]
    }

    /// Fuse the chord with the struts one at a time, in both orders and in
    /// each pose, and check each union: valid (each edge on two faces that
    /// run it opposite ways), three walls and four caps, the volume within
    /// 1e-6 of a line integral, and the class of points about the joints.
    fn fuse_in_both_orders(seams: [[f64; 3]; 3], case: &str) {
        let options = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        for (name, pose) in poses() {
            for strut_5_first in [false, true] {
                let mut topo = Topology::new();
                let (chord, chord_rod) = rod(&mut topo, &pose, 25.0, CHORD, seams[0]);
                let (s5, s5_rod) = rod(&mut topo, &pose, 8.0, STRUT_5, seams[1]);
                let (s7, s7_rod) = rod(&mut topo, &pose, 8.0, STRUT_7, seams[2]);
                let (a, b) = if strut_5_first { (s5, s7) } else { (s7, s5) };
                let label = format!(
                    "{case}, {name}, strut {} first",
                    if strut_5_first { 5 } else { 7 }
                );
                let first =
                    boolean_with_options(&mut topo, BooleanOp::Fuse, chord, a, options).unwrap();
                let fused = boolean_with_options(&mut topo, BooleanOp::Fuse, first, b, options)
                    .expect(&label);

                let report = validate_solid(&topo, fused).unwrap();
                assert!(report.is_valid(), "{label}: {:?}", report.issues);
                assert_eq!(
                    census(&topo, fused),
                    BTreeMap::from([("cylinder", 3), ("plane", 4)]),
                    "{label}"
                );
                let expected = union_volume(&[chord_rod, s5_rod, s7_rod]);
                let volume = solid_volume(&topo, fused, 0.005).unwrap();
                assert!(
                    (volume - expected).abs() < 1e-6 * expected,
                    "{label}: volume {volume} should be {expected}"
                );
                for (p, want) in [
                    ([300.0, -120.0, 120.0], PointClassification::Inside),
                    ([430.0, -100.0, 125.0], PointClassification::Inside),
                    ([430.0, 0.0, 120.0], PointClassification::Inside),
                    ([430.0, 118.0, 120.0], PointClassification::Inside),
                    ([340.0, 0.0, 126.0], PointClassification::Inside),
                    ([400.0, -60.0, 120.0], PointClassification::Outside),
                    ([430.0, -90.0, 130.0], PointClassification::Outside),
                    ([440.0, 125.0, 120.0], PointClassification::Outside),
                ] {
                    let q = posed(&pose, p);
                    let got =
                        classify_point(&topo, fused, Point3::new(q[0], q[1], q[2]), 0.01, 1e-7)
                            .unwrap();
                    assert_eq!(got, want, "{label}: point {p:?}");
                }
            }
        }
    }

    /// The chord's seam runs along its side that faces the struts, under
    /// both of them: strut 5's mouth on the chord has its ends on the seam
    /// line, and so does the section across it.
    #[test]
    fn a_chord_takes_two_struts_across_its_seam() {
        fuse_in_both_orders(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.48, -0.36, 0.8]],
            "seam under both struts",
        );
    }

    /// Strut 5's seam faces strut 7, and the curve where strut 5 leaves the
    /// chord runs against its own curve in the strut's wall.
    #[test]
    fn a_strut_whose_seam_faces_the_other_strut_keeps_its_mouth() {
        fuse_in_both_orders(
            [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.48, -0.36, 0.8]],
            "strut 5's seam toward strut 7",
        );
    }
}
