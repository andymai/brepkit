//! The brepjs playground example `cube-truss-frame`: the 20 cylinders of its
//! `fuseAll`, captured from the call.
//!
//! Four chords of radius 25 run along `x`, and sixteen struts of radius 8
//! join them: an end frame square to the chords at each end of each side,
//! and two diagonals at 53 degrees to the chords on each side. Each strut
//! ends on the axes of two chords, and two to four struts end at one point
//! there.
//!
//! The chords' seams face along `y`, so a strut that comes in along `y`
//! enters a chord across the chord's seam, and the second such strut on a
//! chord met a wall that the first had notched at its seam. Two struts that
//! start at one point have caps whose rims touch the other strut's wall at
//! two points.
//!
//! Data: `cube_truss_struts.bin`, the 20 operands in `fuseAll` order, each
//! as a little-endian `u32` byte count and then the solid's arena bytes.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::compound_ops::fuse_all;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::compound::Compound;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_faces, solid_vertices};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, pose: &Mat4) -> Vec<SolidId> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/cube_truss_struts.bin");
    let bytes = std::fs::read(path).unwrap();
    let mut solids = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let solid = deserialize_solid(&bytes[at + 4..at + 4 + len], topo).unwrap();
        transform_solid(topo, solid, pose).unwrap();
        solids.push(solid);
        at += 4 + len;
    }
    assert_eq!(solids.len(), 20);
    solids
}

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

fn rod(topo: &Topology, solid: SolidId) -> Rod {
    let cylinder = solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .find_map(|f| match topo.face(f).unwrap().surface() {
            FaceSurface::Cylinder(c) => Some(c.clone()),
            _ => None,
        })
        .unwrap();
    let o = cylinder.origin();
    let a = cylinder.axis();
    let (o, a) = ([o.x(), o.y(), o.z()], [a.x(), a.y(), a.z()]);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in solid_vertices(topo, solid).unwrap() {
        let p = topo.vertex(v).unwrap().point();
        let t = dot(sub([p.x(), p.y(), p.z()], o), a);
        lo = lo.min(t);
        hi = hi.max(t);
    }
    Rod {
        start: [o[0] + a[0] * lo, o[1] + a[1] * lo, o[2] + a[2] * lo],
        axis: a,
        radius: cylinder.radius(),
        length: hi - lo,
    }
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

/// The volume of the union of the rods: each rod less the rods before it,
/// integrated over its cross-section along lines parallel to its axis, each
/// line cut exactly by the other rods.
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
        let k = dot(helper, a);
        let e1 = sub(helper, [a[0] * k, a[1] * k, a[2] * k]);
        let n = dot(e1, e1).sqrt();
        let e1 = [e1[0] / n, e1[1] / n, e1[2] / n];
        let e2 = [
            a[1] * e1[2] - a[2] * e1[1],
            a[2] * e1[0] - a[0] * e1[2],
            a[0] * e1[1] - a[1] * e1[0],
        ];
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
                total += 0.5 * (r1 - r0) * g.w * r * free * std::f64::consts::TAU / SPOKES as f64;
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

fn edges_not_used_twice(topo: &Topology, solid: SolidId) -> usize {
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for f in solid_faces(topo, solid).unwrap() {
        let face = topo.face(f).unwrap();
        for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wire).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    uses.values().filter(|&&n| n != 2).count()
}

fn check(topo: &Topology, fused: SolidId, rods: &[Rod], cylinders: usize, name: &str) {
    let report = validate_solid(topo, fused).unwrap();
    assert!(report.is_valid(), "{name}: {:?}", report.issues);
    assert_eq!(edges_not_used_twice(topo, fused), 0, "{name}");
    let shape = census(topo, fused);
    assert_eq!(shape.get("cylinder"), Some(&cylinders), "{name}: {shape:?}");
    assert_eq!(shape.len(), 2, "{name}: {shape:?}");
    let expected = union_volume(rods);
    let volume = solid_volume(topo, fused, 0.005).unwrap();
    assert!(
        (volume - expected).abs() < 1e-6 * expected,
        "{name}: volume {volume} should be {expected}"
    );
}

fn poses() -> [(&'static str, Mat4); 3] {
    [
        ("as captured", Mat4::identity()),
        (
            "turned",
            Mat4::translation(-310.0, 75.0, 940.0)
                * Mat4::rotation_x(0.41)
                * Mat4::rotation_z(1.13),
        ),
        ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
    ]
}

#[test]
fn the_truss_fuses_into_one_exact_solid() {
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let solids = load(&mut topo, &pose);
        let rods: Vec<Rod> = solids.iter().map(|&s| rod(&topo, s)).collect();
        let compound = topo.add_compound(Compound::new(solids));
        let fused = fuse_all(&mut topo, compound).unwrap();
        check(&topo, fused, &rods, 20, name);
    }
}

#[test]
fn struts_that_start_at_one_point_fuse() {
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let solids = load(&mut topo, &pose);
        let (a, b) = (solids[4], solids[6]);
        let rods = [rod(&topo, a), rod(&topo, b)];
        let fused = brepkit_algo::gfa::fuse_n(&mut topo, &[a, b]).unwrap();
        check(&topo, fused, &rods, 2, name);
        let options = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let pair = boolean_with_options(&mut topo, BooleanOp::Fuse, a, b, options).unwrap();
        check(&topo, pair, &rods, 2, &format!("{name}, pairwise"));
    }
}

/// Chord 1 takes struts 5 and 7 one at a time, in either order. The struts
/// meet off the chord, and strut 7 leaves the chord in a closed curve whose
/// pcurve on the strut's wall must run one way round from the wall's seam.
#[test]
fn a_chord_takes_two_struts_that_meet_one_at_a_time() {
    let options = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    for (name, pose) in poses() {
        for order in [[1, 7, 5], [1, 5, 7]] {
            let mut topo = Topology::new();
            let solids = load(&mut topo, &pose);
            let [a, b, c] = order.map(|i| solids[i]);
            let rods = [rod(&topo, a), rod(&topo, b), rod(&topo, c)];
            let first = boolean_with_options(&mut topo, BooleanOp::Fuse, a, b, options).unwrap();
            let both = boolean_with_options(&mut topo, BooleanOp::Fuse, first, c, options).unwrap();
            check(&topo, both, &rods, 3, &format!("{name}, {order:?}"));
        }
    }
}

/// Two struts that meet end to start, or start at one point, on chord 1's
/// axis but not in one plane: struts 6 and 11, 7 and 10, and 7 and 11.
/// Each pair fuses alone, pairwise and through `fuse_n`, and so do chord 1
/// and struts 7 and 11.
#[test]
fn struts_that_meet_on_a_chord_axis_out_of_plane_fuse() {
    let options = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    for (name, pose) in poses() {
        for [i, j] in [[6, 11], [7, 10], [7, 11]] {
            let mut topo = Topology::new();
            let solids = load(&mut topo, &pose);
            let (a, b) = (solids[i], solids[j]);
            let rods = [rod(&topo, a), rod(&topo, b)];
            let label = format!("{name}, struts {i} and {j}");
            let fused = brepkit_algo::gfa::fuse_n(&mut topo, &[a, b]).unwrap();
            check(&topo, fused, &rods, 2, &label);
            let pair = boolean_with_options(&mut topo, BooleanOp::Fuse, a, b, options).unwrap();
            check(&topo, pair, &rods, 2, &format!("{label}, pairwise"));
        }
        let mut topo = Topology::new();
        let solids = load(&mut topo, &pose);
        let three = [solids[1], solids[7], solids[11]];
        let rods = three.map(|s| rod(&topo, s));
        let fused = brepkit_algo::gfa::fuse_n(&mut topo, &three).unwrap();
        check(
            &topo,
            fused,
            &rods,
            3,
            &format!("{name}, chord 1, struts 7 and 11"),
        );
    }
}

/// Three or four struts that end at one corner of the truss, with no chord
/// round them: struts 4, 6, 16 and 18 at `(70, 120, 120)`. Each two of their
/// walls meet in a curve, and each three walls meet at one point inside the
/// three faces, where the three curves cross.
#[test]
fn struts_that_end_at_one_point_fuse_in_one_pass() {
    let sets: [&[usize]; 5] = [
        &[4, 6, 16],
        &[4, 16, 18],
        &[4, 6, 18],
        &[6, 16, 18],
        &[4, 6, 16, 18],
    ];
    let mut volumes: Vec<Vec<f64>> = Vec::new();
    for (name, pose) in poses() {
        let mut row = Vec::new();
        for set in sets {
            let mut topo = Topology::new();
            let solids = load(&mut topo, &pose);
            let struts: Vec<SolidId> = set.iter().map(|&i| solids[i]).collect();
            let rods: Vec<Rod> = struts.iter().map(|&s| rod(&topo, s)).collect();
            let label = format!("{name}, struts {set:?}");
            let fused = brepkit_algo::gfa::fuse_n(&mut topo, &struts).unwrap();
            let report = validate_solid(&topo, fused).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            assert_eq!(edges_not_used_twice(&topo, fused), 0, "{label}");
            let shape = census(&topo, fused);
            assert_eq!(
                shape.get("cylinder"),
                Some(&set.len()),
                "{label}: {shape:?}"
            );
            assert_eq!(shape.len(), 2, "{label}: {shape:?}");
            // The line integral reads these few rods to about 1e-5 of their
            // volume, and differently for each order of the rods.
            let expected = union_volume(&rods);
            let volume = solid_volume(&topo, fused, 0.005).unwrap();
            assert!(
                (volume - expected).abs() < 2e-5 * expected,
                "{label}: volume {volume} should be {expected}"
            );
            row.push(volume);
        }
        volumes.push(row);
    }
    // Every pose holds the same solid, and struts 6 and 18 are mirror
    // images in the plane y = z through the corner, as are 4 and 16.
    let same = |a: f64, b: f64| (a - b).abs() < 1e-9 * a;
    for row in &volumes {
        assert!(same(row[0], row[1]) && same(row[2], row[3]), "{row:?}");
        for (v, w) in row.iter().zip(&volumes[0]) {
            assert!(same(*v, *w), "{volumes:?}");
        }
    }
}
