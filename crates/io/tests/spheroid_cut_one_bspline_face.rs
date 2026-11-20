//! A spheroid written as one rational B-spline face (closed round its axis,
//! a pole at each end) and cut by a plane: the face is bounded by the cut's
//! rim and by a vertex loop at a pole, which says which side of the rim the
//! face lies on.
//!
//! Cut beside the axis, the rim does not wind round the axis: the face is
//! the whole surface but the cap inside the rim, both poles in it. It read
//! as the cap alone, the rim its outer loop (the solid's volume wrong, and
//! every boolean with it). It reads as the lunes a whole spheroid reads as,
//! split off the rim, the rim a hole in one of them.
//!
//! Cut across the axis, the rim winds round it: the face is the dome from
//! the rim to the pole, and it reads as that dome (a guard: it did before).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::fmt::Write as _;

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const R: f64 = 5.0;
const H: f64 = 7.5;

/// The surface's control points and weights, as STEP rows: a quadratic
/// semi-ellipse meridian from the south pole to the north, swept round z as
/// a rational quadratic circle. Its knots round z are written periodic, as
/// some writers write a closed surface (`periodic`), or clamped.
fn surface(points: &mut String, periodic: bool) -> String {
    let meridian = [
        (0.0, -H, 1.0),
        (R, -H, FRAC_1_SQRT_2),
        (R, 0.0, 1.0),
        (R, H, FRAC_1_SQRT_2),
        (0.0, H, 1.0),
    ];
    let round = [
        (1.0, 0.0, 1.0),
        (1.0, 1.0, FRAC_1_SQRT_2),
        (0.0, 1.0, 1.0),
        (-1.0, 1.0, FRAC_1_SQRT_2),
        (-1.0, 0.0, 1.0),
        (-1.0, -1.0, FRAC_1_SQRT_2),
        (0.0, -1.0, 1.0),
        (1.0, -1.0, FRAC_1_SQRT_2),
        (1.0, 0.0, 1.0),
    ];
    let mut rows = Vec::new();
    let mut weights = Vec::new();
    let mut id = 100;
    for &(x, z, wm) in &meridian {
        let mut row = Vec::new();
        let mut wrow = Vec::new();
        for &(cx, cy, wc) in &round {
            writeln!(
                points,
                "#{id} = CARTESIAN_POINT('',({:.15},{:.15},{:.15}));",
                x * cx,
                x * cy,
                z
            )
            .unwrap();
            row.push(format!("#{id}"));
            wrow.push(format!("{:.15}", wm * wc));
            id += 1;
        }
        rows.push(format!("({})", row.join(",")));
        weights.push(format!("({})", wrow.join(",")));
    }
    let (mults, knots) = if periodic {
        ("(1,2,2,2,2,2,1)", "(-1.,0.,1.,2.,3.,4.,5.)")
    } else {
        ("(3,2,2,2,3)", "(0.,1.,2.,3.,4.)")
    };
    format!(
        "( BOUNDED_SURFACE() B_SPLINE_SURFACE(2,2,({}),.UNSPECIFIED.,.F.,.T.,.F.) \
         B_SPLINE_SURFACE_WITH_KNOTS((3,2,3),{mults},(0.,1.,2.),{knots},.UNSPECIFIED.) \
         GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE(({})) \
         REPRESENTATION_ITEM('') SURFACE() )",
        rows.join(","),
        weights.join(","),
    )
}

/// The spheroid's part on one side of a plane, as a STEP file: its curved
/// face (the surface turned out, `.F.`, bounded by the rim and a vertex
/// loop at `pole`) and the flat face. The rim is the plane's ellipse about
/// `center`, its first axis `x` (length `a`) and its second `axis` x `x`
/// (length `b`), run about `axis`, which points out of the part.
#[allow(clippy::too_many_arguments)]
fn cut_step(
    periodic: bool,
    pole: Point3,
    center: Point3,
    axis: [f64; 3],
    x: [f64; 3],
    a: f64,
    b: f64,
) -> String {
    let mut points = String::new();
    let surface = surface(&mut points, periodic);
    let n = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    let start = Point3::new(
        center.x() + a * x[0] / n,
        center.y() + a * x[1] / n,
        center.z() + a * x[2] / n,
    );
    let p = |q: Point3| format!("({:.15},{:.15},{:.15})", q.x(), q.y(), q.z());
    let d = |v: [f64; 3]| format!("({:.15},{:.15},{:.15})", v[0], v[1], v[2]);
    let conic = if (a - b).abs() < 1e-12 {
        format!("CIRCLE('',#40,{a:.15})")
    } else {
        format!("ELLIPSE('',#40,{a:.15},{b:.15})")
    };
    format!(
        "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n\
         #1 = MANIFOLD_SOLID_BREP('',#2);\n\
         #2 = CLOSED_SHELL('',(#3,#20));\n\
         #3 = ADVANCED_FACE('',(#4,#8),#7,.F.);\n\
         #4 = FACE_BOUND('',#5,.F.);\n\
         #5 = VERTEX_LOOP('',#6);\n\
         #6 = VERTEX_POINT('',#60);\n\
         #60 = CARTESIAN_POINT('',{});\n\
         #7 = {surface};\n\
         #8 = FACE_BOUND('',#9,.T.);\n\
         #9 = EDGE_LOOP('',(#10));\n\
         #10 = ORIENTED_EDGE('',*,*,#30,.F.);\n\
         #20 = ADVANCED_FACE('',(#21),#24,.T.);\n\
         #21 = FACE_BOUND('',#22,.T.);\n\
         #22 = EDGE_LOOP('',(#23));\n\
         #23 = ORIENTED_EDGE('',*,*,#30,.T.);\n\
         #24 = PLANE('',#40);\n\
         #30 = EDGE_CURVE('',#31,#31,#33,.T.);\n\
         #31 = VERTEX_POINT('',#32);\n\
         #32 = CARTESIAN_POINT('',{});\n\
         #33 = {conic};\n\
         #40 = AXIS2_PLACEMENT_3D('',#41,#42,#43);\n\
         #41 = CARTESIAN_POINT('',{});\n\
         #42 = DIRECTION('',{});\n\
         #43 = DIRECTION('',{});\n\
         {points}ENDSEC;\nEND-ISO-10303-21;\n",
        p(pole),
        p(start),
        p(center),
        d(axis),
        d(x),
    )
}

fn read(step: &str) -> (Topology, SolidId) {
    let mut topo = Topology::new();
    let solids = brepkit_io::step::read_step(step, &mut topo).unwrap();
    assert_eq!(solids.len(), 1);
    (topo, solids[0])
}

/// Checks the solid is valid, meshes closed and has volume `want`.
fn check(topo: &Topology, solid: SolidId, want: f64, tag: &str) {
    let report = brepkit_operations::validate::validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{tag}: {:?}", report.issues);
    let mesh = brepkit_operations::tessellate::tessellate_solid(topo, solid, 0.05).unwrap();
    assert!(
        brepkit_operations::tessellate::is_watertight(&mesh),
        "{tag}: open mesh"
    );
    let volume = brepkit_operations::measure::solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-3 * want,
        "{tag}: volume {volume}, want {want}"
    );
}

/// The spheroid's volume beyond the plane `t` along an equatorial axis or
/// along its own axis (its semi-axis there `s`): the sphere's cap, stretched.
fn cap(s: f64, t: f64) -> f64 {
    let ball = 4.0 / 3.0 * PI * R * R * H;
    let k = t / s;
    // A unit ball's cap beyond k is pi (1 - k)^2 (2 + k) / 3 of its 4 pi / 3.
    ball * (1.0 - k).powi(2) * (2.0 + k) / 4.0
}

#[test]
fn a_spheroid_cut_beside_its_axis_reads_as_all_but_the_cap() {
    let c = 2.0;
    let shrink = (1.0 - (c / R).powi(2)).sqrt();
    let ball = 4.0 / 3.0 * PI * R * R * H;
    for periodic in [false, true] {
        for south in [true, false] {
            let tag = format!("periodic {periodic}, the pole's loop at the south {south}");
            let pole = Point3::new(0.0, 0.0, if south { -H } else { H });
            // The rim y = c, an ellipse x^2 / R^2 + z^2 / H^2 = 1 - c^2 / R^2
            // (its longer axis first), run about +y (out of the part, which
            // lies at y <= c).
            let step = cut_step(
                periodic,
                pole,
                Point3::new(0.0, c, 0.0),
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                H * shrink,
                R * shrink,
            );
            let (topo, solid) = read(&step);
            check(&topo, solid, ball - cap(R, c), &tag);
        }
    }
}

#[test]
fn a_spheroid_cut_across_its_axis_reads_as_its_dome() {
    let c = 3.0;
    let rho = R * (1.0 - (c / H).powi(2)).sqrt();
    for periodic in [false, true] {
        // The rim's vertex on the surface's seam line, or a turn-fraction
        // off it.
        for turn in [0.0_f64, 0.7, 3.5] {
            let tag = format!("periodic {periodic}, the rim's vertex at {turn}");
            // The rim z = c, a circle of radius rho, run about -z (out of the
            // dome, which lies at z >= c).
            let step = cut_step(
                periodic,
                Point3::new(0.0, 0.0, H),
                Point3::new(0.0, 0.0, c),
                [0.0, 0.0, -1.0],
                [turn.cos(), turn.sin(), 0.0],
                rho,
                rho,
            );
            let (topo, solid) = read(&step);
            check(&topo, solid, cap(H, c), &tag);
        }
    }
}
