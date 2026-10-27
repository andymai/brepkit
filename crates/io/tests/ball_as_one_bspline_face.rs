//! A ball written as one rational B-spline face that closes round the axis
//! and collapses to a pole at each end, bounded by nothing but a vertex loop
//! (as some writers write a sphere or a revolved lemon). It has no edge to
//! bound it, and read "ADVANCED_FACE has no bounds"; it reads as a closed
//! solid of lunes between meridians, at the sphere's volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::fmt::Write as _;

/// The STEP text of a ball of radius `r` about the origin, stretched by
/// `stretch` along z: a quadratic semicircle meridian from the south pole to
/// the north, swept round z as a rational quadratic circle of eight spans'
/// worth of points, its knots round z spaced `v_scale` apart.
fn ball_step(r: f64, stretch: f64, v_scale: f64) -> String {
    let h = r * stretch;
    let meridian = [
        (0.0, -h, 1.0),
        (r, -h, FRAC_1_SQRT_2),
        (r, 0.0, 1.0),
        (r, h, FRAC_1_SQRT_2),
        (0.0, h, 1.0),
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
    let mut points = String::new();
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
    format!(
        "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n\
         #1 = MANIFOLD_SOLID_BREP('',#2);\n\
         #2 = CLOSED_SHELL('',(#3));\n\
         #3 = ADVANCED_FACE('',(#4),#7,.F.);\n\
         #4 = FACE_BOUND('',#5,.F.);\n\
         #5 = VERTEX_LOOP('',#6);\n\
         #6 = VERTEX_POINT('',#100);\n\
         #7 = ( BOUNDED_SURFACE() B_SPLINE_SURFACE(2,2,({}),.UNSPECIFIED.,.F.,.T.,.F.) \
         B_SPLINE_SURFACE_WITH_KNOTS((3,2,3),(3,2,2,2,3),(0.,1.,2.),({}),.UNSPECIFIED.) \
         GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE(({})) REPRESENTATION_ITEM('') SURFACE() );\n\
         {points}ENDSEC;\nEND-ISO-10303-21;\n",
        rows.join(","),
        [0.0, 1.0, 2.0, 3.0, 4.0]
            .map(|k: f64| format!("{:.17E}", k * v_scale))
            .join(","),
        weights.join(","),
    )
}

/// Reads the ball, stretched by `stretch` with knots round z `v_scale`
/// apart, and checks it is one valid solid at its closed-form volume.
fn reads_closed(stretch: f64, v_scale: f64) {
    let r = 5.0;
    let mut topo = brepkit_topology::Topology::new();
    let solids = brepkit_io::step::read_step(&ball_step(r, stretch, v_scale), &mut topo).unwrap();
    assert_eq!(solids.len(), 1);
    let report = brepkit_operations::validate::validate_solid(&topo, solids[0]).unwrap();
    assert!(report.is_valid(), "stretch {stretch}: {:?}", report.issues);
    // Its mesh closes: every mesh edge, welded, is shared by two triangles.
    let (mesh, _) = brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance(
        &topo, solids[0], 0.05, 0.5,
    )
    .unwrap();
    #[allow(clippy::cast_possible_truncation)]
    let key = |p: brepkit_math::vec::Point3| {
        (
            (p.x() * 1e5).round() as i64,
            (p.y() * 1e5).round() as i64,
            (p.z() * 1e5).round() as i64,
        )
    };
    let mut weld = std::collections::HashMap::new();
    let ids: Vec<usize> = mesh
        .positions
        .iter()
        .map(|&p| {
            let n = weld.len();
            *weld.entry(key(p)).or_insert(n)
        })
        .collect();
    let mut uses = std::collections::HashMap::new();
    for t in mesh.indices.chunks_exact(3) {
        for (i, j) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let (a, b) = (ids[i as usize], ids[j as usize]);
            if a != b {
                *uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
    }
    let open = uses.values().filter(|&&n| n == 1).count();
    assert_eq!(open, 0, "stretch {stretch}: {open} open mesh edges");
    let want = 4.0 / 3.0 * PI * r * r * r * stretch;
    let volume = brepkit_operations::measure::solid_volume(&topo, solids[0], 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-3 * want,
        "stretch {stretch}: volume {volume}, want {want}"
    );
}

/// A true sphere reads as the sphere it is.
#[test]
fn a_ball_written_as_one_bspline_face_reads_closed() {
    reads_closed(1.0, 1.0);
}

/// A spheroid is no sphere: it reads as lunes between its meridians.
#[test]
fn a_spheroid_written_as_one_bspline_face_reads_as_lunes() {
    reads_closed(1.5, 1.0);
}

/// Knots 1e-13 apart are still distinct knots: the lunes split at each,
/// and the solid meshes closed.
#[test]
fn a_spheroid_on_a_tiny_parameter_domain_reads_as_lunes() {
    reads_closed(1.5, 1e-13);
    let mut topo = brepkit_topology::Topology::new();
    let solids = brepkit_io::step::read_step(&ball_step(5.0, 1.5, 1e-13), &mut topo).unwrap();
    let faces = brepkit_topology::explorer::solid_faces(&topo, solids[0]).unwrap();
    assert_eq!(faces.len(), 4, "one lune per quarter turn");
}
