//! An `EDGE_CURVE` whose `same_sense` is `.F.` runs from its start vertex to
//! its end against its curve's own direction. Another writer may describe a
//! brepkit arc that way: the circle's axis flipped and the flag false.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use brepkit_io::step::reader::read_step;
use brepkit_io::step::writer::write_step;
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::solid_edges;

/// `#id = TYPE(attrs);` lines of a STEP DATA section, by id.
fn entities(step: &str) -> BTreeMap<u64, (String, String)> {
    step.lines()
        .filter_map(|line| {
            let (id, rest) = line.trim().strip_prefix('#')?.split_once('=')?;
            let (kind, attrs) = rest.trim().split_once('(')?;
            let attrs = attrs.trim_end().strip_suffix(");")?;
            Some((
                id.trim().parse().ok()?,
                (kind.trim().to_string(), attrs.to_string()),
            ))
        })
        .collect()
}

fn refs(attrs: &str) -> Vec<u64> {
    attrs
        .split(|c: char| !c.is_ascii_digit() && c != '#')
        .filter_map(|t| t.strip_prefix('#')?.parse().ok())
        .collect()
}

/// Rewrite every open circular `EDGE_CURVE` as its flipped-axis, `.F.` twin,
/// adding the new direction, placement and circle entities.
fn flip_arcs(step: &str) -> (String, usize) {
    let table = entities(step);
    let mut next = table.keys().max().copied().unwrap_or(0) + 1;
    let mut added = Vec::new();
    let mut rewritten = BTreeMap::new();
    for (&id, (kind, attrs)) in &table {
        if kind != "EDGE_CURVE" {
            continue;
        }
        let r = refs(attrs);
        let (start, end, curve) = (r[0], r[1], r[2]);
        let (curve_kind, curve_attrs) = &table[&curve];
        if curve_kind != "CIRCLE" || start == end {
            continue;
        }
        let placement = refs(curve_attrs)[0];
        let radius = curve_attrs.rsplit(',').next().unwrap().trim();
        let p = refs(&table[&placement].1);
        let (location, axis, reference) = (p[0], p[1], p[2]);
        let direction = &table[&axis].1;
        let inner = direction
            .split_once('(')
            .unwrap()
            .1
            .trim_end_matches(')')
            .split(',')
            .map(|c| -c.trim().parse::<f64>().unwrap())
            .map(|c| format!("{c:.17}"))
            .collect::<Vec<_>>()
            .join(", ");
        let (d, a, c) = (next, next + 1, next + 2);
        next += 3;
        added.push(format!("#{d} = DIRECTION('', ({inner}));"));
        added.push(format!(
            "#{a} = AXIS2_PLACEMENT_3D('', #{location}, #{d}, #{reference});"
        ));
        added.push(format!("#{c} = CIRCLE('', #{a}, {radius});"));
        rewritten.insert(
            id,
            format!("#{id} = EDGE_CURVE('', #{start}, #{end}, #{c}, .F.);"),
        );
    }
    let count = rewritten.len();
    let mut out = String::new();
    for line in step.lines() {
        let id = line
            .trim()
            .strip_prefix('#')
            .and_then(|l| l.split_once('='))
            .and_then(|(id, _)| id.trim().parse::<u64>().ok());
        if let Some(new) = id.and_then(|id| rewritten.get(&id)) {
            out.push_str(new);
        } else if line.trim() == "ENDSEC;" && !added.is_empty() && out.contains("DATA;") {
            for extra in std::mem::take(&mut added) {
                out.push_str(&extra);
                out.push('\n');
            }
            out.push_str(line);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    (out, count)
}

#[test]
fn a_false_same_sense_arc_keeps_its_side() {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, 2.0, 3.0).unwrap();
    let half = make_box(&mut topo, 5.0, 10.0, 5.0).unwrap();
    transform_solid(&mut topo, half, &Mat4::translation(-5.0, -5.0, -1.0)).unwrap();
    let kept = boolean(&mut topo, BooleanOp::Cut, rod, half).unwrap();
    let step = write_step(&topo, &[kept]).unwrap();
    let (flipped, arcs) = flip_arcs(&step);
    assert_eq!(arcs, 4, "the half rod's rim arcs, split at its seam");
    assert!(flipped.contains(".F.);"));

    let mut read = Topology::new();
    let solid = read_step(&flipped, &mut read).unwrap()[0];
    let report = validate_solid(&read, solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    for edge in solid_edges(&read, solid).unwrap() {
        let edge = read.edge(edge).unwrap();
        if let EdgeCurve::Circle(_) = edge.curve() {
            let (start, end) = (
                read.vertex(edge.start()).unwrap().point(),
                read.vertex(edge.end()).unwrap().point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
            let mid = edge
                .curve()
                .evaluate_with_endpoints(0.5 * (t0 + t1), start, end);
            assert!(mid.x() > 1.0, "arc midpoint {mid:?} is off the kept side");
        }
    }
    let volume = solid_volume(&read, solid, 0.001).unwrap();
    let mut plain = Topology::new();
    let plain_solid = read_step(&step, &mut plain).unwrap()[0];
    let truth = solid_volume(&plain, plain_solid, 0.001).unwrap();
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "volume {volume}, as written {truth}"
    );
}
