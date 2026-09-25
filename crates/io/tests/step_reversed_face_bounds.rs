//! A reversed face's bounds in STEP run about the face's normal (ISO
//! 10303-42), while brepkit stores every loop about the surface's normal. A
//! box dimpled by a ball keeps its dimple through a brepkit round trip, when
//! written the way other systems write it (bounds flagged true, loops turned),
//! and when read from an export made before brepkit wrote the standard's
//! bounds.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::Write as _;

use brepkit_io::step::reader::read_step;
use brepkit_io::step::writer::write_step;
use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

/// The box `[-5, 5]³` less a ball of radius 2 at `(0, 0, 5.5)`, as STEP.
fn dimpled_box_step() -> String {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(-5.0, -5.0, -5.0)).unwrap();
    let ball = make_sphere(&mut topo, 2.0, 32).unwrap();
    transform_solid(&mut topo, ball, &Mat4::translation(0.0, 0.0, 5.5)).unwrap();
    let piece = boolean(&mut topo, BooleanOp::Cut, block, ball).unwrap();
    write_step(&topo, &[piece]).unwrap()
}

/// Reads `step` and checks the dimpled box: exact volume, a watertight mesh
/// of the same volume, and a dimple meshed below the box's top.
fn assert_dimpled_box(step: &str, label: &str) {
    let (r, h) = (2.0_f64, 1.5_f64);
    let truth = 1000.0 - PI * h * h * (3.0 * r - h) / 3.0;
    let mut topo = Topology::new();
    let solids = read_step(step, &mut topo).unwrap();
    assert_eq!(solids.len(), 1, "{label}: one solid");
    let piece = solids[0];
    let volume = solid_volume(&topo, piece, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "{label}: volume {volume}, truth {truth}"
    );
    let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
    let meshed = oriented_solid_volume(&topo, piece, 0.005).unwrap();
    assert!(
        (meshed - truth).abs() < 1e-3 * truth,
        "{label}: mesh volume {meshed}, truth {truth}"
    );
    let dimples: Vec<_> = solid_faces(&topo, piece)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
        .collect();
    assert_eq!(dimples.len(), 1, "{label}: one sphere face");
    let dimple = tessellate(&topo, dimples[0], 0.01).unwrap();
    assert!(
        dimple.positions.iter().all(|p| p.z() < 5.0 + 1e-6),
        "{label}: the dimple meshed above the box's top"
    );
}

/// Each `#id = TYPE(attrs);` line of the DATA section, by id.
fn entities(step: &str) -> HashMap<u64, (String, String)> {
    step.lines()
        .filter_map(|line| {
            let (id, rest) = line.strip_prefix('#')?.split_once(" = ")?;
            let (ty, attrs) = rest.split_once('(')?;
            Some((id.parse().ok()?, (ty.to_string(), attrs.to_string())))
        })
        .collect()
}

fn refs(attrs: &str) -> Vec<u64> {
    attrs
        .split('#')
        .skip(1)
        .filter_map(|s| {
            s.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse()
                .ok()
        })
        .collect()
}

#[test]
fn reversed_face_bounds_round_trip() {
    let step = dimpled_box_step();
    assert!(
        step.lines()
            .any(|l| l.contains("FACE_OUTER_BOUND(") && l.ends_with(".F.);")),
        "a reversed face writes its bound flagged false"
    );
    assert_dimpled_box(&step, "round trip");
}

#[test]
fn reversed_face_bounds_as_other_systems_write_them() {
    let step = dimpled_box_step();
    let ents = entities(&step);
    let mut out = String::new();
    for line in step.lines() {
        let Some((id, (ty, attrs))) = line
            .strip_prefix('#')
            .and_then(|l| l.split_once(" = "))
            .and_then(|(id, _)| id.parse::<u64>().ok())
            .and_then(|id| ents.get(&id).map(|e| (id, e)))
        else {
            out.push_str(&line.replace(", 'face bounds per ISO 10303-42'", ""));
            out.push('\n');
            continue;
        };
        // A bound flagged false: flag it true and turn its loop.
        let turned_loop = ents.iter().find_map(|(&bid, (bty, battrs))| {
            (bty.contains("BOUND") && battrs.ends_with(".F.);") && refs(battrs)[0] == id)
                .then_some(bid)
        });
        if ty.contains("BOUND") && attrs.ends_with(".F.);") {
            out.push_str(&line.replace(".F.);", ".T.);"));
        } else if ty == "EDGE_LOOP" && turned_loop.is_some() {
            let mut edges = refs(attrs);
            edges.reverse();
            let list: Vec<String> = edges.iter().map(|e| format!("#{e}")).collect();
            let _ = write!(out, "#{id} = EDGE_LOOP('', ({}));", list.join(", "));
        } else if ty == "ORIENTED_EDGE"
            && ents.iter().any(|(&lid, (lty, lattrs))| {
                lty == "EDGE_LOOP"
                    && refs(lattrs).contains(&id)
                    && ents.values().any(|(bty, battrs)| {
                        bty.contains("BOUND") && battrs.ends_with(".F.);") && refs(battrs)[0] == lid
                    })
            })
        {
            let flipped = if attrs.ends_with(".T.);") {
                line.replace(".T.);", ".F.);")
            } else {
                line.replace(".F.);", ".T.);")
            };
            out.push_str(&flipped);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    let out = out.replace("brepkit", "other");
    assert!(
        !out.lines()
            .any(|l| l.contains("BOUND(") && l.ends_with(".F.);")),
        "every bound flagged true"
    );
    assert_dimpled_box(&out, "other system");
}

#[test]
fn reversed_face_bounds_from_an_earlier_export() {
    let step = dimpled_box_step()
        .replace(", 'face bounds per ISO 10303-42'", "")
        .lines()
        .map(|l| {
            if l.contains("BOUND(") {
                l.replace(".F.);", ".T.);")
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(step.contains("FILE_DESCRIPTION(('brepkit STEP export'), '2;1');"));
    assert_dimpled_box(&step, "earlier export");
}
