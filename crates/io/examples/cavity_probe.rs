//! Per-shell signed volume of a captured solid.
//!
//! A whole-solid volume cannot distinguish "the cavity is missing from the
//! measurement" from "the cavity is in the measurement but encloses nothing".
//! Both read high by the same amount. This prints each shell separately with
//! its face count, bounding box and signed divergence volume, which separates
//! them: compare a shell's signed volume against the volume of its own bbox.
//!
//! ```sh
//! cargo run --release --example cavity_probe -p brepkit-io -- <name.bin> ...
//! ```
//!
//! Names resolve against `crates/io/tests/data`; absolute paths are used as-is.
//! With no arguments it reads the #1536 slotted no-lip operands, where the body
//! reports `inner0 signed=-5263` against a bbox of ~98800 — the cavity is there,
//! correctly signed, and hollow by only 5% of the space it spans.
#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::aabb::Aabb3;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::shell::ShellId;
use brepkit_topology::solid::SolidId;

const DEFLECTION: f64 = 0.05;

fn shell_signed_volume(topo: &Topology, shell: ShellId, defl: f64) -> (f64, Aabb3, usize) {
    let origin = Point3::new(0.0, 0.0, 0.0);
    let mut total = 0.0;
    let mut pts = Vec::new();
    let faces = topo.shell(shell).unwrap().faces().to_vec();
    for fid in &faces {
        let mesh = brepkit_operations::tessellate::tessellate(topo, *fid, defl).unwrap();
        let (idx, pos) = (&mesh.indices, &mesh.positions);
        for t in 0..idx.len() / 3 {
            let a = pos[idx[t * 3] as usize];
            let b = pos[idx[t * 3 + 1] as usize];
            let c = pos[idx[t * 3 + 2] as usize];
            total += (a - origin).dot((b - origin).cross(c - origin));
            pts.extend_from_slice(&[a, b, c]);
        }
    }
    // A shell whose faces all tessellate to nothing still has a signed volume
    // and a face count worth printing, so an empty point set is a zero bbox
    // rather than a panic.
    let bb = Aabb3::try_from_points(pts).unwrap_or(Aabb3 {
        min: origin,
        max: origin,
    });
    (total / 6.0, bb, faces.len())
}

fn report(topo: &Topology, sid: SolidId, name: &str) {
    let sd = topo.solid(sid).unwrap();
    println!("{name}");
    let mut sum = 0.0;
    let shells = std::iter::once(("outer".to_string(), sd.outer_shell())).chain(
        sd.inner_shells()
            .iter()
            .enumerate()
            .map(|(i, &s)| (format!("inner{i}"), s)),
    );
    for (label, shell) in shells {
        let (v, bb, n) = shell_signed_volume(topo, shell, DEFLECTION);
        sum += v;
        let d = bb.max - bb.min;
        let bbox_vol = d.x() * d.y() * d.z();
        println!(
            "  {label:<7} faces={n:<4} signed={v:>12.1}  bbox_vol={bbox_vol:>12.1}  fill={:>5.1}%",
            if bbox_vol > 0.0 {
                100.0 * v.abs() / bbox_vol
            } else {
                0.0
            }
        );
    }
    println!("  shell sum             = {sum:>12.1}");
    println!(
        "  oriented_solid_volume = {:>12.1}",
        brepkit_operations::measure::oriented_solid_volume(topo, sid, DEFLECTION).unwrap()
    );
    println!(
        "  measure::solid_volume = {:>12.1}",
        brepkit_operations::measure::solid_volume(topo, sid, DEFLECTION).unwrap()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let names: Vec<String> = if args.is_empty() {
        vec![
            "slotted_nolip_body.bin".to_string(),
            "slotted_socket_assembly.bin".to_string(),
        ]
    } else {
        args
    };

    let mut topo = Topology::new();
    for name in &names {
        let path = if Path::new(name).is_absolute() {
            PathBuf::from(name)
        } else {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data")
                .join(name)
        };
        let sid = deserialize_solid(&std::fs::read(&path).unwrap(), &mut topo).unwrap();
        report(&topo, sid, name);
    }
}
