//! Per-face anatomy of one shell of a captured solid: surface type, AABB,
//! area, and signed-volume contribution. Written for #1536, where the slotted
//! no-lip body's inner shell is closed and correctly signed yet encloses only
//! 5% of the space it spans — the per-face contributions show which faces
//! cancel and what the void actually is.
//!
//! Usage: `cargo run --release --example shell_face_census -p brepkit-io -- \
//!   <name.bin> [shell-index]`
//! shell-index: 0 = outer (default), 1.. = inner shells in order.
//! Names resolve against `crates/io/tests/data`; absolute paths are used as-is.
#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args
        .first()
        .cloned()
        .unwrap_or_else(|| "slotted_nolip_body.bin".to_string());
    let shell_index: usize = args.get(1).map_or(0, |s| s.parse().unwrap());

    let path = if Path::new(&name).is_absolute() {
        PathBuf::from(&name)
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(&name)
    };
    let mut topo = Topology::new();
    let sid = deserialize_solid(&std::fs::read(&path).unwrap(), &mut topo).unwrap();
    let sd = topo.solid(sid).unwrap();
    let shell = if shell_index == 0 {
        sd.outer_shell()
    } else {
        sd.inner_shells()[shell_index - 1]
    };

    let origin = Point3::new(0.0, 0.0, 0.0);
    let faces = topo.shell(shell).unwrap().faces().to_vec();
    println!("{name} shell#{shell_index}: {} faces", faces.len());
    let mut total = 0.0;
    for fid in &faces {
        let face = topo.face(*fid).unwrap();
        let ty = face.surface().type_tag();
        let mesh = brepkit_operations::tessellate::tessellate(&topo, *fid, 0.05).unwrap();
        let (idx, pos) = (&mesh.indices, &mesh.positions);
        let mut vol6 = 0.0;
        let mut area2 = 0.0;
        let (mut lo, mut hi) = (
            Point3::new(f64::MAX, f64::MAX, f64::MAX),
            Point3::new(f64::MIN, f64::MIN, f64::MIN),
        );
        for t in 0..idx.len() / 3 {
            let a = pos[idx[t * 3] as usize];
            let b = pos[idx[t * 3 + 1] as usize];
            let c = pos[idx[t * 3 + 2] as usize];
            vol6 += (a - origin).dot((b - origin).cross(c - origin));
            area2 += (b - a).cross(c - a).length();
            for p in [a, b, c] {
                lo = Point3::new(lo.x().min(p.x()), lo.y().min(p.y()), lo.z().min(p.z()));
                hi = Point3::new(hi.x().max(p.x()), hi.y().max(p.y()), hi.z().max(p.z()));
            }
        }
        total += vol6 / 6.0;
        println!(
            "  {fid:>8?} {ty:<8} vol={:>10.1} area={:>8.1}  x[{:>7.2},{:>7.2}] y[{:>7.2},{:>7.2}] z[{:>6.2},{:>6.2}]",
            vol6 / 6.0,
            area2 / 2.0,
            lo.x(),
            hi.x(),
            lo.y(),
            hi.y(),
            lo.z(),
            hi.z()
        );
    }
    println!("  shell signed volume = {total:.1}");
}
