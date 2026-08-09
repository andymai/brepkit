//! Per-face census of captured arena `.bin` solids, restricted to a region.
//!
//! Answers "which faces exist here, and which are missing" when a boolean
//! leaves a hole: run it on both operands and on the result (see `OUT=` in
//! `replay_pair`) with the same `BOX` and diff the three listings by eye.
//! Prints surface type, the reversal flag, and the vertex-hull bbox per face;
//! `BOX` keeps a face when its bbox overlaps the given one.
//!
//! ```sh
//! BOX=xmin,ymin,zmin,xmax,ymax,zmax cargo run --release -p brepkit-io \
//!   --example region_probe -- a.bin b.bin
//! ```
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::expect_used,
    clippy::unwrap_used
)]

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

fn face_bbox(topo: &Topology, fid: brepkit_topology::face::FaceId) -> Option<(Point3, Point3)> {
    let face = topo.face(fid).ok()?;
    let mut lo = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut hi = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    let mut any = false;
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let Ok(w) = topo.wire(wid) else { continue };
        for oe in w.edges() {
            let Ok(e) = topo.edge(oe.edge()) else {
                continue;
            };
            for vid in [e.start(), e.end()] {
                let Ok(v) = topo.vertex(vid) else { continue };
                let p = v.point();
                any = true;
                lo = Point3::new(lo.x().min(p.x()), lo.y().min(p.y()), lo.z().min(p.z()));
                hi = Point3::new(hi.x().max(p.x()), hi.y().max(p.y()), hi.z().max(p.z()));
            }
        }
    }
    any.then_some((lo, hi))
}

fn main() {
    let filter: Option<Vec<f64>> = match std::env::var("BOX") {
        Err(_) => None,
        Ok(spec) => {
            let parsed: Result<Vec<f64>, _> =
                spec.split(',').map(|t| t.trim().parse::<f64>()).collect();
            match parsed {
                Ok(v) if v.len() == 6 => Some(v),
                _ => {
                    eprintln!(
                        "BOX must be 6 numbers: xmin,ymin,zmin,xmax,ymax,zmax (got {spec:?})"
                    );
                    std::process::exit(2);
                }
            }
        }
    };

    for path in std::env::args().skip(1) {
        let mut topo = Topology::new();
        let sid = deserialize_solid(&std::fs::read(&path).unwrap(), &mut topo).unwrap();
        println!("== {path}");
        let faces = solid_faces(&topo, sid).unwrap();
        let mut lo_all = Point3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut hi_all = Point3::new(f64::MIN, f64::MIN, f64::MIN);
        for &fid in &faces {
            if let Some((lo, hi)) = face_bbox(&topo, fid) {
                lo_all = Point3::new(
                    lo_all.x().min(lo.x()),
                    lo_all.y().min(lo.y()),
                    lo_all.z().min(lo.z()),
                );
                hi_all = Point3::new(
                    hi_all.x().max(hi.x()),
                    hi_all.y().max(hi.y()),
                    hi_all.z().max(hi.z()),
                );
            }
        }
        println!(
            "   bbox ({:.3},{:.3},{:.3})..({:.3},{:.3},{:.3})",
            lo_all.x(),
            lo_all.y(),
            lo_all.z(),
            hi_all.x(),
            hi_all.y(),
            hi_all.z()
        );
        for &fid in &faces {
            let Ok(face) = topo.face(fid) else { continue };
            let Some((lo, hi)) = face_bbox(&topo, fid) else {
                continue;
            };
            if let Some(f) = &filter
                && (hi.x() < f[0]
                    || hi.y() < f[1]
                    || hi.z() < f[2]
                    || lo.x() > f[3]
                    || lo.y() > f[4]
                    || lo.z() > f[5])
            {
                continue;
            }
            println!(
                "   {fid:?} {:9} rev={:5} ({:.3},{:.3},{:.3})..({:.3},{:.3},{:.3})",
                face.surface().type_tag(),
                face.is_reversed(),
                lo.x(),
                lo.y(),
                lo.z(),
                hi.x(),
                hi.y(),
                hi.z()
            );
        }
    }
}
