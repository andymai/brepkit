//! Probe: do marched quadric x NURBS sections stop at the SSI marcher's
//! 0.1% domain-clamp margin instead of the exact patch boundary?
//!
//! Loads the kumiko strut fixtures, marches the wedge's outer cylinder
//! against each strut patch it touches, and prints each section's endpoint
//! parameters on the patch against the patch domain and the predicted
//! margin (0.001 x span).

#![allow(clippy::print_stdout, clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::nurbs::intersection::intersect_nurbs_nurbs;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

fn main() {
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut topo = Topology::new();
    let a = deserialize_solid(
        &std::fs::read(data.join("kumiko_strut_vertical.bin")).unwrap(),
        &mut topo,
    )
    .unwrap();
    let b = deserialize_solid(
        &std::fs::read(data.join("kumiko_strut_diag_up_ruled.bin")).unwrap(),
        &mut topo,
    )
    .unwrap();

    // Outer cylinder of the wedge = the cylinder face with the larger radius.
    let mut cyl: Option<(
        brepkit_topology::face::FaceId,
        brepkit_math::surfaces::CylindricalSurface,
    )> = None;
    for fid in solid_faces(&topo, a).unwrap() {
        if let FaceSurface::Cylinder(c) = topo.face(fid).unwrap().surface() {
            let replace = cyl
                .as_ref()
                .is_none_or(|(_, prev)| c.radius() > prev.radius());
            if replace {
                cyl = Some((fid, c.clone()));
            }
        }
    }
    let (cyl_fid, cyl) = cyl.expect("no cylinder on A");

    // v-range from the face's vertex extent along the axis, padded.
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    let face = topo.face(cyl_fid).unwrap();
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wid).unwrap().edges() {
            let e = topo.edge(oe.edge()).unwrap();
            for vid in [e.start(), e.end()] {
                let p = topo.vertex(vid).unwrap().point();
                let v = (p - cyl.origin()).dot(cyl.axis());
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    let cyl_nurbs = cyl.to_nurbs(lo - 0.5, hi + 0.5).unwrap();
    println!(
        "cylinder {cyl_fid:?} r={:.4} v=[{lo:.3},{hi:.3}]",
        cyl.radius()
    );

    for fid in solid_faces(&topo, b).unwrap() {
        let FaceSurface::Nurbs(patch) = topo.face(fid).unwrap().surface() else {
            continue;
        };
        let curves = match intersect_nurbs_nurbs(&cyl_nurbs, patch, 32, 0.01) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if curves.is_empty() {
            continue;
        }
        let (u_min, u_max) = patch.domain_u();
        let (v_min, v_max) = patch.domain_v();
        let mu = 0.001 * (u_max - u_min);
        let mv = 0.001 * (v_max - v_min);
        println!(
            "patch {fid:?} u=[{u_min:.4},{u_max:.4}] v=[{v_min:.4},{v_max:.4}] margin=({mu:.6},{mv:.6})"
        );
        for (ci, c) in curves.iter().enumerate() {
            let n = c.points.len();
            if n == 0 {
                continue;
            }
            for (tag, ip) in [("start", &c.points[0]), ("end", &c.points[n - 1])] {
                let (u2, v2) = ip.param2;
                let du = (u2 - u_min).min(u_max - u2);
                let dv = (v2 - v_min).min(v_max - v2);
                println!(
                    "  curve#{ci} {tag} p=({:.6},{:.6},{:.6}) patch(u,v)=({u2:.6},{v2:.6}) dist_to_dom=({du:.6},{dv:.6})",
                    ip.point.x(),
                    ip.point.y(),
                    ip.point.z()
                );
            }
        }
    }
}
