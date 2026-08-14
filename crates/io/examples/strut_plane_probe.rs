//! Probe: how does each wedge side plane meet the strut's end patches?
//!
//! For every (plane of A) x (NURBS patch of B) pair, samples the patch on a
//! grid and reports the signed-distance range to the plane, then runs
//! `intersect_plane_nurbs` and prints each returned curve's endpoints — the
//! discriminant between a near-coincident contact (SD territory) and a
//! transversal crossing the marcher fragments.

#![allow(clippy::print_stdout, clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::nurbs::intersection::intersect_plane_nurbs;
use brepkit_math::vec::Vec3;
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

    let only: Option<String> = std::env::var("PATCH").ok();

    let planes: Vec<(brepkit_topology::face::FaceId, Vec3, f64)> = solid_faces(&topo, a)
        .unwrap()
        .into_iter()
        .filter_map(|fid| {
            if let FaceSurface::Plane { normal, d } = topo.face(fid).unwrap().surface() {
                Some((fid, *normal, *d))
            } else {
                None
            }
        })
        .collect();

    for fid in solid_faces(&topo, b).unwrap() {
        if let Some(want) = &only
            && format!("{fid:?}") != format!("Id({want})")
        {
            continue;
        }
        let FaceSurface::Nurbs(patch) = topo.face(fid).unwrap().surface() else {
            continue;
        };
        let (u0, u1) = patch.domain_u();
        let (v0, v1) = patch.domain_v();
        for &(pid, normal, d) in &planes {
            let mut lo = f64::MAX;
            let mut hi = f64::MIN;
            let n = 12;
            for i in 0..=n {
                for j in 0..=n {
                    let u = u0 + (u1 - u0) * f64::from(i) / f64::from(n);
                    let v = v0 + (v1 - v0) * f64::from(j) / f64::from(n);
                    let p = patch.evaluate(u, v);
                    let s = normal.dot(Vec3::new(p.x(), p.y(), p.z())) - d;
                    lo = lo.min(s);
                    hi = hi.max(s);
                }
            }
            if lo > 0.0 || hi < 0.0 {
                continue;
            }
            println!(
                "patch {fid:?} x plane {pid:?} signed_dist=[{lo:.6},{hi:.6}] span={:.6}",
                hi - lo
            );
            match intersect_plane_nurbs(patch, normal, d, 32) {
                Ok(curves) => {
                    for (ci, c) in curves.iter().enumerate() {
                        let np = c.points.len();
                        if np == 0 {
                            continue;
                        }
                        let s = c.points[0].point;
                        let e = c.points[np - 1].point;
                        println!(
                            "  curve#{ci} pts={np} ({:.6},{:.6},{:.6})->({:.6},{:.6},{:.6}) len~{:.6}",
                            s.x(),
                            s.y(),
                            s.z(),
                            e.x(),
                            e.y(),
                            e.z(),
                            (e - s).length()
                        );
                    }
                }
                Err(err) => println!("  ERR {err}"),
            }
        }
    }
}
