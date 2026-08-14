//! Replay one captured boolean operand pair natively.
//!
//! The tool-side capture writes `<op>-<n>.bin` per operand; this loads two of
//! them and runs the op both through `operations::boolean` (which may fall back
//! to mesh) and through raw GFA (which reports the analytic failure directly).
//!
//! ```sh
//! A=.../op1-fuseWithEvolution-0.bin B=.../op1-fuseWithEvolution-1.bin \
//!   OP=fuse cargo run --release -p brepkit-io --example replay_pair
//! ```
#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn describe(topo: &Topology, sid: SolidId, label: &str) {
    let Ok(faces) = solid_faces(topo, sid) else {
        println!("  {label}: <no faces>");
        return;
    };
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    let mut users: HashMap<EdgeId, Vec<brepkit_topology::face::FaceId>> = HashMap::new();
    let mut mix: HashMap<&'static str, usize> = HashMap::new();
    for &fid in &faces {
        let Ok(face) = topo.face(fid) else { continue };
        *mix.entry(face.surface().type_tag()).or_default() += 1;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let Ok(w) = topo.wire(wid) else { continue };
            for oe in w.edges() {
                *uses.entry(oe.edge()).or_default() += 1;
                users.entry(oe.edge()).or_default().push(fid);
            }
        }
    }
    let free = uses.values().filter(|&&c| c == 1).count();
    let over = uses.values().filter(|&&c| c > 2).count();
    if std::env::var("WIND_DUMP").is_ok() {
        let mut eff: HashMap<EdgeId, Vec<(brepkit_topology::face::FaceId, bool)>> = HashMap::new();
        for &fid in &faces {
            let Ok(face) = topo.face(fid) else { continue };
            let rev = face.is_reversed();
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                let Ok(w) = topo.wire(wid) else { continue };
                for oe in w.edges() {
                    eff.entry(oe.edge())
                        .or_default()
                        .push((fid, oe.is_forward() != rev));
                }
            }
        }
        for (eid, us) in &eff {
            if us.len() == 2
                && us[0].1 == us[1].1
                && let Ok(e) = topo.edge(*eid)
                && let (Ok(a), Ok(b)) = (topo.vertex(e.start()), topo.vertex(e.end()))
            {
                let (a, b) = (a.point(), b.point());
                println!(
                    "  SAMEDIR edge {eid:?} {} ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3}) faces={:?}",
                    e.curve().type_tag(),
                    a.x(),
                    a.y(),
                    a.z(),
                    b.x(),
                    b.y(),
                    b.z(),
                    us
                );
            }
        }
    }
    // TWIN=1: for each free edge, scan the solid for a coincident partner
    // edge (both endpoints within 1e-3, either order) and print the endpoint
    // and midpoint separations — the instrument for seam copies that miss
    // the position-quantized merge.
    if std::env::var("TWIN").is_ok() {
        let all_eids: Vec<EdgeId> = uses.keys().copied().collect();
        let ends = |eid: EdgeId| -> Option<(brepkit_math::vec::Point3, brepkit_math::vec::Point3)> {
            let e = topo.edge(eid).ok()?;
            Some((
                topo.vertex(e.start()).ok()?.point(),
                topo.vertex(e.end()).ok()?.point(),
            ))
        };
        let mid = |eid: EdgeId| -> Option<brepkit_math::vec::Point3> {
            let e = topo.edge(eid).ok()?;
            let (sp, ep) = ends(eid)?;
            let (t0, t1) = e.curve().domain_with_endpoints(sp, ep);
            Some(
                e.curve()
                    .evaluate_with_endpoints(f64::midpoint(t0, t1), sp, ep),
            )
        };
        for (&eid, &n) in &uses {
            if n != 1 {
                continue;
            }
            let Some((sp, ep)) = ends(eid) else { continue };
            for &oid in &all_eids {
                if oid == eid {
                    continue;
                }
                let Some((os, oe2)) = ends(oid) else { continue };
                let fwd = (sp - os).length().max((ep - oe2).length());
                let rev = (sp - oe2).length().max((ep - os).length());
                let d = fwd.min(rev);
                if d < 1e-3 {
                    let md = match (mid(eid), mid(oid)) {
                        (Some(a), Some(b)) => (a - b).length(),
                        _ => f64::NAN,
                    };
                    println!(
                        "  TWIN {eid:?} <-> {oid:?} (uses={}) end_d={d:.3e} mid_d={md:.3e}",
                        uses.get(&oid).copied().unwrap_or(0)
                    );
                    if std::env::var("TWIN").is_ok_and(|v| v == "2") {
                        println!(
                            "    {eid:?} ({:.9},{:.9},{:.9})->({:.9},{:.9},{:.9})",
                            sp.x(),
                            sp.y(),
                            sp.z(),
                            ep.x(),
                            ep.y(),
                            ep.z()
                        );
                        println!(
                            "    {oid:?} ({:.9},{:.9},{:.9})->({:.9},{:.9},{:.9})",
                            os.x(),
                            os.y(),
                            os.z(),
                            oe2.x(),
                            oe2.y(),
                            oe2.z()
                        );
                    }
                }
            }
        }
    }
    if std::env::var("FREE_EDGES").is_ok() {
        for (eid, n) in &uses {
            if *n != 2
                && let Ok(e) = topo.edge(*eid)
                && let (Ok(a), Ok(b)) = (topo.vertex(e.start()), topo.vertex(e.end()))
            {
                let (a, b) = (a.point(), b.point());
                let owners: Vec<String> = users
                    .get(eid)
                    .into_iter()
                    .flatten()
                    .map(|f| {
                        let tag = topo
                            .face(*f)
                            .map(|fc| fc.surface().type_tag())
                            .unwrap_or("?");
                        format!("{f:?}:{tag}")
                    })
                    .collect();
                println!(
                    "  {} edge {eid:?} {} ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3}) used_by={owners:?}",
                    if *n == 1 { "FREE" } else { "OVER" },
                    e.curve().type_tag(),
                    a.x(),
                    a.y(),
                    a.z(),
                    b.x(),
                    b.y(),
                    b.z()
                );
                if std::env::var("FREE_EDGES").is_ok_and(|v| v == "2") {
                    for f in users.get(eid).into_iter().flatten() {
                        let Ok(fc) = topo.face(*f) else { continue };
                        let mut lo = [f64::MAX; 3];
                        let mut hi = [f64::MIN; 3];
                        let mut nedges = 0;
                        for wid in
                            std::iter::once(fc.outer_wire()).chain(fc.inner_wires().iter().copied())
                        {
                            let Ok(w) = topo.wire(wid) else { continue };
                            for oe in w.edges() {
                                nedges += 1;
                                let Ok(e2) = topo.edge(oe.edge()) else {
                                    continue;
                                };
                                for vid in [e2.start(), e2.end()] {
                                    if let Ok(v) = topo.vertex(vid) {
                                        let p = v.point();
                                        for (i, c) in [p.x(), p.y(), p.z()].iter().enumerate() {
                                            lo[i] = lo[i].min(*c);
                                            hi[i] = hi[i].max(*c);
                                        }
                                    }
                                }
                            }
                        }
                        println!(
                            "      owner {f:?} {} rev={} inner={} edges={nedges} bbox=({:.3},{:.3},{:.3})..({:.3},{:.3},{:.3})",
                            fc.surface().type_tag(),
                            fc.is_reversed(),
                            fc.inner_wires().len(),
                            lo[0],
                            lo[1],
                            lo[2],
                            hi[0],
                            hi[1],
                            hi[2]
                        );
                    }
                }
            }
        }
    }
    let mut mix: Vec<_> = mix.into_iter().collect();
    mix.sort_unstable();
    let vol =
        brepkit_operations::measure::oriented_solid_volume(topo, sid, 0.05).unwrap_or(f64::NAN);
    // TESS_BND=1 additionally tessellates at export tolerance (0.01 mm /
    // 5 degrees, matching the tool's STL export) and reports mesh boundary
    // and non-manifold edge counts — the discriminant between a B-Rep leak
    // and a tessellation-parity leak on a clean B-Rep.
    let tess = if std::env::var("TESS_BND").is_ok() {
        match brepkit_operations::tessellate::tessellate_solid_with_tolerance(
            topo,
            sid,
            0.01,
            5.0_f64.to_radians(),
        ) {
            Ok(mesh) => format!(
                " tess_bnd={} tess_nm={}",
                brepkit_operations::tessellate::boundary_edge_count(&mesh),
                brepkit_operations::tessellate::non_manifold_edge_count(&mesh)
            ),
            Err(e) => format!(" tess_err={e}"),
        }
    } else {
        String::new()
    };
    println!(
        "  {label}: F={} mix={mix:?} free={free} over={over} vol={vol:.3}{tess}",
        faces.len()
    );
}

struct Tap;
impl log::Log for Tap {
    fn enabled(&self, m: &log::Metadata) -> bool {
        let max = if std::env::var("BK_TRACE").is_ok() {
            log::Level::Trace
        } else {
            log::Level::Debug
        };
        m.target().starts_with("brepkit_") && m.level() <= max
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            println!("    [log] {}", r.args());
        }
    }
    fn flush(&self) {}
}
static TAP: Tap = Tap;

fn main() {
    let _ = log::set_logger(&TAP);
    log::set_max_level(if std::env::var("BK_TRACE").is_ok() {
        log::LevelFilter::Trace
    } else {
        log::LevelFilter::Debug
    });

    let a_path = PathBuf::from(std::env::var_os("A").expect("A=<path>"));
    let b_path = PathBuf::from(std::env::var_os("B").expect("B=<path>"));
    let op = std::env::var("OP").unwrap_or_else(|_| "fuse".to_string());

    let mut topo = Topology::new();
    let a = deserialize_solid(&std::fs::read(&a_path).unwrap(), &mut topo).unwrap();
    let b = deserialize_solid(&std::fs::read(&b_path).unwrap(), &mut topo).unwrap();
    describe(&topo, a, "A");
    describe(&topo, b, "B");

    if std::env::var("FACE_DUMP").is_ok() {
        for (sid, label) in [(a, "A"), (b, "B")] {
            for fid in solid_faces(&topo, sid).unwrap() {
                let face = topo.face(fid).unwrap();
                let s = face.surface().clone();
                let mut lo = [f64::MAX; 3];
                let mut hi = [f64::MIN; 3];
                for wid in
                    std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
                {
                    let Ok(w) = topo.wire(wid) else { continue };
                    for oe in w.edges() {
                        let Ok(e) = topo.edge(oe.edge()) else {
                            continue;
                        };
                        for vid in [e.start(), e.end()] {
                            let Ok(v) = topo.vertex(vid) else { continue };
                            let p = v.point();
                            for (k, c) in [p.x(), p.y(), p.z()].into_iter().enumerate() {
                                lo[k] = lo[k].min(c);
                                hi[k] = hi[k].max(c);
                            }
                        }
                    }
                }
                let bbox = format!(
                    "bbox=({:.3},{:.3},{:.3})..({:.3},{:.3},{:.3})",
                    lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
                );
                let rev = face.is_reversed();
                if s.is_analytic() || matches!(s, brepkit_topology::face::FaceSurface::Plane { .. })
                {
                    println!("{label} {fid:?} rev={rev} {bbox} {s:?}");
                } else {
                    println!("{label} {fid:?} rev={rev} {bbox} nurbs");
                }
            }
        }
        return;
    }

    if let Ok(want) = std::env::var("MESH_FACE") {
        for fid in solid_faces(&topo, a).unwrap() {
            if format!("{fid:?}") != format!("Id({want})") {
                continue;
            }
            let m = brepkit_operations::tessellate::tessellate_with_uvs(&topo, fid, 0.05)
                .unwrap()
                .mesh;
            let mut nz = 0.0;
            for t in 0..m.indices.len() / 3 {
                let p0 = m.positions[m.indices[t * 3] as usize];
                let p1 = m.positions[m.indices[t * 3 + 1] as usize];
                let p2 = m.positions[m.indices[t * 3 + 2] as usize];
                nz += (p1 - p0).cross(p2 - p0).z();
            }
            println!(
                "A {fid:?} rev={} tris={} sum_tri_normal_z={nz:.3}",
                topo.face(fid).unwrap().is_reversed(),
                m.indices.len() / 3
            );
        }
        return;
    }

    if std::env::var("WINDING_CENSUS").is_ok() {
        for (sid, label) in [(a, "A"), (b, "B")] {
            let mut agree = 0;
            let mut disagree = 0;
            for fid in solid_faces(&topo, sid).unwrap() {
                let face = topo.face(fid).unwrap();
                let brepkit_topology::face::FaceSurface::Plane { normal, .. } = *face.surface()
                else {
                    continue;
                };
                let wire = topo.wire(face.outer_wire()).unwrap();
                let mut pts = Vec::new();
                for oe in wire.edges() {
                    let e = topo.edge(oe.edge()).unwrap();
                    let vid = if oe.is_forward() { e.start() } else { e.end() };
                    pts.push(topo.vertex(vid).unwrap().point());
                }
                if pts.len() < 3 {
                    continue;
                }
                let mut n = brepkit_math::vec::Vec3::new(0.0, 0.0, 0.0);
                for i in 0..pts.len() {
                    let p = pts[i];
                    let q = pts[(i + 1) % pts.len()];
                    n += brepkit_math::vec::Vec3::new(
                        (p.y() - q.y()) * (p.z() + q.z()),
                        (p.z() - q.z()) * (p.x() + q.x()),
                        (p.x() - q.x()) * (p.y() + q.y()),
                    );
                }
                let wire_sign = n.dot(normal) > 0.0;
                let flag_says_forward = !face.is_reversed();
                if wire_sign == flag_says_forward {
                    agree += 1;
                } else {
                    disagree += 1;
                    println!(
                        "{label} {fid:?} DISAGREE rev={} wire_dot_normal={:.3}",
                        face.is_reversed(),
                        n.dot(normal)
                    );
                }
            }
            println!("{label}: planar faces agree={agree} disagree={disagree}");
        }
        return;
    }

    if let Ok(want) = std::env::var("FACE_WIRES") {
        for (sid, label) in [(a, "A"), (b, "B")] {
            for fid in solid_faces(&topo, sid).unwrap() {
                if format!("{fid:?}") != format!("Id({want})") {
                    continue;
                }
                let face = topo.face(fid).unwrap();
                for (wi, wid) in std::iter::once(face.outer_wire())
                    .chain(face.inner_wires().iter().copied())
                    .enumerate()
                {
                    let w = topo.wire(wid).unwrap();
                    println!("{label} {fid:?} wire{wi} ({} edges):", w.edges().len());
                    for oe in w.edges() {
                        let e = topo.edge(oe.edge()).unwrap();
                        let (sv, ev) = (
                            topo.vertex(e.start()).unwrap().point(),
                            topo.vertex(e.end()).unwrap().point(),
                        );
                        println!(
                            "  {:?} {} fwd={} ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3})",
                            oe.edge(),
                            e.curve().type_tag(),
                            oe.is_forward(),
                            sv.x(),
                            sv.y(),
                            sv.z(),
                            ev.x(),
                            ev.y(),
                            ev.z()
                        );
                    }
                }
            }
        }
        return;
    }

    let bop = match op.as_str() {
        "cut" => brepkit_algo::bop::BooleanOp::Cut,
        "intersect" => brepkit_algo::bop::BooleanOp::Intersect,
        _ => brepkit_algo::bop::BooleanOp::Fuse,
    };

    // POINT_IN=x,y,z classifies a point against BOTH operands with the
    // independent operations-level oracle. For a Fuse, a face is needed
    // wherever one side of a surface is inside the union and the other is not.
    if let Ok(spec) = std::env::var("POINT_IN") {
        // Semicolon-separated points, so a batch is classified in one process
        // instead of one process per point.
        // Deflection matters: classify_point tessellates, and this lattice has
        // 0.05mm features that a coarse deflection cannot represent, which
        // makes the verdict itself an artifact of the setting.
        let defl: f64 = std::env::var("POINT_DEFL")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0.01);
        for (i, one) in spec.split(';').filter(|t| !t.trim().is_empty()).enumerate() {
            let c: Vec<f64> = one
                .split(',')
                .filter_map(|t| t.trim().parse().ok())
                .collect();
            if c.len() != 3 {
                continue;
            }
            let p = brepkit_math::vec::Point3::new(c[0], c[1], c[2]);
            let mut row = format!("  POINT_IN[{i}] ({:.3},{:.3},{:.3})", c[0], c[1], c[2]);
            for (label, sid) in [("A", a), ("B", b)] {
                match brepkit_operations::classify::classify_point(&topo, sid, p, defl, 1e-7) {
                    Ok(v) => {
                        let _ = write!(row, "  {label}={v:?}");
                    }
                    Err(_) => {
                        let _ = write!(row, "  {label}=ERR");
                    }
                }
            }
            println!("{row}");
        }
        return;
    }

    // TOOLS=<comma-separated paths> replays a compound_cut, which is how the
    // kumiko wrap chain is actually built; a pairwise replay cannot reach it.
    if let Ok(list) = std::env::var("TOOLS") {
        let tools: Vec<_> = list
            .split(',')
            .filter(|t| !t.trim().is_empty())
            .map(|t| deserialize_solid(&std::fs::read(t.trim()).unwrap(), &mut topo).unwrap())
            .collect();
        if std::env::var("TOOLS_SEQ").is_ok() {
            println!("-- sequential cuts with {} tools --", tools.len());
            let t = std::time::Instant::now();
            let mut cur = a;
            let mut failed = false;
            for (i, &tool) in tools.iter().enumerate() {
                match brepkit_operations::boolean::boolean(
                    &mut topo,
                    brepkit_operations::boolean::BooleanOp::Cut,
                    cur,
                    tool,
                ) {
                    Ok(next) => cur = next,
                    Err(e) => {
                        println!("  cut {i} FAILED: {e}");
                        failed = true;
                        break;
                    }
                }
            }
            if !failed {
                describe(
                    &topo,
                    cur,
                    &format!("sequential {}ms", t.elapsed().as_millis()),
                );
            }
            return;
        }
        println!("-- compound_cut with {} tools --", tools.len());
        let t = std::time::Instant::now();
        match brepkit_operations::boolean::compound_cut(
            &mut topo,
            a,
            &tools,
            brepkit_operations::boolean::BooleanOptions::default(),
        ) {
            Ok(sid) => {
                describe(
                    &topo,
                    sid,
                    &format!("compound_cut {}ms", t.elapsed().as_millis()),
                );
                // CORNER=<path> follows with an intersect against the given
                // solid (the baseplate corner-rounding step of #1488).
                if let Ok(cpath) = std::env::var("CORNER") {
                    let ctool =
                        deserialize_solid(&std::fs::read(&cpath).unwrap(), &mut topo).unwrap();
                    let t2 = std::time::Instant::now();
                    match brepkit_operations::boolean::boolean(
                        &mut topo,
                        brepkit_operations::boolean::BooleanOp::Intersect,
                        sid,
                        ctool,
                    ) {
                        Ok(rid) => describe(
                            &topo,
                            rid,
                            &format!("corner intersect {}ms", t2.elapsed().as_millis()),
                        ),
                        Err(e) => println!(
                            "  corner intersect FAILED in {}ms: {e}",
                            t2.elapsed().as_millis()
                        ),
                    }
                }
            }
            Err(e) => println!(
                "  compound_cut FAILED in {}ms: {e}",
                t.elapsed().as_millis()
            ),
        }
        return;
    }

    // FUSE_ALL=1 replays the wasm `fuseAll` path (compound_ops::fuse_all over
    // {A, B}). A 2-solid group takes the pairwise contract (fallback accepted);
    // 3+ solid clusters route through fuse_cluster, whose mesh-fallback bail
    // rejects a degraded fuse to protect the rest of the batch.
    if std::env::var("FUSE_ALL").is_ok() {
        println!("-- compound_ops::fuse_all {{A, B}} --");
        let compound = topo.add_compound(brepkit_topology::compound::Compound::new(vec![a, b]));
        let before = brepkit_operations::boolean::mesh_fallback_count();
        let t = std::time::Instant::now();
        match brepkit_operations::compound_ops::fuse_all(&mut topo, compound) {
            Ok(sid) => {
                let ms = t.elapsed().as_millis();
                let fell_back = brepkit_operations::boolean::mesh_fallback_count() > before;
                describe(
                    &topo,
                    sid,
                    &format!(
                        "fuse_all {ms}ms{}",
                        if fell_back { " [MESH FALLBACK]" } else { "" }
                    ),
                );
            }
            Err(e) => println!("  fuse_all FAILED in {}ms: {e}", t.elapsed().as_millis()),
        }
        return;
    }

    let raw_only = std::env::var("RAW_ONLY").is_ok();
    if !raw_only {
        println!("-- operations::boolean {op} --");
    }
    let ops_op = match op.as_str() {
        "cut" => brepkit_operations::boolean::BooleanOp::Cut,
        "intersect" => brepkit_operations::boolean::BooleanOp::Intersect,
        _ => brepkit_operations::boolean::BooleanOp::Fuse,
    };
    if !raw_only {
        let before = brepkit_operations::boolean::mesh_fallback_count();
        let t = std::time::Instant::now();
        match brepkit_operations::boolean::boolean(&mut topo, ops_op, a, b) {
            Ok(sid) => {
                let ms = t.elapsed().as_millis();
                let fell_back = brepkit_operations::boolean::mesh_fallback_count() > before;
                describe(
                    &topo,
                    sid,
                    &format!(
                        "OPS {op} {ms}ms{}",
                        if fell_back { " [MESH FALLBACK]" } else { "" }
                    ),
                );
            }
            Err(e) => println!("  OPS {op} FAILED in {}ms: {e}", t.elapsed().as_millis()),
        }
    }

    if std::env::var("VCORNER").is_ok() {
        for i in 0..topo.num_vertices() {
            let Some(vid) = topo.vertex_id_from_index(i) else {
                continue;
            };
            let Ok(v) = topo.vertex(vid) else { continue };
            let q = v.point();
            if (q.x() - 2.9675).abs() < 1e-3
                && (q.y() - 4.0242).abs() < 1e-3
                && (q.z() - 13.962).abs() < 5e-2
            {
                println!("  VCORNER {vid:?} ({:.9},{:.9},{:.9})", q.x(), q.y(), q.z());
            }
        }
    }
    let vcorner_enabled = std::env::var("VCORNER").is_ok();
    let vcorner_scan = |topo: &Topology, tag: &str| {
        if vcorner_enabled {
            for i in 0..topo.num_vertices() {
                let Some(vid) = topo.vertex_id_from_index(i) else {
                    continue;
                };
                let Ok(v) = topo.vertex(vid) else { continue };
                let q = v.point();
                if (q.x() - 2.9675).abs() < 1e-3
                    && (q.y() - 4.0242).abs() < 1e-3
                    && (q.z() - 13.962).abs() < 5e-2
                {
                    println!(
                        "  VCORNER[{tag}] {vid:?} ({:.9},{:.9},{:.9})",
                        q.x(),
                        q.y(),
                        q.z()
                    );
                }
            }
        }
    };
    println!("-- raw GFA {op} --");
    let t = std::time::Instant::now();
    let raw_res = brepkit_algo::gfa::boolean(&mut topo, bop, a, b);
    vcorner_scan(&topo, "post-raw");
    match raw_res {
        Ok(sid) => {
            describe(
                &topo,
                sid,
                &format!("RAW {op} {}ms", t.elapsed().as_millis()),
            );
            // OUT=<path> serializes the RAW result so region probes can run
            // on it without the ops-level heals in between.
            if let Ok(out) = std::env::var("OUT") {
                match brepkit_io::arena_io::serialize_solid(&topo, sid) {
                    Ok(bytes) => {
                        std::fs::write(&out, bytes).unwrap();
                        println!("  wrote raw result to {out}");
                    }
                    Err(e) => println!("  serialize failed: {e}"),
                }
            }
        }
        Err(e) => println!("  RAW {op} FAILED in {}ms: {e}", t.elapsed().as_millis()),
    }
}
