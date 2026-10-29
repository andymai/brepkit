//! Plane-NURBS surface intersection.

use crate::MathError;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::projection::{project_point_to_surface, project_point_to_surface_from};
use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

use super::chaining::{Section, build_curves_from_chains, fit_within};
use super::{IntersectionCurve, IntersectionPoint, MAX_NEWTON_ITER};

/// Intersect a plane with a NURBS surface.
///
/// Returns a list of intersection curves (there may be multiple
/// disconnected branches).
///
/// # Parameters
///
/// - `surface`: The NURBS surface
/// - `plane_normal`: Normal of the cutting plane
/// - `plane_d`: Signed distance from origin (`n · p = d` for points on plane)
/// - `samples`: Grid resolution for finding seed points (e.g., 50)
///
/// # Errors
///
/// Returns an error if NURBS evaluation fails or curve fitting fails.
pub fn intersect_plane_nurbs(
    surface: &NurbsSurface,
    plane_normal: Vec3,
    plane_d: f64,
    samples: usize,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let n = samples.max(10);

    // Phase 1: Sample the signed distance field on a grid.
    let mut distances = vec![vec![0.0_f64; n]; n];

    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();

    #[allow(clippy::cast_precision_loss)]
    let u_step = (u_max - u_min) / (n - 1) as f64;
    #[allow(clippy::cast_precision_loss)]
    let v_step = (v_max - v_min) / (n - 1) as f64;

    let mut reach = 0.0_f64;
    #[allow(clippy::cast_precision_loss)]
    for (i, row) in distances.iter_mut().enumerate() {
        let u = u_min + i as f64 * u_step;
        for (j, dist) in row.iter_mut().enumerate() {
            let v = v_min + j as f64 * v_step;
            let pt = surface.evaluate(u, v);
            let pt_vec = Vec3::new(pt.x(), pt.y(), pt.z());
            reach = reach.max(plane_terms(plane_normal, plane_d, pt));
            *dist = plane_normal.dot(pt_vec) - plane_d;
        }
    }
    // A node within rounding of the plane counts with its positive side.
    let on_plane = 64.0 * f64::EPSILON * reach;
    let below = |f: f64| f < -on_plane;

    // Phase 2: Find zero-crossings on the UNIQUE grid edges (marching
    // squares). Every crossing is computed once and remembers which edge it
    // sits on, so connectivity comes from cell adjacency in phase 3 — not
    // from a distance threshold. Proximity chaining cannot serve this scan:
    // its threshold has no way to tell a sampling gap along one branch from
    // two genuinely-close branches (the kumiko strut end-patch dashes vs the
    // honeycomb thin-wall branches), and each statistic that fixes one side
    // breaks the other. Cell connectivity satisfies both by construction.
    // A node on the plane counts with the positive side, so a section along
    // a seam or a run of nodes still crosses the edges to its negative
    // neighbours, at the nodes themselves.
    let mut crossings: Vec<(f64, f64)> = Vec::new();
    // Which parameter a crossing on the domain's edge lines must keep: a
    // chain ending there ends ON the patch boundary, where the boundary
    // edge's own crossing with the plane (EE/EF) puts its vertex.
    let mut pinned: Vec<Option<Pin>> = Vec::new();
    // horiz[i][j]: crossing on the edge (i,j)->(i+1,j); vert[i][j]: on
    // (i,j)->(i,j+1).
    let mut horiz = vec![vec![None::<usize>; n]; n - 1];
    let mut vert = vec![vec![None::<usize>; n - 1]; n];

    #[allow(clippy::cast_precision_loss)]
    for i in 0..n {
        for j in 0..n {
            let u0 = u_min + i as f64 * u_step;
            let v0 = v_min + j as f64 * v_step;
            if i + 1 < n {
                let (da, db) = (distances[i][j], distances[i + 1][j]);
                if below(da) != below(db) {
                    let t = (da / (da - db)).clamp(0.0, 1.0);
                    let u = u0.mul_add(1.0 - t, (u0 + u_step) * t);
                    horiz[i][j] = Some(crossings.len());
                    let (v, pin) = match j {
                        0 => (v_min, Some(Pin::V)),
                        _ if j + 1 == n => (v_max, Some(Pin::V)),
                        _ => (v0, None),
                    };
                    crossings.push((u, v));
                    pinned.push(pin);
                }
            }
            if j + 1 < n {
                let (da, db) = (distances[i][j], distances[i][j + 1]);
                if below(da) != below(db) {
                    let t = (da / (da - db)).clamp(0.0, 1.0);
                    let v = v0.mul_add(1.0 - t, (v0 + v_step) * t);
                    vert[i][j] = Some(crossings.len());
                    let (u, pin) = match i {
                        0 => (u_min, Some(Pin::U)),
                        _ if i + 1 == n => (u_max, Some(Pin::U)),
                        _ => (u0, None),
                    };
                    crossings.push((u, v));
                    pinned.push(pin);
                }
            }
        }
    }

    let (tiny, touches, valleys) = tiny_loops(
        surface,
        plane_normal,
        plane_d,
        &distances,
        (u_min, u_step),
        (v_min, v_step),
    );
    if crossings.is_empty() && tiny.is_empty() && touches.is_empty() && valleys.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 3: Link crossings through each cell. A cell with two crossing
    // edges connects them; a saddle cell (all four edges crossing) pairs
    // them by the sign of the cell-center sample, unless the section
    // crosses itself there: a point of the cell where the plane's distance
    // stops changing on the plane, and curves away from it on both sides or
    // not at all, joins all four. With every node on one side or the
    // other, a cell has no odd count.
    let noise = rounding(surface, plane_normal, plane_d);
    let mut nodes: Vec<((f64, f64), [usize; 4])> = Vec::new();
    let mut links: Vec<Vec<usize>> = vec![Vec::new(); crossings.len()];
    let connect = |a: usize, b: usize, links: &mut Vec<Vec<usize>>| {
        if !links[a].contains(&b) {
            links[a].push(b);
            links[b].push(a);
        }
    };
    #[allow(clippy::cast_precision_loss)]
    for i in 0..n - 1 {
        for j in 0..n - 1 {
            let bottom = horiz[i][j];
            let top = horiz[i][j + 1];
            let left = vert[i][j];
            let right = vert[i + 1][j];
            let mut present = [0_usize; 4];
            let mut n_present = 0_usize;
            for e in [bottom, top, left, right].into_iter().flatten() {
                present[n_present] = e;
                n_present += 1;
            }
            match n_present {
                2 => connect(present[0], present[1], &mut links),
                4 => {
                    // Saddle: the center sample decides which corners the
                    // curve separates.
                    let uc = u_min + (i as f64 + 0.5) * u_step;
                    let vc = v_min + (j as f64 + 0.5) * v_step;
                    if let Some(node) = extremum(surface, plane_normal, (uc, vc)).filter(|e| {
                        let (fuu, fuv, fvv) = e.hessian;
                        let ((u, v), at) = (e.at, surface.evaluate(e.at.0, e.at.1));
                        let off = plane_normal.dot(Vec3::new(at.x(), at.y(), at.z())) - plane_d;
                        e.settled
                            && (u - uc).abs() <= u_step
                            && (v - vc).abs() <= v_step
                            && off.abs() <= noise
                            && fuu.mul_add(fvv, -(fuv * fuv)) <= 0.0
                    }) {
                        nodes.push((node.at, present));
                        continue;
                    }
                    let pc = surface.evaluate(uc, vc);
                    let dc = plane_normal.dot(Vec3::new(pc.x(), pc.y(), pc.z())) - plane_d;
                    let d00 = distances[i][j];
                    let (b, t, l, r) = (
                        bottom.unwrap_or(usize::MAX),
                        top.unwrap_or(usize::MAX),
                        left.unwrap_or(usize::MAX),
                        right.unwrap_or(usize::MAX),
                    );
                    if below(dc) == below(d00) {
                        // Center joins the (0,0)/(1,1) corners: the curve
                        // cuts off the other two corners separately.
                        connect(b, r, &mut links);
                        connect(t, l, &mut links);
                    } else {
                        connect(b, l, &mut links);
                        connect(t, r, &mut links);
                    }
                }
                _ => {}
            }
        }
    }

    let mut junction = vec![false; crossings.len()];
    for (at, ends) in nodes {
        let node = crossings.len();
        crossings.push(at);
        pinned.push(None);
        links.push(Vec::new());
        junction.push(true);
        for end in ends {
            connect(end, node, &mut links);
        }
    }

    // Phase 4: Walk the links into ordered chains (open chains from
    // degree-1 endpoints first, then from the points where the section
    // crosses itself, then closed loops), Newton-refining each crossing; a
    // refinement failure splits its chain. A chain stops at such a point,
    // which any number of chains may share.
    let mut visited = vec![false; crossings.len()];
    let mut ordered_chains: Vec<Vec<IntersectionPoint>> = Vec::new();
    let walk = |from: Option<usize>, start: usize, visited: &mut Vec<bool>| -> Vec<usize> {
        let mut chain: Vec<usize> = from.into_iter().chain([start]).collect();
        visited[start] = true;
        let (mut prev, mut current) = (from, start);
        while let Some(next) = links[current]
            .iter()
            .copied()
            .find(|&x| Some(x) != prev && (junction[x] || !visited[x]))
        {
            chain.push(next);
            if junction[next] {
                return chain;
            }
            visited[next] = true;
            (prev, current) = (Some(current), next);
        }
        // A loop ends beside its start; closing it keeps the cell between
        // them, and the curve, unbroken.
        if from.is_none() && chain.len() > 2 && links[current].contains(&start) {
            chain.push(start);
        }
        chain
    };
    let degree_one: Vec<usize> = (0..crossings.len())
        .filter(|&x| !junction[x] && links[x].len() == 1)
        .collect();
    let mut index_chains: Vec<Vec<usize>> = Vec::new();
    for start in degree_one {
        if !visited[start] {
            index_chains.push(walk(None, start, &mut visited));
        }
    }
    for node in (0..crossings.len()).filter(|&x| junction[x]) {
        for &start in &links[node] {
            if !visited[start] {
                index_chains.push(walk(Some(node), start, &mut visited));
            }
        }
    }
    for start in 0..crossings.len() {
        if !junction[start] && !visited[start] {
            index_chains.push(walk(None, start, &mut visited));
        }
    }
    for chain in index_chains {
        // A node on the plane amid negative neighbours crosses every edge to
        // them at itself, which links into a chain at that one node: the
        // surface touching the plane there, which the tiny-loop pass finds.
        let first = crossings[chain[0]];
        if chain.iter().all(|&idx| {
            let (u, v) = crossings[idx];
            (u - first.0).abs() <= 1e-12 * u_step && (v - first.1).abs() <= 1e-12 * v_step
        }) {
            continue;
        }
        let mut current: Vec<IntersectionPoint> = Vec::new();
        for idx in chain {
            let (u_guess, v_guess) = crossings[idx];
            if let Some(refined) = refine_plane_surface_point(
                surface,
                plane_normal,
                plane_d,
                (u_guess, v_guess),
                pinned[idx],
            ) {
                current.push(refined);
            } else if current.len() >= 2 {
                ordered_chains.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
        }
        if current.len() >= 2 {
            ordered_chains.push(current);
        }
    }

    if ordered_chains.is_empty() && tiny.is_empty() && touches.is_empty() && valleys.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 5: Fit a curve through each ordered chain.
    let len = plane_normal.length();
    let foot = |p: Point3, near: &IntersectionPoint| {
        project_point_to_surface_from(surface, p, near.param1, FIT_TOLERANCE)
            .or_else(|_| project_point_to_surface(surface, p, FIT_TOLERANCE))
            .ok()
    };
    let off = |p: Point3, near: &IntersectionPoint| {
        let off_plane = (plane_normal.dot(Vec3::new(p.x(), p.y(), p.z())) - plane_d).abs() / len;
        Some(foot(p, near)?.distance.max(off_plane))
    };
    let settle = |p: Point3, near: &IntersectionPoint| {
        let foot = foot(p, near)?;
        refine_plane_surface_point(surface, plane_normal, plane_d, (foot.u, foot.v), None)
    };
    let section = Section {
        off: &off,
        settle: &settle,
    };
    let mut curves = build_curves_from_chains(&ordered_chains, FIT_TOLERANCE, &section)?;
    // A tiny loop runs where the plane all but touches the surface, so a
    // point can lie close to both well off the loop: its stray is how far
    // settling moves it onto the section.
    let to_loop = |p: Point3, near: &IntersectionPoint| Some((p - settle(p, near)?.point).length());
    let tiny_section = Section {
        off: &to_loop,
        settle: &settle,
    };
    // A tiny loop's points lie evenly round it, so none is one met twice,
    // however close: the chains' merge of points within 1e-6 would fold a
    // loop smaller than that into a point.
    for mut chain in tiny {
        let curve = fit_within(&mut chain, FIT_TOLERANCE, &tiny_section)?;
        curves.push(IntersectionCurve {
            curve,
            points: chain,
        });
    }
    // A valley the fit cannot take leaves no curve: the surface only
    // touches the plane there.
    for mut chain in valleys {
        if let Ok(curve) = fit_within(&mut chain, FIT_TOLERANCE, &section) {
            curves.push(IntersectionCurve {
                curve,
                points: chain,
            });
        }
    }
    // A point where the plane touches the surface is a curve that stays on
    // it, with that one point.
    for touch in touches {
        let ends = vec![touch.point, touch.point];
        curves.push(IntersectionCurve {
            curve: NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], ends, vec![1.0, 1.0])?,
            points: vec![touch],
        });
    }
    Ok(curves)
}

/// How far a fitted section may stray from the plane or the surface.
const FIT_TOLERANCE: f64 = 1e-7;

/// Points round each loop [`tiny_loops`] finds.
const TINY_LOOP_POINTS: usize = 32;

/// Closed sections that enclose no grid node, so that no grid edge sees
/// the plane's distance change sign. Each winds round an extremum of the
/// distance whose value has the other sign from the nodes round it. A node
/// holding the least distance of its block, all of one sign, seeds Newton
/// for the extremum anywhere on the patch. From there rays bracket the
/// section and bisection settles each point to rounding, at any scale; a
/// loop that leaves the patch or holds a node is not one of these.
///
/// An extremum within rounding of the plane, where the distance curves
/// away from it both ways, is a point where the surface touches it,
/// returned with the loops. Where it curves away one way only, the floor
/// of a valley lies on the plane, and the surface touches the plane along
/// it: [`trace_valley`] follows it.
#[allow(
    clippy::cast_precision_loss,
    clippy::type_complexity,
    clippy::too_many_lines
)]
fn tiny_loops(
    surface: &NurbsSurface,
    normal: Vec3,
    d: f64,
    distances: &[Vec<f64>],
    (u_min, u_step): (f64, f64),
    (v_min, v_step): (f64, f64),
) -> (
    Vec<Vec<IntersectionPoint>>,
    Vec<IntersectionPoint>,
    Vec<Vec<IntersectionPoint>>,
) {
    let n = distances.len();
    let (u_lo, u_hi) = surface.domain_u();
    let (v_lo, v_hi) = surface.domain_v();
    let grain = f64::EPSILON * u_lo.abs().max(u_hi.abs()).max(v_lo.abs()).max(v_hi.abs());
    let at = |u: f64, v: f64| {
        let p = surface.evaluate(u, v);
        normal.dot(Vec3::new(p.x(), p.y(), p.z())) - d
    };
    // Along `from + t·dir`, where the distance first loses the sign it
    // has at `from`; `None` when the ray leaves the patch before that.
    let crossing = |from: (f64, f64), inner: f64, dir: (f64, f64), start: f64| {
        let along = |t: f64| (dir.0.mul_add(t, from.0), dir.1.mul_add(t, from.1));
        let on_patch =
            |(u, v): (f64, f64)| (u_lo..=u_hi).contains(&u) && (v_lo..=v_hi).contains(&v);
        let (mut lo, mut hi) = (0.0, start);
        for _ in 0..64 {
            let p = along(hi);
            if !on_patch(p) {
                return None;
            }
            if at(p.0, p.1) * inner <= 0.0 {
                break;
            }
            (lo, hi) = (hi, 2.0 * hi);
        }
        let reach = dir.0.abs().max(dir.1.abs());
        for _ in 0..128 {
            if (hi - lo) * reach <= grain {
                break;
            }
            let mid = 0.5 * (lo + hi);
            let p = along(mid);
            let f = at(p.0, p.1);
            if f * inner > 0.0 {
                lo = mid;
            } else if f * inner < 0.0 {
                hi = mid;
            } else {
                return Some(p);
            }
        }
        let (a, b) = (along(lo), along(hi));
        if at(b.0, b.1) * inner > 0.0 {
            return None;
        }
        Some(if at(a.0, a.1).abs() < at(b.0, b.1).abs() {
            a
        } else {
            b
        })
    };
    // An extremum no further past the plane than rounding may only touch
    // it, as along a valley lying on the plane, where rays from it would
    // trace rounding.
    let noise = rounding(surface, normal, d);
    let mut loops: Vec<Vec<IntersectionPoint>> = Vec::new();
    let mut touches: Vec<IntersectionPoint> = Vec::new();
    let mut valleys: Vec<Vec<IntersectionPoint>> = Vec::new();
    let step = 0.5 * u_step.min(v_step);
    for i in 0..n {
        for j in 0..n {
            let block = || {
                (i.saturating_sub(1)..=(i + 1).min(n - 1)).flat_map(|a| {
                    (j.saturating_sub(1)..=(j + 1).min(n - 1)).map(move |b| distances[a][b])
                })
            };
            let mut near = distances[i][j];
            if near.abs() <= noise {
                // A node on the plane seeds where its block lies on one side
                // of the plane but for other nodes on it, as along a valley
                // through a column of nodes; it takes that side.
                near = block().fold(0.0, |m: f64, f| if f.abs() > m.abs() { f } else { m });
                if near.abs() <= noise || block().any(|f| f * near < 0.0 && f.abs() > noise) {
                    continue;
                }
            } else if !block().all(|f| f * near > 0.0 && f.abs() >= near.abs()) {
                continue;
            }
            let seed = (u_min + i as f64 * u_step, v_min + j as f64 * v_step);
            let Some(Extremum {
                at: (u, v),
                settled,
                hessian: (fuu, fuv, fvv),
            }) = extremum(surface, normal, seed)
            else {
                continue;
            };
            let centre = at(u, v);
            let spread = (fuu - fvv).hypot(2.0 * fuv);
            let (big, small) = (0.5 * (fuu + fvv + spread), 0.5 * (fuu + fvv - spread));
            if centre.abs() <= noise {
                let point = surface.evaluate(u, v);
                let known =
                    |q: &IntersectionPoint| (q.param1.0 - u).hypot(q.param1.1 - v) <= 2.0 * step;
                if !settled || valleys.iter().flatten().any(known) {
                    continue;
                }
                if big * near > 0.0
                    && small * near > 0.0
                    && big.abs().min(small.abs()) > 1e-6 * big.abs().max(small.abs())
                    && touches
                        .iter()
                        .all(|t: &IntersectionPoint| (t.point - point).length() > FIT_TOLERANCE)
                {
                    touches.push(IntersectionPoint {
                        point,
                        param1: (u, v),
                        param2: (0.0, 0.0),
                    });
                    continue;
                }
                let b = bend(surface, normal, (u, v));
                if b.major.0 * near > 0.0 && b.minor.0.abs() <= 1e-6 * b.major.0.abs() {
                    let valley = trace_valley(surface, normal, d, (u, v), noise, step);
                    if valley.len() >= 2 {
                        valleys.push(valley);
                    }
                }
                continue;
            }
            if centre * near >= 0.0 || loops.iter().any(|ring| encloses(ring, (u, v))) {
                continue;
            }
            // Near a settled extremum the distance is `centre + ½ xᵀHx`,
            // zero on the ellipse `xᵀHx = −2·centre`: rays out along it
            // start at the section and spread evenly round a long loop.
            let ellipse = settled && big * centre < 0.0 && small * centre < 0.0;
            let (ax, ay) = if fuv.abs() > 0.0 {
                let (x, y) = (big - fvv, fuv);
                let len = x.hypot(y);
                (x / len, y / len)
            } else if fuu >= fvv {
                (1.0, 0.0)
            } else {
                (0.0, 1.0)
            };
            let (r1, r2) = ((-2.0 * centre / big).sqrt(), (-2.0 * centre / small).sqrt());
            let mut chain = Vec::with_capacity(TINY_LOOP_POINTS + 1);
            for k in 0..TINY_LOOP_POINTS {
                let angle = std::f64::consts::TAU * k as f64 / TINY_LOOP_POINTS as f64;
                let (s, c) = angle.sin_cos();
                let (dir, start) = if ellipse {
                    let (x, y) = (r1 * c, r2 * s);
                    ((x * ax - y * ay, x * ay + y * ax), 1.0)
                } else {
                    ((u_step * c, v_step * s), 0.25)
                };
                let Some((pu, pv)) = crossing((u, v), centre, dir, start) else {
                    break;
                };
                chain.push(IntersectionPoint {
                    point: surface.evaluate(pu, pv),
                    param1: (pu, pv),
                    param2: (0.0, 0.0),
                });
            }
            if chain.len() < TINY_LOOP_POINTS {
                continue;
            }
            chain.push(chain[0]);
            let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
            for p in &chain {
                lo = (lo.0.min(p.param1.0), lo.1.min(p.param1.1));
                hi = (hi.0.max(p.param1.0), hi.1.max(p.param1.1));
            }
            let holds_node = (0..n).any(|a| {
                let nu = u_min + a as f64 * u_step;
                (lo.0..=hi.0).contains(&nu)
                    && (0..n).any(|b| {
                        let nv = v_min + b as f64 * v_step;
                        (lo.1..=hi.1).contains(&nv) && encloses(&chain, (nu, nv))
                    })
            });
            if !holds_node {
                loops.push(chain);
            }
        }
    }
    (loops, touches, valleys)
}

/// Steps [`trace_valley`] takes each way at most.
const MAX_VALLEY_STEPS: usize = 4096;

/// The curve along which the surface touches the plane, through `start` on
/// the floor of a valley of the plane's distance lying on the plane. It
/// steps along the valley's flat direction and settles back across the
/// valley each step, both ways until the floor leaves the plane or the
/// patch, or the curve comes round to `start`.
#[allow(clippy::similar_names)]
fn trace_valley(
    surface: &NurbsSurface,
    normal: Vec3,
    d: f64,
    start: (f64, f64),
    noise: f64,
    step: f64,
) -> Vec<IntersectionPoint> {
    let (u_lo, u_hi) = surface.domain_u();
    let (v_lo, v_hi) = surface.domain_v();
    let at = |(u, v): (f64, f64)| {
        let p = surface.evaluate(u, v);
        normal.dot(Vec3::new(p.x(), p.y(), p.z())) - d
    };
    let settle = |(mut u, mut v): (f64, f64)| {
        for _ in 0..MAX_NEWTON_ITER {
            let b = bend(surface, normal, (u, v));
            let ((lambda, e), grad) = (b.major, b.grad);
            if lambda == 0.0 {
                return None;
            }
            let s = grad.0.mul_add(e.0, grad.1 * e.1) / lambda;
            u = s.mul_add(-e.0, u).clamp(u_lo, u_hi);
            v = s.mul_add(-e.1, v).clamp(v_lo, v_hi);
            if s.abs() <= 1e-12 * step {
                break;
            }
        }
        (at((u, v)).abs() <= noise).then_some((u, v))
    };
    let flat = |q: (f64, f64)| bend(surface, normal, q).minor.1;
    let walk = |sign: f64, pts: &mut Vec<(f64, f64)>| {
        let (mut p, e) = (start, flat(start));
        let mut dir = (sign * e.0, sign * e.1);
        for _ in 0..MAX_VALLEY_STEPS {
            // As far as the patch allows, up to one step.
            let reach = |x: f64, dx: f64, lo: f64, hi: f64| {
                if dx > 0.0 {
                    (hi - x) / dx
                } else if dx < 0.0 {
                    (lo - x) / dx
                } else {
                    f64::INFINITY
                }
            };
            let t = step
                .min(reach(p.0, dir.0, u_lo, u_hi))
                .min(reach(p.1, dir.1, v_lo, v_hi));
            if t <= 0.0 {
                return false;
            }
            let Some(next) = settle((t.mul_add(dir.0, p.0), t.mul_add(dir.1, p.1))) else {
                return false;
            };
            if pts.len() > 2 && (next.0 - start.0).hypot(next.1 - start.1) < step {
                return true;
            }
            pts.push(next);
            if t < step {
                return false;
            }
            let e = flat(next);
            dir = if e.0.mul_add(dir.0, e.1 * dir.1) >= 0.0 {
                e
            } else {
                (-e.0, -e.1)
            };
            p = next;
        }
        false
    };
    let mut forward = Vec::new();
    let closed = walk(1.0, &mut forward);
    let mut params = Vec::with_capacity(forward.len() * 2 + 2);
    if closed {
        params.push(start);
        params.extend(forward);
        params.push(start);
    } else {
        let mut back = Vec::new();
        walk(-1.0, &mut back);
        params.extend(back.into_iter().rev());
        params.push(start);
        params.extend(forward);
    }
    // Settling can bring a step back onto the point before it, which the
    // fit cannot take twice.
    let mut points: Vec<IntersectionPoint> = Vec::with_capacity(params.len());
    for (u, v) in params {
        let point = surface.evaluate(u, v);
        if points
            .last()
            .is_none_or(|q| (q.point - point).length() > 1e-9 * step)
        {
            points.push(IntersectionPoint {
                point,
                param1: (u, v),
                param2: (0.0, 0.0),
            });
        }
    }
    points
}

/// How far evaluating the plane's distance on `surface` can round it: a few
/// ulps of the largest sum of its terms over the control points.
fn rounding(surface: &NurbsSurface, normal: Vec3, d: f64) -> f64 {
    4.0 * f64::EPSILON
        * surface
            .control_points()
            .iter()
            .flatten()
            .map(|&p| plane_terms(normal, d, p))
            .fold(0.0, f64::max)
}

/// The sum of the plane's distance's terms at `p`, `Σ|nᵢ·pᵢ| + |d|`, which
/// bounds how far evaluating it rounds: coordinates the normal does not
/// weigh add nothing.
fn plane_terms(normal: Vec3, d: f64, p: Point3) -> f64 {
    (normal.x() * p.x()).abs() + (normal.y() * p.y()).abs() + (normal.z() * p.z()).abs() + d.abs()
}

/// Where the plane's distance stops changing, as Newton from a seed
/// finds it on the patch.
struct Extremum {
    at: (f64, f64),
    /// Whether Newton's last step had shrunk to rounding.
    settled: bool,
    /// The distance's second derivatives `(fuu, fuv, fvv)` there.
    hessian: (f64, f64, f64),
}

/// The plane's distance's gradient and second derivatives at `(u, v)`,
/// its curvature's eigenvalues, the larger in size first, with their unit
/// eigenvectors.
struct Bend {
    grad: (f64, f64),
    hessian: (f64, f64, f64),
    major: (f64, (f64, f64)),
    minor: (f64, (f64, f64)),
}

fn bend(surface: &NurbsSurface, normal: Vec3, (u, v): (f64, f64)) -> Bend {
    let ders = surface.derivatives(u, v, 2);
    let (fu, fv) = (normal.dot(ders[1][0]), normal.dot(ders[0][1]));
    let (fuu, fuv, fvv) = (
        normal.dot(ders[2][0]),
        normal.dot(ders[1][1]),
        normal.dot(ders[0][2]),
    );
    let spread = (fuu - fvv).hypot(2.0 * fuv);
    let mean = 0.5 * (fuu + fvv);
    let (high, low) = (mean + 0.5 * spread, mean - 0.5 * spread);
    // An eigenvector of the 2 by 2 Hessian, from whichever of its rows
    // leaves the longer one.
    let along = |lambda: f64| {
        let (a, b) = ((lambda - fvv, fuv), (fuv, lambda - fuu));
        let (x, y) = if a.0.hypot(a.1) >= b.0.hypot(b.1) {
            a
        } else {
            b
        };
        let len = x.hypot(y);
        if len > 0.0 {
            (x / len, y / len)
        } else {
            (1.0, 0.0)
        }
    };
    let (major, minor) = if high.abs() >= low.abs() {
        (high, low)
    } else {
        (low, high)
    };
    let major_dir = along(major);
    Bend {
        grad: (fu, fv),
        hessian: (fuu, fuv, fvv),
        major: (major, major_dir),
        minor: (minor, (-major_dir.1, major_dir.0)),
    }
}

/// Where the plane's distance stops changing, by Newton from `seed` on the
/// patch. Along a direction where it does not curve the step is left out,
/// so along a valley Newton settles across it.
fn extremum(surface: &NurbsSurface, normal: Vec3, (mut u, mut v): (f64, f64)) -> Option<Extremum> {
    let (u_lo, u_hi) = surface.domain_u();
    let (v_lo, v_hi) = surface.domain_v();
    let mut found = None;
    for _ in 0..MAX_NEWTON_ITER {
        let b = bend(surface, normal, (u, v));
        if b.major.0 == 0.0 {
            break;
        }
        let (mut du, mut dv) = (0.0, 0.0);
        for (lambda, e) in [b.major, b.minor] {
            if lambda.abs() > 1e-9 * b.major.0.abs() {
                let s = b.grad.0.mul_add(e.0, b.grad.1 * e.1) / lambda;
                (du, dv) = (s.mul_add(e.0, du), s.mul_add(e.1, dv));
            }
        }
        u = (u - du).clamp(u_lo, u_hi);
        v = (v - dv).clamp(v_lo, v_hi);
        let settled = du.abs() <= 1e-12 * (u_hi - u_lo) && dv.abs() <= 1e-12 * (v_hi - v_lo);
        found = Some(Extremum {
            at: (u, v),
            settled,
            hessian: b.hessian,
        });
        if settled {
            break;
        }
    }
    found
}

/// Whether the loop of parameters `ring` winds round `(u, v)`.
fn encloses(ring: &[IntersectionPoint], (u, v): (f64, f64)) -> bool {
    let mut inside = false;
    for pair in ring.windows(2) {
        let ((ua, va), (ub, vb)) = (pair[0].param1, pair[1].param1);
        if (va > v) != (vb > v) && u < (v - va) / (vb - va) * (ub - ua) + ua {
            inside = !inside;
        }
    }
    inside
}

/// The parameter a boundary crossing holds fixed during refinement.
#[derive(Clone, Copy)]
pub(super) enum Pin {
    U,
    V,
}

/// Refine a plane-surface intersection point using Newton iteration; a
/// pinned parameter stays on its domain edge.
pub(super) fn refine_plane_surface_point(
    surface: &NurbsSurface,
    plane_normal: Vec3,
    plane_d: f64,
    (u_guess, v_guess): (f64, f64),
    pin: Option<Pin>,
) -> Option<IntersectionPoint> {
    let mut u = u_guess;
    let mut v = v_guess;
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();

    for _ in 0..MAX_NEWTON_ITER {
        let pt = surface.evaluate(u, v);
        let pt_vec = Vec3::new(pt.x(), pt.y(), pt.z());
        let f = plane_normal.dot(pt_vec) - plane_d;
        let on_section = || IntersectionPoint {
            point: pt,
            param1: (u, v),
            param2: (0.0, 0.0),
        };
        if f == 0.0 {
            return Some(on_section());
        }

        // Compute gradient of f w.r.t. (u, v).
        let derivs = surface.derivatives(u, v, 1);
        let du = derivs[1][0]; // dS/du
        let dv = derivs[0][1]; // dS/dv

        let grad_u = match pin {
            Some(Pin::U) => 0.0,
            _ => plane_normal.dot(du),
        };
        let grad_v = match pin {
            Some(Pin::V) => 0.0,
            _ => plane_normal.dot(dv),
        };

        let grad_len_sq = grad_u.mul_add(grad_u, grad_v * grad_v);
        let rounding = f64::EPSILON * plane_normal.length() * (du.length() + dv.length());
        if grad_len_sq <= rounding * rounding {
            break; // Singular.
        }

        // Where the surface runs nearly along the plane, a point far from
        // the section still lies within 1e-12 of the plane: the step left
        // to take says how far off it is.
        let step_size = f / grad_len_sq;
        let left = (du * grad_u + dv * grad_v).length() * step_size.abs();
        if f.abs() < 1e-12 && left < 1e-12 {
            return Some(on_section());
        }

        // Steepest descent step.
        u -= grad_u * step_size;
        v -= grad_v * step_size;

        u = u.clamp(u_min, u_max);
        v = v.clamp(v_min, v_max);
    }

    // Final check.
    let pt = surface.evaluate(u, v);
    let pt_vec = Vec3::new(pt.x(), pt.y(), pt.z());
    let f = plane_normal.dot(pt_vec) - plane_d;

    if f.abs() < 1e-6 {
        Some(IntersectionPoint {
            point: pt,
            param1: (u, v),
            param2: (0.0, 0.0),
        })
    } else {
        None
    }
}
