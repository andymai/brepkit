//! Plane-NURBS surface intersection.

use crate::MathError;
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

    #[allow(clippy::cast_precision_loss)]
    for (i, row) in distances.iter_mut().enumerate() {
        let u = u_min + i as f64 * u_step;
        for (j, dist) in row.iter_mut().enumerate() {
            let v = v_min + j as f64 * v_step;
            let pt = surface.evaluate(u, v);
            let pt_vec = Vec3::new(pt.x(), pt.y(), pt.z());
            *dist = plane_normal.dot(pt_vec) - plane_d;
        }
    }

    // Phase 2: Find zero-crossings on the UNIQUE grid edges (marching
    // squares). Every crossing is computed once and remembers which edge it
    // sits on, so connectivity comes from cell adjacency in phase 3 — not
    // from a distance threshold. Proximity chaining cannot serve this scan:
    // its threshold has no way to tell a sampling gap along one branch from
    // two genuinely-close branches (the kumiko strut end-patch dashes vs the
    // honeycomb thin-wall branches), and each statistic that fixes one side
    // breaks the other. Cell connectivity satisfies both by construction.
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
                if da * db < 0.0 {
                    let t = da / (da - db);
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
                if da * db < 0.0 {
                    let t = da / (da - db);
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

    let tiny = tiny_loops(
        surface,
        plane_normal,
        plane_d,
        &distances,
        (u_min, u_step),
        (v_min, v_step),
    );
    if crossings.is_empty() && tiny.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 3: Link crossings through each cell. A cell with two crossing
    // edges connects them; a saddle cell (all four edges crossing) pairs
    // them by the sign of the cell-center sample. Odd counts (a crossing
    // through a grid node) leave that cell unlinked — the chain simply ends
    // there.
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
                    let pc = surface.evaluate(uc, vc);
                    let dc = plane_normal.dot(Vec3::new(pc.x(), pc.y(), pc.z())) - plane_d;
                    let d00 = distances[i][j];
                    let (b, t, l, r) = (
                        bottom.unwrap_or(usize::MAX),
                        top.unwrap_or(usize::MAX),
                        left.unwrap_or(usize::MAX),
                        right.unwrap_or(usize::MAX),
                    );
                    if dc * d00 >= 0.0 {
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

    // Phase 4: Walk the links into ordered chains (open chains from
    // degree-1 endpoints first, then closed loops), Newton-refining each
    // crossing; a refinement failure splits its chain.
    let mut visited = vec![false; crossings.len()];
    let mut ordered_chains: Vec<Vec<IntersectionPoint>> = Vec::new();
    let walk = |start: usize, visited: &mut Vec<bool>| -> Vec<usize> {
        let mut chain = vec![start];
        visited[start] = true;
        let mut current = start;
        while let Some(&next) = links[current].iter().find(|&&x| !visited[x]) {
            visited[next] = true;
            chain.push(next);
            current = next;
        }
        // A loop ends beside its start; closing it keeps the cell between
        // them, and the curve, unbroken.
        if chain.len() > 2 && links[current].contains(&start) {
            chain.push(start);
        }
        chain
    };
    let degree_one: Vec<usize> = (0..crossings.len())
        .filter(|&x| links[x].len() == 1)
        .collect();
    let mut index_chains: Vec<Vec<usize>> = Vec::new();
    for start in degree_one {
        if !visited[start] {
            index_chains.push(walk(start, &mut visited));
        }
    }
    for start in 0..crossings.len() {
        if !visited[start] {
            index_chains.push(walk(start, &mut visited));
        }
    }
    for chain in index_chains {
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

    if ordered_chains.is_empty() && tiny.is_empty() {
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
    // A tiny loop's points lie evenly round it, so none is one met twice,
    // however close: the chains' merge of points within 1e-6 would fold a
    // loop smaller than that into a point.
    for mut chain in tiny {
        let curve = fit_within(&mut chain, FIT_TOLERANCE, &section)?;
        curves.push(IntersectionCurve {
            curve,
            points: chain,
        });
    }
    Ok(curves)
}

/// How far a fitted section may stray from the plane or the surface.
const FIT_TOLERANCE: f64 = 1e-7;

/// Points round each loop [`tiny_loops`] seeds from.
const TINY_LOOP_POINTS: usize = 32;

/// Closed sections inside one grid cell, which cross no grid edge. Each
/// winds round an extremum of the plane's distance where the distance
/// changes sign between nodes. A node holding the least distance of its
/// 3 by 3 block, all of one sign, seeds Newton for the extremum; where the
/// distance there has the other sign, points round the ellipse that its
/// second derivatives give settle onto the section, closing the loop.
#[allow(clippy::cast_precision_loss)]
fn tiny_loops(
    surface: &NurbsSurface,
    normal: Vec3,
    d: f64,
    distances: &[Vec<f64>],
    (u_min, u_step): (f64, f64),
    (v_min, v_step): (f64, f64),
) -> Vec<Vec<IntersectionPoint>> {
    let n = distances.len();
    let at = |u: f64, v: f64| {
        let p = surface.evaluate(u, v);
        normal.dot(Vec3::new(p.x(), p.y(), p.z())) - d
    };
    let mut loops: Vec<Vec<IntersectionPoint>> = Vec::new();
    let mut centres: Vec<(f64, f64)> = Vec::new();
    for i in 1..n.saturating_sub(1) {
        for j in 1..n.saturating_sub(1) {
            let near = distances[i][j];
            let block_holds = (i - 1..=i + 1)
                .flat_map(|a| (j - 1..=j + 1).map(move |b| (a, b)))
                .all(|(a, b)| distances[a][b] * near > 0.0 && distances[a][b].abs() >= near.abs());
            if !block_holds {
                continue;
            }
            let (u0, v0) = (u_min + i as f64 * u_step, v_min + j as f64 * v_step);
            let (mut u, mut v) = (u0, v0);
            let mut hessian = None;
            for _ in 0..MAX_NEWTON_ITER {
                let ders = surface.derivatives(u, v, 2);
                let (fu, fv) = (normal.dot(ders[1][0]), normal.dot(ders[0][1]));
                let (fuu, fuv, fvv) = (
                    normal.dot(ders[2][0]),
                    normal.dot(ders[1][1]),
                    normal.dot(ders[0][2]),
                );
                let det = fuu.mul_add(fvv, -(fuv * fuv));
                if det.abs() <= f64::EPSILON * (fuu.abs() + fvv.abs()).powi(2) {
                    hessian = None;
                    break;
                }
                let du = fvv.mul_add(fu, -(fuv * fv)) / det;
                let dv = fuu.mul_add(fv, -(fuv * fu)) / det;
                u = (u - du).clamp(u0 - u_step, u0 + u_step);
                v = (v - dv).clamp(v0 - v_step, v0 + v_step);
                hessian = Some((fuu, fuv, fvv));
                if du.abs() <= 1e-15 * u_step && dv.abs() <= 1e-15 * v_step {
                    break;
                }
            }
            let Some((fuu, fuv, fvv)) = hessian else {
                continue;
            };
            let centre = at(u, v);
            if centre * near >= 0.0
                || centres
                    .iter()
                    .any(|&(a, b)| (a - u).abs() <= 1e-9 * u_step && (b - v).abs() <= 1e-9 * v_step)
            {
                continue;
            }
            // Near the extremum the distance is `centre + ½ xᵀHx`, zero on
            // the ellipse `xᵀHx = −2·centre` about its principal axes.
            let spread = (fuu - fvv).hypot(2.0 * fuv);
            let (big, small) = (0.5 * (fuu + fvv + spread), 0.5 * (fuu + fvv - spread));
            if big * centre >= 0.0 || small * centre >= 0.0 {
                continue;
            }
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
                let (x, y) = (r1 * c, r2 * s);
                let guess = (u + x * ax - y * ay, v + x * ay + y * ax);
                if (guess.0 - u0).abs() > u_step || (guess.1 - v0).abs() > v_step {
                    break;
                }
                match refine_plane_surface_point(surface, normal, d, guess, None) {
                    Some(point) => chain.push(point),
                    None => break,
                }
            }
            if chain.len() == TINY_LOOP_POINTS {
                chain.push(chain[0]);
                loops.push(chain);
                centres.push((u, v));
            }
        }
    }
    loops
}

/// The parameter a boundary crossing holds fixed during refinement.
#[derive(Clone, Copy)]
enum Pin {
    U,
    V,
}

/// Refine a plane-surface intersection point using Newton iteration; a
/// pinned parameter stays on its domain edge.
fn refine_plane_surface_point(
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

        if f.abs() < 1e-12 {
            return Some(IntersectionPoint {
                point: pt,
                param1: (u, v),
                param2: (0.0, 0.0),
            });
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
        if grad_len_sq < 1e-20 {
            break; // Singular.
        }

        // Steepest descent step.
        let step_size = f / grad_len_sq;
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
