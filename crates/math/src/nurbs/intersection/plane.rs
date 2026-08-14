//! Plane-NURBS surface intersection.

use crate::MathError;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::Vec3;

use super::chaining::build_curves_from_chains;
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
                    crossings.push((u, v0));
                }
            }
            if j + 1 < n {
                let (da, db) = (distances[i][j], distances[i][j + 1]);
                if da * db < 0.0 {
                    let t = da / (da - db);
                    let v = v0.mul_add(1.0 - t, (v0 + v_step) * t);
                    vert[i][j] = Some(crossings.len());
                    crossings.push((u0, v));
                }
            }
        }
    }

    if crossings.is_empty() {
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
            if let Some(refined) =
                refine_plane_surface_point(surface, plane_normal, plane_d, u_guess, v_guess)
            {
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

    if ordered_chains.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 5: Fit a curve through each ordered chain.
    build_curves_from_chains(&ordered_chains)
}

/// Refine a plane-surface intersection point using Newton iteration.
fn refine_plane_surface_point(
    surface: &NurbsSurface,
    plane_normal: Vec3,
    plane_d: f64,
    u_guess: f64,
    v_guess: f64,
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

        let grad_u = plane_normal.dot(du);
        let grad_v = plane_normal.dot(dv);

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
