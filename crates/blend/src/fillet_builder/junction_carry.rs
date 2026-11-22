//! A walked round carried on to the plane wall of the round it meets.
//!
//! Two rounds meet at a corner of three faces whose third edge is left
//! sharp: round `A`, a walked band between the face `F` they share and a
//! wall `WA`, and round `B` between `F` and a plane wall `WB`. Where `WA`
//! leans in over `F` (the corner between them acute), `A`'s section at the
//! corner stops short of `WB`'s plane on `WA`: `A` reaches farther along
//! `WB` than `B`'s strip, and the two meet along a crease, `A` running on to
//! `WB` and notching it, `B` ending on `A`.
//!
//! This carries `A` on past its end through the sections the ball rolls
//! through there ([`Stripe::beyond`]) up to the one whose contact on `WA`
//! lies on `WB`'s plane, on the third edge, which is solved for exactly; the
//! band runs on a section past it, its contacts ending there. The crossing
//! is then built on the trimmed faces
//! ([`crate::corner::apply_junction_crossings`]).

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::BlendError;
use crate::fillet_plan::{CornerClassification, FilletPlan};
use crate::section::CircSection;
use crate::stripe::{Stripe, StripeResult};
use crate::walker::interpolate_blend_surface;

use super::{band_side, outward_plane, spine_ends_at};

/// A surface point: its parameters (none on a plane) and the point.
type Foot = ((f64, f64), Point3);

/// The face two stripes share, then each one's other face.
fn shared_and_walls(a: &Stripe, b: &Stripe) -> Option<(FaceId, FaceId, FaceId)> {
    let f = [a.face1, a.face2]
        .into_iter()
        .find(|&face| face == b.face1 || face == b.face2)?;
    let other = |s: &Stripe| if s.face1 == f { s.face2 } else { s.face1 };
    let (wa, wb) = (other(a), other(b));
    (wa != wb && wa != f && wb != f).then_some((f, wa, wb))
}

/// The ball of `radius` resting on `f` and `wall` whose contact on `wall`
/// lies on the plane `(n, level)`, from `seed`: its centre found by Newton
/// on the two distances and the contact's offset from the plane.
fn ball_on_plane(
    f: &FaceSurface,
    wall: &FaceSurface,
    radius: f64,
    (n, level): (Vec3, f64),
    seed: Point3,
) -> Option<(Point3, Foot, Foot)> {
    let foot = |surface: &FaceSurface, c: Point3| -> Option<Foot> {
        if let FaceSurface::Plane { normal, d } = surface {
            let unit = normal.normalize().ok()?;
            let off = unit.dot(Vec3::new(c.x(), c.y(), c.z())) - d / normal.length();
            return Some(((f64::NAN, f64::NAN), c - unit * off));
        }
        let uv = surface.project_point(c)?;
        Some((uv, surface.evaluate(uv.0, uv.1)?))
    };
    let to = |p: Point3| Vec3::new(p.x(), p.y(), p.z());
    let residual = |c: Point3| -> Option<[f64; 3]> {
        let (_, on_f) = foot(f, c)?;
        let (_, on_w) = foot(wall, c)?;
        Some([
            (c - on_f).length() - radius,
            (c - on_w).length() - radius,
            n.dot(to(on_w)) - level,
        ])
    };
    let tol = 1e-12 * radius.max(1.0);
    let h = 1e-7 * radius.max(1.0);
    let mut c = seed;
    for _ in 0..50 {
        let g = residual(c)?;
        if g.iter().all(|x| x.abs() <= tol) {
            return Some((c, foot(f, c)?, foot(wall, c)?));
        }
        let mut jac = [[0.0; 3]; 3];
        for (k, dir) in [
            Vec3::new(h, 0.0, 0.0),
            Vec3::new(0.0, h, 0.0),
            Vec3::new(0.0, 0.0, h),
        ]
        .into_iter()
        .enumerate()
        {
            let gp = residual(c + dir)?;
            for row in 0..3 {
                jac[row][k] = (gp[row] - g[row]) / h;
            }
        }
        let step = solve3(jac, [-g[0], -g[1], -g[2]])?;
        let step = Vec3::new(step[0], step[1], step[2]);
        if step.length() > radius {
            return None;
        }
        c = c + step;
    }
    None
}

/// The parameters at `point` of the affine frame that carries each of
/// `known`'s three points to its parameters, in the plane they span.
fn plane_uv(known: &[(Point3, (f64, f64))], point: Point3) -> Option<(f64, f64)> {
    let [(p0, a0), (p1, a1), (p2, a2)] = known else {
        return None;
    };
    let (e1, e2) = (*p1 - *p0, *p2 - *p0);
    let g = [[e1.dot(e1), e1.dot(e2)], [e1.dot(e2), e2.dot(e2)]];
    let det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
    if det.abs() <= 1e-12 * g[0][0].max(g[1][1]).powi(2).max(1e-300) {
        return None;
    }
    let w = point - *p0;
    let (b1, b2) = (w.dot(e1), w.dot(e2));
    let s = (b1 * g[1][1] - b2 * g[0][1]) / det;
    let t = (b2 * g[0][0] - b1 * g[1][0]) / det;
    Some((
        a0.0 + s * (a1.0 - a0.0) + t * (a2.0 - a0.0),
        a0.1 + s * (a1.1 - a0.1) + t * (a2.1 - a0.1),
    ))
}

/// `m x = b` by Cramer's rule; `None` when `m` is singular.
fn solve3(m: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(m);
    let scale = m.iter().flatten().fold(0.0_f64, |a, x| a.max(x.abs()));
    if d.abs() <= 1e-14 * scale.powi(3).max(1e-300) {
        return None;
    }
    let mut out = [0.0; 3];
    for (k, slot) in out.iter_mut().enumerate() {
        let mut mk = m;
        for row in 0..3 {
            mk[row][k] = b[row];
        }
        *slot = det(mk) / d;
    }
    Some(out)
}

/// Carry each walked round meeting a round along a plane wall at a corner,
/// short of that wall, on to it (see the module docs). A round whose ball
/// never reaches the wall, or whose last section cannot be solved, is left
/// as it was.
pub(super) fn carry_walked_rounds_to_junction_walls(
    topo: &Topology,
    plan: &FilletPlan,
    stripe_results: &mut [StripeResult],
) -> Result<(), BlendError> {
    for junction in &plan.junctions {
        if junction.classification != CornerClassification::Junction
            || junction.unselected_sharp_edges.len() != 1
            || junction.face_fan.len() != 3
        {
            continue;
        }
        let indices: Vec<usize> = junction
            .incident_contours
            .iter()
            .filter_map(|&c| {
                let contour = plan.contours.get(c)?;
                stripe_results
                    .iter()
                    .position(|r| r.stripe.spine_edges() == contour.spine.edges())
            })
            .collect();
        let &[i, j] = indices.as_slice() else {
            continue;
        };
        for (ia, ib) in [(i, j), (j, i)] {
            let (a, b) = (&stripe_results[ia].stripe, &stripe_results[ib].stripe);
            if !matches!(a.surface, FaceSurface::Nurbs(_)) {
                continue;
            }
            let Some((f, wa, wb)) = shared_and_walls(a, b) else {
                continue;
            };
            let Some(plane) = outward_plane(topo, wb)? else {
                continue;
            };
            let vertex = topo.vertex(junction.vertex)?.point();
            let end = usize::from(spine_ends_at(topo, a, vertex)?);
            let Some(section) = (if end == 1 {
                a.sections.last()
            } else {
                a.sections.first()
            }) else {
                continue;
            };
            let on_wall = |s: &CircSection| if a.face1 == wa { s.p1 } else { s.p2 };
            let offset = |p: Point3| plane.0.dot(Vec3::new(p.x(), p.y(), p.z())) - plane.1;
            let tol = 1e-7 * section.radius.max(1.0);
            let start = offset(on_wall(section));
            if start >= -tol {
                continue;
            }
            let past = &a.beyond[end];
            let Some(k) = past.iter().position(|s| offset(on_wall(s)) >= 0.0) else {
                continue;
            };
            let before = if k == 0 { section } else { &past[k - 1] };
            let (o0, o1) = (offset(on_wall(before)), offset(on_wall(&past[k])));
            let w = (-o0 / (o1 - o0)).clamp(0.0, 1.0);
            let seed = before.center + (past[k].center - before.center) * w;
            let (fs, ws) = (
                topo.face(f)?.surface().clone(),
                topo.face(wa)?.surface().clone(),
            );
            let Some((centre, (uv_f, p_f), (uv_w, p_w))) =
                ball_on_plane(&fs, &ws, section.radius, plane, seed)
            else {
                continue;
            };
            // A plane has no parameters of its own: the walk's are an affine
            // frame on it, read off the sections before.
            let from_sections = |side_f: bool, point: Point3, uv: (f64, f64)| {
                if !uv.0.is_nan() {
                    return Some(uv);
                }
                let picks: Vec<&CircSection> = a
                    .sections
                    .iter()
                    .chain(past[..k].iter())
                    .rev()
                    .take(3)
                    .collect();
                let side = |s: &CircSection| {
                    let on_f = if a.face1 == f {
                        (s.p1, s.uv1)
                    } else {
                        (s.p2, s.uv2)
                    };
                    let on_w = if a.face1 == f {
                        (s.p2, s.uv2)
                    } else {
                        (s.p1, s.uv1)
                    };
                    if side_f { on_f } else { on_w }
                };
                plane_uv(&picks.iter().map(|s| side(s)).collect::<Vec<_>>(), point)
            };
            let (Some(uv_f), Some(uv_w)) = (
                from_sections(true, p_f, uv_f),
                from_sections(false, p_w, uv_w),
            ) else {
                continue;
            };
            let (p1, p2, uv1, uv2) = if a.face1 == f {
                (p_f, p_w, uv_f, uv_w)
            } else {
                (p_w, p_f, uv_w, uv_f)
            };
            let last = CircSection {
                p1,
                p2,
                center: centre,
                radius: section.radius,
                uv1,
                uv2,
                t: section.t,
            };
            let mut carried_on: Vec<CircSection> = past[..k].to_vec();
            carried_on.push(last.clone());
            let mut sections = Vec::with_capacity(a.sections.len() + carried_on.len());
            if end == 0 {
                sections.extend(carried_on.iter().rev().cloned());
                sections.extend(a.sections.iter().cloned());
            } else {
                sections.extend(a.sections.iter().cloned());
                sections.extend(carried_on.iter().cloned());
            }
            // The band runs on a section past the last one, so that the
            // crease with the other round, which meets this round's last
            // section where that section touches it, lies inside the band;
            // the contacts end at the last section.
            let Some(on) = past[k..]
                .iter()
                .find(|s| (s.center - last.center).length() > 0.1 * last.radius)
            else {
                continue;
            };
            let mut through = sections.clone();
            if end == 0 {
                through.insert(0, on.clone());
            } else {
                through.push(on.clone());
            }
            let Ok(surface) = interpolate_blend_surface(&through) else {
                continue;
            };
            let cut = |row: usize, at: Point3| -> Result<Option<_>, BlendError> {
                let side = band_side(&surface, row)?;
                let found =
                    brepkit_math::nurbs::projection::project_point_to_curve(&side, at, 1e-12)?;
                if found.distance > tol {
                    return Ok(None);
                }
                let (low, high) =
                    brepkit_math::nurbs::knot_ops::curve_split(&side, found.parameter)?;
                Ok(Some(if end == 0 { high } else { low }))
            };
            let rows = surface.control_points().len();
            let (Some(contact1), Some(contact2)) = (cut(0, last.p1)?, cut(rows - 1, last.p2)?)
            else {
                continue;
            };
            let mut carried = a.clone();
            carried.contact1 = contact1;
            carried.contact2 = contact2;
            carried.sections = sections;
            carried.surface = FaceSurface::Nurbs(surface);
            log::debug!(
                "junction {:?}: walked round carried on to the wall of the round it meets",
                junction.vertex
            );
            stripe_results[ia].stripe = carried;
        }
    }
    Ok(())
}
