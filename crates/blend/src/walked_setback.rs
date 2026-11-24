//! Walked rounds set back to their common ball at a convex corner.
//!
//! Three rounds of one radius meeting at a convex corner of three faces
//! share a ball touching all three faces: each round's ball rolls between
//! two of them, so at the corner the three balls are one. A round of a
//! straight edge between planes finds it from the planes' offsets
//! (`corner::convex_trihedral_setback`); a walked round, between curved
//! faces, has no closed form. Its ball is found from the rounds' walked
//! sections (the closest approach of two of their centrelines) and then
//! solved on the faces themselves, so that it lies at the radius from each.
//! Each round is cut back to its section through that ball: its sections
//! past it dropped, the exact section put in their place, and its band,
//! contacts and pcurves rebuilt through them. The corner is then a patch of
//! the ball's sphere, bounded by the three end sections, which are great
//! circles of it.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::BlendError;
use crate::builder_utils::surface_ref_or_adapter;
use crate::section::CircSection;
use crate::stripe::Stripe;

/// How near the solved ball's centre must lie to a round's walked
/// centreline, as a share of the radius: the polyline through the walked
/// centres sags off the true centreline between them.
const ON_CENTRELINE: f64 = 1e-3;

/// The foot of `point` on `face`'s surface.
fn foot(topo: &Topology, face: FaceId, point: Point3) -> Result<(Point3, (f64, f64)), BlendError> {
    let surface = topo.face(face)?.surface().clone();
    if let FaceSurface::Plane { normal, d } = surface {
        let off = normal.dot(Vec3::new(point.x(), point.y(), point.z())) - d;
        let at = point - normal * off;
        let mut adapter = None;
        let ps = surface_ref_or_adapter(&surface, &mut adapter);
        return Ok((at, ps.project_point(at)));
    }
    let mut adapter = None;
    let ps = surface_ref_or_adapter(&surface, &mut adapter);
    let (u, v) = ps.project_point(point);
    Ok((ps.evaluate(u, v), (u, v)))
}

/// The closest points of segments `a0 a1` and `b0 b1`.
fn segments_closest(a0: Point3, a1: Point3, b0: Point3, b1: Point3) -> (Point3, Point3) {
    let (u, v, w) = (a1 - a0, b1 - b0, a0 - b0);
    let (uu, uv, vv, uw, vw) = (u.dot(u), u.dot(v), v.dot(v), u.dot(w), v.dot(w));
    let den = uu * vv - uv * uv;
    let mut s = if den > 1e-300 {
        ((uv * vw - vv * uw) / den).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = if vv > 1e-300 {
        ((uv * s + vw) / vv).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if uu > 1e-300 {
        s = ((uv * t - uw) / uu).clamp(0.0, 1.0);
    }
    if vv > 1e-300 {
        t = ((uv * s + vw) / vv).clamp(0.0, 1.0);
    }
    (a0 + u * s, b0 + v * t)
}

/// Where `point` lies along the polyline through `centres`: the segment, the
/// fraction along it, and how far off it.
fn along_centres(centres: &[Point3], point: Point3) -> Option<(usize, f64, f64)> {
    let mut best: Option<(usize, f64, f64)> = None;
    for (k, pair) in centres.windows(2).enumerate() {
        let d = pair[1] - pair[0];
        let len2 = d.dot(d);
        if len2 <= 1e-300 {
            continue;
        }
        let f = ((point - pair[0]).dot(d) / len2).clamp(0.0, 1.0);
        let off = (pair[0] + d * f - point).length();
        if best.is_none_or(|(_, _, b)| off < b) {
            best = Some((k, f, off));
        }
    }
    best
}

/// The ball of `radius` touching all three `faces`, from a first guess.
fn solve_ball(
    topo: &Topology,
    faces: &[FaceId; 3],
    radius: f64,
    guess: Point3,
) -> Result<Option<Point3>, BlendError> {
    let mut centre = guess;
    for _ in 0..50 {
        let mut rows = [[0.0; 3]; 3];
        let mut miss = [0.0; 3];
        for (row, &face) in faces.iter().enumerate() {
            let (at, _) = foot(topo, face, centre)?;
            let off = centre - at;
            let Ok(n) = off.normalize() else {
                return Ok(None);
            };
            rows[row] = [n.x(), n.y(), n.z()];
            miss[row] = off.length() - radius;
        }
        let Some(inv) = crate::corner::inverse_3x3(rows) else {
            return Ok(None);
        };
        let step = Vec3::new(
            inv[0][0] * miss[0] + inv[0][1] * miss[1] + inv[0][2] * miss[2],
            inv[1][0] * miss[0] + inv[1][1] * miss[1] + inv[1][2] * miss[2],
            inv[2][0] * miss[0] + inv[2][1] * miss[1] + inv[2][2] * miss[2],
        );
        centre = centre - step;
        if step.length() <= 1e-12 * radius.max(1.0) {
            let worst = miss.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            return Ok((worst <= 1e-9 * radius.max(1.0)).then_some(centre));
        }
    }
    Ok(None)
}

/// The ball centre of the walked rounds `indices` at a convex corner whose
/// three faces are `faces`. `None` when the rounds' centrelines do not meet
/// within their walks.
pub fn walked_ball(
    topo: &Topology,
    indices: &[usize],
    stripes: &[Stripe],
    faces: &[FaceId; 3],
    radius: f64,
) -> Result<Option<Point3>, BlendError> {
    let centres: Vec<Vec<Point3>> = indices
        .iter()
        .map(|&i| stripes[i].sections.iter().map(|s| s.center).collect())
        .collect();
    // The closest approach of the first two rounds' walked centrelines.
    let mut guess: Option<(f64, Point3)> = None;
    for a in centres[0].windows(2) {
        for b in centres[1].windows(2) {
            let (p, q) = segments_closest(a[0], a[1], b[0], b[1]);
            let gap = (p - q).length();
            if guess.is_none_or(|(g, _)| gap < g) {
                guess = Some((gap, p + (q - p) * 0.5));
            }
        }
    }
    let Some((_, guess)) = guess else {
        return Ok(None);
    };
    let Some(centre) = solve_ball(topo, faces, radius, guess)? else {
        log::debug!("walked setback: no ball touching all three faces");
        return Ok(None);
    };
    for line in &centres {
        let Some((k, f, off)) = along_centres(line, centre) else {
            return Ok(None);
        };
        // Inside the walk, not at either end.
        let first = k == 0 && f <= 1e-9;
        let last = k + 2 == line.len() && f >= 1.0 - 1e-9;
        if off > ON_CENTRELINE * radius || first || last {
            log::debug!("walked setback: the ball lies {off} off a round's centreline");
            return Ok(None);
        }
    }
    Ok(Some(centre))
}

/// `stripe` cut back to its section through `centre`, keeping the side away
/// from its start (`from_start`) or its end; its band, contacts and pcurves
/// rebuilt through the sections kept.
pub fn cut_back(
    topo: &Topology,
    stripe: &Stripe,
    centre: Point3,
    from_start: bool,
) -> Result<Stripe, BlendError> {
    let sections = &stripe.sections;
    let line: Vec<Point3> = sections.iter().map(|s| s.center).collect();
    let Some((k, f, _)) = along_centres(&line, centre) else {
        return Err(BlendError::PlanningFailure {
            reason: "a walked setback's ball is off its round".to_owned(),
        });
    };
    let (p1, uv1) = foot(topo, stripe.face1, centre)?;
    let (p2, uv2) = foot(topo, stripe.face2, centre)?;
    let exact = CircSection {
        p1,
        p2,
        center: centre,
        radius: sections[k].radius,
        uv1,
        uv2,
        t: sections[k].t + (sections[k + 1].t - sections[k].t) * f,
    };
    // A walked section nearer the cut than a hundredth of its step goes:
    // the band would fold between the two.
    let step = (sections[k + 1].center - sections[k].center).length();
    let near = |s: &CircSection| (s.center - centre).length() <= 1e-2 * step;
    let mut kept: Vec<CircSection> = if from_start {
        std::iter::once(exact)
            .chain(sections[k + 1..].iter().filter(|s| !near(s)).cloned())
            .collect()
    } else {
        sections[..=k]
            .iter()
            .filter(|s| !near(s))
            .cloned()
            .chain(std::iter::once(exact))
            .collect()
    };
    if kept.len() < 2 {
        return Err(BlendError::PlanningFailure {
            reason: "a walked setback leaves a round no sections".to_owned(),
        });
    }
    kept.dedup_by(|a, b| (a.center - b.center).length() <= 1e-12);
    let joins = band_joins(stripe);
    let surface = rebuilt_band(&kept, &joins)?;
    let contact1 = crate::fillet_builder::band_side(&surface, 0)?;
    let contact2 = crate::fillet_builder::band_side(&surface, surface.control_points().len() - 1)?;
    let surfaces = [
        topo.face(stripe.face1)?.surface().clone(),
        topo.face(stripe.face2)?.surface().clone(),
    ];
    let (mut adapter1, mut adapter2) = (None, None);
    let pcurve1 = crate::fillet_builder::build_pcurve_from_contact(
        surface_ref_or_adapter(&surfaces[0], &mut adapter1),
        &contact1,
        crate::fillet_builder::face_u_period(&surfaces[0]),
    )?;
    let pcurve2 = crate::fillet_builder::build_pcurve_from_contact(
        surface_ref_or_adapter(&surfaces[1], &mut adapter2),
        &contact2,
        crate::fillet_builder::face_u_period(&surfaces[1]),
    )?;
    let mut cut = stripe.clone();
    cut.surface = FaceSurface::Nurbs(surface);
    cut.contact1 = contact1;
    cut.contact2 = contact2;
    cut.pcurve1 = pcurve1;
    cut.pcurve2 = pcurve2;
    cut.sections = kept;
    // Nothing rolls on past an end cut back to the corner's ball.
    cut.beyond[usize::from(!from_start)].clear();
    Ok(cut)
}

/// The sections' first contacts at a band's joins: a band built in pieces
/// end to end (about a contact's crossing cut out) has a knot of its degree's
/// multiplicity at each join, where the piece's end section lies.
fn band_joins(stripe: &Stripe) -> Vec<Point3> {
    let FaceSurface::Nurbs(band) = &stripe.surface else {
        return Vec::new();
    };
    let degree = band.degree_v();
    let knots = band.knots_v();
    let (lo, hi) = (knots[0], knots[knots.len() - 1]);
    let mut joins = Vec::new();
    let mut k = degree + 1;
    while k < knots.len() - degree - 1 {
        let mut m = 1;
        while k + m < knots.len() && (knots[k + m] - knots[k]).abs() <= 1e-12 {
            m += 1;
        }
        if m >= degree && knots[k] > lo && knots[k] < hi {
            let (u0, _) = band.domain_u();
            joins.push(band.evaluate(u0, knots[k]));
        }
        k += m;
    }
    joins
}

/// A band through `sections`, in pieces at the sections whose first contact
/// is one of `joins` (as the round was built), else in one.
fn rebuilt_band(
    sections: &[CircSection],
    joins: &[Point3],
) -> Result<brepkit_math::nurbs::surface::NurbsSurface, BlendError> {
    let tol = 1e-7;
    let mut bounds: Vec<usize> = sections
        .iter()
        .enumerate()
        .filter(|(k, s)| {
            *k > 0 && *k + 1 < sections.len() && joins.iter().any(|j| (*j - s.p1).length() <= tol)
        })
        .map(|(k, _)| k)
        .collect();
    bounds.dedup();
    if bounds.is_empty() {
        return crate::walker::interpolate_blend_surface(sections)
            .or_else(|_| crate::walker::approximate_blend_surface(sections));
    }
    let mut cuts = vec![0];
    cuts.extend(bounds);
    cuts.push(sections.len() - 1);
    let mut bands = Vec::with_capacity(cuts.len() - 1);
    let mut lengths = Vec::with_capacity(cuts.len() - 1);
    for pair in cuts.windows(2) {
        let piece = &sections[pair[0]..=pair[1]];
        bands.push(crate::walker::interpolate_blend_surface(piece)?);
        lengths.push(
            piece
                .windows(2)
                .map(|w| {
                    (w[1].p1 - w[0].p1)
                        .length()
                        .max((w[1].p2 - w[0].p2).length())
                })
                .sum::<f64>(),
        );
    }
    crate::fillet_builder::join_bands(&bands, &lengths)
}
