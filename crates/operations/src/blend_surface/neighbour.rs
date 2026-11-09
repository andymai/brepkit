//! The surface beside a chain: its fundamental forms at a point, and which
//! side of a boundary edge its face lies on.

use std::cell::Cell;

use brepkit_math::frame::Frame3;
use brepkit_math::nurbs::projection::{project_point_to_surface, project_point_to_surface_from};
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::wire::WireId;

use super::chain::Chain;
use crate::OperationsError;

const TAU: f64 = std::f64::consts::TAU;

/// A point this far from a face's surface is not on it, as a share of the
/// chain's length; the band has to pass a fitted intersection edge.
const OFF_SURFACE: f64 = 1e-3;

/// `S_u x S_v` smaller than this share of `|S_u| |S_v|` is a parametric
/// singularity (a pole, an apex).
const SINGULAR: f64 = 1e-8;

/// Off-direction angle cosine at a join past which a chain is doubling back.
const DOUBLES_BACK: f64 = -0.97;

/// A normal turning more than this at a break is a crease in the surface.
const CREASE: f64 = 1e-3;

/// The value and first and second partials of a surface at a parameter.
#[derive(Debug, Clone, Copy)]
pub(super) struct Jet {
    pub point: Point3,
    pub su: Vec3,
    pub sv: Vec3,
    pub suu: Vec3,
    pub suv: Vec3,
    pub svv: Vec3,
}

/// A surface's two fundamental forms at a point.
#[derive(Debug, Clone, Copy)]
pub(super) struct Form {
    /// `S_u x S_v` normalised.
    pub normal: Vec3,
    /// The normal the face presents, its own orientation counted.
    pub logical: Vec3,
    pub regular: bool,
    pub gap: f64,
    su: Vec3,
    sv: Vec3,
    e: f64,
    f: f64,
    g: f64,
    l: f64,
    m: f64,
    n: f64,
}

impl Form {
    /// The second fundamental form on the tangent `d`: `II(d, d)`, which is
    /// the normal curvature along `d` times `d . d`.
    pub fn second_form(&self, d: Vec3) -> f64 {
        let det = self.e * self.g - self.f * self.f;
        if det <= 0.0 {
            return 0.0;
        }
        let (ds, dt) = (d.dot(self.su), d.dot(self.sv));
        let du = (self.g * ds - self.f * dt) / det;
        let dv = (self.e * dt - self.f * ds) / det;
        self.l * du * du + 2.0 * self.m * du * dv + self.n * dv * dv
    }
}

/// A face's surface, read at points on it.
#[derive(Debug)]
pub(super) struct Neighbour {
    pub surface: FaceSurface,
    pub reversed: bool,
    plane: Option<Frame3>,
    seed: Cell<Option<(f64, f64)>>,
    /// The `u` domain of a spline that closes across `u`.
    closed_u: Option<(f64, f64)>,
    /// As `closed_u`, for `v`.
    closed_v: Option<(f64, f64)>,
}

impl Neighbour {
    pub fn of(topo: &Topology, face: FaceId) -> Result<Self, OperationsError> {
        let data = topo.face(face)?;
        let surface = data.surface().clone();
        let plane = if let FaceSurface::Plane { normal, d } = &surface {
            let origin = Point3::new(0.0, 0.0, 0.0) + *normal * (*d / normal.length_squared());
            Some(Frame3::from_normal(origin, *normal)?)
        } else {
            None
        };
        let (closed_u, closed_v) = if let FaceSurface::Nurbs(spline) = &surface {
            (
                ParametricSurface::closed_u(spline),
                ParametricSurface::closed_v(spline),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            surface,
            reversed: data.is_reversed(),
            plane,
            seed: Cell::new(None),
            closed_u,
            closed_v,
        })
    }

    /// `(u, v)` wrapped on to a closed spline's domain. A chart runs on past
    /// the seam to stay continuous, and a spline extrapolates there rather
    /// than wrapping round.
    fn on_domain(&self, u: f64, v: f64) -> (f64, f64) {
        let wrap = |x: f64, closed: Option<(f64, f64)>| match closed {
            Some((lo, hi)) if x < lo || x > hi => lo + (x - lo).rem_euclid(hi - lo),
            _ => x,
        };
        (wrap(u, self.closed_u), wrap(v, self.closed_v))
    }

    pub fn jet(&self, u: f64, v: f64) -> Jet {
        let zero = Vec3::new(0.0, 0.0, 0.0);
        if let Some(frame) = &self.plane {
            return Jet {
                point: frame.origin + frame.x * u + frame.y * v,
                su: frame.x,
                sv: frame.y,
                suu: zero,
                suv: zero,
                svv: zero,
            };
        }
        match &self.surface {
            FaceSurface::Plane { .. } => Jet {
                point: Point3::new(0.0, 0.0, 0.0),
                su: zero,
                sv: zero,
                suu: zero,
                suv: zero,
                svv: zero,
            },
            FaceSurface::Nurbs(surface) => {
                let (u, v) = self.on_domain(u, v);
                let d = surface.derivatives(u, v, 2);
                Jet {
                    point: surface.evaluate(u, v),
                    su: d[1][0],
                    sv: d[0][1],
                    suu: d[2][0],
                    suv: d[1][1],
                    svv: d[0][2],
                }
            }
            FaceSurface::Cylinder(c) => {
                let (sin_u, cos_u) = u.sin_cos();
                let radial = c.x_axis() * cos_u + c.y_axis() * sin_u;
                let around = c.y_axis() * cos_u - c.x_axis() * sin_u;
                Jet {
                    point: c.evaluate(u, v),
                    su: around * c.radius(),
                    sv: c.axis(),
                    suu: radial * (-c.radius()),
                    suv: zero,
                    svv: zero,
                }
            }
            FaceSurface::Cone(c) => {
                let (sin_u, cos_u) = u.sin_cos();
                let (sin_a, cos_a) = c.half_angle().sin_cos();
                let radial = c.x_axis() * cos_u + c.y_axis() * sin_u;
                let around = c.y_axis() * cos_u - c.x_axis() * sin_u;
                Jet {
                    point: c.evaluate(u, v),
                    su: around * (v * cos_a),
                    sv: radial * cos_a + c.axis() * sin_a,
                    suu: radial * (-v * cos_a),
                    suv: around * cos_a,
                    svv: zero,
                }
            }
            FaceSurface::Sphere(c) => {
                let (sin_u, cos_u) = u.sin_cos();
                let (sin_v, cos_v) = v.sin_cos();
                let radial = c.x_axis() * cos_u + c.y_axis() * sin_u;
                let around = c.y_axis() * cos_u - c.x_axis() * sin_u;
                let r = c.radius();
                Jet {
                    point: c.evaluate(u, v),
                    su: around * (r * cos_v),
                    sv: (c.z_axis() * cos_v - radial * sin_v) * r,
                    suu: radial * (-r * cos_v),
                    suv: around * (-r * sin_v),
                    svv: (radial * cos_v + c.z_axis() * sin_v) * (-r),
                }
            }
            FaceSurface::Torus(c) => {
                let (sin_u, cos_u) = u.sin_cos();
                let (sin_v, cos_v) = v.sin_cos();
                let radial = c.x_axis() * cos_u + c.y_axis() * sin_u;
                let around = c.y_axis() * cos_u - c.x_axis() * sin_u;
                let r = c.minor_radius();
                let ring = c.major_radius() + r * cos_v;
                Jet {
                    point: c.evaluate(u, v),
                    su: around * ring,
                    sv: (c.z_axis() * cos_v - radial * sin_v) * r,
                    suu: radial * (-ring),
                    suv: around * (-r * sin_v),
                    svv: (radial * cos_v + c.z_axis() * sin_v) * (-r),
                }
            }
        }
    }

    pub fn point(&self, u: f64, v: f64) -> Point3 {
        if let Some(frame) = &self.plane {
            return frame.origin + frame.x * u + frame.y * v;
        }
        let (u, v) = self.on_domain(u, v);
        self.surface
            .evaluate(u, v)
            .unwrap_or(Point3::new(0.0, 0.0, 0.0))
    }

    /// The parameters of the point on the surface nearest `p`, near enough to
    /// the true foot that the surface's forms there carry only rounding.
    pub fn locate(&self, p: Point3) -> Result<(f64, f64), OperationsError> {
        if let Some(frame) = &self.plane {
            let d = p - frame.origin;
            return Ok((d.dot(frame.x), d.dot(frame.y)));
        }
        let uv = match &self.surface {
            FaceSurface::Nurbs(surface) => {
                let seeded = self.seed.get().and_then(|seed| {
                    project_point_to_surface_from(surface, p, seed, 1e-10)
                        .ok()
                        .filter(|found| found.distance < 1e-5)
                });
                let found = match seeded {
                    Some(found) => found,
                    None => project_point_to_surface(surface, p, 1e-10).map_err(|_| {
                        OperationsError::InvalidInput {
                            reason: format!("a point of a chain, {p:?}, is not on its face"),
                        }
                    })?,
                };
                self.polish((found.u, found.v), p)
            }
            FaceSurface::Torus(t) => t.project_point_nearest(p),
            FaceSurface::Cylinder(c) => ParametricSurface::project_point(c, p),
            FaceSurface::Cone(c) => ParametricSurface::project_point(c, p),
            FaceSurface::Sphere(c) => ParametricSurface::project_point(c, p),
            FaceSurface::Plane { .. } => (0.0, 0.0),
        };
        self.seed.set(Some(uv));
        Ok(uv)
    }

    /// Newton steps on the foot of `p`, which an iterative projection leaves
    /// up to its own tolerance away.
    fn polish(&self, (mut u, mut v): (f64, f64), p: Point3) -> (f64, f64) {
        let FaceSurface::Nurbs(surface) = &self.surface else {
            return (u, v);
        };
        let (u0, u1) = surface.domain_u();
        let (v0, v1) = surface.domain_v();
        for _ in 0..6 {
            let jet = self.jet(u, v);
            let r = p - jet.point;
            let (e, f, g) = (jet.su.dot(jet.su), jet.su.dot(jet.sv), jet.sv.dot(jet.sv));
            let det = e * g - f * f;
            if det <= 1e-30 {
                break;
            }
            let (a, b) = (r.dot(jet.su), r.dot(jet.sv));
            let du = (g * a - f * b) / det;
            let dv = (e * b - f * a) / det;
            let (nu, nv) = ((u + du).clamp(u0, u1), (v + dv).clamp(v0, v1));
            let moved = (nu - u).abs().max((nv - v).abs());
            (u, v) = (nu, nv);
            if moved < 1e-15 {
                break;
            }
        }
        (u, v)
    }

    pub fn form_at(&self, p: Point3) -> Result<Form, OperationsError> {
        let (u, v) = self.locate(p)?;
        Ok(self.form_of(u, v, p))
    }

    fn form_of(&self, u: f64, v: f64, p: Point3) -> Form {
        let jet = self.jet(u, v);
        let cross = jet.su.cross(jet.sv);
        let (e, f, g) = (jet.su.dot(jet.su), jet.su.dot(jet.sv), jet.sv.dot(jet.sv));
        let regular = cross.length_squared() > SINGULAR * SINGULAR * e * g && e > 0.0 && g > 0.0;
        let normal = cross
            .normalize()
            .unwrap_or_else(|_| self.surface.normal(u, v));
        let facing = self.surface.normal(u, v);
        let facing = if self.reversed { -facing } else { facing };
        Form {
            normal,
            logical: facing,
            regular,
            gap: (jet.point - p).length(),
            su: jet.su,
            sv: jet.sv,
            e,
            f,
            g,
            l: normal.dot(jet.suu),
            m: normal.dot(jet.suv),
            n: normal.dot(jet.svv),
        }
    }

    /// How far round `u` and `v` wrap, for a surface closed across them.
    pub fn periods(&self) -> (Option<f64>, Option<f64>) {
        match &self.surface {
            FaceSurface::Plane { .. } => (None, None),
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_) | FaceSurface::Sphere(_) => {
                (Some(TAU), None)
            }
            FaceSurface::Torus(_) => (Some(TAU), Some(TAU)),
            FaceSurface::Nurbs(_) => (
                self.closed_u.map(|(a, b)| b - a),
                self.closed_v.map(|(a, b)| b - a),
            ),
        }
    }

    /// The nearest copy of `uv` to `near` on a surface closed across a
    /// parameter.
    pub fn unwrap_near(&self, uv: (f64, f64), near: (f64, f64)) -> (f64, f64) {
        let (pu, pv) = self.periods();
        let wrap = |x: f64, to: f64, period: Option<f64>| match period {
            Some(p) => x - ((x - to) / p).round() * p,
            None => x,
        };
        (wrap(uv.0, near.0, pu), wrap(uv.1, near.1, pv))
    }

    /// A wire's points in `(u, v)`, walked along the wire and kept continuous
    /// across any seam of the surface.
    pub fn chart_loop(
        &self,
        topo: &Topology,
        wire: WireId,
    ) -> Result<Vec<(f64, f64)>, OperationsError> {
        let mut uv: Vec<(f64, f64)> = Vec::new();
        for oe in topo.wire(wire)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let start = topo.vertex(edge.start())?.point();
            let end = topo.vertex(edge.end())?.point();
            let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
            let samples = if matches!(
                edge.curve(),
                brepkit_topology::edge::EdgeCurve::NurbsCurve(_)
            ) {
                32
            } else {
                16
            };
            for k in 0..samples {
                let along = f64::from(k) / f64::from(samples);
                let w = if oe.is_forward() { along } else { 1.0 - along };
                let p = edge
                    .curve()
                    .evaluate_with_endpoints(t0 + (t1 - t0) * w, start, end);
                let at = self.locate(p)?;
                let at = match uv.last() {
                    Some(&near) => self.unwrap_near(at, near),
                    None => at,
                };
                // A pole has no `u` of its own: the wire keeps the one it came
                // with, not a rounding's choice.
                let jet = self.jet(at.0, at.1);
                let at = match uv.last() {
                    Some(&near) if jet.su.length_squared() < 1e-18 => (near.0, at.1),
                    Some(&near) if jet.sv.length_squared() < 1e-18 => (at.0, near.1),
                    _ => at,
                };
                uv.push(at);
            }
        }
        Ok(uv)
    }

    /// Whether the face lies to the left (`1.0`) or right (`-1.0`) of `wire`
    /// as it is walked, seen with `S_u x S_v` towards the viewer.
    ///
    /// A loop that winds round a closed surface bounds no area in `(u, v)`,
    /// and such a wire keeps its face on its left.
    fn wire_side(
        &self,
        topo: &Topology,
        wire: WireId,
        is_outer: bool,
    ) -> Result<f64, OperationsError> {
        let uv = self.chart_loop(topo, wire)?;
        let (pu, pv) = self.periods();
        let (first, last) = (uv[0], uv[uv.len() - 1]);
        let closes_on = |gap: f64, period: Option<f64>| period.is_none_or(|p| gap.abs() < 0.5 * p);
        if !closes_on(last.0 - first.0, pu) || !closes_on(last.1 - first.1, pv) {
            return Ok(1.0);
        }
        let area = shoelace(&uv);
        let (lo, hi) = uv.iter().fold(
            (
                (f64::INFINITY, f64::INFINITY),
                (f64::NEG_INFINITY, f64::NEG_INFINITY),
            ),
            |(lo, hi), &(u, v)| ((lo.0.min(u), lo.1.min(v)), (hi.0.max(u), hi.1.max(v))),
        );
        if area.abs() <= 1e-9 * (hi.0 - lo.0) * (hi.1 - lo.1) {
            return Ok(1.0);
        }
        let around = area.signum();
        Ok(if is_outer { around } else { -around })
    }
}

fn shoelace(polygon: &[(f64, f64)]) -> f64 {
    let n = polygon.len();
    0.5 * (0..n)
        .map(|i| {
            let (a, b) = (polygon[i], polygon[(i + 1) % n]);
            a.0 * b.1 - b.0 * a.1
        })
        .sum::<f64>()
}

/// What a chain sees of its face at one point.
#[derive(Debug, Clone, Copy)]
pub(super) struct Station {
    pub point: Point3,
    pub tangent: Vec3,
    /// The unit normal of the face's surface.
    pub normal: Vec3,
    /// The unit direction in the tangent plane square to the chain, off the
    /// face.
    pub off: Vec3,
    pub form: Form,
}

/// A chain on a face.
#[derive(Debug)]
pub(super) struct Side {
    pub chain: Chain,
    pub neighbour: Neighbour,
    /// `off = sign * (normal x tangent)`.
    pub sign: f64,
}

impl Side {
    /// `edges` as a chain on `face`.
    ///
    /// # Errors
    ///
    /// Returns [`OperationsError::InvalidInput`] if the edges do not make a
    /// chain, an edge is not on the face's boundary or strays from its
    /// surface, an edge is the face's seam, or the face does not lie on one
    /// side of the whole chain.
    pub fn new(topo: &Topology, edges: &[EdgeId], face: FaceId) -> Result<Self, OperationsError> {
        let chain = Chain::from_edges(topo, edges)?;
        let neighbour = Neighbour::of(topo, face)?;
        let sign = face_side(topo, face, &chain, &neighbour)?;
        let side = Self {
            chain,
            neighbour,
            sign,
        };
        side.check_on_surface()?;
        Ok(side)
    }

    fn check_on_surface(&self) -> Result<(), OperationsError> {
        let reach = (OFF_SURFACE * self.chain.length).max(1e-4);
        for segment in &self.chain.segments {
            for k in 0..=16 {
                let s = segment.s0 + (segment.s1 - segment.s0) * f64::from(k) / 16.0;
                let p = self.chain.point(s.min(1.0));
                let form = self.neighbour.form_at(p)?;
                if form.gap > reach {
                    return Err(OperationsError::InvalidInput {
                        reason: format!(
                            "an edge of a chain leaves its face's surface by {:.1e} at {p:?}",
                            form.gap
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn reverse(&mut self) {
        self.chain = self.chain.reversed();
        self.sign = -self.sign;
    }

    /// What the chain sees at `s`, read from inside the piece `(lo, hi)`.
    pub fn station(&self, piece: (f64, f64), s: f64) -> Result<Station, OperationsError> {
        let (point, tangent, inside) = self.chain.point_and_tangent(piece, s)?;
        let mut form = self.neighbour.form_at(inside)?;
        if !form.regular {
            // A pole or apex has no forms of its own; those of its piece's
            // points beside it hold in the limit.
            let inward = if s > 0.5 * (piece.0 + piece.1) {
                piece.0 - piece.1
            } else {
                piece.1 - piece.0
            };
            for step in [1e-6, 1e-5, 1e-4, 1e-3] {
                let (near, _, _) = self.chain.point_and_tangent(piece, s + step * inward)?;
                let beside = self.neighbour.form_at(near)?;
                if beside.regular {
                    form = beside;
                    break;
                }
            }
            if !form.regular {
                return Err(OperationsError::InvalidInput {
                    reason: format!("a chain runs along a singular line of its face at {point:?}"),
                });
            }
        }
        let off = (form.normal.cross(tangent) * self.sign).normalize()?;
        Ok(Station {
            point,
            tangent,
            normal: form.normal,
            off,
            form,
        })
    }

    /// The tilt of the cross-boundary direction at each end of each piece, so
    /// that the direction is continuous across every join.
    pub fn tilts(&self, breaks: &[f64]) -> Result<Vec<(f64, f64)>, OperationsError> {
        let pieces = breaks.len() - 1;
        let mut tilts = vec![(0.0, 0.0); pieces];
        let joins = if self.chain.closed {
            pieces
        } else {
            pieces.saturating_sub(1)
        };
        for j in 0..joins {
            let (left, right) = (j, (j + 1) % pieces);
            let at_left = breaks[left + 1];
            let at_right = breaks[right];
            let a = self.station((breaks[left], breaks[left + 1]), at_left)?;
            let b = self.station((breaks[right], breaks[right + 1]), at_right)?;
            if a.normal.cross(b.normal).length() > CREASE {
                return Err(OperationsError::InvalidInput {
                    reason: format!(
                        "a face has a crease where its chain crosses it, at {:?}",
                        a.point
                    ),
                });
            }
            let cos = a.off.dot(b.off);
            if cos < DOUBLES_BACK {
                return Err(OperationsError::InvalidInput {
                    reason: format!("a chain doubles back on itself at {:?}", a.point),
                });
            }
            if cos > 1.0 - 1e-15 {
                continue;
            }
            let mean = (a.off + b.off).normalize()?;
            tilts[left].1 = mean.dot(a.tangent).atan2(mean.dot(a.off));
            tilts[right].0 = mean.dot(b.tangent).atan2(mean.dot(b.off));
        }
        Ok(tilts)
    }

    /// The `s` where the chain crosses a knot line of a spline surface: the
    /// surface's forms lose smoothness there, however smooth the chain.
    pub fn surface_knot_breaks(&self, breaks: &[f64]) -> Result<Vec<f64>, OperationsError> {
        let FaceSurface::Nurbs(surface) = &self.neighbour.surface else {
            return Ok(Vec::new());
        };
        let distinct = |knots: &[f64], lo: f64, hi: f64| {
            let mut inner: Vec<f64> = knots
                .iter()
                .copied()
                .filter(|k| *k > lo + 1e-9 && *k < hi - 1e-9)
                .collect();
            inner.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
            inner
        };
        let (u0, u1) = surface.domain_u();
        let (v0, v1) = surface.domain_v();
        let knots_u = distinct(surface.knots_u(), u0, u1);
        let knots_v = distinct(surface.knots_v(), v0, v1);
        if knots_u.is_empty() && knots_v.is_empty() {
            return Ok(Vec::new());
        }
        let mut found = Vec::new();
        let per_piece = 8 + 2 * knots_u.len().max(knots_v.len());
        for pair in breaks.windows(2) {
            let piece = (pair[0], pair[1]);
            let at = |k: usize| pair[0] + (pair[1] - pair[0]) * k as f64 / per_piece as f64;
            let mut previous = self.foot(piece, at(0))?;
            for k in 1..=per_piece {
                let next = self.foot(piece, at(k))?;
                for (knots, read, span) in
                    [(&knots_u, 0_usize, u1 - u0), (&knots_v, 1_usize, v1 - v0)]
                {
                    let value = |uv: (f64, f64)| if read == 0 { uv.0 } else { uv.1 };
                    let (a, b) = (value(previous), value(next));
                    if (a - b).abs() > 0.5 * span {
                        continue;
                    }
                    for &knot in knots {
                        if (a >= knot) != (b >= knot)
                            && (a - knot).abs().max((b - knot).abs()) > 1e-8
                        {
                            let (mut lo, mut hi) = (at(k - 1), at(k));
                            for _ in 0..40 {
                                let mid = 0.5 * (lo + hi);
                                if (value(self.foot(piece, mid)?) >= knot) == (a >= knot) {
                                    lo = mid;
                                } else {
                                    hi = mid;
                                }
                            }
                            found.push(0.5 * (lo + hi));
                        }
                    }
                }
                previous = next;
            }
        }
        Ok(found)
    }

    fn foot(&self, piece: (f64, f64), s: f64) -> Result<(f64, f64), OperationsError> {
        let (point, _, _) = self.chain.point_and_tangent(piece, s)?;
        self.neighbour.locate(point)
    }
}

/// `off = sign * (normal x tangent)` along `chain` on `face`, read from the
/// wire each edge runs in.
fn face_side(
    topo: &Topology,
    face: FaceId,
    chain: &Chain,
    neighbour: &Neighbour,
) -> Result<f64, OperationsError> {
    let data = topo.face(face)?;
    let wires: Vec<(WireId, bool)> = std::iter::once((data.outer_wire(), true))
        .chain(data.inner_wires().iter().map(|&w| (w, false)))
        .collect();
    let mut sign = None;
    for segment in &chain.segments {
        let Some(edge) = segment.edge else {
            continue;
        };
        let mut uses = Vec::new();
        for &(wire, is_outer) in &wires {
            for oe in topo.wire(wire)?.edges() {
                if oe.edge() == edge {
                    uses.push((wire, is_outer, oe.is_forward()));
                }
            }
        }
        let &[(wire, is_outer, forward)] = uses.as_slice() else {
            return Err(OperationsError::InvalidInput {
                reason: if uses.is_empty() {
                    format!("edge {} is not on the face beside its chain", edge.index())
                } else {
                    format!(
                        "edge {} is a seam of its face, which lies on both sides of it",
                        edge.index()
                    )
                },
            });
        };
        let left = neighbour.wire_side(topo, wire, is_outer)?;
        // The wire runs the chain's way or not; on the wire's left in `(u, v)`
        // is `normal x walk`, and off the face is the other way from the face.
        let walk = if forward == segment.against {
            -1.0
        } else {
            1.0
        };
        let here = -left * walk;
        match sign {
            None => sign = Some(here),
            Some(earlier) if earlier * here > 0.0 => {}
            Some(_) => {
                return Err(OperationsError::InvalidInput {
                    reason: "a face does not lie on one side of its whole chain".into(),
                });
            }
        }
    }
    Ok(sign.unwrap_or(1.0))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{
        ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
    };
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    /// A chart run on past a closed spline's seam reads the spline a period
    /// on, which has to be the same place with the same jet.
    #[test]
    fn a_closed_spline_reads_the_same_a_period_on() {
        let origin = Point3::new(0.0, 0.0, 0.0);
        let up = Vec3::new(0.0, 0.0, 1.0);
        let spline = CylindricalSurface::new(origin, up, 3.0)
            .unwrap()
            .to_nurbs(0.0, 5.0)
            .unwrap();
        let mut topo = Topology::new();
        let circle = Circle3D::new(origin, up, 3.0).unwrap();
        let at = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
        let rim = topo.add_edge(Edge::new(at, at, EdgeCurve::Circle(circle)));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
        let face = topo.add_face(Face::new(wire, Vec::new(), FaceSurface::Nurbs(spline)));
        let neighbour = Neighbour::of(&topo, face).unwrap();
        let period = neighbour.periods().0.unwrap();
        for (u, v) in [(0.1, 0.3), (0.6, 0.8), (0.95, 0.5)] {
            for turns in [-1.0, 1.0] {
                let (here, there) = (u, u + turns * period);
                let gap = (neighbour.point(here, v) - neighbour.point(there, v)).length();
                assert!(gap < 1e-12, "{u} {turns}: {gap}");
                let (a, b) = (neighbour.jet(here, v), neighbour.jet(there, v));
                for (x, y) in [
                    (a.su, b.su),
                    (a.sv, b.sv),
                    (a.suu, b.suu),
                    (a.suv, b.suv),
                    (a.svv, b.svv),
                ] {
                    assert!((x - y).length() < 1e-9, "{u} {turns}: {x:?} against {y:?}");
                }
            }
        }
    }

    /// The closed forms against central differences of the surface itself.
    fn jets_agree(surface: FaceSurface, (u, v): (f64, f64)) {
        let neighbour = Neighbour {
            surface: surface.clone(),
            reversed: false,
            plane: None,
            seed: Cell::new(None),
            closed_u: None,
            closed_v: None,
        };
        let h = 1e-4;
        let jet = neighbour.jet(u, v);
        let at = |a: f64, b: f64| surface.evaluate(a, b).unwrap();
        let close = |got: Vec3, want: Vec3| {
            assert!((got - want).length() < 1e-6, "{got:?} against {want:?}");
        };
        close(jet.su, (at(u + h, v) - at(u - h, v)) * (0.5 / h));
        close(jet.sv, (at(u, v + h) - at(u, v - h)) * (0.5 / h));
        let su = |a: f64, b: f64| surface.partial_u(a, b).unwrap();
        let sv = |a: f64, b: f64| surface.partial_v(a, b).unwrap();
        close(jet.su, su(u, v));
        close(jet.sv, sv(u, v));
        close(jet.suu, (su(u + h, v) - su(u - h, v)) * (0.5 / h));
        close(jet.suv, (su(u, v + h) - su(u, v - h)) * (0.5 / h));
        close(jet.svv, (sv(u, v + h) - sv(u, v - h)) * (0.5 / h));
    }

    #[test]
    fn analytic_jets_match_their_surfaces() {
        let origin = Point3::new(1.0, -2.0, 0.5);
        let axis = Vec3::new(0.3, 0.2, 0.9);
        let at = (0.7, 0.4);
        jets_agree(
            FaceSurface::Cylinder(CylindricalSurface::new(origin, axis, 2.0).unwrap()),
            at,
        );
        jets_agree(
            FaceSurface::Cone(ConicalSurface::new(origin, axis, 0.6).unwrap()),
            (0.7, 3.0),
        );
        jets_agree(
            FaceSurface::Sphere(SphericalSurface::new(origin, 2.0).unwrap()),
            at,
        );
        jets_agree(
            FaceSurface::Torus(ToroidalSurface::new(origin, 3.0, 1.0).unwrap()),
            at,
        );
    }
}
