//! IGES file writer.
//!
//! Exports B-Rep solids to IGES format (ANSI Y14.26M).
//! The IGES file format uses a fixed 80-column ASCII layout with
//! five sections: Start (S), Global (G), Directory Entry (D),
//! Parameter Data (P), and Terminate (T).
//!
//! Each face goes out as a trimmed surface over its plane or B-spline surface,
//! each of its boundaries as a curve on that surface holding a composite of the
//! boundary's edges in the order they run, so a reader needs nothing else to
//! rebuild the face. Curved analytic faces go out as their exact rational
//! B-spline surfaces. A surface always faces the way its face does, since a
//! trimmed surface has no flag to turn it over.
//!
//! Entity types written:
//! - **102**: Composite Curve (a boundary's edges)
//! - **108**: Plane
//! - **110**: Line
//! - **126**: Rational B-Spline Curve (an arc exactly, as rational quadratics)
//! - **128**: Rational B-Spline Surface
//! - **142**: Curve on a Parametric Surface (a boundary)
//! - **144**: Trimmed (Parametric) Surface (a face)

use std::fmt::Write as _;

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, TAU};

use brepkit_math::nurbs::{NurbsCurve, NurbsSurface};
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::wire::WireId;

use crate::IoError;

/// The status of an entity on its own, in a directory entry.
const INDEPENDENT: &str = "00000000";

/// The status of an entity only another entity's parameters refer to.
const SUBORDINATE: &str = "00010000";

/// Write one or more solids to IGES format.
///
/// Returns the IGES file as a UTF-8 string.
///
/// # Errors
///
/// Returns an error if `solids` is empty, topology lookups fail, or a curved
/// face has no exact B-spline form.
pub fn write_iges(topo: &Topology, solids: &[SolidId]) -> Result<String, IoError> {
    if solids.is_empty() {
        return Err(IoError::InvalidTopology {
            reason: "no solids to export".to_string(),
        });
    }

    let rational = rational_surfaces(topo, solids)?;
    let mut ctx = IgesWriteContext::new();
    for &solid_id in solids {
        for face_id in solid_faces(topo, solid_id).map_err(topo_err)? {
            ctx.write_face(topo, face_id, &rational)?;
        }
    }

    Ok(ctx.finish())
}

/// The exact rational B-spline surface of each curved analytic face. Conversion
/// works in place, so it runs on a copy of the arena, in which faces keep their
/// ids. Everything else is written from the solid as it stands: a plane stays a
/// plane, and an edge stays exact.
fn rational_surfaces(
    topo: &Topology,
    solids: &[SolidId],
) -> Result<HashMap<FaceId, NurbsSurface>, IoError> {
    let mut curved = Vec::new();
    for &solid in solids {
        for face in solid_faces(topo, solid).map_err(topo_err)? {
            match topo.face(face).map_err(topo_err)?.surface() {
                FaceSurface::Cylinder(_)
                | FaceSurface::Cone(_)
                | FaceSurface::Sphere(_)
                | FaceSurface::Torus(_) => curved.push(face),
                FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => {}
            }
        }
    }
    let mut surfaces = HashMap::new();
    if curved.is_empty() {
        return Ok(surfaces);
    }
    let mut copy = topo.clone();
    for &solid in solids {
        brepkit_operations::heal::convert_to_bspline(&mut copy, solid)?;
    }
    for face in curved {
        match copy.face(face).map_err(topo_err)?.surface() {
            FaceSurface::Nurbs(nurbs) => {
                surfaces.insert(face, nurbs.clone());
            }
            FaceSurface::Plane { .. }
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_) => {
                return Err(IoError::InvalidTopology {
                    reason: "a curved face has no exact B-spline form".to_string(),
                });
            }
        }
    }
    Ok(surfaces)
}

/// Whether `nurbs` runs against the way `face` faces, read at a point of its
/// boundary where both surfaces have a normal.
fn runs_against_its_face(
    topo: &Topology,
    face: FaceId,
    nurbs: &NurbsSurface,
) -> Result<bool, IoError> {
    let data = topo.face(face).map_err(topo_err)?;
    let image = FaceSurface::Nurbs(nurbs.clone());
    for oe in topo.wire(data.outer_wire()).map_err(topo_err)?.edges() {
        let edge = topo.edge(oe.edge()).map_err(topo_err)?;
        let a = topo.vertex(edge.start()).map_err(topo_err)?.point();
        let b = topo.vertex(edge.end()).map_err(topo_err)?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        let p = edge.curve().evaluate_with_endpoints(0.5 * (t0 + t1), a, b);
        let (Some((u, v)), Some((iu, iv))) =
            (data.surface().project_point(p), image.project_point(p))
        else {
            continue;
        };
        let (own, written) = (data.surface().normal(u, v), image.normal(iu, iv));
        // A pole or an apex has no normal to read.
        if own.length() < 1e-6 || written.length() < 1e-6 {
            continue;
        }
        let along = own.dot(written);
        if along.abs() > 1e-6 {
            let outward_along = if data.is_reversed() { -along } else { along };
            return Ok(outward_along < 0.0);
        }
    }
    Ok(false)
}

/// IGES writing context: manages entity numbering and sections.
struct IgesWriteContext {
    /// Directory entry lines (D section).
    dir_entries: Vec<String>,
    /// Parameter data lines (P section).
    param_data: Vec<String>,
    /// Next directory entry sequence number (odd, starts at 1).
    next_de: u32,
    /// Next parameter data sequence number (starts at 1).
    next_pd: u32,
}

impl IgesWriteContext {
    const fn new() -> Self {
        Self {
            dir_entries: Vec::new(),
            param_data: Vec::new(),
            next_de: 1,
            next_pd: 1,
        }
    }

    /// Write a face as a trimmed surface over its surface, running its
    /// boundaries the way they run on the face as it faces.
    fn write_face(
        &mut self,
        topo: &Topology,
        face_id: FaceId,
        rational: &HashMap<FaceId, NurbsSurface>,
    ) -> Result<(), IoError> {
        let face = topo.face(face_id).map_err(topo_err)?;
        let reversed = face.is_reversed();

        let surface = match face.surface() {
            FaceSurface::Plane { normal, d } => {
                if reversed {
                    self.write_plane_entity(-*normal, -*d)
                } else {
                    self.write_plane_entity(*normal, *d)
                }
            }
            FaceSurface::Nurbs(nurbs) => {
                // A B-spline face's normal is its own surface's, read through
                // the flag: only the flag can run against it.
                self.write_nurbs_surface_entity(nurbs, reversed)?
            }
            FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_) => {
                let nurbs = rational
                    .get(&face_id)
                    .ok_or_else(|| IoError::InvalidTopology {
                        reason: "a curved face has no exact B-spline form".to_string(),
                    })?;
                let against = runs_against_its_face(topo, face_id, nurbs)?;
                self.write_nurbs_surface_entity(nurbs, against)?
            }
        };

        let outer = self.write_boundary(topo, face.outer_wire(), reversed, surface)?;
        let mut holes = Vec::with_capacity(face.inner_wires().len());
        for &wire in face.inner_wires() {
            holes.push(self.write_boundary(topo, wire, reversed, surface)?);
        }

        // Surface, its own boundary not the one used (1), how many holes, the
        // outer boundary, then each hole.
        let mut params = format!("144,{surface},1,{},{outer}", holes.len());
        for hole in holes {
            let _ = write!(params, ",{hole}");
        }
        params.push(';');
        self.add_entity(144, &params, INDEPENDENT);
        Ok(())
    }

    /// Write one boundary of a face as a curve on `surface`, returning its
    /// directory entry. `backward` runs it against the wire's order.
    fn write_boundary(
        &mut self,
        topo: &Topology,
        wire: WireId,
        backward: bool,
        surface: u32,
    ) -> Result<u32, IoError> {
        let oriented = topo.wire(wire).map_err(topo_err)?.edges();
        let uses: Vec<(_, bool)> = if backward {
            oriented
                .iter()
                .rev()
                .map(|oe| (oe.edge(), !oe.is_forward()))
                .collect()
        } else {
            oriented
                .iter()
                .map(|oe| (oe.edge(), oe.is_forward()))
                .collect()
        };

        let mut composite = format!("102,{}", uses.len());
        for (edge_id, forward) in uses {
            let edge = topo.edge(edge_id).map_err(topo_err)?;
            let a = topo.vertex(edge.start()).map_err(topo_err)?.point();
            let b = topo.vertex(edge.end()).map_err(topo_err)?.point();
            let curve = self.write_edge_curve(edge.curve(), a, b, forward)?;
            let _ = write!(composite, ",{curve}");
        }
        composite.push(';');
        let composite = self.add_entity(102, &composite, SUBORDINATE);

        // Creation unspecified, on `surface`, no parameter-space curve, this
        // model-space curve, neither preferred.
        Ok(self.add_entity(
            142,
            &format!("142,0,{surface},0,{composite},0;"),
            SUBORDINATE,
        ))
    }

    /// Write an edge's curve running the way this use of it does.
    fn write_edge_curve(
        &mut self,
        curve: &EdgeCurve,
        a: Point3,
        b: Point3,
        forward: bool,
    ) -> Result<u32, IoError> {
        let (t0, t1) = curve.domain_with_endpoints(a, b);
        let (from, to) = if forward { (t0, t1) } else { (t1, t0) };
        Ok(match curve {
            EdgeCurve::Line => {
                let (start, end) = if forward { (a, b) } else { (b, a) };
                self.write_line_entity(start, end)
            }
            EdgeCurve::NurbsCurve(nurbs) => {
                if from <= to {
                    self.write_nurbs_curve_entity(nurbs, from, to)
                } else {
                    // Running down its knots: mirrored, each knot `k` becomes
                    // `first + last - k`, and so does the span.
                    let mirrored = curve.reversed().map_err(math_err)?;
                    let EdgeCurve::NurbsCurve(mirrored) = mirrored else {
                        return Err(IoError::InvalidTopology {
                            reason: "a reversed B-spline edge is no B-spline".to_string(),
                        });
                    };
                    let knots = nurbs.knots();
                    let sum = knots[0] + knots[knots.len() - 1];
                    self.write_nurbs_curve_entity(&mirrored, sum - from, sum - to)
                }
            }
            EdgeCurve::Circle(c) => {
                let (from, to) = closed_from_its_vertex(a, b, (from, to), forward, c.project(a));
                let nurbs = rational_arc(
                    c.center(),
                    c.u_axis() * c.radius(),
                    c.v_axis() * c.radius(),
                    from,
                    to,
                )?;
                let last = nurbs.knots()[nurbs.knots().len() - 1];
                self.write_nurbs_curve_entity(&nurbs, 0.0, last)
            }
            EdgeCurve::Ellipse(e) => {
                let (from, to) = closed_from_its_vertex(a, b, (from, to), forward, e.project(a));
                let nurbs = rational_arc(
                    e.center(),
                    e.u_axis() * e.semi_major(),
                    e.v_axis() * e.semi_minor(),
                    from,
                    to,
                )?;
                let last = nurbs.knots()[nurbs.knots().len() - 1];
                self.write_nurbs_curve_entity(&nurbs, 0.0, last)
            }
        })
    }

    /// Write a line entity (type 110).
    fn write_line_entity(&mut self, start: Point3, end: Point3) -> u32 {
        let params = format!(
            "110,{},{},{},{},{},{};",
            fmt_f(start.x()),
            fmt_f(start.y()),
            fmt_f(start.z()),
            fmt_f(end.x()),
            fmt_f(end.y()),
            fmt_f(end.z()),
        );
        self.add_entity(110, &params, SUBORDINATE)
    }

    /// Write a plane entity using IGES type 108 (Plane), unbounded of itself:
    /// the trimmed surface over it holds its boundary.
    fn write_plane_entity(&mut self, normal: Vec3, d: f64) -> u32 {
        // IGES Plane (type 108): A, B, C, D where Ax + By + Cz = D.
        let params = format!(
            "108,{},{},{},{},0,0,0,0;",
            fmt_f(normal.x()),
            fmt_f(normal.y()),
            fmt_f(normal.z()),
            fmt_f(d),
        );
        self.add_entity(108, &params, SUBORDINATE)
    }

    /// Write a NURBS curve entity (type 126) over its span from `from` to `to`.
    fn write_nurbs_curve_entity(&mut self, nurbs: &NurbsCurve, from: f64, to: f64) -> u32 {
        let n = nurbs.control_points().len() - 1; // Index of last control point.
        let degree = nurbs.degree();

        let mut params = format!("126,{n},{degree},0,0,0,0,");
        for k in nurbs.knots() {
            let _ = write!(params, "{},", fmt_f(*k));
        }
        for w in nurbs.weights() {
            let _ = write!(params, "{},", fmt_f(*w));
        }
        for cp in nurbs.control_points() {
            let _ = write!(
                params,
                "{},{},{},",
                fmt_f(cp.x()),
                fmt_f(cp.y()),
                fmt_f(cp.z())
            );
        }
        let _ = write!(params, "{},{},0.,0.,0.;", fmt_f(from), fmt_f(to));

        self.add_entity(126, &params, SUBORDINATE)
    }

    /// Write a NURBS surface entity (type 128), its `u` direction running
    /// backward when `turned`, which turns its normal over.
    fn write_nurbs_surface_entity(
        &mut self,
        nurbs: &NurbsSurface,
        turned: bool,
    ) -> Result<u32, IoError> {
        let cps = nurbs.control_points();
        let weights = nurbs.weights();
        let (nu, nv) = (cps.len(), cps.first().map_or(0, Vec::len));
        if nu == 0 || nv == 0 {
            return Err(IoError::InvalidTopology {
                reason: "a B-spline surface has no control points".to_string(),
            });
        }
        let knots_u: Vec<f64> = if turned {
            let source = nurbs.knots_u();
            let sum = source[0] + source[source.len() - 1];
            source.iter().rev().map(|k| sum - k).collect()
        } else {
            nurbs.knots_u().to_vec()
        };
        let row = |i: usize| if turned { nu - 1 - i } else { i };

        let mut params = format!(
            "128,{},{},{},{},0,0,0,0,0,",
            nu - 1,
            nv - 1,
            nurbs.degree_u(),
            nurbs.degree_v(),
        );
        for k in &knots_u {
            let _ = write!(params, "{},", fmt_f(*k));
        }
        for k in nurbs.knots_v() {
            let _ = write!(params, "{},", fmt_f(*k));
        }
        // Weights, then points: the first index varies fastest.
        for j in 0..nv {
            for i in 0..nu {
                let _ = write!(params, "{},", fmt_f(weights[row(i)][j]));
            }
        }
        for j in 0..nv {
            for i in 0..nu {
                let cp = cps[row(i)][j];
                let _ = write!(
                    params,
                    "{},{},{},",
                    fmt_f(cp.x()),
                    fmt_f(cp.y()),
                    fmt_f(cp.z())
                );
            }
        }
        let kv = nurbs.knots_v();
        let _ = write!(
            params,
            "{},{},{},{};",
            fmt_f(knots_u[0]),
            fmt_f(knots_u[knots_u.len() - 1]),
            fmt_f(kv[0]),
            fmt_f(kv[kv.len() - 1])
        );

        Ok(self.add_entity(128, &params, SUBORDINATE))
    }

    /// Add an entity to the directory and parameter sections, returning its
    /// directory entry's number, which other entities' parameters refer to it by.
    fn add_entity(&mut self, entity_type: u32, params: &str, status: &str) -> u32 {
        let de_num = self.next_de;
        let pd_start = self.next_pd;

        // Split parameter data into 64-char chunks for the P section.
        let pd_lines = split_param_data(params, de_num, &mut self.next_pd);
        #[allow(clippy::cast_possible_truncation)]
        let pd_count = pd_lines.len() as u32;

        // Directory entry (2 lines, each 80 chars).
        let de1 = format!(
            "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}D{:>7}",
            entity_type, pd_start, 0, 0, 0, 0, 0, 0, status, de_num
        );
        let de2 = format!(
            "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}D{:>7}",
            entity_type,
            0,
            0,
            pd_count,
            0,
            " ",
            " ",
            " ",
            0,
            de_num + 1
        );

        self.dir_entries.push(de1);
        self.dir_entries.push(de2);
        self.param_data.extend(pd_lines);

        self.next_de += 2;
        de_num
    }

    /// Assemble the final IGES file.
    fn finish(self) -> String {
        let mut out = String::new();

        // Start section.
        let _ = writeln!(out, "{:<72}S{:>7}", "brepkit IGES export", 1);

        // Global section (split into 72-char lines).
        let global = "1H,,1H;,7Hbrepkit,12Hbrepkit.igs,\
                       7Hbrepkit,7Hbrepkit,32,38,6,38,15,,1.,1,\
                       2HMM,1,0.001,13H000000.000000,,;";
        let g_chars: Vec<char> = global.chars().collect();
        let mut g_seq = 1u32;
        let mut gi = 0;
        while gi < g_chars.len() {
            let end = (gi + 72).min(g_chars.len());
            let chunk: String = g_chars[gi..end].iter().collect();
            let _ = writeln!(out, "{chunk:<72}G{g_seq:>7}");
            g_seq += 1;
            gi = end;
        }

        // Directory section.
        for (i, line) in self.dir_entries.iter().enumerate() {
            let _ = writeln!(out, "{line}");
            let _ = i; // line already includes D sequence number
        }

        // Parameter section.
        for line in &self.param_data {
            let _ = writeln!(out, "{line}");
        }

        // Terminate section.
        let _ = writeln!(
            out,
            "{:>8}{:>8}{:>8}{:>8}{:>40}T{:>7}",
            format!("S{:>7}", 1),
            format!("G{:>7}", 1),
            format!("D{:>7}", self.dir_entries.len()),
            format!("P{:>7}", self.param_data.len()),
            " ",
            1
        );

        out
    }
}

/// Split parameter data into lines of at most 64 characters for the P section,
/// between parameters, as a parameter may not run across lines.
/// Each line is 80 chars: 64 chars data + 8 chars DE pointer + "P" + 7 chars seq.
fn split_param_data(params: &str, de_num: u32, next_pd: &mut u32) -> Vec<String> {
    const WIDTH: usize = 64;
    let mut lines = Vec::new();
    let mut push = |chunk: &str| {
        let seq = *next_pd;
        *next_pd += 1;
        lines.push(format!("{chunk:<64}{de_num:>8}P{seq:>7}"));
    };

    let mut current = String::new();
    for token in params.split_inclusive([',', ';']) {
        if !current.is_empty() && current.len() + token.len() > WIDTH {
            push(&current);
            current.clear();
        }
        current.push_str(token);
    }
    if !current.is_empty() || params.is_empty() {
        push(&current);
    }

    lines
}

/// Format a float for IGES output, with every digit it takes to read back the
/// same: `E` for an exponent and always a decimal point, as IGES reals have.
fn fmt_f(v: f64) -> String {
    if v == 0.0 {
        return "0.".to_string();
    }
    let shortest = format!("{v:?}");
    match shortest.split_once('e') {
        Some((mantissa, exponent)) => {
            if mantissa.contains('.') {
                format!("{mantissa}E{exponent}")
            } else {
                format!("{mantissa}.E{exponent}")
            }
        }
        None => shortest,
    }
}

/// The span a conic edge runs, from `start` at its vertex when it closes: a
/// closed edge's domain starts at its curve's own origin, off the vertex the
/// next edge of its wire starts from.
fn closed_from_its_vertex(
    a: Point3,
    b: Point3,
    span: (f64, f64),
    forward: bool,
    start: f64,
) -> (f64, f64) {
    if (a - b).length() > 1e-10 {
        span
    } else if forward {
        (start, start + TAU)
    } else {
        (start, start - TAU)
    }
}

/// The arc of `center + cos(t) u + sin(t) v` from `from` to `to`, either way
/// round, exactly: rational quadratics of at most a quarter turn each, joined
/// at knots running 0, 1, 2 along it.
fn rational_arc(
    center: Point3,
    u: Vec3,
    v: Vec3,
    from: f64,
    to: f64,
) -> Result<NurbsCurve, IoError> {
    let sweep = to - from;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let pieces = (sweep.abs() / FRAC_PI_2 - 1e-9).ceil().max(1.0) as usize;
    #[allow(clippy::cast_precision_loss)]
    let step = sweep / pieces as f64;
    let at = |t: f64| center + u * t.cos() + v * t.sin();
    let weight = (0.5 * step).cos();

    let mut control_points = vec![at(from)];
    let mut weights = vec![1.0];
    let mut knots = vec![0.0; 3];
    for k in 0..pieces {
        #[allow(clippy::cast_precision_loss)]
        let start = from + step * k as f64;
        let middle = start + 0.5 * step;
        control_points.push(center + (u * middle.cos() + v * middle.sin()) * (1.0 / weight));
        weights.push(weight);
        control_points.push(at(start + step));
        weights.push(1.0);
        #[allow(clippy::cast_precision_loss)]
        let knot = (k + 1) as f64;
        knots.push(knot);
        knots.push(knot);
    }
    // The last knot closes the curve, three times over.
    #[allow(clippy::cast_precision_loss)]
    knots.push(pieces as f64);
    NurbsCurve::new(2, knots, control_points, weights).map_err(math_err)
}

/// Convert a [`brepkit_math::MathError`] into an [`IoError`].
fn math_err(e: brepkit_math::MathError) -> IoError {
    IoError::Operations(brepkit_operations::OperationsError::from(e))
}

/// Convert a [`TopologyError`] into an [`IoError`].
fn topo_err(e: brepkit_topology::TopologyError) -> IoError {
    IoError::Operations(brepkit_operations::OperationsError::from(e))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_non_manifold;

    use super::*;

    #[test]
    fn write_iges_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let iges_str = write_iges(&topo, &[solid]).unwrap();

        assert!(iges_str.contains("S      1"));
        assert!(iges_str.contains("G      1"));
        assert!(iges_str.contains("D      "));
        assert!(iges_str.contains("P      "));
        assert!(iges_str.contains("T      1"));
    }

    #[test]
    fn iges_contains_plane_entities() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let iges_str = write_iges(&topo, &[solid]).unwrap();

        assert!(
            iges_str.contains("108"),
            "should contain plane entity type 108"
        );
    }

    #[test]
    fn iges_contains_line_entities() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let iges_str = write_iges(&topo, &[solid]).unwrap();

        assert!(
            iges_str.contains("110"),
            "should contain line entity type 110"
        );
    }

    #[test]
    fn iges_box_primitive() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let iges_str = write_iges(&topo, &[solid]).unwrap();
        assert!(!iges_str.is_empty());
        assert!(iges_str.contains("brepkit"));
    }

    #[test]
    fn iges_multiple_solids() {
        let mut topo = Topology::new();
        let s1 = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = make_unit_cube_non_manifold(&mut topo);

        let iges_str = write_iges(&topo, &[s1, s2]).unwrap();

        let plane_count = iges_str.matches("     108").count();
        assert!(
            plane_count >= 12,
            "two boxes should have at least 12 plane entities"
        );
    }

    #[test]
    fn iges_empty_solids_error() {
        let topo = Topology::new();
        let result = write_iges(&topo, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn iges_lines_are_80_chars_or_less() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut topo);

        let iges_str = write_iges(&topo, &[solid]).unwrap();

        for line in iges_str.lines() {
            assert!(
                line.len() <= 82, // 80 + possible \r\n
                "IGES line exceeds 80 chars: {} (len={})",
                line,
                line.len()
            );
        }
    }
}
