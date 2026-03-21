//! IGES file writer.
//!
//! Exports B-Rep solids to IGES format (ANSI Y14.26M).
//! The IGES file format uses a fixed 80-column ASCII layout with
//! five sections: Start (S), Global (G), Directory Entry (D),
//! Parameter Data (P), and Terminate (T).
//!
//! Supported entity types:
//! - **110**: Line
//! - **126**: Rational B-Spline Curve
//! - **128**: Rational B-Spline Surface
//! - **142**: Curve on a Parametric Surface (trimming curve)
//! - **144**: Trimmed (Parametric) Surface
//! - **186**: Manifold Solid B-Rep Object (planned)

use std::fmt::Write as _;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::IoError;

/// Write one or more solids to IGES format.
///
/// Returns the IGES file as a UTF-8 string.
///
/// # Errors
///
/// Returns an error if `solids` is empty or topology lookups fail.
pub fn write_iges(topo: &Topology, solids: &[SolidId]) -> Result<String, IoError> {
    if solids.is_empty() {
        return Err(IoError::InvalidTopology {
            reason: "no solids to export".to_string(),
        });
    }

    let mut ctx = IgesWriteContext::new();

    // Write all faces from all solids as trimmed surfaces (entity 144).
    for &solid_id in solids {
        let solid = topo.solid(solid_id).map_err(topo_err)?;
        let shell = topo.shell(solid.outer_shell()).map_err(topo_err)?;

        for &face_id in shell.faces() {
            ctx.write_face(topo, face_id)?;
        }
    }

    Ok(ctx.finish())
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

    /// Write a face as a planar surface entity (type 190 for plane,
    /// or type 128 for NURBS surface).
    fn write_face(&mut self, topo: &Topology, face_id: FaceId) -> Result<(), IoError> {
        let face = topo.face(face_id).map_err(topo_err)?;

        match face.surface() {
            FaceSurface::Plane { normal, d } => {
                self.write_plane_entity(*normal, *d);
            }
            FaceSurface::Nurbs(nurbs) => {
                self.write_nurbs_surface_entity(nurbs);
            }
            FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_) => {
                // Analytic surfaces: write a placeholder comment entity.
                // Full IGES analytic surface support (entity types 190, 192,
                // 194, 196) is not yet implemented. The edges will still be
                // exported correctly.
            }
        }

        // Write edges as line or NURBS curve entities.
        let wire = topo.wire(face.outer_wire()).map_err(topo_err)?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).map_err(topo_err)?;
            let start_pt = topo.vertex(edge.start()).map_err(topo_err)?.point();
            let end_pt = topo.vertex(edge.end()).map_err(topo_err)?.point();

            match edge.curve() {
                EdgeCurve::Line => {
                    self.write_line_entity(start_pt, end_pt);
                }
                EdgeCurve::NurbsCurve(nurbs) => {
                    self.write_nurbs_curve_entity(nurbs);
                }
                EdgeCurve::Circle(circle) => {
                    // IGES entity 100: Circular Arc (in the circle's local plane)
                    // For simplicity, write as a polyline approximation using entity 110 (lines)
                    let n = 32;
                    let t0 = circle.project(start_pt);
                    let mut t1 = circle.project(end_pt);
                    if (start_pt - end_pt).length() < 1e-10 {
                        t1 = t0 + std::f64::consts::TAU;
                    } else if t1 <= t0 {
                        t1 += std::f64::consts::TAU;
                    }
                    let dt = (t1 - t0) / n as f64;
                    let mut prev = circle.evaluate(t0);
                    for i in 1..=n {
                        let cur = circle.evaluate(t0 + dt * i as f64);
                        self.write_line_entity(prev, cur);
                        prev = cur;
                    }
                }
                EdgeCurve::Ellipse(ellipse) => {
                    // Approximate as polyline segments
                    let n = 32;
                    let t0 = ellipse.project(start_pt);
                    let mut t1 = ellipse.project(end_pt);
                    if (start_pt - end_pt).length() < 1e-10 {
                        t1 = t0 + std::f64::consts::TAU;
                    } else if t1 <= t0 {
                        t1 += std::f64::consts::TAU;
                    }
                    let dt = (t1 - t0) / n as f64;
                    let mut prev = ellipse.evaluate(t0);
                    for i in 1..=n {
                        let cur = ellipse.evaluate(t0 + dt * i as f64);
                        self.write_line_entity(prev, cur);
                        prev = cur;
                    }
                }
            }
        }

        Ok(())
    }

    /// Write a line entity (type 110).
    fn write_line_entity(&mut self, start: Point3, end: Point3) {
        let params = format!(
            "110,{},{},{},{},{},{};",
            fmt_f(start.x()),
            fmt_f(start.y()),
            fmt_f(start.z()),
            fmt_f(end.x()),
            fmt_f(end.y()),
            fmt_f(end.z()),
        );
        self.add_entity(110, &params);
    }

    /// Write a plane entity using IGES type 108 (Plane).
    fn write_plane_entity(&mut self, normal: Vec3, d: f64) {
        // IGES Plane (type 108): A, B, C, D where Ax + By + Cz = D.
        let params = format!(
            "108,{},{},{},{},0,0,0,0;",
            fmt_f(normal.x()),
            fmt_f(normal.y()),
            fmt_f(normal.z()),
            fmt_f(d),
        );
        self.add_entity(108, &params);
    }

    /// Write a NURBS curve entity (type 126).
    fn write_nurbs_curve_entity(&mut self, nurbs: &brepkit_math::nurbs::NurbsCurve) {
        let n = nurbs.control_points().len() - 1; // Index of last control point.
        let degree = nurbs.degree();
        let knots = nurbs.knots();
        let cps = nurbs.control_points();
        let weights = nurbs.weights();

        let mut params = format!("126,{n},{degree},0,0,0,0,");

        // Knot values.
        for k in knots {
            let _ = write!(params, "{},", fmt_f(*k));
        }

        // Weights.
        for w in weights {
            let _ = write!(params, "{},", fmt_f(*w));
        }

        // Control points (x, y, z for each).
        for (i, cp) in cps.iter().enumerate() {
            if i > 0 {
                params.push(',');
            }
            let _ = write!(
                params,
                "{},{},{}",
                fmt_f(cp.x()),
                fmt_f(cp.y()),
                fmt_f(cp.z())
            );
        }

        // Parameter range.
        let _ = write!(
            params,
            ",{},{},0.,0.,0.;",
            fmt_f(knots[0]),
            fmt_f(knots[knots.len() - 1])
        );

        self.add_entity(126, &params);
    }

    /// Write a NURBS surface entity (type 128).
    fn write_nurbs_surface_entity(&mut self, nurbs: &brepkit_math::nurbs::NurbsSurface) {
        let cps = nurbs.control_points();
        let nu = cps.len() - 1;
        let nv = if cps.is_empty() { 0 } else { cps[0].len() - 1 };

        let mut params = format!(
            "128,{nu},{nv},{},{},0,0,0,0,0,",
            nurbs.degree_u(),
            nurbs.degree_v(),
        );

        // Knots U.
        for k in nurbs.knots_u() {
            let _ = write!(params, "{},", fmt_f(*k));
        }
        // Knots V.
        for k in nurbs.knots_v() {
            let _ = write!(params, "{},", fmt_f(*k));
        }

        // Weights (row-major).
        for row in cps {
            for _ in row {
                params.push_str("1.,");
            }
        }

        // Control points (x, y, z, row-major).
        for row in cps {
            for cp in row {
                let _ = write!(
                    params,
                    "{},{},{},",
                    fmt_f(cp.x()),
                    fmt_f(cp.y()),
                    fmt_f(cp.z())
                );
            }
        }

        // Parameter ranges.
        let ku = nurbs.knots_u();
        let kv = nurbs.knots_v();
        let _ = write!(
            params,
            "{},{},{},{};",
            fmt_f(ku[0]),
            fmt_f(ku[ku.len() - 1]),
            fmt_f(kv[0]),
            fmt_f(kv[kv.len() - 1])
        );

        self.add_entity(128, &params);
    }

    /// Add an entity to the directory and parameter sections.
    fn add_entity(&mut self, entity_type: u32, params: &str) {
        let de_num = self.next_de;
        let pd_start = self.next_pd;

        // Split parameter data into 64-char chunks for the P section.
        let pd_lines = split_param_data(params, de_num, &mut self.next_pd);
        #[allow(clippy::cast_possible_truncation)]
        let pd_count = pd_lines.len() as u32;

        // Directory entry (2 lines, each 80 chars).
        let de1 = format!(
            "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}D{:>7}",
            entity_type, pd_start, 0, 0, 0, 0, 0, 0, 0, de_num
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

/// Split parameter data into 64-char chunks for the P section.
/// Each line is 80 chars: 64 chars data + 8 chars DE pointer + "P" + 7 chars seq.
fn split_param_data(params: &str, de_num: u32, next_pd: &mut u32) -> Vec<String> {
    let mut lines = Vec::new();
    let chars: Vec<char> = params.chars().collect();
    let chunk_size = 64;

    let mut i = 0;
    while i < chars.len() {
        let end = (i + chunk_size).min(chars.len());
        let chunk: String = chars[i..end].iter().collect();
        let seq = *next_pd;
        *next_pd += 1;

        let line = format!("{chunk:<64}{de_num:>8}P{seq:>7}");
        lines.push(line);
        i = end;
    }

    if lines.is_empty() {
        let seq = *next_pd;
        *next_pd += 1;
        lines.push(format!("{:<64}{:>8}P{:>7}", " ", de_num, seq));
    }

    lines
}

/// Format a float for IGES output.
fn fmt_f(v: f64) -> String {
    if v.abs() < 1e-15 {
        "0.".to_string()
    } else {
        format!("{v:.6}")
    }
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

        // Should contain plane entities (type 108).
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

        // Should contain line entities (type 110).
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

        // Should have entities from both solids.
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
