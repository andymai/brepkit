//! NURBS curve and surface manipulation bindings.

#![allow(clippy::missing_errors_doc, clippy::too_many_arguments)]

use wasm_bindgen::prelude::*;

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::Point3;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::vertex::Vertex;

use crate::error::WasmError;
use crate::handles::{edge_id_to_u32, face_id_to_u32};
use crate::helpers::{TOL, parse_point_grid, parse_points};
use crate::kernel::BrepKernel;

/// Weights within this absolute distance of `1.0` are treated as unit.
const WEIGHT_UNIT_TOL: f64 = 1e-12;

/// Squared linear distance below which control rows/columns are coincident.
const CLOSURE_TOL_SQ: f64 = 1e-14;

/// Split a flat (repeated) knot vector into distinct values + multiplicities.
fn compress_knots(knots: &[f64]) -> (Vec<f64>, Vec<u32>) {
    let mut distinct: Vec<f64> = Vec::new();
    let mut mult: Vec<u32> = Vec::new();
    for &k in knots {
        match distinct.last() {
            Some(&last) if (k - last).abs() <= f64::EPSILON => {
                if let Some(m) = mult.last_mut() {
                    *m += 1;
                }
            }
            _ => {
                distinct.push(k);
                mult.push(1);
            }
        }
    }
    (distinct, mult)
}

/// Serialize a `NurbsCurve` to the read-only extraction object.
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn curve_data_json(curve: &NurbsCurve) -> serde_json::Value {
    let cps: Vec<[f64; 3]> = curve
        .control_points()
        .iter()
        .map(|p| [p.x(), p.y(), p.z()])
        .collect();
    let weights = curve.weights();
    let rational = weights.iter().any(|&w| (w - 1.0).abs() > WEIGHT_UNIT_TOL);
    let (a, b) = curve.domain();
    let (distinct, mult) = compress_knots(curve.knots());

    let closed = curve
        .control_points()
        .first()
        .zip(curve.control_points().last())
        .is_some_and(|(first, last)| (*last - *first).length_squared() < CLOSURE_TOL_SQ);

    serde_json::json!({
        "degree": curve.degree(),
        "controlPoints": cps,
        "weights": weights,
        "knots": curve.knots(),
        "distinctKnots": distinct,
        "multiplicities": mult,
        "rational": rational,
        "closed": closed,
        "periodic": closed,
        "domain": [a, b],
    })
}

/// Serialize a `NurbsSurface` to the read-only extraction object.
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn surface_data_json(surface: &NurbsSurface) -> serde_json::Value {
    let cps: Vec<Vec<[f64; 3]>> = surface
        .control_points()
        .iter()
        .map(|row| row.iter().map(|p| [p.x(), p.y(), p.z()]).collect())
        .collect();
    let weights = surface.weights();
    let rational = weights
        .iter()
        .flatten()
        .any(|&w| (w - 1.0).abs() > WEIGHT_UNIT_TOL);

    let rows = surface.control_points();
    let periodic_u = rows
        .first()
        .zip(rows.last())
        .is_some_and(|(first, last)| rows_coincide(first, last));
    let periodic_v = rows.iter().all(|row| {
        row.first()
            .zip(row.last())
            .is_some_and(|(a, b)| (*b - *a).length_squared() < CLOSURE_TOL_SQ)
    }) && rows.first().is_some_and(|r| r.len() >= 2);

    let (ua, ub) = surface.domain_u();
    let (va, vb) = surface.domain_v();
    let (distinct_u, mult_u) = compress_knots(surface.knots_u());
    let (distinct_v, mult_v) = compress_knots(surface.knots_v());

    serde_json::json!({
        "degreeU": surface.degree_u(),
        "degreeV": surface.degree_v(),
        "controlPoints": cps,
        "weights": weights,
        "knotsU": surface.knots_u(),
        "knotsV": surface.knots_v(),
        "distinctKnotsU": distinct_u,
        "multiplicitiesU": mult_u,
        "distinctKnotsV": distinct_v,
        "multiplicitiesV": mult_v,
        "rational": rational,
        "periodicU": periodic_u,
        "periodicV": periodic_v,
        "domainU": [ua, ub],
        "domainV": [va, vb],
    })
}

/// Serialize a `NurbsSurface` to the parity extraction object.
///
/// The parity shape uses `poles`/`nbPolesU`/`nbPolesV`, distinct `knotsU`/`knotsV`
/// paired with `multiplicitiesU`/`multiplicitiesV`, and `isPeriodicU`/`isPeriodicV`/
/// `isRational`. Distinct knots are emitted (not the flat vector); a consumer
/// rebuilds the flat vector by repeating each value by its multiplicity.
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn surface_data_parity_json(surface: &NurbsSurface) -> serde_json::Value {
    let poles: Vec<Vec<[f64; 3]>> = surface
        .control_points()
        .iter()
        .map(|row| row.iter().map(|p| [p.x(), p.y(), p.z()]).collect())
        .collect();
    let weights = surface.weights();
    let rational = weights
        .iter()
        .flatten()
        .any(|&w| (w - 1.0).abs() > WEIGHT_UNIT_TOL);

    let nb_poles_u = poles.len();
    let nb_poles_v = poles.first().map_or(0, Vec::len);

    let (distinct_u, mult_u) = compress_knots(surface.knots_u());
    let (distinct_v, mult_v) = compress_knots(surface.knots_v());

    serde_json::json!({
        "degreeU": surface.degree_u(),
        "degreeV": surface.degree_v(),
        "nbPolesU": nb_poles_u,
        "nbPolesV": nb_poles_v,
        "poles": poles,
        "weights": weights,
        "knotsU": distinct_u,
        "knotsV": distinct_v,
        "multiplicitiesU": mult_u,
        "multiplicitiesV": mult_v,
        "isPeriodicU": surface.is_periodic_u(),
        "isPeriodicV": surface.is_periodic_v(),
        "isRational": rational,
    })
}

/// Two control rows coincide pointwise within the closure tolerance.
fn rows_coincide(a: &[Point3], b: &[Point3]) -> bool {
    a.len() == b.len()
        && !a.is_empty()
        && a.iter()
            .zip(b)
            .all(|(p, q)| (*q - *p).length_squared() < CLOSURE_TOL_SQ)
}

#[wasm_bindgen]
impl BrepKernel {
    /// Interpolate a NURBS curve through points and create an edge.
    ///
    /// Uses chord-length parameterization with the given degree.
    /// Returns an edge handle (`u32`).
    #[wasm_bindgen(js_name = "interpolatePoints")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn interpolate_points(&mut self, coords: Vec<f64>, degree: u32) -> Result<u32, JsError> {
        if coords.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "coordinate array length must be a multiple of 3, got {}",
                    coords.len()
                ),
            }
            .into());
        }
        let points: Vec<Point3> = coords
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();
        if points.len() < 2 {
            return Err(WasmError::InvalidInput {
                reason: format!("need at least 2 points, got {}", points.len()),
            }
            .into());
        }

        let deg = std::cmp::min(degree as usize, points.len() - 1);
        let curve = brepkit_math::nurbs::fitting::interpolate(&points, deg)?;

        let start = points[0];
        let end = points[points.len() - 1];
        let v_start = self.topo_mut().add_vertex(Vertex::new(start, TOL));
        let v_end = self.topo_mut().add_vertex(Vertex::new(end, TOL));
        let eid = self
            .topo_mut()
            .add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(curve)));
        Ok(edge_id_to_u32(eid))
    }

    /// Approximate a curve through points (least-squares).
    ///
    /// Returns an edge handle.
    #[wasm_bindgen(js_name = "approximateCurve")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn approximate_curve(
        &mut self,
        coords: Vec<f64>,
        degree: u32,
        num_control_points: u32,
    ) -> Result<u32, JsError> {
        let points = parse_points(&coords)?;
        if points.len() < 2 {
            return Err(WasmError::InvalidInput {
                reason: format!("need at least 2 points, got {}", points.len()),
            }
            .into());
        }
        let deg = std::cmp::min(degree as usize, points.len() - 1);
        let curve =
            brepkit_math::nurbs::fitting::approximate(&points, deg, num_control_points as usize)?;
        Ok(edge_id_to_u32(self.nurbs_curve_to_edge(&points, curve)))
    }

    /// Approximate a curve through points using LSPIA (progressive iteration).
    ///
    /// Returns an edge handle.
    #[wasm_bindgen(js_name = "approximateCurveLspia")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn approximate_curve_lspia(
        &mut self,
        coords: Vec<f64>,
        degree: u32,
        num_control_points: u32,
        tolerance: f64,
        max_iterations: u32,
    ) -> Result<u32, JsError> {
        let points = parse_points(&coords)?;
        if points.len() < 2 {
            return Err(WasmError::InvalidInput {
                reason: format!("need at least 2 points, got {}", points.len()),
            }
            .into());
        }
        let deg = std::cmp::min(degree as usize, points.len() - 1);
        let curve = brepkit_math::nurbs::fitting::approximate_lspia(
            &points,
            deg,
            num_control_points as usize,
            tolerance,
            max_iterations as usize,
        )?;
        Ok(edge_id_to_u32(self.nurbs_curve_to_edge(&points, curve)))
    }

    /// Interpolate a grid of points into a NURBS surface.
    ///
    /// `coords` is a flat array `[x,y,z, ...]` of `rows * cols` points.
    /// Returns a face handle.
    #[wasm_bindgen(js_name = "interpolateSurface")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn interpolate_surface(
        &mut self,
        coords: Vec<f64>,
        rows: u32,
        cols: u32,
        degree_u: u32,
        degree_v: u32,
    ) -> Result<u32, JsError> {
        let grid = parse_point_grid(&coords, rows as usize, cols as usize)?;
        let surface = brepkit_math::nurbs::surface_fitting::interpolate_surface(
            &grid,
            degree_u as usize,
            degree_v as usize,
        )?;
        Ok(face_id_to_u32(self.nurbs_surface_to_face(surface)?))
    }

    /// Approximate a grid of points into a NURBS surface using LSPIA.
    ///
    /// Returns a face handle.
    #[wasm_bindgen(js_name = "approximateSurfaceLspia")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn approximate_surface_lspia(
        &mut self,
        coords: Vec<f64>,
        rows: u32,
        cols: u32,
        degree_u: u32,
        degree_v: u32,
        num_cps_u: u32,
        num_cps_v: u32,
        tolerance: f64,
        max_iterations: u32,
    ) -> Result<u32, JsError> {
        let grid = parse_point_grid(&coords, rows as usize, cols as usize)?;
        let surface = brepkit_math::nurbs::surface_fitting::approximate_surface_lspia(
            &grid,
            degree_u as usize,
            degree_v as usize,
            num_cps_u as usize,
            num_cps_v as usize,
            tolerance,
            max_iterations as usize,
        )?;
        Ok(face_id_to_u32(self.nurbs_surface_to_face(surface)?))
    }

    /// Insert a knot into an edge's NURBS curve.
    ///
    /// Returns a new edge handle with the refined curve.
    #[wasm_bindgen(js_name = "curveKnotInsert")]
    pub fn curve_knot_insert(&mut self, edge: u32, knot: f64, times: u32) -> Result<u32, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        let refined =
            brepkit_math::nurbs::knot_ops::curve_knot_insert(&curve, knot, times as usize)?;
        Ok(edge_id_to_u32(
            self.nurbs_curve_to_edge_from_curve(&refined),
        ))
    }

    /// Remove a knot from an edge's NURBS curve.
    ///
    /// Returns a new edge handle with the simplified curve.
    #[wasm_bindgen(js_name = "curveKnotRemove")]
    pub fn curve_knot_remove(
        &mut self,
        edge: u32,
        knot: f64,
        tolerance: f64,
    ) -> Result<u32, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        let simplified = brepkit_math::nurbs::knot_ops::curve_knot_remove(&curve, knot, tolerance)?;
        Ok(edge_id_to_u32(
            self.nurbs_curve_to_edge_from_curve(&simplified),
        ))
    }

    /// Split an edge's NURBS curve at a parameter value.
    ///
    /// Returns two edge handles as `[u32; 2]`.
    #[wasm_bindgen(js_name = "curveSplit")]
    pub fn curve_split(&mut self, edge: u32, u: f64) -> Result<Vec<u32>, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        let (left, right) = brepkit_math::nurbs::knot_ops::curve_split(&curve, u)?;
        let e1 = self.nurbs_curve_to_edge_from_curve(&left);
        let e2 = self.nurbs_curve_to_edge_from_curve(&right);
        Ok(vec![edge_id_to_u32(e1), edge_id_to_u32(e2)])
    }

    /// Elevate the degree of an edge's NURBS curve.
    ///
    /// Returns a new edge handle.
    #[wasm_bindgen(js_name = "curveDegreeElevate")]
    pub fn curve_degree_elevate(&mut self, edge: u32, elevate_by: u32) -> Result<u32, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        let elevated =
            brepkit_math::nurbs::decompose::curve_degree_elevate(&curve, elevate_by as usize)?;
        Ok(edge_id_to_u32(
            self.nurbs_curve_to_edge_from_curve(&elevated),
        ))
    }

    /// Read-only canonical NURBS data for the curve underlying an edge.
    ///
    /// Analytic curves (line, circle, ellipse) are converted to their exact
    /// NURBS form. Returns a JSON string with `degree`, `controlPoints`,
    /// `weights`, the flat `knots` vector, compressed `distinctKnots` /
    /// `multiplicities`, `rational`, `closed` / `periodic`, and `domain`.
    #[wasm_bindgen(js_name = "getNurbsCurveData")]
    pub fn get_nurbs_curve_data(&self, edge: u32) -> Result<String, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        Ok(curve_data_json(&curve).to_string())
    }

    /// Read-only canonical NURBS data for the surface underlying a face.
    ///
    /// Analytic surfaces are converted to NURBS (planes/cylinders exact;
    /// cones/spheres/tori via the exact rational forms). Returns a JSON
    /// string with `degreeU`/`degreeV`, the row-major `controlPoints` grid,
    /// the matching `weights` grid, flat `knotsU`/`knotsV`, compressed
    /// distinct-knots/multiplicities per direction, `rational`,
    /// `periodicU`/`periodicV`, and `domainU`/`domainV`.
    #[wasm_bindgen(js_name = "getNurbsSurfaceData")]
    pub fn get_nurbs_surface_data(&self, face: u32) -> Result<String, JsError> {
        let surface = self.extract_nurbs_surface(face)?;
        Ok(surface_data_json(&surface).to_string())
    }

    /// Type-gated read-only B-Spline/NURBS surface data for a face.
    ///
    /// Unlike `getNurbsSurfaceData`, this never converts analytic surfaces:
    /// faces backed by a plane, cylinder, cone, sphere, or torus return the
    /// JSON literal `null`. Only intrinsically free-form (B-Spline/NURBS) faces
    /// yield a record with `degreeU`/`degreeV`, `nbPolesU`/`nbPolesV`, the
    /// row-major `poles` grid (u-major, v-minor) with the matching `weights`
    /// grid, distinct `knotsU`/`knotsV` paired with `multiplicitiesU`/
    /// `multiplicitiesV`, `isPeriodicU`/`isPeriodicV`, and `isRational`.
    #[wasm_bindgen(js_name = "getNurbsSurfaceDataParity")]
    pub fn get_nurbs_surface_data_parity(&self, face: u32) -> Result<String, JsError> {
        Ok(self.free_form_surface_data_parity(face)?.to_string())
    }
}

impl BrepKernel {
    /// Return parity surface data only for intrinsically free-form faces,
    /// else JSON `null`. Pure: resolves the face without touching the arena.
    pub(crate) fn free_form_surface_data_parity(
        &self,
        face: u32,
    ) -> Result<serde_json::Value, WasmError> {
        use brepkit_topology::face::FaceSurface;

        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        match face_data.surface() {
            FaceSurface::Nurbs(s) => Ok(surface_data_parity_json(s)),
            FaceSurface::Plane { .. }
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_) => Ok(serde_json::Value::Null),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::nurbs::surface_fitting::interpolate_surface;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::builder::{make_nurbs_edge_from_curve, make_nurbs_face};

    use crate::handles::{edge_id_to_u32, face_id_to_u32};
    use crate::helpers::TOL;
    use crate::kernel::BrepKernel;

    use super::*;

    fn dispatch_ok(k: &mut BrepKernel, op: &str, args: serde_json::Value) -> serde_json::Value {
        let batch = serde_json::json!([{ "op": op, "args": args }]);
        let out = k.execute_batch(&batch.to_string());
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&out).unwrap();
        let entry = &parsed[0];
        assert!(
            entry.get("error").is_none(),
            "dispatch {op} errored: {entry}"
        );
        entry["ok"].clone()
    }

    fn knot_distance_curve(degree: usize, cp: usize) -> usize {
        cp + degree + 1
    }

    fn entity_counts(k: &BrepKernel) -> (usize, usize, usize) {
        (
            k.topo.num_faces(),
            k.topo.num_edges(),
            k.topo.num_vertices(),
        )
    }

    #[test]
    fn curve_data_round_trips_cubic_nurbs() {
        let mut k = BrepKernel::new();
        let cps = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(3.0, 2.0, 1.0),
            Point3::new(4.0, 0.0, 0.0),
            Point3::new(6.0, -1.0, 2.0),
        ];
        let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let weights = vec![1.0; 5];
        let curve = NurbsCurve::new(3, knots.clone(), cps.clone(), weights.clone()).unwrap();
        let eid = make_nurbs_edge_from_curve(k.topo_mut(), &curve, TOL);
        let handle = edge_id_to_u32(eid);

        let before = entity_counts(&k);
        let data = dispatch_ok(
            &mut k,
            "getNurbsCurveData",
            serde_json::json!({"edge": handle}),
        );
        assert_eq!(before, entity_counts(&k), "query must not mutate topology");

        assert_eq!(data["degree"].as_u64().unwrap(), 3);
        assert!(!data["rational"].as_bool().unwrap());
        let out_knots: Vec<f64> = serde_json::from_value(data["knots"].clone()).unwrap();
        assert_eq!(out_knots.len(), knot_distance_curve(3, 5));
        for (a, b) in out_knots.iter().zip(&knots) {
            assert!((a - b).abs() < 1e-12);
        }
        let out_cps: Vec<[f64; 3]> = serde_json::from_value(data["controlPoints"].clone()).unwrap();
        assert_eq!(out_cps.len(), 5);
        for (a, b) in out_cps.iter().zip(&cps) {
            assert!((a[0] - b.x()).abs() < 1e-12);
            assert!((a[1] - b.y()).abs() < 1e-12);
            assert!((a[2] - b.z()).abs() < 1e-12);
        }
        let out_w: Vec<f64> = serde_json::from_value(data["weights"].clone()).unwrap();
        assert_eq!(out_w, weights);

        // Domain matches [knot[degree], knot[len-degree-1]].
        let domain: [f64; 2] = serde_json::from_value(data["domain"].clone()).unwrap();
        assert!(domain[1] > domain[0]);

        // Compressed knots expand to the flat vector.
        let dk: Vec<f64> = serde_json::from_value(data["distinctKnots"].clone()).unwrap();
        let mult: Vec<u32> = serde_json::from_value(data["multiplicities"].clone()).unwrap();
        let expanded: Vec<f64> = dk
            .iter()
            .zip(&mult)
            .flat_map(|(&v, &m)| std::iter::repeat_n(v, m as usize))
            .collect();
        assert_eq!(expanded, out_knots);
    }

    #[test]
    fn circle_edge_extracts_rational_quadratic() {
        let mut k = BrepKernel::new();
        let radius = 2.5;
        let circle = brepkit_math::curves::Circle3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            radius,
        )
        .unwrap();
        let start = Point3::new(radius, 0.0, 0.0);
        let v = k.topo_mut().add_vertex(Vertex::new(start, TOL));
        let eid = k.topo_mut().add_edge(Edge::new(
            v,
            v,
            brepkit_topology::edge::EdgeCurve::Circle(circle),
        ));
        let handle = edge_id_to_u32(eid);

        let data = dispatch_ok(
            &mut k,
            "getNurbsCurveData",
            serde_json::json!({"edge": handle}),
        );
        assert_eq!(data["degree"].as_u64().unwrap(), 2);
        assert!(data["rational"].as_bool().unwrap());

        // Reconstruct and sample: every point at distance `radius` from center.
        let out_cps: Vec<Point3> =
            serde_json::from_value::<Vec<[f64; 3]>>(data["controlPoints"].clone())
                .unwrap()
                .into_iter()
                .map(|c| Point3::new(c[0], c[1], c[2]))
                .collect();
        let out_knots: Vec<f64> = serde_json::from_value(data["knots"].clone()).unwrap();
        let out_w: Vec<f64> = serde_json::from_value(data["weights"].clone()).unwrap();
        let degree = data["degree"].as_u64().unwrap() as usize;
        let rebuilt = NurbsCurve::new(degree, out_knots, out_cps, out_w).unwrap();
        let (a, b) = rebuilt.domain();
        for i in 0..=16 {
            let t = a + (b - a) * (i as f64 / 16.0);
            let p = rebuilt.evaluate(t);
            let r = (p.x() * p.x() + p.y() * p.y()).sqrt();
            assert!((r - radius).abs() < 1e-9, "sample r={r} at t={t}");
        }
    }

    #[test]
    fn surface_data_round_trips_nurbs_face() {
        let mut k = BrepKernel::new();
        let mut grid = Vec::new();
        for i in 0..4 {
            let mut row = Vec::new();
            for j in 0..4 {
                let x = i as f64;
                let y = j as f64;
                let z = (x * y).sin() * 0.3;
                row.push(Point3::new(x, y, z));
            }
            grid.push(row);
        }
        let surface = interpolate_surface(&grid, 3, 3).unwrap();
        let fid = make_nurbs_face(k.topo_mut(), surface.clone(), TOL).unwrap();
        let handle = face_id_to_u32(fid);

        let data = dispatch_ok(
            &mut k,
            "getNurbsSurfaceData",
            serde_json::json!({"face": handle}),
        );

        let du = data["degreeU"].as_u64().unwrap() as usize;
        let dv = data["degreeV"].as_u64().unwrap() as usize;
        let knots_u: Vec<f64> = serde_json::from_value(data["knotsU"].clone()).unwrap();
        let knots_v: Vec<f64> = serde_json::from_value(data["knotsV"].clone()).unwrap();
        let cps: Vec<Vec<[f64; 3]>> =
            serde_json::from_value(data["controlPoints"].clone()).unwrap();
        let weights: Vec<Vec<f64>> = serde_json::from_value(data["weights"].clone()).unwrap();

        let nu = cps.len();
        let nv = cps[0].len();
        assert!(cps.iter().all(|row| row.len() == nv), "grid is rectangular");
        assert_eq!(knots_u.len(), nu + du + 1);
        assert_eq!(knots_v.len(), nv + dv + 1);
        assert_eq!(weights.len(), nu);
        assert!(weights.iter().all(|r| r.len() == nv));
        assert!(knots_u.windows(2).all(|w| w[1] >= w[0]));
        assert!(knots_v.windows(2).all(|w| w[1] >= w[0]));

        // Round-trip evaluation against the source surface.
        let rebuilt = NurbsSurface::new(
            du,
            dv,
            knots_u,
            knots_v,
            cps.iter()
                .map(|row| row.iter().map(|c| Point3::new(c[0], c[1], c[2])).collect())
                .collect(),
            weights,
        )
        .unwrap();
        let (ua, ub) = surface.domain_u();
        let (va, vb) = surface.domain_v();
        for i in 0..=4 {
            for j in 0..=4 {
                let u = ua + (ub - ua) * (i as f64 / 4.0);
                let v = va + (vb - va) * (j as f64 / 4.0);
                let p0 = surface.evaluate(u, v);
                let p1 = rebuilt.evaluate(u, v);
                let d = (p1 - p0).length();
                assert!(d < 1e-9, "round-trip mismatch {d} at ({u},{v})");
            }
        }
    }

    #[test]
    fn planar_face_extracts_unit_degree1_grid() {
        let mut k = BrepKernel::new();
        let solid = brepkit_operations::primitives::make_box(k.topo_mut(), 4.0, 6.0, 2.0).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&k.topo, solid).unwrap();
        let handle = face_id_to_u32(faces[0]);

        let data = dispatch_ok(
            &mut k,
            "getNurbsSurfaceData",
            serde_json::json!({"face": handle}),
        );
        assert_eq!(data["degreeU"].as_u64().unwrap(), 1);
        assert_eq!(data["degreeV"].as_u64().unwrap(), 1);
        assert!(!data["rational"].as_bool().unwrap());
        let cps: Vec<Vec<[f64; 3]>> =
            serde_json::from_value(data["controlPoints"].clone()).unwrap();
        assert_eq!(cps.len(), 2);
        assert!(cps.iter().all(|row| row.len() == 2));
        let weights: Vec<Vec<f64>> = serde_json::from_value(data["weights"].clone()).unwrap();
        assert!(weights.iter().flatten().all(|&w| (w - 1.0).abs() < 1e-12));
    }

    #[test]
    fn cylinder_cap_face_extracts_unit_degree1_grid() {
        let mut k = BrepKernel::new();
        let radius = 3.0;
        let height = 5.0;
        let solid =
            brepkit_operations::primitives::make_cylinder(k.topo_mut(), radius, height).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&k.topo, solid).unwrap();
        let cap = faces
            .iter()
            .copied()
            .find(|&f| {
                matches!(
                    k.topo.face(f).unwrap().surface(),
                    brepkit_topology::face::FaceSurface::Plane { .. }
                )
            })
            .unwrap();
        let handle = face_id_to_u32(cap);

        let data = dispatch_ok(
            &mut k,
            "getNurbsSurfaceData",
            serde_json::json!({"face": handle}),
        );
        assert_eq!(data["degreeU"].as_u64().unwrap(), 1);
        assert_eq!(data["degreeV"].as_u64().unwrap(), 1);
        assert!(!data["rational"].as_bool().unwrap());

        let cps: Vec<Vec<[f64; 3]>> =
            serde_json::from_value(data["controlPoints"].clone()).unwrap();
        assert_eq!(cps.len(), 2);
        assert!(cps.iter().all(|row| row.len() == 2));

        // The 2x2 corner grid must enclose the cap disk: the half-diagonal
        // extent in plane must be at least the radius in each direction.
        let xs: Vec<f64> = cps.iter().flatten().map(|c| c[0]).collect();
        let ys: Vec<f64> = cps.iter().flatten().map(|c| c[1]).collect();
        let dx = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - xs.iter().copied().fold(f64::INFINITY, f64::min);
        let dy = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - ys.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(dx >= 2.0 * radius - 1e-9, "cap u-extent {dx}");
        assert!(dy >= 2.0 * radius - 1e-9, "cap v-extent {dy}");
    }

    fn first_analytic_face(k: &BrepKernel, solid: brepkit_topology::solid::SolidId) -> u32 {
        let faces = brepkit_topology::explorer::solid_faces(&k.topo, solid).unwrap();
        face_id_to_u32(faces[0])
    }

    #[test]
    fn parity_planar_face_returns_null() {
        let mut k = BrepKernel::new();
        let solid = brepkit_operations::primitives::make_box(k.topo_mut(), 4.0, 6.0, 2.0).unwrap();
        let handle = first_analytic_face(&k, solid);

        let before = entity_counts(&k);
        let data = dispatch_ok(
            &mut k,
            "getNurbsSurfaceDataParity",
            serde_json::json!({"face": handle}),
        );
        assert_eq!(before, entity_counts(&k), "query must not mutate topology");
        assert!(data.is_null(), "planar face must yield null, got {data}");
    }

    #[test]
    fn parity_analytic_faces_return_null() {
        let mut k = BrepKernel::new();
        let cyl = brepkit_operations::primitives::make_cylinder(k.topo_mut(), 3.0, 5.0).unwrap();
        let cone = brepkit_operations::primitives::make_cone(k.topo_mut(), 3.0, 1.0, 5.0).unwrap();
        let sphere = brepkit_operations::primitives::make_sphere(k.topo_mut(), 2.0, 16).unwrap();
        let torus = brepkit_operations::primitives::make_torus(k.topo_mut(), 4.0, 1.0, 16).unwrap();

        for solid in [cyl, cone, sphere, torus] {
            let faces = brepkit_topology::explorer::solid_faces(&k.topo, solid).unwrap();
            for fid in faces {
                let handle = face_id_to_u32(fid);
                let data = dispatch_ok(
                    &mut k,
                    "getNurbsSurfaceDataParity",
                    serde_json::json!({"face": handle}),
                );
                assert!(
                    data.is_null(),
                    "analytic face {handle} must yield null, got {data}"
                );
            }
        }
    }

    #[test]
    fn parity_nurbs_face_extracts_full_data() {
        let mut k = BrepKernel::new();
        let mut grid = Vec::new();
        for i in 0..4 {
            let mut row = Vec::new();
            for j in 0..4 {
                let x = i as f64;
                let y = j as f64;
                let z = (x * y).sin() * 0.3;
                row.push(Point3::new(x, y, z));
            }
            grid.push(row);
        }
        let surface = interpolate_surface(&grid, 3, 3).unwrap();
        let fid = make_nurbs_face(k.topo_mut(), surface.clone(), TOL).unwrap();
        let handle = face_id_to_u32(fid);

        let before = entity_counts(&k);
        let data = dispatch_ok(
            &mut k,
            "getNurbsSurfaceDataParity",
            serde_json::json!({"face": handle}),
        );
        assert_eq!(before, entity_counts(&k), "query must not mutate topology");
        assert!(!data.is_null());

        let du = data["degreeU"].as_u64().unwrap() as usize;
        let dv = data["degreeV"].as_u64().unwrap() as usize;
        let nb_u = data["nbPolesU"].as_u64().unwrap() as usize;
        let nb_v = data["nbPolesV"].as_u64().unwrap() as usize;
        let poles: Vec<Vec<[f64; 3]>> = serde_json::from_value(data["poles"].clone()).unwrap();
        let weights: Vec<Vec<f64>> = serde_json::from_value(data["weights"].clone()).unwrap();
        let knots_u: Vec<f64> = serde_json::from_value(data["knotsU"].clone()).unwrap();
        let knots_v: Vec<f64> = serde_json::from_value(data["knotsV"].clone()).unwrap();
        let mult_u: Vec<u32> = serde_json::from_value(data["multiplicitiesU"].clone()).unwrap();
        let mult_v: Vec<u32> = serde_json::from_value(data["multiplicitiesV"].clone()).unwrap();

        assert_eq!(poles.len(), nb_u);
        assert!(poles.iter().all(|r| r.len() == nb_v), "rectangular grid");
        assert_eq!(weights.len(), nb_u);
        assert!(weights.iter().all(|r| r.len() == nb_v));
        assert!(!data["isRational"].as_bool().unwrap());

        assert_eq!(knots_u.len(), mult_u.len());
        assert_eq!(knots_v.len(), mult_v.len());
        assert!(
            knots_u.windows(2).all(|w| w[1] > w[0]),
            "strictly increasing"
        );
        assert!(
            knots_v.windows(2).all(|w| w[1] > w[0]),
            "strictly increasing"
        );
        assert_eq!(mult_u.iter().sum::<u32>() as usize, nb_u + du + 1);
        assert_eq!(mult_v.iter().sum::<u32>() as usize, nb_v + dv + 1);

        let flat_u: Vec<f64> = knots_u
            .iter()
            .zip(&mult_u)
            .flat_map(|(&v, &m)| std::iter::repeat_n(v, m as usize))
            .collect();
        let flat_v: Vec<f64> = knots_v
            .iter()
            .zip(&mult_v)
            .flat_map(|(&v, &m)| std::iter::repeat_n(v, m as usize))
            .collect();

        let rebuilt = NurbsSurface::new(
            du,
            dv,
            flat_u,
            flat_v,
            poles
                .iter()
                .map(|row| row.iter().map(|c| Point3::new(c[0], c[1], c[2])).collect())
                .collect(),
            weights,
        )
        .unwrap();
        let (ua, ub) = surface.domain_u();
        let (va, vb) = surface.domain_v();
        for i in 0..=4 {
            for j in 0..=4 {
                let u = ua + (ub - ua) * (i as f64 / 4.0);
                let v = va + (vb - va) * (j as f64 / 4.0);
                let d = (rebuilt.evaluate(u, v) - surface.evaluate(u, v)).length();
                assert!(d < 1e-9, "round-trip mismatch {d} at ({u},{v})");
            }
        }
    }

    #[test]
    fn cylinder_side_face_extracts_periodic_rational() {
        let mut k = BrepKernel::new();
        let radius = 3.0;
        let height = 5.0;
        let solid =
            brepkit_operations::primitives::make_cylinder(k.topo_mut(), radius, height).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&k.topo, solid).unwrap();
        let side = faces
            .iter()
            .copied()
            .find(|&f| {
                matches!(
                    k.topo.face(f).unwrap().surface(),
                    brepkit_topology::face::FaceSurface::Cylinder(_)
                )
            })
            .unwrap();
        let handle = face_id_to_u32(side);

        let data = dispatch_ok(
            &mut k,
            "getNurbsSurfaceData",
            serde_json::json!({"face": handle}),
        );
        assert_eq!(data["degreeU"].as_u64().unwrap(), 2);
        assert_eq!(data["degreeV"].as_u64().unwrap(), 1);
        assert!(data["rational"].as_bool().unwrap());
        assert!(data["periodicU"].as_bool().unwrap());

        let cps: Vec<Vec<[f64; 3]>> =
            serde_json::from_value(data["controlPoints"].clone()).unwrap();

        // Control points span the face's axial extent [0, height] in z even
        // though the v-knot domain is normalized to [0, 1].
        let z_min = cps
            .iter()
            .flatten()
            .map(|c| c[2])
            .fold(f64::INFINITY, f64::min);
        let z_max = cps
            .iter()
            .flatten()
            .map(|c| c[2])
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((z_min - 0.0).abs() < 1e-9, "axial start z {z_min}");
        assert!((z_max - height).abs() < 1e-9, "axial end z {z_max}");

        let knots_u: Vec<f64> = serde_json::from_value(data["knotsU"].clone()).unwrap();
        let knots_v: Vec<f64> = serde_json::from_value(data["knotsV"].clone()).unwrap();
        let weights: Vec<Vec<f64>> = serde_json::from_value(data["weights"].clone()).unwrap();
        let rebuilt = NurbsSurface::new(
            2,
            1,
            knots_u,
            knots_v,
            cps.iter()
                .map(|row| row.iter().map(|c| Point3::new(c[0], c[1], c[2])).collect())
                .collect(),
            weights,
        )
        .unwrap();
        let (ua, ub) = rebuilt.domain_u();
        let (va, vb) = rebuilt.domain_v();
        for i in 0..=8 {
            for j in 0..=2 {
                let u = ua + (ub - ua) * (i as f64 / 8.0);
                let v = va + (vb - va) * (j as f64 / 2.0);
                let p = rebuilt.evaluate(u, v);
                let r = (p.x() * p.x() + p.y() * p.y()).sqrt();
                assert!((r - radius).abs() < 1e-9, "cyl r={r}");
            }
        }
    }
}
