//! NURBS curve and surface manipulation bindings.

#![allow(clippy::missing_errors_doc, clippy::too_many_arguments)]

use wasm_bindgen::prelude::*;

use brepkit_math::vec::Point3;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::vertex::Vertex;

use crate::error::WasmError;
use crate::handles::{edge_id_to_u32, face_id_to_u32};
use crate::helpers::{TOL, parse_point_grid, parse_points};
use crate::kernel::BrepKernel;

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
        let v_start = self.topo.add_vertex(Vertex::new(start, TOL));
        let v_end = self.topo.add_vertex(Vertex::new(end, TOL));
        let eid = self
            .topo
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
}
