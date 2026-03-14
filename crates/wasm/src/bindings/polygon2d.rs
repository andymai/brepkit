//! 2D polygon operation bindings.

#![allow(clippy::missing_errors_doc)]

use wasm_bindgen::prelude::*;

use crate::error::{WasmError, validate_positive};
use crate::helpers::{
    chamfer_polygon_2d, fillet_polygon_2d, find_common_segments, parse_polygon_2d,
    polygons_overlap_2d, sutherland_hodgman_clip,
};
use crate::kernel::BrepKernel;

#[wasm_bindgen]
impl BrepKernel {
    // ── Batch 6: Polygon offset ──────────────────────────────────────

    /// Offset a 2D polygon by a signed distance.
    ///
    /// `coords` is a flat array `[x,y, x,y, ...]` of 2D points.
    /// Returns a flat array of offset polygon coordinates.
    #[wasm_bindgen(js_name = "offsetPolygon2d")]
    #[allow(clippy::needless_pass_by_value, clippy::unused_self)]
    pub fn offset_polygon_2d(
        &self,
        coords: Vec<f64>,
        distance: f64,
        tolerance: f64,
    ) -> Result<Vec<f64>, JsError> {
        if coords.len() % 2 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "2D coordinate array length must be even, got {}",
                    coords.len()
                ),
            }
            .into());
        }
        let points: Vec<brepkit_math::vec::Point2> = coords
            .chunks_exact(2)
            .map(|c| brepkit_math::vec::Point2::new(c[0], c[1]))
            .collect();
        let result = brepkit_math::polygon_offset::offset_polygon_2d(&points, distance, tolerance)?;
        Ok(result.iter().flat_map(|p| [p.x(), p.y()]).collect())
    }

    // ── 2D Blueprint Operations ────────────────────────────────────

    /// Test if a 2D point is inside a closed polygon.
    ///
    /// `polygon_coords` is a flat array `[x,y, x,y, ...]`.
    /// Returns `true` if the point is inside the polygon (winding number test).
    #[wasm_bindgen(js_name = "pointInPolygon2d")]
    #[allow(clippy::unused_self)]
    pub fn point_in_polygon_2d(
        &self,
        polygon_coords: Vec<f64>,
        px: f64,
        py: f64,
    ) -> Result<bool, JsError> {
        if polygon_coords.len() % 2 != 0 || polygon_coords.len() < 6 {
            return Err(WasmError::InvalidInput {
                reason: "polygon needs at least 3 points (6 coordinates)".into(),
            }
            .into());
        }
        let polygon: Vec<brepkit_math::vec::Point2> = polygon_coords
            .chunks_exact(2)
            .map(|c| brepkit_math::vec::Point2::new(c[0], c[1]))
            .collect();
        let point = brepkit_math::vec::Point2::new(px, py);
        Ok(brepkit_math::predicates::point_in_polygon(point, &polygon))
    }

    /// Test if two 2D polygons intersect (overlap).
    ///
    /// Both polygons are flat arrays `[x,y, x,y, ...]`.
    /// Returns `true` if any vertex of one polygon is inside the other
    /// or if any edges cross.
    #[wasm_bindgen(js_name = "polygonsIntersect2d")]
    #[allow(clippy::unused_self)]
    pub fn polygons_intersect_2d(
        &self,
        coords_a: Vec<f64>,
        coords_b: Vec<f64>,
    ) -> Result<bool, JsError> {
        let poly_a = parse_polygon_2d(&coords_a)?;
        let poly_b = parse_polygon_2d(&coords_b)?;
        Ok(polygons_overlap_2d(&poly_a, &poly_b))
    }

    /// Compute the boolean intersection of two 2D polygons.
    ///
    /// Both polygons are flat arrays `[x,y, x,y, ...]`.
    /// Returns a flat array of the intersection polygon coordinates,
    /// or an empty array if they don't intersect.
    ///
    /// Uses the Sutherland-Hodgman algorithm (convex clipper).
    #[wasm_bindgen(js_name = "intersectPolygons2d")]
    #[allow(clippy::unused_self)]
    pub fn intersect_polygons_2d(
        &self,
        coords_a: Vec<f64>,
        coords_b: Vec<f64>,
    ) -> Result<Vec<f64>, JsError> {
        let subject = parse_polygon_2d(&coords_a)?;
        let clip = parse_polygon_2d(&coords_b)?;
        let result = sutherland_hodgman_clip(&subject, &clip);
        Ok(result.iter().flat_map(|p| [p.x(), p.y()]).collect())
    }

    /// Find common (shared) edges between two adjacent 2D polygons.
    ///
    /// Both polygons are flat arrays `[x,y, x,y, ...]`.
    /// Returns a flat array of common segment endpoints `[x1,y1, x2,y2, ...]`,
    /// or an empty array if no common segments exist.
    #[wasm_bindgen(js_name = "commonSegment2d")]
    #[allow(clippy::unused_self)]
    pub fn common_segment_2d(
        &self,
        coords_a: Vec<f64>,
        coords_b: Vec<f64>,
    ) -> Result<Vec<f64>, JsError> {
        let poly_a = parse_polygon_2d(&coords_a)?;
        let poly_b = parse_polygon_2d(&coords_b)?;
        let tolerance = 1e-7;
        let result = find_common_segments(&poly_a, &poly_b, tolerance);
        Ok(result
            .iter()
            .flat_map(|(a, b)| [a.x(), a.y(), b.x(), b.y()])
            .collect())
    }

    /// Round corners of a 2D polygon by inserting arc-approximation vertices.
    ///
    /// `coords` is a flat array `[x,y, x,y, ...]`.
    /// `radius` is the fillet radius.
    /// Returns a flat array of the filleted polygon coordinates.
    #[wasm_bindgen(js_name = "fillet2d")]
    #[allow(clippy::unused_self)]
    pub fn fillet_2d(&self, coords: Vec<f64>, radius: f64) -> Result<Vec<f64>, JsError> {
        validate_positive(radius, "radius")?;
        let polygon = parse_polygon_2d(&coords)?;
        let result = fillet_polygon_2d(&polygon, radius);
        Ok(result.iter().flat_map(|p| [p.x(), p.y()]).collect())
    }

    /// Cut corners of a 2D polygon with flat bevels.
    ///
    /// `coords` is a flat array `[x,y, x,y, ...]`.
    /// `distance` is the chamfer distance from each corner.
    /// Returns a flat array of the chamfered polygon coordinates.
    #[wasm_bindgen(js_name = "chamfer2d")]
    #[allow(clippy::unused_self)]
    pub fn chamfer_2d(&self, coords: Vec<f64>, distance: f64) -> Result<Vec<f64>, JsError> {
        validate_positive(distance, "distance")?;
        let polygon = parse_polygon_2d(&coords)?;
        let result = chamfer_polygon_2d(&polygon, distance);
        Ok(result.iter().flat_map(|p| [p.x(), p.y()]).collect())
    }
}
