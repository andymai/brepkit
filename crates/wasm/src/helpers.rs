//! Shared free functions and constants used across WASM binding modules.

#![allow(
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_operations::boolean::BooleanOp;
use brepkit_operations::tessellate;
use brepkit_topology::Topology;
use wasm_bindgen::prelude::*;

use crate::error::WasmError;
use crate::handles::face_id_to_u32;
use crate::shapes::JsMesh;

/// Default tolerance for vertices created by the kernel.
pub const TOL: f64 = 1e-7;

// ── Parsing helpers ───────────────────────────────────────────────

/// Parse flat `[x,y,z, ...]` coordinates into `Vec<Point3>`.
pub fn parse_points(coords: &[f64]) -> Result<Vec<Point3>, JsError> {
    if coords.len() % 3 != 0 {
        return Err(WasmError::InvalidInput {
            reason: format!(
                "coordinate array length must be a multiple of 3, got {}",
                coords.len()
            ),
        }
        .into());
    }
    Ok(coords
        .chunks_exact(3)
        .map(|c| Point3::new(c[0], c[1], c[2]))
        .collect())
}

/// Parse flat coordinates into a 2D grid of points.
pub fn parse_point_grid(
    coords: &[f64],
    rows: usize,
    cols: usize,
) -> Result<Vec<Vec<Point3>>, JsError> {
    if rows == 0 || cols == 0 {
        return Err(WasmError::InvalidInput {
            reason: format!("rows and cols must be > 0, got {rows}x{cols}"),
        }
        .into());
    }
    let total = rows
        .checked_mul(cols)
        .ok_or_else(|| WasmError::InvalidInput {
            reason: format!("rows*cols overflow: {rows}*{cols}"),
        })?;
    let points = parse_points(coords)?;
    if points.len() != total {
        return Err(WasmError::InvalidInput {
            reason: format!(
                "expected {total} points ({rows}x{cols}), got {}",
                points.len()
            ),
        }
        .into());
    }
    Ok(points.chunks(cols).map(<[Point3]>::to_vec).collect())
}

/// Parse a flat 16-element array into a `Mat4` (row-major).
pub fn parse_mat4(elems: &[f64]) -> Result<Mat4, JsError> {
    if elems.len() != 16 {
        return Err(WasmError::InvalidInput {
            reason: format!("matrix requires 16 elements, got {}", elems.len()),
        }
        .into());
    }
    let rows = std::array::from_fn(|i| std::array::from_fn(|j| elems[i * 4 + j]));
    Ok(Mat4(rows))
}

/// Convert a `Mat4` to a flat 16-element f64 array for JSON (row-major).
pub fn mat4_to_array(mat: &Mat4) -> Vec<f64> {
    let mut out = Vec::with_capacity(16);
    for row in &mat.0 {
        for &v in row {
            out.push(v);
        }
    }
    out
}

/// Parse a boolean operation string to the enum.
pub fn parse_boolean_op(op: &str) -> Result<BooleanOp, JsError> {
    match op {
        "fuse" | "union" => Ok(BooleanOp::Fuse),
        "cut" | "difference" => Ok(BooleanOp::Cut),
        "intersect" | "intersection" => Ok(BooleanOp::Intersect),
        _ => Err(WasmError::InvalidInput {
            reason: format!("unknown boolean op: {op}"),
        }
        .into()),
    }
}

/// Extract a required `f64` value from a JSON object.
pub fn get_f64(args: &serde_json::Value, key: &str) -> Result<f64, String> {
    args[key]
        .as_f64()
        .ok_or_else(|| format!("missing or invalid '{key}'"))
}

/// Extract a required `u32` value from a JSON object.
pub fn get_u32(args: &serde_json::Value, key: &str) -> Result<u32, String> {
    args[key]
        .as_u64()
        .map(|v| v as u32)
        .ok_or_else(|| format!("missing or invalid '{key}'"))
}

/// Extract a `usize` from a JSON value.
pub fn json_usize(val: &serde_json::Value, key: &str) -> Result<usize, JsError> {
    val[key].as_u64().map(|v| v as usize).ok_or_else(|| {
        WasmError::InvalidInput {
            reason: format!("missing or invalid '{key}'"),
        }
        .into()
    })
}

/// Extract an `f64` from a JSON value.
pub fn json_f64(val: &serde_json::Value, key: &str) -> Result<f64, JsError> {
    val[key].as_f64().ok_or_else(|| {
        WasmError::InvalidInput {
            reason: format!("missing or invalid '{key}'"),
        }
        .into()
    })
}

// ── Edge/face helpers ─────────────────────────────────────────────

/// Attempt fillet using the blend crate's unified fillet builder (primary),
/// falling back to rolling-ball then flat bevel on failure.
#[allow(deprecated)]
pub fn try_fillet(
    topo: &mut brepkit_topology::Topology,
    solid_id: brepkit_topology::solid::SolidId,
    edge_ids: &[brepkit_topology::edge::EdgeId],
    radius: f64,
) -> Result<brepkit_topology::solid::SolidId, brepkit_operations::OperationsError> {
    // Primary path: blend crate's FilletBuilder (correct corner geometry)
    brepkit_operations::blend_ops::fillet_v2(topo, solid_id, edge_ids, radius)
        .map(|r| r.solid)
        // Fallback 1: rolling-ball (legacy, has corner overlap issues)
        .or_else(|_| {
            brepkit_operations::fillet::fillet_rolling_ball(topo, solid_id, edge_ids, radius)
        })
        // Fallback 2: flat bevel (simplest)
        .or_else(|_| brepkit_operations::fillet::fillet(topo, solid_id, edge_ids, radius))
}

/// Extract a human-readable message from a `catch_unwind` panic payload.
pub fn panic_message(payload: &Box<dyn std::any::Any + Send>, operation: &str) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        format!("{operation} operation panicked: {s}")
    } else if let Some(s) = payload.downcast_ref::<String>() {
        format!("{operation} operation panicked: {s}")
    } else {
        format!("{operation} operation panicked (unknown cause)")
    }
}

/// Sample a closed periodic curve (period = TAU) into a flat `[x, y, z, ...]` buffer.
///
/// Produces `n` evenly-spaced points in `[0, TAU)` — the endpoint at `TAU` is
/// excluded because it duplicates `t = 0` on periodic curves. Callers that need
/// a closed polyline should append the first point or close the loop in JS.
///
/// Returns an empty buffer if `n == 0`.
pub fn sample_full_period_curve(n: usize, evaluate: impl Fn(f64) -> Point3) -> Vec<f64> {
    if n <= 1 {
        if n == 1 {
            let p = evaluate(0.0);
            return vec![p.x(), p.y(), p.z()];
        }
        return Vec::new();
    }
    let mut result = Vec::with_capacity(n * 3);
    for i in 0..n {
        let t = std::f64::consts::TAU * (i as f64) / (n as f64);
        let p = evaluate(t);
        result.push(p.x());
        result.push(p.y());
        result.push(p.z());
    }
    result
}

/// Create a tiny degenerate polygon face at a point, matching the vertex
/// count of the first existing profile. Used for loft start/end points.
pub fn create_apex_face(
    topo: &mut Topology,
    point: Point3,
    existing_profiles: &[brepkit_topology::face::FaceId],
) -> Result<brepkit_topology::face::FaceId, JsError> {
    // Determine target vertex count from the first profile.
    let n = if let Some(&fid) = existing_profiles.first() {
        let verts = brepkit_operations::boolean::face_polygon(topo, fid)
            .map_err(|e: brepkit_operations::OperationsError| JsError::new(&e.to_string()))?;
        verts.len().max(3)
    } else {
        3
    };

    // Create a tiny polygon at the apex point.
    let epsilon = 1e-6;
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        pts.push(Point3::new(
            point.x() + epsilon * angle.cos(),
            point.y() + epsilon * angle.sin(),
            point.z(),
        ));
    }

    let wire_id = brepkit_topology::builder::make_polygon_wire(topo, &pts, TOL)
        .map_err(|e| JsError::new(&e.to_string()))?;
    let face_id = brepkit_topology::builder::make_face_from_wire(topo, wire_id)
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(face_id)
}

// ── Mesh / tessellation helpers ───────────────────────────────────

/// Build a `TriangleMesh` from flat position/index arrays.
pub fn build_triangle_mesh(
    positions: &[f64],
    indices: &[u32],
) -> Result<tessellate::TriangleMesh, JsError> {
    if positions.len() % 3 != 0 {
        return Err(WasmError::InvalidInput {
            reason: format!(
                "positions length must be a multiple of 3, got {}",
                positions.len()
            ),
        }
        .into());
    }
    let pts: Vec<Point3> = positions
        .chunks_exact(3)
        .map(|c| Point3::new(c[0], c[1], c[2]))
        .collect();
    // Compute normals as zero vectors (mesh_boolean recomputes them)
    let normals = vec![Vec3::new(0.0, 0.0, 0.0); pts.len()];
    Ok(tessellate::TriangleMesh {
        positions: pts,
        normals,
        indices: indices.to_vec(),
    })
}

/// Convert a `TriangleMesh` to `JsMesh`.
pub fn triangle_mesh_to_js(mesh: &tessellate::TriangleMesh) -> JsMesh {
    JsMesh::from(mesh.clone())
}

// ── Classification / serialization ────────────────────────────────

/// Convert a `PointClassification` to a string.
pub fn classify_to_string(c: brepkit_operations::classify::PointClassification) -> String {
    match c {
        brepkit_operations::classify::PointClassification::Inside => "inside".into(),
        brepkit_operations::classify::PointClassification::Outside => "outside".into(),
        brepkit_operations::classify::PointClassification::OnBoundary => "boundary".into(),
    }
}

/// Serialize a `Feature` enum to JSON.
pub fn serialize_feature(
    f: &brepkit_operations::feature_recognition::Feature,
) -> serde_json::Value {
    use brepkit_operations::feature_recognition::Feature;
    match f {
        Feature::Hole { faces, diameter } => serde_json::json!({
            "type": "hole",
            "faces": faces.iter().map(|f| face_id_to_u32(*f)).collect::<Vec<_>>(),
            "diameter": diameter,
        }),
        Feature::Chamfer {
            face,
            adjacent,
            angle,
        } => serde_json::json!({
            "type": "chamfer",
            "face": face_id_to_u32(*face),
            "adjacent": [face_id_to_u32(adjacent.0), face_id_to_u32(adjacent.1)],
            "angle": angle,
        }),
        Feature::FilletLike { face, area } => serde_json::json!({
            "type": "filletLike",
            "face": face_id_to_u32(*face),
            "area": area,
        }),
        Feature::Pocket { floor, walls } => serde_json::json!({
            "type": "pocket",
            "floor": face_id_to_u32(*floor),
            "walls": walls.iter().map(|f| face_id_to_u32(*f)).collect::<Vec<_>>(),
        }),
        Feature::Pattern {
            feature_indices,
            pattern_type,
            count,
            spacing,
        } => serde_json::json!({
            "type": "pattern",
            "featureIndices": feature_indices,
            "patternType": format!("{pattern_type:?}").to_lowercase(),
            "count": count,
            "spacing": spacing,
        }),
    }
}

// ── Sketch constraint parsing ─────────────────────────────────────

/// Parse a sketch constraint from a JSON value.
pub fn parse_sketch_constraint(
    val: &serde_json::Value,
) -> Result<brepkit_operations::sketch::Constraint, JsError> {
    use brepkit_operations::sketch::Constraint;
    let ty = val["type"].as_str().unwrap_or("");
    match ty {
        "coincident" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            Ok(Constraint::Coincident(p1, p2))
        }
        "distance" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            let v = json_f64(val, "value")?;
            Ok(Constraint::Distance(p1, p2, v))
        }
        "fixX" => {
            let p = json_usize(val, "point")?;
            let v = json_f64(val, "value")?;
            Ok(Constraint::FixX(p, v))
        }
        "fixY" => {
            let p = json_usize(val, "point")?;
            let v = json_f64(val, "value")?;
            Ok(Constraint::FixY(p, v))
        }
        "vertical" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            Ok(Constraint::Vertical(p1, p2))
        }
        "horizontal" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            Ok(Constraint::Horizontal(p1, p2))
        }
        "angle" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            // Backward compat: old API was (p1, p2, value) for single-line angle.
            // New API is (p1, p2, p3, p4, value) for angle between two lines.
            // When p3/p4 are absent, default to p1/p2 (zero angle between same line).
            let p3 = val
                .get("p3")
                .and_then(serde_json::Value::as_u64)
                .map_or(p1, |v| v as usize);
            let p4 = val
                .get("p4")
                .and_then(serde_json::Value::as_u64)
                .map_or(p2, |v| v as usize);
            let v = json_f64(val, "value")?;
            Ok(Constraint::Angle(p1, p2, p3, p4, v))
        }
        "perpendicular" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            let p3 = json_usize(val, "p3")?;
            let p4 = json_usize(val, "p4")?;
            Ok(Constraint::Perpendicular(p1, p2, p3, p4))
        }
        "parallel" => {
            let p1 = json_usize(val, "p1")?;
            let p2 = json_usize(val, "p2")?;
            let p3 = json_usize(val, "p3")?;
            let p4 = json_usize(val, "p4")?;
            Ok(Constraint::Parallel(p1, p2, p3, p4))
        }
        _ => Err(WasmError::InvalidInput {
            reason: format!("unknown constraint type: {ty}"),
        }
        .into()),
    }
}

// ── 2D polygon helpers ────────────────────────────────────────────

/// Parse flat `[x,y, ...]` coordinates into `Vec<Point2>`.
pub fn parse_polygon_2d(coords: &[f64]) -> Result<Vec<Point2>, JsError> {
    if coords.len() % 2 != 0 || coords.len() < 6 {
        return Err(WasmError::InvalidInput {
            reason: "polygon needs at least 3 points (6 coordinates)".into(),
        }
        .into());
    }
    Ok(coords
        .chunks_exact(2)
        .map(|c| Point2::new(c[0], c[1]))
        .collect())
}

/// Check if two 2D polygons overlap using vertex containment + edge crossing.
pub fn polygons_overlap_2d(a: &[Point2], b: &[Point2]) -> bool {
    use brepkit_math::predicates::point_in_polygon;

    // Check if any vertex of A is inside B or vice versa.
    for p in a {
        if point_in_polygon(*p, b) {
            return true;
        }
    }
    for p in b {
        if point_in_polygon(*p, a) {
            return true;
        }
    }

    // Check edge crossings.
    for i in 0..a.len() {
        let a1 = a[i];
        let a2 = a[(i + 1) % a.len()];
        for j in 0..b.len() {
            let b1 = b[j];
            let b2 = b[(j + 1) % b.len()];
            if segments_intersect_2d(a1, a2, b1, b2) {
                return true;
            }
        }
    }
    false
}

/// Test if two 2D line segments intersect (proper crossing).
pub fn segments_intersect_2d(a1: Point2, a2: Point2, b1: Point2, b2: Point2) -> bool {
    use brepkit_math::polygon2d::cross_2d;
    let d1 = cross_2d(b1, b2, a1);
    let d2 = cross_2d(b1, b2, a2);
    let d3 = cross_2d(a1, a2, b1);
    let d4 = cross_2d(a1, a2, b2);

    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}
