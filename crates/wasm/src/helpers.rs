//! Shared free functions and constants used across WASM binding modules.

#![allow(
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::NurbsCurve;
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

/// Filter edges to only those shared by two planar faces in a solid.
pub fn filter_planar_edges(
    topo: &brepkit_topology::Topology,
    solid_id: brepkit_topology::solid::SolidId,
    edge_ids: &[brepkit_topology::edge::EdgeId],
) -> Result<Vec<brepkit_topology::edge::EdgeId>, JsError> {
    use std::collections::HashMap;
    let solid_data = topo.solid(solid_id)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut edge_faces: HashMap<usize, Vec<brepkit_topology::face::FaceId>> = HashMap::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            edge_faces.entry(oe.edge().index()).or_default().push(fid);
        }
    }

    let mut result = Vec::new();
    for &eid in edge_ids {
        if let Some(adj_faces) = edge_faces.get(&eid.index()) {
            let all_planar = adj_faces.iter().all(|&fid| {
                topo.face(fid)
                    .map(|f| {
                        matches!(
                            f.surface(),
                            brepkit_topology::face::FaceSurface::Plane { .. }
                        )
                    })
                    .unwrap_or(false)
            });
            if all_planar {
                result.push(eid);
            }
        }
    }
    Ok(result)
}

/// Attempt fillet with rolling-ball, falling back to flat bevel on failure.
#[allow(deprecated)]
pub fn try_fillet(
    topo: &mut brepkit_topology::Topology,
    solid_id: brepkit_topology::solid::SolidId,
    edge_ids: &[brepkit_topology::edge::EdgeId],
    radius: f64,
) -> Result<brepkit_topology::solid::SolidId, brepkit_operations::OperationsError> {
    brepkit_operations::fillet::fillet_rolling_ball(topo, solid_id, edge_ids, radius)
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

// ── NURBS detection ───────────────────────────────────────────────

/// Detect if a NURBS curve represents an analytic curve type.
///
/// Checks if the curve is a circle or ellipse by sampling points
/// and verifying they are coplanar and equidistant from a center.
pub fn detect_nurbs_curve_type(nc: &NurbsCurve) -> &'static str {
    // A rational degree-2 NURBS with specific weight patterns can represent
    // conic sections. Check if all sampled points lie on a circle.
    if nc.degree() < 2 || !nc.is_rational() {
        return "BSPLINE_CURVE";
    }

    let (u_min, u_max) = nc.domain();
    let n_samples = 16;

    // Check if the curve is closed (start ≈ end) to avoid sampling the
    // duplicate endpoint, which would bias the center calculation.
    let start_pt = nc.evaluate(u_min);
    let end_pt = nc.evaluate(u_max);
    let is_closed = (start_pt - end_pt).length() < 1e-6;

    // Sample points along the curve. For closed curves, exclude the
    // last point (t=u_max) since it duplicates the first.
    let mut points = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = if is_closed {
            u_min + (u_max - u_min) * (i as f64) / (n_samples as f64)
        } else {
            u_min + (u_max - u_min) * (i as f64) / ((n_samples - 1) as f64)
        };
        points.push(nc.evaluate(t));
    }

    // Compute center as average of all sampled points
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut cz = 0.0_f64;
    for p in &points {
        cx += p.x();
        cy += p.y();
        cz += p.z();
    }
    let n = points.len() as f64;
    let center = Point3::new(cx / n, cy / n, cz / n);

    // Check if all points are equidistant from center (circle test)
    let distances: Vec<f64> = points.iter().map(|p| (*p - center).length()).collect();
    let avg_dist = distances.iter().sum::<f64>() / n;

    if avg_dist < 1e-10 {
        return "BSPLINE_CURVE";
    }

    let tol = avg_dist * 1e-4; // 0.01% relative tolerance
    let is_circle = distances.iter().all(|d| (d - avg_dist).abs() < tol);

    if is_circle {
        // Check coplanarity — all points should lie in a plane through center
        let v0 = points[0] - center;
        let v1 = points[n_samples / 4] - center;
        let normal = v0.cross(v1);
        let normal_len = normal.length();
        if normal_len < 1e-10 {
            return "BSPLINE_CURVE";
        }
        let normal = Vec3::new(
            normal.x() / normal_len,
            normal.y() / normal_len,
            normal.z() / normal_len,
        );

        let coplanar = points
            .iter()
            .all(|p| ((*p - center).dot(normal)).abs() < tol);

        if coplanar {
            return "CIRCLE";
        }
    }

    // TODO: Could also detect ELLIPSE (non-uniform distances but elliptic pattern)
    "BSPLINE_CURVE"
}

/// Detect if a NURBS surface represents an analytic surface type.
///
/// Checks if the surface is a sphere, cylinder, cone, or torus by
/// sampling a grid of points and analyzing their geometric properties.
pub fn detect_nurbs_surface_type(ns: &brepkit_math::nurbs::surface::NurbsSurface) -> &'static str {
    let (u_min, u_max) = ns.domain_u();
    let (v_min, v_max) = ns.domain_v();
    let n = 8; // 8×8 grid = 64 sample points

    // Sample points on the surface
    let mut points = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let u = u_min + (u_max - u_min) * (i as f64) / ((n - 1) as f64);
            let v = v_min + (v_max - v_min) * (j as f64) / ((n - 1) as f64);
            points.push(ns.evaluate(u, v));
        }
    }

    // Compute center as average
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut cz = 0.0_f64;
    for p in &points {
        cx += p.x();
        cy += p.y();
        cz += p.z();
    }
    let np = points.len() as f64;
    let center = Point3::new(cx / np, cy / np, cz / np);

    // Check if all points equidistant from center (sphere test)
    let distances: Vec<f64> = points.iter().map(|p| (*p - center).length()).collect();
    let avg_dist = distances.iter().sum::<f64>() / np;

    if avg_dist < 1e-10 {
        return "bspline";
    }

    let tol = avg_dist * 1e-3; // 0.1% relative tolerance
    let is_sphere = distances.iter().all(|d| (d - avg_dist).abs() < tol);

    if is_sphere {
        return "sphere";
    }

    // Cylinder test: points should be equidistant from an axis line.
    // Try to find the axis by PCA (direction of maximum variance).
    // For a cylinder, points cluster around a line; cross-section is a circle.
    if let Some(axis_dir) = estimate_cylinder_axis(&points, center) {
        let projected_distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let v = *p - center;
                let along_axis = v.dot(axis_dir);
                let radial = Vec3::new(
                    v.x() - axis_dir.x() * along_axis,
                    v.y() - axis_dir.y() * along_axis,
                    v.z() - axis_dir.z() * along_axis,
                );
                radial.length()
            })
            .collect();

        let avg_r = projected_distances.iter().sum::<f64>() / np;
        if avg_r > 1e-10 {
            let r_tol = avg_r * 1e-3;
            let is_cylinder = projected_distances
                .iter()
                .all(|d| (d - avg_r).abs() < r_tol);
            if is_cylinder {
                return "cylinder";
            }
        }
    }

    "bspline"
}

/// Estimate the cylinder axis direction from a set of surface sample points
/// using a simple PCA-like approach (direction of maximum variance).
fn estimate_cylinder_axis(points: &[Point3], center: Point3) -> Option<Vec3> {
    // Build covariance matrix
    let mut cxx = 0.0_f64;
    let mut cxy = 0.0_f64;
    let mut cxz = 0.0_f64;
    let mut cyy = 0.0_f64;
    let mut cyz = 0.0_f64;
    let mut czz = 0.0_f64;

    for p in points {
        let dx = p.x() - center.x();
        let dy = p.y() - center.y();
        let dz = p.z() - center.z();
        cxx += dx * dx;
        cxy += dx * dy;
        cxz += dx * dz;
        cyy += dy * dy;
        cyz += dy * dz;
        czz += dz * dz;
    }

    // Power iteration to find the principal eigenvector
    let mut v = Vec3::new(1.0, 0.0, 0.0);
    for _ in 0..20 {
        let new_v = Vec3::new(
            v.x().mul_add(cxx, v.y().mul_add(cxy, v.z() * cxz)),
            v.x().mul_add(cxy, v.y().mul_add(cyy, v.z() * cyz)),
            v.x().mul_add(cxz, v.y().mul_add(cyz, v.z() * czz)),
        );
        let len = new_v.length();
        if len < 1e-15 {
            return None;
        }
        v = Vec3::new(new_v.x() / len, new_v.y() / len, new_v.z() / len);
    }
    Some(v)
}

/// Project a 3D point onto a NURBS surface to get (u,v) parameters.
///
/// Uses a simple grid search + Newton refinement.
pub fn project_to_uv(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    point: Point3,
) -> Point2 {
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();
    let n = 10;
    let mut best_u = u_min;
    let mut best_v = v_min;
    let mut best_dist = f64::MAX;
    for i in 0..=n {
        for j in 0..=n {
            let u = u_min + (u_max - u_min) * (i as f64) / (n as f64);
            let v = v_min + (v_max - v_min) * (j as f64) / (n as f64);
            let pt = surface.evaluate(u, v);
            let dx = pt.x() - point.x();
            let dy = pt.y() - point.y();
            let dz = pt.z() - point.z();
            let dist = dx * dx + dy * dy + dz * dz;
            if dist < best_dist {
                best_dist = dist;
                best_u = u;
                best_v = v;
            }
        }
    }
    Point2::new(best_u, best_v)
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
    let d1 = cross_2d(b1, b2, a1);
    let d2 = cross_2d(b1, b2, a2);
    let d3 = cross_2d(a1, a2, b1);
    let d4 = cross_2d(a1, a2, b2);

    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

pub fn cross_2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x())
}

/// Sutherland-Hodgman polygon clipping algorithm.
pub fn sutherland_hodgman_clip(subject: &[Point2], clip: &[Point2]) -> Vec<Point2> {
    let mut output: Vec<Point2> = subject.to_vec();

    for i in 0..clip.len() {
        if output.is_empty() {
            return output;
        }
        let edge_start = clip[i];
        let edge_end = clip[(i + 1) % clip.len()];
        let input = output;
        output = Vec::new();

        for j in 0..input.len() {
            let current = input[j];
            let previous = input[(j + input.len() - 1) % input.len()];

            let curr_inside = cross_2d(edge_start, edge_end, current) >= 0.0;
            let prev_inside = cross_2d(edge_start, edge_end, previous) >= 0.0;

            if curr_inside {
                if !prev_inside {
                    if let Some(p) = line_intersect_2d(previous, current, edge_start, edge_end) {
                        output.push(p);
                    }
                }
                output.push(current);
            } else if prev_inside {
                if let Some(p) = line_intersect_2d(previous, current, edge_start, edge_end) {
                    output.push(p);
                }
            }
        }
    }

    output
}

/// Find the intersection point of two 2D line segments (as infinite lines).
pub fn line_intersect_2d(a1: Point2, a2: Point2, b1: Point2, b2: Point2) -> Option<Point2> {
    let dx_a = a2.x() - a1.x();
    let dy_a = a2.y() - a1.y();
    let dx_b = b2.x() - b1.x();
    let dy_b = b2.y() - b1.y();
    let denom = dx_a * dy_b - dy_a * dx_b;
    if denom.abs() < 1e-15 {
        return None;
    }
    let t = ((b1.x() - a1.x()) * dy_b - (b1.y() - a1.y()) * dx_b) / denom;
    Some(Point2::new(a1.x() + t * dx_a, a1.y() + t * dy_a))
}

/// Find common (collinear, overlapping) edges between two polygons.
pub fn find_common_segments(a: &[Point2], b: &[Point2], tolerance: f64) -> Vec<(Point2, Point2)> {
    let mut results = Vec::new();
    let tol_sq = tolerance * tolerance;

    for i in 0..a.len() {
        let a1 = a[i];
        let a2 = a[(i + 1) % a.len()];
        for j in 0..b.len() {
            let b1 = b[j];
            let b2 = b[(j + 1) % b.len()];

            // Check if edge A and edge B are collinear and overlapping.
            // Both endpoints of B must be close to line through A, or vice versa.
            let dist_b1 = point_to_line_dist_sq_2d(b1, a1, a2);
            let dist_b2 = point_to_line_dist_sq_2d(b2, a1, a2);

            if dist_b1 < tol_sq && dist_b2 < tol_sq {
                // Edges are collinear. Check for overlap by projecting onto A's direction.
                let dx = a2.x() - a1.x();
                let dy = a2.y() - a1.y();
                let len_sq = dx * dx + dy * dy;
                if len_sq < tol_sq {
                    continue;
                }
                let t1 = ((b1.x() - a1.x()) * dx + (b1.y() - a1.y()) * dy) / len_sq;
                let t2 = ((b2.x() - a1.x()) * dx + (b2.y() - a1.y()) * dy) / len_sq;
                let t_min = t1.min(t2).max(0.0);
                let t_max = t1.max(t2).min(1.0);
                if t_max - t_min > tolerance / len_sq.sqrt() {
                    results.push((
                        Point2::new(a1.x() + t_min * dx, a1.y() + t_min * dy),
                        Point2::new(a1.x() + t_max * dx, a1.y() + t_max * dy),
                    ));
                }
            }
        }
    }
    results
}

pub fn point_to_line_dist_sq_2d(p: Point2, a: Point2, b: Point2) -> f64 {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-30 {
        let ex = p.x() - a.x();
        let ey = p.y() - a.y();
        return ex * ex + ey * ey;
    }
    let cross = (p.x() - a.x()) * dy - (p.y() - a.y()) * dx;
    (cross * cross) / len_sq
}

/// Round all corners of a 2D polygon with arc approximations.
pub fn fillet_polygon_2d(polygon: &[Point2], radius: f64) -> Vec<Point2> {
    let n = polygon.len();
    if n < 3 {
        return polygon.to_vec();
    }

    let arc_segments = 8; // Number of segments per fillet arc
    let mut result = Vec::with_capacity(n * (arc_segments + 1));

    for i in 0..n {
        let prev = polygon[(i + n - 1) % n];
        let curr = polygon[i];
        let next = polygon[(i + 1) % n];

        let d_prev = ((prev.x() - curr.x()).powi(2) + (prev.y() - curr.y()).powi(2)).sqrt();
        let d_next = ((next.x() - curr.x()).powi(2) + (next.y() - curr.y()).powi(2)).sqrt();

        let max_r = (d_prev.min(d_next) / 2.0).min(radius);

        if max_r < 1e-10 {
            result.push(curr);
            continue;
        }

        // Direction vectors from corner to adjacent vertices
        let dir_prev_x = (prev.x() - curr.x()) / d_prev;
        let dir_prev_y = (prev.y() - curr.y()) / d_prev;
        let dir_next_x = (next.x() - curr.x()) / d_next;
        let dir_next_y = (next.y() - curr.y()) / d_next;

        // Tangent points on edges
        let t1 = Point2::new(curr.x() + dir_prev_x * max_r, curr.y() + dir_prev_y * max_r);
        let t2 = Point2::new(curr.x() + dir_next_x * max_r, curr.y() + dir_next_y * max_r);

        // Generate arc points from t1 to t2
        for k in 0..=arc_segments {
            let t = k as f64 / arc_segments as f64;
            let x = t2.x().mul_add(t, t1.x() * (1.0 - t));
            let y = t2.y().mul_add(t, t1.y() * (1.0 - t));

            // Push point toward the arc center for a circular approximation
            let mid_x = f64::midpoint(t1.x(), t2.x());
            let mid_y = f64::midpoint(t1.y(), t2.y());
            let to_corner_x = curr.x() - mid_x;
            let to_corner_y = curr.y() - mid_y;
            let corner_dist = (to_corner_x * to_corner_x + to_corner_y * to_corner_y).sqrt();

            if corner_dist > 1e-10 {
                // Compute the bulge: how much to push along the corner bisector
                let chord_half =
                    ((t2.x() - t1.x()).powi(2) + (t2.y() - t1.y()).powi(2)).sqrt() / 2.0;
                let sagitta = if max_r > chord_half {
                    max_r - (max_r * max_r - chord_half * chord_half).sqrt()
                } else {
                    0.0
                };

                // Blend factor: maximum at midpoint (t=0.5), zero at endpoints
                let blend = 4.0 * t * (1.0 - t); // parabolic blend
                let push = sagitta * blend;

                let nx = to_corner_x / corner_dist;
                let ny = to_corner_y / corner_dist;
                result.push(Point2::new(x + nx * push, y + ny * push));
            } else {
                result.push(Point2::new(x, y));
            }
        }
    }

    result
}

/// Cut all corners of a 2D polygon with flat bevels.
pub fn chamfer_polygon_2d(polygon: &[Point2], distance: f64) -> Vec<Point2> {
    let n = polygon.len();
    if n < 3 {
        return polygon.to_vec();
    }

    let mut result = Vec::with_capacity(n * 2);

    for i in 0..n {
        let prev = polygon[(i + n - 1) % n];
        let curr = polygon[i];
        let next = polygon[(i + 1) % n];

        let d_prev = ((prev.x() - curr.x()).powi(2) + (prev.y() - curr.y()).powi(2)).sqrt();
        let d_next = ((next.x() - curr.x()).powi(2) + (next.y() - curr.y()).powi(2)).sqrt();

        let d = (d_prev.min(d_next) / 2.0).min(distance);

        if d < 1e-10 {
            result.push(curr);
            continue;
        }

        // Two chamfer points: one on previous edge, one on next edge
        result.push(Point2::new(
            curr.x() + (prev.x() - curr.x()) / d_prev * d,
            curr.y() + (prev.y() - curr.y()) / d_prev * d,
        ));
        result.push(Point2::new(
            curr.x() + (next.x() - curr.x()) / d_next * d,
            curr.y() + (next.y() - curr.y()) / d_next * d,
        ));
    }

    result
}
