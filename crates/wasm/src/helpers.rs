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
    if !coords.len().is_multiple_of(3) {
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

/// Attempt a fillet, preferring the rolling-ball engine and validating output.
///
/// The rolling-ball engine produces watertight single-edge fillets, does the
/// per-corner setback trimming + spherical patches multi-edge inputs need
/// (the walking `FilletBuilder` over-removes corner material — all 12 box edges
/// drop to ~470 instead of ~975), and now solves true contacts against curved
/// neighbours, so a fillet whose neighbour is a prior fillet's NURBS blend face
/// is watertight too (#834). It runs first; the `FilletBuilder` and a flat
/// bevel are fallbacks.
///
/// Every candidate is validated as a manifold solid before being accepted, so
/// a malformed result is rejected in favour of the next engine and, if none
/// qualifies, the solid is returned unchanged. This guard lets
/// `filter_filletable_edges` be permissive about curved neighbours without ever
/// returning a degenerate solid.
#[allow(deprecated)]
pub fn try_fillet(
    topo: &mut brepkit_topology::Topology,
    solid_id: brepkit_topology::solid::SolidId,
    edge_ids: &[brepkit_topology::edge::EdgeId],
    radius: f64,
) -> Result<brepkit_topology::solid::SolidId, brepkit_operations::OperationsError> {
    // Drop tangent / degenerate edges (e.g. a fillet face's G1 contact line with
    // its planar neighbour). If none qualify, the solid is returned unchanged.
    let edges = brepkit_operations::query::filter_filletable_edges(topo, solid_id, edge_ids)?;
    if edges.is_empty() {
        return Ok(solid_id);
    }
    let edges = edges.as_slice();

    // A candidate is acceptable only if its outer shell is a CLOSED 2-manifold
    // (every edge used by exactly two faces — no free/boundary edges). The
    // weaker `validate_shell_manifold` (only rejects edges shared by 3+ faces)
    // silently accepted open shells: e.g. a primitive cylinder's rim fillet,
    // where the walking engine cannot trim the disc cap to the toroidal blend's
    // contact circle and leaves 8 free edges at the cap/torus seams. An open
    // shell measures a plausible-but-wrong volume, so it must be rejected so the
    // next engine (or the unchanged input) is used instead.
    let is_valid =
        |topo: &brepkit_topology::Topology, s: brepkit_topology::solid::SolidId| -> bool {
            topo.solid(s)
                .and_then(|sd| topo.shell(sd.outer_shell()))
                .map(|sh| brepkit_topology::validation::validate_shell_closed(sh, topo).is_ok())
                .unwrap_or(false)
        };

    // Try engines in preference order; accept the first valid result.
    if let Ok(s) = brepkit_operations::fillet::fillet_rolling_ball(topo, solid_id, edges, radius)
        && is_valid(topo, s)
    {
        return Ok(s);
    }
    if let Ok(r) = brepkit_operations::blend_ops::fillet_v2(topo, solid_id, edges, radius)
        && is_valid(topo, r.solid)
    {
        return Ok(r.solid);
    }
    if let Ok(s) = brepkit_operations::fillet::fillet(topo, solid_id, edges, radius)
        && is_valid(topo, s)
    {
        return Ok(s);
    }

    // No engine produced a valid solid — leave the input unchanged.
    Ok(solid_id)
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
    if !positions.len().is_multiple_of(3) {
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
    if !coords.len().is_multiple_of(2) || coords.len() < 6 {
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

#[cfg(test)]
mod fillet_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::HashSet;

    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::solid::SolidId;

    use super::try_fillet;

    fn solid_edge_ids(topo: &Topology, solid_id: SolidId) -> Vec<EdgeId> {
        let solid = topo.solid(solid_id).expect("solid");
        let shell = topo.shell(solid.outer_shell()).expect("shell");
        let mut seen = HashSet::new();
        let mut edges = Vec::new();
        for &fid in shell.faces() {
            let face = topo.face(fid).expect("face");
            let wire = topo.wire(face.outer_wire()).expect("wire");
            for oe in wire.edges() {
                if seen.insert(oe.edge().index()) {
                    edges.push(oe.edge());
                }
            }
        }
        edges
    }

    // The wasm `fillet` binding (and its batch sibling) route through
    // `try_fillet`. Filleting all 12 box edges must remove only the rounded
    // slivers (volume ≈ 975.6 for a 10³ box at r=1), not excise whole corner
    // octants. This guards the consumer path against regressing to the
    // over-removing walking engine.
    #[test]
    fn try_fillet_all_box_edges_no_corner_over_removal() {
        let mut topo = Topology::new();
        let cube = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let edges = solid_edge_ids(&topo, cube);
        assert_eq!(edges.len(), 12, "box should have 12 edges");

        let result = try_fillet(&mut topo, cube, &edges, 1.0).expect("all-edges fillet");
        let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.01).unwrap();
        assert!(
            vol > 970.0 && vol < 1000.0,
            "filleted box volume should be ≈975.6, got {vol}"
        );
    }

    // gh #967: filleting a plain cylinder's circular rim used to silently
    // remove ~37% of the volume — the deprecated rolling-ball engine collapses
    // a closed circular edge's blend strip and caps to zero-area faces, and the
    // walking engine leaves an open shell. Both are now rejected, so `try_fillet`
    // falls back to returning the input unchanged: a VALID solid with the raw
    // volume, never the silently-corrupt one. (Genuine rim rounding — occt's
    // ~0.1% reduction — is a separate blend-engine feature.)
    #[test]
    fn try_fillet_cylinder_rim_does_not_corrupt_volume() {
        let mut topo = Topology::new();
        let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, 10.0, 20.0).unwrap();
        let raw = brepkit_operations::measure::solid_volume(&topo, cyl, 0.01).unwrap();
        let edges = solid_edge_ids(&topo, cyl);

        let result = try_fillet(&mut topo, cyl, &edges, 0.5).expect("rim fillet must not error");
        let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.01).unwrap();

        // Must NOT be the corrupt ~3978 (−37%); a no-op keeps the raw volume.
        assert!(
            (vol - raw).abs() / raw < 0.02,
            "cylinder rim fillet must keep raw volume {raw:.1} (no-op), got {vol:.1}"
        );
        assert!(
            brepkit_operations::validate::validate_solid(&topo, result)
                .unwrap()
                .is_valid(),
            "fillet result must be a valid solid"
        );
    }

    #[test]
    fn try_fillet_second_pass_does_not_break_solid() {
        use brepkit_topology::face::FaceSurface;

        // #813: a second fillet whose target edge borders the first fillet's
        // NURBS blend face must not produce a self-intersecting solid — the
        // volume grew past the base to 1000.30 before the fix; such edges are
        // now skipped. Checked over *every* result edge so the guard doesn't
        // rely on a particular edge ordering.
        let mut topo = Topology::new();
        let cube = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let edges = solid_edge_ids(&topo, cube);
        let first = try_fillet(&mut topo, cube, &[edges[0], edges[1]], 1.0).expect("first fillet");
        let v1 = brepkit_operations::measure::solid_volume(&topo, first, 0.05).unwrap();

        // The scenario under test only exists if the first fillet produced NURBS
        // blend faces for the later edges to border.
        let sd = topo.solid(first).expect("solid");
        let sh = topo.shell(sd.outer_shell()).expect("shell");
        let has_blend = sh.faces().iter().any(|&fid| {
            topo.face(fid)
                .is_ok_and(|f| matches!(f.surface(), FaceSurface::Nurbs(_)))
        });
        assert!(has_blend, "first fillet should create NURBS blend faces");

        // Filleting any single result edge must stay a manifold solid and must
        // not self-intersect/inflate past the original box volume (the #813 bug
        // grew it to 1000.30). A blend-adjacent *concave* end-cap edge validly
        // *fills* (volume rises toward — but never beyond — the box), so the
        // guard is the box volume, not the pre-fillet volume.
        let _ = v1;
        let r_edges = solid_edge_ids(&topo, first);
        for &e in &r_edges {
            let mut t = topo.clone();
            let s = try_fillet(&mut t, first, &[e], 0.5).expect("second fillet");
            let v2 = brepkit_operations::measure::solid_volume(&t, s, 0.05).unwrap();
            assert!(
                v2 <= 1000.0 + 0.1,
                "second fillet on edge {} inflated past the box: second={v2:.2}",
                e.index()
            );
            let ssd = t.solid(s).expect("result solid");
            let ssh = t.shell(ssd.outer_shell()).expect("shell");
            brepkit_topology::validation::validate_shell_manifold(ssh, &t)
                .expect("second fillet result must remain a manifold solid");
        }
    }

    #[test]
    fn try_fillet_nurbs_blend_neighbor_is_watertight() {
        use std::collections::HashMap;

        use brepkit_topology::face::FaceSurface;
        use brepkit_topology::validation::{validate_shell_closed, validate_shell_manifold};

        // #834 via the consumer path: a single fillet creates a NURBS blend
        // face; `try_fillet` on a non-tangent edge bordering it must round it
        // into a valid watertight manifold (rather than skip it as before).
        let mut topo = Topology::new();
        let cube = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let edges = solid_edge_ids(&topo, cube);
        let first = try_fillet(&mut topo, cube, &[edges[0]], 1.0).expect("first fillet");
        {
            let sh = topo
                .shell(topo.solid(first).unwrap().outer_shell())
                .unwrap();
            validate_shell_closed(sh, &topo).expect("first fillet should be watertight");
        }

        let nurbs: HashSet<usize> = {
            let sh = topo
                .shell(topo.solid(first).unwrap().outer_shell())
                .unwrap();
            sh.faces()
                .iter()
                .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_)))
                .map(|f| f.index())
                .collect()
        };
        assert!(
            !nurbs.is_empty(),
            "first fillet must create a NURBS blend face"
        );

        let mut ef: HashMap<usize, HashSet<usize>> = HashMap::new();
        {
            let sh = topo
                .shell(topo.solid(first).unwrap().outer_shell())
                .unwrap();
            for &fid in sh.faces() {
                for oe in topo
                    .wire(topo.face(fid).unwrap().outer_wire())
                    .unwrap()
                    .edges()
                {
                    ef.entry(oe.edge().index()).or_default().insert(fid.index());
                }
            }
        }

        let r_edges = solid_edge_ids(&topo, first);
        let filletable: HashSet<usize> =
            brepkit_operations::query::filter_filletable_edges(&topo, first, &r_edges)
                .unwrap()
                .iter()
                .map(|e| e.index())
                .collect();
        let target = r_edges
            .iter()
            .copied()
            .find(|e| {
                filletable.contains(&e.index())
                    && ef
                        .get(&e.index())
                        .is_some_and(|fs| fs.iter().any(|f| nurbs.contains(f)))
            })
            .expect("a filletable edge bordering the NURBS blend face");

        let result = try_fillet(&mut topo, first, &[target], 0.5).expect("second fillet");
        assert_ne!(
            result, first,
            "the NURBS-blend-adjacent edge should be filleted, not skipped"
        );
        let sh = topo
            .shell(topo.solid(result).unwrap().outer_shell())
            .unwrap();
        validate_shell_manifold(sh, &topo).expect("second fillet must be manifold");
        validate_shell_closed(sh, &topo)
            .expect("second fillet on a NURBS-blend-adjacent edge must be watertight");
    }
}
