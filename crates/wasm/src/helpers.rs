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

/// Attempt a fillet, preferring the corrected walking `FilletBuilder` engine
/// and validating output.
///
/// The production WASM dispatch runs [`brepkit_operations::blend_ops::fillet_v2`]
/// (the corrected `FilletBuilder`) first. It produces watertight single-edge
/// fillets, per-contour reconstruction through the canonical boundary
/// registry, ordered-fan junction corners (no over-removal at box corners),
/// and closed periodic bands for circular rims — so a fillet whose neighbour
/// is a prior fillet's NURBS blend face is watertight too (#834). The
/// deprecated flat-bevel [`brepkit_operations::fillet::fillet`] remains only
/// as a last-resort fallback for inputs the walking engine cannot represent.
///
/// The deprecated rolling-ball engine
/// ([`brepkit_operations::fillet::fillet_rolling_ball`]) is NOT part of the
/// production dispatch chain: it stays callable as an explicit reference /
/// diagnostic implementation but is never the WASM default.
///
/// Every candidate is validated as a changed, closed (watertight), outward-
/// oriented solid before being accepted, so a malformed result is rejected in
/// favour of the next engine and, if none qualifies, an error is returned.
/// This guard lets `filter_filletable_edges` be permissive about curved
/// neighbours without ever reporting an unchanged input or returning a
/// degenerate solid.
#[allow(deprecated)]
pub fn try_fillet(
    topo: &mut brepkit_topology::Topology,
    solid_id: brepkit_topology::solid::SolidId,
    edge_ids: &[brepkit_topology::edge::EdgeId],
    radius: f64,
) -> Result<brepkit_topology::solid::SolidId, brepkit_operations::OperationsError> {
    // Filtering identifies non-candidate topology seams and tangent/degenerate
    // edges (for example a prior fillet's G1 contact line). Seams are harmless
    // carrier topology; every actual edge candidate remains atomic.
    let filtered = brepkit_operations::query::filter_filletable_edges(topo, solid_id, edge_ids)?;
    if filtered.is_empty() {
        return Err(brepkit_operations::OperationsError::InvalidInput {
            reason: "none of the requested edges are filletable".to_owned(),
        });
    }
    if filtered.len() != edge_ids.len() {
        let adjacency = topo.build_adjacency(solid_id)?;
        let rejected_are_seams =
            edge_ids
                .iter()
                .filter(|edge| !filtered.contains(edge))
                .all(|&edge| {
                    let faces = adjacency.faces_for_edge(edge);
                    faces
                        .first()
                        .is_some_and(|first| faces.iter().all(|face| face == first))
                });
        if !rejected_are_seams {
            return Err(brepkit_operations::OperationsError::InvalidInput {
                reason: "some requested edges are not filletable".to_owned(),
            });
        }
    }
    let edges = filtered.as_slice();

    let orientation_deflection = gate_deflection(topo, solid_id, radius)?;
    let is_valid = |topo: &brepkit_topology::Topology, s: brepkit_topology::solid::SolidId| {
        blend_result_holds(topo, s, orientation_deflection)
    };

    // Try engines in preference order; accept the first changed, valid result.
    // The corrected `FilletBuilder` is the production engine; the deprecated
    // flat bevel is the only fallback. `fillet_rolling_ball` is deliberately
    // absent — it remains an explicit reference/diagnostic API only.
    if let Ok(r) = brepkit_operations::blend_ops::fillet_v2(topo, solid_id, edges, radius)
        && r.solid != solid_id
        && !r.is_partial
        && r.failed.is_empty()
        && r.succeeded.len() == edges.len()
        && is_valid(topo, r.solid)
    {
        return Ok(r.solid);
    }
    if let Ok(s) = brepkit_operations::fillet::fillet(topo, solid_id, edges, radius)
        && s != solid_id
        && is_valid(topo, s)
    {
        return Ok(s);
    }

    Err(brepkit_operations::OperationsError::InvalidInput {
        reason: "no fillet engine produced a changed, closed, outward-oriented result".to_owned(),
    })
}

/// Fillet edges each along its radius law, as [`try_fillet`] fillets them at
/// one radius: the walking engine's band is kept when it is a changed,
/// closed, outward-oriented solid, and the planar rebuild, which closes each
/// band's ends with straight chords, is the fallback.
///
/// # Errors
///
/// Returns an error if neither engine fillets the edges.
pub fn try_fillet_variable(
    topo: &mut brepkit_topology::Topology,
    solid_id: brepkit_topology::solid::SolidId,
    edge_laws: &[(
        brepkit_topology::edge::EdgeId,
        brepkit_operations::fillet::FilletRadiusLaw,
    )],
) -> Result<brepkit_topology::solid::SolidId, brepkit_operations::OperationsError> {
    let widest = edge_laws
        .iter()
        .flat_map(|(_, law)| [0.0, 0.5, 1.0].map(|t| law.evaluate(t)))
        .fold(0.0_f64, f64::max);
    let orientation_deflection = gate_deflection(topo, solid_id, widest)?;
    let laws = edge_laws
        .iter()
        .map(|(edge, law)| (*edge, law.into()))
        .collect();
    if let Ok(r) = brepkit_operations::blend_ops::fillet_v2_variable(topo, solid_id, laws)
        && r.solid != solid_id
        && !r.is_partial
        && r.failed.is_empty()
        && r.succeeded.len() == edge_laws.len()
        && blend_result_holds(topo, r.solid, orientation_deflection)
    {
        return Ok(r.solid);
    }
    let s = brepkit_operations::fillet::fillet_variable(topo, solid_id, edge_laws)?;
    if s != solid_id && blend_result_holds(topo, s, orientation_deflection) {
        return Ok(s);
    }
    Err(brepkit_operations::OperationsError::InvalidInput {
        reason: "no fillet engine produced a changed, closed, outward-oriented result".to_owned(),
    })
}

/// Chamfer edges at an equal `distance` on both faces, as [`try_fillet`]
/// fillets them: the walking engine's chamfer is kept when it is a changed,
/// closed, outward-oriented solid, and the planar chamfer, which rebuilds
/// every face from its polygon and so cannot carry a curved one, is the
/// fallback.
///
/// # Errors
///
/// Returns an error if neither engine chamfers the edges.
pub fn try_chamfer(
    topo: &mut brepkit_topology::Topology,
    solid_id: brepkit_topology::solid::SolidId,
    edge_ids: &[brepkit_topology::edge::EdgeId],
    distance: f64,
) -> Result<brepkit_topology::solid::SolidId, brepkit_operations::OperationsError> {
    let on_solid: std::collections::HashSet<_> =
        brepkit_topology::explorer::solid_edges(topo, solid_id)?
            .into_iter()
            .collect();
    if edge_ids.iter().any(|e| !on_solid.contains(e)) {
        return Err(brepkit_operations::OperationsError::InvalidInput {
            reason: "some requested edges are not on the solid".to_owned(),
        });
    }
    // Neither engine chamfers a cavity's edges: the walking one refuses
    // them, and the planar one would chamfer the rest and drop them.
    let mut on_outer = std::collections::HashSet::new();
    for &face in topo.shell(topo.solid(solid_id)?.outer_shell())?.faces() {
        on_outer.extend(brepkit_topology::explorer::face_edges(topo, face)?);
    }
    if edge_ids.iter().any(|e| !on_outer.contains(e)) {
        return Err(brepkit_operations::OperationsError::InvalidInput {
            reason: "chamfering an edge of a cavity is not supported".to_owned(),
        });
    }
    let orientation_deflection = gate_deflection(topo, solid_id, distance)?;
    if let Ok(r) =
        brepkit_operations::blend_ops::chamfer_v2(topo, solid_id, edge_ids, distance, distance)
        && r.solid != solid_id
        && !r.is_partial
        && r.failed.is_empty()
        && r.succeeded.len() == edge_ids.len()
        && blend_result_holds(topo, r.solid, orientation_deflection)
    {
        return Ok(r.solid);
    }
    let s = brepkit_operations::chamfer::chamfer(topo, solid_id, edge_ids, distance)?;
    if s != solid_id && blend_result_holds(topo, s, orientation_deflection) {
        return Ok(s);
    }
    Err(brepkit_operations::OperationsError::InvalidInput {
        reason: "no chamfer engine produced a changed, closed, outward-oriented result".to_owned(),
    })
}

/// The deflection [`blend_result_holds`] reads a blend of `size` on `solid`
/// at: a tenth of the blend, at most 0.1, and at least a thousandth of the
/// solid's least extent up to 0.01, so a small or slender part's curves
/// still mesh within the check's 1%.
fn gate_deflection(
    topo: &brepkit_topology::Topology,
    solid: brepkit_topology::solid::SolidId,
    size: f64,
) -> Result<f64, brepkit_operations::OperationsError> {
    let floor = (1e-3 * least_extent(topo, solid)?).min(0.01);
    Ok((size * 0.1).clamp(floor, 0.1))
}

/// The solid's least extent across the box that best fits points along its
/// edges, a ball face's six farthest points in its own frame, and a NURBS
/// face's control points, so the part reads as thick however it is turned.
fn least_extent(
    topo: &brepkit_topology::Topology,
    solid: brepkit_topology::solid::SolidId,
) -> Result<f64, brepkit_operations::OperationsError> {
    use brepkit_topology::face::FaceSurface;
    const SAMPLES: u32 = 16;
    let mut points = Vec::new();
    for eid in brepkit_topology::explorer::solid_edges(topo, solid)? {
        let edge = topo.edge(eid)?;
        let start = topo.vertex(edge.start())?.point();
        let end = topo.vertex(edge.end())?.point();
        let curve = edge.curve();
        let (t0, t1) = curve.domain_with_endpoints(start, end);
        points.extend((0..=SAMPLES).map(|k| {
            let t = (t1 - t0).mul_add(f64::from(k) / f64::from(SAMPLES), t0);
            curve.evaluate_with_endpoints(t, start, end)
        }));
    }
    for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
        match topo.face(fid)?.surface() {
            FaceSurface::Sphere(s) => {
                use std::f64::consts::{FRAC_PI_2, PI};
                points.extend(
                    [(0.0, 0.0), (FRAC_PI_2, 0.0), (PI, 0.0), (-FRAC_PI_2, 0.0)]
                        .into_iter()
                        .chain([(0.0, FRAC_PI_2), (0.0, -FRAC_PI_2)])
                        .map(|(u, v)| s.evaluate(u, v)),
                );
            }
            FaceSurface::Nurbs(n) => points.extend(n.control_points().iter().flatten()),
            FaceSurface::Plane { .. }
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Torus(_) => {}
        }
    }
    if points.is_empty() {
        return Ok(0.0);
    }
    let fit = brepkit_math::obb::Obb3::from_points_slice(&points);
    Ok(2.0 * fit.half_extents.into_iter().fold(f64::INFINITY, f64::min))
}

/// Whether a fillet's or chamfer's result is a solid to keep, its volume
/// read at `orientation_deflection`.
///
/// A candidate is acceptable only if its outer shell is a CLOSED 2-manifold
/// (every edge used by exactly two faces — no free/boundary edges) with
/// opposite effective edge senses on adjacent faces and positive oriented
/// volume. Edge senses alone cannot detect a globally inverted shell, which
/// sends the next boolean through a corrupt mesh fallback.
fn blend_result_holds(
    topo: &brepkit_topology::Topology,
    s: brepkit_topology::solid::SolidId,
    orientation_deflection: f64,
) -> bool {
    let Ok(solid) = topo.solid(s) else {
        return false;
    };
    let shell_id = solid.outer_shell();
    let Ok(shell) = topo.shell(shell_id) else {
        return false;
    };
    // Closure by edge id is blind to a face that overlays a neighbour or
    // winds against it; the position-quantized view is the mesh, whose
    // directed edges must all pair. A result that fails this is
    // consumed as garbage downstream (a slot cut through it falls back
    // to a mesh), while a clean failure lets the caller keep its input.
    let watertight = || {
        brepkit_operations::tessellate::tessellate_solid_with_tolerance(
            topo,
            s,
            orientation_deflection,
            10.0_f64.to_radians(),
        )
        .is_ok_and(|mesh| brepkit_operations::tessellate::boundary_edge_count(&mesh) == 0)
    };
    brepkit_topology::validation::validate_shell_closed(shell, topo).is_ok()
        && brepkit_check::validate::shell::check_shell_orientation(topo, shell_id)
            .is_ok_and(|issues| issues.is_empty())
        && brepkit_operations::measure::oriented_solid_volume(topo, s, orientation_deflection)
            .is_ok_and(|meshed| {
                // A solid whose exact volume and meshed volume disagree
                // is not a consistent B-rep (a second-pass band thrown
                // past its neighbouring blend reads 1010.11 exact and
                // 985.42 meshed); a sound one agrees to the mesh's chord
                // error, far inside 1%.
                // Chord error alone can pass 1% at the coarse deflection
                // (a plain r = 10 cylinder meshes 1.24% short at 0.1), so
                // a disagreement is re-read ten times finer, where a sound
                // solid converges and an inconsistent one does not.
                let agrees = |exact: f64, meshed: f64| (exact - meshed).abs() <= 0.01 * exact.abs();
                meshed > 0.0
                    && brepkit_operations::measure::solid_volume(topo, s, orientation_deflection)
                        .is_ok_and(|exact| {
                            agrees(exact, meshed)
                                || brepkit_operations::measure::oriented_solid_volume(
                                    topo,
                                    s,
                                    orientation_deflection * 0.1,
                                )
                                .is_ok_and(|fine| agrees(exact, fine))
                        })
            })
        && watertight()
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

    use super::{try_chamfer, try_fillet, try_fillet_variable};

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

    fn solid_topology_fingerprint(topo: &Topology, solid_id: SolidId) -> Vec<String> {
        let shell = topo
            .shell(topo.solid(solid_id).expect("solid").outer_shell())
            .expect("shell");
        shell
            .faces()
            .iter()
            .map(|&face_id| {
                let face = topo.face(face_id).expect("face");
                let wires = std::iter::once(face.outer_wire())
                    .chain(face.inner_wires().iter().copied())
                    .map(|wire_id| {
                        topo.wire(wire_id)
                            .expect("wire")
                            .edges()
                            .iter()
                            .map(|edge| format!("{}:{}", edge.edge().index(), edge.is_forward()))
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .collect::<Vec<_>>()
                    .join(";");
                format!(
                    "{}:{}:{wires}",
                    face.surface().type_tag(),
                    face.is_reversed()
                )
            })
            .collect()
    }

    // The wasm `fillet` binding (and its batch sibling) route through
    // `try_fillet`. Filleting all 12 box edges must return the exact corrected
    // `FilletBuilder` topology and remove only the rounded slivers (volume
    // ≈975.6 for a 10³ box at r=1), not regress to the builder's historical
    // corner over-removal.
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
        let production_fingerprint = solid_topology_fingerprint(&topo, result);
        let mut direct_topo = Topology::new();
        let direct_cube =
            brepkit_operations::primitives::make_box(&mut direct_topo, 10.0, 10.0, 10.0).unwrap();
        let direct_edges = solid_edge_ids(&direct_topo, direct_cube);
        let direct = brepkit_operations::blend_ops::fillet_v2(
            &mut direct_topo,
            direct_cube,
            &direct_edges,
            1.0,
        )
        .expect("direct FilletBuilder result");
        assert_eq!(
            production_fingerprint,
            solid_topology_fingerprint(&direct_topo, direct.solid),
            "production dispatch must publish the corrected FilletBuilder result directly"
        );
    }

    // gh #967: filleting a plain cylinder's circular rim used to remove ~37%
    // of the volume. The corrected builder now rounds both rims into exact
    // quarter-torus bands, and the production validator accepts only the
    // watertight result (≈6275.7, a ~0.12% rim round) — never the corrupt
    // ~3978 rolling-ball result.
    #[test]
    fn try_fillet_cylinder_rim_rounds_not_corrupts() {
        use brepkit_topology::face::FaceSurface;

        let mut topo = Topology::new();
        let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, 10.0, 20.0).unwrap();
        let raw = brepkit_operations::measure::solid_volume(&topo, cyl, 0.01).unwrap();
        let edges = solid_edge_ids(&topo, cyl);

        let result = try_fillet(&mut topo, cyl, &edges, 0.5).expect("rim fillet");
        let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.01).unwrap();

        // A tiny rim round — well under 1% removed, never the −37% corruption.
        assert!(
            vol < raw && vol > raw * 0.99,
            "cylinder rim fillet should round (~{:.0}), got {vol} vs raw {raw}",
            raw * 0.999
        );
        let sh = topo
            .shell(topo.solid(result).unwrap().outer_shell())
            .unwrap();
        let torus_count = sh
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Torus(_)))
            .count();
        assert_eq!(torus_count, 2, "both rims round into toroidal bands");
        assert!(
            brepkit_operations::validate::validate_solid(&topo, result)
                .unwrap()
                .is_valid(),
            "rounded rim solid must be watertight"
        );
    }

    // An L bracket whose inside edge is filleted at 5, then two outer corner
    // edges chamfered by 3: the chamfers take two 3 x 3 triangles 8 tall,
    // 72 in all, from the filleted solid, as from the bare L.
    #[test]
    fn try_chamfer_cuts_a_filleted_bracket() {
        use brepkit_math::mat::Mat4;
        use brepkit_math::vec::Point3;
        use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};

        let near = |topo: &Topology, s, p: Point3| {
            solid_edge_ids(topo, s)
                .into_iter()
                .min_by(|&a, &b| {
                    let reach = |e| {
                        let edge = topo.edge(e).unwrap();
                        let (x, y) = (
                            topo.vertex(edge.start()).unwrap().point(),
                            topo.vertex(edge.end()).unwrap().point(),
                        );
                        (Point3::new(
                            f64::midpoint(x.x(), y.x()),
                            f64::midpoint(x.y(), y.y()),
                            f64::midpoint(x.z(), y.z()),
                        ) - p)
                            .length()
                    };
                    reach(a).total_cmp(&reach(b))
                })
                .unwrap()
        };
        for fillet_first in [true, false] {
            let mut topo = Topology::new();
            let base =
                brepkit_operations::primitives::make_box(&mut topo, 80.0, 50.0, 8.0).unwrap();
            let wall =
                brepkit_operations::primitives::make_box(&mut topo, 80.0, 8.0, 42.0).unwrap();
            brepkit_operations::transform::transform_solid(
                &mut topo,
                wall,
                &Mat4::translation(0.0, 42.0, 8.0),
            )
            .unwrap();
            let simplify = BooleanOptions {
                unify_faces: true,
                ..Default::default()
            };
            let mut solid =
                boolean_with_options(&mut topo, BooleanOp::Fuse, base, wall, simplify).unwrap();
            if fillet_first {
                let inside = near(&topo, solid, Point3::new(40.0, 42.0, 8.0));
                solid = try_fillet(&mut topo, solid, &[inside], 5.0).unwrap();
                assert!(
                    brepkit_topology::explorer::solid_faces(&topo, solid)
                        .unwrap()
                        .iter()
                        .any(|&f| !topo.face(f).unwrap().surface().is_planar()),
                    "the fillet should leave a round, not a bevel"
                );
            }
            let before = brepkit_operations::measure::solid_volume(&topo, solid, 0.01).unwrap();
            let corners = [
                near(&topo, solid, Point3::new(80.0, 0.0, 4.0)),
                near(&topo, solid, Point3::new(0.0, 0.0, 4.0)),
            ];
            let cut = try_chamfer(&mut topo, solid, &corners, 3.0).unwrap();
            let after = brepkit_operations::measure::solid_volume(&topo, cut, 0.01).unwrap();
            assert!(
                (before - after - 72.0).abs() < 1e-3 * 72.0,
                "fillet first {fillet_first}: {before} became {after}"
            );
            assert!(
                brepkit_operations::validate::validate_solid(&topo, cut)
                    .unwrap()
                    .is_valid()
            );
        }
    }

    /// A 10 cube less a 3 cube cavity at `corner`, and its top edge along
    /// x at y = 0.
    fn hollow_block(
        topo: &mut Topology,
        corner: (f64, f64, f64),
    ) -> (SolidId, brepkit_topology::edge::EdgeId) {
        use brepkit_operations::boolean::{BooleanOp, boolean};

        let block = brepkit_operations::primitives::make_box(topo, 10.0, 10.0, 10.0).unwrap();
        let cube = brepkit_operations::primitives::make_box(topo, 3.0, 3.0, 3.0).unwrap();
        brepkit_operations::transform::transform_solid(
            topo,
            cube,
            &brepkit_math::mat::Mat4::translation(corner.0, corner.1, corner.2),
        )
        .unwrap();
        let hollow = boolean(topo, BooleanOp::Cut, block, cube).unwrap();
        let edge = solid_edge_ids(topo, hollow)
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                [edge.start(), edge.end()].iter().all(|&v| {
                    let p = topo.vertex(v).unwrap().point();
                    p.y().abs() < 1e-9 && (p.z() - 10.0).abs() < 1e-9
                })
            })
            .unwrap();
        (hollow, edge)
    }

    /// A box's top front edge, 40 long, filleted along a radius running 2
    /// to 5, and along a constant 2 given as a law: each is a round inside
    /// the box, its band ending in arcs on the end faces, and the solid
    /// loses `(1 − π/4) ∫ r²` along the edge.
    #[test]
    fn try_fillet_variable_rounds_an_edge() {
        use brepkit_operations::fillet::FilletRadiusLaw;
        use std::f64::consts::FRAC_PI_4;

        for (law, removed) in [
            (
                FilletRadiusLaw::Linear {
                    start: 2.0,
                    end: 5.0,
                },
                (1.0 - FRAC_PI_4) * 40.0 * (4.0 + 10.0 + 25.0) / 3.0,
            ),
            (
                FilletRadiusLaw::Constant(2.0),
                (1.0 - FRAC_PI_4) * 4.0 * 40.0,
            ),
        ] {
            let mut topo = Topology::new();
            let block =
                brepkit_operations::primitives::make_box(&mut topo, 40.0, 30.0, 20.0).unwrap();
            let edge = solid_edge_ids(&topo, block)
                .into_iter()
                .find(|&e| {
                    let edge = topo.edge(e).unwrap();
                    [edge.start(), edge.end()].iter().all(|&v| {
                        let p = topo.vertex(v).unwrap().point();
                        p.y().abs() < 1e-9 && (p.z() - 20.0).abs() < 1e-9
                    })
                })
                .unwrap();
            let rounded = try_fillet_variable(&mut topo, block, &[(edge, law.clone())]).unwrap();
            assert!(
                brepkit_operations::validate::validate_solid(&topo, rounded)
                    .unwrap()
                    .is_valid()
            );
            let arcs = solid_edge_ids(&topo, rounded)
                .into_iter()
                .filter(|&e| {
                    matches!(
                        topo.edge(e).unwrap().curve(),
                        brepkit_topology::edge::EdgeCurve::Circle(_)
                    )
                })
                .count();
            assert_eq!(arcs, 2, "{law:?}");
            let bounds = brepkit_operations::measure::solid_bounding_box(&topo, rounded).unwrap();
            assert!(
                bounds.min.y() > -1e-9 && bounds.max.z() < 20.0 + 1e-9,
                "{bounds:?}"
            );
            let volume = brepkit_operations::measure::solid_volume(&topo, rounded, 0.01).unwrap();
            let expected = 24000.0 - removed;
            assert!(
                (volume - expected).abs() < 1e-2 * removed,
                "{law:?}: volume {volume}, expected {expected}"
            );
        }
    }

    /// A chamfer of a hollow block keeps a cavity of its own: moving the
    /// result leaves the input's cavity where it was. A chamfer that would
    /// open a cavity 0.05 behind the front face is refused.
    #[test]
    fn try_chamfer_keeps_a_cavity_of_its_own() {
        let mut topo = Topology::new();
        let (hollow, edge) = hollow_block(&mut topo, (3.5, 3.5, 3.5));
        let cut = try_chamfer(&mut topo, hollow, &[edge], 1.0).unwrap();
        let volume = brepkit_operations::measure::solid_volume(&topo, cut, 0.01).unwrap();
        assert!((volume - (1000.0 - 27.0 - 5.0)).abs() < 1e-6, "{volume}");
        let input_cavity = topo.solid(hollow).unwrap().inner_shells()[0];
        assert_ne!(topo.solid(cut).unwrap().inner_shells(), &[input_cavity]);
        brepkit_operations::transform::transform_solid(
            &mut topo,
            cut,
            &brepkit_math::mat::Mat4::translation(0.0, 0.0, 100.0),
        )
        .unwrap();
        for &face in topo.shell(input_cavity).unwrap().faces() {
            for v in brepkit_topology::explorer::face_vertices(&topo, face).unwrap() {
                let z = topo.vertex(v).unwrap().point().z();
                assert!(
                    (3.5 - 1e-9..=6.5 + 1e-9).contains(&z),
                    "input cavity at z {z}"
                );
            }
        }

        let mut topo = Topology::new();
        let (near, edge) = hollow_block(&mut topo, (3.5, 0.05, 6.5));
        assert!(try_chamfer(&mut topo, near, &[edge], 1.0).is_err());
    }

    /// A chamfer naming an edge of a hollow block's cavity, alone or with an
    /// outer edge, is refused rather than chamfering the outer edge and
    /// dropping the cavity's.
    #[test]
    fn try_chamfer_refuses_a_cavitys_edge() {
        let mut topo = Topology::new();
        let (hollow, edge) = hollow_block(&mut topo, (3.5, 3.5, 3.5));
        let cavity = topo.solid(hollow).unwrap().inner_shells()[0];
        let cavity_face = topo.shell(cavity).unwrap().faces()[0];
        let cavity_edge = brepkit_topology::explorer::face_edges(&topo, cavity_face).unwrap()[0];
        assert!(try_chamfer(&mut topo, hollow, &[edge, cavity_edge], 0.5).is_err());
        assert!(try_chamfer(&mut topo, hollow, &[cavity_edge], 0.5).is_err());
        assert!(try_chamfer(&mut topo, hollow, &[edge], 0.5).is_ok());
    }

    /// A chamfer naming an edge of another solid is refused rather than
    /// chamfering the rest.
    #[test]
    fn try_chamfer_refuses_an_edge_off_the_solid() {
        let mut topo = Topology::new();
        let a = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let edges = [solid_edge_ids(&topo, a)[0], solid_edge_ids(&topo, b)[0]];
        assert!(try_chamfer(&mut topo, a, &edges, 1.0).is_err());
    }

    /// A 0.002 chamfer round the top rim of a cylinder of radius 0.01 and
    /// height 0.03 is kept: its curves mesh finely enough for the check.
    #[test]
    fn try_chamfer_keeps_a_small_cylinders_rim_chamfer() {
        let mut topo = Topology::new();
        let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, 0.01, 0.03).unwrap();
        let rim = solid_edge_ids(&topo, cyl)
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(c)
                    if c.center().z() > 0.02)
            })
            .unwrap();
        let cut = try_chamfer(&mut topo, cyl, &[rim], 0.002).unwrap();
        let volume = brepkit_operations::measure::solid_volume(&topo, cut, 1e-5).unwrap();
        let full = std::f64::consts::PI * 1e-4 * 0.03;
        assert!(volume < full && volume > 0.97 * full, "{volume} of {full}");
    }

    /// A 0.002 chamfer round the top rim of a cylinder of radius 0.01 and
    /// height 100 is kept: the part is long, but its curves are small.
    #[test]
    fn try_chamfer_keeps_a_slender_cylinders_rim_chamfer() {
        let mut topo = Topology::new();
        let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, 0.01, 100.0).unwrap();
        let rim = solid_edge_ids(&topo, cyl)
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(c)
                    if c.center().z() > 50.0)
            })
            .unwrap();
        let cut = try_chamfer(&mut topo, cyl, &[rim], 0.002).unwrap();
        let volume = brepkit_operations::measure::solid_volume(&topo, cut, 1e-5).unwrap();
        let full = std::f64::consts::PI * 1e-4 * 100.0;
        assert!(volume < full && volume > 0.999 * full, "{volume} of {full}");
    }

    /// The same cylinder turned to run along the cube's diagonal, its top
    /// rim chamfered alike, is kept too, though its axis-aligned box is some
    /// 58 across every way: the gate reads the part's thickness across the
    /// box that fits it, which turning leaves alone.
    #[test]
    fn try_chamfer_keeps_a_turned_slender_cylinders_rim_chamfer() {
        let mut topo = Topology::new();
        let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, 0.01, 100.0).unwrap();
        let tilt = (1.0_f64 / 2.0_f64.sqrt()).atan();
        let turn = brepkit_math::mat::Mat4::rotation_x(-tilt)
            * brepkit_math::mat::Mat4::rotation_y(std::f64::consts::FRAC_PI_4);
        brepkit_operations::transform::transform_solid(&mut topo, cyl, &turn).unwrap();
        let rim = solid_edge_ids(&topo, cyl)
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(c)
                    if c.center().z() > 30.0)
            })
            .unwrap();
        let cut = try_chamfer(&mut topo, cyl, &[rim], 0.002).unwrap();
        let volume = brepkit_operations::measure::solid_volume(&topo, cut, 1e-5).unwrap();
        let full = std::f64::consts::PI * 1e-4 * 100.0;
        assert!(volume < full && volume > 0.999 * full, "{volume} of {full}");
    }

    /// A chamfer of 2e-5 on a box of side 1e-4 is kept: its volume of about
    /// 1e-12 is small only at that scale.
    #[test]
    fn try_chamfer_keeps_a_small_parts_chamfer() {
        let mut topo = Topology::new();
        let b = brepkit_operations::primitives::make_box(&mut topo, 1e-4, 1e-4, 1e-4).unwrap();
        let edge = solid_edge_ids(&topo, b)[0];
        let cut = try_chamfer(&mut topo, b, &[edge], 2e-5).unwrap();
        let volume = brepkit_operations::measure::solid_volume(&topo, cut, 1e-6).unwrap();
        let expected = 1e-12 - 0.5 * 2e-5 * 2e-5 * 1e-4;
        assert!((volume - expected).abs() < 1e-3 * expected, "{volume}");
    }

    // A wide rim's chord error alone reads past 1% at the gate's coarse
    // deflection (a plain r = 10 cylinder meshes 1.24% short at 0.1); the
    // exact round must not be rejected for it.
    #[test]
    fn try_fillet_accepts_a_wide_cylinders_rim_round() {
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, 10.0, 20.0).unwrap();
        let top: Vec<EdgeId> = solid_edge_ids(&topo, cyl)
            .into_iter()
            .filter(|&e| {
                let edge = topo.edge(e).unwrap();
                matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_))
                    && (topo.vertex(edge.start()).unwrap().point().z() - 20.0).abs() < 1e-9
            })
            .collect();
        assert_eq!(top.len(), 1);

        let result = try_fillet(&mut topo, cyl, &top, 2.0).expect("rim fillet");
        let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.001).unwrap();
        let section = (1.0 - PI / 4.0) * 4.0;
        let centroid = 2.0 * 3.0f64.mul_add(-PI, 10.0) / (3.0 * (4.0 - PI));
        let truth = PI * 100.0 * 20.0 - 2.0 * PI * (10.0 - centroid) * section;
        assert!((vol - truth).abs() < 1e-6 * truth, "{vol} vs {truth}");
    }

    #[test]
    fn try_fillet_second_pass_does_not_break_solid() {
        // #813: a second fillet whose target edge borders the first fillet's
        // curved blend face must not produce a self-intersecting solid — the
        // volume grew past the base to 1000.30 before the fix; such edges are
        // now rejected unless the result remains valid. Checked over *every*
        // result edge so the guard does not rely on a particular edge ordering.
        let mut topo = Topology::new();
        let cube = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let edges = solid_edge_ids(&topo, cube);
        let first = try_fillet(&mut topo, cube, &[edges[0], edges[1]], 1.0).expect("first fillet");

        // The scenario under test only exists if the first fillet produced
        // curved blend faces for the later edges to border. Constant-radius
        // planar supports use exact analytic surfaces; walking cases use NURBS.
        let sd = topo.solid(first).expect("solid");
        let sh = topo.shell(sd.outer_shell()).expect("shell");
        let has_blend = sh
            .faces()
            .iter()
            .any(|&fid| topo.face(fid).is_ok_and(|f| !f.surface().is_planar()));
        assert!(has_blend, "first fillet should create curved blend faces");

        // Filleting any filter-eligible single result edge must either return a
        // consistently oriented manifold that does not inflate past the
        // original box (the #813 bug grew it to 1000.30), or report that no
        // engine could construct one. Tangent result edges fail earlier.
        let r_edges = solid_edge_ids(&topo, first);
        let mut attempted = 0usize;
        let mut rejected = 0usize;
        for &e in &r_edges {
            if brepkit_operations::query::filter_filletable_edges(&topo, first, &[e])
                .expect("filter second-pass edge")
                .is_empty()
            {
                continue;
            }
            attempted += 1;
            let mut t = topo.clone();
            match try_fillet(&mut t, first, &[e], 0.5) {
                Ok(s) => {
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
                Err(error) => {
                    assert!(
                        error.to_string().contains(
                            "no fillet engine produced a changed, closed, outward-oriented result"
                        ),
                        "unexpected second-fillet error: {error}"
                    );
                    rejected += 1;
                }
            }
        }
        assert!(attempted > 0, "fixture should retain filter-eligible edges");
        assert!(
            rejected > 0,
            "fixture should exercise explicit rejection instead of unchanged success"
        );
    }

    #[test]
    fn try_fillet_nurbs_blend_neighbor_is_watertight() {
        use std::collections::HashMap;

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

        let blend_faces: HashSet<usize> = {
            let sh = topo
                .shell(topo.solid(first).unwrap().outer_shell())
                .unwrap();
            sh.faces()
                .iter()
                .filter(|&&f| !topo.face(f).unwrap().surface().is_planar())
                .map(|f| f.index())
                .collect()
        };
        assert!(
            !blend_faces.is_empty(),
            "first fillet must create a blend face"
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
                        .is_some_and(|fs| fs.iter().any(|f| blend_faces.contains(f)))
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

    #[test]
    fn try_fillet_rejects_unchanged_batch() {
        let mut topo = Topology::new();
        let cube = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let edges = solid_edge_ids(&topo, cube);
        let first = try_fillet(&mut topo, cube, &[edges[0], edges[1]], 1.0).expect("first fillet");
        let result_edges = solid_edge_ids(&topo, first);
        let supported: HashSet<_> =
            brepkit_operations::query::filter_filletable_edges(&topo, first, &result_edges)
                .expect("filter result edges")
                .iter()
                .map(|edge| edge.index())
                .collect();
        let unsupported: Vec<_> = result_edges
            .iter()
            .copied()
            .filter(|edge| !supported.contains(&edge.index()))
            .collect();
        assert!(
            !unsupported.is_empty(),
            "first fillet should create tangent contact edges"
        );

        let supported_edge = result_edges
            .iter()
            .copied()
            .find(|edge| supported.contains(&edge.index()))
            .expect("first fillet should leave a supported edge");
        let mixed = [supported_edge, unsupported[0]];
        let mixed_error = try_fillet(&mut topo, first, &mixed, 0.5)
            .expect_err("a mixed batch must not publish a filleted subset");
        assert!(
            mixed_error
                .to_string()
                .contains("some requested edges are not filletable"),
            "unexpected error: {mixed_error}"
        );

        let error = try_fillet(&mut topo, first, &unsupported, 0.5)
            .expect_err("an unchanged batch must not report the input solid as success");
        assert!(
            error
                .to_string()
                .contains("none of the requested edges are filletable"),
            "unexpected error: {error}"
        );
    }
}
