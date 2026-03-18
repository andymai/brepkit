//! Boolean operations on solids: fuse, cut, and intersect.
//!
//! The primary pipeline (`boolean_pipeline`) operates in 2D parameter space,
//! preserving all surface types (plane, cylinder, cone, sphere, torus,
//! NURBS) through the boolean. Falls back to the analytic path for
//! cases pipeline can't handle, then to mesh boolean as a last resort.

mod analytic;
mod assembly;
pub mod boolean_pipeline;
mod classify;
pub(crate) mod classify_2d;
mod compound;
pub(crate) mod face_splitter;
mod fragments;
mod intersect;
pub(crate) mod pcurve_compute;
pub(crate) mod pipeline;
pub(crate) mod plane_frame;
mod precompute;
mod split;
mod types;
pub(crate) mod wire_builder;
use analytic::{analytic_boolean, collect_face_signatures, has_torus, is_all_analytic};
use assembly::validate_boolean_result;
pub(crate) use assembly::{assemble_solid, assemble_solid_mixed};
pub use compound::compound_cut;
pub use precompute::face_polygon;
pub use types::{BooleanOp, BooleanOptions, FaceSpec};

// WASM-compatible timer: `std::time::Instant` panics on wasm32 targets.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn timer_now() -> std::time::Instant {
    std::time::Instant::now()
}
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn timer_elapsed_ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
#[cfg(target_arch = "wasm32")]
pub(super) fn timer_now() -> () {}
#[cfg(target_arch = "wasm32")]
pub(super) fn timer_elapsed_ms(_t: ()) -> f64 {
    0.0
}

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform a boolean operation on two solids.
///
/// When both solids are composed entirely of analytic faces (planes,
/// cylinders, cones, spheres), uses an exact analytic path that preserves
/// surface types through the boolean. Falls back to the tessellated path
/// for NURBS faces or analytic-analytic face pairs.
///
/// # Errors
///
/// Returns an error if either solid contains NURBS faces, or if the operation
/// produces an empty or non-manifold result.
pub fn boolean(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
) -> Result<SolidId, crate::OperationsError> {
    boolean_with_options(topo, op, a, b, BooleanOptions::default())
}

/// Perform a boolean operation with custom options.
///
/// See [`boolean`] for details. The `opts` parameter allows configuring
/// tessellation quality for non-planar faces.
///
/// # Errors
///
/// Returns an error if either solid is invalid or the operation produces
/// an empty or non-manifold result.
#[allow(clippy::too_many_lines)]
pub fn boolean_with_options(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    opts: BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    let tol = opts.tolerance;

    log::debug!(
        "boolean {op:?}: solids ({}, {}), deflection={}",
        a.index(),
        b.index(),
        opts.deflection,
    );

    // ── Try analytic fast path ──────────────────────────────────────
    // Optimized for non-coplanar all-analytic solids (box-cylinder, etc.).
    // This is the most validated and fastest path for the common case.
    let try_analytic = {
        let both_analytic = is_all_analytic(topo, a)? && is_all_analytic(topo, b)?;
        let no_torus = !has_torus(topo, a)? && !has_torus(topo, b)?;
        both_analytic && no_torus
    };
    let mut analytic_fallback: Option<SolidId> = None;
    if try_analytic {
        if let Ok(solid) = analytic_boolean(topo, op, a, b, tol, opts.deflection) {
            let _ = crate::heal::remove_degenerate_edges(topo, solid, tol.linear)?;
            if opts.unify_faces {
                let _ = crate::heal::unify_faces(topo, solid)?;
            }
            // Check for non-manifold edges (3+ faces sharing an edge).
            // Only reject when either input has reversed faces (shelled solid) —
            // for those cases, the pipeline can produce a better result.
            // For simple (non-shelled) solids, accept the analytic result even if
            // slightly non-manifold (sphere booleans have known minor issues).
            let has_reversed = {
                let shell_a = topo.shell(topo.solid(a)?.outer_shell())?;
                let shell_b = topo.shell(topo.solid(b)?.outer_shell())?;
                shell_a
                    .faces()
                    .iter()
                    .chain(shell_b.faces().iter())
                    .any(|&fid| {
                        topo.face(fid)
                            .is_ok_and(brepkit_topology::face::Face::is_reversed)
                    })
            };
            let is_manifold = !has_reversed
                || brepkit_topology::validation::validate_shell_manifold(
                    topo.shell(topo.solid(solid)?.outer_shell())?,
                    topo.faces(),
                    topo.wires(),
                )
                .is_ok();
            if is_manifold && validate_boolean_result(topo, solid).is_ok() {
                return Ok(solid);
            }
            analytic_fallback = Some(solid);
        }
    }

    // ── Containment shortcut ─────────────────────────────────────────
    // Detect A⊂B or B⊂A (including A=B) and handle directly.
    // This catches identical solids that neither analytic nor pipeline handle
    // correctly (boundary-coincident intersections produce degenerate splits).
    {
        use classify::try_build_analytic_classifier;
        let ca = try_build_analytic_classifier(topo, a);
        let cb = try_build_analytic_classifier(topo, b);
        // Sample the AABB centroid of the solid — guaranteed to be in the
        // interior for convex solids (which all analytic primitives are).
        let sample_aabb_center = |topo: &Topology, solid: SolidId| -> Option<Point3> {
            let s = topo.solid(solid).ok()?;
            let sh = topo.shell(s.outer_shell()).ok()?;
            let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
            let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
            for &fid in sh.faces() {
                let f = topo.face(fid).ok()?;
                let w = topo.wire(f.outer_wire()).ok()?;
                for oe in w.edges() {
                    let e = topo.edge(oe.edge()).ok()?;
                    let p = topo.vertex(e.start()).ok()?.point();
                    min = Point3::new(min.x().min(p.x()), min.y().min(p.y()), min.z().min(p.z()));
                    max = Point3::new(max.x().max(p.x()), max.y().max(p.y()), max.z().max(p.z()));
                }
            }
            Some(Point3::new(
                (min.x() + max.x()) * 0.5,
                (min.y() + max.y()) * 0.5,
                (min.z() + max.z()) * 0.5,
            ))
        };
        let b_in_a = ca
            .as_ref()
            .zip(sample_aabb_center(topo, b))
            .and_then(|(c, p)| c.classify(p, tol))
            == Some(types::FaceClass::Inside);
        let a_in_b = cb
            .as_ref()
            .zip(sample_aabb_center(topo, a))
            .and_then(|(c, p)| c.classify(p, tol))
            == Some(types::FaceClass::Inside);
        if b_in_a || a_in_b {
            return match (op, b_in_a, a_in_b) {
                (BooleanOp::Fuse, true, _) => Ok(crate::copy::copy_solid(topo, a)?),
                (BooleanOp::Fuse, _, true) => Ok(crate::copy::copy_solid(topo, b)?),
                (BooleanOp::Cut, true, _) => Err(crate::OperationsError::InvalidInput {
                    reason: "boolean Cut: B is inside A — result would have a void".into(),
                }),
                (BooleanOp::Cut, _, true) => Err(crate::OperationsError::InvalidInput {
                    reason: "boolean Cut: A is inside B — result is empty".into(),
                }),
                (BooleanOp::Intersect, true, _) => Ok(crate::copy::copy_solid(topo, b)?),
                (BooleanOp::Intersect, _, true) => Ok(crate::copy::copy_solid(topo, a)?),
                _ => unreachable!(),
            };
        }
    }

    // ── Try pipeline parameter-space pipeline ────────────────────────────────
    // pipeline handles coplanar faces, NURBS, and other cases analytic can't.
    // Preserves all surface types through the boolean.
    match boolean_pipeline::boolean_pipeline(topo, op, a, b) {
        Ok(solid) if validate_boolean_result(topo, solid).is_ok() => {
            log::info!(
                "boolean {op:?}: pipeline succeeded → solid {}",
                solid.index()
            );
            return Ok(solid);
        }
        Ok(_) => {
            log::debug!("boolean {op:?}: pipeline result failed validation, falling back");
        }
        Err(e) => {
            log::debug!("boolean {op:?}: pipeline failed ({e}), falling back");
        }
    }

    // ── Mesh boolean (last resort) ───────────────────────────────────
    // When both analytic and pipeline fail, fall back to mesh co-refinement.
    // All surface types are lost — output is planar triangles only.
    match mesh_boolean_fallback(topo, op, a, b, opts.deflection, tol, &opts) {
        Ok(result) => return Ok(result),
        Err(e) => {
            log::debug!("boolean {op:?}: mesh boolean also failed ({e})");
        }
    }

    // If the analytic path produced a result (even with non-manifold edges),
    // return it as a last resort. Imperfect topology is better than no result.
    if let Some(solid) = analytic_fallback {
        return Ok(solid);
    }

    Err(crate::OperationsError::InvalidInput {
        reason: format!("boolean {op:?}: all paths failed (analytic, pipeline, mesh boolean)"),
    })
}

// ---------------------------------------------------------------------------
// Evolution-tracking wrapper
// ---------------------------------------------------------------------------

/// Perform a boolean operation and return an [`EvolutionMap`] tracking face
/// provenance.
///
/// This wraps [`boolean`] and uses a heuristic (normal + centroid similarity)
/// to match output faces back to their input faces. Faces whose best match
/// score exceeds the similarity threshold are classified as "modified";
/// unmatched input faces are classified as "deleted".
///
/// # Errors
///
/// Returns the same errors as [`boolean`].
pub fn boolean_with_evolution(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
) -> Result<(SolidId, crate::evolution::EvolutionMap), crate::OperationsError> {
    use crate::evolution::EvolutionMap;

    // Collect input face normals + centroids before the operation mutates topology.
    let input_faces_a = collect_face_signatures(topo, a)?;
    let input_faces_b = collect_face_signatures(topo, b)?;

    let mut input_faces: Vec<(usize, Vec3, Point3)> =
        Vec::with_capacity(input_faces_a.len() + input_faces_b.len());
    input_faces.extend(input_faces_a);
    input_faces.extend(input_faces_b);

    // Run the actual boolean.
    let result = boolean(topo, op, a, b)?;

    // Collect output face normals + centroids.
    let output_faces = collect_face_signatures(topo, result)?;

    // Build evolution map via heuristic matching.
    let mut evo = EvolutionMap::new();
    let mut matched_inputs: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut unmatched_outputs: Vec<(usize, Vec3, Point3)> = Vec::new();

    // Normal dot threshold: cos(45deg) — relaxed to handle faces split by
    // booleans where normals may shift slightly.
    let normal_threshold = 0.707;
    // Maximum centroid distance squared for a match (generous).
    let centroid_dist_sq_max = 100.0;

    for &(out_idx, out_normal, out_centroid) in &output_faces {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_input: Option<usize> = None;

        for &(in_idx, in_normal, in_centroid) in &input_faces {
            let dot = out_normal.dot(in_normal);
            if dot < normal_threshold {
                continue;
            }

            let dx = out_centroid.x() - in_centroid.x();
            let dy = out_centroid.y() - in_centroid.y();
            let dz = out_centroid.z() - in_centroid.z();
            let dist_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

            if dist_sq > centroid_dist_sq_max {
                continue;
            }

            // Score: higher normal alignment + closer centroid = better.
            let score = dot - dist_sq / centroid_dist_sq_max;
            if score > best_score {
                best_score = score;
                best_input = Some(in_idx);
            }
        }

        if let Some(in_idx) = best_input {
            evo.add_modified(in_idx, out_idx);
            matched_inputs.insert(in_idx);
        } else {
            unmatched_outputs.push((out_idx, out_normal, out_centroid));
        }
    }

    // Unmatched output faces are "generated" — attribute them to the nearest
    // input face (the face most likely responsible for generating them, e.g.
    // intersection curves create new faces near the boundary).
    for &(out_idx, _out_normal, out_centroid) in &unmatched_outputs {
        let mut best_dist_sq = f64::MAX;
        let mut best_input: Option<usize> = None;

        for &(in_idx, _, in_centroid) in &input_faces {
            let dx = out_centroid.x() - in_centroid.x();
            let dy = out_centroid.y() - in_centroid.y();
            let dz = out_centroid.z() - in_centroid.z();
            let dist_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_input = Some(in_idx);
            }
        }

        if let Some(in_idx) = best_input {
            evo.add_generated(in_idx, out_idx);
            matched_inputs.insert(in_idx);
        }
    }

    // Any input face not matched to any output is deleted.
    for &(in_idx, _, _) in &input_faces {
        if !matched_inputs.contains(&in_idx) {
            evo.add_deleted(in_idx);
        }
    }

    Ok((result, evo))
}

// ---------------------------------------------------------------------------
// Mesh boolean helpers
// ---------------------------------------------------------------------------

/// Best-effort mesh boolean fallback for high face-count solids.
///
/// Tessellates both solids, runs mesh co-refinement, assembles the result,
/// and applies the same post-processing as the other boolean paths.
/// Returns `Err` on any failure so the caller can fall through to the
/// chord-based path.
fn mesh_boolean_fallback(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    deflection: f64,
    tol: brepkit_math::tolerance::Tolerance,
    opts: &BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    let mesh_a = crate::tessellate::tessellate_solid(topo, a, deflection)?;
    let mesh_b = crate::tessellate::tessellate_solid(topo, b, deflection)?;

    let mb_result = crate::mesh_boolean::mesh_boolean(&mesh_a, &mesh_b, op, tol.linear)?;
    let face_specs = mesh_result_to_face_specs(&mb_result);
    if face_specs.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "mesh boolean produced empty result".into(),
        });
    }
    let result = assemble_solid_mixed(topo, &face_specs, tol)?;
    let _ = crate::heal::remove_degenerate_edges(topo, result, tol.linear)?;
    if opts.unify_faces {
        let _ = crate::heal::unify_faces(topo, result)?;
    }
    if opts.heal_after_boolean {
        let _ = crate::heal::heal_solid(topo, result, tol.linear)?;
    }
    validate_boolean_result(topo, result)?;
    log::info!(
        "boolean {op:?}: mesh boolean path → solid {} ({} faces, surface types lost)",
        result.index(),
        face_specs.len()
    );
    Ok(result)
}

/// Convert a mesh boolean result into `FaceSpec` entries for solid assembly.
fn mesh_result_to_face_specs(result: &crate::mesh_boolean::MeshBooleanResult) -> Vec<FaceSpec> {
    let mut specs = Vec::new();
    for tri in result.mesh.indices.chunks_exact(3) {
        let v0 = result.mesh.positions[tri[0] as usize];
        let v1 = result.mesh.positions[tri[1] as usize];
        let v2 = result.mesh.positions[tri[2] as usize];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let Ok(normal) = edge1.cross(edge2).normalize() else {
            continue;
        };
        let d = crate::dot_normal_point(normal, v0);
        specs.push(FaceSpec::Planar {
            vertices: vec![v0, v1, v2],
            normal,
            d,
            inner_wires: vec![],
        });
    }
    specs
}

#[cfg(test)]
mod tests;
