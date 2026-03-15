//! Reproducer tests for gridfinity-layout-tool dual-kernel failures.
//!
//! These tests reproduce the scenarios from issues #258, #259, #260 using
//! WASM contract tests via `execute_batch()`. Tests that reproduce known
//! bugs are marked `#[ignore]` with a comment linking to the issue.
//!
//! # Categories
//!
//! - **A**: Compound boolean crash reproducers (#258)
//! - **B**: Volume / bounding box regression reproducers (#260)
//!
//! Tessellation reproducers (#259) live in
//! `crates/operations/src/tessellate.rs` since `tessellateSolid` is not
//! in the batch dispatcher.
//!
//! # Solid Handle Numbering
//!
//! Solid handles are arena indices, NOT batch result indices.
//! Ops that create new solids: `makeBox`, `makeCylinder`, `makeSphere`,
//! `makeCone`, `makeTorus`, `copyAndTransformSolid`, `copySolid`,
//! `fuse`, `cut`, `intersect`, `compoundCut`, `fillet`, `chamfer`,
//! `extrude`, `revolve`, `sweep`, `loft`, `loftSmooth`, `shell`, `pipe`.
//!
//! Ops that do NOT create new solids (return same handle or a float):
//! `transform`, `volume`, `surfaceArea`, `boundingBox`, `centerOfMass`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::kernel::BrepKernel;

/// Parse the batch result JSON and return the parsed array.
fn parse_batch(result: &str) -> serde_json::Value {
    serde_json::from_str(result).expect("batch result should be valid JSON")
}

/// Check that batch result at `idx` has an `"ok"` field (not an error).
fn assert_ok(parsed: &serde_json::Value, idx: usize) {
    assert!(
        parsed[idx].get("ok").is_some(),
        "expected ok at index {idx}, got: {}",
        parsed[idx]
    );
}

/// Check that batch result at `idx` completed without crash.
fn assert_no_crash(parsed: &serde_json::Value, idx: usize, msg: &str) {
    assert!(
        parsed[idx].get("ok").is_some() || parsed[idx].get("error").is_some(),
        "{msg}: got: {}",
        parsed[idx]
    );
}

/// Extract the `"ok"` value as f64 (volume, area, etc.).
fn ok_f64(parsed: &serde_json::Value, idx: usize) -> f64 {
    parsed[idx]["ok"]
        .as_f64()
        .unwrap_or_else(|| panic!("expected ok f64 at index {idx}, got: {}", parsed[idx]))
}

/// Extract bounding box as `[min_x, min_y, min_z, max_x, max_y, max_z]`.
fn ok_bbox(parsed: &serde_json::Value, idx: usize) -> [f64; 6] {
    let arr = parsed[idx]["ok"]
        .as_array()
        .unwrap_or_else(|| panic!("expected ok array at index {idx}, got: {}", parsed[idx]));
    assert_eq!(arr.len(), 6, "bbox should have 6 elements");
    let mut out = [0.0; 6];
    for (i, v) in arr.iter().enumerate() {
        out[i] = v.as_f64().unwrap();
    }
    out
}

/// Build a row-major translation matrix JSON fragment.
///
/// `Mat4` is row-major: `rows[i][j] = elems[i*4+j]`.
/// Translation goes at `[0][3], [1][3], [2][3]` (flat indices 3, 7, 11).
fn translate_matrix(x: f64, y: f64, z: f64) -> String {
    format!("[1,0,0,{x}, 0,1,0,{y}, 0,0,1,{z}, 0,0,0,1]")
}

// ═══════════════════════════════════════════════════════════════════════
// Category A: Compound Boolean Crash Reproducers (#258)
//
// These reproduce scenarios where compound booleans with 3+ tools cause
// RefCell aliasing panics in the WASM layer. The test passes if the
// operation completes without panic (returning an error is also acceptable).
// ═══════════════════════════════════════════════════════════════════════

/// 4 cylinder magnet sockets cut from a baseplate.
///
/// Solids: 0=box, 1=cylinder, 2-5=copies, 6=compoundCut result
#[test]
fn compound_cut_4_cylinders() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 42, "height": 42, "depth": 5}},
        {"op": "makeCylinder", "args": {"radius": 3.0, "height": 10.0}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,5, 0,1,0,5, 0,0,1,-2.5, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,37, 0,1,0,5, 0,0,1,-2.5, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,5, 0,1,0,37, 0,0,1,-2.5, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,37, 0,1,0,37, 0,0,1,-2.5, 0,0,0,1]}},
        {"op": "compoundCut", "args": {"target": 0, "tools": [2, 3, 4, 5]}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    assert_no_crash(&parsed, 6, "compoundCut with 4 cylinders should not crash");
}

/// 4 box cutouts from each wall side.
///
/// Solids: 0=target, 1=x-template, 2-3=x-cutouts, 4=y-template,
///         5-6=y-cutouts, 7=compoundCut result
#[test]
fn compound_cut_wall_cutouts() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 20, "height": 20, "depth": 10}},
        {"op": "makeBox", "args": {"width": 5, "height": 22, "depth": 5}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,-2, 0,1,0,-1, 0,0,1,2.5, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,17, 0,1,0,-1, 0,0,1,2.5, 0,0,0,1]}},
        {"op": "makeBox", "args": {"width": 22, "height": 5, "depth": 5}},
        {"op": "copyAndTransformSolid", "args": {"solid": 4, "matrix": [1,0,0,-1, 0,1,0,-2, 0,0,1,2.5, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 4, "matrix": [1,0,0,-1, 0,1,0,17, 0,0,1,2.5, 0,0,0,1]}},
        {"op": "compoundCut", "args": {"target": 0, "tools": [2, 3, 5, 6]}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    assert_no_crash(&parsed, 7, "compoundCut with wall cutouts should not crash");
}

/// Mixed insert shapes: cylinder + 2 boxes.
///
/// Solids: 0=target, 1=cylinder, 2=cyl-copy, 3=box-template,
///         4-5=box-copies, 6=compoundCut result
#[test]
fn compound_cut_inserts_mixed() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 30, "height": 30, "depth": 8}},
        {"op": "makeCylinder", "args": {"radius": 2.0, "height": 12.0}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,15, 0,1,0,15, 0,0,1,-2, 0,0,0,1]}},
        {"op": "makeBox", "args": {"width": 4, "height": 4, "depth": 12}},
        {"op": "copyAndTransformSolid", "args": {"solid": 3, "matrix": [1,0,0,5, 0,1,0,5, 0,0,1,-2, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 3, "matrix": [1,0,0,22, 0,1,0,22, 0,0,1,-2, 0,0,0,1]}},
        {"op": "compoundCut", "args": {"target": 0, "tools": [2, 4, 5]}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    assert_no_crash(
        &parsed,
        6,
        "compoundCut with mixed inserts should not crash",
    );
}

/// 6 dividers via sequential cuts (3 in X + 3 in Y).
///
/// Solids: 0=target, 1=x-template, 2-4=x-dividers, 5=y-template,
///         6-8=y-dividers, 9-14=cut results
#[test]
fn sequential_cut_many_dividers() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 40, "height": 40, "depth": 10}},
        {"op": "makeBox", "args": {"width": 1, "height": 42, "depth": 12}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,9.5, 0,1,0,-1, 0,0,1,-1, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,19.5, 0,1,0,-1, 0,0,1,-1, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,29.5, 0,1,0,-1, 0,0,1,-1, 0,0,0,1]}},
        {"op": "makeBox", "args": {"width": 42, "height": 1, "depth": 12}},
        {"op": "copyAndTransformSolid", "args": {"solid": 5, "matrix": [1,0,0,-1, 0,1,0,9.5, 0,0,1,-1, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 5, "matrix": [1,0,0,-1, 0,1,0,19.5, 0,0,1,-1, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 5, "matrix": [1,0,0,-1, 0,1,0,29.5, 0,0,1,-1, 0,0,0,1]}},
        {"op": "cut", "args": {"solidA": 0, "solidB": 2}},
        {"op": "cut", "args": {"solidA": 9, "solidB": 3}},
        {"op": "cut", "args": {"solidA": 10, "solidB": 4}},
        {"op": "cut", "args": {"solidA": 11, "solidB": 6}},
        {"op": "cut", "args": {"solidA": 12, "solidB": 7}},
        {"op": "cut", "args": {"solidA": 13, "solidB": 8}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    for i in 9..=14 {
        assert_no_crash(&parsed, i, &format!("sequential cut step {i}"));
    }
}

/// Compound cut with 5 cylinders (slot pattern).
///
/// Solids: 0=box, 1=cylinder, 2-6=copies, 7=compoundCut result
#[test]
fn compound_cut_slotted() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 30, "height": 10, "depth": 5}},
        {"op": "makeCylinder", "args": {"radius": 1.0, "height": 8.0}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,5, 0,1,0,5, 0,0,1,-1.5, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,10, 0,1,0,5, 0,0,1,-1.5, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,15, 0,1,0,5, 0,0,1,-1.5, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,20, 0,1,0,5, 0,0,1,-1.5, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,25, 0,1,0,5, 0,0,1,-1.5, 0,0,0,1]}},
        {"op": "compoundCut", "args": {"target": 0, "tools": [2, 3, 4, 5, 6]}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    assert_no_crash(&parsed, 7, "compoundCut with 5 slot cylinders");
}

/// Honeycomb pattern: many cylinder tools.
///
/// Solids: 0=box, 1=cylinder, 2-10=copies, 11=compoundCut result
#[test]
fn compound_cut_honeycomb() {
    let mut k = BrepKernel::new();
    let mut ops: Vec<String> = vec![
        r#"{"op": "makeBox", "args": {"width": 30, "height": 30, "depth": 3}}"#.to_string(),
        r#"{"op": "makeCylinder", "args": {"radius": 2.0, "height": 6.0}}"#.to_string(),
    ];
    let mut tool_handles = Vec::new();
    let mut handle = 2u32;
    for row in 0..3 {
        for col in 0..3 {
            let x = 5.0 + col as f64 * 10.0;
            let y = 5.0 + row as f64 * 10.0;
            let mat = translate_matrix(x, y, -1.5);
            ops.push(format!(
                r#"{{"op": "copyAndTransformSolid", "args": {{"solid": 1, "matrix": {mat}}}}}"#
            ));
            tool_handles.push(handle);
            handle += 1;
        }
    }
    let tools_json = serde_json::to_string(&tool_handles).unwrap();
    ops.push(format!(
        r#"{{"op": "compoundCut", "args": {{"target": 0, "tools": {tools_json}}}}}"#
    ));
    let json = format!("[{}]", ops.join(","));

    let result = k.execute_batch(&json);
    let parsed = parse_batch(&result);
    let last_idx = ops.len() - 1;
    assert_no_crash(&parsed, last_idx, "compoundCut with 9 honeycomb cylinders");
}

/// Fillet first, then compound cut — the fillet introduces torus faces.
///
/// Solids: 0=box, 1=filleted, 2=cylinder, 3-5=copies, 6=compoundCut result
#[test]
fn fillet_then_compound_cut() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 20, "height": 20, "depth": 10}},
        {"op": "fillet", "args": {"solid": 0, "radius": 1.0}},
        {"op": "makeCylinder", "args": {"radius": 2.0, "height": 14.0}},
        {"op": "copyAndTransformSolid", "args": {"solid": 2, "matrix": [1,0,0,5, 0,1,0,5, 0,0,1,-2, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 2, "matrix": [1,0,0,15, 0,1,0,5, 0,0,1,-2, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 2, "matrix": [1,0,0,10, 0,1,0,15, 0,0,1,-2, 0,0,0,1]}},
        {"op": "compoundCut", "args": {"target": 1, "tools": [3, 4, 5]}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    assert_no_crash(&parsed, 6, "fillet + compoundCut");
}

/// Sequential 5-cylinder cuts (not compound — one at a time).
///
/// Solids: 0=box, 1=cylinder, 2-6=copies, 7-11=cut results
#[test]
fn sequential_cut_5_cylinders() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 20, "height": 20, "depth": 8}},
        {"op": "makeCylinder", "args": {"radius": 1.5, "height": 12.0}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,4, 0,1,0,10, 0,0,1,-2, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,8, 0,1,0,10, 0,0,1,-2, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,12, 0,1,0,10, 0,0,1,-2, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,16, 0,1,0,10, 0,0,1,-2, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,10, 0,1,0,10, 0,0,1,-2, 0,0,0,1]}},
        {"op": "cut", "args": {"solidA": 0, "solidB": 2}},
        {"op": "cut", "args": {"solidA": 7, "solidB": 3}},
        {"op": "cut", "args": {"solidA": 8, "solidB": 4}},
        {"op": "cut", "args": {"solidA": 9, "solidB": 5}},
        {"op": "cut", "args": {"solidA": 10, "solidB": 6}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    for i in 7..=11 {
        assert_no_crash(&parsed, i, &format!("sequential cut step {i}"));
    }
}

/// Fuse two boxes, then compound-cut the result.
///
/// Solids: 0=box1, 1=box2, 2=box2-copy (translated), 3=fused,
///         4=cylinder, 5-6=cyl-copies, 7=compoundCut result
#[test]
fn compound_cut_after_fuse() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 10, "height": 10, "depth": 10}},
        {"op": "makeBox", "args": {"width": 10, "height": 10, "depth": 10}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,5, 0,1,0,0, 0,0,1,0, 0,0,0,1]}},
        {"op": "fuse", "args": {"solidA": 0, "solidB": 2}},
        {"op": "makeCylinder", "args": {"radius": 1.5, "height": 14.0}},
        {"op": "copyAndTransformSolid", "args": {"solid": 4, "matrix": [1,0,0,5, 0,1,0,5, 0,0,1,-2, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 4, "matrix": [1,0,0,10, 0,1,0,5, 0,0,1,-2, 0,0,0,1]}},
        {"op": "compoundCut", "args": {"target": 3, "tools": [5, 6]}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    assert_no_crash(&parsed, 7, "fuse + compoundCut");
}

/// Full pipeline: box + fuse + fillet + compoundCut (all in one batch).
///
/// Solids: 0=box1, 1=box2, 2=box2-copy (translated), 3=fused,
///         4=filleted, 5=cylinder, 6-8=cyl-copies, 9=compoundCut result
#[test]
fn batch_fuse_cut_fillet_compound() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 10, "height": 10, "depth": 10}},
        {"op": "makeBox", "args": {"width": 10, "height": 10, "depth": 5}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,0, 0,1,0,0, 0,0,1,10, 0,0,0,1]}},
        {"op": "fuse", "args": {"solidA": 0, "solidB": 2}},
        {"op": "fillet", "args": {"solid": 3, "radius": 0.5}},
        {"op": "makeCylinder", "args": {"radius": 1.0, "height": 20.0}},
        {"op": "copyAndTransformSolid", "args": {"solid": 5, "matrix": [1,0,0,5, 0,1,0,5, 0,0,1,-2, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 5, "matrix": [1,0,0,3, 0,1,0,3, 0,0,1,-2, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 5, "matrix": [1,0,0,7, 0,1,0,7, 0,0,1,-2, 0,0,0,1]}},
        {"op": "compoundCut", "args": {"target": 4, "tools": [6, 7, 8]}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    // Fillet may fail (torus faces), but the pipeline should not crash.
    assert_no_crash(&parsed, 9, "full pipeline (fuse+fillet+compoundCut)");
}

/// Compound cut with several tools, then measure.
///
/// Solids: 0=box, 1=cylinder, 2-5=copies, 6=compoundCut result
#[test]
fn compound_cut_then_measure() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 20, "height": 20, "depth": 5}},
        {"op": "makeCylinder", "args": {"radius": 2.0, "height": 8.0}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,5, 0,1,0,5, 0,0,1,-1.5, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,15, 0,1,0,5, 0,0,1,-1.5, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,5, 0,1,0,15, 0,0,1,-1.5, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,15, 0,1,0,15, 0,0,1,-1.5, 0,0,0,1]}},
        {"op": "compoundCut", "args": {"target": 0, "tools": [2, 3, 4, 5]}},
        {"op": "volume", "args": {"solid": 6}},
        {"op": "boundingBox", "args": {"solid": 6}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    // If compoundCut succeeded, volume and bbox should also succeed.
    if parsed[6].get("ok").is_some() {
        assert_ok(&parsed, 7);
        assert_ok(&parsed, 8);
        let vol = ok_f64(&parsed, 7);
        assert!(
            vol > 0.0 && vol < 2000.0,
            "volume should be positive and less than original (2000): {vol}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Category B: Volume / BBox Regression Reproducers (#260)
//
// These test geometric accuracy after boolean operations.
// ═══════════════════════════════════════════════════════════════════════

/// Sequential cylinder cuts: volume should be analytically predictable.
///
/// Solids: 0=box, 1=cylinder, 2-4=copies, 5-7=cut results
#[test]
fn sequential_booleans_volume_accuracy() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 10, "height": 10, "depth": 10}},
        {"op": "volume", "args": {"solid": 0}},
        {"op": "makeCylinder", "args": {"radius": 1.0, "height": 14.0}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,2, 0,1,0,5, 0,0,1,-2, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,5, 0,1,0,5, 0,0,1,-2, 0,0,0,1]}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,8, 0,1,0,5, 0,0,1,-2, 0,0,0,1]}},
        {"op": "cut", "args": {"solidA": 0, "solidB": 2}},
        {"op": "cut", "args": {"solidA": 5, "solidB": 3}},
        {"op": "cut", "args": {"solidA": 6, "solidB": 4}},
        {"op": "volume", "args": {"solid": 7}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    for i in 6..=8 {
        assert_ok(&parsed, i);
    }
    let original_vol = ok_f64(&parsed, 1);
    let result_vol = ok_f64(&parsed, 9);
    // Each cylinder removes π * r² * h_inside = π * 1 * 10 ≈ 31.4 from the box.
    let expected = original_vol - 3.0 * std::f64::consts::PI * 1.0 * 10.0;
    let rel_error = ((result_vol - expected) / expected).abs();
    assert!(
        rel_error < 0.05,
        "volume after 3 cylinder cuts: got {result_vol:.1}, expected {expected:.1}, \
         error {:.1}% (issue #260)",
        rel_error * 100.0
    );
}

/// Fillet should not change the bounding box of a box.
///
/// Solids: 0=box, 1=filleted
#[test]
fn fillet_box_bbox_unchanged() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 10, "height": 10, "depth": 10}},
        {"op": "boundingBox", "args": {"solid": 0}},
        {"op": "fillet", "args": {"solid": 0, "radius": 1.0}},
        {"op": "boundingBox", "args": {"solid": 1}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    assert_ok(&parsed, 1); // boundingBox on unfilleted box
    let bbox_before = ok_bbox(&parsed, 1);

    // Fillet might fail — only check bbox if it succeeded.
    if parsed[2].get("ok").is_some() {
        let bbox_after = ok_bbox(&parsed, 3);
        let tol = 0.01;
        for i in 0..6 {
            assert!(
                (bbox_before[i] - bbox_after[i]).abs() < tol,
                "bbox[{i}] shifted after fillet: {:.3} → {:.3} (issue #260)",
                bbox_before[i],
                bbox_after[i]
            );
        }
    }
}

/// Cylinder cut from center: outer bounding box should not change.
///
/// Solids: 0=box, 1=cylinder, 2=cyl-copy, 3=cut result
#[test]
fn compound_cut_bbox_accurate() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 20, "height": 20, "depth": 10}},
        {"op": "boundingBox", "args": {"solid": 0}},
        {"op": "makeCylinder", "args": {"radius": 2.0, "height": 14.0}},
        {"op": "copyAndTransformSolid", "args": {"solid": 1, "matrix": [1,0,0,10, 0,1,0,10, 0,0,1,-2, 0,0,0,1]}},
        {"op": "cut", "args": {"solidA": 0, "solidB": 2}},
        {"op": "boundingBox", "args": {"solid": 3}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    let bbox_before = ok_bbox(&parsed, 1);

    if parsed[4].get("ok").is_some() {
        let bbox_after = ok_bbox(&parsed, 5);
        let tol = 0.1;
        for i in 0..6 {
            assert!(
                (bbox_before[i] - bbox_after[i]).abs() < tol,
                "bbox[{i}] shifted after cylinder cut: {:.3} → {:.3} (issue #260)",
                bbox_before[i],
                bbox_after[i]
            );
        }
    }
}

/// Box volume sanity check.
#[test]
fn box_volume_sanity() {
    let mut k = BrepKernel::new();
    let result = k.execute_batch(
        r#"[
        {"op": "makeBox", "args": {"width": 10, "height": 10, "depth": 10}},
        {"op": "volume", "args": {"solid": 0}}
    ]"#,
    );
    let parsed = parse_batch(&result);
    assert_ok(&parsed, 0);
    let vol = ok_f64(&parsed, 1);
    let expected = 1000.0;
    let rel_error = ((vol - expected) / expected).abs();
    assert!(
        rel_error < 0.01,
        "box volume should be 1000: got {vol:.1}, error {:.1}%",
        rel_error * 100.0
    );
}
