//! Each batch arm that takes a number checks it as its binding does: a size
//! past the limit, a number JSON wrote as `null` (NaN and the infinities),
//! and a count past its limit are each refused, not built.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::vec::{Point3, Vec3};

use crate::kernel::BrepKernel;

fn run(k: &mut BrepKernel, op: &str) -> serde_json::Value {
    let out: serde_json::Value =
        serde_json::from_str(&k.execute_batch(&format!("[{op}]"))).unwrap();
    out[0].clone()
}

fn made(k: &mut BrepKernel, op: &str) -> u32 {
    let out = run(k, op);
    assert!(out["ok"].is_u64(), "{op}: {out}");
    u32::try_from(out["ok"].as_u64().unwrap()).unwrap()
}

/// A 2-cube, one of its faces and edges, and a hexagonal wire.
fn fixture() -> (BrepKernel, u32, u32, u32, u32) {
    let mut k = BrepKernel::new();
    let solid = made(
        &mut k,
        r#"{"op": "makeBox", "args": {"width": 2, "height": 2, "depth": 2}}"#,
    );
    let face = k.get_solid_faces(solid).unwrap()[0];
    let edge = k.get_solid_edges(solid).unwrap()[0];
    let wire = made(
        &mut k,
        r#"{"op": "makeRegularPolygonWire", "args": {"radius": 1, "nSides": 6}}"#,
    );
    (k, solid, face, edge, wire)
}

fn refused(cases: &[(&str, String)]) {
    for (what, op) in cases {
        let (mut k, ..) = fixture();
        let out = run(&mut k, op);
        let message = out["error"].as_str().unwrap_or_default();
        let checks = [
            "at most",
            "must be positive",
            "must not be below zero",
            "is null",
            "must be a count",
            "is not a handle",
            "missing or invalid",
            "must be in",
            "must be finite",
        ];
        assert!(
            checks.iter().any(|check| message.contains(check)),
            "{what} was not refused by an input check: {out}"
        );
    }
}

#[test]
fn a_size_past_the_limit_is_refused_by_every_arm() {
    let (_, s, f, e, w) = fixture();
    let big = "1e301";
    let scale = format!("[{big}, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]");
    let view = |origin: &str, deflection: &str| {
        format!(
            r#"{{"op": "projectEdges", "args": {{"solid": {s}, "originX": {origin}, "originY": 0, "originZ": 0, "dirX": 0, "dirY": 0, "dirZ": 1, "xAxisX": 1, "xAxisY": 0, "xAxisZ": 0, "deflection": {deflection}}}}}"#
        )
    };
    let cases: Vec<(&str, String)> = vec![
        (
            "makeBox width",
            format!(r#"{{"op": "makeBox", "args": {{"width": {big}, "height": 1, "depth": 1}}}}"#),
        ),
        (
            "makeCylinder radius",
            format!(r#"{{"op": "makeCylinder", "args": {{"radius": {big}, "height": 1}}}}"#),
        ),
        (
            "makeSphere radius",
            format!(r#"{{"op": "makeSphere", "args": {{"radius": {big}}}}}"#),
        ),
        (
            "makeCone height",
            format!(
                r#"{{"op": "makeCone", "args": {{"bottomRadius": 1, "topRadius": 0, "height": {big}}}}}"#
            ),
        ),
        (
            "makeTorus majorRadius",
            format!(r#"{{"op": "makeTorus", "args": {{"majorRadius": {big}, "minorRadius": 1}}}}"#),
        ),
        (
            "makeEllipsoid rx",
            format!(r#"{{"op": "makeEllipsoid", "args": {{"rx": {big}, "ry": 1, "rz": 1}}}}"#),
        ),
        (
            "makeCircleEdge centre",
            format!(r#"{{"op": "makeCircleEdge", "args": {{"cx": {big}, "radius": 1}}}}"#),
        ),
        (
            "makeCircle radius",
            format!(r#"{{"op": "makeCircle", "args": {{"radius": {big}, "segments": 8}}}}"#),
        ),
        (
            "makeTangentArc3d start",
            format!(
                r#"{{"op": "makeTangentArc3d", "args": {{"startX": {big}, "startY": 0, "startZ": 0, "tangentX": 1, "tangentY": 0, "tangentZ": 0, "endX": 1, "endY": 1, "endZ": 0}}}}"#
            ),
        ),
        (
            "extrude direction",
            format!(r#"{{"op": "extrude", "args": {{"face": {f}, "dz": {big}}}}}"#),
        ),
        (
            "extrude distance",
            format!(r#"{{"op": "extrude", "args": {{"face": {f}, "distance": {big}}}}}"#),
        ),
        (
            "extrude offset",
            format!(
                r#"{{"op": "extrude", "args": {{"face": {f}, "dz": 1e200, "distance": 1e200}}}}"#
            ),
        ),
        (
            "extrudeWire distance",
            format!(r#"{{"op": "extrudeWire", "args": {{"wire": {w}, "distance": {big}}}}}"#),
        ),
        (
            "revolve origin",
            format!(
                r#"{{"op": "revolve", "args": {{"face": {f}, "angle": 90, "originX": {big}}}}}"#
            ),
        ),
        (
            "revolveWire origin",
            format!(
                r#"{{"op": "revolveWire", "args": {{"wire": {w}, "angle": 90, "originX": {big}}}}}"#
            ),
        ),
        ("projectEdges origin", view(big, "0.1")),
        ("projectEdges deflection", view("0", big)),
        (
            "chamfer distance",
            format!(
                r#"{{"op": "chamfer", "args": {{"solid": {s}, "edges": [{e}], "distance": {big}}}}}"#
            ),
        ),
        (
            "fillet radius",
            format!(
                r#"{{"op": "fillet", "args": {{"solid": {s}, "edges": [{e}], "radius": {big}}}}}"#
            ),
        ),
        (
            "filletVariable radius",
            format!(
                r#"{{"op": "filletVariable", "args": {{"solid": {s}, "specs": [{{"edge": {e}, "radius": {big}}}]}}}}"#
            ),
        ),
        (
            "filletV2 radius",
            format!(
                r#"{{"op": "filletV2", "args": {{"solid": {s}, "edges": [{e}], "radius": {big}}}}}"#
            ),
        ),
        (
            "filletV2Corners radius",
            format!(
                r#"{{"op": "filletV2Corners", "args": {{"solid": {s}, "edges": [{e}], "radius": {big}, "corners": []}}}}"#
            ),
        ),
        (
            "chamferV2 d1",
            format!(
                r#"{{"op": "chamferV2", "args": {{"solid": {s}, "edges": [{e}], "d1": {big}, "d2": 0.1}}}}"#
            ),
        ),
        (
            "chamferDistanceAngle distance",
            format!(
                r#"{{"op": "chamferDistanceAngle", "args": {{"solid": {s}, "edges": [{e}], "distance": {big}, "angle": 0.3}}}}"#
            ),
        ),
        (
            "shell thickness",
            format!(
                r#"{{"op": "shell", "args": {{"solid": {s}, "faces": [{f}], "thickness": {big}}}}}"#
            ),
        ),
        (
            "mirror point",
            format!(r#"{{"op": "mirror", "args": {{"solid": {s}, "px": {big}}}}}"#),
        ),
        (
            "circularPattern axis",
            format!(
                r#"{{"op": "circularPattern", "args": {{"solid": {s}, "ax": {big}, "count": 2}}}}"#
            ),
        ),
        (
            "gridPattern spacing",
            format!(
                r#"{{"op": "gridPattern", "args": {{"solid": {s}, "spacingX": {big}, "spacingY": 1, "countX": 2, "countY": 2}}}}"#
            ),
        ),
        (
            "linearPattern spacing",
            format!(
                r#"{{"op": "linearPattern", "args": {{"solid": {s}, "spacing": {big}, "count": 2}}}}"#
            ),
        ),
        (
            "linearPattern reach",
            format!(
                r#"{{"op": "linearPattern", "args": {{"solid": {s}, "spacing": 1e299, "count": 20}}}}"#
            ),
        ),
        (
            "transform",
            format!(r#"{{"op": "transform", "args": {{"solid": {s}, "matrix": {scale}}}}}"#),
        ),
        (
            "copyAndTransformSolid",
            format!(
                r#"{{"op": "copyAndTransformSolid", "args": {{"solid": {s}, "matrix": {scale}}}}}"#
            ),
        ),
        (
            "transformWire",
            format!(r#"{{"op": "transformWire", "args": {{"wire": {w}, "matrix": {scale}}}}}"#),
        ),
        (
            "transformFace",
            format!(r#"{{"op": "transformFace", "args": {{"face": {f}, "matrix": {scale}}}}}"#),
        ),
        (
            "volume deflection",
            format!(r#"{{"op": "volume", "args": {{"solid": {s}, "deflection": {big}}}}}"#),
        ),
        (
            "surfaceArea deflection",
            format!(r#"{{"op": "surfaceArea", "args": {{"solid": {s}, "deflection": {big}}}}}"#),
        ),
        (
            "centerOfMass deflection",
            format!(r#"{{"op": "centerOfMass", "args": {{"solid": {s}, "deflection": {big}}}}}"#),
        ),
        (
            "classifyPoint point",
            format!(
                r#"{{"op": "classifyPoint", "args": {{"solid": {s}, "x": {big}, "y": 0, "z": 0}}}}"#
            ),
        ),
        (
            "section point",
            format!(r#"{{"op": "section", "args": {{"solid": {s}, "pz": {big}}}}}"#),
        ),
        (
            "split point",
            format!(r#"{{"op": "split", "args": {{"solid": {s}, "pz": {big}}}}}"#),
        ),
        (
            "draft neutral point",
            format!(
                r#"{{"op": "draft", "args": {{"solid": {s}, "faces": [{f}], "angle": 0.1, "neutralX": {big}}}}}"#
            ),
        ),
        (
            "offsetSolid distance",
            format!(r#"{{"op": "offsetSolid", "args": {{"solid": {s}, "distance": {big}}}}}"#),
        ),
        (
            "offsetSolidV2 distance",
            format!(r#"{{"op": "offsetSolidV2", "args": {{"solid": {s}, "distance": {big}}}}}"#),
        ),
        (
            "thicken thickness",
            format!(r#"{{"op": "thicken", "args": {{"face": {f}, "thickness": {big}}}}}"#),
        ),
        (
            "offsetFace distance",
            format!(r#"{{"op": "offsetFace", "args": {{"face": {f}, "distance": {big}}}}}"#),
        ),
        (
            "offsetWire distance",
            format!(r#"{{"op": "offsetWire", "args": {{"face": {f}, "distance": {big}}}}}"#),
        ),
        (
            "offsetWireWithJoinType distance",
            format!(
                r#"{{"op": "offsetWireWithJoinType", "args": {{"face": {f}, "distance": {big}, "joinType": "arc"}}}}"#
            ),
        ),
        (
            "offsetWire2DWithJoin distance",
            format!(
                r#"{{"op": "offsetWire2DWithJoin", "args": {{"wire": {w}, "distance": {big}, "joinType": "arc"}}}}"#
            ),
        ),
        (
            "sewFaces tolerance",
            format!(r#"{{"op": "sewFaces", "args": {{"faces": [{f}], "tolerance": {big}}}}}"#),
        ),
        (
            "convertToElementary tolerance",
            format!(
                r#"{{"op": "convertToElementary", "args": {{"solid": {s}, "tolerance": {big}}}}}"#
            ),
        ),
        (
            "healSolid tolerance",
            format!(r#"{{"op": "healSolid", "args": {{"solid": {s}, "tolerance": {big}}}}}"#),
        ),
        (
            "repairSolid tolerance",
            format!(r#"{{"op": "repairSolid", "args": {{"solid": {s}, "tolerance": {big}}}}}"#),
        ),
        (
            "fairSurface tolerance",
            format!(r#"{{"op": "fairSurface", "args": {{"face": {f}, "tolerance": {big}}}}}"#),
        ),
    ];
    refused(&cases);
}

#[test]
fn a_size_of_zero_or_less_is_refused_where_a_size_is_asked_for() {
    let (_, s, f, e, _) = fixture();
    refused(&[
        (
            "fillet radius 0",
            format!(r#"{{"op": "fillet", "args": {{"solid": {s}, "edges": [{e}], "radius": 0}}}}"#),
        ),
        (
            "chamfer distance -1",
            format!(
                r#"{{"op": "chamfer", "args": {{"solid": {s}, "edges": [{e}], "distance": -1}}}}"#
            ),
        ),
        (
            "chamferV2 d2 0",
            format!(
                r#"{{"op": "chamferV2", "args": {{"solid": {s}, "edges": [{e}], "d1": 0.1, "d2": 0}}}}"#
            ),
        ),
        (
            "filletVariable radius -1",
            format!(
                r#"{{"op": "filletVariable", "args": {{"solid": {s}, "specs": [{{"edge": {e}, "radius": -1}}]}}}}"#
            ),
        ),
        (
            "shell thickness 0",
            format!(
                r#"{{"op": "shell", "args": {{"solid": {s}, "faces": [{f}], "thickness": 0}}}}"#
            ),
        ),
        (
            "volume deflection 0",
            format!(r#"{{"op": "volume", "args": {{"solid": {s}, "deflection": 0}}}}"#),
        ),
        (
            "projectEdges deflection -1",
            format!(
                r#"{{"op": "projectEdges", "args": {{"solid": {s}, "originX": 0, "originY": 0, "originZ": 0, "dirX": 0, "dirY": 0, "dirZ": 1, "xAxisX": 1, "xAxisY": 0, "xAxisZ": 0, "deflection": -1}}}}"#
            ),
        ),
        (
            "linearPattern spacing 0",
            format!(
                r#"{{"op": "linearPattern", "args": {{"solid": {s}, "spacing": 0, "count": 2}}}}"#
            ),
        ),
        (
            "sewFaces tolerance -1",
            format!(r#"{{"op": "sewFaces", "args": {{"faces": [{f}], "tolerance": -1}}}}"#),
        ),
        (
            "revolve angle 0",
            format!(r#"{{"op": "revolve", "args": {{"face": {f}, "angle": 0}}}}"#),
        ),
    ]);
}

#[test]
fn a_count_past_its_limit_is_refused_by_every_arm() {
    let (_, s, f, e, _) = fixture();
    let past_u32 = "4294967296";
    refused(&[
        (
            "makeSphere segments",
            r#"{"op": "makeSphere", "args": {"radius": 1, "segments": 70000}}"#.to_string(),
        ),
        (
            "makeSphere segments past u32",
            format!(r#"{{"op": "makeSphere", "args": {{"radius": 1, "segments": {past_u32}}}}}"#),
        ),
        (
            "makeTorus segments past u32",
            format!(
                r#"{{"op": "makeTorus", "args": {{"majorRadius": 2, "minorRadius": 1, "segments": {past_u32}}}}}"#
            ),
        ),
        (
            "makeCircle segments",
            r#"{"op": "makeCircle", "args": {"radius": 1, "segments": 70000}}"#.to_string(),
        ),
        (
            "sweepWithOptions segments past u32",
            format!(
                r#"{{"op": "sweepWithOptions", "args": {{"profile": {f}, "pathEdge": {e}, "segments": {past_u32}}}}}"#
            ),
        ),
        (
            "offsetFace samples",
            format!(
                r#"{{"op": "offsetFace", "args": {{"face": {f}, "distance": 0.1, "samples": 300}}}}"#
            ),
        ),
        (
            "offsetFace samples past u32",
            format!(
                r#"{{"op": "offsetFace", "args": {{"face": {f}, "distance": 0.1, "samples": {past_u32}}}}}"#
            ),
        ),
        (
            "circularPattern count",
            format!(r#"{{"op": "circularPattern", "args": {{"solid": {s}, "count": 70000}}}}"#),
        ),
        (
            "gridPattern counts",
            format!(
                r#"{{"op": "gridPattern", "args": {{"solid": {s}, "spacingX": 3, "spacingY": 3, "countX": 300, "countY": 300}}}}"#
            ),
        ),
        (
            "linearPattern count past u32",
            format!(
                r#"{{"op": "linearPattern", "args": {{"solid": {s}, "spacing": 3, "count": {past_u32}}}}}"#
            ),
        ),
        (
            "a handle past u32",
            format!(r#"{{"op": "fuseAll", "args": {{"solids": [{s}, {past_u32}]}}}}"#),
        ),
        (
            "an edge list with a bad entry",
            format!(
                r#"{{"op": "fillet", "args": {{"solid": {s}, "edges": [{e}, -1], "radius": 0.1}}}}"#
            ),
        ),
    ]);
}

/// JSON writes NaN and the infinities as `null`: a `null` where a number is
/// asked for is refused, not read as the default.
#[test]
fn a_number_written_as_null_is_refused() {
    let (_, s, f, e, _) = fixture();
    refused(&[
        (
            "mirror px",
            format!(r#"{{"op": "mirror", "args": {{"solid": {s}, "px": null}}}}"#),
        ),
        (
            "extrude dx",
            format!(r#"{{"op": "extrude", "args": {{"face": {f}, "dx": null}}}}"#),
        ),
        (
            "volume deflection",
            format!(r#"{{"op": "volume", "args": {{"solid": {s}, "deflection": null}}}}"#),
        ),
        (
            "makeSphere segments",
            r#"{"op": "makeSphere", "args": {"radius": 1, "segments": null}}"#.to_string(),
        ),
        (
            "draft neutralX",
            format!(
                r#"{{"op": "draft", "args": {{"solid": {s}, "faces": [{f}], "angle": 0.1, "neutralX": null}}}}"#
            ),
        ),
        (
            "classifyPoint x",
            format!(
                r#"{{"op": "classifyPoint", "args": {{"solid": {s}, "x": null, "y": 0, "z": 0}}}}"#
            ),
        ),
        (
            "fillet radius",
            format!(
                r#"{{"op": "fillet", "args": {{"solid": {s}, "edges": [{e}], "radius": null}}}}"#
            ),
        ),
    ]);
}

/// Absent, an optional number takes its default, so these still build.
#[test]
fn an_absent_number_takes_its_default() {
    let (mut k, s, ..) = fixture();
    for op in [
        format!(r#"{{"op": "volume", "args": {{"solid": {s}}}}}"#),
        format!(r#"{{"op": "mirror", "args": {{"solid": {s}}}}}"#),
        r#"{"op": "makeSphere", "args": {"radius": 1}}"#.to_string(),
    ] {
        let out = run(&mut k, &op);
        assert!(out["ok"].is_number(), "{op}: {out}");
    }
}

/// Each move's input is within the limit, but the moved shape is not: a
/// scale of 1e100 on a shape 2e200 across is refused, and the shape is
/// left as it was.
#[test]
fn a_move_past_the_limit_is_refused_though_each_input_is_within_it() {
    let (mut k, s, ..) = fixture();
    let scale = "[1e100, 0, 0, 0, 0, 1e100, 0, 0, 0, 0, 1e100, 0, 0, 0, 0, 1]";
    let op = format!(r#"{{"op": "transform", "args": {{"solid": {s}, "matrix": {scale}}}}}"#);
    for _ in 0..2 {
        let moved = run(&mut k, &op);
        assert!(moved["ok"].is_number(), "{moved}");
    }
    let refused = run(&mut k, &op);
    assert!(refused["error"].is_string(), "{refused}");
    let bbox = k.bounding_box(s).unwrap();
    assert!((bbox[3] - 2e200).abs() < 1e186, "{bbox:?}");
}

/// The shared bodies refuse NaN, which reaches them from the JS bindings.
#[test]
fn the_shared_bodies_refuse_nan() {
    let (mut k, s, f, e, w) = fixture();
    let nan = f64::NAN;
    assert!(k.extrude_impl(f, Vec3::new(nan, 0.0, 1.0), 1.0).is_err());
    assert!(
        k.extrude_wire_impl(w, Vec3::new(0.0, 0.0, 1.0), nan)
            .is_err()
    );
    assert!(
        k.revolve_impl(
            f,
            Point3::new(nan, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            90.0
        )
        .is_err()
    );
    assert!(
        k.mirror_impl(s, Point3::new(0.0, nan, 0.0), Vec3::new(1.0, 0.0, 0.0))
            .is_err()
    );
    assert!(
        k.section_impl(s, Point3::new(0.0, 0.0, nan), Vec3::new(0.0, 0.0, 1.0))
            .is_err()
    );
    assert!(
        k.split_impl(s, Point3::new(0.0, 0.0, 1.0), Vec3::new(0.0, nan, 1.0))
            .is_err()
    );
    assert!(
        k.draft_impl(
            s,
            &[f],
            Vec3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            nan
        )
        .is_err()
    );
    assert!(k.volume_impl(s, nan).is_err());
    assert!(
        k.classify_point_impl(s, Point3::new(nan, 0.0, 0.0), 1e-7)
            .is_err()
    );
    assert!(k.fillet_impl(s, &[e], nan).is_err());
    assert!(k.chamfer_impl(s, &[e], nan).is_err());
    assert!(k.shell_impl(s, nan, &[f]).is_err());
    assert!(
        k.linear_pattern_impl(s, Vec3::new(1.0, 0.0, 0.0), nan, 2)
            .is_err()
    );
    assert!(
        k.circular_pattern_impl(s, Vec3::new(0.0, 0.0, nan), 2)
            .is_err()
    );
    assert!(k.offset_wire_impl(f, nan, None).is_err());
    assert!(k.sew_faces_impl(&[f], nan).is_err());
    assert!(
        k.project_edges_impl(
            s,
            (
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, nan),
                Vec3::new(1.0, 0.0, 0.0)
            ),
            true,
            0.1,
        )
        .is_err()
    );
    let mut nan_matrix = vec![0.0; 16];
    nan_matrix[0] = nan;
    assert!(k.transform_solid_impl(s, &nan_matrix).is_err());
}
