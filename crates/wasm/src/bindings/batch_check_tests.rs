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
            "is not a number",
            "unknown corner kind",
            "must be two numbers",
            "law must be",
            "must be an array",
            // A move is refused by the engine, before it moves anything: an
            // image past the load checks, or one it cannot build.
            "would not pass the load checks",
            "cannot normalize",
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

/// A spec's field given as `null`, as JSON writes NaN and the infinities,
/// is refused, not read as absent: `{"radius": null}` would take a radius
/// of 1, and a null start would take its alias or the default.
#[test]
fn a_fillet_spec_field_written_as_null_is_refused() {
    let (_, s, _, e, _) = fixture();
    refused(&[
        (
            "radius",
            format!(
                r#"{{"op": "filletVariable", "args": {{"solid": {s}, "specs": [{{"edge": {e}, "radius": null}}]}}}}"#
            ),
        ),
        (
            "start",
            format!(
                r#"{{"op": "filletVariable", "args": {{"solid": {s}, "specs": [{{"edge": {e}, "law": "linear", "start": null, "end": 0.2}}]}}}}"#
            ),
        ),
        (
            "end",
            format!(
                r#"{{"op": "filletVariable", "args": {{"solid": {s}, "specs": [{{"edge": {e}, "law": "linear", "start": 0.1, "endRadius": null}}]}}}}"#
            ),
        ),
        (
            "law",
            format!(
                r#"{{"op": "filletVariable", "args": {{"solid": {s}, "specs": [{{"edge": {e}, "law": null, "radius": 0.1}}]}}}}"#
            ),
        ),
        (
            "an unknown law",
            format!(
                r#"{{"op": "filletVariable", "args": {{"solid": {s}, "specs": [{{"edge": {e}, "law": "cubic", "radius": 0.1}}]}}}}"#
            ),
        ),
    ]);
}

/// A unit patch of degree one, flat at z = 0 but for one control point at
/// `(1/8, 1/8)` raised to 2e299: a bump no quarter of its parameters meets,
/// inside a boundary that stays at z = 0. Its face, and a solid of it alone.
fn bumped_patch(k: &mut BrepKernel) -> (u32, u32) {
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    let points: Vec<Vec<Point3>> = (0..9_u32)
        .map(|i| {
            (0..9_u32)
                .map(|j| {
                    let z = if i == 1 && j == 1 { 2e299 } else { 0.0 };
                    Point3::new(f64::from(i) / 8.0, f64::from(j) / 8.0, z)
                })
                .collect()
        })
        .collect();
    let mut knots = vec![0.0, 0.0];
    knots.extend((1..8_u32).map(|k| f64::from(k) / 8.0));
    knots.extend([1.0, 1.0]);
    let surface =
        NurbsSurface::new(1, 1, knots.clone(), knots, points, vec![vec![1.0; 9]; 9]).unwrap();
    let face = brepkit_topology::builder::make_nurbs_face(k.topo_mut(), surface, 1e-7).unwrap();
    let shell = k.topo_mut().add_shell(Shell::new(vec![face]).unwrap());
    let solid = k.topo_mut().add_solid(Solid::new(shell, vec![]));
    (
        crate::handles::face_id_to_u32(face),
        crate::handles::solid_id_to_u32(solid),
    )
}

/// The bump of [`bumped_patch`] scaled by 10 in z passes the size limit:
/// a move of the solid or of the face alone is refused, though each box
/// sampled from the surface, or drawn from its boundary, stays at z = 0.
#[test]
fn a_move_is_bounded_by_the_control_points_not_samples_of_the_surface() {
    let scale = "[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 10, 0, 0, 0, 0, 1]";
    let mut k = BrepKernel::new();
    let (face, solid) = bumped_patch(&mut k);
    for op in [
        format!(r#"{{"op": "transform", "args": {{"solid": {solid}, "matrix": {scale}}}}}"#),
        format!(
            r#"{{"op": "copyAndTransformSolid", "args": {{"solid": {solid}, "matrix": {scale}}}}}"#
        ),
        format!(r#"{{"op": "transformFace", "args": {{"face": {face}, "matrix": {scale}}}}}"#),
    ] {
        let out = run(&mut k, &op);
        let message = out["error"].as_str().unwrap_or_default();
        assert!(
            message.contains("would not pass the load checks"),
            "{op}: {out}"
        );
    }
}

/// A fillet corner's kind or setbacks given as `null` is refused, not read
/// as absent, and so is a blend's weights.
#[test]
fn a_nested_option_written_as_null_is_refused() {
    let (k, s, f, e, _) = fixture();
    let v = k.get_solid_vertices(s).unwrap()[0];
    let g = k.get_solid_faces(s).unwrap()[1];
    refused(&[
        (
            "a corner's kind",
            format!(
                r#"{{"op": "filletV2Corners", "args": {{"solid": {s}, "edges": [{e}], "radius": 0.1, "corners": [{{"vertex": {v}, "corner": null}}]}}}}"#
            ),
        ),
        (
            "a corner's setbacks",
            format!(
                r#"{{"op": "filletV2Corners", "args": {{"solid": {s}, "edges": [{e}], "radius": 0.1, "corners": [{{"vertex": {v}, "setbacks": null}}]}}}}"#
            ),
        ),
        (
            "a blend's weights",
            format!(
                r#"{{"op": "blendSurface", "args": {{"chainA": [{e}], "faceA": {f}, "chainB": [{e}], "faceB": {g}, "continuity": "g1", "weights": null}}}}"#
            ),
        ),
    ]);
}

/// A line from the origin to (1e200, 1e200, 0) under a matrix whose first
/// row is (1e200, -1e200, 0, 0) maps its far end to an x of inf - inf: not
/// a number. Each moved point is checked, not a box that drops NaN, and
/// the wire is left as it was.
#[test]
fn a_move_that_makes_a_point_not_a_number_is_refused_before_any_change() {
    use brepkit_topology::wire::{OrientedEdge, Wire};
    let mut k = BrepKernel::new();
    let far = Point3::new(1e200, 1e200, 0.0);
    let edge = brepkit_topology::builder::make_line_edge(
        k.topo_mut(),
        Point3::new(0.0, 0.0, 0.0),
        far,
        1e-7,
    )
    .unwrap();
    let wire = k
        .topo_mut()
        .add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], false).unwrap());
    let matrix = [
        1e200, -1e200, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let handle = crate::handles::wire_id_to_u32(wire);
    let refused = k.transform_wire_impl(handle, &matrix);
    assert!(refused.is_err(), "{refused:?}");
    let end = k.topo.edge(edge).unwrap().end();
    assert_eq!(k.topo.vertex(end).unwrap().point(), far);
}

/// The intersection of two boxes apart is a solid with no faces. Moving,
/// copying, mirroring and patterning it is not a panic: it has no extent
/// to check.
#[test]
fn an_empty_solid_is_moved_without_a_panic() {
    let mut k = BrepKernel::new();
    let a = made(
        &mut k,
        r#"{"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}}"#,
    );
    let b = made(
        &mut k,
        r#"{"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}}"#,
    );
    let shift = "[1, 0, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]";
    let moved = run(
        &mut k,
        &format!(r#"{{"op": "transform", "args": {{"solid": {b}, "matrix": {shift}}}}}"#),
    );
    assert!(moved["ok"].is_number(), "{moved}");
    let empty = made(
        &mut k,
        &format!(r#"{{"op": "intersect", "args": {{"solidA": {a}, "solidB": {b}}}}}"#),
    );
    assert!(k.get_solid_faces(empty).unwrap().is_empty());
    let identity = "[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]";
    for op in [
        format!(r#"{{"op": "transform", "args": {{"solid": {empty}, "matrix": {identity}}}}}"#),
        format!(
            r#"{{"op": "copyAndTransformSolid", "args": {{"solid": {empty}, "matrix": {identity}}}}}"#
        ),
        format!(r#"{{"op": "mirror", "args": {{"solid": {empty}}}}}"#),
        format!(
            r#"{{"op": "linearPattern", "args": {{"solid": {empty}, "spacing": 2, "count": 2}}}}"#
        ),
        format!(
            r#"{{"op": "gridPattern", "args": {{"solid": {empty}, "spacingX": 2, "spacingY": 2, "countX": 2, "countY": 2}}}}"#
        ),
        format!(r#"{{"op": "circularPattern", "args": {{"solid": {empty}, "count": 3}}}}"#),
    ] {
        let out = run(&mut k, &op);
        assert!(out.is_object(), "{op}: {out}");
    }
}

/// A rational arc whose middle weight is 1e10, moved 1e295 along x: each
/// control point stays within the size limit, but its largest weight times
/// its largest coordinate, 1e305, passes the load checks' limit of 1e300,
/// so the moved curve would not load. The move is refused, and the curve is
/// left as it was.
#[test]
fn a_move_that_takes_a_weighted_spline_past_the_load_checks_is_refused() {
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::wire::{OrientedEdge, Wire};
    let mut k = BrepKernel::new();
    let arc = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        ],
        vec![1.0, 1e10, 1.0],
    )
    .unwrap();
    assert!(arc.is_well_formed());
    let edge = brepkit_topology::builder::make_nurbs_edge_from_curve(k.topo_mut(), &arc, 1e-7);
    let wire = k
        .topo_mut()
        .add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], false).unwrap());
    let shift = [
        1.0, 0.0, 0.0, 1e295, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let moved = k.transform_wire_impl(crate::handles::wire_id_to_u32(wire), &shift);
    let EdgeCurve::NurbsCurve(after) = k.topo.edge(edge).unwrap().curve() else {
        unreachable!()
    };
    assert!(
        moved.is_err(),
        "moved, and it loads: {}",
        after.is_well_formed()
    );
    assert!(after.is_well_formed());
}

/// A box 1e299 across, set with its corners from (4e299, 0, 8e299) to
/// (5e299, 1e299, 9e299), turned about z: its far corner is 1.03e300 from
/// the origin but every copy keeps each coordinate within the size limit,
/// so the pattern is built. Set from (8e299, 8e299, 0), four copies a
/// quarter turn apart still fit, but a copy turned by 45 degrees would put
/// its far corner 1.27e300 out along y: eight copies are refused.
#[test]
fn a_circular_pattern_is_bounded_by_its_distance_from_the_axis() {
    let mut k = BrepKernel::new();
    let at = |k: &mut BrepKernel, x: &str, y: &str, z: &str| {
        let s = made(
            k,
            r#"{"op": "makeBox", "args": {"width": 1e299, "height": 1e299, "depth": 1e299}}"#,
        );
        let shift = format!("[1, 0, 0, {x}, 0, 1, 0, {y}, 0, 0, 1, {z}, 0, 0, 0, 1]");
        let moved = run(
            k,
            &format!(r#"{{"op": "transform", "args": {{"solid": {s}, "matrix": {shift}}}}}"#),
        );
        assert!(moved["ok"].is_number(), "{moved}");
        s
    };
    let near = at(&mut k, "4e299", "0", "8e299");
    let built = run(
        &mut k,
        &format!(r#"{{"op": "circularPattern", "args": {{"solid": {near}, "count": 4}}}}"#),
    );
    assert!(built["ok"].is_number(), "{built}");
    let far = at(&mut k, "8e299", "8e299", "0");
    let four = run(
        &mut k,
        &format!(r#"{{"op": "circularPattern", "args": {{"solid": {far}, "count": 4}}}}"#),
    );
    assert!(four["ok"].is_number(), "{four}");
    let solids = k.topo.num_solids();
    let refused = run(
        &mut k,
        &format!(r#"{{"op": "circularPattern", "args": {{"solid": {far}, "count": 8}}}}"#),
    );
    assert_eq!(
        k.topo.num_solids(),
        solids,
        "a refused pattern builds no copy"
    );
    assert!(
        refused["error"]
            .as_str()
            .unwrap_or_default()
            .contains("would not pass the load checks"),
        "{refused}"
    );
}

/// A triangle on the plane x + y + z = 1, moved 6e299 along each axis: each
/// vertex stays within the size limit, but the plane's distance from the
/// origin, which a plane face stores, becomes 1.04e300, which the load
/// checks refuse. The move is refused and the face left as it was; so is a
/// section plane at that distance.
#[test]
fn a_plane_moved_past_the_load_checks_is_refused() {
    let mut k = BrepKernel::new();
    let face = k
        .make_polygon(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        .unwrap();
    let shift = "[1, 0, 0, 6e299, 0, 1, 0, 6e299, 0, 0, 1, 6e299, 0, 0, 0, 1]";
    let moved = run(
        &mut k,
        &format!(r#"{{"op": "transformFace", "args": {{"face": {face}, "matrix": {shift}}}}}"#),
    );
    let id = k.resolve_face(face).unwrap();
    let loads = k.topo.face(id).unwrap().surface().is_well_formed();
    assert!(
        moved["error"].is_string(),
        "moved: {moved}, and it loads: {loads}"
    );
    assert!(loads);
    let (mut k, s, ..) = fixture();
    let cut = run(
        &mut k,
        &format!(
            r#"{{"op": "section", "args": {{"solid": {s}, "px": 6e299, "py": 6e299, "pz": 6e299, "nx": 1, "ny": 1, "nz": 1}}}}"#
        ),
    );
    assert!(
        cut["error"]
            .as_str()
            .unwrap_or_default()
            .contains("distance from the origin"),
        "{cut}"
    );
}

/// A scale of 1e300 composed with a scale of 1e-300 is the identity: each
/// is a finite matrix, and so is their product, so neither is refused.
#[test]
fn a_composition_of_large_and_small_scales_is_built() {
    let k = BrepKernel::new();
    let scale = |s: f64| {
        vec![
            s, 0.0, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    };
    let product = k.compose_transforms(scale(1e300), scale(1e-300)).unwrap();
    assert_eq!(product, scale(1.0));
}

/// A triangle on the plane with normal (1, 1, 1)/sqrt(3): its third copy
/// along that normal, 6e299 out on each axis, keeps every vertex within the
/// size limit, but its plane's distance is 1.04e300. The pattern is refused,
/// and no copy of it is left in the arena.
#[test]
fn a_pattern_refused_for_its_last_plane_leaves_no_copy() {
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    let mut k = BrepKernel::new();
    let face = k
        .make_polygon(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        .unwrap();
    let face_id = k.resolve_face(face).unwrap();
    let shell = k.topo_mut().add_shell(Shell::new(vec![face_id]).unwrap());
    let solid = k.topo_mut().add_solid(Solid::new(shell, vec![]));
    let s = crate::handles::solid_id_to_u32(solid);
    let before = (k.topo.num_solids(), k.topo.num_faces());
    let spacing = 6e299 * 3.0_f64.sqrt() / 2.0;
    for op in [
        format!(
            r#"{{"op": "linearPattern", "args": {{"solid": {s}, "dx": 1, "dy": 1, "dz": 1, "spacing": {spacing:e}, "count": 3}}}}"#
        ),
        format!(
            r#"{{"op": "gridPattern", "args": {{"solid": {s}, "dirXx": 1, "dirXy": 1, "dirXz": 1, "dirYx": 0, "dirYy": 0, "dirYz": 1, "spacingX": {spacing:e}, "spacingY": 1, "countX": 3, "countY": 2}}}}"#
        ),
    ] {
        let out = run(&mut k, &op);
        let message = out["error"].as_str().unwrap_or_default();
        assert!(
            message.contains("would not pass the load checks"),
            "{op}: {out}"
        );
        assert_eq!((k.topo.num_solids(), k.topo.num_faces()), before, "{op}");
    }
}
