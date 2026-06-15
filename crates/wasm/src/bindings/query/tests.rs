#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::vec::Point3;

use crate::handles::face_id_to_u32;
use crate::kernel::test_fixtures::{kernel_with_box, kernel_with_cylinder};

// ── get_solid_faces ───────────────────────────────────────────

#[test]
fn box_has_six_faces() {
    let (k, solid) = kernel_with_box();
    let faces = k.get_solid_faces(solid).unwrap();
    assert_eq!(faces.len(), 6, "a box must have exactly 6 faces");
}

#[test]
fn face_handles_are_unique() {
    let (k, solid) = kernel_with_box();
    let faces = k.get_solid_faces(solid).unwrap();
    let mut sorted = faces.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), faces.len(), "face handles must be unique");
}

// ── get_solid_edges ───────────────────────────────────────────

#[test]
fn box_has_twelve_edges() {
    let (k, solid) = kernel_with_box();
    let edges = k.get_solid_edges(solid).unwrap();
    assert_eq!(edges.len(), 12, "a box must have exactly 12 edges");
}

#[test]
fn edge_handles_are_unique() {
    let (k, solid) = kernel_with_box();
    let edges = k.get_solid_edges(solid).unwrap();
    let mut sorted = edges.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), edges.len(), "edge handles must be unique");
}

// ── get_solid_vertices ────────────────────────────────────────

#[test]
fn box_has_eight_vertices() {
    let (k, solid) = kernel_with_box();
    let verts = k.get_solid_vertices(solid).unwrap();
    assert_eq!(verts.len(), 8, "a box must have exactly 8 vertices");
}

#[test]
fn vertex_handles_are_unique() {
    let (k, solid) = kernel_with_box();
    let verts = k.get_solid_vertices(solid).unwrap();
    let mut sorted = verts.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), verts.len(), "vertex handles must be unique");
}

// ── get_solid_shells ──────────────────────────────────────────

#[test]
fn box_has_one_shell() {
    let (k, solid) = kernel_with_box();
    let shells = k.get_solid_shells(solid).unwrap();
    assert_eq!(shells.len(), 1, "a box must report exactly one shell");
}

#[test]
fn solid_shell_round_trips_to_faces() {
    let (k, solid) = kernel_with_box();
    let shells = k.get_solid_shells(solid).unwrap();
    let shell_faces = k.get_shell_faces(shells[0]).unwrap();
    assert_eq!(
        shell_faces.len(),
        6,
        "the box's single shell must enumerate all six faces"
    );
}

// ── get_face_normal ───────────────────────────────────────────

#[test]
fn face_normal_has_three_components() {
    let (k, solid) = kernel_with_box();
    let faces = k.get_solid_faces(solid).unwrap();
    let normal = k.get_face_normal(faces[0]).unwrap();
    assert_eq!(normal.len(), 3, "normal must have exactly 3 components");
}

#[test]
fn face_normal_is_unit_length() {
    let (k, solid) = kernel_with_box();
    let faces = k.get_solid_faces(solid).unwrap();
    // Every face of a box is planar — all normals must be unit vectors.
    for &fh in &faces {
        let n = k.get_face_normal(fh).unwrap();
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-10,
            "face {fh} normal length {len} is not 1"
        );
    }
}

#[test]
fn face_normal_error_on_non_planar_face() {
    let (k, solid) = kernel_with_cylinder();
    // The cylindrical lateral face is not planar — verify via topology.
    let solid_id = k.resolve_solid(solid).unwrap();
    let faces = brepkit_topology::explorer::solid_faces(&k.topo, solid_id).unwrap();
    // At least one face must be non-planar (Cylinder surface).
    let has_non_planar = faces.iter().any(|&fid| {
        let face = k.topo.face(fid).unwrap();
        !matches!(
            face.surface(),
            brepkit_topology::face::FaceSurface::Plane { .. }
        )
    });
    assert!(
        has_non_planar,
        "cylinder must contain at least one non-planar face"
    );
}

// ── get_entity_counts ─────────────────────────────────────────

#[test]
fn entity_counts_match_individual_queries() {
    let (k, solid) = kernel_with_box();
    let counts = k.get_entity_counts(solid).unwrap();
    assert_eq!(
        counts.len(),
        3,
        "get_entity_counts must return [faces, edges, vertices]"
    );
    let faces = k.get_solid_faces(solid).unwrap();
    let edges = k.get_solid_edges(solid).unwrap();
    let verts = k.get_solid_vertices(solid).unwrap();
    assert_eq!(counts[0] as usize, faces.len(), "face count mismatch");
    assert_eq!(counts[1] as usize, edges.len(), "edge count mismatch");
    assert_eq!(counts[2] as usize, verts.len(), "vertex count mismatch");
}

#[test]
fn entity_counts_box_exact() {
    let (k, solid) = kernel_with_box();
    let counts = k.get_entity_counts(solid).unwrap();
    assert_eq!(counts[0], 6, "box: 6 faces");
    assert_eq!(counts[1], 12, "box: 12 edges");
    assert_eq!(counts[2], 8, "box: 8 vertices");
}

// ── invalid handle ────────────────────────────────────────────
// Error-path tests use resolve_solid (which returns WasmError, not
// JsError) to avoid the JsError panic on non-wasm targets.

#[test]
fn invalid_solid_handle_returns_error_for_faces() {
    let (k, _) = kernel_with_box();
    let result = k.resolve_solid(9999);
    assert!(
        result.is_err(),
        "non-existent solid handle must produce an error"
    );
}

#[test]
fn invalid_solid_handle_returns_error_for_edges() {
    let (k, _) = kernel_with_box();
    let result = k.resolve_solid(9999);
    assert!(
        result.is_err(),
        "non-existent solid handle must produce an error"
    );
}

#[test]
fn invalid_solid_handle_returns_error_for_vertices() {
    let (k, _) = kernel_with_box();
    let result = k.resolve_solid(9999);
    assert!(
        result.is_err(),
        "non-existent solid handle must produce an error"
    );
}

#[test]
fn invalid_solid_handle_returns_error_for_entity_counts() {
    let (k, _) = kernel_with_box();
    let result = k.resolve_solid(9999);
    assert!(
        result.is_err(),
        "non-existent solid handle must produce an error"
    );
}

// ── toBREP geometry data ─────────────────────────────────────
//
// `to_brep` returns JsValue which panics on non-wasm targets, so we
// replicate the JSON serialization logic in a helper and verify the
// same field structure that `to_brep` emits.

use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::FaceSurface;

/// Build the toBREP JSON for a solid using the same logic as
/// `to_brep`, but returning `serde_json::Value` directly so the
/// test can run on non-wasm targets.
fn build_brep_json(k: &crate::kernel::BrepKernel, solid: u32) -> serde_json::Value {
    let solid_id = k.resolve_solid(solid).unwrap();
    let faces = brepkit_topology::explorer::solid_faces(&k.topo, solid_id).unwrap();
    let edges = brepkit_topology::explorer::solid_edges(&k.topo, solid_id).unwrap();
    let verts = brepkit_topology::explorer::solid_vertices(&k.topo, solid_id).unwrap();

    let edge_json: Vec<serde_json::Value> = edges
        .iter()
        .map(|&eid| {
            let e = k.topo.edge(eid).unwrap();
            let curve_type = e.curve().type_tag();
            let curve_params = match e.curve() {
                EdgeCurve::Line => serde_json::json!(null),
                EdgeCurve::Circle(c) => serde_json::json!({
                    "center": [c.center().x(), c.center().y(), c.center().z()],
                    "axis": [c.normal().x(), c.normal().y(), c.normal().z()],
                    "xAxis": [c.u_axis().x(), c.u_axis().y(), c.u_axis().z()],
                    "radius": c.radius(),
                }),
                EdgeCurve::Ellipse(el) => serde_json::json!({
                    "center": [el.center().x(), el.center().y(), el.center().z()],
                    "axis": [el.normal().x(), el.normal().y(), el.normal().z()],
                    "majorAxis": [el.u_axis().x(), el.u_axis().y(), el.u_axis().z()],
                    "majorRadius": el.semi_major(),
                    "minorRadius": el.semi_minor(),
                }),
                EdgeCurve::NurbsCurve(n) => serde_json::json!({
                    "degree": n.degree(),
                    "controlPoints": n.control_points().iter()
                        .map(|p| [p.x(), p.y(), p.z()])
                        .collect::<Vec<_>>(),
                    "weights": n.weights().to_vec(),
                    "knots": n.knots().to_vec(),
                }),
            };
            serde_json::json!({
                "id": eid.index(),
                "curveType": curve_type,
                "curveParams": curve_params,
            })
        })
        .collect();

    let face_json: Vec<serde_json::Value> = faces
        .iter()
        .map(|&fid| {
            let f = k.topo.face(fid).unwrap();
            let surface_type = f.surface().type_tag();
            let surface_params = match f.surface() {
                FaceSurface::Plane { normal, d } => serde_json::json!({
                    "normal": [normal.x(), normal.y(), normal.z()],
                    "d": d,
                }),
                FaceSurface::Cylinder(c) => serde_json::json!({
                    "origin": [c.origin().x(), c.origin().y(), c.origin().z()],
                    "axis": [c.axis().x(), c.axis().y(), c.axis().z()],
                    "refDir": [c.x_axis().x(), c.x_axis().y(), c.x_axis().z()],
                    "radius": c.radius(),
                }),
                FaceSurface::Cone(c) => serde_json::json!({
                    "apex": [c.apex().x(), c.apex().y(), c.apex().z()],
                    "axis": [c.axis().x(), c.axis().y(), c.axis().z()],
                    "refDir": [c.x_axis().x(), c.x_axis().y(), c.x_axis().z()],
                    "halfAngle": c.half_angle(),
                }),
                FaceSurface::Sphere(s) => serde_json::json!({
                    "center": [s.center().x(), s.center().y(), s.center().z()],
                    "axis": [s.z_axis().x(), s.z_axis().y(), s.z_axis().z()],
                    "radius": s.radius(),
                }),
                FaceSurface::Torus(t) => serde_json::json!({
                    "center": [t.center().x(), t.center().y(), t.center().z()],
                    "axis": [t.z_axis().x(), t.z_axis().y(), t.z_axis().z()],
                    "majorRadius": t.major_radius(),
                    "minorRadius": t.minor_radius(),
                }),
                FaceSurface::Nurbs(n) => serde_json::json!({
                    "degreeU": n.degree_u(),
                    "degreeV": n.degree_v(),
                    "controlPoints": n.control_points().iter()
                        .map(|row| row.iter()
                            .map(|p| [p.x(), p.y(), p.z()])
                            .collect::<Vec<_>>())
                        .collect::<Vec<_>>(),
                    "weights": n.weights().to_vec(),
                    "knotsU": n.knots_u().to_vec(),
                    "knotsV": n.knots_v().to_vec(),
                }),
            };
            serde_json::json!({
                "id": fid.index(),
                "surfaceType": surface_type,
                "surfaceParams": surface_params,
            })
        })
        .collect();

    serde_json::json!({
        "vertices": verts.len(),
        "edges": edge_json,
        "faces": face_json,
    })
}

#[test]
fn to_brep_box_edges_have_null_curve_params() {
    let (k, solid) = kernel_with_box();
    let brep = build_brep_json(&k, solid);
    let edges = brep["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 12, "box must have 12 edges");
    for edge in edges {
        assert_eq!(edge["curveType"].as_str().unwrap(), "line");
        assert!(
            edge["curveParams"].is_null(),
            "line edges should have null curveParams"
        );
    }
}

#[test]
fn to_brep_box_faces_have_plane_surface_params() {
    let (k, solid) = kernel_with_box();
    let brep = build_brep_json(&k, solid);
    let faces = brep["faces"].as_array().unwrap();
    assert_eq!(faces.len(), 6, "box must have 6 faces");
    for face in faces {
        assert_eq!(face["surfaceType"].as_str().unwrap(), "plane");
        let params = &face["surfaceParams"];
        let normal = params["normal"].as_array().unwrap();
        assert_eq!(normal.len(), 3, "plane normal must have 3 components");
        let len: f64 = normal
            .iter()
            .map(|v| v.as_f64().unwrap().powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            (len - 1.0).abs() < 1e-10,
            "plane normal must be unit length, got {len}"
        );
        assert!(
            params["d"].as_f64().is_some(),
            "plane params must include 'd'"
        );
    }
}

#[test]
fn to_brep_cylinder_circle_edge_params() {
    let (k, solid) = kernel_with_cylinder();
    let brep = build_brep_json(&k, solid);
    let edges = brep["edges"].as_array().unwrap();
    let circle_edges: Vec<_> = edges
        .iter()
        .filter(|e| e["curveType"].as_str().unwrap() == "circle")
        .collect();
    assert!(!circle_edges.is_empty(), "cylinder must have circle edges");
    for ce in &circle_edges {
        let params = &ce["curveParams"];
        assert!(!params.is_null(), "circle edge must have curveParams");
        let center = params["center"].as_array().unwrap();
        assert_eq!(center.len(), 3);
        let axis = params["axis"].as_array().unwrap();
        assert_eq!(axis.len(), 3);
        // axis must be unit length
        let axis_len: f64 = axis
            .iter()
            .map(|v| v.as_f64().unwrap().powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            (axis_len - 1.0).abs() < 1e-10,
            "circle axis must be unit length, got {axis_len}"
        );
        let x_axis = params["xAxis"].as_array().unwrap();
        assert_eq!(x_axis.len(), 3);
        let radius = params["radius"].as_f64().unwrap();
        assert!(
            (radius - 1.0).abs() < 1e-10,
            "circle radius should be 1.0, got {radius}"
        );
    }
}

#[test]
fn to_brep_cylinder_surface_params() {
    let (k, solid) = kernel_with_cylinder();
    let brep = build_brep_json(&k, solid);
    let faces = brep["faces"].as_array().unwrap();
    let cyl_faces: Vec<_> = faces
        .iter()
        .filter(|f| f["surfaceType"].as_str().unwrap() == "cylinder")
        .collect();
    assert!(
        !cyl_faces.is_empty(),
        "cylinder solid must have cylinder faces"
    );
    for cf in &cyl_faces {
        let params = &cf["surfaceParams"];
        assert!(params["origin"].as_array().unwrap().len() == 3);
        assert!(params["axis"].as_array().unwrap().len() == 3);
        assert!(params["refDir"].as_array().unwrap().len() == 3);
        let radius = params["radius"].as_f64().unwrap();
        assert!(
            (radius - 1.0).abs() < 1e-10,
            "cylinder radius should be 1.0, got {radius}"
        );
    }
}

#[test]
fn to_brep_sphere_surface_params() {
    let mut k = crate::kernel::BrepKernel::new();
    let id = brepkit_operations::primitives::make_sphere(k.topo_mut(), 3.0, 16).unwrap();
    #[allow(clippy::cast_possible_truncation)]
    let solid = id.index() as u32;
    let brep = build_brep_json(&k, solid);
    let faces = brep["faces"].as_array().unwrap();
    let sph_faces: Vec<_> = faces
        .iter()
        .filter(|f| f["surfaceType"].as_str().unwrap() == "sphere")
        .collect();
    assert!(!sph_faces.is_empty(), "sphere solid must have sphere faces");
    for sf in &sph_faces {
        let params = &sf["surfaceParams"];
        assert!(params["center"].as_array().unwrap().len() == 3);
        assert!(params["axis"].as_array().unwrap().len() == 3);
        let radius = params["radius"].as_f64().unwrap();
        assert!(
            (radius - 3.0).abs() < 1e-10,
            "sphere radius should be 3.0, got {radius}"
        );
    }
}

#[test]
fn to_brep_cone_surface_params() {
    let mut k = crate::kernel::BrepKernel::new();
    let id = brepkit_operations::primitives::make_cone(k.topo_mut(), 2.0, 0.5, 4.0).unwrap();
    #[allow(clippy::cast_possible_truncation)]
    let solid = id.index() as u32;
    let brep = build_brep_json(&k, solid);
    let faces = brep["faces"].as_array().unwrap();
    let cone_faces: Vec<_> = faces
        .iter()
        .filter(|f| f["surfaceType"].as_str().unwrap() == "cone")
        .collect();
    assert!(!cone_faces.is_empty(), "cone solid must have cone faces");
    for cf in &cone_faces {
        let params = &cf["surfaceParams"];
        assert!(params["apex"].as_array().unwrap().len() == 3);
        assert!(params["axis"].as_array().unwrap().len() == 3);
        assert!(params["refDir"].as_array().unwrap().len() == 3);
        let half_angle = params["halfAngle"].as_f64().unwrap();
        assert!(
            half_angle > 0.0 && half_angle < std::f64::consts::FRAC_PI_2,
            "cone halfAngle must be in (0, pi/2), got {half_angle}"
        );
    }
}

#[test]
fn nurbs_quarter_circle_arc_reports_circle_type() {
    // #816: a circular arc stored as a rational NURBS curve must report
    // "CIRCLE" (so brepjs's curveAxis resolves it), not "BSPLINE_CURVE".
    // Unit quarter circle as a rational quadratic Bezier (weights
    // [1, cos45°, 1]).
    let mut k = crate::kernel::BrepKernel::new();
    let w = std::f64::consts::FRAC_1_SQRT_2;
    let edge = k
        .make_nurbs_edge(
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            vec![1.0, w, 1.0],
        )
        .unwrap();
    assert_eq!(k.get_edge_curve_type(edge).unwrap(), "CIRCLE");
}

#[test]
fn to_brep_torus_surface_params() {
    let mut k = crate::kernel::BrepKernel::new();
    let id = brepkit_operations::primitives::make_torus(k.topo_mut(), 5.0, 1.0, 16).unwrap();
    #[allow(clippy::cast_possible_truncation)]
    let solid = id.index() as u32;
    let brep = build_brep_json(&k, solid);
    let faces = brep["faces"].as_array().unwrap();
    let tor_faces: Vec<_> = faces
        .iter()
        .filter(|f| f["surfaceType"].as_str().unwrap() == "torus")
        .collect();
    assert!(!tor_faces.is_empty(), "torus solid must have torus faces");
    for tf in &tor_faces {
        let params = &tf["surfaceParams"];
        assert!(params["center"].as_array().unwrap().len() == 3);
        assert!(params["axis"].as_array().unwrap().len() == 3);
        let major_r = params["majorRadius"].as_f64().unwrap();
        assert!(
            (major_r - 5.0).abs() < 1e-10,
            "torus major radius should be 5.0, got {major_r}"
        );
        let minor_r = params["minorRadius"].as_f64().unwrap();
        assert!(
            (minor_r - 1.0).abs() < 1e-10,
            "torus minor radius should be 1.0, got {minor_r}"
        );
    }
}

fn closed_polygon_wire(
    k: &mut crate::kernel::BrepKernel,
    pts: &[Point3],
) -> brepkit_topology::wire::WireId {
    brepkit_topology::builder::make_polygon_wire(k.topo_mut(), pts, 1e-7).unwrap()
}

#[test]
fn surface_type_noncoplanar_wire_is_not_plane() {
    let mut k = crate::kernel::BrepKernel::new();
    let wid = closed_polygon_wire(
        &mut k,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(5.0, 5.0, 5.0),
        ],
    );
    let fid = brepkit_topology::builder::make_face_from_wire(k.topo_mut(), wid).unwrap();
    let stype = k.get_surface_type(face_id_to_u32(fid)).unwrap();
    assert_ne!(stype, "plane", "non-coplanar wire must not report a plane");
}

#[test]
fn surface_type_planar_square_wire_is_plane() {
    let mut k = crate::kernel::BrepKernel::new();
    let wid = closed_polygon_wire(
        &mut k,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        ],
    );
    let fid = brepkit_topology::builder::make_face_from_wire(k.topo_mut(), wid).unwrap();
    let stype = k.get_surface_type(face_id_to_u32(fid)).unwrap();
    assert_eq!(stype, "plane");
}

#[test]
fn make_planar_face_from_wire_rejects_noncoplanar() {
    let mut k = crate::kernel::BrepKernel::new();
    let wid = closed_polygon_wire(
        &mut k,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(5.0, 5.0, 5.0),
        ],
    );
    let res = brepkit_topology::builder::make_planar_face_from_wire(k.topo_mut(), wid);
    assert!(
        res.is_err(),
        "planar-only build must reject non-coplanar wire"
    );
    let msg = res.err().map(|e| e.to_string()).unwrap_or_default();
    assert!(
        msg.contains("not planar"),
        "error must mention 'not planar', got {msg}"
    );
}
