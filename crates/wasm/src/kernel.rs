//! The `BrepKernel` — a WASM-exposed modeling context.
//!
//! JavaScript consumers create a single `BrepKernel` instance and call
//! methods on it to build and query geometry. All topological state is
//! owned by the kernel; JS only holds opaque `u32` handles.

#![allow(
    clippy::missing_errors_doc,
    clippy::too_many_arguments,
    clippy::redundant_closure,
    clippy::redundant_closure_for_method_calls,
    clippy::map_unwrap_or,
    clippy::expect_used,
    dead_code
)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure;
use brepkit_operations::revolve::revolve;
use brepkit_operations::sweep::sweep;
use brepkit_operations::tessellate::{self, TriangleMesh};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};
use wasm_bindgen::prelude::*;

use crate::error::{WasmError, validate_finite, validate_positive};
use crate::shapes::JsMesh;

/// Default tolerance for vertices created by the kernel.
const TOL: f64 = 1e-7;

/// The B-Rep modeling kernel.
///
/// Owns all topological state. JavaScript holds this reference and
/// invokes methods to create, transform, and query geometry.
#[wasm_bindgen]
pub struct BrepKernel {
    topo: Topology,
    assemblies: Vec<brepkit_operations::assembly::Assembly>,
    sketches: Vec<SketchState>,
}

/// Internal state for an in-progress sketch.
#[derive(Default)]
struct SketchState {
    points: Vec<brepkit_operations::sketch::SketchPoint>,
    constraints: Vec<brepkit_operations::sketch::Constraint>,
}

#[wasm_bindgen]
impl BrepKernel {
    // ── Construction ────────────────────────────────────────────────

    /// Create a new, empty kernel.
    #[wasm_bindgen(constructor)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            topo: Topology::new(),
            assemblies: Vec::new(),
            sketches: Vec::new(),
        }
    }

    // ── Shape creation ─────────────────────────────────────────────

    /// Create a rectangular face on the XY plane centered at the origin.
    ///
    /// Returns a face handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if `width` or `height` is non-positive, NaN,
    /// or infinite, or if the face geometry cannot be constructed.
    #[wasm_bindgen(js_name = "makeRectangle")]
    pub fn make_rectangle(&mut self, width: f64, height: f64) -> Result<u32, JsError> {
        validate_positive(width, "width")?;
        validate_positive(height, "height")?;

        let hw = width / 2.0;
        let hh = height / 2.0;

        let points = [
            Point3::new(-hw, -hh, 0.0),
            Point3::new(hw, -hh, 0.0),
            Point3::new(hw, hh, 0.0),
            Point3::new(-hw, hh, 0.0),
        ];

        let face_id = self.make_planar_face(&points)?;
        Ok(face_id_to_u32(face_id))
    }

    /// Create a polygonal face from flat coordinate triples `[x,y,z, ...]`.
    ///
    /// Requires at least 3 points (9 `f64` values).
    /// Returns a face handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if `coords` length is not a multiple of 3,
    /// fewer than 3 points are provided, or the face normal is degenerate.
    #[wasm_bindgen(js_name = "makePolygon")]
    #[allow(clippy::needless_pass_by_value)] // wasm-bindgen requires owned Vec
    pub fn make_polygon(&mut self, coords: Vec<f64>) -> Result<u32, JsError> {
        if coords.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "coordinate array length must be a multiple of 3, got {}",
                    coords.len()
                ),
            }
            .into());
        }
        let n = coords.len() / 3;
        if n < 3 {
            return Err(WasmError::InvalidInput {
                reason: format!("polygon requires at least 3 points, got {n}"),
            }
            .into());
        }

        if let Some(pos) = coords.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("coordinate at index {pos} is not finite"),
            }
            .into());
        }

        let points: Vec<Point3> = coords
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();

        let face_id = self.make_planar_face(&points)?;
        Ok(face_id_to_u32(face_id))
    }

    /// Create a circular polygon approximation on the XY plane.
    ///
    /// The circle is centered at the origin with the given `radius`,
    /// approximated by `segments` straight edges.
    /// Returns a face handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 3 segments are specified.
    #[wasm_bindgen(js_name = "makeCircle")]
    pub fn make_circle(&mut self, radius: f64, segments: u32) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        if segments < 3 {
            return Err(WasmError::InvalidInput {
                reason: format!("circle requires at least 3 segments, got {segments}"),
            }
            .into());
        }

        let n = segments as usize;
        let mut points = Vec::with_capacity(n);
        for i in 0..n {
            #[allow(clippy::cast_precision_loss)]
            let angle = 2.0 * PI * (i as f64) / (n as f64);
            points.push(Point3::new(radius * angle.cos(), radius * angle.sin(), 0.0));
        }

        let face_id = self.make_planar_face(&points)?;
        Ok(face_id_to_u32(face_id))
    }

    // ── Primitive shapes ─────────────────────────────────────────

    /// Create a box solid with the given dimensions, centered at the origin.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is non-positive or non-finite.
    #[wasm_bindgen(js_name = "makeBox")]
    pub fn make_box_solid(&mut self, dx: f64, dy: f64, dz: f64) -> Result<u32, JsError> {
        validate_positive(dx, "dx")?;
        validate_positive(dy, "dy")?;
        validate_positive(dz, "dz")?;
        let solid_id = brepkit_operations::primitives::make_box(&mut self.topo, dx, dy, dz)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a cylinder solid centered at the origin, axis along +Z.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if radius or height is non-positive.
    #[wasm_bindgen(js_name = "makeCylinder")]
    pub fn make_cylinder_solid(&mut self, radius: f64, height: f64) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        validate_positive(height, "height")?;
        let solid_id =
            brepkit_operations::primitives::make_cylinder(&mut self.topo, radius, height)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a sphere solid centered at the origin.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if radius is non-positive or segments < 4.
    #[wasm_bindgen(js_name = "makeSphere")]
    pub fn make_sphere_solid(&mut self, radius: f64, segments: u32) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        let solid_id =
            brepkit_operations::primitives::make_sphere(&mut self.topo, radius, segments as usize)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a cone or frustum solid centered at the origin, axis along +Z.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if height is non-positive or both radii are zero.
    #[wasm_bindgen(js_name = "makeCone")]
    pub fn make_cone_solid(
        &mut self,
        bottom_radius: f64,
        top_radius: f64,
        height: f64,
    ) -> Result<u32, JsError> {
        validate_finite(bottom_radius, "bottom_radius")?;
        validate_finite(top_radius, "top_radius")?;
        validate_positive(height, "height")?;
        let solid_id = brepkit_operations::primitives::make_cone(
            &mut self.topo,
            bottom_radius,
            top_radius,
            height,
        )?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a torus solid centered at the origin in the XY plane.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if radii are non-positive or minor >= major.
    #[wasm_bindgen(js_name = "makeTorus")]
    pub fn make_torus_solid(
        &mut self,
        major_radius: f64,
        minor_radius: f64,
        segments: u32,
    ) -> Result<u32, JsError> {
        validate_positive(major_radius, "major_radius")?;
        validate_positive(minor_radius, "minor_radius")?;
        let solid_id = brepkit_operations::primitives::make_torus(
            &mut self.topo,
            major_radius,
            minor_radius,
            segments as usize,
        )?;
        Ok(solid_id_to_u32(solid_id))
    }

    // ── Section ───────────────────────────────────────────────────

    /// Section a solid with a plane, returning cross-section face handles.
    ///
    /// Returns an array of face handles (`u32[]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or the plane doesn't
    /// intersect the solid.
    #[wasm_bindgen(js_name = "section")]
    #[allow(clippy::too_many_arguments)]
    pub fn section_solid(
        &mut self,
        solid: u32,
        px: f64,
        py: f64,
        pz: f64,
        nx: f64,
        ny: f64,
        nz: f64,
    ) -> Result<Vec<u32>, JsError> {
        validate_finite(px, "px")?;
        validate_finite(py, "py")?;
        validate_finite(pz, "pz")?;
        validate_finite(nx, "nx")?;
        validate_finite(ny, "ny")?;
        validate_finite(nz, "nz")?;
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::section::section(
            &mut self.topo,
            solid_id,
            Point3::new(px, py, pz),
            Vec3::new(nx, ny, nz),
        )?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(result.faces.iter().map(|f| f.index() as u32).collect())
    }

    // ── Loft ──────────────────────────────────────────────────────

    /// Loft two or more profile faces into a solid.
    ///
    /// Takes an array of face handles. Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 faces or profiles have
    /// different vertex counts.
    #[wasm_bindgen(js_name = "loft")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn loft_faces(&mut self, faces: Vec<u32>) -> Result<u32, JsError> {
        let face_ids: Vec<brepkit_topology::face::FaceId> = faces
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let solid_id = brepkit_operations::loft::loft(&mut self.topo, &face_ids)?;
        Ok(solid_id_to_u32(solid_id))
    }

    // ── Shell ─────────────────────────────────────────────────────

    /// Hollow a solid with uniform wall thickness.
    ///
    /// `open_faces` is an array of face handles to remove (creating openings).
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if thickness is non-positive or the solid is invalid.
    #[wasm_bindgen(js_name = "shell")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn shell_solid(
        &mut self,
        solid: u32,
        thickness: f64,
        open_faces: Vec<u32>,
    ) -> Result<u32, JsError> {
        validate_positive(thickness, "thickness")?;
        let solid_id = self.resolve_solid(solid)?;
        let open_face_ids: Vec<brepkit_topology::face::FaceId> = open_faces
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let result = brepkit_operations::shell_op::shell(
            &mut self.topo,
            solid_id,
            thickness,
            &open_face_ids,
        )?;
        Ok(solid_id_to_u32(result))
    }

    // ── Chamfer ───────────────────────────────────────────────────

    /// Chamfer edges of a solid.
    ///
    /// `edge_handles` is an array of edge handles. Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if distance is non-positive or edges are invalid.
    #[wasm_bindgen(js_name = "chamfer")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn chamfer_solid(
        &mut self,
        solid: u32,
        edge_handles: Vec<u32>,
        distance: f64,
    ) -> Result<u32, JsError> {
        validate_positive(distance, "distance")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids: Vec<brepkit_topology::edge::EdgeId> = edge_handles
            .iter()
            .map(|&h| self.resolve_edge(h))
            .collect::<Result<_, _>>()?;
        let result =
            brepkit_operations::chamfer::chamfer(&mut self.topo, solid_id, &edge_ids, distance)?;
        Ok(solid_id_to_u32(result))
    }

    // ── Fillet ────────────────────────────────────────────────────

    /// Fillet (round) edges of a solid.
    ///
    /// `edge_handles` is an array of edge handles. Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if radius is non-positive or edges are invalid.
    #[wasm_bindgen(js_name = "fillet")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn fillet_solid(
        &mut self,
        solid: u32,
        edge_handles: Vec<u32>,
        radius: f64,
    ) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_ids: Vec<brepkit_topology::edge::EdgeId> = edge_handles
            .iter()
            .map(|&h| self.resolve_edge(h))
            .collect::<Result<_, _>>()?;
        let result =
            brepkit_operations::fillet::fillet(&mut self.topo, solid_id, &edge_ids, radius)?;
        Ok(solid_id_to_u32(result))
    }

    // ── Operations ─────────────────────────────────────────────────

    /// Extrude a planar face along a direction vector to create a solid.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid or the extrusion fails.
    #[wasm_bindgen(js_name = "extrude")]
    pub fn extrude_face(
        &mut self,
        face: u32,
        dir_x: f64,
        dir_y: f64,
        dir_z: f64,
        distance: f64,
    ) -> Result<u32, JsError> {
        validate_finite(dir_x, "dir_x")?;
        validate_finite(dir_y, "dir_y")?;
        validate_finite(dir_z, "dir_z")?;
        validate_finite(distance, "distance")?;

        let face_id = self.resolve_face(face)?;
        let direction = Vec3::new(dir_x, dir_y, dir_z);
        let solid_id = extrude(&mut self.topo, face_id, direction, distance)?;

        Ok(solid_id_to_u32(solid_id))
    }

    /// Revolve a planar face around an axis to create a solid of revolution.
    ///
    /// The axis is defined by an origin point `(ox, oy, oz)` and a direction
    /// `(dx, dy, dz)`. The angle is in degrees and must be in (0, 360].
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if any input is non-finite, the face handle is
    /// invalid, or the revolve operation fails.
    #[wasm_bindgen(js_name = "revolve")]
    #[allow(clippy::too_many_arguments)]
    pub fn revolve_face(
        &mut self,
        face: u32,
        ox: f64,
        oy: f64,
        oz: f64,
        dx: f64,
        dy: f64,
        dz: f64,
        angle_degrees: f64,
    ) -> Result<u32, JsError> {
        validate_finite(ox, "ox")?;
        validate_finite(oy, "oy")?;
        validate_finite(oz, "oz")?;
        validate_finite(dx, "dx")?;
        validate_finite(dy, "dy")?;
        validate_finite(dz, "dz")?;
        validate_finite(angle_degrees, "angle_degrees")?;
        if angle_degrees <= 0.0 || angle_degrees > 360.0 {
            return Err(WasmError::InvalidInput {
                reason: format!("angle_degrees must be in (0, 360], got {angle_degrees}"),
            }
            .into());
        }

        let face_id = self.resolve_face(face)?;
        let origin = Point3::new(ox, oy, oz);
        let direction = Vec3::new(dx, dy, dz);
        let angle_radians = angle_degrees.to_radians();

        let solid_id = revolve(&mut self.topo, face_id, origin, direction, angle_radians)?;

        Ok(solid_id_to_u32(solid_id))
    }

    /// Sweep a planar face along a NURBS curve path to create a solid.
    ///
    /// The path is specified as flat arrays for JS interop:
    /// - `path_degree` — polynomial degree of the path curve
    /// - `path_knots` — knot vector
    /// - `path_control_points` — flat `[x,y,z, ...]` control point coordinates
    /// - `path_weights` — per-control-point weights
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid, the NURBS arrays have
    /// inconsistent lengths, or the sweep operation fails.
    #[wasm_bindgen(js_name = "sweep")]
    #[allow(clippy::needless_pass_by_value)] // wasm-bindgen requires owned Vec
    pub fn sweep_face(
        &mut self,
        face: u32,
        path_degree: u32,
        path_knots: Vec<f64>,
        path_control_points: Vec<f64>,
        path_weights: Vec<f64>,
    ) -> Result<u32, JsError> {
        // Validate coordinate array length.
        if path_control_points.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "path_control_points length must be a multiple of 3, got {}",
                    path_control_points.len()
                ),
            }
            .into());
        }
        let num_pts = path_control_points.len() / 3;

        if path_weights.len() != num_pts {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "path_weights length ({}) must match number of control points ({num_pts})",
                    path_weights.len()
                ),
            }
            .into());
        }

        // Validate all values are finite.
        if let Some(pos) = path_knots.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("path_knots[{pos}] is not finite"),
            }
            .into());
        }
        if let Some(pos) = path_control_points.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("path_control_points[{pos}] is not finite"),
            }
            .into());
        }
        if let Some(pos) = path_weights.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("path_weights[{pos}] is not finite"),
            }
            .into());
        }

        let face_id = self.resolve_face(face)?;

        let control_points: Vec<Point3> = path_control_points
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();

        let path_curve = NurbsCurve::new(
            path_degree as usize,
            path_knots,
            control_points,
            path_weights,
        )?;

        let solid_id = sweep(&mut self.topo, face_id, &path_curve)?;

        Ok(solid_id_to_u32(solid_id))
    }

    /// Apply a 4×4 affine transform to a solid (in place).
    ///
    /// The `matrix` must contain exactly 16 values in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid, the matrix doesn't
    /// have 16 elements, or the matrix is singular.
    #[wasm_bindgen(js_name = "transformSolid")]
    #[allow(clippy::needless_pass_by_value)] // wasm-bindgen requires owned Vec
    pub fn transform_solid_wasm(&mut self, solid: u32, matrix: Vec<f64>) -> Result<(), JsError> {
        if matrix.len() != 16 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "transform matrix must have 16 elements, got {}",
                    matrix.len()
                ),
            }
            .into());
        }

        if let Some(pos) = matrix.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("matrix element at index {pos} is not finite"),
            }
            .into());
        }

        let solid_id = self.resolve_solid(solid)?;

        let rows = std::array::from_fn(|i| std::array::from_fn(|j| matrix[i * 4 + j]));
        let mat = Mat4(rows);

        transform_solid(&mut self.topo, solid_id, &mat)?;
        Ok(())
    }

    // ── Boolean operations ──────────────────────────────────────────

    /// Fuse (union) two solids into one.
    ///
    /// Returns a new solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty or non-manifold result.
    #[wasm_bindgen(js_name = "fuse")]
    pub fn fuse(&mut self, a: u32, b: u32) -> Result<u32, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let result = boolean(&mut self.topo, BooleanOp::Fuse, a_id, b_id)?;
        Ok(solid_id_to_u32(result))
    }

    /// Cut (subtract) solid `b` from solid `a`.
    ///
    /// Returns a new solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty or non-manifold result.
    #[wasm_bindgen(js_name = "cut")]
    pub fn cut(&mut self, a: u32, b: u32) -> Result<u32, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let result = boolean(&mut self.topo, BooleanOp::Cut, a_id, b_id)?;
        Ok(solid_id_to_u32(result))
    }

    /// Intersect two solids, keeping only their common volume.
    ///
    /// Returns a new solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty result.
    #[wasm_bindgen(js_name = "intersect")]
    pub fn intersect_solids(&mut self, a: u32, b: u32) -> Result<u32, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let result = boolean(&mut self.topo, BooleanOp::Intersect, a_id, b_id)?;
        Ok(solid_id_to_u32(result))
    }

    // ── Export ─────────────────────────────────────────────────────

    /// Export a solid to 3MF format (ZIP archive as bytes).
    ///
    /// Returns a `Uint8Array` in JavaScript containing the `.3mf` file.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[wasm_bindgen(js_name = "export3mf")]
    pub fn export_3mf(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let bytes = brepkit_io::threemf::write_threemf(&self.topo, &[solid_id], deflection)?;
        Ok(bytes)
    }

    /// Export a solid to binary STL format.
    ///
    /// Returns a `Uint8Array` containing the `.stl` file.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[wasm_bindgen(js_name = "exportStl")]
    pub fn export_stl(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let bytes = brepkit_io::stl::writer::write_stl(
            &self.topo,
            &[solid_id],
            deflection,
            brepkit_io::stl::writer::StlFormat::Binary,
        )?;
        Ok(bytes)
    }

    /// Export a solid to OBJ format (UTF-8 string as bytes).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "exportObj")]
    pub fn export_obj(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let obj_str = brepkit_io::obj::write_obj(&self.topo, &[solid_id], deflection)?;
        Ok(obj_str.into_bytes())
    }

    /// Export a solid to glTF binary (.glb) format.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "exportGlb")]
    pub fn export_glb(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let glb = brepkit_io::gltf::write_glb(&self.topo, &[solid_id], deflection)?;
        Ok(glb)
    }

    /// Export a solid to PLY format (binary little-endian).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "exportPly")]
    pub fn export_ply(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let ply = brepkit_io::ply::write_ply(
            &self.topo,
            &[solid_id],
            deflection,
            brepkit_io::ply::writer::PlyFormat::BinaryLittleEndian,
        )?;
        Ok(ply)
    }

    // ── Import ──────────────────────────────────────────────────────

    /// Import an OBJ file and return a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is malformed or mesh import fails.
    #[wasm_bindgen(js_name = "importObj")]
    pub fn import_obj(&mut self, data: &[u8]) -> Result<u32, JsError> {
        let text = std::str::from_utf8(data).map_err(|e| WasmError::InvalidInput {
            reason: format!("OBJ must be valid UTF-8: {e}"),
        })?;
        let mesh = brepkit_io::obj::read_obj(text)?;
        let solid_id = brepkit_io::stl::import::import_mesh(&mut self.topo, &mesh, 1e-7)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(solid_id.index() as u32)
    }

    /// Import a GLB (glTF binary) file and return a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is malformed or mesh import fails.
    #[wasm_bindgen(js_name = "importGlb")]
    pub fn import_glb(&mut self, data: &[u8]) -> Result<u32, JsError> {
        let mesh = brepkit_io::gltf::read_glb(data)?;
        let solid_id = brepkit_io::stl::import::import_mesh(&mut self.topo, &mesh, 1e-7)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(solid_id.index() as u32)
    }

    /// Import an STL file (binary or ASCII) and return a solid handle.
    ///
    /// The mesh triangles are converted to planar B-Rep faces with
    /// vertex merging.
    ///
    /// # Errors
    ///
    /// Returns an error if the STL data is malformed or empty.
    #[wasm_bindgen(js_name = "importStl")]
    pub fn import_stl(&mut self, data: &[u8]) -> Result<u32, JsError> {
        let mesh = brepkit_io::stl::reader::read_stl(data)?;
        let solid_id = brepkit_io::stl::import::import_mesh(&mut self.topo, &mesh, TOL)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Import a 3MF file and return solid handles.
    ///
    /// Returns handles for each object found in the 3MF archive.
    ///
    /// # Errors
    ///
    /// Returns an error if the 3MF data is malformed.
    #[wasm_bindgen(js_name = "import3mf")]
    pub fn import_3mf(&mut self, data: &[u8]) -> Result<Vec<u32>, JsError> {
        let meshes = brepkit_io::threemf::reader::read_threemf(data)?;
        let mut handles = Vec::new();
        for mesh in &meshes {
            let solid_id = brepkit_io::stl::import::import_mesh(&mut self.topo, mesh, TOL)?;
            handles.push(solid_id_to_u32(solid_id));
        }
        Ok(handles)
    }

    /// Export a solid to STEP AP203 format.
    ///
    /// Returns the STEP file as a UTF-8 encoded byte vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[wasm_bindgen(js_name = "exportStep")]
    pub fn export_step(&self, solid: u32) -> Result<Vec<u8>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let step_str = brepkit_io::step::writer::write_step(&self.topo, &[solid_id])?;
        Ok(step_str.into_bytes())
    }

    /// Import a STEP file and return solid handles.
    ///
    /// Returns handles for each solid found in the STEP file.
    ///
    /// # Errors
    ///
    /// Returns an error if the STEP data is malformed.
    #[wasm_bindgen(js_name = "importStep")]
    pub fn import_step(&mut self, data: &[u8]) -> Result<Vec<u32>, JsError> {
        let text = std::str::from_utf8(data)
            .map_err(|e| JsError::new(&format!("STEP data is not valid UTF-8: {e}")))?;
        let solid_ids = brepkit_io::step::reader::read_step(text, &mut self.topo)?;
        Ok(solid_ids.iter().map(|id| solid_id_to_u32(*id)).collect())
    }

    /// Offset a face by a distance along its surface normal.
    ///
    /// Returns the new offset face handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid or the operation fails.
    #[wasm_bindgen(js_name = "offsetFace")]
    pub fn offset_face_wasm(
        &mut self,
        face: u32,
        distance: f64,
        samples: u32,
    ) -> Result<u32, JsError> {
        validate_finite(distance, "distance")?;
        let face_id = self.resolve_face(face)?;
        let result = brepkit_operations::offset_face::offset_face(
            &mut self.topo,
            face_id,
            distance,
            samples as usize,
        )?;
        Ok(face_id_to_u32(result))
    }

    // ── IGES Import/Export ────────────────────────────────────────

    /// Export a solid to IGES format.
    ///
    /// Returns the IGES file as a UTF-8 encoded byte vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[wasm_bindgen(js_name = "exportIges")]
    pub fn export_iges(&self, solid: u32) -> Result<Vec<u8>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let iges_str = brepkit_io::iges::writer::write_iges(&self.topo, &[solid_id])?;
        Ok(iges_str.into_bytes())
    }

    /// Import an IGES file and return solid handles.
    ///
    /// # Errors
    ///
    /// Returns an error if the IGES data is malformed.
    #[wasm_bindgen(js_name = "importIges")]
    pub fn import_iges(&mut self, data: &[u8]) -> Result<Vec<u32>, JsError> {
        let text = std::str::from_utf8(data)
            .map_err(|e| JsError::new(&format!("IGES data is not valid UTF-8: {e}")))?;
        let solid_ids = brepkit_io::iges::reader::read_iges(text, &mut self.topo)?;
        Ok(solid_ids.iter().map(|id| solid_id_to_u32(*id)).collect())
    }

    // ── Helical Sweep ───────────────────────────────────────────

    /// Create a helical sweep of a profile face.
    ///
    /// Sweeps the profile along a helix defined by axis, radius, pitch,
    /// and number of turns. Used for generating thread geometry.
    ///
    /// # Errors
    ///
    /// Returns an error if parameters are invalid or the sweep fails.
    #[wasm_bindgen(js_name = "helicalSweep")]
    #[allow(clippy::too_many_arguments)]
    pub fn helical_sweep_wasm(
        &mut self,
        profile: u32,
        axis_origin_x: f64,
        axis_origin_y: f64,
        axis_origin_z: f64,
        axis_dir_x: f64,
        axis_dir_y: f64,
        axis_dir_z: f64,
        radius: f64,
        pitch: f64,
        turns: f64,
    ) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        validate_positive(pitch, "pitch")?;
        let face_id = self.resolve_face(profile)?;

        let origin = brepkit_math::vec::Point3::new(axis_origin_x, axis_origin_y, axis_origin_z);
        let axis_dir = brepkit_math::vec::Vec3::new(axis_dir_x, axis_dir_y, axis_dir_z);

        let solid_id = brepkit_operations::helix::helical_sweep(
            &mut self.topo,
            face_id,
            origin,
            axis_dir,
            radius,
            pitch,
            turns,
            8,
        )?;
        Ok(solid_id_to_u32(solid_id))
    }

    // ── Copy / Mirror / Pattern ───────────────────────────────────

    /// Deep copy a solid, returning a new independent solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "copySolid")]
    pub fn copy_solid_wasm(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let copy = brepkit_operations::copy::copy_solid(&mut self.topo, solid_id)?;
        Ok(solid_id_to_u32(copy))
    }

    /// Copy a solid and apply a 4×4 row-major affine transform in one pass.
    ///
    /// Equivalent to `copySolid` + `transformSolid` but performs both in a
    /// single topology traversal, avoiding redundant NURBS clones.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid, the matrix doesn't
    /// have 16 elements, or the matrix is singular.
    #[wasm_bindgen(js_name = "copyAndTransformSolid")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn copy_and_transform_solid_wasm(
        &mut self,
        solid: u32,
        matrix: Vec<f64>,
    ) -> Result<u32, JsError> {
        if matrix.len() != 16 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "transform matrix must have 16 elements, got {}",
                    matrix.len()
                ),
            }
            .into());
        }

        if let Some(pos) = matrix.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("matrix element at index {pos} is not finite"),
            }
            .into());
        }

        let solid_id = self.resolve_solid(solid)?;

        let rows = std::array::from_fn(|i| std::array::from_fn(|j| matrix[i * 4 + j]));
        let mat = Mat4(rows);

        let copy =
            brepkit_operations::copy::copy_and_transform_solid(&mut self.topo, solid_id, &mat)?;
        Ok(solid_id_to_u32(copy))
    }

    /// Mirror a solid across a plane.
    ///
    /// Returns a new solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or the normal is zero.
    #[wasm_bindgen(js_name = "mirror")]
    #[allow(clippy::too_many_arguments)]
    pub fn mirror_solid(
        &mut self,
        solid: u32,
        px: f64,
        py: f64,
        pz: f64,
        nx: f64,
        ny: f64,
        nz: f64,
    ) -> Result<u32, JsError> {
        validate_finite(px, "px")?;
        validate_finite(py, "py")?;
        validate_finite(pz, "pz")?;
        validate_finite(nx, "nx")?;
        validate_finite(ny, "ny")?;
        validate_finite(nz, "nz")?;
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::mirror::mirror(
            &mut self.topo,
            solid_id,
            Point3::new(px, py, pz),
            Vec3::new(nx, ny, nz),
        )?;
        Ok(solid_id_to_u32(result))
    }

    /// Create a linear pattern of a solid.
    ///
    /// Returns a compound handle containing all copies.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid.
    #[wasm_bindgen(js_name = "linearPattern")]
    #[allow(clippy::too_many_arguments)]
    pub fn linear_pattern_wasm(
        &mut self,
        solid: u32,
        dx: f64,
        dy: f64,
        dz: f64,
        spacing: f64,
        count: u32,
    ) -> Result<u32, JsError> {
        validate_finite(dx, "dx")?;
        validate_finite(dy, "dy")?;
        validate_finite(dz, "dz")?;
        validate_positive(spacing, "spacing")?;
        let solid_id = self.resolve_solid(solid)?;
        let compound = brepkit_operations::pattern::linear_pattern(
            &mut self.topo,
            solid_id,
            Vec3::new(dx, dy, dz),
            spacing,
            count as usize,
        )?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(compound.index() as u32)
    }

    // ── Split ─────────────────────────────────────────────────────

    /// Split a solid into two halves along a plane.
    ///
    /// Returns `[positive_solid_handle, negative_solid_handle]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the plane doesn't intersect the solid.
    #[wasm_bindgen(js_name = "split")]
    #[allow(clippy::too_many_arguments)]
    pub fn split_solid(
        &mut self,
        solid: u32,
        px: f64,
        py: f64,
        pz: f64,
        nx: f64,
        ny: f64,
        nz: f64,
    ) -> Result<Vec<u32>, JsError> {
        validate_finite(px, "px")?;
        validate_finite(py, "py")?;
        validate_finite(pz, "pz")?;
        validate_finite(nx, "nx")?;
        validate_finite(ny, "ny")?;
        validate_finite(nz, "nz")?;
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::split::split(
            &mut self.topo,
            solid_id,
            Point3::new(px, py, pz),
            Vec3::new(nx, ny, nz),
        )?;
        Ok(vec![
            solid_id_to_u32(result.positive),
            solid_id_to_u32(result.negative),
        ])
    }

    // ── Draft ─────────────────────────────────────────────────────

    /// Apply draft angle to faces of a solid.
    ///
    /// `face_handles` is an array of face handles to draft.
    /// Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if angle is zero or faces are invalid.
    #[wasm_bindgen(js_name = "draft")]
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    pub fn draft_solid(
        &mut self,
        solid: u32,
        face_handles: Vec<u32>,
        pull_x: f64,
        pull_y: f64,
        pull_z: f64,
        neutral_x: f64,
        neutral_y: f64,
        neutral_z: f64,
        angle_degrees: f64,
    ) -> Result<u32, JsError> {
        validate_finite(angle_degrees, "angle_degrees")?;
        let solid_id = self.resolve_solid(solid)?;
        let face_ids: Vec<brepkit_topology::face::FaceId> = face_handles
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let result = brepkit_operations::draft::draft(
            &mut self.topo,
            solid_id,
            &face_ids,
            Vec3::new(pull_x, pull_y, pull_z),
            Point3::new(neutral_x, neutral_y, neutral_z),
            angle_degrees.to_radians(),
        )?;
        Ok(solid_id_to_u32(result))
    }

    // ── Pipe ──────────────────────────────────────────────────────

    /// Pipe sweep: sweep a profile along a NURBS path (no guide).
    ///
    /// Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the face or path is invalid.
    #[wasm_bindgen(js_name = "pipe")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn pipe_solid(
        &mut self,
        face: u32,
        path_degree: u32,
        path_knots: Vec<f64>,
        path_control_points: Vec<f64>,
        path_weights: Vec<f64>,
    ) -> Result<u32, JsError> {
        if path_control_points.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "path_control_points length must be a multiple of 3, got {}",
                    path_control_points.len()
                ),
            }
            .into());
        }

        let face_id = self.resolve_face(face)?;
        let control_points: Vec<Point3> = path_control_points
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();

        let path_curve = NurbsCurve::new(
            path_degree as usize,
            path_knots,
            control_points,
            path_weights,
        )?;

        let solid_id = brepkit_operations::pipe::pipe(&mut self.topo, face_id, &path_curve, None)?;
        Ok(solid_id_to_u32(solid_id))
    }

    // ── Tessellation ───────────────────────────────────────────────

    /// Tessellate a single face into a triangle mesh.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "tessellateFace")]
    pub fn tessellate_face(&self, face: u32, deflection: f64) -> Result<JsMesh, JsError> {
        validate_positive(deflection, "deflection")?;
        let face_id = self.resolve_face(face)?;
        let mesh = tessellate::tessellate(&self.topo, face_id, deflection)?;
        Ok(mesh.into())
    }

    /// Tessellate all faces of a solid into a single merged triangle mesh.
    ///
    /// Includes both the outer shell and any inner shells (voids).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "tessellateSolid")]
    pub fn tessellate_solid(&self, solid: u32, deflection: f64) -> Result<JsMesh, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let solid_data = self.topo.solid(solid_id)?;

        let mut merged = TriangleMesh::default();

        for shell_id in std::iter::once(solid_data.outer_shell())
            .chain(solid_data.inner_shells().iter().copied())
        {
            let shell = self.topo.shell(shell_id)?;
            for &face_id in shell.faces() {
                let face_mesh = tessellate::tessellate(&self.topo, face_id, deflection)?;

                #[allow(clippy::cast_possible_truncation)]
                let offset = merged.positions.len() as u32;

                merged.positions.extend_from_slice(&face_mesh.positions);
                merged.normals.extend_from_slice(&face_mesh.normals);
                merged
                    .indices
                    .extend(face_mesh.indices.iter().map(|i| i + offset));
            }
        }

        Ok(merged.into())
    }

    // ── Topology queries ──────────────────────────────────────────

    /// Get all face handles of a solid.
    ///
    /// Returns an array of face handles (`u32[]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "getSolidFaces")]
    pub fn get_solid_faces(&self, solid: u32) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let faces = brepkit_topology::explorer::solid_faces(&self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(faces.iter().map(|f| f.index() as u32).collect())
    }

    /// Get all edge handles of a solid.
    ///
    /// Returns an array of unique edge handles (`u32[]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "getSolidEdges")]
    pub fn get_solid_edges(&self, solid: u32) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let edges = brepkit_topology::explorer::solid_edges(&self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(edges.iter().map(|e| e.index() as u32).collect())
    }

    /// Get all vertex handles of a solid.
    ///
    /// Returns an array of unique vertex handles (`u32[]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "getSolidVertices")]
    pub fn get_solid_vertices(&self, solid: u32) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let verts = brepkit_topology::explorer::solid_vertices(&self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(verts.iter().map(|v| v.index() as u32).collect())
    }

    /// Get the vertex positions of an edge.
    ///
    /// Returns `[start_x, start_y, start_z, end_x, end_y, end_z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge handle is invalid.
    #[wasm_bindgen(js_name = "getEdgeVertices")]
    pub fn get_edge_vertices(&self, edge: u32) -> Result<Vec<f64>, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        let start = self.topo.vertex(edge_data.start())?.point();
        let end = self.topo.vertex(edge_data.end())?.point();
        Ok(vec![
            start.x(),
            start.y(),
            start.z(),
            end.x(),
            end.y(),
            end.z(),
        ])
    }

    /// Get the position of a vertex.
    ///
    /// Returns `[x, y, z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the vertex handle is invalid.
    #[wasm_bindgen(js_name = "getVertexPosition")]
    pub fn get_vertex_position(&self, vertex: u32) -> Result<Vec<f64>, JsError> {
        let vertex_id = self.resolve_vertex(vertex)?;
        let point = self.topo.vertex(vertex_id)?.point();
        Ok(vec![point.x(), point.y(), point.z()])
    }

    /// Get the face normal of a planar face.
    ///
    /// Returns `[nx, ny, nz]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the face is invalid or NURBS.
    #[wasm_bindgen(js_name = "getFaceNormal")]
    pub fn get_face_normal(&self, face: u32) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        match face_data.surface() {
            brepkit_topology::face::FaceSurface::Plane { normal, .. } => {
                Ok(vec![normal.x(), normal.y(), normal.z()])
            }
            _ => Err(WasmError::InvalidInput {
                reason: "getFaceNormal only works on planar faces".into(),
            }
            .into()),
        }
    }

    /// Get entity counts of a solid: `[faces, edges, vertices]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "getEntityCounts")]
    pub fn get_entity_counts(&self, solid: u32) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let (f, e, v) = brepkit_topology::explorer::solid_entity_counts(&self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(vec![f as u32, e as u32, v as u32])
    }

    // ── Measurement ───────────────────────────────────────────────

    /// Compute the axis-aligned bounding box of a solid.
    ///
    /// Returns `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or has no vertices.
    #[wasm_bindgen(js_name = "boundingBox")]
    pub fn bounding_box(&self, solid: u32) -> Result<Vec<f64>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let aabb = measure::solid_bounding_box(&self.topo, solid_id)?;
        Ok(vec![
            aabb.min.x(),
            aabb.min.y(),
            aabb.min.z(),
            aabb.max.x(),
            aabb.max.y(),
            aabb.max.z(),
        ])
    }

    /// Compute the volume of a solid.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "volume")]
    pub fn volume(&self, solid: u32, deflection: f64) -> Result<f64, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        Ok(measure::solid_volume(&self.topo, solid_id, deflection)?)
    }

    /// Compute the total surface area of a solid.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "surfaceArea")]
    pub fn surface_area(&self, solid: u32, deflection: f64) -> Result<f64, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        Ok(measure::solid_surface_area(
            &self.topo, solid_id, deflection,
        )?)
    }

    /// Compute the area of a single face.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "faceArea")]
    pub fn face_area_wasm(&self, face: u32, deflection: f64) -> Result<f64, JsError> {
        validate_positive(deflection, "deflection")?;
        let face_id = self.resolve_face(face)?;
        Ok(measure::face_area(&self.topo, face_id, deflection)?)
    }

    /// Compute the center of mass of a solid (uniform density).
    ///
    /// Returns `[x, y, z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid has zero volume or tessellation fails.
    #[wasm_bindgen(js_name = "centerOfMass")]
    pub fn center_of_mass(&self, solid: u32, deflection: f64) -> Result<Vec<f64>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let com = measure::solid_center_of_mass(&self.topo, solid_id, deflection)?;
        Ok(vec![com.x(), com.y(), com.z()])
    }

    /// Classify a point relative to a solid: inside, outside, or on boundary.
    ///
    /// Returns `"inside"`, `"outside"`, or `"boundary"`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "classifyPoint")]
    pub fn classify_point(
        &self,
        solid: u32,
        x: f64,
        y: f64,
        z: f64,
        tolerance: f64,
    ) -> Result<String, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let point = brepkit_math::vec::Point3::new(x, y, z);
        let result = brepkit_operations::classify::classify_point(
            &self.topo, solid_id, point, 0.1, tolerance,
        )?;
        Ok(match result {
            brepkit_operations::classify::PointClassification::Inside => "inside".into(),
            brepkit_operations::classify::PointClassification::Outside => "outside".into(),
            brepkit_operations::classify::PointClassification::OnBoundary => "boundary".into(),
        })
    }

    /// Compute the length of an edge.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge handle is invalid.
    #[wasm_bindgen(js_name = "edgeLength")]
    pub fn edge_length_wasm(&self, edge: u32) -> Result<f64, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        Ok(measure::edge_length(&self.topo, edge_id)?)
    }

    /// Compute the perimeter of a face.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid.
    #[wasm_bindgen(js_name = "facePerimeter")]
    pub fn face_perimeter_wasm(&self, face: u32) -> Result<f64, JsError> {
        let face_id = self.resolve_face(face)?;
        Ok(measure::face_perimeter(&self.topo, face_id)?)
    }

    /// Validate a solid, returning the number of errors found.
    ///
    /// Returns 0 if the solid is valid.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "validateSolid")]
    pub fn validate_solid_wasm(&self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let report = brepkit_operations::validate::validate_solid(&self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(report.error_count() as u32)
    }

    // ── Distance ──────────────────────────────────────────────────

    /// Compute minimum distance from a point to a solid.
    ///
    /// Returns `[distance, closest_x, closest_y, closest_z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "pointToSolidDistance")]
    pub fn point_to_solid_distance_wasm(
        &self,
        px: f64,
        py: f64,
        pz: f64,
        solid: u32,
    ) -> Result<Vec<f64>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::distance::point_to_solid_distance(
            &self.topo,
            Point3::new(px, py, pz),
            solid_id,
        )?;
        Ok(vec![
            result.distance,
            result.point_b.x(),
            result.point_b.y(),
            result.point_b.z(),
        ])
    }

    /// Compute minimum distance between two solids.
    ///
    /// Returns `[distance]`.
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid.
    #[wasm_bindgen(js_name = "solidToSolidDistance")]
    pub fn solid_to_solid_distance_wasm(&self, a: u32, b: u32) -> Result<f64, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let result = brepkit_operations::distance::solid_to_solid_distance(&self.topo, a_id, b_id)?;
        Ok(result.distance)
    }

    // ── Sewing ────────────────────────────────────────────────────

    /// Sew loose faces into a connected solid.
    ///
    /// `face_handles` is an array of face handles. Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 faces or sewing fails.
    #[wasm_bindgen(js_name = "sewFaces")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn sew_faces_wasm(
        &mut self,
        face_handles: Vec<u32>,
        tolerance: f64,
    ) -> Result<u32, JsError> {
        let face_ids: Vec<brepkit_topology::face::FaceId> = face_handles
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let solid = brepkit_operations::sew::sew_faces(&mut self.topo, &face_ids, tolerance)?;
        Ok(solid_id_to_u32(solid))
    }

    // ── Shape construction (low-level) ────────────────────────────

    /// Create a vertex at the given position.
    ///
    /// Returns a vertex handle (`u32`).
    #[wasm_bindgen(js_name = "makeVertex")]
    pub fn make_vertex_wasm(&mut self, x: f64, y: f64, z: f64) -> Result<u32, JsError> {
        validate_finite(x, "x")?;
        validate_finite(y, "y")?;
        validate_finite(z, "z")?;
        let id = self
            .topo
            .vertices
            .alloc(Vertex::new(Point3::new(x, y, z), TOL));
        Ok(vertex_id_to_u32(id))
    }

    /// Create a straight-line edge between two points.
    ///
    /// Returns an edge handle (`u32`).
    #[wasm_bindgen(js_name = "makeLineEdge")]
    pub fn make_line_edge_wasm(
        &mut self,
        x1: f64,
        y1: f64,
        z1: f64,
        x2: f64,
        y2: f64,
        z2: f64,
    ) -> Result<u32, JsError> {
        let start = Point3::new(x1, y1, z1);
        let end = Point3::new(x2, y2, z2);
        let eid = brepkit_topology::builder::make_line_edge(&mut self.topo, start, end)?;
        Ok(edge_id_to_u32(eid))
    }

    /// Create a NURBS curve edge.
    ///
    /// Returns an edge handle (`u32`).
    #[wasm_bindgen(js_name = "makeNurbsEdge")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_nurbs_edge_wasm(
        &mut self,
        start_x: f64,
        start_y: f64,
        start_z: f64,
        end_x: f64,
        end_y: f64,
        end_z: f64,
        degree: u32,
        knots: Vec<f64>,
        control_points: Vec<f64>,
        weights: Vec<f64>,
    ) -> Result<u32, JsError> {
        if control_points.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "control_points length must be a multiple of 3, got {}",
                    control_points.len()
                ),
            }
            .into());
        }
        let cp: Vec<Point3> = control_points
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();
        let curve = NurbsCurve::new(degree as usize, knots, cp, weights)?;

        let v_start = self
            .topo
            .vertices
            .alloc(Vertex::new(Point3::new(start_x, start_y, start_z), TOL));
        let v_end = self
            .topo
            .vertices
            .alloc(Vertex::new(Point3::new(end_x, end_y, end_z), TOL));
        let eid = self
            .topo
            .edges
            .alloc(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(curve)));
        Ok(edge_id_to_u32(eid))
    }

    /// Create a closed wire from an ordered array of edge handles.
    ///
    /// Returns a wire handle (`u32`).
    #[wasm_bindgen(js_name = "makeWire")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_wire_wasm(&mut self, edge_handles: Vec<u32>, closed: bool) -> Result<u32, JsError> {
        let oriented: Vec<OrientedEdge> = edge_handles
            .iter()
            .map(|&h| {
                let eid = self.resolve_edge(h)?;
                Ok(OrientedEdge::new(eid, true))
            })
            .collect::<Result<_, WasmError>>()?;
        let wire = Wire::new(oriented, closed)?;
        let wid = self.topo.wires.alloc(wire);
        Ok(wire_id_to_u32(wid))
    }

    /// Create a planar face from a wire (computes normal from first 3 vertices).
    ///
    /// Returns a face handle (`u32`).
    #[wasm_bindgen(js_name = "makeFaceFromWire")]
    pub fn make_face_from_wire_wasm(&mut self, wire: u32) -> Result<u32, JsError> {
        let wid = self.resolve_wire(wire)?;
        let fid = brepkit_topology::builder::make_face_from_wire(&mut self.topo, wid)?;
        Ok(face_id_to_u32(fid))
    }

    /// Create a solid from a shell.
    ///
    /// Returns a solid handle (`u32`).
    #[wasm_bindgen(js_name = "solidFromShell")]
    pub fn solid_from_shell_wasm(&mut self, shell: u32) -> Result<u32, JsError> {
        let shell_id = self.resolve_shell(shell)?;
        let solid = brepkit_topology::solid::Solid::new(shell_id, vec![]);
        let sid = self.topo.solids.alloc(solid);
        Ok(solid_id_to_u32(sid))
    }

    /// Create a compound from multiple solid handles.
    ///
    /// Returns a compound handle (stored as `u32`).
    #[wasm_bindgen(js_name = "makeCompound")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_compound_wasm(&mut self, solid_handles: Vec<u32>) -> Result<u32, JsError> {
        let solid_ids: Vec<brepkit_topology::solid::SolidId> = solid_handles
            .iter()
            .map(|&h| self.resolve_solid(h))
            .collect::<Result<_, _>>()?;
        let compound = brepkit_topology::compound::Compound::new(solid_ids);
        #[allow(clippy::cast_possible_truncation)]
        let cid = self.topo.compounds.alloc(compound);
        Ok(cid.index() as u32)
    }

    // ── Topology queries (extended) ──────────────────────────────

    /// Get the edge handles of a face.
    ///
    /// Returns an array of edge handles (`u32[]`).
    #[wasm_bindgen(js_name = "getFaceEdges")]
    pub fn get_face_edges(&self, face: u32) -> Result<Vec<u32>, JsError> {
        let face_id = self.resolve_face(face)?;
        let edges = brepkit_topology::explorer::face_edges(&self.topo, face_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(edges.iter().map(|e| e.index() as u32).collect())
    }

    /// Get the vertex handles of a face.
    ///
    /// Returns an array of vertex handles (`u32[]`).
    #[wasm_bindgen(js_name = "getFaceVertices")]
    pub fn get_face_vertices(&self, face: u32) -> Result<Vec<u32>, JsError> {
        let face_id = self.resolve_face(face)?;
        let verts = brepkit_topology::explorer::face_vertices(&self.topo, face_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(verts.iter().map(|v| v.index() as u32).collect())
    }

    /// Get the outer wire handle of a face.
    ///
    /// Returns a wire handle (`u32`).
    #[wasm_bindgen(js_name = "getFaceOuterWire")]
    pub fn get_face_outer_wire(&self, face: u32) -> Result<u32, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        Ok(wire_id_to_u32(face_data.outer_wire()))
    }

    /// Get the surface type of a face.
    ///
    /// Returns one of: `"plane"`, `"cylinder"`, `"cone"`, `"sphere"`,
    /// `"torus"`, `"bspline"`.
    #[wasm_bindgen(js_name = "getSurfaceType")]
    pub fn get_surface_type(&self, face: u32) -> Result<String, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        Ok(match face_data.surface() {
            FaceSurface::Plane { .. } => "plane",
            FaceSurface::Nurbs(_) => "bspline",
            FaceSurface::Cylinder(_) => "cylinder",
            FaceSurface::Cone(_) => "cone",
            FaceSurface::Sphere(_) => "sphere",
            FaceSurface::Torus(_) => "torus",
        }
        .into())
    }

    /// Get the curve type of an edge.
    ///
    /// Returns `"line"` or `"bspline"`.
    #[wasm_bindgen(js_name = "getEdgeCurveType")]
    pub fn get_edge_curve_type(&self, edge: u32) -> Result<String, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        Ok(match edge_data.curve() {
            EdgeCurve::Line => "line",
            EdgeCurve::NurbsCurve(_) => "bspline",
        }
        .into())
    }

    /// Get the parameter domain of an edge curve.
    ///
    /// Returns `[t_start, t_end]`.
    /// For line edges: `[0.0, length]`.
    /// For NURBS edges: knot domain.
    #[wasm_bindgen(js_name = "getEdgeCurveParameters")]
    pub fn get_edge_curve_parameters(&self, edge: u32) -> Result<Vec<f64>, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        match edge_data.curve() {
            EdgeCurve::Line => {
                let start = self.topo.vertex(edge_data.start())?.point();
                let end = self.topo.vertex(edge_data.end())?.point();
                let len = (end - start).length();
                Ok(vec![0.0, len])
            }
            EdgeCurve::NurbsCurve(curve) => {
                let (u_start, u_end) = curve.domain();
                Ok(vec![u_start, u_end])
            }
        }
    }

    /// Evaluate a point on an edge curve at parameter `t`.
    ///
    /// Returns `[x, y, z]`.
    #[wasm_bindgen(js_name = "evaluateEdgeCurve")]
    pub fn evaluate_edge_curve(&self, edge: u32, t: f64) -> Result<Vec<f64>, JsError> {
        validate_finite(t, "t")?;
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        let point = match edge_data.curve() {
            EdgeCurve::Line => {
                let start = self.topo.vertex(edge_data.start())?.point();
                let end = self.topo.vertex(edge_data.end())?.point();
                let len = (end - start).length();
                if len < 1e-15 {
                    start
                } else {
                    let frac = t / len;
                    let dir = end - start;
                    Point3::new(
                        start.x() + dir.x() * frac,
                        start.y() + dir.y() * frac,
                        start.z() + dir.z() * frac,
                    )
                }
            }
            EdgeCurve::NurbsCurve(curve) => curve.evaluate(t),
        };
        Ok(vec![point.x(), point.y(), point.z()])
    }

    /// Evaluate a point and tangent on an edge curve at parameter `t`.
    ///
    /// Returns `[px, py, pz, tx, ty, tz]`.
    #[wasm_bindgen(js_name = "evaluateEdgeCurveD1")]
    pub fn evaluate_edge_curve_d1(&self, edge: u32, t: f64) -> Result<Vec<f64>, JsError> {
        validate_finite(t, "t")?;
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        match edge_data.curve() {
            EdgeCurve::Line => {
                let start = self.topo.vertex(edge_data.start())?.point();
                let end = self.topo.vertex(edge_data.end())?.point();
                let dir = end - start;
                let len = dir.length();
                let frac = if len < 1e-15 { 0.0 } else { t / len };
                let point = Point3::new(
                    start.x() + dir.x() * frac,
                    start.y() + dir.y() * frac,
                    start.z() + dir.z() * frac,
                );
                let tangent = if len < 1e-15 {
                    Vec3::new(1.0, 0.0, 0.0)
                } else {
                    Vec3::new(dir.x() / len, dir.y() / len, dir.z() / len)
                };
                Ok(vec![
                    point.x(),
                    point.y(),
                    point.z(),
                    tangent.x(),
                    tangent.y(),
                    tangent.z(),
                ])
            }
            EdgeCurve::NurbsCurve(curve) => {
                let point = curve.evaluate(t);
                let derivs = curve.derivatives(t, 1);
                let tangent = if derivs.len() > 1 {
                    derivs[1]
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                Ok(vec![
                    point.x(),
                    point.y(),
                    point.z(),
                    tangent.x(),
                    tangent.y(),
                    tangent.z(),
                ])
            }
        }
    }

    /// Evaluate a surface normal at (u, v) on a face.
    ///
    /// Returns `[nx, ny, nz]`.
    #[wasm_bindgen(js_name = "evaluateSurfaceNormal")]
    pub fn evaluate_surface_normal(&self, face: u32, u: f64, v: f64) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        match face_data.surface() {
            FaceSurface::Plane { normal, .. } => Ok(vec![normal.x(), normal.y(), normal.z()]),
            FaceSurface::Nurbs(surface) => {
                let derivs = surface.derivatives(u, v, 1);
                let du = if derivs.len() > 1 && !derivs[1].is_empty() {
                    derivs[1][0]
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                let dv = if !derivs.is_empty() && derivs[0].len() > 1 {
                    derivs[0][1]
                } else {
                    Vec3::new(0.0, 1.0, 0.0)
                };
                let n = du.cross(dv);
                match n.normalize() {
                    Ok(normal) => Ok(vec![normal.x(), normal.y(), normal.z()]),
                    Err(_) => Ok(vec![0.0, 0.0, 1.0]),
                }
            }
            FaceSurface::Cylinder(cyl) => {
                let n = cyl.normal(u, v);
                Ok(vec![n.x(), n.y(), n.z()])
            }
            FaceSurface::Cone(cone) => {
                let n = cone.normal(u, v);
                Ok(vec![n.x(), n.y(), n.z()])
            }
            FaceSurface::Sphere(sph) => {
                let n = sph.normal(u, v);
                Ok(vec![n.x(), n.y(), n.z()])
            }
            FaceSurface::Torus(tor) => {
                let n = tor.normal(u, v);
                Ok(vec![n.x(), n.y(), n.z()])
            }
        }
    }

    /// Evaluate a point on a face surface at (u, v).
    ///
    /// Returns `[x, y, z]`.
    #[wasm_bindgen(js_name = "evaluateSurface")]
    pub fn evaluate_surface(&self, face: u32, u: f64, v: f64) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let point = match face_data.surface() {
            FaceSurface::Plane { normal, d } => {
                // Build a point on the plane: p = d * normal + u * x_axis + v * y_axis
                // Choose arbitrary axes perpendicular to normal
                let up = if normal.x().abs() < 0.9 {
                    Vec3::new(1.0, 0.0, 0.0)
                } else {
                    Vec3::new(0.0, 1.0, 0.0)
                };
                let x_axis = normal.cross(up);
                let y_axis = normal.cross(x_axis);
                Point3::new(
                    normal.x() * d + x_axis.x() * u + y_axis.x() * v,
                    normal.y() * d + x_axis.y() * u + y_axis.y() * v,
                    normal.z() * d + x_axis.z() * u + y_axis.z() * v,
                )
            }
            FaceSurface::Nurbs(surface) => surface.evaluate(u, v),
            FaceSurface::Cylinder(cyl) => cyl.evaluate(u, v),
            FaceSurface::Cone(cone) => cone.evaluate(u, v),
            FaceSurface::Sphere(sph) => sph.evaluate(u, v),
            FaceSurface::Torus(tor) => tor.evaluate(u, v),
        };
        Ok(vec![point.x(), point.y(), point.z()])
    }

    /// Heal a solid topology.
    ///
    /// Returns the number of issues fixed.
    #[wasm_bindgen(js_name = "healSolid")]
    pub fn heal_solid_wasm(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let report = brepkit_operations::heal::heal_solid(&mut self.topo, solid_id, TOL)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(
            (report.vertices_merged + report.degenerate_edges_removed + report.orientations_fixed)
                as u32,
        )
    }

    /// Tessellate an edge curve into polyline segments.
    ///
    /// For line edges, returns just start and end points.
    /// For NURBS edges, samples at `num_points` along the curve.
    ///
    /// Returns flattened `[x, y, z, x, y, z, ...]` array.
    #[wasm_bindgen(js_name = "tessellateEdge")]
    pub fn tessellate_edge(&self, edge: u32, num_points: u32) -> Result<Vec<f64>, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;

        match edge_data.curve() {
            EdgeCurve::Line => {
                let start = self.topo.vertex(edge_data.start())?.point();
                let end = self.topo.vertex(edge_data.end())?.point();
                Ok(vec![
                    start.x(),
                    start.y(),
                    start.z(),
                    end.x(),
                    end.y(),
                    end.z(),
                ])
            }
            EdgeCurve::NurbsCurve(curve) => {
                let (u0, u1) = curve.domain();
                let n = std::cmp::max(2, num_points as usize);
                let mut result = Vec::with_capacity(n * 3);
                for i in 0..n {
                    #[allow(clippy::cast_precision_loss)]
                    let t = u0 + (u1 - u0) * (i as f64) / ((n - 1) as f64);
                    let p = curve.evaluate(t);
                    result.push(p.x());
                    result.push(p.y());
                    result.push(p.z());
                }
                Ok(result)
            }
        }
    }

    /// Check if an edge is forward-oriented in a given wire.
    ///
    /// Returns `true` if the edge is forward in the wire, `false` if reversed.
    #[wasm_bindgen(js_name = "isEdgeForwardInWire")]
    pub fn is_edge_forward_in_wire(&self, edge: u32, wire: u32) -> Result<bool, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let wire_id = self.resolve_wire(wire)?;
        let wire_data = self.topo.wire(wire_id)?;

        for oe in wire_data.edges() {
            if oe.edge() == edge_id {
                return Ok(oe.is_forward());
            }
        }

        Err(WasmError::InvalidInput {
            reason: "edge not found in wire".into(),
        }
        .into())
    }

    /// Get the UV parameter domain of a face's surface.
    ///
    /// Returns `[u_min, u_max, v_min, v_max]`.
    #[wasm_bindgen(js_name = "getSurfaceDomain")]
    pub fn get_surface_domain(&self, face: u32) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        match face_data.surface() {
            FaceSurface::Plane { .. } => Ok(vec![-1e6, 1e6, -1e6, 1e6]),
            FaceSurface::Nurbs(surface) => {
                let (u0, u1) = surface.domain_u();
                let (v0, v1) = surface.domain_v();
                Ok(vec![u0, u1, v0, v1])
            }
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => Ok(vec![0.0, 2.0 * PI, -1e6, 1e6]),
            FaceSurface::Sphere(_) => Ok(vec![0.0, 2.0 * PI, -PI / 2.0, PI / 2.0]),
            FaceSurface::Torus(_) => Ok(vec![0.0, 2.0 * PI, 0.0, 2.0 * PI]),
        }
    }

    /// Project a 3D point onto a face surface using Newton iteration.
    ///
    /// Returns `[u, v, px, py, pz, distance]`.
    #[wasm_bindgen(js_name = "projectPointOnSurface")]
    pub fn project_point_on_surface(
        &self,
        face: u32,
        px: f64,
        py: f64,
        pz: f64,
    ) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let target = Point3::new(px, py, pz);

        match face_data.surface() {
            FaceSurface::Plane { normal, d } => {
                // Project onto plane: p - ((p·n - d) * n)
                let dist_to_plane = normal.x() * px + normal.y() * py + normal.z() * pz - d;
                let proj = Point3::new(
                    px - dist_to_plane * normal.x(),
                    py - dist_to_plane * normal.y(),
                    pz - dist_to_plane * normal.z(),
                );
                let dist = (proj - target).length();
                // UV coordinates: project onto plane's local frame
                Ok(vec![proj.x(), proj.y(), proj.x(), proj.y(), proj.z(), dist])
            }
            FaceSurface::Nurbs(surface) => {
                // Newton iteration for closest point on NURBS surface
                let (u0, u1) = surface.domain_u();
                let (v0, v1) = surface.domain_v();
                let mut best_u = f64::midpoint(u0, u1);
                let mut best_v = f64::midpoint(v0, v1);
                let mut best_dist = f64::MAX;

                // Grid search for initial guess
                let n_grid = 8;
                for iu in 0..=n_grid {
                    for iv in 0..=n_grid {
                        #[allow(clippy::cast_precision_loss)]
                        let u = u0 + (u1 - u0) * (iu as f64) / (n_grid as f64);
                        #[allow(clippy::cast_precision_loss)]
                        let v = v0 + (v1 - v0) * (iv as f64) / (n_grid as f64);
                        let p = surface.evaluate(u, v);
                        let d = (p - target).length();
                        if d < best_dist {
                            best_dist = d;
                            best_u = u;
                            best_v = v;
                        }
                    }
                }

                // Newton refinement (5 iterations)
                for _ in 0..5 {
                    let p = surface.evaluate(best_u, best_v);
                    let derivs = surface.derivatives(best_u, best_v, 1);
                    if derivs.len() < 2 || derivs[0].len() < 2 || derivs[1].is_empty() {
                        break;
                    }
                    let du = derivs[1][0]; // ∂S/∂u
                    let dv = derivs[0][1]; // ∂S/∂v
                    let diff = p - target;

                    // Jacobian entries
                    let j00 = du.dot(du);
                    let j01 = du.dot(dv);
                    let j10 = j01;
                    let j11 = dv.dot(dv);
                    let r0 = diff.x() * du.x() + diff.y() * du.y() + diff.z() * du.z();
                    let r1 = diff.x() * dv.x() + diff.y() * dv.y() + diff.z() * dv.z();

                    let det = j00 * j11 - j01 * j10;
                    if det.abs() < 1e-20 {
                        break;
                    }
                    let delta_u = -(j11 * r0 - j01 * r1) / det;
                    let delta_v = -(-j10 * r0 + j00 * r1) / det;

                    best_u = (best_u + delta_u).clamp(u0, u1);
                    best_v = (best_v + delta_v).clamp(v0, v1);
                }

                let proj = surface.evaluate(best_u, best_v);
                let dist = (proj - target).length();
                Ok(vec![best_u, best_v, proj.x(), proj.y(), proj.z(), dist])
            }
            _ => {
                // For analytic surfaces, use grid search (no Newton for now)
                let mut best_u = 0.0;
                let mut best_v = 0.0;
                let mut best_dist = f64::MAX;
                let n_grid = 16;
                for iu in 0..=n_grid {
                    for iv in 0..=n_grid {
                        #[allow(clippy::cast_precision_loss)]
                        let u = 2.0 * PI * (iu as f64) / (n_grid as f64);
                        #[allow(clippy::cast_precision_loss)]
                        let v = -PI + 2.0 * PI * (iv as f64) / (n_grid as f64);
                        let p = match face_data.surface() {
                            FaceSurface::Cylinder(cyl) => cyl.evaluate(u, v),
                            FaceSurface::Cone(cone) => cone.evaluate(u, v),
                            FaceSurface::Sphere(sph) => sph.evaluate(u, v),
                            FaceSurface::Torus(tor) => tor.evaluate(u, v),
                            _ => continue,
                        };
                        let d = (p - target).length();
                        if d < best_dist {
                            best_dist = d;
                            best_u = u;
                            best_v = v;
                        }
                    }
                }
                let proj = match face_data.surface() {
                    FaceSurface::Cylinder(cyl) => cyl.evaluate(best_u, best_v),
                    FaceSurface::Cone(cone) => cone.evaluate(best_u, best_v),
                    FaceSurface::Sphere(sph) => sph.evaluate(best_u, best_v),
                    FaceSurface::Torus(tor) => tor.evaluate(best_u, best_v),
                    _ => target,
                };
                Ok(vec![
                    best_u,
                    best_v,
                    proj.x(),
                    proj.y(),
                    proj.z(),
                    best_dist,
                ])
            }
        }
    }

    /// Add hole wires to an existing face, creating a new face with the same
    /// surface but additional inner wires.
    ///
    /// Returns a new face handle (`u32`).
    #[wasm_bindgen(js_name = "addHolesToFace")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn add_holes_to_face(
        &mut self,
        face: u32,
        hole_wire_handles: Vec<u32>,
    ) -> Result<u32, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let outer_wire = face_data.outer_wire();
        let surface = face_data.surface().clone();
        let mut inner_wires: Vec<brepkit_topology::wire::WireId> = face_data.inner_wires().to_vec();

        for &wh in &hole_wire_handles {
            let wid = self.resolve_wire(wh)?;
            inner_wires.push(wid);
        }

        let new_face = Face::new(outer_wire, inner_wires, surface);
        let fid = self.topo.faces.alloc(new_face);
        Ok(face_id_to_u32(fid))
    }

    /// Sweep a face along a path defined by a chain of edges.
    ///
    /// Collects points from the edges, fits an interpolating NURBS curve,
    /// then sweeps the profile along that curve.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 edges or the fit fails.
    #[wasm_bindgen(js_name = "sweepAlongEdges")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn sweep_along_edges(&mut self, face: u32, edge_handles: Vec<u32>) -> Result<u32, JsError> {
        if edge_handles.is_empty() {
            return Err(WasmError::InvalidInput {
                reason: "sweepAlongEdges requires at least one edge".into(),
            }
            .into());
        }

        // Collect ordered points from the edge chain.
        let mut points = Vec::new();
        for &eh in &edge_handles {
            let eid = self.resolve_edge(eh)?;
            let edge_data = self.topo.edge(eid)?;
            let start = self.topo.vertex(edge_data.start())?.point();

            // Only push start if it's not a duplicate of the last point.
            if points
                .last()
                .is_none_or(|p: &Point3| (*p - start).length() > TOL)
            {
                points.push(start);
            }

            // For NURBS edges, sample interior points for better fidelity.
            if let EdgeCurve::NurbsCurve(curve) = edge_data.curve() {
                let (u0, u1) = curve.domain();
                let n_samples = 4;
                for i in 1..n_samples {
                    #[allow(clippy::cast_precision_loss)]
                    let frac = i as f64 / n_samples as f64;
                    let u = u0 + frac * (u1 - u0);
                    points.push(curve.evaluate(u));
                }
            }

            let end = self.topo.vertex(edge_data.end())?.point();
            points.push(end);
        }

        if points.len() < 2 {
            return Err(WasmError::InvalidInput {
                reason: "sweepAlongEdges: need at least 2 distinct points".into(),
            }
            .into());
        }

        // Fit an interpolating NURBS curve through the points.
        let degree = std::cmp::min(3, points.len() - 1);
        let path_curve = brepkit_math::nurbs::fitting::interpolate(&points, degree)?;

        let face_id = self.resolve_face(face)?;
        let solid_id = sweep(&mut self.topo, face_id, &path_curve)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Build a convex hull solid from a point cloud.
    ///
    /// Uses a simple Quickhull-inspired algorithm for 3D point sets.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 4 points are provided.
    #[wasm_bindgen(js_name = "convexHull")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn convex_hull_wasm(&mut self, coords: Vec<f64>) -> Result<u32, JsError> {
        if coords.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "coordinate array length must be a multiple of 3, got {}",
                    coords.len()
                ),
            }
            .into());
        }
        let points: Vec<Point3> = coords
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();
        if points.len() < 4 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "convex hull requires at least 4 points, got {}",
                    points.len()
                ),
            }
            .into());
        }

        // Build a proper convex hull using Quickhull.
        let hull = brepkit_math::convex_hull::convex_hull_3d(&points).ok_or_else(|| {
            WasmError::InvalidInput {
                reason: "points are coplanar or degenerate — cannot form a 3D convex hull".into(),
            }
        })?;

        // Convert hull to B-Rep: vertices → edges → faces → shell → solid.
        let vertex_ids: Vec<brepkit_topology::vertex::VertexId> = hull
            .vertices
            .iter()
            .map(|p| self.topo.vertices.alloc(Vertex::new(*p, TOL)))
            .collect();

        let mut face_ids = Vec::new();
        for &[a, b, c] in &hull.faces {
            let va = vertex_ids[a];
            let vb = vertex_ids[b];
            let vc = vertex_ids[c];

            let e0 = self.topo.edges.alloc(Edge::new(va, vb, EdgeCurve::Line));
            let e1 = self.topo.edges.alloc(Edge::new(vb, vc, EdgeCurve::Line));
            let e2 = self.topo.edges.alloc(Edge::new(vc, va, EdgeCurve::Line));

            let oriented = vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
            ];
            let wire = Wire::new(oriented, true)?;
            let wid = self.topo.wires.alloc(wire);

            // Compute face normal.
            let pa = hull.vertices[a];
            let pb = hull.vertices[b];
            let pc = hull.vertices[c];
            let ab = pb - pa;
            let ac = pc - pa;
            let normal = ab.cross(ac);
            let normal = match normal.normalize() {
                Ok(n) => n,
                Err(_) => Vec3::new(0.0, 0.0, 1.0),
            };
            let d = normal
                .x()
                .mul_add(pa.x(), normal.y().mul_add(pa.y(), normal.z() * pa.z()));

            let fid =
                self.topo
                    .faces
                    .alloc(Face::new(wid, vec![], FaceSurface::Plane { normal, d }));
            face_ids.push(fid);
        }

        let shell = brepkit_topology::shell::Shell::new(face_ids)?;
        let shell_id = self.topo.shells.alloc(shell);
        let solid = brepkit_topology::solid::Solid::new(shell_id, vec![]);
        let solid_id = self.topo.solids.alloc(solid);

        Ok(solid_id_to_u32(solid_id))
    }

    /// Interpolate a NURBS curve through points and create an edge.
    ///
    /// Uses chord-length parameterization with the given degree.
    /// Returns an edge handle (`u32`).
    #[wasm_bindgen(js_name = "interpolatePoints")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn interpolate_points_wasm(
        &mut self,
        coords: Vec<f64>,
        degree: u32,
    ) -> Result<u32, JsError> {
        if coords.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "coordinate array length must be a multiple of 3, got {}",
                    coords.len()
                ),
            }
            .into());
        }
        let points: Vec<Point3> = coords
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();
        if points.len() < 2 {
            return Err(WasmError::InvalidInput {
                reason: format!("need at least 2 points, got {}", points.len()),
            }
            .into());
        }

        let deg = std::cmp::min(degree as usize, points.len() - 1);
        let curve = brepkit_math::nurbs::fitting::interpolate(&points, deg)?;

        let start = points[0];
        let end = points[points.len() - 1];
        let v_start = self.topo.vertices.alloc(Vertex::new(start, TOL));
        let v_end = self.topo.vertices.alloc(Vertex::new(end, TOL));
        let eid = self
            .topo
            .edges
            .alloc(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(curve)));
        Ok(edge_id_to_u32(eid))
    }

    /// Offset (shell) a solid by a distance.
    ///
    /// Returns a new solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the distance is zero or the solid is invalid.
    #[wasm_bindgen(js_name = "offsetSolid")]
    pub fn offset_solid_wasm(&mut self, solid: u32, distance: f64) -> Result<u32, JsError> {
        validate_finite(distance, "distance")?;
        let solid_id = self.resolve_solid(solid)?;
        let result =
            brepkit_operations::offset_solid::offset_solid(&mut self.topo, solid_id, distance)?;
        Ok(solid_id_to_u32(result))
    }

    /// Build an edge's NURBS curve data for JS consumption.
    ///
    /// Returns `null` for line edges, or a JSON string with
    /// `{degree, knots, controlPoints, weights}` for NURBS edges.
    #[wasm_bindgen(js_name = "getEdgeNurbsData")]
    pub fn get_edge_nurbs_data(&self, edge: u32) -> Result<JsValue, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        match edge_data.curve() {
            EdgeCurve::Line => Ok(JsValue::NULL),
            EdgeCurve::NurbsCurve(curve) => {
                let cp_flat: Vec<f64> = curve
                    .control_points()
                    .iter()
                    .flat_map(|p| [p.x(), p.y(), p.z()])
                    .collect();
                let data = serde_json::json!({
                    "degree": curve.degree(),
                    "knots": curve.knots(),
                    "controlPoints": cp_flat,
                    "weights": curve.weights(),
                });
                Ok(JsValue::from_str(&data.to_string()))
            }
        }
    }

    // ── Batch execution ──────────────────────────────────────────

    /// Execute a batch of operations, crossing the JS/WASM boundary once.
    ///
    /// Accepts a JSON string containing an array of operation objects:
    /// ```json
    /// [
    ///   {"op": "makeBox", "args": {"width": 2.0, "height": 2.0, "depth": 2.0}},
    ///   {"op": "fuse", "args": {"solidA": 0, "solidB": 1}},
    ///   {"op": "volume", "args": {"solid": 2, "deflection": 0.1}}
    /// ]
    /// ```
    ///
    /// Returns a JSON string with an array of results:
    /// ```json
    /// [
    ///   {"ok": 0},
    ///   {"ok": 2},
    ///   {"error": "invalid solid id"}
    /// ]
    /// ```
    ///
    /// Operations are executed sequentially; an error in one does not
    /// prevent execution of subsequent operations.
    #[wasm_bindgen(js_name = "executeBatch")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn execute_batch(&mut self, json: &str) -> String {
        let ops: Vec<serde_json::Value> = match serde_json::from_str(json) {
            Ok(v) => v,
            Err(e) => {
                return serde_json::json!([{"error": format!("invalid JSON: {e}")}]).to_string();
            }
        };

        let results: Vec<serde_json::Value> = ops
            .iter()
            .map(|entry| {
                let op = match entry["op"].as_str() {
                    Some(s) => s,
                    None => return serde_json::json!({"error": "missing or invalid 'op' field"}),
                };
                let args = &entry["args"];
                match self.dispatch_op(op, args) {
                    Ok(val) => serde_json::json!({"ok": val}),
                    Err(msg) => serde_json::json!({"error": msg}),
                }
            })
            .collect();

        serde_json::Value::Array(results).to_string()
    }
}

// ── Private helpers ────────────────────────────────────────────────

impl BrepKernel {
    /// Build a closed planar face from an ordered sequence of points.
    fn make_planar_face(
        &mut self,
        points: &[Point3],
    ) -> Result<brepkit_topology::face::FaceId, WasmError> {
        let n = points.len();

        // Allocate vertices.
        let verts: Vec<_> = points
            .iter()
            .map(|p| self.topo.vertices.alloc(Vertex::new(*p, TOL)))
            .collect();

        // Allocate edges connecting consecutive vertices.
        let edges: Vec<_> = (0..n)
            .map(|i| {
                let next = (i + 1) % n;
                self.topo
                    .edges
                    .alloc(Edge::new(verts[i], verts[next], EdgeCurve::Line))
            })
            .collect();

        // Build oriented edge list and closed wire.
        let oriented: Vec<_> = edges
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let wire = Wire::new(oriented, true)?;
        let wid = self.topo.wires.alloc(wire);

        // Compute the face normal from the first three points.
        let a = points[1] - points[0];
        let b = points[2] - points[0];
        let normal = a.cross(b).normalize()?;

        // Plane equation: n · p = d
        let d = normal.x().mul_add(
            points[0].x(),
            normal
                .y()
                .mul_add(points[0].y(), normal.z() * points[0].z()),
        );

        let face_id =
            self.topo
                .faces
                .alloc(Face::new(wid, vec![], FaceSurface::Plane { normal, d }));

        Ok(face_id)
    }

    /// Resolve a `u32` face handle to a typed `FaceId`.
    fn resolve_face(&self, handle: u32) -> Result<brepkit_topology::face::FaceId, WasmError> {
        let index = handle as usize;
        self.topo
            .faces
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "face",
                index,
            })
    }

    /// Resolve a `u32` vertex handle to a typed `VertexId`.
    fn resolve_vertex(&self, handle: u32) -> Result<brepkit_topology::vertex::VertexId, WasmError> {
        let index = handle as usize;
        self.topo
            .vertices
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "vertex",
                index,
            })
    }

    /// Resolve a `u32` edge handle to a typed `EdgeId`.
    fn resolve_edge(&self, handle: u32) -> Result<brepkit_topology::edge::EdgeId, WasmError> {
        let index = handle as usize;
        self.topo
            .edges
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "edge",
                index,
            })
    }

    /// Resolve a `u32` solid handle to a typed `SolidId`.
    fn resolve_solid(&self, handle: u32) -> Result<brepkit_topology::solid::SolidId, WasmError> {
        let index = handle as usize;
        self.topo
            .solids
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "solid",
                index,
            })
    }

    /// Resolve a `u32` wire handle to a typed `WireId`.
    fn resolve_wire(&self, handle: u32) -> Result<brepkit_topology::wire::WireId, WasmError> {
        let index = handle as usize;
        self.topo
            .wires
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "wire",
                index,
            })
    }

    /// Resolve a `u32` shell handle to a typed `ShellId`.
    fn resolve_shell(&self, handle: u32) -> Result<brepkit_topology::shell::ShellId, WasmError> {
        let index = handle as usize;
        self.topo
            .shells
            .id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "shell",
                index,
            })
    }
}

// ── New WASM bindings (Batches 1–6) ─────────────────────────────────

#[wasm_bindgen]
impl BrepKernel {
    // ── Batch 1: Math WASM bindings ────────────────────────────────

    /// Approximate a curve through points (least-squares).
    ///
    /// Returns an edge handle.
    #[wasm_bindgen(js_name = "approximateCurve")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn approximate_curve(
        &mut self,
        coords: Vec<f64>,
        degree: u32,
        num_control_points: u32,
    ) -> Result<u32, JsError> {
        let points = parse_points(&coords)?;
        if points.len() < 2 {
            return Err(WasmError::InvalidInput {
                reason: format!("need at least 2 points, got {}", points.len()),
            }
            .into());
        }
        let deg = std::cmp::min(degree as usize, points.len() - 1);
        let curve =
            brepkit_math::nurbs::fitting::approximate(&points, deg, num_control_points as usize)?;
        Ok(edge_id_to_u32(self.nurbs_curve_to_edge(&points, curve)))
    }

    /// Approximate a curve through points using LSPIA (progressive iteration).
    ///
    /// Returns an edge handle.
    #[wasm_bindgen(js_name = "approximateCurveLspia")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn approximate_curve_lspia(
        &mut self,
        coords: Vec<f64>,
        degree: u32,
        num_control_points: u32,
        tolerance: f64,
        max_iterations: u32,
    ) -> Result<u32, JsError> {
        let points = parse_points(&coords)?;
        if points.len() < 2 {
            return Err(WasmError::InvalidInput {
                reason: format!("need at least 2 points, got {}", points.len()),
            }
            .into());
        }
        let deg = std::cmp::min(degree as usize, points.len() - 1);
        let curve = brepkit_math::nurbs::fitting::approximate_lspia(
            &points,
            deg,
            num_control_points as usize,
            tolerance,
            max_iterations as usize,
        )?;
        Ok(edge_id_to_u32(self.nurbs_curve_to_edge(&points, curve)))
    }

    /// Interpolate a grid of points into a NURBS surface.
    ///
    /// `coords` is a flat array `[x,y,z, ...]` of `rows * cols` points.
    /// Returns a face handle.
    #[wasm_bindgen(js_name = "interpolateSurface")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn interpolate_surface(
        &mut self,
        coords: Vec<f64>,
        rows: u32,
        cols: u32,
        degree_u: u32,
        degree_v: u32,
    ) -> Result<u32, JsError> {
        let grid = parse_point_grid(&coords, rows as usize, cols as usize)?;
        let surface = brepkit_math::nurbs::surface_fitting::interpolate_surface(
            &grid,
            degree_u as usize,
            degree_v as usize,
        )?;
        Ok(face_id_to_u32(self.nurbs_surface_to_face(surface)?))
    }

    /// Approximate a grid of points into a NURBS surface using LSPIA.
    ///
    /// Returns a face handle.
    #[wasm_bindgen(js_name = "approximateSurfaceLspia")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn approximate_surface_lspia(
        &mut self,
        coords: Vec<f64>,
        rows: u32,
        cols: u32,
        degree_u: u32,
        degree_v: u32,
        num_cps_u: u32,
        num_cps_v: u32,
        tolerance: f64,
        max_iterations: u32,
    ) -> Result<u32, JsError> {
        let grid = parse_point_grid(&coords, rows as usize, cols as usize)?;
        let surface = brepkit_math::nurbs::surface_fitting::approximate_surface_lspia(
            &grid,
            degree_u as usize,
            degree_v as usize,
            num_cps_u as usize,
            num_cps_v as usize,
            tolerance,
            max_iterations as usize,
        )?;
        Ok(face_id_to_u32(self.nurbs_surface_to_face(surface)?))
    }

    /// Insert a knot into an edge's NURBS curve.
    ///
    /// Returns a new edge handle with the refined curve.
    #[wasm_bindgen(js_name = "curveKnotInsert")]
    pub fn curve_knot_insert(&mut self, edge: u32, knot: f64, times: u32) -> Result<u32, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        let refined =
            brepkit_math::nurbs::knot_ops::curve_knot_insert(&curve, knot, times as usize)?;
        Ok(edge_id_to_u32(
            self.nurbs_curve_to_edge_from_curve(&refined),
        ))
    }

    /// Remove a knot from an edge's NURBS curve.
    ///
    /// Returns a new edge handle with the simplified curve.
    #[wasm_bindgen(js_name = "curveKnotRemove")]
    pub fn curve_knot_remove(
        &mut self,
        edge: u32,
        knot: f64,
        tolerance: f64,
    ) -> Result<u32, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        let simplified = brepkit_math::nurbs::knot_ops::curve_knot_remove(&curve, knot, tolerance)?;
        Ok(edge_id_to_u32(
            self.nurbs_curve_to_edge_from_curve(&simplified),
        ))
    }

    /// Split an edge's NURBS curve at a parameter value.
    ///
    /// Returns two edge handles as `[u32; 2]`.
    #[wasm_bindgen(js_name = "curveSplit")]
    pub fn curve_split(&mut self, edge: u32, u: f64) -> Result<Vec<u32>, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        let (left, right) = brepkit_math::nurbs::knot_ops::curve_split(&curve, u)?;
        let e1 = self.nurbs_curve_to_edge_from_curve(&left);
        let e2 = self.nurbs_curve_to_edge_from_curve(&right);
        Ok(vec![edge_id_to_u32(e1), edge_id_to_u32(e2)])
    }

    /// Elevate the degree of an edge's NURBS curve.
    ///
    /// Returns a new edge handle.
    #[wasm_bindgen(js_name = "curveDegreeElevate")]
    pub fn curve_degree_elevate(&mut self, edge: u32, elevate_by: u32) -> Result<u32, JsError> {
        let curve = self.extract_nurbs_curve(edge)?;
        let elevated =
            brepkit_math::nurbs::decompose::curve_degree_elevate(&curve, elevate_by as usize)?;
        Ok(edge_id_to_u32(
            self.nurbs_curve_to_edge_from_curve(&elevated),
        ))
    }

    // ── Batch 2: Topology query bindings ─────────────────────────────

    /// Create a closed polygon wire from flat coordinates.
    ///
    /// Returns a wire handle.
    #[wasm_bindgen(js_name = "makePolygonWire")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_polygon_wire(&mut self, coords: Vec<f64>) -> Result<u32, JsError> {
        let points = parse_points(&coords)?;
        if points.len() < 3 {
            return Err(WasmError::InvalidInput {
                reason: format!("polygon wire needs at least 3 points, got {}", points.len()),
            }
            .into());
        }
        let n = points.len();
        let verts: Vec<_> = points
            .iter()
            .map(|p| self.topo.vertices.alloc(Vertex::new(*p, TOL)))
            .collect();
        let edges: Vec<_> = (0..n)
            .map(|i| {
                self.topo
                    .edges
                    .alloc(Edge::new(verts[i], verts[(i + 1) % n], EdgeCurve::Line))
            })
            .collect();
        let oriented: Vec<_> = edges
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let wire = Wire::new(oriented, true)?;
        let wid = self.topo.wires.alloc(wire);
        Ok(wire_id_to_u32(wid))
    }

    /// Create a regular polygon wire on the XY plane.
    ///
    /// Returns a wire handle.
    #[wasm_bindgen(js_name = "makeRegularPolygonWire")]
    pub fn make_regular_polygon_wire(&mut self, radius: f64, n_sides: u32) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        if n_sides < 3 {
            return Err(WasmError::InvalidInput {
                reason: format!("polygon needs at least 3 sides, got {n_sides}"),
            }
            .into());
        }
        let wid = brepkit_topology::builder::make_regular_polygon_wire(
            &mut self.topo,
            radius,
            n_sides as usize,
        )?;
        Ok(wire_id_to_u32(wid))
    }

    /// Create a circular face on the XY plane (using NURBS arcs).
    ///
    /// Returns a face handle.
    #[wasm_bindgen(js_name = "makeCircleFace")]
    pub fn make_circle_face_wasm(&mut self, radius: f64, segments: u32) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        if segments < 3 {
            return Err(WasmError::InvalidInput {
                reason: format!("circle face needs at least 3 segments, got {segments}"),
            }
            .into());
        }
        let fid =
            brepkit_topology::builder::make_circle_face(&mut self.topo, radius, segments as usize)?;
        Ok(face_id_to_u32(fid))
    }

    /// Get the edge-to-face adjacency map for a solid.
    ///
    /// Returns a JSON string: `{"edgeId": [faceId, ...], ...}`.
    #[wasm_bindgen(js_name = "edgeToFaceMap")]
    pub fn edge_to_face_map(&self, solid: u32) -> Result<String, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let map = brepkit_topology::explorer::edge_to_face_map(&self.topo, solid_id)?;
        let json_map: std::collections::HashMap<String, Vec<u32>> = map
            .into_iter()
            .map(|(edge_idx, face_ids)| {
                let fids: Vec<u32> = face_ids.iter().map(|f| face_id_to_u32(*f)).collect();
                (edge_idx.to_string(), fids)
            })
            .collect();
        Ok(serde_json::json!(json_map).to_string())
    }

    /// Get edges shared between two faces.
    ///
    /// Returns an array of edge handles.
    #[wasm_bindgen(js_name = "sharedEdges")]
    pub fn shared_edges(&self, face_a: u32, face_b: u32) -> Result<Vec<u32>, JsError> {
        let fa = self.resolve_face(face_a)?;
        let fb = self.resolve_face(face_b)?;
        let edges = brepkit_topology::explorer::shared_edges(&self.topo, fa, fb)?;
        Ok(edges.iter().map(|e| edge_id_to_u32(*e)).collect())
    }

    /// Get faces adjacent to a given face within a solid.
    ///
    /// Returns an array of face handles.
    #[wasm_bindgen(js_name = "adjacentFaces")]
    pub fn adjacent_faces(&self, solid: u32, face: u32) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let face_id = self.resolve_face(face)?;
        let map = brepkit_topology::explorer::edge_to_face_map(&self.topo, solid_id)?;
        let adj = brepkit_topology::explorer::adjacent_faces(&self.topo, face_id, &map)?;
        Ok(adj.iter().map(|f| face_id_to_u32(*f)).collect())
    }

    /// Get the wires (outer + inner) of a face.
    ///
    /// Returns an array of wire handles.
    #[wasm_bindgen(js_name = "faceWires")]
    pub fn face_wires(&self, face: u32) -> Result<Vec<u32>, JsError> {
        let face_id = self.resolve_face(face)?;
        let wires = brepkit_topology::explorer::face_wires(&self.topo, face_id)?;
        Ok(wires.iter().map(|w| wire_id_to_u32(*w)).collect())
    }

    // ── Batch 3: Simple operations bindings ──────────────────────────

    /// Create a circular pattern of a solid around an axis.
    ///
    /// Returns a compound handle.
    #[wasm_bindgen(js_name = "circularPattern")]
    pub fn circular_pattern(
        &mut self,
        solid: u32,
        ax: f64,
        ay: f64,
        az: f64,
        count: u32,
    ) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let axis = Vec3::new(ax, ay, az);
        let compound = brepkit_operations::pattern::circular_pattern(
            &mut self.topo,
            solid_id,
            axis,
            count as usize,
        )?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(compound.index() as u32)
    }

    /// Merge coincident vertices in a solid.
    ///
    /// Returns the number of vertices merged.
    #[wasm_bindgen(js_name = "mergeCoincidentVertices")]
    pub fn merge_coincident_vertices(
        &mut self,
        solid: u32,
        tolerance: f64,
    ) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let count = brepkit_operations::heal::merge_coincident_vertices(
            &mut self.topo,
            solid_id,
            tolerance,
        )?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(count as u32)
    }

    /// Remove degenerate (zero-length) edges from a solid.
    ///
    /// Returns the number of edges removed.
    #[wasm_bindgen(js_name = "removeDegenerateEdges")]
    pub fn remove_degenerate_edges(&mut self, solid: u32, tolerance: f64) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let count =
            brepkit_operations::heal::remove_degenerate_edges(&mut self.topo, solid_id, tolerance)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(count as u32)
    }

    /// Fix face orientations to ensure consistent outward normals.
    ///
    /// Returns the number of faces fixed.
    #[wasm_bindgen(js_name = "fixFaceOrientations")]
    pub fn fix_face_orientations(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let count = brepkit_operations::heal::fix_face_orientations(&mut self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(count as u32)
    }

    /// Remove specified faces from a solid (defeaturing).
    ///
    /// `face_handles` is an array of face handles to remove.
    /// Returns a new solid handle.
    #[wasm_bindgen(js_name = "defeature")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn defeature(&mut self, solid: u32, face_handles: Vec<u32>) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let face_ids: Vec<_> = face_handles
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<Vec<_>, _>>()?;
        let result = brepkit_operations::defeature::defeature(&mut self.topo, solid_id, &face_ids)?;
        Ok(solid_id_to_u32(result))
    }

    /// Detect small features (faces below an area threshold).
    ///
    /// Returns an array of face handles.
    #[wasm_bindgen(js_name = "detectSmallFeatures")]
    pub fn detect_small_features(
        &self,
        solid: u32,
        area_threshold: f64,
        deflection: f64,
    ) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let faces = brepkit_operations::defeature::detect_small_features(
            &self.topo,
            solid_id,
            area_threshold,
            deflection,
        )?;
        Ok(faces.iter().map(|f| face_id_to_u32(*f)).collect())
    }

    /// Recognize geometric features in a solid.
    ///
    /// Returns a JSON string describing the recognized features.
    #[wasm_bindgen(js_name = "recognizeFeatures")]
    pub fn recognize_features(&self, solid: u32, deflection: f64) -> Result<String, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let features = brepkit_operations::feature_recognition::recognize_features(
            &self.topo, solid_id, deflection,
        )?;
        let json_features: Vec<serde_json::Value> =
            features.iter().map(|f| serialize_feature(f)).collect();
        Ok(serde_json::Value::Array(json_features).to_string())
    }

    // ── Batch 4: Complex operations bindings ─────────────────────────

    /// Apply variable-radius fillets to edges.
    ///
    /// `json` is a JSON string: `[{"edge": u32, "law": "constant"|"linear"|"scurve", "start": f64, "end": f64}]`
    /// Returns a new solid handle.
    #[wasm_bindgen(js_name = "filletVariable")]
    pub fn fillet_variable(&mut self, solid: u32, json: &str) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let specs: Vec<serde_json::Value> =
            serde_json::from_str(json).map_err(|e| WasmError::InvalidInput {
                reason: format!("invalid JSON: {e}"),
            })?;
        let mut edge_laws = Vec::with_capacity(specs.len());
        for spec in &specs {
            let edge_handle = spec["edge"]
                .as_u64()
                .ok_or_else(|| WasmError::InvalidInput {
                    reason: "missing 'edge' in fillet spec".into(),
                })? as u32;
            let edge_id = self.resolve_edge(edge_handle)?;
            let law_str = spec["law"].as_str().unwrap_or("constant");
            let law = match law_str {
                "linear" => {
                    let s = spec["start"].as_f64().unwrap_or(1.0);
                    let e = spec["end"].as_f64().unwrap_or(1.0);
                    brepkit_operations::fillet::FilletRadiusLaw::Linear { start: s, end: e }
                }
                "scurve" => {
                    let s = spec["start"].as_f64().unwrap_or(1.0);
                    let e = spec["end"].as_f64().unwrap_or(1.0);
                    brepkit_operations::fillet::FilletRadiusLaw::SCurve { start: s, end: e }
                }
                _ => {
                    let r = spec["radius"]
                        .as_f64()
                        .or_else(|| spec["start"].as_f64())
                        .unwrap_or(1.0);
                    brepkit_operations::fillet::FilletRadiusLaw::Constant(r)
                }
            };
            edge_laws.push((edge_id, law));
        }
        let result =
            brepkit_operations::fillet::fillet_variable(&mut self.topo, solid_id, &edge_laws)?;
        Ok(solid_id_to_u32(result))
    }

    /// Sweep a face along a NURBS path with advanced options.
    ///
    /// `contact_mode`: "rmf" (default), "fixed", or "constantNormal:x,y,z"
    /// `scale_values`: flat `[t0,s0,t1,s1,...]` pairs for piecewise-linear scale law.
    /// Returns a solid handle.
    #[wasm_bindgen(js_name = "sweepWithOptions")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn sweep_with_options(
        &mut self,
        profile: u32,
        path_edge: u32,
        contact_mode: &str,
        scale_values: Vec<f64>,
        segments: u32,
    ) -> Result<u32, JsError> {
        use brepkit_operations::sweep::{SweepContactMode, SweepOptions};

        let face_id = self.resolve_face(profile)?;
        let path_curve = self.extract_nurbs_curve(path_edge)?;

        let mode = if contact_mode == "fixed" {
            SweepContactMode::Fixed
        } else if let Some(rest) = contact_mode.strip_prefix("constantNormal:") {
            let parts: Vec<f64> = rest
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if parts.len() >= 3 {
                SweepContactMode::ConstantNormal(Vec3::new(parts[0], parts[1], parts[2]))
            } else {
                SweepContactMode::RotationMinimizing
            }
        } else {
            SweepContactMode::RotationMinimizing
        };

        let scale_law: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>> =
            if scale_values.len() >= 4 && scale_values.len() % 2 == 0 {
                let pairs: Vec<(f64, f64)> =
                    scale_values.chunks_exact(2).map(|c| (c[0], c[1])).collect();
                Some(Box::new(move |t: f64| -> f64 {
                    // Piecewise-linear interpolation
                    if pairs.is_empty() {
                        return 1.0;
                    }
                    if t <= pairs[0].0 {
                        return pairs[0].1;
                    }
                    if t >= pairs[pairs.len() - 1].0 {
                        return pairs[pairs.len() - 1].1;
                    }
                    for w in pairs.windows(2) {
                        if t >= w[0].0 && t <= w[1].0 {
                            let frac = (t - w[0].0) / (w[1].0 - w[0].0);
                            return w[0].1 + frac * (w[1].1 - w[0].1);
                        }
                    }
                    1.0
                }))
            } else {
                None
            };

        let options = SweepOptions {
            contact_mode: mode,
            scale_law,
            segments: segments as usize,
        };

        let result = brepkit_operations::sweep::sweep_with_options(
            &mut self.topo,
            face_id,
            &path_curve,
            &options,
        )?;
        Ok(solid_id_to_u32(result))
    }

    /// Classify a point relative to a solid using generalized winding numbers.
    ///
    /// Returns "inside", "outside", or "boundary".
    #[wasm_bindgen(js_name = "classifyPointWinding")]
    pub fn classify_point_winding(
        &self,
        solid: u32,
        x: f64,
        y: f64,
        z: f64,
        tolerance: f64,
    ) -> Result<String, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let point = Point3::new(x, y, z);
        let result = brepkit_operations::classify::classify_point_winding(
            &self.topo, solid_id, point, 0.1, tolerance,
        )?;
        Ok(classify_to_string(result))
    }

    /// Classify a point using robust dual-method (winding + ray casting).
    ///
    /// Returns "inside", "outside", or "boundary".
    #[wasm_bindgen(js_name = "classifyPointRobust")]
    pub fn classify_point_robust(
        &self,
        solid: u32,
        x: f64,
        y: f64,
        z: f64,
        tolerance: f64,
    ) -> Result<String, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let point = Point3::new(x, y, z);
        let result = brepkit_operations::classify::classify_point_robust(
            &self.topo, solid_id, point, 0.1, tolerance,
        )?;
        Ok(classify_to_string(result))
    }

    /// Perform a mesh boolean on raw triangle data.
    ///
    /// Returns a `JsMesh` with the result.
    #[wasm_bindgen(js_name = "meshBoolean")]
    #[allow(
        clippy::needless_pass_by_value,
        clippy::too_many_arguments,
        clippy::unused_self
    )]
    pub fn mesh_boolean(
        &self,
        positions_a: Vec<f64>,
        indices_a: Vec<u32>,
        positions_b: Vec<f64>,
        indices_b: Vec<u32>,
        op: &str,
        tolerance: f64,
    ) -> Result<JsMesh, JsError> {
        let mesh_a = build_triangle_mesh(&positions_a, &indices_a)?;
        let mesh_b = build_triangle_mesh(&positions_b, &indices_b)?;
        let bool_op = parse_boolean_op(op)?;
        let result =
            brepkit_operations::mesh_boolean::mesh_boolean(&mesh_a, &mesh_b, bool_op, tolerance)?;
        Ok(triangle_mesh_to_js(&result.mesh))
    }

    /// Fill a 4-sided boundary with a Coons patch surface.
    ///
    /// `boundary_coords` is flat `[x,y,z, ...]` for all 4 curves concatenated.
    /// `curve_lengths` is `[n0, n1, n2, n3]` — number of points per curve.
    /// Returns a face handle.
    #[wasm_bindgen(js_name = "fillCoonsPatch")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn fill_coons_patch(
        &mut self,
        boundary_coords: Vec<f64>,
        curve_lengths: Vec<u32>,
    ) -> Result<u32, JsError> {
        if curve_lengths.len() != 4 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "Coons patch requires exactly 4 boundary curves, got {}",
                    curve_lengths.len()
                ),
            }
            .into());
        }
        let points = parse_points(&boundary_coords)?;
        let mut curves: Vec<Vec<Point3>> = Vec::with_capacity(4);
        let mut offset = 0usize;
        for &len in &curve_lengths {
            let l = len as usize;
            if offset + l > points.len() {
                return Err(WasmError::InvalidInput {
                    reason: "curve_lengths exceed total coordinate count".into(),
                }
                .into());
            }
            curves.push(points[offset..offset + l].to_vec());
            offset += l;
        }
        let face_id = brepkit_operations::fill_face::fill_coons_patch(&mut self.topo, &curves)?;
        Ok(face_id_to_u32(face_id))
    }

    /// Untrim a NURBS face by fitting a new surface to the trimmed region.
    ///
    /// Returns a new face handle.
    #[wasm_bindgen(js_name = "untrimFace")]
    pub fn untrim_face(
        &mut self,
        face: u32,
        samples_per_curve: u32,
        interior_samples: u32,
    ) -> Result<u32, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let surface = match face_data.surface() {
            FaceSurface::Nurbs(s) => s.clone(),
            _ => {
                return Err(WasmError::InvalidInput {
                    reason: "untrim only works on NURBS faces".into(),
                }
                .into());
            }
        };
        // Build trim curves from wire edges projected to UV space
        let wire_id = face_data.outer_wire();
        let wire = self.topo.wire(wire_id)?;
        let mut trim_curves = Vec::new();
        for oe in wire.edges() {
            let edge = self.topo.edge(oe.edge())?;
            let v_start = self.topo.vertex(edge.start())?;
            let v_end = self.topo.vertex(edge.end())?;
            // Project endpoints to UV
            let uv_start = project_to_uv(&surface, v_start.point());
            let uv_end = project_to_uv(&surface, v_end.point());
            trim_curves.push(brepkit_operations::untrim::TrimCurve {
                curve: vec![uv_start, uv_end],
            });
        }
        let new_surface = brepkit_operations::untrim::untrim_face(
            &surface,
            &trim_curves,
            samples_per_curve as usize,
            interior_samples as usize,
        )?;
        Ok(face_id_to_u32(self.nurbs_surface_to_face(new_surface)?))
    }

    /// Offset a wire on a planar face.
    ///
    /// Returns a new wire handle.
    #[wasm_bindgen(js_name = "offsetWire")]
    pub fn offset_wire(&mut self, face: u32, distance: f64) -> Result<u32, JsError> {
        let face_id = self.resolve_face(face)?;
        let wire_id =
            brepkit_operations::offset_wire::offset_wire(&mut self.topo, face_id, distance)?;
        Ok(wire_id_to_u32(wire_id))
    }

    /// Get analytic surface parameters for a face.
    ///
    /// Returns a JSON string with type-dependent fields.
    #[wasm_bindgen(js_name = "getAnalyticSurfaceParams")]
    pub fn get_analytic_surface_params(&self, face: u32) -> Result<String, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let json = match face_data.surface() {
            FaceSurface::Plane { normal, d } => serde_json::json!({
                "type": "plane",
                "normal": [normal.x(), normal.y(), normal.z()],
                "d": d,
            }),
            FaceSurface::Nurbs(_) => serde_json::json!({
                "type": "nurbs",
            }),
            FaceSurface::Cylinder(cyl) => serde_json::json!({
                "type": "cylinder",
                "origin": [cyl.origin().x(), cyl.origin().y(), cyl.origin().z()],
                "axis": [cyl.axis().x(), cyl.axis().y(), cyl.axis().z()],
                "radius": cyl.radius(),
            }),
            FaceSurface::Cone(cone) => serde_json::json!({
                "type": "cone",
                "apex": [cone.apex().x(), cone.apex().y(), cone.apex().z()],
                "axis": [cone.axis().x(), cone.axis().y(), cone.axis().z()],
                "halfAngle": cone.half_angle(),
            }),
            FaceSurface::Sphere(sph) => serde_json::json!({
                "type": "sphere",
                "center": [sph.center().x(), sph.center().y(), sph.center().z()],
                "radius": sph.radius(),
            }),
            FaceSurface::Torus(tor) => serde_json::json!({
                "type": "torus",
                "center": [tor.center().x(), tor.center().y(), tor.center().z()],
                "majorRadius": tor.major_radius(),
                "minorRadius": tor.minor_radius(),
            }),
        };
        Ok(json.to_string())
    }

    // ── Batch 5: Assembly & Sketch bindings ──────────────────────────

    /// Create a new empty sketch. Returns a sketch index.
    #[wasm_bindgen(js_name = "sketchNew")]
    pub fn sketch_new(&mut self) -> u32 {
        self.sketches.push(SketchState::default());
        #[allow(clippy::cast_possible_truncation)]
        let idx = (self.sketches.len() - 1) as u32;
        idx
    }

    /// Add a point to a sketch. Returns the point index.
    #[wasm_bindgen(js_name = "sketchAddPoint")]
    pub fn sketch_add_point(
        &mut self,
        sketch: u32,
        x: f64,
        y: f64,
        fixed: bool,
    ) -> Result<u32, JsError> {
        let sk = self
            .sketches
            .get_mut(sketch as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "sketch",
                index: sketch as usize,
            })?;
        let pt = if fixed {
            brepkit_operations::sketch::SketchPoint::fixed(x, y)
        } else {
            brepkit_operations::sketch::SketchPoint::new(x, y)
        };
        sk.points.push(pt);
        #[allow(clippy::cast_possible_truncation)]
        Ok((sk.points.len() - 1) as u32)
    }

    /// Add a constraint to a sketch from a JSON string.
    ///
    /// Formats: `{"type":"coincident","p1":0,"p2":1}`,
    /// `{"type":"distance","p1":0,"p2":1,"value":5.0}`,
    /// `{"type":"fixX","point":0,"value":1.0}`, etc.
    #[wasm_bindgen(js_name = "sketchAddConstraint")]
    pub fn sketch_add_constraint(&mut self, sketch: u32, json: &str) -> Result<(), JsError> {
        let sk = self
            .sketches
            .get_mut(sketch as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "sketch",
                index: sketch as usize,
            })?;
        let val: serde_json::Value =
            serde_json::from_str(json).map_err(|e| WasmError::InvalidInput {
                reason: format!("invalid constraint JSON: {e}"),
            })?;
        let constraint = parse_sketch_constraint(&val)?;
        sk.constraints.push(constraint);
        Ok(())
    }

    /// Solve the sketch constraints.
    ///
    /// Returns a JSON string: `{"converged": bool, "iterations": n, "maxResidual": f, "points": [[x,y], ...]}`.
    #[wasm_bindgen(js_name = "sketchSolve")]
    pub fn sketch_solve(
        &mut self,
        sketch: u32,
        max_iterations: u32,
        tolerance: f64,
    ) -> Result<String, JsError> {
        let sk = self
            .sketches
            .get_mut(sketch as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "sketch",
                index: sketch as usize,
            })?;
        let mut sketch_obj = brepkit_operations::sketch::Sketch {
            points: std::mem::take(&mut sk.points),
            constraints: std::mem::take(&mut sk.constraints),
        };
        let result = sketch_obj.solve(max_iterations as usize, tolerance);
        // Store back the updated points
        sk.points = sketch_obj.points;
        sk.constraints = sketch_obj.constraints;
        let (converged, iterations, max_residual) = match &result {
            Ok(r) => (r.converged, r.iterations, r.max_residual),
            Err(_) => (false, max_iterations as usize, f64::NAN),
        };
        let pts: Vec<serde_json::Value> = sk
            .points
            .iter()
            .map(|p| serde_json::json!([p.x, p.y]))
            .collect();
        Ok(serde_json::json!({
            "converged": converged,
            "iterations": iterations,
            "maxResidual": max_residual,
            "points": pts,
        })
        .to_string())
    }

    /// Create a new empty assembly. Returns an assembly index.
    #[wasm_bindgen(js_name = "assemblyNew")]
    pub fn assembly_new(&mut self, name: &str) -> u32 {
        self.assemblies
            .push(brepkit_operations::assembly::Assembly::new(name));
        #[allow(clippy::cast_possible_truncation)]
        let idx = (self.assemblies.len() - 1) as u32;
        idx
    }

    /// Add a root component to an assembly.
    ///
    /// Returns the component ID.
    #[wasm_bindgen(js_name = "assemblyAddRoot")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn assembly_add_root(
        &mut self,
        assembly: u32,
        name: &str,
        solid: u32,
        matrix: Vec<f64>,
    ) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let mat = parse_mat4(&matrix)?;
        let asm = self
            .assemblies
            .get_mut(assembly as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "assembly",
                index: assembly as usize,
            })?;
        let cid = asm.add_root_component(name, solid_id, mat);
        #[allow(clippy::cast_possible_truncation)]
        Ok(cid as u32)
    }

    /// Add a child component to a parent in an assembly.
    ///
    /// Returns the component ID.
    #[wasm_bindgen(js_name = "assemblyAddChild")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn assembly_add_child(
        &mut self,
        assembly: u32,
        parent: u32,
        name: &str,
        solid: u32,
        matrix: Vec<f64>,
    ) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let mat = parse_mat4(&matrix)?;
        let asm = self
            .assemblies
            .get_mut(assembly as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "assembly",
                index: assembly as usize,
            })?;
        let cid = asm.add_child_component(parent as usize, name, solid_id, mat)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(cid as u32)
    }

    /// Flatten an assembly into `[(solid, matrix), ...]`.
    ///
    /// Returns a JSON string: `[{"solid": u32, "matrix": [16 floats]}, ...]`.
    #[wasm_bindgen(js_name = "assemblyFlatten")]
    pub fn assembly_flatten(&self, assembly: u32) -> Result<String, JsError> {
        let asm = self
            .assemblies
            .get(assembly as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "assembly",
                index: assembly as usize,
            })?;
        let flat = asm.flatten();
        let entries: Vec<serde_json::Value> = flat
            .iter()
            .map(|(solid_id, mat)| {
                serde_json::json!({
                    "solid": solid_id_to_u32(*solid_id),
                    "matrix": mat4_to_array(mat),
                })
            })
            .collect();
        Ok(serde_json::Value::Array(entries).to_string())
    }

    /// Get the bill of materials for an assembly.
    ///
    /// Returns a JSON string: `[{"name": "...", "solidIndex": n, "instanceCount": n}, ...]`.
    #[wasm_bindgen(js_name = "assemblyBom")]
    pub fn assembly_bom(&self, assembly: u32) -> Result<String, JsError> {
        let asm = self
            .assemblies
            .get(assembly as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "assembly",
                index: assembly as usize,
            })?;
        let bom = asm.bill_of_materials();
        let entries: Vec<serde_json::Value> = bom
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "name": entry.name,
                    "solidIndex": entry.solid_index,
                    "instanceCount": entry.instance_count,
                })
            })
            .collect();
        Ok(serde_json::Value::Array(entries).to_string())
    }

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
}

// ── Private helpers for new bindings ────────────────────────────────

impl BrepKernel {
    /// Extract a `NurbsCurve` from an edge, or error if it's a line.
    fn extract_nurbs_curve(&self, edge: u32) -> Result<NurbsCurve, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let edge_data = self.topo.edge(edge_id)?;
        match edge_data.curve() {
            EdgeCurve::NurbsCurve(c) => Ok(c.clone()),
            EdgeCurve::Line => Err(WasmError::InvalidInput {
                reason: "edge is a line, not a NURBS curve".into(),
            }
            .into()),
        }
    }

    /// Create an edge from a `NurbsCurve`, using its endpoints.
    fn nurbs_curve_to_edge(
        &mut self,
        points: &[Point3],
        curve: NurbsCurve,
    ) -> brepkit_topology::edge::EdgeId {
        let start = points[0];
        let end = points[points.len() - 1];
        let v_start = self.topo.vertices.alloc(Vertex::new(start, TOL));
        let v_end = self.topo.vertices.alloc(Vertex::new(end, TOL));
        self.topo
            .edges
            .alloc(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(curve)))
    }

    /// Create an edge from a `NurbsCurve`, evaluating its endpoints.
    fn nurbs_curve_to_edge_from_curve(
        &mut self,
        curve: &NurbsCurve,
    ) -> brepkit_topology::edge::EdgeId {
        let start = curve.evaluate(curve.knots()[0]);
        let end = curve.evaluate(*curve.knots().last().unwrap_or(&1.0));
        let v_start = self.topo.vertices.alloc(Vertex::new(start, TOL));
        let v_end = self.topo.vertices.alloc(Vertex::new(end, TOL));
        self.topo.edges.alloc(Edge::new(
            v_start,
            v_end,
            EdgeCurve::NurbsCurve(curve.clone()),
        ))
    }

    /// Create a face from a `NurbsSurface` with a rectangular domain wire.
    fn nurbs_surface_to_face(
        &mut self,
        surface: brepkit_math::nurbs::surface::NurbsSurface,
    ) -> Result<brepkit_topology::face::FaceId, JsError> {
        // Evaluate corner points from the surface domain
        let (u_min, u_max) = surface.domain_u();
        let (v_min, v_max) = surface.domain_v();
        let corners = [
            surface.evaluate(u_min, v_min),
            surface.evaluate(u_max, v_min),
            surface.evaluate(u_max, v_max),
            surface.evaluate(u_min, v_max),
        ];
        let verts: Vec<_> = corners
            .iter()
            .map(|p| self.topo.vertices.alloc(Vertex::new(*p, TOL)))
            .collect();
        let n = verts.len();
        let edges: Vec<_> = (0..n)
            .map(|i| {
                self.topo
                    .edges
                    .alloc(Edge::new(verts[i], verts[(i + 1) % n], EdgeCurve::Line))
            })
            .collect();
        let oriented: Vec<_> = edges
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let wire = Wire::new(oriented, true)?;
        let wid = self.topo.wires.alloc(wire);
        let face_id = self
            .topo
            .faces
            .alloc(Face::new(wid, vec![], FaceSurface::Nurbs(surface)));
        Ok(face_id)
    }

    /// Dispatch a single batch operation by name.
    #[allow(clippy::too_many_lines)]
    fn dispatch_op(
        &mut self,
        op: &str,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        match op {
            "makeBox" => {
                let w = get_f64(args, "width")?;
                let h = get_f64(args, "height")?;
                let d = get_f64(args, "depth")?;
                let solid = brepkit_operations::primitives::make_box(&mut self.topo, w, h, d)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "makeCylinder" => {
                let r = get_f64(args, "radius")?;
                let h = get_f64(args, "height")?;
                let solid = brepkit_operations::primitives::make_cylinder(&mut self.topo, r, h)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "makeSphere" => {
                let r = get_f64(args, "radius")?;
                let segments = get_u32(args, "segments").unwrap_or(16);
                let solid = brepkit_operations::primitives::make_sphere(
                    &mut self.topo,
                    r,
                    segments as usize,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "makeCone" => {
                let br = get_f64(args, "bottomRadius")?;
                let tr = get_f64(args, "topRadius")?;
                let h = get_f64(args, "height")?;
                let solid = brepkit_operations::primitives::make_cone(&mut self.topo, br, tr, h)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "makeTorus" => {
                let major = get_f64(args, "majorRadius")?;
                let minor = get_f64(args, "minorRadius")?;
                let segments = get_u32(args, "segments").unwrap_or(16);
                let solid = brepkit_operations::primitives::make_torus(
                    &mut self.topo,
                    major,
                    minor,
                    segments as usize,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "fuse" => {
                let a = get_u32(args, "solidA")?;
                let b = get_u32(args, "solidB")?;
                let a_id = self.resolve_solid(a).map_err(|e| e.to_string())?;
                let b_id = self.resolve_solid(b).map_err(|e| e.to_string())?;
                let result = boolean(&mut self.topo, BooleanOp::Fuse, a_id, b_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "cut" => {
                let a = get_u32(args, "solidA")?;
                let b = get_u32(args, "solidB")?;
                let a_id = self.resolve_solid(a).map_err(|e| e.to_string())?;
                let b_id = self.resolve_solid(b).map_err(|e| e.to_string())?;
                let result = boolean(&mut self.topo, BooleanOp::Cut, a_id, b_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "intersect" => {
                let a = get_u32(args, "solidA")?;
                let b = get_u32(args, "solidB")?;
                let a_id = self.resolve_solid(a).map_err(|e| e.to_string())?;
                let b_id = self.resolve_solid(b).map_err(|e| e.to_string())?;
                let result = boolean(&mut self.topo, BooleanOp::Intersect, a_id, b_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "transform" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let matrix = args["matrix"]
                    .as_array()
                    .ok_or("missing or invalid 'matrix'")?;
                if matrix.len() != 16 {
                    return Err(format!(
                        "matrix must have 16 elements, got {}",
                        matrix.len()
                    ));
                }
                let elems: Vec<f64> = matrix
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        v.as_f64()
                            .ok_or_else(|| format!("matrix[{i}] is not a number"))
                    })
                    .collect::<Result<_, _>>()?;
                let rows = std::array::from_fn(|i| std::array::from_fn(|j| elems[i * 4 + j]));
                let mat = Mat4(rows);
                transform_solid(&mut self.topo, solid_id, &mat).map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid_id)))
            }
            "volume" => {
                let s = get_u32(args, "solid")?;
                let deflection = get_f64(args, "deflection").unwrap_or(0.1);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let v = measure::solid_volume(&self.topo, solid_id, deflection)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(v))
            }
            "surfaceArea" => {
                let s = get_u32(args, "solid")?;
                let deflection = get_f64(args, "deflection").unwrap_or(0.1);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let a = measure::solid_surface_area(&self.topo, solid_id, deflection)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(a))
            }
            "boundingBox" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let aabb =
                    measure::solid_bounding_box(&self.topo, solid_id).map_err(|e| e.to_string())?;
                Ok(serde_json::json!([
                    aabb.min.x(),
                    aabb.min.y(),
                    aabb.min.z(),
                    aabb.max.x(),
                    aabb.max.y(),
                    aabb.max.z()
                ]))
            }
            "centerOfMass" => {
                let s = get_u32(args, "solid")?;
                let deflection = get_f64(args, "deflection").unwrap_or(0.1);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let com = measure::solid_center_of_mass(&self.topo, solid_id, deflection)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!([com.x(), com.y(), com.z()]))
            }
            "copySolid" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let copy = brepkit_operations::copy::copy_solid(&mut self.topo, solid_id)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(copy)))
            }
            "copyAndTransformSolid" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let matrix = args["matrix"]
                    .as_array()
                    .ok_or("missing or invalid 'matrix'")?;
                if matrix.len() != 16 {
                    return Err(format!(
                        "matrix must have 16 elements, got {}",
                        matrix.len()
                    ));
                }
                let elems: Vec<f64> = matrix
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        v.as_f64()
                            .ok_or_else(|| format!("matrix[{i}] is not a number"))
                    })
                    .collect::<Result<_, _>>()?;
                let rows = std::array::from_fn(|i| std::array::from_fn(|j| elems[i * 4 + j]));
                let mat = Mat4(rows);
                let copy = brepkit_operations::copy::copy_and_transform_solid(
                    &mut self.topo,
                    solid_id,
                    &mat,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(copy)))
            }
            // ── Batch 8: new batch-dispatched operations ──────────────
            "extrude" => {
                let f = get_u32(args, "face")?;
                let dx = get_f64(args, "dx").unwrap_or(0.0);
                let dy = get_f64(args, "dy").unwrap_or(0.0);
                let dz = get_f64(args, "dz").unwrap_or(1.0);
                let dist = get_f64(args, "distance").unwrap_or(1.0);
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let dir = Vec3::new(dx, dy, dz);
                let solid =
                    extrude(&mut self.topo, face_id, dir, dist).map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "revolve" => {
                let f = get_u32(args, "face")?;
                let angle = get_f64(args, "angle")?;
                let ox = get_f64(args, "originX").unwrap_or(0.0);
                let oy = get_f64(args, "originY").unwrap_or(0.0);
                let oz = get_f64(args, "originZ").unwrap_or(0.0);
                let ax = get_f64(args, "axisX").unwrap_or(0.0);
                let ay = get_f64(args, "axisY").unwrap_or(0.0);
                let az = get_f64(args, "axisZ").unwrap_or(1.0);
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let solid = revolve(
                    &mut self.topo,
                    face_id,
                    Point3::new(ox, oy, oz),
                    Vec3::new(ax, ay, az),
                    angle,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "sweep" => {
                let f = get_u32(args, "face")?;
                let e = get_u32(args, "pathEdge")?;
                let face_id = self.resolve_face(f).map_err(|e| e.to_string())?;
                let edge_id = self.resolve_edge(e).map_err(|e| e.to_string())?;
                let edge_data = self.topo.edge(edge_id).map_err(|e| e.to_string())?;
                let curve = match edge_data.curve() {
                    EdgeCurve::NurbsCurve(c) => c.clone(),
                    EdgeCurve::Line => return Err("sweep path must be a NURBS edge".into()),
                };
                let solid = sweep(&mut self.topo, face_id, &curve).map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid)))
            }
            "chamfer" => {
                let s = get_u32(args, "solid")?;
                let dist = get_f64(args, "distance")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let edge_handles: Vec<u32> = args["edges"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let edge_ids: Vec<_> = edge_handles
                    .iter()
                    .map(|&h| self.resolve_edge(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result =
                    brepkit_operations::chamfer::chamfer(&mut self.topo, solid_id, &edge_ids, dist)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "fillet" => {
                let s = get_u32(args, "solid")?;
                let radius = get_f64(args, "radius")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let edge_handles: Vec<u32> = args["edges"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let edge_ids: Vec<_> = edge_handles
                    .iter()
                    .map(|&h| self.resolve_edge(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result =
                    brepkit_operations::fillet::fillet(&mut self.topo, solid_id, &edge_ids, radius)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "shell" => {
                let s = get_u32(args, "solid")?;
                let thickness = get_f64(args, "thickness")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let face_handles: Vec<u32> = args["faces"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let face_ids: Vec<_> = face_handles
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = brepkit_operations::shell_op::shell(
                    &mut self.topo,
                    solid_id,
                    thickness,
                    &face_ids,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "mirror" => {
                let s = get_u32(args, "solid")?;
                let px = get_f64(args, "px").unwrap_or(0.0);
                let py = get_f64(args, "py").unwrap_or(0.0);
                let pz = get_f64(args, "pz").unwrap_or(0.0);
                let nx = get_f64(args, "nx").unwrap_or(1.0);
                let ny = get_f64(args, "ny").unwrap_or(0.0);
                let nz = get_f64(args, "nz").unwrap_or(0.0);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let result = brepkit_operations::mirror::mirror(
                    &mut self.topo,
                    solid_id,
                    Point3::new(px, py, pz),
                    Vec3::new(nx, ny, nz),
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "healSolid" => {
                let s = get_u32(args, "solid")?;
                let tol = get_f64(args, "tolerance").unwrap_or(1e-7);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                brepkit_operations::heal::heal_solid(&mut self.topo, solid_id, tol)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(solid_id)))
            }
            "classifyPoint" => {
                let s = get_u32(args, "solid")?;
                let x = get_f64(args, "x")?;
                let y = get_f64(args, "y")?;
                let z = get_f64(args, "z")?;
                let tol = get_f64(args, "tolerance").unwrap_or(1e-7);
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let pt = Point3::new(x, y, z);
                let result = brepkit_operations::classify::classify_point(
                    &self.topo, solid_id, pt, 0.1, tol,
                )
                .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(classify_to_string(result)))
            }
            "loft" => {
                let face_handles: Vec<u32> = args["faces"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let face_ids: Vec<_> = face_handles
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result = brepkit_operations::loft::loft(&mut self.topo, &face_ids)
                    .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            "circularPattern" => {
                let s = get_u32(args, "solid")?;
                let ax = get_f64(args, "ax").unwrap_or(0.0);
                let ay = get_f64(args, "ay").unwrap_or(0.0);
                let az = get_f64(args, "az").unwrap_or(1.0);
                let count = get_u32(args, "count")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let axis = Vec3::new(ax, ay, az);
                let compound = brepkit_operations::pattern::circular_pattern(
                    &mut self.topo,
                    solid_id,
                    axis,
                    count as usize,
                )
                .map_err(|e| e.to_string())?;
                #[allow(clippy::cast_possible_truncation)]
                Ok(serde_json::json!(compound.index() as u32))
            }
            "defeature" => {
                let s = get_u32(args, "solid")?;
                let solid_id = self.resolve_solid(s).map_err(|e| e.to_string())?;
                let face_handles: Vec<u32> = args["faces"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                let face_ids: Vec<_> = face_handles
                    .iter()
                    .map(|&h| self.resolve_face(h).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let result =
                    brepkit_operations::defeature::defeature(&mut self.topo, solid_id, &face_ids)
                        .map_err(|e| e.to_string())?;
                Ok(serde_json::json!(solid_id_to_u32(result)))
            }
            _ => Err(format!("unknown operation: {op}")),
        }
    }
}

/// Convert a `FaceId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
const fn face_id_to_u32(id: brepkit_topology::face::FaceId) -> u32 {
    id.index() as u32
}

/// Convert a `SolidId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
const fn solid_id_to_u32(id: brepkit_topology::solid::SolidId) -> u32 {
    id.index() as u32
}

/// Convert a `VertexId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
const fn vertex_id_to_u32(id: brepkit_topology::vertex::VertexId) -> u32 {
    id.index() as u32
}

/// Convert an `EdgeId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
const fn edge_id_to_u32(id: brepkit_topology::edge::EdgeId) -> u32 {
    id.index() as u32
}

/// Convert a `WireId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
const fn wire_id_to_u32(id: brepkit_topology::wire::WireId) -> u32 {
    id.index() as u32
}

/// Convert a `ShellId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
const fn shell_id_to_u32(id: brepkit_topology::shell::ShellId) -> u32 {
    id.index() as u32
}

impl Default for BrepKernel {
    fn default() -> Self {
        Self::new()
    }
}

// ── Batch argument helpers ─────────────────────────────────────────

/// Extract a required `f64` value from a JSON object.
fn get_f64(args: &serde_json::Value, key: &str) -> Result<f64, String> {
    args[key]
        .as_f64()
        .ok_or_else(|| format!("missing or invalid '{key}'"))
}

/// Extract a required `u32` value from a JSON object.
fn get_u32(args: &serde_json::Value, key: &str) -> Result<u32, String> {
    args[key]
        .as_u64()
        .map(|v| v as u32)
        .ok_or_else(|| format!("missing or invalid '{key}'"))
}

// ── Free helper functions for new bindings ────────────────────────

/// Parse flat `[x,y,z, ...]` coordinates into `Vec<Point3>`.
fn parse_points(coords: &[f64]) -> Result<Vec<Point3>, JsError> {
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
fn parse_point_grid(coords: &[f64], rows: usize, cols: usize) -> Result<Vec<Vec<Point3>>, JsError> {
    let points = parse_points(coords)?;
    if points.len() != rows * cols {
        return Err(WasmError::InvalidInput {
            reason: format!(
                "expected {} points ({}x{}), got {}",
                rows * cols,
                rows,
                cols,
                points.len()
            ),
        }
        .into());
    }
    Ok(points.chunks(cols).map(|row| row.to_vec()).collect())
}

/// Serialize a `Feature` enum to JSON.
fn serialize_feature(f: &brepkit_operations::feature_recognition::Feature) -> serde_json::Value {
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

/// Convert a `PointClassification` to a string.
fn classify_to_string(c: brepkit_operations::classify::PointClassification) -> String {
    match c {
        brepkit_operations::classify::PointClassification::Inside => "inside".into(),
        brepkit_operations::classify::PointClassification::Outside => "outside".into(),
        brepkit_operations::classify::PointClassification::OnBoundary => "boundary".into(),
    }
}

/// Build a `TriangleMesh` from flat position/index arrays.
fn build_triangle_mesh(
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

/// Parse a boolean operation string to the enum.
fn parse_boolean_op(op: &str) -> Result<BooleanOp, JsError> {
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

/// Convert a `TriangleMesh` to `JsMesh`.
fn triangle_mesh_to_js(mesh: &tessellate::TriangleMesh) -> JsMesh {
    JsMesh::from(mesh.clone())
}

/// Parse a sketch constraint from a JSON value.
fn parse_sketch_constraint(
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
            let v = json_f64(val, "value")?;
            Ok(Constraint::Angle(p1, p2, v))
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

/// Extract a `usize` from a JSON value.
fn json_usize(val: &serde_json::Value, key: &str) -> Result<usize, JsError> {
    val[key].as_u64().map(|v| v as usize).ok_or_else(|| {
        WasmError::InvalidInput {
            reason: format!("missing or invalid '{key}'"),
        }
        .into()
    })
}

/// Extract an `f64` from a JSON value.
fn json_f64(val: &serde_json::Value, key: &str) -> Result<f64, JsError> {
    val[key].as_f64().ok_or_else(|| {
        WasmError::InvalidInput {
            reason: format!("missing or invalid '{key}'"),
        }
        .into()
    })
}

/// Parse a flat 16-element array into a `Mat4` (row-major).
fn parse_mat4(elems: &[f64]) -> Result<Mat4, JsError> {
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
fn mat4_to_array(mat: &Mat4) -> Vec<f64> {
    let mut out = Vec::with_capacity(16);
    for row in &mat.0 {
        for &v in row {
            out.push(v);
        }
    }
    out
}

/// Project a 3D point onto a NURBS surface to get (u,v) parameters.
///
/// Uses a simple grid search + Newton refinement.
fn project_to_uv(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    point: Point3,
) -> brepkit_math::vec::Point2 {
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();
    let n = 10;
    let mut best_u = u_min;
    let mut best_v = v_min;
    let mut best_dist = f64::MAX;
    for i in 0..=n {
        for j in 0..=n {
            #[allow(clippy::cast_precision_loss)]
            let u = u_min + (u_max - u_min) * (i as f64) / (n as f64);
            #[allow(clippy::cast_precision_loss)]
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
    brepkit_math::vec::Point2::new(best_u, best_v)
}

#[cfg(test)]
mod batch_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn batch_single_op() {
        let mut kernel = BrepKernel::new();
        let result = kernel
            .execute_batch(r#"[{"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}}]"#);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed[0]["ok"].is_number(),
            "expected ok number, got {parsed}"
        );
    }

    #[test]
    fn batch_multiple_ops() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 2, "height": 2, "depth": 2}},
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "volume", "args": {"solid": 0}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 3);
        // First two should be ok with solid ids
        assert!(parsed[0]["ok"].is_number());
        assert!(parsed[1]["ok"].is_number());
        // Third should be a volume value
        assert!(parsed[2]["ok"].is_number());
    }

    #[test]
    fn batch_error_doesnt_stop_rest() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch(
            r#"[
                {"op": "unknownOp", "args": {}},
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed[0]["error"].is_string());
        assert!(parsed[1]["ok"].is_number());
    }

    #[test]
    fn batch_invalid_json() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch("not valid json");
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(
            parsed[0]["error"]
                .as_str()
                .unwrap()
                .contains("invalid JSON")
        );
    }

    #[test]
    fn batch_missing_op_field() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch(r#"[{"args": {"width": 1}}]"#);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed[0]["error"].as_str().unwrap().contains("op"));
    }

    #[test]
    fn batch_boolean_ops() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 2, "height": 2, "depth": 2}},
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "fuse", "args": {"solidA": 0, "solidB": 1}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed[0]["ok"].is_number());
        assert!(parsed[1]["ok"].is_number());
        assert!(parsed[2]["ok"].is_number());
    }

    #[test]
    fn batch_bounding_box() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 2, "height": 4, "depth": 6}},
                {"op": "boundingBox", "args": {"solid": 0}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed[0]["ok"].is_number());
        let bbox = &parsed[1]["ok"];
        assert!(bbox.is_array());
        assert_eq!(bbox.as_array().unwrap().len(), 6);
    }

    #[test]
    fn batch_copy_solid() {
        let mut kernel = BrepKernel::new();
        let result = kernel.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "copySolid", "args": {"solid": 0}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed[0]["ok"].is_number());
        assert!(parsed[1]["ok"].is_number());
        // Copy should have a different handle
        assert_ne!(parsed[0]["ok"].as_u64(), parsed[1]["ok"].as_u64());
    }
}
