//! The `BrepKernel` — a WASM-exposed modeling context.
//!
//! JavaScript consumers create a single `BrepKernel` instance and call
//! methods on it to build and query geometry. All topological state is
//! owned by the kernel; JS only holds opaque `u32` handles.

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

impl Default for BrepKernel {
    fn default() -> Self {
        Self::new()
    }
}
