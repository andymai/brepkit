//! The `BrepKernel` — a WASM-exposed modeling context.
//!
//! JavaScript consumers create a single `BrepKernel` instance and call
//! methods on it to build and query geometry. All topological state is
//! owned by the kernel; JS only holds opaque `u32` handles.

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::extrude::extrude;
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

        // Collect all shell IDs (outer + inner voids).
        let shell_ids: Vec<_> = std::iter::once(solid_data.outer_shell())
            .chain(solid_data.inner_shells().iter().copied())
            .collect();

        let mut merged = TriangleMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            indices: Vec::new(),
        };

        for shell_id in shell_ids {
            let shell = self.topo.shell(shell_id)?;
            for &face_id in shell.faces() {
                let face_mesh = tessellate::tessellate(&self.topo, face_id, deflection)?;

                // Offset indices by the current vertex count in the merged mesh.
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
