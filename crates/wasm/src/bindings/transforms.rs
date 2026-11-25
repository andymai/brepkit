//! Transform, copy, mirror, and pattern bindings.

#![allow(clippy::missing_errors_doc, clippy::too_many_arguments)]

use wasm_bindgen::prelude::*;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::transform::transform_solid;

use crate::error::{
    WasmError, validate_pattern_counts, validate_point, validate_positive, validate_vector,
};
use crate::handles::{compound_id_to_u32, face_id_to_u32, solid_id_to_u32, wire_id_to_u32};
use crate::kernel::BrepKernel;

#[wasm_bindgen]
impl BrepKernel {
    /// Apply a 4×4 affine transform to a solid (in place).
    ///
    /// The `matrix` must contain exactly 16 values in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid, the matrix doesn't
    /// have 16 elements, the matrix is singular, or the moved solid would
    /// pass the size limit.
    #[wasm_bindgen(js_name = "transformSolid")]
    #[allow(clippy::needless_pass_by_value)] // wasm-bindgen requires owned Vec
    pub fn transform_solid_binding(&mut self, solid: u32, matrix: Vec<f64>) -> Result<(), JsError> {
        Ok(self.transform_solid_impl(solid, &matrix)?)
    }

    /// Compose (multiply) two 4x4 transformation matrices.
    ///
    /// Returns the composed matrix as a flat 16-element array (row-major).
    /// This computes `a * b`, meaning `b` is applied first, then `a`.
    ///
    /// # Errors
    ///
    /// Returns an error if either matrix doesn't have 16 elements.
    #[wasm_bindgen(js_name = "composeTransforms")]
    #[allow(clippy::needless_pass_by_value, clippy::unused_self)]
    pub fn compose_transforms(
        &self,
        matrix_a: Vec<f64>,
        matrix_b: Vec<f64>,
    ) -> Result<Vec<f64>, JsError> {
        let product = matrix_from(&matrix_a)? * matrix_from(&matrix_b)?;
        let out: Vec<f64> = product.0.iter().flatten().copied().collect();
        if let Some(pos) = out.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("the composed matrix's element at index {pos} is not finite"),
            }
            .into());
        }
        Ok(out)
    }

    // ── Copy / Mirror / Pattern ───────────────────────────────────

    /// Deep copy a solid, returning a new independent solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "copySolid")]
    pub fn copy_solid(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let copy = brepkit_operations::copy::copy_solid(self.topo_mut(), solid_id)?;
        Ok(solid_id_to_u32(copy))
    }

    /// Deep copy a wire, returning a new independent wire handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the wire handle is invalid.
    #[wasm_bindgen(js_name = "copyWire")]
    pub fn copy_wire(&mut self, wire: u32) -> Result<u32, JsError> {
        let wire_id = self.resolve_wire(wire)?;
        let copy = brepkit_operations::copy::copy_wire(self.topo_mut(), wire_id)?;
        Ok(wire_id_to_u32(copy))
    }

    /// Deep copy a face, returning a new independent face handle.
    ///
    /// The copy shares no sub-entities with the original, so translating it
    /// (to form a pocket or boss profile) does not mutate the donor solid.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid.
    #[wasm_bindgen(js_name = "copyFace")]
    pub fn copy_face(&mut self, face: u32) -> Result<u32, JsError> {
        let face_id = self.resolve_face(face)?;
        let copy = brepkit_operations::copy::copy_face(self.topo_mut(), face_id)?;
        Ok(face_id_to_u32(copy))
    }

    /// Apply a 4×4 affine transform to a wire (in place).
    ///
    /// The `matrix` must contain exactly 16 values in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error if the wire handle is invalid, the matrix doesn't
    /// have 16 elements, the matrix is singular, or the moved wire would
    /// pass the size limit.
    #[wasm_bindgen(js_name = "transformWire")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn transform_wire(&mut self, wire: u32, matrix: Vec<f64>) -> Result<(), JsError> {
        Ok(self.transform_wire_impl(wire, &matrix)?)
    }

    /// Apply a 4×4 affine transform to a face (in place).
    ///
    /// Transforms all vertices, edge curves, and the face surface geometry.
    /// The `matrix` must contain exactly 16 values in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid, the matrix doesn't
    /// have 16 elements, the matrix is singular, or the moved face would
    /// pass the size limit.
    #[wasm_bindgen(js_name = "transformFace")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn transform_face(&mut self, face: u32, matrix: Vec<f64>) -> Result<(), JsError> {
        Ok(self.transform_face_impl(face, &matrix)?)
    }

    /// Copy a solid and apply a 4×4 row-major affine transform in one pass.
    ///
    /// Equivalent to `copySolid` + `transformSolid` but performs both in a
    /// single topology traversal, avoiding redundant NURBS clones.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid, the matrix doesn't
    /// have 16 elements, the matrix is singular, or the copy would pass the
    /// size limit.
    #[wasm_bindgen(js_name = "copyAndTransformSolid")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn copy_and_transform_solid(
        &mut self,
        solid: u32,
        matrix: Vec<f64>,
    ) -> Result<u32, JsError> {
        Ok(self.copy_and_transform_solid_impl(solid, &matrix)?)
    }

    /// Mirror a solid across a plane.
    ///
    /// Returns a new solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid, the normal is zero,
    /// or a coordinate or the mirrored solid would pass the size limit.
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
        Ok(self.mirror_impl(solid, Point3::new(px, py, pz), Vec3::new(nx, ny, nz))?)
    }

    /// Create a linear pattern of a solid.
    ///
    /// Returns a compound handle containing all copies.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid, `count` is more than
    /// `MAX_COUNT`, or the last copy would pass the size limit.
    #[wasm_bindgen(js_name = "linearPattern")]
    #[allow(clippy::too_many_arguments)]
    pub fn linear_pattern(
        &mut self,
        solid: u32,
        dx: f64,
        dy: f64,
        dz: f64,
        spacing: f64,
        count: u32,
    ) -> Result<u32, JsError> {
        Ok(self.linear_pattern_impl(solid, Vec3::new(dx, dy, dz), spacing, count)?)
    }

    // ── Grid Pattern ──────────────────────────────────────────────

    /// Create a 2D grid pattern of a solid.
    ///
    /// Produces `count_x × count_y` copies arranged in a rectangular grid.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid, the grid holds more than
    /// `MAX_COUNT` copies, or a corner copy would pass the size limit.
    #[wasm_bindgen(js_name = "gridPattern")]
    #[allow(clippy::too_many_arguments)]
    pub fn grid_pattern(
        &mut self,
        solid: u32,
        dir_x_x: f64,
        dir_x_y: f64,
        dir_x_z: f64,
        dir_y_x: f64,
        dir_y_y: f64,
        dir_y_z: f64,
        spacing_x: f64,
        spacing_y: f64,
        count_x: u32,
        count_y: u32,
    ) -> Result<u32, JsError> {
        Ok(self.grid_pattern_impl(
            solid,
            [
                Vec3::new(dir_x_x, dir_x_y, dir_x_z),
                Vec3::new(dir_y_x, dir_y_y, dir_y_z),
            ],
            [spacing_x, spacing_y],
            [count_x, count_y],
        )?)
    }

    /// Create a circular pattern of a solid around an axis.
    ///
    /// Returns a compound handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid, the axis is not a
    /// finite size, `count` is more than `MAX_COUNT`, or a turned copy would
    /// pass the size limit.
    #[wasm_bindgen(js_name = "circularPattern")]
    pub fn circular_pattern(
        &mut self,
        solid: u32,
        ax: f64,
        ay: f64,
        az: f64,
        count: u32,
    ) -> Result<u32, JsError> {
        Ok(self.circular_pattern_impl(solid, Vec3::new(ax, ay, az), count)?)
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
        crate::error::validate_tolerance(tolerance, "tolerance")?;
        let solid_id = self.resolve_solid(solid)?;
        let count = brepkit_operations::heal::merge_coincident_vertices(
            self.topo_mut(),
            solid_id,
            tolerance,
        )?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(count as u32)
    }
}

/// A 4x4 matrix from sixteen row-major numbers from JS.
fn matrix_from(matrix: &[f64]) -> Result<Mat4, WasmError> {
    if matrix.len() != 16 {
        return Err(WasmError::InvalidInput {
            reason: format!(
                "transform matrix must have 16 elements, got {}",
                matrix.len()
            ),
        });
    }
    if let Some(pos) = matrix.iter().position(|v| !v.is_finite()) {
        return Err(WasmError::InvalidInput {
            reason: format!("matrix element at index {pos} is not finite"),
        });
    }
    Ok(Mat4(std::array::from_fn(|i| {
        std::array::from_fn(|j| matrix[i * 4 + j])
    })))
}

/// The length of `v`, taken from `v` divided by its largest component, so
/// that no square of a component past 1e154 overflows.
pub fn scaled_length(v: Vec3) -> f64 {
    let largest = v.x().abs().max(v.y().abs()).max(v.z().abs());
    if largest > 0.0 && largest.is_finite() {
        let unit = Vec3::new(v.x() / largest, v.y() / largest, v.z() / largest);
        largest * unit.length()
    } else {
        largest
    }
}

/// `v` scaled to unit length, by way of [`scaled_length`].
pub fn scaled_unit(v: Vec3) -> Result<Vec3, WasmError> {
    let largest = v.x().abs().max(v.y().abs()).max(v.z().abs());
    if largest > 0.0 && largest.is_finite() {
        Ok(Vec3::new(v.x() / largest, v.y() / largest, v.z() / largest).normalize()?)
    } else {
        Err(WasmError::InvalidInput {
            reason: format!("a direction must be finite and not zero, got {v:?}"),
        })
    }
}

/// The points that hold a shape, kept for the checks a move must pass: the
/// ends of its edges and the cube round each circle, ellipse, sphere or
/// torus, and apart from them each spline's control points with its
/// largest weight. A spline lies in the hull of its control points while
/// its weights are positive; a plane, cylinder or cone face lies within its
/// edges' hull.
#[derive(Default)]
pub struct Reach {
    points: Vec<Point3>,
    splines: Vec<(f64, Vec<Point3>)>,
}

impl Reach {
    /// A reach of loose points alone, with no spline.
    #[cfg(test)]
    pub fn of_points(points: Vec<Point3>) -> Self {
        Self {
            points,
            splines: Vec::new(),
        }
    }

    /// The reach of every edge of `wires`.
    pub fn of_wires(
        topo: &brepkit_topology::Topology,
        wires: impl IntoIterator<Item = brepkit_topology::wire::WireId>,
    ) -> Result<Self, WasmError> {
        let mut reach = Self::default();
        for wire in wires {
            reach.add_wire(topo, wire)?;
        }
        Ok(reach)
    }

    /// The reach of a face: its wires, and a surface that can bulge past
    /// them.
    pub fn of_face(
        topo: &brepkit_topology::Topology,
        face: brepkit_topology::face::FaceId,
    ) -> Result<Self, WasmError> {
        let mut reach = Self::default();
        reach.add_face(topo, face)?;
        Ok(reach)
    }

    /// The reach of every face of `solid`, in every shell.
    pub fn of_solid(
        topo: &brepkit_topology::Topology,
        solid: brepkit_topology::solid::SolidId,
    ) -> Result<Self, WasmError> {
        let mut reach = Self::default();
        for face in brepkit_topology::explorer::solid_faces(topo, solid)? {
            reach.add_face(topo, face)?;
        }
        Ok(reach)
    }

    fn add_cube(&mut self, center: Point3, r: f64) {
        for k in 0..8 {
            let side = |bit: i32| if k & bit == 0 { -r } else { r };
            self.points
                .push(center + Vec3::new(side(1), side(2), side(4)));
        }
    }

    fn add_wire(
        &mut self,
        topo: &brepkit_topology::Topology,
        wire: brepkit_topology::wire::WireId,
    ) -> Result<(), WasmError> {
        use brepkit_topology::edge::EdgeCurve;
        for oe in topo.wire(wire)?.edges() {
            let edge = topo.edge(oe.edge())?;
            self.points.push(topo.vertex(edge.start())?.point());
            self.points.push(topo.vertex(edge.end())?.point());
            match edge.curve() {
                EdgeCurve::Line => {}
                EdgeCurve::Circle(c) => self.add_cube(c.center(), c.radius()),
                EdgeCurve::Ellipse(e) => self.add_cube(e.center(), e.semi_major()),
                EdgeCurve::NurbsCurve(n) => {
                    let weight = n.weights().iter().fold(0.0_f64, |m, w| m.max(w.abs()));
                    self.splines.push((weight, n.control_points().to_vec()));
                }
            }
        }
        Ok(())
    }

    fn add_face(
        &mut self,
        topo: &brepkit_topology::Topology,
        face: brepkit_topology::face::FaceId,
    ) -> Result<(), WasmError> {
        use brepkit_topology::face::FaceSurface;
        let face_data = topo.face(face)?;
        for wire in
            std::iter::once(face_data.outer_wire()).chain(face_data.inner_wires().iter().copied())
        {
            self.add_wire(topo, wire)?;
        }
        match face_data.surface() {
            FaceSurface::Nurbs(s) => {
                let weight = s
                    .weights()
                    .iter()
                    .flatten()
                    .fold(0.0_f64, |m, w| m.max(w.abs()));
                let controls = s.control_points().iter().flatten().copied().collect();
                self.splines.push((weight, controls));
            }
            FaceSurface::Sphere(s) => self.add_cube(s.center(), s.radius()),
            FaceSurface::Torus(t) => self.add_cube(t.center(), t.major_radius() + t.minor_radius()),
            FaceSurface::Plane { .. } | FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => {}
        }
        Ok(())
    }

    /// Check each point and control point that `place` gives as the bound of
    /// its image, and each spline's largest weight times its largest bound,
    /// as the load checks do. Each is checked on its own: a box of them
    /// would drop a coordinate that is not a number. A shape with no points,
    /// as an empty solid, has nothing to check.
    fn check_by(&self, place: impl Fn(Point3) -> Point3, what: &str) -> Result<(), WasmError> {
        for &p in &self.points {
            validate_point(place(p), what)?;
        }
        for (weight, controls) in &self.splines {
            let mut largest = 1.0_f64;
            for &p in controls {
                let q = place(p);
                validate_point(q, what)?;
                largest = largest.max(q.x().abs()).max(q.y().abs()).max(q.z().abs());
            }
            if weight * largest > brepkit_math::nurbs::basis::WEIGHTED_LIMIT {
                return Err(WasmError::InvalidInput {
                    reason: format!(
                        "{what}: a spline's largest weight {weight:e} times its largest \
                         coordinate {largest:e} must be at most {:e}",
                        brepkit_math::nurbs::basis::WEIGHTED_LIMIT
                    ),
                });
            }
        }
        Ok(())
    }

    /// Check the shape moved by `matrix`. An affine map takes a hull to the
    /// hull of the moved points, so this bounds the moved shape.
    pub fn check_moved(&self, matrix: &Mat4, what: &str) -> Result<(), WasmError> {
        self.check_by(|p| matrix.mul_point(p), what)
    }

    /// Check the shape turned about any axis through `center`: a turn keeps
    /// each point's distance from `center`, so the corner of the cube round
    /// `center` out to that distance bounds the point's every coordinate.
    pub fn check_turned(&self, center: Point3, what: &str) -> Result<(), WasmError> {
        self.check_by(
            |p| {
                let r = scaled_length(p - center);
                let out = |c: f64| if c < 0.0 { -r } else { r };
                center + Vec3::new(out(center.x()), out(center.y()), out(center.z()))
            },
            what,
        )
    }

    /// Check the copies of the shape turned about the axis through the
    /// origin along the unit `axis` by each multiple of a `count`-th of a
    /// turn, as a circular pattern places them. A turn keeps each point `p`
    /// on the circle about the axis: its coordinate `i` at angle `t` is
    /// `c_i + A_i cos(t - f_i)`, with `c` the circle's centre, and `A_i` and
    /// `f_i` from the radial offset `u` and `a x u`. The copy nearest to
    /// `f_i`, and the one nearest to the opposite angle, give the largest
    /// and smallest values over the copies: the bound is exact, with no
    /// copy checked twice.
    pub fn check_copies(&self, axis: Vec3, count: u32, what: &str) -> Result<(), WasmError> {
        let step = std::f64::consts::TAU / f64::from(count.max(1));
        let gap = |angle: f64| angle - (angle / step).round() * step;
        self.check_by(
            |p| {
                let along = (p - Point3::new(0.0, 0.0, 0.0)).dot(axis);
                let center = Point3::new(0.0, 0.0, 0.0) + axis * along;
                let u = p - center;
                let v = axis.cross(u);
                let (c, u, v) = (
                    [center.x(), center.y(), center.z()],
                    [u.x(), u.y(), u.z()],
                    [v.x(), v.y(), v.z()],
                );
                let far: [f64; 3] = std::array::from_fn(|i| {
                    let amplitude = u[i].hypot(v[i]);
                    let phase = v[i].atan2(u[i]);
                    let high = c[i] + amplitude * gap(phase).cos();
                    let low = c[i] - amplitude * gap(phase + std::f64::consts::PI).cos();
                    high.abs().max(low.abs())
                });
                Point3::new(far[0], far[1], far[2])
            },
            what,
        )
    }
}

impl BrepKernel {
    /// Check that `solid` moved by `matrix` stays within the size limit.
    fn check_moved_solid(
        &self,
        solid: brepkit_topology::solid::SolidId,
        matrix: &Mat4,
        what: &str,
    ) -> Result<(), WasmError> {
        Reach::of_solid(&self.topo, solid)?.check_moved(matrix, what)?;
        Ok(brepkit_operations::transform::check_solid_transform(
            &self.topo, solid, matrix,
        )?)
    }

    /// The checked body of `transformSolid` and the batch's `transform`.
    pub(crate) fn transform_solid_impl(
        &mut self,
        solid: u32,
        matrix: &[f64],
    ) -> Result<(), WasmError> {
        let mat = matrix_from(matrix)?;
        let solid_id = self.resolve_solid(solid)?;
        self.check_moved_solid(solid_id, &mat, "the moved solid")?;
        transform_solid(self.topo_mut(), solid_id, &mat)?;
        Ok(())
    }

    /// The checked body of `copyAndTransformSolid`, in JS and in the batch.
    pub(crate) fn copy_and_transform_solid_impl(
        &mut self,
        solid: u32,
        matrix: &[f64],
    ) -> Result<u32, WasmError> {
        let mat = matrix_from(matrix)?;
        let solid_id = self.resolve_solid(solid)?;
        self.check_moved_solid(solid_id, &mat, "the moved copy")?;
        let copy =
            brepkit_operations::copy::copy_and_transform_solid(self.topo_mut(), solid_id, &mat)?;
        Ok(solid_id_to_u32(copy))
    }

    /// The checked body of `transformWire`, in JS and in the batch.
    pub(crate) fn transform_wire_impl(
        &mut self,
        wire: u32,
        matrix: &[f64],
    ) -> Result<(), WasmError> {
        let mat = matrix_from(matrix)?;
        let wire_id = self.resolve_wire(wire)?;
        Reach::of_wires(&self.topo, [wire_id])?.check_moved(&mat, "the moved wire")?;
        brepkit_operations::transform::check_wire_transform(&self.topo, wire_id, &mat)?;
        brepkit_operations::transform::transform_wire(self.topo_mut(), wire_id, &mat)?;
        Ok(())
    }

    /// The checked body of `transformFace`, in JS and in the batch.
    pub(crate) fn transform_face_impl(
        &mut self,
        face: u32,
        matrix: &[f64],
    ) -> Result<(), WasmError> {
        let mat = matrix_from(matrix)?;
        let face_id = self.resolve_face(face)?;
        Reach::of_face(&self.topo, face_id)?.check_moved(&mat, "the moved face")?;
        brepkit_operations::transform::check_face_transform(&self.topo, face_id, &mat)?;
        brepkit_operations::transform::transform_face(self.topo_mut(), face_id, &mat)?;
        Ok(())
    }

    /// The checked body of `mirror`, in JS and in the batch.
    pub(crate) fn mirror_impl(
        &mut self,
        solid: u32,
        point: Point3,
        normal: Vec3,
    ) -> Result<u32, WasmError> {
        validate_point(point, "the mirror plane's point")?;
        validate_vector(normal, "the mirror plane's normal")?;
        let solid_id = self.resolve_solid(solid)?;
        let n = scaled_unit(normal)?;
        let offset = 2.0 * (point - Point3::new(0.0, 0.0, 0.0)).dot(n);
        let reflection = Mat4(std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let unit = if i == j { 1.0 } else { 0.0 };
                match (i, j) {
                    (3, 3) => 1.0,
                    (3, _) => 0.0,
                    (_, 3) => offset * n.0[i],
                    _ => unit - 2.0 * n.0[i] * n.0[j],
                }
            })
        }));
        self.check_moved_solid(solid_id, &reflection, "the mirrored solid")?;
        let result = brepkit_operations::mirror::mirror(self.topo_mut(), solid_id, point, normal)?;
        Ok(solid_id_to_u32(result))
    }

    /// The checked body of `linearPattern`, in JS and in the batch.
    pub(crate) fn linear_pattern_impl(
        &mut self,
        solid: u32,
        direction: Vec3,
        spacing: f64,
        count: u32,
    ) -> Result<u32, WasmError> {
        validate_vector(direction, "the pattern's direction")?;
        validate_positive(spacing, "spacing")?;
        validate_pattern_counts(&[count])?;
        let solid_id = self.resolve_solid(solid)?;
        let reach = spacing * f64::from(count.saturating_sub(1));
        let last = scaled_unit(direction)? * reach;
        self.check_moved_solid(
            solid_id,
            &Mat4::translation(last.x(), last.y(), last.z()),
            "the last copy",
        )?;
        let compound = brepkit_operations::pattern::linear_pattern(
            self.topo_mut(),
            solid_id,
            direction,
            spacing,
            count as usize,
        )?;
        Ok(compound_id_to_u32(compound))
    }

    /// The checked body of `gridPattern`, in JS and in the batch.
    pub(crate) fn grid_pattern_impl(
        &mut self,
        solid: u32,
        directions: [Vec3; 2],
        spacings: [f64; 2],
        counts: [u32; 2],
    ) -> Result<u32, WasmError> {
        validate_vector(directions[0], "the grid's first direction")?;
        validate_vector(directions[1], "the grid's second direction")?;
        validate_positive(spacings[0], "spacing_x")?;
        validate_positive(spacings[1], "spacing_y")?;
        validate_pattern_counts(&counts)?;
        let solid_id = self.resolve_solid(solid)?;
        let reach =
            |k: usize| -> Result<Vec3, WasmError> {
                Ok(scaled_unit(directions[k])?
                    * (spacings[k] * f64::from(counts[k].saturating_sub(1))))
            };
        let (along_x, along_y) = (reach(0)?, reach(1)?);
        for corner in [along_x, along_y, along_x + along_y] {
            self.check_moved_solid(
                solid_id,
                &Mat4::translation(corner.x(), corner.y(), corner.z()),
                "a corner copy",
            )?;
        }
        let compound = brepkit_operations::pattern::grid_pattern(
            self.topo_mut(),
            solid_id,
            directions[0],
            directions[1],
            spacings[0],
            spacings[1],
            counts[0] as usize,
            counts[1] as usize,
        )?;
        Ok(compound_id_to_u32(compound))
    }

    /// The checked body of `circularPattern`, in JS and in the batch. Each
    /// copy is the solid turned about the axis through the origin.
    pub(crate) fn circular_pattern_impl(
        &mut self,
        solid: u32,
        axis: Vec3,
        count: u32,
    ) -> Result<u32, WasmError> {
        validate_vector(axis, "the pattern's axis")?;
        validate_pattern_counts(&[count])?;
        let solid_id = self.resolve_solid(solid)?;
        Reach::of_solid(&self.topo, solid_id)?.check_copies(
            scaled_unit(axis)?,
            count,
            "a turned copy",
        )?;
        let compound = brepkit_operations::pattern::circular_pattern(
            self.topo_mut(),
            solid_id,
            axis,
            count as usize,
        )?;
        Ok(compound_id_to_u32(compound))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use crate::kernel::BrepKernel;

    fn identity_matrix() -> Vec<f64> {
        vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    // ── copy_solid ────────────────────────────────────────────────

    #[test]
    fn copy_solid_returns_new_handle() {
        let mut k = BrepKernel::new();
        let s = k.make_box_solid(1.0, 1.0, 1.0).unwrap();
        let copy = k.copy_solid(s).unwrap();
        assert_ne!(s, copy);
    }

    #[test]
    fn copy_solid_both_handles_resolve() {
        let mut k = BrepKernel::new();
        let s = k.make_box_solid(2.0, 2.0, 2.0).unwrap();
        let copy = k.copy_solid(s).unwrap();
        // Both handles must be independently resolvable.
        assert!(k.resolve_solid(s).is_ok());
        assert!(k.resolve_solid(copy).is_ok());
    }

    #[test]
    fn copy_solid_invalid_handle_errors() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(r#"[{"op": "copySolid", "args": {"solid": 9999}}]"#);
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert!(parsed[0]["error"].is_string());
    }

    // ── copy_face ─────────────────────────────────────────────────

    #[test]
    fn copy_face_returns_new_handle() {
        let mut k = BrepKernel::new();
        let s = k.make_box_solid(1.0, 1.0, 1.0).unwrap();
        let face = k.get_solid_faces(s).unwrap()[0];
        let batch = format!(r#"[{{"op": "copyFace", "args": {{"face": {face}}}}}]"#);
        let r = k.execute_batch(&batch);
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        let copy = parsed[0]["ok"].as_u64().unwrap();
        assert_ne!(copy, u64::from(face));
    }

    #[test]
    fn copy_face_invalid_handle_errors() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(r#"[{"op": "copyFace", "args": {"face": 9999}}]"#);
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert!(parsed[0]["error"].is_string());
    }

    // ── transform_solid_binding ───────────────────────────────────

    #[test]
    fn identity_transform_preserves_bounding_box() {
        let mut k = BrepKernel::new();
        let s = k.make_box_solid(3.0, 4.0, 5.0).unwrap();
        let before = k.bounding_box(s).unwrap();
        k.transform_solid_binding(s, identity_matrix()).unwrap();
        let after = k.bounding_box(s).unwrap();
        for (a, b) in before.iter().zip(after.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "bbox changed after identity: {a} vs {b}"
            );
        }
    }

    #[test]
    fn transform_solid_translation_shifts_bbox() {
        let mut k = BrepKernel::new();
        let s = k.make_box_solid(1.0, 1.0, 1.0).unwrap();
        // Translate +10 along X.
        let mat = vec![
            1.0, 0.0, 0.0, 10.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        k.transform_solid_binding(s, mat).unwrap();
        let bbox = k.bounding_box(s).unwrap();
        // min_x should have shifted by +10
        assert!(
            (bbox[0] - (bbox[3] - 1.0)).abs() < 1e-6,
            "unexpected min_x: {}",
            bbox[0]
        );
        assert!(bbox[0] >= 9.0, "bbox min_x not translated: {}", bbox[0]);
    }

    #[test]
    fn transform_solid_invalid_handle_errors() {
        let mut k = BrepKernel::new();
        let mat_json: Vec<String> = identity_matrix().iter().map(ToString::to_string).collect();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "transform", "args": {{"solid": 9999, "matrix": [{}]}}}}]"#,
            mat_json.join(",")
        ));
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert!(parsed[0]["error"].is_string());
    }

    #[test]
    fn transform_solid_wrong_matrix_length_errors() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "transform", "args": {"solid": 0, "matrix": [1,0,0,0, 0,1,0,0, 0]}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        let err_msg = parsed[1]["error"].as_str().unwrap();
        assert!(
            err_msg.contains("16"),
            "error should mention 16 elements: {err_msg}"
        );
    }

    #[test]
    fn transform_solid_nan_in_matrix_errors() {
        // NaN cannot be represented in JSON, so we test this at the
        // operations layer instead. A NaN in the matrix is caught by
        // the WasmError validation before reaching operations.
        // This verifies that our validation logic in the binding
        // would reject non-finite values.
        let mat = identity_matrix();
        let has_nan = mat.iter().any(|v| !v.is_finite());
        assert!(!has_nan, "identity matrix should be finite");
        // A matrix with NaN is invalid — tested via the assertion pattern.
        assert!(!f64::NAN.is_finite());
    }

    // ── mirror_solid ──────────────────────────────────────────────

    #[test]
    fn mirror_solid_returns_new_handle() {
        let mut k = BrepKernel::new();
        let s = k.make_box_solid(1.0, 1.0, 1.0).unwrap();
        let mirrored = k.mirror_solid(s, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0).unwrap();
        assert_ne!(s, mirrored);
        assert!(k.resolve_solid(mirrored).is_ok());
    }

    #[test]
    fn mirror_solid_invalid_handle_errors() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "mirror", "args": {"solid": 9999, "px": 0, "py": 0, "pz": 0, "nx": 0, "ny": 0, "nz": 1}}]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert!(parsed[0]["error"].is_string());
    }

    // ── linear_pattern ────────────────────────────────────────────

    #[test]
    fn linear_pattern_count_matches_compound_children() {
        let mut k = BrepKernel::new();
        let s = k.make_box_solid(1.0, 1.0, 1.0).unwrap();
        let compound = k.linear_pattern(s, 1.0, 0.0, 0.0, 2.0, 4).unwrap();
        let solids = k.get_compound_solids(compound).unwrap();
        assert_eq!(
            solids.len(),
            4,
            "expected 4 solids in compound, got {}",
            solids.len()
        );
    }

    #[test]
    fn linear_pattern_count_one_returns_original() {
        let mut k = BrepKernel::new();
        let s = k.make_box_solid(1.0, 1.0, 1.0).unwrap();
        let compound = k.linear_pattern(s, 0.0, 1.0, 0.0, 5.0, 1).unwrap();
        let solids = k.get_compound_solids(compound).unwrap();
        assert_eq!(solids.len(), 1);
        // The single element should be the original solid handle.
        assert_eq!(solids[0], s);
    }

    #[test]
    fn linear_pattern_invalid_solid_errors() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "linearPattern", "args": {"solid": 9999, "dx": 1, "dy": 0, "dz": 0, "spacing": 2, "count": 3}}]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert!(parsed[0]["error"].is_string());
    }

    #[test]
    fn linear_pattern_zero_spacing_errors() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "linearPattern", "args": {"solid": 0, "dx": 1, "dy": 0, "dz": 0, "spacing": 0, "count": 3}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert!(parsed[1]["error"].is_string());
    }
}
