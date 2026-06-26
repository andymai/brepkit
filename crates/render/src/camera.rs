//! Camera and view/projection matrices.
//!
//! All camera state is kept in f64. The render-relative-to-center (RTC)
//! precision scheme uploads vertex positions as f32 offsets from the model
//! AABB center; the f64 center is folded into the view matrix here so the GPU
//! never sees large absolute coordinates.

use brepkit_math::vec::{Point3, Vec3};

/// A perspective camera defined in world space (all coordinates in f64).
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Eye (camera) position in world space.
    pub eye: Point3,
    /// Point the camera looks at, in world space.
    pub target: Point3,
    /// Up direction (need not be exactly orthogonal to the view direction).
    pub up: Vec3,
    /// Vertical field of view, in radians.
    pub fov_y: f64,
    /// Aspect ratio (width / height).
    pub aspect: f64,
    /// Near clip plane distance (> 0).
    pub near: f64,
    /// Far clip plane distance (> near).
    pub far: f64,
}

impl Camera {
    /// World-space direction the camera looks along (`target - eye`, normalized).
    ///
    /// Falls back to `-Z` if eye and target coincide.
    #[must_use]
    pub fn view_direction(&self) -> Vec3 {
        (self.target - self.eye)
            .normalize()
            .unwrap_or_else(|_| Vec3::new(0.0, 0.0, -1.0))
    }
}

/// Build the combined view-projection matrix for `cam`, with `center` folded
/// into the translation so it operates on RTC (center-relative) positions.
///
/// The result, as a 16-element column-major f32 array, maps a center-relative
/// world position directly to wgpu clip space (NDC z in `[0, 1]`).
pub fn view_proj_rtc(cam: &Camera, center: Point3) -> [f32; 16] {
    let view = look_at_rh_rtc(cam.eye, cam.target, cam.up, center);
    let proj = perspective_rh_zo(cam.fov_y, cam.aspect, cam.near, cam.far);
    proj.mul(&view).to_cols_array()
}

/// A column-major 4x4 f32 matrix laid out for direct upload to WGSL.
///
/// WGSL `mat4x4<f32>` is column-major, so `cols[i]` is the i-th column. This
/// type is constructed from f64 math and converted to f32 only at the end,
/// after the large model-center translation has been removed (RTC).
#[derive(Debug, Clone, Copy)]
struct Mat4f {
    cols: [[f32; 4]; 4],
}

impl Mat4f {
    /// Flatten to the 16-element column-major array WGSL expects.
    fn to_cols_array(self) -> [f32; 16] {
        let c = &self.cols;
        [
            c[0][0], c[0][1], c[0][2], c[0][3], c[1][0], c[1][1], c[1][2], c[1][3], c[2][0],
            c[2][1], c[2][2], c[2][3], c[3][0], c[3][1], c[3][2], c[3][3],
        ]
    }
}

/// A column-major 4x4 f64 matrix used for the intermediate camera math.
#[derive(Debug, Clone, Copy)]
struct Mat4d {
    cols: [[f64; 4]; 4],
}

impl Mat4d {
    /// `self * rhs`, returning an f32 matrix ready for upload.
    fn mul(self, rhs: &Self) -> Mat4f {
        let mut out = [[0.0_f64; 4]; 4];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.cols[k][row] * rhs.cols[col][k];
                }
                out[col][row] = sum;
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        Mat4f {
            cols: std::array::from_fn(|c| std::array::from_fn(|r| out[c][r] as f32)),
        }
    }
}

/// Pick a world axis to use as an "up" vector that is not parallel to `f`.
///
/// Returns the axis (`X`, `Y`, or `Z`) whose alignment with `f` is smallest,
/// so the subsequent `f x up` cross product is well-conditioned regardless of
/// the view direction.
fn fallback_up(f: Vec3) -> Vec3 {
    let ax = f.x().abs();
    let ay = f.y().abs();
    let az = f.z().abs();
    if ax <= ay && ax <= az {
        Vec3::new(1.0, 0.0, 0.0)
    } else if ay <= az {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    }
}

/// Right-handed look-at view matrix operating on center-relative positions.
///
/// `eye`, `target`, and `center` are absolute world points; subtracting
/// `center` from both eye and the translated origin keeps every quantity small
/// (RTC), so the f64 -> f32 conversion at upload time loses no meaningful
/// precision even for models far from the origin.
fn look_at_rh_rtc(eye: Point3, target: Point3, up: Vec3, center: Point3) -> Mat4d {
    // Camera basis (right-handed): f points from eye toward target, s is right,
    // u is the recomputed up.
    let f = (target - eye)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 0.0, -1.0));
    // If `up` is parallel (or anti-parallel) to the view direction, f x up
    // collapses to ~zero and the basis is degenerate. Fall back to whichever
    // world axis is least aligned with f, which is guaranteed non-parallel.
    let s = f.cross(up).normalize().unwrap_or_else(|_| {
        f.cross(fallback_up(f))
            .normalize()
            .unwrap_or(Vec3::new(1.0, 0.0, 0.0))
    });
    let u = s.cross(f);

    // Eye expressed relative to the model center.
    let eye_rel = eye - center; // Vec3

    // View matrix rows dot the basis vectors; translation = -(basis . eye_rel).
    let tx = -s.dot(eye_rel);
    let ty = -u.dot(eye_rel);
    let tz = f.dot(eye_rel);

    // Column-major: column 3 holds the translation.
    Mat4d {
        cols: [
            [s.x(), u.x(), -f.x(), 0.0],
            [s.y(), u.y(), -f.y(), 0.0],
            [s.z(), u.z(), -f.z(), 0.0],
            [tx, ty, tz, 1.0],
        ],
    }
}

/// Right-handed perspective projection mapping NDC z to `[0, 1]` (wgpu/WebGPU
/// depth convention, "zero-to-one").
///
/// Inputs are clamped to finite, well-conditioned ranges so a malformed
/// [`Camera`] (zero FOV, zero/negative near or aspect, `far <= near`) yields a
/// usable matrix instead of one full of NaNs/infinities.
fn perspective_rh_zo(fov_y: f64, aspect: f64, near: f64, far: f64) -> Mat4d {
    let fov_y = fov_y.clamp(1.0e-4, std::f64::consts::PI - 1.0e-4);
    let aspect = if aspect.is_finite() && aspect > 0.0 {
        aspect
    } else {
        1.0
    };
    let near = if near.is_finite() && near > 0.0 {
        near
    } else {
        1.0e-3
    };
    let far = if far.is_finite() && far > near {
        far
    } else {
        near * 1000.0
    };

    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);

    Mat4d {
        cols: [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far * nf, -1.0],
            [0.0, 0.0, far * near * nf, 0.0],
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_finite(m: &[f32; 16]) -> bool {
        m.iter().all(|v| v.is_finite())
    }

    #[test]
    fn degenerate_up_parallel_to_view_stays_finite() {
        // up parallel to the view direction would collapse f x up to zero; the
        // fallback-up path must still yield a finite, usable view matrix.
        let cam = Camera {
            eye: Point3::new(0.0, 0.0, 10.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vec3::new(0.0, 0.0, 1.0), // parallel to view dir (-Z)
            fov_y: 45.0_f64.to_radians(),
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        };
        let m = view_proj_rtc(&cam, Point3::new(0.0, 0.0, 0.0));
        assert!(all_finite(&m), "view-proj must be finite for parallel up");
    }

    #[test]
    fn fallback_up_is_not_parallel_to_view() {
        // For each principal view direction, the chosen fallback up must not be
        // (anti-)parallel to it, so the basis cross product is well-conditioned.
        for f in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ] {
            let up = fallback_up(f);
            assert!(
                f.cross(up).length() > 0.5,
                "fallback up {up:?} too aligned with view dir {f:?}"
            );
        }
    }

    #[test]
    fn projection_guards_degenerate_inputs() {
        // Zero FOV, zero aspect, zero near, and far <= near must all be clamped
        // to produce a finite matrix rather than NaNs/infinities.
        let cam = Camera {
            eye: Point3::new(5.0, 5.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vec3::new(0.0, 0.0, 1.0),
            fov_y: 0.0,
            aspect: 0.0,
            near: 0.0,
            far: 0.0,
        };
        let m = view_proj_rtc(&cam, Point3::new(0.0, 0.0, 0.0));
        assert!(
            all_finite(&m),
            "projection must be finite for degenerate inputs"
        );
    }
}
