# brepkit-blend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace brepkit's sample-fit fillet/chamfer engine with an OCCT-style walking-based blend engine in a new `brepkit-blend` crate at L1.5.

**Architecture:** New `brepkit-blend` crate depends only on `brepkit-math` + `brepkit-topology`. Contains a Newton-Raphson walking engine that solves a 4-equation blend constraint system, analytic fast paths for common surface pairs, a dedicated topology builder for face trimming and solid assembly, and a vertex blend corner solver. The operations crate provides thin public API wrappers.

**Tech Stack:** Rust, `thiserror` for errors, `brepkit-math` NURBS/geometry, `brepkit-topology` arena-based topology, OCCT source at `~/Git/OCCT/src/` as algorithmic reference.

**Spec:** `docs/superpowers/specs/2026-03-19-brepkit-blend-design.md`

---

## File Map

### New files (brepkit-blend crate)

| File | Responsibility |
|------|---------------|
| `crates/blend/Cargo.toml` | Crate manifest (deps: math, topology) |
| `crates/blend/src/lib.rs` | Public API, `BlendError`, `BlendResult`, re-exports |
| `crates/blend/src/radius_law.rs` | `RadiusLaw` enum: Constant, Linear, SCurve, Custom |
| `crates/blend/src/spine.rs` | `Spine`: edge chain with arc-length parameterization |
| `crates/blend/src/section.rs` | `CircSection`: cross-section (contact points + center + radius) |
| `crates/blend/src/stripe.rs` | `Stripe`: fillet band (surface + PCurves + contact curves) |
| `crates/blend/src/blend_func.rs` | `BlendFunction` trait + `ConstRadBlend`, `EvolRadBlend`, `ChamferBlend`, `ChamferAngleBlend` |
| `crates/blend/src/walker.rs` | Newton-Raphson walking engine + NURBS surface approximation |
| `crates/blend/src/analytic.rs` | Closed-form fillet/chamfer for plane-plane, plane-cyl, plane-cone, cyl-cyl |
| `crates/blend/src/fillet_builder.rs` | `FilletBuilder`: orchestrates spine construction, stripe computation, assembly |
| `crates/blend/src/chamfer_builder.rs` | `ChamferBuilder`: chamfer variant of fillet builder |
| `crates/blend/src/corner.rs` | Vertex blend: sphere/torus analytic + Coons patch for general cases |
| `crates/blend/src/trimmer.rs` | Face trimming along contact curves + PCurve computation |

### Modified files

| File | Change |
|------|--------|
| `crates/math/src/traits.rs` | Add `partial_u`, `partial_v`, `domain` to `ParametricSurface` |
| `crates/math/src/surfaces.rs` | Implement `partial_u`/`partial_v` for Cylinder, Cone, Sphere, Torus |
| `crates/math/src/nurbs/surface.rs` | Implement `partial_u`/`partial_v` for NurbsSurface |
| `crates/operations/src/lib.rs` | Add `pub mod blend_ops;` |
| `crates/operations/src/blend_ops.rs` | New: thin wrappers calling brepkit-blend |
| `crates/operations/Cargo.toml` | Add `brepkit-blend` dependency |
| `crates/wasm/src/bindings/operations.rs` | Add `fillet_v2`, `chamfer_v2`, `chamfer_distance_angle` bindings |
| `crates/wasm/Cargo.toml` | Add `brepkit-blend` dependency |
| `scripts/check-boundaries.sh` | Add `blend` crate rules |
| `CLAUDE.md` | Add `blend` to layer table and use paths |

---

## Task 1: Extend ParametricSurface trait with partial derivatives

The walker's Jacobian requires `∂S/∂u` and `∂S/∂v` from each surface. This is a prerequisite for the entire blend engine.

**Files:**
- Modify: `crates/math/src/traits.rs`
- Modify: `crates/math/src/surfaces.rs`
- Modify: `crates/math/src/nurbs/surface.rs`

**OCCT reference:** Each analytic surface has explicit derivative formulas. See `Geom_CylindricalSurface::D1()` etc.

- [ ] **Step 1: Write failing tests for partial derivatives**

Add tests in `crates/math/src/traits.rs` (or `surfaces.rs` test module):

```rust
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn cylinder_partial_u_is_tangential() {
        let cyl = CylindricalSurface::new(
            Point3::origin(), Vec3::new(0.0, 0.0, 1.0), 2.0,
        ).unwrap();
        let du = cyl.partial_u(0.0, 0.0);
        let p = cyl.evaluate(0.0, 0.0);
        let n = cyl.normal(0.0, 0.0);
        // du should be perpendicular to normal
        assert!(du.dot(n).abs() < 1e-10, "du·n = {}", du.dot(n));
        // du magnitude should be radius (= 2.0)
        assert!((du.length() - 2.0).abs() < 1e-10, "|du| = {}", du.length());
    }

    #[test]
    fn cylinder_partial_v_is_axial() {
        let cyl = CylindricalSurface::new(
            Point3::origin(), Vec3::new(0.0, 0.0, 1.0), 2.0,
        ).unwrap();
        let dv = cyl.partial_v(0.0, 0.5);
        // dv should be the axis direction
        assert!((dv - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-10);
    }

    #[test]
    fn sphere_partials_are_perpendicular() {
        let sph = SphericalSurface::new(Point3::origin(), 3.0).unwrap();
        let du = sph.partial_u(1.0, 0.5);
        let dv = sph.partial_v(1.0, 0.5);
        assert!(du.dot(dv).abs() < 1e-10, "du·dv = {}", du.dot(dv));
    }

    #[test]
    fn nurbs_partial_matches_finite_difference() {
        // Create a simple bilinear NURBS patch
        let surf = NurbsSurface::new(
            1, 1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        ).unwrap();
        let h = 1e-6;
        let u = 0.5;
        let v = 0.5;
        let du_fd = (surf.evaluate(u + h, v) - surf.evaluate(u - h, v)) * (0.5 / h);
        let du = surf.partial_u(u, v);
        assert!((du - du_fd).length() < 1e-4, "du={du:?} fd={du_fd:?}");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p brepkit-math -- partial_u`
Expected: compilation error — `partial_u` method doesn't exist

- [ ] **Step 3: Add trait methods with default implementations**

In `crates/math/src/traits.rs`, add to the `ParametricSurface` trait:

```rust
/// Partial derivative ∂S/∂u at (u, v).
fn partial_u(&self, u: f64, v: f64) -> Vec3;

/// Partial derivative ∂S/∂v at (u, v).
fn partial_v(&self, u: f64, v: f64) -> Vec3;

/// Parameter domain bounds: ((u_min, u_max), (v_min, v_max)).
/// Returns `None` if the domain is unbounded (e.g., planes).
fn domain(&self) -> Option<((f64, f64), (f64, f64))> {
    None
}
```

- [ ] **Step 4: Implement for CylindricalSurface**

In `crates/math/src/surfaces.rs`, add to the `ParametricSurface` impl for `CylindricalSurface`:

```rust
// S(u,v) = origin + radius*(cos(u)*x_axis + sin(u)*y_axis) + v*axis
// ∂S/∂u = radius*(-sin(u)*x_axis + cos(u)*y_axis)
fn partial_u(&self, u: f64, _v: f64) -> Vec3 {
    let (sin_u, cos_u) = u.sin_cos();
    self.x_axis * (-self.radius * sin_u) + self.y_axis * (self.radius * cos_u)
}

// ∂S/∂v = axis
fn partial_v(&self, _u: f64, _v: f64) -> Vec3 {
    self.axis
}
```

- [ ] **Step 5: Implement for ConicalSurface**

```rust
// S(u,v) = apex + v*(cos(half_angle)*(cos(u)*x_axis + sin(u)*y_axis) + sin(half_angle)*axis)
// ∂S/∂u = v*cos(half_angle)*(-sin(u)*x_axis + cos(u)*y_axis)
fn partial_u(&self, u: f64, v: f64) -> Vec3 {
    let (sin_u, cos_u) = u.sin_cos();
    let scale = v * self.half_angle.cos();
    self.x_axis * (-scale * sin_u) + self.y_axis * (scale * cos_u)
}

// ∂S/∂v = cos(half_angle)*(cos(u)*x_axis + sin(u)*y_axis) + sin(half_angle)*axis
fn partial_v(&self, u: f64, _v: f64) -> Vec3 {
    let (sin_u, cos_u) = u.sin_cos();
    let ca = self.half_angle.cos();
    let sa = self.half_angle.sin();
    self.x_axis * (ca * cos_u) + self.y_axis * (ca * sin_u) + self.axis * sa
}
```

- [ ] **Step 6: Implement for SphericalSurface**

Note: `SphericalSurface` has fields `center`, `radius`, `x_axis`, `y_axis`, `z_axis`.

```rust
// S(u,v) = center + radius*(cos(v)*cos(u)*x_axis + cos(v)*sin(u)*y_axis + sin(v)*z_axis)
// ∂S/∂u = radius*cos(v)*(-sin(u)*x_axis + cos(u)*y_axis)
fn partial_u(&self, u: f64, v: f64) -> Vec3 {
    let (sin_u, cos_u) = u.sin_cos();
    let cv = v.cos();
    self.x_axis * (-self.radius * cv * sin_u) + self.y_axis * (self.radius * cv * cos_u)
}

// ∂S/∂v = radius*(-sin(v)*cos(u)*x_axis - sin(v)*sin(u)*y_axis + cos(v)*z_axis)
fn partial_v(&self, u: f64, v: f64) -> Vec3 {
    let (sin_u, cos_u) = u.sin_cos();
    let (sin_v, cos_v) = v.sin_cos();
    self.x_axis * (-self.radius * sin_v * cos_u)
        + self.y_axis * (-self.radius * sin_v * sin_u)
        + self.z_axis * (self.radius * cos_v)
}
```

- [ ] **Step 7: Implement for ToroidalSurface**

Note: `ToroidalSurface` has fields `center`, `major_radius`, `minor_radius`, `x_axis`, `y_axis`, `z_axis`.

```rust
// S(u,v) = center + (R + r*cos(v))*(cos(u)*x_axis + sin(u)*y_axis) + r*sin(v)*z_axis
// ∂S/∂u = (R + r*cos(v))*(-sin(u)*x_axis + cos(u)*y_axis)
fn partial_u(&self, u: f64, v: f64) -> Vec3 {
    let (sin_u, cos_u) = u.sin_cos();
    let rho = self.major_radius + self.minor_radius * v.cos();
    self.x_axis * (-rho * sin_u) + self.y_axis * (rho * cos_u)
}

// ∂S/∂v = r*(-sin(v))*(cos(u)*x_axis + sin(u)*y_axis) + r*cos(v)*z_axis
fn partial_v(&self, u: f64, v: f64) -> Vec3 {
    let (sin_u, cos_u) = u.sin_cos();
    let (sin_v, cos_v) = v.sin_cos();
    self.x_axis * (-self.minor_radius * sin_v * cos_u)
        + self.y_axis * (-self.minor_radius * sin_v * sin_u)
        + self.z_axis * (self.minor_radius * cos_v)
}
```

- [ ] **Step 8: Implement for NurbsSurface**

In `crates/math/src/nurbs/surface.rs`, add `partial_u`/`partial_v` to the `ParametricSurface` impl. The `NurbsSurface` already has a `derivatives(u, v, order)` method that returns partial derivatives.

```rust
fn partial_u(&self, u: f64, v: f64) -> Vec3 {
    let ders = self.derivatives(u, v, 1);
    ders[1][0] // ∂S/∂u
}

fn partial_v(&self, u: f64, v: f64) -> Vec3 {
    let ders = self.derivatives(u, v, 1);
    ders[0][1] // ∂S/∂v
}
```

Check the actual return type/indexing of `derivatives()` — it may be `Vec<Vec<Vec3>>` or `Vec<Vec<Point3>>`. Adapt accordingly.

- [ ] **Step 9: Handle Plane surfaces**

`FaceSurface::Plane { normal, d }` is NOT a struct — it's an inline enum variant with no
`ParametricSurface` impl and no UV frame. The blend crate needs to construct a local UV
frame from the normal when working with planes.

Create a helper in the blend crate (e.g., in `blend_func.rs` or a shared utils):

```rust
/// Construct an orthonormal UV frame from a plane normal.
/// Returns `(u_axis, v_axis)` where `u_axis × v_axis = normal`.
fn plane_uv_frame(normal: Vec3) -> (Vec3, Vec3) {
    // Pick a non-parallel seed vector
    let seed = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_axis = normal.cross(seed).normalized();
    let v_axis = normal.cross(u_axis);
    (u_axis, v_axis)
}
```

For the blend engine, planes are evaluated as:
- `evaluate(u, v) = origin + u*u_axis + v*v_axis` where `origin` = point on plane
- `partial_u = u_axis` (constant)
- `partial_v = v_axis` (constant)
- `normal = normal` (constant)

The `ParametricSurface` trait extension does NOT need a Plane impl — planes are handled
by the blend functions directly via this helper. The analytic fast path (Task 5) will
destructure `FaceSurface::Plane { normal, d }` and use `plane_uv_frame(normal)`.

Note: `FaceSurface` delegate methods (`evaluate`, `normal`, etc.) already handle the Plane
variant — the blend code can use these delegates for evaluation, and the helper above only
for computing Jacobians where partial derivatives are needed.

- [ ] **Step 10: Run all tests**

Run: `cargo test -p brepkit-math -- partial`
Expected: all new partial derivative tests pass

Run: `cargo test --workspace`
Expected: no regressions (existing code doesn't call partial_u/partial_v yet)

- [ ] **Step 11: Commit**

```bash
git add crates/math/src/traits.rs crates/math/src/surfaces.rs crates/math/src/nurbs/surface.rs
git commit -m "feat(math): add partial_u/partial_v to ParametricSurface trait

Required by the blend engine's Newton-Raphson walker to compute
Jacobians of the blend constraint equation."
```

---

## Task 2: Scaffold brepkit-blend crate

Create the new crate with Cargo.toml, lib.rs, error types, and data structures.

**Files:**
- Create: `crates/blend/Cargo.toml`
- Create: `crates/blend/src/lib.rs`
- Create: `crates/blend/src/radius_law.rs`
- Create: `crates/blend/src/section.rs`
- Create: `crates/blend/src/spine.rs`
- Create: `crates/blend/src/stripe.rs`
- Modify: `scripts/check-boundaries.sh`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add workspace dependency + Create Cargo.toml**

First, add to the root `Cargo.toml` `[workspace.dependencies]` section:
```toml
brepkit-blend = { path = "crates/blend" }
```

Then create `crates/blend/Cargo.toml`:

```toml
[package]
name = "brepkit-blend"
version = "0.1.0"
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
description = "Walking-based fillet and chamfer engine for brepkit"

[dependencies]
brepkit-math.workspace = true
brepkit-topology.workspace = true
thiserror.workspace = true
log.workspace = true

[dev-dependencies]
brepkit-topology = { workspace = true, features = ["test-utils"] }

[lints]
workspace = true
```

- [ ] **Step 2: Create lib.rs with error types and re-exports**

```rust
//! Walking-based fillet and chamfer engine.
//!
//! This crate implements OCCT-style blend surface computation using a
//! Newton-Raphson walking algorithm. It produces G1-continuous fillet
//! and chamfer surfaces for all combinations of analytic and NURBS faces.

pub mod analytic;
pub mod blend_func;
pub mod chamfer_builder;
pub mod corner;
pub mod fillet_builder;
pub mod radius_law;
pub mod section;
pub mod spine;
pub mod stripe;
pub mod trimmer;
pub mod walker;

use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

/// Error type for blend operations.
#[derive(Debug, thiserror::Error)]
pub enum BlendError {
    /// No initial solution found at the spine start.
    #[error("no start solution at edge {edge:?}, t={t}")]
    StartSolutionFailure { edge: EdgeId, t: f64 },

    /// Walker diverged during marching.
    #[error("walking failure at edge {edge:?}, t={t}, residual={residual}")]
    WalkingFailure { edge: EdgeId, t: f64, residual: f64 },

    /// Generated surface is twisted or self-intersecting.
    #[error("twisted surface on stripe {stripe_idx}")]
    TwistedSurface { stripe_idx: usize },

    /// Radius too large for the edge geometry.
    #[error("radius too large for edge {edge:?}: max={max_radius}")]
    RadiusTooLarge { edge: EdgeId, max_radius: f64 },

    /// Face trimming failed.
    #[error("trimming failure on face {face:?}")]
    TrimmingFailure { face: FaceId },

    /// Corner solver failed at vertex.
    #[error("corner failure at vertex {vertex:?}")]
    CornerFailure { vertex: VertexId },

    /// Surface type not supported.
    #[error("unsupported surface on face {face:?}: {surface_tag}")]
    UnsupportedSurface { face: FaceId, surface_tag: String },

    /// Topology error from underlying operations.
    #[error(transparent)]
    Topology(#[from] brepkit_topology::TopologyError),

    /// Math error from underlying computations.
    #[error(transparent)]
    Math(#[from] brepkit_math::MathError),
}

/// Result of a blend operation.
pub struct BlendResult {
    /// The resulting solid.
    pub solid: SolidId,
    /// Edges that were successfully blended.
    pub succeeded: Vec<EdgeId>,
    /// Edges that failed with diagnostic info.
    pub failed: Vec<(EdgeId, BlendError)>,
    /// Whether this is a partial result (some edges failed).
    pub is_partial: bool,
}
```

- [ ] **Step 3: Create radius_law.rs**

```rust
//! Radius law types for variable-radius fillets.

/// Defines how the fillet radius varies along an edge.
#[derive(Debug, Clone)]
pub enum RadiusLaw {
    /// Constant radius.
    Constant(f64),
    /// Linear interpolation from `start` to `end`.
    Linear { start: f64, end: f64 },
    /// Smooth Hermite ramp: `3t² - 2t³`.
    SCurve { start: f64, end: f64 },
    /// Custom law: boxed closure mapping `t ∈ [0,1] → radius`.
    Custom(Box<dyn Fn(f64) -> f64 + Send + Sync>),
}

impl RadiusLaw {
    /// Evaluate the radius at parameter `t ∈ [0, 1]`.
    pub fn evaluate(&self, t: f64) -> f64 {
        match self {
            Self::Constant(r) => *r,
            Self::Linear { start, end } => start + (end - start) * t,
            Self::SCurve { start, end } => {
                let s = t * t * (3.0 - 2.0 * t);
                start + (end - start) * s
            }
            Self::Custom(f) => f(t),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn constant_law_returns_same_value() {
        let law = RadiusLaw::Constant(5.0);
        assert!((law.evaluate(0.0) - 5.0).abs() < f64::EPSILON);
        assert!((law.evaluate(0.5) - 5.0).abs() < f64::EPSILON);
        assert!((law.evaluate(1.0) - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn linear_law_interpolates() {
        let law = RadiusLaw::Linear { start: 1.0, end: 3.0 };
        assert!((law.evaluate(0.0) - 1.0).abs() < f64::EPSILON);
        assert!((law.evaluate(0.5) - 2.0).abs() < f64::EPSILON);
        assert!((law.evaluate(1.0) - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn scurve_law_is_smooth() {
        let law = RadiusLaw::SCurve { start: 1.0, end: 3.0 };
        assert!((law.evaluate(0.0) - 1.0).abs() < f64::EPSILON);
        assert!((law.evaluate(1.0) - 3.0).abs() < f64::EPSILON);
        // Midpoint: 3*(0.5)² - 2*(0.5)³ = 0.5
        assert!((law.evaluate(0.5) - 2.0).abs() < f64::EPSILON);
    }
}
```

- [ ] **Step 4: Create section.rs**

```rust
//! Cross-section of a blend surface at a spine parameter.

use brepkit_math::vec::{Point3, Vec3};

/// A circular cross-section of the blend at a given spine parameter.
#[derive(Debug, Clone)]
pub struct CircSection {
    /// Contact point on surface 1.
    pub p1: Point3,
    /// Contact point on surface 2.
    pub p2: Point3,
    /// Center of the rolling ball.
    pub center: Point3,
    /// Fillet radius at this section.
    pub radius: f64,
    /// Surface 1 parameters (u, v) at the contact point.
    pub uv1: (f64, f64),
    /// Surface 2 parameters (u, v) at the contact point.
    pub uv2: (f64, f64),
    /// Spine parameter where this section was computed.
    pub t: f64,
}

impl CircSection {
    /// Normal direction of the section plane (spine tangent).
    pub fn plane_normal(&self, spine_tangent: Vec3) -> Vec3 {
        spine_tangent.normalized()
    }

    /// Half-angle of the fillet arc (angle from center to each contact).
    pub fn half_angle(&self) -> f64 {
        let d = (self.p1 - self.p2).length();
        if self.radius < f64::EPSILON {
            return 0.0;
        }
        (d / (2.0 * self.radius)).clamp(-1.0, 1.0).asin()
    }
}
```

- [ ] **Step 5: Create spine.rs**

```rust
//! Spine: ordered edge chain with arc-length parameterization.
//!
//! A spine represents the guideline along which a fillet or chamfer is
//! computed. It may consist of multiple edges forming a G1-continuous chain.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::edge::EdgeId;
use brepkit_topology::Topology;

/// An ordered chain of edges forming the fillet guideline.
#[derive(Debug, Clone)]
pub struct Spine {
    /// Ordered edge IDs in the chain.
    edges: Vec<EdgeId>,
    /// Cumulative arc-length at each edge boundary.
    /// `params[0] = 0`, `params[i]` = cumulative length through edge `i-1`.
    params: Vec<f64>,
    /// Total arc length of the spine.
    length: f64,
    /// Whether the chain forms a closed loop.
    is_closed: bool,
}

impl Spine {
    /// Build a spine from a single edge.
    pub fn from_single_edge(topo: &Topology, edge_id: EdgeId) -> Self {
        let edge = topo.edge(edge_id);
        let p_start = topo.vertex(edge.start_vertex).position;
        let p_end = topo.vertex(edge.end_vertex).position;
        let length = (p_end - p_start).length();

        Self {
            edges: vec![edge_id],
            params: vec![0.0, length],
            length,
            is_closed: false,
        }
    }

    /// Build a spine from an ordered chain of edges.
    ///
    /// Edges must be G1-continuous (verified by caller).
    pub fn from_chain(topo: &Topology, edges: Vec<EdgeId>) -> Self {
        let mut params = Vec::with_capacity(edges.len() + 1);
        params.push(0.0);
        let mut cumulative = 0.0;

        for &eid in &edges {
            let edge = topo.edge(eid);
            let p_start = topo.vertex(edge.start_vertex).position;
            let p_end = topo.vertex(edge.end_vertex).position;
            cumulative += (p_end - p_start).length();
            params.push(cumulative);
        }

        let is_closed = if edges.len() >= 2 {
            let first = topo.edge(edges[0]);
            let last = topo.edge(edges[edges.len() - 1]);
            first.start_vertex == last.end_vertex
        } else {
            false
        };

        Self {
            edges,
            params,
            length: cumulative,
            is_closed,
        }
    }

    /// Total arc length.
    pub fn length(&self) -> f64 {
        self.length
    }

    /// Number of edges in the chain.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Whether the spine forms a closed loop.
    pub fn is_closed(&self) -> bool {
        self.is_closed
    }

    /// The edges in order.
    pub fn edges(&self) -> &[EdgeId] {
        &self.edges
    }

    /// Map a global spine parameter `s ∈ [0, length]` to `(edge_index, local_t ∈ [0,1])`.
    pub fn locate(&self, s: f64) -> (usize, f64) {
        let s_clamped = s.clamp(0.0, self.length);
        for i in 0..self.edges.len() {
            let s0 = self.params[i];
            let s1 = self.params[i + 1];
            if s_clamped <= s1 || i == self.edges.len() - 1 {
                let edge_len = s1 - s0;
                let t = if edge_len > f64::EPSILON {
                    (s_clamped - s0) / edge_len
                } else {
                    0.0
                };
                return (i, t.clamp(0.0, 1.0));
            }
        }
        (self.edges.len() - 1, 1.0)
    }

    /// Evaluate the 3D point on the spine at global parameter `s`.
    pub fn evaluate(&self, topo: &Topology, s: f64) -> Point3 {
        let (idx, t) = self.locate(s);
        let edge = topo.edge(self.edges[idx]);
        let p0 = topo.vertex(edge.start_vertex).position;
        let p1 = topo.vertex(edge.end_vertex).position;
        // Linear interpolation for now; curved edges need curve evaluation
        p0 + (p1 - p0) * t
    }

    /// Evaluate the tangent direction on the spine at global parameter `s`.
    pub fn tangent(&self, topo: &Topology, s: f64) -> Vec3 {
        let (idx, _t) = self.locate(s);
        let edge = topo.edge(self.edges[idx]);
        let p0 = topo.vertex(edge.start_vertex).position;
        let p1 = topo.vertex(edge.end_vertex).position;
        (p1 - p0).normalized()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_topology::Topology;

    fn make_line_edge(topo: &mut Topology, a: Point3, b: Point3) -> EdgeId {
        let v0 = topo.add_vertex(a);
        let v1 = topo.add_vertex(b);
        topo.add_edge(
            v0,
            v1,
            brepkit_topology::edge::EdgeCurve::Line,
        )
    }

    #[test]
    fn single_edge_spine_length() {
        let mut topo = Topology::new();
        let eid = make_line_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
        );
        let spine = Spine::from_single_edge(&topo, eid);
        assert!((spine.length() - 10.0).abs() < 1e-10);
        assert_eq!(spine.edge_count(), 1);
        assert!(!spine.is_closed());
    }

    #[test]
    fn locate_maps_parameter_correctly() {
        let mut topo = Topology::new();
        let eid = make_line_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
        );
        let spine = Spine::from_single_edge(&topo, eid);
        let (idx, t) = spine.locate(5.0);
        assert_eq!(idx, 0);
        assert!((t - 0.5).abs() < 1e-10);
    }

    #[test]
    fn evaluate_midpoint() {
        let mut topo = Topology::new();
        let eid = make_line_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
        );
        let spine = Spine::from_single_edge(&topo, eid);
        let mid = spine.evaluate(&topo, 5.0);
        assert!((mid - Point3::new(5.0, 0.0, 0.0)).length() < 1e-10);
    }
}
```

- [ ] **Step 6: Create stripe.rs (initially empty struct)**

```rust
//! Stripe: a fillet band connecting two adjacent faces.

use brepkit_math::curves2d::Curve2D;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::Point3;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::section::CircSection;
use crate::spine::Spine;

/// A fillet band (one per edge or edge chain) connecting two faces.
#[derive(Debug, Clone)]
pub struct Stripe {
    /// The spine (guideline) for this stripe.
    pub spine: Spine,
    /// The blend surface (NURBS or analytic).
    pub surface: FaceSurface,
    /// Contact PCurve on face 1 (UV-space).
    pub pcurve1: Curve2D,
    /// Contact PCurve on face 2 (UV-space).
    pub pcurve2: Curve2D,
    /// 3D contact curve on face 1.
    pub contact1: NurbsCurve,
    /// 3D contact curve on face 2.
    pub contact2: NurbsCurve,
    /// The two adjacent faces.
    pub face1: FaceId,
    pub face2: FaceId,
    /// Cross-sections computed during walking.
    pub sections: Vec<CircSection>,
}

/// Result from computing a single stripe (before topology reconstruction).
pub struct StripeResult {
    /// The blend stripe data.
    pub stripe: Stripe,
    /// New edges created for the blend surface boundaries.
    pub new_edges: Vec<EdgeId>,
}
```

- [ ] **Step 7: Create stub files for modules not yet implemented**

Create empty stub modules so `lib.rs` compiles. Each will be filled in later tasks:

For each of `blend_func.rs`, `walker.rs`, `analytic.rs`, `fillet_builder.rs`, `chamfer_builder.rs`, `corner.rs`, `trimmer.rs`:

```rust
//! [Module description — placeholder].
```

- [ ] **Step 8: Verify crate compiles**

Run: `cargo check -p brepkit-blend`
Expected: compiles with no errors (maybe warnings about unused code)

- [ ] **Step 9: Update check-boundaries.sh**

Read `scripts/check-boundaries.sh`, find the `check_deps` lines, add:

```bash
check_deps "blend"      "brepkit-math" "brepkit-topology"
```

Also update the `operations` line to add `"brepkit-blend"`:

```bash
check_deps "operations" "brepkit-math" "brepkit-topology" "brepkit-algo" "brepkit-blend"
```

And update the `wasm` line to add `"brepkit-blend"`.

- [ ] **Step 10: Update CLAUDE.md layer table**

Add `blend` row to the layer dependency table at the top of CLAUDE.md:

```
| `blend` | `math`, `topology` |
```

And add to the "Allowed `use` paths per crate" section:

```
- `blend/src/**` → `brepkit_math::*`, `brepkit_topology::*`
```

Also update `operations/src/**` to include `brepkit_blend::*`.

- [ ] **Step 11: Run boundary check + tests**

Run: `./scripts/check-boundaries.sh`
Expected: pass (new crate within allowed deps)

Run: `cargo test -p brepkit-blend`
Expected: pass (radius_law + spine + section tests)

- [ ] **Step 12: Commit**

```bash
git add crates/blend/ scripts/check-boundaries.sh CLAUDE.md
git commit -m "feat(blend): scaffold brepkit-blend crate with data structures

New L1.5 crate containing Spine, Stripe, CircSection, RadiusLaw,
BlendError, and BlendResult. Stub modules for walker, blend_func,
analytic, builders, corner, and trimmer."
```

---

## Task 3: Blend constraint functions

Implement the `BlendFunction` trait and the constant-radius fillet constraint (`ConstRadBlend`). This is the mathematical core — 4 equations in 4 unknowns encoding the rolling-ball constraint.

**Files:**
- Create: `crates/blend/src/blend_func.rs`

**OCCT reference:** `~/Git/OCCT/src/ModelingAlgorithms/TKFillet/BlendFunc/BlendFunc_ConstRad.cxx`

The constraint equations:
1. **Planarity:** The ball center lies on a plane perpendicular to the spine — `nplan · (C - guide_point) = 0`
2. **Equidistance:** `C = P₁ + R·N₁ = P₂ + R·N₂` where N₁, N₂ are surface normals projected onto the perpendicular plane

This gives residual vector `F(u1,v1,u2,v2) = [f₁, f₂, f₃, f₄]`:
- `f₁ = nplan · ((P₁ + P₂)/2 - guide_point)` (planarity)
- `f₂₋₄ = (P₁ + R·npn₁) - (P₂ + R·npn₂)` (3 components, equidistance)

where `npn_i` = perpendicular-projected normal = `(nplan × N_i) × nplan / |nplan × N_i|` signed by `nplan · N_i`.

- [ ] **Step 1: Write failing tests for ConstRadBlend**

```rust
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_math::vec::{Point3, Vec3};

    #[test]
    fn const_rad_value_is_zero_at_known_solution() {
        // Two planes at 90°: z=0 and x=0, edge along y-axis
        // Fillet radius 1.0: ball center at (1, y, 1)
        // Contact on z=0 plane at (1, y, 0), on x=0 plane at (0, y, 1)
        let blend = ConstRadBlend { radius: 1.0 };
        let guide = Point3::new(0.0, 5.0, 0.0);
        let nplan = Vec3::new(0.0, 1.0, 0.0); // spine along y

        let ctx = BlendContext {
            guide_point: guide,
            nplan,
        };

        // Plane 1 (z=0): S1(u1,v1) = (u1, v1, 0), N1 = (0,0,1)
        // Contact at (1.0, 5.0, 0.0) → u1=1.0, v1=5.0
        // Plane 2 (x=0): S2(u2,v2) = (0, u2, v2), N2 = (-1,0,0)
        // Contact at (0.0, 5.0, 1.0) → u2=5.0, v2=1.0
        let params = BlendParams {
            u1: 1.0, v1: 5.0,
            u2: 5.0, v2: 1.0,
        };

        // We need actual surface objects for this test.
        // This test verifies the constraint system setup — detailed
        // numerical tests come after the walker is integrated.
    }

    #[test]
    fn const_rad_jacobian_matches_finite_difference() {
        // Numerical Jacobian verification using central differences
        // This is the key correctness test for the Newton solver
    }
}
```

The full tests require surface objects. Write tests using `Plane` surfaces (simplest case) where the analytic solution is known.

- [ ] **Step 2: Implement BlendFunction trait and BlendParams**

```rust
//! Blend constraint functions for fillet and chamfer.
//!
//! Each blend function encodes a system of 4 equations in 4 unknowns
//! `(u1, v1, u2, v2)` parameterized by a spine position. The walker
//! uses Newton-Raphson to solve this system at each step.

use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};
use crate::section::CircSection;

/// Parameters of the blend solution at a single spine point.
#[derive(Debug, Clone, Copy)]
pub struct BlendParams {
    /// Parameter on surface 1.
    pub u1: f64,
    pub v1: f64,
    /// Parameter on surface 2.
    pub u2: f64,
    pub v2: f64,
}

/// Context provided by the spine at the current walking position.
#[derive(Debug, Clone, Copy)]
pub struct BlendContext {
    /// 3D point on the spine (guide curve).
    pub guide_point: Point3,
    /// Normal to the section plane (= spine tangent direction).
    pub nplan: Vec3,
}

/// Blend constraint function interface.
pub trait BlendFunction {
    /// Evaluate residual `F(u1,v1,u2,v2)` at the given context.
    /// Returns `[f64; 4]` — should be zero at a valid solution.
    fn value(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [f64; 4];

    /// Evaluate the 4×4 Jacobian `∂F/∂(u1,v1,u2,v2)`.
    fn jacobian(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [[f64; 4]; 4];

    /// Extract the cross-section from a converged solution.
    fn section(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> CircSection;
}
```

- [ ] **Step 3: Implement ConstRadBlend**

Follow OCCT's `BlendFunc_ConstRad::Values()` logic. The key math:

```rust
/// Constant-radius fillet blend function.
pub struct ConstRadBlend {
    pub radius: f64,
}

impl BlendFunction for ConstRadBlend {
    fn value(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [f64; 4] {
        let p1 = surf1.evaluate(params.u1, params.v1);
        let p2 = surf2.evaluate(params.u2, params.v2);
        let n1 = surf1.normal(params.u1, params.v1);
        let n2 = surf2.normal(params.u2, params.v2);

        // Project normals onto section plane (perpendicular to nplan)
        let npn1 = project_normal_to_section(n1, ctx.nplan);
        let npn2 = project_normal_to_section(n2, ctx.nplan);

        // f1: planarity — midpoint of P1,P2 lies on the section plane
        let mid = (p1 + p2.to_vec()) * 0.5;
        let f1 = ctx.nplan.dot((mid - ctx.guide_point).into());

        // f2-f4: equidistance — P1 + R*npn1 = P2 + R*npn2
        let lhs = p1 + npn1 * self.radius;
        let rhs = p2 + npn2 * self.radius;
        let diff = lhs - rhs;

        [f1, diff.x(), diff.y(), diff.z()]
    }

    fn jacobian(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [[f64; 4]; 4] {
        // Analytic Jacobian using surface partial derivatives.
        // Each column: ∂F/∂u1, ∂F/∂v1, ∂F/∂u2, ∂F/∂v2
        //
        // For the planarity equation (f1):
        //   ∂f1/∂u1 = nplan · (∂P1/∂u1) / 2
        //   ∂f1/∂v1 = nplan · (∂P1/∂v1) / 2
        //   etc.
        //
        // For the equidistance equations (f2-f4):
        //   ∂f/∂u1 = ∂P1/∂u1 + R * ∂npn1/∂u1
        //   where ∂npn1/∂u1 depends on ∂N1/∂u1 (second derivatives)
        //
        // SIMPLIFICATION for v1: use first-order approximation
        // (ignore ∂npn/∂u terms — valid when radius << surface radius of curvature).
        // This matches OCCT's approach for the majority of cases.

        let du1 = surf1.partial_u(params.u1, params.v1);
        let dv1 = surf1.partial_v(params.u1, params.v1);
        let du2 = surf2.partial_u(params.u2, params.v2);
        let dv2 = surf2.partial_v(params.u2, params.v2);

        let nplan = ctx.nplan;

        // Row 0: planarity
        let j00 = nplan.dot(du1) * 0.5;
        let j01 = nplan.dot(dv1) * 0.5;
        let j02 = nplan.dot(du2) * 0.5;
        let j03 = nplan.dot(dv2) * 0.5;

        // Rows 1-3: equidistance (first-order: ∂P1/∂u - ∂P2/∂u, ignoring ∂npn/∂u)
        [
            [j00, j01, j02, j03],
            [du1.x(), dv1.x(), -du2.x(), -dv2.x()],
            [du1.y(), dv1.y(), -du2.y(), -dv2.y()],
            [du1.z(), dv1.z(), -du2.z(), -dv2.z()],
        ]
    }

    fn section(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> CircSection {
        let p1 = surf1.evaluate(params.u1, params.v1);
        let p2 = surf2.evaluate(params.u2, params.v2);
        let n1 = surf1.normal(params.u1, params.v1);
        let npn1 = project_normal_to_section(n1, ctx.nplan);
        let center = p1 + npn1 * self.radius;

        CircSection {
            p1,
            p2,
            center,
            radius: self.radius,
            uv1: (params.u1, params.v1),
            uv2: (params.u2, params.v2),
            t: 0.0, // Caller sets this
        }
    }
}

/// Project a surface normal onto the section plane (perpendicular to nplan).
///
/// Returns a unit vector in the section plane pointing "outward" from the surface.
/// Sign convention: if `nplan · N > 0`, the projected normal points in the
/// same half-space as N projected onto the section plane.
fn project_normal_to_section(normal: Vec3, nplan: Vec3) -> Vec3 {
    let cross = nplan.cross(normal);
    let cross_len = cross.length();
    if cross_len < 1e-15 {
        // Normal is parallel to nplan — degenerate case
        return Vec3::new(0.0, 0.0, 0.0);
    }
    let projected = cross.cross(nplan);
    let sign = if nplan.dot(normal) >= 0.0 { 1.0 } else { -1.0 };
    projected * (sign / cross_len)
}
```

- [ ] **Step 4: Implement ChamferBlend and ChamferAngleBlend**

These use distance-based constraints instead of radius:

```rust
/// Two-distance chamfer blend.
pub struct ChamferBlend {
    pub d1: f64,
    pub d2: f64,
}

/// Distance-angle chamfer blend.
pub struct ChamferAngleBlend {
    pub distance: f64,
    pub angle: f64,
}
```

The constraint equations are simpler: contact points at fixed distances from the spine rather than equidistant from a rolling ball center. Implement `BlendFunction` for both — the structure mirrors `ConstRadBlend` but the residual equations differ.

- [ ] **Step 5: Implement EvolRadBlend (variable radius)**

Wraps `ConstRadBlend` with a `RadiusLaw` that changes the radius at each spine parameter:

```rust
/// Variable-radius fillet blend.
pub struct EvolRadBlend {
    pub law: RadiusLaw,
}
```

The `value` and `jacobian` methods evaluate `law.evaluate(t)` to get the radius at the current spine parameter, then delegate to the same math as `ConstRadBlend`.

- [ ] **Step 6: Write comprehensive tests**

Test the Jacobian via finite differences (central differencing with `h=1e-7`). For two known planes at 90°, verify the constraint residual is near zero at the known analytic solution.

- [ ] **Step 7: Run tests**

Run: `cargo test -p brepkit-blend -- blend_func`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add crates/blend/src/blend_func.rs
git commit -m "feat(blend): implement blend constraint functions

ConstRadBlend (constant-radius fillet), EvolRadBlend (variable-radius),
ChamferBlend (two-distance), ChamferAngleBlend (distance-angle).
Each provides constraint residual + analytic Jacobian for Newton-Raphson."
```

---

## Task 4: Walking engine

Implement the Newton-Raphson marching algorithm that traces the blend surface along the spine.

**Files:**
- Create: `crates/blend/src/walker.rs`

**OCCT reference:** `~/Git/OCCT/src/ModelingAlgorithms/TKFillet/BRepBlend/BRepBlend_Walking.cxx`

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn walk_two_planes_at_90_degrees() {
        // Two planes at 90°: z=0 and x=0, edge along y-axis from (0,0,0) to (0,10,0)
        // Fillet radius 1.0
        // Expected: straight contact lines at x=1 on z=0 plane and z=1 on x=0 plane
        // The walk should produce uniform sections along the spine
    }

    #[test]
    fn find_start_converges_for_simple_case() {
        // Verify Newton-Raphson convergence at the midpoint of a plane-plane fillet
    }

    #[test]
    fn adaptive_step_reduces_on_divergence() {
        // Verify step halving when Newton fails to converge
    }
}
```

- [ ] **Step 2: Implement the 4×4 linear solver**

The walker needs to solve `J·δ = -F` at each Newton step. Implement a simple 4×4 Gaussian elimination with partial pivoting (no external dependency needed):

```rust
/// Solve 4×4 linear system Ax = b via Gaussian elimination with partial pivoting.
/// Returns `None` if the matrix is singular.
fn solve_4x4(a: [[f64; 4]; 4], b: [f64; 4]) -> Option<[f64; 4]> {
    let mut aug = [[0.0; 5]; 4];
    for i in 0..4 {
        for j in 0..4 {
            aug[i][j] = a[i][j];
        }
        aug[i][4] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..4 {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..4 {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None; // Singular
        }
        aug.swap(col, max_row);

        for row in (col + 1)..4 {
            let factor = aug[row][col] / aug[col][col];
            for j in col..5 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = [0.0; 4];
    for i in (0..4).rev() {
        x[i] = aug[i][4];
        for j in (i + 1)..4 {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Some(x)
}
```

- [ ] **Step 3: Implement Walker struct and find_start**

```rust
/// Configuration for the walking engine.
pub struct WalkerConfig {
    /// 3D convergence tolerance.
    pub tol_3d: f64,
    /// Maximum Newton iterations per step.
    pub max_newton_iters: usize,
    /// Maximum step along spine (fraction of spine length).
    pub max_step_fraction: f64,
    /// Minimum step along spine (below this → failure).
    pub min_step: f64,
    /// Maximum number of walking steps.
    pub max_steps: usize,
}

impl Default for WalkerConfig {
    fn default() -> Self {
        Self {
            tol_3d: 1e-7,
            max_newton_iters: 20,
            max_step_fraction: 0.05, // 5% of spine length
            min_step: 1e-10,
            max_steps: 1000,
        }
    }
}

/// Result of a successful walk.
pub struct WalkResult {
    /// Cross-sections collected along the spine.
    pub sections: Vec<CircSection>,
    /// Final blend parameters at the end of the walk.
    pub end_params: BlendParams,
}

/// Walking engine for tracing blend surfaces.
pub struct Walker<'a, F: BlendFunction> {
    func: &'a F,
    surf1: &'a dyn ParametricSurface,
    surf2: &'a dyn ParametricSurface,
    spine: &'a Spine,
    topo: &'a Topology,
    config: WalkerConfig,
}
```

Implement `find_start`:
- Initial guess: project spine midpoint onto both surfaces, use those UV coords
- Newton-Raphson loop: `δ = -J⁻¹·F`, `params += δ`, check `|F| < tol_3d`
- Return `BlendParams` or `BlendError::StartSolutionFailure`

- [ ] **Step 4: Implement the walking loop**

```rust
impl<F: BlendFunction> Walker<'_, F> {
    pub fn walk(
        &self,
        start_params: BlendParams,
        s_start: f64,
        s_end: f64,
    ) -> Result<WalkResult, BlendError> {
        let mut sections = Vec::new();
        let mut params = start_params;
        let mut s = s_start;
        let direction = if s_end > s_start { 1.0 } else { -1.0 };
        let mut step = (s_end - s_start).abs() * self.config.max_step_fraction;
        let mut step_count = 0usize;

        while (s_end - s) * direction > self.config.min_step {
            step_count += 1;
            if step_count > self.config.max_steps {
                return Err(BlendError::WalkingFailure {
                    edge: self.spine.edges()[0],
                    t: s,
                    residual: f64::NAN,
                });
            }
            let s_next = s + step * direction;
            let s_clamped = if direction > 0.0 {
                s_next.min(s_end)
            } else {
                s_next.max(s_end)
            };

            let ctx = self.make_context(s_clamped);

            // Predict: linear extrapolation from previous solution
            // (for first step, use start_params as-is)
            let predicted = params; // TODO: linear extrapolation from last two

            // Correct: Newton-Raphson
            match self.newton_correct(predicted, &ctx) {
                Ok(corrected) => {
                    params = corrected;
                    s = s_clamped;
                    let mut section = self.func.section(
                        self.surf1, self.surf2, &params, &ctx,
                    );
                    section.t = s;
                    sections.push(section);
                    // Increase step on success (up to max)
                    step = (step * 1.5).min(
                        (s_end - s_start).abs() * self.config.max_step_fraction,
                    );
                }
                Err(_) => {
                    // Halve step and retry
                    step *= 0.5;
                    if step < self.config.min_step {
                        return Err(BlendError::WalkingFailure {
                            edge: self.spine.edges()[0],
                            t: s,
                            residual: f64::NAN,
                        });
                    }
                }
            }
        }

        Ok(WalkResult {
            sections,
            end_params: params,
        })
    }

    fn newton_correct(
        &self,
        initial: BlendParams,
        ctx: &BlendContext,
    ) -> Result<BlendParams, ()> {
        let mut params = initial;
        for _ in 0..self.config.max_newton_iters {
            let f = self.func.value(self.surf1, self.surf2, &params, ctx);
            let residual = (f[0]*f[0] + f[1]*f[1] + f[2]*f[2] + f[3]*f[3]).sqrt();
            if residual < self.config.tol_3d {
                return Ok(params);
            }
            let j = self.func.jacobian(self.surf1, self.surf2, &params, ctx);
            let Some(delta) = solve_4x4(j, [-f[0], -f[1], -f[2], -f[3]]) else {
                return Err(());
            };
            params.u1 += delta[0];
            params.v1 += delta[1];
            params.u2 += delta[2];
            params.v2 += delta[3];
        }
        Err(()) // Did not converge
    }

    fn make_context(&self, s: f64) -> BlendContext {
        BlendContext {
            guide_point: self.spine.evaluate(self.topo, s),
            nplan: self.spine.tangent(self.topo, s),
        }
    }
}
```

- [ ] **Step 5: Implement NURBS surface approximation from walked sections**

After walking produces a sequence of `CircSection`s, approximate the blend surface:
- U-direction: rational quadratic arc (exact circular cross-section)
- V-direction: cubic B-spline fit through section control points

```rust
/// Approximate the blend surface from walked cross-sections.
pub fn approximate_blend_surface(
    sections: &[CircSection],
) -> Result<NurbsSurface, BlendError> {
    // For each section, compute 3 control points of the circular arc:
    //   cp0 = p1, cp1 = center + (p1-center+p2-center)/cos(half_angle), cp2 = p2
    //   weight: [1, cos(half_angle), 1]
    //
    // Stack sections along V, fit cubic B-spline through corresponding CPs
    // Result: degree (2, 3) NURBS surface
    todo!()
}
```

This is ~100 LOC using `brepkit-math`'s NURBS fitting infrastructure.

- [ ] **Step 6: Write integration tests**

Test the full walk pipeline on plane-plane (known analytic solution):
- Two perpendicular planes, straight edge
- Verify all sections have correct contact points and radius
- Verify the approximated NURBS surface passes through contact points

- [ ] **Step 7: Run tests**

Run: `cargo test -p brepkit-blend -- walker`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add crates/blend/src/walker.rs
git commit -m "feat(blend): implement Newton-Raphson walking engine

Adaptive-step walker traces blend surface along spine by solving
4x4 constraint system. Includes 4x4 Gaussian elimination solver,
convergence control, and NURBS surface approximation from sections."
```

---

## Task 5: Analytic fast paths

Closed-form fillet/chamfer for common surface pairs (plane-plane, plane-cylinder, plane-cone, cylinder-cylinder).

**Files:**
- Create: `crates/blend/src/analytic.rs`

**OCCT reference:** `~/Git/OCCT/src/ModelingAlgorithms/TKFillet/ChFiKPart/ChFiKPart_ComputeData.cxx` (lines relevant to plane-plane case)

- [ ] **Step 1: Write failing tests for plane-plane fillet**

A plane-plane fillet at 90° with radius R produces a cylinder of radius R whose axis lies along the edge, offset by R from each plane.

```rust
#[test]
fn plane_plane_fillet_produces_cylinder() {
    // Two planes at 90°: z=0 and x=0
    // Edge along y-axis
    // Fillet R=2.0
    // Expected: CylindricalSurface with radius=2, axis along y,
    //           origin at (2, 0, 2) — the rolling ball center line
}
```

- [ ] **Step 2: Implement dispatch function**

```rust
/// Attempt an analytic fillet for the given surface pair.
/// Returns `None` if no analytic solution exists.
pub fn try_analytic_fillet(
    topo: &Topology,
    surf1: &FaceSurface,
    surf2: &FaceSurface,
    spine: &Spine,
    radius: f64,
) -> Option<StripeResult> {
    match (surf1, surf2) {
        (
            FaceSurface::Plane { normal: n1, d: d1 },
            FaceSurface::Plane { normal: n2, d: d2 },
        ) => plane_plane_fillet(topo, *n1, *d1, *n2, *d2, spine, radius),
        (FaceSurface::Plane { normal, d }, FaceSurface::Cylinder(c))
        | (FaceSurface::Cylinder(c), FaceSurface::Plane { normal, d }) => {
            plane_cylinder_fillet(topo, *normal, *d, c, spine, radius)
        }
        // ... plane-cone, cyl-cyl
        _ => None,
    }
}
```

- [ ] **Step 3: Implement plane_plane_fillet**

The fillet surface is a cylinder:
- Axis = edge direction (spine tangent)
- Radius = fillet radius R
- Center line = spine offset by R along the angle bisector of the two plane normals
- Contact lines = offset each plane by R along its normal, intersect with fillet cylinder

```rust
fn plane_plane_fillet(
    topo: &Topology,
    n1: Vec3,
    _d1: f64,
    n2: Vec3,
    _d2: f64,
    spine: &Spine,
    radius: f64,
) -> Option<StripeResult> {
    let edge_dir = spine.tangent(topo, spine.length() * 0.5);

    // Angle bisector in the section plane
    let bisector = (n1 + n2).normalized();
    let half_angle = n1.dot(n2).clamp(-1.0, 1.0).acos() * 0.5;

    // Center offset from edge
    let offset = radius / half_angle.sin();
    let center_line_origin = spine.evaluate(topo, 0.0) + bisector * offset;

    // Build cylindrical surface
    let cyl = CylindricalSurface::new(center_line_origin, edge_dir, radius).ok()?;

    // Build contact curves, PCurves, etc.
    // ...
    todo!()
}
```

- [ ] **Step 4: Implement plane_cylinder_fillet**

The fillet surface between a plane and a cylinder is typically a torus section. Follow OCCT's `ChFiKPart_ComputeData` for the plane-cylinder case.

- [ ] **Step 5: Implement plane_cone_fillet and cylinder_cylinder_fillet**

Similar patterns — offset surfaces, intersect to find the blend surface center locus, construct toric or cylindrical result.

- [ ] **Step 6: Write tests for each surface pair**

For each analytic case, verify:
- The returned surface type is correct (Cylinder, Torus, etc.)
- Contact points lie on both original surfaces
- The blend surface is tangent to both surfaces at contact

- [ ] **Step 7: Run tests and commit**

Run: `cargo test -p brepkit-blend -- analytic`

```bash
git add crates/blend/src/analytic.rs
git commit -m "feat(blend): analytic fast paths for plane-plane, plane-cyl, plane-cone, cyl-cyl

Closed-form fillet/chamfer computation for the 4 most common surface
pairs. Bypasses the walking engine for ~80% of real-world fillets."
```

---

## Task 6: Face trimmer

Trim original faces along fillet contact curves.

**Files:**
- Create: `crates/blend/src/trimmer.rs`

- [ ] **Step 1: Write failing tests**

```rust
#[test]
fn trim_planar_face_along_straight_contact() {
    // A rectangular planar face trimmed by a straight contact line
    // parallel to one edge. Result should be a smaller rectangle.
}

#[test]
fn trim_cylindrical_face_along_contact_curve() {
    // A cylindrical face trimmed by a contact curve.
    // The result should have the contact curve as a new boundary edge.
}
```

- [ ] **Step 2: Implement face trimming for planar faces**

For planar faces, the contact curve projects to a straight line in the face plane. Split the face polygon along this line:

```rust
/// Trim a face along a contact curve, returning the outer (kept) sub-face.
pub fn trim_face(
    topo: &mut Topology,
    face_id: FaceId,
    contact_3d: &NurbsCurve,
    contact_pcurve: &Curve2D,
    keep_side: TrimSide,
) -> Result<FaceId, BlendError> {
    // 1. Find where the contact curve intersects the face boundary edges
    // 2. Split boundary edges at intersection points
    // 3. Build two wire loops: one for the kept side, one for the discarded side
    // 4. Create new face with the kept wire
    todo!()
}
```

- [ ] **Step 3: Implement for periodic (cylindrical/spherical) faces**

Handle periodic UV domains: unwrap parameters, split at seam crossings, rebuild wires.

- [ ] **Step 4: Tests and commit**

```bash
git add crates/blend/src/trimmer.rs
git commit -m "feat(blend): face trimmer for contact curve splitting

Splits original faces along fillet/chamfer contact curves.
Handles planar faces (polygon splitting) and periodic faces
(UV unwrapping + seam crossing)."
```

---

## Task 7: Fillet builder

Orchestrates the full fillet pipeline: spine construction → stripe computation → trimming → assembly.

**Files:**
- Create: `crates/blend/src/fillet_builder.rs`

- [ ] **Step 1: Write integration test**

```rust
#[test]
fn fillet_unit_cube_single_edge() {
    // Create a unit cube, fillet one edge with R=0.1
    // Verify: face count = 7 (6 original - 2 trimmed + 2 trimmed + 1 blend)
    // Verify: solid is manifold and closed
    // Verify: volume < 1.0 (material removed)
}
```

- [ ] **Step 2: Implement G1 chain propagation**

Port the existing `expand_g1_chain()` from `crates/operations/src/fillet.rs` — the logic is the same (BFS through tangent-continuous edges).

- [ ] **Step 3: Implement FilletBuilder**

```rust
pub struct FilletBuilder<'a> {
    topo: &'a mut Topology,
    solid: SolidId,
    stripes: Vec<Stripe>,
    trimmed_faces: Vec<FaceId>,
}

impl<'a> FilletBuilder<'a> {
    pub fn new(topo: &'a mut Topology, solid: SolidId) -> Self { ... }

    /// Add edges to fillet with constant radius.
    pub fn add_edges(&mut self, edges: &[EdgeId], radius: f64) -> &mut Self { ... }

    /// Compute all stripes (blend surfaces).
    pub fn compute(&mut self) -> Result<(), BlendError> {
        // 1. Build spines from edge sets (with G1 propagation)
        // 2. For each spine:
        //    a. Get adjacent face pair
        //    b. Try analytic fast path
        //    c. If no analytic: use walker
        //    d. Store stripe result
        Ok(())
    }

    /// Trim faces and assemble the final solid.
    pub fn build(self) -> Result<BlendResult, BlendError> {
        // 1. Trim each adjacent face along contact curves
        // 2. Collect: trimmed faces + blend surfaces
        // 3. Build new shell → solid
        // 4. Validate manifold + closed
        todo!()
    }
}
```

- [ ] **Step 4: Implement solid assembly**

Build the final solid from trimmed faces + blend surfaces:
- Create edges for blend surface boundaries (contact curves become shared edges)
- Build wires from the edges
- Build faces from wires
- Assemble shell → solid
- Validate result

- [ ] **Step 5: End-to-end test with cube**

Use `brepkit_operations::primitives::make_box` to create a test cube (if accessible as dev-dep), or build topology manually. Fillet one edge, verify volume and topology.

- [ ] **Step 6: Run tests and commit**

```bash
git add crates/blend/src/fillet_builder.rs
git commit -m "feat(blend): fillet builder with G1 propagation and solid assembly

Orchestrates: spine construction → analytic/walking stripe computation
→ face trimming → solid assembly. Full pipeline for constant-radius
and variable-radius fillets."
```

---

## Task 8: Chamfer builder

Similar to fillet builder but with chamfer blend functions.

**Files:**
- Create: `crates/blend/src/chamfer_builder.rs`

- [ ] **Step 1: Write tests**

```rust
#[test]
fn chamfer_unit_cube_single_edge() {
    // Symmetric chamfer d=0.1 on one edge
    // Verify: flat bevel face, volume close to 1.0 - 0.001
}

#[test]
fn chamfer_distance_angle() {
    // Distance-angle chamfer: d=0.2, angle=45°
    // Verify: asymmetric bevel
}
```

- [ ] **Step 2: Implement ChamferBuilder**

Mirrors `FilletBuilder` structure but:
- Uses `ChamferBlend` or `ChamferAngleBlend` instead of `ConstRadBlend`
- Chamfer surface is ruled (degree 1 in cross-section direction) instead of circular arc
- Corner patches are flat (no sphere/torus needed)

- [ ] **Step 3: Tests and commit**

```bash
git add crates/blend/src/chamfer_builder.rs
git commit -m "feat(blend): chamfer builder with two-distance and distance-angle modes

Supports symmetric, asymmetric, and distance-angle chamfer on all
surface types. Reuses walking engine with chamfer blend functions."
```

---

## Task 9: Corner solver (vertex blends)

**Files:**
- Create: `crates/blend/src/corner.rs`

**OCCT reference:** `~/Git/OCCT/src/ModelingAlgorithms/TKFillet/ChFi3d/ChFi3d_Builder_6.cxx`

- [ ] **Step 1: Write tests**

```rust
#[test]
fn corner_3_edges_symmetric_produces_sphere() {
    // 3 edges meeting at corner with equal radii on orthogonal faces
    // → spherical cap vertex blend
}

#[test]
fn corner_3_edges_asymmetric_produces_coons_patch() {
    // 3 edges with different radii
    // → Coons patch interpolating stripe boundaries
}
```

- [ ] **Step 2: Implement corner classification**

```rust
/// Classify the vertex blend type.
pub fn classify_corner(
    vertex: VertexId,
    stripes: &[Stripe],
    topo: &Topology,
) -> CornerType {
    let n = count_stripes_at_vertex(vertex, stripes);
    match n {
        0 | 1 => CornerType::None,
        2 => CornerType::TwoEdge,
        3 => {
            if all_radii_equal(vertex, stripes) && all_faces_orthogonal(vertex, topo) {
                CornerType::SphereCap
            } else {
                CornerType::CoonsPatch
            }
        }
        _ => CornerType::CoonsPatch,
    }
}
```

- [ ] **Step 3: Implement sphere cap for symmetric 3-edge corners**

```rust
fn build_sphere_cap(
    vertex: VertexId,
    stripes: &[Stripe],
    radius: f64,
    topo: &mut Topology,
) -> Result<FaceId, BlendError> {
    // Sphere center = vertex position + R * (n1 + n2 + n3).normalized * offset
    // Cap boundary = circle through 3 stripe endpoint contact points
    // Build as NURBS (rational quadratic spherical patch)
    todo!()
}
```

- [ ] **Step 4: Implement Coons patch for general corners**

Use `brepkit_operations::fill_face` (Coons patch infrastructure) if accessible, or implement directly:
- Collect boundary curves from stripe endpoints at the vertex
- Build bilinear Coons patch: `S(u,v) = C₁(u)·(1-v) + C₂(u)·v + ...`
- Fit as NURBS surface

- [ ] **Step 5: Tests and commit**

```bash
git add crates/blend/src/corner.rs
git commit -m "feat(blend): vertex blend corner solver

Sphere cap for symmetric 3-edge corners, Coons patch interpolation
for asymmetric and N-edge corners. G1 tangent matching with adjacent
fillet stripes."
```

---

## Task 10: Operations wrappers and WASM bindings

**Files:**
- Modify: `crates/operations/Cargo.toml` — add `brepkit-blend` dep
- Create: `crates/operations/src/blend_ops.rs` — thin wrappers
- Modify: `crates/operations/src/lib.rs` — add module
- Modify: `crates/wasm/Cargo.toml` — add `brepkit-blend` dep
- Modify: `crates/wasm/src/bindings/operations.rs` — add bindings

- [ ] **Step 1: Add dependency to operations Cargo.toml**

```toml
brepkit-blend.workspace = true
```

Also add to workspace `Cargo.toml`:
```toml
brepkit-blend = { path = "crates/blend" }
```

- [ ] **Step 2: Add `Blend` variant to `OperationsError`**

In `crates/operations/src/lib.rs`, add to the `OperationsError` enum:
```rust
    #[error("blend: {0}")]
    Blend(#[from] brepkit_blend::BlendError),
```

- [ ] **Step 3: Create blend_ops.rs wrapper module**

```rust
//! Thin wrappers around `brepkit-blend` for the operations API.

use brepkit_blend::{BlendResult, BlendError, RadiusLaw};
use brepkit_topology::edge::EdgeId;
use brepkit_topology::solid::SolidId;
use brepkit_topology::Topology;
use crate::OperationsError;

/// Fillet edges with constant radius (v2 engine).
pub fn fillet_v2(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
) -> Result<BlendResult, OperationsError> {
    if radius <= 0.0 {
        return Err(OperationsError::InvalidInput {
            reason: "radius must be positive".into(),
        });
    }
    if edges.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "no edges specified".into(),
        });
    }
    let mut builder = brepkit_blend::fillet_builder::FilletBuilder::new(topo, solid);
    builder.add_edges(edges, radius);
    builder.compute().map_err(OperationsError::from)?;
    builder.build().map_err(OperationsError::from)
}

/// Fillet edges with variable radius (v2 engine).
pub fn fillet_variable_v2(
    topo: &mut Topology,
    solid: SolidId,
    edge_laws: &[(EdgeId, RadiusLaw)],
) -> Result<BlendResult, OperationsError> {
    // Similar pattern — build, compute, assemble
    todo!()
}

/// Chamfer edges with two distances (v2 engine).
pub fn chamfer_v2(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    d1: f64,
    d2: f64,
) -> Result<BlendResult, OperationsError> {
    todo!()
}

/// Chamfer edges with distance and angle (v2 engine).
pub fn chamfer_distance_angle(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    distance: f64,
    angle: f64,
) -> Result<BlendResult, OperationsError> {
    todo!()
}
```

- [ ] **Step 4: Add module to operations lib.rs**

```rust
pub mod blend_ops;
```

- [ ] **Step 5: Add WASM bindings**

In `crates/wasm/src/bindings/operations.rs`, add:

```rust
/// Fillet edges using v2 walking engine.
#[wasm_bindgen(js_name = "filletV2")]
pub fn fillet_v2(
    &mut self,
    solid: u32,
    edge_handles: Vec<u32>,
    radius: f64,
) -> Result<u32, JsError> {
    validate_positive(radius, "radius")?;
    let solid_id = self.resolve_solid(solid)?;
    let edge_ids: Vec<_> = edge_handles
        .iter()
        .map(|&h| self.resolve_edge(h))
        .collect::<Result<_, _>>()?;
    let result = brepkit_operations::blend_ops::fillet_v2(
        self.topo_mut(), solid_id, &edge_ids, radius,
    )?;
    Ok(solid_id_to_u32(result.solid))
}

/// Chamfer edges using v2 engine with distance-angle mode.
#[wasm_bindgen(js_name = "chamferDistanceAngle")]
pub fn chamfer_distance_angle(
    &mut self,
    solid: u32,
    edge_handles: Vec<u32>,
    distance: f64,
    angle: f64,
) -> Result<u32, JsError> {
    validate_positive(distance, "distance")?;
    validate_positive(angle, "angle")?;
    let solid_id = self.resolve_solid(solid)?;
    let edge_ids: Vec<_> = edge_handles
        .iter()
        .map(|&h| self.resolve_edge(h))
        .collect::<Result<_, _>>()?;
    let result = brepkit_operations::blend_ops::chamfer_distance_angle(
        self.topo_mut(), solid_id, &edge_ids, distance, angle,
    )?;
    Ok(solid_id_to_u32(result.solid))
}
```

- [ ] **Step 6: Build WASM target**

Run: `cargo build -p brepkit-wasm --target wasm32-unknown-unknown`
Expected: compiles

- [ ] **Step 7: Run full test suite**

Run: `cargo test --workspace`
Expected: all pass, no regressions

- [ ] **Step 8: Commit**

```bash
git add crates/operations/Cargo.toml crates/operations/src/blend_ops.rs \
       crates/operations/src/lib.rs crates/wasm/Cargo.toml \
       crates/wasm/src/bindings/operations.rs Cargo.toml
git commit -m "feat(ops,wasm): add v2 fillet/chamfer API wrappers and WASM bindings

Thin operations wrappers delegating to brepkit-blend. WASM bindings
expose filletV2, chamferV2, and chamferDistanceAngle."
```

---

## Task 11: Integration tests and old code deprecation

**Files:**
- Modify: `crates/operations/src/fillet.rs` — mark deprecated
- Modify: `crates/operations/src/chamfer.rs` — mark deprecated
- Create: `crates/operations/tests/blend_integration.rs` — end-to-end tests (lives in operations crate since it needs `make_box` etc.)

- [ ] **Step 1: Write comprehensive integration tests**

```rust
//! End-to-end fillet and chamfer tests using primitives.

#[test]
fn fillet_box_single_edge_plane_plane() { ... }

#[test]
fn fillet_box_all_edges_with_vertex_blends() { ... }

#[test]
fn fillet_cylinder_edge_plane_cylinder() { ... }

#[test]
fn chamfer_box_symmetric() { ... }

#[test]
fn chamfer_box_distance_angle() { ... }

#[test]
fn fillet_variable_radius_linear() { ... }

#[test]
fn fillet_variable_radius_scurve() { ... }

#[test]
fn fillet_on_boolean_result() { ... }

#[test]
fn fillet_g1_chain_propagation() { ... }
```

Each test:
1. Creates geometry (box, cylinder, etc.)
2. Applies fillet/chamfer via v2 API
3. Verifies: manifold, closed shell, face count, volume within tolerance

- [ ] **Step 2: Mark old API as deprecated**

In `crates/operations/src/fillet.rs`:
```rust
#[deprecated(note = "Use brepkit_operations::blend_ops::fillet_v2 instead")]
pub fn fillet(...) { ... }

#[deprecated(note = "Use brepkit_operations::blend_ops::fillet_v2 instead")]
pub fn fillet_rolling_ball(...) { ... }
```

Same for `chamfer.rs`.

- [ ] **Step 3: Run full test suite and boundary check**

Run: `cargo test --workspace`
Run: `./scripts/check-boundaries.sh`
Run: `cargo clippy --all-targets -- -D warnings`
Expected: all pass

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(blend): integration tests + deprecate old fillet/chamfer API

Comprehensive end-to-end tests for all fillet/chamfer modes.
Old fillet() and chamfer() functions marked deprecated in favor
of the v2 walking-based engine in brepkit-blend."
```

---

## Summary of Build Order

| Task | Description | Deps | LOC est. |
|------|-------------|------|----------|
| 1 | ParametricSurface partial derivatives | none | ~150 |
| 2 | Crate scaffold + data structures | Task 1 | ~600 |
| 3 | Blend constraint functions | Task 2 | ~600 |
| 4 | Walking engine | Task 3 | ~800 |
| 5 | Analytic fast paths | Task 2 | ~600 |
| 6 | Face trimmer | Task 4 | ~500 |
| 7 | Fillet builder | Tasks 4,5,6 | ~500 |
| 8 | Chamfer builder | Tasks 3,6 | ~300 |
| 9 | Corner solver | Task 7 | ~900 |
| 10 | Operations + WASM wrappers | Tasks 7,8 | ~400 |
| 11 | Integration tests + deprecation | Task 10 | ~2000 |
| **Total** | | | **~7,350** |
