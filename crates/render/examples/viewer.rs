//! Interactive viewer demo: a box fused with a cylinder.
//!
//! Run (needs a display server and the `window` feature):
//!
//! ```text
//! cargo run -p brepkit-render --example viewer --features window
//! ```
//!
//! Controls:
//! - Left-drag: orbit
//! - Right-drag (or Shift + left-drag): pan
//! - Scroll: zoom
//! - Left click on a face: highlight it (click again to clear)
//!
//! Each highlighted face corresponds to a kernel `FaceId`, read back from the
//! GPU id buffer under the cursor.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_render::{ViewOpts, view_solid};
use brepkit_topology::Topology;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A 40x40x20 box with a cylinder rising through its top, fused into one
    // solid so the viewer can show (and pick) the combined faces.
    let mut topo = Topology::new();

    let box_solid = make_box(&mut topo, 40.0, 40.0, 20.0)?;

    // Cylinder base at z=0; lift it so it spans the box and protrudes above.
    let cyl = make_cylinder(&mut topo, 10.0, 35.0)?;
    transform_solid(&mut topo, cyl, &Mat4::translation(20.0, 20.0, 0.0))?;

    let solid = boolean(&mut topo, BooleanOp::Fuse, box_solid, cyl)?;

    let opts = ViewOpts::new("brepkit viewer — box + cylinder (click a face)");
    view_solid(&topo, solid, &opts)?;
    Ok(())
}
