//! Edge projection with hidden-line removal (HLR).
//!
//! Projects a solid's edges onto a view plane and splits each edge into visible
//! and hidden polylines. Occlusion is an **exact** point-in-solid test — no
//! tessellation of faces: a boundary point is hidden when stepping it toward the
//! camera enters the solid (a face is in front of it). Edges themselves are
//! sampled to polylines because a projected drawing is inherently polygonal.

use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::classify::{PointClassification, classify_point};

/// Projected edges, split into visible and hidden 2D polylines (in the view
/// plane's `(x, y)` coordinates).
#[derive(Debug, Clone, Default)]
pub struct ProjectedEdges {
    /// Polylines that are not occluded by the solid.
    pub visible: Vec<Vec<Point2>>,
    /// Polylines hidden behind the solid (empty when `hidden_lines` is false).
    pub hidden: Vec<Vec<Point2>>,
}

/// Project a solid's edges onto the view plane through `origin` with in-plane
/// x-axis `x_axis`, viewed along `direction` (orthographic), classifying each
/// segment as visible or hidden.
///
/// `direction` points from the camera into the scene. `x_axis` is the horizontal
/// view direction (re-orthonormalized against `direction`). `deflection` controls
/// edge-sampling density (the point-classification tolerance is fixed). When
/// `hidden_lines` is false, hidden segments are dropped and `hidden` is left empty.
///
/// # Errors
///
/// Returns [`crate::OperationsError::InvalidInput`] if `direction` or `x_axis`
/// is degenerate, and propagates topology, sampling, and point-classification
/// errors.
pub fn project_edges(
    topo: &Topology,
    solid: SolidId,
    origin: Point3,
    direction: Vec3,
    x_axis: Vec3,
    hidden_lines: bool,
    deflection: f64,
) -> Result<ProjectedEdges, crate::OperationsError> {
    let view = direction
        .normalize()
        .map_err(|_| crate::OperationsError::InvalidInput {
            reason: "projection direction must be non-zero".into(),
        })?;
    // In-plane orthonormal basis: x re-orthonormalized against the view, y = x × view.
    let x = (x_axis - view * view.dot(x_axis))
        .normalize()
        .map_err(|_| crate::OperationsError::InvalidInput {
            reason: "projection x_axis is parallel to the direction".into(),
        })?;
    let y = x.cross(view);

    let project = |p: Point3| -> Point2 {
        let v = p - origin;
        Point2::new(x.dot(v), y.dot(v))
    };

    // Step length for the occlusion probe — far enough to clear the boundary
    // tolerance, small relative to the model. Keyed to the model extent and
    // capped so a large-coordinate model can't push the probe through thin
    // features.
    let bbox = crate::measure::solid_bounding_box(topo, solid)?;
    let diag = (bbox.max - bbox.min).length();
    let eps = (diag * 1e-4).clamp(1e-6, 1e-2);

    // A boundary point is hidden when stepping toward the camera (−view) lands
    // inside the solid, i.e. a face is between it and the camera. A
    // classification error is propagated rather than silently read as
    // "visible", so a degenerate solid surfaces instead of yielding wrong HLR.
    let is_hidden = |p: Point3| -> Result<bool, crate::OperationsError> {
        Ok(
            classify_point(topo, solid, p - view * eps, deflection, 1e-7)?
                == PointClassification::Inside,
        )
    };

    let lines = crate::tessellate::sample_solid_edges(topo, solid, deflection)?;
    let n_edges = lines.offsets.len();
    let mut result = ProjectedEdges::default();

    for i in 0..n_edges {
        let start = lines.offsets[i];
        let end = if i + 1 < n_edges {
            lines.offsets[i + 1]
        } else {
            lines.positions.len()
        };
        let pts = &lines.positions[start..end];
        if pts.len() < 2 {
            continue;
        }

        // Classify each segment by its midpoint, then merge consecutive
        // same-visibility segments into polylines (adjacent runs share the
        // boundary vertex, so the drawing stays connected).
        let seg_hidden: Vec<bool> = (0..pts.len() - 1)
            .map(|j| {
                let mid = Point3::new(
                    0.5 * (pts[j].x() + pts[j + 1].x()),
                    0.5 * (pts[j].y() + pts[j + 1].y()),
                    0.5 * (pts[j].z() + pts[j + 1].z()),
                );
                is_hidden(mid)
            })
            .collect::<Result<Vec<bool>, _>>()?;

        let mut j = 0;
        while j < seg_hidden.len() {
            let hidden = seg_hidden[j];
            let run_start = j;
            while j < seg_hidden.len() && seg_hidden[j] == hidden {
                j += 1;
            }
            // Run covers segments [run_start, j), i.e. points [run_start, j].
            if hidden && !hidden_lines {
                continue;
            }
            let poly: Vec<Point2> = (run_start..=j).map(|k| project(pts[k])).collect();
            if hidden {
                result.hidden.push(poly);
            } else {
                result.visible.push(poly);
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    // An oblique view along (1,1,1): the three edges meeting at the far corner
    // (10,10,10) are unambiguously occluded by the three near faces.
    fn oblique() -> (Point3, Vec3, Vec3) {
        (
            Point3::new(-100.0, -100.0, -100.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, 0.0),
        )
    }

    #[test]
    fn project_box_oblique_view_has_visible_and_hidden_edges() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let (o, d, x) = oblique();
        let result = project_edges(&topo, solid, o, d, x, true, 0.1).unwrap();
        assert!(
            !result.visible.is_empty(),
            "oblique view must have visible edges"
        );
        assert!(
            !result.hidden.is_empty(),
            "the far corner's edges must be hidden behind the box"
        );
    }

    #[test]
    fn project_box_without_hidden_lines_drops_hidden() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let (o, d, x) = oblique();
        let result = project_edges(&topo, solid, o, d, x, false, 0.1).unwrap();
        assert!(!result.visible.is_empty());
        assert!(
            result.hidden.is_empty(),
            "hidden lines disabled → no hidden polylines"
        );
    }
}
