//! Edge projection with hidden-line removal (HLR).
//!
//! Projects solids' edges onto a view plane and splits each edge into visible
//! and hidden polylines. Occlusion is **exact** — no tessellation of faces: a
//! boundary point is hidden when a solid lies on its line of sight toward the
//! camera. Edges themselves are sampled to polylines because a projected
//! drawing is inherently polygonal.

use brepkit_check::classify::{ClassifyOptions, PointClassification, SolidClassifier};
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

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
    project_solids_edges(
        topo,
        &[solid],
        origin,
        direction,
        x_axis,
        hidden_lines,
        deflection,
    )
}

/// [`project_edges`] for several solids seen together: every solid's edges,
/// each hidden where any of the solids lies on its line of sight.
///
/// # Errors
///
/// As [`project_edges`], and [`crate::OperationsError::InvalidInput`] when
/// `solids` is empty.
pub fn project_solids_edges(
    topo: &Topology,
    solids: &[SolidId],
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
    let mut bounds: Option<brepkit_math::aabb::Aabb3> = None;
    for &solid in solids {
        let b = crate::measure::solid_bounding_box(topo, solid)?;
        bounds = Some(bounds.map_or(b, |a| a.union(b)));
    }
    let bbox = bounds.ok_or_else(|| crate::OperationsError::InvalidInput {
        reason: "projection needs at least one solid".into(),
    })?;
    let diag = (bbox.max - bbox.min).length();
    let eps = (diag * 1e-4).clamp(1e-6, 1e-2);

    // A boundary point is hidden when a solid lies on its line of sight: the
    // step toward the camera lands inside one, or on a face seen edge-on along
    // the line of sight, or the line beyond the step crosses a face. A
    // classification error is propagated rather than silently read as
    // "visible", so a degenerate solid surfaces instead of yielding wrong HLR.
    let classifiers = solids
        .iter()
        .map(|&solid| SolidClassifier::new(topo, solid))
        .collect::<Result<Vec<_>, _>>()?;
    let options = ClassifyOptions {
        tolerance: 1e-7,
        ..ClassifyOptions::default()
    };
    let is_hidden = |p: Point3| -> Result<bool, crate::OperationsError> {
        let probe = p - view * eps;
        for classifier in &classifiers {
            if classifier.classify(probe, &options)? != PointClassification::Outside
                || classifier.ray_crossings(probe, -view)? > 0
            {
                return Ok(true);
            }
        }
        Ok(false)
    };

    let mut result = ProjectedEdges::default();
    for &solid in solids {
        let lines = crate::tessellate::sample_solid_edges(topo, solid, deflection)?;
        let n_edges = lines.offsets.len();
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

    /// A view from `(0, -100, 15)` along +y, with x across.
    fn front() -> (Point3, Vec3, Vec3) {
        (
            Point3::new(0.0, -100.0, 15.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
    }

    /// A `w` x `d` x `h` box with its least corner at `at`.
    fn placed_box(topo: &mut Topology, (w, d, h): (f64, f64, f64), at: Point3) -> SolidId {
        let solid = crate::primitives::make_box(topo, w, d, h).unwrap();
        crate::transform::transform_solid(
            topo,
            solid,
            &brepkit_math::mat::Mat4::translation(at.x(), at.y(), at.z()),
        )
        .unwrap();
        solid
    }

    /// The block from x -20 to 20, y -5 to 5 and z 0 to 30.
    fn block(topo: &mut Topology) -> SolidId {
        placed_box(topo, (40.0, 10.0, 30.0), Point3::new(-20.0, -5.0, 0.0))
    }

    fn length(polylines: &[Vec<Point2>]) -> f64 {
        polylines
            .iter()
            .flat_map(|p| p.windows(2))
            .map(|w| (w[1].x() - w[0].x()).hypot(w[1].y() - w[0].y()))
            .sum()
    }

    /// Square on, a box's back rectangle lies behind its front one: 140 of
    /// edges are visible and 140 hidden.
    #[test]
    fn a_box_seen_square_on_hides_its_back_rectangle() {
        let mut topo = Topology::new();
        let solid = block(&mut topo);
        let (o, d, x) = front();
        let result = project_edges(&topo, solid, o, d, x, true, 0.1).unwrap();
        assert!((length(&result.visible) - 140.0).abs() < 1e-6);
        assert!((length(&result.hidden) - 140.0).abs() < 1e-6);
    }

    /// A 10 cube 20 behind the block is hidden by it, whether the two are
    /// lumps of one solid or two solids seen together: the block's front
    /// rectangle is all that shows, and the cube's 80 joins the block's back
    /// 140 among the hidden edges.
    #[test]
    fn a_cube_behind_the_block_is_hidden() {
        let mut topo = Topology::new();
        let (front_block, cube) = (
            block(&mut topo),
            placed_box(&mut topo, (10.0, 10.0, 10.0), Point3::new(-5.0, 25.0, 10.0)),
        );
        let (o, d, x) = front();
        let together =
            project_solids_edges(&topo, &[front_block, cube], o, d, x, true, 0.1).unwrap();
        let lumps = crate::boolean::boolean(
            &mut topo,
            crate::boolean::BooleanOp::Fuse,
            front_block,
            cube,
        )
        .unwrap();
        let fused = project_edges(&topo, lumps, o, d, x, true, 0.1).unwrap();
        for result in [together, fused] {
            assert!((length(&result.visible) - 140.0).abs() < 1e-6);
            assert!((length(&result.hidden) - 220.0).abs() < 1e-6);
        }
    }

    /// A 40 tall tower fused against the block's back shows above the
    /// block only: nothing visible on its sides' lines x = ±5 lies below
    /// z = 30, where its back verticals and the junction lie behind the
    /// block. The block's front rectangle and the tower's front above the
    /// block, 170 in all, show; the other 290 is hidden.
    #[test]
    fn a_tower_behind_the_block_shows_only_above_it() {
        let mut topo = Topology::new();
        let front_block = block(&mut topo);
        let tower = placed_box(&mut topo, (10.0, 10.0, 40.0), Point3::new(-5.0, 5.0, 0.0));
        let fused = crate::boolean::boolean(
            &mut topo,
            crate::boolean::BooleanOp::Fuse,
            front_block,
            tower,
        )
        .unwrap();
        let (o, d, x) = front();
        let result = project_edges(&topo, fused, o, d, x, true, 0.1).unwrap();
        for poly in &result.visible {
            for w in poly.windows(2) {
                let on_side =
                    (w[0].x().abs() - 5.0).abs() < 1e-9 && (w[1].x() - w[0].x()).abs() < 1e-9;
                assert!(
                    !on_side || w[0].y().min(w[1].y()) >= 15.0 - 1e-9,
                    "visible {w:?}"
                );
            }
        }
        assert!((length(&result.visible) - 170.0).abs() < 1e-6);
        assert!((length(&result.hidden) - 290.0).abs() < 1e-6);
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
