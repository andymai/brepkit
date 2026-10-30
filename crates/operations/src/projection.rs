//! Edge projection with hidden-line removal (HLR).
//!
//! Projects solids' edges onto a view plane and splits each edge into visible
//! and hidden polylines. Occlusion is **exact** — no tessellation of faces: a
//! boundary point is hidden when a solid lies on its line of sight toward the
//! camera. Edges themselves are sampled to polylines because a projected
//! drawing is inherently polygonal.

use brepkit_check::classify::SolidClassifier;
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

    let mut bounds: Option<brepkit_math::aabb::Aabb3> = None;
    for &solid in solids {
        let b = crate::measure::solid_bounding_box(topo, solid)?;
        bounds = Some(bounds.map_or(b, |a| a.union(b)));
    }
    let bbox = bounds.ok_or_else(|| crate::OperationsError::InvalidInput {
        reason: "projection needs at least one solid".into(),
    })?;
    let diag = (bbox.max - bbox.min).length();

    // A point on an edge is hidden when its line of sight toward the camera
    // meets a solid. The line starts a hair toward the camera, clear of the
    // faces the edge bounds yet short of any plate in front. A line that runs
    // along a face or past its edge is caught by the lines a hair to either
    // side across the view: a point behind such a line coincides on the
    // drawing with a nearer edge. A classification error is propagated
    // rather than read as "visible".
    let classifiers = solids
        .iter()
        .map(|&solid| SolidClassifier::new(topo, solid))
        .collect::<Result<Vec<_>, _>>()?;
    // Ten times the classification tolerance: a solid is grown by this much
    // where a line of sight passes it.
    let hair = 1e-6;
    let sideways = [x * hair, x * -hair, y * hair, y * -hair];
    // Each solid's box seen from the camera: its extent across the view and
    // its nearest depth, so a solid that cannot cover a point is passed by.
    let seen_boxes = solids
        .iter()
        .map(|&solid| {
            let b = crate::measure::solid_bounding_box(topo, solid)?;
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for c in 0..8 {
                let corner = Point3::new(
                    if c & 1 == 0 { b.min.x() } else { b.max.x() },
                    if c & 2 == 0 { b.min.y() } else { b.max.y() },
                    if c & 4 == 0 { b.min.z() } else { b.max.z() },
                );
                let d = corner - origin;
                for (k, v) in [x.dot(d), y.dot(d), view.dot(d)].into_iter().enumerate() {
                    lo[k] = lo[k].min(v);
                    hi[k] = hi[k].max(v);
                }
            }
            Ok((lo, hi))
        })
        .collect::<Result<Vec<_>, crate::OperationsError>>()?;
    let is_hidden = |p: Point3, own: usize| -> Result<bool, crate::OperationsError> {
        let d = p - origin;
        let (px, py, depth) = (x.dot(d), y.dot(d), view.dot(d));
        for (k, classifier) in classifiers.iter().enumerate() {
            let (lo, hi) = seen_boxes[k];
            let reach = 2.0 * hair;
            if k != own
                && (px < lo[0] - reach
                    || px > hi[0] + reach
                    || py < lo[1] - reach
                    || py > hi[1] + reach
                    || lo[2] > depth)
            {
                continue;
            }
            let from = p - view * hair;
            if classifier.ray_crossings(from, -view)? > 0 {
                return Ok(true);
            }
            // Lines a hair aside start further out, past where they would
            // clip the faces the edge bounds at a grazing angle.
            for s in &sideways {
                if classifier.ray_crossings(from + *s - view * (100.0 * hair), -view)? > 0 {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    };

    // Each edge is read on its own curve at the drawing's samples, closer
    // where its view is long, and on either side of each point where its
    // view crosses another edge's, where it may pass behind a solid however
    // narrow. It is cut where it passes behind a solid or comes out, found
    // by halving the step that changes.
    let resolution = diag / 64.0;
    let mut drawn = Vec::new();
    for (own, &solid) in solids.iter().enumerate() {
        for (edge_id, samples) in crate::tessellate::drawn_edge_samples(
            topo,
            solid,
            deflection,
            brepkit_math::chord::DEFAULT_ANGULAR_TOL,
            true,
        )? {
            if samples.len() >= 2 {
                let seen: Vec<Point2> = samples.iter().map(|&q| project(q)).collect();
                drawn.push((own, edge_id, seen));
            }
        }
    }
    let views = SegmentGrid::new(&drawn, resolution);
    let mut result = ProjectedEdges::default();
    for (index, (own, edge_id, seen)) in drawn.iter().enumerate() {
        let own = *own;
        let edge = topo.edge(*edge_id)?;
        let a = topo.vertex(edge.start())?.point();
        let b = topo.vertex(edge.end())?.point();
        let curve = edge.curve();
        let (t0, t1) = curve.domain_with_endpoints(a, b);
        let at = |t: f64| curve.evaluate_with_endpoints(t, a, b);
        let seen_length: f64 = seen
            .windows(2)
            .map(|w| (w[1].x() - w[0].x()).hypot(w[1].y() - w[0].y()))
            .sum();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let steps = (seen.len() - 1).max((seen_length / resolution).ceil() as usize);
        #[allow(clippy::cast_precision_loss)]
        let mut params: Vec<f64> = (0..=steps)
            .map(|k| (t1 - t0).mul_add(k as f64 / steps as f64, t0))
            .collect();
        let grid_view: Vec<Point2> = params.iter().map(|&t| project(at(t))).collect();
        let aside = 1e-4 / steps.max(1) as f64 * (t1 - t0);
        let mut extra = Vec::new();
        for (k, w) in grid_view.windows(2).enumerate() {
            for f in views.crossings(w[0], w[1], index) {
                let t = (params[k + 1] - params[k]).mul_add(f, params[k]);
                extra.extend([t - aside, t + aside]);
            }
        }
        let (lo_t, hi_t) = (t0.min(t1), t0.max(t1));
        params.extend(extra.into_iter().filter(|t| *t > lo_t && *t < hi_t));
        params.sort_by(|p, q| {
            if t1 >= t0 {
                p.total_cmp(q)
            } else {
                q.total_cmp(p)
            }
        });
        params.dedup_by(|p, q| (*p - *q).abs() <= 1e-12 * (t1 - t0).abs());
        let last = params.len() - 1;
        // The ends are read a hair inside, off the vertex they share.
        let inset = 1e-6 * (t1 - t0);
        let probe_at = |k: usize| match k {
            0 => t0 + inset,
            k if k == last => t1 - inset,
            k => params[k],
        };
        let hidden = (0..=last)
            .map(|k| is_hidden(at(probe_at(k)), own))
            .collect::<Result<Vec<bool>, _>>()?;
        let mut runs: Vec<(bool, Vec<Point3>)> = vec![(hidden[0], vec![at(t0)])];
        for k in 0..last {
            if hidden[k] != hidden[k + 1] {
                let (mut lo, mut hi) = (probe_at(k), probe_at(k + 1));
                for _ in 0..24 {
                    let mid = f64::midpoint(lo, hi);
                    if is_hidden(at(mid), own)? == hidden[k] {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let cut = at(f64::midpoint(lo, hi));
                if let Some(run) = runs.last_mut() {
                    run.1.push(cut);
                }
                runs.push((hidden[k + 1], vec![cut]));
            }
            if let Some(run) = runs.last_mut() {
                run.1.push(at(params[k + 1]));
            }
        }
        for (is_hidden_run, points) in runs {
            if is_hidden_run && !hidden_lines {
                continue;
            }
            let poly: Vec<Point2> = points.into_iter().map(project).collect();
            if is_hidden_run {
                result.hidden.push(poly);
            } else {
                result.visible.push(poly);
            }
        }
    }

    Ok(result)
}

/// The drawn edges' views on the drawing, filed by cell, to find where one
/// edge's view crosses another's.
struct SegmentGrid {
    cell: f64,
    cells: std::collections::HashMap<(i64, i64), Vec<Segment>>,
}

/// A piece of one drawn edge's view: the edge's index and the piece's ends.
type Segment = (usize, Point2, Point2);

impl SegmentGrid {
    fn new(drawn: &[(usize, brepkit_topology::edge::EdgeId, Vec<Point2>)], cell: f64) -> Self {
        let mut grid = Self {
            cell: cell.max(1e-12),
            cells: std::collections::HashMap::new(),
        };
        for (index, (_, _, seen)) in drawn.iter().enumerate() {
            for w in seen.windows(2) {
                for key in grid.keys(w[0], w[1]) {
                    grid.cells.entry(key).or_default().push((index, w[0], w[1]));
                }
            }
        }
        grid
    }

    #[allow(clippy::cast_possible_truncation)]
    fn keys(&self, a: Point2, b: Point2) -> Vec<(i64, i64)> {
        let at = |v: f64| (v / self.cell).floor() as i64;
        let (x0, x1) = (at(a.x().min(b.x())), at(a.x().max(b.x())));
        let (y0, y1) = (at(a.y().min(b.y())), at(a.y().max(b.y())));
        (x0..=x1)
            .flat_map(|i| (y0..=y1).map(move |j| (i, j)))
            .collect()
    }

    /// The fractions along `a`..`b` where it crosses a view of an edge
    /// other than `skip`.
    fn crossings(&self, a: Point2, b: Point2, skip: usize) -> Vec<f64> {
        let r = (b.x() - a.x(), b.y() - a.y());
        let mut out = Vec::new();
        for key in self.keys(a, b) {
            for &(index, c, d) in self.cells.get(&key).into_iter().flatten() {
                if index == skip {
                    continue;
                }
                let w = (d.x() - c.x(), d.y() - c.y());
                let denom = r.0 * w.1 - r.1 * w.0;
                if denom.abs() <= 1e-300 {
                    continue;
                }
                let (qx, qy) = (c.x() - a.x(), c.y() - a.y());
                let f = (qx * w.1 - qy * w.0) / denom;
                let g = (qx * r.1 - qy * r.0) / denom;
                if (0.0..=1.0).contains(&f) && (0.0..=1.0).contains(&g) {
                    out.push(f);
                }
            }
        }
        out
    }
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
        assert!((length(&result.visible) - 140.0).abs() < 1e-5);
        assert!((length(&result.hidden) - 140.0).abs() < 1e-5);
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
            assert!((length(&result.visible) - 140.0).abs() < 1e-5);
            assert!((length(&result.hidden) - 220.0).abs() < 1e-5);
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
        assert!((length(&result.visible) - 170.0).abs() < 1e-5);
        assert!((length(&result.hidden) - 290.0).abs() < 1e-5);
    }

    /// A second box the block's size 20 behind it, seen with it: every line
    /// of sight from its front rectangle runs along the block's faces or past
    /// its edges, so that rectangle lies behind the block's and is hidden,
    /// with both back rectangles: 140 shows and 420 is hidden.
    #[test]
    fn a_box_aligned_behind_the_block_is_hidden() {
        let mut topo = Topology::new();
        let front_block = block(&mut topo);
        let rear = placed_box(&mut topo, (40.0, 10.0, 30.0), Point3::new(-20.0, 25.0, 0.0));
        let (o, d, x) = front();
        let result = project_solids_edges(&topo, &[front_block, rear], o, d, x, true, 0.1).unwrap();
        assert!((length(&result.visible) - 140.0).abs() < 1e-5);
        assert!((length(&result.hidden) - 420.0).abs() < 1e-5);
    }

    /// The tower against the block's back, as a separate solid: its front
    /// verticals are hidden below the block's top and show above it, so the
    /// block's front, the tower's top front edge and the verticals' upper 10
    /// make the 170 that shows; the other 310 is hidden.
    #[test]
    fn a_separate_tower_behind_the_block_shows_above_it() {
        let mut topo = Topology::new();
        let front_block = block(&mut topo);
        let tower = placed_box(&mut topo, (10.0, 10.0, 40.0), Point3::new(-5.0, 5.0, 0.0));
        let (o, d, x) = front();
        let result =
            project_solids_edges(&topo, &[front_block, tower], o, d, x, true, 0.1).unwrap();
        assert!((length(&result.visible) - 170.0).abs() < 1e-5);
        assert!((length(&result.hidden) - 310.0).abs() < 1e-5);
    }

    /// A 120 square plate 0.002 thick, from y = -0.006 to -0.004, in front
    /// of a 100 cube at y = 0: nearer the cube than the step its own edges
    /// are read from, it still hides the whole cube. Only the plate's front
    /// square, 480, shows.
    #[test]
    fn a_thin_plate_just_in_front_hides_the_cube() {
        let mut topo = Topology::new();
        let cube = placed_box(&mut topo, (100.0, 100.0, 100.0), Point3::new(0.0, 0.0, 0.0));
        let plate = placed_box(
            &mut topo,
            (120.0, 0.002, 120.0),
            Point3::new(-10.0, -0.006, -10.0),
        );
        let (o, d, x) = (
            Point3::new(50.0, -100.0, 50.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let result = project_solids_edges(&topo, &[cube, plate], o, d, x, true, 0.1).unwrap();
        assert!((length(&result.visible) - 480.0).abs() < 1e-5);
    }

    /// A cylinder of radius 10 and height 10 seen side on along +y: each
    /// rim's front half shows as a 20 long line and its back half is hidden
    /// behind it, read on the rim itself rather than on its chords.
    #[test]
    fn a_cylinder_seen_side_on_shows_its_rims_front_halves() {
        let mut topo = Topology::new();
        let cylinder = crate::primitives::make_cylinder(&mut topo, 10.0, 10.0).unwrap();
        let (o, d, x) = (
            Point3::new(0.0, -100.0, 5.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let result = project_edges(&topo, cylinder, o, d, x, true, 0.1).unwrap();
        assert!(
            (length(&result.visible) - 40.0).abs() < 1e-5,
            "{}",
            length(&result.visible)
        );
        assert!((length(&result.hidden) - 40.0).abs() < 1e-5);
    }

    /// The 10 cube seen along (1, 1, 1) shows the 9 edges of its three near
    /// faces, its outline among them, and hides the 3 at its far corner,
    /// each 10 √(2/3) long on the drawing. Lines a hair aside of an outline
    /// edge clip the face it bounds at a grazing angle, which hid it.
    #[test]
    fn a_cube_seen_corner_on_shows_its_outline() {
        let mut topo = Topology::new();
        let cube = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let (o, d, x) = oblique();
        let edge = 10.0 * (2.0_f64 / 3.0).sqrt();
        let shown = project_edges(&topo, cube, o, d, x, false, 0.1).unwrap();
        assert!(
            (length(&shown.visible) - 9.0 * edge).abs() < 1e-5,
            "{}",
            length(&shown.visible)
        );
        let all = project_edges(&topo, cube, o, d, x, true, 0.1).unwrap();
        assert!(
            (length(&all.hidden) - 3.0 * edge).abs() < 1e-5,
            "{}",
            length(&all.hidden)
        );
    }

    /// A bar 0.2 wide in front of a box 100 long hides 0.2 of each of the
    /// box's front horizontals, far narrower than the steps the edges are
    /// read at: their views cross the bar's sides, and are read either side.
    #[test]
    fn a_narrow_bar_hides_its_width_of_a_long_box() {
        let mut topo = Topology::new();
        let long = placed_box(&mut topo, (100.0, 1.0, 1.0), Point3::new(0.0, 10.0, 0.0));
        let bar = placed_box(&mut topo, (0.2, 1.0, 3.0), Point3::new(0.4, 0.0, -1.0));
        let (o, d, x) = (
            Point3::new(50.0, -100.0, 0.5),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        );
        let result = project_solids_edges(&topo, &[long, bar], o, d, x, true, 0.001).unwrap();
        assert!(
            (length(&result.visible) - 208.0).abs() < 1e-5,
            "{}",
            length(&result.visible)
        );
        assert!(
            (length(&result.hidden) - 208.8).abs() < 1e-5,
            "{}",
            length(&result.hidden)
        );
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
