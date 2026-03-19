# GFA Bug Fixes: Same-Domain + Wire Builder Crossing

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two GFA bugs that cause silent fallback to the old boolean pipeline — same-domain over-classification on touching geometry, and wire builder failure on crossing section edges.

**Architecture:** Bug 1 (same-domain) adds a polygon interior overlap check in `same_domain.rs` so edge-touching faces aren't classified as coplanar. Bug 2 (wire builder) adds section edge crossing detection in `face_splitter.rs` that splits crossing edges at their intersection point before passing to the wire builder. Both are isolated changes in `crates/algo/src/builder/`.

**Tech Stack:** Rust, brepkit-algo crate, 2D computational geometry (point-in-polygon, line-line intersection).

---

### Task 1: Fix same-domain over-classification on edge-touching faces

**Root cause:** `detect_same_domain` in `same_domain.rs` checks `face_bboxes_overlap` (AABB with tolerance expansion) + `surfaces_same_domain` (surface equation match). For touching boxes A=[0,1]³ and B=[1,2]×[0,1]², the y=0/y=1/z=0/z=1 face pairs share the same plane equation. Their AABBs overlap within tolerance (they share an edge at x=1). Both get marked `CoplanarSame`, even though their face interiors don't overlap — they only share an edge.

For Cut, `CoplanarSame` from A is not selected → A loses 4 of 6 faces → result has 2 faces instead of 6.

**Fix:** After confirming same surface, project both face boundaries to 2D in the shared plane and verify interior overlap (not just edge contact). If no vertex of either polygon is inside the other, skip same-domain.

**Files:**
- Modify: `crates/algo/src/builder/same_domain.rs`
- Test: `crates/algo/src/pave_filler/tests.rs` (un-ignore `gfa_cut_touching_boxes`)

- [ ] **Step 1: Write failing test — `same_domain_skips_edge_touching_faces`**

Add a unit test in `same_domain.rs` that constructs two sub-faces on the same plane
whose boundary polygons share an edge but don't overlap in area, and asserts they
are NOT classified as same-domain.

```rust
#[test]
fn same_domain_skips_edge_touching_faces() {
    // Two unit squares sharing edge at x=1:
    //   A: [0,1]×[0,1] on y=0 plane
    //   B: [1,2]×[0,1] on y=0 plane
    // They share the same plane but only touch at x=1 — no area overlap.
    use brepkit_topology::Topology;
    use brepkit_math::tolerance::Tolerance;

    let mut topo = Topology::default();
    let tol = Tolerance::new();

    // Build two boxes that touch at x=1
    let a = crate::pave_filler::tests::make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = crate::pave_filler::tests::make_box(&mut topo, [1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);

    let faces_a = brepkit_topology::explorer::solid_faces(&topo, a).unwrap();
    let faces_b = brepkit_topology::explorer::solid_faces(&topo, b).unwrap();

    // Find y=0 faces from each solid
    let find_y0 = |faces: &[brepkit_topology::face::FaceId]| {
        faces.iter().find(|&&fid| {
            let f = topo.face(fid).unwrap();
            matches!(f.surface(), brepkit_topology::face::FaceSurface::Plane { normal, d }
                if normal.y().abs() > 0.9 && d.abs() < 0.01)
        }).copied().unwrap()
    };

    let y0_a = find_y0(&faces_a);
    let y0_b = find_y0(&faces_b);

    // Verify they DON'T have interior overlap
    assert!(
        !faces_have_interior_overlap(&topo, y0_a, y0_b, tol),
        "edge-touching faces should NOT have interior overlap"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p brepkit-algo same_domain_skips`
Expected: FAIL — `faces_have_interior_overlap` doesn't exist yet.

- [ ] **Step 3: Implement `faces_have_interior_overlap`**

Add to `same_domain.rs`:

```rust
/// Check whether two coplanar faces overlap in interior area.
///
/// Projects both face boundaries to 2D in the shared plane, then checks
/// if any vertex of face A is inside face B's boundary polygon (or vice
/// versa). Faces that only share an edge (zero area overlap) return false.
fn faces_have_interior_overlap(topo: &Topology, a: FaceId, b: FaceId, tol: Tolerance) -> bool {
    let poly_a = face_boundary_2d(topo, a, tol);
    let poly_b = face_boundary_2d(topo, b, tol);

    if poly_a.len() < 3 || poly_b.len() < 3 {
        return false;
    }

    // Check if any vertex of A is strictly inside B's polygon
    for pt in &poly_a {
        if point_strictly_inside_polygon(pt, &poly_b, tol.linear) {
            return true;
        }
    }
    // Check if any vertex of B is strictly inside A's polygon
    for pt in &poly_b {
        if point_strictly_inside_polygon(pt, &poly_a, tol.linear) {
            return true;
        }
    }

    // Edge case: faces could overlap without containing each other's vertices
    // (e.g., star-shaped overlap). Check if any edge of A crosses any edge of B.
    for i in 0..poly_a.len() {
        let a0 = &poly_a[i];
        let a1 = &poly_a[(i + 1) % poly_a.len()];
        for j in 0..poly_b.len() {
            let b0 = &poly_b[j];
            let b1 = &poly_b[(j + 1) % poly_b.len()];
            if segments_cross_interior(a0, a1, b0, b1, tol.linear) {
                return true;
            }
        }
    }

    false
}

/// Project a face's outer wire boundary to 2D in the face's plane.
fn face_boundary_2d(
    topo: &Topology,
    face_id: FaceId,
    _tol: Tolerance,
) -> Vec<brepkit_math::vec::Point2> {
    use brepkit_math::vec::{Point2, Vec3};

    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };

    // Get plane normal and a reference point
    let (normal, d) = match face.surface() {
        FaceSurface::Plane { normal, d } => (*normal, *d),
        _ => return Vec::new(), // Non-plane faces use surface.project_point
    };

    // Build a local 2D frame from the plane normal
    let u_axis = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0).cross(normal)
    } else {
        Vec3::new(0.0, 1.0, 0.0).cross(normal)
    };
    let u_len = u_axis.length();
    let u_axis = if u_len > 1e-12 {
        u_axis * (1.0 / u_len)
    } else {
        return Vec::new();
    };
    let v_axis = normal.cross(u_axis);

    let wire = match topo.wire(face.outer_wire()) {
        Ok(w) => w,
        Err(_) => return Vec::new(),
    };

    let mut pts = Vec::new();
    for oe in wire.edges() {
        let edge = match topo.edge(oe.edge()) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let p = match topo.vertex(oe.oriented_start(edge)) {
            Ok(v) => v.point(),
            Err(_) => continue,
        };
        pts.push(Point2::new(p.dot_vec3(u_axis), p.dot_vec3(v_axis)));
    }
    pts
}

/// Check if a point is strictly inside a polygon (not on boundary).
fn point_strictly_inside_polygon(
    pt: &brepkit_math::vec::Point2,
    poly: &[brepkit_math::vec::Point2],
    tol: f64,
) -> bool {
    // First check if point is ON any edge (within tolerance) — that's boundary, not interior.
    let n = poly.len();
    for i in 0..n {
        let a = &poly[i];
        let b = &poly[(i + 1) % n];
        let dist = point_to_segment_dist_2d(pt, a, b);
        if dist < tol {
            return false; // On boundary
        }
    }

    // Ray-casting for interior test
    let mut inside = false;
    let px = pt.x();
    let py = pt.y();
    let mut j = n - 1;
    for i in 0..n {
        let yi = poly[i].y();
        let yj = poly[j].y();
        let xi = poly[i].x();
        let xj = poly[j].x();
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Distance from a 2D point to a line segment.
fn point_to_segment_dist_2d(
    pt: &brepkit_math::vec::Point2,
    a: &brepkit_math::vec::Point2,
    b: &brepkit_math::vec::Point2,
) -> f64 {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-30 {
        return ((pt.x() - a.x()).powi(2) + (pt.y() - a.y()).powi(2)).sqrt();
    }
    let t = ((pt.x() - a.x()) * dx + (pt.y() - a.y()) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj_x = a.x() + t * dx;
    let proj_y = a.y() + t * dy;
    ((pt.x() - proj_x).powi(2) + (pt.y() - proj_y).powi(2)).sqrt()
}

/// Check if two 2D line segments cross in their interiors (not at endpoints).
fn segments_cross_interior(
    a0: &brepkit_math::vec::Point2,
    a1: &brepkit_math::vec::Point2,
    b0: &brepkit_math::vec::Point2,
    b1: &brepkit_math::vec::Point2,
    tol: f64,
) -> bool {
    let d1x = a1.x() - a0.x();
    let d1y = a1.y() - a0.y();
    let d2x = b1.x() - b0.x();
    let d2y = b1.y() - b0.y();
    let det = d1x * d2y - d1y * d2x;
    if det.abs() < tol * tol {
        return false; // Parallel
    }
    let dx = b0.x() - a0.x();
    let dy = b0.y() - a0.y();
    let t = (dx * d2y - dy * d2x) / det;
    let s = (dx * d1y - dy * d1x) / det;
    // Strictly interior: t and s in (eps, 1-eps)
    let eps = 0.01;
    t > eps && t < 1.0 - eps && s > eps && s < 1.0 - eps
}
```

- [ ] **Step 4: Wire up overlap check in `detect_same_domain`**

In `detect_same_domain`, after `surfaces_same_domain` returns `Some(...)`, add:

```rust
if let Some(same_dir) = surfaces_same_domain(surf_i, surf_j, tol) {
    // Verify actual area overlap — faces that only share an edge
    // (e.g., touching boxes) should NOT be classified as same-domain.
    if !faces_have_interior_overlap(topo, sub_faces[i].face_id, sub_faces[j].face_id, tol) {
        continue;
    }
    // ... existing classification logic ...
}
```

- [ ] **Step 5: Handle `dot_vec3` — add helper if needed**

`Point3` may not have `dot_vec3`. Check if it exists; if not, use inline:
```rust
let proj = |p: Point3, axis: Vec3| -> f64 {
    p.x() * axis.x() + p.y() * axis.y() + p.z() * axis.z()
};
```

- [ ] **Step 6: Run unit test**

Run: `cargo test -p brepkit-algo same_domain_skips`
Expected: PASS

- [ ] **Step 7: Un-ignore `gfa_cut_touching_boxes` integration test**

In `crates/algo/src/pave_filler/tests.rs`, remove `#[ignore = ...]` from
`gfa_cut_touching_boxes`.

- [ ] **Step 8: Run integration test**

Run: `cargo test -p brepkit-algo gfa_cut_touching_boxes`
Expected: PASS with 6 faces

- [ ] **Step 9: Run full algo test suite**

Run: `cargo test -p brepkit-algo`
Expected: All pass except `gfa_intersect_overlapping_boxes` (Task 2 fixes that)

- [ ] **Step 10: Commit**

```bash
git add crates/algo/src/builder/same_domain.rs crates/algo/src/pave_filler/tests.rs
git commit -m "fix(algo): same-domain detection requires interior overlap, not just edge contact

Touching faces (shared edge, zero area overlap) were incorrectly classified
as CoplanarSame/CoplanarOpposite, causing Cut to drop faces from the result.
Added polygon interior overlap check: projects boundaries to 2D, tests if any
vertex is strictly inside the other polygon or if edges cross interiors."
```

---

### Task 2: Fix wire builder failure on crossing section edges

**Root cause:** When two overlapping boxes are intersected (A=[0,2]³, B=[1,3]³), each face can have TWO section edges from different face-face intersection curves. For example, face z=2 of box A gets section edges at x=1 and y=1. These two lines cross at point (1, 1, 2) in 3D / at the corresponding UV point in 2D.

The PaveFiller creates section edges as boundary-to-boundary lines, but never splits them at their mutual crossing point. The wire builder receives two independent full-length edges that cross geometrically but share no vertex. Its angular traversal can't split the face into 4 quadrants — it produces 1 sub-face instead of 4.

**Fix:** In `face_splitter.rs`, before calling `build_wire_loops`, detect pairwise crossings of section edges and split each crossing pair into 4 half-edges meeting at a new vertex. This converts the geometric crossing into a topological junction.

**Files:**
- Modify: `crates/algo/src/builder/face_splitter.rs`
- Test: `crates/algo/src/pave_filler/tests.rs` (un-ignore `gfa_intersect_overlapping_boxes`)
- Test: `crates/algo/src/builder/wire_builder.rs` (new unit test)

- [ ] **Step 1: Write failing test — wire builder with crossing edges**

Add test in `wire_builder.rs`:

```rust
#[test]
fn square_with_two_crossing_chords_produces_four_loops() {
    // A 10x10 square with two crossing interior lines:
    //   Horizontal: (0,5)→(10,5) + reverse
    //   Vertical:   (5,0)→(5,10) + reverse
    // These cross at (5,5), creating 4 quadrants.
    // With crossing detection, the chords are split at (5,5) into 4 half-edges.
    let edges = vec![
        // Boundary: bottom, right, top, left (split at section endpoints)
        make_line_edge(Point2::new(0.0, 0.0), Point2::new(5.0, 0.0)),
        make_line_edge(Point2::new(5.0, 0.0), Point2::new(10.0, 0.0)),
        make_line_edge(Point2::new(10.0, 0.0), Point2::new(10.0, 5.0)),
        make_line_edge(Point2::new(10.0, 5.0), Point2::new(10.0, 10.0)),
        make_line_edge(Point2::new(10.0, 10.0), Point2::new(5.0, 10.0)),
        make_line_edge(Point2::new(5.0, 10.0), Point2::new(0.0, 10.0)),
        make_line_edge(Point2::new(0.0, 10.0), Point2::new(0.0, 5.0)),
        make_line_edge(Point2::new(0.0, 5.0), Point2::new(0.0, 0.0)),
        // Horizontal chord + reverse (split at crossing)
        make_line_edge(Point2::new(0.0, 5.0), Point2::new(5.0, 5.0)),
        make_line_edge(Point2::new(5.0, 5.0), Point2::new(10.0, 5.0)),
        make_line_edge(Point2::new(10.0, 5.0), Point2::new(5.0, 5.0)),
        make_line_edge(Point2::new(5.0, 5.0), Point2::new(0.0, 5.0)),
        // Vertical chord + reverse (split at crossing)
        make_line_edge(Point2::new(5.0, 0.0), Point2::new(5.0, 5.0)),
        make_line_edge(Point2::new(5.0, 5.0), Point2::new(5.0, 10.0)),
        make_line_edge(Point2::new(5.0, 10.0), Point2::new(5.0, 5.0)),
        make_line_edge(Point2::new(5.0, 5.0), Point2::new(5.0, 0.0)),
    ];

    let loops = build_wire_loops(&edges, 1e-7, false, false);
    assert_eq!(loops.len(), 4, "expected 4 loops (quadrants), got {}", loops.len());
}
```

- [ ] **Step 2: Run test to verify it passes (wire builder itself is fine with pre-split edges)**

Run: `cargo test -p brepkit-algo square_with_two_crossing_chords`
Expected: PASS — the wire builder handles pre-split edges correctly. The bug is that `face_splitter.rs` doesn't split them.

- [ ] **Step 3: Write failing test — `split_crossing_section_edges` function**

Add test in `face_splitter.rs`:

```rust
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn crossing_sections_are_split() {
        // Two section edges that cross at (5, 5):
        //   Horizontal: (0,5)→(10,5)
        //   Vertical:   (5,0)→(5,10)
        let sections = vec![
            SectionEdge {
                curve_3d: EdgeCurve::Line,
                pcurve_a: dummy_pcurve(),
                pcurve_b: dummy_pcurve(),
                start: Point3::new(0.0, 5.0, 0.0),
                end: Point3::new(10.0, 5.0, 0.0),
                start_uv_a: Some(Point2::new(0.0, 5.0)),
                end_uv_a: Some(Point2::new(10.0, 5.0)),
                start_uv_b: None,
                end_uv_b: None,
                target_face: None,
            },
            SectionEdge {
                curve_3d: EdgeCurve::Line,
                pcurve_a: dummy_pcurve(),
                pcurve_b: dummy_pcurve(),
                start: Point3::new(5.0, 0.0, 0.0),
                end: Point3::new(5.0, 10.0, 0.0),
                start_uv_a: Some(Point2::new(5.0, 0.0)),
                end_uv_a: Some(Point2::new(5.0, 10.0)),
                start_uv_b: None,
                end_uv_b: None,
                target_face: None,
            },
        ];

        let result = split_crossing_section_edges(&sections, 1e-7);
        assert_eq!(result.len(), 4, "2 crossing sections should produce 4 half-edges, got {}", result.len());
    }

    fn dummy_pcurve() -> brepkit_math::curves2d::Curve2D {
        use brepkit_math::curves2d::{Curve2D, Line2D};
        use brepkit_math::vec::{Point2, Vec2};
        Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap())
    }
}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cargo test -p brepkit-algo crossing_sections_are_split`
Expected: FAIL — `split_crossing_section_edges` doesn't exist.

- [ ] **Step 5: Implement `split_crossing_section_edges`**

Add to `face_splitter.rs`:

```rust
/// Split section edges at mutual crossing points.
///
/// When two section edges cross each other on a face (e.g., two intersection
/// curves from different face pairs), the wire builder needs a shared vertex
/// at the crossing point to form correct sub-face loops. This function detects
/// pairwise crossings (using pre-computed UV endpoints) and splits each
/// crossing pair into two half-edges.
///
/// Non-crossing edges pass through unchanged.
pub(crate) fn split_crossing_section_edges(
    sections: &[SectionEdge],
    tol: f64,
) -> Vec<SectionEdge> {
    if sections.len() < 2 {
        return sections.to_vec();
    }

    // Collect all crossing points: (section_idx, t_parameter, crossing_3d_point, crossing_uv)
    struct Crossing {
        t: f64,
        point_3d: Point3,
        point_uv: Point2,
    }

    let mut crossings: Vec<Vec<Crossing>> = vec![Vec::new(); sections.len()];

    for i in 0..sections.len() {
        let si = &sections[i];
        let (si_uv_start, si_uv_end) = match (si.start_uv_a, si.end_uv_a) {
            (Some(s), Some(e)) => (s, e),
            _ => continue,
        };

        for j in (i + 1)..sections.len() {
            let sj = &sections[j];
            let (sj_uv_start, sj_uv_end) = match (sj.start_uv_a, sj.end_uv_a) {
                (Some(s), Some(e)) => (s, e),
                _ => continue,
            };

            // 2D line-line intersection in UV space
            let d1 = Point2::new(
                si_uv_end.x() - si_uv_start.x(),
                si_uv_end.y() - si_uv_start.y(),
            );
            let d2 = Point2::new(
                sj_uv_end.x() - sj_uv_start.x(),
                sj_uv_end.y() - sj_uv_start.y(),
            );
            let det = d1.x() * d2.y() - d1.y() * d2.x();
            if det.abs() < tol * tol {
                continue; // Parallel
            }
            let dx = sj_uv_start.x() - si_uv_start.x();
            let dy = sj_uv_start.y() - si_uv_start.y();
            let ti = (dx * d2.y() - dy * d2.x()) / det;
            let tj = (dx * d1.y() - dy * d1.x()) / det;

            // Both parameters must be strictly interior (not at endpoints)
            let eps = 0.01;
            if ti > eps && ti < 1.0 - eps && tj > eps && tj < 1.0 - eps {
                let cross_uv = Point2::new(
                    si_uv_start.x() + ti * d1.x(),
                    si_uv_start.y() + ti * d1.y(),
                );
                let cross_3d = Point3::new(
                    si.start.x() + ti * (si.end.x() - si.start.x()),
                    si.start.y() + ti * (si.end.y() - si.start.y()),
                    si.start.z() + ti * (si.end.z() - si.start.z()),
                );

                crossings[i].push(Crossing { t: ti, point_3d: cross_3d, point_uv: cross_uv });
                crossings[j].push(Crossing { t: tj, point_3d: cross_3d, point_uv: cross_uv });
            }
        }
    }

    // Split each section edge at its crossing points (sorted by t)
    let mut result = Vec::new();

    for (idx, section) in sections.iter().enumerate() {
        let mut splits = std::mem::take(&mut crossings[idx]);
        if splits.is_empty() {
            result.push(section.clone());
            continue;
        }

        splits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

        // Build sub-edges: start→split1→split2→...→end
        let mut prev_3d = section.start;
        let mut prev_uv = section.start_uv_a.unwrap_or(Point2::new(0.0, 0.0));
        let mut prev_t = 0.0_f64;

        for split in &splits {
            let sub = SectionEdge {
                curve_3d: section.curve_3d.clone(),
                pcurve_a: section.pcurve_a.clone(),
                pcurve_b: section.pcurve_b.clone(),
                start: prev_3d,
                end: split.point_3d,
                start_uv_a: Some(prev_uv),
                end_uv_a: Some(split.point_uv),
                start_uv_b: section.start_uv_b, // simplified — B side not split
                end_uv_b: section.end_uv_b,
                target_face: section.target_face,
            };
            result.push(sub);
            prev_3d = split.point_3d;
            prev_uv = split.point_uv;
            prev_t = split.t;
        }

        // Final segment: last split → end
        let sub = SectionEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve_a: section.pcurve_a.clone(),
            pcurve_b: section.pcurve_b.clone(),
            start: prev_3d,
            end: section.end,
            start_uv_a: Some(prev_uv),
            end_uv_a: section.end_uv_a,
            start_uv_b: section.start_uv_b,
            end_uv_b: section.end_uv_b,
            target_face: section.target_face,
        };
        result.push(sub);
    }

    result
}
```

- [ ] **Step 6: Run unit test**

Run: `cargo test -p brepkit-algo crossing_sections_are_split`
Expected: PASS

- [ ] **Step 7: Wire up in `split_face_2d`**

In `face_splitter.rs`, at the top of `split_face_2d`, after the early returns,
split crossing sections before they're used:

```rust
// Split crossing section edges at their mutual intersection points.
// This converts geometric crossings into topological junctions that
// the wire builder can handle (star pattern instead of 4-way crossing).
let sections = split_crossing_section_edges(sections, tol.linear);
let sections = sections.as_slice();
```

Update all subsequent references to use the local `sections` binding (it shadows
the parameter).

- [ ] **Step 8: Un-ignore `gfa_intersect_overlapping_boxes`**

In `crates/algo/src/pave_filler/tests.rs`, remove `#[ignore = ...]` from
`gfa_intersect_overlapping_boxes`.

- [ ] **Step 9: Run integration test**

Run: `cargo test -p brepkit-algo gfa_intersect_overlapping_boxes`
Expected: PASS with 6 faces

- [ ] **Step 10: Run full algo test suite**

Run: `cargo test -p brepkit-algo`
Expected: All pass, 0 ignored

- [ ] **Step 11: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All pass (gridfinity and other tests unaffected)

- [ ] **Step 12: Commit**

```bash
git add crates/algo/src/builder/face_splitter.rs crates/algo/src/builder/wire_builder.rs crates/algo/src/pave_filler/tests.rs
git commit -m "fix(algo): split crossing section edges before wire builder

When two section edges cross on a face (e.g., box-box intersect with two
face pairs contributing chords), the wire builder needs a shared vertex at
the crossing point. Added split_crossing_section_edges() that detects 2D
UV crossings and splits each pair into half-edges, converting geometric
crossings into topological star junctions."
```

---

### Task 3: Full regression test + clippy

- [ ] **Step 1: Run clippy**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: PASS

- [ ] **Step 2: Run full test suite**

Run: `cargo test --workspace`
Expected: All pass

- [ ] **Step 3: Run gridfinity parity tests specifically**

Run: `cargo test -p brepkit-wasm gridfinity`
Expected: 24 passed, 1 ignored (D4 — unrelated)

- [ ] **Step 4: Check layer boundaries**

Run: `./scripts/check-boundaries.sh`
Expected: PASS

- [ ] **Step 5: Final commit if any fixups needed**
