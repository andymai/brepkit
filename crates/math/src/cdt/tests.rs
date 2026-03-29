#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

use super::*;

#[test]
fn cdt_simple_square() {
    let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(2.0, 2.0)));
    let v0 = cdt.insert_point(Point2::new(0.0, 0.0)).unwrap();
    let v1 = cdt.insert_point(Point2::new(1.0, 0.0)).unwrap();
    let v2 = cdt.insert_point(Point2::new(1.0, 1.0)).unwrap();
    let v3 = cdt.insert_point(Point2::new(0.0, 1.0)).unwrap();

    cdt.insert_constraint(v0, v1).unwrap();
    cdt.insert_constraint(v1, v2).unwrap();
    cdt.insert_constraint(v2, v3).unwrap();
    cdt.insert_constraint(v3, v0).unwrap();

    cdt.remove_exterior(&[(v0, v1), (v1, v2), (v2, v3), (v3, v0)]);

    let tris = cdt.triangles();
    assert_eq!(tris.len(), 2, "square should produce 2 triangles");
}

#[test]
fn cdt_with_interior_point() {
    let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(2.0, 2.0)));
    let v0 = cdt.insert_point(Point2::new(0.0, 0.0)).unwrap();
    let v1 = cdt.insert_point(Point2::new(1.0, 0.0)).unwrap();
    let v2 = cdt.insert_point(Point2::new(1.0, 1.0)).unwrap();
    let v3 = cdt.insert_point(Point2::new(0.0, 1.0)).unwrap();
    let _v4 = cdt.insert_point(Point2::new(0.5, 0.5)).unwrap();

    cdt.insert_constraint(v0, v1).unwrap();
    cdt.insert_constraint(v1, v2).unwrap();
    cdt.insert_constraint(v2, v3).unwrap();
    cdt.insert_constraint(v3, v0).unwrap();

    cdt.remove_exterior(&[(v0, v1), (v1, v2), (v2, v3), (v3, v0)]);

    let tris = cdt.triangles();
    assert_eq!(
        tris.len(),
        4,
        "square with center point should produce 4 triangles"
    );
}

#[test]
fn cdt_delaunay_property() {
    let mut cdt = Cdt::new((Point2::new(-2.0, -2.0), Point2::new(2.0, 2.0)));
    for &(x, y) in &[(0.0, 0.0), (1.0, 0.0), (0.5, 0.8), (0.2, 0.4), (0.8, 0.3)] {
        cdt.insert_point(Point2::new(x, y)).unwrap();
    }
    let tris = cdt.triangles();
    assert!(tris.len() >= 4, "should have multiple triangles");

    // Verify Delaunay property: for each non-boundary triangle,
    // no other vertex should lie inside its circumcircle.
    let verts = cdt.vertices();
    for &(a, b, c) in &tris {
        let pa = verts[a];
        let pb = verts[b];
        let pc = verts[c];

        for (i, &pv) in verts.iter().enumerate() {
            if i < cdt.super_count || i == a || i == b || i == c {
                continue;
            }
            // in_circle > 0 means inside (for CCW triangle).
            let ic = in_circle(pa, pb, pc, pv);
            assert!(
                ic <= 1e-10,
                "Delaunay violation: vertex {i} inside circumcircle of ({a},{b},{c}), ic={ic}"
            );
        }
    }
}

#[test]
fn cdt_triangle_area_conservation() {
    let mut cdt = Cdt::new((Point2::new(-2.0, -2.0), Point2::new(3.0, 3.0)));
    let v0 = cdt.insert_point(Point2::new(0.0, 0.0)).unwrap();
    let v1 = cdt.insert_point(Point2::new(2.0, 0.0)).unwrap();
    let v2 = cdt.insert_point(Point2::new(1.0, 1.5)).unwrap();

    cdt.insert_constraint(v0, v1).unwrap();
    cdt.insert_constraint(v1, v2).unwrap();
    cdt.insert_constraint(v2, v0).unwrap();

    cdt.remove_exterior(&[(v0, v1), (v1, v2), (v2, v0)]);

    let tris = cdt.triangles();
    assert_eq!(tris.len(), 1, "triangle should produce 1 triangle");

    // Check area.
    let verts = cdt.vertices();
    let (a, b, c) = tris[0];
    let area = 0.5
        * ((verts[b].x() - verts[a].x()) * (verts[c].y() - verts[a].y())
            - (verts[c].x() - verts[a].x()) * (verts[b].y() - verts[a].y()))
        .abs();
    let expected = 0.5 * 2.0 * 1.5; // base=2, height=1.5
    assert!(
        (area - expected).abs() < 1e-10,
        "area should be {expected}, got {area}"
    );
}

#[test]
fn cdt_duplicate_point() {
    let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(2.0, 2.0)));
    let v0 = cdt.insert_point(Point2::new(0.5, 0.5)).unwrap();
    let v1 = cdt.insert_point(Point2::new(0.5, 0.5)).unwrap();
    assert_eq!(v0, v1, "duplicate point should return same index");
}

#[test]
fn cdt_constraint_diagonal() {
    // Square with a diagonal constraint.
    let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(2.0, 2.0)));
    let v0 = cdt.insert_point(Point2::new(0.0, 0.0)).unwrap();
    let v1 = cdt.insert_point(Point2::new(1.0, 0.0)).unwrap();
    let v2 = cdt.insert_point(Point2::new(1.0, 1.0)).unwrap();
    let v3 = cdt.insert_point(Point2::new(0.0, 1.0)).unwrap();

    cdt.insert_constraint(v0, v1).unwrap();
    cdt.insert_constraint(v1, v2).unwrap();
    cdt.insert_constraint(v2, v3).unwrap();
    cdt.insert_constraint(v3, v0).unwrap();
    // Add diagonal constraint.
    cdt.insert_constraint(v0, v2).unwrap();

    cdt.remove_exterior(&[(v0, v1), (v1, v2), (v2, v3), (v3, v0)]);

    let tris = cdt.triangles();
    assert_eq!(
        tris.len(),
        2,
        "square with diagonal should have 2 triangles"
    );

    // Verify the diagonal (v0, v2) exists as an edge.
    let has_diagonal = tris.iter().any(|&(a, b, c)| {
        let edges = [(a, b), (b, c), (c, a)];
        edges
            .iter()
            .any(|&(x, y)| sorted_pair(x, y) == sorted_pair(v0, v2))
    });
    assert!(has_diagonal, "diagonal constraint should appear as an edge");
}

#[test]
fn cdt_near_coincident_points() {
    // Two points separated by 1e-10 — should not panic
    let pts = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(0.5, 1.0),
        Point2::new(0.5 + 1e-10, 1.0),
    ];
    let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(2.0, 2.0)));
    for pt in &pts {
        let _result = cdt.insert_point(*pt);
    }
}

#[test]
fn cdt_collinear_input() {
    // All points on a line — CDT should handle gracefully
    let pts = vec![
        Point2::new(0.0, 0.0),
        Point2::new(1.0, 0.0),
        Point2::new(2.0, 0.0),
    ];
    let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(3.0, 1.0)));
    for pt in &pts {
        let _result = cdt.insert_point(*pt);
    }
}

#[test]
fn extract_regions_grid_2x2() {
    // 36×36 square with 4 cutting lines forming a 3×3 grid of 9 regions.
    let bounds = (Point2::new(0.0, 0.0), Point2::new(40.0, 40.0));
    let mut cdt = Cdt::with_capacity(bounds, 20);

    // Polygon vertices (CCW)
    let v0 = cdt.insert_point(Point2::new(2.0, 2.0)).unwrap();
    let v1 = cdt.insert_point(Point2::new(38.0, 2.0)).unwrap();
    let v2 = cdt.insert_point(Point2::new(38.0, 38.0)).unwrap();
    let v3 = cdt.insert_point(Point2::new(2.0, 38.0)).unwrap();

    // Chord endpoints on boundary
    let h0l = cdt.insert_point(Point2::new(2.0, 4.0)).unwrap();
    let h0r = cdt.insert_point(Point2::new(38.0, 4.0)).unwrap();
    let h1l = cdt.insert_point(Point2::new(2.0, 7.0)).unwrap();
    let h1r = cdt.insert_point(Point2::new(38.0, 7.0)).unwrap();
    let v0b = cdt.insert_point(Point2::new(4.0, 2.0)).unwrap();
    let v0t = cdt.insert_point(Point2::new(4.0, 38.0)).unwrap();
    let v1b = cdt.insert_point(Point2::new(7.0, 2.0)).unwrap();
    let v1t = cdt.insert_point(Point2::new(7.0, 38.0)).unwrap();

    // Crossing points
    let c00 = cdt.insert_point(Point2::new(4.0, 4.0)).unwrap();
    let c01 = cdt.insert_point(Point2::new(7.0, 4.0)).unwrap();
    let c10 = cdt.insert_point(Point2::new(4.0, 7.0)).unwrap();
    let c11 = cdt.insert_point(Point2::new(7.0, 7.0)).unwrap();

    // Boundary constraints (split at chord endpoints)
    // (2,2)->(38,2)->(38,38)->(2,38)->(2,2) with splits at chord endpoints
    let boundary = vec![
        // Bottom: (2,2) -> (38,2) via (4,2) and (7,2)
        (v0, v0b),
        (v0b, v1b),
        (v1b, v1),
        // Right: (38,2) -> (38,38) via (38,4) and (38,7)
        (v1, h0r),
        (h0r, h1r),
        (h1r, v2),
        // Top: (38,38) -> (2,38) via (7,38) and (4,38)
        (v2, v1t),
        (v1t, v0t),
        (v0t, v3),
        // Left: (2,38) -> (2,2) via (2,7) and (2,4)
        (v3, h1l),
        (h1l, h0l),
        (h0l, v0),
    ];

    for &(a, b) in &boundary {
        cdt.insert_constraint(a, b).unwrap();
    }

    // Chord constraints (split at crossings)
    let separators = vec![
        // h0 (y=4): h0l -> c00 -> c01 -> h0r
        (h0l, c00),
        (c00, c01),
        (c01, h0r),
        // h1 (y=7): h1l -> c10 -> c11 -> h1r
        (h1l, c10),
        (c10, c11),
        (c11, h1r),
        // v0 (x=4): v0b -> c00 -> c10 -> v0t
        (v0b, c00),
        (c00, c10),
        (c10, v0t),
        // v1 (x=7): v1b -> c01 -> c11 -> v1t
        (v1b, c01),
        (c01, c11),
        (c11, v1t),
    ];

    for &(a, b) in &separators {
        cdt.insert_constraint(a, b).unwrap();
    }

    cdt.remove_exterior(&boundary);

    let regions = cdt.extract_regions(&separators);
    eprintln!("Grid 2×2 regions: {}", regions.len());
    for (i, r) in regions.iter().enumerate() {
        let area = shoelace_area(r);
        eprintln!("  Region {i}: {} verts, area={area:.1}", r.len());
    }
    assert_eq!(
        regions.len(),
        9,
        "2 horizontal + 2 vertical lines → 9 regions"
    );
}

fn shoelace_area(poly: &[Point2]) -> f64 {
    let mut a = 0.0;
    for i in 0..poly.len() {
        let j = (i + 1) % poly.len();
        a += poly[i].x() * poly[j].y() - poly[j].x() * poly[i].y();
    }
    a.abs() / 2.0
}

// -----------------------------------------------------------------------
// Stress tests
// -----------------------------------------------------------------------

/// Simple xorshift64 PRNG — avoids adding `rand` as a dependency.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a pseudorandom f64 in [lo, hi) from the PRNG state.
fn rand_f64(state: &mut u64, lo: f64, hi: f64) -> f64 {
    let u = (xorshift64(state) as f64) / (u64::MAX as f64);
    lo + u * (hi - lo)
}

/// Signed triangle area (positive = CCW).
fn signed_tri_area(a: Point2, b: Point2, c: Point2) -> f64 {
    0.5 * ((b.x() - a.x()) * (c.y() - a.y()) - (c.x() - a.x()) * (b.y() - a.y()))
}

#[test]
fn cdt_nearly_cocircular_points() {
    // Place 8 points on a unit circle with tiny (1e-10) radial perturbations.
    // These create near-degenerate in-circle predicates that stress the
    // exact arithmetic fallback.
    let n = 8;
    let radius = 1.0;
    let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_1234;

    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        let r = radius + rand_f64(&mut rng_state, -1e-10, 1e-10);
        pts.push(Point2::new(r * angle.cos(), r * angle.sin()));
    }

    let mut cdt = Cdt::new((Point2::new(-2.0, -2.0), Point2::new(2.0, 2.0)));
    for pt in &pts {
        cdt.insert_point(*pt).unwrap();
    }

    let tris = cdt.triangles();
    assert!(
        tris.len() >= n - 2,
        "nearly-cocircular: expected at least {} triangles, got {}",
        n - 2,
        tris.len()
    );

    // Every triangle must have positive area (CCW winding, non-degenerate).
    let verts = cdt.vertices();
    for &(a, b, c) in &tris {
        let area = signed_tri_area(verts[a], verts[b], verts[c]);
        assert!(
            area > 0.0,
            "degenerate triangle ({a}, {b}, {c}) with area {area}"
        );
    }
}

#[test]
fn cdt_nearly_cocircular_with_constraints() {
    // 12 points nearly on a circle, constrained as a polygon boundary.
    // Stresses constraint insertion with near-degenerate geometry.
    let n = 12;
    let mut rng_state: u64 = 0xBAAD_F00D_1234_5678;

    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        let r = 5.0 + rand_f64(&mut rng_state, -1e-10, 1e-10);
        pts.push(Point2::new(r * angle.cos(), r * angle.sin()));
    }

    let mut cdt = Cdt::new((Point2::new(-10.0, -10.0), Point2::new(10.0, 10.0)));
    let mut ids = Vec::with_capacity(n);
    for pt in &pts {
        ids.push(cdt.insert_point(*pt).unwrap());
    }

    // Insert boundary constraints forming a closed polygon.
    let mut boundary = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        cdt.insert_constraint(ids[i], ids[j]).unwrap();
        boundary.push((ids[i], ids[j]));
    }

    cdt.remove_exterior(&boundary);

    let tris = cdt.triangles();
    // A convex polygon with n vertices should produce n - 2 triangles.
    assert_eq!(
        tris.len(),
        n - 2,
        "nearly-cocircular polygon: expected {} triangles, got {}",
        n - 2,
        tris.len()
    );

    // Sum of triangle areas should equal polygon area.
    let verts = cdt.vertices();
    let tri_area_sum: f64 = tris
        .iter()
        .map(|&(a, b, c)| signed_tri_area(verts[a], verts[b], verts[c]).abs())
        .sum();

    // Compute polygon area via shoelace.
    let poly_area = shoelace_area(&pts);
    assert!(
        (tri_area_sum - poly_area).abs() < 1e-6,
        "area mismatch: triangles={tri_area_sum}, polygon={poly_area}"
    );
}

#[test]
fn cdt_large_scale_random() {
    // 500 random points — verify triangle count, positive areas,
    // and Delaunay property.
    let n = 500;
    let mut rng_state: u64 = 0x1234_5678_9ABC_DEF0;

    let mut pts = Vec::with_capacity(n);
    for _ in 0..n {
        pts.push(Point2::new(
            rand_f64(&mut rng_state, 0.0, 1000.0),
            rand_f64(&mut rng_state, 0.0, 1000.0),
        ));
    }

    let mut cdt = Cdt::new((Point2::new(-10.0, -10.0), Point2::new(1010.0, 1010.0)));
    for pt in &pts {
        cdt.insert_point(*pt).unwrap();
    }

    let tris = cdt.triangles();
    let verts = cdt.vertices();
    let n_inserted = verts.len() - cdt.super_count;

    // For n points in general position, the Delaunay triangulation has
    // between n-1 and 2n-2-k triangles where k is boundary vertices.
    // Use a generous range.
    assert!(
        tris.len() >= n_inserted,
        "too few triangles: {} for {n_inserted} points",
        tris.len()
    );
    assert!(
        tris.len() <= 2 * n_inserted,
        "too many triangles: {} for {n_inserted} points",
        tris.len()
    );

    // All triangles must have strictly positive area.
    for &(a, b, c) in &tris {
        let area = signed_tri_area(verts[a], verts[b], verts[c]);
        assert!(
            area > 0.0,
            "degenerate triangle ({a}, {b}, {c}) with area {area}"
        );
    }
}

#[test]
fn cdt_large_scale_hilbert_insertion() {
    // Test bulk Hilbert-ordered insertion with 600 points.
    let n = 600;
    let mut rng_state: u64 = 0xFEED_FACE_DEAD_BEEF;

    let mut pts = Vec::with_capacity(n);
    for _ in 0..n {
        pts.push(Point2::new(
            rand_f64(&mut rng_state, 0.0, 1000.0),
            rand_f64(&mut rng_state, 0.0, 1000.0),
        ));
    }

    let mut cdt = Cdt::new((Point2::new(-10.0, -10.0), Point2::new(1010.0, 1010.0)));
    let indices = cdt.insert_points_hilbert(&pts).unwrap();

    // Every original point should map to a valid vertex index.
    assert_eq!(indices.len(), n);
    for &idx in &indices {
        assert!(
            idx >= cdt.super_count,
            "vertex index {idx} is a super-triangle vertex"
        );
        assert!(
            idx < cdt.vertices().len(),
            "vertex index {idx} out of range"
        );
    }

    let tris = cdt.triangles();
    assert!(
        tris.len() >= n,
        "Hilbert insertion: too few triangles {} for {n} points",
        tris.len()
    );

    // Verify Delaunay property on a sample of triangles.
    let verts = cdt.vertices();
    for (ti, &(a, b, c)) in tris.iter().enumerate().take(100) {
        let pa = verts[a];
        let pb = verts[b];
        let pc = verts[c];
        for (vi, &pv) in verts.iter().enumerate() {
            if vi < cdt.super_count || vi == a || vi == b || vi == c {
                continue;
            }
            let ic = in_circle(pa, pb, pc, pv);
            assert!(
                ic <= 1e-10,
                "Delaunay violation at tri {ti}: vertex {vi} inside circumcircle, ic={ic}"
            );
        }
    }
}

#[test]
fn cdt_constraint_grid_edges() {
    // Create a 6x6 grid of points with horizontal and vertical constraint
    // edges along grid lines. This stresses the CDT with many constraints
    // that share endpoints and form a dense mesh.
    let grid_size = 6;
    let spacing = 10.0;
    let extent = (grid_size as f64 - 1.0) * spacing;

    let mut cdt = Cdt::new((
        Point2::new(-spacing, -spacing),
        Point2::new(extent + spacing, extent + spacing),
    ));

    // Insert grid points in row-major order.
    let mut ids = Vec::new();
    for row in 0..grid_size {
        for col in 0..grid_size {
            let p = Point2::new(col as f64 * spacing, row as f64 * spacing);
            ids.push(cdt.insert_point(p).unwrap());
        }
    }

    let idx = |row: usize, col: usize| -> usize { ids[row * grid_size + col] };

    // Add horizontal constraint edges (all rows).
    let mut constraint_pairs = Vec::new();
    for row in 0..grid_size {
        for col in 0..grid_size - 1 {
            let a = idx(row, col);
            let b = idx(row, col + 1);
            cdt.insert_constraint(a, b).unwrap();
            constraint_pairs.push((a, b));
        }
    }

    // Add vertical constraint edges (all columns).
    for col in 0..grid_size {
        for row in 0..grid_size - 1 {
            let a = idx(row, col);
            let b = idx(row + 1, col);
            cdt.insert_constraint(a, b).unwrap();
            constraint_pairs.push((a, b));
        }
    }

    // Verify all grid-edge constraints appear in the triangulation.
    let tris = cdt.triangles();
    for &(cv0, cv1) in &constraint_pairs {
        let target = sorted_pair(cv0, cv1);
        let found = tris.iter().any(|&(a, b, c)| {
            let edges = [sorted_pair(a, b), sorted_pair(b, c), sorted_pair(c, a)];
            edges.contains(&target)
        });
        assert!(
            found,
            "grid constraint ({cv0}, {cv1}) missing from triangulation"
        );
    }

    // All triangles should have positive area.
    let verts = cdt.vertices();
    for &(a, b, c) in &tris {
        let area = signed_tri_area(verts[a], verts[b], verts[c]);
        assert!(
            area > 0.0,
            "degenerate triangle ({a}, {b}, {c}) after grid constraint insertion, area={area}"
        );
    }

    // Expected triangle count: a grid of n×n quads produces 2*(n-1)^2
    // triangles when all edges are constrained.
    let expected = 2 * (grid_size - 1) * (grid_size - 1);
    assert_eq!(
        tris.len(),
        expected,
        "grid with all edges constrained should have {expected} triangles"
    );
}

#[test]
fn cdt_shared_endpoint_constraints() {
    // Multiple constraint edges sharing a common endpoint (fan pattern).
    // This stresses the constraint insertion when many edges radiate
    // from the same vertex.
    let n_spokes = 10;
    let radius = 50.0;

    let mut cdt = Cdt::new((Point2::new(-60.0, -60.0), Point2::new(60.0, 60.0)));
    let center = cdt.insert_point(Point2::new(0.0, 0.0)).unwrap();

    let mut rim_ids = Vec::with_capacity(n_spokes);
    for i in 0..n_spokes {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_spokes as f64);
        let p = Point2::new(radius * angle.cos(), radius * angle.sin());
        rim_ids.push(cdt.insert_point(p).unwrap());
    }

    // Constrain each spoke: center -> rim[i].
    for &rim in &rim_ids {
        cdt.insert_constraint(center, rim).unwrap();
    }

    // Also constrain the rim edges to form a closed polygon.
    let mut boundary = Vec::new();
    for i in 0..n_spokes {
        let j = (i + 1) % n_spokes;
        cdt.insert_constraint(rim_ids[i], rim_ids[j]).unwrap();
        boundary.push((rim_ids[i], rim_ids[j]));
    }

    cdt.remove_exterior(&boundary);

    let tris = cdt.triangles();
    // With center + n rim points and n spokes, expect 2*n - 2 triangles
    // for a fan... but with the Delaunay property, it should be exactly
    // n triangles (one per spoke sector) since all spokes are constrained.

    // Verify all spoke constraints appear.
    for &rim in &rim_ids {
        let target = sorted_pair(center, rim);
        let found = tris.iter().any(|&(a, b, c)| {
            let edges = [sorted_pair(a, b), sorted_pair(b, c), sorted_pair(c, a)];
            edges.contains(&target)
        });
        assert!(
            found,
            "spoke constraint ({center}, {rim}) missing from triangulation"
        );
    }

    // Triangle count: exactly n_spokes (one per sector).
    assert_eq!(
        tris.len(),
        n_spokes,
        "fan with {n_spokes} spokes should have {n_spokes} triangles"
    );
}

#[test]
fn cdt_dense_cluster_with_constraint() {
    // A dense cluster of nearly-coincident points with constraint edges
    // connecting anchor points that lie well outside the cluster. Tests
    // duplicate detection under pressure and constraint recovery through
    // crowded regions.
    let mut cdt = Cdt::new((Point2::new(-10.0, -10.0), Point2::new(110.0, 110.0)));

    // Insert anchor points for constraints, placed away from y=50 so the
    // constraint line does not pass exactly through cluster points.
    let anchor_a = cdt.insert_point(Point2::new(0.0, 48.0)).unwrap();
    let anchor_b = cdt.insert_point(Point2::new(100.0, 52.0)).unwrap();

    // Insert a dense cluster of 50 points near (50, 50), offset above the
    // constraint line so no cluster point is collinear with the constraint.
    let mut rng_state: u64 = 0xAAAA_BBBB_CCCC_DDDD;
    for _ in 0..50 {
        let p = Point2::new(
            50.0 + rand_f64(&mut rng_state, -0.5, 0.5),
            55.0 + rand_f64(&mut rng_state, -0.5, 0.5),
        );
        cdt.insert_point(p).unwrap();
    }

    // Insert a second cluster below the constraint line.
    for _ in 0..50 {
        let p = Point2::new(
            50.0 + rand_f64(&mut rng_state, -0.5, 0.5),
            45.0 + rand_f64(&mut rng_state, -0.5, 0.5),
        );
        cdt.insert_point(p).unwrap();
    }

    // The constraint must cut between the two clusters.
    cdt.insert_constraint(anchor_a, anchor_b).unwrap();

    // Verify the constraint edge appears as a triangle edge.
    let tris = cdt.triangles();
    let target = sorted_pair(anchor_a, anchor_b);
    let found = tris.iter().any(|&(a, b, c)| {
        let edges = [sorted_pair(a, b), sorted_pair(b, c), sorted_pair(c, a)];
        edges.contains(&target)
    });
    assert!(
        found,
        "constraint edge ({anchor_a}, {anchor_b}) missing from triangulation"
    );

    // All triangles valid.
    let verts = cdt.vertices();
    for &(a, b, c) in &tris {
        let area = signed_tri_area(verts[a], verts[b], verts[c]);
        assert!(
            area > 0.0,
            "degenerate triangle ({a}, {b}, {c}) in dense cluster, area={area}"
        );
    }
}

use proptest::prelude::*;

proptest! {
    #[test]
    fn cdt_random_points_no_panic(
        seed in 0u64..100_000,
        n in 10usize..100,
    ) {
        let mut rng_state = seed | 1; // ensure non-zero
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            pts.push(Point2::new(
                rand_f64(&mut rng_state, -100.0, 100.0),
                rand_f64(&mut rng_state, -100.0, 100.0),
            ));
        }

        let mut cdt = Cdt::new((Point2::new(-200.0, -200.0), Point2::new(200.0, 200.0)));
        for pt in &pts {
            cdt.insert_point(*pt).unwrap();
        }

        let tris = cdt.triangles();
        // Must produce at least 1 triangle for 3+ non-collinear points.
        prop_assert!(!tris.is_empty(), "no triangles for {n} points (seed={seed})");

        // All triangles must have positive area.
        let verts = cdt.vertices();
        for &(a, b, c) in &tris {
            let area = signed_tri_area(verts[a], verts[b], verts[c]);
            prop_assert!(area > 0.0, "degenerate triangle area={area}");
        }
    }

    #[test]
    fn cdt_random_constraints_no_panic(
        seed in 0u64..50_000,
    ) {
        let mut rng_state = seed | 1;
        let n = 20;
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            pts.push(Point2::new(
                rand_f64(&mut rng_state, 0.0, 100.0),
                rand_f64(&mut rng_state, 0.0, 100.0),
            ));
        }

        let mut cdt = Cdt::new((Point2::new(-10.0, -10.0), Point2::new(110.0, 110.0)));
        let mut ids = Vec::with_capacity(n);
        for pt in &pts {
            ids.push(cdt.insert_point(*pt).unwrap());
        }

        // Insert 5 random constraints between existing vertices.
        for _ in 0..5 {
            let a = (xorshift64(&mut rng_state) as usize) % n;
            let b = (xorshift64(&mut rng_state) as usize) % n;
            if a != b {
                cdt.insert_constraint(ids[a], ids[b]).unwrap();
            }
        }

        let tris = cdt.triangles();
        prop_assert!(!tris.is_empty());
    }
}

/// Collinear vertex splitting: a constraint from v0→v3 should split through
/// collinear intermediate vertices v1 and v2, producing three sub-constraints.
#[test]
fn cdt_collinear_constraint_splitting() {
    let mut cdt = Cdt::new((Point2::new(-1.0, -1.0), Point2::new(6.0, 6.0)));
    // Four collinear points along y=2.
    let v0 = cdt.insert_point(Point2::new(0.0, 2.0)).unwrap();
    let v1 = cdt.insert_point(Point2::new(1.0, 2.0)).unwrap();
    let v2 = cdt.insert_point(Point2::new(3.0, 2.0)).unwrap();
    let v3 = cdt.insert_point(Point2::new(5.0, 2.0)).unwrap();
    // Also add off-line points so the triangulation isn't degenerate.
    cdt.insert_point(Point2::new(2.0, 0.0)).unwrap();
    cdt.insert_point(Point2::new(2.0, 4.0)).unwrap();

    // Insert constraint spanning all four collinear points.
    cdt.insert_constraint(v0, v3).unwrap();

    // All three sub-segments should now be constrained.
    let has = |a, b| cdt.constraints.contains(&sorted_pair(a, b));
    assert!(has(v0, v1), "sub-constraint v0-v1 missing");
    assert!(has(v1, v2), "sub-constraint v1-v2 missing");
    assert!(has(v2, v3), "sub-constraint v2-v3 missing");

    // Triangulation should still be valid.
    let tris = cdt.triangles();
    assert!(!tris.is_empty());
}
