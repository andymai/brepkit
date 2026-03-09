#![allow(clippy::unwrap_used, missing_docs)]

use brepkit_math::cdt::Cdt;
use brepkit_math::vec::Point2;
use std::time::Instant;

fn run_once() {
    let bounds = (Point2::new(-1.0, -1.0), Point2::new(101.0, 101.0));
    let mut cdt = Cdt::new(bounds);

    let segs = 32usize;
    let mut all_points = vec![
        Point2::new(0.0, 0.0),
        Point2::new(100.0, 0.0),
        Point2::new(100.0, 100.0),
        Point2::new(0.0, 100.0),
    ];
    let mut all_constraints: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (2, 3), (3, 0)];

    for row in 0..8_u32 {
        for col in 0..8_u32 {
            let cx = 6.0 + f64::from(col) * 12.0;
            let cy = 6.0 + f64::from(row) * 12.0;
            let r = 2.0;
            let start = all_points.len();
            for i in 0..segs {
                let theta = 2.0 * std::f64::consts::PI * i as f64 / segs as f64;
                all_points.push(Point2::new(cx + r * theta.cos(), cy + r * theta.sin()));
            }
            for i in 0..segs {
                all_constraints.push((start + i, start + (i + 1) % segs));
            }
        }
    }

    let t0 = Instant::now();
    let mut indices = Vec::with_capacity(all_points.len());
    for p in &all_points {
        indices.push(cdt.insert_point(*p).unwrap());
    }
    let t_insert = t0.elapsed();

    let t1 = Instant::now();
    for &(a, b) in &all_constraints {
        cdt.insert_constraint(indices[a], indices[b]).unwrap();
    }
    let t_constrain = t1.elapsed();

    let t2 = Instant::now();
    let boundary: Vec<(usize, usize)> = all_constraints
        .iter()
        .map(|&(a, b)| (indices[a], indices[b]))
        .collect();
    cdt.remove_exterior(&boundary);
    let t_exterior = t2.elapsed();

    let t3 = Instant::now();
    let tris = cdt.triangles();
    let t_collect = t3.elapsed();

    eprintln!(
        "Pts:{} Tris:{} | insert:{:?} constrain:{:?} exterior:{:?} collect:{:?} total:{:?}",
        all_points.len(),
        tris.len(),
        t_insert,
        t_constrain,
        t_exterior,
        t_collect,
        t0.elapsed()
    );
}

fn main() {
    for _ in 0..3 {
        run_once();
    }
    eprintln!("---");
    for _ in 0..5 {
        run_once();
    }
}
