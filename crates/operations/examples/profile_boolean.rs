#![allow(clippy::unwrap_used, missing_docs)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use std::time::Instant;

fn main() {
    let mut topo = Topology::new();
    let mut result = primitives::make_box(&mut topo, 50.0, 50.0, 10.0).unwrap();

    let positions: &[f64] = &[-15.0, -5.0, 5.0, 15.0];
    let mut cut_num = 0;

    for &x in positions {
        for &y in positions {
            cut_num += 1;
            let cyl = primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();
            let mat = Mat4::translation(x, y, -5.0);
            transform_solid(&mut topo, cyl, &mat).unwrap();

            let t0 = Instant::now();
            result = boolean(&mut topo, BooleanOp::Cut, result, cyl).unwrap();
            let elapsed = t0.elapsed();

            // Count faces in result
            let shell_id = topo.solid(result).unwrap().outer_shell();
            let face_count = topo.shell(shell_id).unwrap().faces().len();
            println!(
                "Cut {:2}: {:>8.2}ms  ({} faces)",
                cut_num,
                elapsed.as_secs_f64() * 1000.0,
                face_count
            );
        }
    }
    let shell_id = topo.solid(result).unwrap().outer_shell();
    let face_count = topo.shell(shell_id).unwrap().faces().len();
    let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.1).unwrap();
    let box_vol = 50.0 * 50.0 * 10.0;
    // 4 of the 16 cylinders actually intersect (x,y in {5,15} within box 0..50)
    let cyl_vol = 4.0 * std::f64::consts::PI * 9.0 * 10.0;
    let expected = box_vol - cyl_vol;
    println!("\nTotal faces: {face_count}");
    println!("Volume:   {vol:.2}");
    println!("Expected: {expected:.2} ({} cylinders removed)", 4);
    println!(
        "Error:    {:.2}%",
        ((vol - expected) / expected * 100.0).abs()
    );
}
