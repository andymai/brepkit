//! A part with B-spline walls whose volume integrals settle only at their
//! samples' noise: each settles, but a face's boundary integral then asks for
//! them without end. The volume must come back promptly, by another way where
//! that face's integral runs past its budget.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::mpsc;
use std::time::Duration;

#[test]
fn a_spline_walled_part_measures_promptly() {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let text = include_str!("data/spline_wall_volume_settles.step");
        let mut topo = brepkit_topology::Topology::new();
        let solids = brepkit_io::step::read_step(text, &mut topo).unwrap();
        assert_eq!(solids.len(), 1);
        let v = brepkit_operations::measure::solid_volume(&topo, solids[0], 0.001).unwrap();
        tx.send(v).unwrap();
    });
    // It takes well under a second; without the budget, minutes.
    let v = rx
        .recv_timeout(Duration::from_secs(60))
        .expect("the volume did not come back within 60 s");
    assert!((v - 149_078.878_8).abs() <= 1e-6 * v, "volume {v}");
}
