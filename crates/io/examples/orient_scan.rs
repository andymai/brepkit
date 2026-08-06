//! Scan captured arena `.bin` solids for orientation-inconsistent shells.
//!
//! Runs `validate_solid_with_options` with `check_orientation` forced on over
//! every path given as an argument and prints the same-sense findings, so a
//! captured boolean chain can be bisected for the op that first emits
//! orientation-inconsistent faces.
//!
//! ```sh
//! cargo run --release -p brepkit-io --example orient_scan -- /tmp/capture/*.bin
//! ```
#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::validate::{ValidationOptions, validate_solid_with_options};
use brepkit_topology::Topology;

fn main() {
    let opts = ValidationOptions {
        check_orientation: true,
        ..Default::default()
    };
    for path in std::env::args().skip(1) {
        let mut topo = Topology::new();
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                println!("{path}: READ FAILED {e}");
                continue;
            }
        };
        let sid = match deserialize_solid(&data, &mut topo) {
            Ok(s) => s,
            Err(e) => {
                println!("{path}: DESERIALIZE FAILED {e}");
                continue;
            }
        };
        match validate_solid_with_options(&topo, sid, &opts) {
            Ok(report) => {
                let orient: Vec<&str> = report
                    .issues
                    .iter()
                    .filter(|i| i.description.contains("orientation"))
                    .map(|i| i.description.as_str())
                    .collect();
                if orient.is_empty() {
                    println!("{path}: clean ({} other issues)", report.issues.len());
                } else {
                    println!("{path}: {}", orient.join("; "));
                }
            }
            Err(e) => println!("{path}: VALIDATE FAILED {e}"),
        }
    }
}
