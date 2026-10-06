//! A caption glyph from the gridfinity tool (`labelPlateBuilder.test.ts`,
//! two-line captions): a prism standing from z 0.8 to 1.21 whose walls are
//! extruded from fitted font curves.
//!
//! A prism holds a point at every height of a column or at none, so a column
//! classifying both ways is a classifier fault. The ray cast refined each
//! NURBS wall crossing by projection, which converges only linearly along a
//! ray oblique to the wall: a crossing came back several times a few 1e-5
//! apart, or not at all, and flipped the ray's parity beside the walls.
//!
//! Data: `glyph_prism.bin`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::{oriented_solid_volume, solid_bounding_box};
use brepkit_topology::Topology;

#[test]
fn every_column_of_a_glyph_prism_classifies_one_way() {
    let mut topo = Topology::new();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/glyph_prism.bin");
    let glyph = deserialize_solid(&std::fs::read(path).unwrap(), &mut topo).unwrap();
    let bbox = solid_bounding_box(&topo, glyph).unwrap();
    let at =
        |lo: f64, hi: f64, k: u32, n: u32| lo + (hi - lo) * (f64::from(k) + 0.5) / f64::from(n);

    let mut columns = 0;
    let mut inside = 0_u32;
    for i in 0..24 {
        for j in 0..24 {
            let (x, y) = (
                at(bbox.min.x(), bbox.max.x(), i, 24),
                at(bbox.min.y(), bbox.max.y(), j, 24),
            );
            let reads: Vec<(f64, bool)> = (0..6)
                .filter_map(|k| {
                    let z = at(bbox.min.z(), bbox.max.z(), k, 6);
                    match classify_point(&topo, glyph, Point3::new(x, y, z), 0.01, 1e-7).unwrap() {
                        PointClassification::Inside => Some((z, true)),
                        PointClassification::Outside => Some((z, false)),
                        PointClassification::OnBoundary => None,
                    }
                })
                .collect();
            assert!(
                !reads.is_empty(),
                "column ({x}, {y}) reads only the boundary"
            );
            assert!(
                reads.iter().all(|r| r.1 == reads[0].1),
                "column ({x}, {y}) reads {reads:?}"
            );
            columns += 1;
            inside += u32::from(reads[0].1);
        }
    }
    assert_eq!(columns, 576);

    // The columns reading inside sample the glyph's outline over its box:
    // their share is the prism's volume over its box's, within the grid's
    // resolution, so reading every point outside fails as surely as a split
    // column.
    let volume = oriented_solid_volume(&topo, glyph, 0.001).unwrap();
    let share = volume
        / ((bbox.max.x() - bbox.min.x())
            * (bbox.max.y() - bbox.min.y())
            * (bbox.max.z() - bbox.min.z()));
    let read = f64::from(inside) / 576.0;
    assert!(
        (read - share).abs() < 0.03,
        "{inside} columns inside, outline share {share:.4}"
    );

    // The two columns the projection refinement misread: one inside the
    // glyph, one outside it.
    for (x, y, held) in [(7.0819, 1.0550, true), (6.78394, 0.37506, false)] {
        for z in [0.85, 0.9, 1.05, 1.19, 1.199] {
            assert_eq!(
                classify_point(&topo, glyph, Point3::new(x, y, z), 0.01, 1e-7).unwrap(),
                if held {
                    PointClassification::Inside
                } else {
                    PointClassification::Outside
                },
                "({x}, {y}, {z})"
            );
        }
    }
}
