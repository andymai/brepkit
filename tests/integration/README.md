# Integration Tests

End-to-end tests that exercise multiple brepkit subsystems together.
These tests verify that primitives, operations, and I/O work correctly
when combined in realistic workflows.

## Where tests live

Integration tests are defined as unit tests inside crate modules (typically
in `crates/operations/` or `crates/io/`), not as standalone workspace-level
files. This directory contains documentation and shared test data.

## Test patterns

### Boolean operation workflow

```rust
#[cfg(test)]
mod integration_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;
    use crate::primitives::{make_box, make_cylinder};
    use crate::boolean::{boolean, BooleanOp};
    use crate::measure::{bounding_box, volume};

    #[test]
    fn boolean_subtract_cylinder_from_box() {
        let mut topo = Topology::new();

        // Create a box and a cylinder
        let box_solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let cyl_solid = make_cylinder(&mut topo, 2.0, 12.0).unwrap();

        // Subtract cylinder from box
        let result = boolean(&mut topo, box_solid, cyl_solid, BooleanOp::Cut).unwrap();

        // Verify: result volume < original box volume
        let result_vol = volume(&topo, result).unwrap();
        let box_vol = 10.0 * 10.0 * 10.0;
        assert!(result_vol < box_vol, "Cut should reduce volume");
        assert!(result_vol > 0.0, "Result should have positive volume");

        // Verify: bounding box unchanged (cylinder is inside box)
        let bbox = bounding_box(&topo, result).unwrap();
        let tolerance = 1e-6;
        assert!((bbox.max.x - 10.0).abs() < tolerance);
    }
}
```

### I/O roundtrip workflow

```rust
#[cfg(test)]
mod io_roundtrip_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::measure::volume;
    use crate::step::{write_step, read_step};
    use crate::stl::writer::write_stl_binary;

    #[test]
    fn step_roundtrip_preserves_topology() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 5.0, 3.0, 2.0).unwrap();
        let original_vol = volume(&topo, solid).unwrap();

        // Export to STEP
        let step_data = write_step(&topo, solid).unwrap();
        assert!(!step_data.is_empty());

        // Re-import from STEP
        let mut topo2 = Topology::new();
        let reimported = read_step(&mut topo2, &step_data).unwrap();

        // Verify volume matches
        let reimported_vol = volume(&topo2, reimported).unwrap();
        let tolerance = 1e-3; // STEP roundtrip may lose some precision
        assert!(
            (reimported_vol - original_vol).abs() < tolerance,
            "Volume mismatch: {reimported_vol} vs {original_vol}"
        );
    }

    #[test]
    fn stl_export_produces_valid_mesh() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let stl_data = write_stl_binary(&topo, solid).unwrap();

        // STL binary header is 80 bytes + 4 bytes triangle count
        assert!(stl_data.len() > 84, "STL should have header + triangles");

        // A box should produce at least 12 triangles (2 per face × 6 faces)
        let tri_count = u32::from_le_bytes([stl_data[80], stl_data[81], stl_data[82], stl_data[83]]);
        assert!(tri_count >= 12, "Box should have >= 12 triangles, got {tri_count}");
    }
}
```

## Key principles

1. **Measure, don't inspect topology** — Use `volume()`, `bounding_box()`, `area()`
   to verify results rather than counting faces/edges (which may vary with implementation)
2. **Use tolerances** — CAD operations accumulate floating-point error. Use appropriate
   tolerances (1e-6 for direct ops, 1e-3 for I/O roundtrips)
3. **Test the workflow, not the internals** — Integration tests should mirror real
   user workflows: create → operate → export → verify
