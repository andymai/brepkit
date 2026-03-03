# Golden File Tests

Golden file tests compare operation output against known-good reference data.
They catch regressions in tessellation, serialization, and geometry computation.

## How it works

1. Create a shape using brepkit operations
2. Produce output (mesh vertices, STEP text, measurements)
3. Compare against a `.golden` file in `tests/golden/data/`
4. If the output differs, the test fails with a diff

## Updating golden files

When intentional changes alter output (e.g., better tessellation), update the
golden files:

```bash
UPDATE_GOLDEN=1 cargo test --workspace golden
```

## Where tests live

Golden tests are defined as unit tests inside crate modules, not as standalone
files. The golden data files live here in `tests/golden/data/`.

Example test (in `crates/operations/src/tessellate.rs` or a dedicated test file):

```rust
#[cfg(test)]
mod golden_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_topology::Topology;
    use crate::primitives::make_box;
    use std::path::Path;

    /// Path to golden data relative to workspace root
    fn golden_path(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/golden/data")
            .join(name)
    }

    /// Compare output against golden file, update if UPDATE_GOLDEN=1
    fn assert_golden(name: &str, actual: &str) {
        let path = golden_path(name);
        if std::env::var("UPDATE_GOLDEN").is_ok() {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, actual).unwrap();
            return;
        }
        let expected = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("Golden file not found: {path:?}\nRun with UPDATE_GOLDEN=1 to create it."));
        assert_eq!(actual.trim(), expected.trim(), "Golden file mismatch: {name}");
    }

    #[test]
    fn golden_box_tessellation() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let mesh = tessellate(&topo, solid, 0.1).unwrap();

        // Format mesh as deterministic string
        let mut output = String::new();
        output.push_str(&format!("vertices: {}\n", mesh.vertices.len()));
        output.push_str(&format!("triangles: {}\n", mesh.indices.len() / 3));
        for v in &mesh.vertices {
            output.push_str(&format!("v {:.6} {:.6} {:.6}\n", v.x, v.y, v.z));
        }
        for chunk in mesh.indices.chunks(3) {
            output.push_str(&format!("f {} {} {}\n", chunk[0], chunk[1], chunk[2]));
        }

        assert_golden("box_mesh.golden", &output);
    }
}
```

## Naming convention

- `{shape}_{operation}.golden` — e.g., `box_mesh.golden`, `cylinder_step.golden`
- Keep golden files small (< 100 lines) by testing representative shapes
