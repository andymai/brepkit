use anyhow::{bail, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Pinned wasm-bindgen-cli version. Must match `wasm-bindgen = "=0.2.114"` in
/// the workspace Cargo.toml.
const WASM_BINDGEN_VERSION: &str = "0.2.114";

/// Minimum number of exported methods expected in the .d.ts file.
/// Based on ~185 methods in the current BrepKernel. Update when the API surface
/// changes significantly.
const MIN_METHOD_COUNT: usize = 170;

/// Valid .wasm file size range (bytes).
const MIN_WASM_SIZE: u64 = 500_000;
const MAX_WASM_SIZE: u64 = 20_000_000;

fn project_root() -> Result<PathBuf> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .context("xtask must be located inside the project root")
}

fn pkg_dir() -> Result<PathBuf> {
    Ok(project_root()?.join("crates/wasm/pkg"))
}

fn pkg_node_dir() -> Result<PathBuf> {
    Ok(project_root()?.join("crates/wasm/pkg-node"))
}

fn run_cmd(cmd: &mut Command) -> Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("failed to run: {cmd:?}"))?;
    if !status.success() {
        bail!("command failed with {status}: {cmd:?}");
    }
    Ok(())
}

fn run_cmd_output(cmd: &mut Command) -> Result<String> {
    let output = cmd
        .output()
        .with_context(|| format!("failed to run: {cmd:?}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("command failed with {}: {stderr}", output.status);
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn command_exists(name: &str) -> bool {
    // Use `which` — standard on Linux/macOS where WASM builds run.
    Command::new("which")
        .arg(name)
        .output()
        .is_ok_and(|o| o.status.success())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Verify required tools are installed and versions match.
pub fn check_tools() -> Result<()> {
    println!("Checking tools...");

    if !command_exists("wasm-pack") {
        bail!(
            "wasm-pack not found. Install with:\n  \
             cargo binstall wasm-pack --no-confirm\n  \
             or: cargo install wasm-pack --locked"
        );
    }

    // wasm-bindgen-cli version check
    if command_exists("wasm-bindgen") {
        let version = run_cmd_output(
            Command::new("wasm-bindgen").arg("--version"),
        )?;
        // Output is like "wasm-bindgen 0.2.114"
        let installed = version.split_whitespace().last().unwrap_or("");
        if installed != WASM_BINDGEN_VERSION {
            bail!(
                "wasm-bindgen-cli version mismatch: installed={installed}, \
                 required={WASM_BINDGEN_VERSION}\n  \
                 Fix: cargo binstall wasm-bindgen-cli@{WASM_BINDGEN_VERSION} --no-confirm"
            );
        }
        println!("  wasm-bindgen-cli {WASM_BINDGEN_VERSION} ok");
    } else {
        println!(
            "  warning: wasm-bindgen-cli not found (wasm-pack bundles its own, \
             but pinned version won't be verified)"
        );
    }

    if command_exists("wasm-opt") {
        println!("  wasm-opt ok");
    } else {
        println!("  warning: wasm-opt not found, optimization will be skipped");
    }

    Ok(())
}

/// Build WASM for both bundler and nodejs targets.
pub fn build_both_targets(simd: bool) -> Result<()> {
    let wasm_crate = project_root()?.join("crates/wasm");

    let mut rustflags = String::from("-Dwarnings");
    if simd {
        rustflags.push_str(" -C target-feature=+simd128");
    }

    println!("\nBuilding WASM (bundler target)...");
    run_cmd(
        Command::new("wasm-pack")
            .args(["build", "--target", "bundler", "--release", "--out-dir", "pkg"])
            .current_dir(&wasm_crate)
            .env("RUSTFLAGS", &rustflags),
    )
    .context("wasm-pack build (bundler) failed")?;

    println!("\nBuilding WASM (nodejs target)...");
    run_cmd(
        Command::new("wasm-pack")
            .args(["build", "--target", "nodejs", "--release", "--out-dir", "pkg-node"])
            .current_dir(&wasm_crate)
            .env("RUSTFLAGS", &rustflags),
    )
    .context("wasm-pack build (nodejs) failed")?;

    Ok(())
}

/// Run wasm-opt on the bundler .wasm file. Skips if wasm-opt is not installed.
pub fn run_wasm_opt() -> Result<()> {
    if !command_exists("wasm-opt") {
        println!("\nSkipping wasm-opt (not installed)");
        return Ok(());
    }

    let wasm_file = pkg_dir()?.join("brepkit_wasm_bg.wasm");
    if !wasm_file.exists() {
        bail!("WASM file not found: {}", wasm_file.display());
    }

    let size_before = fs::metadata(&wasm_file)?.len();
    println!(
        "\nRunning wasm-opt (before: {:.1} KB)...",
        size_before as f64 / 1024.0
    );

    let opt_file = wasm_file.with_extension("wasm.opt");
    run_cmd(
        Command::new("wasm-opt")
            .args(["-O3"])
            .arg(&wasm_file)
            .args(["-o"])
            .arg(&opt_file),
    )
    .context("wasm-opt failed")?;

    fs::rename(&opt_file, &wasm_file).context("replacing wasm with optimized version")?;

    let size_after = fs::metadata(&wasm_file)?.len();
    let reduction = 100.0 * (1.0 - size_after as f64 / size_before as f64);
    println!(
        "  After: {:.1} KB ({reduction:.1}% reduction)",
        size_after as f64 / 1024.0
    );

    Ok(())
}

/// Merge nodejs target into bundler package with proper package.json exports.
pub fn merge_packages() -> Result<()> {
    merge_at(&pkg_dir()?, &pkg_node_dir()?)
}

/// Core merge logic, parameterised on directories for testability.
fn merge_at(pkg: &Path, pkg_node: &Path) -> Result<()> {
    println!("\nMerging dual-target packages...");

    if !pkg_node.exists() {
        bail!(
            "nodejs build output not found: {}\n  \
             Run `build_both_targets` first.",
            pkg_node.display()
        );
    }

    // Copy nodejs entry point, renamed to .cjs so Node treats it as CommonJS
    // even when package.json has "type": "module" (set by bundler target).
    let node_src = pkg_node.join("brepkit_wasm.js");
    let node_dst = pkg.join("brepkit_wasm_node.cjs");
    fs::copy(&node_src, &node_dst).with_context(|| {
        format!(
            "copying nodejs entry: {} -> {}",
            node_src.display(),
            node_dst.display()
        )
    })?;

    // Patch package.json
    let pkg_json_path = pkg.join("package.json");
    let raw = fs::read_to_string(&pkg_json_path).context("reading pkg/package.json")?;
    let mut pkg_json: serde_json::Value =
        serde_json::from_str(&raw).context("parsing pkg/package.json")?;

    patch_package_json(&mut pkg_json)?;

    let output = serde_json::to_string_pretty(&pkg_json).context("serializing package.json")?;
    fs::write(&pkg_json_path, format!("{output}\n")).context("writing pkg/package.json")?;

    println!("  Merged package.json with dual-target exports");

    // Clean up pkg-node (no longer needed)
    if pkg_node.exists() {
        fs::remove_dir_all(pkg_node).context("removing pkg-node")?;
    }

    Ok(())
}

/// Apply dual-target fields to the wasm-pack-generated package.json.
fn patch_package_json(pkg_json: &mut serde_json::Value) -> Result<()> {
    let obj = pkg_json
        .as_object_mut()
        .context("package.json is not an object")?;

    obj.insert("name".into(), serde_json::json!("brepkit-wasm"));
    obj.insert("main".into(), serde_json::json!("brepkit_wasm_node.cjs"));
    obj.insert("module".into(), serde_json::json!("brepkit_wasm.js"));

    obj.insert(
        "exports".into(),
        serde_json::json!({
            ".": {
                "node": "./brepkit_wasm_node.cjs",
                "import": "./brepkit_wasm.js",
                "default": "./brepkit_wasm.js"
            },
            "./brepkit_wasm_bg.wasm": "./brepkit_wasm_bg.wasm"
        }),
    );

    // Ensure files array includes the node entry
    let files = obj
        .entry("files")
        .or_insert_with(|| serde_json::json!([]))
        .as_array_mut()
        .context("package.json files is not an array")?;

    let node_entry = serde_json::json!("brepkit_wasm_node.cjs");
    if !files.contains(&node_entry) {
        files.push(node_entry);
    }

    Ok(())
}

/// Validate the output package meets all quality criteria.
pub fn validate_output() -> Result<()> {
    validate_at(&pkg_dir()?)
}

/// Core validation logic, parameterised on the package directory for testability.
fn validate_at(pkg: &Path) -> Result<()> {
    println!("\nValidating output...");

    let mut errors = Vec::new();

    // 1. Required files exist
    let required_files = [
        "brepkit_wasm_bg.wasm",
        "brepkit_wasm.js",
        "brepkit_wasm_node.cjs",
        "brepkit_wasm.d.ts",
        "package.json",
    ];
    for file in &required_files {
        let path = pkg.join(file);
        if path.exists() {
            println!("  ok {file}");
        } else {
            errors.push(format!("missing required file: {file}"));
        }
    }

    // 2. WASM binary size
    let wasm_path = pkg.join("brepkit_wasm_bg.wasm");
    if wasm_path.exists() {
        let size = fs::metadata(&wasm_path)?.len();
        if size < MIN_WASM_SIZE {
            errors.push(format!(
                ".wasm too small: {size} bytes (min {MIN_WASM_SIZE})"
            ));
        } else if size > MAX_WASM_SIZE {
            errors.push(format!(
                ".wasm too large: {size} bytes (max {MAX_WASM_SIZE})"
            ));
        } else {
            println!("  ok .wasm size: {:.1} KB", size as f64 / 1024.0);
        }
    }

    // 3. Type completeness
    let dts_path = pkg.join("brepkit_wasm.d.ts");
    if dts_path.exists() {
        let dts = fs::read_to_string(&dts_path)?;
        if !dts.contains("export class BrepKernel") {
            errors.push("d.ts missing 'export class BrepKernel'".into());
        }
        let method_count = count_dts_methods(&dts);
        if method_count < MIN_METHOD_COUNT {
            errors.push(format!(
                "d.ts has only {method_count} methods (expected >= {MIN_METHOD_COUNT})"
            ));
        } else {
            println!("  ok d.ts methods: {method_count}");
        }
    }

    // 4. package.json checks
    let pkg_json_path = pkg.join("package.json");
    if pkg_json_path.exists() {
        let pkg_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&pkg_json_path)?)?;
        validate_package_json(&pkg_json, &mut errors);
    }

    if errors.is_empty() {
        println!("\n  All validation checks passed");
        Ok(())
    } else {
        let msg = errors
            .iter()
            .map(|e| format!("  FAIL {e}"))
            .collect::<Vec<_>>()
            .join("\n");
        bail!("Validation failed:\n{msg}");
    }
}

/// Count method declarations in a wasm-bindgen .d.ts file.
/// Matches lines like `  methodName(...): ReturnType;` — any indented line
/// starting with a lowercase identifier followed by `(`.
fn count_dts_methods(dts: &str) -> usize {
    dts.lines()
        .filter(|l| {
            let trimmed = l.trim();
            trimmed
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_lowercase())
                && trimmed.contains('(')
        })
        .count()
}

/// Validate package.json fields, pushing errors into the collector.
fn validate_package_json(pkg_json: &serde_json::Value, errors: &mut Vec<String>) {
    let get_str = |key: &str| pkg_json.get(key).and_then(|v| v.as_str()).unwrap_or("");

    let name = get_str("name");
    if name != "brepkit-wasm" {
        errors.push(format!("package.json name is '{name}', expected 'brepkit-wasm'"));
    } else {
        println!("  ok name: {name}");
    }

    let main = get_str("main");
    if main != "brepkit_wasm_node.cjs" {
        errors.push(format!(
            "package.json main is '{main}', expected 'brepkit_wasm_node.cjs'"
        ));
    } else {
        println!("  ok main: {main}");
    }

    if let Some(exports) = pkg_json.get("exports") {
        if let Some(dot) = exports.get(".") {
            for key in &["node", "import", "default"] {
                if dot.get(key).is_none() {
                    errors.push(format!("exports[\".\"] missing key: {key}"));
                }
            }
            println!("  ok exports[\".\"] has node/import/default");
        } else {
            errors.push("exports missing \".\" entry".into());
        }
    } else {
        errors.push("package.json missing exports".into());
    }

    if let Some(files) = pkg_json.get("files").and_then(|v| v.as_array()) {
        let has_node = files
            .iter()
            .any(|v| v.as_str() == Some("brepkit_wasm_node.cjs"));
        if !has_node {
            errors.push("files array missing 'brepkit_wasm_node.cjs'".into());
        } else {
            println!("  ok files includes brepkit_wasm_node.cjs");
        }
    } else {
        errors.push("package.json missing files array".into());
    }
}

/// Run the Node.js smoke test.
pub fn run_smoke_test() -> Result<()> {
    println!("\nRunning smoke test...");

    let script = project_root()?.join("scripts/test-wasm-smoke.mjs");
    if !script.exists() {
        bail!("Smoke test script not found: {}", script.display());
    }

    run_cmd(Command::new("node").arg(&script)).context("smoke test failed")?;

    println!("  Smoke test passed");
    Ok(())
}

/// Publish the WASM package to npm.
pub fn publish(dry_run: bool) -> Result<()> {
    let pkg = pkg_dir()?;

    let tag_name = std::env::var("TAG_NAME")
        .context("TAG_NAME env var not set — required for publish")?;
    let tag_version = tag_name.strip_prefix('v').unwrap_or(&tag_name);

    let pkg_json: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(pkg.join("package.json"))
            .context("reading package.json for version check")?,
    )?;
    let pkg_version = pkg_json
        .get("version")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if pkg_version != tag_version {
        bail!("Version mismatch: package.json={pkg_version}, tag={tag_version}");
    }

    println!("\nPublishing brepkit-wasm@{pkg_version}...");

    let mut cmd = Command::new("npm");
    cmd.args(["publish", "--provenance", "--access", "public"]);
    if dry_run {
        cmd.arg("--dry-run");
    }
    cmd.current_dir(&pkg);

    run_cmd(&mut cmd).context("npm publish failed")?;

    if dry_run {
        println!("  Dry run complete (nothing published)");
    } else {
        println!("  Published brepkit-wasm@{pkg_version}");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use serde_json::json;

    // -- patch_package_json tests -----------------------------------------

    #[test]
    fn patch_sets_all_required_fields() {
        let mut pkg = json!({
            "name": "wasm-pack-default",
            "version": "0.5.3",
            "files": ["brepkit_wasm_bg.wasm", "brepkit_wasm.js", "brepkit_wasm.d.ts"],
            "module": "brepkit_wasm.js",
            "types": "brepkit_wasm.d.ts",
            "sideEffects": ["./snippets/*"]
        });

        patch_package_json(&mut pkg).unwrap();

        assert_eq!(pkg["name"], "brepkit-wasm");
        assert_eq!(pkg["main"], "brepkit_wasm_node.cjs");
        assert_eq!(pkg["module"], "brepkit_wasm.js");
        assert_eq!(pkg["exports"]["."]["node"], "./brepkit_wasm_node.cjs");
        assert_eq!(pkg["exports"]["."]["import"], "./brepkit_wasm.js");
        assert_eq!(pkg["exports"]["."]["default"], "./brepkit_wasm.js");

        let files = pkg["files"].as_array().unwrap();
        assert!(files.contains(&json!("brepkit_wasm_node.cjs")));
        // Original files preserved
        assert!(files.contains(&json!("brepkit_wasm_bg.wasm")));
    }

    #[test]
    fn patch_does_not_duplicate_node_entry() {
        let mut pkg = json!({
            "files": ["brepkit_wasm_node.cjs", "other.js"]
        });

        patch_package_json(&mut pkg).unwrap();

        assert_eq!(pkg["files"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn patch_creates_files_array_if_missing() {
        let mut pkg = json!({ "name": "test" });

        patch_package_json(&mut pkg).unwrap();

        let files = pkg["files"].as_array().unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], "brepkit_wasm_node.cjs");
    }

    // -- validate_package_json tests --------------------------------------

    #[test]
    fn validate_detects_wrong_name() {
        let pkg = json!({
            "name": "wrong-name",
            "main": "brepkit_wasm_node.cjs",
            "exports": { ".": { "node": "x", "import": "x", "default": "x" } },
            "files": ["brepkit_wasm_node.cjs"]
        });
        let mut errors = Vec::new();
        validate_package_json(&pkg, &mut errors);

        assert!(errors.iter().any(|e| e.contains("wrong-name")));
    }

    #[test]
    fn validate_detects_missing_exports() {
        let pkg = json!({
            "name": "brepkit-wasm",
            "main": "brepkit_wasm_node.cjs",
            "files": ["brepkit_wasm_node.cjs"]
        });
        let mut errors = Vec::new();
        validate_package_json(&pkg, &mut errors);

        assert!(errors.iter().any(|e| e.contains("missing exports")));
    }

    #[test]
    fn validate_passes_correct_json() {
        let mut pkg = json!({
            "name": "wasm-pack-default",
            "version": "0.5.3",
            "files": ["brepkit_wasm_bg.wasm"]
        });
        patch_package_json(&mut pkg).unwrap();

        let mut errors = Vec::new();
        validate_package_json(&pkg, &mut errors);

        assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    }

    // -- count_dts_methods tests ------------------------------------------

    #[test]
    fn count_methods_in_synthetic_dts() {
        let mut dts = String::from("export class BrepKernel {\n");
        for i in 0..200 {
            dts.push_str(&format!("  method{i}(): void;\n"));
        }
        dts.push_str("}\n");

        assert_eq!(count_dts_methods(&dts), 200);
    }

    #[test]
    fn count_methods_ignores_non_method_lines() {
        let dts = "\
export class BrepKernel {
  free(): void;
  /**
   * Some documentation
   */
  makeBox(w: number, h: number, d: number): number;
  readonly positions: Float64Array;
}
";
        // free() and makeBox() are methods; readonly and doc comments are not
        assert_eq!(count_dts_methods(dts), 2);
    }

    // -- merge_at integration test ----------------------------------------

    #[test]
    fn merge_at_copies_and_patches_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let pkg = dir.path().join("pkg");
        let pkg_node = dir.path().join("pkg-node");
        fs::create_dir_all(&pkg).unwrap();
        fs::create_dir_all(&pkg_node).unwrap();

        // Create mock package.json (as wasm-pack would generate)
        let initial = json!({
            "name": "brepkit-wasm",
            "version": "0.5.3",
            "files": ["brepkit_wasm_bg.wasm", "brepkit_wasm.js", "brepkit_wasm.d.ts"],
            "module": "brepkit_wasm.js"
        });
        fs::write(
            pkg.join("package.json"),
            serde_json::to_string_pretty(&initial).unwrap(),
        )
        .unwrap();

        // Create mock nodejs entry
        fs::write(pkg_node.join("brepkit_wasm.js"), "// node CJS entry").unwrap();

        merge_at(&pkg, &pkg_node).unwrap();

        // .cjs file was created
        assert!(pkg.join("brepkit_wasm_node.cjs").exists());
        let content = fs::read_to_string(pkg.join("brepkit_wasm_node.cjs")).unwrap();
        assert_eq!(content, "// node CJS entry");

        // pkg-node was cleaned up
        assert!(!pkg_node.exists());

        // package.json was patched
        let result: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(pkg.join("package.json")).unwrap()).unwrap();
        assert_eq!(result["name"], "brepkit-wasm");
        assert_eq!(result["main"], "brepkit_wasm_node.cjs");
        assert_eq!(result["exports"]["."]["node"], "./brepkit_wasm_node.cjs");
    }

    #[test]
    fn merge_at_fails_if_pkg_node_missing() {
        let dir = tempfile::tempdir().unwrap();
        let pkg = dir.path().join("pkg");
        let pkg_node = dir.path().join("pkg-node");
        fs::create_dir_all(&pkg).unwrap();
        // pkg_node intentionally not created

        let result = merge_at(&pkg, &pkg_node);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("nodejs build output not found"), "got: {err}");
    }
}
