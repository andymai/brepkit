mod wasm;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask", about = "brepkit build automation")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build WASM package with dual targets, merge, and validate.
    WasmBuild {
        /// Enable SIMD optimizations (wasm simd128).
        #[arg(long)]
        simd: bool,

        /// Skip wasm-opt optimization pass.
        #[arg(long)]
        skip_opt: bool,
    },

    /// Build, validate, and publish WASM package to npm.
    WasmPublish {
        /// Run npm publish with --dry-run.
        #[arg(long)]
        dry_run: bool,

        /// Enable SIMD optimizations.
        #[arg(long)]
        simd: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::WasmBuild { simd, skip_opt } => {
            wasm::check_tools()?;
            wasm::build_both_targets(simd)?;
            if !skip_opt {
                wasm::run_wasm_opt()?;
            }
            wasm::merge_packages()?;
            wasm::validate_output()?;
            println!("\n✅ WASM build complete. Run smoke test with:");
            println!("   node scripts/test-wasm-smoke.mjs");
        }
        Command::WasmPublish { dry_run, simd } => {
            wasm::check_tools()?;
            wasm::build_both_targets(simd)?;
            wasm::run_wasm_opt()?;
            wasm::merge_packages()?;
            wasm::validate_output()?;
            wasm::run_smoke_test()?;
            wasm::publish(dry_run)?;
        }
    }

    Ok(())
}
