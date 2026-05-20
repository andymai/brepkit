//! Bridge brepkit's `log` crate calls to JS `console.{log, warn, error}`.
//!
//! Without this, all `log::warn!` / `log::info!` etc. throughout the Rust
//! code are silently dropped under wasm-pack — useful diagnostic output
//! never reaches the consumer's stderr. The bridge sets up a logger that
//! routes by level:
//!
//! - `error` → `console.error`
//! - `warn`  → `console.warn`
//! - `info` / `debug` / `trace` → `console.log` (with level prefix)
//!
//! Initialize once via `BrepKernel.setLogLevel(level)` from JS. Calling
//! again with a different level updates the runtime filter without
//! reinstalling the logger. Default level (if `setLogLevel` is never
//! called) is **off** — zero overhead, no console noise.
//!
//! Works under both `wasm-pack --target nodejs` and `--target web` since
//! `console.*` is a universal JS global.

use core::sync::atomic::{AtomicUsize, Ordering};

use log::{Level, LevelFilter, Log, Metadata, Record};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn console_log(s: &str);
    #[wasm_bindgen(js_namespace = console, js_name = warn)]
    fn console_warn(s: &str);
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(s: &str);
}

/// Runtime level filter. Defaults to `off` (= 0). `set_log_level` writes here
/// rather than mutating `log::max_level` so the level can change after the
/// global logger is installed.
static LEVEL: AtomicUsize = AtomicUsize::new(LevelFilter::Off as usize);

struct ConsoleLogger;

impl Log for ConsoleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        (metadata.level() as usize) <= LEVEL.load(Ordering::Relaxed)
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let msg = format!("[brepkit:{}] {}", record.target(), record.args());
        match record.level() {
            Level::Error => console_error(&msg),
            Level::Warn => console_warn(&msg),
            _ => console_log(&msg),
        }
    }

    fn flush(&self) {}
}

static LOGGER: ConsoleLogger = ConsoleLogger;

/// Install the global logger if it hasn't been set yet, and update the
/// runtime level filter.
///
/// Idempotent — safe to call multiple times. `log::set_logger` succeeds at
/// most once globally; the second call is a no-op but the level update
/// still takes effect.
fn set_log_level(level: LevelFilter) {
    // First call installs the logger; subsequent calls just update the filter.
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(level);
    LEVEL.store(level as usize, Ordering::Relaxed);
}

/// Parse a level string. Accepts `"off"`, `"error"`, `"warn"`, `"info"`,
/// `"debug"`, `"trace"` (case-insensitive). Unknown values default to
/// `Off` so a typo never produces a flood of output.
fn parse_level(s: &str) -> LevelFilter {
    match s.to_ascii_lowercase().as_str() {
        "error" => LevelFilter::Error,
        "warn" => LevelFilter::Warn,
        "info" => LevelFilter::Info,
        "debug" => LevelFilter::Debug,
        "trace" => LevelFilter::Trace,
        _ => LevelFilter::Off,
    }
}

/// Route brepkit's Rust `log::*` calls to JavaScript `console.{log, warn,
/// error}`. Without this every `log::warn!` in the engine is silently
/// dropped under wasm-pack.
///
/// `level` is one of `"off"`, `"error"`, `"warn"`, `"info"`, `"debug"`,
/// `"trace"` (case-insensitive). Default is `"off"` (no log calls reach
/// the console). Idempotent — call as often as you like to change the
/// filter.
///
/// Recommended: call once at app start with `"warn"` to surface boolean /
/// validation diagnostics without flooding the console.
#[wasm_bindgen(js_name = "setLogLevel")]
pub fn js_set_log_level(level: &str) {
    set_log_level(parse_level(level));
}
