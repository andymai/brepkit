//! Bridge brepkit's `log` crate calls to JS `console.{log, warn, error}`.
//!
//! Without this, all `log::warn!` / `log::info!` etc. throughout the Rust
//! code are silently dropped under wasm-pack — useful diagnostic output
//! never reaches the consumer's stderr. The bridge sets up a logger that
//! routes by level:
//!
//! - `error` → `console.error`
//! - `warn`  → `console.warn`
//! - `info` / `debug` / `trace` → `console.log`
//!
//! Initialize once via the standalone `setLogLevel(level)` export from JS.
//! Calling again with a different level updates the runtime filter without
//! reinstalling the logger. Default level (if `setLogLevel` is never
//! called) is **off** — zero overhead, no console noise.
//!
//! Works under both `wasm-pack --target nodejs` and `--target web` since
//! `console.*` is a universal JS global.

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

struct ConsoleLogger;

impl Log for ConsoleLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= log::max_level()
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
/// runtime level filter via `log::set_max_level`.
///
/// Idempotent — safe to call multiple times. `log::set_logger` succeeds at
/// most once globally; the second call is a no-op but the level update
/// still takes effect because `log::set_max_level` is independent of the
/// logger-install handshake.
fn set_log_level(level: LevelFilter) {
    // First call installs the logger; subsequent calls just update the filter.
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(level);
}

/// Parse a level string. Accepts `"off"`, `"error"`, `"warn"`, `"info"`,
/// `"debug"`, `"trace"` (case-insensitive). Returns `None` for unknown
/// values so the caller can surface a clear error to JS rather than
/// silently swallowing a typo.
fn parse_level(s: &str) -> Option<LevelFilter> {
    Some(match s.to_ascii_lowercase().as_str() {
        "off" => LevelFilter::Off,
        "error" => LevelFilter::Error,
        "warn" => LevelFilter::Warn,
        "info" => LevelFilter::Info,
        "debug" => LevelFilter::Debug,
        "trace" => LevelFilter::Trace,
        _ => return None,
    })
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
/// Throws a JS Error if `level` is not one of the recognised values so a
/// typo surfaces immediately instead of producing the same observable
/// behaviour as never calling `setLogLevel` at all.
///
/// Recommended: call once at app start with `"warn"` to surface boolean /
/// validation diagnostics without flooding the console.
#[wasm_bindgen(js_name = "setLogLevel")]
pub fn js_set_log_level(level: &str) -> Result<(), JsError> {
    let parsed = parse_level(level).ok_or_else(|| {
        JsError::new(&format!(
            "setLogLevel: unknown level {level:?}; expected one of \
             \"off\", \"error\", \"warn\", \"info\", \"debug\", \"trace\""
        ))
    })?;
    set_log_level(parsed);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_level_known_values() {
        assert_eq!(parse_level("off"), Some(LevelFilter::Off));
        assert_eq!(parse_level("error"), Some(LevelFilter::Error));
        assert_eq!(parse_level("warn"), Some(LevelFilter::Warn));
        assert_eq!(parse_level("info"), Some(LevelFilter::Info));
        assert_eq!(parse_level("debug"), Some(LevelFilter::Debug));
        assert_eq!(parse_level("trace"), Some(LevelFilter::Trace));
    }

    #[test]
    fn parse_level_is_case_insensitive() {
        assert_eq!(parse_level("WARN"), Some(LevelFilter::Warn));
        assert_eq!(parse_level("Warn"), Some(LevelFilter::Warn));
        assert_eq!(parse_level("Trace"), Some(LevelFilter::Trace));
    }

    #[test]
    fn parse_level_unknown_returns_none() {
        // Typos and whitespace are not silently swallowed — the caller
        // surfaces the error to JS.
        assert_eq!(parse_level("warnn"), None);
        assert_eq!(parse_level(""), None);
        assert_eq!(parse_level(" warn"), None);
        assert_eq!(parse_level("verbose"), None);
    }
}
