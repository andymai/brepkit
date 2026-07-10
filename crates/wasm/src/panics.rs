//! Panic capture for post-mortem diagnosis from JavaScript.
//!
//! On `wasm32-unknown-unknown` the panic strategy is `abort`: a panic inside
//! any kernel method traps mid-call, so the wasm-bindgen borrow flag on the
//! `BrepKernel` object is never released and every later method call throws
//! "recursive use of an object detected which would lead to unsafe aliasing
//! in rust". `catch_unwind` cannot intercept an aborting panic, and the
//! borrow flag cannot be restored from Rust — the only recovery is creating
//! a new `BrepKernel`. What CAN be saved is the panic text: the panic hook
//! still runs before the abort, so this module records the message and
//! location in a static and exposes them through free functions that stay
//! callable after the kernel object is poisoned (they never touch its
//! borrow flag). The hook also forwards the text to `console.error` so it
//! survives JS callers that swallow the trap's `RuntimeError`.

use std::sync::{Mutex, Once};

use wasm_bindgen::prelude::*;

static LAST_PANIC: Mutex<Option<String>> = Mutex::new(None);
static INSTALL: Once = Once::new();

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn console_error(s: &str);
}

/// Install the capturing panic hook. Idempotent; chains the previous hook
/// so native test output is unchanged.
pub(crate) fn install_hook() {
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = info.to_string();
            #[cfg(target_arch = "wasm32")]
            console_error(&format!("[brepkit] panic: {msg}"));
            if let Ok(mut slot) = LAST_PANIC.lock() {
                *slot = Some(msg);
            }
            previous(info);
        }));
    });
}

/// Returns the message and source location of the most recent panic inside
/// the kernel, or `undefined` if none has occurred.
///
/// After a panic the kernel object is unusable (every method throws
/// "recursive use of an object"); this free function remains callable and
/// carries the root-cause text for the failed call.
#[wasm_bindgen(js_name = "lastPanicMessage")]
#[must_use]
pub fn last_panic_message() -> Option<String> {
    LAST_PANIC.lock().ok().and_then(|slot| slot.clone())
}

/// Clears the stored panic message so later reads reflect only new panics.
#[wasm_bindgen(js_name = "clearLastPanicMessage")]
pub fn clear_last_panic_message() {
    if let Ok(mut slot) = LAST_PANIC.lock() {
        *slot = None;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn hook_records_panic_message_and_location() {
        let _kernel = crate::kernel::BrepKernel::new();
        clear_last_panic_message();
        assert_eq!(last_panic_message(), None);

        let caught = std::panic::catch_unwind(|| panic!("panics-module-marker-7391"));
        assert!(caught.is_err());

        let msg = last_panic_message().expect("hook should have recorded the panic");
        assert!(msg.contains("panics-module-marker-7391"), "got: {msg}");
        assert!(msg.contains("panics.rs"), "location missing: {msg}");

        clear_last_panic_message();
        assert_eq!(last_panic_message(), None);
    }
}
