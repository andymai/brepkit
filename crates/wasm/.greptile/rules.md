# wasm Crate Review Rules

## Binding Patterns

Flag public `#[wasm_bindgen]` methods that:
- Missing `#[wasm_bindgen(js_name = "camelCase")]` attribute
- Not validating inputs with helpers from `error.rs` (`validate_positive`, `validate_finite`)
- Not mapping errors to `JsValue::from_str(&e.to_string())`
- Returning entity IDs as anything other than `f64`

See `CLAUDE.md` "Recipe 4: Add a new WASM binding".

## Multiple impl Blocks

Flag attempts to merge `#[wasm_bindgen] impl BrepKernel` blocks — multiple blocks
are required by wasm-bindgen.

## Batch Dispatch

Flag new operations that are not added to `dispatch_op` in `kernel.rs`.
