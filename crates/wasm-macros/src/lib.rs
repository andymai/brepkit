//! Proc macros for brepkit-wasm binding generation.
//!
//! # `#[wasm_binding]`
//!
//! Wraps a `BrepKernel` method with panic safety. The method must return
//! `Result<T, WasmError>` where `T` is any wasm-bindgen-compatible type.
//!
//! - **Standard** (default): `catch_unwind` → set `self.poisoned = true` on panic
//! - **Recoverable**: `catch_unwind` → restore `self.topo` from snapshot on panic
//!
//! ```ignore
//! #[wasm_binding(js_name = "makeBox")]
//! pub fn make_box_solid(&mut self, dx: f64, dy: f64, dz: f64) -> Result<u32, WasmError> { ... }
//!
//! #[wasm_binding(js_name = "fuse", recoverable)]
//! pub fn fuse(&mut self, a: u32, b: u32) -> Result<u32, WasmError> { ... }
//!
//! #[wasm_binding(js_name = "volume")]
//! pub fn volume(&self, solid: u32, deflection: f64) -> Result<f64, WasmError> { ... }
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Ident, ItemFn, LitStr, ReturnType, Token, Type, parse_macro_input};

/// Parsed attributes for `#[wasm_binding(...)]`.
struct WasmBindingArgs {
    js_name: Option<String>,
    recoverable: bool,
}

impl Parse for WasmBindingArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut js_name = None;
        let mut recoverable = false;

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "js_name" => {
                    let _: Token![=] = input.parse()?;
                    let lit: LitStr = input.parse()?;
                    js_name = Some(lit.value());
                }
                "recoverable" => {
                    recoverable = true;
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("unknown attribute: {other}"),
                    ));
                }
            }
            if !input.is_empty() {
                let _: Token![,] = input.parse()?;
                if input.is_empty() {
                    break;
                }
            }
        }

        Ok(Self {
            js_name,
            recoverable,
        })
    }
}

/// Extract the `T` from `Result<T, SomeError>` in a return type.
///
/// Returns `None` if the return type is not `Result<T, _>`.
fn extract_result_ok_type(ret: &ReturnType) -> Option<&Type> {
    let ReturnType::Type(_, ty) = ret else {
        return None;
    };
    let Type::Path(type_path) = ty.as_ref() else {
        return None;
    };
    let last_seg = type_path.path.segments.last()?;
    if last_seg.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &last_seg.arguments else {
        return None;
    };
    let syn::GenericArgument::Type(ok_ty) = args.args.first()? else {
        return None;
    };
    Some(ok_ty)
}

/// Attribute macro for panic-safe WASM bindings.
///
/// Generates a public `#[wasm_bindgen]` wrapper that:
/// 1. Checks the `poisoned` flag
/// 2. Wraps the body in `catch_unwind`
/// 3. On panic: poisons (standard) or restores topology (recoverable)
///
/// The original method body is moved to a private `__<name>_impl` method.
/// The return type is preserved from the original signature — the wrapper
/// converts `Result<T, WasmError>` to `Result<T, JsError>`.
#[proc_macro_attribute]
pub fn wasm_binding(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as WasmBindingArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let fn_name = &sig.ident;
    let body = &input_fn.block;
    let attrs: Vec<_> = input_fn.attrs.iter().collect();

    // Build the impl method name
    let impl_name = Ident::new(&format!("__{fn_name}_impl"), fn_name.span());

    // Extract parameters (skip &self/&mut self)
    let params = &sig.inputs;
    let param_names: Vec<_> = sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let syn::FnArg::Typed(pat) = arg {
                Some(&pat.pat)
            } else {
                None
            }
        })
        .collect();

    // Extract the Ok type from Result<T, WasmError>
    let ok_type = extract_result_ok_type(&sig.output);
    let ok_type = match ok_type {
        Some(ty) => ty.clone(),
        None => {
            return syn::Error::new_spanned(
                &sig.output,
                "#[wasm_binding] requires return type Result<T, WasmError>",
            )
            .to_compile_error()
            .into();
        }
    };

    // Build wasm_bindgen attribute
    let wasm_bindgen_attr = if let Some(ref name) = args.js_name {
        quote! { #[wasm_bindgen(js_name = #name)] }
    } else {
        quote! { #[wasm_bindgen] }
    };

    // Determine the js_name for error messages
    let op_name = args
        .js_name
        .as_deref()
        .unwrap_or(&fn_name.to_string())
        .to_string();

    let expanded = if args.recoverable {
        // Recoverable: clone topology before call, restore on panic
        quote! {
            #wasm_bindgen_attr
            #(#attrs)*
            #vis fn #fn_name(#params) -> Result<#ok_type, JsError> {
                if self.poisoned {
                    return Err(JsError::new("Kernel poisoned after panic. Call reset()."));
                }
                let topo_snapshot = self.topo.clone();
                let assemblies_snapshot = self.assemblies.clone();
                let sketches_snapshot = self.sketches.clone();
                let checkpoints_snapshot = self.checkpoints.clone();
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.#impl_name(#(#param_names),*)
                })) {
                    Ok(inner) => inner.map_err(|e| JsError::new(&e.to_string())),
                    Err(p) => {
                        self.topo = topo_snapshot;
                        self.assemblies = assemblies_snapshot;
                        self.sketches = sketches_snapshot;
                        self.checkpoints = checkpoints_snapshot;
                        Err(JsError::new(&crate::helpers::panic_message(&p, #op_name)))
                    }
                }
            }

            fn #impl_name(#params) -> Result<#ok_type, crate::error::WasmError>
            #body
        }
    } else {
        // Standard: catch_unwind → poison on panic
        quote! {
            #wasm_bindgen_attr
            #(#attrs)*
            #vis fn #fn_name(#params) -> Result<#ok_type, JsError> {
                if self.poisoned {
                    return Err(JsError::new("Kernel poisoned after panic. Call reset()."));
                }
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    self.#impl_name(#(#param_names),*)
                })) {
                    Ok(inner) => inner.map_err(|e| JsError::new(&e.to_string())),
                    Err(p) => {
                        self.poisoned = true;
                        Err(JsError::new(&crate::helpers::panic_message(&p, #op_name)))
                    }
                }
            }

            fn #impl_name(#params) -> Result<#ok_type, crate::error::WasmError>
            #body
        }
    };

    expanded.into()
}
