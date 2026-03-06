# io Crate Review Rules

## Wildcard Match Verification

When a PR adds `EdgeCurve` or `FaceSurface` variants, these io files must be
manually checked since their `_ =>` arms won't cause compiler errors:

- `step/writer.rs`
- `iges/writer.rs`

Also check the corresponding readers handle the new variant on import.
