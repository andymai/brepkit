//! STEP file reader (not yet implemented).

/// Read a STEP file and produce topology.
///
/// # Errors
///
/// Currently always returns [`IoError::ParseError`](crate::IoError::ParseError)
/// because the STEP reader is not yet implemented.
pub fn read_step(_input: &str) -> Result<(), crate::IoError> {
    Err(crate::IoError::ParseError {
        reason: "STEP reader not yet implemented".to_string(),
    })
}
