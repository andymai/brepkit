//! STEP file writer (not yet implemented).

/// Write topology to STEP format.
///
/// # Errors
///
/// Currently always returns [`IoError::ParseError`](crate::IoError::ParseError)
/// because the STEP writer is not yet implemented.
pub fn write_step() -> Result<String, crate::IoError> {
    Err(crate::IoError::ParseError {
        reason: "STEP writer not yet implemented".to_string(),
    })
}
