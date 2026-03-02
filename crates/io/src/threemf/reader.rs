//! 3MF file reader (not yet implemented).

/// Read a 3MF file and produce topology.
///
/// # Errors
///
/// Currently always returns [`IoError::ParseError`](crate::IoError::ParseError)
/// because the 3MF reader is not yet implemented.
pub fn read_threemf(_input: &[u8]) -> Result<(), crate::IoError> {
    Err(crate::IoError::ParseError {
        reason: "3MF reader not yet implemented".to_string(),
    })
}
