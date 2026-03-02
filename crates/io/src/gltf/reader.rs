//! glTF 2.0 binary (.glb) reader.
//!
//! Reads mesh geometry (positions, normals, triangle indices) from GLB files.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::tessellate::TriangleMesh;

/// Read a GLB (glTF binary) file and return a triangle mesh.
///
/// Extracts vertex positions, normals, and triangle indices from
/// the first mesh primitive in the file.
///
/// # Errors
///
/// Returns an error if the file is malformed or uses unsupported features.
pub fn read_glb(data: &[u8]) -> Result<TriangleMesh, crate::IoError> {
    if data.len() < 12 {
        return Err(crate::IoError::ParseError {
            reason: "GLB too short for header".into(),
        });
    }

    // Validate magic
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != 0x4654_6C67 {
        return Err(crate::IoError::ParseError {
            reason: "not a GLB file (invalid magic)".into(),
        });
    }

    // Version
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version != 2 {
        return Err(crate::IoError::ParseError {
            reason: format!("unsupported glTF version {version}"),
        });
    }

    // Parse chunks
    let mut offset = 12;
    let mut json_data: Option<&[u8]> = None;
    let mut bin_data: Option<&[u8]> = None;

    while offset + 8 <= data.len() {
        let chunk_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        let chunk_type = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;

        if offset + chunk_len > data.len() {
            break;
        }

        match chunk_type {
            0x4E4F_534A => json_data = Some(&data[offset..offset + chunk_len]), // JSON
            0x004E_4942 => bin_data = Some(&data[offset..offset + chunk_len]),  // BIN
            _ => {}                                                             // skip unknown
        }

        offset += chunk_len;
    }

    let json_bytes = json_data.ok_or_else(|| crate::IoError::ParseError {
        reason: "GLB missing JSON chunk".into(),
    })?;
    let bin = bin_data.ok_or_else(|| crate::IoError::ParseError {
        reason: "GLB missing BIN chunk".into(),
    })?;

    // Parse JSON to extract accessor info
    let json_str = std::str::from_utf8(json_bytes).map_err(|_| crate::IoError::ParseError {
        reason: "GLB JSON is not valid UTF-8".into(),
    })?;

    // Simple JSON parsing for the fields we need
    let accessors = parse_accessors(json_str);
    let buffer_views = parse_buffer_views(json_str);

    if accessors.len() < 3 || buffer_views.len() < 3 {
        return Err(crate::IoError::ParseError {
            reason: "GLB needs at least 3 accessors and 3 buffer views".into(),
        });
    }

    // Accessor 0 = positions, 1 = normals, 2 = indices (our convention)
    let pos_view = &buffer_views[accessors[0].buffer_view];
    let norm_view = &buffer_views[accessors[1].buffer_view];
    let idx_view = &buffer_views[accessors[2].buffer_view];

    // Read positions (VEC3, float)
    let mut positions = Vec::with_capacity(accessors[0].count);
    let pos_data = &bin[pos_view.byte_offset..pos_view.byte_offset + pos_view.byte_length];
    for chunk in pos_data.chunks_exact(12) {
        let x = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let y = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let z = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
        positions.push(Point3::new(f64::from(x), f64::from(y), f64::from(z)));
    }

    // Read normals (VEC3, float)
    let mut normals = Vec::with_capacity(accessors[1].count);
    let norm_data = &bin[norm_view.byte_offset..norm_view.byte_offset + norm_view.byte_length];
    for chunk in norm_data.chunks_exact(12) {
        let x = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let y = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        let z = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
        normals.push(Vec3::new(f64::from(x), f64::from(y), f64::from(z)));
    }

    // Read indices (SCALAR, uint32)
    let mut indices = Vec::with_capacity(accessors[2].count);
    let idx_data = &bin[idx_view.byte_offset..idx_view.byte_offset + idx_view.byte_length];
    for chunk in idx_data.chunks_exact(4) {
        let idx = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        indices.push(idx);
    }

    normals.resize(positions.len(), Vec3::new(0.0, 0.0, 1.0));

    Ok(TriangleMesh {
        positions,
        normals,
        indices,
    })
}

struct AccessorInfo {
    buffer_view: usize,
    count: usize,
}

struct BufferViewInfo {
    byte_offset: usize,
    byte_length: usize,
}

/// Minimal JSON parsing for accessor array.
fn parse_accessors(json: &str) -> Vec<AccessorInfo> {
    let mut accessors = Vec::new();

    // Find "accessors" array
    let Some(start) = json.find("\"accessors\"") else {
        return accessors;
    };
    let Some(arr_start) = json[start..].find('[') else {
        return accessors;
    };
    let arr_offset = start + arr_start;

    let arr_str = extract_json_array(json, arr_offset);

    for obj in arr_str.split('}') {
        let bv = extract_int(obj, "bufferView");
        let count = extract_int(obj, "count");
        if let (Some(bv), Some(count)) = (bv, count) {
            accessors.push(AccessorInfo {
                buffer_view: bv,
                count,
            });
        }
    }

    accessors
}

/// Minimal JSON parsing for buffer views array.
fn parse_buffer_views(json: &str) -> Vec<BufferViewInfo> {
    let mut views = Vec::new();

    let Some(start) = json.find("\"bufferViews\"") else {
        return views;
    };
    let Some(arr_start) = json[start..].find('[') else {
        return views;
    };
    let arr_offset = start + arr_start;

    let arr_str = extract_json_array(json, arr_offset);

    for obj in arr_str.split('}') {
        let offset = extract_int(obj, "byteOffset").unwrap_or(0);
        let length = extract_int(obj, "byteLength");
        if let Some(length) = length {
            views.push(BufferViewInfo {
                byte_offset: offset,
                byte_length: length,
            });
        }
    }

    views
}

/// Extract the content of a JSON array starting at `arr_offset` (the `[` char).
fn extract_json_array(json: &str, arr_offset: usize) -> &str {
    let mut depth = 0;
    let mut arr_end = arr_offset;
    for (i, ch) in json[arr_offset..].chars().enumerate() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    arr_end = arr_offset + i;
                    break;
                }
            }
            _ => {}
        }
    }
    &json[arr_offset + 1..arr_end]
}

/// Extract an integer value for a given key from a JSON-like string.
fn extract_int(text: &str, key: &str) -> Option<usize> {
    let pattern = format!("\"{key}\"");
    let pos = text.find(&pattern)?;
    let after = &text[pos + pattern.len()..];
    let colon_pos = after.find(':')?;
    let value_str = after[colon_pos + 1..].trim();

    // Read digits
    let digits: String = value_str.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_glb() {
        let mut topo = brepkit_topology::Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let glb = crate::gltf::write_glb(&topo, &[solid], 0.1).unwrap();
        let mesh = read_glb(&glb).unwrap();

        assert!(!mesh.positions.is_empty(), "should have vertices");
        assert!(!mesh.indices.is_empty(), "should have indices");
        assert_eq!(mesh.indices.len() % 3, 0, "should be triangles");
    }

    #[test]
    fn invalid_magic() {
        let data = vec![0u8; 20];
        assert!(read_glb(&data).is_err());
    }

    #[test]
    fn too_short() {
        let data = vec![0u8; 4];
        assert!(read_glb(&data).is_err());
    }

    #[test]
    fn extract_int_works() {
        assert_eq!(extract_int(r#""count":42"#, "count"), Some(42));
        assert_eq!(extract_int(r#""byteOffset":0"#, "byteOffset"), Some(0));
        assert_eq!(extract_int(r"no match", "count"), None);
    }
}
