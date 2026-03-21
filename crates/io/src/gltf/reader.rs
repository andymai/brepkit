//! glTF 2.0 binary (.glb) reader.
//!
//! Reads mesh geometry (positions, normals, triangle indices) from GLB files.
//! Supports multiple meshes/primitives, dynamic accessor indices, and both
//! uint16 (5123) and uint32 (5125) index component types.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::tessellate::TriangleMesh;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// Read a GLB (glTF binary) file and return a triangle mesh.
///
/// Extracts vertex positions, normals, and triangle indices from all
/// mesh primitives in the file, combining them into a single mesh.
///
/// # Errors
///
/// Returns an error if the file is malformed or uses unsupported features.
#[allow(clippy::too_many_lines)]
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
    let primitives = parse_mesh_primitives(json_str);

    if primitives.is_empty() {
        return Err(crate::IoError::ParseError {
            reason: "GLB has no mesh primitives".into(),
        });
    }

    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    for prim in &primitives {
        let vertex_offset = positions.len();

        // Read positions
        if let Some(pos_idx) = prim.position_accessor {
            let accessor = accessors
                .get(pos_idx)
                .ok_or_else(|| crate::IoError::ParseError {
                    reason: format!("POSITION accessor index {pos_idx} out of range"),
                })?;
            let view = buffer_views.get(accessor.buffer_view).ok_or_else(|| {
                crate::IoError::ParseError {
                    reason: format!("buffer view index {} out of range", accessor.buffer_view),
                }
            })?;
            let pos_data = safe_slice(bin, view.byte_offset, view.byte_length)?;
            for chunk in pos_data.chunks_exact(12) {
                let x = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let y = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                let z = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                positions.push(Point3::new(f64::from(x), f64::from(y), f64::from(z)));
            }
        }

        // Read normals
        if let Some(norm_idx) = prim.normal_accessor {
            if let Some(accessor) = accessors.get(norm_idx) {
                if let Some(view) = buffer_views.get(accessor.buffer_view) {
                    if let Ok(norm_data) = safe_slice(bin, view.byte_offset, view.byte_length) {
                        for chunk in norm_data.chunks_exact(12) {
                            let x = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            let y = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                            let z = f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]);
                            normals.push(Vec3::new(f64::from(x), f64::from(y), f64::from(z)));
                        }
                    }
                }
            }
        }

        // Read indices
        if let Some(idx_accessor_idx) = prim.indices_accessor {
            let accessor =
                accessors
                    .get(idx_accessor_idx)
                    .ok_or_else(|| crate::IoError::ParseError {
                        reason: format!("indices accessor index {idx_accessor_idx} out of range"),
                    })?;
            let view = buffer_views.get(accessor.buffer_view).ok_or_else(|| {
                crate::IoError::ParseError {
                    reason: format!("buffer view index {} out of range", accessor.buffer_view),
                }
            })?;
            let idx_data = safe_slice(bin, view.byte_offset, view.byte_length)?;

            #[allow(clippy::cast_possible_truncation)]
            let v_offset = vertex_offset as u32;

            match accessor.component_type {
                5123 => {
                    // uint16
                    for chunk in idx_data.chunks_exact(2) {
                        let idx = u16::from_le_bytes([chunk[0], chunk[1]]);
                        indices.push(u32::from(idx) + v_offset);
                    }
                }
                5125 => {
                    // uint32
                    for chunk in idx_data.chunks_exact(4) {
                        let idx = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        indices.push(idx + v_offset);
                    }
                }
                ct => {
                    return Err(crate::IoError::ParseError {
                        reason: format!(
                            "unsupported index component type {ct} (expected 5123=uint16 or 5125=uint32)"
                        ),
                    });
                }
            }
        }
    }

    // Pad normals to match positions if some primitives lack normals
    normals.resize(positions.len(), Vec3::new(0.0, 0.0, 1.0));

    if positions.is_empty() {
        return Err(crate::IoError::ParseError {
            reason: "GLB contains no vertex data".into(),
        });
    }

    Ok(TriangleMesh {
        positions,
        normals,
        indices,
    })
}

/// Safe sub-slice with bounds checking.
fn safe_slice(data: &[u8], offset: usize, length: usize) -> Result<&[u8], crate::IoError> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| crate::IoError::ParseError {
            reason: "buffer view offset + length overflow".into(),
        })?;
    if end > data.len() {
        return Err(crate::IoError::ParseError {
            reason: format!(
                "buffer view [{offset}..{end}] exceeds binary buffer length {}",
                data.len()
            ),
        });
    }
    Ok(&data[offset..end])
}

struct AccessorInfo {
    buffer_view: usize,
    component_type: u32,
    #[allow(dead_code)]
    count: usize,
}

struct BufferViewInfo {
    byte_offset: usize,
    byte_length: usize,
}

/// A parsed mesh primitive with accessor indices for attributes and indices.
#[allow(clippy::struct_field_names)]
struct MeshPrimitive {
    position_accessor: Option<usize>,
    normal_accessor: Option<usize>,
    indices_accessor: Option<usize>,
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

    for obj in split_json_objects(arr_str) {
        let bv = extract_int(obj, "bufferView");
        let count = extract_int(obj, "count");
        let component_type = extract_int(obj, "componentType");
        if let (Some(bv), Some(count)) = (bv, count) {
            #[allow(clippy::cast_possible_truncation)]
            accessors.push(AccessorInfo {
                buffer_view: bv,
                component_type: component_type.unwrap_or(5126) as u32,
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

    for obj in split_json_objects(arr_str) {
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

/// Parse mesh primitives from the JSON, extracting attribute accessor
/// indices (`POSITION`, `NORMAL`) and the `indices` accessor index.
///
/// Iterates all `meshes[].primitives[]`, so multi-mesh files are supported.
fn parse_mesh_primitives(json: &str) -> Vec<MeshPrimitive> {
    let mut primitives = Vec::new();

    let Some(meshes_start) = json.find("\"meshes\"") else {
        return primitives;
    };
    let Some(arr_start) = json[meshes_start..].find('[') else {
        return primitives;
    };
    let meshes_arr_offset = meshes_start + arr_start;
    let meshes_str = extract_json_array(json, meshes_arr_offset);

    // Find all "primitives" arrays within the meshes array
    let mut search_offset = 0;
    while let Some(prim_key_pos) = meshes_str[search_offset..].find("\"primitives\"") {
        let prim_key_abs = search_offset + prim_key_pos;
        if let Some(prim_arr_start) = meshes_str[prim_key_abs..].find('[') {
            let prim_arr_offset = prim_key_abs + prim_arr_start;
            let prim_arr_str = extract_json_array(meshes_str, prim_arr_offset);

            // Each primitive is a JSON object within the primitives array.
            for prim_obj in split_json_objects(prim_arr_str) {
                let pos = extract_attribute_accessor(prim_obj, "POSITION");
                let norm = extract_attribute_accessor(prim_obj, "NORMAL");
                let idx = extract_int(prim_obj, "indices");

                // Only add if at least POSITION is present
                if pos.is_some() {
                    primitives.push(MeshPrimitive {
                        position_accessor: pos,
                        normal_accessor: norm,
                        indices_accessor: idx,
                    });
                }
            }

            search_offset = prim_arr_offset + 1;
        } else {
            break;
        }
    }

    // Fallback: if no primitives found via mesh parsing, try legacy
    // accessor convention (0=positions, 1=normals, 2=indices)
    if primitives.is_empty() {
        // Check if there are accessors at all — use hardcoded indices as fallback
        if json.contains("\"accessors\"") {
            primitives.push(MeshPrimitive {
                position_accessor: Some(0),
                normal_accessor: Some(1),
                indices_accessor: Some(2),
            });
        }
    }

    primitives
}

/// Extract an accessor index for a named attribute (e.g., `"POSITION":0`).
///
/// Looks for the pattern `"ATTR_NAME":N` within a primitive object string.
fn extract_attribute_accessor(text: &str, attr_name: &str) -> Option<usize> {
    let pattern = format!("\"{attr_name}\"");
    let pos = text.find(&pattern)?;
    let after = &text[pos + pattern.len()..];
    let colon_pos = after.find(':')?;
    let value_str = after[colon_pos + 1..].trim();

    let digits: String = value_str.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
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

/// Split a JSON array's inner text into top-level objects by tracking brace depth.
fn split_json_objects(array_content: &str) -> Vec<&str> {
    let mut objects = Vec::new();
    let mut depth = 0;
    let mut obj_start = None;
    for (i, ch) in array_content.char_indices() {
        match ch {
            '{' => {
                if depth == 0 {
                    obj_start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(start) = obj_start {
                        objects.push(&array_content[start..=i]);
                    }
                    obj_start = None;
                }
            }
            _ => {}
        }
    }
    objects
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

/// Read a GLB (glTF binary) file and import it as a solid with one planar
/// face per triangle.
///
/// This is a convenience wrapper that calls [`read_glb`] followed by
/// [`import_mesh`](crate::stl::import::import_mesh). Vertices within
/// `tolerance` of each other are merged.
///
/// # Errors
///
/// Returns [`IoError`](crate::IoError) if the file is malformed or the mesh
/// cannot be converted to a valid solid.
pub fn read_glb_solid(
    topo: &mut Topology,
    data: &[u8],
    tolerance: f64,
) -> Result<SolidId, crate::IoError> {
    let mesh = read_glb(data)?;
    crate::stl::import::import_mesh(topo, &mesh, tolerance)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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

    #[test]
    fn parse_primitives_from_json() {
        let json = r#"{"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1},"indices":2}]}],"accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3"},{"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"},{"bufferView":2,"componentType":5125,"count":6,"type":"SCALAR"}]}"#;

        let prims = parse_mesh_primitives(json);
        assert_eq!(prims.len(), 1);
        assert_eq!(prims[0].position_accessor, Some(0));
        assert_eq!(prims[0].normal_accessor, Some(1));
        assert_eq!(prims[0].indices_accessor, Some(2));
    }

    #[test]
    fn parse_primitives_non_default_indices() {
        // Accessors in a different order: positions=2, normals=3, indices=4
        let json =
            r#"{"meshes":[{"primitives":[{"attributes":{"POSITION":2,"NORMAL":3},"indices":4}]}]}"#;

        let prims = parse_mesh_primitives(json);
        assert_eq!(prims.len(), 1);
        assert_eq!(prims[0].position_accessor, Some(2));
        assert_eq!(prims[0].normal_accessor, Some(3));
        assert_eq!(prims[0].indices_accessor, Some(4));
    }

    #[test]
    fn parse_multi_mesh_primitives() {
        let json = r#"{"meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1},"indices":2}]},{"primitives":[{"attributes":{"POSITION":3,"NORMAL":4},"indices":5}]}]}"#;

        let prims = parse_mesh_primitives(json);
        assert_eq!(prims.len(), 2);
        assert_eq!(prims[0].position_accessor, Some(0));
        assert_eq!(prims[1].position_accessor, Some(3));
        assert_eq!(prims[1].indices_accessor, Some(5));
    }

    #[test]
    fn parse_accessor_component_type() {
        let json = r#"{"accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3"},{"bufferView":1,"componentType":5123,"count":6,"type":"SCALAR"}]}"#;
        let accessors = parse_accessors(json);
        assert_eq!(accessors.len(), 2);
        assert_eq!(accessors[0].component_type, 5126);
        assert_eq!(accessors[1].component_type, 5123);
    }

    #[test]
    fn extract_attribute_accessor_works() {
        let text = r#""attributes":{"POSITION":0,"NORMAL":1}"#;
        assert_eq!(extract_attribute_accessor(text, "POSITION"), Some(0));
        assert_eq!(extract_attribute_accessor(text, "NORMAL"), Some(1));
        assert_eq!(extract_attribute_accessor(text, "TANGENT"), None);
    }

    #[test]
    fn read_uint16_indices() {
        // Build a minimal GLB with uint16 indices manually.
        // 1 triangle: 3 positions + 3 normals + 3 uint16 indices

        // Positions: 3 vertices (3 * 12 bytes = 36 bytes)
        let positions: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let normals: [f32; 9] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let indices: [u16; 3] = [0, 1, 2];

        let pos_bytes = 36;
        let norm_bytes = 36;
        let idx_bytes = 6;
        // Pad index buffer to 4-byte alignment
        let idx_bytes_padded = 8;
        let buf_len = pos_bytes + norm_bytes + idx_bytes_padded;

        let mut bin_buffer = Vec::with_capacity(buf_len);
        for &v in &positions {
            bin_buffer.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &normals {
            bin_buffer.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &indices {
            bin_buffer.extend_from_slice(&v.to_le_bytes());
        }
        // Pad to 4-byte alignment
        while bin_buffer.len() % 4 != 0 {
            bin_buffer.push(0);
        }

        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1}},"indices":2}}]}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}},{{"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"}},{{"bufferView":2,"componentType":5123,"count":3,"type":"SCALAR"}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{pos_bytes}}},{{"buffer":0,"byteOffset":{norm_off},"byteLength":{norm_bytes}}},{{"buffer":0,"byteOffset":{idx_off},"byteLength":{idx_bytes}}}],"buffers":[{{"byteLength":{buf_len}}}]}}"#,
            pos_bytes = pos_bytes,
            norm_off = pos_bytes,
            norm_bytes = norm_bytes,
            idx_off = pos_bytes + norm_bytes,
            idx_bytes = idx_bytes,
            buf_len = bin_buffer.len(),
        );

        let mut json_bytes = json.into_bytes();
        while json_bytes.len() % 4 != 0 {
            json_bytes.push(b' ');
        }

        let total_len = 12 + 8 + json_bytes.len() + 8 + bin_buffer.len();
        let mut glb = Vec::with_capacity(total_len);

        // Header
        glb.extend_from_slice(&0x4654_6C67_u32.to_le_bytes());
        glb.extend_from_slice(&2_u32.to_le_bytes());
        #[allow(clippy::cast_possible_truncation)]
        glb.extend_from_slice(&(total_len as u32).to_le_bytes());

        // JSON chunk
        #[allow(clippy::cast_possible_truncation)]
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
        glb.extend_from_slice(&json_bytes);

        // BIN chunk
        #[allow(clippy::cast_possible_truncation)]
        glb.extend_from_slice(&(bin_buffer.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E_4942_u32.to_le_bytes());
        glb.extend_from_slice(&bin_buffer);

        let mesh = read_glb(&glb).unwrap();
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.normals.len(), 3);
        assert_eq!(mesh.indices.len(), 3);
        assert_eq!(mesh.indices, vec![0, 1, 2]);
    }

    #[test]
    fn read_multi_primitive_glb() {
        // Two primitives with different accessor indices.
        // Primitive 0: positions=accessor 0, normals=accessor 1, indices=accessor 2
        // Primitive 1: positions=accessor 3, normals=accessor 4, indices=accessor 5

        // 3 vertices each: 2 triangles total
        let positions_0: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let normals_0: [f32; 9] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let indices_0: [u32; 3] = [0, 1, 2];

        let positions_1: [f32; 9] = [2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 1.0, 0.0];
        let normals_1: [f32; 9] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let indices_1: [u32; 3] = [0, 1, 2];

        let bv0_len = 36_usize; // positions_0
        let bv1_len = 36_usize; // normals_0
        let bv2_len = 12_usize; // indices_0
        let bv3_len = 36_usize; // positions_1
        let bv4_len = 36_usize; // normals_1
        let bv5_len = 12_usize; // indices_1

        let bv0_off = 0_usize;
        let bv1_off = bv0_off + bv0_len;
        let bv2_off = bv1_off + bv1_len;
        let bv3_off = bv2_off + bv2_len;
        let bv4_off = bv3_off + bv3_len;
        let bv5_off = bv4_off + bv4_len;
        let buf_len = bv5_off + bv5_len;

        let mut bin_buffer = Vec::with_capacity(buf_len);
        for &v in &positions_0 {
            bin_buffer.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &normals_0 {
            bin_buffer.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &indices_0 {
            bin_buffer.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &positions_1 {
            bin_buffer.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &normals_1 {
            bin_buffer.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &indices_1 {
            bin_buffer.extend_from_slice(&v.to_le_bytes());
        }
        while bin_buffer.len() % 4 != 0 {
            bin_buffer.push(0);
        }

        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0,1]}}],"nodes":[{{"mesh":0}},{{"mesh":1}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1}},"indices":2}}]}},{{"primitives":[{{"attributes":{{"POSITION":3,"NORMAL":4}},"indices":5}}]}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}},{{"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"}},{{"bufferView":2,"componentType":5125,"count":3,"type":"SCALAR"}},{{"bufferView":3,"componentType":5126,"count":3,"type":"VEC3"}},{{"bufferView":4,"componentType":5126,"count":3,"type":"VEC3"}},{{"bufferView":5,"componentType":5125,"count":3,"type":"SCALAR"}}],"bufferViews":[{{"buffer":0,"byteOffset":{bv0_off},"byteLength":{bv0_len}}},{{"buffer":0,"byteOffset":{bv1_off},"byteLength":{bv1_len}}},{{"buffer":0,"byteOffset":{bv2_off},"byteLength":{bv2_len}}},{{"buffer":0,"byteOffset":{bv3_off},"byteLength":{bv3_len}}},{{"buffer":0,"byteOffset":{bv4_off},"byteLength":{bv4_len}}},{{"buffer":0,"byteOffset":{bv5_off},"byteLength":{bv5_len}}}],"buffers":[{{"byteLength":{buf_len}}}]}}"#,
        );

        let mut json_bytes = json.into_bytes();
        while json_bytes.len() % 4 != 0 {
            json_bytes.push(b' ');
        }

        let total_len = 12 + 8 + json_bytes.len() + 8 + bin_buffer.len();
        let mut glb = Vec::with_capacity(total_len);

        glb.extend_from_slice(&0x4654_6C67_u32.to_le_bytes());
        glb.extend_from_slice(&2_u32.to_le_bytes());
        #[allow(clippy::cast_possible_truncation)]
        glb.extend_from_slice(&(total_len as u32).to_le_bytes());

        #[allow(clippy::cast_possible_truncation)]
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
        glb.extend_from_slice(&json_bytes);

        #[allow(clippy::cast_possible_truncation)]
        glb.extend_from_slice(&(bin_buffer.len() as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E_4942_u32.to_le_bytes());
        glb.extend_from_slice(&bin_buffer);

        let mesh = read_glb(&glb).unwrap();

        // Combined: 6 positions, 6 normals, 6 indices
        assert_eq!(mesh.positions.len(), 6, "should have 6 vertices total");
        assert_eq!(mesh.normals.len(), 6, "should have 6 normals total");
        assert_eq!(mesh.indices.len(), 6, "should have 6 indices total");

        // Second primitive indices should be offset by 3
        assert_eq!(mesh.indices[0], 0);
        assert_eq!(mesh.indices[1], 1);
        assert_eq!(mesh.indices[2], 2);
        assert_eq!(mesh.indices[3], 3); // 0 + vertex_offset(3)
        assert_eq!(mesh.indices[4], 4); // 1 + vertex_offset(3)
        assert_eq!(mesh.indices[5], 5); // 2 + vertex_offset(3)

        // Check that second primitive positions are correct
        assert!((mesh.positions[3].x() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn roundtrip_multi_solid_glb() {
        // Write two separate solids to a GLB and read them back
        let mut topo = brepkit_topology::Topology::new();
        let box1 = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let box2 = brepkit_operations::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let glb = crate::gltf::write_glb(&topo, &[box1, box2], 0.1).unwrap();
        let mesh = read_glb(&glb).unwrap();

        assert!(!mesh.positions.is_empty(), "should have vertices");
        assert!(!mesh.indices.is_empty(), "should have indices");
        assert_eq!(mesh.indices.len() % 3, 0, "should be triangles");
    }
}
