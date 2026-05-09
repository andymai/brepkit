//! glTF 2.0 binary (.glb) writer.
//!
//! Produces a single-file GLB containing tessellated mesh geometry
//! with vertex positions, normals, and triangle indices.

use std::io::Write;

use brepkit_operations::tessellate;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

/// Write one or more solids to glTF binary (.glb) format.
///
/// The output is a single GLB file containing a mesh with positions,
/// normals, and triangle indices. No materials or textures are included.
///
/// # Errors
///
/// Returns an error if tessellation fails.
#[allow(clippy::too_many_lines)]
pub fn write_glb(
    topo: &Topology,
    solids: &[SolidId],
    deflection: f64,
) -> Result<Vec<u8>, crate::IoError> {
    // Tessellate all solids into a merged mesh.
    let mut positions: Vec<f32> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let mut vertex_offset: u32 = 0;

    for &solid_id in solids {
        // Walk outer + inner (cavity) shells so hollow solids'
        // cavity surfaces are present in the GLB output.
        let face_ids = solid_faces(topo, solid_id)?;

        for &face_id in &face_ids {
            let mesh = tessellate::tessellate(topo, face_id, deflection)?;
            let offset = vertex_offset;

            for pos in &mesh.positions {
                #[allow(clippy::cast_possible_truncation)]
                {
                    positions.push(pos.x() as f32);
                    positions.push(pos.y() as f32);
                    positions.push(pos.z() as f32);
                }
            }

            for norm in &mesh.normals {
                #[allow(clippy::cast_possible_truncation)]
                {
                    normals.push(norm.x() as f32);
                    normals.push(norm.y() as f32);
                    normals.push(norm.z() as f32);
                }
            }

            for &idx in &mesh.indices {
                indices.push(idx + offset);
            }

            #[allow(clippy::cast_possible_truncation)]
            {
                vertex_offset += mesh.positions.len() as u32;
            }
        }
    }

    let vertex_count = positions.len() / 3;
    let index_count = indices.len();

    if vertex_count == 0 {
        return Err(crate::IoError::InvalidTopology {
            reason: "no vertices to export".into(),
        });
    }

    // Compute bounding box for accessor min/max.
    let mut min_pos = [f32::MAX; 3];
    let mut max_pos = [f32::MIN; 3];
    for chunk in positions.chunks_exact(3) {
        for i in 0..3 {
            min_pos[i] = min_pos[i].min(chunk[i]);
            max_pos[i] = max_pos[i].max(chunk[i]);
        }
    }

    // Build binary buffer: positions + normals + indices
    let pos_bytes = positions.len() * 4;
    let norm_bytes = normals.len() * 4;
    let idx_bytes = indices.len() * 4;
    let total_buffer = pos_bytes + norm_bytes + idx_bytes;

    let mut bin_buffer = Vec::with_capacity(total_buffer);
    for &val in &positions {
        bin_buffer.write_all(&val.to_le_bytes()).ok();
    }
    for &val in &normals {
        bin_buffer.write_all(&val.to_le_bytes()).ok();
    }
    for &val in &indices {
        bin_buffer.write_all(&val.to_le_bytes()).ok();
    }

    // Pad buffer to 4-byte alignment
    while bin_buffer.len() % 4 != 0 {
        bin_buffer.push(0);
    }

    // Build JSON
    let json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"brepkit"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1}},"indices":2}}]}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":{vertex_count},"type":"VEC3","min":[{min0},{min1},{min2}],"max":[{max0},{max1},{max2}]}},{{"bufferView":1,"componentType":5126,"count":{vertex_count},"type":"VEC3"}},{{"bufferView":2,"componentType":5125,"count":{index_count},"type":"SCALAR"}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{pos_bytes}}},{{"buffer":0,"byteOffset":{norm_offset},"byteLength":{norm_bytes}}},{{"buffer":0,"byteOffset":{idx_offset},"byteLength":{idx_bytes}}}],"buffers":[{{"byteLength":{buf_len}}}]}}"#,
        vertex_count = vertex_count,
        index_count = index_count,
        min0 = min_pos[0],
        min1 = min_pos[1],
        min2 = min_pos[2],
        max0 = max_pos[0],
        max1 = max_pos[1],
        max2 = max_pos[2],
        pos_bytes = pos_bytes,
        norm_offset = pos_bytes,
        norm_bytes = norm_bytes,
        idx_offset = pos_bytes + norm_bytes,
        idx_bytes = idx_bytes,
        buf_len = bin_buffer.len(),
    );

    let mut json_bytes = json.into_bytes();
    // Pad JSON to 4-byte alignment with spaces
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }

    // GLB header: magic + version + total length
    let total_len = 12 + 8 + json_bytes.len() + 8 + bin_buffer.len();

    let mut glb = Vec::with_capacity(total_len);

    // Header
    glb.write_all(&0x4654_6C67_u32.to_le_bytes()).ok(); // magic: "glTF"
    glb.write_all(&2_u32.to_le_bytes()).ok(); // version: 2
    #[allow(clippy::cast_possible_truncation)]
    {
        glb.write_all(&(total_len as u32).to_le_bytes()).ok(); // total length
    }

    // JSON chunk
    #[allow(clippy::cast_possible_truncation)]
    {
        glb.write_all(&(json_bytes.len() as u32).to_le_bytes()).ok(); // chunk length
    }
    glb.write_all(&0x4E4F_534A_u32.to_le_bytes()).ok(); // chunk type: "JSON"
    glb.write_all(&json_bytes).ok();

    // BIN chunk
    #[allow(clippy::cast_possible_truncation)]
    {
        glb.write_all(&(bin_buffer.len() as u32).to_le_bytes()).ok(); // chunk length
    }
    glb.write_all(&0x004E_4942_u32.to_le_bytes()).ok(); // chunk type: "BIN\0"
    glb.write_all(&bin_buffer).ok();

    Ok(glb)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use brepkit_topology::Topology;

    use super::*;

    #[test]
    fn write_box_glb() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let glb = write_glb(&topo, &[solid], 0.1).unwrap();

        // GLB starts with magic "glTF"
        assert_eq!(&glb[0..4], &0x4654_6C67_u32.to_le_bytes());

        // Version 2
        assert_eq!(&glb[4..8], &2_u32.to_le_bytes());

        // Total length matches actual length
        let total_len = u32::from_le_bytes([glb[8], glb[9], glb[10], glb[11]]);
        assert_eq!(total_len as usize, glb.len());
    }

    #[test]
    fn glb_has_json_and_bin_chunks() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let glb = write_glb(&topo, &[solid], 0.1).unwrap();

        // After 12-byte header: JSON chunk
        let json_len = u32::from_le_bytes([glb[12], glb[13], glb[14], glb[15]]) as usize;
        let json_type = u32::from_le_bytes([glb[16], glb[17], glb[18], glb[19]]);
        assert_eq!(json_type, 0x4E4F_534A); // "JSON"

        // JSON content should contain "asset" and "version"
        let json_str = std::str::from_utf8(&glb[20..20 + json_len]).unwrap();
        assert!(json_str.contains("\"version\":\"2.0\""));
        assert!(json_str.contains("\"generator\":\"brepkit\""));
        assert!(json_str.contains("POSITION"));
        assert!(json_str.contains("NORMAL"));

        // BIN chunk follows JSON
        let bin_offset = 20 + json_len;
        let bin_type = u32::from_le_bytes([
            glb[bin_offset + 4],
            glb[bin_offset + 5],
            glb[bin_offset + 6],
            glb[bin_offset + 7],
        ]);
        assert_eq!(bin_type, 0x004E_4942); // "BIN\0"
    }

    #[test]
    fn glb_is_4byte_aligned() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let glb = write_glb(&topo, &[solid], 0.1).unwrap();

        // Total length should be 4-byte aligned
        assert_eq!(glb.len() % 4, 0, "GLB should be 4-byte aligned");
    }
}
