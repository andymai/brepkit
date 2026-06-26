//! Headless offscreen-render smoke tests.
//!
//! These tests require a wgpu adapter (a real GPU or a software fallback such
//! as Mesa lavapipe via the Vulkan backend). When no adapter is available they
//! log a skip message and return rather than failing, so the suite stays green
//! on machines without any GPU stack.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::collections::HashSet;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_render::{Camera, RenderOpts, RenderOutput, probe_adapter, render_solid_offscreen};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

/// An isometric-ish camera framing a model of bounding-sphere `radius` centered
/// at `target`, pulled back far enough to fit the model with margin.
fn iso_camera(target: Point3, radius: f64) -> Camera {
    let fov_y = 40.0_f64.to_radians();
    // Distance so a sphere of `radius` subtends ~half the vertical FOV (a ~2x
    // framing margin), keeping the silhouette comfortably inside the frame.
    let dist = radius / (fov_y * 0.5).sin() * 2.0;
    // Isometric-ish unit direction from target to eye.
    let dir = Vec3::new(1.0, 0.9, 1.1).normalize().unwrap();
    let eye = target + Vec3::new(dir.x() * dist, dir.y() * dist, dir.z() * dist);
    Camera {
        eye,
        target,
        up: Vec3::new(0.0, 0.0, 1.0),
        fov_y,
        aspect: 1.0,
        near: (dist - radius).max(radius * 0.01),
        far: dist + radius * 4.0,
    }
}

/// Encode a linear color component to sRGB (matches the GPU's `Rgba8UnormSrgb`
/// write), so background comparison uses the byte the GPU actually stored.
fn linear_to_srgb_u8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

/// Count pixels that differ meaningfully from the background clear color.
fn non_background_pixels(out: &RenderOutput, bg: [f32; 4]) -> usize {
    let bg_rgb = [
        linear_to_srgb_u8(bg[0]),
        linear_to_srgb_u8(bg[1]),
        linear_to_srgb_u8(bg[2]),
    ];
    let mut count = 0;
    for px in out.color.pixels() {
        let d = (i32::from(px[0]) - i32::from(bg_rgb[0])).abs()
            + (i32::from(px[1]) - i32::from(bg_rgb[1])).abs()
            + (i32::from(px[2]) - i32::from(bg_rgb[2])).abs();
        // Tolerate sRGB-encoding rounding around the clear color.
        if d > 12 {
            count += 1;
        }
    }
    count
}

/// Bounding box (min_x, min_y, max_x, max_y) of non-zero id pixels, if any.
fn id_silhouette_bbox(out: &RenderOutput) -> Option<(u32, u32, u32, u32)> {
    let mut bbox: Option<(u32, u32, u32, u32)> = None;
    for y in 0..out.height {
        for x in 0..out.width {
            if out.face_id_at(x, y).is_some() {
                bbox = Some(match bbox {
                    None => (x, y, x, y),
                    Some((minx, miny, maxx, maxy)) => {
                        (minx.min(x), miny.min(y), maxx.max(x), maxy.max(y))
                    }
                });
            }
        }
    }
    bbox
}

/// Render one solid, write a PNG to the scratch dir, and assert the image is a
/// plausible non-blank silhouette whose ids map back to real faces.
fn check_render(topo: &Topology, solid: SolidId, name: &str) {
    let faces = solid_faces(topo, solid).unwrap();
    let valid_ids: HashSet<u32> = faces
        .iter()
        .map(|f| u32::try_from(f.index()).unwrap() + 1)
        .collect();

    // Frame the model from its AABB center.
    let (min, max) = solid_aabb(topo, solid);
    let center = Point3::new(
        (min.x() + max.x()) * 0.5,
        (min.y() + max.y()) * 0.5,
        (min.z() + max.z()) * 0.5,
    );
    let radius =
        ((max.x() - min.x()).powi(2) + (max.y() - min.y()).powi(2) + (max.z() - min.z()).powi(2))
            .sqrt()
            * 0.5;

    let cam = iso_camera(center, radius);
    let opts = RenderOpts::new(512, 512);
    let out = render_solid_offscreen(topo, solid, &cam, &opts).unwrap();

    assert_eq!(out.width, 512);
    assert_eq!(out.height, 512);
    assert_eq!(out.id_buffer.len(), (512 * 512) as usize);

    let path = std::env::temp_dir().join(format!("brepkit_render_{name}.png"));
    out.color.save(&path).unwrap();
    println!("wrote {}", path.display());

    // 1. The image is non-blank: a meaningful fraction of pixels differ from bg.
    let total = (out.width * out.height) as usize;
    let drawn = non_background_pixels(&out, opts.background);
    let frac = drawn as f64 / total as f64;
    assert!(
        frac > 0.05,
        "{name}: only {drawn}/{total} ({frac:.3}) pixels differ from background; expected a visible solid"
    );
    assert!(
        frac < 0.98,
        "{name}: {drawn}/{total} ({frac:.3}) pixels differ from background; silhouette fills the whole frame (camera too close?)"
    );

    // 2. The id silhouette is plausible: present, but not the entire frame.
    let bbox = id_silhouette_bbox(&out).expect("expected at least one face-id pixel");
    let (minx, miny, maxx, maxy) = bbox;
    let bw = maxx - minx + 1;
    let bh = maxy - miny + 1;
    assert!(
        bw >= 32 && bh >= 32,
        "{name}: id silhouette too small: {bw}x{bh}"
    );
    assert!(
        bw < out.width && bh < out.height,
        "{name}: id silhouette fills the entire frame: {bw}x{bh}"
    );

    // 3. Every non-background id maps to a real face of the solid, and at least
    //    one face is actually visible.
    let mut seen: HashSet<u32> = HashSet::new();
    for &id in &out.id_buffer {
        if id != 0 {
            assert!(
                valid_ids.contains(&id),
                "{name}: id buffer holds {id}, which is not a face of the solid"
            );
            seen.insert(id);
        }
    }
    assert!(
        !seen.is_empty(),
        "{name}: no face-id pixels mapped to a real FaceId"
    );

    // 4. Background pixels really are 0 (the corners of an iso view are empty).
    assert_eq!(
        out.face_id_at(0, 0),
        None,
        "{name}: top-left corner should be background (id 0)"
    );
}

/// Axis-aligned bounding box of a solid via its tessellation positions.
fn solid_aabb(topo: &Topology, solid: SolidId) -> (Point3, Point3) {
    let (mesh, _) = brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance(
        topo,
        solid,
        0.05,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
    )
    .unwrap();
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for p in &mesh.positions {
        let c = [p.x(), p.y(), p.z()];
        for i in 0..3 {
            min[i] = min[i].min(c[i]);
            max[i] = max[i].max(c[i]);
        }
    }
    (
        Point3::new(min[0], min[1], min[2]),
        Point3::new(max[0], max[1], max[2]),
    )
}

#[test]
fn render_box_and_cylinder_offscreen() {
    let Some(adapter) = probe_adapter() else {
        println!(
            "SKIP render_box_and_cylinder_offscreen: no wgpu adapter available (no GPU or software fallback in this environment)"
        );
        return;
    };
    println!("using wgpu adapter: {adapter}");

    let mut topo = Topology::new();
    let cube = brepkit_operations::primitives::make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
    check_render(&topo, cube, "box");

    let mut topo = Topology::new();
    let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, 8.0, 20.0).unwrap();
    check_render(&topo, cyl, "cylinder");
}

#[test]
fn invalid_size_is_rejected() {
    // Size validation does not need a GPU, so it always runs.
    let mut topo = Topology::new();
    let cube = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let cam = iso_camera(Point3::new(5.0, 5.0, 5.0), 9.0);
    let opts = RenderOpts {
        width: 0,
        ..RenderOpts::new(0, 256)
    };
    let err = render_solid_offscreen(&topo, cube, &cam, &opts);
    assert!(matches!(
        err,
        Err(brepkit_render::RenderError::InvalidSize { .. })
    ));
}
