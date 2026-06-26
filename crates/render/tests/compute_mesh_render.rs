//! Headless tests for the GPU compute-shader quadric mesher (M2).
//!
//! These render a cylinder lateral face meshed entirely on the GPU from its
//! analytic parameters (no CPU tessellation), then cross-check the result
//! against the M1 solid path (which tessellates on the CPU) and verify the LOD
//! knob. They require a wgpu adapter (a real GPU or a software fallback such as
//! Mesa lavapipe); when none is available they log a skip and return.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use brepkit_math::vec::{Point3, Vec3};
use brepkit_render::{
    Camera, CylinderDescriptor, RenderOpts, RenderOutput, TessFactor, extract_cylinder_descriptor,
    probe_adapter, render_cylinder_compute_offscreen, render_solid_offscreen,
};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 8.0;
const HEIGHT: f64 = 20.0;

/// An isometric-ish camera framing a model of bounding-sphere `radius` centered
/// at `target`. Matches the framing used by the M1 smoke tests so the compute
/// and solid renders share a viewpoint for cross-checking.
fn iso_camera(target: Point3, radius: f64) -> Camera {
    let fov_y = 40.0_f64.to_radians();
    let dist = radius / (fov_y * 0.5).sin() * 2.0;
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

/// The cylinder centered like `make_cylinder` (base at z=0, axis +Z) framed by
/// an iso camera from its AABB center.
fn cylinder_camera() -> (Point3, Camera) {
    let center = Point3::new(0.0, 0.0, HEIGHT * 0.5);
    let bsphere = (RADIUS * RADIUS + (HEIGHT * 0.5) * (HEIGHT * 0.5)).sqrt();
    (center, iso_camera(center, bsphere))
}

/// Encode a linear color component to sRGB (matches the GPU's `Rgba8UnormSrgb`
/// write).
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

/// For each image row, the leftmost and rightmost drawn (non-background) column.
///
/// The left/right profile of a cylinder's side silhouette is governed purely by
/// the analytic radius and axis, so it is the cleanest signal to cross-check
/// the compute mesh against the CPU mesh (the caps differ between the two, but
/// the vertical sides do not).
fn horizontal_profile(out: &RenderOutput) -> Vec<Option<(u32, u32)>> {
    let mut rows = Vec::with_capacity(out.height as usize);
    for y in 0..out.height {
        let mut span: Option<(u32, u32)> = None;
        for x in 0..out.width {
            if out.face_id_at(x, y).is_some() {
                span = Some(match span {
                    None => (x, x),
                    Some((lo, hi)) => (lo.min(x), hi.max(x)),
                });
            }
        }
        rows.push(span);
    }
    rows
}

/// Count background pixels that lie strictly between the left and right drawn
/// columns of their row (interior holes). A watertight, gap-free front surface
/// has zero such pixels; a cracked seam shows up as a vertical run of them.
fn interior_holes(out: &RenderOutput) -> usize {
    let mut holes = 0;
    for (y, span) in horizontal_profile(out).into_iter().enumerate() {
        if let Some((lo, hi)) = span {
            for x in (lo + 1)..hi {
                #[allow(clippy::cast_possible_truncation)]
                if out.face_id_at(x, y as u32).is_none() {
                    holes += 1;
                }
            }
        }
    }
    holes
}

/// Find the cylinder (lateral) face of `solid`.
fn cylinder_face(topo: &Topology, solid: SolidId) -> FaceId {
    solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .expect("cylinder solid has a cylindrical lateral face")
}

/// A side-on orthographic-ish camera: the cylinder axis (+Z) is vertical and
/// perpendicular to the view (looking along +Y), so the lateral surface
/// silhouette is a clean rectangle exactly `2·radius` wide. This isolates the
/// angular faceting: an inscribed N-gon's widest pair of generators sits at
/// `radius·cos(π/N)` from the axis, so a coarse mesh renders *narrower* than a
/// fine one — the rendered width converges up to the analytic `2·radius`.
fn side_camera(center: Point3, radius: f64) -> Camera {
    let fov_y = 30.0_f64.to_radians();
    // Far back so the perspective foreshortening of the width is negligible
    // (a near-orthographic side view), keeping the silhouette a true rectangle.
    let dist = radius / (fov_y * 0.5).sin() * 6.0;
    let eye = center + Vec3::new(0.0, -dist, 0.0);
    Camera {
        eye,
        target: center,
        up: Vec3::new(0.0, 0.0, 1.0),
        fov_y,
        aspect: 1.0,
        near: dist - radius * 2.0,
        far: dist + radius * 2.0,
    }
}

/// Maximum rendered half-width (pixels from image center) of the silhouette
/// across the central band of rows. For the side-on view this is the projected
/// distance of the widest generator from the axis — larger means the mesh
/// reaches closer to the true cylinder radius (rounder).
fn max_side_half_width(out: &RenderOutput) -> f64 {
    let profile = horizontal_profile(out);
    let cx = f64::from(out.width) * 0.5;
    let y_lo = out.height / 3;
    let y_hi = out.height * 2 / 3;
    let mut max_half = 0.0_f64;
    for y in y_lo..y_hi {
        if let Some((lo, hi)) = profile[y as usize] {
            let half = ((f64::from(hi) + 0.5 - cx).abs()).max((cx - f64::from(lo) - 0.5).abs());
            max_half = max_half.max(half);
        }
    }
    max_half
}

#[test]
fn compute_mesh_cylinder_is_nonblank_and_plausible() {
    let Some(adapter) = probe_adapter() else {
        println!("SKIP compute_mesh_cylinder_is_nonblank_and_plausible: no wgpu adapter available");
        return;
    };
    println!("using wgpu adapter: {adapter}");

    let mut topo = Topology::new();
    let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
    let face = cylinder_face(&topo, cyl);
    let desc = extract_cylinder_descriptor(&topo, face).unwrap();

    // Sanity: the extracted descriptor recovers the primitive's geometry.
    assert!(
        (desc.radius - RADIUS).abs() < 1e-9,
        "radius {}",
        desc.radius
    );
    assert!(
        (desc.v0).abs() < 1e-6 && (desc.v1 - HEIGHT).abs() < 1e-6,
        "axial range [{}, {}] should span [0, {HEIGHT}]",
        desc.v0,
        desc.v1
    );

    let (_, cam) = cylinder_camera();
    let opts = RenderOpts::new(512, 512);
    let tess = TessFactor::new(48, 1);
    let out = render_cylinder_compute_offscreen(&desc, tess, 1, &cam, &opts).unwrap();

    let path = std::env::temp_dir().join("brepkit_compute_cylinder.png");
    out.color.save(&path).unwrap();
    println!("wrote {}", path.display());

    let total = (out.width * out.height) as usize;
    let drawn = non_background_pixels(&out, opts.background);
    let frac = drawn as f64 / total as f64;
    assert!(
        frac > 0.05,
        "only {drawn}/{total} ({frac:.3}) pixels differ from background"
    );
    assert!(
        frac < 0.98,
        "{drawn}/{total} ({frac:.3}) pixels differ from background (camera too close?)"
    );

    // A plausible cylinder side: silhouette present and wider than tall is not
    // required, but it must be a real, bounded region inside the frame.
    let (minx, miny, maxx, maxy) = id_silhouette_bbox(&out).expect("expected face-id pixels");
    let bw = maxx - minx + 1;
    let bh = maxy - miny + 1;
    assert!(bw >= 32 && bh >= 32, "silhouette too small: {bw}x{bh}");
    assert!(
        bw < out.width && bh < out.height,
        "silhouette fills the frame: {bw}x{bh}"
    );

    // Every drawn id is the face id we asked for (1).
    for &id in &out.id_buffer {
        assert!(id == 0 || id == 1, "unexpected id {id} in compute mesh");
    }
    assert_eq!(out.face_id_at(0, 0), None, "corner should be background");
}

#[test]
fn compute_mesh_matches_cpu_silhouette() {
    let Some(_) = probe_adapter() else {
        println!("SKIP compute_mesh_matches_cpu_silhouette: no wgpu adapter available");
        return;
    };

    let mut topo = Topology::new();
    let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
    let face = cylinder_face(&topo, cyl);
    let desc = extract_cylinder_descriptor(&topo, face).unwrap();

    let (_, cam) = cylinder_camera();
    let opts = RenderOpts {
        edges: false,
        ..RenderOpts::new(512, 512)
    };

    // CPU path: tessellate the whole solid (M1).
    let cpu = render_solid_offscreen(&topo, cyl, &cam, &opts).unwrap();
    // GPU path: compute-mesh the lateral face only, fine LOD for a tight match.
    let gpu =
        render_cylinder_compute_offscreen(&desc, TessFactor::new(96, 4), 1, &cam, &opts).unwrap();

    let cpu_bbox = id_silhouette_bbox(&cpu).expect("cpu silhouette");
    let gpu_bbox = id_silhouette_bbox(&gpu).expect("gpu silhouette");
    println!("cpu bbox {cpu_bbox:?} gpu bbox {gpu_bbox:?}");

    // The left/right extent is set purely by the analytic radius+axis and must
    // agree closely (the compute mesh has no caps, so the vertical extent can
    // differ slightly, but the horizontal silhouette edges should not).
    let tol = 4_i64; // pixels
    assert!(
        (i64::from(cpu_bbox.0) - i64::from(gpu_bbox.0)).abs() <= tol,
        "left edge differs: cpu {} gpu {}",
        cpu_bbox.0,
        gpu_bbox.0
    );
    assert!(
        (i64::from(cpu_bbox.2) - i64::from(gpu_bbox.2)).abs() <= tol,
        "right edge differs: cpu {} gpu {}",
        cpu_bbox.2,
        gpu_bbox.2
    );

    // The side silhouette half-width should match between CPU and GPU meshes
    // across the central band (caps excluded): compare per-row spans.
    let cpu_prof = horizontal_profile(&cpu);
    let gpu_prof = horizontal_profile(&gpu);
    let mut compared = 0;
    let mut max_diff = 0_i64;
    for y in (cpu.height / 3)..(cpu.height * 2 / 3) {
        if let (Some((cl, cr)), Some((gl, gr))) = (cpu_prof[y as usize], gpu_prof[y as usize]) {
            max_diff = max_diff
                .max((i64::from(cl) - i64::from(gl)).abs())
                .max((i64::from(cr) - i64::from(gr)).abs());
            compared += 1;
        }
    }
    assert!(compared > 50, "too few comparable rows: {compared}");
    assert!(
        max_diff <= 4,
        "central-band side profile differs by up to {max_diff}px"
    );
}

#[test]
fn compute_mesh_lod_scales_triangles_and_smooths_silhouette() {
    let Some(_) = probe_adapter() else {
        println!(
            "SKIP compute_mesh_lod_scales_triangles_and_smooths_silhouette: no wgpu adapter available"
        );
        return;
    };

    let mut topo = Topology::new();
    let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
    let face = cylinder_face(&topo, cyl);
    let desc = extract_cylinder_descriptor(&topo, face).unwrap();

    let center = Point3::new(0.0, 0.0, HEIGHT * 0.5);
    let cam = side_camera(center, RADIUS);
    let opts = RenderOpts {
        edges: false,
        ..RenderOpts::new(512, 512)
    };

    let coarse = TessFactor::new(6, 1);
    let fine = TessFactor::new(48, 1);
    let reference = TessFactor::new(256, 1); // ≈ the analytic circle

    // Triangle count scales with the angular factor (2·n_u·n_v).
    assert_eq!(CylinderDescriptor::triangle_count(coarse), 12);
    assert_eq!(CylinderDescriptor::triangle_count(fine), 96);
    assert!(
        CylinderDescriptor::triangle_count(fine) > CylinderDescriptor::triangle_count(coarse) * 4,
        "fine {} should dwarf coarse {}",
        CylinderDescriptor::triangle_count(fine),
        CylinderDescriptor::triangle_count(coarse)
    );

    let coarse_out = render_cylinder_compute_offscreen(&desc, coarse, 1, &cam, &opts).unwrap();
    let fine_out = render_cylinder_compute_offscreen(&desc, fine, 1, &cam, &opts).unwrap();
    let ref_out = render_cylinder_compute_offscreen(&desc, reference, 1, &cam, &opts).unwrap();

    coarse_out
        .color
        .save(std::env::temp_dir().join("brepkit_compute_cylinder_coarse.png"))
        .unwrap();
    fine_out
        .color
        .save(std::env::temp_dir().join("brepkit_compute_cylinder_fine.png"))
        .unwrap();

    // Side-on, the silhouette half-width is the projected distance of the
    // widest generator from the axis. An inscribed N-gon underestimates the
    // analytic radius; the shortfall (reference − rendered) shrinks with LOD.
    let analytic = max_side_half_width(&ref_out);
    let coarse_half = max_side_half_width(&coarse_out);
    let fine_half = max_side_half_width(&fine_out);
    let coarse_short = analytic - coarse_half;
    let fine_short = analytic - fine_half;
    println!(
        "analytic half={analytic:.2}px  coarse={coarse_half:.2} (short {coarse_short:.2})  fine={fine_half:.2} (short {fine_short:.2})"
    );

    // The finer mesh reaches strictly closer to the analytic radius (rounder
    // silhouette), and the coarse 6-gon shows a clearly visible shortfall.
    assert!(
        coarse_short > 1.5,
        "coarse 6-gon should visibly under-fill the circle, shortfall {coarse_short:.2}px"
    );
    assert!(
        fine_half > coarse_half + 1.0,
        "finer LOD should render a wider (rounder) silhouette: coarse {coarse_half:.2} fine {fine_half:.2}"
    );
    assert!(
        fine_short < coarse_short * 0.5,
        "finer LOD should at least halve the facet shortfall (coarse {coarse_short:.2}, fine {fine_short:.2})"
    );
}

#[test]
fn compute_mesh_seam_is_watertight() {
    let Some(_) = probe_adapter() else {
        println!("SKIP compute_mesh_seam_is_watertight: no wgpu adapter available");
        return;
    };

    let mut topo = Topology::new();
    let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
    let face = cylinder_face(&topo, cyl);
    let desc = extract_cylinder_descriptor(&topo, face).unwrap();

    // Look along -X so the u = 0 seam generator (at +X) sits dead-center facing
    // the camera. If the wrap quad did not reuse column 0's vertices, a crack
    // would open along this central vertical line.
    let center = Point3::new(0.0, 0.0, HEIGHT * 0.5);
    let fov_y = 35.0_f64.to_radians();
    let dist = RADIUS / (fov_y * 0.5).sin() * 3.0;
    let cam = Camera {
        eye: Point3::new(dist, 0.0, HEIGHT * 0.5),
        target: center,
        up: Vec3::new(0.0, 0.0, 1.0),
        fov_y,
        aspect: 1.0,
        near: dist - RADIUS * 2.0,
        far: dist + RADIUS * 2.0,
    };
    let opts = RenderOpts {
        edges: false,
        ..RenderOpts::new(512, 512)
    };

    // A coarse mesh maximizes the chance that a mis-stitched seam shows a gap.
    let out =
        render_cylinder_compute_offscreen(&desc, TessFactor::new(12, 3), 1, &cam, &opts).unwrap();
    out.color
        .save(std::env::temp_dir().join("brepkit_compute_cylinder_seam.png"))
        .unwrap();

    // The front wall fills the whole silhouette interior — no background cracks,
    // in particular none along the central seam line.
    let holes = interior_holes(&out);
    assert_eq!(
        holes, 0,
        "seam/interior shows {holes} background hole pixels"
    );

    // And the seam column itself is drawn (the surface is continuous there).
    let cx = out.width / 2;
    let mut seam_drawn = 0;
    for y in (out.height / 3)..(out.height * 2 / 3) {
        if out.face_id_at(cx, y).is_some() {
            seam_drawn += 1;
        }
    }
    assert!(
        seam_drawn > 50,
        "central seam column should be continuously drawn, got {seam_drawn} rows"
    );
}

#[test]
fn compute_mesh_matches_cpu_for_off_origin_cylinder() {
    let Some(_) = probe_adapter() else {
        println!(
            "SKIP compute_mesh_matches_cpu_for_off_origin_cylinder: no wgpu adapter available"
        );
        return;
    };

    // A cylinder whose axis does NOT pass through the world origin: this
    // exercises the descriptor's `axis_origin` term (a cylinder built at the
    // origin would hide a missing-translation bug).
    let shift = Vec3::new(50.0, -30.0, 12.0);
    let mut topo = Topology::new();
    let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
    brepkit_operations::transform::transform_solid(
        &mut topo,
        cyl,
        &brepkit_math::mat::Mat4::translation(shift.x(), shift.y(), shift.z()),
    )
    .unwrap();

    let face = cylinder_face(&topo, cyl);
    let desc = extract_cylinder_descriptor(&topo, face).unwrap();
    // The extracted axis origin tracks the translation.
    assert!(
        (desc.axis_origin.x() - shift.x()).abs() < 1e-6
            && (desc.axis_origin.y() - shift.y()).abs() < 1e-6,
        "axis_origin {:?} should follow the translation",
        desc.axis_origin
    );

    let center = Point3::new(shift.x(), shift.y(), shift.z() + HEIGHT * 0.5);
    let bsphere = (RADIUS * RADIUS + (HEIGHT * 0.5) * (HEIGHT * 0.5)).sqrt();
    let cam = iso_camera(center, bsphere);
    let opts = RenderOpts {
        edges: false,
        ..RenderOpts::new(512, 512)
    };

    let cpu = render_solid_offscreen(&topo, cyl, &cam, &opts).unwrap();
    let gpu =
        render_cylinder_compute_offscreen(&desc, TessFactor::new(96, 4), 1, &cam, &opts).unwrap();

    let cpu_bbox = id_silhouette_bbox(&cpu).expect("cpu silhouette");
    let gpu_bbox = id_silhouette_bbox(&gpu).expect("gpu silhouette");
    println!("off-origin cpu bbox {cpu_bbox:?} gpu bbox {gpu_bbox:?}");

    // The horizontal silhouette edges (set by the analytic radius + axis
    // location) must agree — proving the GPU honors `axis_origin`.
    let tol = 4_i64;
    assert!(
        (i64::from(cpu_bbox.0) - i64::from(gpu_bbox.0)).abs() <= tol,
        "left edge differs: cpu {} gpu {}",
        cpu_bbox.0,
        gpu_bbox.0
    );
    assert!(
        (i64::from(cpu_bbox.2) - i64::from(gpu_bbox.2)).abs() <= tol,
        "right edge differs: cpu {} gpu {}",
        cpu_bbox.2,
        gpu_bbox.2
    );
}
