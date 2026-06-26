//! Headless tests for screen-space adaptive LOD (M2.1).
//!
//! The compute mesher's tessellation is derived per-frame from the cylinder's
//! projected screen size, so a near view gets a fine mesh and a far view a
//! coarse one — both bounded to a target pixel chord error. These tests render
//! the same cylinder from near/far cameras and verify the triangle count scales
//! with distance and the rendered silhouette stays within the pixel budget of
//! the analytic circle. They require a wgpu adapter; without one they skip.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use brepkit_math::vec::{Point3, Vec3};
use brepkit_render::{
    Camera, CylinderDescriptor, RenderOpts, RenderOutput, TessFactor, extract_cylinder_descriptor,
    probe_adapter, render_cylinder_compute_offscreen, render_cylinder_compute_screen_lod,
    screen_space_tess_factor,
};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 8.0;
const HEIGHT: f64 = 20.0;
const VIEWPORT: u32 = 512;
const TARGET_PX: f64 = 0.5;

/// A side-on near-orthographic camera at `dist` along +Y, axis (+Z) vertical and
/// perpendicular to the view, so the lateral silhouette is a clean rectangle
/// exactly `2·radius` wide. Far back enough that width foreshortening is tiny.
fn side_camera(center: Point3, dist: f64) -> Camera {
    let fov_y = 30.0_f64.to_radians();
    Camera {
        eye: center + Vec3::new(0.0, -dist, 0.0),
        target: center,
        up: Vec3::new(0.0, 0.0, 1.0),
        fov_y,
        aspect: 1.0,
        near: (dist - RADIUS * 2.0).max(0.01),
        far: dist + RADIUS * 2.0,
    }
}

/// Find the cylinder (lateral) face of `solid`.
fn cylinder_face(topo: &Topology, solid: SolidId) -> FaceId {
    solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .expect("cylinder solid has a cylindrical lateral face")
}

/// The default cylinder descriptor (base at z=0, axis +Z), centered on its AABB.
fn cylinder_descriptor() -> (CylinderDescriptor, Point3) {
    let mut topo = Topology::new();
    let cyl = brepkit_operations::primitives::make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
    let face = cylinder_face(&topo, cyl);
    let desc = extract_cylinder_descriptor(&topo, face).unwrap();
    (desc, desc.center)
}

/// For each image row, the leftmost and rightmost drawn (non-background) column.
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

/// Maximum silhouette half-width (pixels from image center) over the central
/// band of rows, for the side-on view.
///
/// All generators of a side-on cylinder sit at radius `r`, so the silhouette
/// edge is the *outermost projected generator*: an inscribed `n_u`-gon renders
/// this at `r_px·cos(φ)` where `φ ∈ [0, π/n_u]` is the offset of the nearest
/// generator to the silhouette tangent. The worst case (an edge facing the
/// silhouette) is `r_px·cos(π/n_u)`, so `r_px − max_half` is the rendered chord
/// error. Using the *max* (not min) avoids the vertical taper of the rounded
/// caps, which would otherwise contaminate a min-based metric at small scales.
fn side_max_half_width(out: &RenderOutput) -> f64 {
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

/// The cylinder radius projected to pixels at `cam`, matching the formula
/// `screen_space_tess_factor` uses: `r_px = r · (H/2) / (d · tan(fov_y/2))`.
fn analytic_projected_radius(desc: &CylinderDescriptor, cam: &Camera, viewport_h: u32) -> f64 {
    let d = (cam.eye - desc.center).length();
    desc.radius * (f64::from(viewport_h) * 0.5) / (d * (cam.fov_y * 0.5).tan())
}

/// Rendered chord error in pixels: how far the silhouette edge falls inside the
/// analytic projected circle (`r_px − rendered max half-width`).
fn chord_error_px(desc: &CylinderDescriptor, cam: &Camera, out: &RenderOutput) -> f64 {
    let r_px = analytic_projected_radius(desc, cam, out.height);
    (r_px - side_max_half_width(out)).max(0.0)
}

#[test]
fn screen_lod_triangle_count_scales_with_distance() {
    let Some(adapter) = probe_adapter() else {
        println!("SKIP screen_lod_triangle_count_scales_with_distance: no wgpu adapter available");
        return;
    };
    println!("using wgpu adapter: {adapter}");

    let (desc, center) = cylinder_descriptor();
    let opts = RenderOpts {
        edges: false,
        ..RenderOpts::new(VIEWPORT, VIEWPORT)
    };

    // Near: the cylinder fills much of the frame (but stays fully inside it, so
    // the silhouette is not clipped by the viewport edge). Far: it is small.
    let near_cam = side_camera(center, RADIUS * 6.5);
    let far_cam = side_camera(center, RADIUS * 40.0);

    let near_tess = screen_space_tess_factor(&desc, &near_cam, (VIEWPORT, VIEWPORT), TARGET_PX);
    let far_tess = screen_space_tess_factor(&desc, &far_cam, (VIEWPORT, VIEWPORT), TARGET_PX);
    let near_tris = CylinderDescriptor::triangle_count(near_tess);
    let far_tris = CylinderDescriptor::triangle_count(far_tess);
    println!(
        "near n_u={} ({near_tris} tris)  far n_u={} ({far_tris} tris)",
        near_tess.n_u, far_tess.n_u
    );

    assert!(
        near_tris > far_tris,
        "near view should produce more triangles than far: near {near_tris} far {far_tris}"
    );

    // Render both via the screen-LOD entry; write PNGs and confirm both are
    // non-blank, bounded silhouettes.
    let near = render_cylinder_compute_screen_lod(&desc, 1, &near_cam, &opts, TARGET_PX).unwrap();
    let far = render_cylinder_compute_screen_lod(&desc, 1, &far_cam, &opts, TARGET_PX).unwrap();
    let near_path = std::env::temp_dir().join("brepkit_lod_near.png");
    let far_path = std::env::temp_dir().join("brepkit_lod_far.png");
    near.color.save(&near_path).unwrap();
    far.color.save(&far_path).unwrap();
    println!("wrote {} and {}", near_path.display(), far_path.display());

    // Both silhouettes stay within ~target_px of the analytic circle: the
    // rendered silhouette edge is within the pixel budget of the projected
    // radius (plus ~1px rasterization slack).
    for (label, cam, out) in [("near", near_cam, &near), ("far", far_cam, &far)] {
        let r_px = analytic_projected_radius(&desc, &cam, VIEWPORT);
        // Guard: the cross-check is only valid if the silhouette is fully inside
        // the frame (otherwise the viewport edge, not the mesh, bounds it).
        assert!(
            r_px < f64::from(VIEWPORT) * 0.5 - 4.0,
            "{label}: projected radius {r_px:.1}px would clip the {VIEWPORT}px frame"
        );
        let err = chord_error_px(&desc, &cam, out);
        println!("{label}: r_px≈{r_px:.2}px  chord error {err:.2}px");
        assert!(
            err <= TARGET_PX + 1.0,
            "{label}: silhouette chord error {err:.2}px exceeds budget {TARGET_PX}px"
        );
    }
}

#[test]
fn screen_lod_bound_holds_and_is_not_wasteful() {
    let Some(_) = probe_adapter() else {
        println!("SKIP screen_lod_bound_holds_and_is_not_wasteful: no wgpu adapter available");
        return;
    };

    let (desc, center) = cylinder_descriptor();
    let opts = RenderOpts {
        edges: false,
        ..RenderOpts::new(VIEWPORT, VIEWPORT)
    };

    // A fixed moderate framing.
    let cam = side_camera(center, RADIUS * 8.0);
    let tess = screen_space_tess_factor(&desc, &cam, (VIEWPORT, VIEWPORT), TARGET_PX);
    println!("chosen n_u={}", tess.n_u);

    // The chosen LOD keeps the measured chord error within budget.
    let out = render_cylinder_compute_screen_lod(&desc, 1, &cam, &opts, TARGET_PX).unwrap();
    let r_px = analytic_projected_radius(&desc, &cam, VIEWPORT);
    let err = chord_error_px(&desc, &cam, &out);
    println!("r_px≈{r_px:.2}px  chosen chord error {err:.2}px");
    assert!(
        err <= TARGET_PX + 1.0,
        "chord error {err:.2}px exceeds budget {TARGET_PX}px"
    );

    // ...and is not wastefully high: a much coarser mesh would visibly violate
    // the budget. Quartering n_u must produce a measurably worse chord error,
    // confirming the chosen factor is near the budget rather than excessive.
    let coarser = TessFactor::new(tess.n_u / 4, 1);
    if coarser.n_u < tess.n_u {
        let coarse_out = render_cylinder_compute_offscreen(&desc, coarser, 1, &cam, &opts).unwrap();
        let coarse_err = chord_error_px(&desc, &cam, &coarse_out);
        println!("coarser n_u={} chord error {coarse_err:.2}px", coarser.n_u);
        assert!(
            coarse_err > err,
            "the chosen LOD should be near-minimal: quartering n_u ({}) did not worsen the error ({coarse_err:.2} vs {err:.2})",
            coarser.n_u
        );
    }
}

#[test]
fn screen_lod_render_entry_matches_manual_factor() {
    let Some(_) = probe_adapter() else {
        println!("SKIP screen_lod_render_entry_matches_manual_factor: no wgpu adapter available");
        return;
    };

    let (desc, center) = cylinder_descriptor();
    let opts = RenderOpts {
        edges: false,
        ..RenderOpts::new(VIEWPORT, VIEWPORT)
    };
    let cam = side_camera(center, RADIUS * 6.0);

    // The screen-LOD entry must pick the same factor the standalone fn reports,
    // so its output equals rendering with that factor explicitly.
    let tess = screen_space_tess_factor(&desc, &cam, (VIEWPORT, VIEWPORT), TARGET_PX);
    let via_entry = render_cylinder_compute_screen_lod(&desc, 1, &cam, &opts, TARGET_PX).unwrap();
    let via_manual = render_cylinder_compute_offscreen(&desc, tess, 1, &cam, &opts).unwrap();

    assert_eq!(
        via_entry.id_buffer, via_manual.id_buffer,
        "screen-LOD entry should render identically to the explicitly-chosen factor"
    );
}
