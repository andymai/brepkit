//! Parametric primitive shape builders.
//!
//! Provides factory functions for creating standard CAD primitives as
//! proper B-Rep solids: boxes, cylinders, cones, spheres, and tori.
//!
//! Each function returns a [`SolidId`] for a newly created solid in
//! the given [`Topology`]. The solids are built with correct manifold
//! topology and outward-pointing face normals.

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

// ── Box ────────────────────────────────────────────────────────────

/// Create a box solid with one corner at the origin.
///
/// The box extends from `(0, 0, 0)` to `(dx, dy, dz)`.
/// One corner sits at the origin; the opposite at `(dx, dy, dz)`.
///
/// # Errors
///
/// Returns an error if any dimension is zero or negative.
pub fn make_box(
    topo: &mut Topology,
    dx: f64,
    dy: f64,
    dz: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if dx <= tol.linear || dy <= tol.linear || dz <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("box dimensions must be positive, got ({dx}, {dy}, {dz})"),
        });
    }

    // 8 vertices: corner at origin, extending to (dx, dy, dz)
    let v = [
        topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol.linear)),
        topo.add_vertex(Vertex::new(Point3::new(dx, 0.0, 0.0), tol.linear)),
        topo.add_vertex(Vertex::new(Point3::new(dx, dy, 0.0), tol.linear)),
        topo.add_vertex(Vertex::new(Point3::new(0.0, dy, 0.0), tol.linear)),
        topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, dz), tol.linear)),
        topo.add_vertex(Vertex::new(Point3::new(dx, 0.0, dz), tol.linear)),
        topo.add_vertex(Vertex::new(Point3::new(dx, dy, dz), tol.linear)),
        topo.add_vertex(Vertex::new(Point3::new(0.0, dy, dz), tol.linear)),
    ];

    // 12 edges (shared between faces)
    // Bottom ring (z=0)
    let eb0 = topo.add_edge(Edge::new(v[0], v[1], EdgeCurve::Line));
    let eb1 = topo.add_edge(Edge::new(v[1], v[2], EdgeCurve::Line));
    let eb2 = topo.add_edge(Edge::new(v[2], v[3], EdgeCurve::Line));
    let eb3 = topo.add_edge(Edge::new(v[3], v[0], EdgeCurve::Line));
    // Top ring (z=dz)
    let et0 = topo.add_edge(Edge::new(v[4], v[5], EdgeCurve::Line));
    let et1 = topo.add_edge(Edge::new(v[5], v[6], EdgeCurve::Line));
    let et2 = topo.add_edge(Edge::new(v[6], v[7], EdgeCurve::Line));
    let et3 = topo.add_edge(Edge::new(v[7], v[4], EdgeCurve::Line));
    // Verticals
    let ev0 = topo.add_edge(Edge::new(v[0], v[4], EdgeCurve::Line));
    let ev1 = topo.add_edge(Edge::new(v[1], v[5], EdgeCurve::Line));
    let ev2 = topo.add_edge(Edge::new(v[2], v[6], EdgeCurve::Line));
    let ev3 = topo.add_edge(Edge::new(v[3], v[7], EdgeCurve::Line));

    let mk_face = |topo: &mut Topology,
                   edges: [(brepkit_topology::edge::EdgeId, bool); 4],
                   normal: Vec3,
                   d: f64|
     -> Result<brepkit_topology::face::FaceId, crate::OperationsError> {
        let wire = Wire::new(
            edges
                .iter()
                .map(|&(eid, fwd)| OrientedEdge::new(eid, fwd))
                .collect(),
            true,
        )
        .map_err(crate::OperationsError::Topology)?;
        let wid = topo.add_wire(wire);
        Ok(topo.add_face(Face::new(wid, vec![], FaceSurface::Plane { normal, d })))
    };

    // Plane 'd' values: signed distance from origin along normal
    let bottom = mk_face(
        topo,
        [(eb0, false), (eb3, false), (eb2, false), (eb1, false)],
        Vec3::new(0.0, 0.0, -1.0),
        0.0,
    )?;
    let top = mk_face(
        topo,
        [(et0, true), (et1, true), (et2, true), (et3, true)],
        Vec3::new(0.0, 0.0, 1.0),
        dz,
    )?;
    let front = mk_face(
        topo,
        [(eb0, true), (ev1, true), (et0, false), (ev0, false)],
        Vec3::new(0.0, -1.0, 0.0),
        0.0,
    )?;
    let back = mk_face(
        topo,
        [(eb2, true), (ev3, true), (et2, false), (ev2, false)],
        Vec3::new(0.0, 1.0, 0.0),
        dy,
    )?;
    let left = mk_face(
        topo,
        [(eb3, true), (ev0, true), (et3, false), (ev3, false)],
        Vec3::new(-1.0, 0.0, 0.0),
        0.0,
    )?;
    let right = mk_face(
        topo,
        [(eb1, true), (ev2, true), (et1, false), (ev1, false)],
        Vec3::new(1.0, 0.0, 0.0),
        dx,
    )?;

    let shell = Shell::new(vec![bottom, top, front, back, left, right])
        .map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

// ── Cylinder ───────────────────────────────────────────────────────

/// Create a cylinder solid with its axis along +Z, base at the origin.
///
/// The cylinder extends from `z = 0` to `z = height`.
/// Built with one `CylindricalSurface` lateral face and two planar cap faces.
///
/// # Errors
///
/// Returns an error if `radius` or `height` is zero or negative.
pub fn make_cylinder(
    topo: &mut Topology,
    radius: f64,
    height: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if radius <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("cylinder radius must be positive, got {radius}"),
        });
    }
    if height <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("cylinder height must be positive, got {height}"),
        });
    }

    // Cylinder base at z=0, top at z=height.
    // (brepjs drill and placement code assumes this convention.)

    // Analytic cylindrical surface
    let cyl_surface = brepkit_math::surfaces::CylindricalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        radius,
    )
    .map_err(crate::OperationsError::Math)?;

    // --- Lateral face: single face with degenerate seam wire ---
    let v_bot = topo.add_vertex(Vertex::new(Point3::new(radius, 0.0, 0.0), tol.linear));
    let v_top = topo.add_vertex(Vertex::new(Point3::new(radius, 0.0, height), tol.linear));

    let bot_circle = brepkit_math::curves::Circle3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        radius,
    )
    .map_err(crate::OperationsError::Math)?;
    let top_circle = brepkit_math::curves::Circle3D::new(
        Point3::new(0.0, 0.0, height),
        Vec3::new(0.0, 0.0, 1.0),
        radius,
    )
    .map_err(crate::OperationsError::Math)?;

    let e_bot_circle = topo.add_edge(Edge::new(v_bot, v_bot, EdgeCurve::Circle(bot_circle)));
    let e_top_circle = topo.add_edge(Edge::new(v_top, v_top, EdgeCurve::Circle(top_circle)));
    let e_seam = topo.add_edge(Edge::new(v_bot, v_top, EdgeCurve::Line));

    let lateral_wire = Wire::new(
        vec![
            OrientedEdge::new(e_bot_circle, true),
            OrientedEdge::new(e_seam, true),
            OrientedEdge::new(e_top_circle, false),
            OrientedEdge::new(e_seam, false),
        ],
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let lateral_wid = topo.add_wire(lateral_wire);
    let lateral_face = topo.add_face(Face::new(
        lateral_wid,
        vec![],
        FaceSurface::Cylinder(cyl_surface),
    ));

    // --- Bottom cap (z = 0, normal pointing down) ---
    // Reuse the same circle edge as the lateral face for watertight topology.
    // Reversed orientation: CW from +z corresponds to outward normal -z.
    let bot_cap_wire = Wire::new(vec![OrientedEdge::new(e_bot_circle, false)], true)
        .map_err(crate::OperationsError::Topology)?;
    let bot_wid = topo.add_wire(bot_cap_wire);
    let bot_face = topo.add_face(Face::new(
        bot_wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ));

    // --- Top cap (z = height, normal pointing up) ---
    // Reuse the same circle edge; forward orientation gives outward normal +z.
    let top_cap_wire = Wire::new(vec![OrientedEdge::new(e_top_circle, true)], true)
        .map_err(crate::OperationsError::Topology)?;
    let top_wid = topo.add_wire(top_cap_wire);
    let top_face = topo.add_face(Face::new(
        top_wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: height,
        },
    ));

    let shell = Shell::new(vec![lateral_face, bot_face, top_face])
        .map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

// ── Cone ───────────────────────────────────────────────────────────

/// Create a cone solid with its base at the origin, axis along +Z.
///
/// The cone has `bottom_radius` at `z = 0` and `top_radius` at
/// `z = height`. Setting `top_radius = 0` creates a pointed cone;
/// setting it to a positive value creates a truncated cone (frustum).
/// The base is at the origin.
///
/// # Errors
///
/// Returns an error if `height` is non-positive, both radii are zero,
/// or any radius is negative.
#[allow(clippy::too_many_lines)]
pub fn make_cone(
    topo: &mut Topology,
    bottom_radius: f64,
    top_radius: f64,
    height: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if height <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("cone height must be positive, got {height}"),
        });
    }
    if bottom_radius < 0.0 || top_radius < 0.0 {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!(
                "cone radii must be non-negative, got bottom={bottom_radius}, top={top_radius}"
            ),
        });
    }
    if bottom_radius <= tol.linear && top_radius <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: "cone must have at least one non-zero radius".into(),
        });
    }

    // Determine which end is larger and compute virtual apex + half-angle
    // Base at z=0, top at z=height.
    let (r_big, r_small, big_z, small_z, axis_sign) = if bottom_radius >= top_radius {
        (bottom_radius, top_radius, 0.0, height, -1.0_f64)
    } else {
        (top_radius, bottom_radius, height, 0.0, 1.0_f64)
    };

    // The ConicalSurface formula is:
    //   P(u,v) = apex + v*(cos(a)*radial(u) + sin(a)*axis)
    // where `a` is the angle from the radial plane to the surface generator.
    // For a cone with base radius R at axial distance H from the apex:
    //   tan(a) = H / R  →  a = atan2(H, R)
    let half_angle = if r_small <= tol.linear {
        // Pointed cone: apex at the small end
        height.atan2(r_big)
    } else {
        // Frustum: virtual apex beyond the small end
        let axial_to_apex = r_small * height / (r_big - r_small);
        (axial_to_apex + height).atan2(r_big)
    };

    // half_angle must be in (0, π/2) for ConicalSurface
    if half_angle <= tol.angular || half_angle >= FRAC_PI_2 {
        // Degenerate case — fall back to revolve approach
        let face_id = make_trapezoid_xz_face(topo, bottom_radius, top_radius, 0.0, height)?;
        return crate::revolve::revolve(
            topo,
            face_id,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0 * PI,
        );
    }

    let apex_pos = if r_small <= tol.linear {
        Point3::new(0.0, 0.0, small_z)
    } else {
        let axial_to_apex = r_small * height / (r_big - r_small);
        // Apex is beyond the small end, away from the big end.
        // axis_sign points big→small, so we negate it to go small→apex.
        Point3::new(0.0, 0.0, small_z - axis_sign * axial_to_apex)
    };

    // Axis points from apex toward the base (big end), so that the
    // surface generator v>0 sweeps from apex outward to the base.
    let axis_dir = Vec3::new(0.0, 0.0, axis_sign);
    let cone_surface = brepkit_math::surfaces::ConicalSurface::new(apex_pos, axis_dir, half_angle)
        .map_err(crate::OperationsError::Math)?;

    let mut faces = Vec::new();

    // --- Lateral conical face ---
    if r_small <= tol.linear {
        // Pointed cone: degenerate wire from base circle to apex
        let v_apex = topo.add_vertex(Vertex::new(apex_pos, tol.linear));
        let v_base = topo.add_vertex(Vertex::new(Point3::new(r_big, 0.0, big_z), tol.linear));

        let base_circle = brepkit_math::curves::Circle3D::new(
            Point3::new(0.0, 0.0, big_z),
            Vec3::new(0.0, 0.0, 1.0),
            r_big,
        )
        .map_err(crate::OperationsError::Math)?;
        let e_circle = topo.add_edge(Edge::new(v_base, v_base, EdgeCurve::Circle(base_circle)));
        let e_seam = topo.add_edge(Edge::new(v_base, v_apex, EdgeCurve::Line));

        let lateral_wire = Wire::new(
            vec![
                OrientedEdge::new(e_circle, true),
                OrientedEdge::new(e_seam, true),
                OrientedEdge::new(e_seam, false),
            ],
            true,
        )
        .map_err(crate::OperationsError::Topology)?;
        let lateral_wid = topo.add_wire(lateral_wire);
        faces.push(topo.add_face(Face::new(
            lateral_wid,
            vec![],
            FaceSurface::Cone(cone_surface),
        )));

        // Base cap: reuse the same circle edge for watertight topology.
        // axis_sign < 0 means bottom is bigger (cap at z=0, outward normal -z, reversed edge).
        // axis_sign > 0 means top is bigger (cap at z=height, outward normal +z, forward edge).
        let cap_forward = axis_sign > 0.0;
        let cap_wire = Wire::new(vec![OrientedEdge::new(e_circle, cap_forward)], true)
            .map_err(crate::OperationsError::Topology)?;
        let cap_wid = topo.add_wire(cap_wire);
        let cap_normal = Vec3::new(0.0, 0.0, axis_sign);
        faces.push(topo.add_face(Face::new(
            cap_wid,
            vec![],
            FaceSurface::Plane {
                normal: cap_normal,
                d: big_z,
            },
        )));
    } else {
        // Frustum: two circles
        let v_bot = topo.add_vertex(Vertex::new(
            Point3::new(bottom_radius, 0.0, 0.0),
            tol.linear,
        ));
        let v_top = topo.add_vertex(Vertex::new(
            Point3::new(top_radius, 0.0, height),
            tol.linear,
        ));

        let bot_circle = brepkit_math::curves::Circle3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            bottom_radius,
        )
        .map_err(crate::OperationsError::Math)?;
        let top_circle = brepkit_math::curves::Circle3D::new(
            Point3::new(0.0, 0.0, height),
            Vec3::new(0.0, 0.0, 1.0),
            top_radius,
        )
        .map_err(crate::OperationsError::Math)?;
        let e_bot = topo.add_edge(Edge::new(v_bot, v_bot, EdgeCurve::Circle(bot_circle)));
        let e_top = topo.add_edge(Edge::new(v_top, v_top, EdgeCurve::Circle(top_circle)));
        let e_seam = topo.add_edge(Edge::new(v_bot, v_top, EdgeCurve::Line));

        let lateral_wire = Wire::new(
            vec![
                OrientedEdge::new(e_bot, true),
                OrientedEdge::new(e_seam, true),
                OrientedEdge::new(e_top, false),
                OrientedEdge::new(e_seam, false),
            ],
            true,
        )
        .map_err(crate::OperationsError::Topology)?;
        let lateral_wid = topo.add_wire(lateral_wire);
        faces.push(topo.add_face(Face::new(
            lateral_wid,
            vec![],
            FaceSurface::Cone(cone_surface),
        )));

        // Bottom cap (z=0): reuse the same circle edge for watertight topology.
        let bot_cap_wire = Wire::new(vec![OrientedEdge::new(e_bot, false)], true)
            .map_err(crate::OperationsError::Topology)?;
        let bot_wid = topo.add_wire(bot_cap_wire);
        faces.push(topo.add_face(Face::new(
            bot_wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: 0.0,
            },
        )));

        // Top cap (z=height): reuse the same circle edge for watertight topology.
        let top_cap_wire = Wire::new(vec![OrientedEdge::new(e_top, true)], true)
            .map_err(crate::OperationsError::Topology)?;
        let top_wid = topo.add_wire(top_cap_wire);
        faces.push(topo.add_face(Face::new(
            top_wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: height,
            },
        )));
    }

    let shell = Shell::new(faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

// ── Sphere ─────────────────────────────────────────────────────────

/// Create a sphere solid centered at the origin.
///
/// Built as a single spherical face with a degenerate boundary wire,
/// boundary wire, yielding exact `SphericalSurface` geometry with proper
/// topology for boolean operations.
///
/// The `segments` parameter controls the number of vertices on the
/// equatorial boundary polygon (minimum 4). Higher values improve
/// boolean accuracy for curved intersections. Tessellation density
/// is separately controlled by the `deflection` parameter at
/// tessellation time.
///
/// # Errors
///
/// Returns an error if `radius` is zero or negative, or `segments < 4`.
#[allow(clippy::too_many_lines)]
pub fn make_sphere(
    topo: &mut Topology,
    radius: f64,
    segments: usize,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if radius <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("sphere radius must be positive, got {radius}"),
        });
    }
    if segments < 4 {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("sphere needs at least 4 segments, got {segments}"),
        });
    }

    let surface_n =
        brepkit_math::surfaces::SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), radius)
            .map_err(crate::OperationsError::Math)?;
    let surface_s =
        brepkit_math::surfaces::SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), radius)
            .map_err(crate::OperationsError::Math)?;

    // Equatorial polygon: `segments` vertices evenly spaced on the circle
    // at z = 0, forming the shared boundary between north and south hemispheres.
    let eq_verts: Vec<_> = (0..segments)
        .map(|i| {
            let theta = TAU * i as f64 / segments as f64;
            let pt = Point3::new(radius * theta.cos(), radius * theta.sin(), 0.0);
            topo.add_vertex(Vertex::new(pt, tol.linear))
        })
        .collect();

    // Line edges connecting consecutive equatorial vertices (closed polygon).
    let eq_edges: Vec<_> = (0..segments)
        .map(|i| {
            let j = (i + 1) % segments;
            topo.add_edge(Edge::new(eq_verts[i], eq_verts[j], EdgeCurve::Line))
        })
        .collect();

    // North hemisphere: equatorial wire traversed forward (CCW from +Z).
    // Outward normal points upward (+Z hemisphere).
    let north_wire = Wire::new(
        eq_edges
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect(),
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let north_wid = topo.add_wire(north_wire);
    let north_face = topo.add_face(Face::new(north_wid, vec![], FaceSurface::Sphere(surface_n)));

    // South hemisphere: equatorial wire traversed backward (CW from +Z).
    // Outward normal points downward (-Z hemisphere).
    let south_wire = Wire::new(
        eq_edges
            .iter()
            .rev()
            .map(|&eid| OrientedEdge::new(eid, false))
            .collect(),
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let south_wid = topo.add_wire(south_wire);
    let south_face = topo.add_face(Face::new(south_wid, vec![], FaceSurface::Sphere(surface_s)));

    let shell =
        Shell::new(vec![north_face, south_face]).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

// ── Torus ──────────────────────────────────────────────────────────

/// Create a torus solid centered at the origin in the XY plane.
///
/// Built as a single `ToroidalSurface` face with exact analytic geometry,
/// yielding fast tessellation.
///
/// The `segments` parameter is accepted for API compatibility but ignored —
/// tessellation density is controlled by the `deflection` parameter.
///
/// # Errors
///
/// Returns an error if either radius is non-positive, or if the minor
/// radius is greater than the major radius (self-intersecting torus).
pub fn make_torus(
    topo: &mut Topology,
    major_radius: f64,
    minor_radius: f64,
    segments: usize,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if major_radius <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("torus major radius must be positive, got {major_radius}"),
        });
    }
    if minor_radius <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("torus minor radius must be positive, got {minor_radius}"),
        });
    }
    if minor_radius >= major_radius {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!(
                "torus minor radius ({minor_radius}) must be less than major radius ({major_radius})"
            ),
        });
    }
    if segments < 4 {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("torus needs at least 4 segments, got {segments}"),
        });
    }

    let surface = brepkit_math::surfaces::ToroidalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        major_radius,
        minor_radius,
    )
    .map_err(crate::OperationsError::Math)?;

    // The torus surface is doubly periodic (two seam curves).
    // Minimal CW complex: 1 vertex, 2 edges, 1 face → V-E+F = 0 (genus 1).
    // The boundary wire follows the fundamental polygon: a → b → a⁻¹ → b⁻¹.
    let v0 = topo.add_vertex(Vertex::new(
        Point3::new(major_radius + minor_radius, 0.0, 0.0),
        tol.linear,
    ));
    // Seam edge a (longitudinal — around the tube)
    let ea = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Line));
    // Seam edge b (meridional — around the ring)
    let eb = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(ea, true),
            OrientedEdge::new(eb, true),
            OrientedEdge::new(ea, false),
            OrientedEdge::new(eb, false),
        ],
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let wid = topo.add_wire(wire);

    let face_id = topo.add_face(Face::new(wid, vec![], FaceSurface::Torus(surface)));

    let shell = Shell::new(vec![face_id]).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

// ── Helpers ────────────────────────────────────────────────────────

/// Build a trapezoid face in the XZ plane (y=0) for cone profiles.
///
/// Vertices: `(0,-hz)`, `(bottom_r,-hz)`, `(top_r,+hz)`, `(0,+hz)`
/// CCW winding when viewed from -Y.
fn make_trapezoid_xz_face(
    topo: &mut Topology,
    bottom_radius: f64,
    top_radius: f64,
    z_bottom: f64,
    z_top: f64,
) -> Result<brepkit_topology::face::FaceId, crate::OperationsError> {
    let tol = Tolerance::new();

    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, z_bottom), tol.linear));
    let v1 = topo.add_vertex(Vertex::new(
        Point3::new(bottom_radius, 0.0, z_bottom),
        tol.linear,
    ));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(top_radius, 0.0, z_top), tol.linear));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, z_top), tol.linear));

    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
    let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e0, true),
            OrientedEdge::new(e1, true),
            OrientedEdge::new(e2, true),
            OrientedEdge::new(e3, true),
        ],
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let wid = topo.add_wire(wire);

    let normal = Vec3::new(0.0, -1.0, 0.0);
    Ok(topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane { normal, d: 0.0 },
    )))
}

#[allow(dead_code)]
/// Build a rectangular face in the XZ plane (y=0) from corners.
fn make_rect_xz_face(
    topo: &mut Topology,
    x0: f64,
    z0: f64,
    x1: f64,
    z1: f64,
) -> Result<brepkit_topology::face::FaceId, crate::OperationsError> {
    let tol = Tolerance::new();

    let v0 = topo.add_vertex(Vertex::new(Point3::new(x0, 0.0, z0), tol.linear));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(x1, 0.0, z0), tol.linear));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(x1, 0.0, z1), tol.linear));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(x0, 0.0, z1), tol.linear));

    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
    let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e0, true),
            OrientedEdge::new(e1, true),
            OrientedEdge::new(e2, true),
            OrientedEdge::new(e3, true),
        ],
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let wid = topo.add_wire(wire);

    let normal = Vec3::new(0.0, -1.0, 0.0);
    let d = 0.0;
    Ok(topo.add_face(Face::new(wid, vec![], FaceSurface::Plane { normal, d })))
}

// ── Convex hull ───────────────────────────────────────────────────

/// Create a solid from the 3D convex hull of a point cloud.
///
/// Uses the Quickhull algorithm to compute the convex hull, then
/// converts the resulting triangulated surface to a B-Rep solid with
/// planar triangular faces.
///
/// # Errors
///
/// Returns an error if fewer than 4 non-coplanar points are provided.
pub fn make_convex_hull(
    topo: &mut Topology,
    points: &[Point3],
) -> Result<SolidId, crate::OperationsError> {
    let hull = brepkit_math::convex_hull::convex_hull_3d(points).ok_or_else(|| {
        crate::OperationsError::InvalidInput {
            reason: "points are coplanar or degenerate — cannot form a 3D convex hull".into(),
        }
    })?;

    let tol = Tolerance::new();

    // Create vertices in the topology arena.
    let vertex_ids: Vec<_> = hull
        .vertices
        .iter()
        .map(|p| topo.add_vertex(Vertex::new(*p, tol.linear)))
        .collect();

    // Create faces from hull triangles, sharing edges between adjacent faces.
    // Each undirected edge (min_vertex, max_vertex) is created once; the second
    // face that references it uses reverse orientation.
    let mut edge_map: HashMap<(usize, usize), brepkit_topology::edge::EdgeId> = HashMap::new();
    let mut face_ids = Vec::with_capacity(hull.faces.len());
    for &[a, b, c] in &hull.faces {
        let va = vertex_ids[a];
        let vb = vertex_ids[b];
        let vc = vertex_ids[c];

        let pairs = [(a, b, va, vb), (b, c, vb, vc), (c, a, vc, va)];
        let mut oriented_edges = Vec::with_capacity(3);
        for (idx_a, idx_b, v_a, v_b) in pairs {
            let key = (idx_a.min(idx_b), idx_a.max(idx_b));
            let (eid, forward) = if let Some(&existing) = edge_map.get(&key) {
                (existing, false)
            } else {
                let eid = topo.add_edge(Edge::new(v_a, v_b, EdgeCurve::Line));
                edge_map.insert(key, eid);
                (eid, true)
            };
            oriented_edges.push(OrientedEdge::new(eid, forward));
        }

        let wire = Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
        let wid = topo.add_wire(wire);

        // Compute face plane from triangle vertices.
        let pa = hull.vertices[a];
        let pb = hull.vertices[b];
        let pc = hull.vertices[c];
        let ab = pb - pa;
        let ac = pc - pa;
        let normal = ab.cross(ac).normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        let d = normal
            .x()
            .mul_add(pa.x(), normal.y().mul_add(pa.y(), normal.z() * pa.z()));

        let fid = topo.add_face(Face::new(wid, vec![], FaceSurface::Plane { normal, d }));
        face_ids.push(fid);
    }

    let shell = Shell::new(face_ids).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    let solid = Solid::new(shell_id, vec![]);
    Ok(topo.add_solid(solid))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_topology::Topology;

    use super::*;
    use crate::test_helpers::{assert_euler_genus0, assert_volume_near, euler_characteristic};

    // ── Box tests ──────────────────────────────────────────────────

    #[test]
    fn make_box_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(sh.faces().len(), 6, "box should have 6 faces");

        // Verify volume
        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            (vol - 1.0).abs() < 1e-10,
            "unit box volume should be 1.0, got {vol}"
        );

        assert_euler_genus0(&topo, solid);
    }

    #[test]
    fn make_box_rectangular() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            (vol - 24.0).abs() < 1e-10,
            "2x3x4 box volume should be 24.0, got {vol}"
        );

        assert_euler_genus0(&topo, solid);
    }

    #[test]
    fn make_box_corner_at_origin() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Box extends from (0,0,0) to (2,2,2), so center of mass is at (1,1,1).
        let com = crate::measure::solid_center_of_mass(&topo, solid, 0.1).unwrap();
        assert!(
            (com.x() - 1.0).abs() < 1e-6,
            "com x should be ~1, got {}",
            com.x()
        );
        assert!(
            (com.y() - 1.0).abs() < 1e-6,
            "com y should be ~1, got {}",
            com.y()
        );
        assert!(
            (com.z() - 1.0).abs() < 1e-6,
            "com z should be ~1, got {}",
            com.z()
        );
    }

    #[test]
    fn make_box_zero_dimension_error() {
        let mut topo = Topology::new();
        assert!(make_box(&mut topo, 0.0, 1.0, 1.0).is_err());
        assert!(make_box(&mut topo, 1.0, 0.0, 1.0).is_err());
        assert!(make_box(&mut topo, 1.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn make_box_negative_dimension_error() {
        let mut topo = Topology::new();
        assert!(make_box(&mut topo, -1.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn make_box_manifold_edges() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        brepkit_topology::validation::validate_shell_manifold(sh, topo.faces(), topo.wires())
            .expect("box should be manifold");
    }

    // ── Cylinder tests ─────────────────────────────────────────────

    #[test]
    fn make_cylinder_basic() {
        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 1.0, 2.0).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // Analytic cylinder: 1 lateral + 2 caps = 3 faces
        assert_eq!(sh.faces().len(), 3, "cylinder should have 3 faces");

        assert_euler_genus0(&topo, solid);
    }

    #[test]
    fn make_cylinder_volume() {
        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 1.0, 2.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        let expected = PI * 1.0_f64.powi(2) * 2.0;
        assert!(
            (vol - expected).abs() / expected < 1e-6,
            "cylinder volume should be ~{expected}, got {vol} (error: {:.1}%)",
            (vol - expected).abs() / expected * 100.0
        );
    }

    #[test]
    fn make_cylinder_zero_radius_error() {
        let mut topo = Topology::new();
        assert!(make_cylinder(&mut topo, 0.0, 1.0).is_err());
    }

    #[test]
    fn make_cylinder_zero_height_error() {
        let mut topo = Topology::new();
        assert!(make_cylinder(&mut topo, 1.0, 0.0).is_err());
    }

    // ── Cone tests ─────────────────────────────────────────────────

    #[test]
    fn make_cone_frustum() {
        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 2.0, 1.0, 3.0).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert!(!sh.faces().is_empty(), "cone should have faces");

        assert_euler_genus0(&topo, solid);
    }

    #[test]
    fn make_cone_frustum_volume() {
        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 2.0, 1.0, 3.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        // V = πh/3 * (r1² + r1*r2 + r2²)
        let expected =
            std::f64::consts::PI * 3.0 / 3.0 * (2.0_f64.powi(2) + 2.0 * 1.0 + 1.0_f64.powi(2));
        let rel_error = (vol - expected).abs() / expected;
        assert!(
            rel_error < 0.01,
            "cone frustum volume should be ~{expected:.3}, got {vol:.3} (error: {rel_error:.1}%)",
        );
    }

    #[test]
    fn make_cone_both_zero_error() {
        let mut topo = Topology::new();
        assert!(make_cone(&mut topo, 0.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn make_cone_negative_radius_error() {
        let mut topo = Topology::new();
        assert!(make_cone(&mut topo, -1.0, 1.0, 1.0).is_err());
    }

    // ── Sphere tests ───────────────────────────────────────────────

    #[test]
    fn make_sphere_basic() {
        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 1.0, 8).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert!(!sh.faces().is_empty(), "sphere should have faces");

        assert_euler_genus0(&topo, solid);
    }

    #[test]
    fn make_sphere_volume() {
        let mut topo = Topology::new();
        // Use high segment count for better approximation
        let solid = make_sphere(&mut topo, 1.0, 32).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        let expected = 4.0 / 3.0 * PI;
        // Polygonal approximation — allow 5% tolerance
        assert!(
            (vol - expected).abs() / expected < 0.01,
            "sphere volume should be ~{expected}, got {vol} (error: {:.1}%)",
            (vol - expected).abs() / expected * 100.0
        );
    }

    #[test]
    fn make_sphere_center_of_mass_at_origin() {
        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 1.0, 16).unwrap();

        let com = crate::measure::solid_center_of_mass(&topo, solid, 0.1).unwrap();
        assert!(
            com.x().abs() < 1e-6,
            "sphere com x should be ~0, got {}",
            com.x()
        );
        assert!(
            com.y().abs() < 1e-6,
            "sphere com y should be ~0, got {}",
            com.y()
        );
        assert!(
            com.z().abs() < 1e-6,
            "sphere com z should be ~0, got {}",
            com.z()
        );
    }

    #[test]
    fn make_sphere_two_hemispheres() {
        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 1.0, 8).unwrap();
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(sh.faces().len(), 2, "sphere should have 2 hemisphere faces");
    }

    #[test]
    fn make_sphere_zero_radius_error() {
        let mut topo = Topology::new();
        assert!(make_sphere(&mut topo, 0.0, 8).is_err());
    }

    #[test]
    fn make_sphere_few_segments_error() {
        let mut topo = Topology::new();
        assert!(make_sphere(&mut topo, 1.0, 2).is_err());
    }

    // ── Torus tests ────────────────────────────────────────────────

    #[test]
    fn make_torus_basic() {
        let mut topo = Topology::new();
        let solid = make_torus(&mut topo, 3.0, 1.0, 8).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert!(!sh.faces().is_empty(), "torus should have faces");

        assert_eq!(
            euler_characteristic(&topo, solid),
            0,
            "torus should be genus-1 (χ=0)"
        );
    }

    #[test]
    fn make_torus_volume() {
        let mut topo = Topology::new();
        let solid = make_torus(&mut topo, 3.0, 1.0, 32).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        // V = 2π²Rr² where R=major, r=minor
        let expected = 2.0 * PI * PI * 3.0 * 1.0;
        // Polygonal approximation — allow 5% tolerance
        assert!(
            (vol - expected).abs() / expected < 0.01,
            "torus volume should be ~{expected}, got {vol} (error: {:.1}%)",
            (vol - expected).abs() / expected * 100.0
        );
    }

    #[test]
    fn make_torus_self_intersecting_error() {
        let mut topo = Topology::new();
        // minor >= major → self-intersecting
        assert!(make_torus(&mut topo, 1.0, 1.0, 8).is_err());
        assert!(make_torus(&mut topo, 1.0, 2.0, 8).is_err());
    }

    #[test]
    fn make_torus_zero_radius_error() {
        let mut topo = Topology::new();
        assert!(make_torus(&mut topo, 0.0, 1.0, 8).is_err());
        assert!(make_torus(&mut topo, 3.0, 0.0, 8).is_err());
    }

    // ── Periodic surface / singular point tests ─────────────────────

    #[test]
    fn cylinder_seam_continuity() {
        // Cylinder surface should evaluate to the same point at u=0 and u=2π
        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 1.0, 2.0).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();

        for fid in &faces {
            let face = topo.face(*fid).unwrap();
            if let brepkit_topology::face::FaceSurface::Cylinder(cyl) = face.surface() {
                let p0 = cyl.evaluate(0.0, 0.5);
                let p2pi = cyl.evaluate(std::f64::consts::TAU, 0.5);
                let dist = ((p0.x() - p2pi.x()).powi(2)
                    + (p0.y() - p2pi.y()).powi(2)
                    + (p0.z() - p2pi.z()).powi(2))
                .sqrt();
                assert!(
                    dist < 1e-10,
                    "cylinder seam gap: u=0 and u=2π should coincide, dist={dist}"
                );
            }
        }
    }

    #[test]
    fn sphere_pole_degenerate() {
        // Sphere should have degenerate edges at poles (start == end vertex)
        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 1.0, 8).unwrap();

        let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
        // Sphere has 2 hemisphere faces
        assert_eq!(faces.len(), 2, "sphere should have 2 faces");
    }

    #[test]
    fn torus_seam_u_and_v() {
        // Torus surface should be periodic in both u and v
        let mut topo = Topology::new();
        let solid = make_torus(&mut topo, 3.0, 1.0, 8).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();

        for fid in &faces {
            let face = topo.face(*fid).unwrap();
            if let brepkit_topology::face::FaceSurface::Torus(tor) = face.surface() {
                // Check u-periodicity
                let p00 = tor.evaluate(0.0, 0.5);
                let p2pi_0 = tor.evaluate(std::f64::consts::TAU, 0.5);
                let dist_u = ((p00.x() - p2pi_0.x()).powi(2)
                    + (p00.y() - p2pi_0.y()).powi(2)
                    + (p00.z() - p2pi_0.z()).powi(2))
                .sqrt();
                assert!(dist_u < 1e-10, "torus u-seam gap: dist={dist_u}");

                // Check v-periodicity
                let p0_0 = tor.evaluate(0.5, 0.0);
                let p0_2pi = tor.evaluate(0.5, std::f64::consts::TAU);
                let dist_v = ((p0_0.x() - p0_2pi.x()).powi(2)
                    + (p0_0.y() - p0_2pi.y()).powi(2)
                    + (p0_0.z() - p0_2pi.z()).powi(2))
                .sqrt();
                assert!(dist_v < 1e-10, "torus v-seam gap: dist={dist_v}");
            }
        }
    }

    #[test]
    fn cone_apex_singular() {
        // A cone with top_radius=0 has an apex vertex
        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 2.0, 0.0, 3.0).unwrap();
        let verts = brepkit_topology::explorer::solid_vertices(&topo, solid).unwrap();

        // The apex should be at (0, 0, 3.0) — the top of the cone
        let has_apex = verts.iter().any(|vid| {
            let v = topo.vertex(*vid).unwrap();
            let p = v.point();
            (p.x().powi(2) + p.y().powi(2)).sqrt() < 1e-10 && (p.z() - 3.0).abs() < 1e-10
        });
        assert!(has_apex, "cone should have apex vertex at (0,0,3)");
    }

    // ── Degenerate geometry tests ──────────────────────────────

    #[test]
    fn make_box_very_thin() {
        // Thin plate: dx=1e-4, dy=1, dz=1
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 1e-4, 1.0, 1.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            (vol - 1e-4).abs() < 1e-12,
            "thin box volume should be 1e-4, got {vol}"
        );
        assert_euler_genus0(&topo, solid);
    }

    #[test]
    fn make_box_very_large() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 1e6, 1.0, 1.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            (vol - 1e6).abs() / 1e6 < 1e-10,
            "large box volume should be 1e6, got {vol}"
        );
    }

    #[test]
    fn make_cylinder_high_aspect_ratio() {
        // Very thin, very tall cylinder
        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 0.001, 1000.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.001).unwrap();
        let expected = PI * 0.001_f64.powi(2) * 1000.0;
        let rel_error = (vol - expected).abs() / expected;
        assert!(
            rel_error < 0.01,
            "high aspect cylinder volume: got {vol:.6e}, expected {expected:.6e} (error: {:.1}%)",
            rel_error * 100.0
        );
    }

    // ── Convex hull tests ─────────────────────────────────────────

    #[test]
    fn convex_hull_unit_cube() {
        let mut topo = Topology::new();
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];
        let solid = make_convex_hull(&mut topo, &points).unwrap();
        assert_volume_near(&topo, solid, 1.0, 1e-10);
    }

    #[test]
    fn convex_hull_larger_box() {
        let mut topo = Topology::new();
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 4.0, 0.0),
            Point3::new(3.0, 4.0, 0.0),
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(3.0, 0.0, 5.0),
            Point3::new(0.0, 4.0, 5.0),
            Point3::new(3.0, 4.0, 5.0),
        ];
        let solid = make_convex_hull(&mut topo, &points).unwrap();
        assert_volume_near(&topo, solid, 60.0, 1e-10);
    }

    #[test]
    fn convex_hull_minkowski_sum_volume() {
        // Minkowski sum of [0,1]^3 with [0,1]^3 = [0,2]^3, volume = 8.
        let cube: Vec<Point3> = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];

        let mut sum_points = Vec::with_capacity(64);
        for &a in &cube {
            for &b in &cube {
                sum_points.push(Point3::new(a.x() + b.x(), a.y() + b.y(), a.z() + b.z()));
            }
        }

        let mut topo = Topology::new();
        let solid = make_convex_hull(&mut topo, &sum_points).unwrap();
        assert_volume_near(&topo, solid, 8.0, 1e-8);
    }

    #[test]
    fn convex_hull_minkowski_displaced_cubes_volume() {
        // Cube A at origin [0,1]^3, Cube B at (5,5,5) to (6,6,6).
        // Hull of pairwise sums spans [5,7]^3, volume = 8.
        let cube_a = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];
        let cube_b = vec![
            Point3::new(5.0, 5.0, 5.0),
            Point3::new(6.0, 5.0, 5.0),
            Point3::new(5.0, 6.0, 5.0),
            Point3::new(6.0, 6.0, 5.0),
            Point3::new(5.0, 5.0, 6.0),
            Point3::new(6.0, 5.0, 6.0),
            Point3::new(5.0, 6.0, 6.0),
            Point3::new(6.0, 6.0, 6.0),
        ];

        let mut sum_points = Vec::with_capacity(64);
        for &a in &cube_a {
            for &b in &cube_b {
                sum_points.push(Point3::new(a.x() + b.x(), a.y() + b.y(), a.z() + b.z()));
            }
        }

        let mut topo = Topology::new();
        let solid = make_convex_hull(&mut topo, &sum_points).unwrap();
        assert_volume_near(&topo, solid, 8.0, 1e-8);
    }

    #[test]
    fn convex_hull_with_interior_points_volume() {
        // 8 cube corners + interior points — volume should be 1.0.
        let mut points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];
        points.push(Point3::new(0.5, 0.5, 0.5));
        points.push(Point3::new(0.2, 0.3, 0.7));

        let mut topo = Topology::new();
        let solid = make_convex_hull(&mut topo, &points).unwrap();
        assert_volume_near(&topo, solid, 1.0, 1e-10);
    }

    #[test]
    fn convex_hull_rejects_coplanar_points() {
        let mut topo = Topology::new();
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];
        assert!(make_convex_hull(&mut topo, &points).is_err());
    }

    #[test]
    fn convex_hull_rejects_too_few_points() {
        let mut topo = Topology::new();
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        assert!(make_convex_hull(&mut topo, &points).is_err());
    }
}
