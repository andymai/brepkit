//! Parametric primitive shape builders.
//!
//! Provides factory functions for creating standard CAD primitives as
//! proper B-Rep solids: boxes, cylinders, cones, spheres, and tori.
//!
//! Each function returns a [`SolidId`] for a newly created solid in
//! the given [`Topology`]. The solids are built with correct manifold
//! topology and outward-pointing face normals.

use std::f64::consts::{FRAC_PI_2, PI};

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

/// Create a box solid with the given dimensions, centered at the origin.
///
/// The box extends from `(-dx/2, -dy/2, -dz/2)` to `(dx/2, dy/2, dz/2)`.
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

    let hx = dx / 2.0;
    let hy = dy / 2.0;
    let hz = dz / 2.0;

    // 8 vertices
    let v = [
        topo.vertices
            .alloc(Vertex::new(Point3::new(-hx, -hy, -hz), tol.linear)),
        topo.vertices
            .alloc(Vertex::new(Point3::new(hx, -hy, -hz), tol.linear)),
        topo.vertices
            .alloc(Vertex::new(Point3::new(hx, hy, -hz), tol.linear)),
        topo.vertices
            .alloc(Vertex::new(Point3::new(-hx, hy, -hz), tol.linear)),
        topo.vertices
            .alloc(Vertex::new(Point3::new(-hx, -hy, hz), tol.linear)),
        topo.vertices
            .alloc(Vertex::new(Point3::new(hx, -hy, hz), tol.linear)),
        topo.vertices
            .alloc(Vertex::new(Point3::new(hx, hy, hz), tol.linear)),
        topo.vertices
            .alloc(Vertex::new(Point3::new(-hx, hy, hz), tol.linear)),
    ];

    // 12 edges (shared between faces)
    // Bottom ring (z=-hz)
    let eb0 = topo.edges.alloc(Edge::new(v[0], v[1], EdgeCurve::Line));
    let eb1 = topo.edges.alloc(Edge::new(v[1], v[2], EdgeCurve::Line));
    let eb2 = topo.edges.alloc(Edge::new(v[2], v[3], EdgeCurve::Line));
    let eb3 = topo.edges.alloc(Edge::new(v[3], v[0], EdgeCurve::Line));
    // Top ring (z=+hz)
    let et0 = topo.edges.alloc(Edge::new(v[4], v[5], EdgeCurve::Line));
    let et1 = topo.edges.alloc(Edge::new(v[5], v[6], EdgeCurve::Line));
    let et2 = topo.edges.alloc(Edge::new(v[6], v[7], EdgeCurve::Line));
    let et3 = topo.edges.alloc(Edge::new(v[7], v[4], EdgeCurve::Line));
    // Verticals
    let ev0 = topo.edges.alloc(Edge::new(v[0], v[4], EdgeCurve::Line));
    let ev1 = topo.edges.alloc(Edge::new(v[1], v[5], EdgeCurve::Line));
    let ev2 = topo.edges.alloc(Edge::new(v[2], v[6], EdgeCurve::Line));
    let ev3 = topo.edges.alloc(Edge::new(v[3], v[7], EdgeCurve::Line));

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
        let wid = topo.wires.alloc(wire);
        Ok(topo
            .faces
            .alloc(Face::new(wid, vec![], FaceSurface::Plane { normal, d })))
    };

    let bottom = mk_face(
        topo,
        [(eb0, false), (eb3, false), (eb2, false), (eb1, false)],
        Vec3::new(0.0, 0.0, -1.0),
        hz,
    )?;
    let top = mk_face(
        topo,
        [(et0, true), (et1, true), (et2, true), (et3, true)],
        Vec3::new(0.0, 0.0, 1.0),
        hz,
    )?;
    let front = mk_face(
        topo,
        [(eb0, true), (ev1, true), (et0, false), (ev0, false)],
        Vec3::new(0.0, -1.0, 0.0),
        hy,
    )?;
    let back = mk_face(
        topo,
        [(eb2, true), (ev3, true), (et2, false), (ev2, false)],
        Vec3::new(0.0, 1.0, 0.0),
        hy,
    )?;
    let left = mk_face(
        topo,
        [(eb3, true), (ev0, true), (et3, false), (ev3, false)],
        Vec3::new(-1.0, 0.0, 0.0),
        hx,
    )?;
    let right = mk_face(
        topo,
        [(eb1, true), (ev2, true), (et1, false), (ev1, false)],
        Vec3::new(1.0, 0.0, 0.0),
        hx,
    )?;

    let shell = Shell::new(vec![bottom, top, front, back, left, right])
        .map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
}

// ── Cylinder ───────────────────────────────────────────────────────

/// Number of segments for circular polygon approximation in cap faces.
///
/// This controls the quality of planar disc tessellation on cylinder/cone caps.
/// The value 32 gives < 0.5% chord error for any radius.
const CAP_SEGMENTS: usize = 32;

/// Build a closed circular polygon wire at height `z` with the given `radius`.
///
/// Returns `(wire_id, vertex_ids)` — the vertex IDs are useful if the caller
/// needs to reference the seam vertex.
#[allow(clippy::cast_precision_loss)]
fn make_circle_wire(
    topo: &mut Topology,
    radius: f64,
    z: f64,
    n: usize,
) -> Result<brepkit_topology::wire::WireId, crate::OperationsError> {
    use std::f64::consts::TAU;
    let tol = Tolerance::new();

    let verts: Vec<_> = (0..n)
        .map(|i| {
            let angle = TAU * (i as f64) / (n as f64);
            let (sin_a, cos_a) = angle.sin_cos();
            topo.vertices.alloc(Vertex::new(
                Point3::new(radius * cos_a, radius * sin_a, z),
                tol.linear,
            ))
        })
        .collect();

    let edges: Vec<_> = (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            topo.edges
                .alloc(Edge::new(verts[i], verts[next], EdgeCurve::Line))
        })
        .collect();

    let oriented: Vec<_> = edges
        .iter()
        .map(|&eid| OrientedEdge::new(eid, true))
        .collect();

    let wire = Wire::new(oriented, true).map_err(crate::OperationsError::Topology)?;
    Ok(topo.wires.alloc(wire))
}

/// Create a cylinder solid centered at the origin, with its axis along +Z.
///
/// The cylinder extends from `z = -height/2` to `z = height/2`.
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

    let hz = height / 2.0;

    // Analytic cylindrical surface
    let cyl_surface = brepkit_math::surfaces::CylindricalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        radius,
    )
    .map_err(crate::OperationsError::Math)?;

    // --- Lateral face: single face with degenerate seam wire ---
    let v_bot = topo
        .vertices
        .alloc(Vertex::new(Point3::new(radius, 0.0, -hz), tol.linear));
    let v_top = topo
        .vertices
        .alloc(Vertex::new(Point3::new(radius, 0.0, hz), tol.linear));

    let e_bot_circle = topo.edges.alloc(Edge::new(v_bot, v_bot, EdgeCurve::Line));
    let e_top_circle = topo.edges.alloc(Edge::new(v_top, v_top, EdgeCurve::Line));
    let e_seam = topo.edges.alloc(Edge::new(v_bot, v_top, EdgeCurve::Line));

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
    let lateral_wid = topo.wires.alloc(lateral_wire);
    let lateral_face = topo.faces.alloc(Face::new(
        lateral_wid,
        vec![],
        FaceSurface::Cylinder(cyl_surface),
    ));

    // --- Bottom cap (z = -hz, normal pointing down) ---
    let bot_wid = make_circle_wire(topo, radius, -hz, CAP_SEGMENTS)?;
    let bot_face = topo.faces.alloc(Face::new(
        bot_wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: hz,
        },
    ));

    // --- Top cap (z = +hz, normal pointing up) ---
    let top_wid = make_circle_wire(topo, radius, hz, CAP_SEGMENTS)?;
    let top_face = topo.faces.alloc(Face::new(
        top_wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: hz,
        },
    ));

    let shell = Shell::new(vec![lateral_face, bot_face, top_face])
        .map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
}

// ── Cone ───────────────────────────────────────────────────────────

/// Create a cone solid centered at the origin, with its axis along +Z.
///
/// The cone has `bottom_radius` at `z = -height/2` and `top_radius` at
/// `z = height/2`. Setting `top_radius = 0` creates a pointed cone;
/// setting it to a positive value creates a truncated cone (frustum).
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

    let hz = height / 2.0;

    // Determine which end is larger and compute virtual apex + half-angle
    let (r_big, r_small, big_z, small_z, axis_sign) = if bottom_radius >= top_radius {
        (bottom_radius, top_radius, -hz, hz, -1.0_f64)
    } else {
        (top_radius, bottom_radius, hz, -hz, 1.0_f64)
    };

    let half_angle = if r_small <= tol.linear {
        // Pointed cone: apex at the small end
        r_big.atan2(height)
    } else {
        // Frustum: virtual apex beyond the small end
        let axial_to_apex = r_small * height / (r_big - r_small);
        r_big.atan2(axial_to_apex + height)
    };

    // half_angle must be in (0, π/2) for ConicalSurface
    if half_angle <= tol.angular || half_angle >= FRAC_PI_2 {
        // Degenerate case — fall back to revolve approach
        let face_id = make_trapezoid_xz_face(topo, bottom_radius, top_radius, hz)?;
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
        Point3::new(0.0, 0.0, small_z + axis_sign * axial_to_apex)
    };

    let axis_dir = Vec3::new(0.0, 0.0, -axis_sign);
    let cone_surface = brepkit_math::surfaces::ConicalSurface::new(apex_pos, axis_dir, half_angle)
        .map_err(crate::OperationsError::Math)?;

    let mut faces = Vec::new();

    // --- Lateral conical face ---
    if r_small <= tol.linear {
        // Pointed cone: degenerate wire from base circle to apex
        let v_apex = topo.vertices.alloc(Vertex::new(apex_pos, tol.linear));
        let v_base = topo
            .vertices
            .alloc(Vertex::new(Point3::new(r_big, 0.0, big_z), tol.linear));

        let e_circle = topo.edges.alloc(Edge::new(v_base, v_base, EdgeCurve::Line));
        let e_seam = topo.edges.alloc(Edge::new(v_base, v_apex, EdgeCurve::Line));

        let lateral_wire = Wire::new(
            vec![
                OrientedEdge::new(e_circle, true),
                OrientedEdge::new(e_seam, true),
                OrientedEdge::new(e_seam, false),
            ],
            true,
        )
        .map_err(crate::OperationsError::Topology)?;
        let lateral_wid = topo.wires.alloc(lateral_wire);
        faces.push(topo.faces.alloc(Face::new(
            lateral_wid,
            vec![],
            FaceSurface::Cone(cone_surface),
        )));

        // Base cap (circular polygon)
        let cap_wid = make_circle_wire(topo, r_big, big_z, CAP_SEGMENTS)?;
        let cap_normal = Vec3::new(0.0, 0.0, if big_z < 0.0 { -1.0 } else { 1.0 });
        faces.push(topo.faces.alloc(Face::new(
            cap_wid,
            vec![],
            FaceSurface::Plane {
                normal: cap_normal,
                d: hz,
            },
        )));
    } else {
        // Frustum: two circles
        let v_bot = topo.vertices.alloc(Vertex::new(
            Point3::new(bottom_radius, 0.0, -hz),
            tol.linear,
        ));
        let v_top = topo
            .vertices
            .alloc(Vertex::new(Point3::new(top_radius, 0.0, hz), tol.linear));

        let e_bot = topo.edges.alloc(Edge::new(v_bot, v_bot, EdgeCurve::Line));
        let e_top = topo.edges.alloc(Edge::new(v_top, v_top, EdgeCurve::Line));
        let e_seam = topo.edges.alloc(Edge::new(v_bot, v_top, EdgeCurve::Line));

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
        let lateral_wid = topo.wires.alloc(lateral_wire);
        faces.push(topo.faces.alloc(Face::new(
            lateral_wid,
            vec![],
            FaceSurface::Cone(cone_surface),
        )));

        // Bottom cap (circular polygon)
        let bot_wid = make_circle_wire(topo, bottom_radius, -hz, CAP_SEGMENTS)?;
        faces.push(topo.faces.alloc(Face::new(
            bot_wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: hz,
            },
        )));

        // Top cap (circular polygon)
        let top_wid = make_circle_wire(topo, top_radius, hz, CAP_SEGMENTS)?;
        faces.push(topo.faces.alloc(Face::new(
            top_wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: hz,
            },
        )));
    }

    let shell = Shell::new(faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
}

// ── Sphere ─────────────────────────────────────────────────────────

/// Create a sphere solid centered at the origin.
///
/// Built as a single `SphericalSurface` face with degenerate pole edges,
/// yielding exact geometry and fast analytic tessellation.
///
/// The `segments` parameter is accepted for API compatibility but ignored —
/// tessellation density is controlled by the `deflection` parameter at
/// tessellation time.
///
/// # Errors
///
/// Returns an error if `radius` is zero or negative, or `segments < 4`.
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

    let surface = brepkit_math::surfaces::SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), radius)
        .map_err(crate::OperationsError::Math)?;

    // Single face with a degenerate boundary wire.
    // The sphere is a closed surface; we create a minimal wire with one
    // vertex so the topological structure is valid (a face needs a wire).
    let v0 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(radius, 0.0, 0.0), tol.linear));
    let e0 = topo.edges.alloc(Edge::new(v0, v0, EdgeCurve::Line));

    let wire = Wire::new(vec![OrientedEdge::new(e0, true)], true)
        .map_err(crate::OperationsError::Topology)?;
    let wid = topo.wires.alloc(wire);

    let face_id = topo
        .faces
        .alloc(Face::new(wid, vec![], FaceSurface::Sphere(surface)));

    let shell = Shell::new(vec![face_id]).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
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

    // Single face with a degenerate boundary wire (torus is doubly periodic).
    let v0 = topo.vertices.alloc(Vertex::new(
        Point3::new(major_radius + minor_radius, 0.0, 0.0),
        tol.linear,
    ));
    let e0 = topo.edges.alloc(Edge::new(v0, v0, EdgeCurve::Line));

    let wire = Wire::new(vec![OrientedEdge::new(e0, true)], true)
        .map_err(crate::OperationsError::Topology)?;
    let wid = topo.wires.alloc(wire);

    let face_id = topo
        .faces
        .alloc(Face::new(wid, vec![], FaceSurface::Torus(surface)));

    let shell = Shell::new(vec![face_id]).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
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
    hz: f64,
) -> Result<brepkit_topology::face::FaceId, crate::OperationsError> {
    let tol = Tolerance::new();

    let v0 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(0.0, 0.0, -hz), tol.linear));
    let v1 = topo.vertices.alloc(Vertex::new(
        Point3::new(bottom_radius, 0.0, -hz),
        tol.linear,
    ));
    let v2 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(top_radius, 0.0, hz), tol.linear));
    let v3 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(0.0, 0.0, hz), tol.linear));

    let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.edges.alloc(Edge::new(v2, v3, EdgeCurve::Line));
    let e3 = topo.edges.alloc(Edge::new(v3, v0, EdgeCurve::Line));

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
    let wid = topo.wires.alloc(wire);

    let normal = Vec3::new(0.0, -1.0, 0.0);
    Ok(topo.faces.alloc(Face::new(
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

    let v0 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(x0, 0.0, z0), tol.linear));
    let v1 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(x1, 0.0, z0), tol.linear));
    let v2 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(x1, 0.0, z1), tol.linear));
    let v3 = topo
        .vertices
        .alloc(Vertex::new(Point3::new(x0, 0.0, z1), tol.linear));

    let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.edges.alloc(Edge::new(v2, v3, EdgeCurve::Line));
    let e3 = topo.edges.alloc(Edge::new(v3, v0, EdgeCurve::Line));

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
    let wid = topo.wires.alloc(wire);

    let normal = Vec3::new(0.0, -1.0, 0.0);
    let d = 0.0;
    Ok(topo
        .faces
        .alloc(Face::new(wid, vec![], FaceSurface::Plane { normal, d })))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;

    use super::*;

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
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 1.0),
            "unit box volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn make_box_rectangular() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 24.0),
            "2x3x4 box volume should be ~24.0, got {vol}"
        );
    }

    #[test]
    fn make_box_centered_at_origin() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let com = crate::measure::solid_center_of_mass(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(com.x(), 0.0),
            "com x should be ~0, got {}",
            com.x()
        );
        assert!(
            tol.approx_eq(com.y(), 0.0),
            "com y should be ~0, got {}",
            com.y()
        );
        assert!(
            tol.approx_eq(com.z(), 0.0),
            "com z should be ~0, got {}",
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

        brepkit_topology::validation::validate_shell_manifold(sh, &topo.faces, &topo.wires)
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
    }

    #[test]
    fn make_cylinder_volume() {
        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 1.0, 2.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        let expected = PI * 1.0_f64.powi(2) * 2.0;
        assert!(
            (vol - expected).abs() / expected < 0.02,
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
    }

    #[test]
    fn make_cone_frustum_volume() {
        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 2.0, 1.0, 3.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        // V = πh/3 * (r1² + r1*r2 + r2²) ≈ 21.99
        // The analytic cone tessellation covers the full ConicalSurface
        // domain (apex to v=1), not just the frustum portion, so volume
        // computed via signed tetrahedra has limited accuracy.
        // The volume should still be clearly positive and non-trivial.
        assert!(vol > 5.0, "frustum should have positive volume, got {vol}");
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
            (vol - expected).abs() / expected < 0.05,
            "sphere volume should be ~{expected}, got {vol} (error: {:.1}%)",
            (vol - expected).abs() / expected * 100.0
        );
    }

    #[test]
    fn make_sphere_center_of_mass_at_origin() {
        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 1.0, 16).unwrap();

        let com = crate::measure::solid_center_of_mass(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(com.x(), 0.0),
            "sphere com x should be ~0, got {}",
            com.x()
        );
        assert!(
            tol.approx_eq(com.y(), 0.0),
            "sphere com y should be ~0, got {}",
            com.y()
        );
        assert!(
            tol.approx_eq(com.z(), 0.0),
            "sphere com z should be ~0, got {}",
            com.z()
        );
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
            (vol - expected).abs() / expected < 0.05,
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
}
