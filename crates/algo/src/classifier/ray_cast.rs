//! Ray-cast point-in-solid classification (canonical implementation).
//!
//! Shoots rays from a sample point and counts boundary crossings
//! to determine inside/outside status.
//!
//! NOTE: `operations/boolean/classify.rs` contains a duplicate of this
//! logic. Bug fixes should be applied here first; the operations copy
//! will be deleted during the GFA step 5 switchover.

use brepkit_math::predicates::point_in_polygon;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;
use std::f64::consts::TAU;

use crate::builder::FaceClass;
use crate::error::AlgoError;

/// Per-face geometry used for ray crossing tests.
enum FaceGeom {
    /// A planar (or planar-approximated) face: boundary polygon, hole
    /// polygons, and the supporting plane.
    Planar {
        verts: Vec<Point3>,
        holes: Vec<Vec<Point3>>,
        /// Holes whose wire is circular, kept ANALYTIC (center, radius): a
        /// chord-sampled hole polygon under-covers the true circle by its
        /// sagitta, and a ray passing through that band counts a phantom
        /// crossing while reading clean (the circleinsert pocket-mouth
        /// sample: a hit 0.03 inside the r=10 pocket rim, sagitta ~0.09).
        circle_holes: Vec<(Point3, f64)>,
        normal: Vec3,
        d: f64,
    },
    /// A full-period cylindrical face (e.g. a bore lateral). Crossings are
    /// computed analytically — a flat polygon approximation counts one
    /// crossing where the real surface has two, flipping the parity.
    ///
    /// `hole_bands` are full-circumference v-ranges carved out of the lateral
    /// (a flush-cap interaction can leave such a holed lateral). A crossing
    /// whose axial parameter falls inside a hole band is excluded.
    Cylinder {
        surface: brepkit_math::surfaces::CylindricalSurface,
        v_min: f64,
        v_max: f64,
        hole_bands: Vec<(f64, f64)>,
        /// For a partial-arc patch (e.g. a rounded-rect corner quarter), the
        /// angular range NOT covered by the face — a crossing whose `u`
        /// (circumferential parameter) falls in this gap is off the patch and
        /// excluded. `None` for a full-period lateral.
        u_gap: Option<(f64, f64)>,
    },
    /// A conical face without inner wires, full-period or partial-arc.
    /// Crossings come from the ray/double-cone quadratic filtered to the
    /// face's slant range (`v` = distance from apex along the generator,
    /// which also rejects mirror-nappe hits) and angular patch. The flat
    /// polygon fallback mis-counts crossings against a strongly curved
    /// tapered corner patch, flipping the parity for nearby points.
    Cone {
        surface: brepkit_math::surfaces::ConicalSurface,
        v_min: f64,
        v_max: f64,
        u_gap: Option<(f64, f64)>,
    },
    /// A toroidal face covering the full major (u) revolution: either the
    /// whole torus (degenerate fundamental-polygon boundary — previously
    /// dropped from parity counting entirely) or a tube-angle band bounded
    /// by full rim circles. Crossings come from the residual-verified
    /// ray/torus quartic in `brepkit_math`, filtered to the tube-angle band.
    /// The flat polygon fallback mis-counts against the doubly-curved
    /// surface (up to four real crossings per ray).
    Torus {
        surface: brepkit_math::surfaces::ToroidalSurface,
        /// Tube-angle band as `(v_start, span)` with `span` in `(0, TAU)`,
        /// membership tested periodically from `v_start`. `None` = the full
        /// tube (whole torus).
        v_band: Option<(f64, f64)>,
    },
    /// A spherical face whose loops each bound a region of the sphere cut out
    /// by planes. Crossings come from the ray/sphere quadratic, kept when the
    /// hit lies on every loop's side. The flat polygon fallback stands the
    /// patch in by the polygon through its boundary, which misreads every
    /// point between the two (an octant's corner region).
    Sphere {
        surface: brepkit_math::surfaces::SphericalSurface,
        loops: Vec<SphereLoop>,
    },
}

/// One loop of a spherical face as the region it leaves on its left about the
/// sphere's outward normal.
enum SphereLoop {
    /// The half-spaces the region lies in: the side each boundary plane
    /// leaves on the loop's left. A loop in one plane is one half-space; a
    /// loop of arcs in several planes bounds either their intersection (a
    /// convex patch) or, running the other way around it, its complement
    /// (`any`: a convex hole).
    HalfSpaces {
        /// `(point on the plane, normal toward the region)`.
        planes: Vec<(Point3, Vec3)>,
        any: bool,
    },
    /// A hole whose rim is neither in one plane nor made of circle arcs (a
    /// bore's rim off the ball's centre), lying within the open hemisphere
    /// about `frame.z` (the frame sits at the sphere's centre): a hemisphere
    /// is a graph over its base plane, so the hole is the part of it whose
    /// projection along `frame.z` falls within the rim's (`flat`).
    Rim {
        frame: brepkit_math::frame::Frame3,
        flat: Vec<Point2>,
    },
}

impl SphereLoop {
    fn admits(&self, p: Point3, slack: f64) -> bool {
        match self {
            Self::HalfSpaces { planes, any } => {
                let inside = |&(c, n): &(Point3, Vec3)| (p - c).dot(n) >= -slack;
                if *any {
                    planes.iter().any(inside)
                } else {
                    planes.iter().all(inside)
                }
            }
            Self::Rim { frame, flat } => {
                (p - frame.origin).dot(frame.z) <= 0.0
                    || !point_in_polygon(rim_flat(frame, p), flat)
            }
        }
    }

    /// Whether `p` lies within `near` of the loop's boundary.
    fn grazes(&self, p: Point3, near: f64) -> bool {
        match self {
            Self::HalfSpaces { planes, .. } => {
                planes.iter().any(|&(c, n)| (p - c).dot(n).abs() <= near)
            }
            Self::Rim { frame, flat } => {
                let q = rim_flat(frame, p);
                flat.iter().zip(flat.iter().cycle().skip(1)).any(|(a, b)| {
                    let (ab, aq) = (*b - *a, q - *a);
                    let t = (aq.dot(ab) / ab.dot(ab).max(f64::MIN_POSITIVE)).clamp(0.0, 1.0);
                    (aq - ab * t).length() <= near
                })
            }
        }
    }
}

/// `p` projected along a rim frame's axis onto its base plane.
fn rim_flat(frame: &brepkit_math::frame::Frame3, p: Point3) -> Point2 {
    let d = p - frame.origin;
    Point2::new(d.dot(frame.x), d.dot(frame.y))
}

/// Classify a point by ray casting against the solid's faces.
///
/// Shoots 3 cardinal rays and uses majority vote; when all three cardinal
/// rays graze degenerate structure the vote is re-cast with 3 fixed
/// generic-direction rays (see `votes_from_geoms`). A point is inside if 2+
/// rays of the deciding triple report an odd crossing count.
///
/// # Errors
///
/// Returns [`AlgoError::ClassificationFailed`] if classification is
/// indeterminate after multiple ray directions.
pub fn classify_ray_cast(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
) -> Result<FaceClass, AlgoError> {
    let inside_votes = ray_cast_inside_votes(topo, solid, point)?;
    if inside_votes >= 2 {
        Ok(FaceClass::Inside)
    } else {
        Ok(FaceClass::Outside)
    }
}

/// Number of rays (of three) reporting an odd crossing count.
///
/// A point is classified inside when 2+ of the three rays agree; the raw count
/// distinguishes a confident 3-vote verdict from a grazing 2-vote tie.
///
/// # Errors
///
/// Returns [`AlgoError::ClassificationFailed`] if no face geometry is collected.
pub fn ray_cast_inside_votes(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
) -> Result<u8, AlgoError> {
    let face_data = collect_face_geoms(topo, solid)?;
    votes_from_geoms(&face_data, point)
}

/// Pre-collected ray-cast geometry for a solid, built once and reused across
/// many point classifications.
///
/// Collecting the geometry samples every face's wire into a polygon (and chains
/// the polylines), which is the dominant cost of a single classification. When
/// classifying many points against the same solid — every sub-face of a boolean
/// against the opposing solid — rebuilding it per point is O(faces) × O(points).
/// Building it once turns that into a single O(faces) pass.
pub struct RayCastGeoms {
    faces: Vec<FaceGeom>,
    /// The solid's closed-form classifier, built once with the geometry
    /// (its convexity guard walks every vertex, too much per point).
    analytic: Option<super::analytic::AnalyticClassifier>,
}

impl RayCastGeoms {
    /// Collect the ray-cast geometry for `solid` once.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if a topology lookup fails.
    pub fn new(topo: &Topology, solid: SolidId) -> Result<Self, AlgoError> {
        Ok(Self {
            faces: collect_face_geoms(topo, solid)?,
            analytic: super::analytic::try_build_analytic_classifier(topo, solid),
        })
    }

    /// The solid's closed-form classifier, if it has one.
    #[must_use]
    pub const fn analytic(&self) -> Option<&super::analytic::AnalyticClassifier> {
        self.analytic.as_ref()
    }
}

/// Inside-vote count for `point` using pre-collected geometry.
///
/// Identical to [`ray_cast_inside_votes`] but skips the per-call geometry
/// collection.
///
/// # Errors
///
/// Returns [`AlgoError`] if no face geometry was collected.
pub fn ray_cast_inside_votes_cached(geoms: &RayCastGeoms, point: Point3) -> Result<u8, AlgoError> {
    votes_from_geoms(&geoms.faces, point)
}

/// Ray-cast classification for `point` using pre-collected geometry.
///
/// # Errors
///
/// Returns [`AlgoError`] if no face geometry was collected.
pub fn classify_ray_cast_cached(
    geoms: &RayCastGeoms,
    point: Point3,
) -> Result<FaceClass, AlgoError> {
    if votes_from_geoms(&geoms.faces, point)? >= 2 {
        Ok(FaceClass::Inside)
    } else {
        Ok(FaceClass::Outside)
    }
}
/// `BK_RAY_POINT=x,y,z[,radius]` — the point (and match radius) for `RAYTRACE`.
///
/// Resolved once: the classifier runs this per sub-face, so an env lookup here
/// would land in a hot path.
fn ray_trace_target() -> Option<(Point3, f64)> {
    static TARGET: std::sync::OnceLock<Option<(Point3, f64)>> = std::sync::OnceLock::new();
    *TARGET.get_or_init(|| {
        let spec = std::env::var("BK_RAY_POINT").ok()?;
        let v: Vec<f64> = spec
            .split(',')
            .map(|t| t.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .ok()?;
        match v.len() {
            3 => Some((Point3::new(v[0], v[1], v[2]), 1e-6)),
            4 => Some((Point3::new(v[0], v[1], v[2]), v[3])),
            _ => None,
        }
    })
}

/// Count the inside votes (of three rays) for a point against pre-collected
/// face geometry: cardinal rays first, re-cast with fixed generic directions
/// only when every cardinal ray grazed degenerate structure.
fn votes_from_geoms(face_data: &[FaceGeom], point: Point3) -> Result<u8, AlgoError> {
    if face_data.is_empty() {
        return Err(AlgoError::ClassificationFailed(
            "no face polygons collected for ray-cast".into(),
        ));
    }

    let tol = Tolerance::new();
    let cardinal_dirs = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    // Escalation directions (normalized √-prime component vectors). CAD models
    // are dominated by axis-aligned feature planes, and a sample point lying ON
    // such a plane sends a cardinal ray along every edge, seam, and tangency in
    // that plane — the crossing parity of that ray is meaningless (a
    // dovetail-nub interior point on the x/y/z planes of a relief-bore tangency
    // lost 2 of 3 cardinal votes and classified Outside). Each ray reports
    // whether any of its hits grazed a face boundary, band limit, or in-plane
    // face (`suspicious`); only when ALL THREE cardinal rays are degenerate is
    // the cardinal instrument unusable, and the vote is re-cast with these
    // fixed generic directions that never run parallel to axis-aligned planes.
    // Any clean cardinal ray keeps the historical verdict, deterministically
    // (coincident-contact landscapes are calibrated against those results).
    let generic_dirs = [
        Vec3::new(
            0.447_213_595_499_957_9,
            0.547_722_557_505_166_1,
            std::f64::consts::FRAC_1_SQRT_2,
        ),
        Vec3::new(-0.5, 0.763_762_615_825_973_4, 0.408_248_290_463_863),
        Vec3::new(
            0.597_614_304_667_196_8,
            -0.377_964_473_009_227_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ),
    ];

    // `BK_RAY_POINT=x,y,z[,radius]`: dump this vote when the classified point is
    // near the given one. The verdict alone cannot say WHY a point classified
    // wrongly — the useful facts are the per-ray crossing parity and which rays
    // reported grazing degenerate structure, since the generic re-cast only
    // fires when ALL THREE cardinal rays are suspicious.
    let traced = ray_trace_target().is_some_and(|(t, r)| (point - t).length() <= r);

    let vote = |dirs: &[Vec3; 3], label: &str| -> [(bool, bool); 3] {
        let mut rays = [(false, false); 3];
        for (i, ray_dir) in dirs.iter().enumerate() {
            let mut crossings = 0i32;
            let mut suspicious = false;
            for (gi, geom) in face_data.iter().enumerate() {
                let (c, s) = ray_geom_crossings(point, *ray_dir, geom, tol);
                if traced && (c != 0 || s) {
                    let tag = match geom {
                        FaceGeom::Planar { verts, .. } => format!("planar[{}v]", verts.len()),
                        FaceGeom::Cylinder { .. } => "cylinder".to_string(),
                        FaceGeom::Cone { .. } => "cone".to_string(),
                        FaceGeom::Torus { .. } => "torus".to_string(),
                        FaceGeom::Sphere { loops, .. } => format!("sphere[{} loops]", loops.len()),
                    };
                    log::debug!(
                        "RAYHIT {label} dir=({:.3},{:.3},{:.3}) geom#{gi} {tag} c={c} susp={s}",
                        ray_dir.x(),
                        ray_dir.y(),
                        ray_dir.z()
                    );
                }
                crossings += c;
                suspicious |= s;
            }
            rays[i] = (crossings % 2 != 0, suspicious);
            if traced {
                log::debug!(
                    "RAYTRACE {label} dir=({:.3},{:.3},{:.3}) crossings={crossings} parity={} suspicious={suspicious}",
                    ray_dir.x(),
                    ray_dir.y(),
                    ray_dir.z(),
                    crossings % 2
                );
            }
        }
        rays
    };
    let count_inside = |rays: &[(bool, bool); 3]| rays.iter().filter(|r| r.0).count() as u8;

    let rays = vote(&cardinal_dirs, "cardinal");
    let cardinal = count_inside(&rays);
    let suspicious = rays.iter().filter(|r| r.1).count() as u8;
    // Clean/suspicious conflict: a clean ray's parity is trustworthy while a
    // suspicious ray's is unreliable by its own report, so a suspicious pair
    // must not silently outvote a clean minority (the O-shape chamfer strip:
    // an interior sample lying in an opposing rim plane sends both horizontal
    // rays grazing that structure, each losing a crossing and voting Inside
    // against the clean vertical ray's Outside). On that signature, re-cast
    // with the generic directions — but adopt the re-cast ONLY when it is
    // unanimous. In exact arithmetic every ray from one point has the same
    // parity, so a split generic vote proves the neighborhood defeats the
    // crossing counter (suspicion detection has false negatives: a honeycomb
    // landscape produced three CLEAN generic rays voting 2/1) and the
    // calibrated historic verdict stands. Mixed suspicious votes or any
    // suspicious ray agreeing with the clean verdict keep the historic result
    // outright.
    let clean_verdicts: Vec<bool> = rays.iter().filter(|r| !r.1).map(|r| r.0).collect();
    let clean_vs_suspicious_conflict = suspicious > 0
        && !clean_verdicts.is_empty()
        && clean_verdicts.iter().all(|&v| v == clean_verdicts[0])
        && rays
            .iter()
            .filter(|r| r.1)
            .all(|r| r.0 != clean_verdicts[0]);
    if traced {
        log::debug!(
            "RAYTRACE point=({:.3},{:.3},{:.3}) faces={} cardinal_inside={cardinal} suspicious={suspicious} conflict={clean_vs_suspicious_conflict}",
            point.x(),
            point.y(),
            point.z(),
            face_data.len(),
        );
    }
    if clean_vs_suspicious_conflict {
        let generic = vote(&generic_dirs, "generic");
        let inside = count_inside(&generic);
        if std::env::var("BK_CONFLICT").is_ok() {
            log::debug!(
                "CONFLICT pt=({:.4},{:.4},{:.4}) cardinal={cardinal} generic={inside} generic_susp={} clean_verdict={}",
                point.x(),
                point.y(),
                point.z(),
                generic.iter().filter(|r| r.1).count(),
                clean_verdicts[0]
            );
        }
        if inside == 0 || inside == 3 {
            return Ok(inside);
        }
        return Ok(cardinal);
    }
    if suspicious < 3 {
        return Ok(cardinal);
    }
    // The generic rays' own suspicion count is deliberately not consulted:
    // when both instruments graze degenerate structure there is no cleaner
    // signal left, and the generic directions are still the less-aligned,
    // better-conditioned of the two.
    let generic = vote(&generic_dirs, "generic");
    Ok(count_inside(&generic))
}

/// Distance from a point to the closed polyline through `verts`.
fn dist_to_polygon_boundary(p: Point3, verts: &[Point3]) -> f64 {
    let mut best = f64::INFINITY;
    let n = verts.len();
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        let ab = b - a;
        let len2 = ab.dot(ab);
        let t = if len2 > 0.0 {
            ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let foot = Point3::new(
            ab.x().mul_add(t, a.x()),
            ab.y().mul_add(t, a.y()),
            ab.z().mul_add(t, a.z()),
        );
        best = best.min((p - foot).length());
    }
    best
}

/// Sample a wire into a polygon by geometrically chaining its edges.
///
/// Wires are not guaranteed to list edges in traversal order (primitive
/// builders store edge sets), so each edge is sampled into a polyline and
/// the polylines are chained by endpoint matching. Closed curved edges
/// (full circles) get dense sampling; open curved edges get interior
/// samples for better coverage.
fn wire_polygon(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
) -> Result<Vec<Point3>, AlgoError> {
    let wire = topo.wire(wire_id)?;

    let mut polylines: Vec<Vec<Point3>> = Vec::with_capacity(wire.edges().len());
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let raw_start = topo.vertex(edge.start())?.point();
        let raw_end = topo.vertex(edge.end())?.point();
        let mut pts = vec![raw_start];
        if !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
            let (t0, t1) = edge.curve().domain_with_endpoints(raw_start, raw_end);
            let is_closed = (raw_start - raw_end).length() < 1e-9;
            let n_samples = if is_closed { 16_i32 } else { 3_i32 };
            for k in 1..=n_samples {
                let t = t0 + (t1 - t0) * f64::from(k) / f64::from(n_samples + 1);
                pts.push(edge.curve().evaluate_with_endpoints(t, raw_start, raw_end));
            }
        }
        pts.push(raw_end);
        if !oe.is_forward() {
            pts.reverse();
        }
        polylines.push(pts);
    }

    let join_tol = 1e-6;
    let mut used = vec![false; polylines.len()];
    let mut verts: Vec<Point3> = Vec::new();
    let Some(first) = polylines.first() else {
        return Ok(verts);
    };
    verts.extend_from_slice(first);
    used[0] = true;
    for _ in 1..polylines.len() {
        let tail = match verts.last() {
            Some(p) => *p,
            None => break,
        };
        let next = polylines.iter().enumerate().find_map(|(i, pl)| {
            if used[i] {
                return None;
            }
            let s = *pl.first()?;
            let e = *pl.last()?;
            if (s - tail).length() < join_tol {
                Some((i, false))
            } else if (e - tail).length() < join_tol {
                Some((i, true))
            } else {
                None
            }
        });
        let Some((idx, rev)) = next else { break };
        used[idx] = true;
        let mut pl = polylines[idx].clone();
        if rev {
            pl.reverse();
        }
        verts.extend_from_slice(&pl[1..]);
    }
    // Append any unchained polylines so no geometry is silently lost
    // (matches the previous behavior of emitting all edge samples).
    for (i, pl) in polylines.iter().enumerate() {
        if !used[i] {
            verts.extend_from_slice(pl);
        }
    }
    // Drop the duplicated closing point.
    if verts.len() >= 2 {
        let first_pt = verts[0];
        if let Some(last) = verts.last()
            && (*last - first_pt).length() < join_tol
        {
            verts.pop();
        }
    }
    Ok(verts)
}

/// Collect per-face ray-cast geometry from a solid.
fn collect_face_geoms(topo: &Topology, solid: SolidId) -> Result<Vec<FaceGeom>, AlgoError> {
    crate::perf::bump_ray_geom_build();
    let faces = brepkit_topology::explorer::solid_faces(topo, solid)?;
    let mut result = Vec::with_capacity(faces.len());

    for fid in faces {
        let face = topo.face(fid)?;

        // Full-period cylindrical faces: the outer wire contains a closed
        // circle edge, so the face wraps the entire circumference and the
        // analytic crossing test applies. Inner wires are accepted only when
        // each is a full-circumference v-band (the shape a flush-cap
        // interaction carves out); any non-banded hole forces the polygon
        // fallback. Partial cylinder patches also fall through.
        if let brepkit_topology::face::FaceSurface::Cylinder(cyl) = face.surface() {
            let wire = topo.wire(face.outer_wire())?;
            let mut has_closed_circle = false;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_))
                    && edge.start() == edge.end()
                {
                    has_closed_circle = true;
                    break;
                }
            }
            if has_closed_circle {
                let verts = wire_polygon(topo, face.outer_wire())?;
                let mut v_min = f64::INFINITY;
                let mut v_max = f64::NEG_INFINITY;
                for p in &verts {
                    let (_, v) = cyl.project_point(*p);
                    v_min = v_min.min(v);
                    v_max = v_max.max(v);
                }
                let hole_bands = cylinder_hole_bands(topo, face, cyl)?;
                let holes_banded = hole_bands.len() == face.inner_wires().len();
                if v_min.is_finite() && v_max > v_min && holes_banded {
                    result.push(FaceGeom::Cylinder {
                        surface: cyl.clone(),
                        v_min,
                        v_max,
                        hole_bands,
                        u_gap: None,
                    });
                    continue;
                }
            }

            // Partial-arc cylinder patch (e.g. a rounded-rect corner quarter):
            // no closed-circle edge, so the full-period path skipped it.
            // Collect it analytically with an angular trim rather than the
            // polygon fallback, whose non-planar boundary mis-counts crossings.
            if face.inner_wires().is_empty() {
                let verts = wire_polygon(topo, face.outer_wire())?;
                if verts.len() >= 3 {
                    let mut pv_min = f64::INFINITY;
                    let mut pv_max = f64::NEG_INFINITY;
                    let mut u_samples = Vec::with_capacity(verts.len());
                    for p in &verts {
                        let (u, v) = cyl.project_point(*p);
                        pv_min = pv_min.min(v);
                        pv_max = pv_max.max(v);
                        u_samples.push(u);
                    }
                    if pv_min.is_finite()
                        && pv_max > pv_min
                        && let Some(gap) = largest_u_gap(&u_samples)
                    {
                        result.push(FaceGeom::Cylinder {
                            surface: cyl.clone(),
                            v_min: pv_min,
                            v_max: pv_max,
                            hole_bands: Vec::new(),
                            u_gap: Some(gap),
                        });
                        continue;
                    }
                }
            }
        }

        // Conical faces without inner wires: collect analytically. `v` is the
        // slant distance from the apex, so projecting the boundary polygon
        // yields the patch's `[v_min, v_max]`; a closed circle edge marks a
        // full-period band (no angular trim), otherwise the largest angular
        // gap trims the patch like the partial-arc cylinder path above.
        if let brepkit_topology::face::FaceSurface::Cone(cone) = face.surface()
            && face.inner_wires().is_empty()
        {
            let wire = topo.wire(face.outer_wire())?;
            let mut has_closed_circle = false;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_))
                    && edge.start() == edge.end()
                {
                    has_closed_circle = true;
                    break;
                }
            }
            let verts = wire_polygon(topo, face.outer_wire())?;
            if verts.len() >= 3 {
                let mut pv_min = f64::INFINITY;
                let mut pv_max = f64::NEG_INFINITY;
                let mut u_samples = Vec::with_capacity(verts.len());
                for p in &verts {
                    let (u, v) = cone.project_point(*p);
                    pv_min = pv_min.min(v);
                    pv_max = pv_max.max(v);
                    u_samples.push(u);
                }
                if pv_min.is_finite() && pv_max > pv_min {
                    let u_gap = if has_closed_circle {
                        Some(None)
                    } else {
                        largest_u_gap(&u_samples).map(Some)
                    };
                    if let Some(u_gap) = u_gap {
                        result.push(FaceGeom::Cone {
                            surface: cone.clone(),
                            v_min: pv_min,
                            v_max: pv_max,
                            u_gap,
                        });
                        continue;
                    }
                }
            }
        }

        // Toroidal faces without inner wires: the whole torus (degenerate
        // fundamental-polygon boundary yields < 3 distinct polygon points and
        // previously fell out of parity counting entirely) or a full-major-
        // revolution tube band bounded by rim circles. Partial-u patches keep
        // the polygon fallback.
        if let brepkit_topology::face::FaceSurface::Torus(t) = face.surface()
            && face.inner_wires().is_empty()
        {
            use std::f64::consts::TAU;
            let verts = wire_polygon(topo, face.outer_wire())?;
            if verts.len() < 3 {
                // Degenerate boundary: the untrimmed whole torus.
                result.push(FaceGeom::Torus {
                    surface: t.clone(),
                    v_band: None,
                });
                continue;
            }
            let wire = topo.wire(face.outer_wire())?;
            let mut has_closed_circle = false;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_))
                    && edge.start() == edge.end()
                {
                    has_closed_circle = true;
                    break;
                }
            }
            if has_closed_circle {
                // Band gated on full u coverage: the boundary must sweep the
                // whole major revolution, else the patch keeps the fallback.
                let mut us: Vec<f64> = Vec::with_capacity(verts.len());
                let mut vs: Vec<f64> = Vec::with_capacity(verts.len());
                for p in &verts {
                    let (u, v) = t.project_point(*p);
                    us.push(u.rem_euclid(TAU));
                    vs.push(v.rem_euclid(TAU));
                }
                // Full major revolution iff the sampled boundary leaves no
                // large angular gap (a genuine partial patch leaves at least
                // its own missing arc; sampled rim circles leave only the
                // inter-sample spacing).
                us.sort_unstable_by(f64::total_cmp);
                let mut u_gap = -1.0_f64;
                for i in 0..us.len() {
                    let next = us[(i + 1) % us.len()];
                    u_gap = u_gap.max((next - us[i]).rem_euclid(TAU));
                }
                if u_gap <= 1.0 {
                    // Tube coverage from the periodic v samples. Boundary
                    // samples fully covering the tube (a seam-only boundary)
                    // OR collapsing to a single v (a full tube cut along one
                    // rim circle) both mean the face spans the whole tube. A
                    // two-rim band is SIDE-AMBIGUOUS from boundary vertices
                    // alone (the rims bound either half), so it keeps the
                    // polygon fallback rather than guessing.
                    vs.sort_unstable_by(f64::total_cmp);
                    let mut gap = -1.0_f64;
                    for i in 0..vs.len() {
                        let next = vs[(i + 1) % vs.len()];
                        let d = (next - vs[i]).rem_euclid(TAU);
                        gap = gap.max(d);
                    }
                    let span = TAU - gap;
                    if gap <= 1e-3 || span <= 1e-3 {
                        result.push(FaceGeom::Torus {
                            surface: t.clone(),
                            v_band: None,
                        });
                        continue;
                    }
                }
            }
        }

        if let brepkit_topology::face::FaceSurface::Sphere(sph) = face.surface()
            && let Some(loops) = sphere_face_loops(topo, face, sph)?
        {
            result.push(FaceGeom::Sphere {
                surface: sph.clone(),
                loops,
            });
            continue;
        }

        let verts = wire_polygon(topo, face.outer_wire())?;
        if verts.len() < 3 {
            continue;
        }

        let mut holes = Vec::with_capacity(face.inner_wires().len());
        let mut circle_holes: Vec<(Point3, f64)> = Vec::new();
        // Analytic holes only on a genuine plane face: the fallback path
        // reaches here for non-planar surfaces via a Newell normal, where a
        // hole's circle need not be coplanar with the polygon and the exact
        // distance test below would be wrong.
        let plane_normal =
            if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() {
                Some(*normal)
            } else {
                None
            };
        for &iw in face.inner_wires() {
            // A hole whose edges all lie on ONE circle IN THE FACE PLANE
            // stays analytic.
            let circular: Option<(Point3, f64)> = (|| {
                let pn = plane_normal?;
                let wire = topo.wire(iw).ok()?;
                let mut c: Option<(Point3, f64)> = None;
                for oe in wire.edges() {
                    let edge = topo.edge(oe.edge()).ok()?;
                    let brepkit_topology::edge::EdgeCurve::Circle(circ) = edge.curve() else {
                        return None;
                    };
                    if circ.normal().cross(pn).length() > 1e-6 {
                        return None;
                    }
                    match &c {
                        None => c = Some((circ.center(), circ.radius())),
                        Some((cc, cr)) => {
                            if (circ.center() - *cc).length() > 1e-6
                                || (circ.radius() - cr).abs() > 1e-6
                            {
                                return None;
                            }
                        }
                    }
                }
                c
            })();
            if let Some(ch) = circular {
                circle_holes.push(ch);
                continue;
            }
            let hole = wire_polygon(topo, iw)?;
            if hole.len() >= 3 {
                holes.push(hole);
            }
        }

        let raw_normal =
            if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() {
                *normal
            } else {
                newell_normal(&verts)
            };
        let normal = if face.is_reversed() {
            -raw_normal
        } else {
            raw_normal
        };

        let d = dot_normal_point(normal, verts[0]);
        result.push(FaceGeom::Planar {
            verts,
            holes,
            circle_holes,
            normal,
            d,
        });
    }

    Ok(result)
}

/// Collect full-circumference v-band holes carved out of a cylindrical face.
///
/// Each inner wire is sampled and projected into `(u, v)`. A wire is treated
/// as a band only when its u-samples wrap the full circumference; the band's
/// `[v_lo, v_hi]` is the projected axial span. Returns one entry per qualifying
/// inner wire — a count short of `face.inner_wires().len()` signals a
/// non-banded hole, which the caller uses to force the polygon fallback.
fn cylinder_hole_bands(
    topo: &Topology,
    face: &brepkit_topology::face::Face,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
) -> Result<Vec<(f64, f64)>, AlgoError> {
    use std::f64::consts::TAU;

    let mut bands = Vec::with_capacity(face.inner_wires().len());
    for &iw in face.inner_wires() {
        let pts = wire_polygon(topo, iw)?;
        if pts.len() < 3 {
            continue;
        }
        let mut v_lo = f64::INFINITY;
        let mut v_hi = f64::NEG_INFINITY;
        let mut u_min = f64::INFINITY;
        let mut u_max = f64::NEG_INFINITY;
        for p in &pts {
            let (u, v) = cyl.project_point(*p);
            v_lo = v_lo.min(v);
            v_hi = v_hi.max(v);
            u_min = u_min.min(u);
            u_max = u_max.max(u);
        }
        // Only full-circumference bands qualify: a partial-arc hole would be
        // over-excluded by a v-band test.
        let wraps_full = (u_max - u_min) >= TAU - 1e-3;
        if wraps_full && v_hi > v_lo {
            bands.push((v_lo, v_hi));
        }
    }
    Ok(bands)
}

/// Count ray crossings against a face geometry.
///
/// The second component reports a degenerate encounter: a hit (accepted or
/// barely rejected) grazing the face's boundary, patch limit, or an in-plane
/// face — the parity contribution of such a ray is unreliable.
#[inline]
fn ray_geom_crossings(
    origin: Point3,
    ray_dir: Vec3,
    geom: &FaceGeom,
    tol: Tolerance,
) -> (i32, bool) {
    match geom {
        FaceGeom::Planar {
            verts,
            holes,
            circle_holes,
            normal,
            d,
        } => ray_face_crossing(
            origin,
            ray_dir,
            verts,
            holes,
            circle_holes,
            *normal,
            *d,
            tol,
        ),
        FaceGeom::Cylinder {
            surface,
            v_min,
            v_max,
            hole_bands,
            u_gap,
        } => ray_cylinder_crossings(
            origin,
            ray_dir,
            surface,
            (*v_min, *v_max),
            hole_bands,
            *u_gap,
            tol,
        ),
        FaceGeom::Cone {
            surface,
            v_min,
            v_max,
            u_gap,
        } => ray_cone_crossings(origin, ray_dir, surface, (*v_min, *v_max), *u_gap, tol),
        FaceGeom::Torus { surface, v_band } => {
            ray_torus_crossings(origin, ray_dir, surface, *v_band, tol)
        }
        FaceGeom::Sphere { surface, loops } => {
            ray_sphere_crossings(origin, ray_dir, surface, loops, tol)
        }
    }
}

/// Count ray crossings with a spherical face: the ray/sphere roots whose hit
/// lies in every loop's region. A tangent ray counts none; a hit within
/// tolerance of a loop's plane marks the ray suspicious.
fn ray_sphere_crossings(
    origin: Point3,
    ray_dir: Vec3,
    surface: &brepkit_math::surfaces::SphericalSurface,
    loops: &[SphereLoop],
    tol: Tolerance,
) -> (i32, bool) {
    let near = 10.0 * tol.linear;
    let r = surface.radius();
    let m = origin - surface.center();
    let a = ray_dir.dot(ray_dir);
    let b = 2.0 * m.dot(ray_dir);
    let c = r.mul_add(-r, m.dot(m));
    let disc = b.mul_add(b, -4.0 * a * c);
    if disc < 1e-12 * a * r * r {
        return (0, false);
    }
    let sqrt_disc = disc.sqrt();
    let mut crossings = 0;
    let mut suspicious = false;
    for t in [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)] {
        if t <= tol.linear {
            continue;
        }
        let hit = origin + ray_dir * t;
        suspicious |= loops.iter().any(|l| l.grazes(hit, near));
        if loops.iter().all(|l| l.admits(hit, tol.linear)) {
            crossings += 1;
        }
    }
    (crossings, suspicious)
}

/// A spherical face's loops as [`SphereLoop`]s, each wire walked in its
/// traversal direction (a region on the left about the sphere's outward
/// normal, reversed faces included). `None` when a loop is neither planar nor
/// made of circle arcs that bound a convex patch or a convex hole.
fn sphere_face_loops(
    topo: &Topology,
    face: &brepkit_topology::face::Face,
    surface: &brepkit_math::surfaces::SphericalSurface,
) -> Result<Option<Vec<SphereLoop>>, AlgoError> {
    use brepkit_topology::edge::EdgeCurve;
    const SAMPLES: usize = 8;
    let r = surface.radius();
    let slack = 1e-9 * r.max(1.0);
    let mut loops = Vec::new();
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let wire = topo.wire(wire_id)?;
        // Each edge's samples in traversal order, and its circle if it is one.
        let mut edges: Vec<(Vec<Point3>, Option<brepkit_math::curves::Circle3D>)> = Vec::new();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let (sp, ep) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
            let mut pts: Vec<Point3> = (0..=SAMPLES)
                .map(|k| {
                    #[allow(clippy::cast_precision_loss)]
                    let f = k as f64 / SAMPLES as f64;
                    edge.curve()
                        .evaluate_with_endpoints(t0 + (t1 - t0) * f, sp, ep)
                })
                .collect();
            if !oe.is_forward() {
                pts.reverse();
            }
            let circle = match edge.curve() {
                EdgeCurve::Circle(c) => Some(c.clone()),
                EdgeCurve::Line | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => None,
            };
            edges.push((pts, circle));
        }
        let all: Vec<Point3> = edges.iter().flat_map(|(p, _)| p.iter().copied()).collect();
        if all.len() < 3 {
            return Ok(None);
        }
        // Twice the loop's vector area: a loop that only runs along a seam and
        // back encloses none, and its direction is rounding noise.
        let area = all.iter().zip(all.iter().cycle().skip(1)).fold(
            Vec3::new(0.0, 0.0, 0.0),
            |acc, (a, b)| {
                acc + Vec3::new(
                    (a.y() - b.y()) * (a.z() + b.z()),
                    (a.z() - b.z()) * (a.x() + b.x()),
                    (a.x() - b.x()) * (a.y() + b.y()),
                )
            },
        );
        if area.length() <= 1e-9 * r * r {
            return Ok(None);
        }
        let normal = area * (1.0 / area.length());
        let centroid = all.iter().fold(Vec3::new(0.0, 0.0, 0.0), |acc, p| {
            acc + Vec3::new(p.x(), p.y(), p.z())
        }) * (1.0 / f64::from(u32::try_from(all.len()).unwrap_or(u32::MAX)));
        let centroid = Point3::new(centroid.x(), centroid.y(), centroid.z());
        if all
            .iter()
            .all(|p| (*p - centroid).dot(normal).abs() <= slack)
        {
            loops.push(SphereLoop::HalfSpaces {
                planes: vec![(centroid, normal)],
                any: false,
            });
            continue;
        }
        if wire_id != face.outer_wire() && edges.iter().any(|(_, circle)| circle.is_none()) {
            match sphere_rim(topo, wire, surface)? {
                Some(rim) => {
                    loops.push(rim);
                    continue;
                }
                None => return Ok(None),
            }
        }
        // Arcs in several planes: each arc's side is the one its left points
        // into at its middle sample.
        let mut planes = Vec::with_capacity(edges.len());
        for (pts, circle) in &edges {
            let Some(circle) = circle else {
                return Ok(None);
            };
            let mid = pts[SAMPLES / 2];
            let tangent = pts[SAMPLES / 2 + 1] - pts[SAMPLES / 2 - 1];
            let Ok(outward) = (mid - surface.center()).normalize() else {
                return Ok(None);
            };
            let side = outward.cross(tangent).dot(circle.normal());
            if side.abs() <= slack {
                return Ok(None);
            }
            planes.push((circle.center(), circle.normal() * side.signum()));
        }
        let holds = |j: usize, want: bool| {
            edges.iter().enumerate().all(|(i, (pts, _))| {
                i == j
                    || pts.iter().all(|p| {
                        let d = (*p - planes[j].0).dot(planes[j].1);
                        if want { d >= -slack } else { d <= slack }
                    })
            })
        };
        let any = if (0..planes.len()).all(|j| holds(j, true)) {
            false
        } else if (0..planes.len()).all(|j| holds(j, false)) {
            true
        } else {
            return Ok(None);
        };
        // The half-spaces bound the loop's region only when the loop's arcs on
        // each circle cover exactly the part of that circle lying on the
        // region's side of the other planes: a column narrower than the ball
        // meets the sphere twice, and its far end would read as part of a
        // dome bounded by its top arcs. Each side meets a circle in one arc,
        // so that part is exact, however short its runs.
        for (j, (_, circle)) in edges.iter().enumerate() {
            let Some(circle) = circle else {
                return Ok(None);
            };
            let rho = circle.radius();
            let mut admitted: AngleSet = vec![(0.0, TAU)];
            for (i, &(c, n)) in planes.iter().enumerate() {
                if i == j {
                    continue;
                }
                let s = if any { -1.0 } else { 1.0 };
                let k = s * (circle.center() - c).dot(n);
                let (a, b) = (
                    s * rho * n.dot(circle.u_axis()),
                    s * rho * n.dot(circle.v_axis()),
                );
                let amp = a.hypot(b);
                let side = if amp <= slack {
                    if k >= -slack {
                        vec![(0.0, TAU)]
                    } else {
                        Vec::new()
                    }
                } else {
                    let t = (-slack - k) / amp;
                    if t <= -1.0 {
                        vec![(0.0, TAU)]
                    } else if t > 1.0 {
                        Vec::new()
                    } else {
                        let half = t.acos();
                        angle_arc(b.atan2(a) - half, 2.0 * half)
                    }
                };
                admitted = angle_intersection(&admitted, &side);
            }
            let mut covered: AngleSet = Vec::new();
            let mut arcs_on_circle = 0_usize;
            for (pts, other) in &edges {
                let same = other.as_ref().is_some_and(|o| {
                    (o.center() - circle.center()).length() <= slack
                        && (o.radius() - rho).abs() <= slack
                        && o.normal().cross(circle.normal()).length() <= 1e-9
                });
                if !same {
                    continue;
                }
                arcs_on_circle += 1;
                let angle = |p: Point3| circle.project(p);
                let (a0, a1) = (angle(pts[0]), angle(pts[pts.len() - 1]));
                let am = angle(pts[pts.len() / 2]);
                let to_end = (a1 - a0).rem_euclid(TAU);
                covered.extend(if (pts[0] - pts[pts.len() - 1]).length() <= slack {
                    angle_arc(0.0, TAU)
                } else if (am - a0).rem_euclid(TAU) <= to_end {
                    angle_arc(a0, to_end)
                } else {
                    angle_arc(a1, TAU - to_end)
                });
            }
            let covered = angle_union(covered);
            let both = angle_intersection(&admitted, &covered);
            let either = angle_union(admitted.iter().chain(&covered).copied().collect());
            #[allow(clippy::cast_precision_loss)]
            let allowance =
                1e-6 * r.max(1.0) / rho * (2 * (admitted.len() + arcs_on_circle)) as f64;
            if admitted.is_empty() || angle_measure(&either) - angle_measure(&both) > allowance {
                return Ok(None);
            }
        }
        loops.push(SphereLoop::HalfSpaces { planes, any });
    }
    Ok(Some(loops))
}

/// A set of angles on a circle, as disjoint sorted intervals of `[0, 2π]`.
type AngleSet = Vec<(f64, f64)>;

/// The arc from `start` running `span` counter-clockwise (a whole turn at
/// most).
fn angle_arc(start: f64, span: f64) -> AngleSet {
    if span >= TAU {
        return vec![(0.0, TAU)];
    }
    let a = start.rem_euclid(TAU);
    let b = a + span;
    if b <= TAU {
        vec![(a, b)]
    } else {
        vec![(0.0, b - TAU), (a, TAU)]
    }
}

fn angle_union(mut set: AngleSet) -> AngleSet {
    set.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut out: AngleSet = Vec::with_capacity(set.len());
    for (a, b) in set {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

fn angle_intersection(x: &AngleSet, y: &AngleSet) -> AngleSet {
    let mut out = Vec::new();
    for &(a0, a1) in x {
        for &(b0, b1) in y {
            let (lo, hi) = (a0.max(b0), a1.min(b1));
            if lo <= hi {
                out.push((lo, hi));
            }
        }
    }
    angle_union(out)
}

fn angle_measure(set: &AngleSet) -> f64 {
    set.iter().map(|(a, b)| b - a).sum()
}

/// A hole's rim as a [`SphereLoop::Rim`], about the direction from the
/// sphere's centre to the rim's centroid. `None` when the rim leaves that
/// open hemisphere, or runs about its projection with the region inside on
/// its left, where the face would be.
fn sphere_rim(
    topo: &Topology,
    wire: &brepkit_topology::wire::Wire,
    surface: &brepkit_math::surfaces::SphericalSurface,
) -> Result<Option<SphereLoop>, AlgoError> {
    const SAMPLES: usize = 64;
    let mut rim: Vec<Point3> = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let (sp, ep) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
        let mut pts: Vec<Point3> = (0..SAMPLES)
            .map(|k| {
                #[allow(clippy::cast_precision_loss)]
                let f = k as f64 / SAMPLES as f64;
                edge.curve()
                    .evaluate_with_endpoints(t0 + (t1 - t0) * f, sp, ep)
            })
            .collect();
        if !oe.is_forward() {
            pts.reverse();
            pts.rotate_right(1);
            if let Some(first) = pts.first_mut() {
                *first = edge.curve().evaluate_with_endpoints(t1, sp, ep);
            }
        }
        rim.extend(pts);
    }
    let center = surface.center();
    let sum = rim
        .iter()
        .fold(Vec3::new(0.0, 0.0, 0.0), |acc, p| acc + (*p - center));
    let Ok(axis) = sum.normalize() else {
        return Ok(None);
    };
    let margin = 1e-6 * surface.radius();
    if rim.iter().any(|p| (*p - center).dot(axis) <= margin) {
        return Ok(None);
    }
    let Ok(frame) = brepkit_math::frame::Frame3::from_normal(center, axis) else {
        return Ok(None);
    };
    let flat: Vec<Point2> = rim.iter().map(|&p| rim_flat(&frame, p)).collect();
    let twice_area: f64 = flat
        .iter()
        .zip(flat.iter().cycle().skip(1))
        .map(|(a, b)| a.x() * b.y() - b.x() * a.y())
        .sum();
    // Seen from outside the sphere down `axis`, a hole's rim runs clockwise
    // about the face on its left.
    if twice_area >= 0.0 {
        return Ok(None);
    }
    Ok(Some(SphereLoop::Rim { frame, flat }))
}

/// Test a single face polygon against a ray for crossing parity.
///
/// Returns +1 for a crossing, 0 for no intersection. Hits inside a hole
/// polygon do not count.
#[inline]
#[allow(clippy::too_many_arguments)]
fn ray_face_crossing(
    origin: Point3,
    ray_dir: Vec3,
    verts: &[Point3],
    holes: &[Vec<Point3>],
    circle_holes: &[(Point3, f64)],
    normal: Vec3,
    d: f64,
    tol: Tolerance,
) -> (i32, bool) {
    let near = 10.0 * tol.linear;
    let denom = normal.dot(ray_dir);
    if denom.abs() < tol.angular {
        // Ray parallel to the plane. If the origin also LIES in the plane the
        // ray travels inside the face's plane — edges and seams there make its
        // parity unreliable.
        let numer = d - dot_normal_point(normal, origin);
        return (0, numer.abs() <= near);
    }
    let numer = d - dot_normal_point(normal, origin);
    let t = numer / denom;
    if t <= tol.linear {
        return (0, false);
    }
    let hit = Point3::new(
        origin.x() + ray_dir.x() * t,
        origin.y() + ray_dir.y() * t,
        origin.z() + ray_dir.z() * t,
    );
    let boundary_graze = dist_to_polygon_boundary(hit, verts) <= near
        || holes
            .iter()
            .any(|h| dist_to_polygon_boundary(hit, h) <= near)
        || circle_holes
            .iter()
            .any(|(c, r)| ((hit - *c).length() - r).abs() <= near);
    if !point_in_face_3d(hit, verts, &normal) {
        return (0, boundary_graze);
    }
    if holes.iter().any(|h| point_in_face_3d(hit, h, &normal)) {
        return (0, boundary_graze);
    }
    if circle_holes.iter().any(|(c, r)| (hit - *c).length() < *r) {
        return (0, boundary_graze);
    }
    (1, boundary_graze)
}

/// Count ray crossings with a bounded full-period cylindrical face.
///
/// Solves the ray/infinite-cylinder quadratic and counts roots whose axial
/// parameter falls within the face's v-range but outside any `hole_bands`
/// (full-circumference v-ranges carved out of the lateral). Tangent grazes
/// (discriminant ≈ 0) count as zero crossings, which preserves parity.
fn ray_cylinder_crossings(
    origin: Point3,
    ray_dir: Vec3,
    surface: &brepkit_math::surfaces::CylindricalSurface,
    v_range: (f64, f64),
    hole_bands: &[(f64, f64)],
    u_gap: Option<(f64, f64)>,
    tol: Tolerance,
) -> (i32, bool) {
    let near = 10.0 * tol.linear;
    let (v_min, v_max) = v_range;
    let axis = surface.axis();
    let m = origin - surface.origin();
    let d_perp = ray_dir - axis * ray_dir.dot(axis);
    let m_perp = m - axis * m.dot(axis);

    let a = d_perp.dot(d_perp);
    if a < 1e-14 {
        return (0, false);
    }
    let b = 2.0 * m_perp.dot(d_perp);
    let c = surface
        .radius()
        .mul_add(-surface.radius(), m_perp.dot(m_perp));
    let disc = b.mul_add(b, -4.0 * a * c);
    // Treat near-tangent rays as misses: counting one graze flips parity.
    if disc < 1e-12 * a * surface.radius() * surface.radius() {
        return (0, false);
    }
    let sqrt_disc = disc.sqrt();
    let mut crossings = 0;
    let mut suspicious = false;
    let near_angle = near / surface.radius().max(near);
    for t in [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)] {
        if t <= tol.linear {
            continue;
        }
        let hit = Point3::new(
            origin.x() + ray_dir.x() * t,
            origin.y() + ray_dir.y() * t,
            origin.z() + ray_dir.z() * t,
        );
        let v = axis.dot(hit - surface.origin());
        suspicious |= (v - v_min).abs() <= near || (v - v_max).abs() <= near;
        if v < v_min - tol.linear || v > v_max + tol.linear {
            continue;
        }
        suspicious |= hole_bands
            .iter()
            .any(|&(lo, hi)| (v - lo).abs() <= near || (v - hi).abs() <= near);
        if hole_bands
            .iter()
            .any(|&(lo, hi)| v > lo + tol.linear && v < hi - tol.linear)
        {
            continue;
        }
        // Angular trim for a partial-arc patch: skip a hit on the off-patch
        // portion of the full cylinder (the rounded-rect corner quarter only
        // covers a 90° arc; the other 3/4 is not a real face).
        if let Some(gap) = u_gap {
            let (u, _) = surface.project_point(hit);
            suspicious |= near_gap_border(u, gap, near_angle);
            if u_in_gap(u, gap) {
                continue;
            }
        }
        crossings += 1;
    }
    (crossings, suspicious)
}

/// Whether `u` lies within `eps` of either border of the excluded gap.
fn near_gap_border(u: f64, gap: (f64, f64), eps: f64) -> bool {
    use std::f64::consts::TAU;
    let u = u.rem_euclid(TAU);
    for border in [gap.0.rem_euclid(TAU), gap.1.rem_euclid(TAU)] {
        let d = (u - border).abs();
        if d.min(TAU - d) <= eps {
            return true;
        }
    }
    false
}

/// Count ray crossings with a full-major-revolution toroidal face.
///
/// Roots come from the residual-verified ray/torus quartic in
/// `brepkit_math`; each accepted root's tube angle must fall inside the
/// face's periodic `v_band` (`None` accepts the whole tube). Near-tangent
/// root pairs and hits near the band borders flag the ray as unreliable,
/// mirroring the cylinder/cone grazing conventions.
fn ray_torus_crossings(
    origin: Point3,
    ray_dir: Vec3,
    surface: &brepkit_math::surfaces::ToroidalSurface,
    v_band: Option<(f64, f64)>,
    tol: Tolerance,
) -> (i32, bool) {
    use std::f64::consts::TAU;
    let near = 10.0 * tol.linear;
    let Ok(dir) = ray_dir.normalize() else {
        return (0, false);
    };
    let roots = brepkit_math::analytic_intersection::intersect_line_torus(surface, origin, dir);
    let near_angle = near / surface.minor_radius().max(near);
    let mut crossings = 0;
    let mut suspicious = false;
    for (i, &t) in roots.iter().enumerate() {
        if t <= tol.linear {
            continue;
        }
        // A close root pair is a graze: its two crossings cancel in parity
        // only if BOTH land in the band, so flag the ray instead of trusting
        // the count.
        for &t2 in &roots[i + 1..] {
            if (t2 - t).abs() <= near {
                suspicious = true;
            }
        }
        if let Some((v_start, span)) = v_band {
            let hit = Point3::new(
                origin.x() + dir.x() * t,
                origin.y() + dir.y() * t,
                origin.z() + dir.z() * t,
            );
            let (_, v) = surface.project_point(hit);
            let vv = (v - v_start).rem_euclid(TAU);
            suspicious |= vv <= near_angle || (span - vv).abs() <= near_angle;
            if vv > span {
                continue;
            }
        }
        crossings += 1;
    }
    (crossings, suspicious)
}

/// Count ray crossings with a bounded conical face.
///
/// Solves the ray/double-cone quadratic (zero set of `cos²a·h² − sin²a·ρ²`
/// around the apex, `h` axial and `ρ` radial) and counts roots whose slant
/// parameter `v` falls within the face's range — mirror-nappe hits project to
/// `v < 0` and are rejected by the same filter. Near-tangent grazes count as
/// zero crossings, preserving parity. A ray along a generator degenerates the
/// quadratic to a linear equation with a single crossing.
fn ray_cone_crossings(
    origin: Point3,
    ray_dir: Vec3,
    surface: &brepkit_math::surfaces::ConicalSurface,
    v_range: (f64, f64),
    u_gap: Option<(f64, f64)>,
    tol: Tolerance,
) -> (i32, bool) {
    let near = 10.0 * tol.linear;
    let (v_min, v_max) = v_range;
    let axis = surface.axis();
    let m = origin - surface.apex();
    let sin_a = surface.half_angle().sin();
    let sin2 = sin_a * sin_a;

    // Cone condition `cos²a·h² = sin²a·ρ²` (h axial, ρ radial) reduces to
    // `h² − sin²a·|w|² = 0` for w around the apex.
    let d_a = ray_dir.dot(axis);
    let m_a = m.dot(axis);
    let a = sin2.mul_add(-ray_dir.dot(ray_dir), d_a * d_a);
    let half_b = sin2.mul_add(-ray_dir.dot(m), d_a * m_a);
    let c = sin2.mul_add(-m.dot(m), m_a * m_a);

    let r_max = surface.radius_at(v_min.abs().max(v_max.abs()));
    let mut roots = [None, None];
    if a.abs() < 1e-14 {
        if half_b.abs() < 1e-14 {
            return (0, false);
        }
        roots[0] = Some(-c / (2.0 * half_b));
    } else {
        let disc = half_b.mul_add(half_b, -(a * c));
        if disc < 1e-12 * a.abs() * r_max * r_max {
            return (0, false);
        }
        let sqrt_disc = disc.sqrt();
        roots[0] = Some((-half_b - sqrt_disc) / a);
        roots[1] = Some((-half_b + sqrt_disc) / a);
    }

    let mut crossings = 0;
    let mut suspicious = false;
    for t in roots.into_iter().flatten() {
        if t <= tol.linear {
            continue;
        }
        let hit = Point3::new(
            origin.x() + ray_dir.x() * t,
            origin.y() + ray_dir.y() * t,
            origin.z() + ray_dir.z() * t,
        );
        let (u, v) = surface.project_point(hit);
        suspicious |= (v - v_min).abs() <= near || (v - v_max).abs() <= near;
        if v < v_min - tol.linear || v > v_max + tol.linear {
            continue;
        }
        if let Some(gap) = u_gap {
            let near_angle = near / surface.radius_at(v).max(near);
            suspicious |= near_gap_border(u, gap, near_angle);
            if u_in_gap(u, gap) {
                continue;
            }
        }
        crossings += 1;
    }
    (crossings, suspicious)
}

/// Whether circumferential parameter `u` lies in the excluded angular gap
/// `(lo, hi)` (CCW from `lo` to `hi`, possibly wrapping past 2π).
pub fn u_in_gap(u: f64, gap: (f64, f64)) -> bool {
    use std::f64::consts::TAU;
    let eps = 1e-6;
    let u = u.rem_euclid(TAU);
    let (lo, hi) = (gap.0.rem_euclid(TAU), gap.1.rem_euclid(TAU));
    if lo <= hi {
        u > lo + eps && u < hi - eps
    } else {
        u > lo + eps || u < hi - eps
    }
}

/// Largest angular gap between sorted circumferential samples — the arc the
/// partial-cylinder face does NOT cover. `None` for too-few samples or a gap
/// too small to be a genuine partial arc.
pub fn largest_u_gap(u_samples: &[f64]) -> Option<(f64, f64)> {
    use std::f64::consts::TAU;
    let mut us: Vec<f64> = u_samples.iter().map(|&u| u.rem_euclid(TAU)).collect();
    us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    us.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    if us.len() < 2 {
        return None;
    }
    let mut best = 0.0_f64;
    let mut gap = (0.0, 0.0);
    for i in 0..us.len() {
        let lo = us[i];
        let hi = if i + 1 < us.len() {
            us[i + 1]
        } else {
            us[0] + TAU
        };
        if hi - lo > best {
            best = hi - lo;
            gap = (lo, hi.rem_euclid(TAU));
        }
    }
    if best > 0.2 { Some(gap) } else { None }
}

/// Test if a 3D point lies inside a planar face polygon by projecting to 2D.
#[must_use]
pub fn point_in_face_3d(point: Point3, polygon: &[Point3], normal: &Vec3) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let (project_point, project_polygon): (Point2, Vec<Point2>) = if az >= ax && az >= ay {
        (
            Point2::new(point.x(), point.y()),
            polygon.iter().map(|p| Point2::new(p.x(), p.y())).collect(),
        )
    } else if ay >= ax {
        (
            Point2::new(point.x(), point.z()),
            polygon.iter().map(|p| Point2::new(p.x(), p.z())).collect(),
        )
    } else {
        (
            Point2::new(point.y(), point.z()),
            polygon.iter().map(|p| Point2::new(p.y(), p.z())).collect(),
        )
    };

    point_in_polygon(project_point, &project_polygon)
}

/// Compute `n . p` treating a `Point3` as a direction vector.
fn dot_normal_point(n: Vec3, p: Point3) -> f64 {
    n.dot(Vec3::new(p.x(), p.y(), p.z()))
}

/// A planar face's sampled boundary: `(outer_polygon, hole_polygons, normal)`.
pub type FacePolygons = (Vec<Point3>, Vec<Vec<Point3>>, Vec3);

/// Build a planar face's boundary as `(outer_polygon, hole_polygons, normal)`.
///
/// The outer wire and inner wires are sampled into polylines (arcs densified
/// via `wire_polygon`) so a rounded-corner cap's true region is captured.
/// Returns `None` if the outer polygon is degenerate (< 3 points).
///
/// # Errors
///
/// Returns [`AlgoError`] on a topology lookup failure.
pub fn planar_face_polygons(
    topo: &Topology,
    face_id: brepkit_topology::face::FaceId,
) -> Result<Option<FacePolygons>, AlgoError> {
    let face = topo.face(face_id)?;
    let verts = wire_polygon(topo, face.outer_wire())?;
    if verts.len() < 3 {
        return Ok(None);
    }
    let raw_normal =
        if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() {
            *normal
        } else {
            newell_normal(&verts)
        };
    let normal = if face.is_reversed() {
        -raw_normal
    } else {
        raw_normal
    };
    let mut holes = Vec::with_capacity(face.inner_wires().len());
    for &iw in face.inner_wires() {
        let hole = wire_polygon(topo, iw)?;
        if hole.len() >= 3 {
            holes.push(hole);
        }
    }
    Ok(Some((verts, holes, normal)))
}

/// Test whether `point` lies inside the planar face's region (inside the outer
/// polygon and outside every hole), projecting along the face normal.
#[must_use]
pub fn point_in_planar_region(
    point: Point3,
    outer: &[Point3],
    holes: &[Vec<Point3>],
    normal: &Vec3,
) -> bool {
    if !point_in_face_3d(point, outer, normal) {
        return false;
    }
    !holes.iter().any(|h| point_in_face_3d(point, h, normal))
}

/// A box around a solid: its boundary edges, each by its whole extent, and
/// the faces that bulge past their edges (a sphere, a torus, a NURBS
/// patch) by their surfaces' boxes.
///
/// # Errors
///
/// Returns [`AlgoError::ClassificationFailed`] if the solid has no boundary.
pub fn compute_solid_bbox(
    topo: &Topology,
    solid: SolidId,
) -> Result<brepkit_math::aabb::Aabb3, AlgoError> {
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::face::FaceSurface;
    let mut points = Vec::new();
    let faces = brepkit_topology::explorer::solid_faces(topo, solid)?;
    for fid in faces {
        let face = topo.face(fid)?;
        let surface = match face.surface() {
            FaceSurface::Sphere(s) => Some(s.aabb()),
            FaceSurface::Torus(t) => Some(t.aabb()),
            FaceSurface::Nurbs(n) => Some(n.aabb()),
            FaceSurface::Plane { .. } | FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => None,
        };
        if let Some(b) = surface {
            points.extend([b.min, b.max]);
        }
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let (sp, ep) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            points.extend([sp, ep]);
            let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
            let bulge = match edge.curve() {
                EdgeCurve::Line => None,
                EdgeCurve::Circle(c) => Some(c.arc_aabb(t0, t1)),
                EdgeCurve::Ellipse(e) => Some(e.arc_aabb(t0, t1)),
                EdgeCurve::NurbsCurve(n) => Some(n.aabb()),
            };
            if let Some(b) = bulge {
                points.extend([b.min, b.max]);
            }
        }
    }
    brepkit_math::aabb::Aabb3::try_from_points(points)
        .ok_or_else(|| AlgoError::ClassificationFailed("solid has no boundary".into()))
}

/// Compute polygon normal via Newell's method.
fn newell_normal(verts: &[Point3]) -> Vec3 {
    let n = verts.len();
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    for i in 0..n {
        let curr = verts[i];
        let next = verts[(i + 1) % n];
        nx += (curr.y() - next.y()) * (curr.z() + next.z());
        ny += (curr.z() - next.z()) * (curr.x() + next.x());
        nz += (curr.x() - next.x()) * (curr.y() + next.y());
    }
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-15 {
        Vec3::new(nx / len, ny / len, nz / len)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// Build a degenerate solid where all faces have < 3 vertices
    /// (single-edge faces). This tests the empty polygon fallback.
    fn make_degenerate_solid(topo: &mut Topology) -> brepkit_topology::solid::SolidId {
        // Create a "solid" with a single face that has only 2 vertices
        // (a degenerate line edge). This will produce < 3 polygon vertices.
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let e01 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e10 = topo.add_edge(Edge::new(v1, v0, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![OrientedEdge::new(e01, true), OrientedEdge::new(e10, true)],
                true,
            )
            .unwrap(),
        );
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let shell = topo.add_shell(Shell::new(vec![face]).unwrap());
        topo.add_solid(Solid::new(shell, vec![]))
    }

    /// A sphere face through `corners`, each boundary arc from one corner to
    /// the next on the circle its plane (`centres[i]`, through the corner
    /// pair) cuts from the sphere, the short way round.
    fn sphere_patch(
        topo: &mut Topology,
        radius: f64,
        corners: &[Point3],
        centres: &[Point3],
    ) -> brepkit_topology::face::FaceId {
        let verts: Vec<_> = corners
            .iter()
            .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
            .collect();
        let mut edges = Vec::new();
        for i in 0..corners.len() {
            let (a, b, c) = (corners[i], corners[(i + 1) % corners.len()], centres[i]);
            let normal = (a - c).cross(b - c).normalize().unwrap();
            let circle = brepkit_math::curves::Circle3D::new(c, normal, (a - c).length()).unwrap();
            let e = topo.add_edge(Edge::new(
                verts[i],
                verts[(i + 1) % corners.len()],
                EdgeCurve::Circle(circle),
            ));
            edges.push(OrientedEdge::new(e, true));
        }
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let sphere =
            brepkit_math::surfaces::SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), radius)
                .unwrap();
        topo.add_face(Face::new(wire, vec![], FaceSurface::Sphere(sphere)))
    }

    /// Arcs in several planes stand for their half-spaces only when each arc
    /// is the whole of its circle on the region's side of the others: an
    /// octant's three arcs are, a dome's four arcs in a column narrower than
    /// the ball are not (the column's far end meets the sphere too).
    #[test]
    fn sphere_arc_loops_bound_only_their_own_region() {
        let mut topo = Topology::default();
        // The far end's arcs are shorter than any sampling step once the
        // column is thin.
        for w in [3.0, 0.1] {
            let h = 2.0f64.mul_add(-w * w, 25.0).sqrt();
            let dome = sphere_patch(
                &mut topo,
                5.0,
                &[
                    Point3::new(w, w, h),
                    Point3::new(-w, w, h),
                    Point3::new(-w, -w, h),
                    Point3::new(w, -w, h),
                ],
                &[
                    Point3::new(0.0, w, 0.0),
                    Point3::new(-w, 0.0, 0.0),
                    Point3::new(0.0, -w, 0.0),
                    Point3::new(w, 0.0, 0.0),
                ],
            );
            let face = topo.face(dome).unwrap();
            let FaceSurface::Sphere(s) = face.surface() else {
                unreachable!()
            };
            assert!(
                sphere_face_loops(&topo, face, s).unwrap().is_none(),
                "the dome's arcs in the column {w} wide declined"
            );
        }

        let origin = Point3::new(0.0, 0.0, 0.0);
        let octant = sphere_patch(
            &mut topo,
            3.0,
            &[
                Point3::new(3.0, 0.0, 0.0),
                Point3::new(0.0, 3.0, 0.0),
                Point3::new(0.0, 0.0, 3.0),
            ],
            &[origin, origin, origin],
        );
        let face = topo.face(octant).unwrap();
        let FaceSurface::Sphere(s) = face.surface() else {
            unreachable!()
        };
        let loops = sphere_face_loops(&topo, face, s)
            .unwrap()
            .expect("the octant's arcs accepted");
        let r = 3.0_f64.sqrt();
        let admits = |p: Point3| loops.iter().all(|l| l.admits(p, 1e-9));
        assert!(admits(Point3::new(r, r, r)), "its own corner");
        assert!(!admits(Point3::new(-r, r, r)), "a neighbouring octant");
        assert!(!admits(Point3::new(0.0, 0.0, -3.0)), "the far pole");

        // A whole sphere written as a meridian seam walked out and back
        // encloses no area: declined rather than read as the seam's planes.
        let (n, e, s) = (
            Point3::new(0.0, 0.0, 3.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, -3.0),
        );
        let seam = sphere_patch(&mut topo, 3.0, &[n, e, s, e], &[origin; 4]);
        let face = topo.face(seam).unwrap();
        let FaceSurface::Sphere(sph) = face.surface() else {
            unreachable!()
        };
        assert!(
            sphere_face_loops(&topo, face, sph).unwrap().is_none(),
            "the seam loop declined"
        );
    }

    #[test]
    fn whole_torus_classifies_inside_and_outside() {
        use brepkit_math::surfaces::ToroidalSurface;
        // Whole torus R=3 r=1 about Z at the origin: a single face with a
        // degenerate point-seam boundary (the untrimmed fundamental polygon).
        let mut topo = Topology::default();
        let t = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0, 1.0).unwrap();
        let seam_p = t.evaluate(0.0, 0.0);
        let v0 = topo.add_vertex(Vertex::new(seam_p, 1e-7));
        let circle = brepkit_math::curves::Circle3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            4.0,
        )
        .unwrap();
        let e = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Circle(circle)));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
        let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Torus(t)));
        let shell = topo.add_shell(Shell::new(vec![face]).unwrap());
        let solid = topo.add_solid(Solid::new(shell, vec![]));

        // In the tube: on the spine circle.
        let inside = classify_ray_cast(&topo, solid, Point3::new(3.0, 0.0, 0.0)).unwrap();
        assert_eq!(inside, crate::builder::face_class::FaceClass::Inside);
        // The donut hole is OUTSIDE the solid.
        let hole = classify_ray_cast(&topo, solid, Point3::new(0.0, 0.0, 0.0)).unwrap();
        assert_eq!(hole, crate::builder::face_class::FaceClass::Outside);
        // Beyond the outer equator.
        let out = classify_ray_cast(&topo, solid, Point3::new(6.0, 0.0, 0.0)).unwrap();
        assert_eq!(out, crate::builder::face_class::FaceClass::Outside);
        // Above the tube.
        let above = classify_ray_cast(&topo, solid, Point3::new(3.0, 0.0, 2.0)).unwrap();
        assert_eq!(above, crate::builder::face_class::FaceClass::Outside);
    }

    #[test]
    fn empty_face_polygons_returns_error() {
        let mut topo = Topology::default();
        let solid = make_degenerate_solid(&mut topo);

        let result = classify_ray_cast(&topo, solid, Point3::new(0.5, 0.5, 0.5));
        assert!(
            result.is_err(),
            "ray-cast with no valid face polygons should return Err, got {result:?}"
        );
    }

    /// Build a unit box for classification tests.
    fn make_box(
        topo: &mut Topology,
        min: [f64; 3],
        max: [f64; 3],
    ) -> brepkit_topology::solid::SolidId {
        let [x0, y0, z0] = min;
        let [x1, y1, z1] = max;
        let v = [
            topo.add_vertex(Vertex::new(Point3::new(x0, y0, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y0, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y1, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x0, y1, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x0, y0, z1), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y0, z1), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y1, z1), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x0, y1, z1), 1e-7)),
        ];
        let mut edge = |a: usize, b: usize| -> brepkit_topology::edge::EdgeId {
            topo.add_edge(Edge::new(v[a], v[b], EdgeCurve::Line))
        };
        let e01 = edge(0, 1);
        let e12 = edge(1, 2);
        let e23 = edge(2, 3);
        let e30 = edge(3, 0);
        let e45 = edge(4, 5);
        let e56 = edge(5, 6);
        let e67 = edge(6, 7);
        let e74 = edge(7, 4);
        let e04 = edge(0, 4);
        let e15 = edge(1, 5);
        let e26 = edge(2, 6);
        let e37 = edge(3, 7);

        let fwd = |eid| OrientedEdge::new(eid, true);
        let rev = |eid| OrientedEdge::new(eid, false);
        let w_bot =
            topo.add_wire(Wire::new(vec![rev(e01), rev(e30), rev(e23), rev(e12)], true).unwrap());
        let w_top =
            topo.add_wire(Wire::new(vec![fwd(e45), fwd(e56), fwd(e67), fwd(e74)], true).unwrap());
        let w_front =
            topo.add_wire(Wire::new(vec![fwd(e01), fwd(e15), rev(e45), rev(e04)], true).unwrap());
        let w_back =
            topo.add_wire(Wire::new(vec![fwd(e23), fwd(e37), rev(e67), rev(e26)], true).unwrap());
        let w_left =
            topo.add_wire(Wire::new(vec![fwd(e30), fwd(e04), rev(e74), rev(e37)], true).unwrap());
        let w_right =
            topo.add_wire(Wire::new(vec![fwd(e12), fwd(e26), rev(e56), rev(e15)], true).unwrap());

        let mk_face =
            |w, n: Vec3, d: f64| Face::new(w, vec![], FaceSurface::Plane { normal: n, d });
        let faces = vec![
            topo.add_face(mk_face(w_bot, Vec3::new(0.0, 0.0, -1.0), -z0)),
            topo.add_face(mk_face(w_top, Vec3::new(0.0, 0.0, 1.0), z1)),
            topo.add_face(mk_face(w_front, Vec3::new(0.0, -1.0, 0.0), -y0)),
            topo.add_face(mk_face(w_back, Vec3::new(0.0, 1.0, 0.0), y1)),
            topo.add_face(mk_face(w_left, Vec3::new(-1.0, 0.0, 0.0), -x0)),
            topo.add_face(mk_face(w_right, Vec3::new(1.0, 0.0, 0.0), x1)),
        ];
        let shell = topo.add_shell(Shell::new(faces).unwrap());
        topo.add_solid(Solid::new(shell, vec![]))
    }

    #[test]
    fn ray_cast_classifies_inside_point() {
        let mut topo = Topology::default();
        let solid = make_box(&mut topo, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);

        let result = classify_ray_cast(&topo, solid, Point3::new(1.0, 1.0, 1.0)).unwrap();
        assert_eq!(result, FaceClass::Inside, "center of box should be Inside");
    }

    #[test]
    fn ray_cast_classifies_outside_point() {
        let mut topo = Topology::default();
        let solid = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        let result = classify_ray_cast(&topo, solid, Point3::new(5.0, 5.0, 5.0)).unwrap();
        assert_eq!(
            result,
            FaceClass::Outside,
            "point far from box should be Outside"
        );
    }
}
