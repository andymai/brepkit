//! A corner set back along its rounds: three stripe ends, three links on the
//! flat faces between them, and a patch over the hexagon they bound.
//!
//! A hexagon with six true corners is no quad, so the patch is two, split along
//! a seam from one corner to the one opposite. Each net's weights vary across
//! its columns only, which keeps a stripe's end an exact circle and leaves the
//! derivative across any row a plain difference of control points. Tangency and
//! the second order on a stripe's cylinder then hold by construction.

use std::collections::HashMap;

use brepkit_math::aabb::Aabb3;
use brepkit_math::curves::Circle3D;
use brepkit_math::frame::Frame3;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::decompose::curve_bezier_between;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::region2d::{Boundary2, point_in_region};
use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::{Shell, ShellId};
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::BlendError;

/// Rows in `v` each patch holds beyond its first: three from each long side.
const ACROSS_DEGREE: usize = 5;
/// Degree along the first patch's arcs: a pair of quadratic denominators, plus
/// the two the sideways drift of a row and its square add to a numerator.
const ALONG_DEGREE: usize = 6;
/// Degree along the second patch's seam side: the seam's tangent field, one
/// degree above the first patch's rows, over the third arc's denominator.
const SEAM_ALONG_DEGREE: usize = 8;
/// How far along its arc a seam end leans, in arc lengths. Measured across
/// setbacks from a twentieth of the radius to twice it, curvature on the patch
/// is at its evenest here: the dome stays convex, and either way ripples start.
const SEAM_LEAN: f64 = 1.6;
/// How far a stripe's rows reach, for each unit of set-back. The third row sits
/// at two fifths of this, which leaves it short of where the contact lines
/// would meet.
const REACH_PER_SETBACK: f64 = 1.875;

/// One stripe's end as a corner sees it, run from `start` to `end` in the
/// corner's own direction of travel.
#[derive(Debug, Clone, Copy)]
pub struct StripeEnd {
    /// Centre of the end's arc.
    pub center: Point3,
    /// Where the arc starts.
    pub start: Point3,
    /// Where the arc ends.
    pub end: Point3,
    /// Along the stripe towards the corner.
    pub inward: Vec3,
    /// How far along `inward` the first three rows reach, in row units.
    pub reach: f64,
}

/// The two faces over a corner's hexagon, and the curves they share with each
/// other and with the flat faces.
#[derive(Debug, Clone)]
pub struct CornerPatches {
    /// Holds the first two arcs and the link between them.
    pub first: NurbsSurface,
    /// Holds the third arc and the two links either side of it.
    pub second: NurbsSurface,
    /// From the first arc's start to the second arc's end.
    pub seam: NurbsCurve,
    /// From the first arc's end to the second arc's start.
    pub first_link: NurbsCurve,
    /// From the second arc's end to the third arc's start.
    pub second_link: NurbsCurve,
    /// From the first arc's start to the third arc's end.
    pub third_link: NurbsCurve,
}

fn choose(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

fn scalars_times(a: &[f64], b: &[f64]) -> Vec<f64> {
    let (m, n) = (a.len() - 1, b.len() - 1);
    let mut out = vec![0.0; m + n + 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] += x * y * choose(m, i) * choose(n, j) / choose(m + n, i + j);
        }
    }
    out
}

fn times(a: &[f64], b: &[Vec3]) -> Vec<Vec3> {
    let (m, n) = (a.len() - 1, b.len() - 1);
    let mut out = vec![Vec3::new(0.0, 0.0, 0.0); m + n + 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            out[i + j] += y * (x * choose(m, i) * choose(n, j) / choose(m + n, i + j));
        }
    }
    out
}

fn raised(b: &[Vec3], degree: usize) -> Vec<Vec3> {
    times(&vec![1.0; degree + 2 - b.len()], b)
}

fn raised_scalars(b: &[f64], degree: usize) -> Vec<f64> {
    scalars_times(&vec![1.0; degree + 2 - b.len()], b)
}

fn plus(a: &[Vec3], b: &[Vec3], scale: f64) -> Vec<Vec3> {
    a.iter().zip(b).map(|(&x, &y)| x + y * scale).collect()
}

/// An arc as a rational quadratic, about the local origin.
struct Arc {
    numerator: [Vec3; 3],
    weight: [f64; 3],
    center: Vec3,
    radius: f64,
    inward: Vec3,
    reach: f64,
    /// Sign that turns `inward x radial` into the arc's direction of travel.
    travel: f64,
    /// How far along the arc the seam's end leans, for the arc that holds one.
    lean: f64,
}

impl Arc {
    fn new(end: &StripeEnd, origin: Point3) -> Option<Self> {
        let center = end.center - origin;
        let (from, to) = (end.start - origin, end.end - origin);
        let radius = (from - center).length();
        let (du, dv) = (
            (from - center).normalize().ok()?,
            (to - center).normalize().ok()?,
        );
        let bisector = (du + dv).normalize().ok()?;
        let cos_half = du.dot(bisector);
        if cos_half <= 1e-9 {
            return None;
        }
        let peak = center + bisector * (radius / cos_half);
        let travel = if end.inward.cross(du).dot(peak - from) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        Some(Self {
            numerator: [from, peak * cos_half, to],
            weight: [1.0, cos_half, 1.0],
            center,
            radius,
            inward: end.inward,
            reach: end.reach,
            travel,
            lean: SEAM_LEAN * radius * 2.0 * cos_half.acos(),
        })
    }

    /// The arc's end row and the two beside it as numerators over `common`,
    /// which is the arc's denominator times `other`. The rows lean along the arc
    /// at its start, dying away by its end, and hold the cylinder to second order
    /// on the way: the lean turns a row's points round the circle, which is what
    /// the last row's bend pays for.
    fn rows(&self, other: &[f64], common: &[f64], degree: usize) -> [Vec<Vec3>; 3] {
        let across = ACROSS_DEGREE as f64;
        let edge = raised(&times(other, &self.numerator), degree);
        let radial: Vec<Vec3> = self
            .numerator
            .iter()
            .zip(self.weight)
            .map(|(&n, w)| (n - self.center * w) * (1.0 / self.radius))
            .collect();
        let tangent: Vec<Vec3> = radial
            .iter()
            .map(|&r| self.inward.cross(r) * self.travel)
            .collect();
        let lean = [self.lean, 0.0];
        let straight: Vec<Vec3> = common
            .iter()
            .map(|&w| self.inward * (self.reach * w))
            .collect();
        let sideways = times(&scalars_times(&lean, other), &tangent);
        let first = plus(&raised(&straight, degree), &raised(&sideways, degree), 1.0);
        let bend = times(&scalars_times(&scalars_times(&lean, &lean), other), &radial);
        let next = plus(&edge, &first, 1.0 / across);
        let doubled: Vec<Vec3> = next.iter().zip(&edge).map(|(&n, &e)| n * 2.0 - e).collect();
        let last = plus(
            &doubled,
            &raised(&bend, degree),
            -1.0 / (self.radius * across * (across - 1.0)),
        );
        [edge, next, last]
    }

    /// The arc's tangent at its start, by parameter.
    fn start_speed(&self) -> f64 {
        2.0 * self.weight[1]
            * (self.numerator[1] * (1.0 / self.weight[1]) - self.numerator[0]).length()
    }
}

fn curve(points: Vec<Point3>) -> Result<NurbsCurve, BlendError> {
    let degree = points.len() - 1;
    let mut knots = vec![0.0; degree + 1];
    knots.extend(vec![1.0; degree + 1]);
    let weights = vec![1.0; points.len()];
    Ok(NurbsCurve::new(degree, knots, points, weights)?)
}

fn clamped(degree: usize) -> Vec<f64> {
    let mut knots = vec![0.0; degree + 1];
    knots.extend(vec![1.0; degree + 1]);
    knots
}

/// The patches over the hexagon three stripe ends bound, or `None` when an arc
/// has no angle to read.
///
/// # Errors
///
/// Returns [`BlendError::Math`] when a net is rejected.
pub fn corner_patches(ends: &[StripeEnd; 3]) -> Result<Option<CornerPatches>, BlendError> {
    let origin = ends[0].center;
    let to_point = |v: Vec3| origin + v;
    let (Some(first), Some(second), Some(third)) = (
        Arc::new(&ends[0], origin),
        Arc::new(
            &StripeEnd {
                start: ends[1].end,
                end: ends[1].start,
                ..ends[1]
            },
            origin,
        ),
        Arc::new(
            &StripeEnd {
                start: ends[2].end,
                end: ends[2].start,
                ..ends[2]
            },
            origin,
        ),
    ) else {
        return Ok(None);
    };
    let across = ACROSS_DEGREE as f64;

    let common = scalars_times(&first.weight, &second.weight);
    let columns = raised_scalars(&common, ALONG_DEGREE);
    let [row0, row1, row2] = first.rows(&second.weight, &common, ALONG_DEGREE);
    let [row5, row4, row3] = second.rows(&first.weight, &common, ALONG_DEGREE);
    let rows = [row0, row1, row2, row3, row4, row5];
    let net: Vec<Vec<Vec3>> = (0..=ALONG_DEGREE)
        .map(|i| rows.iter().map(|row| row[i] * (1.0 / columns[i])).collect())
        .collect();
    let first_patch = NurbsSurface::new(
        ALONG_DEGREE,
        ACROSS_DEGREE,
        clamped(ALONG_DEGREE),
        clamped(ACROSS_DEGREE),
        net.iter()
            .map(|column| column.iter().map(|&p| to_point(p)).collect())
            .collect(),
        columns
            .iter()
            .map(|&w| vec![w; ACROSS_DEGREE + 1])
            .collect(),
    )?;

    let seam_net = net[0].clone();
    let along = ALONG_DEGREE as f64;
    let leaving: Vec<Vec3> = net[1]
        .iter()
        .zip(&seam_net)
        .map(|(&next, &on)| (next - on) * (along * columns[1] / columns[0]))
        .collect();
    let seam_speed: Vec<Vec3> = seam_net
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) * across)
        .collect();
    // The second patch's cross tangent is a mix of the first's and the seam's
    // own, so the normals agree all along. At each end the mix runs along the
    // stripe, into the link on the flat face, which fixes it there.
    let beta = [1.0, -1.0];
    let alpha = [
        -first.lean / first.start_speed(),
        -second.lean / second.start_speed(),
    ];
    let cross = plus(
        &times(&alpha, &leaving),
        &raised(&times(&beta, &seam_speed), ACROSS_DEGREE + 1),
        1.0,
    );

    let seam_columns = raised_scalars(&third.weight, SEAM_ALONG_DEGREE);
    let seam_row = raised(&times(&third.weight, &seam_net), SEAM_ALONG_DEGREE);
    let beside_seam: Vec<Vec3> = plus(&raised(&seam_net, ACROSS_DEGREE + 1), &cross, 1.0 / across);
    let beside_row = raised(&times(&third.weight, &beside_seam), SEAM_ALONG_DEGREE);
    let beyond_row: Vec<Vec3> = beside_row
        .iter()
        .zip(&seam_row)
        .map(|(&b, &s)| b * 2.0 - s)
        .collect();
    let arc_row = raised(&third.numerator, SEAM_ALONG_DEGREE);
    let shift = |k: f64| -> Vec<Vec3> {
        arc_row
            .iter()
            .zip(&seam_columns)
            .map(|(&a, &w)| a + third.inward * (k * third.reach / across * w))
            .collect()
    };
    let second_rows = [
        seam_row,
        beside_row,
        beyond_row,
        shift(2.0),
        shift(1.0),
        arc_row,
    ];
    let second_net: Vec<Vec<Vec3>> = (0..=SEAM_ALONG_DEGREE)
        .map(|i| {
            second_rows
                .iter()
                .map(|row| row[i] * (1.0 / seam_columns[i]))
                .collect()
        })
        .collect();
    let second_patch = NurbsSurface::new(
        SEAM_ALONG_DEGREE,
        ACROSS_DEGREE,
        clamped(SEAM_ALONG_DEGREE),
        clamped(ACROSS_DEGREE),
        second_net
            .iter()
            .map(|column| column.iter().map(|&p| to_point(p)).collect())
            .collect(),
        seam_columns
            .iter()
            .map(|&w| vec![w; ACROSS_DEGREE + 1])
            .collect(),
    )?;

    let points = |column: &[Vec3]| column.iter().map(|&p| to_point(p)).collect::<Vec<_>>();
    Ok(Some(CornerPatches {
        seam: curve(points(&seam_net))?,
        first_link: curve(points(&net[ALONG_DEGREE]))?,
        second_link: curve(points(&second_net[SEAM_ALONG_DEGREE]))?,
        third_link: curve(points(&second_net[0]))?,
        first: first_patch,
        second: second_patch,
    }))
}

const TOL: f64 = 1e-7;

fn refused(reason: impl Into<String>) -> BlendError {
    BlendError::PlanningFailure {
        reason: reason.into(),
    }
}

/// A ball corner's parts around one arc, in the corner's direction of travel.
struct RolledArc {
    edge: EdgeId,
    forward: bool,
    circle: Circle3D,
    from: VertexId,
    to: VertexId,
    stripe: FaceId,
    axis: Vec3,
}

/// A rolled corner: the ball's face, and for each of its arcs the stripe behind
/// it. `flats[i]` holds the arc's start.
struct RolledCorner {
    shell: ShellId,
    face: FaceId,
    center: Point3,
    radius: f64,
    arcs: [RolledArc; 3],
    flats: [FaceId; 3],
}

fn face_wires(face: &Face) -> Vec<WireId> {
    std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect()
}

fn faces_by_edge(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<HashMap<EdgeId, Vec<FaceId>>, BlendError> {
    let mut by_edge: HashMap<EdgeId, Vec<FaceId>> = HashMap::new();
    for &id in faces {
        for wire in face_wires(topo.face(id)?) {
            for oriented in topo.wire(wire)?.edges() {
                by_edge.entry(oriented.edge()).or_default().push(id);
            }
        }
    }
    Ok(by_edge)
}

fn traversed(topo: &Topology, oriented: OrientedEdge) -> Result<(VertexId, VertexId), BlendError> {
    let edge = topo.edge(oriented.edge())?;
    Ok(if oriented.is_forward() {
        (edge.start(), edge.end())
    } else {
        (edge.end(), edge.start())
    })
}

/// The arcs of a three-sided ball face, each with the cylinder stripe behind
/// it, or `None` when the face is not that. They run the way the face does once
/// its reversal is applied, which a new face that is not reversed then follows.
fn rolled_arcs(
    topo: &Topology,
    face: FaceId,
    by_edge: &HashMap<EdgeId, Vec<FaceId>>,
) -> Result<Option<[RolledArc; 3]>, BlendError> {
    let data = topo.face(face)?;
    let reversed = data.is_reversed();
    let wire = topo.wire(data.outer_wire())?;
    if wire.edges().len() != 3 {
        return Ok(None);
    }
    let mut uses = wire.edges().to_vec();
    if reversed {
        uses.reverse();
    }
    let mut arcs = Vec::with_capacity(3);
    for oriented in uses {
        let oriented = OrientedEdge::new(oriented.edge(), oriented.is_forward() != reversed);
        let EdgeCurve::Circle(circle) = topo.edge(oriented.edge())?.curve().clone() else {
            return Ok(None);
        };
        let Some(&stripe) = by_edge
            .get(&oriented.edge())
            .and_then(|faces| faces.iter().find(|&&other| other != face))
        else {
            return Ok(None);
        };
        let FaceSurface::Cylinder(cylinder) = topo.face(stripe)?.surface() else {
            return Ok(None);
        };
        let axis = cylinder.axis();
        let (from, to) = traversed(topo, oriented)?;
        arcs.push(RolledArc {
            edge: oriented.edge(),
            forward: oriented.is_forward(),
            circle,
            from,
            to,
            stripe,
            axis,
        });
    }
    Ok(arcs.try_into().ok())
}

/// The face among `faces` that touches `vertex` and is not one of `excluded`: at
/// a stripe's end, the flat face once the stripes and the ball are excluded.
fn flat_at(
    topo: &Topology,
    vertex: VertexId,
    faces: &[FaceId],
    excluded: &[FaceId],
) -> Result<Option<FaceId>, BlendError> {
    for &id in faces {
        if excluded.contains(&id) {
            continue;
        }
        for wire in face_wires(topo.face(id)?) {
            for oriented in topo.wire(wire)?.edges() {
                let edge = topo.edge(oriented.edge())?;
                if edge.start() == vertex || edge.end() == vertex {
                    return Ok(Some(id));
                }
            }
        }
    }
    Ok(None)
}

/// The ball corner where three flat faces through `vertex` meet.
fn find_rolled_corner(
    topo: &Topology,
    solid: SolidId,
    vertex: Point3,
) -> Result<Option<RolledCorner>, BlendError> {
    let solid_data = topo.solid(solid)?;
    let shells: Vec<ShellId> = std::iter::once(solid_data.outer_shell())
        .chain(solid_data.inner_shells().iter().copied())
        .collect();
    for shell in shells {
        let faces = topo.shell(shell)?.faces().to_vec();
        let by_edge = faces_by_edge(topo, &faces)?;
        for &face in &faces {
            let FaceSurface::Sphere(ball) = topo.face(face)?.surface() else {
                continue;
            };
            let (center, radius) = (ball.center(), ball.radius());
            let Some(arcs) = rolled_arcs(topo, face, &by_edge)? else {
                continue;
            };
            let mut flats = Vec::with_capacity(3);
            for arc in &arcs {
                let stripes: Vec<FaceId> = arcs.iter().map(|a| a.stripe).chain([face]).collect();
                let Some(flat) = flat_at(topo, arc.from, &faces, &stripes)? else {
                    break;
                };
                flats.push(flat);
            }
            let Ok(flats): Result<[FaceId; 3], _> = flats.try_into() else {
                continue;
            };
            let mut through_vertex = true;
            for flat in flats {
                let data = topo.face(flat)?;
                let FaceSurface::Plane { normal, d } = data.surface() else {
                    through_vertex = false;
                    break;
                };
                let height = normal.dot(Vec3::new(vertex.x(), vertex.y(), vertex.z())) - d;
                through_vertex &= height.abs() <= TOL * 100.0;
            }
            if through_vertex {
                return Ok(Some(RolledCorner {
                    shell,
                    face,
                    center,
                    radius,
                    arcs,
                    flats,
                }));
            }
        }
    }
    Ok(None)
}

/// A stripe's contact edge running to `vertex`, shortened to `moved`.
fn shortened(
    topo: &mut Topology,
    edge: EdgeId,
    vertex: VertexId,
    moved: VertexId,
) -> Result<EdgeId, BlendError> {
    let data = topo.edge(edge)?;
    let (start, end) = (data.start(), data.end());
    let far = if start == vertex { end } else { start };
    let (from, to) = (topo.vertex(vertex)?.point(), topo.vertex(far)?.point());
    let (t0, t1) = data.curve().domain_with_endpoints(from, to);
    let middle = data
        .curve()
        .evaluate_with_endpoints(0.5 * (t0 + t1), from, to);
    let chord_middle = from + (to - from) * 0.5;
    if (middle - chord_middle).length() > TOL * 100.0 {
        return Err(refused("a set-back corner needs straight contact lines"));
    }
    let stored = if start == vertex {
        Edge::new(moved, far, EdgeCurve::Line)
    } else {
        Edge::new(far, moved, EdgeCurve::Line)
    };
    Ok(topo.add_edge(stored))
}

/// A wire with each edge swapped as `swaps` says, and a link put in after the
/// edge arriving at the vertex `insert` names, run on from where that edge now
/// stops.
fn rewired(
    topo: &mut Topology,
    wire: WireId,
    swaps: &HashMap<EdgeId, EdgeId>,
    insert: Option<(VertexId, EdgeId)>,
) -> Result<WireId, BlendError> {
    let data = topo.wire(wire)?;
    let closed = data.is_closed();
    let old: Vec<OrientedEdge> = data.edges().to_vec();
    let mut edges = Vec::with_capacity(old.len() + 1);
    for oriented in old {
        let (_, arrives) = traversed(topo, oriented)?;
        let id = swaps
            .get(&oriented.edge())
            .copied()
            .unwrap_or_else(|| oriented.edge());
        let kept = OrientedEdge::new(id, oriented.is_forward());
        edges.push(kept);
        if let Some((at, link)) = insert
            && arrives == at
        {
            let (_, reached) = traversed(topo, kept)?;
            edges.push(OrientedEdge::new(link, topo.edge(link)?.start() == reached));
        }
    }
    Ok(topo.add_wire(Wire::new(edges, closed)?))
}

fn rebuilt_face(
    topo: &mut Topology,
    face: FaceId,
    swaps: &HashMap<EdgeId, EdgeId>,
    insert: Option<(VertexId, EdgeId)>,
) -> Result<FaceId, BlendError> {
    let data = topo.face(face)?;
    let (outer, inner, surface, reversed) = (
        data.outer_wire(),
        data.inner_wires().to_vec(),
        data.surface().clone(),
        data.is_reversed(),
    );
    let outer = rewired(topo, outer, swaps, insert)?;
    let mut holes = Vec::with_capacity(inner.len());
    for hole in inner {
        holes.push(rewired(topo, hole, swaps, insert)?);
    }
    let rebuilt = if reversed {
        Face::new_reversed(outer, holes, surface)
    } else {
        Face::new(outer, holes, surface)
    };
    Ok(topo.add_face(rebuilt))
}

/// The edges of `face` that end at `vertex`, other than `not`.
fn edges_at(
    topo: &Topology,
    face: FaceId,
    vertex: VertexId,
    not: EdgeId,
) -> Result<Vec<EdgeId>, BlendError> {
    let mut found = Vec::new();
    for wire in face_wires(topo.face(face)?) {
        for oriented in topo.wire(wire)?.edges() {
            let edge = topo.edge(oriented.edge())?;
            if oriented.edge() != not && (edge.start() == vertex || edge.end() == vertex) {
                found.push(oriented.edge());
            }
        }
    }
    Ok(found)
}

/// A rational Bezier piece of a curve in a plane, its control points held as
/// `[w x, w y, w]` with every weight positive, so the piece lies in their hull.
#[derive(Debug, Clone)]
struct Piece(Vec<[f64; 3]>);

impl Piece {
    fn points(&self) -> impl Iterator<Item = Point2> + '_ {
        self.0.iter().map(|&[x, y, w]| Point2::new(x / w, y / w))
    }

    fn ends(&self) -> Option<(Point2, Point2)> {
        let (&[x0, y0, w0], &[x1, y1, w1]) = (self.0.first()?, self.0.last()?);
        Some((Point2::new(x0 / w0, y0 / w0), Point2::new(x1 / w1, y1 / w1)))
    }

    /// The least and greatest of its control points along `axis`.
    fn extent(&self, axis: Vec2) -> (f64, f64) {
        self.points()
            .map(|p| axis.x().mul_add(p.x(), axis.y() * p.y()))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| {
                (lo.min(s), hi.max(s))
            })
    }

    /// The diagonal of its control points' box.
    fn size(&self) -> f64 {
        let (x0, x1) = self.extent(Vec2::new(1.0, 0.0));
        let (y0, y1) = self.extent(Vec2::new(0.0, 1.0));
        (x1 - x0).hypot(y1 - y0)
    }

    fn centroid(&self) -> Point2 {
        let n = self.0.len().max(1) as f64;
        let (x, y) = self
            .points()
            .fold((0.0, 0.0), |(x, y), p| (x + p.x(), y + p.y()));
        Point2::new(x / n, y / n)
    }

    /// The farthest a control point lies from the chord: the piece and the
    /// sliver between it and its chord lie no farther.
    fn bound(&self) -> f64 {
        let Some((a, b)) = self.ends() else {
            return f64::INFINITY;
        };
        self.points()
            .map(|p| distance_to_segment(p, a, b))
            .fold(0.0, f64::max)
    }

    fn halves(&self) -> (Self, Self) {
        let (left, right) = halved(&self.0);
        (Self(left), Self(right))
    }
}

/// A rational Bezier's control points, in homogeneous form, split at its
/// middle parameter by de Casteljau.
fn halved<const N: usize>(line: &[[f64; N]]) -> (Vec<[f64; N]>, Vec<[f64; N]>) {
    let mut work = line.to_vec();
    let n = work.len();
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for r in 0..n {
        if r > 0 {
            for j in 0..n - r {
                let (a, b) = (work[j], work[j + 1]);
                work[j] = std::array::from_fn(|k| 0.5 * (a[k] + b[k]));
            }
        }
        left.push(work[0]);
        right.push(work[n - 1 - r]);
    }
    right.reverse();
    (left, right)
}

fn distance_to_segment(p: Point2, a: Point2, b: Point2) -> f64 {
    let (ab, ap) = (b - a, p - a);
    let len2 = ab.dot(ab);
    let t = if len2 > 0.0 {
        (ap.dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (ap - ab * t).length()
}

/// Whether the hulls of `a` and `b` lie more than `margin` apart along one of
/// the axes that tell small pieces apart: the plane's own, each chord's normal,
/// and the line between their centroids.
fn apart(a: &Piece, b: &Piece, margin: f64) -> bool {
    let normal = |piece: &Piece| {
        piece.ends().map_or(Vec2::new(0.0, 0.0), |(s, e)| {
            Vec2::new(s.y() - e.y(), e.x() - s.x())
        })
    };
    [
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        normal(a),
        normal(b),
        b.centroid() - a.centroid(),
    ]
    .into_iter()
    .any(|axis| {
        let Ok(axis) = axis.normalize() else {
            return false;
        };
        let ((a0, a1), (b0, b1)) = (a.extent(axis), b.extent(axis));
        a1 + margin < b0 || b1 + margin < a0
    })
}

/// Whether a piece of `curve` comes within `margin` of a piece of `region`:
/// pairs whose hulls cannot be told apart are halved until they can, and a pair
/// both no larger than `margin`, or a search that runs out of halvings, reads
/// as near.
fn near(curve: &[Piece], region: &[Piece], margin: f64) -> bool {
    const MAX_PAIRS: usize = 1 << 16;
    let mut pending: Vec<(Piece, Piece)> = curve
        .iter()
        .flat_map(|a| region.iter().map(move |b| (a.clone(), b.clone())))
        .collect();
    let mut visited = 0_usize;
    while let Some((a, b)) = pending.pop() {
        visited += 1;
        if visited > MAX_PAIRS {
            return true;
        }
        if apart(&a, &b, margin) {
            continue;
        }
        let (size_a, size_b) = (a.size(), b.size());
        if size_a <= margin && size_b <= margin {
            return true;
        }
        if size_a >= size_b {
            let (first, second) = a.halves();
            pending.push((first, b.clone()));
            pending.push((second, b));
        } else {
            let (first, second) = b.halves();
            pending.push((a.clone(), first));
            pending.push((a, second));
        }
    }
    false
}

/// Whether `p`, more than `margin` from every piece of the closed loop
/// `region`, lies inside it. A piece whose chord lies farther from `p` than
/// `margin` plus the piece's bound winds about `p` as its chord does, so it
/// stands as that chord; a nearer one is halved. A reading in doubt, or one
/// that runs out of halvings, is inside.
fn inside(region: &[Piece], p: Point2, margin: f64) -> bool {
    const MAX_HALVINGS: usize = 4096;
    let mut chords = Vec::new();
    let mut pending: Vec<Piece> = region.to_vec();
    let mut halvings = 0_usize;
    while let Some(piece) = pending.pop() {
        let Some((a, b)) = piece.ends() else {
            return true;
        };
        if distance_to_segment(p, a, b) > margin + piece.bound() {
            chords.push(Boundary2::Segment(a, b));
        } else if halvings == MAX_HALVINGS {
            return true;
        } else {
            halvings += 1;
            let (first, second) = piece.halves();
            pending.push(first);
            pending.push(second);
        }
    }
    point_in_region(&chords, p, margin) != Some(false)
}

/// The rational Bezier pieces of an edge from `from` to `to` along `curve`,
/// each as control points `[w x, w y, w z, w]` whose hull holds it. A circle or
/// ellipse is cut into arcs of a quarter turn at most, each exactly a rational
/// quadratic; a spline into its knot spans. `None` where a spline's weight is
/// zero or less, and its control points no longer hold it.
fn curve_pieces(curve: &EdgeCurve, from: Point3, to: Point3) -> Option<Vec<Vec<[f64; 4]>>> {
    let plain = |p: Point3| [p.x(), p.y(), p.z(), 1.0];
    let (t0, t1) = curve.domain_with_endpoints(from, to);
    let conic = |center: Point3, u: Vec3, v: Vec3, a: f64, b: f64| {
        let at = |t: f64, scale: f64| center + (u * (a * t.cos()) + v * (b * t.sin())) * scale;
        let turns = ((t1 - t0) / std::f64::consts::FRAC_PI_2).ceil().max(1.0);
        let count = turns as usize;
        (0..count)
            .map(|i| {
                let s0 = (t1 - t0).mul_add(i as f64 / turns, t0);
                let s1 = (t1 - t0).mul_add((i + 1) as f64 / turns, t0);
                let half = 0.5 * (s1 - s0);
                let w = half.cos();
                let peak = at(s0 + half, 1.0 / w);
                vec![
                    plain(at(s0, 1.0)),
                    [peak.x() * w, peak.y() * w, peak.z() * w, w],
                    plain(at(s1, 1.0)),
                ]
            })
            .collect()
    };
    match curve {
        EdgeCurve::Line => Some(vec![vec![plain(from), plain(to)]]),
        EdgeCurve::Circle(c) => Some(conic(
            c.center(),
            c.u_axis(),
            c.v_axis(),
            c.radius(),
            c.radius(),
        )),
        EdgeCurve::Ellipse(e) => Some(conic(
            e.center(),
            e.u_axis(),
            e.v_axis(),
            e.semi_major(),
            e.semi_minor(),
        )),
        EdgeCurve::NurbsCurve(n) => {
            let (d0, d1) = n.domain();
            let (lo, hi) = (t0.min(t1).max(d0), t0.max(t1).min(d1));
            let mut breaks: Vec<f64> = std::iter::once(lo)
                .chain(n.knots().iter().copied().filter(|&k| k > lo && k < hi))
                .chain(std::iter::once(hi))
                .collect();
            breaks.dedup();
            if breaks.len() < 2 {
                return None;
            }
            let mut pieces = Vec::with_capacity(breaks.len() - 1);
            for span in breaks.windows(2) {
                let bezier = curve_bezier_between(n, span[0], span[1]).ok()?;
                if bezier.iter().any(|&[.., w]| w <= 0.0) {
                    return None;
                }
                pieces.push(bezier);
            }
            Some(pieces)
        }
    }
}

/// [`curve_pieces`] taken into a plane by `flat`.
fn edge_pieces(
    curve: &EdgeCurve,
    from: Point3,
    to: Point3,
    flat: impl Fn(Point3) -> Point2,
) -> Option<Vec<Piece>> {
    let pieces = curve_pieces(curve, from, to)?;
    Some(
        pieces
            .into_iter()
            .map(|piece| {
                Piece(
                    piece
                        .into_iter()
                        .map(|[x, y, z, w]| {
                            let p = flat(Point3::new(x / w, y / w, z / w));
                            [p.x() * w, p.y() * w, w]
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

/// Whether the outline of `flat` reaches what a set-back cuts away from it: the
/// region between its corner `at`, the points `a` and `b` its contact lines are
/// cut back to, and `link` between them.
///
/// Every edge but the two contact lines ending at `at` is read whole against
/// the region's boundary, and each loop that keeps clear of it by a point of
/// the loop: a hole may lie wholly inside, and the rest of the outline runs on
/// from the contact lines' far ends, past `a` and `b`. A spline edge whose
/// control points no longer hold it reads as reaching.
fn cut_reaches(
    topo: &Topology,
    flat: FaceId,
    contacts: [EdgeId; 2],
    [at, a, b]: [Point3; 3],
    link: &NurbsCurve,
) -> Result<bool, BlendError> {
    let data = topo.face(flat)?;
    let FaceSurface::Plane { normal, .. } = data.surface() else {
        return Ok(true);
    };
    let frame = Frame3::from_normal(at, *normal)?;
    let to_plane = |p: Point3| {
        let d = p - at;
        Point2::new(d.dot(frame.x), d.dot(frame.y))
    };
    let Some(link) = edge_pieces(&EdgeCurve::NurbsCurve(link.clone()), a, b, to_plane) else {
        return Ok(true);
    };
    let plain = |p: Point3| {
        let q = to_plane(p);
        [q.x(), q.y(), 1.0]
    };
    let mut region = vec![Piece(vec![plain(at), plain(a)])];
    region.extend(link);
    region.push(Piece(vec![plain(b), plain(at)]));

    for wire in face_wires(data) {
        let mut first = None;
        for oriented in topo.wire(wire)?.edges() {
            if contacts.contains(&oriented.edge()) {
                continue;
            }
            let edge = topo.edge(oriented.edge())?;
            let (from, to) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let Some(pieces) = edge_pieces(edge.curve(), from, to, to_plane) else {
                return Ok(true);
            };
            if near(&pieces, &region, TOL) {
                return Ok(true);
            }
            first.get_or_insert_with(|| to_plane(from));
        }
        if first.is_some_and(|p| inside(&region, p, TOL)) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// A box holding what a set-back changes: the region the ball, the stripes'
/// ends up to their new arcs, the flat faces' cut-away corners and the patches
/// enclose. It lies in the hull of those faces: the ball and the old arcs on
/// its sphere, the new arcs on their circles, each stripe's end between the
/// two, and the patches and the links on them in their control points. `None`
/// when a patch has a weight of zero or less.
fn changed_bounds(
    corner: &RolledCorner,
    ends: &[StripeEnd; 3],
    patches: &CornerPatches,
) -> Option<Aabb3> {
    let ball = Vec3::new(corner.radius, corner.radius, corner.radius);
    let mut points = vec![corner.center - ball, corner.center + ball];
    for (end, arc) in ends.iter().zip(&corner.arcs) {
        let r = arc.circle.radius();
        let round = Vec3::new(r, r, r);
        points.extend([end.center - round, end.center + round]);
    }
    for patch in [&patches.first, &patches.second] {
        if patch.weights().iter().flatten().any(|&w| w <= 0.0) {
            return None;
        }
        points.extend(patch.control_points().iter().flatten().copied());
    }
    Aabb3::try_from_points(points)
}

/// A rational Bezier piece of a corner patch, its control net held as
/// `[w x, w y, w z, w]` with every weight positive, so the piece lies in the
/// hull of its control points.
#[derive(Debug, Clone)]
struct PatchPiece(Vec<Vec<[f64; 4]>>);

impl PatchPiece {
    /// The whole of a single-span patch, or `None` when a weight is zero or less.
    fn new(patch: &NurbsSurface) -> Option<Self> {
        patch
            .control_points()
            .iter()
            .zip(patch.weights())
            .map(|(row, weights)| {
                row.iter()
                    .zip(weights)
                    .map(|(p, &w)| (w > 0.0).then(|| [p.x() * w, p.y() * w, p.z() * w, w]))
                    .collect()
            })
            .collect::<Option<_>>()
            .map(Self)
    }

    fn points(&self) -> Vec<Point3> {
        self.0
            .iter()
            .flatten()
            .map(|&[x, y, z, w]| Point3::new(x / w, y / w, z / w))
            .collect()
    }

    /// How far its net runs end to end along its first parameter and along its
    /// second, the most of any line of it.
    fn spans(&self) -> (f64, f64) {
        let at = |&[x, y, z, w]: &[f64; 4]| Point3::new(x / w, y / w, z / w);
        let (Some(first), Some(last)) = (self.0.first(), self.0.last()) else {
            return (0.0, 0.0);
        };
        let across = first
            .iter()
            .zip(last)
            .map(|(a, b)| (at(b) - at(a)).length())
            .fold(0.0, f64::max);
        let along = self
            .0
            .iter()
            .filter_map(|row| Some((at(row.first()?) - at(row.last()?)).length()))
            .fold(0.0, f64::max);
        (across, along)
    }

    /// The piece split at the middle of its first parameter, or of its second.
    fn halves(&self, first: bool) -> (Self, Self) {
        if first {
            let columns = self.0.first().map_or(0, Vec::len);
            let (mut left, mut right) = (
                vec![Vec::with_capacity(columns); self.0.len()],
                vec![Vec::with_capacity(columns); self.0.len()],
            );
            for j in 0..columns {
                let line: Vec<[f64; 4]> = self.0.iter().map(|row| row[j]).collect();
                let (a, b) = halved(&line);
                for (i, (p, q)) in a.into_iter().zip(b).enumerate() {
                    left[i].push(p);
                    right[i].push(q);
                }
            }
            (Self(left), Self(right))
        } else {
            let (left, right) = self.0.iter().map(|row| halved(row)).unzip();
            (Self(left), Self(right))
        }
    }
}

/// Points whose hull holds a face, and axes its own shape is told apart along.
type FaceHull = (Vec<Point3>, Vec<Vec3>);

/// Points whose hull holds `face`, and axes its own shape is best told apart
/// along. A plane's face lies in the hull of its outline. A cylinder's, cone's,
/// sphere's or torus's lies in a box of its own frame, as tall along the axis
/// as its outline reaches; a spline surface's in its control points. `None`
/// where a spline's weight is zero or less.
fn face_hull(topo: &Topology, face: FaceId) -> Result<Option<FaceHull>, BlendError> {
    let data = topo.face(face)?;
    let mut outline = Vec::new();
    for wire in face_wires(data) {
        for oriented in topo.wire(wire)?.edges() {
            let edge = topo.edge(oriented.edge())?;
            let (from, to) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let Some(pieces) = curve_pieces(edge.curve(), from, to) else {
                return Ok(None);
            };
            outline.extend(
                pieces
                    .into_iter()
                    .flatten()
                    .map(|[x, y, z, w]| Point3::new(x / w, y / w, z / w)),
            );
        }
    }
    let height = |origin: Point3, axis: Vec3| {
        outline
            .iter()
            .map(|&p| (p - origin).dot(axis))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), h| {
                (lo.min(h), hi.max(h))
            })
    };
    let boxed = |origin: Point3, axes: [Vec3; 3], lo: [f64; 3], hi: [f64; 3]| {
        let points = (0..8)
            .map(|corner: usize| {
                (0..3).fold(origin, |p, k| {
                    p + axes[k] * if (corner >> k) & 1 == 0 { lo[k] } else { hi[k] }
                })
            })
            .collect();
        (points, axes.to_vec())
    };
    Ok(Some(match data.surface() {
        FaceSurface::Plane { normal, .. } => (outline, vec![*normal]),
        FaceSurface::Cylinder(c) => {
            let ((h0, h1), r) = (height(c.origin(), c.axis()), c.radius());
            boxed(
                c.origin(),
                [c.x_axis(), c.y_axis(), c.axis()],
                [-r, -r, h0],
                [r, r, h1],
            )
        }
        FaceSurface::Cone(k) => {
            let (h0, h1) = height(k.apex(), k.axis());
            let r = h0.abs().max(h1.abs()) / k.half_angle().tan();
            boxed(
                k.apex(),
                [k.x_axis(), k.y_axis(), k.axis()],
                [-r, -r, h0],
                [r, r, h1],
            )
        }
        FaceSurface::Sphere(b) => {
            let r = b.radius();
            boxed(
                b.center(),
                [b.x_axis(), b.y_axis(), b.z_axis()],
                [-r; 3],
                [r; 3],
            )
        }
        FaceSurface::Torus(t) => {
            let (wide, tall) = (t.major_radius() + t.minor_radius(), t.minor_radius());
            boxed(
                t.center(),
                [t.x_axis(), t.y_axis(), t.z_axis()],
                [-wide, -wide, -tall],
                [wide, wide, tall],
            )
        }
        FaceSurface::Nurbs(n) => {
            if n.weights().iter().flatten().any(|&w| w <= 0.0) {
                return Ok(None);
            }
            (
                n.control_points().iter().flatten().copied().collect(),
                Vec::new(),
            )
        }
    }))
}

/// Whether the hulls of `a` and `b` lie more than `margin` apart along one of
/// `axes` or the line between their centroids.
fn apart_in_space(a: &[Point3], b: &[Point3], axes: &[Vec3], margin: f64) -> bool {
    let centroid = |points: &[Point3]| {
        let n = points.len().max(1) as f64;
        let sum = points.iter().fold(Vec3::new(0.0, 0.0, 0.0), |sum, &p| {
            sum + Vec3::new(p.x(), p.y(), p.z())
        });
        sum * (1.0 / n)
    };
    let extent = |points: &[Point3], axis: Vec3| {
        points
            .iter()
            .map(|&p| Vec3::new(p.x(), p.y(), p.z()).dot(axis))
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| {
                (lo.min(s), hi.max(s))
            })
    };
    axes.iter()
        .copied()
        .chain([centroid(b) - centroid(a)])
        .any(|axis| {
            let Ok(axis) = axis.normalize() else {
                return false;
            };
            let ((a0, a1), (b0, b1)) = (extent(a, axis), extent(b, axis));
            a1 + margin < b0 || b1 + margin < a0
        })
}

/// Whether one of `patches` could meet the face held in `hull`: a piece whose
/// net cannot be told apart from the hull is halved until it can, and one no
/// larger than `margin`, or a search that runs out of halvings, reads as
/// meeting it.
fn patches_reach(patches: &[PatchPiece], hull: &[Point3], axes: &[Vec3], margin: f64) -> bool {
    const MAX_PIECES: usize = 1 << 14;
    let mut pending = patches.to_vec();
    let mut visited = 0_usize;
    while let Some(piece) = pending.pop() {
        visited += 1;
        if visited > MAX_PIECES {
            return true;
        }
        if apart_in_space(&piece.points(), hull, axes, margin) {
            continue;
        }
        let (across, along) = piece.spans();
        if across.max(along) <= margin {
            return true;
        }
        let (first, second) = piece.halves(across >= along);
        pending.push(first);
        pending.push(second);
    }
    false
}

/// Whether the patches could meet a face of `solid` other than the corner's
/// ball, stripes and flat faces. Another face cannot cross those, and the flat
/// faces' outlines keep clear of their cut-away corners. So it reaches the
/// changed material only through a patch, or, in a cavity, by lying wholly
/// inside it, which the caller reads. A face that ends a stripe short of its
/// new arc takes the arc, and so a patch, through it.
fn patches_meet_other_faces(
    topo: &Topology,
    solid: SolidId,
    corner: &RolledCorner,
    directions: &[Vec3],
    patches: &CornerPatches,
) -> Result<bool, BlendError> {
    let (Some(first), Some(second)) = (
        PatchPiece::new(&patches.first),
        PatchPiece::new(&patches.second),
    ) else {
        return Ok(true);
    };
    let pieces = [first, second];
    let mut own: Vec<FaceId> = corner.arcs.iter().map(|arc| arc.stripe).collect();
    own.push(corner.face);
    own.extend(corner.flats);
    let mut axes = vec![
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    axes.extend_from_slice(directions);
    for &flat in &corner.flats {
        if let FaceSurface::Plane { normal, .. } = topo.face(flat)?.surface() {
            axes.push(*normal);
        }
    }
    let solid_data = topo.solid(solid)?;
    let shells: Vec<ShellId> = std::iter::once(solid_data.outer_shell())
        .chain(solid_data.inner_shells().iter().copied())
        .collect();
    for shell in shells {
        for &face in topo.shell(shell)?.faces() {
            if own.contains(&face) {
                continue;
            }
            let Some((hull, own_axes)) = face_hull(topo, face)? else {
                return Ok(true);
            };
            let face_axes: Vec<Vec3> = axes.iter().copied().chain(own_axes).collect();
            if patches_reach(&pieces, &hull, &face_axes, TOL) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// How far along `direction` from `vertex` a corner's rounds stop today, for
/// each `direction`, or `None` when no ball corner stands at `vertex`.
///
/// # Errors
///
/// Returns [`BlendError`] if the solid cannot be read.
pub fn rolling_distances(
    topo: &Topology,
    solid: SolidId,
    vertex: Point3,
    directions: &[Vec3; 3],
) -> Result<Option<[f64; 3]>, BlendError> {
    let Some(corner) = find_rolled_corner(topo, solid, vertex)? else {
        return Ok(None);
    };
    Ok(Some(directions.map(|d| (corner.center - vertex).dot(d))))
}

/// Stop the three rounds at the ball corner over `vertex` further back.
///
/// `setbacks` says how far from `vertex` along each edge's direction. The hole
/// that leaves is closed with [`corner_patches`] and a link on each flat face.
/// Returns the new solid and a box holding all the material the change takes
/// off or adds.
///
/// # Errors
///
/// Returns [`BlendError::PlanningFailure`] when no ball corner stands at
/// `vertex`, a direction matches no stripe, a set-back is not past the ball or
/// not short of the stripe's far end, a flat face's outline reaches the part
/// of it the corner cuts away, or another face of the solid could meet a patch.
#[allow(clippy::too_many_lines)]
pub fn set_back_corner(
    topo: &mut Topology,
    solid: SolidId,
    vertex: Point3,
    setbacks: &[(Vec3, f64); 3],
) -> Result<(SolidId, Aabb3), BlendError> {
    let mut corner = find_rolled_corner(topo, solid, vertex)?
        .ok_or_else(|| refused("no ball corner between three flat faces stands at that vertex"))?;

    let mut set_for = Vec::with_capacity(3);
    for arc in &corner.arcs {
        let &(away, distance) = setbacks
            .iter()
            .find(|(direction, _)| direction.cross(arc.axis).length() <= 1e-9)
            .ok_or_else(|| refused("a set-back's edge runs along none of the corner's rounds"))?;
        set_for.push((away, distance));
    }
    // Which round the second patch holds moves the seam, and the wire's order
    // says nothing about the part: a mirror image runs it the other way. So it
    // is the round set back furthest, and then the one turning through most.
    let past_of = |k: usize| set_for[k].1 - (corner.center - vertex).dot(set_for[k].0);
    let turn_of = |k: usize| {
        let arc = &corner.arcs[k];
        let (Ok(a), Ok(b)) = (topo.vertex(arc.from), topo.vertex(arc.to)) else {
            return 0.0;
        };
        (a.point() - b.point()).length() / arc.circle.radius()
    };
    let last = (0..3).fold(0, |best, k| {
        let (ours, theirs) = (past_of(k), past_of(best));
        let wider = (ours - theirs).abs() <= 1e-9 && turn_of(k) > turn_of(best) + 1e-9;
        if ours > theirs + 1e-9 || wider {
            k
        } else {
            best
        }
    });
    corner.arcs.rotate_left((last + 1) % 3);
    corner.flats.rotate_left((last + 1) % 3);
    set_for.rotate_left((last + 1) % 3);

    let mut moves = [Vec3::new(0.0, 0.0, 0.0); 3];
    let mut contacts = [[corner.arcs[0].edge; 2]; 3];
    let mut ends = Vec::with_capacity(3);
    let mut starts = Vec::with_capacity(3);
    let mut stops = Vec::with_capacity(3);
    for (i, arc) in corner.arcs.iter().enumerate() {
        let (away, distance) = set_for[i];
        let rolling = (corner.center - vertex).dot(away);
        let past = distance - rolling;
        if past <= arc.circle.radius() * 1e-3 {
            return Err(refused(format!(
                "a set-back of {distance} is not past the ball, which stops its round at {rolling}"
            )));
        }
        for (end, at) in [arc.from, arc.to].into_iter().enumerate() {
            let contact = edges_at(topo, arc.stripe, at, arc.edge)?;
            let [contact] = contact.as_slice() else {
                return Err(refused("a stripe's end has no single contact edge"));
            };
            contacts[i][end] = *contact;
            let edge = topo.edge(*contact)?;
            let far = if edge.start() == at {
                edge.end()
            } else {
                edge.start()
            };
            let room = (topo.vertex(far)?.point() - topo.vertex(at)?.point()).dot(away);
            if past >= room - TOL {
                return Err(refused(format!(
                    "a set-back of {distance} runs past the end of its round's edge"
                )));
            }
        }
        moves[i] = away * past;
        let (start, stop) = (
            topo.vertex(arc.from)?.point() + moves[i],
            topo.vertex(arc.to)?.point() + moves[i],
        );
        ends.push(StripeEnd {
            center: arc.circle.center() + moves[i],
            start,
            end: stop,
            inward: -away,
            reach: REACH_PER_SETBACK * past,
        });
        starts.push(start);
        stops.push(stop);
    }
    let ends: [StripeEnd; 3] = ends
        .try_into()
        .map_err(|_| refused("a corner has three stripe ends"))?;
    let patches =
        corner_patches(&ends)?.ok_or_else(|| refused("a stripe's end has no angle to read"))?;
    // Each flat face's corner, between its two contact lines and the link, is
    // cut away: the rest of its outline must keep clear of it.
    let links = [
        &patches.third_link,
        &patches.first_link,
        &patches.second_link,
    ];
    for k in 0..3 {
        let at = topo.vertex(corner.arcs[k].from)?.point();
        let (a, b) = (starts[k], stops[(k + 2) % 3]);
        let meeting = [contacts[k][0], contacts[(k + 2) % 3][1]];
        if cut_reaches(topo, corner.flats[k], meeting, [at, a, b], links[k])? {
            return Err(refused(
                "a hole or notch in a flat face reaches where the set-back corner cuts it away",
            ));
        }
    }
    let directions: Vec<Vec3> = set_for.iter().map(|&(away, _)| away).collect();
    if patches_meet_other_faces(topo, solid, &corner, &directions, &patches)? {
        return Err(refused(
            "a face of the solid lies where the set-back corner's patches would cut through it",
        ));
    }
    let reach = changed_bounds(&corner, &ends, &patches)
        .ok_or_else(|| refused("a corner patch has a weight of zero or less"))?;

    let mut new_from = Vec::with_capacity(3);
    let mut new_to = Vec::with_capacity(3);
    for i in 0..3 {
        new_from.push(topo.add_vertex(Vertex::new(starts[i], TOL)));
        new_to.push(topo.add_vertex(Vertex::new(stops[i], TOL)));
    }

    let mut swaps: HashMap<EdgeId, EdgeId> = HashMap::new();
    let mut new_arcs = Vec::with_capacity(3);
    for i in 0..3 {
        let arc = &corner.arcs[i];
        let circle = Circle3D::new(
            arc.circle.center() + moves[i],
            arc.circle.normal(),
            arc.circle.radius(),
        )?;
        let (first, last) = if arc.forward {
            (new_from[i], new_to[i])
        } else {
            (new_to[i], new_from[i])
        };
        let new_arc = topo.add_edge(Edge::new(first, last, EdgeCurve::Circle(circle)));
        swaps.insert(arc.edge, new_arc);
        new_arcs.push(new_arc);
        let (from, to, edge, stripe) = (arc.from, arc.to, arc.edge, arc.stripe);
        for (at, moved) in [(from, new_from[i]), (to, new_to[i])] {
            let contacts = edges_at(topo, stripe, at, edge)?;
            let new_contact = shortened(topo, contacts[0], at, moved)?;
            swaps.insert(contacts[0], new_contact);
        }
    }

    let link_edge = |topo: &mut Topology, from: VertexId, to: VertexId, curve: &NurbsCurve| {
        topo.add_edge(Edge::new(from, to, EdgeCurve::NurbsCurve(curve.clone())))
    };
    let seam = link_edge(topo, new_from[0], new_to[1], &patches.seam);
    let first_link = link_edge(topo, new_to[0], new_from[1], &patches.first_link);
    let second_link = link_edge(topo, new_to[1], new_from[2], &patches.second_link);
    let third_link = link_edge(topo, new_from[0], new_to[2], &patches.third_link);

    let arc_use = |i: usize| OrientedEdge::new(new_arcs[i], corner.arcs[i].forward);
    let first_wire = topo.add_wire(Wire::new(
        vec![
            arc_use(0),
            OrientedEdge::new(first_link, true),
            arc_use(1),
            OrientedEdge::new(seam, false),
        ],
        true,
    )?);
    let second_wire = topo.add_wire(Wire::new(
        vec![
            OrientedEdge::new(seam, true),
            OrientedEdge::new(second_link, true),
            arc_use(2),
            OrientedEdge::new(third_link, false),
        ],
        true,
    )?);
    let first_face = topo.add_face(Face::new(
        first_wire,
        Vec::new(),
        FaceSurface::Nurbs(patches.first),
    ));
    let second_face = topo.add_face(Face::new(
        second_wire,
        Vec::new(),
        FaceSurface::Nurbs(patches.second),
    ));

    let flat_links = [
        (corner.arcs[0].from, third_link),
        (corner.arcs[1].from, first_link),
        (corner.arcs[2].from, second_link),
    ];
    let mut replaced: HashMap<FaceId, FaceId> = HashMap::new();
    for (&flat, &(at, link)) in corner.flats.iter().zip(&flat_links) {
        let rebuilt = rebuilt_face(topo, flat, &swaps, Some((at, link)))?;
        replaced.insert(flat, rebuilt);
    }
    for arc in &corner.arcs {
        let rebuilt = rebuilt_face(topo, arc.stripe, &swaps, None)?;
        replaced.insert(arc.stripe, rebuilt);
    }

    let solid_data = topo.solid(solid)?;
    let (outer, inner) = (solid_data.outer_shell(), solid_data.inner_shells().to_vec());
    let mut shell_faces = Vec::new();
    for &face in topo.shell(corner.shell)?.faces() {
        if face == corner.face {
            shell_faces.extend([first_face, second_face]);
        } else {
            shell_faces.push(replaced.get(&face).copied().unwrap_or(face));
        }
    }
    let new_shell = topo.add_shell(Shell::new(shell_faces)?);
    let swap_shell = |shell: ShellId| {
        if shell == corner.shell {
            new_shell
        } else {
            shell
        }
    };
    let new_inner = inner.into_iter().map(swap_shell).collect();
    Ok((
        topo.add_solid(Solid::new(swap_shell(outer), new_inner)),
        reach,
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// A box corner at the origin with the material in the positive octant,
    /// its rounds of radius 2 set back past the ball by `past`, in the direction
    /// of travel a ball's wire runs.
    fn box_corner(past: [f64; 3]) -> [StripeEnd; 3] {
        let r = 2.0;
        let (z, y, x) = (r + past[0], r + past[1], r + past[2]);
        [
            StripeEnd {
                center: Point3::new(r, r, z),
                start: Point3::new(r, 0.0, z),
                end: Point3::new(0.0, r, z),
                inward: Vec3::new(0.0, 0.0, -1.0),
                reach: REACH_PER_SETBACK * past[0],
            },
            StripeEnd {
                center: Point3::new(r, y, r),
                start: Point3::new(0.0, y, r),
                end: Point3::new(r, y, 0.0),
                inward: Vec3::new(0.0, -1.0, 0.0),
                reach: REACH_PER_SETBACK * past[1],
            },
            StripeEnd {
                center: Point3::new(x, r, r),
                start: Point3::new(x, r, 0.0),
                end: Point3::new(x, 0.0, r),
                inward: Vec3::new(-1.0, 0.0, 0.0),
                reach: REACH_PER_SETBACK * past[2],
            },
        ]
    }

    const PAST: [[f64; 3]; 6] = [
        [0.1, 0.1, 0.1],
        [1.0, 1.0, 1.0],
        [4.0, 4.0, 4.0],
        [0.2, 2.0, 1.0],
        [3.0, 0.5, 0.5],
        [0.5, 3.0, 0.5],
    ];

    /// Curvature of a patch along a tangent direction at a parameter point.
    fn normal_curvature(patch: &NurbsSurface, u: f64, v: f64, along: Vec3) -> f64 {
        let d = patch.derivatives(u, v, 2);
        let (su, sv, suu, suv, svv) = (d[1][0], d[0][1], d[2][0], d[1][1], d[0][2]);
        let normal = su.cross(sv).normalize().unwrap();
        let (e, f, g) = (su.dot(su), su.dot(sv), sv.dot(sv));
        let det = e * g - f * f;
        let (x, y) = (along.dot(su), along.dot(sv));
        let (du, dv) = ((g * x - f * y) / det, (e * y - f * x) / det);
        (suu.dot(normal) * du * du + 2.0 * suv.dot(normal) * du * dv + svv.dot(normal) * dv * dv)
            / (e * du * du + 2.0 * f * du * dv + g * dv * dv)
    }

    #[test]
    fn the_patches_hold_their_neighbours_to_the_order_they_claim() {
        const STEPS: u32 = 64;
        let samples =
            (0..=STEPS).map(|i| (f64::from(i) / f64::from(STEPS)).clamp(1e-4, 1.0 - 1e-4));
        for past in PAST {
            let ends = box_corner(past);
            let patches = corner_patches(&ends).unwrap().unwrap();
            for t in samples.clone() {
                // The seam, read just off it from each side.
                let (a, b) = (
                    patches.first.normal(0.0, t).unwrap(),
                    patches.second.normal(t, 0.0).unwrap(),
                );
                assert!(a.cross(b).length() < 1e-9, "{past:?}: seam at {t}");

                // Each flat face along its link.
                for (normal, flat) in [
                    (
                        patches.first.normal(1.0, t).unwrap(),
                        Vec3::new(1.0, 0.0, 0.0),
                    ),
                    (
                        patches.second.normal(0.0, t).unwrap(),
                        Vec3::new(0.0, 1.0, 0.0),
                    ),
                    (
                        patches.second.normal(1.0, t).unwrap(),
                        Vec3::new(0.0, 0.0, 1.0),
                    ),
                ] {
                    assert!(normal.cross(flat).length() < 1e-9, "{past:?}: link at {t}");
                }

                // Each arc: on the cylinder, tangent to it, and curving like it.
                for (patch, v, end, reversed) in [
                    (&patches.first, 0.0, &ends[0], false),
                    (&patches.first, 1.0, &ends[1], true),
                    (&patches.second, 1.0, &ends[2], true),
                ] {
                    let on = patch.evaluate(t, v);
                    let radial = on - end.center;
                    assert!(
                        (radial.length() - 2.0).abs() < 1e-12,
                        "{past:?}: off the arc"
                    );
                    assert!(
                        radial.dot(end.inward).abs() < 1e-12,
                        "{past:?}: off its plane"
                    );
                    let radial = radial.normalize().unwrap();
                    let normal = patch.normal(t, v).unwrap();
                    assert!(
                        normal.cross(radial).length() < 1e-9,
                        "{past:?}: tangent at {t}"
                    );
                    let round = end.inward.cross(radial).normalize().unwrap();
                    let across = normal_curvature(patch, t, v, end.inward);
                    let around = normal_curvature(patch, t, v, round);
                    assert!(
                        across.abs() < 1e-9,
                        "{past:?} rev={reversed}: across {across}"
                    );
                    assert!(
                        (around.abs() - 0.5).abs() < 1e-9,
                        "{past:?}: around {around}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_patches_face_one_way_with_no_fold_or_ripple() {
        const STEPS: u32 = 40;
        let outward = Vec3::new(-1.0, -1.0, -1.0).normalize().unwrap();
        for past in PAST {
            let patches = corner_patches(&box_corner(past)).unwrap().unwrap();
            for patch in [&patches.first, &patches.second] {
                for i in 1..STEPS {
                    for j in 1..STEPS {
                        let (u, v) = (
                            f64::from(i) / f64::from(STEPS),
                            f64::from(j) / f64::from(STEPS),
                        );
                        let normal = patch.normal(u, v).unwrap();
                        assert!(
                            normal.dot(outward) > 0.5,
                            "{past:?}: faces away at ({u}, {v})"
                        );
                    }
                }
            }
        }
    }
}
