//! IGES file reader.
//!
//! Reads trimmed surfaces (144) over planes (108) and B-spline surfaces (128),
//! and bounded planes (108 form 1), each boundary a curve on the surface (142)
//! holding lines (110), B-spline curves (126) or a composite of them (102). The
//! faces are sewn into shells by the positions of their boundaries, each set of
//! connected faces its own solid, or a cavity of the solid around it where it
//! faces inward. Other entities are skipped, and so is a face that needs them.
//!
//! Solids in a file that share an edge come back as one.

use std::collections::{BTreeMap, HashMap};

use brepkit_math::nurbs::{NurbsCurve, NurbsSurface};
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::{Shell, ShellId};
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::IoError;

/// How near two boundary points read as one vertex.
const SEW_TOL: f64 = 1e-6;

/// Read an IGES file and reconstruct topology.
///
/// Returns the list of solid IDs created. Each set of connected faces found in
/// the IGES file is assembled into a solid.
///
/// # Errors
///
/// Returns [`IoError`] if the file is malformed.
pub fn read_iges(input: &str, topo: &mut Topology) -> Result<Vec<SolidId>, IoError> {
    let entities = parse_iges_entities(input)?;
    build_topology(topo, &entities)
}

// ── IGES entity representation ──────────────────────────────────────

/// A parsed IGES entity with its directory entry and parameter data.
#[derive(Debug)]
struct IgesEntity {
    /// Entity type number (e.g., 110 for line, 128 for NURBS surface).
    entity_type: u32,
    /// The raw parameter data string (comma-separated values).
    params: String,
    /// Directory entry sequence number (used for cross-referencing).
    de_seq: u32,
}

// ── Parsing ─────────────────────────────────────────────────────────

/// Parse all entities from an IGES file.
fn parse_iges_entities(input: &str) -> Result<Vec<IgesEntity>, IoError> {
    let mut d_lines: Vec<&str> = Vec::new();
    let mut p_lines: Vec<&str> = Vec::new();

    for line in input.lines() {
        if line.len() < 73 {
            continue;
        }
        let section = line.as_bytes().get(72).copied().unwrap_or(b' ');
        match section {
            b'D' => d_lines.push(line),
            b'P' => p_lines.push(line),
            _ => {} // Skip S, G, T sections.
        }
    }

    // Parse directory entries (pairs of lines).
    let mut dir_entries: Vec<(u32, u32, u32)> = Vec::new(); // (entity_type, pd_start, de_seq)

    let mut i = 0;
    while i + 1 < d_lines.len() {
        let line1 = d_lines[i];

        let entity_type = parse_int_field(line1, 0, 8)?;
        let pd_start = parse_int_field(line1, 8, 16)?;

        // DE sequence number is in columns 73-80.
        let de_seq = parse_int_field(line1, 73, 80)?;

        dir_entries.push((entity_type, pd_start, de_seq));
        i += 2; // Skip the second line of the DE pair.
    }

    // Collect parameter data by DE pointer.
    // P-section lines have format: data (cols 0-63), DE pointer (cols 64-72), "P" + seq.
    let mut pd_by_de: HashMap<u32, String> = HashMap::new();

    for p_line in &p_lines {
        let data_part = if p_line.len() >= 64 {
            &p_line[..64]
        } else {
            p_line
        };
        let de_ptr = if p_line.len() >= 72 {
            parse_int_field(p_line, 64, 72).unwrap_or(0)
        } else {
            0
        };

        pd_by_de
            .entry(de_ptr)
            .or_default()
            .push_str(data_part.trim_end());
    }

    let mut entities = Vec::new();
    for (entity_type, _pd_start, de_seq) in &dir_entries {
        let params = pd_by_de.get(de_seq).cloned().unwrap_or_default();
        // Strip the entity type number prefix from params (e.g., "110,..." → "...").
        let clean_params = strip_entity_prefix(&params, *entity_type);

        entities.push(IgesEntity {
            entity_type: *entity_type,
            params: clean_params,
            de_seq: *de_seq,
        });
    }

    Ok(entities)
}

/// Parse an integer from a fixed-width field in an IGES line.
fn parse_int_field(line: &str, start: usize, end: usize) -> Result<u32, IoError> {
    let end = end.min(line.len());
    if start >= end {
        return Ok(0);
    }
    let field = line[start..end].trim();
    if field.is_empty() {
        return Ok(0);
    }
    field.parse::<u32>().map_err(|e| IoError::ParseError {
        reason: format!("invalid IGES integer field '{field}': {e}"),
    })
}

/// Strip the entity type prefix from parameter data.
/// E.g., "110,1.0,2.0,..." → "1.0,2.0,..."
fn strip_entity_prefix(params: &str, entity_type: u32) -> String {
    let prefix = format!("{entity_type},");
    params.strip_prefix(&prefix).unwrap_or(params).to_string()
}

// ── Reading parameters ──────────────────────────────────────────────

/// A cursor over an entity's parameters.
struct Params {
    values: Vec<f64>,
    next: usize,
}

impl Params {
    fn of(entity: &IgesEntity) -> Self {
        Self {
            values: parse_float_params(&entity.params),
            next: 0,
        }
    }

    fn real(&mut self) -> Result<f64, IoError> {
        let value = self
            .values
            .get(self.next)
            .copied()
            .ok_or_else(|| IoError::ParseError {
                reason: "an IGES entity has too few parameters".to_string(),
            })?;
        self.next += 1;
        Ok(value)
    }

    fn reals(&mut self, count: usize) -> Result<Vec<f64>, IoError> {
        (0..count).map(|_| self.real()).collect()
    }

    fn point(&mut self) -> Result<Point3, IoError> {
        Ok(Point3::new(self.real()?, self.real()?, self.real()?))
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn count(&mut self) -> Result<usize, IoError> {
        let value = self.real()?;
        if value < 0.0 {
            return Err(IoError::ParseError {
                reason: format!("an IGES entity has a negative count, {value}"),
            });
        }
        Ok(value as usize)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn pointer(&mut self) -> Result<u32, IoError> {
        Ok(self.real()?.abs() as u32)
    }
}

/// One stretch of a boundary, running the way the boundary does.
enum Stretch {
    Line(Point3, Point3),
    Curve {
        curve: NurbsCurve,
        from: f64,
        to: f64,
    },
}

impl Stretch {
    fn ends(&self) -> (Point3, Point3) {
        match self {
            Self::Line(a, b) => (*a, *b),
            Self::Curve { curve, from, to } => (curve.evaluate(*from), curve.evaluate(*to)),
        }
    }

    fn at(&self, fraction: f64) -> Point3 {
        match self {
            Self::Line(a, b) => *a + (*b - *a) * fraction,
            Self::Curve { curve, from, to } => curve.evaluate(from + (to - from) * fraction),
        }
    }

    fn into_curve(self) -> EdgeCurve {
        match self {
            Self::Line(..) => EdgeCurve::Line,
            Self::Curve { curve, .. } => EdgeCurve::NurbsCurve(curve),
        }
    }
}

/// An edge and the points along it that tell it from another with its ends,
/// and which way it runs when both ends are one vertex.
struct SewnEdge {
    id: EdgeId,
    start: VertexId,
    quarter: Point3,
    middle: Point3,
}

/// The file's entities by directory entry, and what has been sewn so far.
struct Sewing<'a> {
    entities: HashMap<u32, &'a IgesEntity>,
    /// Vertices in cells of `SEW_TOL`, searched with the cells around.
    vertices: HashMap<(i64, i64, i64), Vec<(Point3, VertexId)>>,
    /// Edges by their two end vertices, lowest first.
    edges: HashMap<(usize, usize), Vec<SewnEdge>>,
}

impl<'a> Sewing<'a> {
    fn new(entities: &'a [IgesEntity]) -> Self {
        Self {
            entities: entities.iter().map(|e| (e.de_seq, e)).collect(),
            vertices: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    fn entity(&self, pointer: u32) -> Result<&'a IgesEntity, IoError> {
        self.entities
            .get(&pointer)
            .copied()
            .ok_or_else(|| IoError::ParseError {
                reason: format!("an IGES entity points at no entity, {pointer}"),
            })
    }

    /// The vertex at `p`, made when none lies near.
    #[allow(clippy::cast_possible_truncation)]
    fn vertex(&mut self, topo: &mut Topology, p: Point3) -> VertexId {
        let cell = |v: f64| (v / SEW_TOL).round() as i64;
        let (x, y, z) = (cell(p.x()), cell(p.y()), cell(p.z()));
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    if let Some(found) = self.vertices.get(&(x + i, y + j, z + k))
                        && let Some(&(_, id)) =
                            found.iter().find(|(q, _)| (*q - p).length() <= SEW_TOL)
                    {
                        return id;
                    }
                }
            }
        }
        let id = topo.add_vertex(Vertex::new(p, SEW_TOL));
        self.vertices.entry((x, y, z)).or_default().push((p, id));
        id
    }

    /// The edge along `stretch`, made when no other runs along it, and whether
    /// this use runs with it.
    fn edge(&mut self, topo: &mut Topology, stretch: Stretch) -> OrientedEdge {
        let (a, b) = stretch.ends();
        let (start, end) = (self.vertex(topo, a), self.vertex(topo, b));
        let key = if start.index() <= end.index() {
            (start.index(), end.index())
        } else {
            (end.index(), start.index())
        };
        let (quarter, middle) = (stretch.at(0.25), stretch.at(0.5));
        let near = |p: Point3, q: Point3| (p - q).length() <= 10.0 * SEW_TOL;
        if let Some(sewn) = self.edges.get(&key) {
            for edge in sewn {
                if near(edge.middle, middle) {
                    let forward = if start == end {
                        near(edge.quarter, quarter)
                    } else {
                        edge.start == start
                    };
                    return OrientedEdge::new(edge.id, forward);
                }
            }
        }
        let id = topo.add_edge(Edge::new(start, end, stretch.into_curve()));
        self.edges.entry(key).or_default().push(SewnEdge {
            id,
            start,
            quarter,
            middle,
        });
        OrientedEdge::new(id, true)
    }

    /// The stretches of a curve entity, in the order they run.
    fn stretches(&self, pointer: u32) -> Result<Vec<Stretch>, IoError> {
        let entity = self.entity(pointer)?;
        let mut params = Params::of(entity);
        Ok(match entity.entity_type {
            102 => {
                let count = params.count()?;
                let mut out = Vec::new();
                for _ in 0..count {
                    out.extend(self.stretches(params.pointer()?)?);
                }
                out
            }
            110 => vec![Stretch::Line(params.point()?, params.point()?)],
            126 => vec![read_curve(&mut params)?],
            other => {
                return Err(IoError::ParseError {
                    reason: format!("IGES curve entity {other} is not read"),
                });
            }
        })
    }

    /// A boundary as a wire, from a curve on a surface or a curve itself.
    fn boundary(&mut self, topo: &mut Topology, pointer: u32) -> Result<WireId, IoError> {
        let entity = self.entity(pointer)?;
        let curve = if entity.entity_type == 142 {
            let mut params = Params::of(entity);
            // Creation, surface, parameter-space curve, then the model-space one.
            params.reals(3)?;
            params.pointer()?
        } else {
            pointer
        };
        let stretches = self.stretches(curve)?;
        let edges = stretches.into_iter().map(|s| self.edge(topo, s)).collect();
        let wire = Wire::new(edges, true).map_err(|e| IoError::ParseError {
            reason: format!("an IGES boundary does not close: {e}"),
        })?;
        Ok(topo.add_wire(wire))
    }

    /// A trimmed surface as a face.
    fn trimmed_surface(
        &mut self,
        topo: &mut Topology,
        entity: &IgesEntity,
    ) -> Result<FaceId, IoError> {
        let mut params = Params::of(entity);
        let surface = params.pointer()?;
        let bounded = params.count()?;
        let holes = params.count()?;
        let outer = params.pointer()?;
        if bounded == 0 || outer == 0 {
            return Err(IoError::ParseError {
                reason: "an IGES trimmed surface uses its surface's own boundary".to_string(),
            });
        }
        let mut hole_pointers = Vec::with_capacity(holes);
        for _ in 0..holes {
            hole_pointers.push(params.pointer()?);
        }

        let surface = self.surface(surface)?;
        let outer = self.boundary(topo, outer)?;
        let mut inner = Vec::with_capacity(holes);
        for hole in hole_pointers {
            inner.push(self.boundary(topo, hole)?);
        }
        Ok(topo.add_face(Face::new(outer, inner, surface)))
    }

    fn surface(&self, pointer: u32) -> Result<FaceSurface, IoError> {
        let entity = self.entity(pointer)?;
        let mut params = Params::of(entity);
        match entity.entity_type {
            108 => read_plane(&mut params),
            128 => read_surface(&mut params),
            other => Err(IoError::ParseError {
                reason: format!("IGES surface entity {other} is not read"),
            }),
        }
    }
}

/// A plane: `A, B, C, D` with `Ax + By + Cz = D`.
fn read_plane(params: &mut Params) -> Result<FaceSurface, IoError> {
    let normal = Vec3::new(params.real()?, params.real()?, params.real()?);
    let d = params.real()?;
    let length = normal.length();
    if length < 1e-10 {
        return Err(IoError::ParseError {
            reason: "IGES plane has zero normal".to_string(),
        });
    }
    Ok(FaceSurface::Plane {
        normal: normal * (1.0 / length),
        d: d / length,
    })
}

/// A rational B-spline curve (126), over the span it gives.
fn read_curve(params: &mut Params) -> Result<Stretch, IoError> {
    let last = params.count()?;
    let degree = params.count()?;
    params.reals(4)?; // Closed, planar, polynomial, periodic.
    let knots = params.reals(last + degree + 2)?;
    let weights = params.reals(last + 1)?;
    let mut points = Vec::with_capacity(last + 1);
    for _ in 0..=last {
        points.push(params.point()?);
    }
    let (from, to) = (params.real()?, params.real()?);
    let curve =
        NurbsCurve::new(degree, knots, points, weights).map_err(|e| IoError::ParseError {
            reason: format!("an IGES curve is malformed: {e}"),
        })?;
    Ok(Stretch::Curve { curve, from, to })
}

/// A rational B-spline surface (128). Its first index varies fastest.
fn read_surface(params: &mut Params) -> Result<FaceSurface, IoError> {
    let (last_u, last_v) = (params.count()?, params.count()?);
    let (degree_u, degree_v) = (params.count()?, params.count()?);
    params.reals(5)?; // Closed both ways, polynomial, periodic both ways.
    let knots_u = params.reals(last_u + degree_u + 2)?;
    let knots_v = params.reals(last_v + degree_v + 2)?;

    let mut weights = vec![vec![0.0; last_v + 1]; last_u + 1];
    for j in 0..=last_v {
        for row in &mut weights {
            row[j] = params.real()?;
        }
    }
    let mut points = vec![vec![Point3::new(0.0, 0.0, 0.0); last_v + 1]; last_u + 1];
    for j in 0..=last_v {
        for row in &mut points {
            row[j] = params.point()?;
        }
    }
    let surface = NurbsSurface::new(degree_u, degree_v, knots_u, knots_v, points, weights)
        .map_err(|e| IoError::ParseError {
            reason: format!("an IGES surface is malformed: {e}"),
        })?;
    Ok(FaceSurface::Nurbs(surface))
}

// ── Topology building ───────────────────────────────────────────────

/// Build topology from parsed IGES entities.
fn build_topology(topo: &mut Topology, entities: &[IgesEntity]) -> Result<Vec<SolidId>, IoError> {
    let mut sewing = Sewing::new(entities);
    let mut face_ids = Vec::new();

    for entity in entities {
        let face = match entity.entity_type {
            144 => sewing.trimmed_surface(topo, entity).map(Some),
            108 => bounded_plane(topo, &mut sewing, entity),
            _ => continue,
        };
        match face {
            Ok(Some(face)) => face_ids.push(face),
            Ok(None) => {}
            Err(e) => log::warn!("skipping IGES entity {}: {e}", entity.de_seq),
        }
    }

    if face_ids.is_empty() {
        return Ok(Vec::new());
    }
    assemble_solids(topo, &face_ids)
}

/// A plane bounding itself (form 1) as a face. A plane a trimmed surface uses,
/// or with no boundary at all, is no face of its own.
fn bounded_plane(
    topo: &mut Topology,
    sewing: &mut Sewing<'_>,
    entity: &IgesEntity,
) -> Result<Option<FaceId>, IoError> {
    let mut params = Params::of(entity);
    let surface = read_plane(&mut params)?;
    let boundary = params.pointer().unwrap_or(0);
    if boundary == 0 {
        return Ok(None);
    }
    let wire = sewing.boundary(topo, boundary)?;
    Ok(Some(topo.add_face(Face::new(wire, Vec::new(), surface))))
}

/// The root of `i` among faces joined so far.
fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Sets of faces connected through shared edges, as solids. A set facing
/// inward is a cavity of the solid whose box holds it.
fn assemble_solids(topo: &mut Topology, face_ids: &[FaceId]) -> Result<Vec<SolidId>, IoError> {
    let mut parent: Vec<usize> = (0..face_ids.len()).collect();
    let mut first_use: HashMap<EdgeId, usize> = HashMap::new();
    for (index, &face) in face_ids.iter().enumerate() {
        let data = topo.face(face).map_err(topo_err)?;
        let wires: Vec<WireId> = std::iter::once(data.outer_wire())
            .chain(data.inner_wires().iter().copied())
            .collect();
        for wire in wires {
            for oe in topo.wire(wire).map_err(topo_err)?.edges() {
                let other = *first_use.entry(oe.edge()).or_insert(index);
                let (a, b) = (root(&mut parent, other), root(&mut parent, index));
                parent[a] = b;
            }
        }
    }
    // Sets keep the order their first faces came in, so a result does not hang
    // on how the joining ran.
    let mut sets: BTreeMap<usize, Vec<FaceId>> = BTreeMap::new();
    let mut first_of_root: HashMap<usize, usize> = HashMap::new();
    for (index, &face) in face_ids.iter().enumerate() {
        let r = root(&mut parent, index);
        let first = *first_of_root.entry(r).or_insert(index);
        sets.entry(first).or_default().push(face);
    }

    let mut outers: Vec<(SolidId, ShellId)> = Vec::new();
    let mut cavities = Vec::new();
    let mut lone = Vec::new();
    for faces in sets.into_values() {
        let shell = topo.add_shell(Shell::new(faces).map_err(|e| IoError::ParseError {
            reason: format!("failed to build shell: {e}"),
        })?);
        let solid = topo.add_solid(Solid::new(shell, Vec::new()));
        // A set that cannot be measured, an open sheet say, is its own solid.
        let extent = brepkit_operations::measure::solid_bounding_box(topo, solid)
            .map_or(1.0, |b| (b.max - b.min).length());
        match brepkit_operations::measure::oriented_solid_volume(topo, solid, extent * 1e-3) {
            Ok(volume) if volume < 0.0 => cavities.push((solid, shell)),
            Ok(_) => outers.push((solid, shell)),
            Err(_) => lone.push(solid),
        }
    }

    let mut held: HashMap<SolidId, Vec<ShellId>> = HashMap::new();
    for (cavity, shell) in cavities {
        let inner = brepkit_operations::measure::solid_bounding_box(topo, cavity)?;
        let mut holder = None;
        for &(solid, _) in &outers {
            let outer = brepkit_operations::measure::solid_bounding_box(topo, solid)?;
            let holds = outer.min.x() <= inner.min.x()
                && outer.min.y() <= inner.min.y()
                && outer.min.z() <= inner.min.z()
                && outer.max.x() >= inner.max.x()
                && outer.max.y() >= inner.max.y()
                && outer.max.z() >= inner.max.z();
            if holds {
                holder = Some(solid);
                break;
            }
        }
        match holder {
            Some(solid) => held.entry(solid).or_default().push(shell),
            None => lone.push(cavity),
        }
    }

    let mut solids = Vec::with_capacity(outers.len() + lone.len());
    for (solid, shell) in outers {
        match held.remove(&solid) {
            Some(inner) => solids.push(topo.add_solid(Solid::new(shell, inner))),
            None => solids.push(solid),
        }
    }
    solids.extend(lone);
    Ok(solids)
}

/// Convert a [`TopologyError`] into an [`IoError`].
fn topo_err(e: brepkit_topology::TopologyError) -> IoError {
    IoError::Operations(brepkit_operations::OperationsError::from(e))
}

/// Parse comma-separated float parameters from IGES parameter data, up to its
/// terminator. A parameter left out reads as 0, its usual default, and an
/// exponent may be written with `D`.
fn parse_float_params(params: &str) -> Vec<f64> {
    let clean = params.split(';').next().unwrap_or("");
    clean
        .split(',')
        .map(|s| {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return 0.0;
            }
            trimmed
                .parse::<f64>()
                .or_else(|_| trimmed.replace(['D', 'd'], "E").parse::<f64>())
                .unwrap_or(0.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_non_manifold;

    use super::*;
    use crate::iges::writer;

    #[test]
    fn roundtrip_unit_cube() {
        let mut write_topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut write_topo);

        let iges_str = writer::write_iges(&write_topo, &[solid]).unwrap();

        let mut read_topo = Topology::new();
        let solids = read_iges(&iges_str, &mut read_topo).unwrap();

        assert_eq!(solids.len(), 1);
        let read_solid = read_topo.solid(solids[0]).unwrap();
        let shell = read_topo.shell(read_solid.outer_shell()).unwrap();
        // Unit cube has 6 plane entities.
        assert_eq!(shell.faces().len(), 6);
    }

    #[test]
    fn roundtrip_box_primitive() {
        let mut write_topo = Topology::new();
        let solid =
            brepkit_operations::primitives::make_box(&mut write_topo, 2.0, 3.0, 4.0).unwrap();

        let iges_str = writer::write_iges(&write_topo, &[solid]).unwrap();

        let mut read_topo = Topology::new();
        let solids = read_iges(&iges_str, &mut read_topo).unwrap();

        assert_eq!(solids.len(), 1);
    }

    #[test]
    fn empty_file_returns_empty() {
        let mut topo = Topology::new();
        let solids = read_iges("", &mut topo).unwrap();
        assert!(solids.is_empty());
    }

    #[test]
    fn parse_float_params_basic() {
        let floats = parse_float_params("1.0,2.5,-3.0,0.;");
        assert_eq!(floats.len(), 4);
        assert!((floats[0] - 1.0).abs() < 1e-10);
        assert!((floats[1] - 2.5).abs() < 1e-10);
        assert!((floats[2] - (-3.0)).abs() < 1e-10);
        assert!((floats[3]).abs() < 1e-10);
    }

    #[test]
    fn parse_int_field_basic() {
        let val = parse_int_field("     108       1", 0, 8).unwrap();
        assert_eq!(val, 108);
    }
}
