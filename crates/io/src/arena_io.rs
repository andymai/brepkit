//! Exact serialization of a single solid's topology sub-arena.
//!
//! Captures every entity reachable from a [`SolidId`] — vertices (with exact
//! `Point3` and tolerance), edges (curve + analytic params), wires, faces
//! (surface + analytic params + reversed flag), shells, the solid, and the
//! pcurves on the captured (edge, face) pairs — and replays them into a fresh
//! [`Topology`] with byte-identical f64 values.
//!
//! Unlike the geometry-exchange formats (STEP, IGES), this preserves the
//! kernel's in-memory representation verbatim: no curve/surface re-derivation,
//! no tolerance normalization, no vertex welding. It exists so an in-memory
//! operand captured from a live session (e.g. a WASM kernel) can be replayed
//! in a native Rust harness with the *exact* floating-point state that drives
//! sub-ULP-sensitive boolean behavior.
//!
//! Entity ids are remapped to dense local indices in deterministic discovery
//! order, so the dump is compact and self-contained (independent of the
//! source arena's global id layout).

use std::collections::HashMap;

use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::curves2d::Curve2D;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::surfaces::{
    ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
};
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::pcurve::PCurve;
use brepkit_topology::shell::{Shell, ShellId};
use brepkit_topology::solid::SolidId;
use brepkit_topology::topology::Topology;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};
use serde::{Deserialize, Serialize};

use crate::IoError;

/// Serialized form of an [`EdgeCurve`].
#[derive(Debug, Clone, Serialize, Deserialize)]
enum SerEdgeCurve {
    Line,
    NurbsCurve(NurbsCurve),
    Circle(Circle3D),
    Ellipse(Ellipse3D),
}

impl SerEdgeCurve {
    fn from_curve(curve: &EdgeCurve) -> Self {
        match curve {
            EdgeCurve::Line => Self::Line,
            EdgeCurve::NurbsCurve(c) => Self::NurbsCurve(c.clone()),
            EdgeCurve::Circle(c) => Self::Circle(c.clone()),
            EdgeCurve::Ellipse(e) => Self::Ellipse(e.clone()),
        }
    }

    fn into_curve(self) -> EdgeCurve {
        match self {
            Self::Line => EdgeCurve::Line,
            Self::NurbsCurve(c) => EdgeCurve::NurbsCurve(c),
            Self::Circle(c) => EdgeCurve::Circle(c),
            Self::Ellipse(e) => EdgeCurve::Ellipse(e),
        }
    }
}

/// Serialized form of a [`FaceSurface`].
#[derive(Debug, Clone, Serialize, Deserialize)]
enum SerFaceSurface {
    Plane { normal: Vec3, d: f64 },
    Nurbs(NurbsSurface),
    Cylinder(CylindricalSurface),
    Cone(ConicalSurface),
    Sphere(SphericalSurface),
    Torus(ToroidalSurface),
}

impl SerFaceSurface {
    fn from_surface(surface: &FaceSurface) -> Self {
        match surface {
            FaceSurface::Plane { normal, d } => Self::Plane {
                normal: *normal,
                d: *d,
            },
            FaceSurface::Nurbs(s) => Self::Nurbs(s.clone()),
            FaceSurface::Cylinder(s) => Self::Cylinder(s.clone()),
            FaceSurface::Cone(s) => Self::Cone(s.clone()),
            FaceSurface::Sphere(s) => Self::Sphere(s.clone()),
            FaceSurface::Torus(s) => Self::Torus(s.clone()),
        }
    }

    fn into_surface(self) -> FaceSurface {
        match self {
            Self::Plane { normal, d } => FaceSurface::Plane { normal, d },
            Self::Nurbs(s) => FaceSurface::Nurbs(s),
            Self::Cylinder(s) => FaceSurface::Cylinder(s),
            Self::Cone(s) => FaceSurface::Cone(s),
            Self::Sphere(s) => FaceSurface::Sphere(s),
            Self::Torus(s) => FaceSurface::Torus(s),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerVertex {
    point: Point3,
    tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerEdge {
    start: usize,
    end: usize,
    curve: SerEdgeCurve,
    tolerance: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerOrientedEdge {
    edge: usize,
    forward: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerWire {
    edges: Vec<SerOrientedEdge>,
    closed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerFace {
    outer_wire: usize,
    inner_wires: Vec<usize>,
    surface: SerFaceSurface,
    reversed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerShell {
    faces: Vec<usize>,
}

/// A pcurve attached to a captured (edge, face) pair, keyed by local indices.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerPCurve {
    edge: usize,
    face: usize,
    curve: Curve2D,
    t_start: f64,
    t_end: f64,
}

/// Self-contained, byte-exact dump of a solid's topology sub-arena.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedSolid {
    /// Format version, so a future change can be detected on load.
    version: u32,
    vertices: Vec<SerVertex>,
    edges: Vec<SerEdge>,
    wires: Vec<SerWire>,
    faces: Vec<SerFace>,
    shells: Vec<SerShell>,
    /// Local index of the solid's outer shell.
    outer_shell: usize,
    /// Local indices of the solid's inner (cavity) shells.
    inner_shells: Vec<usize>,
    pcurves: Vec<SerPCurve>,
}

const FORMAT_VERSION: u32 = 1;

/// Discovers and remaps a solid's reachable entities into dense local indices.
struct Builder<'a> {
    topo: &'a Topology,
    vertices: Vec<SerVertex>,
    edges: Vec<SerEdge>,
    wires: Vec<SerWire>,
    faces: Vec<SerFace>,
    shells: Vec<SerShell>,
    vertex_map: HashMap<usize, usize>,
    edge_map: HashMap<usize, usize>,
    wire_map: HashMap<usize, usize>,
    face_map: HashMap<usize, usize>,
    shell_map: HashMap<usize, usize>,
}

impl<'a> Builder<'a> {
    fn new(topo: &'a Topology) -> Self {
        Self {
            topo,
            vertices: Vec::new(),
            edges: Vec::new(),
            wires: Vec::new(),
            faces: Vec::new(),
            shells: Vec::new(),
            vertex_map: HashMap::new(),
            edge_map: HashMap::new(),
            wire_map: HashMap::new(),
            face_map: HashMap::new(),
            shell_map: HashMap::new(),
        }
    }

    fn intern_vertex(&mut self, id: brepkit_topology::vertex::VertexId) -> Result<usize, IoError> {
        if let Some(&local) = self.vertex_map.get(&id.index()) {
            return Ok(local);
        }
        let v = self.topo.vertex(id)?;
        let local = self.vertices.len();
        self.vertices.push(SerVertex {
            point: v.point(),
            tolerance: v.tolerance(),
        });
        self.vertex_map.insert(id.index(), local);
        Ok(local)
    }

    fn intern_edge(&mut self, id: EdgeId) -> Result<usize, IoError> {
        if let Some(&local) = self.edge_map.get(&id.index()) {
            return Ok(local);
        }
        let e = self.topo.edge(id)?;
        let start = self.intern_vertex(e.start())?;
        let end = self.intern_vertex(e.end())?;
        let curve = SerEdgeCurve::from_curve(e.curve());
        let tolerance = e.tolerance();
        let local = self.edges.len();
        self.edges.push(SerEdge {
            start,
            end,
            curve,
            tolerance,
        });
        self.edge_map.insert(id.index(), local);
        Ok(local)
    }

    fn intern_wire(&mut self, id: brepkit_topology::wire::WireId) -> Result<usize, IoError> {
        if let Some(&local) = self.wire_map.get(&id.index()) {
            return Ok(local);
        }
        let w = self.topo.wire(id)?;
        let mut edges = Vec::with_capacity(w.edges().len());
        for oe in w.edges() {
            let edge = self.intern_edge(oe.edge())?;
            edges.push(SerOrientedEdge {
                edge,
                forward: oe.is_forward(),
            });
        }
        let closed = w.is_closed();
        let local = self.wires.len();
        self.wires.push(SerWire { edges, closed });
        self.wire_map.insert(id.index(), local);
        Ok(local)
    }

    fn intern_face(&mut self, id: brepkit_topology::face::FaceId) -> Result<usize, IoError> {
        if let Some(&local) = self.face_map.get(&id.index()) {
            return Ok(local);
        }
        let f = self.topo.face(id)?;
        let outer_wire = self.intern_wire(f.outer_wire())?;
        let mut inner_wires = Vec::with_capacity(f.inner_wires().len());
        for &iw in f.inner_wires() {
            inner_wires.push(self.intern_wire(iw)?);
        }
        let surface = SerFaceSurface::from_surface(f.surface());
        let reversed = f.is_reversed();
        let local = self.faces.len();
        self.faces.push(SerFace {
            outer_wire,
            inner_wires,
            surface,
            reversed,
        });
        self.face_map.insert(id.index(), local);
        Ok(local)
    }

    fn intern_shell(&mut self, id: ShellId) -> Result<usize, IoError> {
        if let Some(&local) = self.shell_map.get(&id.index()) {
            return Ok(local);
        }
        let s = self.topo.shell(id)?;
        let mut faces = Vec::with_capacity(s.faces().len());
        for &fid in s.faces() {
            faces.push(self.intern_face(fid)?);
        }
        let local = self.shells.len();
        self.shells.push(SerShell { faces });
        self.shell_map.insert(id.index(), local);
        Ok(local)
    }

    /// Collects all pcurves whose (edge, face) are both in the captured set.
    fn collect_pcurves(&self) -> Vec<SerPCurve> {
        let mut out = Vec::new();
        for (&global_face, &local_face) in &self.face_map {
            let Some(fid) = self.topo.face_id_from_index(global_face) else {
                continue;
            };
            for (eid, pc) in self.topo.pcurves().pcurves_for_face(fid) {
                if let Some(&local_edge) = self.edge_map.get(&eid.index()) {
                    out.push(SerPCurve {
                        edge: local_edge,
                        face: local_face,
                        curve: pc.curve().clone(),
                        t_start: pc.t_start(),
                        t_end: pc.t_end(),
                    });
                }
            }
        }
        // Deterministic order so the dump is reproducible across runs
        // (HashMap iteration order is randomized per process).
        out.sort_unstable_by_key(|p| (p.face, p.edge));
        out
    }
}

/// Serializes a solid's complete topology sub-arena to a byte buffer.
///
/// The result captures every vertex, edge, wire, face, shell reachable from
/// `solid_id`, plus the pcurves on the captured (edge, face) pairs, with
/// byte-identical f64 values. Load it with [`deserialize_solid`].
///
/// # Errors
///
/// Returns [`IoError`] if any referenced entity is missing or serialization
/// fails.
pub fn serialize_solid(topo: &Topology, solid_id: SolidId) -> Result<Vec<u8>, IoError> {
    let solid = topo.solid(solid_id)?;
    let mut builder = Builder::new(topo);

    let outer_shell = builder.intern_shell(solid.outer_shell())?;
    let mut inner_shells = Vec::with_capacity(solid.inner_shells().len());
    for &sh in solid.inner_shells() {
        inner_shells.push(builder.intern_shell(sh)?);
    }
    let pcurves = builder.collect_pcurves();

    let dump = SerializedSolid {
        version: FORMAT_VERSION,
        vertices: builder.vertices,
        edges: builder.edges,
        wires: builder.wires,
        faces: builder.faces,
        shells: builder.shells,
        outer_shell,
        inner_shells,
        pcurves,
    };

    serde_json::to_vec(&dump).map_err(|e| IoError::ParseError {
        reason: format!("arena serialization failed: {e}"),
    })
}

/// Reconstructs a solid from a buffer produced by [`serialize_solid`] into
/// `topo`, returning the new [`SolidId`].
///
/// All entities are appended to `topo` as fresh ids. Floating-point values are
/// restored byte-for-byte; analytic curves and surfaces are rebuilt by direct
/// field population (no constructor re-derivation), so the parametric frame is
/// preserved exactly.
///
/// # Errors
///
/// Returns [`IoError`] if the buffer is malformed, references an out-of-range
/// local index, holds a curve, surface or pcurve that is not well formed (see
/// [`EdgeCurve::is_well_formed`], [`FaceSurface::is_well_formed`] and
/// [`Curve2D::is_well_formed`]), or any entity construction fails.
pub fn deserialize_solid(bytes: &[u8], topo: &mut Topology) -> Result<SolidId, IoError> {
    let dump: SerializedSolid = serde_json::from_slice(bytes).map_err(|e| IoError::ParseError {
        reason: format!("arena deserialization failed: {e}"),
    })?;
    if dump.version != FORMAT_VERSION {
        return Err(IoError::ParseError {
            reason: format!(
                "unsupported arena dump version {} (expected {FORMAT_VERSION})",
                dump.version
            ),
        });
    }

    let mut vertex_ids = Vec::with_capacity(dump.vertices.len());
    for v in dump.vertices {
        vertex_ids.push(topo.add_vertex(Vertex::new(v.point, v.tolerance)));
    }

    let mut edge_ids = Vec::with_capacity(dump.edges.len());
    for (index, e) in dump.edges.into_iter().enumerate() {
        let start = *vertex_ids
            .get(e.start)
            .ok_or_else(|| index_err("vertex", e.start))?;
        let end = *vertex_ids
            .get(e.end)
            .ok_or_else(|| index_err("vertex", e.end))?;
        let curve = e.curve.into_curve();
        if !curve.is_well_formed() {
            return Err(IoError::ParseError {
                reason: format!("arena dump holds a malformed curve of edge {index}"),
            });
        }
        edge_ids.push(topo.add_edge(Edge::with_tolerance(start, end, curve, e.tolerance)));
    }

    let mut wire_ids = Vec::with_capacity(dump.wires.len());
    for w in dump.wires {
        let mut oriented = Vec::with_capacity(w.edges.len());
        for oe in w.edges {
            let edge = *edge_ids
                .get(oe.edge)
                .ok_or_else(|| index_err("edge", oe.edge))?;
            oriented.push(OrientedEdge::new(edge, oe.forward));
        }
        wire_ids.push(topo.add_wire(Wire::new(oriented, w.closed)?));
    }

    let mut face_ids = Vec::with_capacity(dump.faces.len());
    for (index, f) in dump.faces.into_iter().enumerate() {
        let outer = *wire_ids
            .get(f.outer_wire)
            .ok_or_else(|| index_err("wire", f.outer_wire))?;
        let mut inner = Vec::with_capacity(f.inner_wires.len());
        for iw in f.inner_wires {
            inner.push(*wire_ids.get(iw).ok_or_else(|| index_err("wire", iw))?);
        }
        let surface = f.surface.into_surface();
        if !surface.is_well_formed() {
            return Err(IoError::ParseError {
                reason: format!("arena dump holds a malformed surface of face {index}"),
            });
        }
        let mut face = Face::new(outer, inner, surface);
        face.set_reversed(f.reversed);
        face_ids.push(topo.add_face(face));
    }

    let mut shell_ids = Vec::with_capacity(dump.shells.len());
    for s in dump.shells {
        let mut faces = Vec::with_capacity(s.faces.len());
        for fid in s.faces {
            faces.push(*face_ids.get(fid).ok_or_else(|| index_err("face", fid))?);
        }
        shell_ids.push(topo.add_shell(Shell::new(faces)?));
    }

    for pc in dump.pcurves {
        let edge = *edge_ids
            .get(pc.edge)
            .ok_or_else(|| index_err("edge", pc.edge))?;
        let face = *face_ids
            .get(pc.face)
            .ok_or_else(|| index_err("face", pc.face))?;
        if !(pc.t_start.is_finite() && pc.t_end.is_finite() && pc.curve.is_well_formed()) {
            return Err(IoError::ParseError {
                reason: format!(
                    "arena dump holds a malformed pcurve of edge {} on face {}",
                    pc.edge, pc.face
                ),
            });
        }
        topo.pcurves_mut()
            .set(edge, face, PCurve::new(pc.curve, pc.t_start, pc.t_end));
    }

    let outer = *shell_ids
        .get(dump.outer_shell)
        .ok_or_else(|| index_err("shell", dump.outer_shell))?;
    let inner = dump
        .inner_shells
        .iter()
        .map(|&i| {
            shell_ids
                .get(i)
                .copied()
                .ok_or_else(|| index_err("shell", i))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(topo.add_solid(brepkit_topology::solid::Solid::new(outer, inner)))
}

fn index_err(kind: &str, index: usize) -> IoError {
    IoError::ParseError {
        reason: format!("arena dump references out-of-range {kind} index {index}"),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_topology::explorer::solid_faces;

    fn face_type_histogram(
        topo: &Topology,
        solid: SolidId,
    ) -> std::collections::BTreeMap<&'static str, usize> {
        let mut hist = std::collections::BTreeMap::new();
        for fid in solid_faces(topo, solid).unwrap() {
            let tag = topo.face(fid).unwrap().surface().type_tag();
            *hist.entry(tag).or_insert(0) += 1;
        }
        hist
    }

    #[test]
    fn roundtrip_box_preserves_counts_and_exact_bits() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 10.0, 20.0, 30.0).unwrap();

        let bytes = serialize_solid(&topo, solid).unwrap();
        let mut topo2 = Topology::new();
        let solid2 = deserialize_solid(&bytes, &mut topo2).unwrap();

        // Entity counts identical for the sub-arena.
        assert_eq!(topo2.num_vertices(), topo.num_vertices());
        assert_eq!(topo2.num_edges(), topo.num_edges());
        assert_eq!(topo2.num_wires(), topo.num_wires());
        assert_eq!(topo2.num_faces(), topo.num_faces());
        assert_eq!(topo2.num_shells(), topo.num_shells());
        assert_eq!(topo2.num_solids(), 1);

        // Face-type breakdown identical.
        assert_eq!(
            face_type_histogram(&topo2, solid2),
            face_type_histogram(&topo, solid)
        );

        // Sampled vertex position must match bit-for-bit.
        let orig: Vec<Point3> = solid_faces(&topo, solid)
            .unwrap()
            .iter()
            .flat_map(|&fid| {
                let w = topo.face(fid).unwrap().outer_wire();
                topo.wire(w)
                    .unwrap()
                    .edges()
                    .iter()
                    .map(|oe| topo.edge(oe.edge()).unwrap().start())
                    .map(|vid| topo.vertex(vid).unwrap().point())
                    .collect::<Vec<_>>()
            })
            .collect();
        let restored: Vec<Point3> = solid_faces(&topo2, solid2)
            .unwrap()
            .iter()
            .flat_map(|&fid| {
                let w = topo2.face(fid).unwrap().outer_wire();
                topo2
                    .wire(w)
                    .unwrap()
                    .edges()
                    .iter()
                    .map(|oe| topo2.edge(oe.edge()).unwrap().start())
                    .map(|vid| topo2.vertex(vid).unwrap().point())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(orig.len(), restored.len());
        for (a, b) in orig.iter().zip(&restored) {
            assert_eq!(a.x().to_bits(), b.x().to_bits(), "x bits differ");
            assert_eq!(a.y().to_bits(), b.y().to_bits(), "y bits differ");
            assert_eq!(a.z().to_bits(), b.z().to_bits(), "z bits differ");
        }
    }

    #[test]
    fn roundtrip_keeps_floats_a_best_effort_parser_rounds_off() {
        // Each offset parses one ulp off without exact float parsing.
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 10.0, 20.0, 30.0).unwrap();
        brepkit_operations::transform::transform_solid(
            &mut topo,
            solid,
            &brepkit_math::mat::Mat4::translation(
                1.055_261_488_752_658_3e-14,
                20.029_999_999_999_966,
                -7.035_076_591_683_869e-14,
            ),
        )
        .unwrap();

        let bytes = serialize_solid(&topo, solid).unwrap();
        let mut topo2 = Topology::new();
        let solid2 = deserialize_solid(&bytes, &mut topo2).unwrap();
        assert!(serialize_solid(&topo2, solid2).unwrap() == bytes);
    }

    #[test]
    fn roundtrip_cylinder_preserves_analytic_surface_exact() {
        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 7.5, 12.5).unwrap();

        let bytes = serialize_solid(&topo, solid).unwrap();
        let mut topo2 = Topology::new();
        let solid2 = deserialize_solid(&bytes, &mut topo2).unwrap();

        // The cylindrical surface must round-trip with the exact same frame.
        let mut cyl_orig = None;
        for fid in solid_faces(&topo, solid).unwrap() {
            if let FaceSurface::Cylinder(c) = topo.face(fid).unwrap().surface() {
                cyl_orig = Some(c.clone());
            }
        }
        let mut cyl_restored = None;
        for fid in solid_faces(&topo2, solid2).unwrap() {
            if let FaceSurface::Cylinder(c) = topo2.face(fid).unwrap().surface() {
                cyl_restored = Some(c.clone());
            }
        }
        let a = cyl_orig.expect("orig has cylinder");
        let b = cyl_restored.expect("restored has cylinder");
        assert_eq!(a.radius().to_bits(), b.radius().to_bits());
        for i in 0..3 {
            assert_eq!(a.origin().0[i].to_bits(), b.origin().0[i].to_bits());
            assert_eq!(a.axis().0[i].to_bits(), b.axis().0[i].to_bits());
            assert_eq!(a.x_axis().0[i].to_bits(), b.x_axis().0[i].to_bits());
            assert_eq!(a.y_axis().0[i].to_bits(), b.y_axis().0[i].to_bits());
        }
    }

    /// A dump holds whatever data its bytes hold. A B-spline curve or surface
    /// whose degree passes its control points, whose knots fall, or whose
    /// weights are not positive normal floats panics, hangs or gives a wrong
    /// solid when it is evaluated, and so does a circle or cylinder of no
    /// radius or with no frame: such a dump is refused. The dump's text
    /// cannot hold a number that is not finite, so its parse refuses one.
    /// The dumps they were made from load and validate.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn a_dump_with_a_malformed_curve_or_surface_is_refused() {
        use serde_json::{Value, json};

        let mut topo = Topology::new();
        let plain = make_cylinder(&mut topo, 3.0, 4.0).unwrap();
        let spline = make_cylinder(&mut topo, 3.0, 4.0).unwrap();
        brepkit_operations::heal::convert_to_bspline(&mut topo, spline).unwrap();
        let plain = serialize_solid(&topo, plain).unwrap();
        let spline = serialize_solid(&topo, spline).unwrap();
        for bytes in [&plain, &spline] {
            let mut fresh = Topology::new();
            let solid = deserialize_solid(bytes, &mut fresh).unwrap();
            let report = brepkit_operations::validate::validate_solid(&fresh, solid).unwrap();
            assert!(report.is_valid(), "{:?}", report.issues);
        }

        // The first entity in `list` whose `field` holds a `variant`, edited.
        let tamper = |bytes: &[u8], list: &str, field: &str, variant: &str| {
            let dump: Value = serde_json::from_slice(bytes).unwrap();
            let at = dump[list]
                .as_array()
                .unwrap()
                .iter()
                .position(|entity| entity[field].get(variant).is_some())
                .unwrap();
            let (list, field, variant) = (list.to_owned(), field.to_owned(), variant.to_owned());
            move |edit: &dyn Fn(&mut Value)| {
                let mut dump = dump.clone();
                edit(&mut dump[&list][at][&field][&variant]);
                serde_json::to_vec(&dump).unwrap()
            }
        };
        let curve = tamper(&spline, "edges", "curve", "NurbsCurve");
        let surface = tamper(&spline, "faces", "surface", "Nurbs");
        let circle = tamper(&plain, "edges", "curve", "Circle");
        let cylinder = tamper(&plain, "faces", "surface", "Cylinder");
        let plane = tamper(&plain, "faces", "surface", "Plane");
        let count = |value: &Value| value.as_array().unwrap().len();
        // One knot vector of `degree` over `points`, from 0 to 1.
        let clamped = |degree: usize, points: usize| {
            let mut knots = vec![0.0; degree + 1];
            knots.extend(vec![1.0; points]);
            json!(knots)
        };
        let reversed = |knots: &mut Value| {
            let mut values: Vec<f64> = serde_json::from_value(knots.clone()).unwrap();
            values.reverse();
            *knots = json!(values);
        };
        // A number the text cannot hold where `edit` wrote 0.123456789.
        let unbounded = |bytes: Vec<u8>| {
            String::from_utf8(bytes)
                .unwrap()
                .replacen("0.123456789", "1e400", 1)
                .into_bytes()
        };

        let malformed_curves = [
            curve(&|c| {
                let n = count(&c["control_points"]);
                c["degree"] = json!(n);
                c["knots"] = clamped(n, n);
            }),
            curve(&|c| c["degree"] = json!(count(&c["control_points"]) + 2)),
            curve(&|c| reversed(&mut c["knots"])),
            curve(&|c| c["weights"][1] = json!(0.0)),
            curve(&|c| c["weights"][1] = json!(-1.0)),
            curve(&|c| c["weights"][1] = json!(1e-310)),
            curve(&|c| c["weights"] = json!([1.0])),
            circle(&|c| c["radius"] = json!(0.0)),
            circle(&|c| c["radius"] = json!(-3.0)),
            circle(&|c| {
                for axis in ["normal", "u_axis", "v_axis"] {
                    c[axis] = json!([0.0, 0.0, 0.0]);
                }
            }),
        ];
        let malformed_surfaces = [
            surface(&|s| {
                let n = count(&s["control_points"]);
                s["degree_u"] = json!(n);
                s["knots_u"] = clamped(n, n);
            }),
            surface(&|s| s["degree_v"] = json!(count(&s["control_points"][0]) + 2)),
            surface(&|s| reversed(&mut s["knots_u"])),
            surface(&|s| reversed(&mut s["knots_v"])),
            surface(&|s| s["weights"][1][0] = json!(0.0)),
            surface(&|s| s["weights"][1][0] = json!(-1.0)),
            surface(&|s| s["weights"][1][0] = json!(1e-310)),
            surface(&|s| {
                s["control_points"][1].as_array_mut().unwrap().pop();
            }),
            surface(&|s| {
                s["weights"][0].as_array_mut().unwrap().pop();
            }),
            cylinder(&|c| c["radius"] = json!(0.0)),
            cylinder(&|c| c["radius"] = json!(-3.0)),
            cylinder(&|c| {
                for axis in ["axis", "x_axis", "y_axis"] {
                    c[axis] = json!([0.0, 0.0, 0.0]);
                }
            }),
            plane(&|p| p["normal"] = json!([0.0, 0.0, 0.0])),
        ];
        let unreadable = [
            unbounded(curve(&|c| c["knots"][4] = json!(0.123_456_789))),
            unbounded(curve(&|c| c["control_points"][1][0] = json!(0.123_456_789))),
            unbounded(surface(&|s| s["knots_v"][1] = json!(0.123_456_789))),
            unbounded(surface(&|s| {
                s["control_points"][1][0][2] = json!(0.123_456_789);
            })),
            curve(&|c| c["knots"][4] = Value::Null),
            surface(&|s| s["control_points"][1][0][2] = Value::Null),
        ];

        let refused = |bad: &[u8], why: &str| {
            let found = deserialize_solid(bad, &mut Topology::new());
            assert!(
                matches!(&found, Err(IoError::ParseError { reason }) if reason.contains(why)),
                "{why}: {found:?}"
            );
        };
        for bad in &malformed_curves {
            refused(bad, "malformed curve");
        }
        for bad in &malformed_surfaces {
            refused(bad, "malformed surface");
        }
        for bad in &unreadable {
            refused(bad, "arena deserialization failed");
        }
    }

    /// A circle or a surface of revolution whose frame has two axes alike,
    /// axes not at right angles, or a left-handed turn is not the shape its
    /// type names: a dump that holds one is refused. A knot span or weights
    /// whose arithmetic overflows are refused too. The dumps they were made
    /// from load.
    #[test]
    fn a_dump_with_a_skewed_frame_or_an_overflowing_spline_is_refused() {
        use brepkit_operations::primitives::{make_cone, make_sphere, make_torus};
        use serde_json::{Value, json};

        let mut topo = Topology::new();
        let solids = [
            make_cylinder(&mut topo, 3.0, 4.0).unwrap(),
            make_cone(&mut topo, 3.0, 1.0, 4.0).unwrap(),
            make_sphere(&mut topo, 2.5, 8).unwrap(),
            make_torus(&mut topo, 4.0, 1.0, 8).unwrap(),
        ];
        let spline = make_cylinder(&mut topo, 3.0, 4.0).unwrap();
        brepkit_operations::heal::convert_to_bspline(&mut topo, spline).unwrap();
        let dumps: Vec<Value> = solids
            .iter()
            .chain([&spline])
            .map(|&s| {
                let bytes = serialize_solid(&topo, s).unwrap();
                assert!(deserialize_solid(&bytes, &mut Topology::new()).is_ok());
                serde_json::from_slice(&bytes).unwrap()
            })
            .collect();
        // The dumps with `edit` made to the first entity in `list` whose
        // `field` holds a `variant`, if any does.
        let edited = |list: &str, field: &str, variant: &str, edit: &dyn Fn(&mut Value)| {
            dumps
                .iter()
                .filter_map(|dump| {
                    let at = dump[list]
                        .as_array()
                        .unwrap()
                        .iter()
                        .position(|entity| entity[field].get(variant).is_some())?;
                    let mut dump = dump.clone();
                    edit(&mut dump[list][at][field][variant]);
                    Some(serde_json::to_vec(&dump).unwrap())
                })
                .collect::<Vec<_>>()
        };
        let same = |a: &'static str, b: &'static str| move |g: &mut Value| g[b] = g[a].clone();
        let flipped = |a: &'static str| {
            move |g: &mut Value| {
                let v: Vec<f64> = serde_json::from_value(g[a].clone()).unwrap();
                g[a] = json!(v.iter().map(|c| -c).collect::<Vec<_>>());
            }
        };
        let shrunk = |key: &'static str| {
            move |g: &mut Value| {
                let v: Vec<f64> = serde_json::from_value(g[key].clone()).unwrap();
                g[key] = json!(v.iter().map(|k| k * 1e-310).collect::<Vec<_>>());
            }
        };
        let kinds = [
            edited("edges", "curve", "Circle", &same("u_axis", "v_axis")),
            edited("edges", "curve", "Circle", &flipped("normal")),
            edited("faces", "surface", "Cylinder", &same("x_axis", "y_axis")),
            edited("faces", "surface", "Cylinder", &flipped("axis")),
            edited("faces", "surface", "Cone", &same("x_axis", "y_axis")),
            edited("faces", "surface", "Cone", &flipped("axis")),
            edited("faces", "surface", "Sphere", &same("x_axis", "y_axis")),
            edited("faces", "surface", "Sphere", &flipped("z_axis")),
            edited("faces", "surface", "Torus", &same("x_axis", "y_axis")),
            edited("faces", "surface", "Torus", &flipped("z_axis")),
            edited("edges", "curve", "NurbsCurve", &shrunk("knots")),
            edited("faces", "surface", "Nurbs", &shrunk("knots_v")),
            edited("edges", "curve", "NurbsCurve", &|c| {
                c["weights"] = json!(vec![1e308; c["weights"].as_array().unwrap().len()]);
            }),
            edited("faces", "surface", "Nurbs", &|s| {
                let rows: Vec<Vec<f64>> = serde_json::from_value(s["weights"].clone()).unwrap();
                s["weights"] = json!(
                    rows.iter()
                        .map(|row| vec![1e308; row.len()])
                        .collect::<Vec<_>>()
                );
            }),
        ];
        assert!(kinds.iter().all(|dumps| !dumps.is_empty()));
        let malformed = kinds.concat();
        for bad in &malformed {
            let found = deserialize_solid(bad, &mut Topology::new());
            assert!(
                matches!(&found, Err(IoError::ParseError { reason }) if reason.contains("malformed")),
                "{found:?}"
            );
        }
    }

    /// A dump holds whatever data its bytes hold. A pcurve whose B-spline
    /// has more degrees than control points, or knots that fall, would read
    /// past its arrays when evaluated: such a dump is refused. The dump it
    /// was made from loads.
    #[test]
    fn a_dump_with_a_malformed_pcurve_is_refused() {
        use brepkit_math::curves2d::NurbsCurve2D;
        use brepkit_math::vec::Point2;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 3.0, 4.0).unwrap();
        let wall = solid_faces(&topo, solid)
            .unwrap()
            .into_iter()
            .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
            .unwrap();
        let rim = brepkit_topology::explorer::face_edges(&topo, wall).unwrap()[0];
        let line = NurbsCurve2D::from_line(Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)).unwrap();
        topo.pcurves_mut()
            .set(rim, wall, PCurve::new(Curve2D::Nurbs(line), 0.0, 1.0));
        let bytes = serialize_solid(&topo, solid).unwrap();
        assert!(deserialize_solid(&bytes, &mut Topology::new()).is_ok());

        let tamper = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut dump: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            let nurbs = dump["pcurves"][0]["curve"]
                .get_mut("Nurbs")
                .expect("the pcurve is a B-spline");
            edit(nurbs);
            serde_json::to_vec(&dump).unwrap()
        };
        for bad in [
            tamper(&|nurbs| {
                nurbs["degree"] = serde_json::json!(3);
                nurbs["knots"] = serde_json::json!([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
            }),
            tamper(&|nurbs| nurbs["knots"] = serde_json::json!([1.0, 1.0, 0.0, 0.0])),
            tamper(&|nurbs| nurbs["weights"] = serde_json::json!([1.0])),
        ] {
            let found = deserialize_solid(&bad, &mut Topology::new());
            assert!(
                matches!(&found, Err(IoError::ParseError { reason }) if reason.contains("malformed pcurve")),
                "{found:?}"
            );
        }
    }

    /// The constructors put an ellipse's larger semi-axis first, and code
    /// that reads its size from `semi_major` alone relies on that order. A
    /// dump holds whatever its bytes hold: an edge or a pcurve ellipse with
    /// its smaller semi-axis first is refused. With its axes in order, the
    /// dump loads.
    #[test]
    fn a_dump_with_an_ellipse_whose_smaller_axis_comes_first_is_refused() {
        use brepkit_math::curves2d::Ellipse2D;
        use brepkit_math::vec::Point2;
        use serde_json::Value;

        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 3.0, 4.0).unwrap();
        let wall = solid_faces(&topo, solid)
            .unwrap()
            .into_iter()
            .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
            .unwrap();
        let rim = brepkit_topology::explorer::face_edges(&topo, wall).unwrap()[0];
        let flat = Ellipse2D::new(Point2::new(0.5, 0.5), 0.4, 0.1, 0.0).unwrap();
        topo.pcurves_mut()
            .set(rim, wall, PCurve::new(Curve2D::Ellipse(flat), 0.0, 1.0));
        let ellipse = Ellipse3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            3.0,
            2.0,
        )
        .unwrap();
        topo.edge_mut(rim)
            .unwrap()
            .set_curve(EdgeCurve::Ellipse(ellipse));
        let bytes = serialize_solid(&topo, solid).unwrap();
        assert!(deserialize_solid(&bytes, &mut Topology::new()).is_ok());

        // The dump with the semi-axes of the ellipse that `pick` finds swapped.
        let swapped = |pick: &dyn Fn(&mut Value) -> &mut Value| {
            let mut dump: Value = serde_json::from_slice(&bytes).unwrap();
            let ellipse = pick(&mut dump);
            let (major, minor) = (ellipse["semi_major"].clone(), ellipse["semi_minor"].clone());
            ellipse["semi_major"] = minor;
            ellipse["semi_minor"] = major;
            serde_json::to_vec(&dump).unwrap()
        };
        let edge = swapped(&|dump| {
            dump["edges"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find_map(|e| e["curve"].get_mut("Ellipse"))
                .unwrap()
        });
        let pcurve = swapped(&|dump| dump["pcurves"][0]["curve"].get_mut("Ellipse").unwrap());
        for (bad, why) in [(edge, "malformed curve"), (pcurve, "malformed pcurve")] {
            let found = deserialize_solid(&bad, &mut Topology::new());
            assert!(
                matches!(&found, Err(IoError::ParseError { reason }) if reason.contains(why)),
                "{why}: {found:?}"
            );
        }
    }
}
