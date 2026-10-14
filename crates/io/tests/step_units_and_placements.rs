//! A STEP file's length unit and assembly placements decide where its
//! shapes are. The reader returns them in millimetres, each placed where its
//! assembly puts it, once for every time the assembly uses it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "support/step_text.rs"]
mod step_text;

use std::f64::consts::PI;

use brepkit_io::step::reader::read_step;
use brepkit_math::vec::Point3;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;
use step_text::{StepText, Unit};

const R: f64 = 10.0;
const H: f64 = 30.0;

/// A seamed cylinder of radius `R` and height `H` up its local +Z, in the
/// file's unit: the solid, and its vertices at `(R, 0, 0)` and `(R, 0, H)`.
fn cylinder(s: &mut StepText) -> u64 {
    let a = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let surface = s.add(format!("CYLINDRICAL_SURFACE('',#{a},{R:?})"));
    let (v0, v1) = (s.vertex([R, 0.0, 0.0]), s.vertex([R, 0.0, H]));
    let c0 = s.circle([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], R);
    let c1 = s.circle([0.0, 0.0, H], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], R);
    let (e0, e1) = (s.edge(v0, v0, c0), s.edge(v1, v1, c1));
    let up = s.line([R, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let seam = s.edge(v0, v1, up);
    let floor_loop = s.edge_loop(&[(e0, false)]);
    let floor_plane = s.plane([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let floor = s.face(&[floor_loop], floor_plane, false);
    let wall_loop = s.edge_loop(&[(e0, true), (seam, true), (e1, false), (seam, false)]);
    let wall = s.face(&[wall_loop], surface, true);
    let roof_loop = s.edge_loop(&[(e1, true)]);
    let roof_plane = s.plane([0.0, 0.0, H], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let roof = s.face(&[roof_loop], roof_plane, true);
    s.solid(&[floor, wall, roof])
}

/// A frame: origin, z axis, x axis.
type Frame = ([f64; 3], [f64; 3], [f64; 3]);

fn unit(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `p` in `frame`'s coordinates, in the frame's parent's.
fn place((o, z, x): Frame, p: [f64; 3]) -> [f64; 3] {
    let (z, x) = (unit(z), unit(x));
    let y = cross(z, x);
    [0, 1, 2].map(|i| o[i] + x[i] * p[0] + y[i] * p[1] + z[i] * p[2])
}

/// A product with a shape representation `rep`: its product definition.
fn product(s: &mut StepText, name: &str, rep: u64) -> u64 {
    let app = s.add("APPLICATION_CONTEXT('automotive design')");
    let pc = s.add(format!("PRODUCT_CONTEXT('',#{app},'mechanical')"));
    let pdc = s.add(format!(
        "PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')"
    ));
    let p = s.add(format!("PRODUCT('{name}','{name}','',(#{pc}))"));
    let pdf = s.add(format!("PRODUCT_DEFINITION_FORMATION('','',#{p})"));
    let pd = s.add(format!("PRODUCT_DEFINITION('design','',#{pdf},#{pdc})"));
    let pds = s.add(format!("PRODUCT_DEFINITION_SHAPE('','',#{pd})"));
    s.add(format!("SHAPE_DEFINITION_REPRESENTATION(#{pds},#{rep})"));
    pd
}

/// Use the component (`child_pd`, `child_rep`) in the assembly
/// (`parent_pd`, `parent_rep`), its `child_origin` placed on `at`.
/// `reversed` writes the relationship parent first.
#[allow(clippy::too_many_arguments)]
fn use_component(
    s: &mut StepText,
    (parent_pd, parent_rep): (u64, u64),
    (child_pd, child_rep, child_origin): (u64, u64, u64),
    at: u64,
    reversed: bool,
) {
    let nauo = s.add(format!(
        "NEXT_ASSEMBLY_USAGE_OCCURRENCE('#1','use','',#{parent_pd},#{child_pd},$)"
    ));
    let pds = s.add(format!("PRODUCT_DEFINITION_SHAPE('','',#{nauo})"));
    let (reps, items) = if reversed {
        ((parent_rep, child_rep), (at, child_origin))
    } else {
        ((child_rep, parent_rep), (child_origin, at))
    };
    let idt = s.add(format!(
        "ITEM_DEFINED_TRANSFORMATION('','',#{},#{})",
        items.0, items.1
    ));
    let rel = s.add(format!(
        "( REPRESENTATION_RELATIONSHIP('','',#{},#{}) \
         REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{idt}) \
         SHAPE_REPRESENTATION_RELATIONSHIP() )",
        reps.0, reps.1
    ));
    s.add(format!(
        "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{rel},#{pds})"
    ));
}

/// Every vertex position of `solid`.
fn vertices(topo: &Topology, solid: SolidId) -> Vec<Point3> {
    let mut out: Vec<Point3> = Vec::new();
    for f in solid_faces(topo, solid).unwrap() {
        let face = topo.face(f).unwrap();
        for w in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(w).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                for v in [e.start(), e.end()] {
                    let p = topo.vertex(v).unwrap().point();
                    if !out.iter().any(|q| (*q - p).length() < 1e-9) {
                        out.push(p);
                    }
                }
            }
        }
    }
    out
}

/// `solid` is the cylinder scaled by `scale` and placed by `frames`, the
/// innermost first.
fn assert_cylinder(topo: &Topology, solid: SolidId, scale: f64, frames: &[Frame], label: &str) {
    let expected: Vec<[f64; 3]> = [[R, 0.0, 0.0], [R, 0.0, H]]
        .into_iter()
        .map(|p| {
            let p = p.map(|c| c * scale);
            frames.iter().fold(p, |p, &f| place(f, p))
        })
        .collect();
    let got = vertices(topo, solid);
    assert_eq!(got.len(), 2, "{label}: two vertices");
    for e in expected {
        let e = Point3::new(e[0], e[1], e[2]);
        assert!(
            got.iter()
                .any(|g| (*g - e).length() < 1e-9 * (1.0 + scale * H)),
            "{label}: no vertex at {e:?}, got {got:?}"
        );
    }
    let truth = PI * R * R * H * scale.powi(3);
    let volume = solid_volume(topo, solid, 0.001 * scale).unwrap();
    assert!(
        (volume - truth).abs() < 1e-6 * truth,
        "{label}: volume {volume}, truth {truth}"
    );
    assert!(
        is_watertight(&tessellate_solid(topo, solid, 0.01 * scale).unwrap()),
        "{label}: open mesh"
    );
}

fn read(step: &str) -> (Topology, Vec<SolidId>) {
    let mut topo = Topology::new();
    let solids = read_step(step, &mut topo).unwrap();
    (topo, solids)
}

#[test]
fn geometry_comes_out_in_millimetres() {
    for (unit, scale) in [
        (Unit::Millimetre, 1.0),
        (Unit::Inch, 25.4),
        (Unit::Metre, 1000.0),
    ] {
        let mut s = StepText::with_unit(unit);
        let solid = cylinder(&mut s);
        s.representation("ADVANCED_BREP_SHAPE_REPRESENTATION", &[solid]);
        let (topo, solids) = read(&s.finish());
        assert_eq!(solids.len(), 1);
        assert_cylinder(&topo, solids[0], scale, &[], &format!("x{scale}"));
    }
}

/// A one-cylinder part, and an assembly using it once at each of `frames`.
fn assembly_step(frames: &[Frame], reversed: bool) -> String {
    let mut s = StepText::new();
    let solid = cylinder(&mut s);
    let part_origin = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let part_rep = s.representation("ADVANCED_BREP_SHAPE_REPRESENTATION", &[part_origin, solid]);
    let part_pd = product(&mut s, "part", part_rep);
    let asm_origin = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let ats: Vec<u64> = frames
        .iter()
        .map(|&(o, z, x)| s.placement(o, z, x))
        .collect();
    let mut items = vec![asm_origin];
    items.extend(&ats);
    let asm_rep = s.representation("SHAPE_REPRESENTATION", &items);
    let asm_pd = product(&mut s, "assembly", asm_rep);
    for at in ats {
        use_component(
            &mut s,
            (asm_pd, asm_rep),
            (part_pd, part_rep, part_origin),
            at,
            reversed,
        );
    }
    s.finish()
}

const TILTED: Frame = ([40.0, -25.0, 10.0], [1.0, 1.0, 1.0], [1.0, -1.0, 0.0]);
const TURNED: Frame = ([60.0, 0.0, 10.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);

#[test]
fn an_assembly_places_its_part() {
    for reversed in [false, true] {
        let (topo, solids) = read(&assembly_step(&[TILTED], reversed));
        assert_eq!(solids.len(), 1);
        assert_cylinder(
            &topo,
            solids[0],
            1.0,
            &[TILTED],
            &format!("reversed {reversed}"),
        );
    }
}

#[test]
fn a_part_used_twice_comes_out_twice() {
    let (topo, solids) = read(&assembly_step(&[TILTED, TURNED], false));
    assert_eq!(solids.len(), 2);
    let mut matched = [false, false];
    for &solid in &solids {
        let low = vertices(&topo, solid);
        for (k, frame) in [TILTED, TURNED].into_iter().enumerate() {
            let e = place(frame, [R, 0.0, 0.0]);
            if low
                .iter()
                .any(|g| (*g - Point3::new(e[0], e[1], e[2])).length() < 1e-9)
            {
                assert_cylinder(&topo, solid, 1.0, &[frame], &format!("use {k}"));
                matched[k] = true;
            }
        }
    }
    assert_eq!(matched, [true, true], "one solid at each use");
}

#[test]
fn placements_compose_through_a_subassembly() {
    let inner: Frame = ([0.0, 50.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
    let outer: Frame = ([100.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]);
    let mut s = StepText::new();
    let solid = cylinder(&mut s);
    let part_origin = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let part_rep = s.representation("ADVANCED_BREP_SHAPE_REPRESENTATION", &[part_origin, solid]);
    let part_pd = product(&mut s, "part", part_rep);
    let sub_origin = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let sub_at = s.placement(inner.0, inner.1, inner.2);
    let sub_rep = s.representation("SHAPE_REPRESENTATION", &[sub_origin, sub_at]);
    let sub_pd = product(&mut s, "sub", sub_rep);
    let top_origin = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let top_at = s.placement(outer.0, outer.1, outer.2);
    let top_rep = s.representation("SHAPE_REPRESENTATION", &[top_origin, top_at]);
    let top_pd = product(&mut s, "top", top_rep);
    use_component(
        &mut s,
        (sub_pd, sub_rep),
        (part_pd, part_rep, part_origin),
        sub_at,
        false,
    );
    use_component(
        &mut s,
        (top_pd, top_rep),
        (sub_pd, sub_rep, sub_origin),
        top_at,
        false,
    );
    let (topo, solids) = read(&s.finish());
    assert_eq!(solids.len(), 1);
    assert_cylinder(&topo, solids[0], 1.0, &[inner, outer], "nested");
}

#[test]
fn a_mapped_item_places_its_representation() {
    let mut s = StepText::new();
    let solid = cylinder(&mut s);
    let part_origin = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let part_rep = s.representation("ADVANCED_BREP_SHAPE_REPRESENTATION", &[part_origin, solid]);
    let map = s.add(format!("REPRESENTATION_MAP(#{part_origin},#{part_rep})"));
    let target = s.placement(TILTED.0, TILTED.1, TILTED.2);
    let item = s.add(format!("MAPPED_ITEM('',#{map},#{target})"));
    s.representation("SHAPE_REPRESENTATION", &[item]);
    let (topo, solids) = read(&s.finish());
    assert_eq!(solids.len(), 1);
    assert_cylinder(&topo, solids[0], 1.0, &[TILTED], "mapped item");
}

#[test]
fn a_millimetre_part_in_an_inch_assembly() {
    let mut s = StepText::with_unit(Unit::Inch);
    // The part's own context, in millimetres.
    let mm = s.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )");
    let part_context = s.add(format!(
        "( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#{mm},#2,#3)) \
         REPRESENTATION_CONTEXT('','3D') )"
    ));
    let solid = cylinder(&mut s);
    let part_origin = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let part_rep = s.add(format!(
        "ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{part_origin},#{solid}),#{part_context})"
    ));
    let part_pd = product(&mut s, "part", part_rep);
    let asm_origin = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let at = s.placement([2.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let asm_rep = s.representation("SHAPE_REPRESENTATION", &[asm_origin, at]);
    let asm_pd = product(&mut s, "assembly", asm_rep);
    use_component(
        &mut s,
        (asm_pd, asm_rep),
        (part_pd, part_rep, part_origin),
        at,
        false,
    );
    let (topo, solids) = read(&s.finish());
    assert_eq!(solids.len(), 1);
    let shifted: Frame = ([50.8, 0.0, 25.4], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    assert_cylinder(&topo, solids[0], 1.0, &[shifted], "mixed units");
}

#[test]
fn an_assembly_that_doubles_at_every_level_is_refused() {
    // Each level uses the one below twice: 2^25 placements of one cylinder.
    let mut s = StepText::new();
    let solid = cylinder(&mut s);
    let origin = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let mut child_rep = s.representation("ADVANCED_BREP_SHAPE_REPRESENTATION", &[origin, solid]);
    let mut child = (product(&mut s, "part", child_rep), child_rep, origin);
    for level in 0..25 {
        let origin = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        let left = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        let right = s.placement([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        let rep = s.representation("SHAPE_REPRESENTATION", &[origin, left, right]);
        let pd = product(&mut s, &format!("level {level}"), rep);
        for at in [left, right] {
            use_component(&mut s, (pd, rep), child, at, false);
        }
        child_rep = rep;
        child = (pd, child_rep, origin);
    }
    let mut topo = Topology::new();
    let started = std::time::Instant::now();
    let refused = read_step(&s.finish(), &mut topo);
    assert!(refused.is_err(), "a placement bomb is refused");
    assert!(
        started.elapsed().as_secs() < 5,
        "refused without listing the placements"
    );
}

#[test]
fn an_assembly_wider_than_the_allowance_is_refused() {
    // 400 uses of a part in a subassembly used 300 times: 120,000 copies.
    let mut s = StepText::new();
    let solid = cylinder(&mut s);
    let part_origin = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let part_rep = s.representation("ADVANCED_BREP_SHAPE_REPRESENTATION", &[part_origin, solid]);
    let mut child = (product(&mut s, "part", part_rep), part_rep, part_origin);
    for (level, uses) in [400, 300].into_iter().enumerate() {
        let origin = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        let ats: Vec<u64> = (0..uses)
            .map(|k| s.placement([f64::from(k), 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]))
            .collect();
        let mut items = vec![origin];
        items.extend(&ats);
        let rep = s.representation("SHAPE_REPRESENTATION", &items);
        let pd = product(&mut s, &format!("level {level}"), rep);
        for at in ats {
            use_component(&mut s, (pd, rep), child, at, false);
        }
        child = (pd, rep, origin);
    }
    let mut topo = Topology::new();
    assert!(
        read_step(&s.finish(), &mut topo).is_err(),
        "too many copies are refused"
    );
}
