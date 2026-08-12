//! #1538 coplanar-interface fuse family probe: fuse a plate carrying a
//! through-hole onto a block below it, sharing the interface plane. The
//! plate's interface face has a hole; the block's top face is full. The
//! correct fuse is a solid with a blind pocket whose floor is part of the
//! block's top plane.
//!
//! Modes: `circle` (round hole), `rect` (rectangular hole), `pocket`
//! (blind pocket in the plate reaching exactly the interface — the
//! circle-insert configuration), `circle4`/`pocket4` (same, but the
//! cylinder tool's wall is pre-split into four quarter faces with
//! quarter-arc rims, the way the layout tool's extruded circle profiles
//! arrive), `pocket4r` (pocket4 with the tool profile wound CW — the
//! extrude winding-normalization repro), `roundpocket4` (rounded plate +
//! centered quartered cylinder). SYNTHETIC inputs only: both operands are
//! validation-clean, which splits "the interface fuse is broken" from
//! "the captured operands carry winding taint".
//!
//! `cargo run --release --example interface_fuse_probe -p brepkit-operations -- <mode>`
#![allow(
    clippy::print_stdout,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic
)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn report(topo: &Topology, sid: SolidId, label: &str) {
    let faces = brepkit_topology::explorer::solid_faces(topo, sid).unwrap();
    let mut mix: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    let mut uses: std::collections::HashMap<brepkit_topology::edge::EdgeId, usize> =
        std::collections::HashMap::new();
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        *mix.entry(face.surface().type_tag()).or_default() += 1;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    let free = uses.values().filter(|&&c| c == 1).count();
    let over = uses.values().filter(|&&c| c > 2).count();
    if std::env::var("BK_WINDING").is_ok() {
        // Per shared edge: the directed uses from each face. Same direction
        // twice = the winding defect the census cannot see.
        let mut dir_uses: std::collections::HashMap<
            brepkit_topology::edge::EdgeId,
            Vec<(brepkit_topology::face::FaceId, bool, bool)>,
        > = std::collections::HashMap::new();
        for &fid in &faces {
            let face = topo.face(fid).unwrap();
            let wires: Vec<_> = std::iter::once((face.outer_wire(), true))
                .chain(face.inner_wires().iter().map(|&w| (w, false)))
                .collect();
            for (wid, is_outer) in wires {
                for oe in topo.wire(wid).unwrap().edges() {
                    dir_uses.entry(oe.edge()).or_default().push((
                        fid,
                        oe.is_forward() != face.is_reversed(),
                        is_outer,
                    ));
                }
            }
        }
        for (eid, us) in &dir_uses {
            let closed = topo.edge(*eid).is_ok_and(|e| e.start() == e.end());
            if us.len() == 2
                && (us[0].1 == us[1].1
                    || (closed && std::env::var("BK_WINDING").is_ok_and(|v| v == "2")))
            {
                let e = topo.edge(*eid).unwrap();
                let a = topo.vertex(e.start()).unwrap().point();
                let b = topo.vertex(e.end()).unwrap().point();
                if let brepkit_topology::edge::EdgeCurve::Circle(c) = e.curve() {
                    println!(
                        "    axis=({:.2},{:.2},{:.2})",
                        c.normal().x(),
                        c.normal().y(),
                        c.normal().z()
                    );
                }
                println!(
                    "  SAMEDIR {eid:?} ({:.1},{:.1},{:.1})->({:.1},{:.1},{:.1}) uses={us:?}",
                    a.x(),
                    a.y(),
                    a.z(),
                    b.x(),
                    b.y(),
                    b.z()
                );
                for (fid, _, _) in us {
                    let f = topo.face(*fid).unwrap();
                    println!(
                        "    {fid:?} {} reversed={} inner_wires={}",
                        f.surface().type_tag(),
                        f.is_reversed(),
                        f.inner_wires().len()
                    );
                }
            }
        }
    }
    let vol = brepkit_operations::measure::oriented_solid_volume(topo, sid, 0.05).unwrap();
    let report = brepkit_operations::validate::validate_solid(topo, sid).unwrap();
    println!(
        "{label}: F={} mix={mix:?} free={free} over={over} vol={vol:.3} valid={} {:?}",
        faces.len(),
        report.is_valid(),
        report.issues
    );
}

fn rounded_rect_plate(topo: &mut Topology, w: f64, r: f64, z0: f64, h: f64) -> SolidId {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};
    let hw = w / 2.0;
    let c = hw - r;
    let t = 1e-7;
    let z = Vec3::new(0.0, 0.0, 1.0);
    let v = |topo: &mut Topology, x: f64, y: f64| {
        topo.add_vertex(Vertex::new(Point3::new(x, y, z0), t))
    };
    let v0 = v(topo, hw, -c);
    let v1 = v(topo, hw, c);
    let v2 = v(topo, c, hw);
    let v3 = v(topo, -c, hw);
    let v4 = v(topo, -hw, c);
    let v5 = v(topo, -hw, -c);
    let v6 = v(topo, -c, -hw);
    let v7 = v(topo, c, -hw);
    let arc = |topo: &mut Topology, a, b, cx: f64, cy: f64| {
        let circle = Circle3D::new(Point3::new(cx, cy, z0), z, r).unwrap();
        topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle)))
    };
    let edges = [
        topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line)),
        arc(topo, v1, v2, c, c),
        topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line)),
        arc(topo, v3, v4, -c, c),
        topo.add_edge(Edge::new(v4, v5, EdgeCurve::Line)),
        arc(topo, v5, v6, -c, -c),
        topo.add_edge(Edge::new(v6, v7, EdgeCurve::Line)),
        arc(topo, v7, v0, c, -c),
    ];
    let wire = Wire::new(
        edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let fid = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z0,
        },
    ));
    brepkit_operations::extrude::extrude(topo, fid, Vec3::new(0.0, 0.0, 1.0), h).unwrap()
}

fn quartered_cylinder(
    topo: &mut Topology,
    cx: f64,
    cy: f64,
    r: f64,
    z0: f64,
    h: f64,
    cw_profile: bool,
) -> SolidId {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};
    let z = Vec3::new(0.0, 0.0, 1.0);
    let t = 1e-7;
    let v = |topo: &mut Topology, x: f64, y: f64| {
        topo.add_vertex(Vertex::new(Point3::new(x, y, z0), t))
    };
    let v0 = v(topo, cx + r, cy);
    let v1 = v(topo, cx, cy + r);
    let v2 = v(topo, cx - r, cy);
    let v3 = v(topo, cx, cy - r);
    let arc = |topo: &mut Topology, a, b| {
        let circle = Circle3D::new(Point3::new(cx, cy, z0), z, r).unwrap();
        topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle)))
    };
    let edges = [
        arc(topo, v0, v1),
        arc(topo, v1, v2),
        arc(topo, v2, v3),
        arc(topo, v3, v0),
    ];
    // The CW variant traverses the same CCW-stored arcs reversed, in reverse
    // order: v0 -> v3 -> v2 -> v1 -> v0.
    let oes: Vec<OrientedEdge> = if cw_profile {
        edges
            .iter()
            .rev()
            .map(|&e| OrientedEdge::new(e, false))
            .collect()
    } else {
        edges.iter().map(|&e| OrientedEdge::new(e, true)).collect()
    };
    let wire = Wire::new(oes, true).unwrap();
    let wid = topo.add_wire(wire);
    let fid = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z0,
        },
    ));
    brepkit_operations::extrude::extrude(topo, fid, Vec3::new(0.0, 0.0, 1.0), h).unwrap()
}

fn dump_faces(topo: &Topology, sid: SolidId) {
    for fid in brepkit_topology::explorer::solid_faces(topo, sid).unwrap() {
        let face = topo.face(fid).unwrap();
        println!(
            "FACE {fid:?} {} reversed={}",
            face.surface().type_tag(),
            face.is_reversed()
        );
        for (wi, wid) in std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .enumerate()
        {
            println!("  w{wi}");
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = topo.vertex(e.start()).unwrap().point();
                let b = topo.vertex(e.end()).unwrap().point();
                let ax = if let brepkit_topology::edge::EdgeCurve::Circle(c) = e.curve() {
                    format!(" axis_z={:.0}", c.normal().z())
                } else {
                    String::new()
                };
                println!(
                    "    {:?} fwd={} ({:.1},{:.1},{:.1})->({:.1},{:.1},{:.1}){ax}",
                    oe.edge(),
                    oe.is_forward(),
                    a.x(),
                    a.y(),
                    a.z(),
                    b.x(),
                    b.y(),
                    b.z()
                );
            }
        }
    }
}

fn main() {
    env_logger::init();
    let mode = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "circle".to_string());
    let mut topo = Topology::new();

    // Plate: 80 x 80 x 5 spanning z 5..10.
    let plate = brepkit_operations::primitives::make_box(&mut topo, 80.0, 80.0, 5.0).unwrap();
    brepkit_operations::transform::transform_solid(
        &mut topo,
        plate,
        &Mat4::translation(0.0, 0.0, 5.0),
    )
    .unwrap();

    let plate = match mode.as_str() {
        "circle" => {
            // Through-hole: cylinder r=10 through the plate.
            let hole = brepkit_operations::primitives::make_cylinder(&mut topo, 10.0, 7.0).unwrap();
            brepkit_operations::transform::transform_solid(
                &mut topo,
                hole,
                &Mat4::translation(40.0, 40.0, 4.0),
            )
            .unwrap();
            boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap()
        }
        "rect" => {
            let hole =
                brepkit_operations::primitives::make_box(&mut topo, 20.0, 20.0, 7.0).unwrap();
            brepkit_operations::transform::transform_solid(
                &mut topo,
                hole,
                &Mat4::translation(30.0, 30.0, 4.0),
            )
            .unwrap();
            boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap()
        }
        "roundpocket" => {
            // Rounded plate + interior box cutter flush at the bottom plane:
            // the deep-cutout z=0 configuration with arc corners.
            let rplate = rounded_rect_plate(&mut topo, 80.0, 3.75, 5.0, 5.0);
            let hole =
                brepkit_operations::primitives::make_box(&mut topo, 20.0, 20.0, 6.0).unwrap();
            brepkit_operations::transform::transform_solid(
                &mut topo,
                hole,
                &Mat4::translation(-38.0, -38.0, 5.0),
            )
            .unwrap();
            boolean::boolean(&mut topo, BooleanOp::Cut, rplate, hole).unwrap()
        }
        "rectpocket" => {
            // All-line analog of `pocket`: the box cutter's bottom cap is
            // COINCIDENT with the plate's bottom plane (the deep-cutout z=0
            // configuration).
            let hole =
                brepkit_operations::primitives::make_box(&mut topo, 20.0, 20.0, 6.0).unwrap();
            brepkit_operations::transform::transform_solid(
                &mut topo,
                hole,
                &Mat4::translation(30.0, 30.0, 5.0),
            )
            .unwrap();
            boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap()
        }
        "circle4" => {
            // Through-hole cut by a quarter-split cylinder wall.
            let hole = quartered_cylinder(&mut topo, 40.0, 40.0, 10.0, 4.0, 7.0, false);
            boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap()
        }
        "roundpocket4" => {
            // Rounded plate + quarter-split cylinder through-cut at the
            // center, bottom cap coincident with the plate bottom: both
            // bounding planes of the captured circle-insert floor cut are
            // rounded rects.
            let rplate = rounded_rect_plate(&mut topo, 80.0, 3.75, 5.0, 5.0);
            let hole = quartered_cylinder(&mut topo, 0.0, 0.0, 10.0, 5.0, 6.0, false);
            boolean::boolean(&mut topo, BooleanOp::Cut, rplate, hole).unwrap()
        }
        "pocket4" | "pocket4r" => {
            // Quarter-split cylinder whose bottom cap is COINCIDENT with the
            // plate's bottom plane and whose top clears the plate top: the
            // captured circle-insert floor through-cut. `pocket4r` winds the
            // tool profile CW, probing seed-direction dependence of the
            // pure-arc rim loops.
            let cw = mode == "pocket4r";
            let hole = quartered_cylinder(&mut topo, 40.0, 40.0, 10.0, 5.0, 6.0, cw);
            if std::env::var("BK_DUMP_TOOL").is_ok() {
                dump_faces(&topo, hole);
            }
            report(&topo, hole, "tool");
            boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap()
        }
        "pocket" => {
            // Blind pocket from the plate top reaching EXACTLY the interface
            // plane z=5 (the circle-insert configuration: cutDepth == floor).
            let hole = brepkit_operations::primitives::make_cylinder(&mut topo, 10.0, 6.0).unwrap();
            brepkit_operations::transform::transform_solid(
                &mut topo,
                hole,
                &Mat4::translation(40.0, 40.0, 5.0),
            )
            .unwrap();
            boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap()
        }
        other => panic!("unknown mode {other}"),
    };
    if std::env::var("BK_DUMP_CYL").is_ok() {
        for fid in brepkit_topology::explorer::solid_faces(&topo, plate).unwrap() {
            let face = topo.face(fid).unwrap();
            let dump_all = std::env::var("BK_DUMP_CYL").is_ok_and(|v| v == "all");
            if face.surface().type_tag() != "cylinder" && !dump_all {
                continue;
            }
            println!(
                "FACE {fid:?} {} reversed={}",
                face.surface().type_tag(),
                face.is_reversed()
            );
            for (wi, wid) in std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .enumerate()
            {
                println!("  w{wi}");
                for oe in topo.wire(wid).unwrap().edges() {
                    let e = topo.edge(oe.edge()).unwrap();
                    let a = topo.vertex(e.start()).unwrap().point();
                    let b = topo.vertex(e.end()).unwrap().point();
                    let ax = if let brepkit_topology::edge::EdgeCurve::Circle(c) = e.curve() {
                        format!(" axis_z={:.0}", c.normal().z())
                    } else {
                        String::new()
                    };
                    println!(
                        "    {:?} fwd={} ({:.1},{:.1},{:.1})->({:.1},{:.1},{:.1}){ax}",
                        oe.edge(),
                        oe.is_forward(),
                        a.x(),
                        a.y(),
                        a.z(),
                        b.x(),
                        b.y(),
                        b.z()
                    );
                }
            }
        }
    }
    report(&topo, plate, "plate with hole");

    // Block below: 80 x 80 x 5 spanning z 0..5, sharing the z=5 plane.
    let block = brepkit_operations::primitives::make_box(&mut topo, 80.0, 80.0, 5.0).unwrap();
    report(&topo, block, "block");

    let before = boolean::mesh_fallback_count();
    let ops_result = boolean::boolean(&mut topo, BooleanOp::Fuse, plate, block).unwrap();
    report(&topo, ops_result, "ops fuse");
    println!("fallbacks: {}", boolean::mesh_fallback_count() - before);

    // GFA deep-copies operands into its own store, so the same solids can be
    // replayed raw in the same topology.
    let raw =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, plate, block)
            .unwrap();
    report(&topo, raw, "raw GFA fuse");
}
