//! Rounds of the long top edges of a sheared prism: its plan is the
//! parallelogram `(0, 0)`, `(2, 0)`, `(3 + s, d)`, `(1 + s, d)`, `d` wide,
//! extruded `H` up. The short sides lean, so each round runs on to them as
//! to tilted caps, and its contact on the top runs `(1 + s) r` along the
//! edge from where the edge ends. A shear along the edges moves each
//! section of the round along them and keeps its shape, so the volume a
//! round takes off is that of the unsheared prism (`s = -1`): its section
//! times the edge's length.
//!
//! - One round of `r`: the section is `(1 - π/4) r²`.
//! - Both rounds of `r` past half the width: the second rests on its wall
//!   and pivots on the first's contact on the top, `r` in from the first
//!   edge. Its ball's centre stands `d - r` in from its wall and `k` below
//!   the top, with `(2r - d)² + k² = r²`, and it takes off the corner of the
//!   `(d - r)` by `k` rectangle outside its arc.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_blend::fillet_builder::FilletBuilder;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::make_planar_face_from_wire;
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::explorer::solid_edges;
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    const D: f64 = 1.0;
    const H: f64 = 3.0;
    const LENGTH: f64 = 2.0;

    fn poses() -> Vec<(&'static str, Mat4)> {
        vec![
            ("upright", Mat4::translation(0.0, 0.0, 0.0)),
            (
                "tilted",
                Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    /// The prism sheared by `s`, and its top edges at `y = 0` and `y = D`.
    fn prism(topo: &mut Topology, s: f64) -> (SolidId, [EdgeId; 2]) {
        let corners = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(LENGTH, 0.0, 0.0),
            Point3::new(LENGTH + 1.0 + s, D, 0.0),
            Point3::new(1.0 + s, D, 0.0),
        ];
        let vertices = corners.map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
        let edges: Vec<OrientedEdge> = (0..4)
            .map(|i| {
                let edge = Edge::new(vertices[i], vertices[(i + 1) % 4], EdgeCurve::Line);
                OrientedEdge::new(topo.add_edge(edge), true)
            })
            .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let face = make_planar_face_from_wire(topo, wire).unwrap();
        let solid = extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), H).unwrap();
        let top = |y: f64| {
            solid_edges(topo, solid)
                .unwrap()
                .into_iter()
                .find(|&e| {
                    let edge = topo.edge(e).unwrap();
                    [edge.start(), edge.end()].iter().all(|&v| {
                        let p = topo.vertex(v).unwrap().point();
                        (p.y() - y).abs() < 1e-9 && (p.z() - H).abs() < 1e-9
                    })
                })
                .unwrap()
        };
        (solid, [top(0.0), top(D)])
    }

    /// The section the pivoting second round takes off.
    fn pivot_section(r: f64) -> f64 {
        let k = (r * r - (2.0 * r - D).powi(2)).sqrt();
        let below =
            |u: f64| 0.5 * u * (r * r - u * u).max(0.0).sqrt() + 0.5 * r * r * (u / r).asin();
        (D - r) * k - (below(r) - below(2.0 * r - D))
    }

    fn check(s: f64, r: f64, both: bool, expected: f64) {
        for (name, pose) in poses() {
            let what = format!("s = {s}, r = {r}, both = {both}, {name}");
            let mut topo = Topology::new();
            let (solid, [a, b]) = prism(&mut topo, s);
            transform_solid(&mut topo, solid, &pose).unwrap();
            let before = solid_volume(&topo, solid, 1e-3).unwrap();
            let mut builder = FilletBuilder::new(&mut topo, solid);
            if both {
                builder.add_edges(&[a, b], r);
            } else {
                builder.add_edges(&[a], r);
            }
            let built = builder.build().unwrap();
            assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
            assert!(
                validate_solid(&topo, built.solid).unwrap().is_valid(),
                "{what}: not valid"
            );
            let removed = before - solid_volume(&topo, built.solid, 1e-3).unwrap();
            assert!(
                (removed - expected).abs() < 1e-5 * expected,
                "{what}: took off {removed}, expected {expected}"
            );
            let mesh = tessellate_solid(&topo, built.solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{what}: open mesh");
        }
    }

    /// The contact on the top runs on past the whole length of the edge, or
    /// exactly that far, before it meets the leaning side.
    #[test]
    fn a_round_whose_contact_runs_on_past_its_edge_s_length_takes_off_its_strip() {
        for (s, r) in [(-1.0, 0.4), (1.5, 0.4), (4.0, 0.4), (5.0, 0.4)] {
            check(s, r, false, LENGTH * (1.0 - PI / 4.0) * r * r);
        }
    }

    /// Both rounds past half the width, the second pivoting on the first's
    /// contact, on prisms sheared so far that their edges share no span.
    #[test]
    fn crossing_rounds_of_a_sheared_prism_take_off_what_the_unsheared_one_loses() {
        let r = 0.8;
        let expected = LENGTH * ((1.0 - PI / 4.0) * r * r + pivot_section(r));
        for s in [-1.0, 0.5, 1.5, 2.0] {
            check(s, r, true, expected);
        }
    }
}
