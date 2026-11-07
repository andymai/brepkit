//! A sweep along a chain of edges follows the chain, whichever way each edge
//! happens to be stored: a wire keeps the direction it runs an edge separately
//! from the edge, so a list of a wire's edges can hold them either way round.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::FRAC_PI_2;

    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
    use brepkit_operations::sweep::sweep_along_edges;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::face::{Face, FaceId, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// A 2 x 2 square in `x = 0`, centred on the axis, its normal along `x`.
    fn square(topo: &mut Topology) -> FaceId {
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        let vertices: Vec<_> = corners
            .iter()
            .map(|&(y, z)| topo.add_vertex(Vertex::new(Point3::new(0.0, y, z), 1e-7)))
            .collect();
        let edges: Vec<_> = (0..4)
            .map(|i| {
                let edge = Edge::new(vertices[i], vertices[(i + 1) % 4], EdgeCurve::Line);
                OrientedEdge::new(topo.add_edge(edge), true)
            })
            .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        topo.add_face(Face::new(
            wire,
            Vec::new(),
            FaceSurface::Plane {
                normal: Vec3::new(1.0, 0.0, 0.0),
                d: 0.0,
            },
        ))
    }

    /// Lines through `points`, each stored backward where `flips` says.
    fn lines(topo: &mut Topology, points: &[Point3], flips: &[bool]) -> Vec<EdgeId> {
        let vertices: Vec<_> = points
            .iter()
            .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
            .collect();
        flips
            .iter()
            .enumerate()
            .map(|(i, &flip)| {
                let (a, b) = if flip {
                    (vertices[i + 1], vertices[i])
                } else {
                    (vertices[i], vertices[i + 1])
                };
                topo.add_edge(Edge::new(a, b, EdgeCurve::Line))
            })
            .collect()
    }

    const FLIPS: [[bool; 2]; 4] = [[false, false], [true, false], [false, true], [true, true]];

    fn assert_holds(
        topo: &Topology,
        solid: brepkit_topology::solid::SolidId,
        volume: f64,
        what: &str,
    ) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        let exact = solid_volume(topo, solid, 0.01).unwrap();
        assert!(
            (exact - volume).abs() < 1e-3 * volume,
            "{what}: volume {exact}, expected {volume}"
        );
        let meshed = oriented_solid_volume(topo, solid, 0.05).unwrap();
        assert!(
            (meshed - volume).abs() < 2e-2 * volume,
            "{what}: meshed volume {meshed}, expected {volume}"
        );
    }

    /// Two collinear lines: the prism's section times the path's length.
    #[test]
    fn a_straight_path_sweeps_its_length_however_its_edges_are_stored() {
        let path = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(30.0, 0.0, 0.0),
        ];
        for flips in FLIPS {
            let mut topo = Topology::new();
            let face = square(&mut topo);
            let edges = lines(&mut topo, &path, &flips);
            let solid = sweep_along_edges(&mut topo, face, &edges).unwrap();
            assert_holds(&topo, solid, 120.0, &format!("{flips:?}"));
        }
    }

    /// A bent path reads the same in any storage as with every edge forward.
    #[test]
    fn a_bent_path_sweeps_the_same_however_its_edges_are_stored() {
        let path = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(20.0, 10.0, 0.0),
        ];
        let read = |flips: &[bool; 2]| {
            let mut topo = Topology::new();
            let face = square(&mut topo);
            let edges = lines(&mut topo, &path, flips);
            let solid = sweep_along_edges(&mut topo, face, &edges).unwrap();
            let report = validate_solid(&topo, solid).unwrap();
            assert!(report.is_valid(), "{flips:?}: {:?}", report.issues);
            solid_volume(&topo, solid, 0.01).unwrap()
        };
        let forward = read(&FLIPS[0]);
        for flips in &FLIPS[1..] {
            let volume = read(flips);
            assert!(
                (volume - forward).abs() < 1e-6 * forward,
                "{flips:?}: {volume} against {forward}"
            );
        }
    }

    /// A line into a quarter arc, the arc stored from its far end or its near
    /// one, turning the same way along the chain. A stored-backward arc samples
    /// its own span, then reads it reversed.
    #[test]
    fn an_arc_in_a_chain_sweeps_the_same_however_it_is_stored() {
        let read = |flip_arc: bool| {
            let mut topo = Topology::new();
            let face = square(&mut topo);
            let (a, b, c) = (
                topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7)),
                topo.add_vertex(Vertex::new(Point3::new(10.0, 0.0, 0.0), 1e-7)),
                topo.add_vertex(Vertex::new(Point3::new(15.0, 5.0, 0.0), 1e-7)),
            );
            let line = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
            // Counterclockwise about +z from b to c, about (10, 5).
            let center = Point3::new(10.0, 5.0, 0.0);
            let up = Vec3::new(0.0, 0.0, 1.0);
            let arc = if flip_arc {
                // The same span, clockwise about -z from c to b.
                let circle = Circle3D::new(center, -up, 5.0).unwrap();
                topo.add_edge(Edge::new(c, b, EdgeCurve::Circle(circle)))
            } else {
                let circle = Circle3D::new(center, up, 5.0).unwrap();
                topo.add_edge(Edge::new(b, c, EdgeCurve::Circle(circle)))
            };
            let solid = sweep_along_edges(&mut topo, face, &[line, arc]).unwrap();
            solid_volume(&topo, solid, 0.01).unwrap()
        };
        let (forward, flipped) = (read(false), read(true));
        // 10 straight, and a quarter turn of radius 5 by Pappus.
        let truth = 4.0 * (10.0 + 5.0 * FRAC_PI_2);
        assert!(
            (forward - truth).abs() < 2e-2 * truth,
            "{forward} against {truth}"
        );
        assert!(
            (flipped - forward).abs() < 1e-6 * forward,
            "{flipped} against {forward}"
        );
    }

    /// A gap in the chain is refused rather than crossed by a segment nobody
    /// asked for, and neither are edges out of order taken as given.
    #[test]
    fn a_broken_chain_is_refused() {
        let mut topo = Topology::new();
        let face = square(&mut topo);
        let v: Vec<_> = [0.0, 10.0, 12.0, 20.0]
            .iter()
            .map(|&x| topo.add_vertex(Vertex::new(Point3::new(x, 0.0, 0.0), 1e-7)))
            .collect();
        let first = topo.add_edge(Edge::new(v[0], v[1], EdgeCurve::Line));
        let second = topo.add_edge(Edge::new(v[2], v[3], EdgeCurve::Line));
        let error = sweep_along_edges(&mut topo, face, &[first, second]).unwrap_err();
        assert!(error.to_string().contains("does not meet"), "{error}");
    }
}
