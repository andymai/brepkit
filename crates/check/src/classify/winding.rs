//! Generalized winding number classifier.
//!
//! Computes the winding number of a point with respect to a closed surface
//! by summing the signed solid angles of triangulated faces. More robust
//! than ray casting for imperfect geometry (small gaps, T-junctions).

use std::f64::consts::PI;

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::CheckError;

/// Compute the generalized winding number of a point relative to a solid.
///
/// Returns a value close to 1.0 for inside points, 0.0 for outside.
///
/// The algorithm triangulates each face of the solid's outer shell from
/// its wire polygon, then sums the signed solid angle subtended by each
/// triangle at the query point. The total is divided by 4pi to yield the
/// winding number.
///
/// # Errors
///
/// Returns an error if the solid or its faces contain invalid topology
/// references.
pub fn winding_number(topo: &Topology, solid: SolidId, point: Point3) -> Result<f64, CheckError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let faces = shell.faces().to_vec();

    let mut total = 0.0;
    for fid in &faces {
        total += face_winding_contribution(topo, *fid, point)?;
    }

    Ok(total / (4.0 * PI))
}

/// Compute the winding contribution of a single face.
///
/// Triangulates the face via fan triangulation from its wire polygon and
/// sums solid angles. The face orientation (`is_reversed`) determines the
/// sign of the contribution.
fn face_winding_contribution(
    topo: &Topology,
    face_id: FaceId,
    point: Point3,
) -> Result<f64, CheckError> {
    let reversed = topo.face(face_id)?.is_reversed();
    let polygon = crate::util::face_polygon(topo, face_id)?;

    if polygon.len() < 3 {
        return Ok(0.0);
    }

    let mut contribution = 0.0;
    for i in 1..polygon.len() - 1 {
        let omega = solid_angle(point, polygon[0], polygon[i], polygon[i + 1]);
        contribution += if reversed { -omega } else { omega };
    }

    Ok(contribution)
}

/// Compute the signed solid angle subtended by triangle (a, b, c) at point p.
///
/// Uses the Van Oosterom & Strackee (1983) formula:
///
/// `tan(omega/2) = det(a', b', c') / (|a'||b'||c'| + (a'.b')|c'| + (a'.c')|b'| + (b'.c')|a'|)`
///
/// where `a' = a - p`, etc. Returns the solid angle in steradians.
fn solid_angle(p: Point3, a: Point3, b: Point3, c: Point3) -> f64 {
    let pa = a - p;
    let pb = b - p;
    let pc = c - p;

    let la = pa.length();
    let lb = pb.length();
    let lc = pc.length();

    // Point coincides with a triangle vertex — degenerate.
    if la < 1e-15 || lb < 1e-15 || lc < 1e-15 {
        return 0.0;
    }

    // Numerator: scalar triple product det(pa, pb, pc) = pa . (pb x pc)
    let num = pa.x() * (pb.y() * pc.z() - pb.z() * pc.y())
        + pa.y() * (pb.z() * pc.x() - pb.x() * pc.z())
        + pa.z() * (pb.x() * pc.y() - pb.y() * pc.x());

    // Denominator
    let den = la
        .mul_add(lb * lc, pa.dot(pb) * lc)
        .mul_add(1.0, pa.dot(pc) * lb + pb.dot(pc) * la);

    2.0 * num.atan2(den)
}
