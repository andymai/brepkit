//! Edge-to-edge distance computation.

use brepkit_math::vec::Point3;

/// Minimum distance between two line segments [p0, p1] and [q0, q1].
///
/// Returns (distance, closest\_on\_seg1, closest\_on\_seg2).
#[allow(clippy::similar_names)]
pub fn segment_segment_distance(
    p0: Point3,
    p1: Point3,
    q0: Point3,
    q1: Point3,
) -> (f64, Point3, Point3) {
    let d1 = p1 - p0;
    let d2 = q1 - q0;
    let r = p0 - q0;

    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);

    if a < 1e-30 && e < 1e-30 {
        return ((p0 - q0).length(), p0, q0);
    }

    let (s, t) = if a < 1e-30 {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = d1.dot(r);
        if e < 1e-30 {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = d1.dot(d2);
            let denom = a.mul_add(e, -(b * b));
            let s_unclamp = if denom.abs() > 1e-30 {
                (b.mul_add(f, -(c * e)) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let t_val = (b.mul_add(s_unclamp, f) / e).clamp(0.0, 1.0);
            // Recompute s from clamped t
            let s_val = if a.abs() > 1e-30 {
                (b.mul_add(t_val, -c) / a).clamp(0.0, 1.0)
            } else {
                s_unclamp
            };
            (s_val, t_val)
        }
    };

    let cp = Point3::new(
        d1.x().mul_add(s, p0.x()),
        d1.y().mul_add(s, p0.y()),
        d1.z().mul_add(s, p0.z()),
    );
    let cq = Point3::new(
        d2.x().mul_add(t, q0.x()),
        d2.y().mul_add(t, q0.y()),
        d2.z().mul_add(t, q0.z()),
    );

    ((cp - cq).length(), cp, cq)
}
