//! Closed-form geometric properties for primitive solids.
//!
//! Each function returns a [`GProps`] with volume, center of mass, and inertia
//! tensor for a canonical primitive placement (origin-based, axis-aligned).

use std::f64::consts::PI;

use brepkit_math::vec::Point3;

use super::accumulator::GProps;

/// Properties of a rectangular box with dimensions `(dx, dy, dz)`, corner at origin.
///
/// Volume = `dx * dy * dz`, center at `(dx/2, dy/2, dz/2)`.
#[must_use]
pub fn box_props(dx: f64, dy: f64, dz: f64) -> GProps {
    let v = dx * dy * dz;
    let center = Point3::new(dx / 2.0, dy / 2.0, dz / 2.0);
    // Inertia at CoM for uniform density=1 box:
    // Ixx = V/12 * (dy^2 + dz^2), etc.
    let ixx = v / 12.0 * (dy * dy + dz * dz);
    let iyy = v / 12.0 * (dx * dx + dz * dz);
    let izz = v / 12.0 * (dx * dx + dy * dy);
    GProps {
        mass: v,
        center,
        inertia: [ixx, iyy, izz, 0.0, 0.0, 0.0],
    }
}

/// Properties of a sphere with given `radius`, centered at origin.
///
/// Volume = `4/3 * pi * r^3`.
#[must_use]
pub fn sphere_props(radius: f64) -> GProps {
    let v = 4.0 / 3.0 * PI * radius.powi(3);
    let center = Point3::new(0.0, 0.0, 0.0);
    // I = 2/5 * m * r^2 (all axes, by symmetry)
    let i = 2.0 / 5.0 * v * radius * radius;
    GProps {
        mass: v,
        center,
        inertia: [i, i, i, 0.0, 0.0, 0.0],
    }
}

/// Properties of a cylinder with given `radius` and `height`, base at z=0.
///
/// Volume = `pi * r^2 * h`, center at `(0, 0, h/2)`.
#[must_use]
pub fn cylinder_props(radius: f64, height: f64) -> GProps {
    let v = PI * radius * radius * height;
    let center = Point3::new(0.0, 0.0, height / 2.0);
    // Ixx = Iyy = V/12 * (3*r^2 + h^2)
    let ixx = v / 12.0 * (3.0 * radius * radius + height * height);
    let iyy = ixx;
    // Izz = V/2 * r^2
    let izz = v / 2.0 * radius * radius;
    GProps {
        mass: v,
        center,
        inertia: [ixx, iyy, izz, 0.0, 0.0, 0.0],
    }
}

/// Properties of a cone frustum with `r_bottom`, `r_top`, `height`, base at z=0.
///
/// When `r_top = 0` this is a full cone.
/// Volume = `pi * h / 3 * (rb^2 + rb*rt + rt^2)`.
#[must_use]
#[allow(clippy::similar_names)]
pub fn cone_props(r_bottom: f64, r_top: f64, height: f64) -> GProps {
    let rb2 = r_bottom * r_bottom;
    let rt2 = r_top * r_top;
    let rbrt = r_bottom * r_top;
    let r_sum2 = rb2 + rbrt + rt2;

    let v = PI * height / 3.0 * r_sum2;
    // CoM for frustum:
    // z_com = h * (rb^2 + 2*rb*rt + 3*rt^2) / (4*(rb^2 + rb*rt + rt^2))
    let z_com = height * (rb2 + 2.0 * rbrt + 3.0 * rt2) / (4.0 * r_sum2);
    let center = Point3::new(0.0, 0.0, z_com);

    // Izz for frustum: 3V/10 * (rb^5 - rt^5) / (rb^3 - rt^3) when rb != rt
    // For full cone (rt=0): Izz = 3/10 * V * rb^2
    // General formula: Izz = 3*V/10 * (rb^4 + rb^3*rt + rb^2*rt^2 + rb*rt^3 + rt^4) / (rb^2 + rb*rt + rt^2)
    let r_sum4 = rb2 * rb2 + rb2 * rbrt + rb2 * rt2 + rbrt * rt2 + rt2 * rt2;
    let izz = 3.0 * v / 10.0 * r_sum4 / r_sum2;

    // Ixx = Iyy for frustum (about CoM):
    // Ixx_origin = 3V/80 * (4*(rb^4+rb^3*rt+rb^2*rt^2+rb*rt^3+rt^4)/(rb^2+rb*rt+rt^2) + h^2*(rb^2+3*rb*rt+6*rt^2)/(rb^2+rb*rt+rt^2))
    // Then shift to CoM via parallel axis
    let ixx_about_base = 3.0 * v / 20.0 * r_sum4 / r_sum2
        + v * height * height * (rb2 + 3.0 * rbrt + 6.0 * rt2) / (10.0 * r_sum2);
    // Shift from base-axis to CoM: Ixx_com = Ixx_base - V * z_com^2
    let ixx = ixx_about_base - v * z_com * z_com;
    let iyy = ixx;

    GProps {
        mass: v,
        center,
        inertia: [ixx, iyy, izz, 0.0, 0.0, 0.0],
    }
}

/// Properties of a torus with `major_r` (center to tube center) and `minor_r`
/// (tube radius), centered at origin in the XY plane.
///
/// Volume = `2 * pi^2 * R * r^2`.
#[must_use]
pub fn torus_props(major_r: f64, minor_r: f64) -> GProps {
    let v = 2.0 * PI * PI * major_r * minor_r * minor_r;
    let center = Point3::new(0.0, 0.0, 0.0);
    // Izz = V * (R^2 + 3/4 * r^2)
    let izz = v * (major_r * major_r + 0.75 * minor_r * minor_r);
    // Ixx = Iyy = V * (R^2/2 + 5/8 * r^2)
    let ixx = v * (major_r * major_r / 2.0 + 5.0 / 8.0 * minor_r * minor_r);
    let iyy = ixx;
    GProps {
        mass: v,
        center,
        inertia: [ixx, iyy, izz, 0.0, 0.0, 0.0],
    }
}

/// Surface area of a box with dimensions `(dx, dy, dz)`.
#[must_use]
pub fn box_area(dx: f64, dy: f64, dz: f64) -> f64 {
    2.0 * (dx * dy + dy * dz + dx * dz)
}

/// Surface area of a sphere with given `radius`.
#[must_use]
pub fn sphere_area(radius: f64) -> f64 {
    4.0 * PI * radius * radius
}

/// Surface area of a cylinder (including caps) with given `radius` and `height`.
#[must_use]
pub fn cylinder_area(radius: f64, height: f64) -> f64 {
    2.0 * PI * radius * height + 2.0 * PI * radius * radius
}

/// Surface area of a torus with `major_r` and `minor_r`.
#[must_use]
pub fn torus_area(major_r: f64, minor_r: f64) -> f64 {
    4.0 * PI * PI * major_r * minor_r
}
