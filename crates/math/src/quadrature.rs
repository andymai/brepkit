// Surface integration helpers are pub(crate) but not yet called internally.
#![allow(dead_code)]
//! Gauss-Legendre quadrature for numerical integration.
//!
//! Provides pre-computed quadrature points and weights for orders 1–20,
//! plus tensor-product surface integration functions for area and volume
//! computation via the divergence theorem.
//!
//! Instead of tessellating a face and summing triangle contributions,
//! we integrate the surface directly in parametric space using Gauss
//! quadrature.

use crate::vec::{Point3, Vec3};

/// Maximum supported Gauss-Legendre order.
pub const MAX_ORDER: usize = 20;

/// Gauss-Legendre quadrature point (position on [-1,1] and weight).
#[derive(Debug, Clone, Copy)]
pub struct GaussPoint {
    /// Position on [-1, 1].
    pub x: f64,
    /// Quadrature weight.
    pub w: f64,
}

/// Return Gauss-Legendre quadrature points and weights for the given order.
///
/// Order `n` exactly integrates polynomials of degree `2n - 1`.
/// Supported range: 1..=20. Orders outside this range are clamped.
///
/// Points are on the interval [-1, 1]. To integrate over [a, b], apply
/// the affine transform `t = (b-a)/2 * x + (a+b)/2` and scale weights
/// by `(b-a)/2`.
#[must_use]
pub fn gauss_legendre_points(order: usize) -> &'static [GaussPoint] {
    let n = order.clamp(1, MAX_ORDER);
    GAUSS_TABLES[n - 1]
}

/// Integrate a scalar function over a parametric surface patch using
/// tensor-product Gauss-Legendre quadrature.
///
/// The evaluator `f(u, v)` must return `(point, unnormalized_normal)` where
/// the normal is `∂S/∂u × ∂S/∂v` (its magnitude is the Jacobian `dA`).
///
/// Returns the surface area of the patch `[u0,u1] × [v0,v1]`.
pub(crate) fn gauss_surface_area(
    f: &impl Fn(f64, f64) -> (Point3, Vec3),
    u_range: (f64, f64),
    v_range: (f64, f64),
    nu: usize,
    nv: usize,
) -> f64 {
    let pts_u = gauss_legendre_points(nu);
    let pts_v = gauss_legendre_points(nv);

    let u_half = (u_range.1 - u_range.0) * 0.5;
    let u_mid = (u_range.1 + u_range.0) * 0.5;
    let v_half = (v_range.1 - v_range.0) * 0.5;
    let v_mid = (v_range.1 + v_range.0) * 0.5;

    let mut area = 0.0;
    for gu in pts_u {
        let u = u_half.mul_add(gu.x, u_mid);
        for gv in pts_v {
            let v = v_half.mul_add(gv.x, v_mid);
            let (_pt, normal) = f(u, v);
            // ||normal|| = ||∂S/∂u × ∂S/∂v|| is the surface element dA
            area += normal.length() * gu.w * gv.w;
        }
    }
    area * u_half * v_half
}

/// Compute the volume contribution of a surface patch via the divergence
/// theorem: `V = (1/3) ∫∫ P · N dA`.
///
/// The evaluator `f(u, v)` must return `(point, unnormalized_normal)`.
/// The returned value is the *signed* volume contribution (positive when
/// the normal points outward from the enclosed volume).
pub(crate) fn gauss_surface_volume(
    f: &impl Fn(f64, f64) -> (Point3, Vec3),
    u_range: (f64, f64),
    v_range: (f64, f64),
    nu: usize,
    nv: usize,
) -> f64 {
    let pts_u = gauss_legendre_points(nu);
    let pts_v = gauss_legendre_points(nv);

    let u_half = (u_range.1 - u_range.0) * 0.5;
    let u_mid = (u_range.1 + u_range.0) * 0.5;
    let v_half = (v_range.1 - v_range.0) * 0.5;
    let v_mid = (v_range.1 + v_range.0) * 0.5;

    let mut vol = 0.0;
    for gu in pts_u {
        let u = u_half.mul_add(gu.x, u_mid);
        for gv in pts_v {
            let v = v_half.mul_add(gv.x, v_mid);
            let (pt, normal) = f(u, v);
            // P · N where N is unnormalized (includes Jacobian)
            let p_dot_n = pt.x() * normal.x() + pt.y() * normal.y() + pt.z() * normal.z();
            vol += p_dot_n * gu.w * gv.w;
        }
    }
    vol * u_half * v_half / 3.0
}

/// Integrate surface area over multiple (u,v) sub-patches (knot spans).
///
/// Each pair of consecutive u-knots and v-knots defines a sub-patch.
/// This is critical for NURBS where each knot span is a polynomial piece:
/// integrating per-span avoids the accuracy loss of integrating across
/// knot boundaries where derivatives are discontinuous.
pub(crate) fn gauss_surface_area_spans(
    f: &impl Fn(f64, f64) -> (Point3, Vec3),
    u_knots: &[f64],
    v_knots: &[f64],
    nu: usize,
    nv: usize,
) -> f64 {
    let mut total = 0.0;
    for u_win in u_knots.windows(2) {
        let u_range = (u_win[0], u_win[1]);
        if (u_range.1 - u_range.0).abs() < 1e-15 {
            continue;
        }
        for v_win in v_knots.windows(2) {
            let v_range = (v_win[0], v_win[1]);
            if (v_range.1 - v_range.0).abs() < 1e-15 {
                continue;
            }
            total += gauss_surface_area(f, u_range, v_range, nu, nv);
        }
    }
    total
}

/// Integrate volume contribution over multiple (u,v) sub-patches.
///
/// See [`gauss_surface_area_spans`] for the per-knot-span rationale.
pub(crate) fn gauss_surface_volume_spans(
    f: &impl Fn(f64, f64) -> (Point3, Vec3),
    u_knots: &[f64],
    v_knots: &[f64],
    nu: usize,
    nv: usize,
) -> f64 {
    let mut total = 0.0;
    for u_win in u_knots.windows(2) {
        let u_range = (u_win[0], u_win[1]);
        if (u_range.1 - u_range.0).abs() < 1e-15 {
            continue;
        }
        for v_win in v_knots.windows(2) {
            let v_range = (v_win[0], v_win[1]);
            if (v_range.1 - v_range.0).abs() < 1e-15 {
                continue;
            }
            total += gauss_surface_volume(f, u_range, v_range, nu, nv);
        }
    }
    total
}

// ── Pre-computed Gauss-Legendre tables ────────────────────────────────
//
// Generated from the standard three-term recurrence for Legendre
// polynomials. These are the well-known tabulated values used in
// numerical analysis (Abramowitz & Stegun Table 25.4).

/// Lookup table indexed by `order - 1`. Each entry is a slice of
/// `GaussPoint` for that order.
static GAUSS_TABLES: [&[GaussPoint]; MAX_ORDER] = [
    // Order 1
    &[GaussPoint { x: 0.0, w: 2.0 }],
    // Order 2
    &[
        GaussPoint {
            x: -0.577_350_269_189_625_7,
            w: 1.0,
        },
        GaussPoint {
            x: 0.577_350_269_189_625_7,
            w: 1.0,
        },
    ],
    // Order 3
    &[
        GaussPoint {
            x: -0.774_596_669_241_483_4,
            w: 0.555_555_555_555_555_6,
        },
        GaussPoint {
            x: 0.0,
            w: 0.888_888_888_888_889,
        },
        GaussPoint {
            x: 0.774_596_669_241_483_4,
            w: 0.555_555_555_555_555_6,
        },
    ],
    // Order 4
    &[
        GaussPoint {
            x: -0.861_136_311_594_052_6,
            w: 0.347_854_845_137_453_9,
        },
        GaussPoint {
            x: -0.339_981_043_584_856_3,
            w: 0.652_145_154_862_546_1,
        },
        GaussPoint {
            x: 0.339_981_043_584_856_3,
            w: 0.652_145_154_862_546_1,
        },
        GaussPoint {
            x: 0.861_136_311_594_052_6,
            w: 0.347_854_845_137_453_9,
        },
    ],
    // Order 5
    &[
        GaussPoint {
            x: -0.906_179_845_938_664,
            w: 0.236_926_885_056_189_1,
        },
        GaussPoint {
            x: -0.538_469_310_105_683,
            w: 0.478_628_670_499_366_5,
        },
        GaussPoint {
            x: 0.0,
            w: 0.568_888_888_888_889,
        },
        GaussPoint {
            x: 0.538_469_310_105_683,
            w: 0.478_628_670_499_366_5,
        },
        GaussPoint {
            x: 0.906_179_845_938_664,
            w: 0.236_926_885_056_189_1,
        },
    ],
    // Order 6
    &[
        GaussPoint {
            x: -0.932_469_514_203_152,
            w: 0.171_324_492_379_170_3,
        },
        GaussPoint {
            x: -0.661_209_386_466_265,
            w: 0.360_761_573_048_139,
        },
        GaussPoint {
            x: -0.238_619_186_083_197,
            w: 0.467_913_934_572_691,
        },
        GaussPoint {
            x: 0.238_619_186_083_197,
            w: 0.467_913_934_572_691,
        },
        GaussPoint {
            x: 0.661_209_386_466_265,
            w: 0.360_761_573_048_139,
        },
        GaussPoint {
            x: 0.932_469_514_203_152,
            w: 0.171_324_492_379_170_3,
        },
    ],
    // Order 7
    &[
        GaussPoint {
            x: -0.949_107_912_342_759,
            w: 0.129_484_966_168_870,
        },
        GaussPoint {
            x: -0.741_531_185_599_394,
            w: 0.279_705_391_489_277,
        },
        GaussPoint {
            x: -0.405_845_151_377_397,
            w: 0.381_830_050_505_119,
        },
        GaussPoint {
            x: 0.0,
            w: 0.417_959_183_673_469,
        },
        GaussPoint {
            x: 0.405_845_151_377_397,
            w: 0.381_830_050_505_119,
        },
        GaussPoint {
            x: 0.741_531_185_599_394,
            w: 0.279_705_391_489_277,
        },
        GaussPoint {
            x: 0.949_107_912_342_759,
            w: 0.129_484_966_168_870,
        },
    ],
    // Order 8
    &[
        GaussPoint {
            x: -0.960_289_856_497_536,
            w: 0.101_228_536_290_376,
        },
        GaussPoint {
            x: -0.796_666_477_413_627,
            w: 0.222_381_034_453_374,
        },
        GaussPoint {
            x: -0.525_532_409_916_329,
            w: 0.313_706_645_877_887,
        },
        GaussPoint {
            x: -0.183_434_642_495_650,
            w: 0.362_683_783_378_362,
        },
        GaussPoint {
            x: 0.183_434_642_495_650,
            w: 0.362_683_783_378_362,
        },
        GaussPoint {
            x: 0.525_532_409_916_329,
            w: 0.313_706_645_877_887,
        },
        GaussPoint {
            x: 0.796_666_477_413_627,
            w: 0.222_381_034_453_374,
        },
        GaussPoint {
            x: 0.960_289_856_497_536,
            w: 0.101_228_536_290_376,
        },
    ],
    // Order 9
    &[
        GaussPoint {
            x: -0.968_160_239_507_626,
            w: 0.081_274_388_361_574,
        },
        GaussPoint {
            x: -0.836_031_107_326_636,
            w: 0.180_648_160_694_857,
        },
        GaussPoint {
            x: -0.613_371_432_700_590,
            w: 0.260_610_696_402_935,
        },
        GaussPoint {
            x: -0.324_253_423_403_809,
            w: 0.312_347_077_040_003,
        },
        GaussPoint {
            x: 0.0,
            w: 0.330_239_355_001_260,
        },
        GaussPoint {
            x: 0.324_253_423_403_809,
            w: 0.312_347_077_040_003,
        },
        GaussPoint {
            x: 0.613_371_432_700_590,
            w: 0.260_610_696_402_935,
        },
        GaussPoint {
            x: 0.836_031_107_326_636,
            w: 0.180_648_160_694_857,
        },
        GaussPoint {
            x: 0.968_160_239_507_626,
            w: 0.081_274_388_361_574,
        },
    ],
    // Order 10
    &[
        GaussPoint {
            x: -0.973_906_528_517_172,
            w: 0.066_671_344_308_688,
        },
        GaussPoint {
            x: -0.865_063_366_688_985,
            w: 0.149_451_349_150_581,
        },
        GaussPoint {
            x: -0.679_409_568_299_024,
            w: 0.219_086_362_515_982,
        },
        GaussPoint {
            x: -0.433_395_394_129_247,
            w: 0.269_266_719_309_996,
        },
        GaussPoint {
            x: -0.148_874_338_981_631,
            w: 0.295_524_224_714_753,
        },
        GaussPoint {
            x: 0.148_874_338_981_631,
            w: 0.295_524_224_714_753,
        },
        GaussPoint {
            x: 0.433_395_394_129_247,
            w: 0.269_266_719_309_996,
        },
        GaussPoint {
            x: 0.679_409_568_299_024,
            w: 0.219_086_362_515_982,
        },
        GaussPoint {
            x: 0.865_063_366_688_985,
            w: 0.149_451_349_150_581,
        },
        GaussPoint {
            x: 0.973_906_528_517_172,
            w: 0.066_671_344_308_688,
        },
    ],
    // Order 11
    &[
        GaussPoint {
            x: -0.978_228_658_146_057,
            w: 0.055_668_567_116_174,
        },
        GaussPoint {
            x: -0.887_062_599_768_095,
            w: 0.125_580_369_464_905,
        },
        GaussPoint {
            x: -0.730_152_005_574_049,
            w: 0.186_290_210_927_734,
        },
        GaussPoint {
            x: -0.519_096_129_206_812,
            w: 0.233_193_764_591_990,
        },
        GaussPoint {
            x: -0.269_543_155_952_345,
            w: 0.262_804_544_510_247,
        },
        GaussPoint {
            x: 0.0,
            w: 0.272_925_086_777_901,
        },
        GaussPoint {
            x: 0.269_543_155_952_345,
            w: 0.262_804_544_510_247,
        },
        GaussPoint {
            x: 0.519_096_129_206_812,
            w: 0.233_193_764_591_990,
        },
        GaussPoint {
            x: 0.730_152_005_574_049,
            w: 0.186_290_210_927_734,
        },
        GaussPoint {
            x: 0.887_062_599_768_095,
            w: 0.125_580_369_464_905,
        },
        GaussPoint {
            x: 0.978_228_658_146_057,
            w: 0.055_668_567_116_174,
        },
    ],
    // Order 12
    &[
        GaussPoint {
            x: -0.981_560_634_246_719,
            w: 0.047_175_336_386_512,
        },
        GaussPoint {
            x: -0.904_117_256_370_475,
            w: 0.106_939_325_995_318,
        },
        GaussPoint {
            x: -0.769_902_674_194_305,
            w: 0.160_078_328_543_346,
        },
        GaussPoint {
            x: -0.587_317_954_286_617,
            w: 0.203_167_426_723_066,
        },
        GaussPoint {
            x: -0.367_831_498_998_180,
            w: 0.233_492_536_538_355,
        },
        GaussPoint {
            x: -0.125_233_408_511_469,
            w: 0.249_147_045_813_403,
        },
        GaussPoint {
            x: 0.125_233_408_511_469,
            w: 0.249_147_045_813_403,
        },
        GaussPoint {
            x: 0.367_831_498_998_180,
            w: 0.233_492_536_538_355,
        },
        GaussPoint {
            x: 0.587_317_954_286_617,
            w: 0.203_167_426_723_066,
        },
        GaussPoint {
            x: 0.769_902_674_194_305,
            w: 0.160_078_328_543_346,
        },
        GaussPoint {
            x: 0.904_117_256_370_475,
            w: 0.106_939_325_995_318,
        },
        GaussPoint {
            x: 0.981_560_634_246_719,
            w: 0.047_175_336_386_512,
        },
    ],
    // Order 13
    &[
        GaussPoint {
            x: -0.984_183_054_718_588,
            w: 0.040_484_004_765_316,
        },
        GaussPoint {
            x: -0.917_598_399_222_978,
            w: 0.092_121_499_837_728,
        },
        GaussPoint {
            x: -0.801_578_090_733_310,
            w: 0.138_873_510_219_787,
        },
        GaussPoint {
            x: -0.642_349_339_440_340,
            w: 0.178_145_980_761_946,
        },
        GaussPoint {
            x: -0.448_492_751_036_447,
            w: 0.207_816_047_536_889,
        },
        GaussPoint {
            x: -0.230_458_315_955_135,
            w: 0.226_283_180_262_897,
        },
        GaussPoint {
            x: 0.0,
            w: 0.232_551_553_230_874,
        },
        GaussPoint {
            x: 0.230_458_315_955_135,
            w: 0.226_283_180_262_897,
        },
        GaussPoint {
            x: 0.448_492_751_036_447,
            w: 0.207_816_047_536_889,
        },
        GaussPoint {
            x: 0.642_349_339_440_340,
            w: 0.178_145_980_761_946,
        },
        GaussPoint {
            x: 0.801_578_090_733_310,
            w: 0.138_873_510_219_787,
        },
        GaussPoint {
            x: 0.917_598_399_222_978,
            w: 0.092_121_499_837_728,
        },
        GaussPoint {
            x: 0.984_183_054_718_588,
            w: 0.040_484_004_765_316,
        },
    ],
    // Order 14
    &[
        GaussPoint {
            x: -0.986_283_808_696_812,
            w: 0.035_119_460_331_752,
        },
        GaussPoint {
            x: -0.928_434_883_663_574,
            w: 0.080_158_087_159_760,
        },
        GaussPoint {
            x: -0.827_201_315_069_765,
            w: 0.121_518_570_687_903,
        },
        GaussPoint {
            x: -0.687_292_904_811_685,
            w: 0.157_203_167_158_194,
        },
        GaussPoint {
            x: -0.515_248_636_358_154,
            w: 0.185_538_397_477_938,
        },
        GaussPoint {
            x: -0.319_112_368_927_890,
            w: 0.205_198_463_721_296,
        },
        GaussPoint {
            x: -0.108_054_948_707_344,
            w: 0.215_263_853_463_158,
        },
        GaussPoint {
            x: 0.108_054_948_707_344,
            w: 0.215_263_853_463_158,
        },
        GaussPoint {
            x: 0.319_112_368_927_890,
            w: 0.205_198_463_721_296,
        },
        GaussPoint {
            x: 0.515_248_636_358_154,
            w: 0.185_538_397_477_938,
        },
        GaussPoint {
            x: 0.687_292_904_811_685,
            w: 0.157_203_167_158_194,
        },
        GaussPoint {
            x: 0.827_201_315_069_765,
            w: 0.121_518_570_687_903,
        },
        GaussPoint {
            x: 0.928_434_883_663_574,
            w: 0.080_158_087_159_760,
        },
        GaussPoint {
            x: 0.986_283_808_696_812,
            w: 0.035_119_460_331_752,
        },
    ],
    // Order 15
    &[
        GaussPoint {
            x: -0.987_992_518_020_485,
            w: 0.030_753_241_996_117,
        },
        GaussPoint {
            x: -0.937_273_392_400_706,
            w: 0.070_366_047_488_108,
        },
        GaussPoint {
            x: -0.848_206_583_410_427,
            w: 0.107_159_220_467_172,
        },
        GaussPoint {
            x: -0.724_417_731_360_170,
            w: 0.139_570_677_926_154,
        },
        GaussPoint {
            x: -0.570_972_172_608_539,
            w: 0.166_269_205_816_994,
        },
        GaussPoint {
            x: -0.394_151_347_077_563,
            w: 0.186_161_000_015_562,
        },
        GaussPoint {
            x: -0.201_194_093_997_435,
            w: 0.198_431_485_327_111,
        },
        GaussPoint {
            x: 0.0,
            w: 0.202_578_241_925_561,
        },
        GaussPoint {
            x: 0.201_194_093_997_435,
            w: 0.198_431_485_327_111,
        },
        GaussPoint {
            x: 0.394_151_347_077_563,
            w: 0.186_161_000_015_562,
        },
        GaussPoint {
            x: 0.570_972_172_608_539,
            w: 0.166_269_205_816_994,
        },
        GaussPoint {
            x: 0.724_417_731_360_170,
            w: 0.139_570_677_926_154,
        },
        GaussPoint {
            x: 0.848_206_583_410_427,
            w: 0.107_159_220_467_172,
        },
        GaussPoint {
            x: 0.937_273_392_400_706,
            w: 0.070_366_047_488_108,
        },
        GaussPoint {
            x: 0.987_992_518_020_485,
            w: 0.030_753_241_996_117,
        },
    ],
    // Order 16
    &[
        GaussPoint {
            x: -0.989_400_934_991_650,
            w: 0.027_152_459_411_754,
        },
        GaussPoint {
            x: -0.944_575_023_073_233,
            w: 0.062_253_523_938_648,
        },
        GaussPoint {
            x: -0.865_631_202_387_832,
            w: 0.095_158_511_682_493,
        },
        GaussPoint {
            x: -0.755_404_408_355_003,
            w: 0.124_628_971_255_534,
        },
        GaussPoint {
            x: -0.617_876_244_402_644,
            w: 0.149_595_988_816_577,
        },
        GaussPoint {
            x: -0.458_016_777_657_227,
            w: 0.169_156_519_395_003,
        },
        GaussPoint {
            x: -0.281_603_550_779_259,
            w: 0.182_603_415_044_924,
        },
        GaussPoint {
            x: -0.095_012_509_837_637,
            w: 0.189_450_610_455_069,
        },
        GaussPoint {
            x: 0.095_012_509_837_637,
            w: 0.189_450_610_455_069,
        },
        GaussPoint {
            x: 0.281_603_550_779_259,
            w: 0.182_603_415_044_924,
        },
        GaussPoint {
            x: 0.458_016_777_657_227,
            w: 0.169_156_519_395_003,
        },
        GaussPoint {
            x: 0.617_876_244_402_644,
            w: 0.149_595_988_816_577,
        },
        GaussPoint {
            x: 0.755_404_408_355_003,
            w: 0.124_628_971_255_534,
        },
        GaussPoint {
            x: 0.865_631_202_387_832,
            w: 0.095_158_511_682_493,
        },
        GaussPoint {
            x: 0.944_575_023_073_233,
            w: 0.062_253_523_938_648,
        },
        GaussPoint {
            x: 0.989_400_934_991_650,
            w: 0.027_152_459_411_754,
        },
    ],
    // Order 17
    &[
        GaussPoint {
            x: -0.990_575_475_314_417,
            w: 0.024_148_302_868_548,
        },
        GaussPoint {
            x: -0.950_675_521_768_768,
            w: 0.055_459_529_373_987,
        },
        GaussPoint {
            x: -0.880_239_153_726_986,
            w: 0.085_036_148_317_179,
        },
        GaussPoint {
            x: -0.781_514_003_896_801,
            w: 0.111_883_847_193_404,
        },
        GaussPoint {
            x: -0.657_671_159_216_691,
            w: 0.135_136_368_468_525,
        },
        GaussPoint {
            x: -0.512_690_537_086_477,
            w: 0.154_045_761_076_810,
        },
        GaussPoint {
            x: -0.351_231_763_453_876,
            w: 0.168_004_102_156_450,
        },
        GaussPoint {
            x: -0.178_484_181_495_848,
            w: 0.176_562_705_366_993,
        },
        GaussPoint {
            x: 0.0,
            w: 0.179_446_470_356_207,
        },
        GaussPoint {
            x: 0.178_484_181_495_848,
            w: 0.176_562_705_366_993,
        },
        GaussPoint {
            x: 0.351_231_763_453_876,
            w: 0.168_004_102_156_450,
        },
        GaussPoint {
            x: 0.512_690_537_086_477,
            w: 0.154_045_761_076_810,
        },
        GaussPoint {
            x: 0.657_671_159_216_691,
            w: 0.135_136_368_468_525,
        },
        GaussPoint {
            x: 0.781_514_003_896_801,
            w: 0.111_883_847_193_404,
        },
        GaussPoint {
            x: 0.880_239_153_726_986,
            w: 0.085_036_148_317_179,
        },
        GaussPoint {
            x: 0.950_675_521_768_768,
            w: 0.055_459_529_373_987,
        },
        GaussPoint {
            x: 0.990_575_475_314_417,
            w: 0.024_148_302_868_548,
        },
    ],
    // Order 18
    &[
        GaussPoint {
            x: -0.991_565_168_420_931,
            w: 0.021_616_013_526_483,
        },
        GaussPoint {
            x: -0.955_823_949_571_398,
            w: 0.049_714_548_894_970,
        },
        GaussPoint {
            x: -0.892_602_466_497_556,
            w: 0.076_425_730_254_890,
        },
        GaussPoint {
            x: -0.803_704_958_972_523,
            w: 0.100_942_044_106_287,
        },
        GaussPoint {
            x: -0.691_687_043_060_353,
            w: 0.122_555_206_711_478,
        },
        GaussPoint {
            x: -0.559_770_831_073_948,
            w: 0.140_642_914_670_651,
        },
        GaussPoint {
            x: -0.411_751_161_462_843,
            w: 0.154_684_675_126_265,
        },
        GaussPoint {
            x: -0.251_886_225_691_506,
            w: 0.164_276_483_745_833,
        },
        GaussPoint {
            x: -0.084_775_013_041_735,
            w: 0.169_142_382_963_144,
        },
        GaussPoint {
            x: 0.084_775_013_041_735,
            w: 0.169_142_382_963_144,
        },
        GaussPoint {
            x: 0.251_886_225_691_506,
            w: 0.164_276_483_745_833,
        },
        GaussPoint {
            x: 0.411_751_161_462_843,
            w: 0.154_684_675_126_265,
        },
        GaussPoint {
            x: 0.559_770_831_073_948,
            w: 0.140_642_914_670_651,
        },
        GaussPoint {
            x: 0.691_687_043_060_353,
            w: 0.122_555_206_711_478,
        },
        GaussPoint {
            x: 0.803_704_958_972_523,
            w: 0.100_942_044_106_287,
        },
        GaussPoint {
            x: 0.892_602_466_497_556,
            w: 0.076_425_730_254_890,
        },
        GaussPoint {
            x: 0.955_823_949_571_398,
            w: 0.049_714_548_894_970,
        },
        GaussPoint {
            x: 0.991_565_168_420_931,
            w: 0.021_616_013_526_483,
        },
    ],
    // Order 19
    &[
        GaussPoint {
            x: -0.992_406_843_843_584,
            w: 0.019_461_788_229_726,
        },
        GaussPoint {
            x: -0.960_208_152_134_830,
            w: 0.044_814_226_765_699,
        },
        GaussPoint {
            x: -0.903_155_903_614_818,
            w: 0.069_044_542_737_641,
        },
        GaussPoint {
            x: -0.822_714_656_537_143,
            w: 0.091_490_021_622_450,
        },
        GaussPoint {
            x: -0.720_966_177_335_229,
            w: 0.111_566_645_547_334,
        },
        GaussPoint {
            x: -0.600_545_304_661_681,
            w: 0.128_753_962_539_336,
        },
        GaussPoint {
            x: -0.464_570_741_375_961,
            w: 0.142_606_702_173_607,
        },
        GaussPoint {
            x: -0.316_564_099_963_630,
            w: 0.152_766_042_065_860,
        },
        GaussPoint {
            x: -0.160_358_645_640_225,
            w: 0.158_968_843_393_954,
        },
        GaussPoint {
            x: 0.0,
            w: 0.161_054_449_848_784,
        },
        GaussPoint {
            x: 0.160_358_645_640_225,
            w: 0.158_968_843_393_954,
        },
        GaussPoint {
            x: 0.316_564_099_963_630,
            w: 0.152_766_042_065_860,
        },
        GaussPoint {
            x: 0.464_570_741_375_961,
            w: 0.142_606_702_173_607,
        },
        GaussPoint {
            x: 0.600_545_304_661_681,
            w: 0.128_753_962_539_336,
        },
        GaussPoint {
            x: 0.720_966_177_335_229,
            w: 0.111_566_645_547_334,
        },
        GaussPoint {
            x: 0.822_714_656_537_143,
            w: 0.091_490_021_622_450,
        },
        GaussPoint {
            x: 0.903_155_903_614_818,
            w: 0.069_044_542_737_641,
        },
        GaussPoint {
            x: 0.960_208_152_134_830,
            w: 0.044_814_226_765_699,
        },
        GaussPoint {
            x: 0.992_406_843_843_584,
            w: 0.019_461_788_229_726,
        },
    ],
    // Order 20
    &[
        GaussPoint {
            x: -0.993_128_599_185_095,
            w: 0.017_614_007_139_152,
        },
        GaussPoint {
            x: -0.963_971_927_277_914,
            w: 0.040_601_429_800_387,
        },
        GaussPoint {
            x: -0.912_234_428_251_326,
            w: 0.062_672_048_334_109,
        },
        GaussPoint {
            x: -0.839_116_971_822_219,
            w: 0.083_276_741_576_705,
        },
        GaussPoint {
            x: -0.746_331_906_460_151,
            w: 0.101_930_119_817_240,
        },
        GaussPoint {
            x: -0.636_053_680_726_515,
            w: 0.118_194_531_961_518,
        },
        GaussPoint {
            x: -0.510_867_001_950_827,
            w: 0.131_688_638_449_177,
        },
        GaussPoint {
            x: -0.373_706_088_715_420,
            w: 0.142_096_109_318_382,
        },
        GaussPoint {
            x: -0.227_785_851_141_645,
            w: 0.149_172_986_472_604,
        },
        GaussPoint {
            x: -0.076_526_521_133_497,
            w: 0.152_753_387_130_726,
        },
        GaussPoint {
            x: 0.076_526_521_133_497,
            w: 0.152_753_387_130_726,
        },
        GaussPoint {
            x: 0.227_785_851_141_645,
            w: 0.149_172_986_472_604,
        },
        GaussPoint {
            x: 0.373_706_088_715_420,
            w: 0.142_096_109_318_382,
        },
        GaussPoint {
            x: 0.510_867_001_950_827,
            w: 0.131_688_638_449_177,
        },
        GaussPoint {
            x: 0.636_053_680_726_515,
            w: 0.118_194_531_961_518,
        },
        GaussPoint {
            x: 0.746_331_906_460_151,
            w: 0.101_930_119_817_240,
        },
        GaussPoint {
            x: 0.839_116_971_822_219,
            w: 0.083_276_741_576_705,
        },
        GaussPoint {
            x: 0.912_234_428_251_326,
            w: 0.062_672_048_334_109,
        },
        GaussPoint {
            x: 0.963_971_927_277_914,
            w: 0.040_601_429_800_387,
        },
        GaussPoint {
            x: 0.993_128_599_185_095,
            w: 0.017_614_007_139_152,
        },
    ],
];

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use std::f64::consts::PI;

    /// Verify weights sum to 2.0 for each order (integration of f(x)=1 over [-1,1]).
    #[test]
    fn weights_sum_to_two() {
        for order in 1..=MAX_ORDER {
            let pts = gauss_legendre_points(order);
            let sum: f64 = pts.iter().map(|p| p.w).sum();
            assert!(
                (sum - 2.0).abs() < 1e-12,
                "order {order}: weight sum = {sum}"
            );
        }
    }

    /// Verify symmetry: x values are symmetric, weights at symmetric positions match.
    #[test]
    fn points_are_symmetric() {
        for order in 1..=MAX_ORDER {
            let pts = gauss_legendre_points(order);
            let n = pts.len();
            for i in 0..n / 2 {
                let j = n - 1 - i;
                assert!(
                    (pts[i].x + pts[j].x).abs() < 1e-14,
                    "order {order}: x[{i}] + x[{j}] = {}",
                    pts[i].x + pts[j].x
                );
                assert!(
                    (pts[i].w - pts[j].w).abs() < 1e-14,
                    "order {order}: w[{i}] != w[{j}]"
                );
            }
        }
    }

    /// Integrate x^2 over [-1,1] = 2/3. Exact for order >= 2.
    #[test]
    fn integrate_x_squared() {
        for order in 2..=MAX_ORDER {
            let pts = gauss_legendre_points(order);
            let result: f64 = pts.iter().map(|p| p.x * p.x * p.w).sum();
            assert!(
                (result - 2.0 / 3.0).abs() < 1e-12,
                "order {order}: integral = {result}"
            );
        }
    }

    /// Integrate sin(x) over [0, pi] = 2.0. Use affine transform.
    #[test]
    fn integrate_sin_on_zero_to_pi() {
        let pts = gauss_legendre_points(8);
        let a = 0.0;
        let b = PI;
        let half = (b - a) / 2.0;
        let mid = f64::midpoint(b, a);
        let result: f64 = pts
            .iter()
            .map(|p| half * p.w * (half * p.x + mid).sin())
            .sum();
        assert!(
            (result - 2.0).abs() < 1e-10,
            "integral of sin(x) on [0,pi] = {result}"
        );
    }

    /// Surface area of a unit sphere via Gauss quadrature = 4*pi.
    #[test]
    fn sphere_area_via_quadrature() {
        let eval = |u: f64, v: f64| -> (Point3, Vec3) {
            // Sphere: P(u,v) = (cos(v)*cos(u), cos(v)*sin(u), sin(v))
            // ∂P/∂u × ∂P/∂v = -cos(v) * P (unnormalized, magnitude = cos(v))
            let cu = u.cos();
            let su = u.sin();
            let cv = v.cos();
            let sv = v.sin();
            let pt = Point3::new(cv * cu, cv * su, sv);
            // Unnormalized normal: magnitude = cos(v) (the Jacobian factor)
            let n = Vec3::new(-cv * cv * cu, -cv * cv * su, -cv * sv);
            (pt, n)
        };

        // Integrate over [0, 2*pi] x [-pi/2, pi/2] with subdivisions
        // 3 u-spans, 2 v-spans, order 4 each
        let u_knots = [0.0, 2.0 * PI / 3.0, 4.0 * PI / 3.0, 2.0 * PI];
        let v_knots = [-PI / 2.0, 0.0, PI / 2.0];
        let area = gauss_surface_area_spans(&eval, &u_knots, &v_knots, 4, 4);
        let expected = 4.0 * PI;
        assert!(
            (area - expected).abs() < 1e-6,
            "sphere area = {area}, expected {expected}"
        );
    }

    /// Volume of a unit sphere via divergence theorem quadrature = 4*pi/3.
    #[test]
    fn sphere_volume_via_quadrature() {
        let eval = |u: f64, v: f64| -> (Point3, Vec3) {
            let cu = u.cos();
            let su = u.sin();
            let cv = v.cos();
            let sv = v.sin();
            let pt = Point3::new(cv * cu, cv * su, sv);
            // Outward-pointing normal (∂P/∂v × ∂P/∂u for outward convention)
            let n = Vec3::new(cv * cv * cu, cv * cv * su, cv * sv);
            (pt, n)
        };

        let u_knots = [0.0, 2.0 * PI / 3.0, 4.0 * PI / 3.0, 2.0 * PI];
        let v_knots = [-PI / 2.0, 0.0, PI / 2.0];
        let vol = gauss_surface_volume_spans(&eval, &u_knots, &v_knots, 4, 4);
        let expected = 4.0 * PI / 3.0;
        assert!(
            (vol - expected).abs() < 1e-6,
            "sphere volume = {vol}, expected {expected}"
        );
    }

    /// Area of a flat unit square [0,1]x[0,1] = 1.0.
    #[test]
    fn flat_square_area() {
        let eval = |u: f64, v: f64| -> (Point3, Vec3) {
            let pt = Point3::new(u, v, 0.0);
            // ∂P/∂u × ∂P/∂v = (1,0,0) × (0,1,0) = (0,0,1)
            let n = Vec3::new(0.0, 0.0, 1.0);
            (pt, n)
        };
        let area = gauss_surface_area(&eval, (0.0, 1.0), (0.0, 1.0), 2, 2);
        assert!((area - 1.0).abs() < 1e-14, "area = {area}");
    }
}
