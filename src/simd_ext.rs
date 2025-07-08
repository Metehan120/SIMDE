use core::arch::x86_64::*;

#[cfg(feature = "fast_math")]
#[inline(always)]
pub fn simd_exp_f32(x: __m256) -> __m256 {
    unsafe {
        let ln2 = _mm256_set1_ps(0.69314718056);
        let inv_ln2 = _mm256_set1_ps(1.44269504089);

        let c0 = _mm256_set1_ps(1.0);
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(0.16666667);

        let k_float = _mm256_mul_ps(x, inv_ln2);
        let k_int = _mm256_cvtps_epi32(k_float);
        let k_float_rounded = _mm256_cvtepi32_ps(k_int);

        let r = _mm256_sub_ps(x, _mm256_mul_ps(k_float_rounded, ln2));
        let r2 = _mm256_mul_ps(r, r);
        let r3 = _mm256_mul_ps(r2, r);

        let poly = _mm256_add_ps(
            c0,
            _mm256_add_ps(
                _mm256_mul_ps(c1, r),
                _mm256_add_ps(_mm256_mul_ps(c2, r2), _mm256_mul_ps(c3, r3)),
            ),
        );

        let bias = _mm256_set1_epi32(127);
        let exp_bits = _mm256_add_epi32(k_int, bias);
        let exp_shifted = _mm256_slli_epi32(exp_bits, 23);
        let power_of_2 = _mm256_castsi256_ps(exp_shifted);

        _mm256_mul_ps(power_of_2, poly)
    }
}

#[cfg(feature = "fast_math")]
#[inline(always)]
pub fn simd_ln_f32(x: __m256) -> __m256 {
    unsafe {
        let exp_mask = _mm256_set1_epi32(0x7F800000u32 as i32);
        let mant_mask = _mm256_set1_epi32(0x007FFFFFu32 as i32);
        let bias = _mm256_set1_epi32(127);

        let x_int = _mm256_castps_si256(x);
        let exp_bits = _mm256_and_si256(x_int, exp_mask);
        let exp_shifted = _mm256_srli_epi32(exp_bits, 23);
        let exp_unbiased = _mm256_sub_epi32(exp_shifted, bias);
        let exp_f = _mm256_cvtepi32_ps(exp_unbiased);

        let mant = _mm256_or_si256(
            _mm256_and_si256(x_int, mant_mask),
            _mm256_set1_epi32(0x3F800000u32 as i32),
        );
        let mant_f = _mm256_castsi256_ps(mant);
        let t = _mm256_sub_ps(mant_f, _mm256_set1_ps(1.0));

        let t2 = _mm256_mul_ps(t, t);
        let t3 = _mm256_mul_ps(t2, t);

        let poly = _mm256_add_ps(
            t,
            _mm256_add_ps(
                _mm256_mul_ps(t2, _mm256_set1_ps(-0.5)),
                _mm256_mul_ps(t3, _mm256_set1_ps(0.33333333)),
            ),
        );

        let ln_2 = _mm256_set1_ps(0.69314718056);
        let exp_part = _mm256_mul_ps(exp_f, ln_2);

        _mm256_add_ps(exp_part, poly)
    }
}

#[cfg(feature = "fast_math")]
#[inline(always)]
pub fn avx2_sin_f32(x: __m256) -> __m256 {
    unsafe {
        let two_pi = _mm256_set1_ps(6.28318530718);
        let pi = _mm256_set1_ps(3.14159265359);
        let half_pi = _mm256_set1_ps(1.57079632679);

        // Basit range reduction
        let div = _mm256_mul_ps(x, _mm256_set1_ps(0.15915494309)); // 1/(2*pi)
        let floor_div = _mm256_floor_ps(div);
        let reduced_x = _mm256_sub_ps(x, _mm256_mul_ps(floor_div, two_pi));

        let abs_reduced = _mm256_andnot_ps(_mm256_set1_ps(-0.0), reduced_x);
        let gt_pi = _mm256_cmp_ps(abs_reduced, pi, _CMP_GT_OQ);
        let gt_half_pi = _mm256_cmp_ps(abs_reduced, half_pi, _CMP_GT_OQ);

        let mut y = abs_reduced;
        let mut sign = _mm256_set1_ps(1.0);

        let pi_minus_x = _mm256_sub_ps(pi, abs_reduced);
        y = _mm256_blendv_ps(y, pi_minus_x, _mm256_andnot_ps(gt_pi, gt_half_pi));

        let x_minus_pi = _mm256_sub_ps(abs_reduced, pi);
        y = _mm256_blendv_ps(y, x_minus_pi, gt_pi);

        let neg_mask = _mm256_or_ps(
            gt_pi,
            _mm256_cmp_ps(reduced_x, _mm256_setzero_ps(), _CMP_LT_OQ),
        );
        sign = _mm256_blendv_ps(sign, _mm256_set1_ps(-1.0), neg_mask);

        let x2 = _mm256_mul_ps(y, y);
        let x3 = _mm256_mul_ps(x2, y);
        let x5 = _mm256_mul_ps(x3, x2);

        let result = _mm256_add_ps(
            y,
            _mm256_add_ps(
                _mm256_mul_ps(x3, _mm256_set1_ps(-0.16666667)),
                _mm256_mul_ps(x5, _mm256_set1_ps(0.00833333)),
            ),
        );

        _mm256_mul_ps(result, sign)
    }
}

#[cfg(feature = "fast_math")]
#[inline(always)]
pub fn avx2_cos_f32(x: __m256) -> __m256 {
    unsafe {
        let half_pi = _mm256_set1_ps(1.57079632679);
        let x_shifted = _mm256_add_ps(x, half_pi);
        avx2_sin_f32(x_shifted)
    }
}

#[cfg(feature = "fast_math")]
#[inline(always)]
pub fn avx2_tan_f32(x: __m256) -> __m256 {
    unsafe {
        let pi = _mm256_set1_ps(3.14159265359);
        let half_pi = _mm256_set1_ps(1.57079632679);

        let div = _mm256_div_ps(x, pi);
        let floor_div = _mm256_floor_ps(div);
        let reduced_x = _mm256_sub_ps(x, _mm256_mul_ps(floor_div, pi));

        let gt_half_pi = _mm256_cmp_ps(reduced_x, half_pi, _CMP_GT_OQ);
        let adjusted_x = _mm256_sub_ps(reduced_x, pi);
        let y = _mm256_blendv_ps(reduced_x, adjusted_x, gt_half_pi);

        let x2 = _mm256_mul_ps(y, y);
        let x3 = _mm256_mul_ps(x2, y);
        let x5 = _mm256_mul_ps(x3, x2);

        let result = _mm256_add_ps(
            y,
            _mm256_add_ps(
                _mm256_mul_ps(x3, _mm256_set1_ps(0.33333333)),
                _mm256_mul_ps(x5, _mm256_set1_ps(0.13333333)),
            ),
        );

        result
    }
}

#[cfg(feature = "fast_math")]
#[inline(always)]
pub fn avx2_asin_f32(x: __m256) -> __m256 {
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        let x5 = _mm256_mul_ps(x3, x2);

        let result = _mm256_add_ps(
            x,
            _mm256_add_ps(
                _mm256_mul_ps(x3, _mm256_set1_ps(0.16666667)),
                _mm256_mul_ps(x5, _mm256_set1_ps(0.075)),
            ),
        );

        // Range check
        let valid_range = _mm256_cmp_ps(abs_x, one, _CMP_LE_OQ);
        let nan_val = _mm256_set1_ps(f32::NAN);
        _mm256_blendv_ps(nan_val, result, valid_range)
    }
}

#[cfg(feature = "fast_math")]
#[inline(always)]
pub fn avx2_acos_f32(x: __m256) -> __m256 {
    unsafe {
        let pi_half = _mm256_set1_ps(1.57079632679);
        let asin_result = avx2_asin_f32(x);
        _mm256_sub_ps(pi_half, asin_result)
    }
}

#[cfg(feature = "fast_math")]
#[inline(always)]
pub fn avx2_atan_f32(x: __m256) -> __m256 {
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let pi_half = _mm256_set1_ps(1.57079632679);
        let zero = _mm256_setzero_ps();

        let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
        let use_reciprocal = _mm256_cmp_ps(abs_x, one, _CMP_GT_OQ);
        let reciprocal_x = _mm256_div_ps(one, abs_x);
        let y = _mm256_blendv_ps(abs_x, reciprocal_x, use_reciprocal);

        let y2 = _mm256_mul_ps(y, y);
        let y3 = _mm256_mul_ps(y2, y);
        let y5 = _mm256_mul_ps(y3, y2);

        let mut result = _mm256_add_ps(
            y,
            _mm256_add_ps(
                _mm256_mul_ps(y3, _mm256_set1_ps(-0.33333333)),
                _mm256_mul_ps(y5, _mm256_set1_ps(0.2)),
            ),
        );

        let corrected_result = _mm256_sub_ps(pi_half, result);
        result = _mm256_blendv_ps(result, corrected_result, use_reciprocal);

        let neg_mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
        let neg_result = _mm256_sub_ps(zero, result);
        _mm256_blendv_ps(result, neg_result, neg_mask)
    }
}

#[cfg(feature = "precise_math")]
#[inline(always)]
pub fn simd_exp_f32(x: __m256) -> __m256 {
    unsafe {
        let ln2 = _mm256_set1_ps(0.6931471805599453);
        let inv_ln2 = _mm256_set1_ps(1.4426950408889634);
        let c0 = _mm256_set1_ps(1.0);
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(0.16666666666666666);
        let c4 = _mm256_set1_ps(0.041666666666666664);
        let c5 = _mm256_set1_ps(0.008333333333333333);

        let k_float = _mm256_mul_ps(x, inv_ln2);
        let k_int = _mm256_cvtps_epi32(k_float);
        let k_float_rounded = _mm256_cvtepi32_ps(k_int);

        let r = _mm256_sub_ps(x, _mm256_mul_ps(k_float_rounded, ln2));

        let r2 = _mm256_mul_ps(r, r);
        let r3 = _mm256_mul_ps(r2, r);
        let r4 = _mm256_mul_ps(r3, r);
        let r5 = _mm256_mul_ps(r4, r);

        let poly = _mm256_add_ps(
            c0,
            _mm256_add_ps(
                _mm256_mul_ps(c1, r),
                _mm256_add_ps(
                    _mm256_mul_ps(c2, r2),
                    _mm256_add_ps(
                        _mm256_mul_ps(c3, r3),
                        _mm256_add_ps(_mm256_mul_ps(c4, r4), _mm256_mul_ps(c5, r5)),
                    ),
                ),
            ),
        );

        let bias = _mm256_set1_epi32(127);
        let exp_bits = _mm256_add_epi32(k_int, bias);
        let exp_shifted = _mm256_slli_epi32(exp_bits, 23);
        let power_of_2 = _mm256_castsi256_ps(exp_shifted);

        _mm256_mul_ps(power_of_2, poly)
    }
}

#[cfg(feature = "precise_math")]
#[inline(always)]
pub fn simd_ln_f32(x: __m256) -> __m256 {
    unsafe {
        let exp_mask = _mm256_set1_epi32(0x7F800000u32 as i32);
        let mant_mask = _mm256_set1_epi32(0x007FFFFFu32 as i32);
        let bias = _mm256_set1_epi32(127);

        let x_int = _mm256_castps_si256(x);

        let exp_bits = _mm256_and_si256(x_int, exp_mask);
        let exp_shifted = _mm256_srli_epi32(exp_bits, 23);
        let exp_unbiased = _mm256_sub_epi32(exp_shifted, bias);
        let exp_f = _mm256_cvtepi32_ps(exp_unbiased);

        let mant = _mm256_or_si256(
            _mm256_and_si256(x_int, mant_mask),
            _mm256_set1_epi32(0x3F800000u32 as i32),
        );
        let mant_f = _mm256_castsi256_ps(mant);

        let one = _mm256_set1_ps(1.0);
        let t = _mm256_sub_ps(mant_f, one);

        let t2 = _mm256_mul_ps(t, t);
        let t3 = _mm256_mul_ps(t2, t);
        let t4 = _mm256_mul_ps(t3, t);
        let t5 = _mm256_mul_ps(t4, t);

        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(-0.5);
        let c3 = _mm256_set1_ps(0.3333333333333333);
        let c4 = _mm256_set1_ps(-0.25);
        let c5 = _mm256_set1_ps(0.2);

        let poly = _mm256_add_ps(
            _mm256_mul_ps(c1, t),
            _mm256_add_ps(
                _mm256_mul_ps(c2, t2),
                _mm256_add_ps(
                    _mm256_mul_ps(c3, t3),
                    _mm256_add_ps(_mm256_mul_ps(c4, t4), _mm256_mul_ps(c5, t5)),
                ),
            ),
        );

        let ln_2 = _mm256_set1_ps(0.6931471805599453);
        let exp_part = _mm256_mul_ps(exp_f, ln_2);

        _mm256_add_ps(exp_part, poly)
    }
}

#[cfg(feature = "precise_math")]
const SIN_COEFFS: [f32; 4] = [
    -1.6666667163e-01,
    8.3333337680e-03,
    -1.9841270114e-04,
    2.7557314297e-06,
];

#[cfg(feature = "precise_math")]
#[inline(always)]
pub fn avx2_sin_f32(x: __m256) -> __m256 {
    unsafe {
        let pi = _mm256_set1_ps(std::f32::consts::PI);
        let two_pi = _mm256_set1_ps(2.0 * std::f32::consts::PI);
        let half_pi = _mm256_set1_ps(std::f32::consts::PI * 0.5);
        let three_half_pi = _mm256_set1_ps(1.5 * std::f32::consts::PI);

        let div = _mm256_div_ps(x, two_pi);
        let floor_div = _mm256_floor_ps(div);
        let reduced_x = _mm256_sub_ps(x, _mm256_mul_ps(floor_div, two_pi));

        let gt_half_pi = _mm256_cmp_ps(reduced_x, half_pi, _CMP_GT_OQ);
        let gt_pi = _mm256_cmp_ps(reduced_x, pi, _CMP_GT_OQ);
        let gt_three_half_pi = _mm256_cmp_ps(reduced_x, three_half_pi, _CMP_GT_OQ);

        let mut y = reduced_x;
        let mut sign = _mm256_set1_ps(1.0);

        // Q2: Use π - x, keep positive
        let mask_q2 = _mm256_andnot_ps(gt_pi, gt_half_pi);
        let q2_adjusted = _mm256_sub_ps(pi, reduced_x);
        y = _mm256_blendv_ps(y, q2_adjusted, mask_q2);

        let mask_q3 = _mm256_andnot_ps(gt_three_half_pi, gt_pi);
        let q3_adjusted = _mm256_sub_ps(reduced_x, pi);
        y = _mm256_blendv_ps(y, q3_adjusted, mask_q3);

        let mask_q4 = gt_three_half_pi;
        let q4_adjusted = _mm256_sub_ps(two_pi, reduced_x);
        y = _mm256_blendv_ps(y, q4_adjusted, mask_q4);

        let mask_negative = _mm256_or_ps(mask_q3, mask_q4);
        let neg_sign = _mm256_set1_ps(-1.0);
        sign = _mm256_blendv_ps(sign, neg_sign, mask_negative);

        let x2 = _mm256_mul_ps(y, y);
        let x3 = _mm256_mul_ps(x2, y);
        let x5 = _mm256_mul_ps(x3, x2);
        let x7 = _mm256_mul_ps(x5, x2);

        let c1 = _mm256_set1_ps(SIN_COEFFS[0]);
        let c2 = _mm256_set1_ps(SIN_COEFFS[1]);
        let c3 = _mm256_set1_ps(SIN_COEFFS[2]);
        let c4 = _mm256_set1_ps(SIN_COEFFS[3]);

        let mut result = y;
        result = _mm256_add_ps(result, _mm256_mul_ps(x3, c1));
        result = _mm256_add_ps(result, _mm256_mul_ps(x5, c2));
        result = _mm256_add_ps(result, _mm256_mul_ps(x7, c3));
        result = _mm256_add_ps(result, _mm256_mul_ps(_mm256_mul_ps(x7, x2), c4));

        _mm256_mul_ps(result, sign)
    }
}

#[cfg(feature = "precise_math")]
const TAN_COEFFS: [f32; 5] = [
    3.3333334327e-01,
    1.3333334029e-01,
    5.3968253968e-02,
    2.1869488294e-02,
    8.8632398605e-03,
];

#[cfg(feature = "precise_math")]
#[inline(always)]
pub fn avx2_tan_f32(x: __m256) -> __m256 {
    unsafe {
        let pi = _mm256_set1_ps(std::f32::consts::PI);
        let half_pi = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
        let quarter_pi = _mm256_set1_ps(std::f32::consts::FRAC_PI_4);

        let div = _mm256_div_ps(x, pi);
        let floor_div = _mm256_floor_ps(div);
        let reduced_x = _mm256_sub_ps(x, _mm256_mul_ps(floor_div, pi));

        let gt_half_pi = _mm256_cmp_ps(reduced_x, half_pi, _CMP_GT_OQ);
        let adjusted_x = _mm256_sub_ps(reduced_x, pi);
        let y = _mm256_blendv_ps(reduced_x, adjusted_x, gt_half_pi);

        let period_shift = _mm256_castsi256_ps(_mm256_and_si256(
            _mm256_castps_si256(floor_div),
            _mm256_set1_epi32(1),
        ));
        let mut sign = _mm256_set1_ps(1.0);
        let neg_sign = _mm256_set1_ps(-1.0);
        let period_neg_mask = _mm256_cmp_ps(period_shift, _mm256_setzero_ps(), _CMP_NEQ_OQ);
        sign = _mm256_blendv_ps(sign, neg_sign, period_neg_mask);

        let abs_y = _mm256_andnot_ps(_mm256_set1_ps(-0.0), y);
        let use_cotangent = _mm256_cmp_ps(abs_y, quarter_pi, _CMP_GT_OQ);

        let cot_arg = _mm256_sub_ps(half_pi, abs_y);
        let tan_arg = _mm256_blendv_ps(y, cot_arg, use_cotangent);

        let y_sign_mask = _mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_LT_OQ);
        let signed_arg = _mm256_blendv_ps(
            _mm256_andnot_ps(_mm256_set1_ps(-0.0), tan_arg),
            _mm256_or_ps(
                _mm256_set1_ps(-0.0),
                _mm256_andnot_ps(_mm256_set1_ps(-0.0), tan_arg),
            ),
            y_sign_mask,
        );

        let x2 = _mm256_mul_ps(signed_arg, signed_arg);
        let x3 = _mm256_mul_ps(x2, signed_arg);
        let x5 = _mm256_mul_ps(x3, x2);
        let x7 = _mm256_mul_ps(x5, x2);
        let x9 = _mm256_mul_ps(x7, x2);

        let mut result = signed_arg;
        result = _mm256_add_ps(result, _mm256_mul_ps(x3, _mm256_set1_ps(TAN_COEFFS[0])));
        result = _mm256_add_ps(result, _mm256_mul_ps(x5, _mm256_set1_ps(TAN_COEFFS[1])));
        result = _mm256_add_ps(result, _mm256_mul_ps(x7, _mm256_set1_ps(TAN_COEFFS[2])));
        result = _mm256_add_ps(result, _mm256_mul_ps(x9, _mm256_set1_ps(TAN_COEFFS[3])));

        let one = _mm256_set1_ps(1.0);
        let reciprocal = _mm256_div_ps(one, result);
        result = _mm256_blendv_ps(result, reciprocal, use_cotangent);

        _mm256_mul_ps(result, sign)
    }
}

#[cfg(feature = "precise_math")]
const COS_COEFFS: [f32; 5] = [
    -0.5,
    0.041666666667,
    -0.001388888889,
    0.000024801587,
    -0.000000275573,
];

#[cfg(feature = "precise_math")]
#[inline(always)]
pub fn avx2_cos_f32(x: __m256) -> __m256 {
    unsafe {
        let pi = _mm256_set1_ps(std::f32::consts::PI);
        let two_pi = _mm256_set1_ps(2.0 * std::f32::consts::PI);
        let half_pi = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
        let three_half_pi = _mm256_set1_ps(1.5 * std::f32::consts::PI);

        let div = _mm256_div_ps(x, two_pi);
        let floor_div = _mm256_floor_ps(div);
        let reduced_x = _mm256_sub_ps(x, _mm256_mul_ps(floor_div, two_pi));

        let gt_half_pi = _mm256_cmp_ps(reduced_x, half_pi, _CMP_GT_OQ);
        let gt_pi = _mm256_cmp_ps(reduced_x, pi, _CMP_GT_OQ);
        let gt_three_half_pi = _mm256_cmp_ps(reduced_x, three_half_pi, _CMP_GT_OQ);

        let mut y = reduced_x;
        let mut sign = _mm256_set1_ps(1.0);

        let mask_q2 = _mm256_andnot_ps(gt_pi, gt_half_pi);
        let q2_adjusted = _mm256_sub_ps(pi, reduced_x);
        y = _mm256_blendv_ps(y, q2_adjusted, mask_q2);

        let mask_q3 = _mm256_andnot_ps(gt_three_half_pi, gt_pi);
        let q3_adjusted = _mm256_sub_ps(reduced_x, pi);
        y = _mm256_blendv_ps(y, q3_adjusted, mask_q3);

        let mask_q4 = gt_three_half_pi;
        let q4_adjusted = _mm256_sub_ps(two_pi, reduced_x);
        y = _mm256_blendv_ps(y, q4_adjusted, mask_q4);

        let mask_negative = _mm256_or_ps(mask_q2, mask_q3);
        let neg_sign = _mm256_set1_ps(-1.0);
        sign = _mm256_blendv_ps(sign, neg_sign, mask_negative);

        let x2 = _mm256_mul_ps(y, y);
        let x4 = _mm256_mul_ps(x2, x2);
        let x6 = _mm256_mul_ps(x4, x2);
        let x8 = _mm256_mul_ps(x6, x2);
        let x10 = _mm256_mul_ps(x8, x2);

        let one = _mm256_set1_ps(1.0);
        let mut result = one;
        result = _mm256_add_ps(result, _mm256_mul_ps(x2, _mm256_set1_ps(COS_COEFFS[0])));
        result = _mm256_add_ps(result, _mm256_mul_ps(x4, _mm256_set1_ps(COS_COEFFS[1])));
        result = _mm256_add_ps(result, _mm256_mul_ps(x6, _mm256_set1_ps(COS_COEFFS[2])));
        result = _mm256_add_ps(result, _mm256_mul_ps(x8, _mm256_set1_ps(COS_COEFFS[3])));
        result = _mm256_add_ps(result, _mm256_mul_ps(x10, _mm256_set1_ps(COS_COEFFS[4])));

        _mm256_mul_ps(result, sign)
    }
}

#[cfg(feature = "precise_math")]
const ASIN_COEFFS: [f32; 6] = [
    1.0,             // x^1 katsayısı
    1.0 / 6.0,       // x^3 katsayısı
    3.0 / 40.0,      // x^5 katsayısı
    15.0 / 336.0,    // x^7 katsayısı
    105.0 / 3456.0,  // x^9 katsayısı
    945.0 / 42240.0, // x^11 katsayısı
];

#[cfg(feature = "precise_math")]
#[inline(always)]
pub fn avx2_asin_f32(x: __m256) -> __m256 {
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let half = _mm256_set1_ps(0.5);
        let zero = _mm256_setzero_ps();
        let pi_half = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

        let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

        let valid_range = _mm256_cmp_ps(abs_x, one, _CMP_LE_OQ);

        let use_alternative = _mm256_cmp_ps(abs_x, half, _CMP_GT_OQ);

        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        let x5 = _mm256_mul_ps(x3, x2);
        let x7 = _mm256_mul_ps(x5, x2);
        let x9 = _mm256_mul_ps(x7, x2);
        let x11 = _mm256_mul_ps(x9, x2);

        let mut taylor_result = x;
        taylor_result = _mm256_add_ps(
            taylor_result,
            _mm256_mul_ps(x3, _mm256_set1_ps(ASIN_COEFFS[1])),
        );
        taylor_result = _mm256_add_ps(
            taylor_result,
            _mm256_mul_ps(x5, _mm256_set1_ps(ASIN_COEFFS[2])),
        );
        taylor_result = _mm256_add_ps(
            taylor_result,
            _mm256_mul_ps(x7, _mm256_set1_ps(ASIN_COEFFS[3])),
        );
        taylor_result = _mm256_add_ps(
            taylor_result,
            _mm256_mul_ps(x9, _mm256_set1_ps(ASIN_COEFFS[4])),
        );
        taylor_result = _mm256_add_ps(
            taylor_result,
            _mm256_mul_ps(x11, _mm256_set1_ps(ASIN_COEFFS[5])),
        );

        // Büyük değerler için: asin(x) = π/2 - 2*asin(√((1-x)/2))
        let one_minus_abs_x = _mm256_sub_ps(one, abs_x);
        let half_expr = _mm256_mul_ps(one_minus_abs_x, half);
        let sqrt_expr = _mm256_sqrt_ps(half_expr);

        let sqrt_x2 = _mm256_mul_ps(sqrt_expr, sqrt_expr);
        let sqrt_x3 = _mm256_mul_ps(sqrt_x2, sqrt_expr);
        let sqrt_x5 = _mm256_mul_ps(sqrt_x3, sqrt_x2);

        let mut sqrt_asin = sqrt_expr;
        sqrt_asin = _mm256_add_ps(
            sqrt_asin,
            _mm256_mul_ps(sqrt_x3, _mm256_set1_ps(ASIN_COEFFS[1])),
        );
        sqrt_asin = _mm256_add_ps(
            sqrt_asin,
            _mm256_mul_ps(sqrt_x5, _mm256_set1_ps(ASIN_COEFFS[2])),
        );

        let two = _mm256_set1_ps(2.0);
        let alternative_result = _mm256_sub_ps(pi_half, _mm256_mul_ps(two, sqrt_asin));

        let mut result = _mm256_blendv_ps(taylor_result, alternative_result, use_alternative);

        let neg_mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
        let neg_result = _mm256_sub_ps(zero, result);
        result = _mm256_blendv_ps(result, neg_result, neg_mask);

        let nan_val = _mm256_set1_ps(f32::NAN);
        _mm256_blendv_ps(nan_val, result, valid_range)
    }
}

#[cfg(feature = "precise_math")]
#[inline(always)]
pub fn avx2_acos_f32(x: __m256) -> __m256 {
    unsafe {
        let pi_half = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
        let one = _mm256_set1_ps(1.0);

        let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
        let valid_range = _mm256_cmp_ps(abs_x, one, _CMP_LE_OQ);

        let asin_result = avx2_asin_f32(x);
        let result = _mm256_sub_ps(pi_half, asin_result);

        let nan_val = _mm256_set1_ps(f32::NAN);
        _mm256_blendv_ps(nan_val, result, valid_range)
    }
}

#[cfg(feature = "precise_math")]
const ATAN_COEFFS: [f32; 5] = [
    1.0,        // x katsayısı
    -1.0 / 3.0, // x³ katsayısı
    1.0 / 5.0,  // x⁵ katsayısı
    -1.0 / 7.0, // x⁷ katsayısı
    1.0 / 9.0,  // x⁹ katsayısı
];

#[cfg(feature = "precise_math")]
#[inline(always)]
pub fn avx2_atan_f32(x: __m256) -> __m256 {
    unsafe {
        let one = _mm256_set1_ps(1.0);
        let pi_half = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
        // let pi_quarter = _mm256_set1_ps(std::f32::consts::FRAC_PI_4);
        let zero = _mm256_setzero_ps();

        let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

        let use_reciprocal = _mm256_cmp_ps(abs_x, one, _CMP_GT_OQ);
        let reciprocal_x = _mm256_div_ps(one, abs_x);
        let y = _mm256_blendv_ps(abs_x, reciprocal_x, use_reciprocal);

        let y2 = _mm256_mul_ps(y, y);
        let y3 = _mm256_mul_ps(y2, y);
        let y5 = _mm256_mul_ps(y3, y2);
        let y7 = _mm256_mul_ps(y5, y2);
        let y9 = _mm256_mul_ps(y7, y2);

        let mut result = y;
        result = _mm256_add_ps(result, _mm256_mul_ps(y3, _mm256_set1_ps(ATAN_COEFFS[1])));
        result = _mm256_add_ps(result, _mm256_mul_ps(y5, _mm256_set1_ps(ATAN_COEFFS[2])));
        result = _mm256_add_ps(result, _mm256_mul_ps(y7, _mm256_set1_ps(ATAN_COEFFS[3])));
        result = _mm256_add_ps(result, _mm256_mul_ps(y9, _mm256_set1_ps(ATAN_COEFFS[4])));

        let corrected_result = _mm256_sub_ps(pi_half, result);
        result = _mm256_blendv_ps(result, corrected_result, use_reciprocal);

        let neg_mask = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
        let neg_result = _mm256_sub_ps(zero, result);
        result = _mm256_blendv_ps(result, neg_result, neg_mask);

        result
    }
}
