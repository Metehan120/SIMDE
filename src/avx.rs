use core::arch::x86_64::*;
#[cfg(any(feature = "alloc", feature = "std"))]
use core::error::Error;
use core::mem::zeroed;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;

use crate::simd_ext::*;

#[cfg(any(feature = "alloc", feature = "std"))]
pub type Result<T> = core::result::Result<T, Box<dyn Error>>;

#[cfg(all(not(feature = "alloc"), not(feature = "std")))]
pub type Result<T> = core::result::Result<T, &'static str>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingleOperations {
    Sqrt,
    RSqrt,
    Floor,
    Ceil,
    Rcp,
    Ln,
    Log2,
    Log10,
    Exp,
    Sin,
    Tan,
    Cos,
    ASin,
    ATan,
    ACos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add,
    Sub,
    HAdd,
    HSub,
    Mul,
    And,
    Or,
    Xor,
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOperation {
    Compare,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatOperation {
    MulFloat32,
    MulFloat64,
    DivFloat32,
    DivFloat64,
    AddFloat32,
    AddFloat64,
    SubFloat32,
    SubFloat64,
    XorFloat32,
    XorFloat64,
    OrFloat32,
    OrFloat64,
    AndFloat32,
    AndFloat64,
    HAddFloat32,
    HAddFloat64,
    HSubFloat32,
    HSubFloat64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmaOperation {
    FmaAdd,
    FmaSub,
    FnmaAdd,
    FnmaSub,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOperation {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

#[inline(always)]
pub fn avx2_mod_constant_i32<const U: usize>(a: __m256i, divisor: i32) -> [i32; U] {
    unsafe {
        let mut output = [0i32; U];
        if divisor > 0 && (divisor & (divisor - 1)) == 0 {
            let mask = _mm256_set1_epi32(divisor - 1);
            let result = _mm256_and_si256(a, mask);
            _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
            return output;
        }

        let a_f32 = _mm256_cvtepi32_ps(a);
        let divisor_f32 = _mm256_set1_ps(divisor as f32);
        let div = _mm256_div_ps(a_f32, divisor_f32);
        let floored = _mm256_floor_ps(div);
        let mul = _mm256_mul_ps(floored, divisor_f32);
        let result_f32 = _mm256_sub_ps(a_f32, mul);
        let result = _mm256_cvtps_epi32(result_f32);
        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
        output
    }
}

#[inline(always)]
pub fn avx2_fmod_constant_f32<const U: usize>(a: __m256, divisor: f32) -> [f32; U] {
    unsafe {
        let mut output = [0f32; U];
        let b = _mm256_set1_ps(divisor);

        let div = _mm256_div_ps(a, b);
        let floored = _mm256_floor_ps(div);
        let mul = _mm256_mul_ps(floored, b);
        let result = _mm256_sub_ps(a, mul);
        _mm256_storeu_ps(output.as_mut_ptr() as *mut f32, result);
        output
    }
}

#[inline(always)]
pub fn avx2_fmod_constant_f64<const U: usize>(a: __m256d, divisor: f64) -> [f64; U] {
    unsafe {
        let mut output = [0f64; U];
        let b = _mm256_set1_pd(divisor);

        let div = _mm256_div_pd(a, b);
        let floored = _mm256_floor_pd(div);
        let mul = _mm256_mul_pd(floored, b);
        let result = _mm256_sub_pd(a, mul);
        _mm256_storeu_pd(output.as_mut_ptr() as *mut f64, result);
        output
    }
}

#[inline(always)]
pub fn avx2_reverse_32bit(a: __m256i) -> __m256i {
    unsafe {
        let shuffle_mask = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

        _mm256_permutevar8x32_epi32(a, shuffle_mask)
    }
}

#[inline(always)]
pub fn avx2_reverse_16bit(a: __m256i) -> __m256i {
    unsafe {
        let swapped = _mm256_permute2x128_si256(a, a, 0x01);

        let shuffle_lo = _mm256_shufflelo_epi16(swapped, 0b00011011);
        let shuffle_hi = _mm256_shufflehi_epi16(shuffle_lo, 0b00011011);

        shuffle_hi
    }
}

#[inline(always)]
pub fn avx2_reverse_8bit(a: __m256i) -> __m256i {
    unsafe {
        let swapped = _mm256_permute2x128_si256(a, a, 0x01);

        let reverse_mask = _mm256_setr_epi8(
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7,
            6, 5, 4, 3, 2, 1, 0,
        );

        _mm256_shuffle_epi8(swapped, reverse_mask)
    }
}

#[inline(always)]
pub fn avx2_reverse_f32(a: __m256) -> __m256 {
    unsafe {
        let indices = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

        _mm256_permutevar8x32_ps(a, indices)
    }
}

#[inline(always)]
pub fn avx2_reverse_f64(a: __m256d) -> __m256d {
    unsafe {
        let swapped = _mm256_permute2f128_pd(a, a, 0x01);

        _mm256_shuffle_pd(swapped, swapped, 0b0101)
    }
}

#[inline(always)]
pub fn avx2_compare_f32<T: 'static + Copy, const U: usize>(
    a: __m256,
    b: __m256,
    op: CompareOperation,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    unsafe {
        let result = match op {
            CompareOperation::Equal => _mm256_cmp_ps(a, b, _CMP_EQ_US),
            CompareOperation::NotEqual => _mm256_cmp_ps(a, b, _CMP_NEQ_US),
            CompareOperation::Greater => _mm256_cmp_ps(a, b, _CMP_GT_OS),
            CompareOperation::GreaterEqual => _mm256_cmp_ps(a, b, _CMP_GE_OS),
            CompareOperation::Less => _mm256_cmp_ps(a, b, _CMP_LT_OS),
            CompareOperation::LessEqual => _mm256_cmp_ps(a, b, _CMP_LE_OS),
        };

        _mm256_storeu_ps(output.as_mut_ptr() as *mut f32, result);
    }

    Ok(output)
}

#[inline(always)]
pub fn avx2_compare_f64<T: 'static + Copy, const U: usize>(
    a: __m256d,
    b: __m256d,
    op: CompareOperation,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    unsafe {
        let result = match op {
            CompareOperation::Equal => _mm256_cmp_pd(a, b, _CMP_EQ_US),
            CompareOperation::NotEqual => _mm256_cmp_pd(a, b, _CMP_NEQ_US),
            CompareOperation::Greater => _mm256_cmp_pd(a, b, _CMP_GT_OS),
            CompareOperation::GreaterEqual => _mm256_cmp_pd(a, b, _CMP_GE_OS),
            CompareOperation::Less => _mm256_cmp_pd(a, b, _CMP_LT_OS),
            CompareOperation::LessEqual => _mm256_cmp_pd(a, b, _CMP_LE_OS),
        };

        _mm256_storeu_pd(output.as_mut_ptr() as *mut f64, result);
    }

    Ok(output)
}

#[inline(always)]
pub fn avx2_blend_f32<T: 'static + Copy, const U: usize>(
    a: __m256,
    b: __m256,
    mask: __m256,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    unsafe {
        let result = _mm256_blendv_ps(a, b, mask);

        _mm256_storeu_ps(output.as_mut_ptr() as *mut f32, result);

        Ok(output)
    }
}

#[inline(always)]
pub fn avx2_blend_f64<T: 'static + Copy, const U: usize>(
    a: __m256d,
    b: __m256d,
    mask: __m256d,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    unsafe {
        let result = _mm256_blendv_pd(a, b, mask);

        _mm256_storeu_pd(output.as_mut_ptr() as *mut f64, result);

        Ok(output)
    }
}

#[inline(always)]
pub fn avx2_pow_f32<T: 'static + Copy, const U: usize>(x: __m256, y: __m256) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    unsafe {
        let ln_x = simd_ln_f32(x);
        let y_ln_x = _mm256_mul_ps(y, ln_x);
        let result = simd_exp_f32(y_ln_x);

        _mm256_storeu_ps(output.as_mut_ptr() as *mut f32, result);

        Ok(output)
    }
}

#[inline(always)]
pub fn fma_f32_avx2<T: 'static + Copy, const U: usize>(
    a: __m256,
    b: __m256,
    c: __m256,
    op: FmaOperation,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    unsafe {
        let result = match op {
            FmaOperation::FmaAdd => _mm256_fmadd_ps(a, b, c),
            FmaOperation::FmaSub => _mm256_fmsub_ps(a, b, c),
            FmaOperation::FnmaAdd => _mm256_fnmadd_ps(a, b, c),
            FmaOperation::FnmaSub => _mm256_fnmsub_ps(a, b, c),
        };

        _mm256_storeu_ps(output.as_mut_ptr() as *mut f32, result);

        Ok(output)
    }
}

#[inline(always)]
pub fn fma_f64_avx2<T: 'static + Copy, const U: usize>(
    a: __m256d,
    b: __m256d,
    c: __m256d,
    op: FmaOperation,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    unsafe {
        let result = match op {
            FmaOperation::FmaAdd => _mm256_fmadd_pd(a, b, c),
            FmaOperation::FmaSub => _mm256_fmsub_pd(a, b, c),
            FmaOperation::FnmaAdd => _mm256_fnmadd_pd(a, b, c),
            FmaOperation::FnmaSub => _mm256_fnmsub_pd(a, b, c),
        };

        _mm256_storeu_pd(output.as_mut_ptr() as *mut f64, result);

        Ok(output)
    }
}

#[inline(always)]
pub fn avx2_shift_left_16bit<T: 'static + Copy, const U: usize, const VALUE: usize>(
    a: __m256i,
) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let shift_count = _mm_set_epi64x(0, VALUE as i64);
        let result = _mm256_sll_epi16(a, shift_count);
        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx2_shift_right_16bit<T: 'static + Copy, const U: usize, const VALUE: usize>(
    a: __m256i,
) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let shift_count = _mm_set_epi64x(0, VALUE as i64);
        let result = _mm256_sra_epi16(a, shift_count);
        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx2_shift_left_32bit<T: 'static + Copy, const U: usize, const VALUE: usize>(
    a: __m256i,
) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let shift_count = _mm_set_epi64x(0, VALUE as i64);
        let result = _mm256_sll_epi32(a, shift_count);
        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx2_shift_right_32bit<T: 'static + Copy, const U: usize, const VALUE: usize>(
    a: __m256i,
) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let shift_count = _mm_set_epi64x(0, VALUE as i64);
        let result = _mm256_sra_epi32(a, shift_count);
        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx_shift_left_16bit<T: 'static + Copy, const U: usize, const VALUE: usize>(
    a: __m128i,
) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let shift_count = _mm_set_epi64x(0, VALUE as i64);
        let result = _mm_sll_epi16(a, shift_count);
        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx_shift_right_16bit<T: 'static + Copy, const U: usize, const VALUE: usize>(
    a: __m128i,
) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let shift_count = _mm_set_epi64x(0, VALUE as i64);
        let result = _mm_srl_epi16(a, shift_count);
        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx_shift_left_32bit<T: 'static + Copy, const U: usize, const VALUE: usize>(
    a: __m128i,
) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let shift_count = _mm_set_epi64x(0, VALUE as i64);
        let result = _mm_sll_epi32(a, shift_count);
        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx_shift_right_32bit<T: 'static + Copy, const U: usize, const VALUE: usize>(
    a: __m128i,
) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let shift_count = _mm_set_epi64x(0, VALUE as i64);
        let result = _mm_srl_epi32(a, shift_count);
        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx_shift_left_64bit<T: 'static + Copy, const U: usize, const VALUE: usize>(
    a: __m128i,
) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let shift_count = _mm_set_epi64x(0, VALUE as i64);
        let result = _mm_sll_epi64(a, shift_count);
        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx_shift_right_64bit<T: 'static + Copy, const U: usize, const VALUE: usize>(
    a: __m128i,
) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let shift_count = _mm_set_epi64x(0, VALUE as i64);
        let result = _mm_srl_epi64(a, shift_count);
        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx2_abs_8bit<T: 'static + Copy, const U: usize>(a: __m256i) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let result = _mm256_abs_epi8(a);
        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx2_abs_16bit<T: 'static + Copy, const U: usize>(a: __m256i) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let result = _mm256_abs_epi16(a);
        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx2_abs_32bit<T: 'static + Copy, const U: usize>(a: __m256i) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let result = _mm256_abs_epi32(a);
        _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx2_single_f32<T: 'static + Copy, const U: usize>(
    a: __m256,
    operation: SingleOperations,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    unsafe {
        let result = match operation {
            SingleOperations::Sqrt => _mm256_sqrt_ps(a),
            SingleOperations::RSqrt => _mm256_rsqrt_ps(a),
            SingleOperations::Floor => _mm256_floor_ps(a),
            SingleOperations::Ceil => _mm256_ceil_ps(a),
            SingleOperations::Rcp => _mm256_rcp_ps(a),
            SingleOperations::Ln => simd_ln_f32(a),
            SingleOperations::Log2 => {
                let ln_result = simd_ln_f32(a);
                let ln2 = _mm256_set1_ps(core::f32::consts::LN_2);
                _mm256_div_ps(ln_result, ln2)
            }
            SingleOperations::Log10 => {
                let ln_result = simd_ln_f32(a);
                let ln10 = _mm256_set1_ps(core::f32::consts::LN_10);
                _mm256_div_ps(ln_result, ln10)
            }
            SingleOperations::Exp => simd_exp_f32(a),
            SingleOperations::Sin => avx2_sin_f32(a),
            SingleOperations::Tan => avx2_tan_f32(a),
            SingleOperations::Cos => avx2_cos_f32(a),
            SingleOperations::ASin => avx2_asin_f32(a),
            SingleOperations::ATan => avx2_atan_f32(a),
            SingleOperations::ACos => avx2_acos_f32(a),
        };

        _mm256_storeu_ps(output.as_mut_ptr() as *mut f32, result);

        Ok(output)
    }
}

#[inline(always)]
pub fn avx2_single_f64<T: 'static + Copy, const U: usize>(
    a: __m256d,
    operation: SingleOperations,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    unsafe {
        let result = match operation {
            SingleOperations::Sqrt => _mm256_sqrt_pd(a),
            SingleOperations::Floor => _mm256_floor_pd(a),
            SingleOperations::Ceil => _mm256_ceil_pd(a),
            _ => return Err("Operation not supported".into()),
        };

        _mm256_storeu_pd(output.as_mut_ptr() as *mut f64, result);

        Ok(output)
    }
}

#[inline(always)]
pub fn avx_abs_8bit<T: 'static + Copy, const U: usize>(a: __m128i) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let result = _mm_abs_epi8(a);
        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx_abs_16bit<T: 'static + Copy, const U: usize>(a: __m128i) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let result = _mm_abs_epi16(a);
        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx_abs_32bit<T: 'static + Copy, const U: usize>(a: __m128i) -> Result<[T; U]> {
    unsafe {
        let mut output = [core::mem::zeroed(); U];
        let result = _mm_abs_epi32(a);
        _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
        Ok(output)
    }
}

#[inline(always)]
pub fn avx_single_f32<T: 'static + Copy, const U: usize>(
    a: __m128,
    operation: SingleOperations,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    unsafe {
        let result = match operation {
            SingleOperations::Sqrt => _mm_sqrt_ps(a),
            SingleOperations::Floor => _mm_floor_ps(a),
            _ => return Err("Operation not supported".into()),
        };

        _mm_storeu_ps(output.as_mut_ptr() as *mut f32, result);

        Ok(output)
    }
}

#[inline(always)]
pub fn avx_single_f64<T: 'static + Copy, const U: usize>(
    a: __m128d,
    operation: SingleOperations,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    unsafe {
        let result = match operation {
            SingleOperations::Sqrt => _mm_sqrt_pd(a),
            SingleOperations::Floor => _mm_floor_pd(a),
            _ => return Err("Operation not supported".into()),
        };

        _mm_storeu_pd(output.as_mut_ptr() as *mut f64, result);

        Ok(output)
    }
}

#[inline(always)]
pub fn avx2_8bit<T: 'static + Copy, const U: usize>(
    a: __m256i,
    b: __m256i,
    operation: Operation,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    let data = unsafe {
        match operation {
            Operation::Add => _mm256_add_epi8(a, b),
            Operation::Sub => _mm256_sub_epi8(a, b),
            Operation::And => _mm256_and_si256(a, b),
            Operation::Or => _mm256_or_si256(a, b),
            Operation::Xor => _mm256_xor_si256(a, b),
            Operation::Min => _mm256_min_epi8(a, b),
            Operation::Max => _mm256_max_epi8(a, b),
            _ => return Err("Operation not supported".into()),
        }
    };

    unsafe { _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, data) }

    Ok(output)
}

#[inline(always)]
pub fn avx2_16bit<T: 'static + Copy, const U: usize>(
    a: __m256i,
    b: __m256i,
    operation: Operation,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    let data = unsafe {
        match operation {
            Operation::Add => _mm256_add_epi16(a, b),
            Operation::HAdd => _mm256_hadd_epi16(a, b),
            Operation::HSub => _mm256_hsub_epi16(a, b),
            Operation::Sub => _mm256_sub_epi16(a, b),
            Operation::Mul => _mm256_mullo_epi16(a, b),
            Operation::And => _mm256_and_si256(a, b),
            Operation::Or => _mm256_or_si256(a, b),
            Operation::Xor => _mm256_xor_si256(a, b),
            Operation::Min => _mm256_min_epi16(a, b),
            Operation::Max => _mm256_max_epi16(a, b),
        }
    };

    unsafe { _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, data) }

    Ok(output)
}

#[inline(always)]
pub fn avx2_32bit<T: 'static + Copy, const U: usize>(
    a: __m256i,
    b: __m256i,
    operation: Operation,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    let data = unsafe {
        match operation {
            Operation::Add => _mm256_add_epi32(a, b),
            Operation::HAdd => _mm256_hadd_epi32(a, b),
            Operation::HSub => _mm256_hsub_epi32(a, b),
            Operation::Sub => _mm256_sub_epi32(a, b),
            Operation::Mul => _mm256_mullo_epi32(a, b),
            Operation::And => _mm256_and_si256(a, b),
            Operation::Or => _mm256_or_si256(a, b),
            Operation::Xor => _mm256_xor_si256(a, b),
            Operation::Min => _mm256_min_epi32(a, b),
            Operation::Max => _mm256_max_epi32(a, b),
        }
    };

    unsafe { _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, data) }

    Ok(output)
}

#[inline(always)]
pub fn avx2_64bit<T: 'static + Copy, const U: usize>(
    a: __m256i,
    b: __m256i,
    operation: Operation,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    let data = unsafe {
        match operation {
            Operation::Add => _mm256_add_epi64(a, b),
            Operation::Sub => _mm256_sub_epi64(a, b),
            Operation::Mul => _mm256_mul_epi32(a, b),
            Operation::And => _mm256_and_si256(a, b),
            Operation::Or => _mm256_or_si256(a, b),
            Operation::Xor => _mm256_xor_si256(a, b),
            _ => return Err("Operation not supported".into()),
        }
    };

    unsafe { _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, data) }

    Ok(output)
}

#[inline(always)]
pub fn avx_8bit<T: 'static + Copy, const U: usize>(
    a: __m128i,
    b: __m128i,
    operation: Operation,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    let data = unsafe {
        match operation {
            Operation::Add => _mm_add_epi8(a, b),
            Operation::Sub => _mm_sub_epi8(a, b),
            Operation::And => _mm_and_si128(a, b),
            Operation::Or => _mm_or_si128(a, b),
            Operation::Xor => _mm_xor_si128(a, b),
            Operation::Min => _mm_min_epi8(a, b),
            Operation::Max => _mm_max_epi8(a, b),
            _ => return Err("Operation not supported".into()),
        }
    };

    unsafe { _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, data) }

    Ok(output)
}

#[inline(always)]
pub fn avx_16bit<T: 'static + Copy, const U: usize>(
    a: __m128i,
    b: __m128i,
    operation: Operation,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    let data = unsafe {
        match operation {
            Operation::Add => _mm_add_epi16(a, b),
            Operation::HAdd => _mm_hadd_epi16(a, b),
            Operation::HSub => _mm_hsub_epi16(a, b),
            Operation::Sub => _mm_sub_epi16(a, b),
            Operation::Mul => _mm_mullo_epi16(a, b),
            Operation::And => _mm_and_si128(a, b),
            Operation::Or => _mm_or_si128(a, b),
            Operation::Xor => _mm_xor_si128(a, b),
            Operation::Min => _mm_min_epi16(a, b),
            Operation::Max => _mm_max_epi16(a, b),
        }
    };

    unsafe { _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, data) }

    Ok(output)
}

#[inline(always)]
pub fn avx_32bit<T: 'static + Copy, const U: usize>(
    a: __m128i,
    b: __m128i,
    operation: Operation,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    let data = unsafe {
        match operation {
            Operation::Add => _mm_add_epi32(a, b),
            Operation::HAdd => _mm_hadd_epi32(a, b),
            Operation::HSub => _mm_hsub_epi32(a, b),
            Operation::Sub => _mm_sub_epi32(a, b),
            Operation::Mul => _mm_mullo_epi32(a, b),
            Operation::And => _mm_and_si128(a, b),
            Operation::Or => _mm_or_si128(a, b),
            Operation::Xor => _mm_xor_si128(a, b),
            Operation::Min => _mm_min_epi32(a, b),
            Operation::Max => _mm_max_epi32(a, b),
        }
    };

    unsafe { _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, data) }

    Ok(output)
}

#[inline(always)]
pub fn avx_64bit<T: 'static + Copy, const U: usize>(
    a: __m128i,
    b: __m128i,
    operation: Operation,
) -> Result<[T; U]> {
    let mut output = [unsafe { core::mem::zeroed() }; U];

    let data = unsafe {
        match operation {
            Operation::Add => _mm_add_epi64(a, b),
            Operation::Sub => _mm_sub_epi64(a, b),
            Operation::Mul => _mm_mul_epi32(a, b),
            Operation::And => _mm_and_si128(a, b),
            Operation::Or => _mm_or_si128(a, b),
            Operation::Xor => _mm_xor_si128(a, b),
            _ => return Err("Operation not supported".into()),
        }
    };

    unsafe { _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, data) }

    Ok(output)
}

#[inline(always)]
pub fn avx2_f32<T: Copy, const U: usize>(
    a: __m256,
    b: __m256,
    operation: FloatOperation,
) -> Result<[T; U]> {
    let mut output = [unsafe { zeroed() }; U];

    let data = unsafe {
        match operation {
            FloatOperation::DivFloat32 => _mm256_div_ps(a, b),
            FloatOperation::MulFloat32 => _mm256_mul_ps(a, b),
            FloatOperation::AddFloat32 => _mm256_add_ps(a, b),
            FloatOperation::SubFloat32 => _mm256_sub_ps(a, b),
            FloatOperation::OrFloat32 => _mm256_or_ps(a, b),
            FloatOperation::XorFloat32 => _mm256_xor_ps(a, b),
            FloatOperation::AndFloat32 => _mm256_and_ps(a, b),
            FloatOperation::HAddFloat32 => _mm256_hadd_ps(a, b),
            FloatOperation::HSubFloat32 => _mm256_hsub_ps(a, b),
            _ => return Err("Operation not supported".into()),
        }
    };

    unsafe {
        _mm256_storeu_ps(output.as_mut_ptr() as *mut f32, data);
    }

    Ok(output)
}

#[inline(always)]
pub fn avx2_f64<T: Copy, const U: usize>(
    a: __m256d,
    b: __m256d,
    operation: FloatOperation,
) -> Result<[T; U]> {
    let mut output = [unsafe { zeroed() }; U];

    let data = unsafe {
        match operation {
            FloatOperation::DivFloat64 => _mm256_div_pd(a, b),
            FloatOperation::MulFloat64 => _mm256_mul_pd(a, b),
            FloatOperation::AddFloat64 => _mm256_add_pd(a, b),
            FloatOperation::SubFloat64 => _mm256_sub_pd(a, b),
            FloatOperation::OrFloat64 => _mm256_or_pd(a, b),
            FloatOperation::XorFloat64 => _mm256_xor_pd(a, b),
            FloatOperation::AndFloat64 => _mm256_and_pd(a, b),
            FloatOperation::HAddFloat64 => _mm256_hadd_pd(a, b),
            FloatOperation::HSubFloat64 => _mm256_hsub_pd(a, b),
            _ => return Err("Operation not supported".into()),
        }
    };

    unsafe {
        _mm256_storeu_pd(output.as_mut_ptr() as *mut f64, data);
    }

    Ok(output)
}

#[inline(always)]
pub fn avx_f32<T: Copy, const U: usize>(
    a: __m128,
    b: __m128,
    operation: FloatOperation,
) -> Result<[T; U]> {
    let mut output = [unsafe { zeroed() }; U];

    let data = unsafe {
        match operation {
            FloatOperation::DivFloat32 => _mm_div_ps(a, b),
            FloatOperation::MulFloat32 => _mm_mul_ps(a, b),
            FloatOperation::AddFloat32 => _mm_add_ps(a, b),
            FloatOperation::SubFloat32 => _mm_sub_ps(a, b),
            FloatOperation::OrFloat32 => _mm_or_ps(a, b),
            FloatOperation::XorFloat32 => _mm_xor_ps(a, b),
            FloatOperation::AndFloat32 => _mm_and_ps(a, b),
            FloatOperation::HAddFloat32 => _mm_hadd_ps(a, b),
            FloatOperation::HSubFloat32 => _mm_hsub_ps(a, b),
            _ => return Err("Operation not supported".into()),
        }
    };

    unsafe {
        _mm_storeu_ps(output.as_mut_ptr() as *mut f32, data);
    }

    Ok(output)
}

#[inline(always)]
pub fn avx_f64<T: Copy, const U: usize>(
    a: __m128d,
    b: __m128d,
    operation: FloatOperation,
) -> Result<[T; U]> {
    let mut output = [unsafe { zeroed() }; U];

    let data = unsafe {
        match operation {
            FloatOperation::DivFloat64 => _mm_div_pd(a, b),
            FloatOperation::MulFloat64 => _mm_mul_pd(a, b),
            FloatOperation::AddFloat64 => _mm_add_pd(a, b),
            FloatOperation::SubFloat64 => _mm_sub_pd(a, b),
            FloatOperation::OrFloat64 => _mm_or_pd(a, b),
            FloatOperation::XorFloat64 => _mm_xor_pd(a, b),
            FloatOperation::AndFloat64 => _mm_and_pd(a, b),
            FloatOperation::HAddFloat64 => _mm_hadd_pd(a, b),
            FloatOperation::HSubFloat64 => _mm_hsub_pd(a, b),
            _ => return Err("Operation not supported".into()),
        }
    };

    unsafe {
        _mm_storeu_pd(output.as_mut_ptr() as *mut f64, data);
    }

    Ok(output)
}

#[inline(always)]
pub fn load_avx2<T>(a: &[T]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr() as *const __m256i) }
}

#[inline(always)]
pub fn load_avx<T>(a: &[T]) -> __m128i {
    unsafe { _mm_loadu_si128(a.as_ptr() as *const __m128i) }
}

#[inline(always)]
pub fn load_avx2_f32<T>(a: &[T]) -> __m256 {
    unsafe { _mm256_loadu_ps(a.as_ptr() as *const f32) }
}

#[inline(always)]
pub fn load_avx_f32<T>(a: &[T]) -> __m128 {
    unsafe { _mm_loadu_ps(a.as_ptr() as *const f32) }
}

#[inline(always)]
pub fn load_avx2_f64<T>(a: &[T]) -> __m256d {
    unsafe { _mm256_loadu_pd(a.as_ptr() as *const f64) }
}

#[inline(always)]
pub fn load_avx_f64<T>(a: &[T]) -> __m128d {
    unsafe { _mm_loadu_pd(a.as_ptr() as *const f64) }
}

#[inline(always)]
pub fn store_avx2<T: Copy, const U: usize>(a: __m256i) -> [T; U] {
    let mut output = [unsafe { zeroed() }; U];

    unsafe { _mm256_storeu_si256(output.as_mut_ptr() as *mut __m256i, a) }

    output
}

#[inline(always)]
pub fn store_avx<T: Copy, const U: usize>(a: __m128i) -> [T; U] {
    let mut output = [unsafe { zeroed() }; U];

    unsafe { _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, a) }

    output
}

#[inline(always)]
pub fn store_avx2_f32<T: Copy, const U: usize>(a: __m256) -> [T; U] {
    let mut output = [unsafe { zeroed() }; U];

    unsafe { _mm256_storeu_ps(output.as_mut_ptr() as *mut f32, a) }

    output
}

#[inline(always)]
pub fn store_avx_f32<T: Copy, const U: usize>(a: __m128) -> [T; U] {
    let mut output = [unsafe { zeroed() }; U];

    unsafe { _mm_storeu_ps(output.as_mut_ptr() as *mut f32, a) }

    output
}

#[inline(always)]
pub fn store_avx2_f64<T: Copy, const U: usize>(a: __m256d) -> [T; U] {
    let mut output = [unsafe { zeroed() }; U];

    unsafe { _mm256_storeu_pd(output.as_mut_ptr() as *mut f64, a) }

    output
}

#[inline(always)]
pub fn store_avx_f64<T: Copy, const U: usize>(a: __m128d) -> [T; U] {
    let mut output = [unsafe { zeroed() }; U];

    unsafe { _mm_storeu_pd(output.as_mut_ptr() as *mut f64, a) }

    output
}
