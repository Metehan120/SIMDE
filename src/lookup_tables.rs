use crate::{has_avx, has_avx2};
#[cfg(any(feature = "alloc", feature = "std"))]
use alloc::boxed::Box;
use core::arch::x86_64::*;
#[cfg(any(feature = "alloc", feature = "std"))]
use core::error::Error;
#[cfg(any(feature = "alloc", feature = "std"))]
extern crate alloc;

#[cfg(any(feature = "alloc", feature = "std"))]
pub type Result<T> = core::result::Result<T, Box<dyn Error>>;

#[cfg(all(not(feature = "alloc"), not(feature = "std")))]
pub type Result<T> = core::result::Result<T, &'static str>;

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdTableI32<const U: usize>([i32; U]);

impl<const U: usize> SimdTableI32<U>
where
    [i32; U]: Sized,
{
    pub fn new(data: [i32; U]) -> Self {
        Self(data)
    }

    pub fn preload(&self) -> Result<()> {
        const PREFETCH_DISTANCE: usize = 64;
        let chunks = U / PREFETCH_DISTANCE;

        for i in 0..chunks {
            unsafe {
                _mm_prefetch(
                    self.0.as_ptr().add(i * PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }

        Ok(())
    }

    #[inline(always)]
    pub fn read_offsets<const Y: usize>(&self, offsets: &[i32; Y]) -> [i32; Y] {
        unsafe {
            let mut result = [0i32; Y];
            let mut processed = 0;

            if has_avx2() {
                while processed + 8 <= Y {
                    let chunk_offsets =
                        _mm256_loadu_si256(offsets.as_ptr().add(processed) as *const __m256i);
                    let chunk_result =
                        _mm256_i32gather_epi32::<4>(self.0.as_ptr() as *const i32, chunk_offsets);
                    _mm256_storeu_si256(
                        result.as_mut_ptr().add(processed) as *mut __m256i,
                        chunk_result,
                    );
                    processed += 8;
                }
            }

            if has_avx() {
                while processed + 4 <= Y {
                    let chunk_offsets =
                        _mm_loadu_si128(offsets.as_ptr().add(processed) as *const __m128i);
                    let chunk_result =
                        _mm_i32gather_epi32::<4>(self.0.as_ptr() as *const i32, chunk_offsets);
                    _mm_storeu_si128(
                        result.as_mut_ptr().add(processed) as *mut __m128i,
                        chunk_result,
                    );
                    processed += 4;
                }
            }

            while processed < Y {
                result[processed] = self.0[offsets[processed] as usize];
                processed += 1;
            }

            result
        }
    }
}

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimdTableF32<const U: usize>([f32; U]);

impl<const U: usize> SimdTableF32<U>
where
    [f32; U]: Sized,
{
    pub fn new(data: [f32; U]) -> Self {
        Self(data)
    }

    pub fn preload(&self) -> Result<()> {
        const PREFETCH_DISTANCE: usize = 64;
        let chunks = U / PREFETCH_DISTANCE;

        for i in 0..chunks {
            unsafe {
                _mm_prefetch(
                    self.0.as_ptr().add(i * PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }

        Ok(())
    }

    #[inline(always)]
    pub fn read_offsets<const Y: usize>(&self, offsets: &[i32; Y]) -> [f32; Y] {
        unsafe {
            let mut result = [0f32; Y];
            let mut processed = 0;

            if has_avx2() {
                while processed + 8 <= Y {
                    let chunk_offsets =
                        _mm256_loadu_si256(offsets.as_ptr().add(processed) as *const __m256i);
                    let chunk_result =
                        _mm256_i32gather_ps::<4>(self.0.as_ptr() as *const f32, chunk_offsets);
                    _mm256_storeu_ps(result.as_mut_ptr().add(processed) as *mut f32, chunk_result);
                    processed += 8;
                }
            }

            if has_avx() {
                while processed + 4 <= Y {
                    let chunk_offsets =
                        _mm_loadu_si128(offsets.as_ptr().add(processed) as *const __m128i);
                    let chunk_result =
                        _mm_i32gather_ps::<4>(self.0.as_ptr() as *const f32, chunk_offsets);
                    _mm_storeu_ps(result.as_mut_ptr().add(processed) as *mut f32, chunk_result);
                    processed += 4;
                }
            }

            while processed < Y {
                result[processed] = self.0[offsets[processed] as usize];
                processed += 1;
            }

            result
        }
    }
}

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimdTableF64<const U: usize>([f64; U]);

impl<const U: usize> SimdTableF64<U>
where
    [f64; U]: Sized,
{
    pub fn new(data: [f64; U]) -> Self {
        Self(data)
    }

    pub fn preload(&self) -> Result<()> {
        const PREFETCH_DISTANCE: usize = 64;
        let chunks = U / PREFETCH_DISTANCE;

        for i in 0..chunks {
            unsafe {
                _mm_prefetch(
                    self.0.as_ptr().add(i * PREFETCH_DISTANCE) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }

        Ok(())
    }

    #[inline(always)]
    pub fn read_offsets<const Y: usize>(&self, offsets: &[i64; Y]) -> [f64; Y] {
        unsafe {
            let mut result = [0f64; Y];
            let mut processed = 0;

            if has_avx2() {
                while processed + 8 <= Y {
                    let chunk_offsets =
                        _mm256_loadu_si256(offsets.as_ptr().add(processed) as *const __m256i);
                    let chunk_result =
                        _mm256_i64gather_pd::<4>(self.0.as_ptr() as *const f64, chunk_offsets);
                    _mm256_storeu_pd(result.as_mut_ptr().add(processed) as *mut f64, chunk_result);
                    processed += 8;
                }
            }

            if has_avx() {
                while processed + 4 <= Y {
                    let chunk_offsets =
                        _mm_loadu_si128(offsets.as_ptr().add(processed) as *const __m128i);
                    let chunk_result =
                        _mm_i64gather_pd::<4>(self.0.as_ptr() as *const f64, chunk_offsets);
                    _mm_storeu_pd(result.as_mut_ptr().add(processed) as *mut f64, chunk_result);
                    processed += 4;
                }
            }

            while processed < Y {
                result[processed] = self.0[offsets[processed] as usize];
                processed += 1;
            }

            result
        }
    }
}
