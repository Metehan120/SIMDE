//! # SIMDE - High-Performance SIMD Library for Rust
//!
//! **SIMDE** is a comprehensive, zero-cost abstraction SIMD library that provides
//! ergonomic wrappers around Intel AVX/AVX2 intrinsics. Built for maximum performance
//! while maintaining type safety and ease of use.
//!
//! ## Why SIMDE?
//!
//! SIMDE solves these problems by providing:
//! - **Human-readable API**: `data.add(&other)` instead of `_mm256_add_epi32`
//! - **Automatic optimization**: Runtime CPU detection with optimal instruction selection
//! - **Type safety**: Generic implementations with compile-time guarantees
//! - **Memory alignment**: Built-in utilities for SIMD-friendly data layout
//! - **Comprehensive coverage**: Support for all major SIMD operations
//!
//! ## Quick Start
//!
//! Add SIMDE to your `Cargo.toml`:
//! ```toml
//! [dependencies]
//! simde = "0.1"
//! ```
//!
//! ### Basic Usage
//!
//! ```rust
//! use simde::*;
//!
//! // Simple arithmetic operations
//! let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
//!
//! let sum = a.add(&b)?;        // SIMD addition
//! let product = a.mul(&b)?;    // SIMD multiplication
//! let sqrt_a = a.sqrt()?;      // SIMD square root
//! ```
//!
//! ### Advanced Operations
//!
//! ```rust
//! // Mathematical functions
//! let angles = [0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.14159];
//! let sines = angles.sin()?;
//! let cosines = angles.cos()?;
//! let logarithms = angles.ln()?;
//!
//! // Fused multiply-add for maximum precision
//! let c = [1.0f32; 8];
//! let result = a.fma_add(&b, &c)?;  // (a * b) + c
//!
//! // Comparison and blending
//! let mask = a.gt(&b)?;
//! let blended = a.blend(&b, &mask)?;
//! ```
//!
//! ## Supported Operations
//!
//! ### Integer Operations (i8, i16, i32, i64, u8, u16, u32, u64)
//! - **Arithmetic**: `add`, `sub`, `mul`, `hadd`, `hsub`
//! - **Bitwise**: `and`, `or`, `xor`
//! - **Comparison**: `min`, `max`
//! - **Utility**: `abs`, `modulo`, `reverse_simd`
//! - **Bit shifting**: `shift_left`, `shift_right`
//!
//! ### Float Operations (f32, f64)
//! - **Arithmetic**: `add`, `sub`, `mul`, `div`, `hadd`, `hsub`
//! - **Advanced Math**: `sqrt`, `rsqrt`, `pow`, `rcp`
//! - **Transcendental**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
//! - **Logarithmic**: `ln`, `log2`, `log10`, `exp`
//! - **Rounding**: `floor`, `ceil`
//! - **Comparison**: `eq`, `not_eq`, `lt`, `lte`, `gt`, `gte`
//! - **FMA**: `fma_add`, `fma_sub`, `fnma_add`, `fnma_sub`
//! - **Utility**: `blend`, `modulo`, `reverse_simd`
//!
//! ### Lookup Table Operations
//! - **Fast Gather**: SIMD-accelerated table lookups using gather instructions
//! - **Prefetching**: Automatic memory prefetching for large tables
//! - **Multiple Types**: Support for i32, f32, f64 lookup tables
//!
//! ### Memory Alignment
//! - **AlignData<T, N>**: Generic aligned array wrapper with 32-byte alignment
//! - **Type Aliases**: Convenient shortcuts like `AlignF32x8`, `AlignI32x8`
//! - **Transparent Access**: Works like normal arrays with `Deref` traits
//!
//! ## Performance
//!
//! SIMDE provides significant performance improvements over scalar operations:
//! - **2-20x speedup** for basic operations depending on data size
//! - **Zero overhead** function calls with inline optimization
//! - **Automatic vectorization** with optimal instruction selection
//! - **Memory throughput** improvements up to 100GB/s on modern CPUs
//!
//! ### SIMD Lane Utilization
//! SIMDE operates on fixed-size arrays optimized for 256-bit SIMD registers:
//! - **8-bit types**: 32 elements per operation
//! - **16-bit types**: 16 elements per operation
//! - **32-bit types**: 8 elements per operation
//! - **64-bit types**: 4 elements per operation
//!
//! ## CPU Feature Detection
//!
//! SIMDE automatically detects and utilizes the best available instruction set:
//!
//! ```rust
//! use simde::{has_avx, has_avx2};
//!
//! if has_avx2() {
//!     // Uses 256-bit AVX2 instructions for maximum performance
//! } else if has_avx() {
//!     // Falls back to 128-bit AVX instructions
//! } else {
//!     // Returns error - SIMD not supported
//! }
//! ```
//!
//! ### Feature Flags
//!
//! Configure SIMDE's behavior through Cargo features:
//!
//! ```toml
//! [features]
//! default = ["std", "runtime_detection"]
//! std = []                    # Standard library support
//! alloc = []                  # Heap allocation support
//! avx2 = ["avx"]              # Enable AVX2 instructions
//! avx = []                    # Enable AVX instructions
//! runtime_detection = []      # Automatic CPU feature detection
//! no_main = []                # For no_main environments
//! ```
//!
//! ## Memory Alignment
//!
//! SIMDE provides utilities for proper SIMD memory alignment:
//!
//! ```rust
//! use simde::{AlignData, AlignF32x8};
//!
//! // Automatic 32-byte alignment
//! let aligned_data = AlignF32x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
//!
//! // Guaranteed alignment for optimal performance
//! assert_eq!(aligned_data.as_ptr() as usize % 32, 0);
//!
//! // Works like a normal array
//! let result = aligned_data.add(&other_data)?;
//! ```
//!
//! ### Type Aliases for Common Patterns
//!
//! ```rust
//! use simde::*;
//!
//! type AlignF32x8 = AlignData<f32, 8>;   // 8 x f32 (256-bit)
//! type AlignF64x4 = AlignData<f64, 4>;   // 4 x f64 (256-bit)
//! type AlignI32x8 = AlignData<i32, 8>;   // 8 x i32 (256-bit)
//! type AlignI16x16 = AlignData<i16, 16>; // 16 x i16 (256-bit)
//! type AlignU8x32 = AlignData<u8, 32>;   // 32 x u8 (256-bit)
//! ```
//!
//! ## Lookup Tables
//!
//! SIMDE provides high-performance lookup table implementations:
//!
//! ```rust
//! use simde::lookup_tables::SimdTableF32;
//!
//! // Create a lookup table for sin values
//! let sin_table: Vec<f32> = (0..1024)
//!     .map(|i| (i as f32 * std::f32::consts::PI / 512.0).sin())
//!     .collect();
//!
//! let table = SimdTableF32::new(sin_table.try_into().unwrap());
//! table.preload()?; // Preload into CPU cache
//!
//! // Fast SIMD lookups using gather instructions
//! let indices = [0i32, 128, 256, 384, 512, 640, 768, 896];
//! let sin_values = table.read_offsets(&indices);
//! ```
//!
//! ## Error Handling
//!
//! All SIMD operations return `Result` types for safe error handling:
//!
//! ```rust
//! use simde::Result;
//!
//! fn process_data(input: &[f32; 8]) -> Result<[f32; 8]> {
//!     let normalized = input.div(&[2.0; 8])?;
//!     let roots = normalized.sqrt()?;
//!     let scaled = roots.mul(&[10.0; 8])?;
//!     Ok(scaled)
//! }
//! ```
//!
//! Common error conditions:
//! - **Unsupported operations**: e.g., `mul` on 8-bit integers
//! - **Missing CPU features**: Operations requiring AVX2 on AVX-only systems
//! - **Invalid parameters**: e.g., negative values for `sqrt`
//!
//! ## No-STD Support
//!
//! SIMDE works in `no_std` environments:
//!
//! ```toml
//! [dependencies]
//! simde = { version = "0.1", default-features = false, features = ["alloc"] }
//! ```
//!
//! For bare metal environments:
//! ```toml
//! simde = { version = "0.1", default-features = false, features = ["no_main"] }
//! ```
//!
//! ## Architecture Support
//!
//! **Minimum Requirements:**
//! - x86_64 architecture
//! - SSE2 support (available on all x86_64 CPUs)
//!
//! **Recommended:**
//! - Intel Haswell (2013+) or AMD Excavator (2015+) for AVX2 support
//! - Modern CPUs with AVX-512 (future support)
//!
//! **Supported Platforms:**
//! - Linux (x86_64-unknown-linux-gnu)
//! - Windows (x86_64-pc-windows-msvc)
//! - macOS (x86_64-apple-darwin)
//!
//! ## Examples
//!
//! ### Image Processing
//! ```rust
//! use simde::*;
//!
//! fn brightness_adjust(pixels: &mut [u8; 32], adjustment: u8) -> Result<()> {
//!     let adj_array = [adjustment; 32];
//!     let result = pixels.add(&adj_array)?;
//!     *pixels = result;
//!     Ok(())
//! }
//! ```
//!
//! ### Scientific Computing
//! ```rust
//! use simde::*;
//!
//! fn vector_magnitude(x: &[f32; 8], y: &[f32; 8], z: &[f32; 8]) -> Result<[f32; 8]> {
//!     let x_squared = x.mul(x)?;
//!     let y_squared = y.mul(y)?;
//!     let z_squared = z.mul(z)?;
//!
//!     let sum = x_squared.add(&y_squared)?.add(&z_squared)?;
//!     sum.sqrt()
//! }
//! ```
//!
//! ### Digital Signal Processing
//! ```rust
//! use simde::*;
//!
//! fn apply_window_function(signal: &[f32; 8], window: &[f32; 8]) -> Result<[f32; 8]> {
//!     signal.mul(window)
//! }
//!
//! fn compute_phase(real: &[f32; 8], imag: &[f32; 8]) -> Result<[f32; 8]> {
//!     let ratio = imag.div(real)?;
//!     ratio.atan()
//! }
//! ```
//!
//! ## Best Practices
//!
//! ### Memory Layout
//! - Use `AlignData<T, N>` for optimal SIMD performance
//! - Process data in chunks that match SIMD lane counts
//! - Prefer contiguous memory access patterns
//!
//! ### Error Handling
//! - Always handle `Result` return values
//! - Check CPU feature availability for optional optimizations
//! - Provide scalar fallbacks for unsupported operations
//!
//! ### Performance Tips
//! - Use `preload()` for large lookup tables
//! - Prefer FMA operations for better precision and performance
//! - Process multiple SIMD vectors in tight loops for better throughput
//!
//! ## Contributing
//!
//! SIMDE welcomes contributions! Key areas for improvement:
//! - Additional instruction set support (AVX-512, ARM NEON)
//! - More mathematical functions
//! - Platform-specific optimizations
//! - Documentation and examples
//!
//! ## License
//!
//! SIMDE is dual-licensed under MIT license.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "no_main", no_main)]

// I HATE YOU RUST JUST LEMME USE MACROS (After 10 hours I decided to use good old COPY + PASTE)

use core::{
    arch::x86_64::{
        __m128, __m128d, __m128i, __m256, __m256d, __m256i, _mm256_storeu_pd, _mm256_storeu_ps,
        _mm256_storeu_si256,
    },
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicBool, Ordering},
};

#[cfg(any(feature = "alloc", feature = "std"))]
use core::error::Error;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;

use crate::avx::*;

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod avx;
pub mod lookup_tables;
mod simd_ext;

pub const X8_TYPE: usize = 32;
pub const X16_TYPE: usize = 16;
pub const X32_TYPE: usize = 8;
pub const X64_TYPE: usize = 4;

#[cfg(all(feature = "avx2", not(feature = "runtime_detection")))]
const _HAS_AVX2: bool = true;
#[cfg(not(all(feature = "avx2", not(feature = "runtime_detection"))))]
const _HAS_AVX2: bool = false;

#[cfg(all(feature = "avx", not(feature = "runtime_detection")))]
const _HAS_AVX: bool = true;
#[cfg(not(all(feature = "avx", not(feature = "runtime_detection"))))]
const _HAS_AVX: bool = false;

#[cfg(feature = "runtime_detection")]
static AVX2_DETECTED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "runtime_detection")]
static AVX_DETECTED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "runtime_detection")]
static DETECTION_DONE: AtomicBool = AtomicBool::new(false);

#[cfg(not(feature = "runtime_detection"))]
#[inline(always)]
pub fn has_avx2() -> bool {
    _HAS_AVX2
}

#[cfg(not(feature = "runtime_detection"))]
#[inline(always)]
pub fn has_avx() -> bool {
    _HAS_AVX
}

#[cfg(feature = "runtime_detection")]
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn detect_and_cache() {
    if DETECTION_DONE.load(Ordering::Relaxed) {
        return;
    }

    unsafe {
        use core::arch::x86_64::__cpuid;

        let cpuid1 = __cpuid(1);
        let has_avx = (cpuid1.ecx & (1 << 28)) != 0;

        let cpuid7 = __cpuid(7);
        let has_avx2 = (cpuid7.ebx & (1 << 5)) != 0;

        AVX_DETECTED.store(has_avx, Ordering::Relaxed);
        AVX2_DETECTED.store(has_avx2, Ordering::Relaxed);
        DETECTION_DONE.store(true, Ordering::Release);
    }
}

#[cfg(feature = "runtime_detection")]
#[inline(always)]
pub fn has_avx2() -> bool {
    if !DETECTION_DONE.load(Ordering::Acquire) {
        detect_and_cache();
    }
    AVX2_DETECTED.load(Ordering::Relaxed)
}

#[cfg(feature = "runtime_detection")]
#[inline(always)]
pub fn has_avx() -> bool {
    if !DETECTION_DONE.load(Ordering::Acquire) {
        detect_and_cache();
    }
    AVX_DETECTED.load(Ordering::Relaxed)
}

#[cfg(any(feature = "alloc", feature = "std"))]
pub type Result<T> = core::result::Result<T, Box<dyn Error>>;

#[cfg(all(not(feature = "alloc"), not(feature = "std")))]
pub type Result<T> = core::result::Result<T, &'static str>;

pub trait SimdLoadable {
    fn load_avx2(&self) -> __m256i;
    fn load_avx2_f32(&self) -> __m256;
    fn load_avx2_f64(&self) -> __m256d;
    fn load_avx(&self) -> __m128i;
    fn load_avx_f32(&self) -> __m128;
    fn load_avx_f64(&self) -> __m128d;
}

impl<T> SimdLoadable for [T] {
    #[inline(always)]
    fn load_avx2(&self) -> __m256i {
        load_avx2(self)
    }

    #[inline(always)]
    fn load_avx2_f32(&self) -> __m256 {
        load_avx2_f32(self)
    }

    #[inline(always)]
    fn load_avx2_f64(&self) -> __m256d {
        load_avx2_f64(self)
    }

    #[inline(always)]
    fn load_avx(&self) -> __m128i {
        load_avx(self)
    }

    #[inline(always)]
    fn load_avx_f32(&self) -> __m128 {
        load_avx_f32(self)
    }

    #[inline(always)]
    fn load_avx_f64(&self) -> __m128d {
        load_avx_f64(self)
    }
}

pub trait StoreAVXBitWise<T, const U: usize> {
    fn store_avx(&self) -> [T; U];
}

pub trait StoreAVX2BitWise<T, const U: usize> {
    fn store_avx2(&self) -> [T; U];
}

pub trait StoreAVXFloat32<T, const U: usize> {
    fn store_avx_f32(&self) -> [T; U];
}

pub trait StoreAVX2Float32<T, const U: usize> {
    fn store_avx2_f32(&self) -> [T; U];
}

pub trait StoreAVXFloat64<T, const U: usize> {
    fn store_avx_f64(&self) -> [T; U];
}

pub trait StoreAVX2Float64<T, const U: usize> {
    fn store_avx2_f64(&self) -> [T; U];
}

impl<T: Copy, const U: usize> StoreAVX2BitWise<T, U> for __m256i {
    fn store_avx2(&self) -> [T; U] {
        store_avx2(*self)
    }
}

impl<T: Copy, const U: usize> StoreAVXBitWise<T, U> for __m128i {
    fn store_avx(&self) -> [T; U] {
        store_avx(*self)
    }
}

impl<T: Copy, const U: usize> StoreAVX2Float32<T, U> for __m256 {
    fn store_avx2_f32(&self) -> [T; U] {
        store_avx2_f32(*self)
    }
}

impl<T: Copy, const U: usize> StoreAVXFloat32<T, U> for __m128 {
    fn store_avx_f32(&self) -> [T; U] {
        store_avx_f32(*self)
    }
}

impl<T: Copy, const U: usize> StoreAVX2Float64<T, U> for __m256d {
    fn store_avx2_f64(&self) -> [T; U] {
        store_avx2_f64(*self)
    }
}

impl<T: Copy, const U: usize> StoreAVXFloat64<T, U> for __m128d {
    fn store_avx_f64(&self) -> [T; U] {
        store_avx_f64(*self)
    }
}

/// A wrapper struct that ensures proper alignment for SIMD operations.
///
/// This struct automatically aligns data to 32-byte boundaries, which is optimal
/// for AVX2 operations. It provides transparent access to the underlying array
/// while guaranteeing SIMD-friendly memory layout.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignData<T, const U: usize>([T; U]);

impl<T, const U: usize> AlignData<T, U> {
    /// Creates a new aligned data structure from an array.
    ///
    /// # Example
    /// ```rust
    /// let data = AlignData::new([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    /// assert_eq!(data.as_ptr() as usize % 32, 0); // Guaranteed 32-byte alignment
    /// ```
    #[inline]
    pub const fn new(data: [T; U]) -> Self {
        Self(data)
    }

    /// Creates a new aligned data structure filled with the given value.
    ///
    /// # Example
    /// ```rust
    /// let data = AlignData::<f32, 8>::filled(1.0);
    /// ```
    #[inline]
    pub fn filled(value: T) -> Self
    where
        T: Copy,
    {
        Self([value; U])
    }

    /// Returns a reference to the inner array.
    #[inline]
    pub const fn as_array(&self) -> &[T; U] {
        &self.0
    }

    /// Returns a mutable reference to the inner array.
    #[inline]
    pub fn as_array_mut(&mut self) -> &mut [T; U] {
        &mut self.0
    }

    /// Returns a pointer to the data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    /// Returns a mutable pointer to the data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }

    /// Returns the length of the array.
    #[inline]
    pub const fn len(&self) -> usize {
        U
    }

    /// Returns true if the array is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        U == 0
    }

    /// Consumes the wrapper and returns the inner array.
    #[inline]
    pub fn into_inner(self) -> [T; U] {
        self.0
    }
}

impl<T, const U: usize> Deref for AlignData<T, U> {
    type Target = [T; U];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const U: usize> DerefMut for AlignData<T, U> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T, const U: usize> AsRef<[T]> for AlignData<T, U> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<T, const U: usize> AsMut<[T]> for AlignData<T, U> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T, const U: usize> From<[T; U]> for AlignData<T, U> {
    #[inline]
    fn from(data: [T; U]) -> Self {
        Self::new(data)
    }
}

impl<T, const U: usize> Into<[T; U]> for AlignData<T, U> {
    #[inline]
    fn into(self) -> [T; U] {
        self.into_inner()
    }
}

// Default implementation for types that support it
impl<T: Default + Copy, const U: usize> Default for AlignData<T, U> {
    #[inline]
    fn default() -> Self {
        Self([T::default(); U])
    }
}

// Convenient type aliases for common SIMD sizes
pub type AlignF32x8 = AlignData<f32, 8>;
pub type AlignF64x4 = AlignData<f64, 4>;
pub type AlignI32x8 = AlignData<i32, 8>;
pub type AlignI64x4 = AlignData<i64, 4>;
pub type AlignU32x8 = AlignData<u32, 8>;
pub type AlignU64x4 = AlignData<u64, 4>;
pub type AlignI16x16 = AlignData<i16, 16>;
pub type AlignU16x16 = AlignData<u16, 16>;
pub type AlignI8x32 = AlignData<i8, 32>;
pub type AlignU8x32 = AlignData<u8, 32>;

/// SIMD operations for integer types.
///
/// This trait provides vectorized arithmetic, bitwise, and comparison operations
/// for integer arrays. All operations are optimized using AVX/AVX2 instructions
/// when available, with automatic fallback to scalar operations.
///
/// # Type Parameters
/// - `T`: The integer element type (i8, i16, i32, i64, u8, u16, u32, u64)
/// - `U`: The array size (must match SIMD lane count for optimal performance)
///
/// # Supported Array Sizes
/// - 8-bit types: 32 elements (256-bit SIMD)
/// - 16-bit types: 16 elements (256-bit SIMD)
/// - 32-bit types: 8 elements (256-bit SIMD)
/// - 64-bit types: 4 elements (256-bit SIMD)
///
/// # Examples
/// ```rust
/// use simde::*;
///
/// let a = [1i32, 2, 3, 4, 5, 6, 7, 8];
/// let b = [8i32, 7, 6, 5, 4, 3, 2, 1];
///
/// // Basic arithmetic
/// let sum = a.add(&b)?;        // [9, 9, 9, 9, 9, 9, 9, 9]
/// let diff = a.sub(&b)?;       // [-7, -5, -3, -1, 1, 3, 5, 7]
/// let product = a.mul(&b)?;    // [8, 14, 18, 20, 20, 18, 14, 8]
///
/// // Bitwise operations
/// let and_result = a.and(&b)?; // Bitwise AND
/// let or_result = a.or(&b)?;   // Bitwise OR
/// let xor_result = a.xor(&b)?; // Bitwise XOR
///
/// // Min/Max operations
/// let minimum = a.min(&b)?;    // [1, 2, 3, 4, 4, 3, 2, 1]
/// let maximum = a.max(&b)?;    // [8, 7, 6, 5, 5, 6, 7, 8]
/// ```
pub trait SimdIntegerOps<T, const U: usize> {
    /// Performs element-wise addition of two integer arrays.
    ///
    /// # Arguments
    /// * `b` - The second operand array
    ///
    /// # Returns
    /// A `Result` containing the sum array or an error if SIMD is not supported
    ///
    /// # Examples
    /// ```rust
    /// let a = [1, 2, 3, 4];
    /// let b = [5, 6, 7, 8];
    /// let result = a.add(&b)?; // [6, 8, 10, 12]
    /// ```
    fn add(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs element-wise subtraction of two integer arrays.
    fn sub(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs element-wise multiplication of two integer arrays.
    ///
    /// Note: Not supported for 8-bit integers due to hardware limitations.
    fn mul(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs horizontal addition within 128-bit lanes.
    ///
    /// Adds adjacent pairs of elements within each 128-bit lane.
    /// Not supported for 8-bit integers or 64-bit integers.
    fn hadd(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs horizontal subtraction within 128-bit lanes.
    ///
    /// Subtracts adjacent pairs of elements within each 128-bit lane.
    /// Not supported for 8-bit integers or 64-bit integers.
    fn hsub(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs element-wise bitwise AND operation.
    fn and(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs element-wise bitwise OR operation.
    fn or(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs element-wise bitwise XOR operation.
    fn xor(&self, b: &[T]) -> Result<[T; U]>;

    /// Returns the element-wise minimum of two arrays.
    fn min(&self, b: &[T]) -> Result<[T; U]>;

    /// Returns the element-wise maximum of two arrays.
    fn max(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs modulo operation with a constant divisor.
    ///
    /// Only supported for signed 32-bit integers. For power-of-2 divisors,
    /// uses optimized bitwise AND operation.
    ///
    /// # Arguments
    /// * `b` - The divisor (must be positive for signed integers)
    fn modulo(&self, b: i32) -> Result<[T; U]>;
}

/// SIMD operations for floating-point types.
///
/// This trait provides comprehensive vectorized operations for floating-point arrays,
/// including arithmetic, comparison, blending, and transcendental functions.
///
/// # Type Parameters
/// - `T`: The floating-point element type (f32, f64)
/// - `U`: The array size (8 for f32, 4 for f64)
///
/// # Examples
/// ```rust
/// use simde::*;
///
/// let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let b = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
///
/// // Arithmetic operations
/// let sum = a.add(&b)?;
/// let quotient = a.div(&b)?;
/// let power = a.pow(&b)?;
///
/// // Comparison operations (returns mask)
/// let greater = a.gt(&b)?;     // [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] (all true)
/// let equal = a.eq(&b)?;       // [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (all false)
///
/// // Conditional blending
/// let mask = a.gt(&b)?;
/// let blended = a.blend(&b, &mask)?; // Select from a where mask is true, b otherwise
/// ```
pub trait SimdFloatOps<T, const U: usize> {
    /// Performs element-wise addition.
    fn add(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs element-wise subtraction.
    fn sub(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs element-wise multiplication.
    fn mul(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs element-wise division.
    fn div(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs horizontal addition within 128-bit lanes.
    fn hadd(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs horizontal subtraction within 128-bit lanes.
    fn hsub(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs element-wise bitwise AND (useful for masking).
    fn and(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs element-wise bitwise OR.
    fn or(&self, b: &[T]) -> Result<[T; U]>;

    /// Performs element-wise bitwise XOR.
    fn xor(&self, b: &[T]) -> Result<[T; U]>;

    /// Computes element-wise power operation (a^b).
    ///
    /// Currently only supported for f32 with AVX2.
    /// Uses the identity: a^b = exp(b * ln(a))
    fn pow(&self, b: &[T]) -> Result<[T; U]>;

    /// Element-wise equality comparison.
    ///
    /// Returns a mask where each element is all 1s (true) or all 0s (false).
    fn eq(&self, b: &[T]) -> Result<[T; U]>;

    /// Element-wise inequality comparison.
    fn not_eq(&self, b: &[T]) -> Result<[T; U]>;

    /// Element-wise less-than comparison.
    fn lt(&self, b: &[T]) -> Result<[T; U]>;

    /// Element-wise less-than-or-equal comparison.
    fn lte(&self, b: &[T]) -> Result<[T; U]>;

    /// Element-wise greater-than comparison.
    fn gt(&self, b: &[T]) -> Result<[T; U]>;

    /// Element-wise greater-than-or-equal comparison.
    fn gte(&self, b: &[T]) -> Result<[T; U]>;

    /// Conditional blending of two arrays based on a mask.
    ///
    /// For each element position, selects from `b` if the corresponding
    /// mask element has its high bit set, otherwise selects from `self`.
    ///
    /// # Arguments
    /// * `b` - The second source array
    /// * `c` - The mask array (typically result of comparison operation)
    fn blend(&self, b: &[T], c: &[T]) -> Result<[T; U]>;

    /// Performs floating-point modulo operation with a constant.
    ///
    /// Computes `self % b` for each element using the identity:
    /// `a % b = a - floor(a/b) * b`
    fn modulo(&self, b: T) -> Result<[T; U]>;
}

/// Single-operand SIMD operations.
///
/// This trait provides vectorized mathematical functions that operate on a single
/// input array, such as square root, trigonometric functions, and logarithms.
///
/// # Type Parameters
/// - `T`: The element type (integer or floating-point)
/// - `U`: The array size
///
/// # Examples
/// ```rust
/// use simde::*;
///
/// let values = [1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0];
///
/// // Mathematical functions
/// let roots = values.sqrt()?;      // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
/// let logs = values.ln()?;         // Natural logarithm
/// let exponentials = values.exp()?; // e^x
///
/// // Trigonometric functions
/// let angles = [0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.14159];
/// let sines = angles.sin()?;
/// let cosines = angles.cos()?;
/// let tangents = angles.tan()?;
///
/// // Rounding functions
/// let decimals = [1.2f32, 2.7, 3.1, 4.9, 5.5, 6.8, 7.3, 8.6];
/// let floors = decimals.floor()?;  // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
/// let ceils = decimals.ceil()?;    // [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
/// ```
pub trait SimdSingleOps<T, const U: usize> {
    /// Computes the absolute value of each element.
    ///
    /// For integers: `|x|`
    /// For floats: requires special implementation (not currently supported)
    fn abs(&self) -> Result<[T; U]>;

    /// Computes the square root of each element.
    ///
    /// Only supported for floating-point types.
    fn sqrt(&self) -> Result<[T; U]>;

    /// Computes the reciprocal square root (1/√x) of each element.
    ///
    /// Fast approximation, suitable for applications where precision
    /// can be traded for performance.
    fn rsqrt(&self) -> Result<[T; U]>;

    /// Rounds each element down to the nearest integer.
    fn floor(&self) -> Result<[T; U]>;

    /// Rounds each element up to the nearest integer.
    fn ceil(&self) -> Result<[T; U]>;

    /// Computes the reciprocal (1/x) of each element.
    ///
    /// Fast approximation with reduced precision.
    fn rcp(&self) -> Result<[T; U]>;

    /// Computes the natural logarithm (ln) of each element.
    ///
    /// Requires AVX2 support for f32.
    fn ln(&self) -> Result<[T; U]>;

    /// Computes the base-2 logarithm of each element.
    fn log2(&self) -> Result<[T; U]>;

    /// Computes the base-10 logarithm of each element.
    fn log10(&self) -> Result<[T; U]>;

    /// Computes e^x for each element.
    fn exp(&self) -> Result<[T; U]>;

    /// Computes the sine of each element (input in radians).
    fn sin(&self) -> Result<[T; U]>;

    /// Computes the tangent of each element (input in radians).
    fn tan(&self) -> Result<[T; U]>;

    /// Computes the cosine of each element (input in radians).
    fn cos(&self) -> Result<[T; U]>;

    /// Computes the arcsine of each element (result in radians).
    fn asin(&self) -> Result<[T; U]>;

    /// Computes the arctangent of each element (result in radians).
    fn atan(&self) -> Result<[T; U]>;

    /// Computes the arccosine of each element (result in radians).
    fn acos(&self) -> Result<[T; U]>;
}

/// Fused Multiply-Add (FMA) operations.
///
/// This trait provides high-precision fused multiply-add operations that compute
/// `(a * b) ± c` in a single instruction, avoiding intermediate rounding errors.
///
/// FMA operations are only available with AVX2 support and provide better
/// numerical accuracy and performance compared to separate multiply and add operations.
///
/// # Type Parameters
/// - `T`: The floating-point element type (f32, f64)
/// - `U`: The array size
///
/// # Examples
/// ```rust
/// use simde::*;
///
/// let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let b = [2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
/// let c = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
///
/// // Fused multiply-add: (a * b) + c
/// let result1 = a.fma_add(&b, &c)?;  // [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]
///
/// // Fused multiply-subtract: (a * b) - c
/// let result2 = a.fma_sub(&b, &c)?;  // [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
///
/// // Negated fused multiply-add: -(a * b) + c
/// let result3 = a.fnma_add(&b, &c)?; // [-1.0, -3.0, -5.0, -7.0, -9.0, -11.0, -13.0, -15.0]
///
/// // Negated fused multiply-subtract: -(a * b) - c
/// let result4 = a.fnma_sub(&b, &c)?; // [-3.0, -5.0, -7.0, -9.0, -11.0, -13.0, -15.0, -17.0]
/// ```
pub trait SimdFmaOps<T, const U: usize> {
    /// Fused multiply-add: `(self * b) + c`
    ///
    /// Computes the operation in a single instruction with no intermediate rounding.
    fn fma_add(&self, b: &[T], c: &[T]) -> Result<[T; U]>;

    /// Fused multiply-subtract: `(self * b) - c`
    fn fma_sub(&self, b: &[T], c: &[T]) -> Result<[T; U]>;

    /// Negated fused multiply-add: `-(self * b) + c`
    fn fnma_add(&self, b: &[T], c: &[T]) -> Result<[T; U]>;

    /// Negated fused multiply-subtract: `-(self * b) - c`
    fn fnma_sub(&self, b: &[T], c: &[T]) -> Result<[T; U]>;
}

/// Bit shift operations for integer types.
///
/// This trait provides vectorized bit shifting operations with compile-time
/// shift amounts for optimal performance.
///
/// # Type Parameters
/// - `T`: The integer element type (i16, i32, u16, u32)
/// - `U`: The array size
///
/// # Examples
/// ```rust
/// use simde::*;
///
/// let values = [1u32, 2, 4, 8, 16, 32, 64, 128];
///
/// // Left shift by 2 bits (multiply by 4)
/// let shifted_left = values.shift_left::<2>()?;  // [4, 8, 16, 32, 64, 128, 256, 512]
///
/// // Right shift by 1 bit (divide by 2)
/// let shifted_right = values.shift_right::<1>()?; // [0, 1, 2, 4, 8, 16, 32, 64]
/// ```
pub trait SimdShiftOps<T, const U: usize> {
    /// Performs left bit shift by a compile-time constant.
    ///
    /// # Type Parameters
    /// * `SHIFT` - The number of bits to shift (0-31 for 32-bit types, 0-15 for 16-bit types)
    ///
    /// Equivalent to multiplying by 2^SHIFT for each element.
    fn shift_left<const SHIFT: usize>(&self) -> Result<[T; U]>;

    /// Performs right bit shift by a compile-time constant.
    ///
    /// For signed types, performs arithmetic right shift (sign extension).
    /// For unsigned types, performs logical right shift (zero fill).
    ///
    /// Equivalent to dividing by 2^SHIFT for each element.
    fn shift_right<const SHIFT: usize>(&self) -> Result<[T; U]>;
}

impl SimdIntegerOps<u8, 32> for [u8] {
    #[inline(always)]
    fn add(&self, b: &[u8]) -> Result<[u8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::Add)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::Add)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn sub(&self, b: &[u8]) -> Result<[u8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::Sub)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::Sub)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn mul(&self, _: &[u8]) -> Result<[u8; 32]> {
        return Err("Mul not supported for 8-bit integers".into());
    }
    #[inline(always)]
    fn hadd(&self, _: &[u8]) -> Result<[u8; 32]> {
        return Err("HAdd not supported for 8-bit integers".into());
    }
    #[inline(always)]
    fn hsub(&self, _: &[u8]) -> Result<[u8; 32]> {
        return Err("HSub not supported for 8-bit integers".into());
    }
    #[inline(always)]
    fn and(&self, b: &[u8]) -> Result<[u8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::And)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::And)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn or(&self, b: &[u8]) -> Result<[u8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::Or)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::Or)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn xor(&self, b: &[u8]) -> Result<[u8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::Xor)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::Xor)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn max(&self, b: &[u8]) -> Result<[u8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::Max)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::Max)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn min(&self, b: &[u8]) -> Result<[u8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::Min)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::Min)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    fn modulo(&self, _: i32) -> Result<[u8; 32]> {
        return Err("Modulo not supported for unsigned integers".into());
    }
}

impl SimdIntegerOps<u16, 16> for [u16] {
    #[inline(always)]
    fn add(&self, b: &[u16]) -> Result<[u16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Add)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Add)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn sub(&self, b: &[u16]) -> Result<[u16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Sub)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Sub)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn mul(&self, b: &[u16]) -> Result<[u16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Mul)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Mul)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hadd(&self, b: &[u16]) -> Result<[u16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::HAdd)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::HAdd)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hsub(&self, b: &[u16]) -> Result<[u16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::HSub)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::HSub)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn and(&self, b: &[u16]) -> Result<[u16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::And)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::And)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn or(&self, b: &[u16]) -> Result<[u16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Or)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Or)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn xor(&self, b: &[u16]) -> Result<[u16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Xor)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Xor)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn max(&self, b: &[u16]) -> Result<[u16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Max)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Max)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn min(&self, b: &[u16]) -> Result<[u16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Min)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Min)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    fn modulo(&self, _: i32) -> Result<[u16; 16]> {
        return Err("Modulo not supported for unsigned integers".into());
    }
}

impl SimdIntegerOps<u32, 8> for [u32] {
    #[inline(always)]
    fn add(&self, b: &[u32]) -> Result<[u32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Add)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Add)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn sub(&self, b: &[u32]) -> Result<[u32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Sub)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Sub)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn mul(&self, b: &[u32]) -> Result<[u32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Mul)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Mul)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hadd(&self, b: &[u32]) -> Result<[u32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::HAdd)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::HAdd)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hsub(&self, b: &[u32]) -> Result<[u32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::HSub)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::HSub)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn and(&self, b: &[u32]) -> Result<[u32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::And)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::And)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn or(&self, b: &[u32]) -> Result<[u32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Or)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Or)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn xor(&self, b: &[u32]) -> Result<[u32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Xor)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Xor)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn max(&self, b: &[u32]) -> Result<[u32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Max)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Max)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn min(&self, b: &[u32]) -> Result<[u32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Min)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Min)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    fn modulo(&self, _: i32) -> Result<[u32; 8]> {
        return Err("Modulo not supported for unsigned integers".into());
    }
}

impl SimdIntegerOps<u64, 4> for [u64] {
    #[inline(always)]
    fn add(&self, b: &[u64]) -> Result<[u64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_64bit(a, b, avx::Operation::Add)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_64bit(a, b, avx::Operation::Add)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn sub(&self, b: &[u64]) -> Result<[u64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_64bit(a, b, avx::Operation::Sub)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_64bit(a, b, avx::Operation::Sub)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn mul(&self, b: &[u64]) -> Result<[u64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_64bit(a, b, avx::Operation::Mul)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_64bit(a, b, avx::Operation::Mul)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hadd(&self, _: &[u64]) -> Result<[u64; 4]> {
        return Err("HAdd not supported for 64-bit integers".into());
    }
    #[inline(always)]
    fn hsub(&self, _: &[u64]) -> Result<[u64; 4]> {
        return Err("HSub not supported for 64-bit integers".into());
    }
    #[inline(always)]
    fn and(&self, b: &[u64]) -> Result<[u64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_64bit(a, b, avx::Operation::And)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_64bit(a, b, avx::Operation::And)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn or(&self, b: &[u64]) -> Result<[u64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_64bit(a, b, avx::Operation::Or)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_64bit(a, b, avx::Operation::Or)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn xor(&self, b: &[u64]) -> Result<[u64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_64bit(a, b, avx::Operation::Xor)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_64bit(a, b, avx::Operation::Xor)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn max(&self, _: &[u64]) -> Result<[u64; 4]> {
        return Err("Max not supported for 64-bit integers".into());
    }
    #[inline(always)]
    fn min(&self, _: &[u64]) -> Result<[u64; 4]> {
        return Err("Min not supported for 64-bit integers".into());
    }
    fn modulo(&self, _: i32) -> Result<[u64; 4]> {
        return Err("Modulo not supported for unsigned integers".into());
    }
}

impl SimdIntegerOps<i8, 32> for [i8] {
    #[inline(always)]
    fn add(&self, b: &[i8]) -> Result<[i8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::Add)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::Add)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn sub(&self, b: &[i8]) -> Result<[i8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::Sub)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::Sub)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn mul(&self, _: &[i8]) -> Result<[i8; 32]> {
        return Err("Mul not supported for 8-bit integers".into());
    }
    #[inline(always)]
    fn hadd(&self, _: &[i8]) -> Result<[i8; 32]> {
        return Err("HAdd not supported for 8-bit integers".into());
    }
    #[inline(always)]
    fn hsub(&self, _: &[i8]) -> Result<[i8; 32]> {
        return Err("HSub not supported for 8-bit integers".into());
    }
    #[inline(always)]
    fn and(&self, b: &[i8]) -> Result<[i8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::And)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::And)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn or(&self, b: &[i8]) -> Result<[i8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::Or)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::Or)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn xor(&self, b: &[i8]) -> Result<[i8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::Xor)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::Xor)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn max(&self, b: &[i8]) -> Result<[i8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::Max)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::Max)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn min(&self, b: &[i8]) -> Result<[i8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_8bit(a, b, avx::Operation::Min)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_8bit(a, b, avx::Operation::Min)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    fn modulo(&self, _: i32) -> Result<[i8; 32]> {
        return Err("Modulo not supported for 8-bit integers".into());
    }
}

impl SimdIntegerOps<i16, 16> for [i16] {
    #[inline(always)]
    fn add(&self, b: &[i16]) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Add)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Add)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn sub(&self, b: &[i16]) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Sub)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Sub)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn mul(&self, b: &[i16]) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Mul)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Mul)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hadd(&self, b: &[i16]) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::HAdd)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::HAdd)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hsub(&self, b: &[i16]) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::HSub)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::HSub)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn and(&self, b: &[i16]) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::And)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::And)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn or(&self, b: &[i16]) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Or)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Or)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn xor(&self, b: &[i16]) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Xor)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Xor)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn max(&self, b: &[i16]) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Max)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Max)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn min(&self, b: &[i16]) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_16bit(a, b, avx::Operation::Min)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_16bit(a, b, avx::Operation::Min)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    fn modulo(&self, _: i32) -> Result<[i16; 16]> {
        return Err("Modulo not supported for 16-bit integers".into());
    }
}

impl SimdIntegerOps<i32, 8> for [i32] {
    #[inline(always)]
    fn add(&self, b: &[i32]) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Add)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Add)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn sub(&self, b: &[i32]) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Sub)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Sub)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn mul(&self, b: &[i32]) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Mul)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Mul)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hadd(&self, b: &[i32]) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::HAdd)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::HAdd)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hsub(&self, b: &[i32]) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::HSub)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::HSub)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn and(&self, b: &[i32]) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::And)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::And)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn or(&self, b: &[i32]) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Or)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Or)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn xor(&self, b: &[i32]) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Xor)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Xor)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn max(&self, b: &[i32]) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Max)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Max)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn min(&self, b: &[i32]) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_32bit(a, b, avx::Operation::Min)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_32bit(a, b, avx::Operation::Min)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    fn modulo(&self, b: i32) -> Result<[i32; 8]> {
        match has_avx2() {
            true => Ok(avx2_mod_constant_i32(self.load_avx2(), b)),
            false => return Err("Ln requires AVX2 support".into()),
        }
    }
}

impl SimdIntegerOps<i64, 4> for [i64] {
    #[inline(always)]
    fn add(&self, b: &[i64]) -> Result<[i64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_64bit(a, b, avx::Operation::Add)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_64bit(a, b, avx::Operation::Add)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn sub(&self, b: &[i64]) -> Result<[i64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_64bit(a, b, avx::Operation::Sub)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_64bit(a, b, avx::Operation::Sub)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn mul(&self, b: &[i64]) -> Result<[i64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_64bit(a, b, avx::Operation::Mul)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_64bit(a, b, avx::Operation::Mul)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hadd(&self, _: &[i64]) -> Result<[i64; 4]> {
        return Err("HAdd not supported for 64-bit integers".into());
    }
    #[inline(always)]
    fn hsub(&self, _: &[i64]) -> Result<[i64; 4]> {
        return Err("HSub not supported for 64-bit integers".into());
    }
    #[inline(always)]
    fn and(&self, b: &[i64]) -> Result<[i64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_64bit(a, b, avx::Operation::And)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_64bit(a, b, avx::Operation::And)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn or(&self, b: &[i64]) -> Result<[i64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_64bit(a, b, avx::Operation::Or)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_64bit(a, b, avx::Operation::Or)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn xor(&self, b: &[i64]) -> Result<[i64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                let b = load_avx2(b);
                avx2_64bit(a, b, avx::Operation::Xor)
            }
            (true, _) => {
                let a = load_avx(self);
                let b = load_avx(b);
                avx_64bit(a, b, avx::Operation::Xor)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn max(&self, _: &[i64]) -> Result<[i64; 4]> {
        return Err("Max not supported for 64-bit integers".into());
    }
    #[inline(always)]
    fn min(&self, _: &[i64]) -> Result<[i64; 4]> {
        return Err("Min not supported for 64-bit integers".into());
    }
    fn modulo(&self, _: i32) -> Result<[i64; 4]> {
        return Err("Modulo not supported for 64-bit integers".into());
    }
}

impl SimdFloatOps<f32, 8> for [f32] {
    #[inline(always)]
    fn add(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_f32(a, b, avx::FloatOperation::AddFloat32)
            }
            (true, _) => {
                let a = load_avx_f32(self);
                let b = load_avx_f32(b);
                avx_f32(a, b, avx::FloatOperation::AddFloat32)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn sub(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_f32(a, b, avx::FloatOperation::SubFloat32)
            }
            (true, _) => {
                let a = load_avx_f32(self);
                let b = load_avx_f32(b);
                avx_f32(a, b, avx::FloatOperation::SubFloat32)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn mul(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_f32(a, b, avx::FloatOperation::MulFloat32)
            }
            (true, _) => {
                let a = load_avx_f32(self);
                let b = load_avx_f32(b);
                avx_f32(a, b, avx::FloatOperation::MulFloat32)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn div(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_f32(a, b, avx::FloatOperation::DivFloat32)
            }
            (true, _) => {
                let a = load_avx_f32(self);
                let b = load_avx_f32(b);
                avx_f32(a, b, avx::FloatOperation::DivFloat32)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hadd(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_f32(a, b, avx::FloatOperation::HAddFloat32)
            }
            (true, _) => {
                let a = load_avx_f32(self);
                let b = load_avx_f32(b);
                avx_f32(a, b, avx::FloatOperation::HAddFloat32)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hsub(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_f32(a, b, avx::FloatOperation::HSubFloat32)
            }
            (true, _) => {
                let a = load_avx_f32(self);
                let b = load_avx_f32(b);
                avx_f32(a, b, avx::FloatOperation::HSubFloat32)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn and(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_f32(a, b, avx::FloatOperation::AndFloat32)
            }
            (true, _) => {
                let a = load_avx_f32(self);
                let b = load_avx_f32(b);
                avx_f32(a, b, avx::FloatOperation::AndFloat32)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn or(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_f32(a, b, avx::FloatOperation::OrFloat32)
            }
            (true, _) => {
                let a = load_avx_f32(self);
                let b = load_avx_f32(b);
                avx_f32(a, b, avx::FloatOperation::OrFloat32)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn xor(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_f32(a, b, avx::FloatOperation::XorFloat32)
            }
            (true, _) => {
                let a = load_avx_f32(self);
                let b = load_avx_f32(b);
                avx_f32(a, b, avx::FloatOperation::XorFloat32)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn pow(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_pow_f32(a, b)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn eq(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_compare_f32(a, b, avx::CompareOperation::Equal)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn not_eq(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_compare_f32(a, b, avx::CompareOperation::NotEqual)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn gt(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_compare_f32(a, b, avx::CompareOperation::Greater)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn gte(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_compare_f32(a, b, avx::CompareOperation::GreaterEqual)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn lt(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_compare_f32(a, b, avx::CompareOperation::Less)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn lte(&self, b: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                avx2_compare_f32(a, b, avx::CompareOperation::LessEqual)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn blend(&self, b: &[f32], c: &[f32]) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                let c = load_avx2_f32(c);
                avx2_blend_f32(a, b, c)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    fn modulo(&self, b: f32) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => Ok(avx2_fmod_constant_f32(self.load_avx2_f32(), b)),
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
}

impl SimdFloatOps<f64, 4> for [f64] {
    #[inline(always)]
    fn add(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_f64(a, b, avx::FloatOperation::AddFloat64)
            }
            (true, _) => {
                let a = load_avx_f64(self);
                let b = load_avx_f64(b);
                avx_f64(a, b, avx::FloatOperation::AddFloat64)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn sub(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_f64(a, b, avx::FloatOperation::SubFloat64)
            }
            (true, _) => {
                let a = load_avx_f64(self);
                let b = load_avx_f64(b);
                avx_f64(a, b, avx::FloatOperation::SubFloat64)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn mul(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_f64(a, b, avx::FloatOperation::MulFloat64)
            }
            (true, _) => {
                let a = load_avx_f64(self);
                let b = load_avx_f64(b);
                avx_f64(a, b, avx::FloatOperation::MulFloat64)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn div(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_f64(a, b, avx::FloatOperation::DivFloat64)
            }
            (true, _) => {
                let a = load_avx_f64(self);
                let b = load_avx_f64(b);
                avx_f64(a, b, avx::FloatOperation::DivFloat64)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hadd(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_f64(a, b, avx::FloatOperation::HAddFloat64)
            }
            (true, _) => {
                let a = load_avx_f64(self);
                let b = load_avx_f64(b);
                avx_f64(a, b, avx::FloatOperation::HAddFloat64)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn hsub(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_f64(a, b, avx::FloatOperation::HSubFloat64)
            }
            (true, _) => {
                let a = load_avx_f64(self);
                let b = load_avx_f64(b);
                avx_f64(a, b, avx::FloatOperation::HSubFloat64)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn and(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_f64(a, b, avx::FloatOperation::AndFloat64)
            }
            (true, _) => {
                let a = load_avx_f64(self);
                let b = load_avx_f64(b);
                avx_f64(a, b, avx::FloatOperation::AndFloat64)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn or(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_f64(a, b, avx::FloatOperation::OrFloat64)
            }
            (true, _) => {
                let a = load_avx_f64(self);
                let b = load_avx_f64(b);
                avx_f64(a, b, avx::FloatOperation::OrFloat64)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn xor(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_f64(a, b, avx::FloatOperation::XorFloat64)
            }
            (true, _) => {
                let a = load_avx_f64(self);
                let b = load_avx_f64(b);
                avx_f64(a, b, avx::FloatOperation::XorFloat64)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn pow(&self, _: &[f64]) -> Result<[f64; 4]> {
        return Err("Pow not supported for f64 in current implementation".into());
    }
    #[inline(always)]
    fn eq(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_compare_f64(a, b, avx::CompareOperation::Equal)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn not_eq(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_compare_f64(a, b, avx::CompareOperation::NotEqual)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn gt(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_compare_f64(a, b, avx::CompareOperation::Greater)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn gte(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_compare_f64(a, b, avx::CompareOperation::GreaterEqual)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn lt(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_compare_f64(a, b, avx::CompareOperation::Less)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn lte(&self, b: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                avx2_compare_f64(a, b, avx::CompareOperation::LessEqual)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn blend(&self, b: &[f64], c: &[f64]) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                let c = load_avx2_f64(c);
                avx2_blend_f64(a, b, c)
            }
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    fn modulo(&self, b: f64) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => Ok(avx2_fmod_constant_f64(self.load_avx2_f64(), b)),
            (true, _) => return Err("AVX2 not supported on this instruction".into()),
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
}

impl SimdSingleOps<i8, 32> for [i8] {
    #[inline(always)]
    fn abs(&self) -> Result<[i8; 32]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                avx2_abs_8bit(a)
            }
            (true, _) => {
                let a = load_avx(self);
                avx_abs_8bit(a)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    fn sqrt(&self) -> Result<[i8; 32]> {
        return Err("Sqrt not supported for integer types".into());
    }
    fn floor(&self) -> Result<[i8; 32]> {
        return Err("Floor not supported for integer types".into());
    }
    fn ln(&self) -> Result<[i8; 32]> {
        return Err("Ln not supported for integer types".into());
    }
    fn log2(&self) -> Result<[i8; 32]> {
        return Err("Log2 not supported for integer types".into());
    }
    fn log10(&self) -> Result<[i8; 32]> {
        return Err("Log10 not supported for integer types".into());
    }
    fn exp(&self) -> Result<[i8; 32]> {
        return Err("Exp not supported for integer types".into());
    }
    fn cos(&self) -> Result<[i8; 32]> {
        return Err("Cos not supported for integer types".into());
    }
    fn sin(&self) -> Result<[i8; 32]> {
        return Err("Sin not supported for integer types".into());
    }
    fn tan(&self) -> Result<[i8; 32]> {
        return Err("Tan not supported for integer types".into());
    }
    fn acos(&self) -> Result<[i8; 32]> {
        return Err("ACos not supported for integer types".into());
    }
    fn asin(&self) -> Result<[i8; 32]> {
        return Err("ASin not supported for integer types".into());
    }
    fn atan(&self) -> Result<[i8; 32]> {
        return Err("ATan not supported for integer types".into());
    }
    fn ceil(&self) -> Result<[i8; 32]> {
        return Err("Ceil not supported for integer types".into());
    }
    fn rcp(&self) -> Result<[i8; 32]> {
        return Err("Rcp not supported for integer types".into());
    }
    fn rsqrt(&self) -> Result<[i8; 32]> {
        return Err("Rsqrt not supported for integer types".into());
    }
}

impl SimdSingleOps<i16, 16> for [i16] {
    #[inline(always)]
    fn abs(&self) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                avx2_abs_16bit(a)
            }
            (true, _) => {
                let a = load_avx(self);
                avx_abs_16bit(a)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    fn sqrt(&self) -> Result<[i16; 16]> {
        return Err("Sqrt not supported for integer types".into());
    }
    fn floor(&self) -> Result<[i16; 16]> {
        return Err("Floor not supported for integer types".into());
    }
    fn ln(&self) -> Result<[i16; 16]> {
        return Err("Ln not supported for integer types".into());
    }
    fn log2(&self) -> Result<[i16; 16]> {
        return Err("Log2 not supported for integer types".into());
    }
    fn log10(&self) -> Result<[i16; 16]> {
        return Err("Log10 not supported for integer types".into());
    }
    fn exp(&self) -> Result<[i16; 16]> {
        return Err("Exp not supported for integer types".into());
    }
    fn cos(&self) -> Result<[i16; 16]> {
        return Err("Cos not supported for integer types".into());
    }
    fn sin(&self) -> Result<[i16; 16]> {
        return Err("Sin not supported for integer types".into());
    }
    fn tan(&self) -> Result<[i16; 16]> {
        return Err("Tan not supported for integer types".into());
    }
    fn acos(&self) -> Result<[i16; 16]> {
        return Err("ACos not supported for integer types".into());
    }
    fn asin(&self) -> Result<[i16; 16]> {
        return Err("ASin not supported for integer types".into());
    }
    fn atan(&self) -> Result<[i16; 16]> {
        return Err("ATan not supported for integer types".into());
    }
    fn ceil(&self) -> Result<[i16; 16]> {
        return Err("Ceil not supported for integer types".into());
    }
    fn rcp(&self) -> Result<[i16; 16]> {
        return Err("Rcp not supported for integer types".into());
    }
    fn rsqrt(&self) -> Result<[i16; 16]> {
        return Err("Rsqrt not supported for integer types".into());
    }
}

impl SimdSingleOps<i32, 8> for [i32] {
    #[inline(always)]
    fn abs(&self) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                avx2_abs_32bit(a)
            }
            (true, _) => {
                let a = load_avx(self);
                avx_abs_32bit(a)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    fn sqrt(&self) -> Result<[i32; 8]> {
        return Err("Sqrt not supported for integer types".into());
    }
    fn floor(&self) -> Result<[i32; 8]> {
        return Err("Floor not supported for integer types".into());
    }
    fn ln(&self) -> Result<[i32; 8]> {
        return Err("Ln not supported for integer types".into());
    }
    fn log2(&self) -> Result<[i32; 8]> {
        return Err("Log2 not supported for integer types".into());
    }
    fn log10(&self) -> Result<[i32; 8]> {
        return Err("Log10 not supported for integer types".into());
    }
    fn exp(&self) -> Result<[i32; 8]> {
        return Err("Exp not supported for integer types".into());
    }
    fn cos(&self) -> Result<[i32; 8]> {
        return Err("Cos not supported for integer types".into());
    }
    fn sin(&self) -> Result<[i32; 8]> {
        return Err("Sin not supported for integer types".into());
    }
    fn tan(&self) -> Result<[i32; 8]> {
        return Err("Tan not supported for integer types".into());
    }
    fn acos(&self) -> Result<[i32; 8]> {
        return Err("ACos not supported for integer types".into());
    }
    fn asin(&self) -> Result<[i32; 8]> {
        return Err("ASin not supported for integer types".into());
    }
    fn atan(&self) -> Result<[i32; 8]> {
        return Err("ATan not supported for integer types".into());
    }
    fn ceil(&self) -> Result<[i32; 8]> {
        return Err("Ceil not supported for integer types".into());
    }
    fn rcp(&self) -> Result<[i32; 8]> {
        return Err("Rcp not supported for integer types".into());
    }
    fn rsqrt(&self) -> Result<[i32; 8]> {
        return Err("Rsqrt not supported for integer types".into());
    }
}

impl SimdSingleOps<f32, 8> for [f32] {
    #[inline(always)]
    fn abs(&self) -> Result<[f32; 8]> {
        return Err("Abs for f32 requires special implementation".into());
    }
    #[inline(always)]
    fn sqrt(&self) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::Sqrt)
            }
            (true, _) => {
                let a = load_avx_f32(self);
                avx_single_f32(a, avx::SingleOperations::Sqrt)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn floor(&self) -> Result<[f32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::Floor)
            }
            (true, _) => {
                let a = load_avx_f32(self);
                avx_single_f32(a, avx::SingleOperations::Floor)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn ln(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::Ln)
            }
            false => return Err("Ln requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn log2(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::Log2)
            }
            false => return Err("Log2 requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn log10(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::Log10)
            }
            false => return Err("Log10 requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn exp(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::Exp)
            }
            false => return Err("Exp requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn cos(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::Cos)
            }
            false => return Err("Cos requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn sin(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::Sin)
            }
            false => return Err("Sin requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn tan(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::Tan)
            }
            false => return Err("Tan requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn acos(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::ACos)
            }
            false => return Err("ACos requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn asin(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::ASin)
            }
            false => return Err("ASin requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn atan(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::ATan)
            }
            false => return Err("ATan requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn ceil(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::Ceil)
            }
            false => return Err("Ceil requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn rcp(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::Rcp)
            }
            false => return Err("Rcp requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn rsqrt(&self) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                avx2_single_f32(a, avx::SingleOperations::RSqrt)
            }
            false => return Err("Rsqrt requires AVX2 support".into()),
        }
    }
}

impl SimdSingleOps<f64, 4> for [f64] {
    fn abs(&self) -> Result<[f64; 4]> {
        return Err("Abs for f64 requires special implementation".into());
    }
    #[inline(always)]
    fn sqrt(&self) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                avx2_single_f64(a, avx::SingleOperations::Sqrt)
            }
            (true, _) => {
                let a = load_avx_f64(self);
                avx_single_f64(a, avx::SingleOperations::Sqrt)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn floor(&self) -> Result<[f64; 4]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2_f64(self);
                avx2_single_f64(a, avx::SingleOperations::Floor)
            }
            (true, _) => {
                let a = load_avx_f64(self);
                avx_single_f64(a, avx::SingleOperations::Floor)
            }
            _ => return Err("SIMD not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn ceil(&self) -> Result<[f64; 4]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f64(self);
                avx2_single_f64(a, avx::SingleOperations::Ceil)
            }
            false => return Err("Ceil requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn rcp(&self) -> Result<[f64; 4]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f64(self);
                avx2_single_f64(a, avx::SingleOperations::Rcp)
            }
            false => return Err("Rcp requires AVX2 support".into()),
        }
    }
    fn rsqrt(&self) -> Result<[f64; 4]> {
        return Err("rsqrt not supported for f64 in current implementation".into());
    }
    fn ln(&self) -> Result<[f64; 4]> {
        return Err("Ln not supported for f64 in current implementation".into());
    }
    fn log2(&self) -> Result<[f64; 4]> {
        return Err("Log2 not supported for f64 in current implementation".into());
    }
    fn log10(&self) -> Result<[f64; 4]> {
        return Err("Log10 not supported for f64 in current implementation".into());
    }
    fn exp(&self) -> Result<[f64; 4]> {
        return Err("Exp not supported for f64 in current implementation".into());
    }
    fn cos(&self) -> Result<[f64; 4]> {
        return Err("Cos not supported for f64 in current implementation".into());
    }
    fn sin(&self) -> Result<[f64; 4]> {
        return Err("Sin not supported for f64 in current implementation".into());
    }
    fn tan(&self) -> Result<[f64; 4]> {
        return Err("Tan not supported for f64 in current implementation".into());
    }
    fn acos(&self) -> Result<[f64; 4]> {
        return Err("ACos not supported for f64 in current implementation".into());
    }
    fn asin(&self) -> Result<[f64; 4]> {
        return Err("ASin not supported for f64 in current implementation".into());
    }
    fn atan(&self) -> Result<[f64; 4]> {
        return Err("ATan not supported for f64 in current implementation".into());
    }
}

impl SimdFmaOps<f32, 8> for [f32] {
    #[inline(always)]
    fn fma_add(&self, b: &[f32], c: &[f32]) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                let c = load_avx2_f32(c);
                fma_f32_avx2(a, b, c, avx::FmaOperation::FmaAdd)
            }
            false => return Err("FMA requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn fma_sub(&self, b: &[f32], c: &[f32]) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                let c = load_avx2_f32(c);
                fma_f32_avx2(a, b, c, avx::FmaOperation::FmaSub)
            }
            false => return Err("FMA requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn fnma_add(&self, b: &[f32], c: &[f32]) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                let c = load_avx2_f32(c);
                fma_f32_avx2(a, b, c, avx::FmaOperation::FnmaAdd)
            }
            false => return Err("FMA requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn fnma_sub(&self, b: &[f32], c: &[f32]) -> Result<[f32; 8]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f32(self);
                let b = load_avx2_f32(b);
                let c = load_avx2_f32(c);
                fma_f32_avx2(a, b, c, avx::FmaOperation::FnmaSub)
            }
            false => return Err("FMA requires AVX2 support".into()),
        }
    }
}

impl SimdFmaOps<f64, 4> for [f64] {
    #[inline(always)]
    fn fma_add(&self, b: &[f64], c: &[f64]) -> Result<[f64; 4]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                let c = load_avx2_f64(c);
                fma_f64_avx2(a, b, c, avx::FmaOperation::FmaAdd)
            }
            false => return Err("FMA requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn fma_sub(&self, b: &[f64], c: &[f64]) -> Result<[f64; 4]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                let c = load_avx2_f64(c);
                fma_f64_avx2(a, b, c, avx::FmaOperation::FmaSub)
            }
            false => return Err("FMA requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn fnma_add(&self, b: &[f64], c: &[f64]) -> Result<[f64; 4]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                let c = load_avx2_f64(c);
                fma_f64_avx2(a, b, c, avx::FmaOperation::FnmaAdd)
            }
            false => return Err("FMA requires AVX2 support".into()),
        }
    }
    #[inline(always)]
    fn fnma_sub(&self, b: &[f64], c: &[f64]) -> Result<[f64; 4]> {
        match has_avx2() {
            true => {
                let a = load_avx2_f64(self);
                let b = load_avx2_f64(b);
                let c = load_avx2_f64(c);
                fma_f64_avx2(a, b, c, avx::FmaOperation::FnmaSub)
            }
            false => return Err("FMA requires AVX2 support".into()),
        }
    }
}

impl SimdShiftOps<i16, 16> for [i16] {
    #[inline(always)]
    fn shift_left<const SHIFT: usize>(&self) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                avx2_shift_left_16bit::<i16, 16, SHIFT>(a)
            }
            _ => return Err("AVX2 not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn shift_right<const SHIFT: usize>(&self) -> Result<[i16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                avx2_shift_right_16bit::<i16, 16, SHIFT>(a)
            }
            _ => return Err("AVX2 not supported on this device".into()),
        }
    }
}

impl SimdShiftOps<i32, 8> for [i32] {
    #[inline(always)]
    fn shift_left<const SHIFT: usize>(&self) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                avx2_shift_left_32bit::<i32, 8, SHIFT>(a)
            }
            _ => return Err("AVX2 not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn shift_right<const SHIFT: usize>(&self) -> Result<[i32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                avx2_shift_right_32bit::<i32, 8, SHIFT>(a)
            }
            _ => return Err("AVX2 not supported on this device".into()),
        }
    }
}

impl SimdShiftOps<u16, 16> for [u16] {
    #[inline(always)]
    fn shift_left<const SHIFT: usize>(&self) -> Result<[u16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                avx2_shift_left_16bit::<u16, 16, SHIFT>(a)
            }
            _ => return Err("AVX2 not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn shift_right<const SHIFT: usize>(&self) -> Result<[u16; 16]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                avx2_shift_right_16bit::<u16, 16, SHIFT>(a)
            }
            _ => return Err("AVX2 not supported on this device".into()),
        }
    }
}

impl SimdShiftOps<u32, 8> for [u32] {
    #[inline(always)]
    fn shift_left<const SHIFT: usize>(&self) -> Result<[u32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                avx2_shift_left_32bit::<u32, 8, SHIFT>(a)
            }
            _ => return Err("AVX2 not supported on this device".into()),
        }
    }
    #[inline(always)]
    fn shift_right<const SHIFT: usize>(&self) -> Result<[u32; 8]> {
        match (has_avx(), has_avx2()) {
            (_, true) => {
                let a = load_avx2(self);
                avx2_shift_right_32bit::<u32, 8, SHIFT>(a)
            }
            _ => return Err("AVX2 not supported on this device".into()),
        }
    }
}

/// Array reversal operations using SIMD instructions.
///
/// This trait provides vectorized array reversal that can be significantly
/// faster than scalar implementations for supported types and sizes.
///
/// # Type Parameters
/// - `T`: The element type (integers and floats)
/// - `U`: The array size (must match SIMD lane count)
///
/// # Examples
/// ```rust
/// use simde::*;
///
/// let forward = [1i32, 2, 3, 4, 5, 6, 7, 8];
/// let backward = forward.reverse_simd()?; // [8, 7, 6, 5, 4, 3, 2, 1]
///
/// let floats = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let reversed = floats.reverse_simd()?;  // [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
/// ```
pub trait SimdReverseOps<T, const U: usize> {
    /// Reverses the order of elements in the array using SIMD instructions.
    ///
    /// This operation uses specialized shuffle instructions to reverse the array
    /// in-place much faster than a scalar loop.
    ///
    /// Requires AVX2 support.
    fn reverse_simd(&self) -> Result<[T; U]>;
}

impl SimdReverseOps<i32, 8> for [i32] {
    #[inline(always)]
    fn reverse_simd(&self) -> Result<[i32; 8]> {
        if !has_avx2() {
            return Err("AVX2 not supported".into());
        }

        unsafe {
            let loaded = load_avx2(self);
            let reversed = avx2_reverse_32bit(loaded);

            let mut result = [0i32; 8];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, reversed);
            Ok(result)
        }
    }
}
impl SimdReverseOps<i16, 16> for [i16] {
    #[inline(always)]
    fn reverse_simd(&self) -> Result<[i16; 16]> {
        if !has_avx2() {
            return Err("AVX2 not supported".into());
        }

        unsafe {
            let loaded = load_avx2(self);
            let reversed = avx2_reverse_16bit(loaded);

            let mut result = [0i16; 16];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, reversed);
            Ok(result)
        }
    }
}

impl SimdReverseOps<i8, 32> for [i8] {
    #[inline(always)]
    fn reverse_simd(&self) -> Result<[i8; 32]> {
        if !has_avx2() {
            return Err("AVX2 not supported".into());
        }

        unsafe {
            let loaded = load_avx2(self);
            let reversed = avx2_reverse_8bit(loaded);

            let mut result = [0i8; 32];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, reversed);
            Ok(result)
        }
    }
}

impl SimdReverseOps<u32, 8> for [u32] {
    #[inline(always)]
    fn reverse_simd(&self) -> Result<[u32; 8]> {
        if !has_avx2() {
            return Err("AVX2 not supported".into());
        }

        unsafe {
            let loaded = load_avx2(self);
            let reversed = avx2_reverse_32bit(loaded);

            let mut result = [0u32; 8];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, reversed);
            Ok(result)
        }
    }
}
impl SimdReverseOps<u16, 16> for [u16] {
    #[inline(always)]
    fn reverse_simd(&self) -> Result<[u16; 16]> {
        if !has_avx2() {
            return Err("AVX2 not supported".into());
        }

        unsafe {
            let loaded = load_avx2(self);
            let reversed = avx2_reverse_16bit(loaded);

            let mut result = [0u16; 16];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, reversed);
            Ok(result)
        }
    }
}

impl SimdReverseOps<u8, 32> for [u8] {
    #[inline(always)]
    fn reverse_simd(&self) -> Result<[u8; 32]> {
        if !has_avx2() {
            return Err("AVX2 not supported".into());
        }

        unsafe {
            let loaded = load_avx2(self);
            let reversed = avx2_reverse_8bit(loaded);

            let mut result = [0u8; 32];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, reversed);
            Ok(result)
        }
    }
}

impl SimdReverseOps<f64, 4> for [f64] {
    #[inline(always)]
    fn reverse_simd(&self) -> Result<[f64; 4]> {
        if !has_avx2() {
            return Err("AVX2 not supported".into());
        }

        unsafe {
            let loaded = load_avx2_f64(self);
            let reversed = avx2_reverse_f64(loaded);

            let mut result = [0.0f64; 4];
            _mm256_storeu_pd(result.as_mut_ptr(), reversed);
            Ok(result)
        }
    }
}

impl SimdReverseOps<f32, 8> for [f32] {
    #[inline(always)]
    fn reverse_simd(&self) -> Result<[f32; 8]> {
        if !has_avx2() {
            return Err("AVX2 not supported".into());
        }

        unsafe {
            let loaded = load_avx2_f32(self);
            let reversed = avx2_reverse_f32(loaded);

            let mut result = [0.0f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), reversed);
            Ok(result)
        }
    }
}
