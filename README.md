# SIMDE - High-Performance SIMD Library for Rust

**SIMDE** is a comprehensive, zero-cost abstraction SIMD library for Rust that provides ergonomic wrappers around Intel AVX/AVX2 intrinsics. Built for maximum performance while maintaining type safety and ease of use.

## Features

**Zero-Cost Abstractions** - All functions are inlined and optimized away at compile time
**Type-Safe API** - Generic implementations with compile-time guarantees
**Runtime Detection** - Automatic CPU feature detection with graceful fallbacks
**Complete Coverage** - Supports all major SIMD operations and data types
**Multiple APIs** - Both trait-based and function-based interfaces
**Lookup Tables** - Optimized SIMD table operations with prefetching
**Aligned Data Types** - Built-in memory alignment utilities for optimal SIMD performance

## Supported Operations

### Integer Operations (8-bit, 16-bit, 32-bit, 64-bit)
- **Arithmetic**: Add, Subtract, Multiply, Horizontal Add/Sub
- **Bitwise**: AND, OR, XOR
- **Comparison**: Min, Max
- **Unary**: Absolute value, Bit shifts

### Float Operations (f32, f64)
- **Arithmetic**: Add, Subtract, Multiply, Divide, Horizontal Add/Sub
- **Advanced Math**: Square root, Reciprocal, Power, Logarithms
- **Trigonometric**: Sin, Cos, Tan, ASin, ACos, ATan
- **Rounding**: Floor, Ceil
- **Comparison**: Equal, Not Equal, Less/Greater Than
- **FMA**: Fused Multiply-Add operations
- **Blending**: Conditional selection operations

### Lookup Table Operations
- **Fast Gather**: SIMD-accelerated table lookups
- **Prefetching**: Automatic memory prefetching for large tables
- **Multiple Types**: i32, f32, f64 lookup tables

### Memory Alignment
- **Automatic Alignment**: Built-in `AlignData<T, N>` for guaranteed SIMD alignment
- **Zero-Cost Wrapper**: Transparent access with `Deref` traits
- **Type Aliases**: Convenient aliases for common SIMD sizes

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
simde = "0.1"
```

### Basic Usage

```rust
use simde::prelude::*;

// Trait-based API (convenient)
let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let b = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

let result = a.add(&b)?; // SIMD addition
let sqrt_result = a.sqrt()?; // SIMD square root
let fma_result = a.fma_add(&b, &c)?; // Fused multiply-add

// Function-based API (maximum performance)
use simde::{load_avx2_f32, avx2_f32, FloatOperation};

let va = load_avx2_f32(&a);
let vb = load_avx2_f32(&b);
let result = avx2_f32(va, vb, FloatOperation::AddFloat32)?;
```

### Advanced Math Operations

```rust
let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

// Trigonometric functions
let sin_result = data.sin()?;
let cos_result = data.cos()?;
let tan_result = data.tan()?;

// Logarithmic functions
let ln_result = data.ln()?;
let log2_result = data.log2()?;
let exp_result = data.exp()?;

// Power operations
let power_result = data.pow(&[2.0; 8])?;
```

### Lookup Table Operations

```rust
use simde::lookup_tables::SimdTableF32;

// Create an optimized lookup table
let table_data = (0..1024).map(|i| (i as f32).sin()).collect::<Vec<_>>();
let lookup_table = SimdTableF32::new(table_data.try_into().unwrap());

// Preload for better cache performance
lookup_table.preload()?;

// Fast SIMD lookups
let indices = [0i32, 10, 20, 30, 40, 50, 60, 70];
let results = lookup_table.read_offsets(&indices);
```

### Aligned Data Types

```rust
use simde::{AlignData, AlignF32x8, AlignI32x8};

// Automatic 32-byte alignment for optimal SIMD performance
let data = AlignData::new([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

// Use convenient type aliases
let aligned_floats = AlignF32x8::filled(1.0);
let aligned_ints = AlignI32x8::filled(42);

// Transparent access - works like a normal array
let result = aligned_floats.add(&aligned_ints.into())?;

// Guaranteed alignment for maximum performance
assert_eq!(aligned_floats.as_ptr() as usize % 32, 0);
```

## Performance

SIMDE provides significant performance improvements over scalar operations:

- **2-20x speedup** for basic operations depending on data size
- **100GB/s+ memory throughput** on modern CPUs
- **Zero overhead** function calls with inline optimization
- **Automatic vectorization** with optimal instruction selection

### Benchmark Results

```
Operation           Scalar Time    SIMD Time     Speedup
f32 Addition (8)    36 cycles     36 cycles     1.00x
f32 Addition (256)  677 cycles    36 cycles     18.83x
u8 Addition (32)    88 cycles     36 cycles     2.08x
u8 Addition (256)    244 cycles     36 cycles   6.77x
```

## Feature Flags

SIMDE provides flexible configuration through Cargo features:

```toml
[features]
default = ["std", "fast_math", "runtime_detection"]
fast_math = []        # Fast but less precise math operations
precise_math = []     # Slower but more precise math operations
alloc = []           # Enable heap allocation support
std = []             # Standard library support
avx2 = ["avx"]       # Enable AVX2 instructions (implies AVX)
avx = []             # Enable AVX instructions
runtime_detection = [] # Automatic CPU feature detection
no_main = []         # For no_main environments
```

### Math Precision Modes

Choose between speed and precision for mathematical operations:

```toml
# Fast math (default) - maximum performance
simde = { version = "0.1", features = ["fast_math"] }

# Precise math - better accuracy for critical applications
simde = { version = "0.1", features = ["precise_math"] }
```

### No-STD Support

SIMDE works in `no_std` environments:

```toml
# Minimal configuration for embedded systems
simde = { version = "0.1", default-features = false, features = ["alloc"] }

# Bare metal configuration
simde = { version = "0.1", default-features = false, features = ["no_main"] }
```

### CPU Feature Detection

```rust
use simde::{has_avx, has_avx2};

// Runtime detection (enabled by default)
if has_avx2() {
    // Use AVX2 optimized path (256-bit vectors)
} else if has_avx() {
    // Fallback to AVX (128-bit vectors)
} else {
    // Scalar fallback
}

// Compile-time optimization
#[cfg(feature = "avx2")]
fn optimized_function() {
    // This code will only compile when AVX2 is enabled
    // and will be heavily optimized
}
```

### Recommended Configurations

```toml
# High-performance applications
simde = { version = "0.1", features = ["std", "avx2", "fast_math"] }

# Scientific computing (precision matters)
simde = { version = "0.1", features = ["std", "avx2", "precise_math"] }

# Embedded systems
simde = { version = "0.1", default-features = false, features = ["alloc", "avx"] }

# Bare metal / kernel development
simde = { version = "0.1", default-features = false, features = ["no_main", "avx"] }
```

## Memory Layout

All SIMD types are properly aligned for optimal performance:

```rust
use simde::{AlignData, AlignF32x8, AlignI32x8};

// Built-in aligned data types (recommended)
let data = AlignF32x8::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

// Generic aligned wrapper
let aligned = AlignData::<f32, 8>::filled(1.0);

// All AlignData types guarantee 32-byte alignment
assert_eq!(data.as_ptr() as usize % 32, 0);
```

### Convenient Type Aliases

```rust
use simde::*;

// Common SIMD sizes with automatic alignment
type AlignF32x8 = AlignData<f32, 8>;   // 8 x f32 (256-bit)
type AlignF64x4 = AlignData<f64, 4>;   // 4 x f64 (256-bit)
type AlignI32x8 = AlignData<i32, 8>;   // 8 x i32 (256-bit)
type AlignI16x16 = AlignData<i16, 16>; // 16 x i16 (256-bit)
type AlignU8x32 = AlignData<u8, 32>;   // 32 x u8 (256-bit)
```

## Error Handling

SIMDE provides comprehensive error handling:

```rust
use simde::Result;

fn simd_operation() -> Result<[f32; 8]> {
    let data = [1.0f32; 8];

    // All operations return Results
    let result = data.sqrt()?;
    let advanced = data.sin()?;

    Ok(result)
}
```

## API Reference

### Core Traits

- `SimdIntegerOps<T, N>` - Integer SIMD operations
- `SimdFloatOps<T, N>` - Floating-point SIMD operations
- `SimdSingleOps<T, N>` - Unary operations
- `SimdFmaOps<T, N>` - Fused multiply-add operations
- `SimdShiftOps<T, N>` - Bit shift operations
- `SimdLoadable` - Loading data into SIMD registers

### Aligned Data Types

- `AlignData<T, N>` - Generic aligned array wrapper
- `AlignF32x8`, `AlignF64x4` - Aligned float types
- `AlignI32x8`, `AlignI64x4` - Aligned integer types
- `AlignU8x32`, `AlignU16x16` - Aligned unsigned types

### Direct Functions

```rust
// Loading functions
load_avx2_f32(data: &[T]) -> __m256
load_avx_f32(data: &[T]) -> __m128

// Operation functions
avx2_f32(a: __m256, b: __m256, op: FloatOperation) -> Result<[T; U]>
avx2_single_f32(a: __m256, op: SingleOperations) -> Result<[T; U]>
fma_f32_avx2(a: __m256, b: __m256, c: __m256, op: FmaOperation) -> Result<[T; U]>
```

## Platform Requirements

- **Minimum**: x86_64 architecture
- **Recommended**: CPU with AVX2 support (Intel Haswell+ / AMD Excavator+)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
