# Disk Layout for Scalar Quantizer and Scalar Quantized Vectors

## 1. Motivation

We need a compact on-disk representation for:

- **Scalar-quantized (SQ) vectors** — the compressed data itself  
- **SQ quantizer parameters** — metadata needed to dequantize, this includes any codebook as well if required

**Requirements:**

- Compactness for large datasets  
- Versioned schema for forward-compatibility  
- Easy parsing in Rust code; “good to have” in other languages  
- Agnostic to the existing PQ disk layout.
- Optional human inspection when troubleshooting  

**Out of Scope:**

- Redesign of the disk layout for PQ quantizer and PQ vectors
- Disk layout for BQ, and RaBitQ are out of scope for this RFC, as the schema for upcoming quantizers is still not well-defined. Additionally, I believe we should allow flexibility in deciding the disk layout for each quantizer type independently, since they can vary significantly in terms of data size and optional fields.

## 2. Scope

Covers two files, both sharing a common prefix `index_path_prefix`:

- `index_path_prefix_sq_compressed.bin` — compressed vectors  
- `index_path_prefix_sq_quantizer.{json | bin}` — quantizer parameters  

> **Note:**
> The metadata schema will include a `compressed_data_file_name` field, which stores the exact name of the compressed-data file. This lets clients perform the entire loading process by supplying only the full path to the metadata file.
> The compressed-data file itself continues to use the `index_path_prefix_sq_compressed.bin` naming convention, making it easy to locate for any downstream tasks (e.g., uploading to blob storage).

## 3. Compressed-Data File Layout (`.bin`)

Binary, little-endian:

| Offset | Size      | Type    | Description                                                    |
|-------:|----------:|--------:|----------------------------------------------------------------|
| 0      | 4         | `u32`   | `number_of_points` — total vectors stored                      |
| 4      | 4         | `u32`   | `dim_compressed` — number of `u8` per vector                   |
| 8      | N × D × 1 | `u8[]`  | raw quantized bytes: point 0 (D bytes), point 1, …, point N−1  |

> **Note:**  
> `dim_compressed = ⌈(bits_per_original_dimension × original_dim) / 8⌉`

This exactly matches the format we currently use for storing Product Quantized compressed vectors.

## 4. Quantizer-Params File Layout

Here’s how the Scalar Quantizer is defined in Rust:

```rust
pub struct ScalarQuantizer<const NBITS: usize> {
    /// The scaling parameter applied to each vector component.
    scale: f32,

    /// The amount each data point is shifted.
    ///
    /// This is computed as the dataset mean subtracted by the scaling parameter.
    /// The additional subtraction is needed to ensure we can map encodings into an unsigned
    /// integer.
    ///
    /// For datasets that have components with non-zero mean, this can greatly improve the
    /// quality of quantization by decreasing the observed dynamic range across all vector
    /// component, but this shift must be applied regardless of whether or not the mean
    /// is calculated.
    shift: Vec<f32>,

    /// The square norm of the shift.
    /// This quantity is useful when computing dot-products.
    shift_square_norm: f32,

    /// When processing queries, it may be beneficial to modify the query norm to match the
    /// dataset norm.
    ///
    /// This is only applicable when `InnerProduct` and `Cosine` are used, but serves to
    /// move the query into the dynamic range of the quantization.
    mean_norm: Option<f32>,
}
```

The quantizer file grows in `O(num_of_dimensions)`

### 4.1 Protobuf Schema (`.proto`)

Compact, cross-language, and schema-evolvable:

```proto
syntax = "proto3";

message SQQuantizer {
  float version       = 1;
  uint32 nbits        = 2;
  float scale         = 3;
  repeated float shift      = 4;  // length = original_dim
  optional float mean_norm = 5;  // absent if unset
  string compressed_data_file_name = 6;  // e.g. "<prefix>_sq_compressed.bin"
}
```

#### Pros

- Built-in versioning and optional fields
- Extensible  
- Flexible to accommodate more complex fields
- Small footprint  
- Language-agnostic  

#### Cons

- Production dependency on proto libraries like [prost](https://docs.rs/prost/latest/prost/) (the project already has a production dependency on this)

### 4.2 Alternative: JSON Schema (`.json`)

Human-readable, self-describing:

```json
{
  "version": 0.1,
  "nbits": 1,
  "scale": 0.123,
  "shift": [0.10, -0.20, 0.05, ...],
  "mean_norm": 0.5678,
  "compressed_data_file_name": "<prefix>_sq_compressed.bin"
}
```

#### Pros

- Extensible  
- Easy to read/edit  

#### Cons

- Text overhead  
- Production dependencies on [`serde_json`](https://docs.rs/serde_json/latest/serde_json/) (already a dev-dependency of the library)
- JSON parse cost  

### 4.3 Alternative: Custom Binary Schema (.bin)

Compact, fixed-layout; little-endian:

| Offset          | Size       | Type     | Description                                    |
|----------------:|-----------:|---------:|------------------------------------------------|
| 0               | 4          | `u32`    | `version_major`                                |
| 4               | 4          | `u32`    | `version_minor`                                |
| 8               | 4          | `u32`    | `nbits` — bits per original dimension          |
| 12              | 4          | `f32`    | `scale`                                        |
| 16              | 4          | `u32`    | `dim` — number of floats in `shift`            |
| 20              | 4 × `dim`  | `f32[]`  | `shift[0…dim−1]`                               |
| 20 + 4 × `dim`  | 4          | `f32`    | `mean_norm` (or sentinel `NaN` if missing)     |
| -               | variable   | `string` | `compressed_data_file_name` (null-terminated)  |

#### Pros

- Minimal overhead  
- Very fast to parse  

#### Cons

- Custom reader required  
- Less flexible for future fields  

## 5. Trade-Off Comparison

| Aspect             | JSON Schema                       | Custom Binary Schema                 | Protobuf Schema                                    |
| ------------------ | --------------------------------- | ------------------------------------ | -------------------------------------------------- |
| **Size on disk**   | ≈ 100 B + text overhead           | ≈ (24 + 4 × dim) B + filename length | ≈ (1 B tag + 4 B float)×4 + 4 × dim bytes + filename length |
| **Parse latency**  | moderate (string → numbers)       | minimal (memmap + struct read)       | moderate (generated parsers)                       |
| **Extensibility**  | add fields arbitrarily            | must version and pad or extend       | backward-compatible adds fields                    |
| **Tooling**        | universal JSON libraries          | custom code or `struct` unpacking    | standard protobuf generators                       |
| **Human-readable** | yes                               | no                                   | yes (possible with [textprotos](https://protobuf.dev/reference/protobuf/textformat-spec/)) |
| **Error tolerance**| tolerant of unknown keys          | brittle if layout changes            | tolerant of unknown fields                         |

## 6. Versioning & Backwards Compatibility

- **JSON:** bump `version` field; unknown keys are ignored.  
- **Binary:** use major/minor; readers skip unknown-versioned blocks or fall back to defaults.  
- **Protobuf:** add new fields with higher tags; old readers ignore unknown tags. Additionally we have a `version` field as well to handle any incompatible change.

### 7. Recommendation

After evaluating all formats, we recommend the **Protobuf Schema** for SQ quantizer parameters:

- **Extensibility:** New fields can be added without breaking existing readers; unknown tags are safely ignored.  
- **Error tolerance:** Built-in tag-based parsing skips unrecognized fields, avoiding brittle failures when schemas evolve.  
- **Parse latency:** Generated parsers (e.g., via `prost` or `protobuf`) offer moderate performance, acceptable for small quantizer files.
- **Memory efficiency:** Protobuf’s compact binary encoding minimizes disk and memory footprint.

**Trade-offs:**

- Requires adding a protobuf runtime dependency (e.g., [`prost`](https://docs.rs/prost/latest/prost/) or [`protobuf`](https://docs.rs/protobuf/latest/protobuf/)) to DiskANN’s Rust crate.  
- Slightly higher parse cost compared to raw binary, but gains in maintainability and cross-language support.  
