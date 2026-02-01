# Binary Model Format Specification (M2BN v1.0)
## Model-To-Binary Format for ARM On-Device Inference

---

## Overview

**M2BN** is a binary format designed for efficient loading of neural network models on ARM edge devices with constrained memory. It supports:
- Multiple data types (FP32, FP16, INT8)
- Streaming loading
- Memory-efficient tensor management
- Fast validation and integrity checking

---

## File Structure

```
┌─────────────────────────────────────┐
│  Header (16 bytes)                  │
├─────────────────────────────────────┤
│  Layer 1 Metadata (64 bytes)        │
├─────────────────────────────────────┤
│  Layer 1 Tensor Data                │
├─────────────────────────────────────┤
│  Layer 2 Metadata (64 bytes)        │
├─────────────────────────────────────┤
│  Layer 2 Tensor Data                │
│  ... (repeat for each layer)        │
├─────────────────────────────────────┤
│  Checksum (8 bytes) - Optional      │
└─────────────────────────────────────┘
```

---

## Header Format (16 bytes)

```cpp
struct model_binary_header_t {
    char magic[4];              // Offset 0: "M2BN"
    uint32_t version;           // Offset 4: Format version (1)
    uint32_t num_layers;        // Offset 8: Number of layers
    uint32_t num_parameters;    // Offset 12: Total parameters
    uint32_t data_type;         // Offset 16: 0=FP32, 1=FP16, 2=INT8
    uint64_t total_size;        // Offset 20: Total file size
};
```

### Header Details

| Field | Offset | Size | Type | Range | Example |
|-------|--------|------|------|-------|---------|
| magic | 0 | 4 | char[4] | "M2BN" | "M2BN" |
| version | 4 | 4 | uint32_t | 1 | 0x00000001 |
| num_layers | 8 | 4 | uint32_t | 1-65535 | 12 |
| num_parameters | 12 | 4 | uint32_t | 1+ | 268435456 |
| data_type | 16 | 4 | uint32_t | 0,1,2 | 0 (FP32) |
| total_size | 20 | 8 | uint64_t | 1MB-1GB | 268435456 |

---

## Layer Metadata Format (64 bytes per layer)

```cpp
struct layer_metadata_t {
    uint32_t layer_id;              // 0: Layer sequence number
    char layer_name[32];            // 4: Layer name (null-terminated)
    uint32_t layer_type;            // 36: 0=dense, 1=conv, 2=embed, 3=rnn, 4=other
    uint32_t num_parameters;        // 40: Parameters in this layer
    uint32_t num_tensors;           // 44: Number of tensors (weights, bias, etc)
    uint64_t offset;                // 48: Byte offset to data
    uint64_t size;                  // 56: Size of layer data in bytes
};
```

### Layer Type Mapping

```
0 = Dense/Linear layer
1 = Convolution layer
2 = Embedding layer
3 = RNN/LSTM/GRU layer
4 = Normalization (BatchNorm, LayerNorm)
5 = Activation (ReLU, GELU, etc)
6 = Attention layer
7 = Other/Custom
```

---

## Tensor Information Format

Each layer contains metadata about its tensors:

```cpp
struct tensor_info_t {
    uint32_t tensor_id;             // Tensor index in layer
    char tensor_name[32];           // Tensor name (e.g., "weight", "bias")
    uint32_t rank;                  // Number of dimensions (1-6)
    uint32_t shape[6];              // Dimension sizes
    uint32_t dtype;                 // 0=FP32, 1=FP16, 2=INT8
    uint64_t size_bytes;            // Size in bytes
    uint64_t offset;                // Offset in layer data
};
```

---

## Data Type Encodings

### FP32 (dtype=0)
- IEEE 754 single precision floating point
- 4 bytes per element
- Range: ±1.17e-38 to ±3.4e+38
- Use for: High precision, training checkpoints

### FP16 (dtype=1)
- IEEE 754 half precision floating point
- 2 bytes per element
- Range: ±6.1e-5 to ±6.5e+4
- Use for: Memory efficient inference, moderate precision

### INT8 (dtype=2)
- Signed 8-bit integer
- 1 byte per element
- Range: -128 to +127
- Use for: Extreme memory efficiency, reduced precision
- **Requires scaling factor for dequantization**

### INT8 Quantization Metadata
```cpp
struct int8_scale_t {
    float scale;                    // Quantization scale factor
    float zero_point;               // Zero point offset
};
```

Dequantization: `float_value = (int8_value - zero_point) * scale`

---

## Example: IndicTrans2 Model

### Encoder Embedding Layer
```
Layer ID: 0
Layer Name: "encoder.embeddings"
Type: Embedding (2)
Offset: 1024 bytes

Tensors:
  0. weight: [50000 x 512] FP32 = 50000*512*4 = 102,400,000 bytes
  1. bias: (optional)

Total size: ~102 MB
```

### Encoder Attention Layer
```
Layer ID: 1
Layer Name: "encoder.attention.0"
Type: Attention (6)
Offset: 102,401,024 bytes

Tensors:
  0. query_weight: [512 x 512] FP32
  1. key_weight: [512 x 512] FP32
  2. value_weight: [512 x 512] FP32
  3. output_weight: [512 x 512] FP32
  (+ bias terms)

Total size: ~4 MB per head * 8 heads = ~32 MB
```

---

## Reading Algorithm

### Step 1: Read Header
```cpp
FILE* fp = fopen("model.m2bn", "rb");
model_binary_header_t header;
fread(&header, sizeof(header), 1, fp);

// Validate
assert(strncmp(header.magic, "M2BN", 4) == 0);
assert(header.version == 1);
```

### Step 2: Validate Checksum (if present)
```cpp
uint8_t file_data[file_size];
fread(file_data, 1, file_size - 8, fp);
uint64_t stored_checksum;
fread(&stored_checksum, 8, 1, fp);

uint64_t computed = compute_checksum(file_data, file_size - 8);
assert(computed == stored_checksum);
```

### Step 3: Stream Layer Data
```cpp
for (int i = 0; i < header.num_layers; i++) {
    // Seek to layer metadata
    fseek(fp, 16 + i * 64, SEEK_SET);
    
    layer_metadata_t layer_meta;
    fread(&layer_meta, sizeof(layer_meta), 1, fp);
    
    // Allocate memory
    float* layer_data = malloc(layer_meta.size);
    
    // Read tensor data
    fseek(fp, layer_meta.offset, SEEK_SET);
    fread(layer_data, 1, layer_meta.size, fp);
    
    // Process layer...
}
```

---

## Conversion: PyTorch → M2BN

### Python Converter
```python
import torch
import struct

def save_pytorch_to_m2bn(model, tokenizer, output_path, dtype=0):
    """
    Convert PyTorch model to M2BN format
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer object
        output_path: Output .m2bn file path
        dtype: 0=FP32, 1=FP16, 2=INT8
    """
    
    with open(output_path, 'wb') as fp:
        # Write header
        magic = b'M2BN'
        version = struct.pack('<I', 1)
        num_layers = struct.pack('<I', len(list(model.named_parameters())))
        num_params = struct.pack('<I', sum(p.numel() for p in model.parameters()))
        data_type = struct.pack('<I', dtype)
        total_size = struct.pack('<Q', 0)  # Will update later
        
        fp.write(magic + version + num_layers + num_params + data_type + total_size)
        
        # Write layer data
        layer_id = 0
        for name, param in model.named_parameters():
            # Convert to numpy
            data = param.detach().cpu().numpy()
            
            # Quantize if needed
            if dtype == 1:  # FP16
                data = data.astype(np.float16)
            elif dtype == 2:  # INT8
                scale = np.max(np.abs(data)) / 127.0
                data = (data / scale).astype(np.int8)
            
            # Write layer metadata
            layer_size = data.nbytes
            fp.write(struct.pack('<I', layer_id))
            fp.write(name.encode() + b'\0' * (32 - len(name)))
            fp.write(struct.pack('<I', 0))  # layer_type
            fp.write(struct.pack('<I', data.size))
            fp.write(struct.pack('<Q', fp.tell() + 16))  # offset
            fp.write(struct.pack('<Q', layer_size))
            
            # Write tensor data
            fp.write(data.tobytes())
            
            layer_id += 1
```

---

## File Size Examples

### IndicTrans2 Medium (256M FP32)
```
Encoder Embeddings:     102 MB
Encoder Layers (6):      90 MB
Decoder Embeddings:      10 MB
Decoder Layers (6):      40 MB
Output Projection:        5 MB
─────────────────────────────
Total:                  247 MB
```

### Quantized to FP16 (128MB)
```
Same structure but 2x smaller
```

### Quantized to INT8 (64MB)
```
Same structure but 4x smaller
(with per-tensor scaling)
```

---

## Validation & Error Handling

### Checksum Verification
```cpp
uint64_t compute_checksum(const uint8_t* data, size_t size) {
    uint64_t sum = 0;
    for (size_t i = 0; i < size; i++) {
        sum = ((sum << 5) + sum) ^ data[i];  // Rolling XOR+shift
    }
    return sum;
}
```

### Error Codes
```cpp
#define M2BN_OK              0
#define M2BN_ERR_INVALID_MAGIC   -1
#define M2BN_ERR_VERSION         -2
#define M2BN_ERR_CHECKSUM        -3
#define M2BN_ERR_MEMORY          -4
#define M2BN_ERR_TRUNCATED       -5
```

---

## Performance Characteristics

### Loading Speed
- **Sequential read**: ~500 MB/s (typical ARM SSD)
- **1 second per 512MB model**: Standard expectation
- **Streaming without full load**: Supported via layer-wise offsets

### Memory Usage During Load
```
Model Size (FP32):     256 MB
Load Buffer:            10 MB
Metadata:                1 MB
─────────────────────────────
Peak Memory:           267 MB
```

---

## Version History

### v1.0 (Current)
- Initial format
- Support for FP32, FP16, INT8
- Per-layer metadata
- Optional checksum validation

### v2.0 (Future)
- Compression support
- Multiple quantization schemes
- Incremental loading
- Distributed layers

---

## Design Rationale

1. **Magic Number**: Ensures file format validation
2. **Per-Layer Metadata**: Enables streaming/lazy loading
3. **Offset-Based**: Allows random access to layers
4. **Checksum Optional**: Can be skipped on trusted systems
5. **Simple Structure**: Easy to parse in embedded systems
6. **Extensible**: Room for v2.0 enhancements

---

## Usage in S2S System

```cpp
#include "model_loader.h"

// Create loader
s2s::ModelLoader loader;

// Load model
loader.load("indicTrans2.m2bn");

// Get specific tensor
const float* encoder_weight = loader.get_tensor("encoder.embeddings.weight", 
                                                &shape, rank);

// Get memory usage
printf("Model: %.1f MB\n", loader.get_model_size() / 1e6);
printf("Runtime: %.1f MB\n", loader.get_memory_used() / 1e6);
```

---

**Specification Version**: 1.0  
**Last Updated**: January 30, 2026  
**Compatibility**: ARM 32/64-bit systems
