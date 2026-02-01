# Project Index & Navigation Guide
## ARM Real-Time Speech-to-Speech Translation System

---

## 📚 Documentation Index

### Getting Started
1. **[QUICKSTART.md](QUICKSTART.md)** - **START HERE** ⭐
   - 5-minute build guide
   - Basic code examples
   - Common issues & solutions
   - Performance targets

2. **[PHASE2_COMPLETION_SUMMARY.md](PHASE2_COMPLETION_SUMMARY.md)**
   - Project overview
   - What was built (8 components)
   - Architecture summary
   - Achievements checklist

### Comprehensive References
3. **[PHASE2_IMPLEMENTATION.md](PHASE2_IMPLEMENTATION.md)** - Deep dive reference
   - Complete component documentation
   - All APIs explained
   - Usage examples
   - Performance characteristics

4. **[BINARY_FORMAT_SPEC.md](BINARY_FORMAT_SPEC.md)**
   - M2BN binary format specification
   - PyTorch conversion guide
   - File structure details
   - Validation procedures

---

## 📁 Source Code Organization

### Pipeline Orchestration
- **[src/pipeline/pipeline.h](src/pipeline/pipeline.h)** - Pipeline interface
  - Ring buffer implementation
  - Streaming audio processing
  - Statistics tracking
  - ~800 LOC

### Speech Recognition (ASR)
- **[src/asr/asr.h](src/asr/asr.h)** - ASR interface
- **[src/asr/asr.cpp](src/asr/asr.cpp)** - Full implementation
  - Feature extraction (MFCC, Mel-spectrogram)
  - Voice Activity Detection (VAD)
  - CTC decoding (greedy + beam search)
  - ~600 LOC

### Machine Translation (IndicTrans2)
- **[src/mt/mt.h](src/mt/mt.h)** - MT interface
- **[src/mt/mt.cpp](src/mt/mt.cpp)** - Full implementation
  - Tokenizer (BPE)
  - Transformer encoder/decoder
  - Beam search
  - ~700 LOC

### Core Utilities
- **[src/utils/matrix_ops.cpp](src/utils/matrix_ops.cpp)** - Matrix operations
  - GEMM, batched GEMM
  - Activations, normalization
  - NEON/SME2 backends
  - ~600 LOC

- **[src/utils/model_converter.cpp](src/utils/model_converter.cpp)** - Model converter
  - PyTorch to binary format
  - Quantization (FP16, INT8)
  - Validation
  - ~300 LOC

- **[src/utils/model_loader.cpp](src/utils/model_loader.cpp)** - Model loader
  - Binary format parsing
  - Tensor allocation
  - Memory management
  - ~300 LOC

### Optimization Kernels

**ARM NEON (128-bit SIMD)**
- **[src/asr/kernels/neon/simd_kernels.S](src/asr/kernels/neon/simd_kernels.S)**
  - SGEMM (matrix multiply)
  - ReLU, batch normalization
  - Softmax
  - ~300 LOC

**ARM SME2 (Scalable Matrix Extension)**
- **[src/asr/kernels/sme2/matrix_gemm.S](src/asr/kernels/sme2/matrix_gemm.S)**
  - SMOPA (outer product accumulate)
  - Streaming matrix operations
  - ~250 LOC

### Public Headers
- **[include/s2s/s2s.h](include/s2s/s2s.h)** - Main public API
- **[include/s2s/matrix_ops.h](include/s2s/matrix_ops.h)** - Matrix operations
- **[include/s2s/model_converter.h](include/s2s/model_converter.h)** - Converter
- **[include/s2s/model_loader.h](include/s2s/model_loader.h)** - Loader
- **[include/s2s/logger.h](include/s2s/logger.h)** - Logging

### Build System
- **[CMakeLists.txt](CMakeLists.txt)** - Root configuration
  - Architecture detection
  - NEON/SME2 options
  - Test/benchmark configuration

- **[src/CMakeLists.txt](src/CMakeLists.txt)** - Source configuration
  - Component definitions
  - Compiler flags
  - Library creation

### Testing
- **[tests/run_tests.py](tests/run_tests.py)** - Test runner
  - Component tests
  - Performance validation
  - 8+ test categories

---

## 🔍 Quick File Reference

### By Component

#### Audio Input
- `src/audio/audio_backend.h` - Backend interface
- `src/audio/audio_alsa.cpp` - ALSA backend
- `src/audio/audio_jack.cpp` - JACK backend

#### Feature Extraction
- Search: **`FeatureExtractor`** class in `src/asr/asr.cpp`
  - MFCC extraction
  - Mel-spectrogram
  - Delta features

#### Voice Activity Detection
- Search: **`StreamingSegmenter`** class in `src/asr/asr.cpp`
  - Energy-based VAD
  - Utterance boundaries
  - Streaming support

#### Acoustic Model
- Search: **`AcousticModel`** class in `src/asr/asr.cpp`
  - Forward inference
  - State management

#### CTC Decoder
- Search: **`CTCDecoder`** class in `src/asr/asr.cpp`
  - Greedy decoding
  - Beam search

#### Tokenizer
- Search: **`Tokenizer`** class in `src/mt/mt.cpp`
  - BPE encoding
  - Vocabulary management

#### Transformer Encoder
- Search: **`TransformerEncoder`** class in `src/mt/mt.cpp`
  - Self-attention
  - Feed-forward layers

#### Transformer Decoder
- Search: **`TransformerDecoder`** class in `src/mt/mt.cpp`
  - Cross-attention
  - Autoregressive decoding

#### Beam Search
- Search: **`BeamSearchDecoder`** class in `src/mt/mt.cpp`
  - Hypothesis tracking
  - Top-k sampling

#### Ring Buffer
- Search: **`RingBuffer`** class in `src/pipeline/pipeline.cpp`
  - Circular buffering
  - Wrap-around handling

#### Pipeline Orchestration
- Search: **`S2SPipeline`** class in `src/pipeline/pipeline.cpp`
  - ASR → MT → TTS flow
  - Statistics tracking

---

## 🎯 Usage Scenarios

### Scenario 1: Convert PyTorch Model
```cpp
// Location: src/utils/model_converter.cpp
s2s::ModelBinaryConverter converter;
converter.convert_pytorch_to_binary("model.pt", "model.m2bn", 0);
```

### Scenario 2: Load Binary Model
```cpp
// Location: src/utils/model_loader.cpp
s2s::ModelLoader loader;
loader.load("model.m2bn");
```

### Scenario 3: Perform Matrix Multiplication
```cpp
// Location: src/utils/matrix_ops.cpp
matops_init();  // Auto-detect backend
matops_gemm(0, 0, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
```

### Scenario 4: Process Audio Stream
```cpp
// Location: src/pipeline/pipeline.cpp
pipeline_create(&config);
pipeline_process_audio_chunk(audio, num_samples, output, &len);
pipeline_destroy();
```

### Scenario 5: Direct ASR Usage
```cpp
// Location: src/asr/asr.cpp
asr_context_t* ctx = asr_create("model.m2bn");
asr_process(ctx, audio, samples, result, &is_final);
```

### Scenario 6: Direct MT Usage
```cpp
// Location: src/mt/mt.cpp
mt_context_t* ctx = mt_create("model.m2bn", "en", "hi");
mt_translate(ctx, input_text, output_text, max_len);
```

---

## 📊 Component Statistics

| Component | File | LOC | Dependencies |
|-----------|------|-----|--------------|
| Pipeline | `pipeline.cpp` | 800 | ASR, MT, utils |
| ASR | `asr.cpp` | 600 | matrix_ops |
| MT | `mt.cpp` | 700 | matrix_ops |
| Matrix Ops | `matrix_ops.cpp` | 600 | NEON, SME2 |
| Model Converter | `model_converter.cpp` | 300 | - |
| Model Loader | `model_loader.cpp` | 300 | - |
| NEON Kernels | `simd_kernels.S` | 300 | - |
| SME2 Kernels | `matrix_gemm.S` | 250 | - |
| **TOTAL** | | **~4550** | |

---

## 🚀 Getting Started (Recommended Path)

### Day 1: Setup & Build
1. Read: [QUICKSTART.md](QUICKSTART.md)
2. Build: `cmake -DENABLE_NEON=ON && make -j4`
3. Test: `python3 tests/run_tests.py`

### Day 2: Understand Components
1. Read: [PHASE2_IMPLEMENTATION.md](PHASE2_IMPLEMENTATION.md) sections 1-3
2. Review: `include/s2s/` headers
3. Explore: Component interfaces

### Day 3: Deep Dive
1. Study: Source implementations in `src/`
2. Review: Algorithm implementations
3. Understand: Optimization techniques

### Day 4: Optimization
1. Learn: NEON and SME2 kernels
2. Profile: Performance benchmarks
3. Tune: Model parameters

### Day 5: Integration
1. Integrate: Into your application
2. Deploy: On target ARM device
3. Optimize: For your specific use case

---

## 🔗 Cross-References

### Matrix Operations
- **Interface**: `include/s2s/matrix_ops.h`
- **Implementation**: `src/utils/matrix_ops.cpp`
- **NEON Backend**: `src/asr/kernels/neon/simd_kernels.S`
- **SME2 Backend**: `src/asr/kernels/sme2/matrix_gemm.S`
- **Tests**: `tests/run_tests.py` → test_matrix_ops()

### Model Management
- **Converter**: `src/utils/model_converter.cpp`
- **Loader**: `src/utils/model_loader.cpp`
- **Spec**: `BINARY_FORMAT_SPEC.md`

### ASR Pipeline
- **Header**: `src/asr/asr.h`
- **Implementation**: `src/asr/asr.cpp`
- **Components**: FeatureExtractor, StreamingSegmenter, AcousticModel, CTCDecoder

### MT Pipeline
- **Header**: `src/mt/mt.h`
- **Implementation**: `src/mt/mt.cpp`
- **Components**: Tokenizer, TransformerEncoder, TransformerDecoder, BeamSearchDecoder

### S2S Pipeline
- **Header**: `src/pipeline/pipeline.h`
- **Implementation**: `src/pipeline/pipeline.cpp`
- **Components**: RingBuffer, S2SPipeline

---

## 📖 Documentation Layers

```
┌─ QUICKSTART.md (5 min read)
│  └─ Fast overview + build
│
├─ PHASE2_COMPLETION_SUMMARY.md (10 min read)
│  └─ What was built + achievements
│
├─ PHASE2_IMPLEMENTATION.md (30 min read)
│  └─ Complete reference guide
│
├─ BINARY_FORMAT_SPEC.md (15 min read)
│  └─ Model format details
│
└─ Source code headers (2 hours)
   └─ API documentation + examples
```

---

## 🎓 Learning Path

1. **Beginner**: Read QUICKSTART.md → Build → Run tests
2. **Intermediate**: Read PHASE2_IMPLEMENTATION.md → Review headers
3. **Advanced**: Study source code → Modify components → Add features
4. **Expert**: Optimize kernels → Deploy → Profile

---

## 🔗 Interconnections

```
User Application
      ↓
s2s.h (main API)
      ↓
┌─────┴─────┬──────────┐
│           │          │
pipeline    ASR        MT
│           │          │
├─ Ring     ├─ Feature ├─ Tokenizer
│  Buffer   │  Extract │
├─ Orch     ├─ VAD     ├─ Encoder
│  estrate  ├─ Acoustic├─ Decoder
├─ Stats    └─ CTC     └─ Beam Search
│
matrix_ops
│
├─ Reference
├─ NEON kernels
└─ SME2 kernels
```

---

## 📞 Support Matrix

| Question | Answer Location |
|----------|-----------------|
| How do I build? | QUICKSTART.md |
| What was built? | PHASE2_COMPLETION_SUMMARY.md |
| How do I use component X? | PHASE2_IMPLEMENTATION.md + src/X/X.h |
| What's the binary format? | BINARY_FORMAT_SPEC.md |
| How do I optimize? | PHASE2_IMPLEMENTATION.md → Optimization section |
| How do I test? | tests/run_tests.py |
| What's the architecture? | PHASE2_IMPLEMENTATION.md → Architecture |

---

## 🎯 Key Files by Use Case

### Just Want to Build
- `CMakeLists.txt`
- `QUICKSTART.md`

### Want to Understand Everything
- `PHASE2_IMPLEMENTATION.md`
- `PHASE2_COMPLETION_SUMMARY.md`
- All source headers

### Want to Add a New Model
- `BINARY_FORMAT_SPEC.md`
- `src/utils/model_converter.cpp`
- `src/utils/model_loader.cpp`

### Want to Optimize Performance
- `src/asr/kernels/neon/simd_kernels.S`
- `src/asr/kernels/sme2/matrix_gemm.S`
- `src/utils/matrix_ops.cpp`

### Want to Deploy
- `QUICKSTART.md` → Build section
- `PHASE2_IMPLEMENTATION.md` → Deployment notes
- Test on target device

---

**Generated**: January 30, 2026  
**Project Status**: Phase 2 Complete ✅  
**Next**: Phase 3 (TTS, deployment, optimization)

---

## Navigation Quick Links

[📖 Documentation](.) | [🚀 Quick Start](QUICKSTART.md) | [📋 Summary](PHASE2_COMPLETION_SUMMARY.md) | [🔍 Implementation](PHASE2_IMPLEMENTATION.md) | [📦 Binary Format](BINARY_FORMAT_SPEC.md) | [💻 Source](src/)
