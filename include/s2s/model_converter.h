#ifndef MODEL_CONVERTER_H
#define MODEL_CONVERTER_H

#include <cstdint>
#include <cstddef>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Model Binary Format Header
 * Magic: "M2BN" (Model To Binary)
 * Version: 1.0
 */
typedef struct {
    char magic[4];                  // "M2BN"
    uint32_t version;               // Format version
    uint32_t num_layers;            // Number of network layers
    uint32_t num_parameters;        // Total number of parameters
    uint32_t data_type;             // 0=float32, 1=float16, 2=int8
    uint64_t total_size;            // Total binary size in bytes
} model_binary_header_t;

/**
 * Layer metadata for binary model
 */
typedef struct {
    uint32_t layer_id;
    char layer_name[256];
    uint32_t layer_type;            // 0=dense, 1=conv, 2=embedding, etc.
    uint32_t num_parameters;
    uint64_t offset;                // Offset in binary file
    uint64_t size;                  // Size in bytes
} layer_metadata_t;

/**
 * Tensor information for efficient loading
 */
typedef struct {
    uint32_t tensor_id;
    char tensor_name[256];
    uint32_t rank;                  // Number of dimensions
    uint32_t* shape;                // Shape array [rank]
    uint64_t size_bytes;            // Total bytes for this tensor
    uint64_t offset;                // Offset in binary file
} tensor_info_t;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

namespace s2s {

/**
 * IndicTrans2 Model Converter
 * Converts PyTorch checkpoint to optimized binary format
 */
class ModelBinaryConverter {
public:
    ModelBinaryConverter();
    ~ModelBinaryConverter();

    /**
     * Convert PyTorch model checkpoint to binary format
     * @param pytorch_path Path to .pt or .pth file
     * @param output_binary_path Path to write .m2bn binary
     * @param data_type 0=float32, 1=float16 (for quantization)
     * @return 0 on success, <0 on error
     */
    int convert_pytorch_to_binary(
        const char* pytorch_path,
        const char* output_binary_path,
        uint32_t data_type = 0
    );

    /**
     * Extract model metadata without loading full weights
     * Useful for architecture inspection
     */
    int extract_model_metadata(
        const char* model_path,
        model_binary_header_t* header,
        layer_metadata_t** layers
    );

    /**
     * Validate binary model format
     */
    int validate_binary_model(const char* binary_path);

private:
    uint64_t total_parameters;
    std::vector<tensor_info_t> tensor_registry;

    int parse_pytorch_checkpoint(const char* path);
    int serialize_to_binary(const char* output_path, uint32_t data_type);
    int write_header(FILE* fp, const model_binary_header_t& header);
    int write_tensor_data(FILE* fp, const tensor_info_t& tensor);
};

/**
 * Quantization utilities for model compression
 */
class ModelQuantizer {
public:
    /**
     * Quantize float32 to float16
     */
    static int quantize_float32_to_float16(
        const float* input,
        uint16_t* output,
        size_t num_elements
    );

    /**
     * Quantize float32 to int8 with scaling
     */
    static int quantize_float32_to_int8(
        const float* input,
        int8_t* output,
        float* scale,
        size_t num_elements
    );

    /**
     * Dequantize int8 back to float32
     */
    static int dequantize_int8_to_float32(
        const int8_t* input,
        float* output,
        float scale,
        size_t num_elements
    );
};

} // namespace s2s

#endif // __cplusplus

#endif // MODEL_CONVERTER_H
