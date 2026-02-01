#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "model_converter.h"
#include <cstdint>
#include <cstddef>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Binary model loader context
 */
typedef struct model_loader_s model_loader_t;

/**
 * Tensor data structure
 */
typedef struct {
    float* data;
    uint32_t rank;
    uint32_t* shape;        // [rank] dimensions
    size_t size;            // Total number of elements
    size_t size_bytes;      // Total bytes
    char name[256];
} tensor_t;

/**
 * Create a model loader for binary models
 */
model_loader_t* model_loader_create();

/**
 * Load binary model from file
 * @param loader Model loader context
 * @param model_path Path to .m2bn binary file
 * @return 0 on success, <0 on error
 */
int model_loader_load(model_loader_t* loader, const char* model_path);

/**
 * Get tensor by name
 */
tensor_t* model_loader_get_tensor(model_loader_t* loader, const char* tensor_name);

/**
 * Get number of tensors in model
 */
uint32_t model_loader_get_tensor_count(model_loader_t* loader);

/**
 * Get tensor by index
 */
tensor_t* model_loader_get_tensor_by_index(model_loader_t* loader, uint32_t index);

/**
 * Get model header metadata
 */
const model_binary_header_t* model_loader_get_header(model_loader_t* loader);

/**
 * Verify model integrity (checksums)
 */
int model_loader_verify(model_loader_t* loader);

/**
 * Free a tensor (if allocated)
 */
void model_loader_free_tensor(tensor_t* tensor);

/**
 * Cleanup and free resources
 */
void model_loader_destroy(model_loader_t* loader);

#ifdef __cplusplus
}

namespace s2s {

/**
 * C++ wrapper for model loading with memory management
 */
class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();

    /**
     * Load IndicTrans2 model from binary format
     * @param model_path Path to .m2bn file
     * @return true on success
     */
    bool load(const char* model_path);

    /**
     * Get tensor by name
     */
    const float* get_tensor(const char* name, const uint32_t** shape, uint32_t& rank);

    /**
     * Get encoder embedding layer
     */
    const float* get_encoder_embedding() const;

    /**
     * Get decoder embedding layer
     */
    const float* get_decoder_embedding() const;

    /**
     * Get model configuration
     */
    const model_binary_header_t& get_header() const;

    /**
     * Check if model is loaded
     */
    bool is_loaded() const { return loaded; }

    /**
     * Get total model size in bytes
     */
    uint64_t get_model_size() const;

    /**
     * Get memory footprint of loaded tensors
     */
    uint64_t get_memory_used() const;

    /**
     * Prefetch tensors into fast memory (cache optimization)
     */
    int prefetch_tensors(const std::vector<const char*>& tensor_names);

private:
    struct Tensor {
        std::vector<float> data;
        std::vector<uint32_t> shape;
        uint32_t rank;
        std::string name;
    };

    model_binary_header_t header;
    std::vector<Tensor> tensors;
    std::vector<std::string> tensor_names;
    bool loaded;
    uint64_t memory_used;

    int parse_binary_file(const char* path);
    int allocate_tensor(Tensor& tensor, size_t num_elements);
};

} // namespace s2s

#endif // __cplusplus

#endif // MODEL_LOADER_H
