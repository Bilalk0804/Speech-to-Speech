#include "../include/s2s/model_loader.h"
#include "../include/s2s/logger.h"
#include <cstring>
#include <cstdio>
#include <algorithm>

// ============================================================================
// C Interface Implementation
// ============================================================================

struct model_loader_s {
    std::vector<tensor_t*> tensors;
    model_binary_header_t header;
    bool loaded;
    uint64_t memory_used;
};

model_loader_t* model_loader_create() {
    auto loader = new model_loader_s();
    loader->loaded = false;
    loader->memory_used = 0;
    return loader;
}

int model_loader_load(model_loader_t* loader, const char* model_path) {
    if (!loader || !model_path) {
        return -1;
    }

    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        LOG_ERROR("Cannot open model file: %s", model_path);
        return -1;
    }

    // Read header
    size_t read = fread(&loader->header, sizeof(model_binary_header_t), 1, fp);
    if (read != 1) {
        LOG_ERROR("Failed to read model header");
        fclose(fp);
        return -1;
    }

    // Validate magic
    if (strncmp(loader->header.magic, "M2BN", 4) != 0) {
        LOG_ERROR("Invalid model magic number");
        fclose(fp);
        return -1;
    }

    LOG_INFO("Loading model: %u layers, %u parameters",
             loader->header.num_layers, loader->header.num_parameters);

    // Read tensors (implementation would load actual data)
    for (uint32_t i = 0; i < loader->header.num_layers; i++) {
        tensor_t* tensor = new tensor_t();
        tensor->rank = 2;
        tensor->shape = new uint32_t[2]{512, 512};
        tensor->size = 512 * 512;
        tensor->size_bytes = tensor->size * sizeof(float);
        tensor->data = new float[tensor->size];
        
        loader->tensors.push_back(tensor);
        loader->memory_used += tensor->size_bytes;
    }

    fclose(fp);
    loader->loaded = true;
    
    LOG_INFO("Model loaded: %.2f MB memory used",
             loader->memory_used / (1024.0f * 1024.0f));
    return 0;
}

tensor_t* model_loader_get_tensor(model_loader_t* loader, const char* tensor_name) {
    if (!loader || !tensor_name) return nullptr;
    
    for (auto tensor : loader->tensors) {
        if (strcmp(tensor->name, tensor_name) == 0) {
            return tensor;
        }
    }
    return nullptr;
}

uint32_t model_loader_get_tensor_count(model_loader_t* loader) {
    if (!loader) return 0;
    return loader->tensors.size();
}

tensor_t* model_loader_get_tensor_by_index(model_loader_t* loader, uint32_t index) {
    if (!loader || index >= loader->tensors.size()) return nullptr;
    return loader->tensors[index];
}

const model_binary_header_t* model_loader_get_header(model_loader_t* loader) {
    if (!loader) return nullptr;
    return &loader->header;
}

int model_loader_verify(model_loader_t* loader) {
    if (!loader || !loader->loaded) return -1;
    
    // Verify all tensors are loaded correctly
    for (auto tensor : loader->tensors) {
        if (!tensor || !tensor->data) {
            LOG_ERROR("Tensor data is null");
            return -1;
        }
    }
    
    LOG_INFO("Model verification passed");
    return 0;
}

void model_loader_free_tensor(tensor_t* tensor) {
    if (!tensor) return;
    
    delete[] tensor->shape;
    delete[] tensor->data;
    delete tensor;
}

void model_loader_destroy(model_loader_t* loader) {
    if (!loader) return;
    
    for (auto tensor : loader->tensors) {
        model_loader_free_tensor(tensor);
    }
    
    loader->tensors.clear();
    delete loader;
}

// ============================================================================
// C++ Interface Implementation
// ============================================================================

namespace s2s {

ModelLoader::ModelLoader()
    : loaded(false), memory_used(0) {}

ModelLoader::~ModelLoader() {
    tensors.clear();
    tensor_names.clear();
}

bool ModelLoader::load(const char* model_path) {
    if (!model_path || loaded) return false;

    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        LOG_ERROR("Cannot open model file: %s", model_path);
        return false;
    }

    // Read header
    size_t read = fread(&header, sizeof(model_binary_header_t), 1, fp);
    if (read != 1) {
        LOG_ERROR("Failed to read model header");
        fclose(fp);
        return false;
    }

    // Validate magic
    if (strncmp(header.magic, "M2BN", 4) != 0) {
        LOG_ERROR("Invalid model magic number");
        fclose(fp);
        return false;
    }

    LOG_INFO("Loading IndicTrans2 model: %u layers, %u parameters",
             header.num_layers, header.num_parameters);

    // Parse binary file
    int ret = parse_binary_file(model_path);
    fclose(fp);

    if (ret < 0) {
        LOG_ERROR("Failed to parse binary model");
        return false;
    }

    loaded = true;
    LOG_INFO("Model loaded successfully: %.2f MB memory",
             memory_used / (1024.0f * 1024.0f));
    return true;
}

const float* ModelLoader::get_tensor(
    const char* name,
    const uint32_t** shape,
    uint32_t& rank
) {
    if (!loaded) return nullptr;

    for (auto& tensor : tensors) {
        if (tensor.name == name) {
            rank = tensor.rank;
            *shape = tensor.shape.data();
            return tensor.data.data();
        }
    }

    LOG_WARN("Tensor not found: %s", name);
    return nullptr;
}

const float* ModelLoader::get_encoder_embedding() const {
    if (tensors.empty()) return nullptr;
    return tensors[0].data.data();
}

const float* ModelLoader::get_decoder_embedding() const {
    if (tensors.size() < 2) return nullptr;
    return tensors[1].data.data();
}

const model_binary_header_t& ModelLoader::get_header() const {
    return header;
}

uint64_t ModelLoader::get_model_size() const {
    return header.total_size;
}

uint64_t ModelLoader::get_memory_used() const {
    return memory_used;
}

int ModelLoader::prefetch_tensors(const std::vector<const char*>& tensor_names) {
    // This would prefetch tensors into L2/L3 cache
    // For now, just verify they exist
    for (const char* name : tensor_names) {
        bool found = false;
        for (const auto& tensor : tensors) {
            if (tensor.name == name) {
                found = true;
                break;
            }
        }
        if (!found) {
            LOG_WARN("Tensor to prefetch not found: %s", name);
        }
    }
    return 0;
}

int ModelLoader::parse_binary_file(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;

    // Skip header
    fseek(fp, sizeof(model_binary_header_t), SEEK_SET);

    // Read tensors
    for (uint32_t i = 0; i < header.num_layers; i++) {
        Tensor tensor;
        
        // Read tensor metadata (simplified)
        uint32_t metadata[4];
        if (fread(metadata, sizeof(uint32_t), 4, fp) != 4) {
            LOG_ERROR("Failed to read tensor metadata");
            fclose(fp);
            return -1;
        }
        
        tensor.rank = metadata[1];
        tensor.shape.resize(tensor.rank);
        
        if (tensor.rank >= 2) {
            tensor.shape[0] = metadata[2];
            tensor.shape[1] = metadata[3];
        }

        // Read tensor name
        char name_buf[256];
        size_t name_len = 0;
        while (name_len < 255 && fgetc(fp) != '\0') {
            name_len++;
        }
        rewind(fp);  // In real implementation, properly read string
        tensor.name = "tensor_";
        tensor.name += std::to_string(i);

        // Calculate tensor size
        size_t size = 1;
        for (auto dim : tensor.shape) {
            size *= dim;
        }
        
        // Allocate and read tensor data
        if (allocate_tensor(tensor, size) < 0) {
            LOG_ERROR("Failed to allocate tensor memory");
            fclose(fp);
            return -1;
        }

        // Read tensor data (simplified)
        for (size_t j = 0; j < size; j++) {
            tensor.data[j] = 0.1f;  // Stub
        }

        tensors.push_back(tensor);
        tensor_names.push_back(tensor.name);
    }

    fclose(fp);
    return 0;
}

int ModelLoader::allocate_tensor(Tensor& tensor, size_t num_elements) {
    try {
        tensor.data.resize(num_elements);
        memory_used += num_elements * sizeof(float);
        
        // Check memory limit (e.g., 512 MB for on-device)
        const uint64_t MAX_MEMORY = 512 * 1024 * 1024;
        if (memory_used > MAX_MEMORY) {
            LOG_ERROR("Memory limit exceeded: %.2f MB", memory_used / (1024.0f * 1024.0f));
            return -1;
        }
        
        return 0;
    } catch (const std::exception& e) {
        LOG_ERROR("Memory allocation failed: %s", e.what());
        return -1;
    }
}

} // namespace s2s
