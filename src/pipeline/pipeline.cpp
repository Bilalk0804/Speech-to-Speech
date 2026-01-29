#include "pipeline.h"

Pipeline::Pipeline() : initialized(false) {}

Pipeline::~Pipeline() {}

bool Pipeline::initialize() {
    initialized = true;
    return true;
}

void Pipeline::process() {
    if (!initialized) return;
}

void Pipeline::shutdown() {
    initialized = false;
}
