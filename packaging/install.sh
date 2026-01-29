#!/bin/bash
# Installation script for S2S On-Device

set -e

echo "Installing S2S On-Device..."

# Create build directory
mkdir -p build
cd build

# Configure
cmake ..

# Build
make -j$(nproc)

# Install
make install

echo "Installation complete!"
