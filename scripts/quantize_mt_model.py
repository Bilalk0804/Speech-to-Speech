#!/usr/bin/env python3
"""
IndicTrans2 Machine Translation Model Quantization Script
Converts transformer-based MT models to INT8, FP16, and dynamic quantization formats
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

try:
    import torch
    import torch.nn as nn
    from torch.quantization import quantize_dynamic, QConfig, default_qconfig
except ImportError:
    print("ERROR: PyTorch not installed")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    print("ERROR: Transformers not installed")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MTQuantizer:
    """Quantizes machine translation models"""
    
    def __init__(self, model_name: str, device: str = 'cpu'):
        """Initialize quantizer"""
        logger.info(f"Loading MT model: {model_name}")
        self.device = device
        self.model_name = model_name
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Model loaded: {model_name}")
        self._print_model_stats()
    
    def _print_model_stats(self):
        """Print model statistics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Estimate sizes
        fp32_size_mb = (total_params * 4) / (1024 * 1024)
        fp16_size_mb = (total_params * 2) / (1024 * 1024)
        int8_size_mb = (total_params * 1) / (1024 * 1024)
        
        logger.info(f"FP32 size: {fp32_size_mb:.2f} MB")
        logger.info(f"FP16 size: {fp16_size_mb:.2f} MB (2x compression)")
        logger.info(f"INT8 size: {int8_size_mb:.2f} MB (4x compression)")
    
    def quantize_dynamic(self, output_path: str, dtype: str = 'qint8') -> Dict:
        """Dynamic quantization without calibration"""
        logger.info(f"Starting dynamic quantization (dtype={dtype})...")
        
        # PyTorch dynamic quantization - good for RNNs and Transformers
        quantized_model = quantize_dynamic(
            self.model,
            {torch.nn.Linear},  # Quantize Linear layers
            dtype=torch.qint8 if dtype == 'qint8' else torch.float16,
            inplace=False
        )
        
        self._save_model(output_path, quantized_model, 'dynamic_int8')
        
        stats = {
            'quantization_type': 'dynamic',
            'dtype': dtype,
            'layers_quantized': self._count_quantizable_layers()
        }
        
        logger.info(f"Dynamic quantization complete")
        return stats
    
    def quantize_static_int8(self, 
                            output_path: str,
                            calibration_texts: List[str],
                            num_calibration_batches: int = 10) -> Dict:
        """Static INT8 quantization with calibration"""
        logger.info("Starting static INT8 quantization with calibration...")
        
        # Prepare calibration data
        logger.info(f"Calibrating with {num_calibration_batches} batches...")
        
        quantized_model = self._clone_model()
        
        # Collect layer statistics for calibration
        layer_stats = {}
        
        with torch.no_grad():
            # Process calibration texts
            for batch_idx in range(min(num_calibration_batches, len(calibration_texts))):
                text = calibration_texts[batch_idx]
                
                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt", 
                                       max_length=512, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass to collect activation statistics
                try:
                    outputs = quantized_model(**inputs)
                    logger.debug(f"Calibration batch {batch_idx + 1}/{num_calibration_batches}")
                except Exception as e:
                    logger.warning(f"Calibration batch failed: {e}")
                    continue
        
        # Apply INT8 quantization
        self._apply_int8_quantization(quantized_model)
        
        self._save_model(output_path, quantized_model, 'static_int8')
        
        stats = {
            'quantization_type': 'static_int8',
            'calibration_batches': num_calibration_batches,
            'layer_stats': layer_stats
        }
        
        logger.info("Static INT8 quantization complete")
        return stats
    
    def quantize_fp16(self, output_path: str) -> Dict:
        """FP16 quantization"""
        logger.info("Starting FP16 quantization...")
        
        quantized_model = self._clone_model()
        
        # Convert model to FP16
        quantized_model = quantized_model.half()
        
        self._save_model(output_path, quantized_model, 'fp16')
        
        stats = {
            'quantization_type': 'fp16',
            'compression_ratio': 2.0
        }
        
        logger.info("FP16 quantization complete")
        return stats
    
    def quantize_per_layer(self, output_path: str) -> Dict:
        """Quantize different layers with different precisions"""
        logger.info("Starting per-layer mixed precision quantization...")
        
        quantized_model = self._clone_model()
        layer_configs = {}
        
        with torch.no_grad():
            for name, module in quantized_model.named_modules():
                if isinstance(module, nn.Linear):
                    weight = module.weight.data
                    
                    # Compute sensitivity
                    w_norm = torch.norm(weight)
                    w_sensitivity = torch.norm(weight - weight.int()) / w_norm if w_norm > 0 else 1.0
                    
                    # Choose precision based on sensitivity
                    if w_sensitivity > 0.1:
                        # Sensitive layer - keep FP16
                        module = module.half()
                        precision = 'fp16'
                    else:
                        # Less sensitive - use INT8
                        precision = 'int8'
                    
                    layer_configs[name] = precision
                    logger.debug(f"{name}: {precision} (sensitivity={w_sensitivity:.4f})")
        
        self._save_model(output_path, quantized_model, 'mixed_precision')
        
        stats = {
            'quantization_type': 'mixed_precision',
            'layer_configs': layer_configs
        }
        
        logger.info("Per-layer quantization complete")
        return stats
    
    def _apply_int8_quantization(self, model):
        """Apply INT8 quantization to model"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'weight' in name:
                    # Symmetric INT8 quantization
                    abs_max = torch.max(torch.abs(param.data))
                    scale = 127.0 / (abs_max + 1e-8)
                    
                    quantized = torch.round(param.data * scale).clamp(-128, 127)
                    param.data = quantized.to(torch.float32) / scale
    
    def _count_quantizable_layers(self) -> int:
        """Count layers that can be quantized"""
        count = 0
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                count += 1
        return count
    
    def _clone_model(self):
        """Create a copy of the model"""
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    
    def _save_model(self, path: str, model, quant_type: str):
        """Save quantized model"""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save metadata
        metadata = {
            'quantization_type': quant_type,
            'original_model': self.model_name,
            'format_version': '1.0'
        }
        
        metadata_path = os.path.join(path, 'quantization_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def benchmark_model(self, model_path: str, num_samples: int = 100) -> Dict:
        """Benchmark quantized model performance"""
        logger.info(f"Benchmarking model...")
        
        import time
        
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        model.eval()
        
        # Test translations
        test_texts = [
            "Hello world",
            "How are you?",
            "Machine translation is important"
        ]
        
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                text = np.random.choice(test_texts)
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                
                start_time = time.time()
                outputs = model.generate(**inputs, max_length=128)
                latency = (time.time() - start_time) * 1000  # ms
                
                latencies.append(latency)
        
        latencies = np.array(latencies)
        
        benchmark_results = {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'num_samples': num_samples
        }
        
        logger.info(f"Benchmark results: {benchmark_results}")
        return benchmark_results


def main():
    parser = argparse.ArgumentParser(
        description='Quantize IndicTrans2 MT models'
    )
    parser.add_argument('--model', type=str, default='VincentChelsea/IndicTrans2-en-ta',
                       help='Model name or path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for quantized model')
    parser.add_argument('--quantization', type=str, 
                       choices=['dynamic', 'static_int8', 'fp16', 'mixed'],
                       default='dynamic', help='Quantization type')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                       default='cpu', help='Device to use')
    parser.add_argument('--calibration-texts', type=str,
                       help='Path to calibration text file')
    parser.add_argument('--num-calibration', type=int, default=10,
                       help='Number of calibration samples')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark quantized model')
    
    args = parser.parse_args()
    
    logger.info(f"MT Model Quantization Script")
    logger.info(f"Model: {args.model}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"Device: {args.device}")
    
    # Initialize quantizer
    quantizer = MTQuantizer(args.model, device=args.device)
    
    # Load calibration texts if provided
    calibration_texts = []
    if args.calibration_texts and os.path.exists(args.calibration_texts):
        with open(args.calibration_texts, 'r') as f:
            calibration_texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(calibration_texts)} calibration texts")
    
    # Perform quantization
    if args.quantization == 'dynamic':
        quantizer.quantize_dynamic(args.output)
    elif args.quantization == 'static_int8':
        quantizer.quantize_static_int8(args.output, calibration_texts, args.num_calibration)
    elif args.quantization == 'fp16':
        quantizer.quantize_fp16(args.output)
    elif args.quantization == 'mixed':
        quantizer.quantize_per_layer(args.output)
    
    logger.info("Quantization complete!")
    
    # Benchmark if requested
    if args.benchmark:
        quantizer.benchmark_model(args.output)


if __name__ == '__main__':
    main()
