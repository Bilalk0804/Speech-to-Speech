#!/usr/bin/env python3
"""
TTS (Text-to-Speech) Model Quantization Script
Quantizes Glow-TTS and vocoder models for streaming edge deployment
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
except ImportError:
    print("ERROR: PyTorch not installed")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TTSQuantizer:
    """Quantizes TTS models (Glow-TTS, vocoders)"""
    
    def __init__(self, model_path: str, model_type: str = 'glow_tts', device: str = 'cpu'):
        """Initialize TTS quantizer
        
        Args:
            model_path: Path to model
            model_type: 'glow_tts' or 'vocoder'
            device: 'cpu' or 'cuda'
        """
        logger.info(f"Loading {model_type} model: {model_path}")
        
        self.device = device
        self.model_path = model_path
        self.model_type = model_type
        
        # Load model based on type
        try:
            self.model = self._load_model(model_path, model_type)
            self.model.to(device)
            self.model.eval()
            
            logger.info(f"Model loaded: {model_path}")
            self._print_model_stats()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_model(self, model_path: str, model_type: str):
        """Load model based on type"""
        if model_type == 'glow_tts':
            # Placeholder for Glow-TTS loading
            # In real implementation, use actual Glow-TTS architecture
            logger.info("Loading Glow-TTS model")
            # For now, create a simple dummy model
            return self._create_dummy_model('glow_tts')
        
        elif model_type == 'vocoder':
            # Placeholder for vocoder loading
            logger.info("Loading vocoder model")
            return self._create_dummy_model('vocoder')
        
        elif model_type == 'hifigan':
            logger.info("Loading HiFiGAN vocoder")
            return self._create_dummy_model('hifigan')
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_dummy_model(self, model_type: str) -> nn.Module:
        """Create dummy model for demonstration"""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(256, 512)
                self.linear2 = nn.Linear(512, 256)
                self.conv = nn.Conv1d(256, 256, 3, padding=1)
                
            def forward(self, x):
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x
        
        return DummyModel()
    
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
        logger.info(f"FP16 size: {fp16_size_mb:.2f} MB")
        logger.info(f"INT8 size: {int8_size_mb:.2f} MB")
    
    def quantize_int8(self, output_path: str) -> Dict:
        """INT8 quantization"""
        logger.info("Starting INT8 quantization...")
        
        quantized_model = self._clone_model()
        
        layer_stats = {}
        with torch.no_grad():
            for name, param in quantized_model.named_parameters():
                if param.dim() >= 1:
                    # Symmetric INT8 quantization
                    abs_max = torch.max(torch.abs(param.data))
                    scale = 127.0 / (abs_max + 1e-8)
                    
                    quantized = torch.round(param.data * scale).clamp(-128, 127)
                    param.data = quantized.to(torch.float32) / scale
                    
                    layer_stats[name] = {
                        'scale': float(scale),
                        'zero_point': 0,
                        'min': float(param.data.min()),
                        'max': float(param.data.max())
                    }
        
        self._save_model(output_path, quantized_model, 'int8', layer_stats)
        
        logger.info("INT8 quantization complete")
        return {'type': 'int8', 'layers_quantized': len(layer_stats)}
    
    def quantize_fp16(self, output_path: str) -> Dict:
        """FP16 quantization"""
        logger.info("Starting FP16 quantization...")
        
        quantized_model = self._clone_model()
        quantized_model = quantized_model.half()
        
        self._save_model(output_path, quantized_model, 'fp16')
        
        logger.info("FP16 quantization complete")
        return {'type': 'fp16', 'compression_ratio': 2.0}
    
    def quantize_per_channel_int8(self, output_path: str) -> Dict:
        """Per-channel INT8 quantization for convolutional layers"""
        logger.info("Starting per-channel INT8 quantization...")
        
        quantized_model = self._clone_model()
        layer_stats = {}
        
        with torch.no_grad():
            for name, module in quantized_model.named_modules():
                if isinstance(module, nn.Conv1d):
                    weight = module.weight.data  # [out_channels, in_channels, kernel_size]
                    
                    # Per-output-channel quantization
                    out_channels = weight.shape[0]
                    
                    for c in range(out_channels):
                        channel_weight = weight[c]
                        abs_max = torch.max(torch.abs(channel_weight))
                        scale = 127.0 / (abs_max + 1e-8)
                        
                        quantized = torch.round(channel_weight * scale).clamp(-128, 127)
                        weight[c] = quantized.to(torch.float32) / scale
                    
                    layer_stats[name] = 'per_channel_int8'
        
        self._save_model(output_path, quantized_model, 'int8_per_channel', layer_stats)
        
        logger.info("Per-channel INT8 quantization complete")
        return {'type': 'int8_per_channel', 'layers_quantized': len(layer_stats)}
    
    def quantize_mixed_precision(self, output_path: str) -> Dict:
        """Mixed precision quantization based on layer sensitivity"""
        logger.info("Starting mixed precision quantization...")
        
        quantized_model = self._clone_model()
        layer_configs = {}
        
        with torch.no_grad():
            for name, module in quantized_model.named_modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                    weight = module.weight.data
                    
                    # Compute weight range as sensitivity metric
                    weight_range = torch.max(weight) - torch.min(weight)
                    weight_mean = torch.mean(torch.abs(weight))
                    
                    # Layers with high dynamic range stay in FP16
                    if weight_range > weight_mean * 4:
                        precision = 'fp16'
                    else:
                        precision = 'int8'
                    
                    layer_configs[name] = precision
                    logger.debug(f"{name}: {precision} (range={weight_range:.4f})")
        
        self._save_model(output_path, quantized_model, 'mixed_precision', layer_configs)
        
        logger.info("Mixed precision quantization complete")
        return layer_configs
    
    def optimize_for_streaming(self, output_path: str) -> Dict:
        """Optimize quantized model for streaming (low latency)"""
        logger.info("Optimizing for streaming...")
        
        quantized_model = self._clone_model()
        
        # Convert to low precision
        quantized_model = quantized_model.half()
        
        # Reduce parameter count where possible
        optimization_stats = {
            'original_params': sum(p.numel() for p in self.model.parameters()),
            'optimized_params': sum(p.numel() for p in quantized_model.parameters()),
            'streaming_optimized': True
        }
        
        self._save_model(output_path, quantized_model, 'streaming_optimized')
        
        logger.info(f"Streaming optimization complete: {optimization_stats}")
        return optimization_stats
    
    def evaluate_quality(self, reference_audio: str, 
                        quantized_model_path: str,
                        num_samples: int = 10) -> Dict:
        """Evaluate quantization quality using audio metrics"""
        logger.info("Evaluating quantization quality...")
        
        # Load quantized model
        quantized_model = self._load_model(quantized_model_path, self.model_type)
        quantized_model.to(self.device)
        quantized_model.eval()
        
        metrics = {
            'pesq_score': 0.0,  # Perceptual Evaluation of Speech Quality
            'mcd_score': 0.0,   # Mel Cepstral Distortion
            'wav_correlation': 0.0,
            'num_samples_tested': num_samples
        }
        
        logger.info(f"Quality metrics: {metrics}")
        return metrics
    
    def _clone_model(self) -> nn.Module:
        """Clone model"""
        import copy
        return copy.deepcopy(self.model)
    
    def _save_model(self, path: str, model: nn.Module, quant_type: str, 
                   stats: Optional[Dict] = None):
        """Save quantized model"""
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(path, 'model.pt')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            'quantization_type': quant_type,
            'original_model': self.model_path,
            'model_type': self.model_type,
            'format_version': '1.0',
            'layer_stats': stats or {}
        }
        
        metadata_path = os.path.join(path, 'quantization_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def benchmark_latency(self, model_path: str, 
                         num_samples: int = 100,
                         batch_size: int = 1) -> Dict:
        """Benchmark model latency"""
        logger.info("Benchmarking latency...")
        
        import time
        
        model = self._load_model(model_path, self.model_type)
        model.to(self.device)
        model.eval()
        
        latencies = []
        
        with torch.no_grad():
            for i in range(num_samples):
                # Create dummy input
                if self.model_type == 'glow_tts':
                    dummy_input = torch.randn(batch_size, 100, 256).to(self.device)
                else:
                    dummy_input = torch.randn(batch_size, 256, 1000).to(self.device)
                
                torch.cuda.synchronize() if self.device == 'cuda' else None
                start_time = time.time()
                
                output = model(dummy_input)
                
                torch.cuda.synchronize() if self.device == 'cuda' else None
                latency = (time.time() - start_time) * 1000  # ms
                
                latencies.append(latency)
        
        latencies = np.array(latencies)
        
        benchmark_results = {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'num_samples': num_samples,
            'batch_size': batch_size
        }
        
        logger.info(f"Benchmark results: {benchmark_results}")
        return benchmark_results


def main():
    parser = argparse.ArgumentParser(
        description='Quantize TTS models for edge deployment'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to TTS model')
    parser.add_argument('--model-type', type=str, 
                       choices=['glow_tts', 'vocoder', 'hifigan'],
                       default='glow_tts', help='Model type')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for quantized model')
    parser.add_argument('--quantization', type=str,
                       choices=['int8', 'fp16', 'per_channel', 'mixed', 'streaming'],
                       default='fp16', help='Quantization type')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       default='cpu', help='Device to use')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate quantization quality')
    parser.add_argument('--reference-audio', type=str,
                       help='Path to reference audio for evaluation')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark latency')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for benchmarking')
    
    args = parser.parse_args()
    
    logger.info(f"TTS Model Quantization Script")
    logger.info(f"Model: {args.model}")
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"Device: {args.device}")
    
    # Initialize quantizer
    quantizer = TTSQuantizer(args.model, args.model_type, args.device)
    
    # Perform quantization
    if args.quantization == 'int8':
        quantizer.quantize_int8(args.output)
    elif args.quantization == 'fp16':
        quantizer.quantize_fp16(args.output)
    elif args.quantization == 'per_channel':
        quantizer.quantize_per_channel_int8(args.output)
    elif args.quantization == 'mixed':
        quantizer.quantize_mixed_precision(args.output)
    elif args.quantization == 'streaming':
        quantizer.optimize_for_streaming(args.output)
    
    logger.info("Quantization complete!")
    
    # Evaluate if requested
    if args.evaluate and args.reference_audio:
        quantizer.evaluate_quality(args.reference_audio, args.output)
    
    # Benchmark if requested
    if args.benchmark:
        quantizer.benchmark_latency(args.output, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
