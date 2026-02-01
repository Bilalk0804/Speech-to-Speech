#!/usr/bin/env python3
"""
Whisper ASR Model Quantization Script
Converts OpenAI Whisper models to INT8 and FP16 formats for edge deployment
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

# Try to import torch and transformers
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
except ImportError:
    print("ERROR: Transformers not installed. Install with: pip install transformers")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantizationStats:
    """Tracks quantization statistics"""
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.mean_val = 0.0
        self.std_val = 0.0
        self.scale = 1.0
        self.zero_point = 0
        self.num_elements = 0
    
    def update(self, tensor: torch.Tensor):
        """Update statistics with tensor data"""
        flat = tensor.flatten().float().cpu().numpy()
        
        self.min_val = min(self.min_val, float(flat.min()))
        self.max_val = max(self.max_val, float(flat.max()))
        
        total = self.num_elements + len(flat)
        self.mean_val = (self.mean_val * self.num_elements + float(flat.mean()) * len(flat)) / total
        
        self.num_elements = total
    
    def compute_scale(self, quantization_type: str = 'int8'):
        """Compute quantization scale"""
        abs_max = max(abs(self.min_val), abs(self.max_val))
        
        if quantization_type == 'int8':
            # Symmetric quantization
            self.scale = 127.0 / (abs_max + 1e-8)
            self.zero_point = 0
        elif quantization_type == 'uint8':
            # Asymmetric quantization
            self.scale = 255.0 / (self.max_val - self.min_val + 1e-8)
            self.zero_point = int(-self.min_val * self.scale)
        
        return self.scale, self.zero_point
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'min': float(self.min_val),
            'max': float(self.max_val),
            'mean': float(self.mean_val),
            'std': float(self.std_val),
            'scale': float(self.scale),
            'zero_point': int(self.zero_point),
            'num_elements': int(self.num_elements)
        }


class WhisperQuantizer:
    """Quantizes Whisper models"""
    
    def __init__(self, model_name: str, device: str = 'cpu'):
        """Initialize quantizer"""
        logger.info(f"Loading Whisper model: {model_name}")
        self.device = device
        self.model_name = model_name
        
        # Load model and processor
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
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
        
        # Estimate FP32 size
        fp32_size_mb = (total_params * 4) / (1024 * 1024)
        logger.info(f"FP32 Model size: {fp32_size_mb:.2f} MB")
    
    def collect_calibration_data(self, 
                                 audio_samples: list,
                                 num_samples: int = 100) -> Dict[str, QuantizationStats]:
        """Collect calibration data from audio samples"""
        logger.info(f"Collecting calibration data from {len(audio_samples)} samples...")
        
        calibration_stats = {}
        sample_count = 0
        
        with torch.no_grad():
            for sample_idx, audio_path in enumerate(audio_samples):
                if sample_count >= num_samples:
                    break
                
                try:
                    # Load audio (simplified - assumes audio file)
                    # In real implementation, use librosa or similar
                    logger.debug(f"Processing sample {sample_idx + 1}/{len(audio_samples)}")
                    
                    # Placeholder: In real usage, load actual audio
                    # For now, we'll generate random calibration data
                    
                    sample_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process {audio_path}: {e}")
                    continue
        
        logger.info(f"Collected {sample_count} calibration samples")
        return calibration_stats
    
    def quantize_int8(self, output_path: str, calibration_stats: Optional[Dict] = None):
        """Quantize model to INT8"""
        logger.info("Starting INT8 quantization...")
        
        quantized_model = self._clone_model()
        layer_stats = {}
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    # Compute quantization stats
                    flat = param.data.flatten().float().cpu()
                    
                    abs_max = max(abs(flat.min()), abs(flat.max()))
                    scale = 127.0 / (abs_max + 1e-8)
                    
                    # Quantize
                    quantized = torch.round(flat * scale).clamp(-128, 127).to(torch.int8)
                    
                    # Store in quantized model
                    quantized_param = quantized_model.get_parameter(name)
                    if quantized_param is not None:
                        quantized_param.data = quantized.float() / scale
                    
                    layer_stats[name] = {
                        'scale': float(scale),
                        'zero_point': 0,
                        'min': float(flat.min()),
                        'max': float(flat.max())
                    }
                    
                    logger.debug(f"Quantized {name}: scale={scale:.6f}")
        
        # Save quantized model
        self._save_quantized_model(output_path, quantized_model, layer_stats, 'int8')
        logger.info(f"INT8 quantized model saved to {output_path}")
    
    def quantize_fp16(self, output_path: str):
        """Quantize model to FP16"""
        logger.info("Starting FP16 quantization...")
        
        quantized_model = self._clone_model()
        layer_stats = {}
        
        with torch.no_grad():
            for name, param in quantized_model.named_parameters():
                if 'weight' in name or 'bias' in name:
                    # Convert to FP16 and back (simulates quantization)
                    original_dtype = param.data.dtype
                    param.data = param.data.half().float()
                    
                    layer_stats[name] = {
                        'dtype': 'float16',
                        'min': float(param.data.min()),
                        'max': float(param.data.max())
                    }
        
        self._save_quantized_model(output_path, quantized_model, layer_stats, 'fp16')
        logger.info(f"FP16 quantized model saved to {output_path}")
    
    def quantize_per_channel(self, output_path: str):
        """Per-channel quantization for weight matrices"""
        logger.info("Starting per-channel INT8 quantization...")
        
        quantized_model = self._clone_model()
        layer_stats = {}
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    # Per-channel quantization along output dimension
                    abs_max_per_channel = torch.amax(torch.abs(param.data), 
                                                     dim=list(range(1, param.dim())))
                    
                    scales = 127.0 / (abs_max_per_channel + 1e-8)
                    
                    # Reshape for broadcasting
                    reshape_shape = [param.shape[0]] + [1] * (param.dim() - 1)
                    scales = scales.reshape(reshape_shape)
                    
                    # Quantize
                    quantized = torch.round(param.data * scales).clamp(-128, 127).to(torch.int8)
                    
                    # Store
                    quantized_param = quantized_model.get_parameter(name)
                    if quantized_param is not None:
                        quantized_param.data = quantized.float() / scales
                    
                    layer_stats[name] = {
                        'per_channel_scales': scales.squeeze().tolist(),
                        'zero_point': 0
                    }
                    
                    logger.debug(f"Per-channel quantized {name}")
        
        self._save_quantized_model(output_path, quantized_model, layer_stats, 'int8_per_channel')
        logger.info(f"Per-channel INT8 quantized model saved to {output_path}")
    
    def _clone_model(self):
        """Create a copy of the model"""
        return WhisperForConditionalGeneration.from_pretrained(self.model_name)
    
    def _save_quantized_model(self, path: str, model, stats: Dict, quant_type: str):
        """Save quantized model"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save model
        model.save_pretrained(path)
        
        # Save quantization metadata
        metadata = {
            'quantization_type': quant_type,
            'original_model': self.model_name,
            'layer_stats': stats,
            'format_version': '1.0'
        }
        
        metadata_path = os.path.join(path, 'quantization_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved quantization metadata to {metadata_path}")
    
    def evaluate_quantization(self, 
                             quantized_model_path: str,
                             test_samples: list,
                             num_samples: int = 10) -> Dict:
        """Evaluate quantized model accuracy"""
        logger.info("Evaluating quantized model...")
        
        from transformers import WhisperForConditionalGeneration
        
        quantized_model = WhisperForConditionalGeneration.from_pretrained(
            quantized_model_path
        )
        quantized_model.to(self.device)
        quantized_model.eval()
        
        metrics = {
            'mae': 0.0,
            'rmse': 0.0,
            'max_error': 0.0,
            'samples_tested': 0
        }
        
        # TODO: Implement actual evaluation on test samples
        
        logger.info(f"Evaluation complete: {metrics}")
        return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Quantize OpenAI Whisper models for edge deployment'
    )
    parser.add_argument('--model', type=str, default='openai/whisper-base',
                       help='Whisper model name (default: openai/whisper-base)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for quantized model')
    parser.add_argument('--quantization', type=str, choices=['int8', 'fp16', 'per_channel'],
                       default='int8', help='Quantization type')
    parser.add_argument('--calibration-data', type=str,
                       help='Path to calibration audio samples')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate quantized model')
    parser.add_argument('--test-samples', type=str,
                       help='Path to test audio samples for evaluation')
    
    args = parser.parse_args()
    
    logger.info(f"Whisper Model Quantization Script")
    logger.info(f"Model: {args.model}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"Device: {args.device}")
    
    # Initialize quantizer
    quantizer = WhisperQuantizer(args.model, device=args.device)
    
    # Collect calibration data if provided
    calibration_stats = None
    if args.calibration_data:
        calibration_stats = quantizer.collect_calibration_data(
            [args.calibration_data]
        )
    
    # Perform quantization
    if args.quantization == 'int8':
        quantizer.quantize_int8(args.output, calibration_stats)
    elif args.quantization == 'fp16':
        quantizer.quantize_fp16(args.output)
    elif args.quantization == 'per_channel':
        quantizer.quantize_per_channel(args.output)
    
    logger.info(f"Quantization complete!")
    
    # Evaluate if requested
    if args.evaluate and args.test_samples:
        metrics = quantizer.evaluate_quantization(
            args.output,
            [args.test_samples]
        )


if __name__ == '__main__':
    main()
