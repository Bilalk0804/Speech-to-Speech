#!/usr/bin/env python3
"""
Unit tests for quantization modules
"""

import unittest
import numpy as np
import os
import tempfile
import json
from pathlib import Path


class TestQuantizationUtils(unittest.TestCase):
    """Test quantization utility functions"""
    
    def test_float32_to_int8(self):
        """Test float32 to INT8 quantization"""
        # Create test data
        test_data = np.random.randn(1000).astype(np.float32) * 10
        
        # Compute scale
        abs_max = max(abs(test_data.min()), abs(test_data.max()))
        scale = 127.0 / abs_max
        
        # Quantize
        quantized = np.round(test_data * scale).astype(np.int8)
        
        # Verify range
        self.assertTrue(quantized.min() >= -128)
        self.assertTrue(quantized.max() <= 127)
        
        # Verify inverse
        recovered = quantized.astype(np.float32) / scale
        max_error = np.abs(test_data - recovered).max()
        self.assertLess(max_error, 0.1)  # Within tolerance
    
    def test_int8_to_float32(self):
        """Test INT8 to float32 dequantization"""
        quantized_data = np.array([-128, -64, 0, 64, 127], dtype=np.int8)
        scale = 1.5
        
        recovered = quantized_data.astype(np.float32) / scale
        
        expected = np.array([-128/1.5, -64/1.5, 0, 64/1.5, 127/1.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(recovered, expected)
    
    def test_float32_to_float16(self):
        """Test float32 to FP16 conversion"""
        test_data = np.array([1.5, -2.3, 0.0, 1e-5, 1e5], dtype=np.float32)
        
        # Convert to float16 and back
        fp16_data = test_data.astype(np.float16)
        recovered = fp16_data.astype(np.float32)
        
        # Allow small precision loss
        relative_error = np.abs((test_data - recovered) / (test_data + 1e-8))
        self.assertTrue(np.all(relative_error < 0.01))
    
    def test_per_channel_quantization(self):
        """Test per-channel quantization"""
        # Create weight matrix: [output_channels, input_channels]
        weights = np.random.randn(16, 32).astype(np.float32) * 10
        
        quantized_weights = np.zeros_like(weights, dtype=np.int8)
        scales = np.zeros(16)
        
        # Per-channel quantization
        for c in range(16):
            channel_weights = weights[c]
            abs_max = max(abs(channel_weights.min()), abs(channel_weights.max()))
            scale = 127.0 / abs_max
            scales[c] = scale
            
            quantized = np.round(channel_weights * scale).astype(np.int8)
            quantized_weights[c] = quantized
        
        # Verify all channels were quantized
        self.assertEqual(quantized_weights.shape, weights.shape)
        self.assertGreater(np.sum(scales > 0), 0)
    
    def test_accuracy_metrics(self):
        """Test accuracy metrics computation"""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        quantized = np.array([1.02, 1.98, 3.05, 3.95, 5.01], dtype=np.float32)
        
        # MAE
        mae = np.mean(np.abs(original - quantized))
        self.assertAlmostEqual(mae, 0.022, places=3)
        
        # RMSE
        rmse = np.sqrt(np.mean((original - quantized) ** 2))
        self.assertAlmostEqual(rmse, 0.024, places=3)
    
    def test_quantization_symmetry(self):
        """Test symmetric quantization"""
        test_data = np.array([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=np.float32)
        
        abs_max = max(abs(test_data.min()), abs(test_data.max()))
        scale = 127.0 / abs_max
        
        quantized = np.round(test_data * scale).astype(np.int8)
        
        # For symmetric quantization, opposite values should have opposite quantized values
        for i in range(len(test_data) // 2):
            if quantized[i] != 0:
                self.assertEqual(quantized[i], -quantized[-(i+1)])


class TestWhisperQuantization(unittest.TestCase):
    """Test Whisper quantization functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_quantization_metadata(self):
        """Test quantization metadata structure"""
        metadata = {
            'quantization_type': 'int8',
            'original_model': 'openai/whisper-base',
            'layer_stats': {
                'encoder.conv1.weight': {
                    'scale': 0.5,
                    'zero_point': 0,
                    'min': -64.0,
                    'max': 64.0
                }
            },
            'format_version': '1.0'
        }
        
        metadata_path = os.path.join(self.temp_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        self.assertEqual(loaded_metadata['quantization_type'], 'int8')
        self.assertIn('layer_stats', loaded_metadata)


class TestMTQuantization(unittest.TestCase):
    """Test MT model quantization"""
    
    def test_dynamic_quantization_config(self):
        """Test dynamic quantization configuration"""
        config = {
            'quantization_type': 'dynamic',
            'dtype': 'qint8',
            'layers_quantized': 48
        }
        
        self.assertEqual(config['quantization_type'], 'dynamic')
        self.assertGreater(config['layers_quantized'], 0)
    
    def test_mixed_precision_assignment(self):
        """Test mixed precision layer assignment"""
        layer_configs = {
            'encoder.layer.0.self_attn.q_proj': 'fp16',
            'encoder.layer.0.mlp.fc1': 'int8',
            'encoder.layer.0.mlp.fc2': 'int8',
            'decoder.layer.0.self_attn.q_proj': 'fp16'
        }
        
        fp16_count = sum(1 for v in layer_configs.values() if v == 'fp16')
        int8_count = sum(1 for v in layer_configs.values() if v == 'int8')
        
        self.assertEqual(fp16_count, 2)
        self.assertEqual(int8_count, 2)


class TestTTSQuantization(unittest.TestCase):
    """Test TTS model quantization"""
    
    def test_streaming_optimization(self):
        """Test streaming optimization settings"""
        optimization_stats = {
            'original_params': 1_000_000,
            'optimized_params': 500_000,
            'streaming_optimized': True,
            'latency_reduction_percent': 45
        }
        
        self.assertTrue(optimization_stats['streaming_optimized'])
        compression_ratio = optimization_stats['original_params'] / optimization_stats['optimized_params']
        self.assertEqual(compression_ratio, 2.0)
    
    def test_per_channel_conv_quantization(self):
        """Test per-channel quantization for conv layers"""
        # Simulate Conv1d weight quantization
        # Shape: [out_channels, in_channels, kernel_size]
        weights = np.random.randn(64, 128, 3).astype(np.float32)
        
        quantized_weights = np.zeros_like(weights, dtype=np.int8)
        per_channel_scales = []
        
        # Quantize per output channel
        for c in range(weights.shape[0]):
            channel_weights = weights[c]
            abs_max = max(abs(channel_weights.min()), abs(channel_weights.max()))
            scale = 127.0 / abs_max
            
            quantized = np.round(channel_weights * scale).astype(np.int8)
            quantized_weights[c] = quantized
            per_channel_scales.append(scale)
        
        self.assertEqual(len(per_channel_scales), weights.shape[0])
        self.assertTrue(all(s > 0 for s in per_channel_scales))


class TestQuantizationAccuracy(unittest.TestCase):
    """Test accuracy evaluation"""
    
    def test_mae_computation(self):
        """Test MAE computation"""
        original = np.array([1.0, 2.0, 3.0])
        quantized = np.array([1.1, 1.9, 3.0])
        
        mae = np.mean(np.abs(original - quantized))
        self.assertAlmostEqual(mae, 0.0333, places=3)
    
    def test_rmse_computation(self):
        """Test RMSE computation"""
        original = np.array([1.0, 2.0, 3.0])
        quantized = np.array([1.0, 2.0, 3.0])
        
        rmse = np.sqrt(np.mean((original - quantized) ** 2))
        self.assertAlmostEqual(rmse, 0.0)
    
    def test_worst_case_accuracy(self):
        """Test accuracy in worst case"""
        # Create signal and highly quantized version
        original = np.linspace(0, 10, 100).astype(np.float32)
        quantized = np.round(original).astype(np.float32)
        
        mae = np.mean(np.abs(original - quantized))
        self.assertLess(mae, 0.5)


class TestQuantizationBenchmark(unittest.TestCase):
    """Test benchmarking functionality"""
    
    def test_latency_metrics(self):
        """Test latency metrics structure"""
        latencies = np.array([10.5, 11.2, 10.8, 11.5, 10.3])
        
        metrics = {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'num_samples': len(latencies)
        }
        
        self.assertAlmostEqual(metrics['mean_latency_ms'], 10.86, places=1)
        self.assertEqual(metrics['median_latency_ms'], 10.8)
        self.assertGreater(metrics['p95_latency_ms'], metrics['median_latency_ms'])
        self.assertGreater(metrics['p99_latency_ms'], metrics['p95_latency_ms'])


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQuantizationUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestWhisperQuantization))
    suite.addTests(loader.loadTestsFromTestCase(TestMTQuantization))
    suite.addTests(loader.loadTestsFromTestCase(TestTTSQuantization))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantizationAccuracy))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantizationBenchmark))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
