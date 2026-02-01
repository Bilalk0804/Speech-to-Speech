#!/usr/bin/env python3
"""
Master Quantization Runner
Orchestrates quantization of all models (Whisper, IndicTrans2, TTS)
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List
import subprocess
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantizationRunner:
    """Orchestrates model quantization"""
    
    def __init__(self, config_path: str = None):
        """Initialize runner"""
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.quantization_results = {}
        self.start_time = None
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'output_dir': 'models/quantized',
            'whisper': {
                'enabled': True,
                'model': 'openai/whisper-base',
                'quantization_types': ['int8', 'fp16', 'per_channel'],
                'device': 'cpu'
            },
            'mt_model': {
                'enabled': True,
                'model': 'VincentChelsea/IndicTrans2-en-ta',
                'quantization_types': ['dynamic', 'static_int8', 'fp16', 'mixed'],
                'device': 'cpu',
                'calibration_samples': 100
            },
            'tts': {
                'enabled': True,
                'models': [
                    {
                        'name': 'glow_tts',
                        'path': 'models/glow_tts.pt',
                        'quantization_types': ['fp16', 'mixed']
                    },
                    {
                        'name': 'hifigan_vocoder',
                        'path': 'models/hifigan_vocoder.pt',
                        'quantization_types': ['int8', 'per_channel', 'streaming']
                    }
                ],
                'device': 'cpu'
            },
            'evaluation': {
                'benchmark': True,
                'accuracy_check': True,
                'num_test_samples': 10
            }
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON"""
        logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def run_all(self):
        """Run all quantizations"""
        self.start_time = time.time()
        logger.info("=" * 70)
        logger.info("Starting Comprehensive Model Quantization")
        logger.info("=" * 70)
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Quantize each model type
        if self.config['whisper']['enabled']:
            self.quantize_whisper()
        
        if self.config['mt_model']['enabled']:
            self.quantize_mt_model()
        
        if self.config['tts']['enabled']:
            self.quantize_tts()
        
        # Print summary
        self.print_summary()
    
    def quantize_whisper(self):
        """Quantize Whisper ASR model"""
        logger.info("\n" + "=" * 70)
        logger.info("QUANTIZING WHISPER ASR MODEL")
        logger.info("=" * 70)
        
        whisper_config = self.config['whisper']
        model_name = whisper_config['model'].replace('/', '_')
        
        for quant_type in whisper_config['quantization_types']:
            output_path = os.path.join(
                self.config['output_dir'],
                f"whisper_{model_name}_{quant_type}"
            )
            
            logger.info(f"\nQuantizing Whisper to {quant_type.upper()}...")
            
            try:
                cmd = [
                    'python3', 'scripts/quantize_whisper.py',
                    '--model', whisper_config['model'],
                    '--output', output_path,
                    '--quantization', quant_type,
                    '--device', whisper_config['device']
                ]
                
                if self.config['evaluation']['benchmark']:
                    cmd.append('--evaluate')
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"✓ Whisper {quant_type} quantization completed")
                    self.quantization_results[f'whisper_{quant_type}'] = {
                        'status': 'success',
                        'output_path': output_path
                    }
                else:
                    logger.error(f"✗ Whisper {quant_type} quantization failed")
                    logger.error(result.stderr)
                    self.quantization_results[f'whisper_{quant_type}'] = {
                        'status': 'failed',
                        'error': result.stderr
                    }
            
            except Exception as e:
                logger.error(f"Exception during Whisper quantization: {e}")
                self.quantization_results[f'whisper_{quant_type}'] = {
                    'status': 'error',
                    'error': str(e)
                }
    
    def quantize_mt_model(self):
        """Quantize Machine Translation model"""
        logger.info("\n" + "=" * 70)
        logger.info("QUANTIZING MACHINE TRANSLATION MODEL")
        logger.info("=" * 70)
        
        mt_config = self.config['mt_model']
        model_name = mt_config['model'].replace('/', '_')
        
        for quant_type in mt_config['quantization_types']:
            output_path = os.path.join(
                self.config['output_dir'],
                f"mt_{model_name}_{quant_type}"
            )
            
            logger.info(f"\nQuantizing MT model to {quant_type.upper()}...")
            
            try:
                cmd = [
                    'python3', 'scripts/quantize_mt_model.py',
                    '--model', mt_config['model'],
                    '--output', output_path,
                    '--quantization', quant_type,
                    '--device', mt_config['device'],
                    '--num-calibration', str(mt_config['calibration_samples'])
                ]
                
                if self.config['evaluation']['benchmark']:
                    cmd.append('--benchmark')
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"✓ MT model {quant_type} quantization completed")
                    self.quantization_results[f'mt_{quant_type}'] = {
                        'status': 'success',
                        'output_path': output_path
                    }
                else:
                    logger.error(f"✗ MT model {quant_type} quantization failed")
                    logger.error(result.stderr)
                    self.quantization_results[f'mt_{quant_type}'] = {
                        'status': 'failed',
                        'error': result.stderr
                    }
            
            except Exception as e:
                logger.error(f"Exception during MT quantization: {e}")
                self.quantization_results[f'mt_{quant_type}'] = {
                    'status': 'error',
                    'error': str(e)
                }
    
    def quantize_tts(self):
        """Quantize TTS models"""
        logger.info("\n" + "=" * 70)
        logger.info("QUANTIZING TTS MODELS")
        logger.info("=" * 70)
        
        tts_config = self.config['tts']
        
        for model_info in tts_config['models']:
            model_name = model_info['name']
            model_path = model_info['path']
            
            if not os.path.exists(model_path):
                logger.warning(f"TTS model not found: {model_path}, skipping")
                continue
            
            for quant_type in model_info['quantization_types']:
                output_path = os.path.join(
                    self.config['output_dir'],
                    f"tts_{model_name}_{quant_type}"
                )
                
                logger.info(f"\nQuantizing {model_name} to {quant_type.upper()}...")
                
                try:
                    cmd = [
                        'python3', 'scripts/quantize_tts_model.py',
                        '--model', model_path,
                        '--model-type', model_name,
                        '--output', output_path,
                        '--quantization', quant_type,
                        '--device', tts_config['device']
                    ]
                    
                    if self.config['evaluation']['benchmark']:
                        cmd.append('--benchmark')
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"✓ {model_name} {quant_type} quantization completed")
                        self.quantization_results[f'tts_{model_name}_{quant_type}'] = {
                            'status': 'success',
                            'output_path': output_path
                        }
                    else:
                        logger.error(f"✗ {model_name} {quant_type} quantization failed")
                        logger.error(result.stderr)
                        self.quantization_results[f'tts_{model_name}_{quant_type}'] = {
                            'status': 'failed',
                            'error': result.stderr
                        }
                
                except Exception as e:
                    logger.error(f"Exception during TTS quantization: {e}")
                    self.quantization_results[f'tts_{model_name}_{quant_type}'] = {
                        'status': 'error',
                        'error': str(e)
                    }
    
    def print_summary(self):
        """Print quantization summary"""
        elapsed_time = time.time() - self.start_time
        
        logger.info("\n" + "=" * 70)
        logger.info("QUANTIZATION SUMMARY")
        logger.info("=" * 70)
        
        successful = sum(1 for r in self.quantization_results.values() if r['status'] == 'success')
        failed = sum(1 for r in self.quantization_results.values() if r['status'] != 'success')
        
        logger.info(f"\nTotal quantizations: {len(self.quantization_results)}")
        logger.info(f"✓ Successful: {successful}")
        logger.info(f"✗ Failed: {failed}")
        logger.info(f"Total time: {elapsed_time:.1f}s")
        
        # Detailed results
        logger.info("\nDetailed Results:")
        for name, result in self.quantization_results.items():
            status_symbol = "✓" if result['status'] == 'success' else "✗"
            logger.info(f"  {status_symbol} {name}: {result['status']}")
            if result['status'] == 'success':
                logger.info(f"    Output: {result['output_path']}")
        
        # Save summary to file
        summary_path = os.path.join(self.config['output_dir'], 'quantization_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_time_seconds': elapsed_time,
                'successful': successful,
                'failed': failed,
                'results': self.quantization_results
            }, f, indent=2)
        
        logger.info(f"\nSummary saved to: {summary_path}")
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Master Quantization Runner - Quantize all S2S models'
    )
    parser.add_argument('--config', type=str,
                       help='Configuration JSON file')
    parser.add_argument('--output-dir', type=str, default='models/quantized',
                       help='Output directory for quantized models')
    parser.add_argument('--whisper-only', action='store_true',
                       help='Only quantize Whisper')
    parser.add_argument('--mt-only', action='store_true',
                       help='Only quantize MT model')
    parser.add_argument('--tts-only', action='store_true',
                       help='Only quantize TTS models')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = QuantizationRunner(args.config)
    
    # Override config if arguments provided
    if args.output_dir:
        runner.config['output_dir'] = args.output_dir
    
    if args.device:
        runner.config['whisper']['device'] = args.device
        runner.config['mt_model']['device'] = args.device
        runner.config['tts']['device'] = args.device
    
    # Disable models if specific ones requested
    if args.whisper_only:
        runner.config['mt_model']['enabled'] = False
        runner.config['tts']['enabled'] = False
    elif args.mt_only:
        runner.config['whisper']['enabled'] = False
        runner.config['tts']['enabled'] = False
    elif args.tts_only:
        runner.config['whisper']['enabled'] = False
        runner.config['mt_model']['enabled'] = False
    
    # Run quantization
    runner.run_all()


if __name__ == '__main__':
    main()
