import os
import torch
import torch.nn as nn
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, ListConfig
import logging
import json
import timm
from transformers import ViTModel
import torch.cuda.profiler as profiler
import gc

log = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    model_name: str
    model_config: str
    model_size_mb: float
    throughput_gbps: float
    latency_ms: float
    gpu_mem_usage_mb: float

class ModelFactory:
    @staticmethod
    def create_linear(input_size: int, hidden_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size)
        )

    @staticmethod
    def create_mlp(input_size: int, hidden_sizes: List[int]) -> nn.Module:
        layers = []
        prev_size = input_size
        # Convert OmegaConf list to Python list if necessary
        if isinstance(hidden_sizes, ListConfig):
            hidden_sizes = [int(size) for size in hidden_sizes]
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        return nn.Sequential(nn.Flatten(), *layers)

    @staticmethod
    def create_resnet(in_channels: int, num_blocks: int, base_channels: int) -> nn.Module:
        from models import ResNetScaler
        return ResNetScaler(in_channels, num_blocks, base_channels)

    @staticmethod
    def create_convnext(variant: str, in_channels: int) -> nn.Module:
        return timm.create_model(variant, pretrained=False, in_chans=in_channels)

    @staticmethod
    def create_vit(config: dict, image_size: int, in_channels: int) -> nn.Module:
        # Convert OmegaConf dict to Python dict if necessary
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config)
        return ViTModel(
            image_size=image_size,
            patch_size=16,
            num_channels=in_channels,
            **config
        )

def get_model_size(model: nn.Module) -> float:
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024

def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int,
    num_warmup: int,
    num_iterations: int,
    device: str
) -> BenchmarkResult:
    """Run benchmark for a single model configuration"""
    # Clear GPU cache and run garbage collection
    torch.cuda.empty_cache()
    gc.collect()

    model = model.to(device)
    model.eval()

    # Store model info before potential cleanup
    model_name = model.__class__.__name__
    model_config = str(model)

    # Create random input data
    inputs = torch.randn(batch_size, *input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(inputs)

    torch.cuda.synchronize()

    # Reset GPU stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    # Calculate metrics
    avg_latency = np.mean(latencies)
    throughput = (batch_size * np.prod(input_shape) * 4) / (avg_latency / 1000) / 1e9  # GB/s
    model_size = get_model_size(model)
    gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # Clean up
    del model, inputs
    torch.cuda.empty_cache()
    gc.collect()

    return BenchmarkResult(
        model_name=model_name,
        model_config=model_config,
        model_size_mb=model_size,
        throughput_gbps=throughput,
        latency_ms=avg_latency,
        gpu_mem_usage_mb=gpu_mem
    )

@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_benchmark(cfg: DictConfig) -> None:
    log.info(f"Running benchmark with config:\n{OmegaConf.to_yaml(cfg)}")

    input_size = np.prod(cfg.input.shape)
    results = []

    try:
        if cfg.model.type == "linear":
            log.info("Benchmarking Linear models")
            for hidden_size in cfg.model.hidden_sizes:
                log.info(f"Testing Linear model with hidden size: {hidden_size}")
                model = ModelFactory.create_linear(input_size, int(hidden_size))
                result = benchmark_model(
                    model,
                    tuple(cfg.input.shape),
                    cfg.input.batch_size,
                    cfg.benchmark.num_warmup,
                    cfg.benchmark.num_iterations,
                    cfg.benchmark.device
                )
                results.append(result)

        elif cfg.model.type == "mlp":
            log.info("Benchmarking MLP models")
            for hidden_sizes in cfg.model.architectures:
                log.info(f"Testing MLP with architecture: {hidden_sizes}")
                # Convert OmegaConf list to Python list
                hidden_sizes_list = [int(size) for size in hidden_sizes]
                model = ModelFactory.create_mlp(input_size, hidden_sizes_list)
                result = benchmark_model(
                    model,
                    tuple(cfg.input.shape),
                    cfg.input.batch_size,
                    cfg.benchmark.num_warmup,
                    cfg.benchmark.num_iterations,
                    cfg.benchmark.device
                )
                results.append(result)

        elif cfg.model.type == "resnet":
            log.info("Benchmarking ResNet models")
            for num_blocks in cfg.model.num_blocks:
                log.info(f"Testing ResNet with {num_blocks} blocks")
                model = ModelFactory.create_resnet(
                    cfg.input.shape[0],
                    int(num_blocks),
                    int(cfg.model.base_channels)
                )
                result = benchmark_model(
                    model,
                    tuple(cfg.input.shape),
                    cfg.input.batch_size,
                    cfg.benchmark.num_warmup,
                    cfg.benchmark.num_iterations,
                    cfg.benchmark.device
                )
                results.append(result)

        elif cfg.model.type == "convnext":
            log.info("Benchmarking ConvNeXt models")
            for variant in cfg.model.variants:
                log.info(f"Testing ConvNeXt variant: {variant}")
                try:
                    model = ModelFactory.create_convnext(
                        str(variant),
                        cfg.input.shape[0]
                    )
                    result = benchmark_model(
                        model,
                        tuple(cfg.input.shape),
                        cfg.input.batch_size,
                        cfg.benchmark.num_warmup,
                        cfg.benchmark.num_iterations,
                        cfg.benchmark.device
                    )
                    results.append(result)
                except Exception as e:
                    log.error(f"Error testing ConvNeXt variant {variant}: {str(e)}")
                    continue

        elif cfg.model.type == "vit":
            log.info("Benchmarking Vision Transformer models")
            for config in cfg.model.configs:
                log.info(f"Testing ViT with config: {config}")
                try:
                    model = ModelFactory.create_vit(
                        config,
                        max(cfg.input.shape[1:]),  # Use larger dimension as image size
                        cfg.input.shape[0]
                    )
                    result = benchmark_model(
                        model,
                        tuple(cfg.input.shape),
                        cfg.input.batch_size,
                        cfg.benchmark.num_warmup,
                        cfg.benchmark.num_iterations,
                        cfg.benchmark.device
                    )
                    results.append(result)
                except Exception as e:
                    log.error(f"Error testing ViT config {config}: {str(e)}")
                    continue

        else:
            raise ValueError(f"Unknown model type: {cfg.model.type}")

    except Exception as e:
        log.error(f"Error during benchmark: {str(e)}")
        raise

    # Save results
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump([vars(r) for r in results], f, indent=2)

    # Print summary
    log.info("\nBenchmark Results:")
    log.info(f"{'Model':<20} {'Size (MB)':<12} {'Throughput (GB/s)':<18} {'Latency (ms)':<14} {'GPU Mem (MB)':<12}")
    log.info("-" * 80)
    for r in results:
        log.info(f"{r.model_name:<20} {r.model_size_mb:<12.2f} {r.throughput_gbps:<18.2f} {r.latency_ms:<14.2f} {r.gpu_mem_usage_mb:<12.2f}")

if __name__ == "__main__":
    run_benchmark()
