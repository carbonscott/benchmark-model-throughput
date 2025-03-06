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
from transformers import ViTModel, ViTConfig, ConvNextV2Config, ConvNextV2Model
import torch.cuda.profiler as profiler
import gc

log = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    model_name: str
    model_config: str
    model_size_mb: float
    num_params: int
    throughput_gbps: float
    latency_ms: float
    gpu_mem_usage_mb: float

def get_num_params(model: nn.Module) -> int:
    """Get total number of parameters in the model"""
    return sum(p.numel() for p in model.parameters())

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
    def create_convnextv2_from_config(config: dict) -> nn.Module:
        return ConvNextV2Model(
            ConvNextV2Config(**config)
        )

    @staticmethod
    def create_vit(config: dict) -> nn.Module:
        # Create and return the ViT model
        return ViTModel(
            ViTConfig(**config)
        )

def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "fp8": torch.float8_e4m3fn,
    }
    return dtype_map[dtype_str.lower()]

def calibration_loop(model, input_shape=(3, 224, 224)):
    """Calibration loop that uses the correct input shape"""
    device = next(model.parameters()).device
    for _ in range(10):  # 10 calibration batches
        calib_data = torch.randn(16, *input_shape, device=device)
        with torch.no_grad():
            _ = model(calib_data)

def convert_model_dtype(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Convert model to specified dtype"""
    if dtype in [torch.float16, torch.bfloat16]:
        model = model.to(dtype)
    return model

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
    device: str,
    compile_cfg: DictConfig,
    precision_cfg: DictConfig,
    quant_cfg: DictConfig = None,
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
    num_params = get_num_params(model)

    # Convert model dtype if needed
    dtype = get_dtype(precision_cfg.dtype)
    model = convert_model_dtype(model, dtype)

    # Create random input data
    inputs = torch.randn(batch_size, *input_shape, device=device)
    if dtype in [torch.float16, torch.bfloat16]:
        inputs = inputs.to(dtype)

    # Apply compilation optimizations
    if compile_cfg.backend == "tensorrt":
        try:
            import modelopt.torch.quantization as mtq
            from modelopt.torch.quantization.utils import export_torch_mode
            import torch_tensorrt as torchtrt

            # Apply quantization if the precision dtype indicates INT8 or FP8
            if precision_cfg.dtype == "int8":
                log.info(f"Applying INT8 quantization with input shape {input_shape}")
                quant_cfg = mtq.INT8_DEFAULT_CFG
                mtq.quantize(model, quant_cfg, 
                             forward_loop=lambda m: calibration_loop(m, input_shape=input_shape))
            elif precision_cfg.dtype == "fp8":
                log.info(f"Applying FP8 quantization with input shape {input_shape}")
                quant_cfg = mtq.FP8_DEFAULT_CFG
                mtq.quantize(model, quant_cfg,
                             forward_loop=lambda m: calibration_loop(m, input_shape=input_shape))

            # Export the model (with quantization modifications in place)
            with torch.no_grad(), export_torch_mode():
                from torch.export._trace import _export
                exp_program = _export(model, (inputs,))

                # Set up enabled precisions
                enabled_precisions = set(get_dtype(precision_cfg.dtype))

                # Compile with TensorRT using the quantized model
                model = torchtrt.dynamo.compile(
                    exp_program,
                    inputs=[inputs],
                    enabled_precisions=enabled_precisions,
                    min_block_size=5,
                    workspace_size=1 << 30,
                )
            log.info(f"Model compiled with TensorRT using {precision_cfg.dtype}")
        except Exception as e:
            log.warning(f"Failed to compile model with TensorRT: {str(e)}")
    elif compile_cfg.backend == "torch":
        try:
            model = torch.compile(
                model,
                mode=compile_cfg.mode,
                fullgraph=compile_cfg.fullgraph,
                dynamic=compile_cfg.dynamic
            )
            log.info(f"Model compiled with mode: {compile_cfg.mode}")
        except Exception as e:
            log.warning(f"Failed to compile model: {str(e)}")

    # Setup autocast if enabled
    autocast_ctx = (
        torch.autocast(device_type='cuda', dtype=dtype)
        if precision_cfg.autocast
        else nullcontext()
    )

    # Warmup - extra warmup iterations for compiled models
    num_compile_warmup = num_warmup * 2 if compile_cfg.backend is not None else num_warmup
    with torch.no_grad(), autocast_ctx:
        for i in range(num_compile_warmup):
            _ = model(inputs)
            if i == 0:
                torch.cuda.synchronize()  # Ensure first compilation is done

    torch.cuda.synchronize()

    # Reset GPU stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    latencies = []
    with torch.no_grad(), autocast_ctx:
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
        model_name=f"{model_name}_{precision_cfg.dtype}",
        model_config=model_config,
        model_size_mb=model_size,
        num_params=num_params,
        throughput_gbps=throughput,
        latency_ms=avg_latency,
        gpu_mem_usage_mb=gpu_mem
    )

@hydra.main(config_path="configs", config_name="config", version_base=None)
def run_benchmark(cfg: DictConfig) -> None:
    log.info(f"Running benchmark with config:\n{OmegaConf.to_yaml(cfg)}")

    # Check if the GPU supports the requested dtype
    if cfg.benchmark.precision.dtype in ["bfloat16", "float16"]:
        if cfg.benchmark.precision.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BFloat16 is not supported on this GPU")

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
                    cfg.benchmark.device,
                    cfg.benchmark.compile,
                    cfg.benchmark.precision,
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
                    cfg.benchmark.device,
                    cfg.benchmark.compile,
                    cfg.benchmark.precision,
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
                    cfg.benchmark.device,
                    cfg.benchmark.compile,
                    cfg.benchmark.precision,
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
                    cfg.benchmark.device,
                    cfg.benchmark.compile,
                    cfg.benchmark.precision,
                )
                results.append(result)

        elif cfg.model.type == "convnext":
            log.info("Benchmarking ConvNeXt models")
            # Handle custom configurations
            for custom_config in cfg.model.configs:
                log.info(f"Testing custom ConvNeXt config: {custom_config.name}")
                try:
                    custom_config.config_params.image_size = cfg.input.shape[-1]
                    custom_config.config_params.num_channels = cfg.input.shape[0]

                    # Convert OmegaConf dict to Python dict if necessary
                    if isinstance(custom_config.config_params, DictConfig):
                        custom_config_params = OmegaConf.to_container(custom_config.config_params)

                    # Create model from config
                    model = ModelFactory.create_convnextv2_from_config(custom_config_params)

                    result = benchmark_model(
                        model,
                        tuple(cfg.input.shape),
                        cfg.input.batch_size,
                        cfg.benchmark.num_warmup,
                        cfg.benchmark.num_iterations,
                        cfg.benchmark.device,
                        cfg.benchmark.compile,
                        cfg.benchmark.precision,
                    )
                    results.append(result)
                except Exception as e:
                    log.error(f"Error testing custom config: {str(e)}")
                    continue

        elif cfg.model.type == "vit":
            log.info("Benchmarking Vision Transformer models")
            for custom_config in cfg.model.configs:
                log.info(f"Testing ViT with config: {custom_config.name}")
                try:
                    # Apply input shape parameters to the config
                    custom_config.config_params.image_size = cfg.input.shape[-1]
                    custom_config.config_params.num_channels = cfg.input.shape[0]

                    # Convert OmegaConf dict to Python dict if necessary
                    if isinstance(custom_config.config_params, DictConfig):
                        custom_config_params = OmegaConf.to_container(custom_config.config_params)

                    # Create model from config
                    model = ModelFactory.create_vit(custom_config_params)

                    result = benchmark_model(
                        model,
                        tuple(cfg.input.shape),
                        cfg.input.batch_size,
                        cfg.benchmark.num_warmup,
                        cfg.benchmark.num_iterations,
                        cfg.benchmark.device,
                        cfg.benchmark.compile,
                        cfg.benchmark.precision,
                    )
                    results.append(result)
                except Exception as e:
                    log.error(f"Error testing ViT config {custom_config.name}: {str(e)}")
                    continue

        else:
            raise ValueError(f"Unknown model type: {cfg.model.type}")

    except Exception as e:
        log.error(f"Error during benchmark: {str(e)}")
        raise

    # Save results
    compile_method = "eager" if cfg.benchmark.compile.backend is None else cfg.benchmark.compile.backend
    output_file = f"benchmark_results_{cfg.model.type}_bs{cfg.input.batch_size}_{cfg.benchmark.precision.dtype}_{compile_method}.json"
    with open(output_file, 'w') as f:
        json.dump([vars(r) for r in results], f, indent=2)

    # Print summary with parameter count
    log.info(f"\nBenchmark Results (Precision: {cfg.benchmark.precision.dtype}):")
    log.info(f"{'Model':<30} {'Params':<12} {'Size (MB)':<12} {'Throughput (GB/s)':<18} {'Latency (ms)':<14} {'GPU Mem (MB)':<12}")
    log.info("-" * 102)  # Adjusted line length
    for r in results:
        params_str = f"{r.num_params:,}"
        log.info(f"{r.model_name:<30} {params_str:<12} {r.model_size_mb:<12.2f} {r.throughput_gbps:<18.2f} {r.latency_ms:<14.2f} {r.gpu_mem_usage_mb:<12.2f}")

if __name__ == "__main__":
    run_benchmark()
