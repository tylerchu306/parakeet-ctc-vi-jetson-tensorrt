import os
import time
import resource

import torch
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr


model_path = os.environ["PARAKEET_MODEL_PATH"]

print("model_path=", model_path)
print("cuda_available=", torch.cuda.is_available())
print("device_name=", torch.cuda.get_device_name(0))

started = time.perf_counter()

model = nemo_asr.models.ASRModel.restore_from(
    restore_path=model_path,
    map_location="cpu",
)

cpu_loaded = time.perf_counter()
print("model_class=", type(model).__name__)
print("cpu_load_seconds=", round(cpu_loaded - started, 3))

parameter_count = sum(p.numel() for p in model.parameters())
print("parameter_count=", parameter_count)
print("parameter_millions=", round(parameter_count / 1_000_000, 3))
print("initial_parameter_dtype=", next(model.parameters()).dtype)

model.eval()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

gpu_started = time.perf_counter()
model = model.to("cuda")
torch.cuda.synchronize()
gpu_loaded = time.perf_counter()

print("gpu_transfer_seconds=", round(gpu_loaded - gpu_started, 3))
print("gpu_parameter_dtype=", next(model.parameters()).dtype)
print(
    "gpu_allocated_gb=",
    round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
)
print(
    "gpu_reserved_gb=",
    round(torch.cuda.memory_reserved() / (1024 ** 3), 3),
)
print(
    "gpu_peak_allocated_gb=",
    round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
)

config_paths = [
    "sample_rate",
    "preprocessor.sample_rate",
    "preprocessor.features",
    "encoder._target_",
    "encoder.d_model",
    "encoder.n_layers",
    "encoder.subsampling",
    "encoder.subsampling_factor",
    "decoder._target_",
    "decoder.num_classes",
]

for path in config_paths:
    value = OmegaConf.select(model.cfg, path, default=None)
    print(f"config.{path}=", value)

input_types = getattr(model, "input_types", None)
output_types = getattr(model, "output_types", None)

print(
    "input_names=",
    list(input_types.keys()) if input_types else None,
)
print(
    "output_names=",
    list(output_types.keys()) if output_types else None,
)
print("has_export=", callable(getattr(model, "export", None)))
print(
    "process_max_rss_gb=",
    round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 3),
)
