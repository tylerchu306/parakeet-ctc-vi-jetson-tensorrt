import os
import time
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr


model_path = os.environ["PARAKEET_MODEL_PATH"]
output_path = Path(os.environ["PARAKEET_ONNX_PATH"])
output_path.parent.mkdir(parents=True, exist_ok=True)

torch.set_grad_enabled(False)

print("restoring_model=", model_path)
started = time.perf_counter()

model = nemo_asr.models.ASRModel.restore_from(
    restore_path=model_path,
    map_location="cpu",
)
model.eval()
model.encoder.sync_max_audio_length = False

# NeMo 2.7.3 exposes bypass_pre_encode as an optional deployment
# input although ExportableEncDecModel.forward_for_export does not
# accept it. Fix the exported input list for this process only.
encoder_class = type(model.encoder)
original_disabled_property = encoder_class.disabled_deployment_input_names

encoder_class.disabled_deployment_input_names = property(
    lambda self: original_disabled_property.fget(self)
    | {"bypass_pre_encode"}
)

input_example = model.encoder.input_example(
    max_batch=1,
    max_dim=500,
)

print("model_class=", type(model).__name__)
print("input_names=", model.input_names)
print("output_names=", model.output_names)
print("dynamic_axes=", model.dynamic_shapes_for_export())

for index, tensor in enumerate(input_example):
    print(
        "input_example",
        index,
        "shape=",
        tuple(tensor.shape),
        "dtype=",
        tensor.dtype,
    )

print("export_started")

result = model.export(
    output=str(output_path),
    input_example=input_example,
    verbose=False,
    do_constant_folding=True,
    onnx_opset_version=17,
    check_trace=False,
    use_dynamo=False,
)

elapsed = time.perf_counter() - started

print("export_result=", result)
print("export_seconds=", round(elapsed, 3))
print("onnx_path=", output_path)
print("onnx_exists=", output_path.exists())

if output_path.exists():
    print("onnx_bytes=", output_path.stat().st_size)
