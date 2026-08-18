import json
import os
import time
from pathlib import Path

import soundfile as sf
import torch
from jiwer import cer, wer
import nemo.collections.asr as nemo_asr


model_path = os.environ["PARAKEET_MODEL_PATH"]
audio_path = Path(os.environ["PARAKEET_AUDIO_PATH"])
reference_path = Path(os.environ["PARAKEET_REFERENCE_PATH"])
output_path = Path(os.environ["PARAKEET_OUTPUT_PATH"])

audio_info = sf.info(str(audio_path))
audio_seconds = float(audio_info.frames / audio_info.samplerate)
reference = reference_path.read_text(encoding="utf-8").strip()

torch.set_grad_enabled(False)

load_started = time.perf_counter()

model = nemo_asr.models.ASRModel.restore_from(
    restore_path=model_path,
    map_location="cpu",
)
model.eval()
model.encoder.sync_max_audio_length = False
model = model.to("cuda")
torch.cuda.synchronize()

load_seconds = time.perf_counter() - load_started

def transcribe_once():
    started = time.perf_counter()
    result = model.transcribe(
        [str(audio_path)],
        batch_size=1,
        return_hypotheses=False,
        verbose=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    item = result[0]
    text = item.text if hasattr(item, "text") else str(item)
    return text.strip(), elapsed


# First run includes CUDA kernel and data-pipeline warm-up.
warmup_text, warmup_seconds = transcribe_once()

torch.cuda.reset_peak_memory_stats()

transcript, inference_seconds = transcribe_once()

metrics = {
    "backend": "nemo-pytorch-fp32",
    "model_class": type(model).__name__,
    "audio_path": str(audio_path),
    "audio_seconds": round(audio_seconds, 6),
    "reference": reference,
    "transcript": transcript,
    "load_seconds": round(load_seconds, 6),
    "warmup_seconds": round(warmup_seconds, 6),
    "inference_seconds": round(inference_seconds, 6),
    "rtf": round(inference_seconds / audio_seconds, 6),
    "wer": round(wer(reference.lower(), transcript.lower()), 6),
    "cer": round(cer(reference.lower(), transcript.lower()), 6),
    "gpu_allocated_gb": round(
        torch.cuda.memory_allocated() / (1024 ** 3), 6
    ),
    "gpu_reserved_gb": round(
        torch.cuda.memory_reserved() / (1024 ** 3), 6
    ),
    "gpu_peak_allocated_gb": round(
        torch.cuda.max_memory_allocated() / (1024 ** 3), 6
    ),
}

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("report=", output_path)
