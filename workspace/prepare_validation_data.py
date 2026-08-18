import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr


model_path = os.environ["PARAKEET_MODEL_PATH"]
audio_path = Path(os.environ["PARAKEET_AUDIO_PATH"])
output_dir = Path(os.environ["PARAKEET_TEST_DATA_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

audio, sample_rate = sf.read(
    str(audio_path),
    dtype="float32",
    always_2d=False,
)

if sample_rate != 16000:
    raise ValueError(f"Expected 16000 Hz, got {sample_rate}")

if audio.ndim == 2:
    audio = audio.mean(axis=1)

audio = np.ascontiguousarray(audio, dtype=np.float32)

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

signal = torch.from_numpy(audio).unsqueeze(0).to("cuda")
signal_length = torch.tensor(
    [audio.shape[0]],
    dtype=torch.int64,
    device="cuda",
)

with torch.no_grad():
    processed_signal, processed_length = model.preprocessor(
        input_signal=signal,
        length=signal_length,
    )

    inference_started = time.perf_counter()
    logprobs = model.forward_for_export(
        processed_signal,
        processed_length,
    )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started

mel = np.ascontiguousarray(
    processed_signal.detach().cpu().numpy(),
    dtype=np.float32,
)
mel_length = np.ascontiguousarray(
    processed_length.detach().cpu().numpy(),
    dtype=np.int64,
)
reference_logprobs = np.ascontiguousarray(
    logprobs.detach().cpu().numpy(),
    dtype=np.float32,
)

argmax_ids = reference_logprobs.argmax(axis=-1)[0].tolist()
blank_id = reference_logprobs.shape[-1] - 1

collapsed_ids = []
previous = None
for token_id in argmax_ids:
    if token_id != previous and token_id != blank_id:
        collapsed_ids.append(int(token_id))
    previous = token_id

transcript = model.tokenizer.ids_to_text(collapsed_ids)

mel_path = output_dir / "test_vieneu_mel_fp32.bin"
length_path = output_dir / "test_vieneu_mel_length_int64.bin"
logprobs_path = output_dir / "test_vieneu_pytorch_logprobs.npy"
metadata_path = output_dir / "test_vieneu_validation_metadata.json"

mel.tofile(mel_path)
mel_length.tofile(length_path)
np.save(logprobs_path, reference_logprobs)

metadata = {
    "audio_path": str(audio_path),
    "sample_rate": sample_rate,
    "audio_samples": int(audio.shape[0]),
    "audio_seconds": round(audio.shape[0] / sample_rate, 6),
    "mel_shape": list(mel.shape),
    "mel_dtype": str(mel.dtype),
    "mel_length": mel_length.tolist(),
    "logprobs_shape": list(reference_logprobs.shape),
    "logprobs_dtype": str(reference_logprobs.dtype),
    "blank_id": int(blank_id),
    "collapsed_token_ids": collapsed_ids,
    "transcript": transcript,
    "model_load_seconds": round(load_seconds, 6),
    "pytorch_graph_seconds": round(inference_seconds, 6),
}

metadata_path.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(metadata, ensure_ascii=False, indent=2))
print("mel_file=", mel_path)
print("length_file=", length_path)
print("logprobs_file=", logprobs_path)
print("metadata_file=", metadata_path)
