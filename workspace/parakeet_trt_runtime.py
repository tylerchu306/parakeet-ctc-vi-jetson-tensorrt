import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import resampy
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr

from trt_ctypes_runner import TensorRTRunner


class ParakeetTensorRTRuntime:
    SAMPLE_RATE = 16000
    MAX_MEL_FRAMES = 2000

    def __init__(self, model_path, engine_path):
        self.model_path = str(model_path)
        self.engine_path = str(engine_path)

        torch.set_grad_enabled(False)

        restore_started = time.perf_counter()

        model = nemo_asr.models.ASRModel.restore_from(
            restore_path=self.model_path,
            map_location="cpu",
        )
        model.eval()
        model.encoder.sync_max_audio_length = False

        self.preprocessor = model.preprocessor.eval().to("cuda")
        self.tokenizer = model.tokenizer

        classes_with_blank = getattr(
            model.decoder,
            "num_classes_with_blank",
            None,
        )
        if classes_with_blank is not None:
            self.blank_id = int(classes_with_blank) - 1
        else:
            self.blank_id = int(model.decoder.num_classes)

        # Release the 608M-parameter PyTorch encoder/decoder. Only the
        # lightweight preprocessor and tokenizer are retained.
        model.encoder = None
        model.decoder = None
        model.loss = None
        model.wer = None
        model.spec_augmentation = None

        del model
        gc.collect()
        torch.cuda.empty_cache()

        self.model_restore_seconds = (
            time.perf_counter() - restore_started
        )

        engine_started = time.perf_counter()
        self.runner = TensorRTRunner(self.engine_path)
        self.engine_load_seconds = (
            time.perf_counter() - engine_started
        )

        warmup_mel = np.zeros(
            (1, 80, 500),
            dtype=np.float32,
        )
        warmup_length = np.array(
            [500],
            dtype=np.int64,
        )

        warmup_started = time.perf_counter()
        self.runner.infer({
            "audio_signal": warmup_mel,
            "length": warmup_length,
        })
        self.runner.infer({
            "audio_signal": warmup_mel,
            "length": warmup_length,
        })
        self.warmup_seconds = (
            time.perf_counter() - warmup_started
        )

    @staticmethod
    def _load_audio(audio_path):
        audio, sample_rate = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True,
        )

        audio = audio.mean(axis=1)

        if sample_rate != ParakeetTensorRTRuntime.SAMPLE_RATE:
            audio = resampy.resample(
                audio,
                sample_rate,
                ParakeetTensorRTRuntime.SAMPLE_RATE,
            )
            sample_rate = ParakeetTensorRTRuntime.SAMPLE_RATE

        audio = np.ascontiguousarray(
            audio,
            dtype=np.float32,
        )

        return audio, sample_rate

    def _preprocess(self, audio):
        signal = torch.from_numpy(audio).unsqueeze(0).to("cuda")
        signal_length = torch.tensor(
            [audio.shape[0]],
            dtype=torch.int64,
            device="cuda",
        )

        with torch.no_grad():
            mel, mel_length = self.preprocessor(
                input_signal=signal,
                length=signal_length,
            )

        torch.cuda.synchronize()

        mel_numpy = np.ascontiguousarray(
            mel.detach().cpu().numpy(),
            dtype=np.float32,
        )
        length_numpy = np.ascontiguousarray(
            mel_length.detach().cpu().numpy(),
            dtype=np.int64,
        )

        if mel_numpy.shape[2] > self.MAX_MEL_FRAMES:
            raise ValueError(
                "Audio is longer than the TensorRT profile: "
                f"{mel_numpy.shape[2]} mel frames > "
                f"{self.MAX_MEL_FRAMES}"
            )

        return mel_numpy, length_numpy

    def _decode(self, logprobs):
        argmax_ids = logprobs.argmax(axis=-1)[0]

        collapsed_ids = []
        previous = None

        for token_id in argmax_ids.tolist():
            token_id = int(token_id)

            if (
                token_id != previous
                and token_id != self.blank_id
            ):
                collapsed_ids.append(token_id)

            previous = token_id

        text = self.tokenizer.ids_to_text(collapsed_ids)

        return text.strip(), collapsed_ids

    def transcribe(self, audio_path):
        total_started = time.perf_counter()

        load_started = time.perf_counter()
        audio, sample_rate = self._load_audio(audio_path)
        audio_load_seconds = time.perf_counter() - load_started

        preprocess_started = time.perf_counter()
        mel, mel_length = self._preprocess(audio)
        preprocess_seconds = (
            time.perf_counter() - preprocess_started
        )

        inference_started = time.perf_counter()
        outputs = self.runner.infer({
            "audio_signal": mel,
            "length": mel_length,
        })
        inference_seconds = (
            time.perf_counter() - inference_started
        )

        decode_started = time.perf_counter()
        text, token_ids = self._decode(outputs["logprobs"])
        decode_seconds = time.perf_counter() - decode_started

        total_seconds = time.perf_counter() - total_started
        audio_seconds = audio.shape[0] / sample_rate

        return {
            "backend": "tensorrt-fp16-10.11",
            "audio_path": str(audio_path),
            "sample_rate": sample_rate,
            "audio_seconds": round(audio_seconds, 6),
            "mel_shape": list(mel.shape),
            "mel_length": mel_length.tolist(),
            "logprobs_shape": list(
                outputs["logprobs"].shape
            ),
            "blank_id": self.blank_id,
            "token_ids": token_ids,
            "text": text,
            "metrics": {
                "audio_load_seconds": round(
                    audio_load_seconds,
                    6,
                ),
                "preprocess_seconds": round(
                    preprocess_seconds,
                    6,
                ),
                "tensorrt_seconds": round(
                    inference_seconds,
                    6,
                ),
                "decode_seconds": round(
                    decode_seconds,
                    6,
                ),
                "total_seconds": round(
                    total_seconds,
                    6,
                ),
                "rtf": round(
                    total_seconds / audio_seconds,
                    6,
                ),
            },
        }

    def close(self):
        self.runner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    runtime_started = time.perf_counter()
    runtime = ParakeetTensorRTRuntime(
        model_path=args.model,
        engine_path=args.engine,
    )
    runtime_ready_seconds = (
        time.perf_counter() - runtime_started
    )

    result = runtime.transcribe(args.audio)
    result["startup"] = {
        "model_restore_seconds": round(
            runtime.model_restore_seconds,
            6,
        ),
        "engine_load_seconds": round(
            runtime.engine_load_seconds,
            6,
        ),
        "warmup_seconds": round(
            runtime.warmup_seconds,
            6,
        ),
        "runtime_ready_seconds": round(
            runtime_ready_seconds,
            6,
        ),
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    runtime.close()


if __name__ == "__main__":
    main()
