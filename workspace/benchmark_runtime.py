import json
import statistics

from parakeet_trt_runtime import ParakeetTensorRTRuntime


MODEL_PATH = (
    "/workspace/cache/huggingface/hub/"
    "models--nvidia--parakeet-ctc-0.6b-Vietnamese/"
    "snapshots/"
    "b0493142b49458810324e3db8be9e8e07b4ebc17/"
    "parakeet-ctc-0.6b-vi.nemo"
)

ENGINE_PATH = (
    "/workspace/engines/"
    "parakeet_ctc_vi_fp16_trt1011.engine"
)

AUDIO_PATH = "/workspace/audio/test_vieneu_16k.wav"

runtime = ParakeetTensorRTRuntime(
    model_path=MODEL_PATH,
    engine_path=ENGINE_PATH,
)

results = []

for index in range(6):
    result = runtime.transcribe(AUDIO_PATH)
    results.append(result)

    print(json.dumps({
        "request": index + 1,
        "text": result["text"],
        **result["metrics"],
    }, ensure_ascii=False))

warm_results = results[1:]

for metric in [
    "audio_load_seconds",
    "preprocess_seconds",
    "tensorrt_seconds",
    "decode_seconds",
    "total_seconds",
    "rtf",
]:
    values = [
        item["metrics"][metric]
        for item in warm_results
    ]

    print(
        metric,
        "mean=",
        round(statistics.mean(values), 6),
        "min=",
        round(min(values), 6),
        "max=",
        round(max(values), 6),
    )

print(
    "all_transcripts_equal=",
    len({item["text"] for item in results}) == 1,
)

runtime.close()
