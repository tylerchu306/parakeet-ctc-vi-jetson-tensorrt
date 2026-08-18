import argparse
import json
from pathlib import Path

import numpy as np


def collapse_ctc(frame_ids, blank_id):
    collapsed = []
    previous = None
    for token_id in frame_ids.tolist():
        token_id = int(token_id)
        if token_id != previous and token_id != blank_id:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


def load_trtexec_tensor(path, tensor_name):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    item = next(value for value in items if value["name"] == tensor_name)
    shape = tuple(int(value) for value in item["dimensions"].split("x"))
    return np.asarray(item["values"], dtype=np.float32).reshape(shape)


def main():
    parser = argparse.ArgumentParser(
        description="Compare trtexec JSON logprobs with the PyTorch reference tensor."
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--tensorrt-json", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    reference = np.load(args.reference).astype(np.float32, copy=False)
    tensorrt = load_trtexec_tensor(args.tensorrt_json, "logprobs")
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))

    if reference.shape != tensorrt.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} != {tensorrt.shape}")

    difference = reference - tensorrt
    reference_flat = reference.astype(np.float64).ravel()
    tensorrt_flat = tensorrt.astype(np.float64).ravel()
    denominator = np.linalg.norm(reference_flat) * np.linalg.norm(tensorrt_flat)
    cosine = float(np.dot(reference_flat, tensorrt_flat) / (denominator + 1e-12))

    reference_frames = reference.argmax(axis=-1)[0]
    tensorrt_frames = tensorrt.argmax(axis=-1)[0]
    blank_id = int(metadata["blank_id"])
    reference_tokens = collapse_ctc(reference_frames, blank_id)
    tensorrt_tokens = collapse_ctc(tensorrt_frames, blank_id)

    result = {
        "reference_shape": list(reference.shape),
        "tensorrt_shape": list(tensorrt.shape),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "cosine": cosine,
        "frame_argmax_agreement": float(np.mean(reference_frames == tensorrt_frames)),
        "reference_tokens": reference_tokens,
        "tensorrt_tokens": tensorrt_tokens,
        "collapsed_tokens_equal": reference_tokens == tensorrt_tokens,
        "reference_transcript": metadata.get("transcript"),
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")

    if not result["collapsed_tokens_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
