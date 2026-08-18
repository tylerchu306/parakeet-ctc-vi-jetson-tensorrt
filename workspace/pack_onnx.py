import argparse
import time
from pathlib import Path

import onnx


def main():
    parser = argparse.ArgumentParser(
        description="Pack an ONNX model into one graph file and one external data file."
    )
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument(
        "--data-name",
        help="External data filename. Defaults to <output filename>.data.",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    data_name = args.data_name or f"{output.name}.data"

    started = time.perf_counter()
    print("loading=", source)
    model = onnx.load(str(source), load_external_data=True)

    print("saving=", output)
    print("external_data=", output.parent / data_name)
    onnx.save_model(
        model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )

    packed = onnx.load(str(output), load_external_data=False)
    onnx.checker.check_model(packed)

    print("checker=PASSED")
    print("elapsed_seconds=", round(time.perf_counter() - started, 3))
    print("model_bytes=", output.stat().st_size)
    print("data_bytes=", (output.parent / data_name).stat().st_size)


if __name__ == "__main__":
    main()
