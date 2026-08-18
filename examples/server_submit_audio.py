#!/usr/bin/env python3

import argparse
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

import requests


CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


def detect_content_type(audio_path: Path) -> str:
    content_type = CONTENT_TYPES.get(
        audio_path.suffix.lower()
    )

    if content_type:
        return content_type

    guessed, _ = mimetypes.guess_type(audio_path.name)

    return guessed or "application/octet-stream"


def transcribe_audio(
    audio_path: Path,
    api_url: str,
    api_token: str,
    timeout_seconds: float = 10.0,
) -> tuple[dict, float]:
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    content_type = detect_content_type(audio_path)

    headers = {
        "Authorization": f"Bearer {api_token}",
    }

    started = time.perf_counter()

    with audio_path.open("rb") as audio_file:
        response = requests.post(
            api_url,
            headers=headers,
            files={
                "audio": (
                    audio_path.name,
                    audio_file,
                    content_type,
                ),
            },
            timeout=(5.0, timeout_seconds),
        )

    client_e2e_seconds = time.perf_counter() - started

    response.raise_for_status()

    payload = response.json()

    if payload.get("schema_version") != "1.0":
        raise RuntimeError(
            "Unsupported ASR response schema: "
            f"{payload.get('schema_version')!r}"
        )

    if payload.get("status") != "completed":
        raise RuntimeError(
            "ASR job did not complete successfully"
        )

    return payload, client_e2e_seconds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Send an audio file to the Jetson ASR API "
            "and receive JSON schema v1."
        )
    )
    parser.add_argument("audio")
    parser.add_argument(
        "--url",
        default=os.environ.get(
            "ASR_URL",
            "http://127.0.0.1:8770/transcribe",
        ),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("ASR_API_TOKEN"),
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
    )

    args = parser.parse_args()

    if not args.token:
        parser.error(
            "Set ASR_API_TOKEN or pass --token"
        )

    payload, client_seconds = transcribe_audio(
        Path(args.audio),
        args.url,
        args.token,
        args.timeout,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        f"client_e2e_seconds={client_seconds:.6f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
