import asyncio
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, Header, HTTPException, UploadFile

from parakeet_trt_runtime import ParakeetTensorRTRuntime


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


def require_environment(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def verify_token(authorization):
    expected = require_environment("ASR_API_TOKEN")

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
        )

    scheme, separator, supplied = authorization.partition(" ")

    valid = (
        separator
        and scheme.lower() == "bearer"
        and secrets.compare_digest(supplied, expected)
    )

    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token",
        )


def ensure_soundfile_readable(source_path, work_dir):
    try:
        sf.info(str(source_path))
        return source_path, 0.0, False
    except Exception:
        normalized_path = work_dir / "normalized.wav"

        started = time.perf_counter()

        process = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(normalized_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

        elapsed = time.perf_counter() - started

        if process.returncode != 0:
            message = process.stderr.decode(
                "utf-8",
                errors="replace",
            )
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported or invalid audio: {message[-500:]}",
            )

        return normalized_path, elapsed, True


@asynccontextmanager
async def lifespan(app):
    model_path = require_environment("PARAKEET_MODEL_PATH")
    engine_path = require_environment("PARAKEET_ENGINE_PATH")

    started = time.perf_counter()

    runtime = ParakeetTensorRTRuntime(
        model_path=model_path,
        engine_path=engine_path,
    )

    app.state.runtime = runtime
    app.state.inference_lock = asyncio.Lock()
    app.state.ready_seconds = time.perf_counter() - started

    print(json.dumps({
        "event": "asr_api_ready",
        "runtime_ready_seconds": round(
            app.state.ready_seconds,
            6,
        ),
        "backend": "tensorrt-fp16-10.11",
    }))

    try:
        yield
    finally:
        runtime.close()


app = FastAPI(
    title="Parakeet CTC Vietnamese TensorRT API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ready",
        "backend": "tensorrt-fp16-10.11",
        "sample_rate": 16000,
        "max_mel_frames": 2000,
        "runtime_ready_seconds": round(
            app.state.ready_seconds,
            6,
        ),
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    verify_token(authorization)

    request_started = time.perf_counter()
    job_id = str(uuid.uuid4())

    temp_root = Path(
        os.environ.get(
            "ASR_TEMP_DIR",
            "/workspace/api_tmp",
        )
    )
    temp_root.mkdir(parents=True, exist_ok=True)

    work_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{job_id}-",
            dir=str(temp_root),
        )
    )

    filename = Path(audio.filename or "audio.bin").name
    suffix = Path(filename).suffix[:16] or ".bin"
    source_path = work_dir / f"upload{suffix}"

    upload_started = time.perf_counter()
    uploaded_bytes = 0

    try:
        with source_path.open("wb") as destination:
            while True:
                chunk = await audio.read(UPLOAD_CHUNK_BYTES)

                if not chunk:
                    break

                uploaded_bytes += len(chunk)

                if uploaded_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Audio upload exceeds 25 MiB",
                    )

                destination.write(chunk)

        upload_seconds = time.perf_counter() - upload_started

        normalized_path, normalize_seconds, converted = (
            ensure_soundfile_readable(
                source_path,
                work_dir,
            )
        )

        queue_entered = time.perf_counter()

        async with app.state.inference_lock:
            queue_wait_seconds = (
                time.perf_counter() - queue_entered
            )

            try:
                result = app.state.runtime.transcribe(
                    normalized_path
                )
            except ValueError as error:
                raise HTTPException(
                    status_code=400,
                    detail=str(error),
                ) from error
            except RuntimeError as error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio decode failed: {error}",
                ) from error

        server_total_seconds = (
            time.perf_counter() - request_started
        )

        request_metrics = {
            "upload_seconds": round(upload_seconds, 6),
            "normalize_seconds": round(normalize_seconds, 6),
            "queue_wait_seconds": round(queue_wait_seconds, 6),
            "server_total_seconds": round(
                server_total_seconds,
                6,
            ),
        }

        target_seconds = 1.5

        response = {
            "schema_version": "1.0",
            "job_id": job_id,
            "status": "completed",
            "text": result["text"],
            "language": "vi-VN",
            "backend": result["backend"],
            "audio": {
                "filename": filename,
                "content_type": (
                    audio.content_type
                    or "application/octet-stream"
                ),
                "uploaded_bytes": uploaded_bytes,
                "duration_seconds": result["audio_seconds"],
                "sample_rate": result["sample_rate"],
                "converted_by_ffmpeg": converted,
            },
            "latency": {
                "upload_seconds": request_metrics[
                    "upload_seconds"
                ],
                "normalize_seconds": request_metrics[
                    "normalize_seconds"
                ],
                "queue_wait_seconds": request_metrics[
                    "queue_wait_seconds"
                ],
                "preprocess_seconds": result["metrics"][
                    "preprocess_seconds"
                ],
                "tensorrt_seconds": result["metrics"][
                    "tensorrt_seconds"
                ],
                "decode_seconds": result["metrics"][
                    "decode_seconds"
                ],
                "inference_total_seconds": result["metrics"][
                    "total_seconds"
                ],
                "server_total_seconds": request_metrics[
                    "server_total_seconds"
                ],
                "target_seconds": target_seconds,
                "passed": (
                    request_metrics["server_total_seconds"]
                    < target_seconds
                ),
            },
        }

        return response

    finally:
        await audio.close()
        shutil.rmtree(work_dir, ignore_errors=True)
