# Server integration guide

## Contract

The simplest supported flow is synchronous HTTP:

1. The application server finishes receiving or recording one audio segment.
2. It sends `multipart/form-data` to the Jetson `POST /transcribe` endpoint.
3. It keeps the HTTP request open.
4. The Jetson normalizes audio if necessary, performs TensorRT inference and
   returns JSON.
5. The application server reads `text` and continues its own workflow.

There is no callback or polling. If the application server is not ready, it does
not submit the audio yet.

## Request

```http
POST /transcribe HTTP/1.1
Authorization: Bearer <ASR_API_TOKEN>
Content-Type: multipart/form-data; boundary=...

audio=<binary audio file>
```

Tested input types include WAV and WebM/Opus. MP3, M4A, OGG and FLAC are accepted
when the bundled FFmpeg can decode them. Files directly readable by libsndfile
skip FFmpeg. Other formats are converted to mono, 16 kHz, PCM WAV.

## Response schema v1

```json
{
  "schema_version": "1.0",
  "job_id": "e79474c7-e95a-4fba-9918-8c9fac6f0de9",
  "status": "completed",
  "text": "Xin chào ...",
  "language": "vi-VN",
  "backend": "tensorrt-fp16-10.11",
  "audio": {
    "filename": "microphone.webm",
    "content_type": "audio/webm",
    "uploaded_bytes": 19294,
    "duration_seconds": 4.64,
    "sample_rate": 16000,
    "converted_by_ffmpeg": true
  },
  "latency": {
    "upload_seconds": 0.000247,
    "normalize_seconds": 0.113371,
    "queue_wait_seconds": 0.000032,
    "preprocess_seconds": 0.011409,
    "tensorrt_seconds": 0.054833,
    "decode_seconds": 0.000356,
    "inference_total_seconds": 0.068856,
    "server_total_seconds": 0.183475,
    "target_seconds": 1.5,
    "passed": true
  }
}
```

Treat `schema_version`, `status`, `text`, `language`, `audio` and `latency` as
the stable integration surface. Do not parse Uvicorn logs to obtain results.

## Python integration

Copy `examples/server_submit_audio.py` into the server project or import it as a
small module.

```bash
python3 -m pip install 'requests>=2.31,<3'
export ASR_URL='http://JETSON_ADDRESS:8770/transcribe'
export ASR_API_TOKEN='<secret loaded from a secret manager>'
```

```python
import os
from pathlib import Path

from server_submit_audio import transcribe_audio


payload, client_e2e_seconds = transcribe_audio(
    audio_path=Path("microphone.webm"),
    api_url=os.environ["ASR_URL"],
    api_token=os.environ["ASR_API_TOKEN"],
    timeout_seconds=10.0,
)

if payload["schema_version"] != "1.0":
    raise RuntimeError("Unsupported ASR schema")

recognized_text = payload["text"]
within_budget = payload["latency"]["passed"]

process_text(recognized_text)
```

The client uses a 5-second connection timeout and a configurable response
timeout (10 seconds by default). The measured inference latency is normally far
below that; the larger timeout allows for temporary queueing.

## Curl integration

```bash
curl --fail-with-body --silent --show-error \
  --request POST \
  --header "Authorization: Bearer $ASR_API_TOKEN" \
  --form "audio=@microphone.webm;type=audio/webm" \
  "$ASR_URL"
```

## Readiness

Check readiness before sending work:

```http
GET /health
```

```json
{
  "status": "ready",
  "backend": "tensorrt-fp16-10.11",
  "sample_rate": 16000,
  "max_mel_frames": 2000,
  "runtime_ready_seconds": 16.43
}
```

Startup is intentionally separate from request latency. The container restores
the `.nemo` package for its preprocessor/tokenizer, loads the engine and warms
TensorRT before Uvicorn reports startup complete.

## Error handling

| HTTP status | Meaning | Server action |
|---|---|---|
| 200 | Completed; JSON schema v1 returned | Process `text` |
| 400 | Invalid/unsupported audio or profile too long | Reject or split input |
| 401 | Missing or invalid bearer token | Fix credentials; do not retry blindly |
| 413 | Upload exceeds 25 MiB | Compress or split audio |
| 5xx / timeout | Runtime or transport failure | Retry with bounded exponential backoff |

For a synchronous retry, retain the application's own request ID. The API
generates a new `job_id` for every attempt, so deduplicate in the application
server if downstream processing is not idempotent.

## Concurrency and backpressure

The deployed engine profile is batch 1. FastAPI accepts concurrent connections,
but GPU inference is protected by one `asyncio.Lock`; requests wait in-process
and expose that time as `queue_wait_seconds`.

Recommended initial policy:

- keep one Uvicorn worker;
- limit outstanding requests at the application server;
- reject or queue work before Jetson when latency requirements would be missed;
- alert when `queue_wait_seconds` grows materially;
- add a bounded external queue only when real traffic requires it.

Do not start multiple Uvicorn workers to increase throughput without measuring
GPU memory and creating independent TensorRT execution contexts deliberately.

## Network placement

The tested container binds to `127.0.0.1:8770`, so only the Jetson can call it.
This is the safest default while developing. For a remote application server,
choose one of these explicitly:

1. Put the server and Jetson in the same Tailscale network and bind Uvicorn to
   the Jetson Tailscale address.
2. Place an authenticated TLS reverse proxy in front of the API on a private
   network.
3. Make Jetson open an outbound authenticated connection to the application
   server when inbound access is prohibited.

Do not expose raw unauthenticated HTTP port 8770 to the public Internet. Keep the
bearer token in a secret manager or a mode-600 environment file, rotate it when
needed, and never commit it.

## Microphone guidance

For push-to-talk or transcription after the user stops speaking, HTTP WebM/Opus
is appropriate and compact. The 4.64-second fixture shrank from about 149 KB WAV
to about 19 KB WebM. FFmpeg normalization added roughly 113 ms in the acceptance
run, while the complete client call remained about 192 ms on loopback.

For live partial captions while the user is still speaking, this offline CTC
engine/API contract is not streaming. That use case needs chunk boundaries,
voice activity detection and a streaming/cached model design.

## Operations

```bash
sudo docker ps --filter name=parakeet-asr-api
sudo docker logs --follow parakeet-asr-api
sudo docker restart parakeet-asr-api
sudo docker stop parakeet-asr-api
sudo docker start parakeet-asr-api
```

Use `--restart unless-stopped` so the API returns after Docker/Jetson restarts.
Wait for `/health` to report `ready` before routing requests.
