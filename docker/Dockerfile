FROM nvcr.io/nvidia/pytorch:25.06-py3-igpu

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    "nemo_toolkit[asr]==2.7.3"

# Compatibility pins for NVIDIA Jetson PyTorch.
RUN python3 -m pip install --no-cache-dir \
       "numpy==1.26.4" \
       "packaging==24.2" \
       "markdown-it-py==3.0.0" \
       "docopt==0.6.2" \
    && python3 -m pip uninstall -y apex \
    && python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
       "pyannote-core==5.0.0" \
       "pyannote-database==5.1.3" \
       "pyannote-metrics==3.2.1"

# Disable automatic imports of distributed training strategies.
# Required submodules are still imported directly by ASR ModelPT.
RUN printf "%s\n" \
    "\"\"\"Inference-only NeMo Lightning package init for Jetson." \
    "" \
    "The Jetson PyTorch build has no torch.distributed support." \
    "ASR inference imports required submodules directly." \
    "\"\"\"" \
    > /usr/local/lib/python3.12/dist-packages/nemo/lightning/__init__.py

ENV NEMO_TELEMETRY_ENABLED=false
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

CMD ["/bin/bash"]
