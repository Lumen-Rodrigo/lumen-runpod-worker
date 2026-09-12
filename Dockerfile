FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/runpod-volume/huggingface LUMEN_MODEL_ID=Wan-AI/Wan2.1-VACE-1.3B-diffusers
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /worker
COPY requirements.txt .
RUN python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
COPY handler.py .
CMD ["python3", "-u", "handler.py"]
