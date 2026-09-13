FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/runpod-volume/huggingface LUMEN_MODEL_ID=Wan-AI/Wan2.1-VACE-1.3B-diffusers
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /worker
COPY requirements.txt .
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 \
 && python3 -m pip install -r requirements.txt \
 && python3 -c "import torch; assert torch.version.cuda == '12.8' and 'sm_120' in torch.cuda.get_arch_list(), (torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"
COPY handler.py .
CMD ["python3", "-u", "handler.py"]
