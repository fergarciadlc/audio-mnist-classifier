# CPU image. Use docker/Dockerfile.gpu for CUDA on the Linux server.
FROM python:3.11-slim

# System deps: libsndfile + ffmpeg cover the audio decoders librosa/soundfile rely on.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -r requirements.txt

COPY src /app/src
COPY pipelines /app/pipelines

CMD ["bash"]
