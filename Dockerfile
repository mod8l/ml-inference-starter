FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --upgrade pip && \
    pip install --user --no-cache-dir .

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r inference && useradd -r -g inference inference

COPY --from=builder /root/.local /home/inference/.local
COPY data ./data

ENV PATH=/home/inference/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN chown -R inference:inference /app /home/inference
USER inference

EXPOSE 8000

CMD ["uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "8000"]
