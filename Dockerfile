# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Install uv from the official image and use lockfile-based sync.
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy training code and data, then train inside the image
COPY train.py predict.py features.py config.py ./
COPY data ./data
RUN uv run python train.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
