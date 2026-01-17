FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e .

COPY configs ./configs
COPY scripts ./scripts

CMD ["python", "-m", "aet.cli", "--config", "configs/base.yaml", "--stage", "train"]
