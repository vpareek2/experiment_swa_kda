FROM nvcr.io/nvidia/pytorch:25.10-py3

RUN python -m pip install --no-cache-dir uv==0.11.16
WORKDIR /opt/research
COPY pyproject.toml uv.lock README.md ./
RUN UV_PROJECT_ENVIRONMENT=/opt/research-venv uv sync --frozen --extra gpu --no-dev --no-install-project

ENV PATH="/opt/research-venv/bin:${PATH}"
WORKDIR /workspace
ENTRYPOINT []
