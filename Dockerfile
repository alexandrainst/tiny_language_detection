FROM python:3.12-slim-bookworm AS backend
COPY --from=ghcr.io/astral-sh/uv:0.6.11 /uv /uvx /bin/
ENV PATH="/root/.local/bin:/project/.venv/bin/:${PATH}"
WORKDIR /project
RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --locked --no-install-project
COPY . /project
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /project
USER appuser
CMD ["bash"]
