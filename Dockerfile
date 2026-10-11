# Main project image: `uv sync`'s runtime plus git and the Docker CLI.
#
# Talks to the *host*'s Docker daemon via a mounted /var/run/docker.sock
# (Docker-outside-of-Docker). The Commit0 benchmark launches its agent and
# grading containers through it, translating bind-mount sources with
# HOST_PROJECT_ROOT (compose.yaml). OpenHands is not installed here: it
# lives in each library's agent image (commit0.build_agent_image). data/
# is a runtime volume mount, never baked into this image.
FROM ghcr.io/astral-sh/uv:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        build-essential \
        docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# compose.yaml bind-mounts the whole repo over /app at runtime (live code
# edits without a rebuild) -- putting the venv outside /app keeps it from
# being shadowed by that mount.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Dependency layer first, cached across source-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

CMD ["bash"]
