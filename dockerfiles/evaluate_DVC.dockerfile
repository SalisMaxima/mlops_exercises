
# Base image
FROM python:3.12.3-slim


# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY LICENSE LICENSE
COPY README.md README.md


# Install dependencies
WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

# Set up DVC (but don't pull data yet - will pull at runtime)
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc ./
RUN dvc config core.no_scm true

# Copy entrypoint script that will pull data at runtime
COPY dockerfiles/evaluate_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
