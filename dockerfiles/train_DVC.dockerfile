
# Base image with CUDA support (compatible with CUDA 13.1 driver)
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Copenhagen

# Install Python 3.12 and build tools
RUN apt update && \
    apt install --no-install-recommends -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install --no-install-recommends -y python3.12 python3.12-venv build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Set python3.12 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Copy project files
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY LICENSE LICENSE
COPY README.md README.md


# Install dependencies
WORKDIR /
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
RUN pip install . --no-deps --no-cache-dir

# Set up DVC (but don't pull data yet - will pull at runtime)
RUN dvc init --no-scm
COPY .dvc/config .dvc/config
COPY *.dvc ./
RUN dvc config core.no_scm true

# Copy entrypoint script that will pull data at runtime
COPY dockerfiles/train_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--epochs", "5"]
