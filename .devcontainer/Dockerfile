FROM mcr.microsoft.com/devcontainers/python:3.13-jammy

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install jupyterlab notebook ipykernel

USER vscode

WORKDIR /workspaces/${localWorkspaceFolderBasename}
