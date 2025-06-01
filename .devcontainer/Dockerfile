FROM mcr.microsoft.com/devcontainers/python:3.13-jammy

# Install any extra system packages you need (optional: cmake, etc.)
USER root
RUN apt-get update && apt-get install -y cmake

# Install Jupyter
RUN pip install --no-cache-dir notebook

# Switch back to the vscode user for development
USER vscode

# Set the workspace folder
WORKDIR /workspaces/${localWorkspaceFolderBasename}