{
  "name": "Ames House Price ML Dev Container",
  "build": {
    "dockerfile": "Dockerfile"
  },
  "features": {},
  "postCreateCommand": "apk update && apk add gcc musl-dev python3-dev linux-headers && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python -m ipykernel install --user --name=venv --display-name 'Python (venv)'",
  "customizations": {
    "vscode": {
      "settings": {
        "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
        "python.terminal.activateEnvironment": true
      },
      "extensions": [
        "ms-azuretools.vscode-docker",
        "ms-vscode-remote.remote-containers",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "ms-toolsai.jupyter-slideshow",
        "ms-toolsai.jupyter-renderers",
        "ms-vscode.powershell",
        "ms-toolsai.datawrangler"
      ]
    }
  }
}
