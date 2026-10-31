# Running Omni-Agent Analytics with Docker

One command starts the whole system: the app, the local LLM server (Ollama) and the one-time model download. Nothing is sent to any cloud service.

## What you need

| | |
|---|---|
| Docker | Docker Desktop (Windows / macOS; on Windows use the WSL2 backend) or Docker Engine (Linux), with Compose v2 |
| Internet | Needed for the **first** run only: building the image and downloading the model (about 5GB in total). After that it works offline. |
| Disk | Roughly 15GB free (rough estimate: image + Ollama + the 4.7GB model) |
| Memory | 16GB RAM recommended. In Docker Desktop, allow it at least 8GB (Settings, Resources). |
| GPU | Optional. Without one the model runs on the CPU: it works, but answers take much longer. |

## Quick start

```bash
cd docker
docker compose up --build
```

Then open **http://localhost:8000**.

The first start takes a while: it builds the image, then downloads `llama3:8b` (~4.7GB) into a Docker volume. The app waits for the download to finish, so a blank page at first is normal. Watch progress with `docker compose logs -f ollama-pull`. Later starts take seconds.

## Using a GPU (NVIDIA)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Needs an NVIDIA GPU and, on Linux, the NVIDIA Container Toolkit (on Windows: Docker Desktop with WSL2 and a current NVIDIA driver).

## Using an Ollama you already have running

If Ollama is already installed on this machine with `llama3:8b` pulled, skip the bundled copy (saves the duplicate 4.7GB download and uses your GPU directly).

PowerShell:
```powershell
$env:OLLAMA_HOST_URL = "http://host.docker.internal:11434"
docker compose up --no-deps --build app
```
Bash:
```bash
OLLAMA_HOST_URL=http://host.docker.internal:11434 docker compose up --no-deps --build app
```

To use a different model, set `OLLAMA_MODEL` the same way (for example `llama3.1:8b`); it must already be pulled in that Ollama.

## Everyday commands

| Task | Command |
|---|---|
| Stop (keeps all data) | `docker compose down` |
| Start again | `docker compose up` |
| Rebuild after changing code | `docker compose up --build` |
| Watch the app's log | `docker compose logs -f app` |
| **Wipe everything** (uploads, uploaded database, index AND the downloaded model) | `docker compose down -v` |

## Where your data lives

Two Docker volumes keep data between restarts:

- `app_data` holds uploaded PDFs, the uploaded database, and the vector index.
- `ollama_models` holds the downloaded model.

On first start the document index is built automatically from the bundled PDFs. An index that already has content is never touched, so documents uploaded through the UI survive restarts.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Page won't load right after `up` | The model is still downloading: `docker compose logs -f ollama-pull` |
| "model requires more system memory" | Raise Docker Desktop's memory limit (Settings, Resources, or `memory=` in `.wslconfig` for WSL2) |
| Port 8000 already in use | In `docker-compose.yml` change `"127.0.0.1:8000:8000"` to e.g. `"127.0.0.1:8080:8000"` |
| Answers take minutes | You are on the CPU. Use a GPU, or expect this. |
| Build fails during `pip install` | Check the internet connection; re-run `docker compose up --build` and read the failing line |
| Document questions say "No indexed documents" | Check the app log for `[startup]` lines; make sure `data/pdfs/` contains PDFs |

## Security notes

- The app has **no login** and is a single-user tool: one shared document index and one active database. By default it is reachable only from this computer (`127.0.0.1`). Do not expose port 8000 to a network without adding authentication.
- Uploaded files are validated, but treat uploaded PDFs and `.sql` scripts as untrusted content.

## Sharing with a client

1. Send the source **without** your own data: `data/chroma`, `data/pdfs/uploads` and `data/db/uploads` are git-ignored and excluded from the image, so a `git clone` is clean. If you zip the folder instead, delete those three folders first.
2. Before delivery, pin exact library versions: run `pip freeze` in your working environment and copy the versions of the packages in `docker/requirements.docker.txt` (and `torch`) into that file, so the build is reproducible months later.
3. Tell the client: the hardware table above, that it is single-user with no login, and that the first run needs internet.
4. Check the licenses that apply to them: Docker Desktop's terms (free for personal use, education, and small businesses; larger companies need a paid plan; Docker Engine on Linux is free) and the Meta Llama 3 Community License for the model.