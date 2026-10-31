"""
Thin wrapper around a local Ollama server.

Member C's Phase 1 task: "Install and configure Ollama; run Llama 3 8B /
Mistral 7B quantized." This module assumes Ollama is running locally
(`ollama serve`, default http://localhost:11434) with a model pulled
(`ollama pull llama3:8b` or `ollama pull mistral:7b`).
"""
import os
import time
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")


class LLMError(Exception):
    pass


# def generate(prompt: str, model: str = None, system: str = None,
#              temperature: float = 0.0, timeout: int = 60) -> str:
#     """Single-shot generation against Ollama's /api/generate endpoint."""
#     payload = {
#         "model": model or DEFAULT_MODEL,
#         "prompt": prompt,
#         "stream": False,
#         "options": {"temperature": temperature},
#     }
#     if system:
#         payload["system"] = system

#     try:
#         resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=timeout)
#         resp.raise_for_status()
#     except requests.RequestException as e:
#         raise LLMError(
#             f"Could not reach Ollama at {OLLAMA_HOST}. Is `ollama serve` running "
#             f"and is the model pulled? Original error: {e}"
#         )
#     return resp.json().get("response", "").strip()

def generate(prompt: str, model: str = None, system: str = None,
             temperature: float = 0.0, timeout: int = 60) -> str:
    """Single-shot generation against Ollama's /api/generate endpoint."""

    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        },
    }

    if system:
        payload["system"] = system

    url = f"{OLLAMA_HOST}/api/generate"

    try:
        resp = requests.post(
            url,
            json=payload,
            timeout=timeout
        )

        if not resp.ok:
            raise LLMError(
                f"Ollama returned HTTP {resp.status_code}\n"
                f"URL: {url}\n"
                f"Model: {payload['model']}\n"
                f"Response: {resp.text}"
            )

        data = resp.json()

        if "response" not in data:
            raise LLMError(
                f"Ollama response does not contain 'response': {data}"
            )

        return data["response"].strip()

    except requests.RequestException as e:
        raise LLMError(
            f"Could not reach Ollama at {OLLAMA_HOST}. "
            f"Original error: {e}"
        )

def benchmark(prompt: str = "Say hello in one word.", n: int = 3, model: str = None) -> dict:
    """Member C's baseline latency benchmark (Phase 1 milestone)."""
    latencies = []
    for _ in range(n):
        start = time.time()
        generate(prompt, model=model)
        latencies.append(time.time() - start)
    return {
        "model": model or DEFAULT_MODEL,
        "runs": n,
        "avg_seconds": sum(latencies) / len(latencies),
        "min_seconds": min(latencies),
        "max_seconds": max(latencies),
    }


if __name__ == "__main__":
    print(benchmark())
