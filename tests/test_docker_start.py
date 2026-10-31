import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("docker_start", ROOT / "scripts" / "docker_start.py")
docker_start = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(docker_start)


class _FakeCollection:
    def __init__(self, n):
        self._n = n

    def count(self):
        return self._n


def _run(existing, built):
    calls = []

    def fake_build(reset=False):
        calls.append(reset)
        return built

    logs = []
    result = docker_start.ensure_index(lambda: _FakeCollection(existing), fake_build, logs.append)
    return result, calls, logs


def test_existing_index_is_never_touched():
    result, calls, _ = _run(existing=42, built=99)
    assert result == "existing"
    assert calls == []  # build_index must not even be called


def test_empty_index_gets_built_without_wiping_anything():
    result, calls, logs = _run(existing=0, built=13)
    assert result == "built"
    assert calls == [False]  # reset=False: never destructive
    assert any("13 chunks" in line for line in logs)


def test_no_pdfs_is_reported_not_fatal():
    result, _, logs = _run(existing=0, built=0)
    assert result == "empty"
    assert any("WARNING" in line for line in logs)


def test_startup_continues_to_serve_even_if_index_preparation_crashes(monkeypatch, capsys):
    started = {}
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, host, port: started.update(app=app, host=host, port=port)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    def broken():
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(docker_start, "ensure_index", broken)
    monkeypatch.delenv("PORT", raising=False)
    docker_start.main()

    assert started == {"app": "backend.app:app", "host": "0.0.0.0", "port": 8000}
    assert "chroma exploded" in capsys.readouterr().out