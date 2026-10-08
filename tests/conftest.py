"""
Shared pytest setup.

src.pipeline transitively imports src.rag.explainer, which imports
`google.genai` (the Gemini client) at module load. That package is only
needed to call the LLM, which these tests never do since they mock or avoid
the network entirely. To let the pipeline module import in a plain test
environment (CI, a fresh checkout without the Gemini SDK installed), we
register a minimal stub for `google.genai` in sys.modules before any test
imports the pipeline. If the real package is installed, we leave it alone.
"""

import sys
import types


def _install_genai_stub():
    try:
        import google.genai  # noqa: F401  (real package present - use it)
        return
    except Exception:  # noqa: BLE001
        pass

    google = sys.modules.get("google") or types.ModuleType("google")
    genai = types.ModuleType("google.genai")

    class _Client:  # pragma: no cover - must never be constructed in tests
        def __init__(self, *a, **k):
            raise RuntimeError("stub google.genai.Client must not be used in tests")

    genai.Client = _Client
    genai.types = types.SimpleNamespace()
    google.genai = genai
    sys.modules["google"] = google
    sys.modules["google.genai"] = genai


_install_genai_stub()


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_test_reaches_a_model(monkeypatch):
    """A test that forgets to stand in for the model must fail here, not send
    a request with whatever key is in the developer's .env."""
    from src.rag import explainer

    def refuse(*args, **kwargs):
        raise RuntimeError("a test tried to call a model provider")

    monkeypatch.setattr(explainer, "_call_openai_compatible", refuse)
    monkeypatch.setattr(explainer, "get_client", refuse)
