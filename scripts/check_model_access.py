"""Report whether a language-model provider can be reached, and why not.

The dashboard and the pipeline fall back quietly from one provider to the
next, so a wrong key, an exhausted quota and a provider outage all look the
same on screen ("LLM synthesis unavailable"). This prints what each provider
actually answered.

Key values are never printed: only whether a key is set, and how long it is.

    python -m scripts.check_model_access
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def describe_key(name: str) -> str:
    """Whether the key is set and how it looks - never its value."""
    raw = os.environ.get(name)
    if raw is None:
        return "not set"
    value = raw.strip()
    notes = []
    if value != raw:
        notes.append("has surrounding spaces or a line break")
    if value[:1] in ("'", '"') or value[-1:] in ("'", '"'):
        notes.append("is wrapped in quotes - remove them")
    if not value:
        return "set but empty"
    return f"set, {len(value)} characters" + (f" ({'; '.join(notes)})" if notes else "")


def main() -> int:
    from dotenv import load_dotenv

    env_path = REPO_ROOT / ".env"
    print(f".env file: {'found' if env_path.exists() else 'NOT FOUND'} at {env_path}")
    if not env_path.exists():
        print("  On Windows, check the file is called .env and not .env.txt "
              "(File Explorer hides known extensions by default).")
    load_dotenv(env_path)

    from src.rag import explainer

    print("\nKeys:")
    for name in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"):
        print(f"  {name}: {describe_key(name)}")

    # A model that reasons before answering spends output tokens on the
    # reasoning, so a small budget truncates the JSON and the provider
    # answers 400 rather than refusing the key. The pipeline allows 3,000
    # (blind_evidence.EXTRACTION_MAX_OUTPUT_TOKENS); 400 is ample here.
    prompt = 'Reply with exactly this JSON and nothing else: {"ok": true}'
    budget = 400
    print("\nAsking each configured provider for one short answer:")
    reached_one = False

    if os.environ.get("GEMINI_API_KEY"):
        for model in explainer.MODEL_CANDIDATES:
            try:
                explainer.call_one_model("gemini", model, prompt, max_output_tokens=budget)
                print(f"  gemini / {model}: OK")
                reached_one = True
                break
            except Exception as exc:  # noqa: BLE001 - every provider error shape
                print(f"  gemini / {model}: {type(exc).__name__}: {str(exc)[:160]}")

    for config in explainer.FALLBACK_CONFIGS:
        if not os.environ.get(config["env_key"]):
            continue
        for model in config["models"]:
            try:
                explainer.call_one_model(config["provider_name"], model, prompt,
                                         max_output_tokens=budget)
                print(f"  {config['provider_name']} / {model}: OK")
                reached_one = True
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  {config['provider_name']} / {model}: "
                      f"{type(exc).__name__}: {str(exc)[:160]}")

    print("\nHow to read the answer above:")
    print("  OK                       this provider works; a run can use it")
    print("  401 / invalid_api_key    the key is wrong, or the file was not reloaded")
    print("  429 / rate limit / quota the key works; the allowance is used up for now")
    print("  503 / ServerError        the provider is down; try the other one")
    print("  400 / failed to validate  this check's own answer was cut short, not a key")
    print("                           problem; the pipeline allows far more room")
    if not reached_one:
        print("\nNo provider answered. A run started now would end in a fault, "
              "not a refusal.")
    return 0 if reached_one else 1


if __name__ == "__main__":
    raise SystemExit(main())
