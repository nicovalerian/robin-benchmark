#!/usr/bin/env python3
"""
Test which DigitalOcean serverless inference models are accessible with the current key.

Three API formats:
  - chat_completions : POST /v1/chat/completions  (DO-hosted models + Anthropic via DO)
  - responses        : POST /v1/responses         (OpenAI models via DO only)
  - anthropic_msgs   : POST /v1/messages          (Anthropic native format via DO)
"""

import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_URL = "https://inference.do-ai.run"
PROBE = "Reply with exactly: OK"
TIMEOUT = 60

MODELS = [
    # (display_name, model_id, api_format)
    # --- DO-hosted (Chat Completions) ---
    ("gemma-4-31b",              "gemma-4-31B-it",                   "chat"),
    ("qwen3-32b",                "alibaba-qwen3-32b",                 "chat"),
    ("kimi-k2.5",                "kimi-k2.5",                         "chat"),
    ("kimi-k2.6",                "kimi-k2.6",                         "chat"),
    ("llama3.3-70b",             "llama3.3-70b-instruct",             "chat"),
    ("llama-4-maverick",         "llama-4-maverick",                  "chat"),
    ("deepseek-r1-llama-70b",    "deepseek-r1-distill-llama-70b",    "chat"),
    ("deepseek-v4-pro",          "deepseek-v4-pro",                   "chat"),
    ("deepseek-3.2",             "deepseek-3.2",                      "chat"),
    ("minimax-m2.5",             "minimax-m2.5",                      "chat"),
    ("mistral-3-14b",            "mistral-3-14B",                     "chat"),
    ("nemotron-super-120b",      "nvidia-nemotron-3-super-120b",      "chat"),
    ("glm-5",                    "glm-5",                             "chat"),
    ("gpt-oss-120b",             "openai-gpt-oss-120b",               "chat"),
    ("gpt-oss-20b",              "openai-gpt-oss-20b",                "chat"),
    ("qwen3.5-397b",             "qwen3.5-397b-a17b",                 "chat"),
    ("qwen3-coder-flash",        "qwen3-coder-flash",                 "chat"),
    ("arcee-trinity-large",      "arcee-trinity-large-thinking",      "chat"),
    # --- Anthropic via DO (Chat Completions proxy) ---
    ("claude-haiku-4.5",         "anthropic-claude-haiku-4.5",        "chat"),
    ("claude-sonnet-4.6",        "anthropic-claude-4.6-sonnet",       "chat"),
    # --- OpenAI via DO (Responses API) ---
    ("gpt-5-nano",               "openai-gpt-5-nano",                 "responses"),
    ("gpt-5-mini",               "openai-gpt-5-mini",                 "responses"),
    ("gpt-4o-mini",              "openai-gpt-4o-mini",                "responses"),
    ("gpt-4.1",                  "openai-gpt-4.1",                    "responses"),
    ("gpt-5",                    "openai-gpt-5",                      "responses"),
]


async def probe_chat(session: aiohttp.ClientSession, api_key: str, model_id: str) -> tuple[bool, str, float]:
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROBE}],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT),
        ) as resp:
            latency = (time.perf_counter() - t0) * 1000
            data = await resp.json()
            if resp.status == 200:
                msg = data["choices"][0]["message"]
                # Some reasoning models return null content with reasoning_content
                text = msg.get("content") or msg.get("reasoning_content") or ""
                if text is None:
                    text = ""
                return True, text.strip()[:80], latency
            err = data.get("error", {})
            error_str = err.get("message", str(data)) if isinstance(err, dict) else str(err)
            return False, error_str[:120], latency
    except Exception as e:
        return False, str(e)[:120], (time.perf_counter() - t0) * 1000


async def probe_responses(session: aiohttp.ClientSession, api_key: str, model_id: str) -> tuple[bool, str, float]:
    url = f"{BASE_URL}/v1/responses"
    payload = {
        "model": model_id,
        "input": PROBE,
        "max_output_tokens": 16,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=TIMEOUT),
        ) as resp:
            latency = (time.perf_counter() - t0) * 1000
            data = await resp.json()
            if resp.status == 200:
                # Responses API output format
                output = data.get("output", [])
                text = ""
                for item in output:
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                text += c.get("text", "")
                if not text:
                    # fallback: choices-style (some DO models return both)
                    choices = data.get("choices", [])
                    if choices:
                        text = choices[0].get("message", {}).get("content", "")
                return True, text.strip()[:80], latency
            err = data.get("error", {})
            msg = err.get("message", str(data)) if isinstance(err, dict) else str(err)
            return False, msg[:120], latency
    except Exception as e:
        return False, str(e)[:120], (time.perf_counter() - t0) * 1000


async def probe_model(session, api_key, display_name, model_id, api_format):
    if api_format == "responses":
        ok, detail, latency = await probe_responses(session, api_key, model_id)
    else:
        ok, detail, latency = await probe_chat(session, api_key, model_id)

    status = "OK " if ok else "ERR"
    lat_str = f"{latency:6.0f}ms"
    print(f"  [{status}] {lat_str}  {display_name:<26} ({model_id})")
    if not ok:
        print(f"           -> {detail}")
    return {
        "name": display_name,
        "model_id": model_id,
        "api_format": api_format,
        "ok": ok,
        "latency_ms": round(latency),
        "detail": detail,
    }


async def main():
    api_key = os.getenv("DIGITALOCEAN_INFERENCE_KEY", "")
    if not api_key or len(api_key) < 10:
        print("ERROR: DIGITALOCEAN_INFERENCE_KEY not set or too short.")
        sys.exit(1)

    print(f"\nProbing {len(MODELS)} models on DigitalOcean Serverless Inference...")
    print(f"Base URL : {BASE_URL}")
    print(f"Probe    : '{PROBE}'")
    print(f"Timeout  : {TIMEOUT}s per model\n")

    sem = asyncio.Semaphore(8)

    async def limited(session, *args):
        async with sem:
            return await probe_model(session, *args)

    async with aiohttp.ClientSession() as session:
        tasks = [limited(session, api_key, dn, mid, fmt) for dn, mid, fmt in MODELS]
        results = await asyncio.gather(*tasks)

    working = [r for r in results if r["ok"]]
    failed  = [r for r in results if not r["ok"]]

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(working)}/{len(results)} models working")
    print(f"{'='*60}")

    if working:
        print("\nWorking models (copy-paste ready for full_config.yaml):")
        for r in working:
            print(f"  - name: \"{r['name']}\"")
            print(f"    provider: \"digitalocean\"")
            print(f"    model_id: \"{r['model_id']}\"")
            print(f"    api_format: \"{r['api_format']}\"   # latency ~{r['latency_ms']}ms")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  [ERR] {r['name']}: {r['detail'][:100]}")

    out_path = Path(__file__).parent.parent / "data" / "processed" / "do_model_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    import platform
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
