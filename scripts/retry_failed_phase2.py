#!/usr/bin/env python3
"""Retry exact Phase-2 cells listed in a JSONL file."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phase2.inference_runner import InferenceRunner, checkpoint_key, prepare_model_config  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n}: invalid JSON: {e}")
    return out


def choose_model(models_cfg: list[dict], token: str) -> dict:
    """Robust exact selection by name or model_id, with useful diagnostics."""
    if not isinstance(models_cfg, list):
        raise SystemExit(f"Config 'models' must be a list, got {type(models_cfg).__name__}")

    configured = []
    for m in models_cfg:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name", "")).strip()
        model_id = str(m.get("model_id", "")).strip()
        configured.append((name, model_id, m))

    # Exact match first.
    for name, model_id, m in configured:
        if token == name or token == model_id:
            return dict(m)

    # Normalized fallback (hyphen/underscore/case differences only).
    norm = lambda s: s.lower().replace("_", "-").strip()
    nt = norm(token)
    fuzzy = [m for name, model_id, m in configured if nt in (norm(name), norm(model_id))]
    if len(fuzzy) == 1:
        return dict(fuzzy[0])

    available = "\n".join(f"  - {name!r}  ({model_id!r})" for name, model_id, _ in configured)
    raise SystemExit(
        f"Model {token!r} not found in {Path.cwd()} config.\n"
        f"Configured models are:\n{available or '  <none>'}"
    )


def completed_keys(path: Path, cfg: dict) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for r in load_jsonl(path):
        if r.get("success") is not True:
            continue
        if r.get("model_name") != cfg["name"]:
            continue
        if r.get("protocol_hash") != cfg["_protocol_hash"]:
            continue
        done.add(checkpoint_key(
            cfg.get("provider", "openai-compatible"),
            cfg["_deployment_id"],
            cfg["_resolved_model_id"],
            r["sample_id"],
            int(r["level"]),
            cfg["_protocol_hash"],
        ))
    return done


async def main_async(args):
    load_dotenv()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    inference = cfg.get("inference") or {}
    raw_model = choose_model(cfg.get("models") or [], args.model)

    prepared = await prepare_model_config(
        raw_model,
        "",
        temperature=inference.get("temperature", 0.0),
        max_output_tokens=int(inference.get("max_output_tokens", 1024)),
        timeout_seconds=float(inference.get("timeout_seconds", 240)),
    )

    retry_rows = load_jsonl(Path(args.keys))
    prompts, seen = [], set()
    for row in retry_rows:
        sid, level = str(row["sample_id"]), int(row["level"])
        cell = (sid, level)
        if cell in seen:
            raise SystemExit(f"Duplicate retry key: {cell}")
        seen.add(cell)
        prompts.append({
            "id": sid,
            "level": level,
            "category": row.get("category", ""),
            "text": row["prompt"],
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(output, prepared)

    def ck(p):
        return checkpoint_key(
            prepared.get("provider", "openai-compatible"),
            prepared["_deployment_id"],
            prepared["_resolved_model_id"],
            p["id"],
            p["level"],
            prepared["_protocol_hash"],
        )

    remaining = sum(1 for p in prompts if ck(p) not in done)
    provider_rules = ((prepared.get("extra_body") or {}).get("provider") or {})

    print("=" * 64)
    print("ROBIN targeted Phase 2 retry")
    print("=" * 64)
    print(f"Config          : {cfg_path}")
    print(f"Model           : {prepared['name']} ({prepared['_resolved_model_id']})")
    print(f"Retry cells     : {len(prompts)}")
    print(f"Already clean  : {len(prompts)-remaining}")
    print(f"Remaining       : {remaining}")
    print(f"Provider rules  : {json.dumps(provider_rules, ensure_ascii=False)}")
    print(f"Protocol hash   : {prepared['_protocol_hash']}")
    print(f"Output          : {output}")
    print()

    runner = InferenceRunner(
        models_config=[prepared],
        api_key="",
        temperature=inference.get("temperature", 0.0),
        max_output_tokens=int(inference.get("max_output_tokens", 1024)),
        timeout_seconds=float(inference.get("timeout_seconds", 240)),
        max_concurrency=int(inference.get("max_concurrent", 8)),
        rate_limit_per_minute=int(inference.get("rate_limit_per_minute", 120)),
        max_retries=int(inference.get("max_retries", 4)),
    )

    lock = asyncio.Lock()
    bar = tqdm(total=len(prompts), initial=len(prompts)-remaining, unit="resp")
    try:
        await runner.run_model(
            model_cfg=prepared,
            prompts=prompts,
            output_path=output,
            completed=done,
            lock=lock,
            progress_cb=lambda: bar.update(1),
        )
    finally:
        bar.close()

    # Latest record per requested cell under current protocol.
    latest = {}
    if output.exists():
        for r in load_jsonl(output):
            if r.get("model_name") == prepared["name"] and r.get("protocol_hash") == prepared["_protocol_hash"]:
                latest[(r.get("sample_id"), int(r.get("level")))] = r

    requested = {(p["id"], p["level"]) for p in prompts}
    successful = {cell for cell, r in latest.items() if cell in requested and r.get("success") is True}
    missing = sorted(requested - successful)

    routes = {}
    reasoning = 0
    for cell, r in latest.items():
        if cell not in requested:
            continue
        if r.get("reasoning_violation"):
            reasoning += 1
        route = r.get("openrouter_selected_provider") or "metadata-unavailable"
        routes[route] = routes.get(route, 0) + 1

    print("\n" + "=" * 64)
    print("TARGETED RETRY COMPLETE")
    print("=" * 64)
    print(f"Requested cells      : {len(requested)}")
    print(f"Successful cells     : {len(successful)}")
    print(f"Still unresolved     : {len(missing)}")
    print(f"Reasoning violations : {reasoning}")
    print("Routes               : " + ", ".join(f"{k}={v}" for k, v in sorted(routes.items())))

    if missing:
        mp = output.with_suffix(".missing.jsonl")
        source = {(p["id"], p["level"]): p for p in prompts}
        with mp.open("w", encoding="utf-8") as f:
            for cell in missing:
                p = source[cell]
                f.write(json.dumps({
                    "sample_id": p["id"],
                    "level": p["level"],
                    "category": p.get("category", ""),
                    "prompt": p["text"],
                }, ensure_ascii=False) + "\n")
        print(f"Missing-key file     : {mp}")
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
