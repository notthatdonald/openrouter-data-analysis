"""Flatten a raw OpenRouter snapshot (see fetch.py) into pandas tables.

    from tables import load_tables
    t = load_tables()            # latest snapshot
    t.models, t.endpoints, t.providers

Prices are converted from USD/token strings to USD per million tokens.

Run as a script to also write Parquet files to data/processed/<date>/.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"

VARIANT_SUFFIXES = ("free", "batch")
TEXT_OUTPUT_PARAMS = {
    "supports_tools": "tools",
    "supports_structured_outputs": "structured_outputs",
    "supports_response_format": "response_format",
    "supports_reasoning": "reasoning",
    "supports_logprobs": "logprobs",
    "supports_web_search": "web_search_options",
}


@dataclass
class Snapshot:
    date: str
    models: pd.DataFrame
    endpoints: pd.DataFrame
    providers: pd.DataFrame


def snapshot_dates() -> list[str]:
    return sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir())


def _read(date: str, name: str):
    path = RAW_DIR / date / f"{name}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _per_m(value) -> float:
    """USD/token string -> USD per million tokens (NaN if absent)."""
    if value is None or value == "":
        return float("nan")
    return float(value) * 1_000_000


def split_variant(model_id: str) -> tuple[str, str | None]:
    """'qwen/qwen3.8-27b:free' -> ('qwen/qwen3.8-27b', 'free')."""
    base, _, suffix = model_id.partition(":")
    return base, (suffix or None)


def _models(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for m in raw:
        arch = m.get("architecture") or {}
        price = m.get("pricing") or {}
        top = m.get("top_provider") or {}
        params = set(m.get("supported_parameters") or [])
        reasoning = m.get("reasoning") or {}
        aa = ((m.get("benchmarks") or {}).get("artificial_analysis")) or {}
        ins = set(arch.get("input_modalities") or [])
        outs = set(arch.get("output_modalities") or [])
        base_id, variant = split_variant(m["id"])
        price_in, price_out = _per_m(price.get("prompt")), _per_m(price.get("completion"))
        is_alias = m["id"].startswith("~")
        # Routers (openrouter/auto etc.) publish -1 as "price depends on the model picked";
        # openrouter/free routes to the free models, so it is priced at 0 instead.
        is_router = price_in < 0 or price_out < 0 or base_id.startswith("openrouter/")
        row = {
            "id": m["id"],
            "base_id": base_id,
            "variant": variant,
            "author": base_id.lstrip("~").split("/")[0],
            "name": m.get("name"),
            "created": pd.to_datetime(m.get("created"), unit="s", utc=True),
            "context_length": m.get("context_length"),
            "max_completion_tokens": top.get("max_completion_tokens"),
            "is_moderated": top.get("is_moderated"),
            "modality": arch.get("modality"),
            "tokenizer": arch.get("tokenizer"),
            "input_modalities": ",".join(sorted(ins)),
            "output_modalities": ",".join(sorted(outs)),
            **{f"input_{k}": k in ins for k in ("image", "file", "audio", "video")},
            **{f"output_{k}": k in outs for k in ("image", "audio")},
            "price_in": price_in,
            "price_out": price_out,
            "price_cache_read": _per_m(price.get("input_cache_read")),
            "has_tiered_pricing": bool(price.get("overrides")),
            **{col: p in params for col, p in TEXT_OUTPUT_PARAMS.items()},
            "reasoning_mandatory": bool(reasoning.get("mandatory")),
            "aa_intelligence": aa.get("intelligence_index"),
            "aa_coding": aa.get("coding_index"),
            "aa_agentic": aa.get("agentic_index"),
            "hugging_face_id": m.get("hugging_face_id") or None,
            "knowledge_cutoff": m.get("knowledge_cutoff"),
            "expiration_date": m.get("expiration_date"),
            "is_alias": is_alias,
            "is_router": is_router,
            "is_free": (not is_router) and price_in == 0 and price_out == 0,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df["open_weights"] = df["hugging_face_id"].notna()
    # 3:1 input:output blend - the usual convention for a single "price per M tokens".
    df["price_blended"] = (3 * df["price_in"] + df["price_out"]) / 4
    # One row per underlying model: drop aliases, routers and :batch/:free duplicates.
    df["is_primary"] = ~df["is_alias"] & ~df["is_router"] & df["variant"].isna()
    return df


def _endpoints(raw: dict, zdr: list[dict]) -> pd.DataFrame:
    zdr_keys = {(e["model_id"], e["tag"]) for e in zdr}
    rows = []
    for model_id, payload in raw.items():
        for e in payload.get("endpoints", []):
            price = e.get("pricing") or {}
            params = set(e.get("supported_parameters") or [])
            rows.append({
                "model_id": model_id,
                "base_id": split_variant(model_id)[0],
                "variant": split_variant(model_id)[1],
                "provider": e.get("provider_name"),
                "tag": e.get("tag"),
                "quantization": e.get("quantization") or "unknown",
                "context_length": e.get("context_length"),
                "max_completion_tokens": e.get("max_completion_tokens"),
                "price_in": _per_m(price.get("prompt")),
                "price_out": _per_m(price.get("completion")),
                "price_cache_read": _per_m(price.get("input_cache_read")),
                "discount": price.get("discount") or 0,
                "status": e.get("status"),
                "uptime_5m": e.get("uptime_last_5m"),
                "uptime_30m": e.get("uptime_last_30m"),
                "uptime_1d": e.get("uptime_last_1d"),
                "latency_30m": e.get("latency_last_30m"),
                "throughput_30m": e.get("throughput_last_30m"),
                "supports_tools": "tools" in params,
                "supports_implicit_caching": bool(e.get("supports_implicit_caching")),
                "zdr": (model_id, e.get("tag")) in zdr_keys,
            })
    df = pd.DataFrame(rows)
    df["price_blended"] = (3 * df["price_in"] + df["price_out"]) / 4
    return df


def _providers(raw: list[dict], endpoints: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(raw).rename(columns={"name": "provider"})
    df["datacenters"] = df["datacenters"].apply(lambda v: ",".join(v) if isinstance(v, list) else None)
    served = endpoints.groupby("provider").agg(
        n_endpoints=("model_id", "size"),
        n_models=("base_id", "nunique"),
        zdr_share=("zdr", "mean"),
        median_uptime_1d=("uptime_1d", "median"),
    )
    return df.merge(served, left_on="provider", right_index=True, how="left")


def load_tables(date: str | None = None) -> Snapshot:
    date = date or snapshot_dates()[-1]
    models = _models(_read(date, "models")["data"])
    zdr = (_read(date, "endpoints_zdr") or {"data": []})["data"]
    endpoints = _endpoints(_read(date, "endpoints") or {}, zdr)
    providers = _providers(_read(date, "providers")["data"], endpoints)
    n_prov = endpoints.groupby("model_id")["provider"].nunique()
    models["n_providers"] = models["id"].map(n_prov).fillna(0).astype(int)
    return Snapshot(date, models, endpoints, providers)


def main() -> None:
    ap = argparse.ArgumentParser(description="Write Parquet tables for a snapshot.")
    ap.add_argument("--date", help="snapshot date (default: latest)")
    args = ap.parse_args()
    snap = load_tables(args.date)
    out = DATA_DIR / "processed" / snap.date
    out.mkdir(parents=True, exist_ok=True)
    for name in ("models", "endpoints", "providers"):
        df = getattr(snap, name)
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(df)} rows -> {out / f'{name}.parquet'}")


if __name__ == "__main__":
    main()
