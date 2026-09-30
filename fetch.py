"""Download a dated snapshot of OpenRouter's public (no-key) metadata endpoints.

Writes gzipped raw JSON to data/raw/<YYYY-MM-DD>/:
  models.json.gz          GET /api/v1/models
  providers.json.gz       GET /api/v1/providers
  endpoints_zdr.json.gz   GET /api/v1/endpoints/zdr
  endpoints.json.gz       {model_id: GET /api/v1/models/<model_id>/endpoints}

Usage:
  python fetch.py                  # full snapshot (~450 requests, ~10-30s)
  python fetch.py --skip-endpoints # catalogue only (3 requests)
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = "https://openrouter.ai/api/v1"
RAW_DIR = Path(__file__).parent / "data" / "raw"

session = requests.Session()
session.headers["User-Agent"] = "openrouter-catalogue-analysis/0.1"


def get(path: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = session.get(f"{BASE}/{path}", timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} for {path}")
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def write_json(path: Path, obj) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def fetch_model_endpoints(model_ids: list[str], workers: int) -> dict:
    """Per-model provider endpoints. Failures are recorded, not fatal."""
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(get, f"models/{mid}/endpoints"): mid for mid in model_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            mid = futures[fut]
            try:
                out[mid] = fut.result()["data"]
            except Exception as e:  # keep going; note the failure in the snapshot
                out[mid] = {"error": str(e)}
            if i % 50 == 0:
                print(f"  endpoints: {i}/{len(model_ids)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-endpoints", action="store_true", help="skip the per-model endpoint fan-out")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests for the fan-out")
    args = ap.parse_args()

    out_dir = RAW_DIR / dt.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    models = get("models")
    write_json(out_dir / "models.json.gz", models)
    print(f"models: {len(models['data'])}")

    for name, path in [("providers", "providers"), ("endpoints_zdr", "endpoints/zdr")]:
        data = get(path)
        write_json(out_dir / f"{name}.json.gz", data)
        print(f"{name}: {len(data['data'])}")

    if not args.skip_endpoints:
        # Aliases (~author/...-latest) just point at another model in the list.
        ids = [m["id"] for m in models["data"] if not m["id"].startswith("~")]
        endpoints = fetch_model_endpoints(ids, args.workers)
        write_json(out_dir / "endpoints.json.gz", endpoints)
        failed = [k for k, v in endpoints.items() if "error" in v]
        print(f"endpoints: {len(endpoints) - len(failed)} ok, {len(failed)} failed")

    print(f"snapshot written to {out_dir}")


if __name__ == "__main__":
    main()
