# OpenRouter catalogue analysis

Explores OpenRouter's public metadata API: which models exist, what they cost, what the free tier offers, and how providers compare on price, quantization, uptime and data retention.

No API key is needed. All endpoints used are public.

## Layout

| File | Purpose |
|---|---|
| `fetch.py` | Downloads a dated snapshot to `data/raw/<YYYY-MM-DD>/` (gzipped JSON, about 300 KB) |
| `tables.py` | Flattens a snapshot into pandas tables (`models`, `endpoints`, `providers`). Run it directly to write Parquet files to `data/processed/` |
| `viz.py` | Shared chart styling |
| `notebooks/01_catalogue_free_providers.ipynb` | Analysis: the catalogue, the free models and provider comparisons |

## Quick start

```bash
pip install -r requirements.txt
python fetch.py            # about 450 requests, ~10-30s
jupyter nbconvert --to notebook --execute --inplace notebooks/01_catalogue_free_providers.ipynb
```

The notebook always loads the latest snapshot in `data/raw/`.

## Endpoints used

| Endpoint | Contents |
|---|---|
| `GET /api/v1/models` | The catalogue: pricing, context and output limits, input/output types, supported parameters, reasoning settings, Artificial Analysis benchmarks, knowledge cutoff, retirement dates |
| `GET /api/v1/models/{id}/endpoints` | Every provider endpoint for one model: price, quantization, limits, uptime (5m/30m/1d) and status |
| `GET /api/v1/endpoints/zdr` | Endpoints with zero data retention |
| `GET /api/v1/providers` | Provider headquarters, data centres, and links to policies and status pages |

## Caveats

- Each snapshot records a single moment. The daily workflow builds the history needed to track price changes and how long free models last.
- `latency_last_30m` and `throughput_last_30m` are blank in the public responses.
- `created` is when a model was added to OpenRouter, not its original release date.
- "Open weights" means the listing has a Hugging Face link, which is a proxy.
