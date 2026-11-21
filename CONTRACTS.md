# System Contracts & Anti-Cheat Rules

## 1. Directory Ownership & Git Workflow

- Thread `i` (`1..10`) owns **`predict{i}/`** only. Never edit another predictor's `predict{j}/` directory, `scores.json`, or `CONTRACTS.md`.
- Any thread may propose additive, backward-compatible updates to `datacollect/` or `verify/` (the orchestrator audits all changes before merging into `master`).
- Do **not** push directly to `origin/master`; commit or stage changes in your own orb and report back to the orchestrator thread. Pull `origin/master` at the start of each new round.

## 2. Anti-Cheat & Temporal Causality Rules (Strictly Enforced)

1. **No year `X+1` leakage in `datacollect` or `predict{i}`:**
   - `datacollect/run.py --year <X>` may ONLY collect observations dated `<= {X}-12-31`. Never query `{X+1}` or later.
   - `predict{i}/run.py` must run **offline** against `--data-dir` and files inside `predict{i}/` (exception: `predict1` and `predict2` may call Tinker inference using strictly `<= X` data from `--data-dir`).
   - Do **not** hardcode historical weather outcomes, per-year answer tables, or future-year lookups.
2. **Tinker API budget (`TINKER_API_KEY`):**
   - ONLY **`predict1`** and **`predict2`** (two Opus 5.5 threads) are authorized to use `TINKER_API_KEY`, each with a hard **$50.00 USD** total spend cap across the entire session. Log estimated/observed spend in `predict{i}/tinker_budget.json`.
   - **`predict3` through `predict10` are strictly forbidden** from importing `tinker` or reading/calling `TINKER_API_KEY`.
3. **No tampering:** Never inspect `verify/` ground-truth outputs at prediction time or alter `verify/` to bias scoring toward a specific predictor.

## 3. CLI Interfaces

Python `3.13` and `uv` are installed. If a directory needs packages beyond Python stdlib, place a `requirements.txt` inside that directory (e.g. `predict1/requirements.txt`). Keep runtime fast (< 3 minutes per `predict{i}` invocation for a given year `X`).

### `datacollect/run.py`
```bash
python3 datacollect/run.py --year <X> --out <data_dir>
```
- Writes `manifest.json` and year `<= X` weather/climate datasets into `<data_dir>`.
- Must support any historical integer `X` in `1980..2024`.

### `predict{i}/run.py`
```bash
python3 predict{i}/run.py --year-x <X> --data-dir <data_dir> --out <claims_json>
```
- Reads `<data_dir>` (output of `datacollect/run.py`) and writes a JSON array of claim objects to `<claims_json>`.

### `verify/run.py`
```bash
python3 verify/run.py --year-x <X> --claims-dir <claims_dir> --out <report_json>
```
- Reads `<claims_dir>/predict1.json` .. `predict10.json`.
- Detects pairwise contradictions between predictors on the same canonical target.
- Evaluates ground truth for year `X+1` (independently of `datacollect/`).
- Writes `<report_json>` with `scores_delta` (`+1` winner, `-1` loser, `0` if both wrong or unresolvable) and per-contradiction breakdown.

## 4. Claim JSON Schema

Each entry in `<claims_json>` must follow:

```json
{
  "claim_id": "p1_001",
  "target": {
    "location_id": "london_uk",
    "lat": 51.5074,
    "lon": -0.1278,
    "window_start": "2019-06-01",
    "window_end": "2019-08-31",
    "variable": "temp_mean_c"
  },
  "assertion": {
    "kind": "threshold",
    "op": "gt",
    "value": 17.5
  },
  "confidence": 0.72,
  "rationale": "Short explanation"
}
```

### Canonical `target` fields
- `location_id`: canonical string from the shared catalog below (or `grid_{lat:.2f}_{lon:.2f}` for custom coordinates).
- `lat`, `lon`: WGS84 decimal degrees.
- `window_start`, `window_end`: ISO dates `YYYY-MM-DD` strictly inside year `X+1` (`{X+1}-01-01` .. `{X+1}-12-31`).
- `variable`: canonical variable name (extensible via `verify/`):
  - `temp_mean_c`: mean daily `temperature_2m_mean` (°C)
  - `temp_max_c`: max daily `temperature_2m_max` (°C)
  - `temp_min_c`: min daily `temperature_2m_min` (°C)
  - `temp_mean_max_c`: mean daily `temperature_2m_max` (°C)
  - `temp_mean_min_c`: mean daily `temperature_2m_min` (°C)
  - `precip_sum_mm`: total `precipitation_sum` (mm)
  - `precip_max_day_mm`: max single-day `precipitation_sum` (mm)
  - `wind_max_kmh`: max daily `wind_speed_10m_max` (km/h)
  - `hot_days_ge_30c`: count of days with `temperature_2m_max >= 30.0`
  - `frost_days_lt_0c`: count of days with `temperature_2m_min < 0.0`
  - `rainy_days_ge_1mm`: count of days with `precipitation_sum >= 1.0`
  - `wettest_month`: `"01"`..`"12"` (month with highest `precip_sum_mm` within window)
  - `hottest_month`: `"01"`..`"12"` (month with highest `temp_mean_c` within window)
  - `warmer_than_year_x`: boolean (`temp_mean_c(X+1 window) > temp_mean_c(same month-day window in X)`)
  - `wetter_than_year_x`: boolean (`precip_sum_mm(X+1 window) > precip_sum_mm(same month-day window in X)`)

### Supported `assertion` kinds
1. `threshold`: `{"kind": "threshold", "op": "gt"|"ge"|"lt"|"le", "value": <number>}`
2. `interval`: `{"kind": "interval", "low": <number>, "high": <number>}` (inclusive)
3. `boolean`: `{"kind": "boolean", "value": true|false}`
4. `categorical`: `{"kind": "categorical", "value": "<string>"}`

### Anti-spam & Scoring Rules
- **At most 1 claim per `(predictor, location_id, window_start, window_end, variable)`** and **at most 150 claims per predictor per round**. Duplicates or excess claims are rejected by `verify`.
- For each unordered pair `(predict{i}, predict{j})` with a claim on the exact same canonical `target` whose assertions are mathematically disjoint (cannot both be true):
  - If `i` is True and `j` is False: `predict{i}` gets `+1`, `predict{j}` gets `-1`.
  - If `j` is True and `i` is False: `predict{j}` gets `+1`, `predict{i}` gets `-1`.
  - If both are False: `0` / `0`.

## 5. Suggested Shared Benchmark Locations

Using shared `location_id`s ensures head-to-head contradictions occur across diverse climate zones (though predictors may also add custom coordinates):

| `location_id` | City / Region | `lat` | `lon` |
|---|---|---|---|
| `london_uk` | London, UK | 51.5074 | -0.1278 |
| `new_york_us` | New York, USA | 40.7128 | -74.0060 |
| `tokyo_jp` | Tokyo, Japan | 35.6762 | 139.6503 |
| `sydney_au` | Sydney, Australia | -33.8688 | 151.2093 |
| `cairo_eg` | Cairo, Egypt | 30.0444 | 31.2357 |
| `mumbai_in` | Mumbai, India | 19.0760 | 72.8777 |
| `sao_paulo_br` | Sao Paulo, Brazil | -23.5505 | -46.6333 |
| `reykjavik_is` | Reykjavik, Iceland | 64.1466 | -21.9426 |
| `nairobi_ke` | Nairobi, Kenya | -1.2921 | 36.8219 |
| `phoenix_us` | Phoenix, USA | 33.4484 | -112.0740 |
| `singapore_sg` | Singapore | 1.3521 | 103.8198 |
| `chicago_us` | Chicago, USA | 41.8781 | -87.6298 |
| `moscow_ru` | Moscow, Russia | 55.7558 | 37.6173 |
| `cape_town_za` | Cape Town, South Africa | -33.9249 | 18.4241 |
| `lima_pe` | Lima, Peru | -12.0464 | -77.0428 |
| `bangkok_th` | Bangkok, Thailand | 13.7563 | 100.5018 |
| `anchorage_us` | Anchorage, USA | 61.2181 | -149.9003 |
| `madrid_es` | Madrid, Spain | 40.4168 | -3.7038 |
| `seoul_kr` | Seoul, South Korea | 37.5665 | 126.9780 |
| `toronto_ca` | Toronto, Canada | 43.6532 | -79.3832 |
