# Long-Horizon Weather Prediction Arena

Multi-agent iterative scientific benchmark: predict weather in year `X+1` strictly from information available up to `Dec 31` of year `X`.

## Repository Structure

- `datacollect/`: collects input data strictly for year `X` (and historical baselines `<= X`).
- `predict1/` .. `predict10/`: 10 competing predictor codebases that ingest year `X` data and emit verifiable weather claims about year `X+1`.
- `verify/`: detects pairwise contradictions among `predict1..10` claims, fetches ground-truth weather for `X+1`, and scores `+1` (winner) / `-1` (loser) per resolved disagreement.
- `scores.json`: cumulative leaderboard and round-by-round history maintained by the orchestrator.
- `CONTRACTS.md`: CLI interfaces, claim JSON schema, shared benchmark locations, and anti-cheat rules.
