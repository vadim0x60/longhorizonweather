"""Round-one forecast: rank plausible disagreements, not just easy events."""

import argparse
import calendar
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import NormalDist, median


def read_history(directory, x):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['year_x'] != x or date.fromisoformat(manifest['max_date']).year > x:
        raise ValueError('manifest does not match forecast cutoff')
    coords = {}
    # Sums, valid-day counts and extrema by location, year and calendar month.
    months = defaultdict(lambda: defaultdict(dict))
    with (directory / 'daily_history.csv').open(newline='') as src:
        for row in csv.DictReader(src):
            d = date.fromisoformat(row['date'])
            if d.year > x:
                raise ValueError('observation beyond forecast cutoff')
            loc = row['location_id']
            coords[loc] = (float(row['lat']), float(row['lon']))
            m = months[loc][d.year].setdefault(d.month, [0., 0, 0., 0, -math.inf, 0, math.inf, 0])
            for i, col in enumerate(('temperature_2m_mean', 'precipitation_sum',
                                     'temperature_2m_max', 'temperature_2m_min')):
                try:
                    v = float(row[col])
                except ValueError:
                    continue
                if math.isfinite(v):
                    k = 2 * i
                    m[k] = m[k] + v if i < 2 else (max if i == 2 else min)(m[k], v)
                    m[k + 1] += 1
    return coords, months


def superiority(history, current, x, rain):
    """Predict P(new > current), with year X excluded from the reference sample."""
    years = sorted(history)
    obs = [history[y] for y in years]
    center = median(obs)
    spread = 1.4826 * median(abs(v - center) for v in obs)
    slope = 0.
    if not rain and len(obs) >= 6:
        # Robust local temperature drift, regularized against short noisy records.
        raw = median((history[b] - history[a]) / (b - a) for a in years for b in years if b > a)
        slope = max(-0.12, min(0.12, raw)) * len(obs) / (len(obs) + 12)
    smooth = max(0.15, spread * 0.35)
    normal = NormalDist()
    votes = 0.
    for y, v in zip(years, obs):
        if rain and (current == 0 or v == 0):
            votes += v > current  # Ties at zero are false, not half a win.
        else:
            votes += normal.cdf((v + slope * (x + 1 - y) - current) / smooth)
    return (1 + votes) / (2 + len(years))


def predict(directory, x):
    coords, locations = read_history(directory, x)
    candidates = []
    windows = [(1, 12, 1.), (6, 8, .82), (3, 5, .68), (9, 11, .68),
               (1, 3, .54), (4, 6, .54), (7, 9, .54), (10, 12, .54)]
    windows += [(i, i, .43) for i in range(1, 13)]
    for loc, data in sorted(locations.items()):
        lat, lon = coords[loc]
        for start, end, overlap in windows:
            first = date(x + 1, start, 1).isoformat()
            last_day = calendar.monthrange(x + 1, end)[1]
            last = date(x + 1, end, last_day).isoformat()

            def add(var, assertion, p, weight, rationale):
                if p >= .64:
                    claim = dict(target=dict(location_id=loc, lat=lat, lon=lon, window_start=first,
                                             window_end=last, variable=var), assertion=assertion,
                                 confidence=round(p, 3), rationale=rationale)
                    candidates.append(((2 * p - 1) * overlap * weight, claim))

            for i, var in enumerate(('warmer_than_year_x', 'wetter_than_year_x',
                                     'hot_days_ge_30c', 'frost_days_lt_0c')):
                k = 2 * i
                values, winners = {}, {}
                for y, months in data.items():
                    ms = range(start, end + 1)
                    if not all(m in months and months[m][k+1] == calendar.monthrange(y, m)[1] for m in ms):
                        continue
                    if i < 2:
                        total = sum(months[m][k] for m in ms)
                        values[y] = total / sum(months[m][1] for m in ms) if i == 0 else total
                        if start == 1 and end == 12:
                            winners[y] = max(ms, key=lambda m: months[m][k] / (months[m][1] if i == 0 else 1))
                    else:
                        values[y] = (max if i == 2 else min)(months[m][k] for m in ms)
                if x not in values:
                    continue
                if winners:
                    month, count = Counter(winners[y] for y in sorted(winners)).most_common(1)[0]
                    p = (count + .5) / (len(winners) + 1.5)
                    add(('hottest_month', 'wettest_month')[i], dict(kind='categorical', value=f'{month:02}'),
                        p, 1.05 if winners[x] != month else .7,
                        f'Month {month:02} ranked first in {count} of {len(winners)} complete years.')
                if i >= 2:
                    # A safety margin in degrees lets even a single cold/hot year inform count claims.
                    margin = 30 - max(values.values()) if i == 2 else min(values.values())
                    if margin >= 3:
                        p = 1 - 1 / (2 + len(values) + min(10, margin) / 2)
                        add(var, dict(kind='threshold', op='le', value=0), p, .28,
                            f'All {len(values)} complete windows stayed ≥{margin:.1f} °C from the count threshold.')
                else:
                    current = values.pop(x)
                    if len(values) < 2 or (end == 2 and (last_day == 29 or calendar.isleap(x))):
                        continue
                    p = superiority(values, current, x, rain=i == 1)
                    above = sum(v > current for v in values.values())
                    add(var, dict(kind='boolean', value=p > .5), max(p, 1-p), 1.,
                        f'{above}/{len(values)} prior windows exceeded year X; '
                        + ('shrunken temperature trend.' if i == 0 else 'empirical rain distribution, ties not increases.'))
    ranked = sorted(candidates, key=lambda item: -item[0])[:150]
    return [dict(claim_id=f'p10_{j:03}', **c) for j, (_, c) in enumerate(ranked, 1)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year-x', type=int, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(predict(args.data_dir, args.year_x), indent=2, allow_nan=False) + '\n')
