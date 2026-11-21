"""Local seasonal climatology, slow thermal drift and decaying late-year memory."""

import argparse
import calendar
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import median, NormalDist


# Geographic exposure, not a table of climate observations. Intermediate sites
# (including estuaries, or inland sites near large lakes) use the middle prior.
MARITIME = {'london_uk', 'tokyo_jp', 'sydney_au', 'mumbai_in', 'reykjavik_is',
            'singapore_sg', 'cape_town_za', 'lima_pe'}
CONTINENTAL = {'moscow_ru', 'phoenix_us', 'madrid_es', 'nairobi_ke', 'cairo_eg'}


def read_months(folder, x):
    manifest = json.loads((folder / 'manifest.json').read_text())
    cutoff = date(x, 12, 31)
    if manifest['year_x'] != x or date.fromisoformat(manifest['max_date']) > cutoff:
        raise ValueError('manifest does not match forecast cutoff')
    rows = defaultdict(dict)
    coords = {}
    for filename in ('daily_history.csv', 'daily_year_x.csv'):
        with (folder / filename).open() as f:
            for row in csv.DictReader(f):
                d = date.fromisoformat(row['date'])
                if d > cutoff:
                    raise ValueError('observation after forecast cutoff')
                loc = row['location_id']
                coords[loc] = float(row['lat']), float(row['lon'])
                rows[loc, d.year, d.month][d.day] = (
                    float(row['temperature_2m_mean'] or 'nan'),
                    float(row['precipitation_sum'] or 'nan'))
    months = defaultdict(dict)
    for (loc, y, m), days in rows.items():
        n = calendar.monthrange(y, m)[1]
        if len(days) == n:
            t, p = zip(*days.values())
            months[loc][y, m] = sum(t) / n, sum(p) if min(p) >= 0 else math.nan
    return months, coords


def window(data, a, b, index, years):
    result = {}
    for y in years:
        values = [data.get((y, m), (math.nan, math.nan))[index] for m in range(a, b + 1)]
        if all(map(math.isfinite, values)):
            if index == 0:
                days = [calendar.monthrange(y, m)[1] for m in range(a, b + 1)]
                result[y] = sum(v * n for v, n in zip(values, days)) / sum(days)
            else:
                result[y] = sum(values)
    return result


def claims(x, folder):
    months, coords = read_months(folder, x)
    candidates = []
    for loc, data in sorted(months.items()):
        lat, lon = coords[loc]
        years = sorted({y for y, _ in data})
        annual = window(data, 1, 12, 0, years)
        slopes = [(annual[t] - annual[s]) / (t - s) for s in annual for t in annual if t - s >= 4]
        # Short records mix forced warming and internal variability. Shrink
        # multi-year slopes toward a small background drift and limit extrapolation.
        w = len(slopes) / (len(slopes) + 80)
        slope = (1 - w) * .02 + w * max(-.08, min(.12, median(slopes) if slopes else 0))
        sea = 1 if loc in MARITIME else 0 if loc in CONTINENTAL else .5
        ond = [window(data, 10, 12, i, years) for i in range(2)]

        def anomaly(series, drift):
            prior = [v + drift * (x - y) for y, v in series.items() if y < x]
            return series[x] - median(prior) if x in series and len(prior) >= 3 else 0

        heat = .7 * anomaly(ond[0], slope) + .3 * anomaly(annual, slope)
        log_rain = {y: math.log1p(v) for y, v in ond[1].items()}
        moisture = max(-1, min(1, anomaly(log_rain, 0)))

        def add(a, b, var, assertion, prob, rationale):
            if prob < .64:
                return
            target = dict(location_id=loc, lat=lat, lon=lon,
                          window_start=date(x + 1, a, 1).isoformat(),
                          window_end=date(x + 1, b, calendar.monthrange(x + 1, b)[1]).isoformat(),
                          variable=var)
            # Prefer commonly compared windows at similar confidence.
            rank = prob + (.08 if (a, b) == (1, 12) else .035 if b > a else 0)
            candidates.append((rank, dict(target=target, assertion=assertion,
                                          confidence=round(prob, 3), rationale=rationale)))

        for i, var in enumerate(('hottest_month', 'wettest_month')):
            winners = Counter()
            for y in years:
                vals = [data.get((y, m), (math.nan, math.nan))[i] for m in range(1, 13)]
                if all(map(math.isfinite, vals)) and vals.count(max(vals)) == 1:
                    winners[vals.index(max(vals)) + 1] += 1
            if winners:
                m, n = winners.most_common(1)[0]
                add(1, 12, var, dict(kind='categorical', value=f'{m:02}'),
                    (n + 1 / 12) / (winners.total() + 1),
                    f'Month {m} peaked in {n}/{winners.total()} complete, untied local years.')

        spans = [(1, 12), (1, 3), (3, 5), (6, 8), (9, 11), (10, 12)]
        for a, b in spans + [(m, m) for m in range(1, 13)]:
            mid = (a + b) / 2
            decay = math.exp(-mid / (1.7 + 2.5 * sea))
            for i, var in enumerate(('warmer_than_year_x', 'wetter_than_year_x')):
                obs = window(data, a, b, i, years)
                if i == 0 and (a, b) == (1, 12) and x in obs and len(obs) < 4:
                    # At the historical boundary a direction is not identifiable.
                    # Forecast annual mean temperature with a broad physical prior.
                    center = median(obs.values())
                    radius = 2 + (1 - .6 * sea) * abs(lat) / 35
                    add(a, b, 'temp_mean_c', dict(kind='interval', low=round(center - radius, 2),
                                                  high=round(center + radius, 2)), .75,
                        'Short record: annual mean with a broad latitude- and exposure-based margin.')
                if x not in obs or len(obs) < 4:
                    continue
                if i == 0:
                    ref = [v + slope * (x + 1 - y) for y, v in obs.items() if y < x]
                    center = median(ref)
                    shift = decay * (.18 + .32 * sea) * max(-3, min(3, heat))
                    # Continental winter temperature has greater unresolved variability.
                    winter = (1 + math.cos((mid - (1 if lat >= 0 else 7)) * math.pi / 6)) / 2
                    floor = .35 + .8 * abs(math.sin(math.radians(lat))) * (1 - .5 * sea) * winter
                    comparison = obs[x]
                    why = 'Local thermal climatology with shrunk drift and decaying OND/annual heat anomaly.'
                else:
                    n = lambda y: sum(calendar.monthrange(y, m)[1] for m in range(a, b + 1))
                    ref = [math.log1p(v * n(x + 1) / n(y)) for y, v in obs.items() if y < x]
                    center = median(ref)
                    shift = .12 * math.exp(-mid / 2.5) * moisture
                    floor = .16
                    comparison = math.log1p(obs[x])
                    why = 'Local seasonal rainfall distribution with a short-lived OND moisture anomaly.'
                spread = max(floor, 1.4826 * median(abs(v - center) for v in ref))
                p = NormalDist().cdf((center + shift - comparison) / spread)
                # Reconcile the parametric tail with observed support, especially
                # useful at very dry sites with repeated exactly-zero totals.
                empirical = sum(v + shift > comparison for v in ref) / len(ref)
                prob = .55 * p + .45 * empirical
                prob = .5 + (prob - .5) * len(ref) / (len(ref) + 2)
                add(a, b, var, dict(kind='boolean', value=prob > .5),
                    max(prob, 1 - prob), why)

    ordered = sorted(candidates, key=lambda item: -item[0])[:150]
    return [dict(claim_id=f'p9_{j:03}', **c) for j, (_, c) in enumerate(ordered, 1)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year-x', required=True, type=int)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(claims(args.year_x, args.data_dir), indent=2) + '\n')
