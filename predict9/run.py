"""Local seasonal climatology, slow thermal drift and decaying late-year memory."""

import argparse
import calendar
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, median, NormalDist


# Geographic exposure, not a table of climate observations. Intermediate sites
# (including estuaries, or inland sites near large lakes) use the middle prior.
MARITIME = {'london_uk', 'tokyo_jp', 'sydney_au', 'mumbai_in', 'reykjavik_is',
            'singapore_sg', 'cape_town_za', 'lima_pe'}
CONTINENTAL = {'moscow_ru', 'phoenix_us', 'madrid_es', 'nairobi_ke', 'cairo_eg'}


def index_states(folder, x):
    """December ocean/atmosphere state, including historical forecast origins."""
    path = folder / 'indices_monthly.json'
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    if doc['year_x'] != x or doc['max_month'] > f'{x}-12':
        raise ValueError('indices do not match forecast cutoff')
    out = defaultdict(dict)
    for name, rows in doc['series'].items():
        months = defaultdict(dict)
        for row in rows:
            d = date.fromisoformat(row['date'] + '-01')
            if d.year > x:
                raise ValueError('index after forecast cutoff')
            if d.month >= 10 and math.isfinite(row['value']):
                months[d.year][d.month] = row['value']
        for y, vals in months.items():
            # SSTs evolve slowly; modes of atmospheric circulation decorrelate
            # quickly, so only the latest month carries across the forecast origin.
            if name in ('nao', 'ao'):
                v = vals.get(12, 0.)
            elif 12 in vals and len(vals) >= 2:
                v = mean(vals.values())
            else:
                continue
            out[y][name] = max(-3., min(3., v))
    return out


def loading(loc, lat, lon, rain):
    """First-season response in °C or log rainfall, per unit of index."""
    tropical = abs(lat) < 25
    north_pacific = loc in ('anchorage_us', 'tokyo_jp', 'seoul_kr')
    europe = -15 < lon < 45 and lat > 47
    if rain:
        # Walker circulation: warming brings convection eastward away from
        # the Maritime Continent. Avoid assuming one rain sign across Europe.
        enso = (.35 if loc == 'lima_pe' else -.22 if loc in ('singapore_sg', 'sydney_au', 'bangkok_th')
                else .17 if loc in ('phoenix_us', 'nairobi_ke') else 0.)
        nao = .14 if loc == 'london_uk' else -.18 if loc == 'madrid_es' else 0.
        return enso, .07 if loc == 'phoenix_us' else 0., nao, 0.
    enso = (1.0 if loc == 'lima_pe' else .42 if tropical else .48 if loc == 'anchorage_us'
            else -.25 if loc == 'phoenix_us' else .15 if north_pacific else .12)
    pdo = .55 if loc == 'anchorage_us' else -.20 if loc in ('tokyo_jp', 'seoul_kr') else 0.
    # NAO and AO share an annular component: split their loading in Europe.
    return enso, pdo, .65 if europe else 0., .45 if europe or (lat > 38 and -90 < lon < -60) else 0.


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
                    float(row['precipitation_sum'] or 'nan'),
                    float(row['temperature_2m_max'] or 'nan'),
                    float(row['temperature_2m_min'] or 'nan'))
    months = defaultdict(dict)
    for (loc, y, m), days in rows.items():
        n = calendar.monthrange(y, m)[1]
        if len(days) == n:
            t, p, hi, lo = zip(*days.values())
            months[loc][y, m] = (sum(t) / n, sum(p) if min(p) >= 0 else math.nan,
                                 sum(hi) / n, sum(lo) / n)
    return months, coords


def window(data, a, b, index, years):
    result = {}
    for y in years:
        values = [data.get((y, m), (math.nan,) * 4)[index] for m in range(a, b + 1)]
        if all(map(math.isfinite, values)):
            if index != 1:
                days = [calendar.monthrange(y, m)[1] for m in range(a, b + 1)]
                result[y] = sum(v * n for v, n in zip(values, days)) / sum(days)
            else:
                result[y] = sum(values)
    return result


def drift(series, prior, shrink):
    slopes = [(series[t] - series[s]) / (t - s) for s in series for t in series if t - s >= 4]
    w = len(slopes) / (len(slopes) + shrink)
    return (1 - w) * prior + w * max(-.12, min(.16, median(slopes) if slopes else 0))


def claims(x, folder):
    months, coords = read_months(folder, x)
    states = index_states(folder, x)
    candidates = []
    for loc, data in sorted(months.items()):
        lat, lon = coords[loc]
        years = sorted({y for y, _ in data})
        annual = window(data, 1, 12, 0, years)
        # Short records mix forced warming and internal variability. Shrink
        # multi-year slopes toward a small background drift and limit extrapolation.
        slope = drift(annual, .02, 80)
        sea = 1 if loc in MARITIME else 0 if loc in CONTINENTAL else .5
        ond = [window(data, 10, 12, i, years) for i in range(2)]

        def anomaly(series, drift):
            prior = [v + drift * (x - y) for y, v in series.items() if y < x]
            return series[x] - median(prior) if x in series and len(prior) >= 3 else 0

        heat = .7 * anomaly(ond[0], slope) + .3 * anomaly(annual, slope)
        log_rain = {y: math.log1p(v) for y, v in ond[1].items()}
        moisture = max(-1, min(1, anomaly(log_rain, 0)))

        def add(a, b, var, assertion, prob, rationale, priority=0.):
            if prob < (.56 if priority > 0 else .64):
                return
            target = dict(location_id=loc, lat=lat, lon=lon,
                          window_start=date(x + 1, a, 1).isoformat(),
                          window_end=date(x + 1, b, calendar.monthrange(x + 1, b)[1]).isoformat(),
                          variable=var)
            # Prefer commonly compared windows at similar confidence.
            rank = prob + priority + (.14 if (a, b) == (1, 12) else .07 if b > a else 0)
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

        spans = [(1, 12), (1, 2), (1, 3), (3, 5), (6, 8), (9, 11), (10, 12)]
        for a, b in spans + [(m, m) for m in range(1, 13)]:
            mid = (a + b) / 2
            decay = math.exp(-mid / (1.7 + 2.5 * sea))
            for i, var in enumerate(('temp_mean_c', 'precip_sum_mm', 'temp_mean_max_c', 'temp_mean_min_c')):
                rain = i == 1
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
                if not rain:
                    trend = drift(obs, slope, 120)
                    ref = [v + trend * (x + 1 - y) for y, v in obs.items() if y < x]
                    shift = decay * (.18 + .32 * sea) * max(-3, min(3, heat))
                    # More stored water drives evaporation by day, narrowing the
                    # diurnal range; moisture/cloud weakly buffer night cooling.
                    shift += (0., 0., -.45, .16)[i] * moisture * math.exp(-mid / 2.5)
                    # Continental winter temperature has greater unresolved variability.
                    winter = (1 + math.cos((mid - (1 if lat >= 0 else 7)) * math.pi / 6)) / 2
                    floor = .35 + .8 * abs(math.sin(math.radians(lat))) * (1 - .5 * sea) * winter
                    comparison = obs[x]
                    why = f'Shrunk local drift {trend:+.3f} °C/yr; heat/moisture carryover {shift:+.2f} °C.'
                else:
                    n = lambda y: sum(calendar.monthrange(y, m)[1] for m in range(a, b + 1))
                    ref = [math.log1p(v * n(x + 1) / n(y)) for y, v in obs.items() if y < x]
                    shift = .12 * math.exp(-mid / 2.5) * moisture
                    floor = .16
                    comparison = math.log1p(obs[x])
                    why = 'Local seasonal rainfall distribution with a short-lived OND moisture anomaly.'
                # Compare like with like: remove the forcing at each historical
                # origin and apply this December's state, not future index values.
                # Average persistence over the window rather than its midpoint.
                weights = [c * mean(math.exp(-(m - .5) / tau) for m in range(a, b + 1))
                           for c, tau in zip(loading(loc, lat, lon, i == 1), (4.5, 8., 1.1, 1.1))]

                def forcing(origin):
                    return sum(w * states.get(origin, {}).get(k, 0.) for w, k in
                               zip(weights, ('nino34_anom', 'pdo', 'nao', 'ao')) if k in states.get(x, {}))

                past = [forcing(y - 1) for y in obs if y < x]
                current = forcing(x)
                adjustment = current - median(past)
                ref = [v + shift + current - old for v, old in zip(ref, past)]
                if i == 1:
                    ref = [max(0., v) if obs[y] > 0 else 0. for v, y in zip(ref, (y for y in obs if y < x))]
                center = median(ref)
                spread = max(floor, 1.4826 * median(abs(v - center) for v in ref))
                unit = 'log1p(mm)' if rain else '°C'
                why += f' Index adjustment {adjustment:+.2f} {unit} versus historical origins.'

                def probability(bound, *, up=True):
                    sign = 1 if up else -1
                    p = NormalDist().cdf(sign * (center - bound) / spread)
                    empirical = mean(sign * (v - bound) > 0 for v in ref)
                    return .5 + (.55 * p + .45 * empirical - .5) * len(ref) / (len(ref) + 2)

                prob = probability(comparison)
                naive_up = median(obs.values()) > obs[x]
                contested = (prob > .5) != naive_up
                # With a Feb 28 end in X+1 the comparator omits X's leap day;
                # do not claim that comparison from a complete-February total.
                if i < 2 and not (b == 2 and calendar.isleap(x)):
                    add(a, b, 'wetter_than_year_x' if rain else 'warmer_than_year_x', dict(kind='boolean', value=prob > .5),
                        max(prob, 1 - prob), why, .33 if contested else -.12)

                # Published climatologies commonly use 15 full years or all
                # 16 available years. Take the stricter line for the chosen side.
                lines = [round(mean(v for y, v in obs.items() if y >= x - n + 1), 2) for n in (15, 16)]
                up = center > (math.log1p(mean(lines)) if rain else mean(lines))
                line = max(lines) if up else min(lines)
                if rain and not up and line <= 0:
                    continue
                p = probability(math.log1p(line) if rain else line, up=up)
                # Distinguish predicted cooling from a skewed sample whose
                # median happens to lie below its mean. Only elevate the former.
                signal = shift + adjustment + (0 if rain else trend * (x + 1 - mean(obs)))
                bonus = .14 + (.28 * min(1., max(0., -signal / spread)) if not rain and not up else 0.)
                # If the model supports retaining most of an observed anomaly,
                # make the stronger claim against a halfway-reversion forecast.
                retain = round(.60 * obs[x] + .40 * line, 2)
                q = probability(math.log1p(retain) if rain else retain, up=up)
                if (retain > line if up else retain < line) and q >= .64:
                    line, p = retain, q
                    why += ' Forecast retains at least 60% of the year-X anomaly.'
                    bonus += .12
                add(a, b, var, dict(kind='threshold', op='gt' if up else 'lt', value=line), p, why, bonus)

    ordered = sorted(candidates, key=lambda item: -item[0])[:150]
    return [dict(claim_id=f'p9_{j:03}', **c) for j, (_, c) in enumerate(ordered, 1)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year-x', required=True, type=int)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(claims(args.year_x, args.data_dir), indent=2) + '\n')
