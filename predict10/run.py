"""Price public forecast boundaries from local weather history; no live rival inputs."""

import argparse
import calendar
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import NormalDist, mean, median, pstdev


FIELDS = ('temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum')
VARS = ('temp_mean_c', 'temp_mean_max_c', 'temp_mean_min_c', 'precip_sum_mm',
        'hot_days_ge_30c', 'frost_days_lt_0c')
WINDOWS = [(1, 12), (1, 2), (3, 5), (6, 8), (9, 11), (1, 3), (4, 6), (7, 9), (10, 12)]
WINDOWS += [(m, m) for m in range(1, 13)]
NORMAL = NormalDist()


def read_inputs(directory, x):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['year_x'] != x or manifest['max_date'] > f'{x}-12-31':
        raise ValueError('manifest exceeds cutoff or mismatches year')
    coords, monthly = {}, defaultdict(lambda: defaultdict(dict))
    with (directory / 'daily_history.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            d = date.fromisoformat(row['date'])
            if d.year > x:
                raise ValueError('observation beyond forecast cutoff')
            loc = row['location_id']
            coords[loc] = (float(row['lat']), float(row['lon']))
            # Six sufficient statistics, four non-missing observation counts.
            sums = monthly[loc][d.year].setdefault(d.month, [0.] * 10)
            for i, field in enumerate(FIELDS):
                if row[field] and math.isfinite(v := float(row[field])):
                    sums[i] += v
                    sums[i + 6] += 1
                    if i == 1:
                        sums[4] += v >= 30
                    elif i == 2:
                        sums[5] += v < 0
    enso = {}
    if 'indices_monthly.json' in manifest.get('files', []):
        indices = json.loads((directory / 'indices_monthly.json').read_text())
        if indices['year_x'] != x or indices['max_month'] > f'{x}-12':
            raise ValueError('index metadata exceeds cutoff or mismatches year')
        ond = defaultdict(dict)
        for row in indices['series'].get('nino34_anom', []):
            y, m = map(int, row['date'].split('-'))
            if y > x:
                raise ValueError('index observation beyond forecast cutoff')
            if m in (10, 11, 12) and math.isfinite(row['value']):
                ond[y][m] = row['value']
        enso = {y: mean(v.values()) for y, v in ond.items() if len(v) == 3}
    return coords, monthly, enso


def series(data, a, b, i):
    out = {}
    for y, months in data.items():
        counts = (6, 7, 8, 9, 7, 8)[i]
        ms = range(a, b + 1)
        if all(m in months and months[m][counts] == calendar.monthrange(y, m)[1] for m in ms):
            total = sum(months[m][i] for m in ms)
            out[y] = total / sum(months[m][counts] for m in ms) if i < 3 else total
    return out


def slopes(obs):
    return sorted((vb - va) / (b - a) for a, va in obs.items() for b, vb in obs.items() if b > a)


def conditioned_draws(obs, x, drift, signals, weights):
    """Transport each analogue's residual to today's climate state with damped local regressions."""
    draws = {y: v + drift * (x+1-y) for y, v in obs.items()}
    for signal, strength in signals:
        years = [y for y in obs if y-1 in signal]
        if x not in signal or len(years) < 8:
            continue
        center = mean(signal[y-1] for y in years)
        scale = pstdev(signal[y-1] for y in years)
        if scale < .1:
            continue
        level = mean(draws[y] for y in years)
        # Five unit-variance pseudo-observations shrink a one-predictor regression toward no effect.
        beta = sum((draws[y]-level)*(signal[y-1]-center)/scale for y in years)/(len(years)+5)
        today = max(-2.5, min(2.5, (signal[x]-center)/scale))
        for y in draws:
            draws[y] += strength * beta * (today - (signal.get(y-1, center)-center)/scale)
    return [(v, weights(y)) for y, v in draws.items()]


def analog_line(obs, x, rain, annual):
    """Estimate the price set by the round-one annual-drift/jitter analogue strategy."""
    ann_slopes = slopes(annual)
    n = len(annual)
    drift = (n * ann_slopes[len(ann_slopes)//2] + .02 * 30) / (n + 30) if n >= 8 else .02
    recent = [annual[y] + drift * (x - y) for y in annual if y > x - 10]
    carry = .05 * (annual[x] - mean(recent))
    jitter = (-1.2816, -.5244, 0., .5244, 1.2816)
    if n < 8:
        jitter = [NORMAL.inv_cdf((j + .5) / 15) for j in range(15)]
    draws = sorted(((v * math.exp(z * (.12 if n >= 8 else .2773)) if rain else
                     v + drift * (x + 1 - y) + carry + z * (.5 if n >= 8 else .7810)),
                    math.exp((y - x) / 25)) for y, v in obs.items() for z in jitter)
    total = sum(w for _, w in draws)

    def quantile(q):
        cumul = 0.
        for v, w in draws:
            cumul += w
            if cumul >= q * total:
                return v
        return draws[-1][0]

    up = quantile(.5) >= median(obs.values())
    return round(quantile(.38 if up else .62), 1), up


def forecasts(coords, monthly, enso, x, reprice):
    best = {}
    for loc, data in sorted(monthly.items()):
        annual = series(data, 1, 12, 0)
        if x not in annual:
            continue
        annual_slopes = slopes(annual)
        background = .02 if not annual_slopes else .012 + .4 * max(-.10, min(.15, median(annual_slopes)))
        autumn = {y: v + background*(x-y) for y,v in series(data, 10, 12, 0).items()}
        for a, b in WINDOWS:
            ann = (a, b) == (1, 12)
            popular = 1. if ann else .7 if (a, b) in WINDOWS[1:5] else .28
            # Match each analogue's outcome in y to its predecessor's autumn, not y's autumn.
            # ENSO's influence is diluted after spring. A weight floor avoids tiny analogue sets.
            def weight(y):
                w = math.exp((y - x) / 9)
                if x in enso and y-1 in enso:
                    phase = math.exp(-.5 * ((enso[x] - enso[y-1]) / .8) ** 2)
                    w *= 1 + math.exp(-(a+b-2)/15) * (phase - .5)
                return w

            def add(i, assertion, p, score, why):
                key = (loc, a, b, i)
                if score > 0 and (key not in best or score > best[key][0]):
                    target = dict(location_id=loc, lat=coords[loc][0], lon=coords[loc][1], variable=i,
                                  window_start=date(x+1, a, 1).isoformat(),
                                  window_end=date(x+1, b, calendar.monthrange(x+1, b)[1]).isoformat())
                    best[key] = score, dict(target=target, assertion=assertion, confidence=round(p, 3), rationale=why)

            for i, var in enumerate(VARS):
                obs = series(data, a, b, i)
                if x not in obs:
                    continue
                n, count, rain = len(obs), i >= 4, i == 3
                if n < 3:
                    # With no interannual dispersion, only near-impossible temperature counts are useful.
                    temp = series(data, a, b, 1 if i == 4 else 2) if count else {}
                    if count and all(v == 0 for v in obs.values()) and temp:
                        margin = 30 - max(temp.values()) if i == 4 else min(temp.values())
                        if margin > 12:
                            add(var, dict(kind='threshold', op='le', value=0), .7, popular * min(margin, 35)/100,
                                'No qualifying days in the available record; window mean far from threshold.')
                    continue
                local = median(slopes(obs))
                drift = .70*background + .30*max(-.15, min(.15, local)) if i < 3 else 0.
                draws = [(v + drift * (x+1-y), weight(y)) for y, v in obs.items()]
                if i < 3:
                    # Carryover of local heat fades faster than ocean-temperature teleconnections.
                    # Learn the sign at each site: a positive Niño anomaly need not predict local warmth.
                    signals = [(enso, math.exp(-(a-1)/8)), (autumn, .75*math.exp(-(a+b-2)/9))]
                    draws = conditioned_draws(obs, x, drift, signals, weight)
                spread = pstdev(v for v, _ in draws)
                bandwidth = max(.12 if i < 3 else .01, .32 * spread)

                def prob_le(t):
                    s = 0.
                    for v, w in draws:
                        if count or (rain and (v == 0 or t == 0)):
                            s += w * (v <= t)
                        else:
                            s += w * NORMAL.cdf((t - v)/bandwidth)
                    return (.8 + s) / (1.6 + sum(w for _, w in draws))

                # Different rival families announce different prices on the same target.
                # Test each complement independently; never spend a slot on a negative estimated edge.
                def offer(line, rival_up, encounter, name, strict_above=False):
                    if (count or rain) and not rival_up and line <= 0 and not strict_above:
                        return  # A nonnegative-domain tautology cannot win a genuine disagreement.
                    if count and rival_up and line >= sum(calendar.monthrange(x+1,m)[1] for m in range(a,b+1)):
                        return
                    boundary = line - 1 if count and not rival_up else line
                    p = prob_le(boundary)
                    if not rival_up:
                        p = 1 - p
                    if p > .505 and (encounter >= .8 or p >= .54):
                        op = 'le' if rival_up else ('gt' if strict_above else 'ge')
                        p = reprice(loc, a, b, i, name, op, line, p, encounter)
                        add(var, dict(kind='threshold', op=op, value=line), p, (2*p-1)*encounter,
                            f'{n} local analogues, lagged climate signals and past pricing errors; P({op} {line})={p:.2f} at {name}.')

                if (ann and i < 4 or (a,b) in WINDOWS[2:5] and i == 0) and len(annual) >= 3:
                    line, up = analog_line(obs, x, rain, annual)
                    offer(line, up, 1., 'analogue quantile')
                base = mean(obs.values())
                if (ann or (a, b) == (6, 8)) and i in (0, 3) and n >= 6 and spread > 0:
                    # Predict4 uses a Gaussian trend/AR forecast and bids halfway back to X.
                    # A few very wet years can put even that midpoint above the local rain median.
                    mid = mean(obs)
                    ols = sum((y-mid)*(v-base) for y,v in obs.items()) / sum((y-mid)**2 for y in obs)
                    trend = (n*ols + 20*(.025 if i == 0 else 0)) / (n+20)
                    residual = {y: v-base-trend*(y-mid) for y,v in obs.items()}
                    pairs = [(residual[y-1], v) for y,v in residual.items() if y-1 in residual]
                    den = sum(u*u for u,_ in pairs)
                    ar = max(0., min(.5, sum(u*v for u,v in pairs)/den*len(pairs)/(10+len(pairs)))) if den else 0.
                    center = base + trend*(x+1-mid) + ar*residual[x]
                    if rain:
                        center = max(0., center)
                    if abs(center-obs[x]) >= .6 * spread:
                        offer(round((center+obs[x])/2, 2), center > obs[x], .82, 'Gaussian reversion midpoint', True)
                if not rain and n >= 10 and (not count or base >= 3):
                    # Predict5 announces the mean (floored for hot days, ceiled for frost days).
                    # A recent slope of the expected sign makes that claim more likely to survive its cap.
                    line = (math.floor(base) if i == 4 else math.ceil(base) if i == 5 else round(base, 2))
                    z = (1 if i != 5 else -1) * local * n / (2 * max(spread, .1))
                    offer(line, i != 5, max(.015, min(.9, (z-.35)*1.4)), 'warming-model baseline')
                # Predict1 prices its thresholds at a 15-year mean. Both sides are possible.
                base15 = round(mean(v for y, v in obs.items() if y >= x-14), 2)
                if len(set(obs.values())) > 1:
                    for up in (True, False):
                        line = (math.floor(base15) if up else math.ceil(base15)) if count else base15
                        offer(line, up, (.035 if count else .085) * popular, 'trailing climatology')
                if i in (0, 3) and not (b == 2 and calendar.isleap(x)):
                    p = prob_le(obs[x])
                    val = p < .5
                    p = max(p, 1-p)
                    if p >= .58:
                        # A forecast people already agree with earns nothing. Discount predictable reversion.
                        prior = [v for y, v in obs.items() if y < x]
                        usual = obs[x] < median(prior)
                        add(('warmer_than_year_x' if i == 0 else 'wetter_than_year_x'),
                            dict(kind='boolean', value=val), p, (2*p-1)*popular*(.95 if usual != val else .10),
                            f'{n} local windows with recency, temperature drift and available lagged ENSO weights.')
                if i in (0, 3) and b > a and (ann or (a,b) in WINDOWS[1:5]):
                    raw, weighted = Counter(), Counter()
                    for y in obs:
                        vals = {m: round(data[y][m][i] / (data[y][m][6] if i == 0 else 1), 6) for m in range(a,b+1)}
                        top = max(vals.values())
                        peaks = [m for m in vals if vals[m] == top]
                        if len(peaks) == 1:
                            raw[peaks[0]] += 1
                            weighted[peaks[0]] += weight(y)
                    if weighted:
                        m, votes = weighted.most_common(1)[0]
                        flat = raw.most_common(1)[0][0]
                        p = (votes + .5/(b-a+1)) / (weighted.total() + .5)
                        edge = (votes - weighted[flat])/(weighted.total()+.5) if flat != m else .06*(2*p-1)
                        if p >= .35:
                            add('hottest_month' if i == 0 else 'wettest_month', dict(kind='categorical', value=f'{m:02}'),
                                p, edge*popular, f'Month {m:02} leads recency/ENSO-weighted peaks ({raw[m]}/{raw.total()} unweighted).')
    ranked = sorted(best.values(), key=lambda z: -z[0])[:150]
    return [dict(claim_id=f'p10_{j:03}', **c) for j, (_, c) in enumerate(ranked, 1)]


def predict(directory, x):
    coords, monthly, enso = read_inputs(directory, x)
    years = sorted({y for site in monthly.values() for y in site})
    errors = defaultdict(list)

    def family(name, i, op):
        return (name, 'temperature' if i < 3 else VARS[i], op == 'le')

    for cutoff in years[5:-1][-8:]:
        if cutoff + 1 not in years:
            continue
        # Explicitly separate fit inputs from the one completed year used to evaluate that fold.
        past = {loc: {y: v for y, v in site.items() if y <= cutoff} for loc, site in monthly.items()}
        past_enso = {y: v for y, v in enso.items() if y <= cutoff}
        observed = {loc: {cutoff+1: site[cutoff+1]} for loc, site in monthly.items() if cutoff+1 in site}
        by_city = defaultdict(list)

        def record(loc, a, b, i, name, op, line, p, encounter):
            value = series(observed.get(loc, {}), a, b, i).get(cutoff+1)
            if value is not None and encounter >= .08:
                value = round(value, 6)
                hit = value <= line if op == 'le' else value >= line if op == 'ge' else value > line
                by_city[(family(name, i, op), loc)].append(hit - p)
            return p

        forecasts(coords, past, past_enso, cutoff, record)
        by_family = defaultdict(list)
        for (group, loc), residuals in by_city.items():
            by_family[group].append(mean(residuals))
        for group, city_errors in by_family.items():
            # Cities and windows within a year are correlated. Don't count them as independent years.
            errors[group].append(mean(city_errors))

    def calibrate(loc, a, b, i, name, op, line, p, encounter):
        past = errors[family(name, i, op)]
        # Start mildly skeptical of taking the low-probability side of an analogue quantile
        # or Gaussian reversion line. Then incorporate forward, not fitted, pricing errors.
        initial = .42 if name in ('analogue quantile', 'Gaussian reversion midpoint') else .5
        correction = (2*(initial-p) + sum(past)) / (4+len(past))
        return max(.01, min(.99, p + correction))

    return forecasts(coords, monthly, enso, x, calibrate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year-x', type=int, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(predict(args.data_dir, args.year_x), indent=2, allow_nan=False) + '\n')
