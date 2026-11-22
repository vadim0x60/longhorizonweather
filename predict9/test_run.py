"""Exercise the public CLI with fixed local weather and changing climate indices."""

import csv
from datetime import date, timedelta
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class IndexForecastTest(unittest.TestCase):
    def test_winter_response_and_cutoff(self):
        sites = [('lima_pe', -12.0464, -77.0428), ('anchorage_us', 61.2181, -149.9003),
                 ('london_uk', 51.5074, -.1278), ('toronto_ca', 43.6532, -79.3832)]
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            command = [sys.executable, '-I', '-S', str(Path(__file__).with_name('run.py').resolve()),
                       '--year-x', '1999', '--data-dir', tmp, '--out', str(d/'out')]
            for (loc, lat, lon), index in zip(sites, ('nino34_anom', 'pdo', 'nao', 'ao')):
                with self.subTest(index=index):
                    (d/'manifest.json').write_text(json.dumps(dict(year_x=1999, max_date='1999-12-31',
                        locations=[dict(location_id=loc, lat=lat, lon=lon)])))
                    with (d/'daily_history.csv').open('w') as hist, (d/'daily_year_x.csv').open('w') as last:
                        writers = [csv.writer(f) for f in (hist, last)]
                        for w in writers:
                            w.writerow('location_id lat lon date temperature_2m_mean temperature_2m_max temperature_2m_min precipitation_sum wind_speed_10m_max'.split())
                        day = date(1990, 1, 1)
                        while day.year <= 1999:
                            row = (loc, lat, lon, day, 12, 14, 10, 2, 20)
                            writers[0].writerow(row)
                            if day.year == 1999:
                                writers[1].writerow(row)
                            day += timedelta(days=1)
                    for phase, direction in ((-2., 'lt'), (2., 'gt')):
                        rows = [dict(date=f'1999-{m}', value=phase) for m in (10, 11, 12)]
                        doc = dict(year_x=1999, max_month='1999-12', series={index:rows})
                        (d/'indices_monthly.json').write_text(json.dumps(doc))
                        subprocess.run(command, check=True, cwd=d)
                        claims = json.loads((d/'out').read_text())
                        jan = {c['target']['variable']:c['assertion'] for c in claims
                               if c['target']['window_end']=='2000-01-31'}
                        self.assertEqual(jan['temp_mean_c'], dict(kind='threshold', op=direction, value=12.))
                        if loc=='lima_pe':
                            self.assertEqual(jan['precip_sum_mm'], dict(kind='threshold', op=direction, value=62.))
            rows.append(dict(date='2000-01', value=1))
            (d/'indices_monthly.json').write_text(json.dumps(doc))
            (d/'out').unlink()
            fail = subprocess.run(command, capture_output=True, text=True, cwd=d)
            self.assertNotEqual(fail.returncode, 0)
            self.assertIn('index after forecast cutoff', fail.stderr)
            self.assertFalse((d/'out').exists())


if __name__ == '__main__':
    unittest.main()
