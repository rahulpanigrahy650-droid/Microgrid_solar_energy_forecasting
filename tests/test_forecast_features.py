import math
import unittest

from app import build_feature_row


class ForecastFeatureTests(unittest.TestCase):
    def test_build_feature_row_uses_recent_history_for_lags(self):
        history = [1.0, 2.0, 3.0, 4.0, 5.0]

        row = build_feature_row(
            hour_val=14,
            day_of_year=172,
            temperature=27.5,
            humidity=55.0,
            cloud_cover=20.0,
            wind_speed=11.0,
            ghi=650.0,
            history=history,
        )

        self.assertEqual(row["prev_solar_kw"], 5.0)
        self.assertEqual(row["solar_roll_3h"], 4.0)
        self.assertEqual(row["solar_lag_24h"], 0.0)
        self.assertEqual(row["hour_sin"], math.sin(2 * math.pi * 14 / 24))
        self.assertEqual(row["hour_cos"], math.cos(2 * math.pi * 14 / 24))


if __name__ == "__main__":
    unittest.main()
