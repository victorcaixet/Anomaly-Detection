import unittest

import pandas as pd

from insulator_anomaly.evaluation import val_threshold
from insulator_anomaly.figures import pick_cases


class ShowcaseTests(unittest.TestCase):
    def test_threshold_from_validation_normals_and_extreme_cases(self):
        self.assertEqual(val_threshold(pd.Series(range(20)), 0.05), 19)
        test = pd.DataFrame(dict(y_true=[0, 0, 1, 1], pred_score=[1, 5, 3, 9]))
        got = {kind: row.pred_score for kind, row in pick_cases(test, 4)}
        self.assertEqual(got, dict(TP=9, FN=3, FP=5, TN=1))


if __name__ == "__main__":
    unittest.main()
