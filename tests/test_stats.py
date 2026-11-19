import unittest

from graft import stats


class LowerBound(unittest.TestCase):
    def test_all_pass_is_the_rule_of_three_family(self):
        self.assertAlmostEqual(stats.lower_bound(10, 10), 0.05 ** 0.1, places=9)
        self.assertAlmostEqual(stats.lower_bound(300, 300), 0.05 ** (1 / 300), places=9)

    def test_matches_clopper_pearson_tables(self):
        # one-sided 95% lower limits for 8 of 10 and 45 of 50
        self.assertAlmostEqual(stats.lower_bound(8, 10), 0.49310, places=4)
        self.assertAlmostEqual(stats.lower_bound(45, 50), 0.80165, places=3)

    def test_edges(self):
        self.assertEqual(stats.lower_bound(0, 10), 0.0)
        self.assertEqual(stats.lower_bound(0, 0), 0.0)
        self.assertLess(stats.lower_bound(10, 10, 0.99), stats.lower_bound(10, 10, 0.95))

    def test_the_bound_is_where_the_tail_probability_meets_alpha(self):
        bound = stats.lower_bound(17, 20)
        tail = 1.0 - stats._binom_cdf(16, 20, bound)
        self.assertAlmostEqual(tail, 0.05, places=6)


class SignTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(stats.sign_test(0, 0), 1.0)
        self.assertEqual(stats.sign_test(5, 5), 1.0)
        self.assertAlmostEqual(stats.sign_test(7, 0), 2 * 0.5 ** 7)
        self.assertAlmostEqual(stats.sign_test(0, 7), 2 * 0.5 ** 7)
        self.assertAlmostEqual(stats.sign_test(9, 1), 2 * 11 / 1024)


if __name__ == "__main__":
    unittest.main()
