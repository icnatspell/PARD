import unittest

from scripts.check_prepared_dataset import check_sequence


class PreparedBoundaryTest(unittest.TestCase):
    def check(self, ids, limit=100):
        return check_sequence(ids, assistant_header=[10, 11, 12],
                              start_id=10, end_id=13, max_length=limit)

    def test_final_assistant_answer_passes(self):
        self.assertIsNone(self.check([10, 20, 12, 2, 13, 10, 11, 12, 3, 13]))

    def test_final_user_turn_fails(self):
        self.assertIsNotNone(self.check([10, 11, 12, 3, 13, 10, 20, 12, 2, 13]))

    def test_truncation_cannot_reuse_an_earlier_assistant_answer(self):
        ids = [10, 11, 12, 3, 13, 10, 20, 12, 2, 13, 10, 11, 12, 4, 13]
        self.assertIsNotNone(self.check(ids, limit=8))

    def test_partial_answer_is_still_supervised(self):
        self.assertIsNone(self.check([10, 11, 12, 3, 4, 5, 13], limit=5))

    def test_empty_answer_fails(self):
        self.assertIsNotNone(self.check([10, 11, 12, 13]))


if __name__ == "__main__":
    unittest.main()
