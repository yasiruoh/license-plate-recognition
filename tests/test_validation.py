import unittest
from src.validation import check_bytes, threshold_value, count_labels, validate_rows

class ValidationTests(unittest.TestCase):
    def test_empty_input(self):
        with self.assertRaises(ValueError): check_bytes(b"")
    def test_limit(self):
        with self.assertRaises(ValueError): check_bytes(b"abc", 2)
    def test_valid_input(self): self.assertEqual(check_bytes(b"abc"), b"abc")
    def test_threshold(self): self.assertEqual(threshold_value(0.5), 0.5)
    def test_bad_threshold(self):
        for v in [-1, 2, float("nan")]:
            with self.assertRaises(ValueError): threshold_value(v)
    def test_counts(self):
        self.assertEqual(count_labels([{"label":"cow"},{"label":"cow"}]), {"cow":2})
    def test_schema_order(self):
        self.assertEqual(validate_rows([{"b":2,"a":1}], ["a","b"]), [[1.,2.]])
    def test_missing_feature(self):
        with self.assertRaises(ValueError): validate_rows([{"a":1}], ["a","b"])
    def test_nonfinite(self):
        with self.assertRaises(ValueError): validate_rows([{"a":"nan"}], ["a"])

if __name__ == "__main__": unittest.main()
