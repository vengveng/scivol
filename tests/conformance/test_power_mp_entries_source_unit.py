"""Source-only control regression; imports no MP, AD, native or live test module."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


def load_checker():
    path = Path(__file__).with_name("test_power_phase_split_math.py")
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "assert_mp_entries_close")
    namespace = {"mp": SimpleNamespace(isfinite=math.isfinite)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


class MPEntriesSourceUnit(unittest.TestCase):
    def test_later_nan_is_rejected_on_either_side(self):
        checker = load_checker()
        for position in (1, 30):
            for side in (0, 1):
                with self.subTest(position=position, side=side):
                    entries = [[0.0] * 31, [0.0] * 31]
                    entries[side][position] = math.nan
                    with self.assertRaises(AssertionError):
                        checker(*entries, 1e-20)

    def test_later_infinity_is_rejected(self):
        checker = load_checker()
        with self.assertRaises(AssertionError):
            checker((0.0, math.inf), (0.0, math.inf), 1e-35)

    def test_finite_entries_inside_bound_pass(self):
        load_checker()((0.0, 1e-21), (0.0, 0.0), 1e-20)

    def test_bound_remains_strict(self):
        with self.assertRaises(AssertionError):
            load_checker()((0.0, 1e-20), (0.0, 0.0), 1e-20)

    def test_missing_entry_is_rejected(self):
        with self.assertRaises(ValueError):
            load_checker()((0.0, 0.0), (0.0,), 1e-20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
