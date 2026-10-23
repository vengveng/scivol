"""Source-only runner; reuse the qualified complete-scalar symbolic/MP oracle.

After an authorized production-flag build, pass the selected-chain executable
and the live tests/native/unit_map_range_mp.py path. Never imports JAX/scivol.
"""

import importlib.util
import json
import subprocess
import sys


def main():
    assert len(sys.argv) == 3, "Pass prebuilt chain executable and qualified MP helper."
    spec = importlib.util.spec_from_file_location("qualified_unit_range_mp", sys.argv[2])
    assert spec is not None and spec.loader is not None
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    completed = subprocess.run(
        [sys.argv[1], "--emit"], check=True, text=True, capture_output=True, timeout=120
    )
    records = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    expected = {
        "egarch_symmetric_cancellation",
        "aparch_original_omega",
        "aparch_zero_omega_mixed_curvature",
        "aparch_rounded_internal_omega",
    }
    assert len(records) == len(expected)
    assert {record["name"] for record in records} == expected
    for record in records:
        assert record["inner_h"][1] == record["inner_h"][2], "A scalar Hessian must be symmetric."
        assert oracle.verify(record), "No algebra-only skips are allowed in this matrix."
    print("selected chain symbolic/MP: 4 complete scalar compositions at 200 digits")


if __name__ == "__main__":
    main()
