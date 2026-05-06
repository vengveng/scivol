from __future__ import annotations

import os
import sys

from setuptools import Extension, setup


def _compile_args() -> list[str]:
    if sys.platform.startswith("win"):
        return ["/O2", "/fp:fast"]

    args = ["-O3", "-ffast-math", "-ffp-contract=fast", "-funroll-loops"]
    if os.environ.get("SCIVOL_PORTABLE_BUILD") != "1":
        args.append("-march=native")
    return args


ext_modules = [
    Extension(
        "scivol._core",
        sources=[
            "scivol/_core.c",
            "scivol/_csrc/variance_garch.c",
            "scivol/_csrc/likelihood_garch.c",
            "scivol/_csrc/likelihood_normal.c",
            "scivol/_csrc/likelihood_studentt.c",
            "scivol/_csrc/likelihood_skewt.c",
            "scivol/_csrc/errors_garch.c",
            "scivol/_csrc/transforms_logspace.c",
            "scivol/_csrc/arma_garch.c",
            "scivol/_csrc/arma.c",
            "scivol/_csrc/variance_gjr_garch.c",
            "scivol/_csrc/likelihood_gjr_garch.c",
            "scivol/_csrc/errors_gjr_garch.c",
            "scivol/_csrc/log_wrappers.c",
            "scivol/_csrc/dcc_gaussian.c",
        ],
        include_dirs=["scivol/_csrc"],
        extra_compile_args=_compile_args(),
    )
]


setup(ext_modules=ext_modules)
