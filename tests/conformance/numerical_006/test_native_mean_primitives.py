from pathlib import Path
import re

def test_gh_preparation_shares_equations_and_reuses_scalar_bessel_values():
    root = Path(__file__).resolve().parents[3]
    source = (root / "core/src/math/gh_math.c").read_text()
    preparation = source.split("static int gh_shape_cache_init_order(", 1)[1].split(
        "int gh_shape_cache_init(", 1
    )[0]
    assert '#include "gh_equations.hpp"' in preparation
    assert "kappa" not in preparation
    for name in ("gh_first_order.hpp", "aparch_moment_precision.cpp"):
        consumer = (root / "core/src" / ({"gh_first_order.hpp": "math/gh_first_order.hpp", "aparch_moment_precision.cpp": "models/power/aparch_moment_precision.cpp"}[name])).read_text()
        recipe = consumer.split("#define SCIVOL_GH_PREPARE_EQUATIONS", 1)[1].split(
            "#undef SCIVOL_GH_PREPARE_EQUATIONS", 1
        )[0]
        assert re.search(r'#include "(?:../../math/)?gh_equations[.]hpp"', recipe)
    scalar = source.split("static int scalar_standardized_parameters(", 1)[1].split(
        "static int gh_shape_cache_init_order(", 1
    )[0]
    for intention in ("STANDARDIZE", "NORMALIZER"):
        selector = f"SCIVOL_GH_SCALAR_{intention}_EQUATIONS"
        recipe = scalar.split(f"#define {selector}", 1)[1].split(
            f"#undef {selector}", 1
        )[0]
        assert recipe.count('#include "gh_equations.hpp"') == 1
    assert scalar.count("if (!scalar_standardized_parameters(") == 1
    assert "log_bessel_k_scalar(" not in scalar
    equations = (root / "core/src/math/gh_equations.hpp").read_text()
    preparation_equations = equations.split(
        "#elif defined(SCIVOL_GH_OBSERVATION_EQUATIONS)", 1
    )[0]
    assert "#define SCIVOL_GH_LOG_BESSEL(a,b) log_bessel_k_scalar(a,b)" in preparation_equations
    bessel_k1 = (
        "SCIVOL_GH_LOG_BESSEL(\n"
        "    SCIVOL_GH_ADD(SCIVOL_GH_FIELD(lambda), SCIVOL_GH_ONE), SCIVOL_GH_FIELD(zeta))"
    )
    assert preparation_equations.count(bessel_k1) == 1
    assert f"const N log_k1 = {bessel_k1};" in preparation_equations
    assert "SCIVOL_GH_SUB(log_k1, log_k0)" in preparation_equations
    assert "SCIVOL_GH_SUB(SCIVOL_GH_LOG_K2, log_k1)" in preparation_equations

def test_gh_observation_loops_do_not_rebuild_point_normalizers():
    root = Path(__file__).resolve().parents[3]
    first = (root / "core/src/math/gh_first_order.hpp").read_text()
    observations = first.split("variance_gamma_logpdf(", 1)[1]
    assert "ad::lgamma(" not in observations
    assert "ad::log(cache.alpha)" not in observations
    assert "ad::log(cache.vg_argument_scale)" not in observations
    equations = (root / "core/src/math/gh_equations.hpp").read_text()
    observations = equations.split("#elif defined(SCIVOL_GH_OBSERVATION_EQUATIONS)", 1)[1]
    assert "ad::lgamma(" not in observations
    assert "ad::log(cache.alpha)" not in observations
    assert "log_bessel_k<" not in observations
    equations = (root / "core/src/math/vg_equations.hpp").read_text()
    observations = equations.split("#elif defined(SCIVOL_VG_OBSERVATION_EQUATIONS)", 1)[1]
    assert "ad::lgamma(" not in observations
    assert "ad::log(cache.vg_argument_scale)" not in observations
    scalar = (root / "core/src/math/gh_math.c").read_text()
    observations = scalar.split("static gh_jet_t vg_logpdf_jet(", 1)[1].split(
        "int gh_abs_moment_derivatives(", 1
    )[0]
    assert "lgamma(" not in observations
    assert "jet_log(cache->alpha)" not in observations
    assert "log(cache->alpha" not in observations
    assert "log(cache->argument_scale)" not in observations
    assert "vg_standardized_logpdf_scalar(" not in observations
