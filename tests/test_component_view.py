import numpy as np
import pytest
from volkit.components import GARCH, Normal, CompositeSpec, Role

# ------------------------------------------------------------------
# 1. Lone component auto-adds Normal in __str__
# ------------------------------------------------------------------
def test_single_component_string_and_spec():
    g = GARCH(2, 1)
    assert str(g) == "GARCH(2,1)+Normal"        # implicit density
    spec = g.spec
    assert isinstance(spec, CompositeSpec)
    assert spec.get_component(Role.DENSITY).__class__ is Normal

# ------------------------------------------------------------------
# 2. Equality / hash contract with CompositeSpec
# ------------------------------------------------------------------
def test_component_equals_compositespec():
    g   = GARCH(1, 1)
    cmp = CompositeSpec(GARCH(1, 1))           # explicit wrap
    assert g == cmp
    assert hash(g) == hash(cmp)

# ------------------------------------------------------------------
# 3. Duplicate-density guard still works
# ------------------------------------------------------------------
def test_density_replacement_on_lone_component():
    g = GARCH(1, 1)
    spec = g + Normal()      # explicit Normal duplicates implicit one
    # spec should still have only ONE density (because placeholder replaced)
    density_count = sum(1 for c in spec.components if c.role is Role.DENSITY)
    assert density_count == 1

# # ------------------------------------------------------------------
# # 4. Fitted-method calls still succeed
# # ------------------------------------------------------------------
# def test_fit_through_component(tmp_path, monkeypatch):
#     """
#     Mock a very small 'fit' path so we don't need the optimiser here.
#     We only check that the convenience wrapper calls CompositeSpec.fit().
#     """
#     called = {}

#     def fake_fit(self, data, estimator=None, **kw):
#         called["ok"] = True
#         return "dummy-result"

#     monkeypatch.setattr("volkit.components.CompositeSpec.fit", fake_fit, raising=True)
#     g = GARCH(1, 1)
#     res = g.fit(np.ones(50))          # uses Component.fit -> CompositeSpec.fit
#     assert res == "dummy-result"
#     assert called