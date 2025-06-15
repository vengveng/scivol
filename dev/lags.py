import numpy as np
from scipy.stats import chi2
import statsmodels.api as sm
import time
# -------------------------------------------------------------------
# Demo comparing to Statsmodels
# -------------------------------------------------------------------
if __name__ == "__main__":
    
    # Example data
    ut_data = np.array([2.1, 2.5, 1.9, 2.2, 2.0, 2.3, 2.4, 1.8, 2.2, 2.0, 2.1, 2.3])
    K = 2
    # Compare to statsmodels
    time_start = time.time()
    ut_mean = ut_data - ut_data.mean()
    lags_sm = sm.tsa.lagmat(ut_mean, maxlag=K, trim="both")
    lags_sm = sm.add_constant(lags_sm)
    dep_var = ut_mean[K:]

    mod = sm.OLS(dep_var, lags_sm)
    res = mod.fit()

    lm_sm = len(dep_var) * res.rsquared
    pval_sm = 1 - chi2.cdf(lm_sm, df=K)

    print("\n=== Statsmodels + SciPy ===")
    print(f"R^2       = {res.rsquared:.6f}")
    print(f"LM stat   = {lm_sm:.6f}")
    print(f"p-value   = {pval_sm:.6f}")
    print(f"Elapsed   = {time.time() - time_start:.6f} sec")

    from statsmodels.stats.diagnostic import acorr_lm

    # Time acorr_lm approach
    time_start = time.time()
    lm_lm, pval_lm, _, _ = acorr_lm(ut_data, nlags=K)

    print("\n=== Statsmodels acorr_lm() ===")
    print(f"LM stat   = {lm_lm:.6f}")
    print(f"p-value   = {pval_lm:.6f}")
    print(f"Elapsed   = {time.time() - time_start:.6f} sec")