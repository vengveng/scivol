import statsmodels.api as sm
from scipy.stats import chisquare, chi2, norm, probplot, t
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_lm

class Test:
    def __init__(self):
        self.results = None
        self.dist = None

    @staticmethod
    def pit(result=None):
        assert result.__class__.__name__ == 'ModelResult'

        distr = result.distr
        std_resid = result.std_resid

        if distr == 'normal':
            ut = norm.cdf(std_resid)
        elif distr == 't':
            ut = t.cdf(std_resid, result.secondary['nu'])
        else:
            raise ValueError(f"Invalid distribution: {distr} not implemented.")
        
        test_result = Test.pit_adequacy_test(ut, internal=True)
        

        
    @staticmethod
    def pit_adequacy_test(ut, K=4, N=10, internal=False):
        #TODO: check optimal N
        """
        Perform PIT adequacy test using LM test for serial correlation and Pearson's chi-squared test for uniformity.
        
        Parameters:
        - ut: np.ndarray of PIT-transformed values.
        - K: int, number of lags for LM test.
        - N: int, number of bins for Pearson's chi-squared test.
        """

        lm_stat, lm_pval, _, _ = acorr_lm(ut, nlags=K)

        counts, _ = np.histogram(ut, bins=np.linspace(0, 1, N + 1))
        expected_counts = len(ut) / N
        chi2_stat, chi2_pval = chisquare(counts, f_exp=[expected_counts] * N)

        if not internal:
            result = {
            "LM Statistic": lm_stat,
            "LM P-value": lm_pval,
            "Chi2 Statistic": chi2_stat,
            "Chi2 P-value": chi2_pval}
        
        else:
            result = {
            'lms': lm_stat, 
            'lmp': lm_pval, 
            'c2s': chi2_stat, 
            'c2p': chi2_pval}

        return result