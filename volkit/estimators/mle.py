from .._kernels import get_routine
from .base import Estimator

class MLE(Estimator):
    """
    Dispatcher: validates inputs, fetches and calls the Routine, stores last_result.
    """

    def fit(self, spec, data, **kwargs):
        spec = self._validate_spec(spec)
        data = self._validate_data(data)
        self._warn_small_sample(spec, data)

        routine = get_routine(str(spec))
        result  = routine.fit(data, **kwargs)
        self._last_result = result
        return result