"""Parameter-independent history inputs for the existing reference recursions."""


def supplied_innovations(case):
    return len(case.get("innovation_history", ())) != 0


def presample_log_variance(case, log_h0, convert=float, *, repeated_seed=False):
    """Fixed conditioning in the case's units; never infer a variance from data."""
    if repeated_seed or supplied_innovations(case):
        return log_h0
    return convert(case.get("implicit_log_variance", 0.0))


def initial_innovations(case, count, convert=float):
    values = case.get("innovation_history", ())
    if len(values) and len(values) < count:
        raise ValueError("The declared innovation history is shorter than its consumer")
    return tuple(convert(value) for value in values[:count]) if len(values) else tuple(convert(0) for _ in range(count))


def lagged(values, initial, begin, time, lag, seed=0):
    row = time - lag - 1
    if row < begin:
        return initial[begin - row - 1] if len(initial) else seed
    return values[row]
