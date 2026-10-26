"""Model search on the native restriction graph and one comparison sample."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import product
import os
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _scivolapi as api
from ._fit import (_COORDINATES, PreparedSample, native_spec, present_fit,
                   spec_from_native)
from ._validation import validate_data, validate_hold_back, validate_regressors
from .components import ARX, APARCH, EGARCH, GJRGARCH, Normal, SkewT
from .result import EstimationResult
from .roles import Role
from .spec import CompositeSpec


def _level(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between zero and one")


@dataclass(frozen=True)
class InformationCriterion:
    """Rank a declared search by AIC, BIC or HQIC; declare adequacy explicitly."""
    name: str = "bic"
    adequacy: bool | None = None
    max_fits: int = 96
    diagnostic_level: float = 0.01

    def __post_init__(self):
        if self.name not in ("aic", "bic", "hqic"):
            raise ValueError("information criterion must be 'aic', 'bic' or 'hqic'")
        if self.adequacy is not None and type(self.adequacy) is not bool:
            raise TypeError("adequacy must be True or False")
        _validate_budget(self.max_fits)
        _level(self.diagnostic_level, "diagnostic_level")


@dataclass(frozen=True)
class GeneralToSpecific:
    reduction_level: float = 0.01
    encompassing_level: float = 0.01
    diagnostic_level: float = 0.01
    prefer: str = "fewest_parameters"
    tie_break: InformationCriterion = field(default_factory=InformationCriterion)
    max_fits: int = 96

    def __post_init__(self):
        for name in ("reduction_level", "encompassing_level", "diagnostic_level"):
            _level(getattr(self, name), name)
        if self.prefer not in ("fewest_parameters", "information_criterion"):
            raise ValueError("prefer must be 'fewest_parameters' or 'information_criterion'")
        if not isinstance(self.tie_break, InformationCriterion):
            raise TypeError("tie_break must be an InformationCriterion")
        _validate_budget(self.max_fits)


@dataclass(frozen=True)
class ConfidenceSet:
    reduction: GeneralToSpecific | InformationCriterion = field(default_factory=GeneralToSpecific)
    holdout: float = 0.2
    confidence: float = 0.90
    loss: str = "log_score"
    refit_every: int = 20
    bootstrap_draws: int = 4999
    seed: int | None = None

    def __post_init__(self):
        _level(self.holdout, "holdout")
        _level(self.confidence, "confidence")
        if self.loss not in ("log_score", "qlike"):
            raise ValueError("loss must be 'log_score' or 'qlike'")
        for name in ("refit_every", "bootstrap_draws"):
            _validate_budget(getattr(self, name), name)
        if not isinstance(self.reduction, (GeneralToSpecific, InformationCriterion)):
            raise TypeError("reduction must be GeneralToSpecific or InformationCriterion")


def _validate_budget(value, name="max_fits"):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass
class Candidate:
    key: tuple
    model: api.Model
    specification: CompositeSpec
    fit: api.FitResult | None = None
    diagnostics: dict = field(default_factory=dict)
    failure_reason: str | None = None
    encompassing: dict = field(default_factory=dict)
    elapsed: float = 0.0
    warm_started: bool = False
    free: bool = False

    @property
    def adequate(self):
        return self.certified and self.diagnostics.get("status") == "adequate"

    @property
    def certified(self):
        return self.fit is not None and self.fit.success

    @property
    def information_criteria(self):
        if self.fit is None or not self.fit.estimate_available:
            return dict.fromkeys(("aic", "bic", "hqic"), np.inf)
        return dict(zip(("aic", "bic", "hqic"), api.information_criteria(
            self.fit.log_likelihood, self.model.dimension, self.fit.scored)))


@dataclass
class SelectionResult:
    selected: EstimationResult | None = None
    selected_model: CompositeSpec | None = None
    terminals: tuple[Candidate, ...] = ()
    paths: tuple[tuple[tuple, ...], ...] = ()
    tests: list[dict] = field(default_factory=list)
    candidates: tuple[Candidate, ...] = ()
    confidence_set: tuple[Candidate, ...] = ()
    status: str = "no_adequate_model"
    search_complete: bool = True
    frontier: tuple[tuple, ...] = ()
    fit_counts: dict = field(default_factory=dict)
    predictive: dict = field(default_factory=dict)

    def table(self):
        """Candidate rows as a pandas DataFrame, or records without pandas."""
        terminals = {c.key for c in self.terminals}
        confidence = {c.key for c in self.confidence_set}
        rows = [dict(model=str(c.specification), key=c.key,
                     parameters=c.model.dimension,
                     log_likelihood=None if c.fit is None else c.fit.log_likelihood,
                     **c.information_criteria,
                     adequacy=c.diagnostics.get("status", "not_evaluated"),
                     encompassing=dict(c.encompassing), certified=c.certified,
                     terminal=c.key in terminals, confidence_set=c.key in confidence,
                     failure_reason=c.failure_reason) for c in self.candidates]
        try:
            import pandas as pd
        except ImportError:
            return rows
        return pd.DataFrame(rows)

    def summary(self) -> str:
        choice = str(self.selected_model) if self.selected_model is not None else "none"
        return (f"Selection: {self.status}; selected: {choice}\n"
                f"Search complete: {self.search_complete}; terminals: {len(self.terminals)}; "
                f"optimizer invocations: {self.fit_counts.get('optimizer_invocations', 0)}")


class Search:
    """Declare maximal component generators and their native downward closure."""
    def __init__(self, *, mean=None, volatility=None, density=None,
                 constant_variance=True, closure=True):
        self.mean = tuple([None] if mean is None else mean)
        self.volatility = tuple([] if volatility is None else volatility)
        self.density = tuple([Normal()] if density is None else density)
        if type(closure) is not bool or type(constant_variance) is not bool:
            raise TypeError("closure and constant_variance must be bool")
        self.constant_variance, self.closure = constant_variance, closure
        for parts, role in ((self.mean, Role.MEAN), (self.volatility, Role.VOLATILITY),
                            (self.density, Role.DENSITY)):
            for part in parts:
                if part is None and role == Role.MEAN:
                    continue
                if getattr(part, "role", None) != role:
                    raise TypeError(f"search components must have role {role.name}")
        if not self.mean or not self.density or (not self.volatility and not constant_variance):
            raise ValueError("search must declare at least one candidate")

    def fit(self, y, x=None, *, procedure=None, method="mle", n_jobs=None,
            hold_back=0, coordinates="automatic", maxiter=1000, accuracy=1e-10):
        procedure = GeneralToSpecific() if procedure is None else procedure
        if not isinstance(procedure, (GeneralToSpecific, InformationCriterion, ConfidenceSet)):
            raise TypeError("procedure must be GeneralToSpecific, InformationCriterion or ConfidenceSet")
        if method not in ("mle", "qmle"):
            raise ValueError("method must be 'mle' or 'qmle'")
        reduction = procedure.reduction if isinstance(procedure, ConfidenceSet) else procedure
        if isinstance(reduction, GeneralToSpecific) and method == "qmle":
            raise ValueError("GeneralToSpecific requires MLE; QMLE LR calibration is unavailable")
        if isinstance(reduction, InformationCriterion) and reduction.adequacy is None:
            raise ValueError("standalone InformationCriterion requires explicit adequacy=True or adequacy=False")
        if coordinates not in _COORDINATES:
            raise ValueError("coordinates must be 'automatic', 'z' or 'theta'")
        workers = min(4, os.cpu_count() or 1) if n_jobs in (None, -1) else n_jobs
        _validate_budget(workers, "n_jobs")
        y = np.ascontiguousarray(validate_data(y))
        x = None if x is None else validate_regressors(x, len(y))
        options = dict(method=method, coordinates=coordinates, maxiter=maxiter, accuracy=accuracy)
        if isinstance(procedure, ConfidenceSet):
            return self._predict(y, x, procedure, workers, hold_back, options)
        engine = _Reduction(self, y, x, reduction, workers, hold_back, options)
        result = engine.run()
        engine.deploy(result)
        return result

    def _definitions(self, columns):
        volatilities = self.volatility + ((None,) if self.constant_variance else ())
        definitions = []
        for mean, vol, density in product(self.mean, volatilities, self.density):
            parts = [part for part in (mean, vol, density) if part is not None]
            definition = native_spec(CompositeSpec(*parts), columns)
            if definition is None:
                raise ValueError("unsupported search component combination")
            definitions.append(definition)
        return definitions

    def _predict(self, y, x, procedure, workers, hold_back, options):
        first = int(len(y)*(1-procedure.holdout))
        if len(y)-first < 500:
            raise ValueError("ConfidenceSet requires at least 500 holdout observations")
        engine = _Reduction(self, y[:first], None if x is None else x[:first],
                            procedure.reduction, workers, hold_back, options)
        result = engine.run()
        frozen = result.terminals
        if not frozen:
            return result
        def predict(candidate):
            counts = dict(optimizer_invocations=0, warm_started_fits=0, predictive_refits=0)
            try:
                losses = self._predict_candidate(candidate, y, x, first, procedure, options, counts, engine.plan)
                return losses, counts, None
            except (ValueError, RuntimeError) as exc:
                return None, counts, str(exc)

        columns, failed = [], []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(predict, frozen))
        for candidate, (losses, counts, error) in zip(frozen, outputs):
            for name, count in counts.items():
                result.fit_counts[name] = result.fit_counts.get(name, 0) + count
            if error:
                failed.append((candidate.key, error))
            else:
                columns.append(losses)
        if failed:
            result.status = "predictive_fit_failed"
            result.predictive = dict(failures=failed)
            return result
        losses = np.ascontiguousarray(np.column_stack(columns))
        length = api.mcs_block_length(len(losses))
        uniforms = np.random.Generator(np.random.PCG64(procedure.seed)).random(
            (procedure.bootstrap_draws, (len(losses)+length-1)//length))
        mcs = api.model_confidence_set(losses, uniforms, length, 1-procedure.confidence, "R")
        result.confidence_set = tuple(frozen[i] for i in mcs["included"])
        result.predictive = dict(losses=losses, holdout_start=first, block_length=length,
                                 bootstrap="circular", statistic="R", seed=procedure.seed,
                                 confidence=procedure.confidence, **mcs)
        result.status = "predictive_confidence_set"
        if result.confidence_set:
            chosen = min(result.confidence_set, key=lambda c: (c.model.dimension, c.information_criteria["bic"], c.key))
            result.selected_model = chosen.specification
            result.selected = chosen.specification.fit(y, x=x, initial_params=np.array(chosen.fit.theta), **options)
            result.fit_counts["full_sample_refits"] += 1
            result.fit_counts["optimizer_invocations"] += 1
            result.fit_counts["warm_started_fits"] += 1
            if not result.selected.converged:
                result.status = "full_sample_refit_uncertified"
        return result

    @staticmethod
    def _predict_candidate(candidate, y, x, first, procedure, options, counts, training_plan):
        losses = []
        theta = np.array(candidate.fit.theta)
        # Freeze the training initialization. Each block filters observed returns
        # through its own origin; future observations enter no earlier forecast.
        for start in range(first, len(y), procedure.refit_every):
            stop = min(start+procedure.refit_every, len(y))
            if start != first:
                train = api.ComparisonPlan(y[:start], training_plan.first_scored,
                                          None if x is None else x[:start],
                                          variance_seed=training_plan.variance_seed)
                counts["predictive_refits"] = counts.get("predictive_refits", 0) + 1
                counts["optimizer_invocations"] += 1
                counts["warm_started_fits"] += 1
                fitted = api.fit(candidate.model, train, _options(options, theta))
                if not fitted.success:
                    raise RuntimeError("uncertified predictive refit")
                theta = np.array(fitted.theta)
            plan = api.ComparisonPlan(y[:stop], training_plan.first_scored,
                                      None if x is None else x[:stop],
                                      variance_seed=training_plan.variance_seed)
            values = api.predictive_losses(api.Jet(candidate.model, plan), theta, procedure.loss)
            losses.extend(values[start-training_plan.first_scored:stop-training_plan.first_scored])
        return np.array(losses)


class Auto:
    """AR(1)+intercept x {GJR, EGARCH, APARCH}(1,1) x SkewT, plus constant variance."""
    def fit(self, y, x=None, **options):
        return Search(mean=[ARX.up_to(1)], volatility=[GJRGARCH.up_to(1, 1),
                      EGARCH.up_to(1, 1), APARCH.up_to(1, 1)], density=[SkewT()]).fit(
                          y, x, procedure=GeneralToSpecific(), **options)


def _options(values, start=None):
    options = api.FitOptions()
    options.estimator = api.Estimator.mle if values["method"] == "mle" else api.Estimator.qmle
    options.coordinates = _COORDINATES[values["coordinates"]]
    options.maximum_iterations = values["maxiter"]
    options.accuracy = values["accuracy"]
    if start is not None:
        options.start = np.asarray(start).tolist()
    return options


class _Reduction:
    def __init__(self, search, y: NDArray[np.float64], x: NDArray[np.float64] | None,
                 procedure, workers: int, hold_back: int, options: dict):
        self.search, self.y, self.x = search, y, x
        self.procedure, self.workers, self.options = procedure, workers, options
        self.nodes, self.edges = {}, {}
        declared = []
        for definition in search._definitions(0 if x is None else x.shape[1]):
            model = api.Model(definition)
            declared.append(model.key)
            self._compile(model)
        self.allowed = set(self.nodes) if search.closure else set(declared)
        descendants = {child for key in declared for child in self._reachable(key)}
        self.roots = tuple(sorted(set(declared)-descendants))
        first = max(validate_hold_back(hold_back), *(self.nodes[k].model.spec.mean.conditioning for k in declared))
        self.plan = api.ComparisonPlan(y, first, x)
        self.first = first
        self.counts = dict(optimizer_invocations=0, unique_fits=0, free_children=0,
                           warm_started_fits=0, parent_refits=0, full_sample_refits=0)
        self.tests, self.frontier = [], set()
        self.deployment_reserve = 0
        self.visited, self.terminals, self.accepted = set(), set(), {}
        self.started = time.perf_counter()

    def _compile(self, model):
        key = model.key
        if key in self.nodes:
            return
        self.nodes[key] = Candidate(key, model, spec_from_native(model.spec))
        self.edges[key] = []
        for restriction in model.restrictions():
            child = api.Model(restriction.child)
            self.edges[key].append((restriction, child.key))
            self._compile(child)
        self.edges[key].sort(key=lambda item: (item[1], item[0].name))

    def _reachable(self, key):
        seen, queue = set(), list(self.edges[key])
        while queue:
            r, child = queue.pop()
            if child in seen:
                continue
            seen.add(child)
            yield child
            queue.extend(self.edges[child])

    def _permitted_edges(self, key, prefix=None):
        for edge, child in self.edges[key]:
            restriction = edge if prefix is None else prefix.then(edge)
            if child in self.allowed:
                yield restriction, child
            else:
                yield from self._permitted_edges(child, restriction)

    def run(self):
        needs_adequacy = isinstance(self.procedure, GeneralToSpecific) or self.procedure.adequacy
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            self.pool = pool
            if isinstance(self.procedure, InformationCriterion):
                self._fit_nodes([(k, None, None) for k in sorted(self.allowed)], needs_adequacy)
                eligible = [k for k in self.allowed if self.nodes[k].certified
                            and (not needs_adequacy or self.nodes[k].adequate)]
                self.terminals.update(eligible)
            else:
                self._fit_nodes([(k, None, None) for k in self.roots], True)
                while True:
                    retries = self.counts["parent_refits"]
                    self.visited.clear()
                    self.terminals.clear()
                    self.accepted.clear()
                    for root in self.roots:
                        if self.nodes[root].adequate:
                            self._reduce(root, root, None)
                    if self.counts["parent_refits"] == retries:
                        break
                    for record in self.tests:
                        if record["decision"] == "accepted":
                            record["decision"] = "superseded_parent_refit"
                            record["failure_reason"] = "Rechecking the graph after a parent refit"

        available = [self.nodes[k] for k in self.allowed if self.nodes[k].fit is not None
                     or self.nodes[k].failure_reason]
        terminals = tuple(sorted((self.nodes[k] for k in self.terminals if self.nodes[k].certified
                                  and (not needs_adequacy or self.nodes[k].adequate)), key=self._rank))
        reached = set(self.roots)
        for children in self.accepted.values():
            reached.update(children)
        if isinstance(self.procedure, InformationCriterion):
            reached = self.allowed
        adequate = [c for c in available if c.key in reached and c.certified
                    and (not needs_adequacy or c.adequate)]
        choices = terminals or tuple(sorted(adequate, key=self._rank))
        result = SelectionResult(terminals=terminals, tests=self.tests,
                                 candidates=tuple(sorted(available, key=lambda c: c.key)),
                                 paths=self._paths(), search_complete=not self.frontier,
                                 frontier=tuple(sorted(self.frontier)), fit_counts=self.counts)
        if choices:
            result.selected_model = choices[0].specification
            result.status = "selected" if result.search_complete else "budget_exhausted"
        elif self.frontier:
            result.status = "budget_exhausted"
        elif any(c.diagnostics.get("status") == "adequacy_unresolved" for c in available):
            result.status = "adequacy_unresolved"
        return result

    def _rank(self, candidate):
        ic = self.procedure if isinstance(self.procedure, InformationCriterion) else self.procedure.tie_break
        criterion = candidate.information_criteria[ic.name]
        if isinstance(self.procedure, GeneralToSpecific) and self.procedure.prefer == "fewest_parameters":
            return candidate.model.dimension, criterion, candidate.key
        return criterion, candidate.key

    def _fit_nodes(self, requests, diagnostics):
        jobs, seen = [], set()
        for key, parent, restriction in sorted(requests, key=lambda r: r[0]):
            node = self.nodes[key]
            if node.fit is not None or node.failure_reason or key in seen:
                continue
            seen.add(key)
            start = None
            if parent is not None:
                source = self.nodes[parent]
                reuse = api.restrict_fit(source.model, source.fit, restriction, self.plan)
                if reuse.status == api.RestrictFitStatus.reused:
                    reserve = max(self.deployment_reserve, int(node.model.spec.mean.conditioning != self.first))
                    if self.counts["optimizer_invocations"] + reserve > self.procedure.max_fits:
                        self.frontier.add(key)
                        continue
                    self.deployment_reserve = reserve
                    self.frontier.discard(key)
                    node.fit, node.free = reuse.result, True
                    self.counts["free_children"] += 1
                    self._diagnose(node, diagnostics)
                    continue
                start = restriction.map_start(np.array(source.fit.theta))
            reserve = max(self.deployment_reserve, int(node.model.spec.mean.conditioning != self.first))
            if self.counts["optimizer_invocations"] + 1 + reserve > self.procedure.max_fits:
                self.frontier.add(key)
                continue
            self.deployment_reserve = reserve
            self.frontier.discard(key)
            self.counts["optimizer_invocations"] += 1
            self.counts["unique_fits"] += 1
            self.counts["warm_started_fits"] += start is not None
            node.warm_started = start is not None
            jobs.append((node, _options(self.options, start)))
        for node, output, elapsed, failure in self.pool.map(self._fit_one, jobs):
            node.fit, node.elapsed, node.failure_reason = output, elapsed, failure
            self._diagnose(node, diagnostics)

    def _fit_one(self, job):
        node, options = job
        started = time.perf_counter()
        try:
            return node, api.fit(node.model, self.plan, options), time.perf_counter()-started, None
        except (ValueError, RuntimeError) as exc:
            return node, None, time.perf_counter()-started, str(exc)

    def _diagnose(self, node, required):
        if not node.certified:
            node.diagnostics = dict(status="uncertified")
            return
        if required:
            try:
                node.diagnostics = api.selection_diagnostics(api.Jet(node.model, self.plan),
                    np.array(node.fit.theta), self.procedure.diagnostic_level, node.fit.estimator)
            except (ValueError, RuntimeError) as exc:
                node.diagnostics = dict(status="adequacy_unresolved", reason=str(exc))
        else:
            node.diagnostics = dict(status="disabled")
        node.encompassing = node.diagnostics.pop("encompassing", {})

    def _reduce(self, root, key, joint):
        state = root, key
        if state in self.visited:
            return
        self.visited.add(state)
        edges = list(self._permitted_edges(key))
        requests = [(child, key, r) for r, child in edges
                    if r.identification != api.Identification.unidentified_under_null]
        self._fit_nodes(requests, True)
        accepted = []
        pending = False
        for restriction, child in edges:
            if restriction.identification == api.Identification.unidentified_under_null:
                self.tests.append(self._record(key, child, restriction, "unsupported", restriction.condition))
                continue
            if child in self.frontier:
                pending = True
                self.tests.append(self._record(key, child, restriction, "budget_exhausted", "fit budget exhausted"))
                continue
            record = self._test(key, child, restriction, self.procedure.reduction_level)
            self.tests.append(record)
            if record["decision"] != "not_rejected":
                continue
            node = self.nodes[child]
            if not node.adequate:
                record["decision"] = "child_inadequate"
                record["failure_reason"] = node.diagnostics.get("status")
                continue
            combined = restriction if joint is None else joint.then(restriction)
            backtest = self._test(root, child, combined, self.procedure.encompassing_level)
            backtest["purpose"] = "gum_backtest"
            self.tests.append(backtest)
            node.encompassing[root] = backtest["decision"]
            if backtest["decision"] == "not_rejected":
                record["decision"] = "accepted"
                accepted.append((record["p_value"], child, combined))
            else:
                record["decision"] = "encompassing_failed"
                record["failure_reason"] = backtest["failure_reason"] or backtest["decision"]
        self.accepted[state] = tuple(child for _, child, _ in sorted(accepted, key=lambda a: (-a[0], a[1])))
        if not accepted and not pending:
            self.terminals.add(key)
        for _, child, combined in sorted(accepted, key=lambda a: (-a[0], a[1])):
            self._reduce(root, child, combined)

    def _record(self, parent, child, restriction, decision, reason=None):
        p, c = self.nodes[parent], self.nodes[child]
        return dict(parent=parent, child=child, restriction=restriction.name,
                    parent_log_likelihood=None if p.fit is None else p.fit.log_likelihood,
                    child_log_likelihood=None if c.fit is None else c.fit.log_likelihood,
                    statistic=None, law=None, weights=(), p_value=None,
                    level=self.procedure.reduction_level, decision=decision,
                    failure_reason=reason, diagnostics=c.diagnostics,
                    assumptions=restriction.condition, purpose="reduction")

    def _test(self, parent, child, restriction, level):
        p, c = self.nodes[parent], self.nodes[child]
        record = self._record(parent, child, restriction, "unresolved")
        record["level"] = level
        if not p.certified or not c.certified:
            record["failure_reason"] = "Both fits must be certified"
            return record
        theta = restriction.embed(np.array(c.fit.theta))
        if theta is None:
            record["failure_reason"] = "Child cannot be embedded in parent domain"
            return record
        output = api.Jet(p.model, self.plan).theta(theta, api.HESSIAN)
        if output["status"] != "ok":
            record["failure_reason"] = "Invalid parent Hessian at restricted estimate"
            return record
        test = api.restriction_test(p.model, c.model, restriction, p.fit, c.fit,
                                    output["hessian"], level)
        if test.status == api.LRStatus.negative_statistic and self.counts["optimizer_invocations"] + self.deployment_reserve < self.procedure.max_fits:
            self.counts["optimizer_invocations"] += 1
            self.counts["parent_refits"] += 1
            self.counts["warm_started_fits"] += 1
            p.fit = api.fit(p.model, self.plan, _options(self.options, theta))
            self._diagnose(p, True)
            test = api.restriction_test(p.model, c.model, restriction, p.fit, c.fit,
                                        output["hessian"], level)
        if test.status == api.LRStatus.negative_statistic and self.counts["optimizer_invocations"] + self.deployment_reserve >= self.procedure.max_fits:
            self.frontier.add(parent)
        record.update(statistic=test.statistic, law=test.law.name,
                      weights=tuple(test.weights), p_value=test.p_value if np.isfinite(test.p_value) else None,
                      interior_df=test.interior_df, numerical_error=test.numerical_error,
                      parent_log_likelihood=p.fit.log_likelihood,
                      failure_reason=None if test.status == api.LRStatus.ok else test.message,
                      decision=("not_rejected" if test.p_value > level else "rejected")
                      if test.status == api.LRStatus.ok else test.status.name)
        return record

    def _paths(self):
        paths = []
        def walk(root, key, path):
            children = self.accepted.get((root, key), ())
            if not children:
                paths.append(path)
            for child in children:
                walk(root, child, path+(child,))
        for root in self.roots:
            if self.nodes[root].fit is not None:
                walk(root, root, (root,))
        return tuple(paths)

    def deploy(self, result):
        if result.selected_model is None:
            return
        node = next(c for c in result.candidates if c.specification is result.selected_model)
        # Identical full-sample plan fits use the ordinary result presenter.
        # Otherwise deployment is an explicit normal full-sample refit.
        if node.model.spec.mean.conditioning == self.first:
            columns = node.model.spec.mean.exog
            x = None if not columns else np.ascontiguousarray(self.x[:, :columns])
            sample = PreparedSample(node.model.spec, self.y, x, 0, self.first, len(self.y), 1.0)
            result.selected = present_fit(node.specification, sample, node.fit,
                maxiter=self.options["maxiter"], warm_started=node.warm_started, elapsed=node.elapsed)
        else:
            result.selected = node.specification.fit(self.y, x=self.x,
                initial_params=np.array(node.fit.theta), **self.options)
            self.counts["full_sample_refits"] += 1
            self.counts["warm_started_fits"] += 1
            self.counts["optimizer_invocations"] += 1
        if not result.selected.converged:
            result.status = "full_sample_refit_uncertified"
