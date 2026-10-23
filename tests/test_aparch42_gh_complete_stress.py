"""Independent MP AD of complete GH/NIG APARCH stress jets.

The Bessel integral follows the refined GH moment oracle, using DLMF 10.32.9
with y=sqrt(x0)*t. The tracked MP Jet supplies arithmetic, not native coefficients.
No JAX fixed-node Bessel rule, finite differences, floors, or localdev imports.
"""

from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp
import numpy as np
import pytest

from test_native_anchored_mean_chart import Jet
from test_native_aparch_subset_curvature import exp_jet, log_jet, power_jet, jet_outputs


STRESS = (.15, 1.7, -.994, 24.999999, -.5)
DATA = np.array([.2, -.4, .5, -1., .7, .3, -.6])
H0 = .8
NLL_ATOL = 2e-11
G_RTOL, G_ATOL = 4e-7, 4e-9
H_RTOL, H_ATOL = 4e-6, 4e-8
MOMENT_TARGET = (
    "0.93593426035233407829942113248797319202758378269392307275632575",
    ("0.31809463443152624864427894144050501476714792118349655326807733",
     "0.36265421746892202971593896028582661237121123838921201011062881",
     "-0.01594090493892320425785678004558609578219760883417083696548324",
     "-0.00021752690478713893840431940024737109390819794388011453306035",
     "-0.00007930231974166852874990842970927544116840538906920076395257"),
    ("1.07106076377356209323182441789320877590141467966205999145241613",
     "0.77058890521132312494996250621523149971990335670991980369381010",
     "-0.15552788243900022030509880503490598784359055414825505250643024",
     "-0.00297420719414660183731993281174278233020220316792216124624658",
     "-0.00187342195649098690781314684017798861961220787299271250363509",
     "0.77058890521132312494996250621523149971990335670991980369381010",
     "0.50096474924372408881171586568294120191240284556350065277956211",
     "-0.07349464383111562707601673302764849437213954521610705775684620",
     "-0.00163413824984764138283814933929929103739906993809100380389288",
     "-0.00118183732668812494030324261313319291818532932639394457832360",
     "-0.15552788243900022030509880503490598784359055414825505250643024",
     "-0.07349464383111562707601673302764849437213954521610705775684620",
     "-0.00830407039486704785592786608952931898940980345878628356468511",
     "0.00015085421949527053599473790413951640181361034679566594936558",
     "0.00099604031407532440385956479363567525593110974124781696799625",
     "-0.00297420719414660183731993281174278233020220316792216124624658",
     "-0.00163413824984764138283814933929929103739906993809100380389288",
     "0.00015085421949527053599473790413951640181361034679566594936558",
     "0.00000782599176340073973976575528269621215384595930721954409318",
     "0.00000035190546690268220822567426509242794285324071877285814543",
     "-0.00187342195649098690781314684017798861961220787299271250363509",
     "-0.00118183732668812494030324261313319291818532932639394457832360",
     "0.00099604031407532440385956479363567525593110974124781696799625",
     "0.00000035190546690268220822567426509242794285324071877285814543",
     "-0.00001347625261540865278226777727506594509605320332879309228346"),
)


@lru_cache(maxsize=12)
def legendre_rule(count, digits):
    with mp.workdps(digits):
        nodes, weights = mp.gauss_quadrature(count, "legendre")
        return tuple((nodes[i], weights[i]) for i in range(count))


def compose_partials(value, gradient, hessian, arguments):
    """Compose exact local derivatives at the arguments' current scalar point."""
    width, local = len(arguments[0].gradient), len(arguments)
    return Jet(value,
        tuple(sum(gradient[a]*arguments[a].gradient[i] for a in range(local))
              for i in range(width)),
        tuple(sum(gradient[a]*arguments[a].hessian[i*width+j] for a in range(local))
              + sum(hessian[a*local+b]*arguments[a].gradient[i]*arguments[b].gradient[j]
                    for a in range(local) for b in range(local))
              for i in range(width) for j in range(width)))


@dataclass(frozen=True)
class BesselIntegral:
    count: int
    digits: int
    cutoff: int = 24

    def partials(self, order, argument):
        scale = mp.sqrt(argument)
        sums = [mp.mpf(0)]*6
        for left in range(0, self.cutoff, 4):
            for node, weight in legendre_rule(self.count, self.digits):
                t = (left+2*(node+1))/scale
                excess = 2*mp.sinh(t/2)**2
                kernel = 2*weight*mp.exp(-argument*excess)*mp.cosh(order*t)
                slope = t*mp.tanh(order*t)
                for i, term in enumerate((1, slope, excess, t*t,
                                          slope*excess, excess*excess)):
                    sums[i] += kernel*term
        _, dn, dx, dnn, dnx, dxx = [v/sums[0] for v in sums]
        return (-argument-mp.log(scale)+mp.log(sums[0]), (dn, -1-dx),
                (dnn-dn*dn, -dnx+dn*dx, -dnx+dn*dx, dxx-dx*dx))

    def logk(self, order, argument):
        return compose_partials(*self.partials(order.value, argument.value), (order, argument))


@dataclass(frozen=True)
class NormalizedGH:
    alpha: Jet
    beta: Jet
    scale: Jet
    location: Jet
    order: Jet
    normalizer: Jet
    bessel: BesselIntegral

    @classmethod
    def prepare(cls, rho, zeta, lam, bessel):
        logk = [bessel.logk(lam+i, zeta) for i in range(3)]
        first, second = [exp_jet(entry-logk[0]) for entry in logk[1:]]
        gap = 1-rho*rho
        deviation = power_jet(gap*first/zeta+rho*rho*(second-first*first), mp.mpf(".5"))
        alpha = zeta*deviation/gap
        scale = power_jet(gap, mp.mpf(".5"))/deviation
        normalizer = lam*log_jet(zeta/(scale*scale))-mp.log(2*mp.pi)/2-logk[0]
        return cls(alpha, rho*alpha, scale, -rho*first/deviation,
                   lam-mp.mpf(".5"), normalizer, bessel)

    def logpdf(self, innovation):
        centered = innovation-self.location
        radius = power_jet(self.scale*self.scale+centered*centered, mp.mpf(".5"))
        return (self.normalizer+self.beta*centered
                +self.bessel.logk(self.order, self.alpha*radius)
                +self.order*log_jet(radius/self.alpha))


@lru_cache(maxsize=16)
def normalized_moment(point, digits=60, outer=32, inner=32, cutoff=24):
    """Independent GH moment AD at the exact supplied MP point, all 31 entries."""
    with mp.workdps(digits):
        assert len(point) == 5
        gamma, delta, rho, zeta, lam = [Jet.variable(v, 5, i) for i, v in enumerate(point)]
        assert abs(gamma.value) < 1 and delta.value > 0
        assert abs(rho.value) < 1 and zeta.value > 0
        density = NormalizedGH.prepare(rho, zeta, lam, BesselIntegral(inner, digits, cutoff))
        factors = [exp_jet(delta*log_jet(1+sign*gamma)) for sign in (-1, 1)]
        cuts = [mp.mpf(0)]+[2/mp.pi*mp.atan(mp.mpf(x)**mp.mpf(".25"))
            for x in (".25", 1, 2, 3, 4, "4.5", 5, "5.5", 6, 7, 9, 12, 20, 40)]+[mp.mpf(1)]
        result = Jet.constant(0, 5)
        for left, right in zip(cuts, cuts[1:]):
            half = (right-left)/2
            for node, weight in legendre_rule(outer, digits):
                tangent = mp.tan(mp.pi*(left+half*(node+1))/2)
                magnitude = tangent**4
                jacobian = 2*mp.pi*tangent**3*(1+tangent*tangent)
                power = exp_jet(delta*mp.log(magnitude))
                mass = sum(factor*exp_jet(density.logpdf(sign*magnitude))
                           for factor, sign in zip(factors, (1, -1)))
                result += power*mass*(half*weight*jacobian)
        return result


def moment_at_arguments(arguments, digits, outer, inner, cutoff):
    point = tuple(argument.value for argument in arguments)
    local = normalized_moment(point, digits, outer, inner, cutoff)
    return compose_partials(local.value, local.gradient, local.hessian, arguments)


def parameter_layout(orders, mean_width=None):
    ar, ma, p, q = orders if orders is not None else (0, 0, 1, 1)
    mean = (0 if orders is None else 1+ar+ma) if mean_width is None else mean_width
    alpha = mean+1
    gamma, beta = alpha+p, alpha+2*p
    delta = beta+q
    return ar, ma, p, q, mean, alpha, gamma, beta, delta


def complete_likelihood(parameters, data, orders, density, h0, bessel,
                        design=None, first_scored=None):
    mean_width = None if design is None else len(design[0])
    ar, ma, p, q, mean, alpha, gamma, beta, delta = parameter_layout(orders, mean_width)
    width = len(parameters)
    sample = [mp.mpf(float(value)) for value in data]
    first = (0 if orders is None else max(orders)) if design is None else first_scored
    residuals = ([Jet.constant(value, width) for value in sample] if orders is None and design is None
                 else [Jet.constant(0, width) for _ in sample])
    if design is not None:
        assert orders is None and 0 <= first < len(sample)
        assert len(design) == len(sample) and all(len(row) == mean for row in design)
        for t in range(first, len(sample)):
            predicted = sum(parameters[j]*mp.mpf(float(design[t][j])) for j in range(mean))
            residuals[t] = sample[t]-predicted
    elif orders is not None:
        for t in range(first, len(sample)):
            predicted = parameters[0]
            predicted += sum(parameters[1+i]*sample[t-1-i] for i in range(ar))
            predicted += sum(parameters[1+ar+i]*residuals[t-1-i] for i in range(ma))
            residuals[t] = sample[t]-predicted
    power = [exp_jet(parameters[delta]*mp.log(mp.mpf(float(h0)))/2)]*len(sample)
    for t in range(max(first, 1), len(sample)):
        current = parameters[mean]
        for lag in range(p):
            index = t-1-lag
            if index < first:
                continue
            residual = residuals[index]
            assert residual.value != 0, "Stress fixture must remain differentiable"
            shock = (1 if residual.value > 0 else -1)*residual-parameters[gamma+lag]*residual
            current += parameters[alpha+lag]*exp_jet(parameters[delta]*log_jet(shock))
        current += sum(parameters[beta+lag]*power[t-1-lag] for lag in range(q) if t > lag)
        power[t] = current
    rho, zeta = parameters[delta+1:delta+3]
    lam = parameters[-1] if density == "gh" else Jet.constant(mp.mpf("-.5"), width)
    normalized = NormalizedGH.prepare(rho, zeta, lam, bessel)
    result = Jet.constant(0, width)
    for t in range(first, len(sample)):
        log_sigma = log_jet(power[t])/parameters[delta]
        innovation = residuals[t]*exp_jet(-log_sigma)
        result += log_sigma-normalized.logpdf(innovation)
    return result


def mapped_parameters(coordinates, orders, density, digits=60, outer=32, inner=32, cutoff=24):
    ar, ma, p, q, mean, alpha, gamma, beta, delta = parameter_layout(orders)
    width = len(coordinates)
    point = [Jet.variable(mp.mpf(float(v)), width, i) for i, v in enumerate(coordinates)]
    mapped = point.copy()

    def tanh(value):
        exponential = exp_jet(2*value)
        return (exponential-1)/(exponential+1)

    def bounded(value, lower, upper):
        # Differentiate the original coordinate, including rounded endpoint tails.
        tail = exp_jet(-value if value.value >= 0 else value)
        tail = tail/(1+tail)
        return upper-(upper-lower)*tail if value.value >= 0 else lower+(upper-lower)*tail

    if orders is not None:
        for start, order, sign in ((1, ar, 1), (1+ar, ma, -1)):
            coefficients = []
            for stage in range(order):
                reflection = tanh(point[start+stage])
                coefficients = [coefficients[i]-reflection*coefficients[stage-1-i]
                                for i in range(stage)]+[reflection]
            mapped[start:start+order] = [sign*value for value in coefficients]
    tiny = mp.mpf(float(np.finfo(float).tiny))
    mapped[mean] = tiny+log_jet(1+exp_jet(point[mean]))
    mapped[gamma:gamma+p] = [tanh(value) for value in point[gamma:gamma+p]]
    mapped[delta] = tiny+log_jet(1+exp_jet(point[delta]))
    mapped[delta+1] = mp.mpf(.995)*tanh(point[delta+1])
    mapped[delta+2] = bounded(point[delta+2], mp.mpf(.1), mp.mpf(25))
    lam = (bounded(point[-1], mp.mpf(-6), mp.mpf(6)) if density == "gh"
           else Jet.constant(mp.mpf("-.5"), width))
    if density == "gh":
        mapped[-1] = lam
    weights = [exp_jet(value) for value in point[alpha:alpha+p]+point[beta:beta+q]]
    limit = mp.mpf(float(1-8*(p+q+1)*np.finfo(float).eps))
    denominator = 1+sum(weights)
    contributions = [limit*weight/denominator for weight in weights]
    for lag in range(p):
        moment = moment_at_arguments((mapped[gamma+lag], mapped[delta],
            mapped[delta+1], mapped[delta+2], lam), digits, outer, inner, cutoff)
        mapped[alpha+lag] = contributions[lag]/moment
    mapped[beta:beta+q] = contributions[p:]
    return mapped


def complete_oracle(theta, data, orders, density, h0=H0, coordinates=None,
                    digits=60, outer=32, inner=32, cutoff=24,
                    design=None, first_scored=None):
    with mp.workdps(digits):
        assert design is None or coordinates is None, "Linear adapter is theta-space only"
        if coordinates is None:
            parameters = [Jet.variable(mp.mpf(float(v)), len(theta), i)
                          for i, v in enumerate(theta)]
        else:
            parameters = mapped_parameters(coordinates, orders, density, digits, outer, inner, cutoff)
            # Anchor F to actual mapped binary64 theta; retain original-z map G/H.
            parameters = [Jet(mp.mpf(float(value)), parameter.gradient, parameter.hessian)
                          for value, parameter in zip(theta, parameters, strict=True)]
        return complete_likelihood(parameters, data, orders, density, h0,
                                   BesselIntegral(inner, digits, cutoff), design, first_scored)


def stress_point(orders, density, *, transformed=False, plateau=False):
    ar, ma, p, q, mean, alpha, gamma, beta, delta = parameter_layout(orders)
    shape = [-.994, 24.999999]+([-.5] if density == "gh" else [])
    theta = np.r_[([] if orders is None else [.03, *([.12]*ar), *([-.08]*ma)]),
                  .08, np.full(p, .07/p), np.full(p, .15), np.full(q, .82/max(q, 1)), 1.7, shape]
    if not transformed:
        return theta
    z = theta.copy()
    if orders is not None:
        # Independent inverse Schur recursion, not native coefficients.
        for start, order, sign in ((1, ar, 1), (1+ar, ma, -1)):
            coefficients = list(sign*theta[start:start+order])
            for stage in range(order-1, -1, -1):
                reflection = coefficients[stage]
                z[start+stage] = np.arctanh(reflection)
                coefficients = [(coefficients[i]+reflection*coefficients[stage-1-i])
                                /(1-reflection*reflection) for i in range(stage)]
        if plateau:
            z[1] = 18.5
            if ma:
                z[1+ar] = 18.5
    z[mean], z[delta] = np.log(np.expm1(theta[mean])), np.log(np.expm1(theta[delta]))
    z[gamma:gamma+p] = np.arctanh(theta[gamma:gamma+p])
    z[delta+1] = np.arctanh(theta[delta+1]/.995)
    z[delta+2] = np.log((theta[delta+2]-.1)/(25-theta[delta+2]))
    if density == "gh":
        z[-1] = np.log((theta[-1]+6)/(6-theta[-1]))
    moment = float(MOMENT_TARGET[0])
    limit = 1-8*(p+q+1)*np.finfo(float).eps
    contributions = np.r_[theta[alpha:alpha+p]*moment, theta[beta:beta+q]]
    logits = np.log(contributions/(limit-contributions.sum()))
    z[alpha:alpha+p], z[beta:beta+q] = logits[:p], logits[p:]
    return z


def assert_requested(actual, expected, mask):
    for bit, output, reference in zip((1, 2, 4), actual, expected, strict=True):
        if not mask & bit:
            assert np.isnan(output).all()
            continue
        assert np.isfinite(output).all() and np.isfinite(reference).all()
        if bit == 1:
            np.testing.assert_allclose(output, reference, rtol=0, atol=NLL_ATOL)
        else:
            np.testing.assert_allclose(output, reference, rtol=G_RTOL if bit == 2 else H_RTOL,
                                       atol=G_ATOL if bit == 2 else H_ATOL)


def test_refined_normalized_moment_matches_all_31_independent_targets():
    with mp.workdps(60):
        point = tuple(mp.mpf(float(value)) for value in STRESS)
        result = normalized_moment(point)
        target = (mp.mpf(MOMENT_TARGET[0]), *map(mp.mpf, MOMENT_TARGET[1]),
                  *map(mp.mpf, MOMENT_TARGET[2]))
        assert max(abs(a-b) for a, b in zip(
            (result.value, *result.gradient, *result.hessian), target, strict=True)) < mp.mpf("1e-20")


def test_stress_density_bessel_and_exact_nig_reduction():
    with mp.workdps(80):
        bessel = BesselIntegral(40, 80)
        for order, argument in ((-1, 25), (mp.mpf("-.5"), 4000)):
            order, argument = mp.mpf(order), mp.mpf(argument)
            value, gradient, hessian = bessel.partials(order, argument)
            exact = mp.besselk(order, argument)
            slope = -(mp.besselk(order-1, argument)+mp.besselk(order+1, argument))/(2*exact)
            curvature = 1+order*order/(argument*argument)-slope/argument-slope*slope
            assert abs(value-mp.log(exact)) < mp.mpf("1e-35")
            assert abs(gradient[1]-slope) < mp.mpf("1e-35")
            assert abs(hessian[-1]-curvature) < mp.mpf("1e-35")
        x, rho, zeta = [Jet.variable(mp.mpf(float(v)), 3, i)
                        for i, v in enumerate((-.4, -.994, 24.999999))]
        half = Jet.constant(mp.mpf("-.5"), 3)
        general = NormalizedGH.prepare(rho, zeta, half, bessel).logpdf(x)
        root = power_jet(zeta, mp.mpf(".5"))
        gap = 1-rho*rho
        scale = power_jet(zeta*gap, mp.mpf(".5"))
        alpha = root/gap
        centered = x+rho*root
        radius = power_jet(scale*scale+centered*centered, mp.mpf(".5"))
        reduced = (zeta+log_jet(scale)-mp.log(mp.pi)+rho*alpha*centered
                   +bessel.logk(Jet.constant(-1, 3), alpha*radius)-log_jet(radius/alpha))
        assert max(abs(a-b) for a, b in zip(
            (general.value, *general.gradient, *general.hessian),
            (reduced.value, *reduced.gradient, *reduced.hessian), strict=True)) < mp.mpf("1e-35")


CASES = ((None, False), ((1, 1, 1, 1), False), ((1, 1, 1, 1), True),
         ((2, 1, 3, 2), False), ((0, 0, 1, 1), False))


def qualify_stress_case(density, code, orders, plateau):
    from test_native_aparch import _evaluate, _forced_runtime, _map, _native_core

    theta = stress_point(orders, density)
    z = stress_point(orders, density, transformed=True, plateau=plateau)
    mapped = _map(z, code, orders=orders, forward=True)
    if plateau:
        assert _map(mapped, code, orders=orders)[1] != z[1]
    rows = []
    for transformed, point, anchor in ((False, theta, theta), (True, z, mapped)):
        expected = jet_outputs(complete_oracle(anchor, DATA, orders, density,
                                               coordinates=z if transformed else None))
        full = _evaluate(point, DATA, code, orders=orders, transformed=transformed, initial_variance=H0)
        assert full[:2] == (0, 7), (orders, density, transformed, full[:2])
        assert_requested(full[2:], expected, 7)
        dispatch = (_native_core._aparch_dispatch_kind(1, 1, code, int(transformed)) if orders is None
                    else _native_core._arma_aparch_dispatch_kind(*orders, code, int(transformed)))
        assert dispatch == int(orders is None or orders == (1, 1, 1, 1))
        curvature_maximum = None
        if transformed:
            with mp.workdps(60):
                theta_jet = complete_oracle(anchor, DATA, orders, density)
                parameters = mapped_parameters(z, orders, density)
                curvature = [sum(coefficient*parameter.hessian[i]
                                 for coefficient, parameter in zip(theta_jet.gradient, parameters))
                             for i in range(anchor.size*anchor.size)]
                curvature_maximum = float(max(abs(value) for value in curvature))
                assert curvature_maximum > 1e-6, "Fixture must detect omitted map curvature"
        if orders is not None:
            mean = 1+sum(orders[:2])
            end = mean+2+2*orders[2]+orders[3]
            for left, right in ((slice(0, mean), slice(mean, end)),
                                (slice(0, mean), slice(end, None)),
                                (slice(mean, end), slice(end, None))):
                assert np.max(np.abs(expected[2][left, right])) > 1e-6
        before = point.tobytes(), DATA.tobytes()
        for mask in range(1, 8):
            actual = _evaluate(point, DATA, code, mask, orders=orders,
                               transformed=transformed, initial_variance=H0)
            assert actual[:2] == (0, mask)
            runtime = _forced_runtime(point, DATA, code, orders=orders, request=mask,
                                      transformed=transformed, initial_variance=H0)
            assert_requested(actual[2:], expected, mask)
            assert_requested(runtime, expected, mask)
            assert_requested(actual[2:], full[2:], mask)
            for bit, fixed, general in zip((1, 2, 4), actual[2:], runtime, strict=True):
                if mask & bit:
                    np.testing.assert_allclose(fixed, general, rtol=2e-12, atol=2e-12)
            if plateau and transformed:
                tails = [1, 1+orders[0]]
                for outputs in (actual[2:], runtime):
                    if mask & 2:
                        np.testing.assert_allclose(outputs[1][tails], expected[1][tails], rtol=G_RTOL, atol=0)
                    if mask & 4:
                        np.testing.assert_allclose(outputs[2][tails, :], expected[2][tails, :], rtol=H_RTOL, atol=0)
            assert before == (point.tobytes(), DATA.tobytes())
            rows.append(dict(density=density, orders=orders, plateau=plateau,
                transformed=transformed, mask=mask, dispatch=dispatch,
                theta_hex=[float(value).hex() for value in anchor],
                coordinate_hex=[float(value).hex() for value in point],
                F_error=abs(float(actual[2])-float(expected[0][0])) if mask & 1 else None,
                G_error=float(np.max(np.abs(actual[3]-expected[1]))) if mask & 2 else None,
                H_error=float(np.max(np.abs(actual[4]-expected[2]))) if mask & 4 else None,
                map_curvature_maximum=curvature_maximum,
                inputs_preserved=True, unused_outputs_untouched=True, runtime_parity=True))
    return rows


@pytest.mark.parametrize("density,code", [("gh", 4), ("nig", 5)])
@pytest.mark.parametrize("orders,plateau", [
    pytest.param(None, False, id="standalone11"),
    pytest.param((1, 1, 1, 1), False, id="joint1111"),
    pytest.param((1, 1, 1, 1), True, id="original-z-plateau"),
    pytest.param((2, 1, 3, 2), False, id="runtime2132"),
    pytest.param((0, 0, 1, 1), False, id="zero-mean-orders"),
])
def test_complete_stress_theta_and_original_z_all_masks(density, code, orders, plateau):
    qualify_stress_case(density, code, orders, plateau)


@pytest.mark.parametrize("orders", [None, (1, 1, 1, 1), (2, 1, 3, 2), (0, 0, 1, 1)])
def test_complete_nig_is_common_gh_theta_projection(orders):
    from test_native_aparch import _evaluate, _forced_runtime

    nig = stress_point(orders, "nig")
    gh = np.r_[nig, -.5]
    with mp.workdps(60):
        gh_jet = complete_oracle(gh, DATA, orders, "gh")
        nig_jet = complete_oracle(nig, DATA, orders, "nig")
        width, common = gh.size, nig.size
        common_hessian = tuple(gh_jet.hessian[i*width+j] for i in range(common) for j in range(common))
        assert max(abs(a-b) for a, b in zip(
            (gh_jet.value, *gh_jet.gradient[:common], *common_hessian),
            (nig_jet.value, *nig_jet.gradient, *nig_jet.hessian), strict=True)) < mp.mpf("1e-35")
        expected = jet_outputs(gh_jet)
        assert abs(expected[1][-1]) > 1e-6
        assert np.max(np.abs(expected[2][-1, :-1])) > 1e-6
    for mask in range(1, 8):
        left = _evaluate(nig, DATA, 5, mask, orders=orders, initial_variance=H0)
        right = _evaluate(gh, DATA, 4, mask, orders=orders, initial_variance=H0)
        assert left[:2] == right[:2] == (0, mask)
        assert_requested(left[2:], (right[2], right[3][:-1], right[4][:-1, :-1]), mask)
        assert_requested(right[2:], expected, mask)
        runtime_nig = _forced_runtime(nig, DATA, 5, request=mask, orders=orders, initial_variance=H0)
        runtime_gh = _forced_runtime(gh, DATA, 4, request=mask, orders=orders, initial_variance=H0)
        assert_requested(runtime_nig, (runtime_gh[0], runtime_gh[1][:-1], runtime_gh[2][:-1, :-1]), mask)
        assert_requested(runtime_gh, expected, mask)


@pytest.mark.parametrize("orders,plateau", [
    (None, False), ((1, 1, 1, 1), True), ((2, 1, 3, 2), False), ((0, 0, 1, 1), False),
])
def test_complete_nig_is_common_gh_original_z_projection(orders, plateau):
    from test_native_aparch import _evaluate, _forced_runtime, _map

    nig = stress_point(orders, "nig", transformed=True, plateau=plateau)
    gh = np.r_[nig, np.log(5.5/6.5)]
    anchor = _map(gh, 4, orders=orders, forward=True)
    assert anchor[-1] == -.5
    expected = jet_outputs(complete_oracle(anchor, DATA, orders, "gh", coordinates=gh))
    assert abs(expected[1][-1]) > 1e-6
    assert np.max(np.abs(expected[2][-1, :-1])) > 1e-6
    for mask in range(1, 8):
        left = _evaluate(nig, DATA, 5, mask, transformed=True, orders=orders, initial_variance=H0)
        right = _evaluate(gh, DATA, 4, mask, transformed=True, orders=orders, initial_variance=H0)
        assert left[:2] == right[:2] == (0, mask)
        assert_requested(left[2:], (right[2], right[3][:-1], right[4][:-1, :-1]), mask)
        assert_requested(right[2:], expected, mask)
        runtime_nig = _forced_runtime(nig, DATA, 5, request=mask,
            transformed=True, orders=orders, initial_variance=H0)
        runtime_gh = _forced_runtime(gh, DATA, 4, request=mask,
            transformed=True, orders=orders, initial_variance=H0)
        assert_requested(runtime_nig, (runtime_gh[0], runtime_gh[1][:-1], runtime_gh[2][:-1, :-1]), mask)
        assert_requested(runtime_gh, expected, mask)
        if plateau:
            tails = [1, 1+orders[0]]
            for outputs in (right[2:], runtime_gh):
                if mask & 2:
                    np.testing.assert_allclose(outputs[1][tails], expected[1][tails], rtol=G_RTOL, atol=0)
                if mask & 4:
                    np.testing.assert_allclose(outputs[2][tails, :], expected[2][tails, :], rtol=H_RTOL, atol=0)


@pytest.mark.parametrize("code", [4, 5])
def test_zero_volatility_order_is_outside_declared_aparch_model(code):
    from test_native_aparch import _map, _native_core

    orders = (0, 0, 1, 0)
    z = stress_point(orders, "gh" if code == 4 else "nig", transformed=True)
    with pytest.raises(ValueError, match="coordinate map rejected"):
        _map(z, code, orders=orders, forward=True)
    # The Python binding returns a one-byte allocation sentinel for invalid layouts.
    assert _native_core._arma_aparch_workspace_bytes(*orders, code, 0, 7) == 1
