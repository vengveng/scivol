#include <scivolapi/scivolapi.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <numbers>
#include <vector>

using namespace scivolapi;

#define CHECK(condition) do { if (!(condition)) { std::cerr << "failed at line " << __LINE__ << "\n"; std::abort(); } } while (false)

int main()
{
    ModelSpec s;
    s.mean.constant = true;
    s.mean.ar = 2;
    s.mean.exog = 2;
    s.volatility = VolatilityKind::gjr;
    s.p = 2;
    s.density = DensityKind::skew_t;
    const Model parent(s);
    std::vector<double> y(100), x(200);
    for (std::size_t t = 0; t < y.size(); ++t) {
        y[t] = std::sin(1.3 * t) + 0.1;
        x[2 * t] = std::cos(0.7 * t);
        x[2 * t + 1] = std::sin(0.3 * t);
    }
    const ComparisonPlan plan(y, 8, x, 2, 1.7);
    const Sample sample = parent.prepare(plan);
    CHECK(sample.first_scored == 8 && sample.scored() == 92 && sample.variance_seed == 1.7);
    auto edges = parent.restrictions();
    for (const auto& r : edges) {
        const Model child(r.child);
        const Sample reduced = child.prepare(plan);
        const auto theta = child.start(reduced);
        const auto lifted = r.embed(theta);
        CHECK(lifted && parent.feasible(*lifted) && r.on_face(*lifted));
        const auto back = r.map_start(*lifted);
        CHECK(back && *back == theta);
        Jet pjet(parent, sample), cjet(child, reduced);
        JetOutput po, co;
        CHECK(pjet.theta(*lifted, value, po).valid == value);
        CHECK(cjet.theta(theta, value, co).valid == value);
        CHECK(std::abs(po.value - co.value) < 1e-10);
        if (r.name == "volatility.shock") {
            CHECK(r.types.size() == 2 && r.types[0] == RestrictionType::closed_boundary);
            const auto d = parent.dimension();
            CHECK(r.directions[d + 1 + s.mean.width() + s.p - 1] == 1.0);
        }
    }
    for (auto kind : {VolatilityKind::garch, VolatilityKind::gjr, VolatilityKind::egarch, VolatilityKind::aparch})
        for (std::size_t p : {0u, 1u}) for (std::size_t q : {0u, 1u}) {
            ModelSpec zero;
            zero.volatility = kind;
            zero.p = p;
            zero.q = q;
            const Model model(zero);
            const auto data = model.prepare(y);
            const auto theta = model.start(data);
            CHECK(model.feasible(theta));
            Jet jet(model, data);
            std::vector<double> g(model.dimension()), h(g.size() * g.size());
            JetOutput out;
            out.gradient = g;
            out.hessian = h;
            const auto result = jet.theta(theta, value | gradient | hessian, out);
            CHECK(result.valid == (value | gradient | hessian));
        }
    for (std::size_t m = 1; m <= 8; ++m) {
        std::vector<double> identity(m * m, 0.0);
        for (std::size_t i = 0; i < m; ++i) identity[i * m + i] = 1.0;
        const auto w = chi_bar_weights(identity, m);
        CHECK(w.status == LRStatus::ok);
        double binomial = std::pow(0.5, static_cast<double>(m));
        for (std::size_t k = 0; k <= m; ++k) {
            CHECK(std::abs(w.weights[k] - binomial) < 1e-12);
            binomial *= static_cast<double>(m - k) / (k + 1);
        }
    }
    ModelSpec ts;
    ts.volatility = VolatilityKind::constant;
    ts.p = ts.q = 0;
    ts.density = DensityKind::student_t;
    const Model tail(ts);
    const Restriction normal = tail.restrictions().front();
    const Model gaussian(normal.child);
    FitResult pf, cf;
    pf.fitted_spec = tail.spec();
    cf.fitted_spec = gaussian.spec();
    pf.comparison = cf.comparison = plan.identity();
    pf.theta = {1.0, 0.1};
    cf.theta = {1.0};
    pf.success = cf.success = true;
    pf.observations = cf.observations = 100;
    pf.scored = cf.scored = 92;
    pf.log_likelihood = -99.0;
    cf.log_likelihood = -100.0;
    const std::vector<double> info{3.0, 0.8, 0.8, 2.0};
    auto lr = restriction_test(tail, gaussian, normal, pf, cf, info);
    CHECK(lr.status == LRStatus::ok && lr.statistic == 2.0 && lr.interior_df == 0);
    CHECK(std::abs(lr.p_value - 0.5 * chi_square_survival(2.0, 1)) < 1e-14);
    pf.log_likelihood = -101.0;
    lr = restriction_test(tail, gaussian, normal, pf, cf, info);
    CHECK(lr.status == LRStatus::negative_statistic && lr.statistic == -2.0);
    pf.log_likelihood = -99.0;
    pf.estimator = Estimator::qmle;
    CHECK(restriction_test(tail, gaussian, normal, pf, cf, info).status == LRStatus::unsupported);
    pf.estimator = Estimator::mle;
    CHECK(restriction_test(tail, gaussian, normal, pf, cf, std::vector<double>{1, 1, 1, 1}).status
          == LRStatus::singular_information);
    const double threshold = 0.5 * chi_square_survival(2.0, 1);
    CHECK(restriction_test(tail, gaussian, normal, pf, cf, info, threshold).status == LRStatus::unresolved);

    ModelSpec ss = ts;
    ss.density = DensityKind::skew_t;
    const Model skew(ss);
    Restriction mixed;
    mixed.parent = ss;
    mixed.child = gaussian.spec();
    mixed.types = {RestrictionType::closed_boundary, RestrictionType::interior_equality};
    mixed.offset = {0, 0, 0};
    mixed.embedding = {1, 0, 0};
    mixed.directions = {0, 1, 0, 0, 0, 1};
    mixed.null_values = {0, 0};
    FitResult sf = pf;
    sf.fitted_spec = ss;
    sf.theta = {1.0, 0.1, 0.1};
    const std::vector<double> mixed_info{3, .4, .2, .4, 2, .3, .2, .3, 1};
    lr = restriction_test(skew, gaussian, mixed, sf, cf, mixed_info);
    CHECK(lr.status == LRStatus::ok && lr.interior_df == 1);
    CHECK(std::abs(lr.p_value - 0.5 * (chi_square_survival(2.0, 1) + chi_square_survival(2.0, 2))) < 1e-14);
    mixed.identification = Identification::unidentified_under_null;
    CHECK(restriction_test(skew, gaussian, mixed, sf, cf, mixed_info).status == LRStatus::unsupported);

    ModelSpec cancelled;
    cancelled.p = cancelled.q = 2;
    cancelled.density = DensityKind::student_t;
    const Model overfit(cancelled);
    const auto nesting = overfit.restrictions();
    const auto tail_edge = std::find_if(nesting.begin(), nesting.end(), [](const Restriction& r) {
        return r.name == "density.normal";
    });
    CHECK(tail_edge != nesting.end());
    CHECK(!tail_edge->identified(std::vector<double>{.1, .1, .02, .5, .14}));
    cancelled.mean.ar = cancelled.mean.ma = 1;
    const Model arma(cancelled);
    const auto mean_nesting = arma.restrictions();
    const auto mean_tail = std::find_if(mean_nesting.begin(), mean_nesting.end(), [](const Restriction& r) {
        return r.name == "density.normal";
    });
    CHECK(!mean_tail->identified(std::vector<double>{.2, -.2, .1, .1, .03, .5, .14}));

    const auto w = chi_bar_weights(std::vector<double>{1.0, -0.5, -0.5, 1.0}, 2);
    CHECK(w.status == LRStatus::ok);
    CHECK(std::abs(w.weights[2] - 1.0 / 3.0) < 1e-13);
    CHECK(std::abs(chi_square_survival(2.705543454095404, 1) - 0.1) < 1e-12);
    CHECK(chi_square_survival(0.0, 0) == 1.0);
}
