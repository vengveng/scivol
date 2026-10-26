#include <scivolapi/scivolapi.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main()
{
    using namespace scivolapi;
    const std::vector<double> identity{1, 0, 0, 1}, tested{1, 0};
    const std::vector<double> retained{0, 1};
    const auto ordinary = cone_projection_test(identity, 2, retained, tested, 3.0);
    assert(ordinary.status == LRStatus::ok);
    assert(std::abs(ordinary.p_value-chi_square_survival(3.0, 1)) < ordinary.numerical_error);
    const auto boundary = cone_projection_test(identity, 2, identity, tested, 3.0);
    assert(boundary.status == LRStatus::ok);
    assert(std::abs(boundary.p_value-.5*chi_square_survival(3.0, 1)) < boundary.numerical_error);
    const auto repeated = cone_projection_test(identity, 2, identity, tested, 3.0);
    assert(repeated.p_value == boundary.p_value && repeated.numerical_error == boundary.numerical_error);
    const auto threshold = cone_projection_test(identity, 2, identity, tested, 3.0, boundary.p_value);
    assert(threshold.status == LRStatus::unresolved);
    const auto zero = cone_projection_test(identity, 2, identity, tested, 0.0);
    assert(zero.status == LRStatus::ok && zero.p_value == 1 && zero.numerical_error == 0);
    assert(cone_projection_test(std::vector<double>{1, 1, 1, 1}, 2, retained, tested, 3).status
           == LRStatus::singular_information);
    assert(cone_projection_test(std::vector<double>{1, 0, 1, 1}, 2, retained, tested, 3).status
           == LRStatus::invalid_argument);
    // Units of the original parameters cannot change either projection law.
    const auto scaled = cone_projection_test(std::vector<double>{1e-18, 0, 0, 1e18}, 2, identity, tested, 3);
    assert(scaled.status == LRStatus::ok && scaled.p_value == boundary.p_value);

    // The model-driven path removes an exactly orthogonal retained tail face
    // analytically, leaving the ordinary half-mixture for a tested beta.
    ModelSpec spec;
    spec.density = DensityKind::student_t;
    const Model parent(spec);
    const auto edges = parent.restrictions();
    const auto edge = std::find_if(edges.begin(), edges.end(), [](const Restriction& r) {
        return r.name == "volatility.beta";
    });
    assert(edge != edges.end());
    const Model child(edge->child);
    const std::vector<double> y(30, 1.0);
    const ComparisonPlan plan(y, 0);
    FitResult pf, cf;
    pf.success = cf.success = true;
    pf.fitted_spec = parent.spec(); cf.fitted_spec = child.spec();
    pf.comparison = cf.comparison = plan.identity();
    pf.observations = cf.observations = pf.scored = cf.scored = 30;
    pf.theta = {.1, .2, .5, .1}; cf.theta = {.1, .2, 0};
    pf.log_likelihood = -99; cf.log_likelihood = -100;
    std::vector<double> info(16, 0);
    for (std::size_t i = 0; i < 4; ++i) info[5*i] = 1;
    const auto law = restriction_test(parent, child, *edge, pf, cf, info);
    assert(law.status == LRStatus::ok && law.law == ReferenceLaw::chi_bar_square);
    assert(law.weights == std::vector<double>({.5, .5}));
    assert(std::abs(law.p_value-.5*chi_square_survival(2, 1)) < 1e-14);
}
