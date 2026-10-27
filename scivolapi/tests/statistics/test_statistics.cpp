#ifdef NDEBUG
#undef NDEBUG
#endif
#include <scivolapi/scivolapi.hpp>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <new>
#include <random>
#include <vector>

namespace { bool watching = false; std::size_t allocations = 0; }
void* operator new(std::size_t size) {
    if (watching) ++allocations;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace scivolapi;

int main()
{
    assert(std::abs(chi_square_survival(20.0, 2.0) / std::exp(-10.0) - 1.0) < 1e-13);
    assert(std::abs(student_survival(1.0, 1.0) - 0.25) < 1e-14);
    assert(normal_survival(37.0) > 0.0);
    std::vector<double> series(80), variance(80, 1.0);
    for (std::size_t t = 0; t < series.size(); ++t) series[t] = std::sin(double(t) * 1.7) + 0.3;
    const auto test = portmanteau(series, 5, 2);
    assert(test.ljung_box.df == 3 && test.ljung_box.status == TestStatus::ok);
    assert(portmanteau(variance, 5).ljung_box.status == TestStatus::not_estimable);
    const auto ic = information_criteria(-100.0, 3, 100);
    assert(ic.aic == 206.0 && std::abs(ic.bic - (200.0 + 3 * std::log(100.0))) < 1e-13);

    ModelSpec spec;
    spec.mean.constant = true;
    spec.mean.ar = spec.mean.ma = 1;
    const Model model(spec);
    const Sample sample = model.prepare(series);
    Jet jet(model, sample);
    const std::vector<double> theta{0.1, 0.2, 0.15, 0.1, 0.1, 0.8};
    assert(std::abs(model.unconditional_mean(theta)-0.125) < 1e-14);
    assert(std::abs(model.unconditional_variance(theta)-1.0) < 1e-14);
    const std::size_t n = sample.scored(), d = model.dimension();
    std::vector<double> log_variance(n*d), residual(n*d), sentinel(d, 12345.0), hessian_buffer(d*d, 6789.0);
    JetOutput out;
    out.gradient = sentinel;
    out.hessian = hessian_buffer;
    out.log_variance_sensitivities = log_variance;
    out.residual_sensitivities = residual;
    const unsigned request = log_variance_sensitivities | residual_sensitivities;
    watching = true;
    const auto evaluated = jet.theta(theta, request, out);
    watching = false;
    assert(allocations == 0 && evaluated.valid == request);
    assert(out.value != out.value && sentinel == std::vector<double>(d, 12345.0));
    assert(hessian_buffer == std::vector<double>(d*d, 6789.0));
    assert(log_variance[0] == 0.0 && residual[0] == -1.0);
    std::vector<double> z(d);
    assert(model.to_z(theta, z));
    watching = true;
    const auto mapped = jet.z(z, request, out);
    watching = false;
    assert(allocations == 0 && mapped.valid == request);
    out.log_variance_sensitivities = {};
    assert(jet.theta(theta, request, out).status == Status::invalid_argument);

    const Model face_model(ModelSpec{});
    assert(face_model.closed_face_normals(std::vector<double>{0.1, 0.2, 0.0}).size() == 3);
    // A representational persistence margin is not a closed model face.
    assert(face_model.closed_face_normals(std::vector<double>{0.1, 0.2, std::nextafter(0.8, 0.0)}).empty());

    ModelSpec constant_spec;
    constant_spec.volatility = VolatilityKind::constant;
    constant_spec.p = constant_spec.q = 0;
    Model constant(constant_spec);
    std::mt19937_64 generator(402);
    std::normal_distribution<double> normal;
    std::vector<double> arch_series(600);
    double variance_now = 1.0, previous_error = 0.0;
    for (double& x : arch_series) {
        variance_now = 0.2 + 0.6 * previous_error * previous_error;
        x = previous_error = std::sqrt(variance_now) * normal(generator);
    }
    const Sample constant_sample = constant.prepare(arch_series);
    Jet constant_jet(constant, constant_sample);
    double sigma2 = 0.0;
    for (double x : arch_series) sigma2 += x*x / arch_series.size();
    const std::vector<double> constant_theta{sigma2};
    const auto battery = adequacy(constant_jet, constant_theta);
    assert(battery.checks.size() == 5 && battery.volatility_gum == "unavailable");
    assert(battery.status == AdequacyStatus::rejected);
    std::vector<double> standardized(arch_series);
    for (double& x : standardized) x /= std::sqrt(sigma2);
    const auto arch = absolute_li_mak(constant_jet, constant_theta, 10).test;
    bool found_arch = false;
    for (const auto& check : battery.checks) {
        if (check.name == "absolute_li_mak[10]") {
            found_arch = true;
            assert(check.test.status == TestStatus::ok);
            assert(std::abs(check.test.statistic - arch.statistic) < 1e-10);
        }
        assert(check.name != "berkowitz" && check.name != "dgt");
    }
    assert(found_arch);

    std::vector<double> losses(80*3, 0.0), uniforms(199*20, 0.5);
    for (std::size_t t = 0; t < 80; ++t) losses[t*3+2] = 0.5;
    for (auto method : {MCSStatistic::maximum, MCSStatistic::range}) {
        const auto mcs = model_confidence_set(losses, 80, 3, uniforms, 199, 4, 0.05, method);
        assert(mcs.included == std::vector<std::size_t>({0, 1}));
        assert(mcs.eliminated == std::vector<std::size_t>({2}));
        assert(mcs.p_values[2] < 0.05 && mcs.p_values[0] == 1.0 && mcs.p_values[1] == 1.0);
    }
}
