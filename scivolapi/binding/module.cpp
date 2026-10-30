// Python binding for scivolapi: thin, checked, and free of mathematics.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include <scivolapi/scivolapi.hpp>

#include "../src/density/distribution.hpp"
#include "../src/density/moments.hpp"
#include "../src/optimize/kinks.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace scivolapi;

namespace {

using Vector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Tensor3 = nb::ndarray<const double, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using NumpyVector = nb::ndarray<nb::numpy, double, nb::ndim<1>>;
using NumpyMatrix = nb::ndarray<nb::numpy, double, nb::ndim<2>>;
using ReadOnlyVector = nb::ndarray<nb::numpy, const double, nb::ndim<1>>;
using ReadOnlyMatrix = nb::ndarray<nb::numpy, const double, nb::ndim<2>>;

// Read-only NumPy views into a FitResult; the Python result object stays alive
// for as long as any view does (rv_policy::reference_internal).
ReadOnlyVector vector_view(const std::vector<double>& v) { return ReadOnlyVector(v.data(), {v.size()}); }
ReadOnlyMatrix matrix_view(const std::vector<double>& v, std::size_t d)
{
    return ReadOnlyMatrix(v.data(), {v.empty() ? 0 : d, v.empty() ? 0 : d});
}

std::span<const double> view(const Vector& v) { return {v.data(), v.shape(0)}; }

// Exogenous regressors (rows x columns), or none.
std::vector<double> exogenous(const std::optional<Matrix>& x)
{
    if (!x) return {};
    return std::vector<double>(x->data(), x->data() + x->shape(0) * x->shape(1));
}

// Owns copies of the data so the borrowed Sample can never outlive them.
struct BoundJet {
    std::shared_ptr<const Model> model;
    std::vector<double> y, x;
    std::optional<ComparisonPlan> comparison;
    Sample sample;
    Jet jet;

    BoundJet(std::shared_ptr<const Model> m, std::span<const double> data, std::vector<double> regressors)
        : model(std::move(m)), y(data.begin(), data.end()), x(std::move(regressors)), sample(model->prepare(y, x)),
          jet(*model, sample)
    {
    }

    BoundJet(std::shared_ptr<const Model> m, const ComparisonPlan& plan)
        : model(std::move(m)), comparison(plan), sample(model->prepare(*comparison)), jet(*model, sample)
    {
    }
};

template<class T>
nb::object owned_vector(std::vector<T> values)
{
    auto* heap = new std::vector<T>(std::move(values));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::cast(NumpyVector(heap->data(), {heap->size()}, owner));
}

nb::object owned_matrix(std::vector<double> values, std::size_t rows, std::size_t cols)
{
    auto* heap = new std::vector<double>(std::move(values));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    return nb::cast(NumpyMatrix(heap->data(), {rows, cols}, owner));
}

const char* status_name(Status s)
{
    switch (s) {
    case Status::ok: return "ok";
    case Status::invalid_argument: return "invalid_argument";
    case Status::not_evaluable: return "not_evaluable";
    case Status::numerical_range: return "numerical_range";
    }
    return "unknown";
}

// Owns its residuals and sample; the jet borrows both.
struct BoundCorrelationJet {
    std::shared_ptr<const CorrelationModel> model;
    std::vector<double> eps;
    CorrelationSample sample;
    CorrelationJet jet;

    BoundCorrelationJet(std::shared_ptr<const CorrelationModel> m, const Matrix& residuals,
                        std::vector<double> target, const std::vector<InnovationDensity>& marginals)
        : model(std::move(m)), eps(residuals.data(), residuals.data() + residuals.shape(0) * residuals.shape(1)),
          sample(model->prepare(eps, residuals.shape(1), target, marginals)), jet(*model, sample)
    {
    }
};

std::vector<double> square(const std::optional<Matrix>& m)
{
    if (!m) return {};
    return std::vector<double>(m->data(), m->data() + m->shape(0) * m->shape(1));
}

nb::dict evaluate_correlation(BoundCorrelationJet& self, const Vector& point, unsigned request, bool in_z)
{
    const std::size_t d = self.model->dimension(), n = self.sample.rows(), k = self.sample.series;
    std::vector<double> g((request & gradient) ? d : 0), h((request & hessian) ? d * d : 0);
    std::vector<double> s((request & scores) ? n * d : 0), terms_((request & terms) ? n : 0);
    std::vector<double> r((request & paths) ? n * k * k : 0);
    JetOutput out;
    out.gradient = g;
    out.hessian = h;
    out.scores = s;
    out.terms = terms_;
    const JetResult result = in_z ? self.jet.z(view(point), request, out) : self.jet.theta(view(point), request, out, r);
    nb::dict d_out;
    d_out["status"] = status_name(result.status);
    d_out["valid"] = result.valid;
    if (result.valid & value) d_out["value"] = out.value;
    if (result.valid & gradient) d_out["gradient"] = owned_vector(std::move(g));
    if (result.valid & hessian) d_out["hessian"] = owned_matrix(std::move(h), d, d);
    if (result.valid & scores) d_out["scores"] = owned_matrix(std::move(s), n, d);
    if (result.valid & terms) d_out["terms"] = owned_vector(std::move(terms_));
    if (result.valid & paths) {
        auto* heap = new std::vector<double>(std::move(r));
        nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
        d_out["correlations"] = nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<3>>(heap->data(), {n, k, k}, owner));
    }
    return d_out;
}

template<std::size_t D>
nb::object owned_tensor(std::vector<double> values, std::array<std::size_t, D> shape)
{
    auto* heap = new std::vector<double>(std::move(values));
    nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    return nb::cast(nb::ndarray<nb::numpy, double>(heap->data(), D, shape.data(), owner));
}

nb::dict simulation_dict(Simulation s)
{
    nb::dict out;
    const std::size_t paths = s.paths, horizon = s.horizon;
    out["y"] = owned_matrix(std::move(s.y), paths, horizon);
    out["mean"] = owned_matrix(std::move(s.mean), paths, horizon);
    out["residual"] = owned_matrix(std::move(s.residual), paths, horizon);
    out["variance"] = owned_matrix(std::move(s.variance), paths, horizon);
    return out;
}

std::span<const double> view(const Matrix& m) { return {m.data(), m.shape(0) * m.shape(1)}; }

// Pickle state of a specification (Models are rebuilt from it).
using SpecState = std::tuple<bool, std::size_t, std::vector<std::size_t>, std::size_t, std::size_t, int, std::size_t,
                             std::size_t, int, int>;

SpecState spec_state(const ModelSpec& s)
{
    return {s.mean.constant, s.mean.ar, s.mean.har, s.mean.exog, s.mean.ma, static_cast<int>(s.volatility), s.p, s.q,
            static_cast<int>(s.density), static_cast<int>(s.stationarity)};
}

ModelSpec spec_from(const SpecState& t)
{
    ModelSpec s;
    std::tie(s.mean.constant, s.mean.ar, s.mean.har, s.mean.exog, s.mean.ma, std::ignore, s.p, s.q, std::ignore,
             std::ignore) = t;
    s.volatility = static_cast<VolatilityKind>(std::get<5>(t));
    s.density = static_cast<DensityKind>(std::get<8>(t));
    s.stationarity = static_cast<Stationarity>(std::get<9>(t));
    return s;
}

nb::dict evaluate(BoundJet& self, const Vector& point, unsigned request, bool in_z)
{
    const std::size_t d = self.model->dimension(), n = self.sample.scored();
    std::vector<double> g((request & gradient) ? d : 0), h((request & hessian) ? d * d : 0);
    std::vector<double> e((request & paths) ? n : 0), v(e.size()), z(e.size()), s((request & scores) ? n * d : 0),
        terms_((request & terms) ? n : 0), lv((request & log_variance_sensitivities) ? n * d : 0),
        re((request & residual_sensitivities) ? n * d : 0);
    JetOutput out;
    out.gradient = g;
    out.hessian = h;
    out.paths = {e, v, z};
    out.scores = s;
    out.terms = terms_;
    out.log_variance_sensitivities = lv;
    out.residual_sensitivities = re;
    std::unique_ptr<optimize::KinkWorkspace> kink;
    if (request & kinks) {
        kink = std::make_unique<optimize::KinkWorkspace>(n, d);
        out.kink = kink->out;
    }
    const JetResult r = in_z ? self.jet.z(view(point), request, out) : self.jet.theta(view(point), request, out);

    nb::dict result;
    result["status"] = status_name(r.status);
    result["valid"] = r.valid;
    if (r.valid & value) result["value"] = out.value;
    if (r.valid & gradient) result["gradient"] = owned_vector(std::move(g));
    if (r.valid & hessian) result["hessian"] = owned_matrix(std::move(h), d, d);
    if (r.valid & paths) {
        result["residual"] = owned_vector(std::move(e));
        result["variance"] = owned_vector(std::move(v));
        result["innovation"] = owned_vector(std::move(z));
    }
    if (r.valid & scores) result["scores"] = owned_matrix(std::move(s), n, d);
    if (r.valid & terms) result["terms"] = owned_vector(std::move(terms_));
    if (r.valid & log_variance_sensitivities) result["log_variance_sensitivities"] = owned_matrix(std::move(lv), n, d);
    if (r.valid & residual_sensitivities) result["residual_sensitivities"] = owned_matrix(std::move(re), n, d);
    if (r.valid & kinks) {
        const auto count = out.kink.count;
        kink->rows.resize(count); kink->weights.resize(count);
        kink->residuals.resize(count); kink->normals.resize(count*d); kink->curvatures.resize(count*d*d);
        result["kink_rows"] = kink->rows;
        result["kink_supported"] = out.kink.supported;
        result["kink_residuals"] = owned_vector(std::move(kink->residuals));
        result["kink_weights"] = owned_vector(std::move(kink->weights));
        result["kink_normals"] = owned_matrix(std::move(kink->normals), count, d);
        result["kink_curvatures"] = owned_tensor<3>(std::move(kink->curvatures), {count, d, d});
        if (count && out.kink.supported) {
            result["smooth_gradient"] = owned_vector(std::move(kink->gradient));
            result["smooth_hessian"] = owned_matrix(std::move(kink->hessian), d, d);
        }
    }
    return result;
}

nb::dict test_dictionary(const TestStatistic& test, double alpha = 0.05)
{
    nb::dict out;
    out["statistic"] = test.statistic;
    out["p_value"] = test.p_value;
    out["df"] = test.df;
    out["status"] = nb::cast(test.status).attr("name");
    out["log_p_value"] = test.log_p_value;
    out["reject"] = test.rejects(alpha);
    return out;
}

nb::dict portmanteau_dictionary(Portmanteau test, std::size_t lags, double alpha = 0.05)
{
    nb::dict out = test_dictionary(test.ljung_box, alpha);
    out["q_stat"] = test.ljung_box.statistic;
    out["lags"] = lags;
    out["box_pierce"] = test_dictionary(test.box_pierce, alpha);
    out["autocorrelations"] = owned_vector(std::move(test.autocorrelations));
    return out;
}

nb::dict sign_bias_dictionary(const SignBias& test, double alpha = 0.05)
{
    nb::dict out;
    out["sign"] = test_dictionary(test.individual[0], alpha);
    out["negative_size"] = test_dictionary(test.individual[1], alpha);
    out["positive_size"] = test_dictionary(test.individual[2], alpha);
    out["joint"] = test_dictionary(test.joint, alpha);
    return out;
}

nb::dict berkowitz_dictionary(const Berkowitz& test, double alpha = 0.05)
{
    nb::dict out = test_dictionary(test.test, alpha);
    out["intercept"] = test.mean;
    out["variance"] = test.variance;
    out["ar1"] = test.ar1;
    out["likelihood"] = "conditional_on_first_observation";
    return out;
}

nb::dict li_mak_dictionary(LiMak test, std::size_t lags, std::size_t parameters, double alpha = 0.05)
{
    nb::dict out = test_dictionary(test.test, alpha);
    out["lags"] = lags;
    out["autocorrelations"] = owned_vector(std::move(test.autocorrelations));
    if (!test.covariance.empty()) out["covariance"] = owned_matrix(std::move(test.covariance), lags, lags);
    if (!test.derivative.empty()) out["derivative"] = owned_matrix(std::move(test.derivative), lags, parameters);
    out["calibration"] = "joint_score_influence_sandwich";
    return out;
}

}  // namespace

NB_MODULE(_scivolapi, m)
{
    m.doc() = "scivolapi: native volatility-model mathematics";

    nb::enum_<AdequacyStatus>(m, "AdequacyStatus")
        .value("adequate", AdequacyStatus::adequate)
        .value("rejected", AdequacyStatus::rejected)
        .value("adequacy_unresolved", AdequacyStatus::adequacy_unresolved);

    nb::enum_<TestStatus>(m, "TestStatus")
        .value("ok", TestStatus::ok)
        .value("not_estimable", TestStatus::not_estimable)
        .value("probability_underflow", TestStatus::probability_underflow)
        .value("numerical_range", TestStatus::numerical_range)
        .value("infeasible_parameters", TestStatus::infeasible_parameters)
        .value("fourth_moment_unavailable", TestStatus::fourth_moment_unavailable)
        .value("invalid_derivatives", TestStatus::invalid_derivatives)
        .value("invalid_qmle_derivatives", TestStatus::invalid_qmle_derivatives)
        .value("at_boundary", TestStatus::at_boundary)
        .value("singular_information", TestStatus::singular_information)
        .value("singular_moment_covariance", TestStatus::singular_moment_covariance)
        .value("pit_endpoint", TestStatus::pit_endpoint);

    nb::enum_<VolatilityKind>(m, "VolatilityKind")
        .value("garch", VolatilityKind::garch)
        .value("gjr", VolatilityKind::gjr)
        .value("egarch", VolatilityKind::egarch)
        .value("aparch", VolatilityKind::aparch)
        .value("constant", VolatilityKind::constant);
    nb::enum_<Stationarity>(m, "Stationarity")
        .value("covariance", Stationarity::covariance)
        .value("integrated", Stationarity::integrated)
        .value("none", Stationarity::none);
    nb::enum_<DensityKind>(m, "DensityKind")
        .value("normal", DensityKind::normal)
        .value("student_t", DensityKind::student_t)
        .value("skew_t", DensityKind::skew_t)
        .value("ged", DensityKind::ged)
        .value("generalized_hyperbolic", DensityKind::generalized_hyperbolic)
        .value("nig", DensityKind::nig)
        .value("hyperbolic", DensityKind::hyperbolic)
        .value("variance_gamma", DensityKind::variance_gamma);

    m.attr("VALUE") = static_cast<unsigned>(value);
    m.attr("GRADIENT") = static_cast<unsigned>(gradient);
    m.attr("HESSIAN") = static_cast<unsigned>(hessian);
    m.attr("PATHS") = static_cast<unsigned>(paths);
    m.attr("SCORES") = static_cast<unsigned>(scores);
    m.attr("TERMS") = static_cast<unsigned>(terms);
    m.attr("LOG_VARIANCE_SENSITIVITIES") = static_cast<unsigned>(log_variance_sensitivities);
    m.attr("RESIDUAL_SENSITIVITIES") = static_cast<unsigned>(residual_sensitivities);
    m.attr("KINKS") = static_cast<unsigned>(kinks);

    nb::class_<MeanSpec>(m, "MeanSpec")
        .def(nb::init<>())
        .def_rw("constant", &MeanSpec::constant)
        .def_rw("ar", &MeanSpec::ar)
        .def_rw("har", &MeanSpec::har)
        .def_rw("exog", &MeanSpec::exog)
        .def_rw("ma", &MeanSpec::ma)
        .def_prop_ro("conditioning", &MeanSpec::conditioning);

    nb::class_<ModelSpec>(m, "ModelSpec")
        .def(nb::init<>())
        .def("__deepcopy__", [](const ModelSpec& s, nb::dict) { return s; }, "memo"_a)
        .def_rw("mean", &ModelSpec::mean)
        .def_rw("volatility", &ModelSpec::volatility)
        .def_rw("p", &ModelSpec::p)
        .def_rw("q", &ModelSpec::q)
        .def_rw("stationarity", &ModelSpec::stationarity)
        .def_rw("density", &ModelSpec::density)
        .def("__eq__", [](const ModelSpec& a, const ModelSpec& b) { return a == b; });

    auto model_binding = nb::class_<Model>(m, "Model")
        .def("__init__", [](Model* self, const ModelSpec& spec) { new (self) Model(spec); }, "spec"_a)
        .def("__getstate__", [](const Model& m) { return spec_state(m.spec()); })
        .def("__setstate__", [](Model& self, const SpecState& state) { new (&self) Model(spec_from(state)); })
        .def_prop_ro("dimension", &Model::dimension)
        .def_prop_ro("parameter_names", &Model::parameter_names)
        .def_prop_ro("residual_degrees_of_freedom", &Model::residual_degrees_of_freedom)
        .def("feasible", [](const Model& m, const Vector& theta) { return m.feasible(view(theta)); })
        .def("at_open_bound", [](const Model& m, const Vector& theta) { return m.at_open_bound(view(theta)); })
        .def_prop_ro("constraint_names", &Model::constraint_names)
        .def("open_constraints", [](const Model& m) {
            std::vector<char> open(m.constraint_count());
            m.open_constraints(open);
            return std::vector<bool>(open.begin(), open.end());
        })
        .def("bounds", [](const Model& m) {
            std::vector<double> lower(m.dimension()), upper(m.dimension());
            m.bounds(lower, upper);
            return nb::make_tuple(owned_vector(std::move(lower)), owned_vector(std::move(upper)));
        })
        .def("open_bounds", [](const Model& m) {
            std::vector<char> lower(m.dimension()), upper(m.dimension());
            m.open_bounds(lower, upper);
            return nb::make_tuple(std::vector<bool>(lower.begin(), lower.end()), std::vector<bool>(upper.begin(), upper.end()));
        }, "Which finite bounds are excluded from the domain (strict inequalities).")
        .def_prop_ro("capabilities", &Model::capabilities)
        .def("persistence", [](const Model& m, const Vector& theta) { return m.persistence(view(theta)); }, "theta"_a)
        .def("unconditional_variance", [](const Model& m, const Vector& theta) { return m.unconditional_variance(view(theta)); }, "theta"_a)
        .def("unconditional_mean", [](const Model& m, const Vector& theta) { return m.unconditional_mean(view(theta)); }, "theta"_a)
        .def("innovation_power_moment", [](const Model& m, const Vector& theta, double gamma, double power) {
            return m.innovation_power_moment(view(theta), gamma, power);
        }, "theta"_a, "gamma"_a, "power"_a)
        .def("innovation_density", [](const Model& m, const Vector& theta) { return m.innovation_density(view(theta)); },
             "theta"_a)
        .def("constraints", [](const Model& m, const Vector& theta) {
            const std::size_t c = m.constraint_count(), d = m.dimension();
            std::vector<double> values(c), jacobian(c * d);
            m.constraints(view(theta), values, jacobian);
            return nb::make_tuple(owned_vector(std::move(values)), owned_matrix(std::move(jacobian), c, d));
        }, "theta"_a, "Constraint values c(theta) >= 0 beyond the bounds, and their Jacobian.")
        .def("rescale", [](const Model& m, const Vector& theta, double factor) {
            std::vector<double> scaled(m.dimension());
            m.rescale(view(theta), factor, scaled);
            return owned_vector(std::move(scaled));
        }, "theta"_a, "factor"_a)
        .def("to_theta", [](const Model& m, const Vector& z) {
            std::vector<double> theta(m.dimension());
            m.to_theta(view(z), theta);
            return owned_vector(std::move(theta));
        })
        .def("to_z", [](const Model& m, const Vector& theta) -> nb::object {
            std::vector<double> z(m.dimension());
            if (!m.to_z(view(theta), z)) return nb::none();
            return owned_vector(std::move(z));
        })
        .def("start", [](const Model& m, const Vector& y, const std::optional<Matrix>& x) {
            std::vector<double> data(view(y).begin(), view(y).end()), regressors = exogenous(x);
            return owned_vector(m.start(m.prepare(data, regressors)));
        }, "y"_a, "x"_a = nb::none());

    nb::enum_<Coordinates>(m, "Coordinates")
        .value("automatic", Coordinates::automatic)
        .value("z", Coordinates::z)
        .value("theta", Coordinates::theta);
    nb::enum_<Termination>(m, "Termination")
        .value("converged", Termination::converged)
        .value("iteration_limit", Termination::iteration_limit)
        .value("line_search_failed", Termination::line_search_failed)
        .value("stalled", Termination::stalled)
        .value("open_bound", Termination::open_bound)
        .value("failed", Termination::failed);
    nb::enum_<Inference>(m, "Inference")
        .value("available", Inference::available)
        .value("not_feasible", Inference::not_feasible)
        .value("not_stationary", Inference::not_stationary)
        .value("at_boundary", Inference::at_boundary)
        .value("at_kink", Inference::at_kink)
        .value("at_excluded_boundary", Inference::at_excluded_boundary)
        .value("not_positive_definite", Inference::not_positive_definite)
        .value("ill_conditioned", Inference::ill_conditioned);

    nb::enum_<Estimator>(m, "Estimator")
        .value("mle", Estimator::mle)
        .value("qmle", Estimator::qmle);

    nb::class_<FitOptions>(m, "FitOptions")
        .def(nb::init<>())
        .def_rw("estimator", &FitOptions::estimator)
        .def_rw("coordinates", &FitOptions::coordinates)
        .def_rw("accuracy", &FitOptions::accuracy)
        .def_rw("maximum_iterations", &FitOptions::maximum_iterations)
        .def_rw("start", &FitOptions::start);

    nb::class_<FitPhase>(m, "FitPhase")
        .def_ro("name", &FitPhase::name)
        .def_ro("termination", &FitPhase::termination)
        .def_ro("iterations", &FitPhase::iterations)
        .def_ro("value_requests", &FitPhase::value_requests)
        .def_ro("gradient_requests", &FitPhase::gradient_requests)
        .def_ro("hessian_requests", &FitPhase::hessian_requests)
        .def_ro("value", &FitPhase::value);

    constexpr auto internal = nb::rv_policy::reference_internal;
    nb::class_<FitResult>(m, "FitResult")
        .def_ro("estimate_available", &FitResult::estimate_available)
        .def_ro("names", &FitResult::names)
        .def_prop_ro("theta", [](const FitResult& r) { return vector_view(r.theta); }, internal)
        .def_ro("log_likelihood", &FitResult::log_likelihood)
        .def_prop_ro("gradient", [](const FitResult& r) { return vector_view(r.gradient); }, internal)
        .def_prop_ro("hessian", [](const FitResult& r) { return matrix_view(r.hessian, r.theta.size()); }, internal)
        .def_prop_ro("covariance", [](const FitResult& r) { return matrix_view(r.covariance, r.theta.size()); }, internal)
        .def_prop_ro("std_errors", [](const FitResult& r) { return vector_view(r.std_errors); }, internal)
        .def_ro("estimator", &FitResult::estimator)
        .def_prop_ro("opg", [](const FitResult& r) { return matrix_view(r.opg, r.theta.size()); }, internal)
        .def_prop_ro("robust_covariance",
                     [](const FitResult& r) { return matrix_view(r.robust_covariance, r.theta.size()); }, internal)
        .def_prop_ro("robust_std_errors", [](const FitResult& r) { return vector_view(r.robust_std_errors); },
                     internal)
        .def_prop_ro("residual", [](const FitResult& r) { return vector_view(r.residual); }, internal)
        .def_prop_ro("variance", [](const FitResult& r) { return vector_view(r.variance); }, internal)
        .def_prop_ro("innovation", [](const FitResult& r) { return vector_view(r.innovation); }, internal)
        .def_ro("observations", &FitResult::observations)
        .def_ro("scored", &FitResult::scored)
        .def_ro("data_scale", &FitResult::data_scale)
        .def_ro("feasible", &FitResult::feasible)
        .def_ro("termination", &FitResult::termination)
        .def_ro("kkt_residual", &FitResult::kkt_residual)
        .def_ro("kink_point", &FitResult::kink_point)
        .def_ro("active_kink_rows", &FitResult::active_kink_rows)
        .def_ro("excluded_boundary", &FitResult::excluded_boundary)
        .def_ro("kkt_tolerance", &FitResult::kkt_tolerance)
        .def_ro("stationary", &FitResult::stationary)
        .def_ro("inference", &FitResult::inference)
        .def_ro("success", &FitResult::success)
        .def_ro("phases", &FitResult::phases)
        .def_ro("warnings", &FitResult::warnings);

    m.def("fit", [](const Model& model, const Vector& y, const FitOptions& options, const std::optional<Matrix>& x) {
              const std::vector<double> regressors = exogenous(x);
              nb::gil_scoped_release release;
              return fit(model, view(y), regressors, options);
          },
          "model"_a, "y"_a, "options"_a = FitOptions(), "x"_a = nb::none());

    m.def("simulate",
          [](const Model& model, const Vector& theta, const Vector& y, const Matrix& z, const std::optional<Matrix>& x,
             const std::optional<Matrix>& future_x) {
              const std::vector<double> rows = exogenous(x), future = exogenous(future_x);
              return simulation_dict(simulate(model, view(theta), view(y), rows, future, view(z), z.shape(0)));
          },
          "model"_a, "theta"_a, "y"_a, "z"_a, "x"_a = nb::none(), "future_x"_a = nb::none());
    m.def("simulate_series",
          [](const Model& model, const Vector& theta, const Matrix& z, const std::optional<Matrix>& x) {
              const std::vector<double> rows = exogenous(x);
              return simulation_dict(simulate_series(model, view(theta), rows, view(z), z.shape(0)));
          },
          "model"_a, "theta"_a, "z"_a, "x"_a = nb::none());
    m.def("forecast",
          [](const Model& model, const Vector& theta, const Vector& y, std::size_t horizon,
             const std::optional<Matrix>& x, const std::optional<Matrix>& future_x, const std::optional<Matrix>& z) {
              const std::vector<double> rows = exogenous(x), future = exogenous(future_x), draws = exogenous(z);
              Forecast f = forecast(model, view(theta), view(y), rows, future, horizon, draws, z ? z->shape(0) : 0);
              nb::dict out;
              out["mean"] = owned_vector(std::move(f.mean));
              out["residual_variance"] = owned_vector(std::move(f.residual_variance));
              out["variance"] = owned_vector(std::move(f.variance));
              out["simulated"] = f.simulated;
              return out;
          },
          "model"_a, "theta"_a, "y"_a, "horizon"_a, "x"_a = nb::none(), "future_x"_a = nb::none(),
          "z"_a = nb::none());

    nb::enum_<CorrelationDensity>(m, "CorrelationDensity")
        .value("normal", CorrelationDensity::normal)
        .value("student_t", CorrelationDensity::student_t);
    nb::class_<CorrelationSpec>(m, "CorrelationSpec")
        .def(nb::init<>())
        .def_rw("p", &CorrelationSpec::p)
        .def_rw("q", &CorrelationSpec::q)
        .def_rw("density", &CorrelationSpec::density);
    nb::class_<CorrelationModel>(m, "CorrelationModel")
        .def("__init__", [](CorrelationModel* self, const CorrelationSpec& spec) { new (self) CorrelationModel(spec); },
             "spec"_a)
        .def("__getstate__", [](const CorrelationModel& m) {
            return std::make_tuple(m.spec().p, m.spec().q, static_cast<int>(m.spec().density));
        })
        .def("__setstate__", [](CorrelationModel& self, const std::tuple<std::size_t, std::size_t, int>& state) {
            CorrelationSpec spec;
            spec.p = std::get<0>(state);
            spec.q = std::get<1>(state);
            spec.density = static_cast<CorrelationDensity>(std::get<2>(state));
            new (&self) CorrelationModel(spec);
        })
        .def_prop_ro("dimension", &CorrelationModel::dimension)
        .def_prop_ro("parameter_names", &CorrelationModel::parameter_names)
        .def("innovation_density", [](const CorrelationModel& m, const Vector& theta) { return m.innovation_density(view(theta)); })
        .def("feasible", [](const CorrelationModel& m, const Vector& theta) { return m.feasible(view(theta)); })
        .def("start", [](const CorrelationModel& m) { return owned_vector(m.start()); })
        .def("to_theta", [](const CorrelationModel& m, const Vector& z) {
            std::vector<double> theta(m.dimension());
            m.to_theta(view(z), theta);
            return owned_vector(std::move(theta));
        })
        .def("to_z", [](const CorrelationModel& m, const Vector& theta) -> nb::object {
            std::vector<double> z(m.dimension());
            if (!m.to_z(view(theta), z)) return nb::none();
            return owned_vector(std::move(z));
        });
    nb::class_<BoundCorrelationJet>(m, "CorrelationJet")
        .def("__init__",
             [](BoundCorrelationJet* self, const CorrelationModel& model, const Matrix& eps,
                const std::optional<Matrix>& target, const std::vector<InnovationDensity>& marginals) {
                 new (self) BoundCorrelationJet(std::make_shared<const CorrelationModel>(model), eps, square(target), marginals);
             },
             "model"_a, "eps"_a, "target"_a = nb::none(), "marginals"_a = std::vector<InnovationDensity>{})
        .def("target", [](BoundCorrelationJet& j, const Vector& theta) {
            std::vector<double> target(j.sample.series * j.sample.series);
            if (j.jet.target(view(theta), target) != Status::ok)
                throw std::domain_error("the correlation target could not be computed at theta");
            return owned_matrix(std::move(target), j.sample.series, j.sample.series);
        }, "theta"_a)
        .def("theta", [](BoundCorrelationJet& j, const Vector& theta, unsigned request) {
            return evaluate_correlation(j, theta, request, false);
        }, "theta"_a, "request"_a)
        .def("z", [](BoundCorrelationJet& j, const Vector& z, unsigned request) {
            return evaluate_correlation(j, z, request, true);
        }, "z"_a, "request"_a);
    m.def("distribution",
          [](DensityKind kind, const std::vector<double>& shape, const Vector& z) {
              const std::size_t n = z.shape(0);
              std::vector<double> lower(n), upper(n);
              for (std::size_t i = 0; i < n; ++i)
                  if (detail::distribution(kind, shape.empty() ? nullptr : shape.data(), z.data()[i], lower[i], upper[i])
                      != detail::ExpectationStatus::ok)
                      throw std::domain_error("the distribution function could not be computed accurately");
              return nb::make_tuple(owned_vector(std::move(lower)), owned_vector(std::move(upper)));
          },
          "density"_a, "shape"_a, "z"_a, "Both tails P(Z <= z), P(Z > z) of the standardized density.");
    m.def("negative_second_moment",
          [](DensityKind kind, const std::vector<double>& shape) {
              detail::MomentResult r;
              if (!detail::negative_second_moment(kind, shape.empty() ? nullptr : shape.data(), 0, r))
                  throw std::domain_error("E[z^2 1(z < 0)] is not available at these shapes");
              return r.value;
          },
          "density"_a, "shape"_a, "E[z^2 1(z < 0)] of the standardized density (GJR's persistence weight).");
    m.def("quantiles",
          [](DensityKind kind, const std::vector<double>& shape, const Vector& u, bool upper) {
              const std::size_t n = u.shape(0);
              std::vector<double> out(n);
              const double* s = shape.empty() ? nullptr : shape.data();
              bool ok = true;
              if (upper) {
                  for (std::size_t i = 0; i < n && ok; ++i)
                      ok = detail::quantile_upper(kind, s, u.data()[i], out[i]) == detail::ExpectationStatus::ok;
              } else {
                  ok = detail::quantiles(kind, s, u.data(), n, out.data()) == detail::ExpectationStatus::ok;
              }
              if (!ok) throw std::domain_error("a quantile could not be computed accurately");
              return owned_vector(std::move(out));
          },
          "density"_a, "shape"_a, "u"_a, "upper"_a = false,
          "Quantiles of the standardized density at lower-tail (or, with upper, upper-tail) probabilities.");
    nb::class_<ModelCapabilities>(m, "ModelCapabilities")
        .def_ro("chart_covers_domain", &ModelCapabilities::chart_covers_domain)
        .def_ro("density_free_recursion", &ModelCapabilities::density_free_recursion)
        .def_ro("likelihood_can_have_gradient_kinks", &ModelCapabilities::likelihood_can_have_gradient_kinks)
        .def_ro("residual_kink_decomposition", &ModelCapabilities::residual_kink_decomposition)
        .def_ro("exact_variance_horizon", &ModelCapabilities::exact_variance_horizon);
    nb::class_<InnovationDensity>(m, "InnovationDensity")
        .def("__getstate__", [](const InnovationDensity& d) { return std::make_tuple(static_cast<int>(d.density), d.shape); })
        .def("__setstate__", [](InnovationDensity& self, const std::tuple<int, std::vector<double>>& state) {
            new (&self) InnovationDensity{static_cast<DensityKind>(std::get<0>(state)), std::get<1>(state)};
        })
        .def("__init__", [](InnovationDensity* self, DensityKind density, std::vector<double> shape) {
            new (self) InnovationDensity{density, std::move(shape)};
        }, "density"_a, "shape"_a = std::vector<double>{})
        .def_rw("density", &InnovationDensity::density)
        .def_rw("shape", &InnovationDensity::shape)
        .def_prop_ro("named", [](const InnovationDensity& d) {
            nb::dict out;
            for (const auto& [name, value] : named_shapes(d)) out[name.c_str()] = value;
            return out;
        }, "Every shape by name, estimated and fixed by the density's definition.")
        .def_prop_ro("derived", [](const InnovationDensity& d) {
            nb::dict out;
            for (const auto& [name, value] : derived_shapes(d)) out[name.c_str()] = value;
            return out;
        }, "Derived shape quantities for reporting.");
    m.def("simulate_correlation",
          [](const CorrelationModel& model, const Vector& theta, const Matrix& eps,
             const std::vector<InnovationDensity>& marginals, const Tensor3& normals, const std::optional<Matrix>& chi2,
             const std::optional<Matrix>& target) {
              const std::vector<double> residuals(eps.data(), eps.data() + eps.shape(0) * eps.shape(1));
              const std::vector<double> qbar = square(target), c = square(chi2);
              const std::size_t paths = normals.shape(0), horizon = normals.shape(1), series = eps.shape(1);
              CorrelationPaths r = simulate_correlation(
                  model, view(theta), residuals, series, qbar, marginals,
                  {normals.data(), paths * horizon * normals.shape(2)}, c, paths);
              nb::dict out;
              out["innovation"] = owned_tensor<3>(std::move(r.innovation), {paths, horizon, series});
              out["correlation"] = owned_tensor<4>(std::move(r.correlation), {paths, horizon, series, series});
              return out;
          },
          "model"_a, "theta"_a, "eps"_a, "marginals"_a, "normals"_a, "chi2"_a = nb::none(), "target"_a = nb::none());
    m.def("fit_correlation",
          [](const CorrelationModel& model, const Matrix& eps, const FitOptions& options,
             const std::optional<Matrix>& target, const std::vector<InnovationDensity>& marginals) {
              const std::vector<double> residuals(eps.data(), eps.data() + eps.shape(0) * eps.shape(1));
              const std::vector<double> qbar = square(target);
              const std::size_t series = eps.shape(1);
              nb::gil_scoped_release release;
              return fit(model, residuals, series, qbar, options, marginals);
          },
          "model"_a, "eps"_a, "options"_a = FitOptions(), "target"_a = nb::none(),
          "marginals"_a = std::vector<InnovationDensity>{});

    nb::class_<BoundJet>(m, "Jet")
        .def("__init__",
             [](BoundJet* self, const Model& model, const Vector& y, const std::optional<Matrix>& x) {
                 new (self) BoundJet(std::make_shared<const Model>(model), view(y), exogenous(x));
             },
             "model"_a, "y"_a, "x"_a = nb::none())
        .def("__init__", [](BoundJet* self, const Model& model, const ComparisonPlan& plan) {
            new (self) BoundJet(std::make_shared<const Model>(model), plan);
        }, "model"_a, "plan"_a)
        .def_prop_ro("first_scored", [](const BoundJet& j) { return j.sample.first_scored; })
        .def_prop_ro("variance_seed", [](const BoundJet& j) { return j.sample.variance_seed; })
        .def("theta", [](BoundJet& j, const Vector& theta, unsigned request) { return evaluate(j, theta, request, false); },
             "theta"_a, "request"_a)
        .def("z", [](BoundJet& j, const Vector& z, unsigned request) { return evaluate(j, z, request, true); },
             "z"_a, "request"_a);

    nb::class_<ComparisonPlan>(m, "ComparisonPlan")
        .def("__init__", [](ComparisonPlan* self, const Vector& y, std::size_t first,
                            const std::optional<Matrix>& x, std::optional<double> seed) {
            const auto rows = exogenous(x);
            if (x && x->shape(0) != y.shape(0)) throw std::invalid_argument("comparison x and y must have equal rows");
            new (self) ComparisonPlan(view(y), first, rows, x ? x->shape(1) : 0,
                                     seed.value_or(std::numeric_limits<double>::quiet_NaN()));
        }, "y"_a, "first_scored"_a, "x"_a = nb::none(), "variance_seed"_a = nb::none())
        .def_prop_ro("first_scored", &ComparisonPlan::first_scored)
        .def_prop_ro("variance_seed", &ComparisonPlan::variance_seed)
        .def_prop_ro("columns", &ComparisonPlan::columns)
        .def_prop_ro("y", [](const ComparisonPlan& p) { return owned_vector(std::vector<double>(p.y().begin(), p.y().end())); })
        .def_prop_ro("x", [](const ComparisonPlan& p) {
            return owned_matrix(std::vector<double>(p.x().begin(), p.x().end()), p.y().size(), p.columns());
        });
    nb::enum_<RestrictionType>(m, "RestrictionType")
        .value("interior_equality", RestrictionType::interior_equality)
        .value("closed_boundary", RestrictionType::closed_boundary);
    nb::enum_<Identification>(m, "Identification")
        .value("conditional", Identification::conditional)
        .value("unidentified_under_null", Identification::unidentified_under_null);
    nb::class_<Restriction>(m, "Restriction")
        .def_ro("name", &Restriction::name)
        .def_prop_ro("parent", [](const Restriction& r) { return r.parent; })
        .def_prop_ro("child", [](const Restriction& r) { return r.child; })
        .def_ro("types", &Restriction::types)
        .def_ro("identification", &Restriction::identification)
        .def_ro("condition", &Restriction::condition)
        .def_ro("offset", &Restriction::offset)
        .def_ro("embedding", &Restriction::embedding)
        .def_ro("directions", &Restriction::directions)
        .def_ro("null_values", &Restriction::null_values)
        .def("then", &Restriction::then)
        .def("embed", [](const Restriction& r, const Vector& t) -> nb::object {
            auto result = r.embed(view(t));
            return result ? owned_vector(std::move(*result)) : nb::none();
        })
        .def("map_start", [](const Restriction& r, const Vector& t) -> nb::object {
            auto result = r.map_start(view(t));
            return result ? owned_vector(std::move(*result)) : nb::none();
        })
        .def("warm_start", [](const Restriction& r, const Vector& t) -> nb::object {
            auto result = r.warm_start(view(t));
            return result ? owned_vector(std::move(*result)) : nb::none();
        })
        .def("on_face", [](const Restriction& r, const Vector& t) { return r.on_face(view(t)); })
        .def("identified", [](const Restriction& r, const Vector& t) { return r.identified(view(t)); });
    model_binding.def_prop_ro("key", [](const Model& m) {
            const auto& s = m.spec();
            return nb::make_tuple(s.mean.constant, s.mean.ar, nb::tuple(nb::cast(s.mean.har)),
                                  s.mean.exog, s.mean.ma, static_cast<int>(s.volatility), s.p, s.q,
                                  static_cast<int>(s.density), static_cast<int>(s.stationarity));
        })
        .def_prop_ro("constant_variance", &Model::constant_variance)
        .def_prop_ro("components", [](const Model& m) {
            const auto& s = m.spec();
            nb::list out;
            nb::dict kw;
            kw["constant"] = s.mean.constant;
            if (s.mean.ma) out.append(nb::make_tuple("ARMA", nb::make_tuple(s.mean.ar, s.mean.ma), kw));
            else if (!s.mean.har.empty()) out.append(nb::make_tuple("HARX", nb::make_tuple(s.mean.har), kw));
            else if (s.mean.width()) out.append(nb::make_tuple("ARX", nb::make_tuple(s.mean.ar), kw));
            const char* vol = nullptr;
            switch (s.volatility) {
                case VolatilityKind::garch: vol = "GARCH"; break;
                case VolatilityKind::gjr: vol = "GJRGARCH"; break;
                case VolatilityKind::egarch: vol = "EGARCH"; break;
                case VolatilityKind::aparch: vol = "APARCH"; break;
                case VolatilityKind::constant: break;
            }
            if (vol) {
                nb::dict settings;
                if (s.stationarity == Stationarity::integrated) settings["stationarity"] = "integrated";
                if (s.stationarity == Stationarity::none) settings["stationarity"] = "none";
                out.append(nb::make_tuple(vol, nb::make_tuple(s.p, s.q), settings));
            }
            const char* density = nullptr;
            switch (s.density) {
                case DensityKind::normal: density = "Normal"; break;
                case DensityKind::student_t: density = "StudentT"; break;
                case DensityKind::skew_t: density = "SkewT"; break;
                case DensityKind::ged: density = "GED"; break;
                case DensityKind::generalized_hyperbolic: density = "GeneralizedHyperbolic"; break;
                case DensityKind::nig: density = "NormalInverseGaussian"; break;
                case DensityKind::hyperbolic: density = "Hyperbolic"; break;
                case DensityKind::variance_gamma: density = "VarianceGamma"; break;
            }
            out.append(nb::make_tuple(density, nb::tuple(), nb::dict()));
            return out;
        })
        .def("restrictions", &Model::restrictions)
        .def_prop_ro("spec", [](const Model& model) { return model.spec(); })
        .def("start", [](const Model& m, const ComparisonPlan& p) { return owned_vector(m.start(m.prepare(p))); }, "plan"_a);
    nb::enum_<RestrictFitStatus>(m, "RestrictFitStatus")
        .value("reused", RestrictFitStatus::reused)
        .value("fit_needed", RestrictFitStatus::fit_needed)
        .value("parent_uncertified", RestrictFitStatus::parent_uncertified)
        .value("incompatible", RestrictFitStatus::incompatible)
        .value("child_uncertified", RestrictFitStatus::child_uncertified);
    nb::class_<RestrictedFit>(m, "RestrictedFit")
        .def_ro("status", &RestrictedFit::status)
        .def_ro("result", &RestrictedFit::result);
    nb::enum_<LRStatus>(m, "LRStatus")
        .value("ok", LRStatus::ok).value("unsupported", LRStatus::unsupported)
        .value("incompatible", LRStatus::incompatible).value("uncertified", LRStatus::uncertified)
        .value("negative_statistic", LRStatus::negative_statistic)
        .value("singular_information", LRStatus::singular_information)
        .value("unresolved", LRStatus::unresolved).value("invalid_argument", LRStatus::invalid_argument);
    nb::enum_<ReferenceLaw>(m, "ReferenceLaw")
        .value("chi_square", ReferenceLaw::chi_square).value("chi_bar_square", ReferenceLaw::chi_bar_square)
        .value("cone_projection", ReferenceLaw::cone_projection);
    nb::class_<LRResult>(m, "LRResult")
        .def_ro("status", &LRResult::status).def_ro("statistic", &LRResult::statistic)
        .def_ro("law", &LRResult::law).def_ro("interior_df", &LRResult::interior_df)
        .def_ro("weights", &LRResult::weights).def_ro("p_value", &LRResult::p_value)
        .def_ro("numerical_error", &LRResult::numerical_error).def_ro("message", &LRResult::message);
    nb::class_<LRWeights>(m, "LRWeights")
        .def_ro("status", &LRWeights::status).def_ro("weights", &LRWeights::weights)
        .def_ro("numerical_error", &LRWeights::numerical_error);
    m.def("fit", [](const Model& model, const ComparisonPlan& plan, const FitOptions& options) {
        nb::gil_scoped_release release;
        return fit(model, plan, options);
    }, "model"_a, "plan"_a, "options"_a = FitOptions());
    m.def("restrict_fit", [](const Model& model, const FitResult& fit, const Restriction& r, const ComparisonPlan& p) {
        nb::gil_scoped_release release;
        return restrict_fit(model, fit, r, p);
    }, "parent"_a, "parent_fit"_a, "restriction"_a, "plan"_a);
    m.def("restriction_test", [](const Model& parent, const Model& child, const Restriction& r,
                                 const FitResult& pf, const FitResult& cf, const Matrix& h,
                                 double level, double tolerance, std::size_t cells) {
        if (h.shape(0) != parent.dimension() || h.shape(1) != parent.dimension())
            throw std::invalid_argument("parent information has the wrong shape");
        nb::gil_scoped_release release;
        return restriction_test(parent, child, r, pf, cf, view(h), level, tolerance, cells);
    }, "parent"_a, "child"_a, "restriction"_a, "parent_fit"_a, "child_fit"_a, "parent_information"_a,
       "level"_a = 0.01, "tolerance"_a = 1e-6, "maximum_cells"_a = 32768);
    m.def("restriction_test", [](const Model& parent, const Model& child, const Restriction& r,
                                 const FitResult& pf, const FitResult& cf, const ComparisonPlan& plan,
                                 double level, double tolerance, std::size_t draws) {
        nb::gil_scoped_release release;
        return restriction_test(parent, child, r, pf, cf, plan, level, tolerance, draws);
    }, "parent"_a, "child"_a, "restriction"_a, "parent_fit"_a, "child_fit"_a, "plan"_a,
       "level"_a = 0.01, "tolerance"_a = 1e-6, "maximum_cells"_a = 32768);
    m.def("cone_projection_test", [](const Matrix& information, const Matrix& normals, const Matrix& equalities,
                                     double statistic, double level, std::size_t draws) {
        const auto d = information.shape(0);
        if (information.shape(1) != d || normals.shape(1) != d || equalities.shape(1) != d)
            throw std::invalid_argument("cone matrices have inconsistent dimensions");
        nb::gil_scoped_release release;
        return cone_projection_test(view(information), d, view(normals), view(equalities), statistic, level, draws);
    }, "information"_a, "normals"_a, "equalities"_a, "statistic"_a,
       "level"_a = 0.01, "draws"_a = 32768);
    m.def("chi_bar_weights", [](const Matrix& h, double tolerance, std::size_t cells) {
        if (h.shape(0) != h.shape(1)) throw std::invalid_argument("information must be square");
        nb::gil_scoped_release release;
        return chi_bar_weights(view(h), h.shape(0), tolerance, cells);
    }, "information"_a, "tolerance"_a = 1e-6, "maximum_cells"_a = 32768);

    m.def("normal_survival", &normal_survival, "x"_a);
    m.def("student_survival", &student_survival, "x"_a, "df"_a);
    m.def("chi_square_survival", &chi_square_survival, "x"_a, "df"_a);
    m.def("coefficient_test", [](double coefficient, double se, std::size_t df) {
        return test_dictionary(coefficient_test(coefficient, se, df));
    }, "coefficient"_a, "standard_error"_a, "df"_a);
    m.def("portmanteau", [](const Vector& x, std::size_t lags, std::size_t df) {
        return portmanteau_dictionary(portmanteau(view(x), lags, df), lags);
    }, "series"_a, "lags"_a, "df_adjustment"_a = 0);
    m.def("dgt", [](const Vector& u, std::size_t cells) {
        nb::dict out = test_dictionary(dgt(view(u), cells));
        out["n_cells"] = cells;
        out["chi2_stat"] = out["statistic"];
        return out;
    }, "pit"_a, "cells"_a = 40);
    m.def("arch_lm", [](const Vector& z, std::size_t lags) {
        return test_dictionary(arch_lm(view(z), lags));
    }, "standardized"_a, "lags"_a = 10);
    m.def("sign_bias", [](const Vector& z, const Vector& e) { return sign_bias_dictionary(sign_bias(view(z), view(e))); },
          "standardized"_a, "residual"_a);
    m.def("berkowitz", [](const Vector& u) { return berkowitz_dictionary(berkowitz(view(u))); }, "pit"_a);
    m.def("li_mak", [](BoundJet& jet, const Vector& theta, std::size_t lags, Estimator estimator, bool estimated) {
        return li_mak_dictionary(li_mak(jet.jet, view(theta), lags, estimator, estimated), lags, jet.model->dimension());
    }, "jet"_a, "theta"_a, "lags"_a = 10, "estimator"_a = Estimator::mle, "estimated"_a = true);
    m.def("absolute_li_mak", [](BoundJet& jet, const Vector& theta, std::size_t lags, Estimator estimator, bool estimated) {
        return li_mak_dictionary(absolute_li_mak(jet.jet, view(theta), lags, estimator, estimated), lags, jet.model->dimension());
    }, "jet"_a, "theta"_a, "lags"_a = 10, "estimator"_a = Estimator::mle, "estimated"_a = true);
    m.def("absolute_sign_bias", [](BoundJet& jet, const Vector& theta, Estimator estimator, bool estimated) {
        return test_dictionary(absolute_sign_bias(jet.jet, view(theta), estimator, estimated));
    }, "jet"_a, "theta"_a, "estimator"_a = Estimator::mle, "estimated"_a = true);
    m.def("diagnostics", [](BoundJet& jet, const Vector& theta, std::size_t cells, std::size_t lags,
                            const std::optional<std::size_t>& df, Estimator estimator, bool estimated, double alpha) {
        if (!(alpha > 0.0 && alpha < 1.0)) throw std::invalid_argument("alpha must be between zero and one");
        const std::size_t adjustment = df.value_or(estimated ? jet.model->residual_degrees_of_freedom() : 0);
        Diagnostics tests = diagnostics(jet.jet, view(theta), cells, lags, adjustment, estimator, estimated, df.value_or(0));
        nb::dict out, moments, cell = test_dictionary(tests.cell, alpha);
        cell["n_cells"] = cells;
        cell["chi2_stat"] = tests.cell.statistic;
        out["dgt"] = cell;
        for (std::size_t j = 0; j < 4; ++j) moments[nb::int_(j + 1)] = portmanteau_dictionary(std::move(tests.pit_moments[j]), lags, alpha);
        out["ljung_box"] = moments;
        out["residual_ljung_box"] = portmanteau_dictionary(std::move(tests.residual), lags, alpha);
        out["pit_normal_ljung_box"] = portmanteau_dictionary(std::move(tests.pit_normal), lags, alpha);
        out["li_mak"] = li_mak_dictionary(std::move(tests.squared_residual), lags, jet.model->dimension(), alpha);
        out["sign_bias"] = sign_bias_dictionary(tests.asymmetry, alpha);
        out["berkowitz"] = berkowitz_dictionary(tests.density, alpha);
        out["pit"] = owned_vector(std::move(tests.pit));
        out["n_obs"] = jet.sample.scored();
        out["alpha"] = alpha;
        out["df_adjustment"] = adjustment;
        out["pit_df_adjustment"] = df.value_or(0);
        out["pit_calibration"] = "fixed_parameter_reference; in-sample parameter estimation is not calibrated";
        return out;
    }, "jet"_a, "theta"_a, "cells"_a = 40, "lags"_a = 10, "df_adjustment"_a = nb::none(),
       "estimator"_a = Estimator::mle, "estimated"_a = true, "alpha"_a = 0.05);
    m.def("selection_diagnostics", [](BoundJet& jet, const Vector& theta, double alpha, Estimator estimator) {
        Adequacy tests;
        {
            nb::gil_scoped_release release;
            tests = adequacy(jet.jet, view(theta), alpha, estimator);
        }
        nb::list checks;
        for (const auto& check : tests.checks) {
            nb::dict record = test_dictionary(check.test, tests.level);
            record["name"] = check.name;
            record["holm_reject"] = check.holm_reject;
            checks.append(record);
        }
        nb::dict out, encompassing;
        out["checks"] = checks;
        out["status"] = nb::cast(tests.status).attr("name");
        out["holm_level"] = tests.level;
        if (!tests.volatility_gum.empty()) encompassing["volatility_gum"] = tests.volatility_gum;
        out["encompassing"] = encompassing;
        return out;
    }, "jet"_a, "theta"_a, "alpha"_a = 0.01, "estimator"_a = Estimator::mle);
    m.def("predictive_losses", [](BoundJet& jet, const Vector& theta, const std::string& loss) {
        if (loss != "log_score" && loss != "qlike") throw std::invalid_argument("loss must be log_score or qlike");
        const std::size_t n = jet.sample.scored();
        std::vector<double> values(n), e(n), h(n), z(n);
        JetOutput out;
        out.terms = values;
        out.paths = {e, h, z};
        const auto result = jet.jet.theta(view(theta), loss == "log_score" ? terms : paths, out);
        if (result.status != Status::ok) throw std::domain_error("Predictive loss cannot be evaluated at these parameters");
        if (loss == "qlike")
            for (std::size_t i = 0; i < n; ++i) values[i] = std::log(h[i]) + e[i]*e[i]/h[i];
        return owned_vector(std::move(values));
    }, "jet"_a, "theta"_a, "loss"_a = "log_score");
    m.def("diebold_mariano", [](const Vector& a, const Vector& b, std::size_t horizon, int bandwidth) {
        const DieboldMariano test = diebold_mariano(view(a), view(b), horizon, bandwidth);
        nb::dict out = test_dictionary(test.test);
        out["long_run_variance"] = test.long_run_variance;
        out["bandwidth"] = test.bandwidth;
        return out;
    }, "loss_a"_a, "loss_b"_a, "horizon"_a = 1, "bandwidth"_a = -1);
    m.def("mcs_block_length", &mcs_block_length, "rows"_a, "requested"_a = -1);
    m.def("model_confidence_set", [](const Matrix& losses, const Matrix& uniforms, std::size_t block_length,
                                    double alpha, const std::string& statistic) {
        if (statistic != "max" && statistic != "R") throw std::invalid_argument("MCS statistic must be 'max' or 'R'");
        const auto test = model_confidence_set({losses.data(), losses.size()}, losses.shape(0), losses.shape(1),
                                               {uniforms.data(), uniforms.size()}, uniforms.shape(0), block_length, alpha,
                                               statistic == "max" ? MCSStatistic::maximum : MCSStatistic::range);
        nb::dict out;
        out["included"] = test.included;
        out["eliminated"] = test.eliminated;
        out["p_values"] = owned_vector(test.p_values);
        out["final_p_value"] = test.final_p_value;
        return out;
    }, "losses"_a, "uniforms"_a, "block_length"_a, "alpha"_a = 0.05, "statistic"_a = "max");
    m.def("information_criteria", [](double log_likelihood, std::size_t parameters, std::size_t rows) {
        const auto ic = information_criteria(log_likelihood, parameters, rows);
        return nb::make_tuple(ic.aic, ic.bic, ic.hqic);
    }, "log_likelihood"_a, "parameters"_a, "rows"_a);
    m.def("location_scale_quantiles", [](double mean, double variance, const InnovationDensity& law, const Vector& u) {
        return owned_vector(location_scale_quantiles(mean, variance, law, view(u)));
    });
}
