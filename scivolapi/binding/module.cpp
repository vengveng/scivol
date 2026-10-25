// Python binding for scivolapi: thin, checked, and free of mathematics.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include <scivolapi/scivolapi.hpp>

#include "../src/density/distribution.hpp"

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
    Sample sample;
    Jet jet;

    BoundJet(std::shared_ptr<const Model> m, std::span<const double> data, std::vector<double> regressors)
        : model(std::move(m)), y(data.begin(), data.end()), x(std::move(regressors)), sample(model->prepare(y, x)),
          jet(*model, sample)
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
                        std::vector<double> target)
        : model(std::move(m)), eps(residuals.data(), residuals.data() + residuals.shape(0) * residuals.shape(1)),
          sample(model->prepare(eps, residuals.shape(1), target)), jet(*model, sample)
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
                             std::size_t, int>;

SpecState spec_state(const ModelSpec& s)
{
    return {s.mean.constant, s.mean.ar, s.mean.har, s.mean.exog, s.mean.ma, static_cast<int>(s.volatility), s.p, s.q,
            static_cast<int>(s.density)};
}

ModelSpec spec_from(const SpecState& t)
{
    ModelSpec s;
    std::tie(s.mean.constant, s.mean.ar, s.mean.har, s.mean.exog, s.mean.ma, std::ignore, s.p, s.q, std::ignore) = t;
    s.volatility = static_cast<VolatilityKind>(std::get<5>(t));
    s.density = static_cast<DensityKind>(std::get<8>(t));
    return s;
}

nb::dict evaluate(BoundJet& self, const Vector& point, unsigned request, bool in_z)
{
    const std::size_t d = self.model->dimension(), n = self.sample.scored();
    std::vector<double> g((request & gradient) ? d : 0), h((request & hessian) ? d * d : 0);
    std::vector<double> e((request & paths) ? n : 0), v(e.size()), z(e.size()), s((request & scores) ? n * d : 0),
        terms_((request & terms) ? n : 0);
    JetOutput out;
    out.gradient = g;
    out.hessian = h;
    out.paths = {e, v, z};
    out.scores = s;
    out.terms = terms_;
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
    return result;
}

}  // namespace

NB_MODULE(_scivolapi, m)
{
    m.doc() = "scivolapi: native volatility-model mathematics";

    nb::enum_<VolatilityKind>(m, "VolatilityKind")
        .value("garch", VolatilityKind::garch)
        .value("gjr", VolatilityKind::gjr)
        .value("egarch", VolatilityKind::egarch)
        .value("aparch", VolatilityKind::aparch)
        .value("constant", VolatilityKind::constant);
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
        .def_rw("mean", &ModelSpec::mean)
        .def_rw("volatility", &ModelSpec::volatility)
        .def_rw("p", &ModelSpec::p)
        .def_rw("q", &ModelSpec::q)
        .def_rw("density", &ModelSpec::density);

    nb::class_<Model>(m, "Model")
        .def("__init__", [](Model* self, const ModelSpec& spec) { new (self) Model(spec); }, "spec"_a)
        .def("__getstate__", [](const Model& m) { return spec_state(m.spec()); })
        .def("__setstate__", [](Model& self, const SpecState& state) { new (&self) Model(spec_from(state)); })
        .def_prop_ro("dimension", &Model::dimension)
        .def_prop_ro("parameter_names", &Model::parameter_names)
        .def("feasible", [](const Model& m, const Vector& theta) { return m.feasible(view(theta)); })
        .def("bounds", [](const Model& m) {
            std::vector<double> lower(m.dimension()), upper(m.dimension());
            m.bounds(lower, upper);
            return nb::make_tuple(owned_vector(std::move(lower)), owned_vector(std::move(upper)));
        })
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
        .value("failed", Termination::failed);
    nb::enum_<Inference>(m, "Inference")
        .value("available", Inference::available)
        .value("not_feasible", Inference::not_feasible)
        .value("not_stationary", Inference::not_stationary)
        .value("at_boundary", Inference::at_boundary)
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
                const std::optional<Matrix>& target) {
                 new (self) BoundCorrelationJet(std::make_shared<const CorrelationModel>(model), eps, square(target));
             },
             "model"_a, "eps"_a, "target"_a = nb::none())
        .def_prop_ro("target", [](const BoundCorrelationJet& j) {
            return owned_matrix(j.sample.target, j.sample.series, j.sample.series);
        })
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
    nb::class_<MarginalDensity>(m, "MarginalDensity")
        .def("__getstate__", [](const MarginalDensity& d) { return std::make_tuple(static_cast<int>(d.density), d.shape); })
        .def("__setstate__", [](MarginalDensity& self, const std::tuple<int, std::vector<double>>& state) {
            new (&self) MarginalDensity{static_cast<DensityKind>(std::get<0>(state)), std::get<1>(state)};
        })
        .def("__init__", [](MarginalDensity* self, DensityKind density, std::vector<double> shape) {
            new (self) MarginalDensity{density, std::move(shape)};
        }, "density"_a, "shape"_a = std::vector<double>{})
        .def_rw("density", &MarginalDensity::density)
        .def_rw("shape", &MarginalDensity::shape);
    m.def("simulate_correlation",
          [](const CorrelationModel& model, const Vector& theta, const Matrix& eps,
             const std::vector<MarginalDensity>& marginals, const Tensor3& normals, const std::optional<Matrix>& chi2,
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
             const std::optional<Matrix>& target) {
              const std::vector<double> residuals(eps.data(), eps.data() + eps.shape(0) * eps.shape(1));
              const std::vector<double> qbar = square(target);
              const std::size_t series = eps.shape(1);
              nb::gil_scoped_release release;
              return fit(model, residuals, series, qbar, options);
          },
          "model"_a, "eps"_a, "options"_a = FitOptions(), "target"_a = nb::none());

    nb::class_<BoundJet>(m, "Jet")
        .def("__init__",
             [](BoundJet* self, const Model& model, const Vector& y, const std::optional<Matrix>& x) {
                 new (self) BoundJet(std::make_shared<const Model>(model), view(y), exogenous(x));
             },
             "model"_a, "y"_a, "x"_a = nb::none())
        .def_prop_ro("first_scored", [](const BoundJet& j) { return j.sample.first_scored; })
        .def_prop_ro("variance_seed", [](const BoundJet& j) { return j.sample.variance_seed; })
        .def("theta", [](BoundJet& j, const Vector& theta, unsigned request) { return evaluate(j, theta, request, false); },
             "theta"_a, "request"_a)
        .def("z", [](BoundJet& j, const Vector& z, unsigned request) { return evaluate(j, z, request, true); },
             "z"_a, "request"_a);
}
