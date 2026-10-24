// Python binding for scivolapi: thin, checked, and free of mathematics.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <vector>

#include <scivolapi/scivolapi.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace scivolapi;

namespace {

using Vector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
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

// Owns a copy of the series so the borrowed Sample can never outlive its data.
struct BoundJet {
    std::shared_ptr<const Model> model;
    std::vector<double> y;
    Sample sample;
    Jet jet;

    BoundJet(std::shared_ptr<const Model> m, std::span<const double> data)
        : model(std::move(m)), y(data.begin(), data.end()), sample(model->prepare(y)), jet(*model, sample)
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

nb::dict evaluate(BoundJet& self, const Vector& point, unsigned request, bool in_z)
{
    const std::size_t d = self.model->dimension(), n = self.sample.size();
    std::vector<double> g((request & gradient) ? d : 0), h((request & hessian) ? d * d : 0);
    std::vector<double> e((request & paths) ? n : 0), v(e.size()), z(e.size());
    JetOutput out;
    out.gradient = g;
    out.hessian = h;
    out.paths = {e, v, z};
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
    return result;
}

}  // namespace

NB_MODULE(_scivolapi, m)
{
    m.doc() = "scivolapi: native volatility-model mathematics";

    nb::enum_<MeanKind>(m, "MeanKind").value("zero", MeanKind::zero);
    nb::enum_<VolatilityKind>(m, "VolatilityKind")
        .value("garch", VolatilityKind::garch)
        .value("gjr", VolatilityKind::gjr)
        .value("egarch", VolatilityKind::egarch)
        .value("aparch", VolatilityKind::aparch);
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

    nb::class_<ModelSpec>(m, "ModelSpec")
        .def(nb::init<>())
        .def_rw("mean", &ModelSpec::mean)
        .def_rw("volatility", &ModelSpec::volatility)
        .def_rw("p", &ModelSpec::p)
        .def_rw("q", &ModelSpec::q)
        .def_rw("density", &ModelSpec::density);

    nb::class_<Model>(m, "Model")
        .def("__init__", [](Model* self, const ModelSpec& spec) { new (self) Model(spec); }, "spec"_a)
        .def_prop_ro("dimension", &Model::dimension)
        .def_prop_ro("parameter_names", &Model::parameter_names)
        .def("feasible", [](const Model& m, const Vector& theta) { return m.feasible(view(theta)); })
        .def("bounds", [](const Model& m) {
            std::vector<double> lower(m.dimension()), upper(m.dimension());
            m.bounds(lower, upper);
            return nb::make_tuple(owned_vector(std::move(lower)), owned_vector(std::move(upper)));
        })
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
        .def("start", [](const Model& m, const Vector& y) {
            std::vector<double> data(view(y).begin(), view(y).end());
            return owned_vector(m.start(m.prepare(data)));
        });

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

    nb::class_<FitOptions>(m, "FitOptions")
        .def(nb::init<>())
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

    m.def("fit", [](const Model& model, const Vector& y, const FitOptions& options) {
              nb::gil_scoped_release release;
              return fit(model, view(y), options);
          },
          "model"_a, "y"_a, "options"_a = FitOptions());

    nb::class_<BoundJet>(m, "Jet")
        .def("__init__",
             [](BoundJet* self, const Model& model, const Vector& y) {
                 new (self) BoundJet(std::make_shared<const Model>(model), view(y));
             },
             "model"_a, "y"_a)
        .def_prop_ro("variance_seed", [](const BoundJet& j) { return j.sample.variance_seed; })
        .def("theta", [](BoundJet& j, const Vector& theta, unsigned request) { return evaluate(j, theta, request, false); },
             "theta"_a, "request"_a)
        .def("z", [](BoundJet& j, const Vector& z, unsigned request) { return evaluate(j, z, request, true); },
             "z"_a, "request"_a);
}
