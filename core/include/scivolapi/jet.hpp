#pragma once

#include <scivolapi/problem.hpp>

namespace scivolapi {

enum class NumericalIntent : std::uint32_t { model_default, validated_binary64, validated_extended };
struct NumericalAccuracy {
    NumericalIntent intent = NumericalIntent::model_default;
    std::optional<double> absolute_error_target;
};
// fixed_input describes a stage evaluation scheme, not a fitted estimate.
enum class Estimator : std::uint32_t { mle, two_step_qmle, fixed_input };
enum class Stage : std::uint32_t { joint, normal_volatility, conditional_shape, dependence };
enum class ObjectiveScope : std::uint32_t { full_data_likelihood, conditional_likelihood, stage_contribution };
enum class DerivativeTarget : std::uint32_t { full_theta, fixed_free_restriction, exact_profile };
enum class ParameterDisposition : std::uint32_t { free, fixed, profiled };
enum class Scaling : std::uint32_t { additive_nll, per_scored_observation };
struct NamedParameter { std::string_view name; double value; };
struct ObjectiveRestriction {
    // Exact ambient partition. Free-axis order defines coordinate and derivative order.
    std::span<const ParameterId> free_axes;
    std::span<const NamedParameter> fixed_values;
};
struct ObjectiveRequest {
    Estimator estimator = Estimator::mle;
    Stage stage = Stage::joint;
    DerivativeTarget target = DerivativeTarget::full_theta;
    Coordinates coordinates = Coordinates::theta;
    ChartId chart = 0;
    ObjectiveRestriction restriction;
    ProfilePolicy profile = ProfilePolicy::none;
    NumericalAccuracy numerical_accuracy;
};
struct ObjectiveKey {
    ProblemId problem;
    Estimator estimator;
    Stage stage;
    ObjectiveScope scope;
    DerivativeTarget target;
    LayoutId derivative_layout;
    std::uint64_t restriction_id, profile_policy_id, stage_input_id, accuracy_intent_id;
    ChartId chart;
    Scaling scaling;
    bool operator==(const ObjectiveKey&) const = default;
};
struct DerivativeLayout {
    LayoutId id;
    DerivativeTarget target;
    std::span<const ParameterId> axes;
    std::span<const ParameterDisposition> full_parameter_partition;
};
class ObjectiveView {
public:
    const ObjectiveKey& key() const noexcept;
    const DerivativeLayout& derivative_layout() const noexcept;
    const PreparedProblem& problem() const noexcept;
private:
    ObjectiveView() noexcept = default;
    std::shared_ptr<const detail::ObjectiveState> state_;
    friend struct detail::Access;
};
Outcome<ObjectiveView> resolve_objective(const PreparedProblem&, ObjectiveRequest = {});
// Affine restrictions support theta coordinates only; restricted original charts
// and restriction/profile combinations remain unsupported pending semantic review.

enum class Feasibility : std::uint32_t { feasible, infeasible, not_checked };
enum class NumericalSupport : std::uint32_t { supported, not_representable, unavailable };
enum class Differentiability : std::uint32_t { smooth, genuine_kink, unavailable };
struct StatisticalParameterCounts {
    // Full statistical axes partition, distinct from derivative dimension and
    // any later inference/IC regularity qualification.
    std::uint64_t total, free, fixed, profiled;
};
struct ProfileReconstruction {
    ProfilePolicy policy;
    // Parallel views in full-theta axis identity; validity concerns values,
    // not F/G/H derivatives. These views borrow the associated point's owners.
    std::span<const ParameterId> nuisance_axes;
    std::span<const double> values;
    std::span<const Status> validity;
    std::uint64_t n_eff;
    StatisticalParameterCounts statistical_parameters;
};
struct PointEvidence {
    std::uint64_t point_id;
    ObjectiveKey objective;
    LayoutId full_theta_layout;
    Feasibility theoretical_feasibility;
    NumericalSupport numerical_support;
    Differentiability differentiability;
    std::uint32_t valid_mask;
    std::optional<double> additive_nll, attained_absolute_error_bound;
    std::optional<ProfileReconstruction> profile_reconstruction;
};
class PointView {
public:
    const ObjectiveKey& objective() const noexcept;
    const DerivativeLayout& derivative_layout() const noexcept;
    std::span<const double> theta() const noexcept;
    std::span<const double> original_coordinates() const noexcept;
    const PointEvidence& evidence() const noexcept;
private:
    PointView() noexcept = default;
    const detail::PointState* state_ = nullptr;
    const detail::WorkspaceState* workspace_ = nullptr;
    std::uint64_t generation_ = 0;
    friend struct detail::Access;
};
class Point {
public:
    PointView view() const noexcept;
    const ObjectiveKey& objective() const noexcept;
    const DerivativeLayout& derivative_layout() const noexcept;
    std::span<const double> theta() const noexcept;
    std::span<const double> original_coordinates() const noexcept;
    const PointEvidence& evidence() const noexcept;
private:
    Point() noexcept = default;
    std::shared_ptr<const detail::PointState> state_;
    friend struct detail::Access;
};
// Profile inputs contain the reduced mean axes only. Returned theta is always
// the complete statistical point, including the reconstructed nuisance sigma2.
// Original profile coordinates contain only the reduced axes, never a sigma2 z.
// Full/restricted theta inputs contain all ambient parameters, including matching
// fixed entries. Restricted coordinates contain only the ordered free theta axes.
Outcome<Point> from_theta(const ObjectiveView&, std::span<const NamedParameter>);
Outcome<Point> from_coordinates(const ObjectiveView&, std::span<const double>);

struct Requirements {
    std::uint64_t bytes = 0, alignment = 0;
    ProblemId problem = 0;
    LayoutId derivative_layout = 0;
    std::uint64_t work_plan_id = 0;
private:
    std::shared_ptr<const detail::ObjectiveState> owner_;
    JetRequest maximum_request_ = JetRequest::F;
    friend struct detail::Access;
};
class Workspace {
public:
    Workspace(Workspace&&) noexcept;
    Workspace& operator=(Workspace&&) noexcept;
    ~Workspace();
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
private:
    Workspace() noexcept;
    std::unique_ptr<detail::WorkspaceState> state_;
    friend struct detail::Access;
};
Outcome<Requirements> requirements(const ObjectiveView&, JetRequest maximum_request);
Outcome<Workspace> reserve(Requirements);
Outcome<Workspace> reserve(const ObjectiveView&, JetRequest maximum_request);
// Caller storage is an exclusive lease until Workspace destruction. Do not mutate
// it outside the point factories/evaluator, including retained theta/original z.
Outcome<Workspace> bind_workspace(Requirements, std::span<std::byte>);
struct PointBinding { Status status; std::optional<PointView> point; };
// Positional inputs follow derivative_layout().axes for exact profiles; ordinary
// full-theta objectives retain their complete model order. Rebinding expires views.
// Restrictions require complete ambient theta for from_theta, ordered free axes
// for from_coordinates, and always return the complete statistical theta point.
PointBinding from_theta(const ObjectiveView&, std::span<const double>, Workspace&) noexcept;
PointBinding from_coordinates(const ObjectiveView&, std::span<const double>, Workspace&) noexcept;

struct JetBuffers {
    LayoutId derivative_layout;
    std::span<double> value, gradient;
    DenseSymmetricView hessian;
};
class JetStorage {
public:
    JetStorage(JetStorage&&) noexcept;
    JetStorage& operator=(JetStorage&&) noexcept;
    ~JetStorage();
    JetStorage(const JetStorage&) = delete;
    JetStorage& operator=(const JetStorage&) = delete;
    JetBuffers outputs() noexcept;
    double value() const noexcept;
    std::span<const double> gradient() const noexcept;
    std::span<const double> hessian() const noexcept;
private:
    JetStorage() noexcept;
    std::unique_ptr<detail::JetStorageState> state_;
    friend struct detail::Access;
};
Outcome<JetStorage> allocate_jet_storage(const ObjectiveView&, JetRequest maximum_request);
JetValidity evaluate(const ObjectiveView&, PointView, JetRequest, Workspace&, JetBuffers) noexcept;

} // namespace scivolapi
