#pragma once

#include <scivolapi/detail/types.hpp>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace scivolapi {

struct ZeroMean {};
struct ARMA { std::uint64_t p_ar, q_ma; bool intercept = true; };
struct LinearMean { bool intercept = true; };
struct ARX { std::uint64_t lags = 0; bool intercept = true; };
struct HARX { std::vector<std::uint64_t> horizons; bool intercept = true; };
struct ConstantVariance {};
struct GARCH { std::uint64_t p_vol, q_vol; };
struct GJRGARCH { std::uint64_t p_vol, q_vol; };
struct EGARCH { std::uint64_t p_vol, q_vol; };
struct APARCH { std::uint64_t p_vol, q_vol; };
struct Normal {};
struct StudentT {};
struct SkewT {};
struct GED {};
struct GeneralizedHyperbolic {};
struct NIG {};
struct Hyperbolic {};
struct VarianceGamma {};
enum class DensityOrigin : std::uint32_t { unspecified, implicit_normal, explicit_component };
struct ModelSpec {
    std::optional<std::variant<ZeroMean, ARMA, LinearMean, ARX, HARX>> mean;
    std::optional<std::variant<ConstantVariance, GARCH, GJRGARCH, EGARCH, APARCH>> volatility;
    std::optional<std::variant<Normal, StudentT, SkewT, GED, GeneralizedHyperbolic,
                               NIG, Hyperbolic, VarianceGamma>> density;
    DensityOrigin density_origin = DensityOrigin::unspecified;
};
// Caller exogenous width only, excluding generated ARX/HARX columns and intercept.
struct InputSchema { std::uint64_t design_width = 0; };
enum class FixedResidualLaw : std::uint32_t {
    normal_density_ratio, studentt_density_ratio
};
struct FixedInputDCCSpec {
    std::uint64_t p_corr = 1, q_corr = 1;
    FixedResidualLaw law = FixedResidualLaw::normal_density_ratio;
};
struct DependenceInputSchema { std::uint64_t series = 0; };
using ModelSpecification = std::variant<ModelSpec, FixedInputDCCSpec>;
struct ParameterEntry { ParameterId id; std::string_view name; std::uint64_t index; };
struct ParameterBlock { std::string_view name; std::uint64_t offset, width; };
struct ParameterLayout {
    LayoutId id;
    std::span<const ParameterEntry> entries;
    std::span<const ParameterBlock> blocks;
    std::uint64_t full_theta_width;
};
enum class DispatchKind : std::uint32_t { runtime, fixed, unavailable };
struct JetCapabilities {
    std::string_view registration_id;
    std::uint32_t request_mask;
    bool original_chart;
    DispatchKind theta_dispatch, chart_dispatch;
};

class Model {
public:
    const ModelSpecification& spec() const noexcept;
    const ParameterLayout& layout() const noexcept;
    const JetCapabilities& capabilities() const noexcept;
private:
    Model() noexcept = default;
    std::shared_ptr<const detail::ModelState> state_;
    friend struct detail::Access;
};
Outcome<Model> compile(const ModelSpec&, const InputSchema& = {});
Outcome<Model> compile(const FixedInputDCCSpec&, const DependenceInputSchema&);

} // namespace scivolapi
