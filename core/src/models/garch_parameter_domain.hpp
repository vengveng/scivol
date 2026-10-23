// SCIVOL-NATIVE: CORE
#ifndef SCIVOL_EXPERIMENT_GARCH_PARAMETER_DOMAIN_HPP
#define SCIVOL_EXPERIMENT_GARCH_PARAMETER_DOMAIN_HPP

namespace vlk::experiment::domain {

// The stationary chart uses the exact unit open simplex, with no padded target.
inline constexpr double stationary_chart_limit = 1.0;

// Mathematical lower endpoints, not shifted representability buffers.
inline constexpr double omega_chart_lower = 0.0;
inline constexpr double student_t_chart_lower = 2.0;
inline constexpr double ged_chart_lower = 0.0;

} // namespace vlk::experiment::domain

#endif
