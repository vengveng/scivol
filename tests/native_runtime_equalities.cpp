// TEST_ONLY: shared SQP rejection, equalities, workspaces and constraint composition.
#include "runtime_kraft_slsqp.hpp"
#include "diagonal_optimizer_scaling.hpp"
#include "affine_optimizer_constraints.hpp"
#include "mean_solver.hpp"
#include "restricted_optimizer_coordinates.hpp"
#include <cassert>
#include <vector>

using namespace vlk::experiment;

struct Quadratic {
    std::size_t width;
    std::size_t dimension() const noexcept { return width; }
    bool evaluate(kraft::ObjectiveRequest request,const double* x,double* f,
                  double* g,bool) const noexcept {
        if (request!=kraft::ObjectiveRequest::gradient) *f=0.0;
        for (std::size_t j=0;j<width;++j) {
            if (request!=kraft::ObjectiveRequest::gradient) *f+=0.5*x[j]*x[j];
            if (request!=kraft::ObjectiveRequest::value) g[j]=x[j];
        }
        return true;
    }
};

struct Restrictions {
    std::size_t width,rows,eq;
    bool identity=false;
    double identity_scale=1.0;
    std::size_t dimension() const noexcept { return width; }
    std::size_t count() const noexcept { return rows; }
    std::size_t equalities() const noexcept { return eq; }
    bool evaluate(kraft::ConstraintRequest request,const double* x,double* values,
                  double* jacobian) const noexcept {
        if (request!=kraft::ConstraintRequest::value)
            std::fill_n(jacobian,rows*width,0.0);
        for (std::size_t i=0;i<rows;++i) {
            if (identity) {
                if (request!=kraft::ConstraintRequest::jacobian) values[i]=x[i]-(i+1)*identity_scale;
                if (request!=kraft::ConstraintRequest::value) jacobian[i+i*rows]=1.0;
            } else if (i==0 && eq) {
                if (request!=kraft::ConstraintRequest::jacobian) {
                    values[i]=-1.0;
                    for (std::size_t j=0;j<width;++j) values[i]+=x[j];
                }
                if (request!=kraft::ConstraintRequest::value)
                    for (std::size_t j=0;j<width;++j) jacobian[i+j*rows]=1.0;
            } else {
                if (request!=kraft::ConstraintRequest::jacobian) values[i]=x[0]-0.8;
                if (request!=kraft::ConstraintRequest::value) jacobian[i]=1.0;
            }
        }
        return true;
    }
};

// Older inequality-only adapters need not acquire a new member.
struct InequalitiesOnly {
    Restrictions inner;
    std::size_t dimension() const noexcept { return inner.dimension(); }
    std::size_t count() const noexcept { return inner.count(); }
    bool evaluate(kraft::ConstraintRequest request,const double* x,double* c,double* a) const noexcept
    { return inner.evaluate(request,x,c,a); }
};

template<class Constraints>
void verify(const Constraints& constraints,const std::vector<double>& expected) {
    for (bool reset : {false, true}) {
    const auto n=constraints.dimension(),m=constraints.count();
    const auto eq=runtime::runtime_kraft_detail::equality_count(constraints);
    const auto bytes=runtime::runtime_kraft_workspace_bytes(n,m,eq);
    assert(bytes);
    std::vector<std::byte> memory(bytes+64,std::byte{0x5a});
    runtime::RuntimeKraftWorkspace workspace;
    assert(!runtime::bind_runtime_kraft_workspace(n,m,memory.data(),bytes-1,workspace,eq));
    assert(runtime::bind_runtime_kraft_workspace(n,m,memory.data(),bytes,workspace,eq));
    std::vector<double> x(n,0.0),lower(n,NAN),upper(n,NAN);
    const auto result=runtime::minimize_runtime_kraft(Quadratic{n},constraints,x.data(),
        {lower.data(),upper.data()},kraft::Options{1e-12,100,reset},workspace);
    assert(result.mode==0);
    assert(result.constraint_curvature_resets==0);
    assert(result.maximum_violation<1e-10);
    for (std::size_t j=0;j<n;++j) assert(std::abs(x[j]-expected[j])<1e-9);
    for (std::size_t j=bytes;j<memory.size();++j) assert(memory[j]==std::byte{0x5a});
    if (eq) {
        for (double sign : {-1.0,1.0}) {
            workspace.constraint_values[0]=sign*2;
            assert(runtime::runtime_kraft_detail::maximum_violation(
                x.data(),n,m,{lower.data(),upper.data()},workspace,eq)>=2);
        }
    }
    }
}

struct RejectedTrial {
    int failure;
    bool recover, speculate, gradient_only;
    mutable std::size_t values=0,gradients=0;
    mutable std::size_t rejected_values=0;
    mutable double first_rejected[2]{};
    std::size_t dimension() const noexcept { return 1; }
    bool speculate_trial_gradient(std::int64_t) const noexcept { return speculate; }
    bool evaluate(kraft::ObjectiveRequest request, const double* x, double* f,
                  double* g, bool) const noexcept {
        const bool outside = recover ? x[0] <= 0 : x[0] > 0;
        const bool wants_value = request != kraft::ObjectiveRequest::gradient;
        const bool wants_gradient = request != kraft::ObjectiveRequest::value;
        values+=wants_value;gradients+=wants_gradient;
        const double curvature = recover ? 1e6 : 1;
        if (wants_value) *f = 0.5 * curvature * (x[0] - 1) * (x[0] - 1);
        if (wants_gradient) g[0] = curvature * (x[0] - 1);
        if (outside && (gradient_only ? wants_gradient : wants_value)) {
            if (wants_value && rejected_values++ < 2)
                first_rejected[rejected_values-1] = x[0];
            if (failure == 0) return false;
            if (gradient_only) g[0] = failure == 1 ? NAN : INFINITY;
            else *f = failure == 1 ? NAN : INFINITY;
        }
        return true;
    }
};

struct FixedRejectedTrial {
    static constexpr std::size_t dimension = 1;
    RejectedTrial inner;
    template<kraft::ObjectiveRequest Request>
    bool evaluate(const double* x, double* f, double* g, bool same = false) const noexcept
    { return inner.evaluate(Request, x, f, g, same); }
};

void verify_value_only_checkpoint() {
    const auto bytes=runtime::runtime_kraft_workspace_bytes(1,0);
    std::vector<std::byte> memory(bytes);
    runtime::RuntimeKraftWorkspace workspace;
    assert(runtime::bind_runtime_kraft_workspace(1,0,memory.data(),bytes,workspace));
    double x=0,lower=NAN,upper=NAN;
    const RejectedTrial objective{0,false,false,true};
    const auto result=runtime::minimize_runtime_kraft(
        objective,Restrictions{1,0,0},&x,
        {&lower,&upper},kraft::Options{1e-12,100},workspace);
    assert(result.function_evaluations==objective.values);
    assert(result.gradient_evaluations==objective.gradients);
    assert(result.mode!=0);
    assert(x==1 && result.value==0);
    assert(!result.derivatives_current && std::isnan(workspace.gradient[0]));
    assert(result.retained_feasible_trial);
    for (std::size_t i=0;i<workspace.multiplier_count;++i)
        assert(std::isnan(workspace.multipliers[i]));

    for (double scale : {0.5,1.0,2.0}) {
        double scratch;
        x=0;
        const auto scaled=runtime::minimize_runtime_kraft(
            runtime::DiagonalObjective{RejectedTrial{0,false,false,true},&scale,&scratch},
            runtime::DiagonalConstraints{Restrictions{1,0,0},&scale,&scratch},&x,
            {&lower,&upper},kraft::Options{1e-12,100},workspace);
        assert(scaled.mode!=0 && scaled.retained_feasible_trial);
        assert(x>0 && scaled.value<0.5);
        assert(std::abs(scaled.value-0.5*(scale*x-1)*(scale*x-1))<1e-14);
        assert(!scaled.derivatives_current && std::isnan(workspace.gradient[0]));
    }
}

struct TwoWellObjective {
    bool missing_deeper_gradient=false;
    mutable double best_value=INFINITY,best_point=NAN;
    mutable std::size_t values=0,gradients=0;
    std::size_t dimension() const noexcept { return 1; }
    bool evaluate(kraft::ObjectiveRequest request,const double* x,double* f,
                  double* g,bool) const noexcept {
        constexpr double a=0.01,b=0.5049,c=1.0,scale=1.0/(a*b*c);
        const double t=x[0];
        if (request!=kraft::ObjectiveRequest::gradient) {
            ++values;
            *f=scale*t*(t*(t*(t/4-(a+b+c)/3)+(a*b+a*c+b*c)/2)-a*b*c);
            if (*f<best_value) { best_value=*f; best_point=t; }
        }
        if (request!=kraft::ObjectiveRequest::value) {
            ++gradients;
            *g=scale*(t-a)*(t-b)*(t-c);
            if (missing_deeper_gradient && std::abs(t-1.0)<1e-12) { *g=NAN; return false; }
        }
        return true;
    }
};

void verify_success_does_not_discard_a_better_trial() {
    const auto bytes=runtime::runtime_kraft_workspace_bytes(1,0);
    std::vector<std::byte> memory(bytes);
    runtime::RuntimeKraftWorkspace workspace;
    assert(runtime::bind_runtime_kraft_workspace(1,0,memory.data(),bytes,workspace));
    for (bool missing_gradient : {false,true}) {
    double x=0,lower=NAN,upper=NAN;
    const TwoWellObjective objective{missing_gradient};
    const auto result=runtime::minimize_runtime_kraft_phase(
        objective,Restrictions{1,0,0},&x,
        {&lower,&upper},kraft::Options{1e-12,100},workspace);
    // The first trial visits the deeper well but fails sufficient decrease.
    assert(std::abs(objective.best_point-1.0)<1e-12);
    assert(result.value==objective.best_value && x==objective.best_point);
    assert(result.retained_feasible_trial && result.mode==kraft::retained_point_mode);
    assert(result.derivatives_current==!missing_gradient);
    assert(missing_gradient ? std::isnan(workspace.gradient[0])
                            : std::abs(workspace.gradient[0])<1e-11);
    assert(result.maximum_violation==0.0);
    assert(result.function_evaluations==objective.values);
    assert(result.gradient_evaluations==objective.gradients);
    for (std::size_t i=0;i<workspace.multiplier_count;++i)
        assert(std::isnan(workspace.multipliers[i]));
    x=0;
    const TwoWellObjective resumed_objective{missing_gradient};
    const auto resumed=runtime::minimize_runtime_kraft(
        resumed_objective,Restrictions{1,0,0},&x,
        {&lower,&upper},kraft::Options{1e-12,100},workspace);
    assert(x==resumed_objective.best_point && resumed.value==objective.best_value);
    assert(resumed.mode==(missing_gradient?kraft::retained_point_mode:0));
    assert(resumed.checkpoint_restarts==(missing_gradient?0u:1u));
    assert(resumed.iterations>=result.iterations && resumed.iterations<=result.iterations+1);
    assert(resumed.function_evaluations==resumed_objective.values);
    assert(resumed.gradient_evaluations==resumed_objective.gradients);
    assert(resumed.retained_feasible_trial);
    x=0;
    const TwoWellObjective budgeted_objective{missing_gradient};
    const auto budgeted=runtime::minimize_runtime_kraft(
        budgeted_objective,Restrictions{1,0,0},&x,
        {&lower,&upper},kraft::Options{1e-12,result.iterations},workspace);
    assert(budgeted.iterations<=result.iterations && budgeted.checkpoint_restarts==0);
    assert(budgeted.mode!=0 && budgeted.retained_feasible_trial);
    assert(x==budgeted_objective.best_point && budgeted.value==objective.best_value);
    }
}

struct CheckpointConstraint {
    bool failed_jacobian;
    mutable std::size_t values=0,jacobians=0;
    std::size_t dimension() const noexcept { return 1; }
    std::size_t count() const noexcept { return 1; }
    bool evaluate(kraft::ConstraintRequest request,const double* x,double* c,
                  double* jac) const noexcept {
        values+=request!=kraft::ConstraintRequest::jacobian;
        jacobians+=request!=kraft::ConstraintRequest::value;
        if (request!=kraft::ConstraintRequest::jacobian)
            *c=failed_jacobian?1:-x[0]*x[0];
        if (request!=kraft::ConstraintRequest::value) {
            *jac=failed_jacobian?0:-2*x[0];
            if (failed_jacobian && x[0]==0) {
                *jac=NAN;
                return false;
            }
        }
        return true;
    }
};

struct RejectedJacobianConstraint {
    std::size_t dimension() const noexcept { return 1; }
    std::size_t count() const noexcept { return 1; }
    bool evaluate(kraft::ConstraintRequest request,const double* x,double* c,
                  double* jac) const noexcept {
        if (request!=kraft::ConstraintRequest::jacobian) *c=-(x[0]-1)*(x[0]-1);
        if (request!=kraft::ConstraintRequest::value) {
            *jac=x[0]==1?0:NAN;
            return x[0]==1;
        }
        return true;
    }
};

void verify_checkpoint_constraint_evidence() {
    const auto bytes=runtime::runtime_kraft_workspace_bytes(1,1);
    std::vector<std::byte> memory(bytes+64,std::byte{0x5a});
    runtime::RuntimeKraftWorkspace workspace;
    assert(runtime::bind_runtime_kraft_workspace(1,1,memory.data(),bytes,workspace));
    double x=0,lower=NAN,upper=NAN;
    const auto infeasible=runtime::minimize_runtime_kraft(
        RejectedTrial{0,false,false,true},CheckpointConstraint{false},&x,
        {&lower,&upper},kraft::Options{1e-12,100},workspace);
    assert(infeasible.mode!=0 && x==0 && infeasible.value==0.5);
    assert(!infeasible.retained_feasible_trial && infeasible.derivatives_current);

    x=1;
    const CheckpointConstraint constraint{true};
    const auto missing_jac=runtime::minimize_runtime_kraft(
        Quadratic{1},constraint,&x,
        {&lower,&upper},kraft::Options{1e-12,100},workspace);
    assert(missing_jac.constraint_evaluations==constraint.values);
    assert(missing_jac.jacobian_evaluations==constraint.jacobians);
    assert(missing_jac.mode==-102 && x==0 && missing_jac.value==0);
    assert(missing_jac.retained_feasible_trial && !missing_jac.derivatives_current);
    assert(workspace.gradient[0]==0 && std::isnan(workspace.jacobian[0]));
    assert(workspace.constraint_values[0]==1 && missing_jac.maximum_violation==0);

    const CheckpointConstraint initial_constraint{true};
    const auto failed_initial=runtime::minimize_runtime_kraft(
        Quadratic{1},initial_constraint,&x,
        {&lower,&upper},kraft::Options{1e-12,100},workspace);
    assert(failed_initial.mode==-102);
    assert(failed_initial.constraint_evaluations==initial_constraint.values);
    assert(failed_initial.jacobian_evaluations==initial_constraint.jacobians);

    x=1;
    const auto rejected_jac=runtime::minimize_runtime_kraft(
        Quadratic{1},RejectedJacobianConstraint{},&x,
        {&lower,&upper},kraft::Options{1e-12,100},workspace);
    assert(rejected_jac.mode==-102 && x==1 && rejected_jac.value==0.5);
    assert(rejected_jac.derivatives_current && workspace.gradient[0]==1);
    assert(workspace.jacobian[0]==0 && rejected_jac.maximum_violation==0);
    for (std::size_t i=bytes;i<memory.size();++i) assert(memory[i]==std::byte{0x5a});
}

struct CancelAfterTrial {
    mutable unsigned polls = 0;
    kraft::StopReason poll() const noexcept {
        return ++polls == 3 ? kraft::StopReason::cancelled : kraft::StopReason::none;
    }
};

void verify_rejected_trials() {
    const auto bytes = runtime::runtime_kraft_workspace_bytes(1, 0);
    std::vector<std::byte> memory(bytes);
    runtime::RuntimeKraftWorkspace workspace;
    assert(runtime::bind_runtime_kraft_workspace(1, 0, memory.data(), bytes, workspace));
    double lower = NAN, upper = NAN;
    for (const auto policy : {kraft::InvalidTrialPolicy::interpolate,
                              kraft::InvalidTrialPolicy::minimum_step})
    for (const bool reset : {false, true})
    for (const double accuracy : {1e-8, 1e-12})
    for (const int failure : {0, 1, 2}) {
        for (const bool recover : {false, true}) {
            for (const bool speculate : {false, true}) {
                for (const bool gradient_only : {false, true}) {
                    if (gradient_only && !speculate) continue;
                    const RejectedTrial objective{failure, recover, speculate, gradient_only};
                    double x = recover ? 2 : 0;
                    const auto result = runtime::minimize_runtime_kraft(objective,
                        Restrictions{1, 0, 0}, &x, {&lower, &upper},
                        kraft::Options{accuracy, 100, reset, policy}, workspace);
                    assert(result.function_evaluations==objective.values);
                    assert(result.gradient_evaluations==objective.gradients);
                    assert(result.constraint_curvature_resets == 0);
                    if (recover) {
                        assert(result.mode == 0);
                        assert(std::abs(x - 1) < 1e-10);
                        assert(result.function_evaluations > 2);
                    } else {
                        assert(result.mode == VLK_SLSQP_INVALID_OBJECTIVE);
                        if (gradient_only) {
                            assert(x == 1 && result.value == 0);
                            assert(!result.derivatives_current);
                            assert(std::isnan(workspace.gradient[0]));
                            continue;
                        }
                        assert(x == 0 && result.value == 0.5);
                        assert(workspace.gradient[0] == -1);
                        assert(result.derivatives_current);
                    }
                    double f, g;
                    assert(objective.evaluate(kraft::ObjectiveRequest::value_gradient,
                        &x, &f, &g, false));
                    assert(f == result.value && g == workspace.gradient[0]);
                }
            }
            kraft::Workspace<1, 0, 0> fixed_workspace;
            kraft::Bounds<1> bounds;
            double x = recover ? 2 : 0;
            const auto result = kraft::minimize(
                FixedRejectedTrial{{failure, recover, false, false}},
                kraft::NoConstraints<1>{}, &x, bounds,
                kraft::Options{accuracy, 100, reset, policy}, fixed_workspace);
            assert(result.constraint_curvature_resets == 0);
            assert(result.mode == (recover ? 0 : VLK_SLSQP_INVALID_OBJECTIVE));
            if (recover) assert(std::abs(x - 1) < 1e-10);
            else assert(std::isnan(result.value));
        }
    }
    kraft::Workspace<1, 0, 0> fixed_workspace;
    kraft::Bounds<1> bounds;
    double x = 0;
    const auto cancelled = kraft::minimize(FixedRejectedTrial{{0, false, false, false}},
        kraft::NoConstraints<1>{}, &x, bounds, kraft::Options{}, fixed_workspace,
        CancelAfterTrial{});
    assert(cancelled.mode == kraft::cancelled_mode);
    assert(std::isnan(cancelled.value));
}

void verify_invalid_step_policy() {
    const auto bytes = runtime::runtime_kraft_workspace_bytes(1, 0);
    std::vector<std::byte> memory(bytes);
    runtime::RuntimeKraftWorkspace workspace;
    assert(runtime::bind_runtime_kraft_workspace(1, 0, memory.data(), bytes, workspace));
    for (const auto policy : {kraft::InvalidTrialPolicy::interpolate,
                              kraft::InvalidTrialPolicy::minimum_step}) {
        const auto options = kraft::Options{1e-12, 100, false, policy};
        const RejectedTrial objective{0, true, false, false};
        double x = 2, lower = NAN, upper = NAN;
        const auto result = runtime::minimize_runtime_kraft(objective,
            Restrictions{1, 0, 0}, &x, {&lower, &upper}, options, workspace);
        assert(result.mode == 0 && objective.rejected_values >= 2);
        const double ratio = (objective.first_rejected[1]-2)/(objective.first_rejected[0]-2);
        if (policy == kraft::InvalidTrialPolicy::minimum_step)
            assert(std::abs(ratio-0.1) < 1e-14);
        else
            assert(ratio > 0.1 && ratio <= 0.5);
        const FixedRejectedTrial fixed{{0, true, false, false}};
        kraft::Workspace<1, 0, 0> fixed_workspace;
        kraft::Bounds<1> bounds;
        double fixed_x = 2;
        const auto fixed_result = kraft::minimize(fixed, kraft::NoConstraints<1>{},
            &fixed_x, bounds, options, fixed_workspace);
        assert(fixed_result.mode == 0 && fixed_x == x);
        assert(fixed.inner.rejected_values == objective.rejected_values);
        for (int j = 0; j < 2; ++j)
            assert(fixed.inner.first_rejected[j] == objective.first_rejected[j]);
    }
}

void verify_curvature_prediction() {
    ScivolSqpCurvatureReset prediction{};
    prediction.enabled=1;
    double lower[]{NAN}, upper[]{NAN};
    for (int variable_exponent : {-200,-20,0,20,200})
    for (int constraint_exponent : {-200,-20,0,20,200})
    for (int objective_exponent : {-200,-20,0,20,200}) {
        const double variable=std::exp2(variable_exponent);
        const double constraint=std::exp2(constraint_exponent);
        const double objective=std::exp2(objective_exponent);
        const double base[]{0.5/variable}, trial[]{1.5/variable};
        const double c[]{0.25*constraint}, jac[]{constraint*variable}, dual[]{objective/constraint};
        scivol_sqp_reset_begin(&prediction);
        scivol_sqp_reset_predict(&prediction,1,1,c,jac,dual,base,trial,lower,upper);
        const double curved[]{2.25*constraint};
        assert(scivol_sqp_prediction_failed(&prediction,1,1,curved,dual,
            -0.1*objective,objective,1.5*objective));
        const double clipped_upper[]{0.7/variable};
        scivol_sqp_reset_predict(&prediction,1,1,c,jac,dual,base,trial,lower,clipped_upper);
        const double affine[]{std::fma(jac[0],clipped_upper[0]-base[0],c[0])};
        assert(!scivol_sqp_prediction_failed(&prediction,1,1,affine,dual,
            -1e-25*objective,objective,objective));
        for (double invalid : {NAN,INFINITY,-INFINITY}) {
            const double bad[]{invalid};
            scivol_sqp_reset_predict(&prediction,1,1,c,jac,dual,base,trial,lower,upper);
            assert(!scivol_sqp_prediction_failed(&prediction,1,1,bad,dual,
                -0.1*objective,objective,objective));
            scivol_sqp_reset_predict(&prediction,1,1,c,bad,dual,base,trial,lower,upper);
            assert(!scivol_sqp_prediction_failed(&prediction,1,1,curved,dual,
                -0.1*objective,objective,objective));
        }
    }
    scivol_sqp_reset_begin(&prediction);
    const double c[]{0.25}, jac[]{1}, dual[]{1}, base[]{0.5}, trial[]{1.5}, curved[]{2.25};
    scivol_sqp_reset_predict(&prediction,1,1,c,jac,dual,base,trial,lower,upper);
    const auto observe=[&](bool valid, double change) {
        scivol_sqp_reset_observe(&prediction,1,1,curved,dual,valid,change,-0.1,1,1.5);
    };
    observe(true,0.5); assert(!prediction.pending && prediction.consecutive_failures==1);
    observe(true,-0.02); assert(prediction.consecutive_failures==0);
    observe(true,0.5); observe(false,0.5);
    assert(!prediction.pending && prediction.consecutive_failures==0);
    observe(true,0.5); observe(true,0.5); assert(prediction.pending);
    assert(scivol_sqp_reset_take(&prediction) && prediction.count==1);
    assert(!scivol_sqp_reset_take(&prediction));
    observe(true,0.5); observe(true,0.5);
    assert(!scivol_sqp_reset_take(&prediction));
    scivol_sqp_reset_begin(&prediction);
    assert(prediction.count==0 && !prediction.pending && prediction.enabled);
}

struct ConstraintDomainObjective {
    bool recover;
    std::size_t dimension() const noexcept { return 1; }
    bool evaluate(kraft::ObjectiveRequest request,const double* x,double* f,
                  double* g,bool) const noexcept {
        const double curvature=recover?1e6:1.0;
        if (request!=kraft::ObjectiveRequest::gradient) *f=0.5*curvature*(x[0]-1)*(x[0]-1);
        if (request!=kraft::ObjectiveRequest::value) *g=curvature*(x[0]-1);
        return true;
    }
};

struct FailedConstraintValue {
    bool recover;
    int failure;
    mutable unsigned invalid_jacobians=0;
    std::size_t dimension() const noexcept { return 1; }
    std::size_t count() const noexcept { return 1; }
    bool evaluate(kraft::ConstraintRequest request,const double* x,double* c,
                  double* jac) const noexcept {
        const bool invalid=recover?x[0]<=0:x[0]>0;
        if (request!=kraft::ConstraintRequest::value) {
            *jac=0;
            invalid_jacobians+=invalid;
        }
        if (request!=kraft::ConstraintRequest::jacobian) {
            *c=1;
            if (invalid) {
                if (!failure) return false;
                *c=failure==1?NAN:INFINITY;
            }
        }
        return true;
    }
};

void verify_invalid_constraint_trials() {
    const auto bytes=runtime::runtime_kraft_workspace_bytes(1,1);
    std::vector<std::byte> memory(bytes);
    runtime::RuntimeKraftWorkspace workspace;
    assert(runtime::bind_runtime_kraft_workspace(1,1,memory.data(),bytes,workspace));
    for (bool reset : {false,true})
    for (bool recover : {false,true})
    for (int failure : {0,1,2}) {
        double x=recover?2:0, lower=NAN, upper=NAN;
        const FailedConstraintValue constraints{recover,failure};
        const auto result=runtime::minimize_runtime_kraft(ConstraintDomainObjective{recover},
            constraints,&x,{&lower,&upper},{1e-12,100,reset},workspace);
        assert(constraints.invalid_jacobians==0);
        assert(result.constraint_curvature_resets==0);
        if (recover) {
            assert(result.mode==0 && std::abs(x-1)<1e-10);
        } else {
            assert(result.mode==-102);
            assert(x==0 && result.value==0.5 && result.derivatives_current);
        }
    }
}

struct CoupledQuadratic {
    bool invalid_fixed_gradient=false;
    std::size_t dimension() const noexcept {return 3;}
    bool evaluate(kraft::ObjectiveRequest request,const double* x,double* f,double* g,bool) const noexcept {
        const double sum=x[0]+2*x[1]+x[2];
        if (request!=kraft::ObjectiveRequest::gradient)
            *f=.5*sum*sum+.5*(x[2]-3)*(x[2]-3)+1e200*x[1];
        if (request!=kraft::ObjectiveRequest::value) {
            g[0]=sum;g[1]=invalid_fixed_gradient?NAN:2*sum+1e200;g[2]=sum+x[2]-3;
        }
        return true;
    }
};

struct InvalidHeldJacobian : Restrictions {
    bool evaluate(kraft::ConstraintRequest request,const double* x,double* c,double* a) const noexcept {
        Restrictions::evaluate(request,x,c,a);
        if (request!=kraft::ConstraintRequest::value) a[rows]=NAN;
        return true;
    }
};

void check_restricted_coordinates()
{
    const std::size_t indices[]{0,2},duplicate[]{0,0},outside[]{0,3};
    double point[]{-.5,0.,1.5},lower[]{-2.,0.,-1.},upper[]{2.,NAN,4.},scales[]{.5,1.,2.};
    const auto bytes=runtime::CoordinateRestriction::bytes(3,1);
    std::vector<std::byte> memory(bytes,std::byte{0x47});
    runtime::CoordinateRestriction map;
    assert(!map.bind(3,2,1,indices,point,{lower,upper},memory.data(),bytes-1,scales));
    for (auto value:memory) assert(value==std::byte{0x47});
    assert(!map.bind(3,2,1,duplicate,point,{lower,upper},memory.data(),bytes,scales));
    assert(!map.bind(3,2,1,outside,point,{lower,upper},memory.data(),bytes,scales));
    assert(!map.bind(3,0,1,indices,point,{lower,upper},memory.data(),bytes,scales));
    std::vector<std::byte> unaligned(bytes+1,std::byte{0x47});
    assert(!map.bind(3,2,1,indices,point,{lower,upper},unaligned.data()+1,bytes,scales));
    for (auto value:unaligned) assert(value==std::byte{0x47});
    const double invalid_scales[]{0.,-1.,NAN,INFINITY,1e-320};
    for (double invalid_scale : invalid_scales) {
        const double bad_scales[]{invalid_scale,1.,2.};
        assert(!map.bind(3,2,1,indices,point,{lower,upper},memory.data(),bytes,bad_scales));
        for (auto value:memory) assert(value==std::byte{0x47});
    }
    assert(map.bind(3,2,1,indices,point,{lower,upper},memory.data(),bytes,scales));
    const CoupledQuadratic objective;
    const Restrictions constraints{3,1,1};
    const runtime::RestrictedObjective restricted_objective{objective,map};
    const runtime::RestrictedConstraints restricted_constraints{constraints,map};
    assert(restricted_constraints.equalities()==1 && restricted_constraints.dimension()==2);
    double f=31.,g[]{47.,47.},c=31.,j[]{47.,47.};
    assert(restricted_objective.evaluate(kraft::ObjectiveRequest::gradient,map.point,&f,g,false));
    assert(f==31. && g[0]==.5 && g[1]==-1.);
    assert(restricted_objective.evaluate(kraft::ObjectiveRequest::value,map.point,&f,g,false));
    assert(f==1.625 && g[0]==.5 && g[1]==-1.);
    assert(restricted_constraints.evaluate(kraft::ConstraintRequest::jacobian,map.point,&c,j));
    assert(c==31. && j[0]==.5 && j[1]==2.);
    assert(restricted_constraints.evaluate(kraft::ConstraintRequest::value,map.point,&c,j));
    assert(c==0. && j[0]==.5 && j[1]==2.);
    const CoupledQuadratic invalid{true};
    const runtime::RestrictedObjective bad{invalid,map};
    assert(!bad.evaluate(kraft::ObjectiveRequest::gradient,map.point,&f,g,false));
    const InvalidHeldJacobian invalid_constraint{{3,1,1}};
    const runtime::RestrictedConstraints bad_constraint{invalid_constraint,map};
    assert(bad_constraint.evaluate(kraft::ConstraintRequest::value,map.point,&c,j));
    assert(!bad_constraint.evaluate(kraft::ConstraintRequest::jacobian,map.point,&c,j));
    assert(!bad_constraint.evaluate(kraft::ConstraintRequest::value_jacobian,map.point,&c,j));
    const double outside_point[]{100.,0.};
    assert(!restricted_objective.evaluate(kraft::ObjectiveRequest::value,outside_point,&f,g,false));
    const auto solver_bytes=runtime::runtime_kraft_workspace_bytes(2,1,1);
    std::vector<std::byte> solver_memory(solver_bytes);
    runtime::RuntimeKraftWorkspace state;
    assert(runtime::bind_runtime_kraft_workspace(2,1,solver_memory.data(),solver_bytes,state,1));
    const runtime::RestrictedCoordinateKraft solve{map};
    const auto result=solve(objective,constraints,point,{lower,upper},{1e-12,50},state);
    assert(result.mode==0 && result.maximum_violation<1e-12);
    assert(std::abs(point[0]+2)<1e-10 && point[1]==0. && std::abs(point[2]-3)<1e-10);
    assert(map.origin[0]==-.5 && map.origin[2]==1.5);
    assert(runtime::CoordinateRestriction::bytes(SIZE_MAX,2)==0);
}

void verify_augmented_workspace() {
    const std::size_t n=64,m=1;
    const auto bytes=runtime::runtime_kraft_workspace_bytes(n,m);
    std::vector<double> memory((bytes+7)/8+2,919.0);
    runtime::RuntimeKraftWorkspace work;
    assert(runtime::bind_runtime_kraft_workspace(n,m,memory.data()+1,bytes,work));
    std::vector<double> x(n,0.0),lower(n,-2.0),upper(n,2.0);
    std::fill_n(work.gradient,n,1.0);
    std::fill_n(work.jacobian,n*m,0.0);
    work.constraint_values[0]=-1.0;
    SLSQP_vars state{};
    state.n=n;state.m=m;state.acc=1e-12;state.itermax=10;
    double value=0.0;
    __slsqp_body(&state,&value,work.gradient,work.jacobian,work.constraint_values,
        x.data(),work.multipliers,lower.data(),upper.data(),work.buffer,work.indices,work.nnls);
    // The impossible linearized row forces the extra feasibility variable.
    assert(state.h4<1e-8 && state.mode!=2);
    assert(memory.front()==919 && memory.back()==919);
}

struct SelectiveConstraint {
    std::size_t dimension() const noexcept {return 3;}
    std::size_t count() const noexcept {return 1;}
    bool evaluate(kraft::ConstraintRequest r,const double* x,double* c,double* j,
                  runtime::ConstraintColumns columns={}) const noexcept {
        if (r!=kraft::ConstraintRequest::jacobian) *c=x[0]+2*x[2];
        if (r!=kraft::ConstraintRequest::value) {
            j[0]=1.; j[1]=NAN; j[2]=2.;
            if (columns.contains(1)) return false;
        }
        return true;
    }
};

void verify_composed_constraint_columns() {
    double x[]{1.,2.,3.},scales[]{2.,3.,4.},scaled[3];
    const double normal[]{1.,2.,3.};
    const runtime::AffineEquality affine{normal,3,0.};
    const SelectiveConstraint raw;
    const runtime::AffineEqualityPrefix prefixed{raw,affine};
    const runtime::DiagonalConstraints diagonal{prefixed,scales,scaled};
    const std::size_t selected[]{0,2};
    double lower[]{NAN,NAN,NAN},upper[]{NAN,NAN,NAN};
    const auto bytes=runtime::CoordinateRestriction::bytes(3,2);
    std::vector<std::byte> scratch(bytes);
    runtime::CoordinateRestriction map;
    assert(map.bind(3,2,2,selected,x,{lower,upper},scratch.data(),bytes));
    const runtime::RestrictedConstraints restricted{diagonal,map};
    double c[]{91.,92.},j[]{93.,94.,95.,96.};
    assert(restricted.evaluate(kraft::ConstraintRequest::jacobian,map.point,c,j));
    assert(c[0]==91. && c[1]==92.);
    assert(j[0]==2. && j[1]==2. && j[2]==12. && j[3]==8.);
    assert(restricted.evaluate(kraft::ConstraintRequest::value,map.point,c,j));
    assert(c[0]==50. && c[1]==26.);
    assert(j[0]==2. && j[1]==2. && j[2]==12. && j[3]==8.);
    assert(restricted.evaluate(kraft::ConstraintRequest::value_jacobian,map.point,c,j));
    double full[6];
    assert(!runtime::runtime_kraft_detail::evaluate_constraints(
        diagonal,kraft::ConstraintRequest::jacobian,x,c,full));
}

int main() {
    for (double scale : {1e-200,1.,1e200}) {
        kraft::Workspace<2,2,2> workspace;
        workspace.jacobian={scale,0.,0.,scale};
        assert(kraft::detail::equality_jacobian_has_full_row_rank(workspace));
        workspace.jacobian={scale,scale,0.,0.};
        assert(!kraft::detail::equality_jacobian_has_full_row_rank(workspace));
    }
    verify_composed_constraint_columns();
    verify_augmented_workspace();
    std::vector<double> wide_target(64);
    for (std::size_t i=0;i<wide_target.size();++i) wide_target[i]=(i+1)/64.0;
    verify(Restrictions{64,64,0,true,1.0/64},wide_target);
    check_restricted_coordinates();
    verify_curvature_prediction();
    verify_invalid_constraint_trials();
    verify_value_only_checkpoint();
    verify_success_does_not_discard_a_better_trial();
    verify_checkpoint_constraint_evidence();
    verify_rejected_trials();
    verify_invalid_step_policy();
    verify(Restrictions{2,1,1},{0.5,0.5});
    verify(Restrictions{2,2,1},{0.8,0.2});
    verify(Restrictions{3,3,3,true},{1.0,2.0,3.0});
    verify(Restrictions{8,1,1},std::vector<double>(8,0.125));
    verify(InequalitiesOnly{Restrictions{2,1,0}},{0.8,0.0});
    const Restrictions physical{2,1,1};
    const double scales[]{2.0,0.5};
    double physical_point[2];
    verify(runtime::DiagonalConstraints{physical,scales,physical_point},{8.0/17,2.0/17});
    const double normal[]{1.0,1.0};
    const runtime::AffineEquality affine{normal,2,1.0};
    assert(affine.valid(2));
    const Restrictions nonnegative{2,1,0};
    verify(runtime::AffineEqualityPrefix{nonnegative,affine},{0.8,0.2});
    verify(runtime::AffineEqualityPrefix{nonnegative,{}},{0.8,0.0});
    const Restrictions empty{8,0,0};
    verify(runtime::AffineEqualityPrefix{empty,affine},{0.5,0.5,0,0,0,0,0,0});
    const double first[]{1.0};
    const runtime::AffineEqualityPrefix two_equalities{physical,runtime::AffineEquality{first,1,0.8}};
    verify(two_equalities,{0.8,0.2});
    verify(runtime::DiagonalConstraints{two_equalities,scales,physical_point},{0.4,0.4});
    double x[]{0.2,0.3},full[2],jacobian[4];
    assert(two_equalities.evaluate(kraft::ConstraintRequest::value_jacobian,x,full,jacobian));
    double value[2]{},derivative[4]{};
    assert(two_equalities.evaluate(kraft::ConstraintRequest::value,x,value,nullptr));
    assert(two_equalities.evaluate(kraft::ConstraintRequest::jacobian,x,nullptr,derivative));
    for (unsigned i=0;i<2;++i) assert(value[i]==full[i]);
    for (unsigned i=0;i<4;++i) assert(derivative[i]==jacobian[i]);
    const mean::PolynomialDomain polynomial{1,1,nullptr};
    std::vector<std::byte> polynomial_work(polynomial.workspace_bytes(true));
    double polynomial_jacobian[2];
    const mean::PolynomialConstraints mean_constraints{
        2,{{polynomial,1},{}},polynomial_jacobian,polynomial_work.data(),polynomial_work.size()};
    const runtime::AffineEqualityPrefix mean_with_equality{mean_constraints,affine};
    double combined_values[3],combined_jacobian[6],saved_values[3];
    assert(mean_with_equality.evaluate(kraft::ConstraintRequest::value_jacobian,
        x,combined_values,combined_jacobian));
    std::copy_n(combined_values,3,saved_values);
    assert(mean_with_equality.evaluate(kraft::ConstraintRequest::jacobian,
        x,combined_values,combined_jacobian));
    for (unsigned i=0;i<3;++i) assert(combined_values[i]==saved_values[i]);
    assert(mean_with_equality.evaluate(kraft::ConstraintRequest::jacobian,
        x,nullptr,combined_jacobian));
    double saved_jacobian[6];
    std::copy_n(combined_jacobian,6,saved_jacobian);
    assert(mean_with_equality.evaluate(kraft::ConstraintRequest::value,
        x,combined_values,combined_jacobian));
    for (unsigned i=0;i<6;++i) assert(combined_jacobian[i]==saved_jacobian[i]);
    verify(mean_with_equality,{0.5,0.5});
    assert((!runtime::AffineEquality{normal,3,0}.valid(2)));
    assert((!runtime::AffineEquality{normal,2,NAN}.valid(2)));
    const double zero[]{0.0,0.0},nonfinite[]{1.0,INFINITY};
    assert((!runtime::AffineEquality{zero,2,0}.valid(2)));
    assert((!runtime::AffineEquality{nonfinite,2,0}.valid(2)));
    // Several affine rows: minimum-norm solution of [1 1 0; 0 1 1] x = [1; 2].
    const double faces[]{1.0,1.0,0.0,0.0,1.0,1.0},face_targets[]{1.0,2.0};
    const runtime::AffineEquality two_faces{faces,3,0.0,2,face_targets};
    assert(two_faces.valid(4) && two_faces.count()==2);
    const Restrictions four{4,0,0};
    verify(runtime::AffineEqualityPrefix{four,two_faces},{0.0,1.0,1.0,0.0});
    const Restrictions three_nonnegative{3,1,0};
    const runtime::AffineEqualityPrefix faces_first{three_nonnegative,two_faces};
    assert(faces_first.count()==3 && faces_first.equalities()==2);
    double y[]{0.1,0.2,0.3},face_values[3],face_jacobian[9],only_values[3]{},only_jacobian[9]{};
    assert(faces_first.evaluate(kraft::ConstraintRequest::value_jacobian,y,face_values,face_jacobian));
    assert(faces_first.evaluate(kraft::ConstraintRequest::value,y,only_values,nullptr));
    assert(faces_first.evaluate(kraft::ConstraintRequest::jacobian,y,nullptr,only_jacobian));
    assert(std::abs(face_values[0]-(0.3-1.0))<1e-15 && std::abs(face_values[1]-(0.5-2.0))<1e-15);
    for (unsigned i=0;i<3;++i) assert(only_values[i]==face_values[i]);
    for (unsigned i=0;i<9;++i) assert(only_jacobian[i]==face_jacobian[i]);
    // Column-major: the two equality rows precede the inner inequality in every column.
    for (unsigned j=0;j<3;++j) {
        assert(face_jacobian[3*j]==faces[j] && face_jacobian[3*j+1]==faces[3+j]);
    }
    assert((!runtime::AffineEquality{faces,3,0.0,2,nullptr}.valid(4)));
    assert((!runtime::AffineEquality{faces,3,0.0,4,face_targets}.valid(4)));
    const double second_zero[]{1.0,1.0,0.0,0.0,0.0,0.0};
    assert((!runtime::AffineEquality{second_zero,3,0.0,2,face_targets}.valid(4)));
    assert(runtime::runtime_kraft_workspace_bytes(2,1,2)==0);
    assert(runtime::runtime_kraft_workspace_bytes(2,3,3)==0);
    const auto old_bytes=runtime::runtime_kraft_workspace_bytes(8,1);
    const auto eq_bytes=runtime::runtime_kraft_workspace_bytes(8,1,1);
    assert(eq_bytes>old_bytes);
    std::vector<std::byte> memory(old_bytes);
    runtime::RuntimeKraftWorkspace workspace;
    assert(!runtime::bind_runtime_kraft_workspace(8,1,memory.data(),old_bytes,workspace,1));
    assert(runtime::bind_runtime_kraft_workspace(8,1,memory.data(),old_bytes,workspace));
    double point[8]{},lower[8],upper[8];
    std::fill_n(lower,8,NAN); std::fill_n(upper,8,NAN);
    const auto invalid=runtime::minimize_runtime_kraft(Quadratic{8},Restrictions{8,1,1},
        point,{lower,upper},kraft::Options{},workspace);
    assert(invalid.mode==-100 && invalid.function_evaluations==0);
    const auto nnls_workspace=workspace.nnls;
    for (unsigned failure=0;failure<3;++failure) {
        workspace.nnls=nnls_workspace;
        if (failure==0) workspace.nnls.memory=nullptr;
        if (failure==1) --workspace.nnls.bytes;
        if (failure==2) workspace.nnls.memory=static_cast<std::byte*>(workspace.nnls.memory)+1;
        std::fill_n(point,8,2.0);
        const auto bad=runtime::minimize_runtime_kraft(Quadratic{8},Restrictions{8,1,0},
            point,{lower,upper},kraft::Options{},workspace);
        assert(bad.mode==-100 && bad.function_evaluations==0);
        for (double value:point) assert(value==2.0);
    }
}
