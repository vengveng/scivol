// SCIVOL-NATIVE: CORE
#pragma once
#include "aparch_multi_phase_api.hpp"
#include "mean_anchor_selection.hpp"
#include "residual_mean_chart.hpp"
#include "residual_mean_mapper.hpp"
#include "aparch_runtime_jet.hpp"
#include "bounded_direction.hpp"
#include "runtime_reflection.hpp"
#include <cstring>
#include <span>
#include <type_traits>

namespace vlk::experiment::runtime::aparch::multi_phase {
namespace ap = vlk::experiment::runtime::aparch;
namespace rt = vlk::experiment::runtime;
namespace mean = vlk::experiment::mean;
namespace gen = vlk::experiment::generated;
namespace bd = vlk::experiment::bounded_direction;
using Real = vlk::experiment::wide::Real;
using Sample = vlk::experiment::mean_screen::ObservationSummary;
struct alignas(16) SelectionAlignment {};

constexpr std::size_t free_direction_bytes(std::size_t n) noexcept {
    const auto bounded = bd::make_layout(n);
    rt::WorkspaceSizer size;
    if (!bounded.bytes || !size.add<double>(n) || !size.add<double>(n)
        || !size.add<double>(bounded.square) || !size.add<double>(n)
        || !size.add<std::byte>(bounded.bytes)) return 0;
    return size.size();
}

// Diagonal Hessian units are a proposal metric only; G/H remain the exact chart jet.
template<std::size_t Fixed = 0>
bd::Result free_direction(std::size_t runtime, bd::Input g, bd::Input h, double radius,
    bd::Output d, void* memory, std::size_t bytes) noexcept {
    bd::Result result;
    const auto n = bd::dimension<Fixed>(runtime);
    auto layout = bd::make_layout(n);
    layout.bytes = free_direction_bytes(n);
    if ((Fixed && runtime != Fixed) || !std::isfinite(radius) || radius <= 0) return result;
    result.status = bd::validate_buffers(layout,g,h,d,memory,bytes);
    if (result.status != bd::Status::ok) return result;
    double gscale = 0, hscale = 0;
    if (!bd::validate_values<Fixed>(n,g,h,gscale,hscale)) {
        result.status = bd::Status::invalid_input; return result;
    }
    rt::WorkspaceArena arena(memory,bytes);
    auto* units = arena.take<double>(n);
    auto* scaled_g = arena.take<double>(n);
    auto* scaled_h = arena.take<double>(layout.square);
    auto* proposal = arena.take<double>(n);
    const auto bounded = bd::make_layout(n);
    auto* scratch = arena.take<std::byte>(bounded.bytes);
    if (!scratch) { result.status = bd::Status::short_buffer; return result; }
    for (std::size_t i = 0; i < n; ++i) {
        const double diagonal = std::abs(h.data[i*n+i]);
        units[i] = diagonal > 0 ? 1/std::sqrt(diagonal) : 1.;
        scaled_g[i] = g.data[i]*units[i];
    }
    for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j)
        scaled_h[i*n+j] = h.data[i*n+j]*units[i]*units[j];
    result = bd::propose<Fixed>(n,{scaled_g,n},{scaled_h,layout.square},radius,
        {proposal,n},scratch,bounded.bytes);
    if (result.status != bd::Status::ok) return result;
    for (std::size_t i = 0; i < n; ++i) {
        proposal[i] *= units[i];
        if (!std::isfinite(proposal[i])) { result.status = bd::Status::numerical_failure; return result; }
    }
    std::copy_n(proposal,n,d.data);
    // Result norm/slope/curvature describe the scaled proposal, as in bounded_direction.
    return result;
}

struct Extents {
    std::size_t n = 0, m = 0, k = 0, d = 0, square = 0, reduced = 0;
    std::size_t centers = 0, trials = 0, evidence = 0, seen = 0, mean_cache = 0;
    std::size_t evidence_cache = 0, root = 0, trace = 0;
};
constexpr Extents extents(const ap::Layout& l, std::size_t k,
    Strategy strategy = Strategy::held_residuals) noexcept {
    Extents e;
    if (!l.valid || !l.joint || !k || k > l.ma_start || k >= l.parameters) return e;
    if (strategy != Strategy::held_residuals && strategy != Strategy::free_chart) return e;
    e.n = l.parameters; e.m = l.mean_parameters; e.k = k;
    e.d = strategy == Strategy::free_chart ? e.n : e.n-k;
    if (!rt::checked_product(e.n,e.n,e.square) || !rt::checked_product(e.d,e.d,e.reduced)
        || !rt::checked_sum(e.d,1,e.centers) || !rt::checked_product(e.centers,5,e.centers)
        || !rt::checked_sum(e.centers,15,e.centers) || !rt::checked_product(e.centers,2,e.trials)
        || !rt::checked_product(k,8,e.evidence) || !rt::checked_sum(e.evidence,2,e.evidence)) return {};
    if (strategy == Strategy::free_chart) {
        std::size_t coordinate_centers = 0;
        if (!rt::checked_product(e.n,5,coordinate_centers)
            || !rt::checked_sum(32,coordinate_centers,e.centers)) return {};
        if (!rt::checked_product(e.n,2,e.trials) || !rt::checked_sum(e.trials,e.centers,e.trials)) return {};
    }
    if (!rt::checked_product(e.trials,e.n,e.seen)
        || !rt::checked_product(e.centers,e.m,e.mean_cache)
        || !rt::checked_product(e.centers,e.evidence,e.evidence_cache)
        || !rt::checked_product(e.n,2,e.trace) || !rt::checked_sum(e.trace,12,e.trace)) return {};
    e.root = std::max(l.p_ar,l.q_ma);
    return e;
}
struct Plan {
    Extents e; Strategy strategy = Strategy::held_residuals;
    std::size_t core = 0, primal = 0, chart = 0, mapper = 0, direction = 0, selector = 0, bytes = 0;
};
struct Counts {
    std::size_t values = 0, curvatures = 0, mapper_prepares = 0, mappings = 0;
    std::size_t proposals = 0, unique = 0, rejected = 0, accepted = 0, iterations = 0;
    std::size_t gradients = 0, restarts = 0, reselections = 0;
};

template<class Density, class Orders, std::size_t FixedK = mean::dynamic>
class Phase {
    static constexpr const ap::Layout* fixed = Orders::fixed ? &ap::fixed_layout<Orders,Density,true> : nullptr;
    using Primal = mean::ArmaResidualPrecision<ap::Layout,fixed>;
    using Chart = vlk::experiment::mean_chart::ResidualPreservingChart<2,ap::Layout,fixed,FixedK>;
    using Mapper = vlk::experiment::mean_chart::OffOriginMapper<ap::Layout,fixed,FixedK>;
    using Selector = vlk::experiment::mean_chart::anchor_selection::Selector<ap::Layout,fixed,FixedK>;
    static constexpr Extents constant = [] {
        if constexpr (fixed && FixedK != mean::dynamic) return extents(*fixed,FixedK);
        return Extents{};
    }();
    static constexpr Extents free_constant = [] {
        if constexpr (fixed && FixedK != mean::dynamic) return extents(*fixed,FixedK,Strategy::free_chart);
        return Extents{};
    }();
    const ap::Layout& supplied_; const Plan& p_;
    const double* data_; std::size_t n_; double h0_;
    const std::size_t *anchors_, *columns_;
    Sample sample_;
    void *core_ = nullptr, *primal_ = nullptr, *chart_ = nullptr, *mapper_ = nullptr, *direction_memory_ = nullptr;
    void* selector_ = nullptr;
    double* theta_g_ = nullptr;
    double *current_ = nullptr, *best_ = nullptr, *selected_ = nullptr, *coordinates_ = nullptr;
    double *mapped_ = nullptr, *trial_ = nullptr, *g_ = nullptr, *h_ = nullptr;
    double *reduced_g_ = nullptr, *reduced_h_ = nullptr, *direction_ = nullptr, *seen_ = nullptr;
    double *mean_keys_ = nullptr, *mean_values_ = nullptr, *mean_evidence_ = nullptr, *evidence_ = nullptr;
    double *root_current_ = nullptr, *root_next_ = nullptr;
    std::size_t* free_ = nullptr;
    std::size_t *retained_anchors_ = nullptr, *retained_columns_ = nullptr;
    int* map_status_ = nullptr;
    std::size_t seen_count_ = 0, means_ = 0;
    double current_value_ = INFINITY, best_value_ = INFINITY, selected_value_ = INFINITY;
    double selected_slope_ = 0, selected_radius_ = 0, selected_fraction_ = 0;
    std::size_t selected_source_ = 0;
    Counts counts_;

    const ap::Layout& layout() const noexcept { return ap::evaluation_layout<Orders,Density,true>(supplied_); }
    bool free_chart() const noexcept { return p_.strategy == Strategy::free_chart; }
    Extents e() const noexcept {
        if constexpr (fixed && FixedK != mean::dynamic) return free_chart() ? free_constant : constant;
        return p_.e;
    }
    bool is_selected(std::size_t column) const noexcept {
        for (std::size_t r = 0; r < e().k; ++r) if (columns_[r] == column) return true;
        return false;
    }
    bool feasible(const double* theta) noexcept {
        const auto& l = layout();
        return std::all_of(theta,theta+e().n,[](double v){return std::isfinite(v);})
            && ap::valid_theta<Density,fixed>(l,theta)
            && (!l.p_ar || rt::reflection_inverse_values(theta+l.ar_start,l.p_ar,1.,nullptr,root_current_,root_next_))
            && (!l.q_ma || rt::reflection_inverse_values(theta+l.ma_start,l.q_ma,-1.,nullptr,root_current_,root_next_));
    }
    bool value(const double* theta, double& f) noexcept {
        request_accounting::PhysicalRequest accounting(1u,false);
        ++counts_.values;
        Primal primal(layout(),theta,sample_);
        request_accounting::prepare(request_accounting::Preparation::mean_primal,0u);
        if (!primal.prepare(primal_,p_.primal)) return accounting.finish_bool(false);
        return accounting.finish_bool(ap::evaluate_theta_order<0,true,Orders,Density,true,false,false,void,false,void,false,Primal>(
            layout(),theta,data_,n_,h0_,core_,p_.core,&f,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,&primal)
            && std::isfinite(f));
    }
    bool derivatives() noexcept {
        request_accounting::PhysicalRequest accounting(6u,false);
        ++counts_.curvatures;
        Primal primal(layout(),current_,sample_);
        request_accounting::prepare(request_accounting::Preparation::mean_primal,0u);
        if (!primal.prepare(primal_,p_.primal)) return accounting.finish_bool(false);
        Chart chart(layout(),current_,data_,n_,anchors_,columns_,e().k,&primal);
        request_accounting::prepare(request_accounting::Preparation::residual_chart,2u);
        if (!chart.prepare(chart_,p_.chart)) return accounting.finish_bool(false);
        // The accepted point already owns its canonical F; only G and H are missing.
        if (!ap::evaluate_theta_order<2,true,Orders,Density,false,true,true,void,false,Chart>(
                layout(),current_,data_,n_,h0_,core_,p_.core,nullptr,g_,h_,nullptr,nullptr,nullptr,nullptr,&chart)) return accounting.finish_bool(false);
        for (std::size_t i = 0; i < e().d; ++i) {
            reduced_g_[i] = g_[free_[i]];
            for (std::size_t j = 0; j < e().d; ++j)
                reduced_h_[i*e().d+j] = h_[free_[i]*e().n+free_[j]];
        }
        return accounting.finish_bool(true);
    }
    bool gradient(const double* theta) noexcept {
        request_accounting::PhysicalRequest accounting(2u,false);
        ++counts_.gradients;
        Primal primal(layout(),theta,sample_);
        request_accounting::prepare(request_accounting::Preparation::mean_primal,0u);
        return accounting.finish_bool(primal.prepare(primal_,p_.primal)
            && ap::evaluate_theta_order<1,true,Orders,Density,false,true,false,void,false,void,false,Primal>(
                layout(),theta,data_,n_,h0_,core_,p_.core,nullptr,theta_g_,nullptr,
                nullptr,nullptr,nullptr,nullptr,nullptr,&primal));
    }
    double actual_slope(const double* point, bool changed_mean) const noexcept {
        Real sum(0);
        for (std::size_t i = 0; i < e().n; ++i) {
            Real displacement = Real(point[i])-Real(current_[i]);
            for (std::size_t r = 0; r < e().k; ++r) if (columns_[r] == i) {
                displacement = Real(0);
                if (changed_mean) { displacement.high = evidence_[8*r+6]; displacement.low = evidence_[8*r+7]; }
            }
            sum += Real(g_[i])*displacement;
        }
        return static_cast<double>(sum);
    }
    void consider(const double* point, bool changed_mean, std::size_t source, double radius, double fraction,
        const double* theta_gradient = nullptr) noexcept {
        ++counts_.proposals;
        for (std::size_t i = 0; i < seen_count_; ++i)
            if (std::memcmp(point,seen_+i*e().n,e().n*sizeof(double)) == 0) return;
        if (seen_count_ == e().trials) return;
        std::copy_n(point,e().n,seen_+seen_count_++*e().n); ++counts_.unique;
        double f = INFINITY;
        if (!feasible(point) || !value(point,f)) { ++counts_.rejected; return; }
        if (f < best_value_) { best_value_ = f; std::copy_n(point,e().n,best_); }
        double slope = actual_slope(point,changed_mean);
        if (theta_gradient) {
            Real sum(0);
            for (std::size_t i = 0; i < e().n; ++i)
                sum += Real(theta_gradient[i])*(Real(point[i])-Real(current_[i]));
            slope = static_cast<double>(sum);
        }
        volatile double increment = 1e-4*slope;
        if (std::isfinite(slope) && slope < 0 && f <= current_value_+increment && f < selected_value_) {
            selected_value_ = f; selected_slope_ = slope;
            selected_source_ = source; selected_radius_ = radius; selected_fraction_ = fraction;
            std::copy_n(point,e().n,selected_);
        }
    }
    void propose(Mapper& mapper, std::size_t source, double radius, double fraction) noexcept {
        std::copy_n(current_,e().n,coordinates_);
        for (std::size_t r = 0; r < e().k; ++r) coordinates_[columns_[r]] = 0.;
        for (std::size_t i = 0; i < e().d; ++i) {
            volatile double shift = fraction*direction_[i];
            coordinates_[free_[i]] += shift;
        }
        std::size_t center = 0;
        while (center < means_ && std::memcmp(coordinates_,mean_keys_+center*e().m,e().m*sizeof(double))) ++center;
        if (center == means_) {
            if (means_ == e().centers) return;
            ++counts_.mappings;
            const int status = mapper.map(coordinates_,mapped_,evidence_);
            std::copy_n(coordinates_,e().m,mean_keys_+center*e().m);
            map_status_[center] = status;
            if (!status) {
                std::copy_n(mapped_,e().m,mean_values_+center*e().m);
                std::copy_n(evidence_,e().evidence,mean_evidence_+center*e().evidence);
            }
            ++means_;
        }
        if (map_status_[center]) { ++counts_.rejected; return; }
        std::copy_n(coordinates_,e().n,mapped_);
        std::copy_n(mean_values_+center*e().m,e().m,mapped_);
        std::copy_n(mean_evidence_+center*e().evidence,e().evidence,evidence_);
        consider(mapped_,true,source,radius,fraction);
        if (free_chart()) return;
        std::copy_n(mapped_,e().n,trial_); std::copy_n(current_,e().m,trial_);
        consider(trial_,false,source,radius,fraction);
    }
    void write_summary(int stop, double* result) const noexcept {
        const double fields[] = {double(stop),best_value_,current_value_,double(counts_.iterations),
            double(counts_.accepted),double(counts_.values),double(counts_.curvatures),double(counts_.mapper_prepares),
            double(counts_.mappings),double(counts_.proposals),double(counts_.unique),double(counts_.rejected)};
        std::copy_n(fields,12,result);
        if (free_chart()) {
            result[12]=double(counts_.gradients); result[13]=double(counts_.restarts);
            result[14]=double(counts_.reselections); result[15]=double(p_.strategy);
        }
    }
    bool proposals(Mapper& mapper) noexcept {
        if (free_chart()) {
            constexpr std::size_t dimension = fixed && FixedK != mean::dynamic ? free_constant.d : 0;
            const auto proposal = free_direction<dimension>(e().d,{reduced_g_,e().d},
                {reduced_h_,e().reduced},1.,{direction_,e().d},direction_memory_,p_.direction);
            if (proposal.status != bd::Status::ok) return false;
            for (std::size_t scale = 0; scale < 32; ++scale) propose(mapper,0,1.,std::ldexp(1.,-int(scale)));
            for (std::size_t i = 0; i < e().n; ++i) for (double side : {-INFINITY,INFINITY}) {
                std::copy_n(current_,e().n,trial_);
                trial_[i] = std::nextafter(trial_[i],side);
                consider(trial_,false,1+i,0.,0.,theta_g_);
            }
            // A rounded coupled chart step can lose a decrease in other coordinates.
            // These are proposals for the same joint objective, not a conditional solve.
            if (selected_value_ >= current_value_-2e-11 && best_value_ >= current_value_-2e-11) {
                for (std::size_t i = 0; i < e().d; ++i) {
                    std::fill_n(direction_,e().d,0.);
                    const auto coordinate = free_direction<1>(1,{reduced_g_+i,1},
                        {reduced_h_+i*e().d+i,1},1.,{direction_+i,1},direction_memory_,p_.direction);
                    if (coordinate.status != bd::Status::ok) continue;
                    for (std::size_t scale = 0; scale < 5; ++scale)
                        propose(mapper,e().n+2+i,1.,std::ldexp(1.,-int(scale)));
                }
            }
            return true;
        }
        constexpr std::size_t dimension = fixed && FixedK != mean::dynamic ? constant.d : 0;
        for (double radius : {.01,.1,1.}) {
            const auto result = bd::propose<dimension>(e().d,{reduced_g_,e().d},{reduced_h_,e().reduced},radius,
                {direction_,e().d},direction_memory_,p_.direction);
            if (result.status != bd::Status::ok) return false;
            for (std::size_t scale = 0; scale < 5; ++scale) propose(mapper,0,radius,std::ldexp(1.,-int(scale)));
        }
        double square = 0;
        for (std::size_t i = 0; i < e().d; ++i) { volatile double term = reduced_g_[i]*reduced_g_[i]; square += term; }
        const double length = std::sqrt(square);
        if (length > 0 && std::isfinite(length))
            for (std::size_t source = 0; source <= e().d; ++source)
                for (double scale : {1e-4,1e-5,1e-6,1e-7,1e-8}) {
                    for (std::size_t i = 0; i < e().d; ++i) {
                        const double unit = !source ? -reduced_g_[i]/length : source == i+1
                            ? (reduced_g_[i]>0 ? -1. : reduced_g_[i]<0 ? 1. : 0.) : 0.;
                        direction_[i] = scale*unit;
                    }
                    propose(mapper,source+1,scale,1.);
                }
        return true;
    }
public:
    static Plan workspace(const ap::Layout& supplied, std::size_t k,
        Strategy strategy = Strategy::held_residuals) noexcept {
        const auto& l = ap::evaluation_layout<Orders,Density,true>(supplied);
        Plan p; p.strategy = strategy; p.e = extents(l,mean::extent<FixedK>(k),strategy);
        const auto& e = p.e;
        if (!e.n) return {};
        p.core = ap::core_workspace_bytes<2,true,fixed,false,false>(l,std::is_same_v<Density,gen::NormalDensity>);
        p.primal = Primal::workspace_bytes(l); p.chart = Chart::workspace_bytes(l,e.k);
        p.mapper = Mapper::workspace_layout(l,e.k).bytes;
        p.direction = strategy == Strategy::free_chart ? free_direction_bytes(e.d) : bd::make_layout(e.d).bytes;
        p.selector = strategy == Strategy::free_chart ? Selector::workspace_bytes(l,e.k) : 0;
        if (!p.core || !p.chart || !p.mapper || !p.direction) return {};
        rt::WorkspaceSizer size;
        for (auto bytes : {p.core,p.primal,p.chart,p.mapper,p.direction})
            if (!size.add<Real>(0) || !size.add<std::byte>(bytes)) return {};
        for (auto count : {e.n,e.n,e.n,e.n,e.n,e.n,e.n,e.square,e.d,e.reduced,e.d,e.seen,
                e.mean_cache,e.mean_cache,e.evidence_cache,e.evidence,e.root,e.root})
            if (!size.add<double>(count)) return {};
        if (!size.add<std::size_t>(e.d) || !size.add<int>(e.centers)) return {};
        if (strategy == Strategy::free_chart && (!p.selector || !size.add<double>(e.n)
            || !size.add<std::size_t>(e.k) || !size.add<std::size_t>(e.k)
            || !size.add<SelectionAlignment>(0) || !size.add<std::byte>(p.selector))) return {};
        p.bytes = size.size(); return p;
    }
    Phase(const ap::Layout& l, const Plan& p, const double* data, std::size_t n, double h0,
        const std::size_t* anchors, const std::size_t* columns) noexcept
        : Phase(l,p,Sample::prepare(data,n),h0,anchors,columns) {
        request_accounting::prepare(request_accounting::Preparation::sample,0u);
    }
    Phase(const ap::Layout& l, const Plan& p, const Sample& sample, double h0,
        const std::size_t* anchors, const std::size_t* columns) noexcept
        : supplied_(l),p_(p),data_(sample.data()),n_(sample.size()),h0_(h0),
          anchors_(anchors),columns_(columns),sample_(sample) {}
    bool bind(void* memory, std::size_t bytes) noexcept {
        const auto alignment = free_chart() ? alignof(SelectionAlignment) : alignof(Real);
        if (!memory || !p_.bytes || bytes < p_.bytes || reinterpret_cast<std::uintptr_t>(memory)%alignment) return false;
        rt::WorkspaceArena arena(memory,bytes);
        const auto block = [&](std::size_t size) { (void)arena.take<Real>(0); return arena.take<std::byte>(size); };
        core_ = block(p_.core); primal_ = block(p_.primal); chart_ = block(p_.chart);
        mapper_ = block(p_.mapper); direction_memory_ = block(p_.direction);
        current_=arena.take<double>(e().n); best_=arena.take<double>(e().n); selected_=arena.take<double>(e().n);
        coordinates_=arena.take<double>(e().n); mapped_=arena.take<double>(e().n); trial_=arena.take<double>(e().n);
        g_=arena.take<double>(e().n); h_=arena.take<double>(e().square);
        reduced_g_=arena.take<double>(e().d); reduced_h_=arena.take<double>(e().reduced); direction_=arena.take<double>(e().d);
        seen_=arena.take<double>(e().seen); mean_keys_=arena.take<double>(e().mean_cache); mean_values_=arena.take<double>(e().mean_cache);
        mean_evidence_=arena.take<double>(e().evidence_cache); evidence_=arena.take<double>(e().evidence);
        root_current_=arena.take<double>(e().root); root_next_=arena.take<double>(e().root);
        free_=arena.take<std::size_t>(e().d); map_status_=arena.take<int>(e().centers);
        if (!map_status_) return false;
        if (free_chart()) {
            theta_g_=arena.take<double>(e().n);
            retained_anchors_=arena.take<std::size_t>(e().k); retained_columns_=arena.take<std::size_t>(e().k);
            (void)arena.take<SelectionAlignment>(0); selector_=arena.take<std::byte>(p_.selector);
            if (!selector_) return false;
            std::copy_n(anchors_,e().k,retained_anchors_); std::copy_n(columns_,e().k,retained_columns_);
            anchors_=retained_anchors_; columns_=retained_columns_;
        }
        std::size_t free = 0;
        for (std::size_t i = 0; i < e().n; ++i) if (free_chart() || !is_selected(i)) free_[free++] = i;
        return free == e().d;
    }
    void copy_selection(std::size_t* anchors, std::size_t* columns) const noexcept {
        std::copy_n(anchors_,e().k,anchors); std::copy_n(columns_,e().k,columns);
    }
    int run(const double* initial, std::size_t steps, double* continuation, double* best,
        double* result, double* trace) noexcept {
        std::copy_n(initial,e().n,current_); std::copy_n(initial,e().n,best_);
        if (!feasible(initial) || !value(initial,current_value_)) {
            current_value_ = best_value_ = INFINITY;
            write_summary(-2,result);
            return -2;
        }
        best_value_ = current_value_;
        int stop = 0;
        for (std::size_t iteration = 0; iteration < steps; ++iteration) {
            if (free_chart()) {
                if (iteration) {
                    ++counts_.reselections;
                    const auto selected = Selector::select(layout(),current_,sample_,e().k,selector_,p_.selector);
                    if (selected.status != vlk::experiment::mean_chart::anchor_selection::Status::ready) { stop = -7; break; }
                    // A failed later selection must not corrupt the last usable indices.
                    std::copy_n(selected.anchors,e().k,retained_anchors_);
                    std::copy_n(selected.columns,e().k,retained_columns_);
                }
                if (!gradient(current_)) { stop = -6; break; }
            }
            if (!derivatives()) { stop = -3; break; }
            Mapper mapper(layout(),current_,sample_,anchors_,columns_,e().k);
            ++counts_.mapper_prepares;
            if (mapper.prepare(mapper_,p_.mapper)) { stop = -4; break; }
            selected_value_ = INFINITY; selected_slope_ = 0; seen_count_ = means_ = 0;
            selected_source_ = 0; selected_radius_ = selected_fraction_ = 0;
            if (!proposals(mapper)) { stop = -5; break; }
            bool restart = false;
            if (free_chart()) {
                // Best-value retention alone never changes the continuation state.
                if (best_value_ < selected_value_-2e-11 && best_value_ < current_value_-2e-11
                    && gradient(best_)) {
                    selected_value_ = best_value_; std::copy_n(best_,e().n,selected_);
                    selected_source_ = e().n+1; selected_slope_ = 0;
                    selected_radius_ = selected_fraction_ = 0; restart = true;
                } else if (std::isfinite(selected_value_) && !gradient(selected_)) {
                    selected_value_ = INFINITY;
                }
            }
            ++counts_.iterations;
            const bool accepted = std::isfinite(selected_value_) && selected_value_ < current_value_-2e-11;
            double* row = trace+iteration*e().trace;
            row[0]=current_value_; row[1]=selected_value_; row[2]=selected_slope_; row[3]=best_value_;
            row[4]=double(selected_source_); row[5]=selected_radius_; row[6]=selected_fraction_;
            row[7]=double(seen_count_); row[8]=double(means_); row[9]=double(counts_.values);
            row[10]=double(counts_.curvatures); row[11]=double(accepted);
            std::copy_n(accepted?selected_:current_,e().n,row+12); std::copy_n(best_,e().n,row+12+e().n);
            if (!accepted) { stop = 1; break; }
            if (restart) ++counts_.restarts;
            ++counts_.accepted; current_value_ = selected_value_; std::copy_n(selected_,e().n,current_);
        }
        std::copy_n(current_,e().n,continuation); std::copy_n(best_,e().n,best);
        write_summary(stop,result); return 0;
    }
};
} // namespace vlk::experiment::runtime::aparch::multi_phase
