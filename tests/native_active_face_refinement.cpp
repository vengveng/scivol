// Generic active-face math and callback contracts, independent of model kernels.
#include "active_face_refinement.hpp"
#include "native_active_face_phase.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>
#include <vector>

namespace face = vlk::experiment::curvature;
namespace kraft = vlk::experiment::kraft;

static std::size_t allocations = 0;
void* operator new(std::size_t bytes)
{
    ++allocations;
    if (void* memory = std::malloc(bytes ? bytes : 1)) return memory;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

namespace {

std::vector<double> read_values(std::size_t count)
{
    std::vector<double> values(count);
    for (double& value : values) {
        std::string token;
        assert(bool(std::cin >> token));
        char* end = nullptr;
        value = std::strtod(token.c_str(), &end);
        assert(end && *end == '\0');
    }
    return values;
}

void number(double value)
{
    if (std::isfinite(value)) std::cout << value;
    else std::cout << '"' << (std::isnan(value) ? "nan" : value < 0 ? "-inf" : "inf") << '"';
}

template<class T>
void array(const T* values, std::size_t count)
{
    std::cout << '[';
    for (std::size_t i = 0; i < count; ++i) {
        if (i) std::cout << ',';
        if constexpr (std::is_floating_point_v<T>) number(values[i]);
        else std::cout << values[i];
    }
    std::cout << ']';
}

struct GuardedArena {
    std::size_t bytes;
    std::vector<double> values;
    explicit GuardedArena(std::size_t size) : bytes(size), values((size + 7) / 8 + 2, 817.) {}
    void* data() { return values.data() + 1; }
    void check() const { assert(values.front() == 817. && values.back() == 817.); }
};

struct CallbackAudit {
    std::size_t objective[8]{}, constraints[3]{}, curvature = 0;
    std::size_t protocol_errors = 0, noninitial_gradients = 0;
    double minimum_scored_slack = std::numeric_limits<double>::infinity();
};

// f = 3.25 + (x-center)' H (x-center)/2 + linear'x. Constraints are
// affine with an optional strict first row, a strict disk, or x0-x1^2 > 0.
struct AnalyticModel {
    std::size_t n, m;
    unsigned geometry, failure;
    double scale;
    const double *hessian, *center, *linear, *units, *offsets, *normals, *initial;
    CallbackAudit* audit;

    std::size_t dimension() const noexcept { return n; }
    std::size_t count() const noexcept { return m; }
    double objective_scale() const noexcept { return scale; }

    bool at_initial(const double* x) const noexcept
    {
        for (std::size_t i = 0; i < n; ++i) if (x[i] != initial[i]) return false;
        return true;
    }

    double constraint_value(std::size_t row, const double* x) const noexcept
    {
        if (geometry == 1 && row == 0) return offsets[0] - x[0]*x[0] - x[1]*x[1];
        if (geometry == 2 && row == 0) return x[0] - x[1]*x[1];
        double value = offsets[row];
        for (std::size_t i = 0; i < n; ++i) value += normals[row*n+i]*x[i];
        return value;
    }

    bool objective(unsigned mask, const double* x, double* f, double* g, double* h) const noexcept
    {
        ++audit->objective[mask];
        audit->protocol_errors += bool(f) != bool(mask & 1);
        audit->protocol_errors += bool(g) != bool(mask & 2);
        audit->protocol_errors += bool(h) != bool(mask & 4);
        const bool initial_point = at_initial(x);
        if ((mask & 1) && ((failure == 19 && !initial_point) || failure == 20)) return false;
        if (mask & 2) audit->noninitial_gradients += !initial_point;
        if ((mask & 2) && ((failure == 1 && !initial_point) || failure == 6)) return false;
        if ((mask & 4) && failure == 2) return false;
        if (mask & 1) {
            for (std::size_t row = 0; row < m; ++row)
                audit->minimum_scored_slack = std::min(audit->minimum_scored_slack, constraint_value(row, x));
            *f = 3.25;
            for (std::size_t i = 0; i < n; ++i) {
                *f += linear[i]*x[i];
                for (std::size_t j = 0; j < n; ++j)
                    *f += .5*(x[i]-center[i])*hessian[i*n+j]*(x[j]-center[j]);
            }
            if ((failure == 3 && !initial_point) || failure == 7) *f = NAN;
            // A finite uphill F must be rejected without requesting G.
            if (failure == 15 && !initial_point) *f += 1000.;
        }
        if (mask & 2) for (std::size_t i = 0; i < n; ++i) {
            g[i] = linear[i];
            for (std::size_t j = 0; j < n; ++j) g[i] += hessian[i*n+j]*(x[j]-center[j]);
            if (failure == 16) g[i] = NAN;
        }
        if (mask & 4) {
            std::copy_n(hessian, n*n, h);
            if (failure == 10) h[0] = NAN;
        }
        return true;
    }

    bool constraints(kraft::ConstraintRequest request, const double* x, double* c, double* j) const noexcept
    {
        const unsigned index = request == kraft::ConstraintRequest::value ? 0 :
            request == kraft::ConstraintRequest::jacobian ? 1 : 2;
        ++audit->constraints[index];
        if (failure == 4 && index != 0) return false;
        if (failure == 18 && index != 0 && !at_initial(x)) return false;
        if (index != 1 && (failure == 21 || (failure == 22 && !at_initial(x)))) return false;
        for (std::size_t row = 0; row < m; ++row) {
            if (index != 1) {
                c[row] = constraint_value(row, x);
                if (failure == 8 || (failure == 14 && !at_initial(x))) c[row] = NAN;
            }
            if (index != 0) for (std::size_t i = 0; i < n; ++i) {
                double derivative = normals[row*n+i];
                if (row == 0 && geometry == 1) derivative = i < 2 ? -2*x[i] : 0.;
                if (row == 0 && geometry == 2) derivative = i == 0 ? 1. : i == 1 ? -2*x[1] : 0.;
                j[row+i*m] = failure == 9 ? NAN : derivative;
            }
        }
        return true;
    }

    bool feasible(const double*, const double* c) const noexcept
    {
        for (std::size_t row = 0; row < m; ++row)
            if (geometry && row == 0 ? !(c[row] > 0) : !(c[row] >= 0)) return false;
        return true;
    }

    bool parameter_units(const double*, double* output) const noexcept
    {
        if (failure == 23) return false;
        std::copy_n(units, n, output);
        if (failure == 11) output[0] = 0.;
        if (failure == 17) output[0] = NAN;
        return true;
    }

    bool constraint_curvature(const double*, const double* weights, double* h) const noexcept
    {
        ++audit->curvature;
        std::fill_n(h, n*n, 0.);
        if (m && geometry == 1) h[0] = h[n+1] = -2*weights[0];
        if (m && geometry == 2) h[n+1] = -2*weights[0];
        if (failure == 12) h[0] = NAN;
        return failure != 5;
    }
};

void phase()
{
    std::size_t n, m;
    unsigned geometry, failure, backtracks, corrections;
    long short_by;
    assert(bool(std::cin >> n >> m >> geometry >> failure >> backtracks >> corrections >> short_by));
    const auto settings = read_values(2);
    const auto hessian = read_values(n*n), center = read_values(n), linear = read_values(n);
    const auto units = read_values(n), offsets = read_values(m), normals = read_values(m*n);
    auto point = read_values(n);
    const auto initial = point, lower = read_values(n), upper = read_values(n);
    std::vector<double> best(n, 771.);
    CallbackAudit audit;
    AnalyticModel model{n,m,geometry,failure,settings[0],hessian.data(),center.data(),
        linear.data(),units.data(),offsets.data(),normals.data(),initial.data(),&audit};
    GuardedArena arena(face::refinement_workspace_bytes(n,m));
    const auto before = allocations;
    const auto result = face::refine_active_face(model,point.data(),best.data(),lower.data(),upper.data(),
        {backtracks,corrections,settings[1]},arena.data(),arena.bytes-short_by);
    const auto allocated = allocations-before;
    arena.check();
    assert(!allocated && !audit.protocol_errors);
    for (unsigned i = 0; i < 8; ++i) assert(result.objective_requests[i] == audit.objective[i]);
    for (unsigned i = 0; i < 3; ++i) assert(result.constraint_requests[i] == audit.constraints[i]);
    assert(result.curvature_requests == audit.curvature);
    std::cout << "{\"status\":" << int(result.status) << ",\"point\":";
    array(point.data(),n); std::cout << ",\"best\":"; array(best.data(),n);
    std::cout << ",\"evidence\":";
    const double evidence[]{result.value,result.bound,result.best_value,result.best_bound};
    array(evidence,4);
    std::cout << ",\"best_valid\":" << result.best_valid << ",\"best_accepted\":" << result.best_is_accepted;
    std::cout << ",\"requests\":"; array(result.objective_requests,8);
    std::cout << ",\"constraints\":"; array(result.constraint_requests,3);
    std::cout << ",\"curvature\":" << result.curvature_requests << ",\"projections\":" << result.projections
        << ",\"projection_iterations\":" << result.projection_iterations << ",\"trials\":" << result.trials
        << ",\"corrections\":" << result.corrections << ",\"active\":" << result.active
        << ",\"resolved\":" << result.resolved << ",\"allocations\":" << allocated
        << ",\"noninitial_gradients\":" << audit.noninitial_gradients << ",\"minimum_scored_slack\":";
    number(audit.minimum_scored_slack); std::cout << "}\n";
}

void direction_or_correction(bool correction)
{
    std::size_t n, rows;
    long short_by;
    assert(bool(std::cin >> n >> rows >> short_by));
    const auto matrix = read_values(correction ? 0 : n*n);
    const auto gradient = read_values(correction ? rows : n), normals = read_values(rows*n);
    std::vector<double> output(n+2,717.), eigenvalues(n,617.);
    GuardedArena arena(face::active_face_workspace_bytes(n));
    std::size_t resolved = 999;
    const auto before = allocations;
    const bool valid = correction ? face::normal_correction(n,rows,normals.data(),gradient.data(),
        output.data()+1,arena.data(),arena.bytes-short_by) :
        face::active_face_direction(n,rows,normals.data(),matrix.data(),gradient.data(),output.data()+1,
            eigenvalues.data(),resolved,arena.data(),arena.bytes-short_by);
    assert(allocations == before);
    arena.check(); assert(output.front() == 717. && output.back() == 717.);
    std::cout << "{\"valid\":" << valid << ",\"output\":";
    array(output.data()+1,n); std::cout << ",\"eigenvalues\":"; array(eigenvalues.data(),n);
    std::cout << ",\"resolved\":" << resolved << "}\n";
}

void projection_or_stationarity(bool stationarity)
{
    std::size_t n, rows;
    long short_by;
    assert(bool(std::cin >> n >> rows >> short_by));
    const auto gradient = read_values(n), lower = read_values(stationarity ? n : 0);
    const auto upper = read_values(stationarity ? n : 0), normals = read_values(rows*n), gaps = read_values(rows);
    std::vector<double> point(n+2,717.), dual(rows+2,617.);
    GuardedArena arena(stationarity ? face::stationarity_workspace_bytes(n,rows) : face::projection_workspace_bytes(n,rows));
    const auto before = allocations;
    if (stationarity) {
        face::ProjectionGeometry geometry;
        const auto result = face::stationarity_bound(n,rows,gradient.data(),lower.data(),upper.data(),
            normals.data(),gaps.data(),arena.data(),arena.bytes-short_by,&geometry);
        assert(allocations == before); arena.check();
        std::cout << "{\"bound\":"; number(result.bound);
        std::cout << ",\"projected\":" << result.projected << ",\"dimension\":" << geometry.dimension
            << ",\"rows\":" << geometry.rows << ",\"gaps\":";
        array(geometry.gaps,geometry.rows); std::cout << ",\"identities\":";
        array(geometry.identities,geometry.rows); std::cout << "}\n";
    } else {
        const auto result = face::projection_multipliers(n,rows,gradient.data(),normals.data(),gaps.data(),
            point.data()+1,dual.data()+1,arena.data(),arena.bytes-short_by);
        const double bound = face::primal_dual_bound(n,rows,gradient.data(),normals.data(),gaps.data(),
            point.data()+1,dual.data()+1);
        assert(allocations == before); arena.check();
        assert(point.front() == 717. && point.back() == 717. && dual.front() == 617. && dual.back() == 617.);
        std::cout << "{\"mode\":" << result.mode << ",\"point\":";
        array(point.data()+1,n); std::cout << ",\"dual\":"; array(dual.data()+1,rows);
        std::cout << ",\"bound\":"; number(bound); std::cout << "}\n";
    }
}

void certificate()
{
    std::size_t n, rows;
    assert(bool(std::cin >> n >> rows));
    const auto gradient=read_values(n), normals=read_values(rows*n), gaps=read_values(rows);
    const auto point=read_values(n), dual=read_values(rows);
    const auto before=allocations;
    const double bound=face::primal_dual_bound(n,rows,gradient.data(),normals.data(),
        gaps.data(),point.data(),dual.data());
    assert(allocations==before);
    std::cout << "{\"bound\":"; number(bound); std::cout << "}\n";
}

void conditioning()
{
    std::size_t n, rows;
    assert(bool(std::cin >> n >> rows));
    const auto hessian = read_values(n*n), gradient = read_values(n), normals = read_values(rows*n);
    auto units = read_values(n);
    std::vector<double> h(n*n), g(n), a(rows*n), exponents(n);
    const auto before = allocations;
    const bool valid = face::condition_curvature(n,rows,hessian.data(),gradient.data(),normals.data(),
        units.data(),h.data(),g.data(),a.data(),exponents.data());
    assert(allocations == before);
    std::cout << "{\"valid\":" << valid << ",\"units\":"; array(units.data(),n);
    std::cout << ",\"hessian\":"; array(h.data(),n*n);
    std::cout << ",\"gradient\":"; array(g.data(),n);
    std::cout << ",\"normals\":"; array(a.data(),rows*n); std::cout << "}\n";
}

void input_guards()
{
    const auto maximum = std::numeric_limits<std::size_t>::max();
    for (auto n : {std::size_t(0),maximum}) {
        assert(!face::active_face_workspace_bytes(n));
        assert(!face::refinement_workspace_bytes(n,1));
        assert(!face::projection_workspace_bytes(n,1));
        assert(!face::stationarity_workspace_bytes(n,1));
    }
    assert(!face::refinement_workspace_bytes(3,maximum));
    assert(!face::projection_workspace_bytes(3,maximum));
    assert(!face::stationarity_workspace_bytes(3,maximum));
    GuardedArena arena(face::refinement_workspace_bytes(3,0));
    const double hessian[]{1,0,0,0,1,0,0,0,1}, center[]{-1,2,0}, linear[]{0,0,0}, units[]{1,2,.5};
    double lower[]{0,-INFINITY,-INFINITY}, upper[]{INFINITY,INFINITY,INFINITY};
    double point[]{0,1,3}, best[]{771,771,771};
    CallbackAudit audit;
    AnalyticModel model{3,0,0,0,.25,hessian,center,linear,units,nullptr,nullptr,point,&audit};
    const auto before = allocations;
    for (unsigned failure = 0; failure < 15; ++failure) {
        auto options = face::RefinementOptions{};
        model.scale = .25;
        lower[0] = 0.; upper[0] = INFINITY;
        point[0] = 0.;
        if (failure == 4) model.scale = 0.;
        if (failure == 5) model.scale = NAN;
        if (failure == 6) options.stationarity_target = -1.;
        if (failure == 7) options.stationarity_target = NAN;
        if (failure == 8) options.backtracks = 1025;
        if (failure == 9) lower[0] = NAN;
        if (failure == 10) lower[0] = 1., upper[0] = 0.;
        if (failure == 11) point[0] = NAN;
        const auto result = face::refine_active_face(model,failure == 0 ? nullptr : point,
            failure == 1 ? nullptr : failure == 2 ? point : best,
            failure == 12 ? nullptr : lower,failure == 13 ? nullptr : upper,options,
            failure == 3 ? nullptr : failure == 14 ? static_cast<std::byte*>(arena.data())+1 : arena.data(),
            arena.bytes);
        assert(result.status == face::RefinementStatus::invalid_input && !result.best_valid);
        assert(best[0] == 771 && best[1] == 771 && best[2] == 771);
    }
    assert(allocations == before && !audit.objective[1] && !audit.constraints[0]);
    arena.check();
    std::cout << "{\"guards\":15,\"allocations\":0}\n";
}

} // namespace

int main(int argc, char** argv)
{
    assert(argc == 2);
    std::cout.precision(17);
    const std::string command(argv[1]);
    if (command == "phase") phase();
    else if (command == "direction") direction_or_correction(false);
    else if (command == "correction") direction_or_correction(true);
    else if (command == "projection") projection_or_stationarity(false);
    else if (command == "stationarity") projection_or_stationarity(true);
    else if (command == "certificate") certificate();
    else if (command == "conditioning") conditioning();
    else if (command == "guards") input_guards();
    else return 2;
}
