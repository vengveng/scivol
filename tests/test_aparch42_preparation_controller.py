"""Held-power interface contracts, not likelihood or optimizer qualification.

Before merge set SCIVOL_APARCH_PREPARATION_NATIVE to the staged native directory
and SCIVOL_APARCH_PREPARATION_BASE to an explicitly verified sealed source root.
The standalone probe uses the real controller/maps and a quadratic objective with
scripted SQP termination. It does not compile or load the complete extension.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(os.environ.get("SCIVOL_APARCH_PREPARATION_NATIVE", ROOT / "scivol/_native"))
BASE = Path(os.environ.get("SCIVOL_APARCH_PREPARATION_BASE", ROOT))

PROBE = r'''
#include "aparch_preparation.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace rt=vlk::experiment::runtime;
namespace prep=rt::aparch::preparation;
namespace ctl=vlk::experiment::controller;
namespace kr=vlk::experiment::kraft;
namespace reuse=vlk::experiment::request_reuse;
static bool watching=false;
static std::size_t allocations=0;
void* operator new(std::size_t n) {if(watching) ++allocations;auto* p=std::malloc(n?n:1);if(!p)std::abort();return p;}
void* operator new[](std::size_t n) {return ::operator new(n);}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete[](void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
void operator delete[](void* p,std::size_t) noexcept {std::free(p);}

// Selected power rows must never evaluate a density. These linked guards abort
// if the general map fallback is entered; they do not emulate a math backend.
extern "C" int gh_shape_value_cache_init(double,double,double,gh_shape_value_cache_t*) {std::abort();}
extern "C" int vg_shape_value_cache_init(double,double,vg_shape_value_cache_t*) {std::abort();}
extern "C" int gh_bessel_value_plan_init(double,gh_bessel_value_plan_t*) {std::abort();}
extern "C" int gh_bessel_first_plan_init(double,int,gh_bessel_first_plan_t*) {std::abort();}
extern "C" int gh_log_bessel_product(const gh_bessel_product_plan_t*,double,unsigned,double*) {std::abort();}
extern "C" int gh_log_bessel_k_first(double,double,int,double*,double*,double*) {std::abort();}
extern "C" int gh_log_bessel_k_first_planned(const gh_bessel_first_plan_t*,double,double*,double*,double*) {std::abort();}
extern "C" int gh_log_bessel_k_value_planned(const gh_bessel_value_plan_t*,double,double*) {std::abort();}
extern "C" double vg_obs_nll_value_planned(double,double,const vg_shape_value_cache_t*,const gh_bessel_value_plan_t*) {std::abort();}

struct Arena {
    std::vector<unsigned char> storage;
    unsigned char* data;
    std::size_t size;
    explicit Arena(std::size_t n):storage(n+64,0xA5),size(n) {
        data=storage.data()+16;
        data+=(-reinterpret_cast<std::uintptr_t>(data))%16;
    }
    bool guards() const {
        for(auto* p=storage.data();p<data;++p) if(*p!=0xA5)return false;
        for(auto* p=data+size;p<storage.data()+storage.size();++p)if(*p!=0xA5)return false;
        return true;
    }
};
struct Counts {
    std::size_t f=0,g=0,h=0,speculation=0,events=0;
    std::uint64_t hash=14695981039346656037ull;
    void event(unsigned mask,unsigned coordinate,const double* x,std::size_t n) {
        f+=bool(mask&1);g+=bool(mask&2);h+=bool(mask&4);++events;
        hash=(hash^mask^coordinate*16)*1099511628211ull;
        for(std::size_t i=0;i<n;++i)hash=(hash^std::bit_cast<std::uint64_t>(x[i]))*1099511628211ull;
    }
};
struct Objective {
    std::size_t n,delta;
    Counts* counts;
    double target_delta;
    bool fail_h=false;
    double scale=1;
    reuse::Bank* bank=nullptr;
    std::size_t dimension() const noexcept {return n;}
    Objective scaled(double s) const noexcept {auto copy=*this;copy.scale*=s;return copy;}
    Objective with_request_reuse(reuse::Bank* b) const noexcept {auto copy=*this;copy.bank=b;return copy;}
    bool speculate_trial_gradient(std::int64_t) const noexcept {++counts->speculation;return true;}
    bool jet(unsigned mask,unsigned coordinate,const double* x,double* f,double* g,double* h) const noexcept {
        counts->event(mask,coordinate,x,n);
        if(bank && bank->get(x,mask,coordinate,{f,g,h}))return true;
        if((mask&4) && fail_h) {std::fill_n(g,n,999.);return false;}
        double value=10.;
        for(std::size_t i=0;i<n;++i) {
            const double r=x[i]-(i==delta?target_delta:0.);
            value+=.5*r*r;
            if(mask&2)g[i]=r;
        }
        if(mask&1)*f=value;
        if(mask&4) {
            std::fill_n(h,n*n,0.);
            for(std::size_t i=0;i<n;++i)h[i*n+i]=1.;
        }
        if(bank)bank->put(x,mask,coordinate,{f,g,h},0,mask);
        return true;
    }
    bool evaluate(kr::ObjectiveRequest r,const double* x,double* f,double* g,bool) const noexcept {
        const unsigned mask=r==kr::ObjectiveRequest::value?1:r==kr::ObjectiveRequest::gradient?2:3;
        if(!jet(mask,1,x,f,g,nullptr))return false;
        if(mask&1)*f*=scale;
        if(mask&2)for(std::size_t i=0;i<n;++i)g[i]*=scale;
        return true;
    }
    bool evaluate_curvature(const double* x,double* f,double* g,double* h) const noexcept {
        if(!jet(7,1,x,f,g,h))return false;
        *f*=scale;
        for(std::size_t i=0;i<n;++i)g[i]*=scale;
        for(std::size_t i=0;i<n*n;++i)h[i]*=scale;
        return true;
    }
    double scientific_kkt(const double* x,double* theta,double* g,double) const noexcept {
        std::copy_n(x,n,theta);
        if(!jet(2,0,x,nullptr,g,nullptr))return INFINITY;
        double residual=0.;
        for(std::size_t i=0;i<n;++i)residual=std::max(residual,std::abs(g[i]*scale));
        return residual;
    }
    bool boundary_reached(const double*,const double*,double) const noexcept {return false;}
};

static prep::Request active;
static rt::aparch::Layout metadata;
static unsigned bodies=0;
extern "C" void __slsqp_body(SLSQP_vars* s,double*,double*,double*,double*,double* x,
    double*,double* lower,double* upper,double*,CBLAS_INT*,ScivolNnlsWorkspace) {
    ++bodies;
    const bool coupled=active.density==1 || active.density==2;
    assert(s->m==int(coupled) && s->meq==int(coupled));
    if(coupled) {
        assert(lower[metadata.delta_index]<upper[metadata.delta_index]);
        assert(lower[metadata.model_parameters]<upper[metadata.model_parameters]);
    } else assert(lower[metadata.delta_index]==upper[metadata.delta_index]);
    if(s->mode==0) {
        s->iter=1;
        if(coupled) {
            x[metadata.model_parameters]+=.125;
            assert(prep::dispatch(active,[&]<class D,class S,class O>() {
                return rt::aparch_transform::ParameterRows<D,S,O>::retarget_power(metadata,x,active.power);
            }));
        } else x[0]*=.5;
        s->mode=1;
    } else if(s->mode==1) s->mode=-1;
    else {assert(s->mode==-1);s->mode=0;}
}

int main(int argc,char** argv) {
    assert(argc==5);
    const unsigned density=std::strtoul(argv[1],nullptr,10);
    const unsigned layout=std::strtoul(argv[2],nullptr,10);
    const unsigned scenario=std::strtoul(argv[3],nullptr,10);
    const bool raw=std::strtoul(argv[4],nullptr,10);
    prep::Request r{1,1,1,1,density,0,0,31,1.,1.,1e-12,800};
    if(layout==1){r.ar=2;r.ma=1;r.p=3;r.q=2;}
    if(layout==2){r.ar=3;r.ma=2;r.p=2;r.q=3;}
    if(layout==3){r.force_runtime_map=1;}
    if(layout==4){r.ar=0;r.ma=0;r.p=1;r.q=1;}
    r.strategy=raw?prep::Strategy::kraft_equality_or_box:prep::Strategy::box_full_controller_reuse;
    if(!prep::layout(&r,metadata)) {
        assert(!raw && (density==1 || density==2));
        assert(prep::workspace_bytes(&r)==0);
        std::puts("{\"rejected_coupled\":true}");return 0;
    }
    r.width=metadata.parameters;active=r;
    const auto n=r.width;
    std::vector<double> incoming(n,0.),lo(n,-10.),hi(n,10.),continuation(n,123.),best(n,456.),data(r.count,1.);
    incoming[0]=scenario==1?0.:.75;
    if(density==1 || density==2)incoming[metadata.model_parameters]=5.;
    assert(prep::dispatch(r,[&]<class D,class S,class O>() {
        return rt::aparch_transform::ParameterRows<D,S,O>::retarget_power(metadata,incoming.data(),2.);
    }));
    const auto bytes=prep::workspace_bytes(&r);
    Arena arena(bytes),jet(256);
    Counts counts;
    Objective inner{n,metadata.delta_index,&counts,incoming[metadata.delta_index],scenario==2};
    prep::Result result;
    const auto output_before=result;
    const auto storage_before=arena.storage;
    const auto jet_before=jet.storage;
    const auto start_before=incoming,lower_before=lo,upper_before=hi;
    bool validated=false;
    const bool invalid=scenario>=10;
    std::size_t supplied_bytes=bytes,jet_bytes=256;
    void* supplied_memory=arena.data;
    double* supplied_best=best.data();
    if(scenario==10)--supplied_bytes;
    if(scenario==11)jet_bytes=128;
    if(scenario==12)supplied_memory=arena.data+1;
    if(scenario==13)supplied_best=continuation.data();
    if(scenario==14)r.strategy=static_cast<prep::Strategy>(2);
    watching=true;
    const auto status=prep::run_checked(&r,inner,[&]{validated=true;return true;},
        incoming.data(),data.data(),lo.data(),hi.data(),supplied_memory,supplied_bytes,
        jet.data,jet_bytes,[&]{return std::size_t{256};},continuation.data(),n,supplied_best,n,&result);
    watching=false;
    assert(allocations==0 && arena.guards() && jet.guards());
    assert(incoming==start_before && lo==lower_before && hi==upper_before);
    if(invalid) {
        assert(status==-1 && !validated && counts.events==0 && bodies==0);
        assert(arena.storage==storage_before && jet.storage==jet_before);
        assert(std::all_of(continuation.begin(),continuation.end(),[](double x){return x==123.;}));
        assert(std::all_of(best.begin(),best.end(),[](double x){return x==456.;}));
        assert(std::memcmp(&result,&output_before,sizeof(result))==0);
        std::puts("{\"atomic_rejection\":true,\"allocations\":0}");return 0;
    }
    assert(status==0 && validated && result.eligible && result.equality_error<=1e-10);
    assert(result.values==counts.f && result.gradients==counts.g && result.hessians==counts.h);
    assert(result.iterations<=r.iterations);
    if(raw) {
        assert(result.hessians==0 && result.refinement_restarts==0 && result.reuse_queries==0);
        if(density==1 || density==2) {
            assert(continuation[metadata.model_parameters]!=incoming[metadata.model_parameters]);
            assert(result.constraints>0 && result.jacobians>0);
        }
    } else {
        assert(result.mode==kr::retained_point_mode && result.hessians>0);
        assert(result.refinement_restarts==2 && counts.speculation>0);
        assert(result.values==result.solver.function_evaluations+3);
        assert(result.gradients==result.solver.gradient_evaluations);
        auto direct=incoming;
        auto direct_lo=lo,direct_hi=hi;
        Arena solver(ctl::workspace_bytes(n));
        Arena cache_memory(reuse::Bank::bytes(n,prep::controller_reuse_buckets));
        reuse::Bank cache;
        assert(cache.bind(n,prep::controller_reuse_buckets,cache_memory.data,cache_memory.size));
        Counts direct_counts;
        const double scale=1./double(r.count-metadata.maximum_lag);
        auto objective=Objective{n,metadata.delta_index,&direct_counts,incoming[metadata.delta_index],scenario==2}.with_request_reuse(&cache).scaled(scale);
        double value=0.,kkt=0.;std::size_t restarts=0,curvatures=0,projections=0;
        watching=true;
        assert(objective.evaluate(kr::ObjectiveRequest::value,direct.data(),&value,nullptr,false));
        assert(prep::dispatch(r,[&]<class D,class S,class O>() {
            return rt::aparch_transform::ParameterRows<D,S,O>::retarget_power(metadata,direct.data(),r.power);
        }));
        direct_lo[metadata.delta_index]=direct_hi[metadata.delta_index]=direct[metadata.delta_index];
        assert(objective.evaluate(kr::ObjectiveRequest::value,direct.data(),&value,nullptr,false));
        const auto solved=ctl::minimize_objective(objective,direct.data(),direct_lo.data(),direct_hi.data(),
            r.accuracy,r.iterations,solver.data,solver.size,kkt,restarts,curvatures,projections);
        assert(objective.evaluate(kr::ObjectiveRequest::value,direct.data(),&value,nullptr,false));
        watching=false;
        assert(allocations==0 && solver.guards() && cache_memory.guards());
        assert(direct==continuation && solved.value==result.solver.value && solved.mode==result.solver.mode);
        assert(solved.iterations==result.iterations && kkt==result.unrestricted_coordinate_kkt);
        assert(restarts==result.refinement_restarts && curvatures==result.curvature_steps && projections==result.scientific_projection_steps);
        assert(direct_counts.hash==counts.hash && direct_counts.events==counts.events);
        assert(direct_counts.f==counts.f && direct_counts.g==counts.g && direct_counts.h==counts.h);
        assert(cache.queries()==result.reuse_queries && cache.hits()==result.reuse_hits);
        if(scenario==1) {
            assert(best==incoming && best!=continuation && result.best_value<result.continuation_value);
        }
    }
    std::printf("{\"status\":%d,\"allocations\":%zu,\"hessians\":%zu,\"restarts\":%zu,\"map_dispatch\":%zu,\"independent\":%zu}",
        status,allocations,result.hessians,result.refinement_restarts,result.map_dispatch,result.independent_power);
}
'''


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    if NATIVE.resolve() != (ROOT / "scivol/_native").resolve():
        assert "SCIVOL_APARCH_PREPARATION_BASE" in os.environ, (
            "Staged checks require an explicitly verified sealed source root; live fallback is forbidden"
        )
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++20 compiler required")
    directory = tmp_path_factory.mktemp("aparch_preparation_controller")
    source, executable = directory / "probe.cpp", directory / "probe"
    source.write_text(PROBE)
    frame = (["-Wframe-larger-than=16384", "-Werror=frame-larger-than"]
             if "clang" in Path(compiler).name else ["-Werror=frame-larger-than=16384"])
    command = [compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
               "-march=native", "-fno-exceptions", "-fno-rtti", "-fstack-usage", *frame,
               "-I", str(NATIVE), "-I", str(BASE / "scivol/_native"),
               "-I", str(BASE / "scivol/_csrc"),
               "-I", str(BASE / "scivol/_native/third_party/boost"),
               "-I", str(BASE / "scivol/_native/third_party/tlfloat"),
               str(source), "-o", str(executable)]
    built = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert built.returncode == 0, built.stdout + built.stderr
    usage = list(directory.glob("*.su"))
    assert usage, "Compiler did not emit frame evidence"
    for record in usage:
        for line in record.read_text().splitlines():
            _, size, kind = line.rsplit("\t", 2)
            assert int(size) <= 16384 and kind == "static", line
    return executable


def invoke(native, density=0, layout=0, scenario=0, raw=False):
    result = subprocess.run([str(native), str(density), str(layout), str(scenario), str(int(raw))],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("density", [0, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("layout,dispatch", [(0, 1), (1, 2), (2, 0), (3, 0), (4, 0)])
def test_full_controller_request_and_endpoint_parity(native, density, layout, dispatch):
    result = invoke(native, density, layout)
    assert result["map_dispatch"] == dispatch
    assert result["independent"] == 1 and result["allocations"] == 0


@pytest.mark.parametrize("scenario", [1, 2])
def test_retention_and_failed_curvature_preserve_controller_behavior(native, scenario):
    assert invoke(native, scenario=scenario)["restarts"] == 2


@pytest.mark.parametrize("scenario", [10, 11, 12, 13, 14])
def test_preflight_rejection_is_atomic(native, scenario):
    assert invoke(native, scenario=scenario)["atomic_rejection"]


@pytest.mark.parametrize("density", [1, 2])
def test_coupled_power_rejects_box_controller(native, density):
    assert invoke(native, density)["rejected_coupled"]


@pytest.mark.parametrize("density", range(8))
def test_legacy_raw_strategy_keeps_equality_and_free_nu(native, density):
    assert invoke(native, density, raw=True)["hessians"] == 0
