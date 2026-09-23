// Independent workspace, KKT and reentrancy checks for the shared NNLS backend.
#include "runtime_kraft_slsqp.hpp"
#include <cassert>
#include <thread>
#include <vector>

using namespace vlk::experiment;

void solve_diagonal(std::size_t n,bool duplicate)
{
    const auto columns=duplicate?2*n:n;
    std::vector<double> a(n*columns,0.0),b(n),x(columns),dual(columns),residual(n);
    std::vector<CBLAS_INT> passive(columns);
    for (std::size_t i=0;i<n;++i) {
        a[i+i*n]=1.0;
        if (duplicate) a[i+(n+i)*n]=1.0;
        b[i]=(i%2?-1.0:1.0)*(1+i)*0.125;
    }
    const auto bytes=nnls::workspace_bytes(n,columns);
    assert(bytes);
    std::vector<double> arena((bytes+7)/8+2,919.0);
    std::fill(arena.begin()+1,arena.end()-1,NAN);
    double norm=0;
    std::int64_t mode=0;
    __nnls(n,columns,a.data(),b.data(),x.data(),dual.data(),residual.data(),
        passive.data(),0,&norm,&mode,{arena.data()+1,bytes});
    assert(mode==1 && arena.front()==919.0 && arena.back()==919.0);
    double squared=0;
    for (std::size_t i=0;i<n;++i) {
        const auto fitted=x[i]+(duplicate?x[n+i]:0.0);
        assert(std::abs(fitted-std::max(0.0,b[i]))<1e-13);
        assert(std::abs(residual[i]-(b[i]-fitted))<1e-13);
        squared+=residual[i]*residual[i];
    }
    assert(std::abs(norm-std::sqrt(squared))<1e-12);
    for (std::size_t i=0;i<columns;++i) {
        assert(x[i]>=0 && dual[i]<=1e-13);
        assert(std::abs(x[i]*dual[i])<1e-12);
    }
}

void reject_invalid_workspace()
{
    constexpr auto bytes=nnls::workspace_bytes(2,2);
    static_assert(bytes>0);
    alignas(double) std::array<std::byte,bytes+8> arena;
    double a[4]={1,0,0,1},b[2]={1,2},x[2],dual[2],residual[2],norm;
    CBLAS_INT passive[2];
    for (unsigned failure=0;failure<4;++failure) {
        std::fill_n(x,2,717.0);
        std::fill_n(dual,2,617.0);
        std::fill_n(residual,2,517.0);
        std::fill_n(passive,2,417);
        std::int64_t mode;
        const ScivolNnlsWorkspace work{
            failure==1?nullptr:arena.data()+(failure==2?1:0),
            failure==0?bytes-1:bytes};
        __nnls(2,2,a,b,x,dual,residual,passive,
            failure==3?std::numeric_limits<std::int64_t>::max():0,&norm,&mode,work);
        assert(mode==2);
        for (unsigned i=0;i<2;++i)
            assert(x[i]==717 && dual[i]==617 && residual[i]==517 && passive[i]==417);
    }
    assert(nnls::workspace_bytes(0,1)==0);
    assert(nnls::workspace_bytes(CBLAS_INT_MAX,2)==0);
    assert(nnls::sqp_workspace_bytes(std::numeric_limits<std::size_t>::max(),1)==0);
}

int main()
{
    reject_invalid_workspace();
    for (auto n : {1u,2u,5u,32u,64u,128u}) {
        solve_diagonal(n,false);
        solve_diagonal(n,true);
    }
    std::array<std::thread,4> workers;
    for (auto& worker:workers) worker=std::thread([] {
        for (unsigned i=0;i<10;++i) solve_diagonal(19,true);
    });
    for (auto& worker:workers) worker.join();
}
