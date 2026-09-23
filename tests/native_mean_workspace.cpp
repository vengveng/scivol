// SCIVOL-NATIVE: TEST_ONLY; arbitrary generated orders and allocation-free requests.
#include "constant_mean_jet.hpp"
#include "mean_parameter_domain.hpp"
#include "garch_transform_policy.hpp"
#include <cstdlib>
#include <new>
#include <vector>

static bool forbid_allocation = false;
void* operator new(std::size_t size) {
    if (forbid_allocation) std::abort();
    if (void* result=std::malloc(size)) return result;
    std::abort();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer,std::size_t) noexcept { std::free(pointer); }

template<class Density> int check_density()
{
    using namespace vlk::experiment;
    const auto plan=mean::layout(4,2);
    const auto bytes=mean::workspace_bytes(plan,7);
    std::vector<std::byte> memory(bytes);
    std::vector<double> y(300);
    for (std::size_t i=0;i<y.size();++i) y[i]=std::sin(double(i)*.731);
    constexpr std::size_t width=7+Density::shape_parameters;
    double theta[width]={.02,.1,-.04,.02,.15,-.03,1.1};
    constexpr auto code=mean::density_code<Density>();
    if constexpr (code==1 || code==2) theta[7]=7.;
    if constexpr (code==2) theta[8]=.2;
    if constexpr (code==3) theta[7]=1.7;
    if constexpr (code>=4) { theta[7]=.12; theta[8]=code==7?2.:1.8; }
    if constexpr (code==4) theta[9]=.4;
    double f,g[width],h[width*width],f2,g2[width],h2[width*width];
    forbid_allocation=true;
    for (int repeat=0;repeat<3;++repeat) {
        if (!mean::evaluate<Density,7,mean::ArmaDesign<3>,2,7>(
                plan,{y.data(),3},theta,y.data(),y.size(),3,memory.data(),bytes,&f,g,h))
            return 1;
        if (!mean::evaluate<Density,7>(
                plan,mean::ArmaDesign<>{y.data(),3},theta,y.data(),y.size(),3,
                memory.data(),bytes,&f2,g2,h2))
            return 2;
    }
    if (std::abs(f-f2)>1e-11) return 3;
    for (std::size_t i=0;i<width;++i)
        if (std::abs(g[i]-g2[i])>1e-10+1e-11*std::abs(g[i])) return 4;
    for (std::size_t i=0;i<width*width;++i)
        if (std::abs(h[i]-h2[i])>1e-10+1e-11*std::abs(h[i])) return 5;
    if (mean::evaluate<Density,7>(
            plan,mean::ArmaDesign<>{y.data(),3},theta,y.data(),y.size(),3,
            memory.data(),bytes-1,&f,g,h)) return 6;
    if (mean::layout(std::size_t(-1),2).valid) return 7;
    forbid_allocation=false;
    return 0;
}

int check_positive_coordinate_map()
{
    using namespace vlk::experiment;
    // Independently evaluated at 60 decimal digits, then rounded to binary64.
    constexpr double points[][4] = {
        {21.0,21.000000000758256,0.999999999241744,7.582560416412863e-10},
        {-21.0,7.582560425037146e-10,7.582560422162384e-10,7.582560416412863e-10},
        {40.0,40.0,1.0,4.248354255291589e-18},
        {-40.0,4.248354255291589e-18,4.248354255291589e-18,4.248354255291589e-18},
    };
    const auto close=[](double a,double b) {return std::abs(a-b)<=3e-15*std::abs(b);};
    for (const auto& point:points) {
        const auto value=generated::softplus(point[0]);
        const auto scalar=ad::softplus(ad::Differential<0,1>::variable(point[0],0));
        const auto first=ad::softplus(ad::Differential<1,1>::variable(point[0],0));
        const auto jet=ad::softplus(ad::Differential<2,1>::variable(point[0],0));
        if (value!=scalar.value || value!=first.value || value!=jet.value
            || first.gradient[0]!=jet.gradient[0]) return 6;
        if (!close(value,point[1]) || !close(jet.value,point[1])
            || !close(jet.gradient[0],point[2]) || !close(jet.hessian[0],point[3])) return 1;
        if (!close(generated::softplus_first_from_positive(value),point[2])
            || !close(generated::softplus_second_from_positive(value),point[3])) return 2;
        if (!close(generated::inverse_softplus(value),point[0])) return 3;
    }
    for (double boundary:{-20.0,20.0}) {
        const double below=std::nextafter(boundary,-INFINITY),above=std::nextafter(boundary,INFINITY);
        if (!(generated::softplus(below)<=generated::softplus(boundary)
              && generated::softplus(boundary)<=generated::softplus(above))) return 4;
    }
    if (generated::softplus(1000.0)!=1000.0
        || generated::inverse_softplus(1000.0)!=1000.0) return 5;
    return 0;
}

int main()
{
    using namespace vlk::experiment::generated;
    return check_positive_coordinate_map()
        || check_density<NormalDensity>() || check_density<StudentTDensity>()
        || check_density<SkewTDensity>() || check_density<GedDensity>()
        || check_density<GeneralizedHyperbolicDensity>()
        || check_density<NormalInverseGaussianDensity>()
        || check_density<HyperbolicDensity>() || check_density<VarianceGammaDensity>();
}
