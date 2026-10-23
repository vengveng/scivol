#include "../../core/src/math/mean/signed_residual_germ.hpp"
#include "../../core/src/models/mean/mean_equation.hpp"
#include "../../core/src/models/mean/ged_center_composition.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#define CHECK(...) do { if (!(__VA_ARGS__)) { std::fprintf(stderr,"check failed at %d\n",__LINE__); std::abort(); } } while (false)

struct Basis {
    // Ordered active axes can include shapes/variance; the MA coefficient is 7.
    std::array<std::size_t,4> axes{12,7,2,9};
    bool depends(std::size_t i) const { for(auto x:axes)if(x==i)return true; return false; }
    std::size_t zero_factor_axis(std::size_t i) const {
        for(std::size_t a=0;a<axes.size();++a)if(axes[a]==i)return a;
        return std::size_t(-1);
    }
};
template<std::size_t Q>
void shared_traversal_cancellation() {
    using namespace vlk::experiment::mean;
    const std::size_t q=Q==dynamic ? 5 : Q;
    const std::array<std::size_t,1> axes{1};
    const auto basis=*scivolapi::detail::ActiveSensitivityBasis::select(q+3,axes);
    std::array<double,8> theta{},y{},values{},first{},second{};
    std::array<bool,8> dependent{},exact{};
    std::array<int,8> sign{};
    std::array<std::size_t,8> order{};
    std::array<unsigned char,8> odd{};
    theta[2]=.25; y[q]=1; y[q+2]=.25;
    std::fill_n(order.data(),q,SignedResidualGerm::infinite_order);
    std::fill_n(exact.data(),q,true);
    History history;
    history.values=values.data(); history.first=first.data(); history.second=second.data();
    history.slots=q; history.dependent=dependent.data(); history.germ_sign=sign.data();
    history.germ_order=order.data(); history.germ_odd=odd.data(); history.germ_exact_constant=exact.data();
    double g=0,h=0; unsigned char bit=0;
    State current; current.first=&g; current.second=&h; current.germ.odd=&bit; current.germ.width=1;
    const ArmaDesign<0> design{y.data(),0,{}};
    for(std::size_t t=q;t<q+3;++t) {
        advance<2,Q,1>(current,history,design,theta.data(),y[t],t,q,1,basis);
        retain<2,Q,1>(current,history,t,q,1);
    }
    CHECK(current.value==0 && g==0 && h==2);
    const auto evidence=current.germ.center_evidence(current.value);
    CHECK(evidence.fixed_sign==1 && evidence.order_lower==2 && evidence.odd_factors==0);
    CHECK(!current.germ.exact_constant); // m*m is not a constant despite rounded J=0.
}
void classical_center_evidence() {
    namespace mean=vlk::experiment::mean;
    namespace generated=vlk::experiment::generated;
    using Germ=mean::SignedResidualGerm;
    const double shape[]{1};
    const auto first=generated::GedDensity::prepare<false,true,false,true>(shape,1);
    const auto second=generated::GedDensity::prepare<false,false,true,true>(shape,1);
    const Germ::CenterEvidence cubic{0,3,1,true};
    CHECK(mean::classify_ged_center<false,false>(1,0,true,cubic).kind==mean::GedCenterKind::ordinary);
    const auto gradient=mean::prepared_ged_center<false,true,false,true>(first,2,std::log(2.),0,true,cubic);
    CHECK(gradient && gradient->valid && gradient->composed_residual_center);
    // x^3 and x^2*y have the same retained evidence. Only the former has C2 |r|.
    const auto hessian=mean::prepared_ged_center<false,false,true,true>(second,2,std::log(2.),0,true,cubic);
    CHECK(hessian && !hessian->valid
        && hessian->request_status==generated::DensityRequestStatus::unsupported_differentiation);
    const Germ::CenterEvidence square{1,2,0,true};
    const auto smooth=mean::prepared_ged_center<false,false,true,true>(second,2,std::log(2.),0,true,square);
    CHECK(smooth && smooth->valid && smooth->composed_residual_center && smooth->composed_residual_slope>0);
    const Germ::CenterEvidence mixed{0,2,2,true};
    const auto kink=mean::prepared_ged_center<false,false,true,true>(second,2,std::log(2.),0,true,mixed);
    CHECK(kink && !kink->valid
        && kink->request_status==generated::DensityRequestStatus::nondifferentiable);
    const double smooth_shape[]{1.5};
    const auto smooth_outer=generated::GedDensity::prepare<false,false,true,true>(smooth_shape,1);
    const auto flat=mean::prepared_ged_center<false,false,true,true>(smooth_outer,2,std::log(2.),0,true,cubic);
    CHECK(flat && flat->valid && flat->composed_residual_center && flat->composed_residual_slope==0);
}

void unresolved_sign_controls() {
    using Germ=vlk::experiment::mean::SignedResidualGerm;
    Basis basis;
    std::array<unsigned char,4> x_bits{},y_bits{},sum_bits{};
    Germ x{Germ::unknown,0,x_bits.data(),4}, y{Germ::unknown,0,y_bits.data(),4}, sum{Germ::unknown,0,sum_bits.data(),4};
    x.reset(0); x.subtract_product(0,7,1,1,nullptr,basis); x.finish(0); // -x
    y.reset(0); y.subtract_product(0,2,-1,-1,nullptr,basis); y.finish(0); // +y
    sum.reset(.25);
    sum.subtract_product(0,7,0,x.sign,x.odd,basis,x.order_lower); // +x*x
    sum.subtract_product(0,2,0,y.sign,y.odd,basis,y.order_lower); // -y*y
    sum.subtract_product(.25,0,1,1,nullptr,basis,0,true);
    sum.finish(0);
    CHECK(sum.order_lower==2 && !sum.center_evidence(0).known_leading_sign
        && sum.fixed_sign_at_zero(0)==0 && !sum.exact_constant);
    // Opposite variable expressions are not folded into a convenient zero.
    sum.reset(0);
    sum.subtract_product(1,0,0,x.sign,x.odd,basis,x.order_lower);
    sum.subtract_product(-1,0,0,x.sign,x.odd,basis,x.order_lower);
    sum.finish(0);
    CHECK(!sum.center_evidence(0).known_leading_sign && !sum.exact_constant);
}

int main() {
    using Germ=vlk::experiment::mean::SignedResidualGerm;
    Basis basis;
    std::array<unsigned char,4> odd1{},odd2{},odd3{};
    Germ first{Germ::unknown,0,odd1.data(),4},second{Germ::unknown,0,odd2.data(),4},third{Germ::unknown,0,odd3.data(),4};
    first.reset(1); first.finish(1);
    second.reset(0); second.subtract_product(0,7,1,first.sign,first.odd,basis,first.order_lower);
    second.finish(0);
    CHECK(second.fixed_sign_at_zero(0)==0); // -m crosses zero.
    third.reset(0); third.subtract_product(0,7,0,second.sign,second.odd,basis,second.order_lower);
    third.finish(0);
    CHECK(third.fixed_sign_at_zero(0)==1); // m*m is nonnegative.
    CHECK(third.center_evidence(0).order_lower==2);
    second.reset(0); second.subtract_product(0,2,1,first.sign,first.odd,basis,first.order_lower);
    second.finish(0);
    third.reset(0); third.subtract_product(0,7,0,second.sign,second.odd,basis,second.order_lower);
    third.finish(0);
    CHECK(third.fixed_sign_at_zero(0)==0); // Different free factors m*n.
    CHECK(third.center_evidence(0).known_leading_sign && third.center_evidence(0).odd_factors==2
        && third.center_evidence(0).order_lower==2);
    first.reset(-1); first.finish(-1);
    second.reset(0); second.subtract_product(0,7,-1,first.sign,first.odd,basis,first.order_lower);
    second.finish(0);
    third.reset(0); third.subtract_product(0,7,0,second.sign,second.odd,basis,second.order_lower);
    third.finish(0);
    CHECK(third.fixed_sign_at_zero(0)==-1); // -m*m is nonpositive.
    first.reset(.25);
    first.subtract_product(.25,0,1,1,nullptr,basis,0,true);
    first.finish(0);
    CHECK(first.sign==0 && first.exact_constant && first.order_lower==Germ::infinite_order);
    // Fixed offsets cancel around either signed square and around a mixed product.
    for(double seed:{1.,-1.}) {
        first.reset(seed); first.finish(seed);
        second.reset(0); second.subtract_product(0,7,seed,first.sign,first.odd,basis,0);
        second.finish(0);
        third.reset(.25);
        third.subtract_product(0,7,0,second.sign,second.odd,basis,second.order_lower);
        third.subtract_product(.25,0,1,1,nullptr,basis,0,true);
        third.finish(0);
        CHECK(third.fixed_sign_at_zero(0)==int(seed) && third.order_lower==2);
    }
    first.reset(1); first.finish(1);
    second.reset(0); second.subtract_product(0,2,1,first.sign,first.odd,basis,0); second.finish(0);
    third.reset(.25); third.subtract_product(0,7,0,second.sign,second.odd,basis,1);
    third.subtract_product(.25,0,1,1,nullptr,basis,0,true); third.finish(0);
    CHECK(third.center_evidence(0).odd_factors==2 && third.order_lower==2);
    third.reset(.25); third.subtract_product(0,7,1,1,nullptr,basis,0);
    third.subtract_product(.25,0,1,1,nullptr,basis,0,true); third.finish(0);
    CHECK(third.center_evidence(0).odd_factors==1 && third.order_lower==1);
    // No attempt to cancel variable expressions or unrepresentable fixed proofs.
    first.reset(0);
    first.subtract_product(.1,0,.1,1,nullptr,basis,0,true);
    first.subtract_product(-.1,0,.1,1,nullptr,basis,0,true); first.finish(0);
    CHECK(!first.center_evidence(0).known_leading_sign && !first.exact_constant);
    // A rounded zero is not a mathematical zero, including exact tiny products.
    first.reset(.1); first.subtract_product(.1,0,.1,1,nullptr,basis,0,true); first.finish(0);
    CHECK(!first.fixed.valid && !first.exact_constant && !first.center_evidence(0).known_leading_sign);
    first.reset(0);
    const auto tiny=std::numeric_limits<double>::denorm_min();
    first.subtract_product(tiny,0,tiny,1,nullptr,basis,0,true); first.finish(0);
    CHECK(first.fixed.valid && first.fixed.magnitude && !first.exact_constant);
    CHECK(!first.center_evidence(0).known_leading_sign && first.fixed_sign_at_zero(0)==0);
    auto gap=vlk::experiment::mean::FixedResidualDyadic::decode(1);
    gap.subtract(vlk::experiment::mean::FixedResidualDyadic::decode(0x1p-64));
    CHECK(!gap.valid);
    auto fixed=vlk::experiment::mean::FixedResidualDyadic::decode(1);
    fixed.subtract(vlk::experiment::mean::FixedResidualDyadic::decode(.5));
    fixed.subtract(vlk::experiment::mean::FixedResidualDyadic::decode(.5));
    CHECK(fixed.equals(0) && fixed.equals(-0.));
    fixed=vlk::experiment::mean::FixedResidualDyadic::decode(-.25);
    fixed.subtract(vlk::experiment::mean::FixedResidualDyadic::decode(-.5));
    CHECK(fixed.equals(.25));
    fixed=vlk::experiment::mean::FixedResidualDyadic::decode(0x1p63);
    fixed.subtract(vlk::experiment::mean::FixedResidualDyadic::decode(-1));
    CHECK(fixed.valid);
    fixed.subtract(vlk::experiment::mean::FixedResidualDyadic::decode(-0x1p63));
    CHECK(!fixed.valid); // 2^64+1 cannot fit a normalized uint64 significand.
    CHECK(!vlk::experiment::mean::FixedResidualDyadic::decode(std::numeric_limits<double>::infinity()).valid);
    CHECK(!vlk::experiment::mean::FixedResidualDyadic::decode(std::numeric_limits<double>::quiet_NaN()).valid);
    shared_traversal_cancellation<2>();
    shared_traversal_cancellation<3>();
    shared_traversal_cancellation<vlk::experiment::mean::dynamic>();
    classical_center_evidence();
    unresolved_sign_controls();
}
