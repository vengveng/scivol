// SCIVOL-NATIVE: CORE; weighted coordinate derivatives, no model-domain padding.
#pragma once

#include <cmath>
#include <limits>

namespace vlk::experiment::positive_coordinate {

class ScaledLogistic {
    double coordinate_,tail_;
    mutable double range_mantissa_=NAN,centered_=NAN;
    mutable int range_exponent_=0;

    double range_product(bool exponential,double denominator,double a,double b,double c=1.) const noexcept {
        if (!std::isfinite(coordinate_) || !std::isfinite(a)
            || !std::isfinite(b) || !std::isfinite(c)) return NAN;
        if (a==0. || b==0. || c==0.) return 0.;
        int x=0,y=0,z=0;
        const double am=std::frexp(a,&x),bm=std::frexp(b,&y),cm=std::frexp(c,&z);
        double product=(am*bm)*cm;
        if (exponential) {
            // Quarter the exponent before exp can underflow. With at most three
            // binary64 factors, a zero quarter-tail implies a zero final product.
            if (std::isnan(range_mantissa_))
                range_mantissa_=std::frexp(std::exp(-std::abs(coordinate_)*.25),&range_exponent_);
            const double square=range_mantissa_*range_mantissa_;
            product=(product*square)*square;
            z+=4*range_exponent_;
        }
        return std::scalbn(product/denominator,x+y+z);
    }

    double weighted_value(bool positive,double a,double b) const noexcept {
        const double denominator=1.+tail_;
        const double probability=positive ? 1./denominator : tail_/denominator;
        const double partial=a*probability,ordinary=partial*b;
        if ((positive || tail_>=std::numeric_limits<double>::min())
            && std::isnormal(partial) && std::isnormal(ordinary))
            return ordinary;
        return range_product(!positive,denominator,a,b);
    }

public:
    explicit ScaledLogistic(double coordinate) noexcept
        :coordinate_(coordinate),tail_(std::isfinite(coordinate)?std::exp(-std::abs(coordinate)):NAN) {}

    double value(double a=1.,double b=1.) const noexcept {
        return weighted_value(coordinate_>=0.,a,b);
    }
    double complement(double a=1.,double b=1.) const noexcept {
        return weighted_value(coordinate_<0.,a,b);
    }
    double first(double a=1.,double b=1.) const noexcept {
        const double d=1.+tail_,denominator=d*d;
        const double ordinary=(a*b)*(tail_/denominator);
        if (tail_>=std::numeric_limits<double>::min() && std::isnormal(ordinary)) return ordinary;
        return range_product(true,denominator,a,b);
    }
    double second(double a=1.,double b=1.) const noexcept {
        const double d=1.+tail_;
        double denominator=d*d,factor=1.-2.*value();
        if (std::abs(coordinate_)<.25) {
            if (std::isnan(centered_)) centered_=std::expm1(-std::abs(coordinate_));
            factor=coordinate_>=0.?centered_:-centered_;
            denominator*=d;
        }
        const double ordinary=((a*b)*(tail_/denominator))*factor;
        if (tail_>=std::numeric_limits<double>::min() && std::isnormal(ordinary)) return ordinary;
        return range_product(true,denominator,a,b,factor);
    }
};

} // namespace vlk::experiment::positive_coordinate
