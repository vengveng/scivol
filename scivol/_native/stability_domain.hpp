// SCIVOL-NATIVE: mean-model mathematics; promotion gates tracked in MODELS.md.
#ifndef SCIVOL_MEAN_STABILITY_DOMAIN_HPP
#define SCIVOL_MEAN_STABILITY_DOMAIN_HPP

#include "runtime_reflection.hpp"
#include "runtime_workspace.hpp"

#if defined(_MSC_VER)
#define VLK_SCHUR_ROW_INLINE [[msvc::forceinline]]
#else
#define VLK_SCHUR_ROW_INLINE __attribute__((always_inline))
#endif

namespace vlk::experiment::mean {

// Coefficients of the AR polynomial are basis * parameters. MA uses its negative.
struct PolynomialDomain {
    std::size_t order;
    std::size_t parameters;
    const double* basis;
    double identity_scale = 1.0;

    void prepare_coefficients(const double* theta,double* coefficients,
                              double* jacobian=nullptr) const noexcept
    {
        if (jacobian && !basis) std::fill_n(jacobian,order*parameters,0.0);
        for (std::size_t row=0;row<order;++row) {
            double value=0;
            if (basis) {
                for (std::size_t column=0;column<parameters;++column) {
                    const double b=basis[row*parameters+column];
                    value+=b*theta[column];
                    if (jacobian) jacobian[row*parameters+column]=b;
                }
            } else {
                value=identity_scale*theta[row];
                if (jacobian) jacobian[row*parameters+row]=identity_scale;
            }
            coefficients[row]=value;
        }
    }

    std::size_t workspace_bytes(bool derivatives) const noexcept
    {
        if (!order || !parameters || (!basis && order!=parameters))
            return 0;
        std::size_t count;
        if (!runtime::checked_product(order, parameters, count))
            return 0;
        const auto work = derivatives
            ? runtime::reflection_inverse_scratch_doubles(order, parameters)
            : runtime::reflection_inverse_scratch_doubles(order, 0);
        if (!work)
            return 0;
        runtime::WorkspaceSizer size;
        size.add<double>(order);
        size.add<double>(order);
        if (derivatives)
            size.add<double>(count);
        size.add<double>(work);
        return size.size();
    }

    bool evaluate(const double* theta, double* constraints, double* jacobian,
                  void* memory, std::size_t bytes, bool require_stable) const noexcept
    {
        const auto required = workspace_bytes(jacobian != nullptr);
        if (!required || !theta || (!constraints && !jacobian) || !memory || bytes < required)
            return false;
        runtime::WorkspaceArena arena(memory,bytes);
        double* coefficients = arena.take<double>(order);
        double* reflection = arena.take<double>(order);
        double* local_jacobian = jacobian ? arena.take<double>(order*parameters) : nullptr;
        double* work = arena.take<double>(jacobian
            ? runtime::reflection_inverse_scratch_doubles(order,parameters)
            : runtime::reflection_inverse_scratch_doubles(order,0));
        if (!work)
            return false;
        prepare_coefficients(theta,coefficients);
        const bool valid = !jacobian
            ? runtime::reflection_inverse_values(coefficients,order,1.0,
                reflection,work,work+order,require_stable)
            : basis ? runtime::reflection_inverse_composed(coefficients,order,basis,parameters,
                reflection,local_jacobian,work,require_stable)
            : require_stable ? runtime::reflection_inverse(coefficients,order,
                reflection,local_jacobian,work)
            : runtime::reflection_inverse_unrestricted(coefficients,order,
                reflection,local_jacobian,work);
        if (!valid)
            return false;
        for (std::size_t row=0; row<order; ++row) {
            if (constraints) {
                constraints[2*row]=1.0-reflection[row];
                constraints[2*row+1]=1.0+reflection[row];
            }
            if (jacobian) {
                for (std::size_t column=0; column<parameters; ++column) {
                    const double value=local_jacobian[row*parameters+column]
                        * (basis ? 1.0 : identity_scale);
                    if (!std::isfinite(value))
                        return false;
                    jacobian[(2*row)*parameters+column]=-value;
                    jacobian[(2*row+1)*parameters+column]=value;
                }
            }
        }
        return true;
    }

    std::size_t curvature_workspace_bytes() const noexcept
    {
        std::size_t next,product,gradients,hessians;
        if (!order || !parameters || (!basis && order!=parameters)
            || !runtime::checked_sum(parameters,1,next)
            || !runtime::checked_product(parameters,next,product)
            || !runtime::checked_product(order,parameters,gradients)
            || !runtime::checked_product(order,product/2,hessians)) return 0;
        runtime::WorkspaceSizer size;
        for (unsigned slot=0;slot<2;++slot) {
            size.add<double>(order);
            size.add<double>(gradients);
            size.add<double>(hessians);
        }
        return size.size();
    }

    // Sum weights[i] * Hessian(c_i), in row-major parameter coordinates.
    // Value/Jacobian requests never allocate or traverse this second-order state.
    bool weighted_curvature(const double* theta,const double* weights,double* output,
                            void* memory,std::size_t bytes) const noexcept
    {
        const auto required=curvature_workspace_bytes(),n=order,p=parameters;
        if (!required || bytes<required || !theta || !weights || !output || !memory)
            return false;
        const auto packed=p*(p+1)/2;
        runtime::WorkspaceArena arena(memory,bytes);
        double* value=arena.take<double>(n);
        double* gradient=arena.take<double>(n*p);
        double* hessian=arena.take<double>(n*packed);
        double* next_value=arena.take<double>(n);
        double* next_gradient=arena.take<double>(n*p);
        double* next_hessian=arena.take<double>(n*packed);
        if (!next_hessian) return false;
        for (std::size_t j=0;j<p;++j) if (!std::isfinite(theta[j])) return false;
        for (std::size_t row=0;row<n;++row)
            if (!std::isfinite(weights[2*row]) || !std::isfinite(weights[2*row+1]))
                return false;
        prepare_coefficients(theta,value,gradient);
        for (std::size_t i=0;i<n*p;++i) if (!std::isfinite(gradient[i])) return false;
        std::fill_n(hessian,n*packed,0.0);
        std::fill_n(output,p*p,0.0);
        const auto index=[p](std::size_t i,std::size_t j) {return i*p-i*(i+1)/2+j;};
        for (std::size_t stage=n;stage>0;--stage) {
            const auto last=stage-1;
            const double k=value[last];
            if (!std::isfinite(k)) return false;
            const double weight=weights[2*last+1]-weights[2*last];
            for (std::size_t i=0;i<p;++i)
                for (std::size_t j=i;j<p;++j)
                    output[i*p+j]+=weight*hessian[last*packed+index(i,j)];
            if (!last) break;
            const runtime::reflection_detail::InverseSchurStep step(k);
            if (!step.valid(last)) return false;
            for (std::size_t row=0;row<last;++row) {
                const auto reverse=last-1-row;
                const auto evaluate_row=[&](const auto& equation) VLK_SCHUR_ROW_INLINE -> bool {
                    if (!std::isfinite(equation.value)) return false;
                    next_value[row]=equation.value;
                    for (std::size_t i=0;i<p;++i) {
                        next_gradient[row*p+i]=equation.first(gradient[row*p+i],
                            gradient[reverse*p+i],gradient[last*p+i]);
                        if (!std::isfinite(next_gradient[row*p+i])) return false;
                    }
                    for (std::size_t i=0;i<p;++i) for (std::size_t j=i;j<p;++j) {
                        const auto ij=index(i,j);
                        const double bij=equation.second(hessian[row*packed+ij],
                            hessian[reverse*packed+ij],gradient[reverse*p+i],
                            gradient[reverse*p+j],gradient[last*p+i],gradient[last*p+j],
                            hessian[last*packed+ij],next_gradient[row*p+i],next_gradient[row*p+j]);
                        if (!std::isfinite(bij)) return false;
                        next_hessian[row*packed+ij]=bij;
                    }
                    return true;
                };
                const bool valid=row==reverse ? evaluate_row(step.center_row(value[row]))
                    : evaluate_row(step.row(value[row],value[reverse]));
                if (!valid) return false;
            }
            std::swap(value,next_value);
            std::swap(gradient,next_gradient);
            std::swap(hessian,next_hessian);
        }
        for (std::size_t i=0;i<p;++i) for (std::size_t j=i;j<p;++j) {
            if (!std::isfinite(output[i*p+j])) return false;
            output[j*p+i]=output[i*p+j];
        }
        return true;
    }
};

}  // namespace vlk::experiment::mean
#undef VLK_SCHUR_ROW_INLINE
#endif
