#include "test_support.hpp"
#include "optimize/kinks.hpp"
#include "fit_problem.hpp"

using scivolapi::KinkOutput;

// A curved residual manifold exercises the residual Hessian in the equality
// Lagrangian. The finite branch interval is independent of the smooth mean.
static Problem curved_kink(double tilt)
{
    Problem p;
    p.dimension=2; p.objective_can_have_gradient_kinks=true; p.kink_capacity=1;
    p.objective=[tilt](const double* x,double& f,double* g) {
        const double e=x[0]-x[1]*x[1], y=x[1]-1;
        f=std::abs(e)+tilt*e+.5*(e*e+y*y);
        if (g) {
            if (e==0) return false;
            const double b=(e>0?1:-1)+tilt+e;
            g[0]=b; g[1]=y-2*x[1]*b;
        }
        return true;
    };
    p.hessian=[tilt](const double* x,double* h) {
        const double e=x[0]-x[1]*x[1],b=(e>0?1:-1)+tilt+e;
        if(e==0) return false;
        h[0]=1; h[1]=h[2]=-2*x[1]; h[3]=4*x[1]*x[1]+1-2*b;
        return true;
    };
    p.faces=[tilt](const double* x,double radius,KinkOutput& k) {
        const double e=x[0]-x[1]*x[1];
        k.count=radius<0 || std::abs(e)<=std::max(radius,64*std::numeric_limits<double>::epsilon())
            *(std::abs(x[0])+x[1]*x[1]) ? 1 : 0;
        k.supported=true;
        if(!k.count) return true;
        k.rows[0]=0; k.residuals[0]=e; k.normals[0]=1; k.normals[1]=-2*x[1];
        std::fill_n(k.curvatures.data(),4,0.0); k.curvatures[3]=-2;
        if(radius!=0) return true;
        k.weights[0]=1;
        k.smooth_gradient[0]=tilt+e;
        k.smooth_gradient[1]=x[1]-1-2*x[1]*(tilt+e);
        k.smooth_hessian[0]=1;
        k.smooth_hessian[1]=k.smooth_hessian[2]=-2*x[1];
        k.smooth_hessian[3]=4*x[1]*x[1]+1-2*(tilt+e);
        return true;
    };
    return p;
}

int main()
{
    auto p=curved_kink(.3);
    auto r=refine_newton(p,std::array{.25,.5},{});
    assert(r.termination==Termination::converged && r.accepted.kink);
    close(r.accepted.x[0],1,1e-6); close(r.accepted.x[1],1,1e-6);
    KinkWorkspace w(1,2);
    assert(p.faces(r.accepted.x.data(),0,w.out));
    assert(kink_stationarity(p,r.accepted.x,w).residual<=1e-5);
    assert(std::abs(w.coefficients[0])<=w.weights[0]);
    assert(r.accepted.gradient.empty());

    // A domain normal and an absolute-value subgradient must be solved jointly.
    p.lower={std::numeric_limits<double>::quiet_NaN(),1.2};
    r=refine_newton(p,std::array{1.44,1.2},{});
    assert(r.termination==Termination::converged && r.accepted.kink);
    close(r.accepted.x[0],1.44,1e-6); close(r.accepted.x[1],1.2,1e-6);
    assert(p.faces(r.accepted.x.data(),0,w.out));
    assert(kink_stationarity(p,r.accepted.x,w).residual<=1e-5);

    // An out-of-interval multiplier releases the face and resumes the original
    // smooth problem. The optimum has e=-1, outside the manifold.
    p=curved_kink(2);
    r=refine_newton(p,std::array{1.0,1.0},{});
    assert(r.termination==Termination::converged && !r.accepted.kink);
    close(r.accepted.x[0],0,1e-5); close(r.accepted.x[1],1,1e-5);

    // A concave kink has no certificate, even when its Clarke convex hull
    // happens to contain zero.
    p=curved_kink(0);
    assert(p.faces(std::array{1.0,1.0}.data(),0,w.out));
    w.weights[0]=-1;
    assert(!std::isfinite(kink_stationarity(p,std::array{1.0,1.0},w).residual));

    // Unavailable one-sided derivatives retain the finite point without
    // manufacturing an accepted iterate, a gradient or convergence.
    const auto faces=p.faces;
    p.faces=[faces](const double* x,double radius,KinkOutput& k) {
        return radius==0 ? false : faces(x,radius,k);
    };
    r=refine_newton(p,std::array{1.0,1.0},{});
    assert(r.termination==Termination::stalled && r.accepted.x.empty());
    assert(r.best_feasible.x.size()==2 && r.best_feasible.gradient.empty());
    close(r.best_feasible.value,0);

    // Two simultaneous, independent residual faces leave one smooth direction.
    p={}; p.dimension=3; p.kink_capacity=2;
    p.objective_can_have_gradient_kinks=true;
    p.objective=[](const double* x,double& f,double* g) {
        const double z=x[2]-2;
        f=std::abs(x[0])+std::abs(x[1])+.3*x[0]-.4*x[1]
            +.5*(x[0]*x[0]+x[1]*x[1]+z*z);
        if(g) {
            if(x[0]==0 || x[1]==0) return false;
            g[0]=x[0]+.3+(x[0]>0?1:-1);
            g[1]=x[1]-.4+(x[1]>0?1:-1); g[2]=z;
        }
        return true;
    };
    p.hessian=[](const double*,double* h) {
        std::fill_n(h,9,0.0); h[0]=h[4]=h[8]=1; return true;
    };
    p.faces=[](const double* x,double radius,KinkOutput& k) {
        k.count=0; k.supported=true;
        // The smooth part excludes the active rows' |x_j| and keeps the others
        // with their signs, as the jet's decomposition does.
        bool active[2]={};
        for(std::size_t j=0;j<2;++j) {
            if(radius>=0 && std::abs(x[j])>std::max(radius,64*std::numeric_limits<double>::epsilon())) continue;
            active[j]=true;
            const auto at=k.count++;
            k.rows[at]=j; k.residuals[at]=x[j]; k.weights[at]=1;
            std::fill_n(k.normals.data()+at*3,3,0.0); k.normals[at*3+j]=1;
            if(!k.curvatures.empty()) std::fill_n(k.curvatures.data()+at*9,9,0.0);  // as the jet: only when requested
        }
        k.smooth_gradient[0]=x[0]+.3+(active[0] ? 0 : x[0]>0 ? 1 : -1);
        k.smooth_gradient[1]=x[1]-.4+(active[1] ? 0 : x[1]>0 ? 1 : -1);
        k.smooth_gradient[2]=x[2]-2;
        std::fill_n(k.smooth_hessian.data(),9,0.0);
        k.smooth_hessian[0]=k.smooth_hessian[4]=k.smooth_hessian[8]=1;
        return true;
    };
    r=refine_newton(p,std::array{0.0,0.0,.2},{});
    assert(r.termination==Termination::converged && r.accepted.kink);
    close(r.accepted.x[0],0); close(r.accepted.x[1],0); close(r.accepted.x[2],2);

    // On the first face with the second residual 1e-6 away (outside the
    // proposal radius), the step along the face is blocked by the second kink;
    // the refinement joins it and certifies the intersection.
    r=refine_newton(p,std::array{0.0,1e-6,.2},{});
    assert(r.termination==Termination::converged && r.accepted.kink);
    close(r.accepted.x[0],0); close(r.accepted.x[1],0); close(r.accepted.x[2],2);

    // Rows with the same residual function (identical regressors) are one
    // kink: they merge into the first, with the summed subgradient weight.
    {
        KinkWorkspace m(4,3);
        m.out.count=3;
        const double normal[3]={1,-.5,0};
        for(std::size_t j=0;j<3;++j) {
            m.out.rows[j]=10+j; m.out.residuals[j]=j==1 ? 2e-3 : 1e-13*(1+1e-16*j); m.out.weights[j]=.25+j;
            std::copy_n(normal,3,m.out.normals.data()+j*3);
            std::fill_n(m.out.curvatures.data()+j*9,9,0.0);
        }
        m.out.normals[1*3+1]=-.6;  // row 11 differs
        scivolapi::detail::merge_identical_faces(m.out,3,true);
        assert(m.out.count==2 && m.out.rows[0]==10 && m.out.rows[1]==11);
        close(m.out.weights[0],.25+2.25,1e-15); close(m.out.weights[1],1.25,1e-15);
    }
}
