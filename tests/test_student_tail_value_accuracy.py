"""Student/Skew tail values and the shared linear residual under production FMA flags."""

from __future__ import annotations

import ctypes
from fractions import Fraction
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = r"""
#include "garch_extended_density_policy.hpp"
#include "mean_equation.hpp"

using namespace vlk::experiment;

extern "C" double shared_tail(double square, double scale) {
    return vlk_student_log_tail(square, scale, square+scale, std::log(scale));
}

template<class Density, unsigned Mask>
void local(const double* point, double* f, double* g, double* h) {
    constexpr bool F=Mask&1, G=Mask&2, H=Mask&4;
    constexpr std::size_t width=2+Density::shape_parameters;
    const auto prepared=Density::template prepare<F,G,H,true>(point+2,1);
    const auto observation=Density::template observe<F,G,H,true>(
        prepared,point[1],point[0],point[0]*point[0]);
    if constexpr (F) {
        auto total=Density::initial_value(prepared);
        Density::accumulate_value(total,observation);
        *f=Density::finalize_value(total,prepared);
    }
    if constexpr (G) {
        g[0]=observation.residual_gradient;
        g[1]=observation.variance_gradient;
        for (std::size_t i=0;i<Density::shape_parameters;++i) {
            g[2+i]=observation.shape_gradient[i];
            if constexpr (requires {prepared.observation_constant;})
                g[2+i]+=prepared.observation_constant.gradient[i];
        }
    }
    if constexpr (H) {
        h[0]=observation.residual_hessian;
        h[1]=h[width]=observation.residual_variance_hessian;
        h[width+1]=observation.variance_hessian;
        for (std::size_t i=0;i<Density::shape_parameters;++i) {
            h[2+i]=h[(2+i)*width]=observation.residual_shape_hessian[i];
            h[width+2+i]=h[(2+i)*width+1]=observation.variance_shape_hessian[i];
            for (std::size_t j=i;j<Density::shape_parameters;++j) {
                const auto k=ad::packed_index<Density::shape_parameters>(i,j);
                double entry=observation.shape_hessian[k];
                if constexpr (requires {prepared.observation_constant;})
                    entry+=prepared.observation_constant.hessian[k];
                h[(2+i)*width+2+j]=h[(2+j)*width+2+i]=entry;
            }
        }
    }
}

template<class Density>
void requests(unsigned mask,const double* p,double* f,double* g,double* h) {
    switch(mask) {
    case 1:local<Density,1>(p,f,g,h);break;
    case 2:local<Density,2>(p,f,g,h);break;
    case 3:local<Density,3>(p,f,g,h);break;
    case 4:local<Density,4>(p,f,g,h);break;
    case 5:local<Density,5>(p,f,g,h);break;
    case 6:local<Density,6>(p,f,g,h);break;
    case 7:local<Density,7>(p,f,g,h);break;
    }
}

extern "C" void density_local(bool skew,unsigned mask,const double* p,double* f,double* g,double* h) {
    if (skew) requests<generated::SkewTDensity>(mask,p,f,g,h);
    else requests<generated::StudentTDensity>(mask,p,f,g,h);
}

template<int Order>
double residual(const double* theta,const double* row,double y,std::size_t columns) {
    double first[32]{}, second[528]{};
    mean::State state{0.,first,second};
    const mean::History history{};
    mean::advance<Order,0>(state,history,mean::LinearDesign<>{row,columns},theta,y,0,0,columns);
    return state.value;
}

extern "C" void residual_requests(const double* theta,const double* row,double y,
    std::size_t columns,double* values) {
    values[0]=residual<0>(theta,row,y,columns);
    values[1]=residual<1>(theta,row,y,columns);
    values[2]=residual<2>(theta,row,y,columns);
}
"""


@pytest.fixture(scope="module")
def tail_bridge(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required")
    headers = Path(os.environ.get("SCIVOL_TAIL_HEADER_ROOT", ROOT))
    directory = tmp_path_factory.mktemp("tail_value_accuracy")
    source, image = directory / "local.cpp", directory / "local.so"
    source.write_text(NATIVE_SOURCE)
    command = [compiler, "-std=c++20", "-O3", "-ffp-contract=fast", "-funroll-loops",
               "-fno-exceptions", "-fno-rtti", "-fPIC", "-shared",
               "-I" + str(headers / "scivol/_native"),
               "-I" + str(headers / "scivol/_csrc"),
               "-I" + str(ROOT / "scivol/_native/third_party/boost"),
               "-I" + str(ROOT / "scivol/_native/third_party/tlfloat"),
               str(source), "-o", str(image)]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    lib = ctypes.CDLL(str(image))
    ptr = ctypes.c_void_p
    lib.shared_tail.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.shared_tail.restype = ctypes.c_double
    lib.density_local.argtypes = [ctypes.c_bool, ctypes.c_uint, ptr, ptr, ptr, ptr]
    lib.residual_requests.argtypes = [ptr, ptr, ctypes.c_double, ctypes.c_size_t, ptr]
    return lib


@pytest.mark.parametrize("nu", [7., 9432.377308349021, 1e10, 1e15])
@pytest.mark.parametrize("ratio", [np.nextafter(1e-4, 0), 1e-4, np.nextafter(1e-4, np.inf), .003, 2.])
def test_shared_tail_value_avoids_shape_amplified_cancellation(tail_bridge, nu, ratio):
    mp = pytest.importorskip("mpmath")
    square, scale = ratio * (nu - 2), nu - 2
    with mp.workdps(120):
        expected = float((mp.mpf(nu)+1)*mp.log1p(mp.mpf(square)/mp.mpf(scale))/2)
    actual = (nu+1)*tail_bridge.shared_tail(square, scale)/2
    np.testing.assert_allclose(actual, expected, rtol=3*np.finfo(float).eps, atol=0)


def test_shared_tail_overflow_ratio_retains_finite_logarithm(tail_bridge):
    mp = pytest.importorskip("mpmath")
    square, scale = 1e308, 1e-300
    with mp.workdps(120):
        expected = float(mp.log1p(mp.mpf(square)/mp.mpf(scale)))
    np.testing.assert_allclose(tail_bridge.shared_tail(square, scale), expected, rtol=0, atol=3e-13)


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("nu", [7., 9432.377308349021, 1e10, 1e15])
def test_local_tail_all_sparse_masks_match_independent_mp_ad(tail_bridge, skew, nu):
    mp = pytest.importorskip("mpmath")
    from test_native_anchored_mean_chart import Jet
    from test_native_aparch_subset_curvature import exp_jet, log_jet, loggamma_jet, jet_outputs

    point = np.array([.93, .7, nu, *([.33934824082115433] if skew else [])])
    expected = []
    for digits in (80,120):
        with mp.workdps(digits):
            e,h,degrees = [Jet.variable(float(x),len(point),i) for i,x in enumerate(point[:3])]
            constant = (loggamma_jet(degrees/2)-loggamma_jet((degrees+1)/2)
                        +log_jet((degrees-2)*mp.pi)/2)
            z = e*exp_jet(-log_jet(h)/2)
            if skew:
                lam = Jet.variable(float(point[3]),len(point),3)
                a = 4*lam*exp_jet(-constant)*(degrees-2)/(degrees-1)
                b = exp_jet(log_jet(1+3*lam*lam-a*a)/2)
                u = b*z+a
                z = u/(1+lam if u.value>=0 else 1-lam)
                constant -= log_jet(b)
            total = constant+log_jet(h)/2+(degrees+1)*log_jet(1+z*z/(degrees-2))/2
            expected.append(jet_outputs(total))
    for low,high in zip(*expected):
        np.testing.assert_array_equal(low,high)
    for mask in range(1,8):
        outputs = (np.full(1,np.nan),np.full(len(point),np.nan),np.full((len(point),len(point)),np.nan))
        tail_bridge.density_local(skew,mask,point.ctypes.data,*(x.ctypes.data for x in outputs))
        for bit,actual,target in zip((1,2,4),outputs,expected[1]):
            if mask&bit:
                assert np.isfinite(actual).all()
                np.testing.assert_allclose(actual,target,rtol=0 if bit==1 else 3e-10,atol=2e-11)
            else:
                assert np.isnan(actual).all()
        if mask&2:
            np.testing.assert_allclose(outputs[1][2],expected[1][1][2],rtol=3e-10,atol=0)
        if mask&4:
            np.testing.assert_allclose(outputs[2][2,2],expected[1][2][2,2],rtol=3e-10,atol=0)


@pytest.mark.parametrize("nu", [7.,9432.377308349021,1e10])
def test_zero_skew_reduces_to_student_on_common_value_and_derivatives(tail_bridge, nu):
    values = []
    for skew in (False,True):
        point = np.array([.93,.7,nu,*([0.] if skew else [])])
        f,g,h = np.full(1,np.nan),np.full(len(point),np.nan),np.full((len(point),len(point)),np.nan)
        tail_bridge.density_local(skew,7,point.ctypes.data,f.ctypes.data,g.ctypes.data,h.ctypes.data)
        values.append((f,g[:3],h[:3,:3]))
    for bit,a,b in zip((1,2,4),*values):
        np.testing.assert_allclose(a,b,rtol=0 if bit==1 else 3e-10,atol=2e-11)


def retained_linear_sample():
    from scivol import APARCH, HARX, StudentT
    from scivol._joint_mean_fit import JointMeanModel
    from scivol._validation import prepare_series_input

    rng = np.random.default_rng(432)
    x = rng.normal(size=(96,3))
    y = rng.standard_t(7,size=96)*.65+x@np.linspace(.18,-.12,3)
    for time in range(1,len(y)):
        y[time]+=.18*y[time-1]
    spec = HARX([1,3])+APARCH(2,4)+StudentT()
    sample = prepare_series_input(spec,y,x=x)
    return JointMeanModel(spec,sample.data,sample.x)


RETAINED_THETA = np.array([
    .0729304896691508,.3396260138964587,-.5335915627961436,.25619259623263535,
    .14725235603413567,-.2206575395344216,.9787239199900514,.0007863616140008254,
    .0034458432036007553,.1581055595324266,.1753861545449,.0022817201392144946,
    .0029172196653149563,.01175252749301474,1.860119310362138e-12,
    .0005868401736661569,23.95757886551101])


def test_shared_linear_residual_owner_retains_conditional_scale_accuracy(tail_bridge):
    model = retained_linear_sample()
    for time in (0,1,2,10):
        row = np.ascontiguousarray(model.features[time])
        exact = Fraction(float(model.data[time]))-sum(
            (Fraction(float(x))*Fraction(float(a)) for x,a in zip(row,RETAINED_THETA[:model.columns])),Fraction(0))
        values = np.full(3,np.nan)
        tail_bridge.residual_requests(RETAINED_THETA.ctypes.data,row.ctypes.data,
                                     model.data[time],model.columns,values.ctypes.data)
        np.testing.assert_array_equal(values,np.full(3,float(exact)))


def test_retained_student_harx_general_exact_additive_value():
    model = retained_linear_sample()
    values = [model.evaluate(RETAINED_THETA,mask)[0] for mask in (1,3,5,7)]
    np.testing.assert_allclose(values,60.73660888743031,rtol=0,atol=2e-11)


def test_normal_public_score_keeps_G_FG_rounding_identical():
    from scivol import ARX, APARCH, Normal
    from scivol._joint_mean_fit import JointMeanModel

    rng = np.random.default_rng(214)
    y = rng.normal(size=96)
    x = rng.normal(size=(96,2))
    model = JointMeanModel(ARX(1)+APARCH(3,2)+Normal(),y[5:]*2,x[5:])
    theta = np.array([.5039229709333392,.005126124767175115,.00019356986989024254,
        -.0021151791473364213,.27542684708262793,.024,.028181826334211017,
        .024000000000000004,-.00012850871681422942,-.0011746371095916908,
        .001176778705595566,.4300874033706335,.4308056202490993,1.9974353706675203])
    np.testing.assert_array_equal(model.evaluate(theta,2)[1],model.evaluate(theta,3)[1])
    np.testing.assert_array_equal(model.spec.score(y,theta,x=x,hold_back=5,scale=2),-model.evaluate(theta,2)[1])


@pytest.mark.parametrize("family", [0,1,2,3])
def test_other_joint_families_share_the_repaired_linear_residual(family):
    from scivol import HARX, StudentT
    from scivol._joint_mean_fit import FAMILIES, JointMeanModel
    from scivol._validation import prepare_series_input

    original = retained_linear_sample()
    spec = HARX([1,3])+FAMILIES[family](2,4)+StudentT()
    model = JointMeanModel(spec,original.data,np.ascontiguousarray(original.features[:,-3:]))
    theta = model.initialize()
    theta[:model.columns] = RETAINED_THETA[:model.columns]
    exact = [float(Fraction(float(y))-sum(
        (Fraction(float(x))*Fraction(float(a)) for x,a in zip(row,theta[:model.columns])),Fraction(0)))
        for y,row in zip(model.data,model.features)]
    for mask in range(1,8):
        _,_,_,residuals,_ = model.evaluate(theta,mask,paths=True)
        np.testing.assert_allclose(residuals,exact,rtol=3*np.finfo(float).eps,atol=0)
