"""Selected APARCH phase ABI and complete-point retention, not fit certification."""

from __future__ import annotations

import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

SIZE, PTR, UINT = ct.c_size_t, ct.c_void_p, ct.c_uint32
MAX_SIZE = SIZE(-1).value
SENTINEL = 271828.125
SHAPES = ((), (7.,), (7., .23), (1.6,), (.12,1.5,.4), (.12,1.5), (.12,1.5), (.12,2.))
ORDERS = ((1,1,1,1), (2,1,3,2), (3,2,2,1), (0,0,1,1))


class Specification(ct.Structure):
    _fields_ = [(name,SIZE) for name in ("ar","ma","p","q","k")]+[("density",UINT),("force_runtime",ct.c_bool)]


@pytest.fixture(scope="session")
def native_libraries():
    isolated = os.environ.get("SCIVOL_APARCH_MULTI_BUILD_ROOT")
    if not isolated:
        from scivol import _native_core
        assert hasattr(_native_core,"_arma_aparch_multi_phase_run"), "Apply the production multi-phase patch first"
        return {density:(_native_core,ct.CDLL(_native_core.__file__)) for density in range(8)}
    root=Path(isolated)
    spec=importlib.util.spec_from_file_location("multi_contract_builder",root / "2_build.py")
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    builder=helper.configure_builder()
    core=json.loads(helper.CORE.read_text())
    builder.verify_pins({path:value["sha256"] for path,value in core["inputs_before"].items()
        if Path(path).is_relative_to(helper.ROOT / "scivol") and Path(path).suffix in (".c",".cpp",".h",".hpp")})
    libraries={}
    for density in range(8):
        folder=root / f"build-{density}-002"
        library,record=builder.load_bridge(folder / "build.json")
        binding=json.loads((folder / "binding.json").read_text())
        builder.verify_pins({binding["image"]:binding["image_sha256"],
            binding["python_executable"]:binding["python_sha256"]})
        spec=importlib.util.spec_from_file_location("_aparch_multi_contract",binding["image"])
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert record["scoped_density"]==density and record["build_passed"]
        binding_library,owners,layout=helper.open_owned_library(builder,Path(binding["image"]),binding["image_definitions"])
        assert owners==binding["symbol_owners"] and layout==binding["layout_contract"]
        libraries[density]=module,binding_library
    return libraries


@pytest.fixture(scope="session")
def canonical_core(native_libraries):
    isolated=os.environ.get("SCIVOL_APARCH_MULTI_BUILD_ROOT")
    if not isolated:
        return native_libraries[0][1]
    root=Path(isolated).resolve().parents[2]
    manifest=root / "localdev/aparch42_chart_integration_20261010/full_controller/attempt-019/build.json"
    record=json.loads(manifest.read_text())
    image=Path(record["image"])
    assert record["image_sha256"].startswith("04886f80")
    assert hashlib.sha256(image.read_bytes()).hexdigest()==record["image_sha256"]
    return ct.CDLL(str(image),mode=os.RTLD_LOCAL | os.RTLD_NOW)


def canonical_outputs(core,orders,density,theta,data,mask=1):
    workspace=core.vlk_runtime_arma_aparch_jet_workspace_bytes
    workspace.argtypes=[SIZE,SIZE,SIZE,SIZE,UINT,UINT,UINT]
    workspace.restype=SIZE
    evaluate=core.vlk_runtime_arma_aparch_jet_evaluate
    evaluate.argtypes=[PTR,SIZE,PTR,SIZE,ct.c_double,SIZE,SIZE,SIZE,SIZE,UINT,UINT,UINT,
        PTR,SIZE,PTR,PTR,PTR,ct.POINTER(UINT)]
    evaluate.restype=ct.c_int32
    size=workspace(*orders,density,0,mask)
    assert size>0
    arena=np.full(size+128,0xA5,dtype=np.uint8)
    offset=32+(-arena.ctypes.data%16)
    buffers=[np.full(count+2,SENTINEL) for count in (1,theta.size,theta.size**2)]
    outputs=[x[1:-1] for x in buffers]
    valid=UINT()
    status=evaluate(theta.ctypes.data,theta.size,data.ctypes.data,data.size,.5,*orders,density,0,mask,
        arena.ctypes.data+offset,size,*(x.ctypes.data for x in outputs),ct.byref(valid))
    assert status==0 and valid.value==mask,(orders,density,mask,status,valid.value)
    assert np.all(arena[:offset]==0xA5) and np.all(arena[offset+size:]==0xA5)
    for bit,buffer,output in zip((1,2,4),buffers,outputs):
        assert np.all(buffer[[0,-1]]==SENTINEL)
        assert np.isfinite(output).all() if mask & bit else np.all(output==SENTINEL)
    return outputs


def phase_sample(orders, density):
    ar,ma,p,q=orders
    theta=np.ascontiguousarray(np.r_[.04,np.linspace(.09,-.06,ar),np.linspace(-.08,.04,ma),
        .12,np.full(p,.06/p),np.linspace(-.16,.13,p),np.full(q,.6/q),1.4,SHAPES[density]])
    data=np.ascontiguousarray(np.random.default_rng(712+sum(orders)).normal(.1,.8,13))
    return theta,data


def prepare_call(binding, orders, density, k, runtime=False, steps=1):
    theta,data=phase_sample(orders,density)
    size=binding._arma_aparch_multi_phase_workspace_bytes(*orders,density,k,runtime)
    assert size>0
    arena=np.full(size+128,0xA5,dtype=np.uint8)
    offset=32+(-arena.ctypes.data%16)
    indices=[np.full(k+2,0xABCDEF,dtype=np.uintp) for _ in range(2)]
    outputs=[np.full(count+2,SENTINEL) for count in (theta.size,theta.size,12,steps*(12+2*theta.size))]
    args=[theta.ctypes.data,theta.size,data.ctypes.data,data.size,.5,*orders,density,k,steps,
        arena.ctypes.data+offset,size,indices[0][1:].ctypes.data,k,indices[1][1:].ctypes.data,k,
        outputs[0][1:].ctypes.data,theta.size,outputs[1][1:].ctypes.data,theta.size,
        outputs[2][1:].ctypes.data,12,outputs[3][1:].ctypes.data,outputs[3].size-2,runtime]
    return args,theta,data,arena,offset,size,indices,outputs


def call_phase(binding,orders,density,k,runtime=False,steps=1,change=None):
    call=prepare_call(binding,orders,density,k,runtime,steps)
    args,theta,data,arena,offset,size,indices,outputs=call
    before=theta.tobytes(),data.tobytes()
    if change:
        change(args,theta,data)
        before=theta.tobytes(),data.tobytes()
    status=binding._arma_aparch_multi_phase_run(*args)
    assert before==(theta.tobytes(),data.tobytes())
    assert np.all(arena[:offset]==0xA5) and np.all(arena[offset+size:]==0xA5)
    assert all(np.all(x[[0,-1]]==0xABCDEF) for x in indices)
    assert all(np.all(x[[0,-1]]==SENTINEL) for x in outputs)
    if change:
        assert status==-1
        assert np.all(arena==0xA5)
        assert all(np.all(x==0xABCDEF) for x in indices)
        assert all(np.all(x==SENTINEL) for x in outputs)
    else:
        assert status==0,(orders,density,k,status)
    return status,[x[1:-1].copy() for x in outputs],[x[1:-1].copy() for x in indices]


def bind_c(library):
    workspace=library.vlk_arma_aparch_multi_phase_workspace_bytes
    workspace.argtypes=[ct.POINTER(Specification)]
    workspace.restype=SIZE
    kind=library.vlk_arma_aparch_multi_phase_dispatch_kind
    kind.argtypes=[ct.POINTER(Specification)]
    kind.restype=ct.c_int32
    run=library.vlk_arma_aparch_multi_phase_run
    run.argtypes=[ct.POINTER(Specification),PTR,SIZE,PTR,SIZE,ct.c_double,SIZE,PTR,SIZE,
        PTR,SIZE,PTR,SIZE,PTR,SIZE,PTR,SIZE,PTR,SIZE,PTR,SIZE]
    run.restype=ct.c_int32
    return workspace,kind,run


@pytest.mark.parametrize("density",range(8))
@pytest.mark.parametrize("orders,k",[(ORDERS[0],1),(ORDERS[0],2),(ORDERS[1],2),(ORDERS[2],2),(ORDERS[3],1)])
def test_selected_phase_retains_complete_points(native_libraries,canonical_core,density,orders,k):
    binding,library=native_libraries[density]
    _,out,index=call_phase(binding,orders,density,k)
    continuation,best,summary,trace=out
    assert np.isfinite(continuation).all() and np.isfinite(best).all()
    assert summary[0] in (-3,-4,-5,0,1)
    assert summary[1]<=summary[2] and summary[3]==1
    assert summary[4] in (0,1) and summary[5]>=1 and summary[6]==1
    assert summary[7]==1 and summary[9]>=summary[10]>=0
    assert len(set(index[0]))==k and len(set(index[1]))==k
    assert np.all(index[0]>=max(orders)) and np.all(index[0]<13)
    assert np.all(index[1]<1+orders[0])
    np.testing.assert_array_equal(trace[12:12+best.size],continuation)
    np.testing.assert_array_equal(trace[12+best.size:],best)
    workspace,kind,_=bind_c(library)
    specification=Specification(*orders,k,density,False)
    expected=2 if orders==ORDERS[0] else 1 if orders==ORDERS[1] else 0
    assert kind(ct.byref(specification))==expected
    if os.environ.get("SCIVOL_APARCH_MULTI_BUILD_ROOT"):
        library.phase_reached_orders.restype=SIZE
        library.phase_reached_k.restype=SIZE
        assert library.phase_reached_orders()==(1111 if expected==2 else 2132 if expected==1 else 0)
        assert library.phase_reached_k()==(k if expected else MAX_SIZE)
    assert workspace(ct.byref(specification))>0
    theta,data=phase_sample(orders,density)
    initial_f=canonical_outputs(canonical_core,orders,density,theta,data)[0][0]
    assert summary[1]<=initial_f+2e-11
    for point,value in ((best,summary[1]),(continuation,summary[2])):
        actual=canonical_outputs(canonical_core,orders,density,point,data)[0][0]
        assert abs(actual-value)<=2e-11,(density,orders,actual,value)


@pytest.mark.parametrize("density",range(8))
@pytest.mark.parametrize("orders",ORDERS)
def test_canonical_sparse_requests_remain_valid(canonical_core,density,orders):
    theta,data=phase_sample(orders,density)
    full=canonical_outputs(canonical_core,orders,density,theta,data,7)
    for mask in range(1,8):
        actual=canonical_outputs(canonical_core,orders,density,theta,data,mask)
        for bit,output,reference in zip((1,2,4),actual,full):
            if mask & bit:
                np.testing.assert_allclose(output,reference,rtol=0 if bit==1 else 2e-10,
                    atol=2e-11 if bit==1 else 3e-9)


@pytest.mark.parametrize("density",range(8))
@pytest.mark.parametrize("orders,k",[(ORDERS[0],1),(ORDERS[0],2),(ORDERS[1],2)])
def test_fixed_runtime_stage_parity(native_libraries,density,orders,k):
    binding,_=native_libraries[density]
    _,fixed,indices=call_phase(binding,orders,density,k)
    _,runtime,runtime_indices=call_phase(binding,orders,density,k,True)
    for actual,reference in zip(fixed,runtime):
        np.testing.assert_allclose(actual,reference,rtol=2e-10,atol=3e-9)
    for actual,reference in zip(indices,runtime_indices):
        np.testing.assert_array_equal(actual,reference)


FAULTS={
    "short_arena":lambda a,t,d:a.__setitem__(13,a[13]-1),
    "unaligned_arena":lambda a,t,d:a.__setitem__(12,a[12]+1),
    "wrong_width":lambda a,t,d:a.__setitem__(1,a[1]-1),
    "short_anchors":lambda a,t,d:a.__setitem__(15,a[15]-1),
    "short_columns":lambda a,t,d:a.__setitem__(17,a[17]-1),
    "short_continuation":lambda a,t,d:a.__setitem__(19,a[19]-1),
    "short_best":lambda a,t,d:a.__setitem__(21,a[21]-1),
    "short_summary":lambda a,t,d:a.__setitem__(23,11),
    "short_trace":lambda a,t,d:a.__setitem__(25,a[25]-1),
    "alias_points":lambda a,t,d:a.__setitem__(20,a[18]),
    "alias_indices":lambda a,t,d:a.__setitem__(16,a[14]),
    "alias_input":lambda a,t,d:a.__setitem__(18,a[0]),
    "alias_arena":lambda a,t,d:a.__setitem__(22,a[12]),
    "null_data":lambda a,t,d:a.__setitem__(2,0),
    "null_output":lambda a,t,d:a.__setitem__(20,0),
    "unaligned_output":lambda a,t,d:a.__setitem__(20,a[20]+1),
    "unaligned_index":lambda a,t,d:a.__setitem__(14,a[14]+1),
    "overflow_data":lambda a,t,d:a.__setitem__(3,MAX_SIZE),
    "overflow_trace":lambda a,t,d:a.__setitem__(25,MAX_SIZE),
    "overflow_steps":lambda a,t,d:a.__setitem__(11,MAX_SIZE),
    "overflow_output":lambda a,t,d:a.__setitem__(21,MAX_SIZE),
    "wrapped_output":lambda a,t,d:a.__setitem__(20,MAX_SIZE-7),
    "empty_sample":lambda a,t,d:a.__setitem__(3,1),
    "zero_steps":lambda a,t,d:a.__setitem__(11,0),
    "bad_seed":lambda a,t,d:a.__setitem__(4,float("nan")),
    "nonfinite_parameter":lambda a,t,d:t.__setitem__(0,float("inf")),
    "nonfinite_observation":lambda a,t,d:d.__setitem__(-1,float("nan")),
}


@pytest.mark.parametrize("density",range(8))
@pytest.mark.parametrize("fault",FAULTS)
def test_checked_boundary_rejects_without_output_writes(native_libraries,density,fault):
    binding,_=native_libraries[density]
    call_phase(binding,ORDERS[0],density,2,change=FAULTS[fault])


@pytest.mark.parametrize("density",range(8))
def test_typed_metadata_checked_before_dereference(native_libraries,density):
    _,library=native_libraries[density]
    workspace,kind,run=bind_c(library)
    for pointer in (0,MAX_SIZE-7):
        metadata=ct.cast(pointer,ct.POINTER(Specification))
        assert workspace(metadata)==0 and kind(metadata)==-1
        assert run(metadata,0,0,0,0,.5,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0)==-1
    specification=Specification(MAX_SIZE,1,1,1,2,density,False)
    assert workspace(ct.byref(specification))==0 and kind(ct.byref(specification))==-1


@pytest.mark.parametrize("density",range(8))
def test_metadata_output_alias_rejected_without_writes(native_libraries,density):
    binding,library=native_libraries[density]
    _,_,run=bind_c(library)
    args,theta,data,arena,offset,size,indices,outputs=prepare_call(binding,ORDERS[0],density,2)
    specification=Specification(*ORDERS[0],2,density,False)
    before=bytes(specification),theta.tobytes(),data.tobytes()
    native_args=[ct.byref(specification),*args[:5],args[11],args[12],args[13],
        *args[14:22],ct.addressof(specification),12,*args[24:26]]
    assert run(*native_args)==-1
    assert before==(bytes(specification),theta.tobytes(),data.tobytes())
    assert np.all(arena==0xA5)
    assert all(np.all(x==0xABCDEF) for x in indices)
    assert all(np.all(x==SENTINEL) for x in outputs)


@pytest.mark.parametrize("density",range(8))
def test_python_parser_errors_are_not_density_truncation(native_libraries,density):
    binding,_=native_libraries[density]
    with pytest.raises(TypeError):
        binding._arma_aparch_multi_phase_run()
    with pytest.raises(TypeError):
        binding._arma_aparch_multi_phase_workspace_bytes()
    with pytest.raises((ValueError,OverflowError)):
        binding._arma_aparch_multi_phase_workspace_bytes(1,1,1,1,2**32,2,False)
    with pytest.raises(OverflowError):
        binding._arma_aparch_multi_phase_workspace_bytes(-1,1,1,1,density,2,False)
    class BadBoolean:
        def __bool__(self):
            raise RuntimeError("bad boolean")
    with pytest.raises(RuntimeError,match="bad boolean"):
        binding._arma_aparch_multi_phase_workspace_bytes(1,1,1,1,density,2,BadBoolean())


@pytest.mark.parametrize("density",range(8))
def test_initial_unavailable_reports_attempted_work_not_points(native_libraries,density):
    binding,_=native_libraries[density]
    args,theta,data,arena,offset,size,indices,outputs=prepare_call(binding,ORDERS[3],density,1)
    data.fill(1e308)
    before=theta.tobytes(),data.tobytes()
    status=binding._arma_aparch_multi_phase_run(*args)
    assert status==-2
    summary=outputs[2][1:-1]
    assert summary[0]==-2 and np.isposinf(summary[1:3]).all()
    assert summary[5]==1
    assert np.all(summary[[3,4,6,7,8,9,10,11]]==0)
    assert all(np.all(x==SENTINEL) for x in (outputs[0],outputs[1],outputs[3]))
    assert all(np.all(x==0xABCDEF) for x in indices)
    assert before==(theta.tobytes(),data.tobytes())
    assert np.all(arena[:offset]==0xA5) and np.all(arena[offset+size:]==0xA5)
