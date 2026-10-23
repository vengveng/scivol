"""Optional accounting contracts; full-image H remains unknown until qualified."""

from __future__ import annotations

import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("requested", "cache_hits", "failures", "physical", "physical_failures",
          "theta_passes", "preparation", "phase_entries", "solver_requests", "solver_failures")
SENTINEL = 271828.125

# The probe consumes each requested output and leaves every other slot untouched.
PROBE = r'''
#include "native_request_accounting.hpp"
namespace ac = vlk::experiment::request_accounting;
extern "C" std::size_t probe_bytes() { return sizeof(ac::Counts); }
extern "C" std::size_t probe_preparations() { return ac::preparations; }
extern "C" unsigned probe(ac::Counts* counts,unsigned mask,bool fail,bool hit,double* outputs) {
#ifndef ACCOUNTING_PLAIN
    ac::Binding binding(counts);
    ac::PhaseScope phase(ac::Phase::refinement);
    ac::LogicalRequest logical(mask,false);
    if (hit) logical.hit();
    else {
        ac::PhysicalRequest physical(mask,false);
        ac::PhysicalRequest nested(mask,false);
        ac::SolverRequest solver(mask);
        for (std::size_t kind=0;kind<ac::preparations;++kind)
            ac::prepare(static_cast<ac::Preparation>(kind),mask&4?2:mask&2?1:0);
        ac::theta_pass(mask);
        nested.finish_bool(!fail);
        physical.finish_bool(solver.finish(!fail));
    }
    logical.finish(fail?1:0,fail?0:mask);
#endif
    unsigned consumed=0;
    if (!fail) for (unsigned bit=0;bit<3;++bit) if (mask&(1u<<bit)) {
        outputs[bit]=double(11+bit); consumed+=unsigned(outputs[bit]);
    }
    return consumed;
}
extern "C" std::size_t probe_export(const ac::Counts* counts,unsigned field,unsigned long long* output) {
    constexpr std::uint64_t ac::Requests::* members[]={&ac::Requests::requested,&ac::Requests::cache_hits,
        &ac::Requests::failures,&ac::Requests::physical,&ac::Requests::physical_failures,&ac::Requests::theta_passes};
    std::size_t n=0;
    if (field<6) for (const auto& phase:counts->requests)
        for (const auto& coordinate:phase) for (const auto& row:coordinate) output[n++]=row.*members[field];
    if (field==6) for (const auto& phase:counts->preparation)
        for (const auto& kind:phase) for (auto value:kind) output[n++]=value;
    if (field==7) for (auto value:counts->phase_entries) output[n++]=value;
    if (field==8) for (const auto& phase:counts->solver_requests) for (auto value:phase) output[n++]=value;
    if (field==9) for (const auto& phase:counts->solver_failures) for (auto value:phase) output[n++]=value;
    return n;
}
'''


def ptr(array):
    return 0 if array is None else array.ctypes.data


def arena(size):
    storage = np.zeros(size + 32, dtype=np.uint8)
    start = -storage.ctypes.data % 16
    return storage, storage[start:start + size]


def export_counts(lib, counts, prefix):
    export = getattr(lib, prefix + "_export")
    kinds = getattr(lib, prefix + "_preparations")()
    result = {}
    for field, name in enumerate(FIELDS):
        shape = (5, 2, 8) if field < 6 else (5, kinds, 3) if field == 6 else (5,) if field == 7 else (5, 8)
        values = np.zeros(max(105, 5 * kinds * 3), dtype=np.uint64)
        n = export(ptr(counts), field, ptr(values))
        assert n == np.prod(shape)
        result[name] = values[:n].reshape(shape)
    return result


def assembly_body(assembly, symbol):
    start = assembly.index(symbol + ":")
    block = assembly[start:]
    end = re.search(r"(?m)^\s*(?:\.cfi_endproc|\.size\s)", block)
    assert end, symbol
    instructions = []
    for line in block[:end.start()].splitlines()[1:]:
        code = line.split("//", 1)[0].split("# ", 1)[0].strip()
        if code and not code.startswith((".", "//", "#")) and not code.endswith(":"):
            instructions.append(re.sub(r"LBB\d+_", "LBB_", code))
    return instructions


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("A C++20 compiler is required")
    native = Path(os.environ.get("SCIVOL_APARCH42_ACCOUNTING_SOURCE", ROOT / "scivol/_native"))
    source = tmp_path_factory.mktemp("phase-accounting") / "probe.cpp"
    source.write_text(PROBE)
    result, assemblies = {}, {}
    for name, defines in (("default", []), ("plain", ["-DACCOUNTING_PLAIN=1"]),
                          ("enabled", ["-DVLK_NATIVE_REQUEST_ACCOUNTING=1"])):
        image = source.with_name(name + ".so")
        flags = [compiler, "-std=c++20", "-O3", "-fPIC", "-I", str(native), *defines]
        subprocess.run([*flags, "-shared", str(source), "-o", str(image)],
                       check=True, capture_output=True, text=True, timeout=20)
        assemblies[name] = subprocess.run([*flags, "-S", str(source), "-o", "-"],
                                         check=True, capture_output=True, text=True, timeout=20).stdout
        lib = ct.CDLL(str(image))
        lib.probe_bytes.restype = lib.probe_preparations.restype = ct.c_size_t
        lib.probe.argtypes = [ct.c_void_p, ct.c_uint, ct.c_bool, ct.c_bool, ct.c_void_p]
        lib.probe.restype = ct.c_uint
        lib.probe_export.argtypes = [ct.c_void_p, ct.c_uint, ct.c_void_p]
        lib.probe_export.restype = ct.c_size_t
        symbols = subprocess.run(["nm", str(image)], capture_output=True, text=True, check=True, timeout=5).stdout
        if name != "enabled":
            assert "request_accounting7context" not in symbols
        result[name] = lib
    symbol = "_probe" if sys.platform == "darwin" else "probe"
    assert assembly_body(assemblies["default"], symbol) == assembly_body(assemblies["plain"], symbol)
    return result


@pytest.mark.parametrize("mask", range(1, 8))
@pytest.mark.parametrize("fail,hit", ((False, False), (True, False), (False, True)))
def test_all_masks_nested_owner_failures_hits_and_default_erasure(probes, mask, fail, hit):
    observed = {}
    for name, lib in probes.items():
        counts = arena(lib.probe_bytes())[1]
        outputs = np.full(5, SENTINEL)
        consumed = lib.probe(ptr(counts), mask, fail, hit, outputs[1:4].ctypes.data)
        for bit in range(3):
            assert outputs[bit + 1] == (11 + bit if not fail and mask & (1 << bit) else SENTINEL)
        assert outputs[0] == outputs[4] == SENTINEL
        assert consumed == sum(11 + bit for bit in range(3) if not fail and mask & (1 << bit))
        observed[name] = consumed, outputs.tobytes(), export_counts(lib, counts, "probe")
    assert observed["default"][:2] == observed["plain"][:2] == observed["enabled"][:2]
    for name in ("default", "plain"):
        assert all(not value.any() for value in observed[name][2].values())
    counts = observed["enabled"][2]
    index = (2, 0, mask)
    assert counts["requested"][index] == counts["requested"].sum() == 1
    assert counts["physical"][index] == counts["physical"].sum() == int(not hit)
    assert counts["cache_hits"][index] == int(hit)
    assert counts["failures"][index] == int(fail)
    assert counts["physical_failures"][index] == int(fail)
    assert counts["theta_passes"][index] == int(not hit)
    order = 2 if mask & 4 else 1 if mask & 2 else 0
    assert np.all(counts["preparation"][2, :, order] == int(not hit))
    assert counts["solver_requests"][2, mask] == int(not hit)
    assert counts["solver_failures"][2, mask] == int(fail)


@pytest.fixture(scope="module")
def phase_images():
    manifest = os.environ.get("SCIVOL_APARCH42_PHASE_ACCOUNTING_BUILD")
    if not manifest:
        pytest.skip("Selected phase header images are supplied by an isolated diagnostic build")
    record = json.loads(Path(manifest).read_text())
    assert record["passed"] and record["scope"] == "selected-header-accounting"
    assert record["h_accounting_status"] == "no-H-known"
    images = {}
    for name, entry in record["variants"].items():
        image = Path(entry["image"])
        assert hashlib.sha256(image.read_bytes()).hexdigest() == entry["image_sha256"]
        lib = ct.CDLL(str(image))
        pointer, size, integer, boolean = ct.c_void_p, ct.c_size_t, ct.c_uint, ct.c_bool
        for method in ("count_bytes", "preparations", "rows_bytes"):
            getattr(lib, "pac_" + method).restype = size
        for method in ("width", "held_bytes", "primal_bytes", "multi_bytes"):
            getattr(lib, "pac_" + method).argtypes = [integer]
            getattr(lib, "pac_" + method).restype = size
        lib.pac_export.argtypes = [pointer, integer, pointer]
        lib.pac_export.restype = size
        lib.pac_jet_bytes.argtypes = [integer, integer, boolean]
        lib.pac_jet_bytes.restype = size
        lib.pac_jet.argtypes = [integer, integer, boolean, pointer, pointer, size, ct.c_double,
                              pointer, size, pointer, pointer, pointer, pointer, pointer, boolean]
        lib.pac_held.argtypes = [integer, pointer, pointer, size, ct.c_double, pointer, pointer,
                               pointer, size, pointer, size, pointer, size, pointer, pointer, pointer, pointer]
        lib.pac_multi.argtypes = [integer, pointer, pointer, size, ct.c_double, pointer, size,
                                pointer, pointer, pointer, pointer, pointer, pointer, pointer]
        lib.pac_rows.argtypes = [pointer, pointer, size, pointer, pointer, pointer]
        images[name] = lib
    assert images["enabled"].pac_preparations() == 7
    return images


def sample(kind, transformed=False):
    ar, ma, p, q = (2, 1, 3, 2) if kind == 2 else (3, 2, 2, 1) if kind == 3 else (1, 1, 1, 1)
    if transformed:
        point = np.r_[.05, np.full(ar, .07), np.full(ma, .08), -.8,
                      np.full(p, -2.5), np.linspace(-.2, .3, p), np.full(q, .15), 1.2]
    else:
        point = np.r_[.05, np.full(ar, .07), np.full(ma, .08), .12,
                      np.full(p, .06 / p), np.linspace(-.2, .3, p), np.full(q, .35 / q), 1.6]
    data = np.array([.8, -.4, 1.1, .3, -1.2, .7, .6, -.2, 1.4, -.9, .5, .2, -.6, .9])
    return point, data


@pytest.mark.parametrize("kind", range(4))
@pytest.mark.parametrize("transformed", (False, True))
@pytest.mark.parametrize("mask", range(1, 8))
@pytest.mark.parametrize("outer", (False, True))
def test_header_all_masks_consume_outputs_and_do_not_count_wrappers(phase_images, kind, transformed, mask, outer):
    point, data = sample(kind, transformed)
    outputs = {}
    for name, lib in phase_images.items():
        workspace = arena(lib.pac_jet_bytes(kind, mask, transformed))[1]
        counts = arena(lib.pac_count_bytes())[1]
        f, g, h = np.full(1, SENTINEL), np.full(point.size, SENTINEL), np.full((point.size, point.size), SENTINEL)
        valid = np.zeros(1, dtype=np.uint32)
        result = lib.pac_jet(kind, mask, transformed, ptr(point), ptr(data), data.size, 1.1,
            ptr(workspace), workspace.size, ptr(counts), ptr(f) if mask & 1 else 0,
            ptr(g) if mask & 2 else 0, ptr(h) if mask & 4 else 0, ptr(valid), outer)
        assert result == 0 and valid[0] == mask
        for bit, values in enumerate((f, g, h)):
            assert np.isfinite(values).all()
            if not mask & (1 << bit):
                assert np.all(values == SENTINEL)
        outputs[name] = f.tobytes(), g.tobytes(), h.tobytes(), export_counts(lib, counts, "pac")
    assert outputs["baseline"][:3] == outputs["disabled"][:3] == outputs["enabled"][:3]
    counts = outputs["enabled"][3]
    assert counts["requested"][0, int(transformed), mask] == counts["requested"].sum() == 1
    assert counts["physical"][0, int(transformed), mask] == counts["physical"].sum() == 1
    theta_mask = mask | 2 if transformed and mask & 4 else mask
    assert counts["theta_passes"][0, 0, theta_mask] == counts["theta_passes"].sum() == 1
    assert not counts["physical_failures"].any()


@pytest.mark.parametrize("kind", range(4))
def test_header_held_power_and_multi_anchor_exact_lifetimes(phase_images, kind):
    for phase in ("held", "multi"):
        point, data = sample(kind, phase == "held")
        outputs = {}
        for name, lib in phase_images.items():
            counts = arena(lib.pac_count_bytes())[1]
            continuation, best, summary = np.full_like(point, SENTINEL), np.full_like(point, SENTINEL), np.full(12, SENTINEL)
            workspace = arena(getattr(lib, "pac_" + phase + "_bytes")(kind))[1]
            if phase == "held":
                jet = arena(lib.pac_jet_bytes(kind, 3, True))[1]
                primal = arena(lib.pac_primal_bytes(kind))[1]
                lower, upper = np.full_like(point, -100.), np.full_like(point, 100.)
                result = lib.pac_held(kind, ptr(point), ptr(data), data.size, 1.1, ptr(lower), ptr(upper),
                    ptr(workspace), workspace.size, ptr(jet), jet.size, ptr(primal), primal.size,
                    ptr(counts), ptr(continuation), ptr(best), ptr(summary))
                extra = ()
            else:
                anchors, columns = np.full(2, 99, dtype=np.uintp), np.full(2, 99, dtype=np.uintp)
                trace = np.full(12 + 2 * point.size, SENTINEL)
                result = lib.pac_multi(kind, ptr(point), ptr(data), data.size, 1.1,
                    ptr(workspace), workspace.size, ptr(counts), ptr(anchors), ptr(columns),
                    ptr(continuation), ptr(best), ptr(summary), ptr(trace))
                extra = anchors.tobytes(), columns.tobytes(), trace.tobytes()
            assert result == 0 and np.isfinite(summary).all()
            assert np.isfinite(continuation).all() and np.isfinite(best).all()
            outputs[name] = continuation.tobytes(), best.tobytes(), summary.tobytes(), extra, export_counts(lib, counts, "pac")
        assert outputs["baseline"][:4] == outputs["disabled"][:4] == outputs["enabled"][:4]
        counts = outputs["enabled"][4]
        summary = np.frombuffer(outputs["enabled"][2], dtype=np.float64)
        if phase == "held":
            assert counts["preparation"][1, 0, 0] == counts["preparation"][:, 0].sum() == 1
            assert counts["requested"][:, :, 4:].sum() == 0
            assert sum(counts["requested"][:, :, mask].sum() for mask in (1, 3)) == summary[6]
            assert sum(counts["requested"][:, :, mask].sum() for mask in (2, 3)) == summary[7]
            assert summary[8] == 0 and counts["solver_requests"].sum() > 0
            assert counts["preparation"][1, 6, 0] > 0
            assert counts["preparation"][:, :, 2].sum() == 0
        else:
            assert counts["preparation"][2, 0, 0] == counts["preparation"][:, 0].sum() == 1
            assert counts["requested"][2, 0, 1] == counts["physical"][2, 0, 1] == summary[5]
            assert counts["requested"][2, 0, 6] == counts["physical"][2, 0, 6] == summary[6]
            assert counts["requested"].sum() == summary[5] + summary[6]
            assert counts["theta_passes"][2, 0, 1] == summary[5]
            assert counts["theta_passes"][2, 0, 6] == summary[6]
            assert counts["preparation"][2, 1, 0] == summary[5]
            assert counts["preparation"][2, 1, 2] == summary[6]
            assert counts["preparation"][2, 5, 0] == summary[7]
            assert counts["preparation"][2, 4, 1] == 1 + summary[7] + summary[8]
            assert counts["preparation"][2, 4, 2] == summary[6]
            assert counts["preparation"][2, 3, 0] == 1 + summary[5] + summary[6] + summary[7] + summary[8]
            assert not counts["solver_requests"].any() and not counts["requested"][2, 1].any()


def test_header_partial_parameter_rows_count_misses_not_reuses(phase_images):
    point = np.r_[sample(0, True)[0], 1.4]
    results = {}
    for name, lib in phase_images.items():
        memory, counts = arena(lib.pac_rows_bytes())[1], arena(lib.pac_count_bytes())[1]
        output, local = np.zeros(5), np.zeros(3, dtype=np.uint64)
        assert lib.pac_rows(ptr(point), ptr(memory), memory.size, ptr(counts), ptr(output), ptr(local)) == 0
        assert np.isfinite(output).all() and np.all(output != 0)
        results[name] = output.tobytes(), local.tobytes(), export_counts(lib, counts, "pac")
    assert results["baseline"][:2] == results["disabled"][:2] == results["enabled"][:2]
    assert np.frombuffer(results["enabled"][1], dtype=np.uint64).tolist() == [2, 3, 10]
    counts = results["enabled"][2]
    assert counts["preparation"][1, 6].tolist() == [2, 3, 0]
    assert counts["preparation"].sum() == 5
    assert not counts["physical"].any()


@pytest.fixture(scope="module")
def accounting_serializer(tmp_path_factory):
    """Compile the actual private wrapper with synthetic, counted entry bodies."""
    native = Path(os.environ.get("SCIVOL_APARCH42_ACCOUNTING_SOURCE", ROOT / "scivol/_native"))
    content = (native / "native_core.cpp").read_text()
    start = content.index("PyObject* py_request_accounting_call(")
    end = content.index("\nPyObject* py_skewt_absolute_moment(", start)
    wrapper = content[start:end]
    if "py_arma_aparch_prepare, py_arma_aparch_multi_phase_run" not in wrapper:
        pytest.skip("The isolated selector 8/9 patch has not been integrated")
    compiler = shutil.which("clang++")
    if not compiler:
        pytest.skip("A C++20 compiler is required")
    directory = tmp_path_factory.mktemp("accounting-serializer")
    prefix = r'''
#include <Python.h>
#include <iterator>
#include <utility>
#include "native_request_accounting.hpp"
namespace ac = vlk::experiment::request_accounting;
template<int Selector> PyObject* synthetic_entry(PyObject*,PyObject* const* args,Py_ssize_t nargs) {
#if VLK_NATIVE_REQUEST_ACCOUNTING
    auto& counts=*ac::context.counts;
    unsigned index=0;
    for (auto& phase:counts.requests) for (auto& coordinate:phase) for (auto& row:coordinate) {
        const unsigned base=1000*Selector+(++index);
        row={base,base+100,base+200,base+300,base+400,base+500};
    }
    index=0;
    for (auto& phase:counts.preparation) for (auto& kind:phase)
        for (auto& order:kind) order=1000*Selector+(++index);
    index=0;
    for (auto& phase:counts.solver_requests) for (auto& mask:phase) mask=1000*Selector+(++index);
    index=0;
    for (auto& phase:counts.solver_failures) for (auto& mask:phase) mask=2000*Selector+(++index);
#endif
    PyObject* output=PyTuple_New(nargs+1);
    if (!output) return nullptr;
    PyTuple_SET_ITEM(output,0,PyLong_FromLong(Selector));
    for (Py_ssize_t i=0;i<nargs;++i) {Py_INCREF(args[i]);PyTuple_SET_ITEM(output,i+1,args[i]);}
    return output;
}
'''
    names = ("py_fit_arma_aparch", "py_fit_arma_aparch_reuse", "py_arma_aparch_jet",
             "py_arma_aparch_filter_state", "py_arma_aparch_mean_refine", "py_fit_aparch",
             "py_aparch_jet", "py_aparch_filter_state", "py_arma_aparch_prepare",
             "py_arma_aparch_multi_phase_run")
    prefix += "\n".join(f"constexpr auto {name}=synthetic_entry<{index}>;" for index, name in enumerate(names))
    prefix += r'''
namespace scivol::linear_joint {
PyObject* accounting_prevalidated(PyObject* self,PyObject* args) {
    return synthetic_entry<10>(self,PySequence_Fast_ITEMS(args),PyTuple_GET_SIZE(args));
}
PyObject* accounting_fit(PyObject* self,PyObject* args) {
    return synthetic_entry<11>(self,PySequence_Fast_ITEMS(args),PyTuple_GET_SIZE(args));
}
PyObject* accounting_refine(PyObject* self,PyObject* args) {
    return synthetic_entry<12>(self,PySequence_Fast_ITEMS(args),PyTuple_GET_SIZE(args));
}
}
'''
    modules = {}
    for enabled in (False, True):
        name = "_accounting_serializer_" + str(int(enabled))
        suffix = f'''
static PyMethodDef methods[]={{{{"call",(PyCFunction)py_request_accounting_call,METH_FASTCALL,nullptr}},{{nullptr,nullptr,0,nullptr}}}};
static PyModuleDef module={{PyModuleDef_HEAD_INIT,"{name}",nullptr,-1,methods}};
PyMODINIT_FUNC PyInit_{name}() {{return PyModule_Create(&module);}}
'''
        source = directory / (name + ".cpp")
        source.write_text(prefix + "\n" + wrapper + suffix)
        image = directory / (name + sysconfig.get_config_var("EXT_SUFFIX"))
        command = [compiler, "-std=c++20", "-O3", "-fPIC", "-fstack-protector-all",
                   "-I", str(native), "-I", sysconfig.get_paths()["include"]]
        if enabled:
            command += ["-DVLK_NATIVE_REQUEST_ACCOUNTING=1"]
        command += (["-bundle", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared"])
        subprocess.run([*command, str(source), "-o", str(image)], check=True,
                       capture_output=True, text=True, timeout=20)
        symbols = subprocess.run(["nm", str(image)], check=True, capture_output=True, text=True, timeout=5).stdout
        if not enabled:
            assert "request_accounting7context" not in symbols
        spec = importlib.util.spec_from_file_location(name, image)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[enabled] = module
    return modules


@pytest.mark.parametrize("selector", range(13))
@pytest.mark.parametrize("phase", range(5))
def test_actual_serializer_selectors_and_full_capacity(accounting_serializer, selector, phase):
    marker = object()
    result, record = accounting_serializer[True].call(selector, (marker, 17), phase)
    assert result == (selector, marker, 17) and result[1] is marker
    assert set(record) == set(FIELDS)
    for field, name in enumerate(FIELDS[:6]):
        assert record[name] == tuple(1000 * selector + 100 * field + index for index in range(1, 81))
    assert record["preparation"] == tuple(1000 * selector + index for index in range(1, 106))
    assert record["solver_requests"] == tuple(1000 * selector + index for index in range(1, 41))
    assert record["solver_failures"] == tuple(2000 * selector + index for index in range(1, 41))
    assert record["phase_entries"] == tuple(int(index == phase) for index in range(5))


def test_actual_serializer_disabled_and_invalid_entries(accounting_serializer):
    with pytest.raises(RuntimeError, match="disabled"):
        accounting_serializer[False].call(8, (), 1)
    for selector, phase in ((13, 0), (9, 5)):
        with pytest.raises(ValueError, match="selector or phase"):
            accounting_serializer[True].call(selector, (), phase)
    with pytest.raises(TypeError):
        accounting_serializer[True].call(8, [], 1)


@pytest.fixture(scope="module")
def controller_ledger():
    path = os.environ.get("SCIVOL_APARCH42_ACCOUNTING_LEDGER")
    if not path:
        pytest.skip("The isolated controller ledger is supplied explicitly")
    spec = importlib.util.spec_from_file_location("accounting_controller_ledger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_controller_ledger_composes_all_masks_without_charging_theta_passes(controller_ledger):
    ledger = controller_ledger
    records = []
    for stage, phase in (("baseline", 1), ("refinement", 2), ("held", 1), ("multi", 2), ("inference", 4)):
        counts = {name: [0] * int(np.prod(shape)) for name, shape in ledger.SHAPES.items()}
        for mask in range(1, 8):
            row = (phase * 2) * 8 + mask
            counts["requested"][row] = 3
            counts["cache_hits"][row] = 1
            counts["physical"][row] = 2
            counts["theta_passes"][row] = 2
        records.append(dict(stage=stage, method="test", counts=counts))
    result = ledger.compose_controller_ledger(records, tuple(row["stage"] for row in records))
    assert result["capture_complete"] and result["missing_stages"] == []
    assert result["observed_physical_h"] == 5 * 4 * 2
    assert sum(result["totals"]["physical"]) == 5 * 7 * 2
    assert sum(result["totals"]["requested"]) == 5 * 7 * 3
    assert result["h_accounting_status"] == "no-H-known"
    assert result["hessian_evaluations"] is None and result["timing_status"] == "unmeasured"
    assert not ledger.compose_controller_ledger(records, ("filter",))["capture_complete"]
    assert not ledger.compose_controller_ledger([])["capture_complete"]


def test_controller_proxy_keeps_stage_boundaries_and_native_results(controller_ledger):
    ledger = controller_ledger
    marker = object()

    class Core:
        def __getattr__(self, name):
            return lambda *args: marker

        def _request_accounting_call(self, selector, args, phase):
            assert args == (marker,)
            record = {name: [0] * int(np.prod(shape)) for name, shape in ledger.SHAPES.items()}
            mask = 6 if selector == 9 else 3
            record["requested"][(phase * 2) * 8 + mask] = 1
            record["physical"][(phase * 2) * 8 + mask] = 1
            return marker, record

    proxy = ledger.ControllerAccounting(Core())
    with proxy.stage_scope("baseline"):
        assert proxy._fit_arma_aparch(marker) is marker
        with proxy.stage_scope("multi"):
            assert proxy._arma_aparch_multi_phase_run(marker) is marker
        assert proxy.stage == "baseline"
    with proxy.stage_scope("held"):
        assert proxy._arma_aparch_prepare(marker) is marker
    with proxy.stage_scope("inference", 4):
        assert proxy._arma_aparch_jet(marker) is marker
    assert proxy.stage == "unassigned" and proxy.phase is None
    assert proxy._map_arma_aparch(marker) is marker
    assert [row["selector"] for row in proxy.records] == [0, 9, 8, 2]
    assert all(row["native_result"] is marker for row in proxy.records)
    result = ledger.compose_controller_ledger(proxy.records, ("baseline", "multi", "held", "inference"))
    assert result["capture_complete"] and result["observed_physical_h"] == 1
    assert [row["requests"][0]["phase"] for row in result["stages"]] == ["optimization", "refinement", "optimization", "inference"]


@pytest.mark.parametrize("fault", ("old_schema", "negative", "fractional", "extra", "excess_hits"))
def test_controller_ledger_rejects_incomplete_or_inconsistent_records(controller_ledger, fault):
    ledger = controller_ledger
    counts = {name: [0] * int(np.prod(shape)) for name, shape in ledger.SHAPES.items()}
    if fault == "old_schema":
        counts["preparation"] = [0] * 60
    elif fault == "negative":
        counts["physical"][0] = -1
    elif fault == "fractional":
        counts["physical"][0] = .5
    elif fault == "extra":
        counts["extra"] = []
    else:
        counts["cache_hits"][0] = 1
    with pytest.raises(ValueError):
        ledger.validate_record(counts)


@pytest.fixture(scope="module")
def enabled_integrated_core():
    manifest = os.environ.get("SCIVOL_APARCH42_ACCOUNTING_INTEGRATED_BUILD")
    if not manifest:
        pytest.skip("No accounting-enabled integrated image is qualified yet; no-H-known")
    record = json.loads(Path(manifest).read_text())
    assert record["complete"] and record["build_qualification_passed"]
    assert record["request_accounting_enabled"] is True
    image = Path(record["image"])
    assert hashlib.sha256(image.read_bytes()).hexdigest() == record["image_sha256"]
    spec = importlib.util.spec_from_file_location("_native_core", image)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


@pytest.mark.parametrize("density", (0, 1, 2))
@pytest.mark.parametrize("orders", ((1, 1, 1, 1), (2, 1, 3, 2), (3, 2, 2, 1)))
@pytest.mark.parametrize("phase", ("held", "multi"))
def test_integrated_selectors_outputs_masks_and_sample_lifetime(enabled_integrated_core, density, orders, phase):
    from test_aparch42_integrated_phase_guards import held_call, multi_call
    core = enabled_integrated_core
    call = held_call(core, orders, density) if phase == "held" else multi_call(core, orders, density, steps=1)
    plain = call.invoke()
    outputs = [view.tobytes() for _, view in call.outputs]
    result, record = core._request_accounting_call(8 if phase == "held" else 9, tuple(call.args), 1 if phase == "held" else 2)
    assert result == plain
    assert outputs == [view.tobytes() for _, view in call.outputs]
    call.assert_canaries()
    requests = np.array(record["requested"], dtype=np.uint64).reshape(5, 2, 8)
    physical = np.array(record["physical"], dtype=np.uint64).reshape(5, 2, 8)
    preparation = np.array(record["preparation"], dtype=np.uint64).reshape(5, 7, 3)
    assert preparation[:, 0].sum() == 1
    if phase == "held":
        assert requests[:, :, 4:].sum() == physical[:, :, 4:].sum() == result[8] == 0
        assert sum(requests[:, :, mask].sum() for mask in (1, 3)) == result[6]
        assert sum(requests[:, :, mask].sum() for mask in (2, 3)) == result[7]
    else:
        summary = call.outputs[4][1]
        assert requests[2, 0, 1] == physical[2, 0, 1] == summary[5]
        assert requests[2, 0, 6] == physical[2, 0, 6] == summary[6]
        assert requests.sum() == summary[5] + summary[6]
        assert preparation[2, 5, 0] == summary[7]


@pytest.mark.parametrize("density", (0, 3, 4, 5, 6, 7))
@pytest.mark.parametrize("orders,runtime", (((1, 1, 1, 1), False),
                                          ((2, 1, 3, 2), False),
                                          ((3, 2, 2, 1), True)))
@pytest.mark.parametrize("iterations", (1, 8))
def test_integrated_full_held_controller_logical_and_physical_work(
    enabled_integrated_core, density, orders, runtime, iterations,
):
    from test_aparch42_integrated_phase_guards import held_call, full_strategy_report

    core = enabled_integrated_core
    call = held_call(core, orders, density, runtime, iterations, strategy=1)
    plain = call.invoke()
    outputs = [view.tobytes() for _, view in call.outputs]
    result, record = core._request_accounting_call(8, tuple(call.args), 1)
    assert result == plain
    assert outputs == [view.tobytes() for _, view in call.outputs]
    call.assert_canaries()
    report, diagnostics = full_strategy_report(result)
    requested, hits, physical = (np.array(record[name], dtype=np.uint64).reshape(5, 2, 8)
                                 for name in ("requested", "cache_hits", "physical"))
    preparation = np.array(record["preparation"], dtype=np.uint64).reshape(5, 7, 3)
    assert preparation[:, 0].sum() == 1
    np.testing.assert_array_equal(requested - hits, physical)
    for bit, reported in zip((1, 2, 4), report[6:9]):
        assert sum(requested[:, :, mask].sum() for mask in range(1, 8) if mask & bit) == reported
    assert hits.sum() == diagnostics["reuse_hits"]
    assert requested.sum() == diagnostics["reuse_queries"]
    assert not any(record["failures"]) and not any(record["physical_failures"])
