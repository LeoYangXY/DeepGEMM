#pragma once

// 兼容层：老版本 PyTorch（< 2.2）的 `torch/csrc/utils/pybind.h` 里没有为
// `c10::ScalarType` 提供 pybind11 的 type_caster，导致：
//   1. 形如 `py::arg("logits_dtype") = torch::kFloat32` 的默认参数在模块初始化时
//      抛出 `arg(): could not convert default argument into a Python object`；
//   2. Python 侧传入 `torch.float32` 无法自动转换成 `at::ScalarType`。
// 这里在检测到缺失时补一个等价实现（与上游 PyTorch 的实现保持一致）。
//
// 本机环境为 torch 2.1.2+cu121，因此需要该兼容层。

#include <torch/python.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <c10/core/ScalarType.h>
// NOTES: torch 2.6+ moved TORCH_VERSION_* macros into `torch/headeronly/version.h`
// and stopped pulling it in through the top-level `torch/version.h` in some
// builds.  Include it explicitly so the version gate below works on all
// supported PyTorch versions.
#if __has_include(<torch/headeronly/version.h>)
#  include <torch/headeronly/version.h>
#endif

// PyTorch 2.2 起自带该 caster，用版本宏区分，避免重复定义
#if !defined(TORCH_VERSION_MAJOR) || \
    (TORCH_VERSION_MAJOR < 2) || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 2)

namespace pybind11::detail {

template <>
struct type_caster<c10::ScalarType> {
public:
    PYBIND11_TYPE_CASTER(c10::ScalarType, _("torch.dtype"));

    // torch.dtype -> at::ScalarType
    bool load(handle src, bool) {
        PyObject* obj = src.ptr();
        if (THPDtype_Check(obj)) {
            value = reinterpret_cast<THPDtype*>(obj)->scalar_type;
            return true;
        }
        return false;
    }

    // at::ScalarType -> torch.dtype
    static handle cast(const c10::ScalarType& src, return_value_policy, handle) {
        auto* dtype = torch::getTHPDtype(src);
        Py_INCREF(reinterpret_cast<PyObject*>(dtype));
        return handle(reinterpret_cast<PyObject*>(dtype));
    }
};

} // namespace pybind11::detail

#endif
