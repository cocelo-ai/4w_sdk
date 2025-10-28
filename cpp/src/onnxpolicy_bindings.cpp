// onnxpolicy_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "onnxpolicy.hpp"

namespace py = pybind11;

// state를 1차원 list/tuple 또는 1차원 numpy.ndarray로만 허용해 vector<float>로 변환
static std::vector<float> vec1d_from_py(const py::object& obj, const char* who) {
    // 1) numpy 배열
    if (py::isinstance<py::array>(obj)) {
        py::array arr = py::cast<py::array>(obj);
        if (arr.ndim() != 1) {
            throw py::value_error(std::string(who) + ": expected a 1D numpy.ndarray; got ndim="
                                  + std::to_string(arr.ndim()));
        }
        // c-contiguous 강제 + float으로 캐스팅(필요 시 dtype 변환)
        py::array_t<float, py::array::c_style | py::array::forcecast> a = obj.cast<
            py::array_t<float, py::array::c_style | py::array::forcecast>>();
        const float* data = a.data();
        return std::vector<float>(data, data + a.size());
    }

    // 2) list/tuple
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
        std::vector<float> v;
        v.reserve(seq.size());
        for (py::ssize_t i = 0; i < seq.size(); ++i) {
            py::handle it = seq[i];
            // 중첩 시퀀스/배열 차단 -> 2D 이상 금지
            if (py::isinstance<py::list>(it) || py::isinstance<py::tuple>(it) || py::isinstance<py::array>(it)) {
                throw py::value_error(std::string(who) + ": expected a 1D list/tuple of numbers; "
                                        "found a nested sequence/array at index " + std::to_string(i));
            }
            v.push_back(py::cast<float>(it)); // 변환 실패 시 pybind가 예외 발생
        }
        return v;
    }

    // 3) 기타 타입은 거부
    throw py::type_error(std::string(who) + ": expected a 1D list/tuple or 1D numpy.ndarray of numbers");
}

PYBIND11_MODULE(onnxpolicy, m) {
    m.doc() = "C++ port of policy.py";

    py::class_<onnxpolicy::MLPPolicy>(m, "MLPPolicy")
        .def(py::init<const std::string&>(), py::arg("path"),
             "Create MLPPolicy from an ONNX file path.")
        .def("inference",
             [](onnxpolicy::MLPPolicy& self, const py::object& state) {
                 auto v = vec1d_from_py(state, "MLPPolicy.inference");
                 return py::cast(self.inference(v));
             },
             py::arg("state"),
             "Accepts a 1D list/tuple or 1D numpy.ndarray of numbers.");

    py::class_<onnxpolicy::LSTMPolicy>(m, "LSTMPolicy")
        .def(py::init<const std::string&>(), py::arg("path"),
             "Create LSTMPolicy from an ONNX file path.")
        .def("inference",
             [](onnxpolicy::LSTMPolicy& self, const py::object& state) {
                 auto v = vec1d_from_py(state, "LSTMPolicy.inference");
                 return py::cast(self.inference(v));
             },
             py::arg("state"),
             "Accepts a 1D list/tuple or 1D numpy.ndarray of numbers.");
}
