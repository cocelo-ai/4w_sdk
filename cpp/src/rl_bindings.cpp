#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "rl.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rl, m) {
    m.doc() = "C++ port of rl.py";

    py::class_<rl::RL>(m, "RL")
      .def(py::init<>())
      .def("add_mode", &rl::RL::add_mode, py::arg("mode"))
      .def("set_mode", &rl::RL::set_mode, py::arg("mode_id") = py::none())
      .def("build_state", &rl::RL::build_state, py::arg("obs"), py::arg("cmd"), py::arg("scaled_last_action") = py::none())
      .def("select_action", &rl::RL::select_action, py::arg("state"));
}
