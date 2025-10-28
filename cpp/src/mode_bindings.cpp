
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mode.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mode, m) {
    m.doc() = "C++ port of mode.py";

    // Expose custom exception
    py::register_exception<ModeConfigError>(m, "ModeConfigError");

    // Free functions mirroring the Python module-level helpers
    m.def(
        "get_obs_to_length_map",
        [](){ return get_obs_to_length_map(); },
        "Return the observation-to-length map"
    );

    m.def(
        "get_proper_scale_form",
        [](py::object scale, std::size_t length){ return get_proper_scale_form(scale, length); },
        py::arg("scale"), py::arg("length"),
        "Normalize scale to a fixed-length list[float]; accepts number or list/tuple"
    );

    // Mode class
    py::class_<Mode>(m, "Mode")
        .def(py::init<py::object>(), py::arg("mode_cfg") = py::none())
        .def_readonly("obs_to_length", &Mode::obs_to_length)
        .def_readonly("id", &Mode::id)
        .def_readonly("stacked_obs_order", &Mode::stacked_obs_order)
        .def_readonly("non_stacked_obs_order", &Mode::non_stacked_obs_order)
        .def_readonly("obs_scale", &Mode::obs_scale)
        .def_readonly("action_scale", &Mode::action_scale)
        .def_readonly("stack_size", &Mode::stack_size)
        .def_readonly("policy_path", &Mode::policy_path)
        .def_readonly("policy_type", &Mode::policy_type)
        .def_readonly("cmd_vector_length", &Mode::cmd_vector_length)
        .def_readonly("cmd_scale", &Mode::cmd_scale)
        // Return the underlying Python-side policy object so .inference works as-is
        .def_property_readonly("policy", [](Mode &self){ return self.policy; });
}
