#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "robot.hpp"

namespace py = pybind11;
using namespace robot;

PYBIND11_MODULE(robot, m) {
    m.doc() = "C++ port of robot.py";

    // 예외 바인딩
    py::register_exception<RobotEStopError>(m, "RobotEStopError");
    py::register_exception<RobotSetGainsError>(m, "RobotSetGainsError");
    py::register_exception<RobotSleepError>(m, "RobotSleepError");

    py::class_<Robot>(m, "Robot")
        .def(py::init<>())

        .def("set_gains", &Robot::set_gains, py::arg("kp"), py::arg("kd"),
             "Set PD gains")

        .def("check_safety", &Robot::check_safety)

        // 1D action 검사: 첫 원소가 시퀀스면 즉시 estop
        .def("do_action",
             [](Robot& self, py::object action, bool torque_ctrl) {
                 if (py::isinstance<py::sequence>(action)) {
                     py::sequence seq = action;
                     if (py::len(seq) > 0 && py::isinstance<py::sequence>(seq[0])) {
                         self.estop("action must be a 1D list");
                     }
                 }
                 std::vector<float> a = action.cast<std::vector<float>>();
                 self.do_action(a, torque_ctrl);
             },
             py::arg("action"), py::arg("torque_ctrl") = false)

        .def("get_obs", &Robot::get_obs)

        .def("estop", &Robot::estop, py::arg("msg") = std::string())
        .def("sleep", &Robot::sleep)
        .def("stand", &Robot::stand)
        .def("precise_stop", &Robot::precise_stop);
}