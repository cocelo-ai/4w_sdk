#pragma once

// C++ port of mode.py

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>


namespace py = pybind11;

// ------------------------- Exceptions -------------------------
class ModeConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ------------------------- Helpers -------------------------
inline std::unordered_map<std::string, std::size_t> get_obs_to_length_map() {
    return {
        {"dof_pos", 12},
        {"dof_vel", 16},
        {"lin_vel", 3},
        {"ang_vel", 3},
        {"proj_grav", 3},
        {"last_action", 16},
        {"height_map", 144}
    };
}

inline bool is_number(const py::handle &h) {
    return py::isinstance<py::int_>(h) || py::isinstance<py::float_>(h);
}

inline bool is_list_or_tuple(const py::handle &h) {
    return py::isinstance<py::list>(h) || py::isinstance<py::tuple>(h);
}

inline std::vector<double> get_proper_scale_form(const py::handle &scale, std::size_t length) {
    std::vector<double> out;

    if (is_number(scale)) {
        double v = py::cast<double>(scale);
        out.assign(length, v);
    } else if (is_list_or_tuple(scale)) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(scale);
        out.reserve(seq.size());

        std::size_t idx = 0;
        for (py::handle item : seq) {
            if (py::isinstance<py::bool_>(item)) {
                throw ModeConfigError(
                    "`scale` must contain only numeric (int/float) elements; "
                    "bool found at index " + std::to_string(idx)
                );
            }

            try {
                out.push_back(py::cast<double>(item));
            } catch (const py::cast_error&) {
                const std::string r = py::str(py::repr(item));
                throw ModeConfigError(
                    "`scale` must contain only numbers; non-numeric element at index " +
                    std::to_string(idx) + ": " + r
                );
            }
            ++idx;
        }
    } else {
        throw ModeConfigError("`scale` must be a number or a sequence (list or tuple) of numbers");
    }

    if (out.size() != length) {
        throw ModeConfigError(
            "scale length mismatch, got: " + std::to_string(out.size()) +
            ", expected: " + std::to_string(length)
        );
    }

    return out;
}

// ------------------------- Mode -------------------------
class Mode {
public:
    using ObsLengthMap = std::unordered_map<std::string, std::size_t>;
    py::object policy;  // <- Python-based module

    // Public fields to mirror the Python class attributes
    ObsLengthMap obs_to_length;
    int id = 0;
    std::vector<std::string> stacked_obs_order;
    std::vector<std::string> non_stacked_obs_order;
    std::unordered_map<std::string, std::vector<double>> obs_scale; // normalized
    std::vector<double> action_scale; // normalized
    int stack_size = 1;
    std::string policy_path;
    std::string policy_type = "MLP";
    int cmd_vector_length = 0;
    std::vector<double> cmd_scale; // normalized

    explicit Mode(py::object mode_cfg = py::none()) {
        // mode_cfg = mode_cfg or {}
        py::dict cfg;
        if (!mode_cfg.is_none()) {
            if (!py::isinstance<py::dict>(mode_cfg)) {
                throw ModeConfigError("mode_cfg must be a dict or None");
            }
            cfg = mode_cfg.cast<py::dict>();
        } else {
            cfg = py::dict();
        }

        obs_to_length = get_obs_to_length_map();

        // id (required)
        if (!cfg.contains("id") || cfg["id"].is_none()) {
            throw ModeConfigError("mode_cfg must include required field 'id'");
        }
        py::handle id_obj = cfg["id"];
        if (py::isinstance<py::str>(id_obj)) {
            throw ModeConfigError("'id' must be an integer (1..16), not a string: " +
                                  std::string(py::str(py::repr(id_obj))));
        }
        id = cfg["id"].cast<int>();
        if (id < 1 || id > 16) {
            throw ModeConfigError("'id' must be between >=1 and <=16, but got " + std::to_string(id));
        }

        // orders
        if (cfg.contains("stacked_obs_order"))
            stacked_obs_order = cfg["stacked_obs_order"].cast<std::vector<std::string>>();
        if (cfg.contains("non_stacked_obs_order"))
            non_stacked_obs_order = cfg["non_stacked_obs_order"].cast<std::vector<std::string>>();

        // cmd_vector_length
        if (cfg.contains("cmd_vector_length")) {
            py::handle cmd_vector_length_obj = cfg["cmd_vector_length"];

            if (py::isinstance<py::str>(cmd_vector_length_obj)) {
                throw ModeConfigError("'cmd_vector_length' must be an integer, not a string: " +
                                      std::string(py::str(py::repr(cmd_vector_length_obj))));
            }
            cmd_vector_length = py::cast<int>(cmd_vector_length_obj);
        }

        // obs_scale (raw, may contain scalars or lists)
        py::dict obs_scale_in = (cfg.contains("obs_scale") && !cfg["obs_scale"].is_none())
            ? cfg["obs_scale"].cast<py::dict>()
            : py::dict();

        // cmd_scale normalization from obs_scale["command"] or default ones
        {
            py::object cmd_scale_obj = obs_scale_in.contains("command") ? py::object(obs_scale_in["command"]) : py::none();
            if (cmd_vector_length < 0){
                throw ModeConfigError("cmd_vector_length must be >= 0, but got " + std::to_string(cmd_vector_length));
            }
            if (cmd_scale_obj.is_none()) {
                cmd_scale.assign(static_cast<std::size_t>(cmd_vector_length), 1.0);
            } else {
                cmd_scale = get_proper_scale_form(cmd_scale_obj, static_cast<std::size_t>(cmd_vector_length));
            }
        }

        // action_scale from cfg or default ones of length last_action
        {
            std::size_t last_action_len = obs_to_length.at("last_action");
            py::object action_scale_obj = (cfg.contains("action_scale") ? py::object(cfg["action_scale"]) : py::none());
            if (action_scale_obj.is_none()) {
                action_scale.assign(last_action_len, 1.0);
            } else {
                action_scale = get_proper_scale_form(action_scale_obj, last_action_len);
            }
            if (action_scale.size() != last_action_len) {
                throw ModeConfigError(
                    "action scale length mismatch, got: " + std::to_string(action_scale.size()) +
                    ", expected: " + std::to_string(last_action_len)
                );
            }
        }

        // Normalize obs_scale entries for all obs seen in orders (except "command")
        auto validate_and_norm_obs = [&](const std::string &obs){
            if (obs == "command") return; // handled above
            auto it = obs_to_length.find(obs);
            if (it == obs_to_length.end()) {
                // Build message: valid keys (sorted)
                std::vector<std::string> keys; keys.reserve(obs_to_length.size());
                for (auto &kv : obs_to_length) keys.push_back(kv.first);
                std::sort(keys.begin(), keys.end());
                std::string joined;
                for (size_t i = 0; i < keys.size(); ++i) { if (i) joined += ", "; joined += keys[i]; }
                throw py::key_error("unknown observation key: '" + obs + "'. valid keys: " + joined);
            }
            std::size_t length = it->second;
            py::object this_obs_scale = obs_scale_in.contains(obs.c_str()) ? py::object(obs_scale_in[obs.c_str()]) : py::none();
            if (this_obs_scale.is_none()) {
                obs_scale[obs] = std::vector<double>(length, 1.0);
            } else {
                obs_scale[obs] = get_proper_scale_form(this_obs_scale, length);
            }
        };
        for (const auto &obs : stacked_obs_order) validate_and_norm_obs(obs);
        for (const auto &obs : non_stacked_obs_order) validate_and_norm_obs(obs);

        // stack_size
        if (cfg.contains("stack_size")) {
            py::handle stack_size_obj = cfg["stack_size"];
            if (py::isinstance<py::str>(stack_size_obj)) {
                throw ModeConfigError("'stack_size' must be an integer >= 1, not a string: " +  std::string(py::str(py::repr(stack_size_obj))));
            }
            stack_size = py::cast<int>(stack_size_obj);
        }
        if (stack_size < 1) {
            throw ModeConfigError("stack_size must be >= 1, but got " + std::to_string(stack_size));
        }

        // policy_path
        if (!cfg.contains("policy_path") || cfg["policy_path"].is_none()) {
            throw ModeConfigError("policy_path is required but missing");
        }
        policy_path = cfg["policy_path"].cast<std::string>();

        namespace fs = std::filesystem;
        fs::path p(policy_path);
        if (!fs::exists(p)) {
            throw std::runtime_error("policy_path does not exist: " + p.string());
        }
        if (!fs::is_regular_file(p)) {
            throw std::runtime_error("policy_path is not a file: " + p.string());
        }
        {
            std::string ext = p.has_extension() ? p.extension().string() : std::string();
            std::string lower = ext;
            std::transform(ext.begin(), ext.end(), lower.begin(), [](unsigned char c){ return std::tolower(c); });
            if (lower != ".onnx") {
                throw ModeConfigError("policy_path must be a .onnx file, but got '" + ext + "'");
            }
        }

        // policy_type
        if (cfg.contains("policy_type") && !cfg["policy_type"].is_none())
            policy_type = cfg["policy_type"].cast<std::string>();
        std::string type_lower = policy_type;
        std::transform(type_lower.begin(), type_lower.end(), type_lower.begin(), [](unsigned char c){ return std::tolower(c); });

        // Instantiate policy class via package __init__.py
        py::module_ pkg = py::module_::import("w4_sdk");
        py::object cls;
        if (type_lower == "mlp") {
            cls = pkg.attr("MLPPolicy");
        } else if (type_lower == "lstm") {
            cls = pkg.attr("LSTMPolicy");
        } else {
            throw ModeConfigError("Unsupported policy_type: " + policy_type);
        }
        policy = cls(policy_path);

        // obs & policy validation
        obs_to_length["command"] = static_cast<std::size_t>(cmd_vector_length);
        std::size_t state_len = 0;
        for (const std::string& obs : stacked_obs_order) {
            auto it = obs_to_length.find(obs);
            state_len += it->second;
        }
        state_len *= static_cast<std::size_t>(stack_size);

        for (const std::string& obs : non_stacked_obs_order) {
            auto it = obs_to_length.find(obs);
            state_len += it->second;
        }

        std::vector<double> dummy_state(state_len, 0.0);
        try {
            // inference 결과는 항상 1D Python list 여야함 (onnxpolicy.hpp에서 보장)
            py::list out = policy.attr("inference")(py::cast(dummy_state));
            std::size_t got = static_cast<std::size_t>(py::len(out));
            std::size_t expected = obs_to_length.at("last_action");

            if (got != expected) {
                throw ModeConfigError(
                    "Policy 'inference' output length mismatch: got " + std::to_string(got) +
                    ", expected " + std::to_string(expected) + " ('last_action' length)"
                );
            }
        } catch (const py::error_already_set& e) {
            throw ModeConfigError(
                std::string("Policy inference failed. ")
                + "Hint: the state length or dtype may not match the ONNX model's input. "
                + e.what()
            );
        }
    }
};
