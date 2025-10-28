#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

namespace py = pybind11;

namespace rl {

class RL {
public:
    RL() {
        obs_to_length_ = {
            {"dof_pos",   12},
            {"dof_vel",   16},
            {"lin_vel",   3},
            {"ang_vel",   3},
            {"proj_grav", 3},
            {"last_action", 16},
            {"height_map", 144}
        };
        last_action_len_ = obs_to_length_.at("last_action");
        last_action_.assign(last_action_len_, 0.0f);
        scaled_action_.assign(last_action_len_, 0.0f);
        single_frame_len_ = 0;
    }

    void add_mode(const py::object& mode) {
        int new_id = get_mode_id_(mode);
        for (size_t i = 0; i < modes_.size(); ++i) {
            if (get_mode_id_(modes_[i]) == new_id) {
                modes_[i] = mode;
                return;
            }
        }
        modes_.push_back(mode);
    }

    void set_mode(const py::object& mode_id_obj) {
        if (!mode_id_obj || mode_id_obj.is_none()) return;
        int mode_id = mode_id_obj.cast<int>();

        for (const auto& m : modes_) {
            if (get_mode_id_(m) == mode_id) {
                // command 길이
                int cmd_len = m.attr("cmd_vector_length").cast<int>();
                obs_to_length_["command"] = cmd_len;

                // 스택 구간 길이
                cached_stacked_order_ = m.attr("stacked_obs_order").cast<std::vector<std::string>>();
                size_t single_len = 0;
                for (const auto& key : cached_stacked_order_) single_len += get_obs_len_(key);
                single_frame_len_ = single_len;
                single_frame_.assign(single_frame_len_, 0.0f);

                // 전체 상태 길이
                cached_stack_size_ = m.attr("stack_size").cast<int>();
                size_t state_len = single_len * static_cast<size_t>(cached_stack_size_);

                cached_non_stacked_order_ = m.attr("non_stacked_obs_order").cast<std::vector<std::string>>();
                for (const auto& key : cached_non_stacked_order_) state_len += get_obs_len_(key);

                state_.assign(state_len, 0.0f);
                last_action_.assign(last_action_len_, 0.0f);
                scaled_action_.assign(last_action_len_, 0.0f);

                mode_ = m;

                // --- 캐시 (모드 종속) ---
                cached_policy_         = mode_.attr("policy");
                cached_action_scale_   = mode_.attr("action_scale").cast<std::vector<float>>();
                cached_cmd_scale_      = mode_.attr("cmd_scale").cast<std::vector<float>>();

                // obs_scale 캐시
                cached_obs_scale_map_.clear();
                try {
                    py::dict scales = mode_.attr("obs_scale").cast<py::dict>();
                    for (auto item : scales) {
                        std::string k = py::cast<std::string>(item.first);
                        std::vector<float> v = py::cast<std::vector<float>>(item.second);
                        cached_obs_scale_map_[k] = std::move(v);
                    }
                } catch (...) {
                    // 없음/형식 오류 → 비워둠(기본 1.0 사용)
                }

                if (cached_action_scale_.size() < last_action_len_) {
                    throw std::runtime_error("action_scale length is smaller than last_action length for current mode.");
                }
                return;
            }
        }
        // 등록 안 되어 있으면 무시
    }

    // obs: Dict[str, List], cmd: Dict[str, Any]
    std::vector<float> build_state(const py::dict& obs, const py::dict& cmd, py::object scaled_last_action) {
        ensure_mode_();

        // cmd["mode_id"]가 있으면 즉시 모드 전환
        if (cmd.contains("mode_id")) {
            py::object v = cmd["mode_id"];
            if (!v.is_none()) set_mode(v);
        }

        //  scaled_last_action이 None 아니면 벡터로 캐스팅해서 길이 확인 후 반영
        if (!scaled_last_action.is_none()) {
            py::sequence seq;
            try {
                seq = scaled_last_action.cast<py::sequence>();
            } catch (...) {
                throw py::type_error("scaled_last_action must be a 1D array/list.");
            }
            const size_t n = seq.size();
            if (n != last_action_len_) {
                throw py::value_error("scaled_last_action length must be " + std::to_string(last_action_len_) + " (got " + std::to_string(n) + ")");
            }
            if (n > 0) {
                py::handle first = seq[0];
                if (py::isinstance<py::list>(first) ||
                    py::isinstance<py::tuple>(first) ||
                    py::isinstance<py::array>(first)) {
                    throw py::value_error("scaled_last_action must be 1D array/list.");
                }
            }
            std::vector<float> v(n);
            for (size_t i = 0; i < n; ++i) {
                v[i] = py::cast<float>(seq[i]);
            }
            last_action_ = std::move(v);
        }

        // 1) 싱글 프레임 구성
        size_t i = 0;
        const auto& stacked = cached_stacked_order_;
        for (const auto& key : stacked) {
            size_t obs_len = get_obs_len_(key);

            const std::vector<float>& scale = (key == "command")
                ? cached_cmd_scale_
                : get_obs_scale_(key, obs_len);

            py::object obs_obj = py::none();
            if (key == "command") {
                if (cmd.contains("cmd_vector")) obs_obj = cmd["cmd_vector"];
            } else if (key == "last_action") {
                // 내부 last_action_ 사용
            } else {
                if (obs.contains(py::str(key))) obs_obj = obs[py::str(key)];
            }

            if (obs_obj.is_none() && key != "last_action") {
                for (size_t k = 0; k < obs_len; ++k) {
                    single_frame_[i] = state_[i];
                    ++i;
                }
            } else {
                const std::vector<float>* src_vec = nullptr;
                std::vector<float> tmp;
                if (key == "last_action") {
                    src_vec = &last_action_;
                } else {
                    tmp = obs_obj.cast<std::vector<float>>();
                    src_vec = &tmp;
                }
                for (size_t j = 0; j < obs_len; ++j) {
                    single_frame_[i] = (*src_vec)[j] * scale[j];
                    ++i;
                }
            }
        }

        // 2) 스택 쉬프트
        const size_t L = single_frame_len_;
        const int S = cached_stack_size_;
        if (S > 1) {
            for (int k = S - 1; k > 0; --k) {
                std::copy(state_.begin() + (k - 1) * L,
                          state_.begin() + k * L,
                          state_.begin() + k * L);
            }
        }
        std::copy(single_frame_.begin(), single_frame_.end(), state_.begin());

        // 3) 비스택 구간
        size_t base = L * static_cast<size_t>(S);
        const auto& non_stacked = cached_non_stacked_order_;
        for (const auto& key : non_stacked) {
            size_t obs_len = get_obs_len_(key);

            const std::vector<float>& scale = (key == "command")
                ? cached_cmd_scale_
                : get_obs_scale_(key, obs_len);

            py::object obs_obj = py::none();
            if (key == "command") {
                if (cmd.contains("cmd_vector")) obs_obj = cmd["cmd_vector"];
            } else if (key == "last_action") {
                // 내부 last_action_ 사용
            } else {
                if (obs.contains(py::str(key))) obs_obj = obs[py::str(key)];
            }

            if (obs_obj.is_none() && key != "last_action") {
                base += obs_len;
            } else {
                const std::vector<float>* src_vec = nullptr;
                std::vector<float> tmp;
                if (key == "last_action") {
                    src_vec = &last_action_;
                } else {
                    tmp = obs_obj.cast<std::vector<float>>(); // 입력은 매 호출 변함
                    src_vec = &tmp;
                }
                for (size_t j = 0; j < obs_len; ++j) {
                    state_[base + j] = (*src_vec)[j] * scale[j];
                }
                base += obs_len;
            }
        }

        return state_;
    }

    // policy.inference(state) → [-1,1] → action_scale 적용 (검사 없음: set_mode에서 이미 확인)
    std::vector<float> select_action(const std::vector<float>& state) {
        ensure_mode_();

        // 캐시 사용 (유효성 검사는 set_mode에서 끝냄)
        py::object py_action = cached_policy_.attr("inference")(state);
        std::vector<float> action = py_action.cast<std::vector<float>>();

        const size_t n = last_action_len_;
        for (size_t i = 0; i < n; ++i) {
            scaled_action_[i] = action[i] * cached_action_scale_[i];
        }
        last_action_ = action;
        return scaled_action_;
    }

private:
    // --- 내부 상태 ---
    std::unordered_map<std::string, size_t> obs_to_length_;
    py::object mode_;                 // 현재 모드 (None 가능)
    std::vector<py::object> modes_;   // 등록된 모드들

    std::vector<float> single_frame_;
    size_t single_frame_len_{0};

    std::vector<float> state_;
    std::vector<float> last_action_;    // [-1, 1]
    std::vector<float> scaled_action_;  // [X, Y] (scale 적용)

    // --- 캐시 (모드 종속, set_mode()에서만 갱신) ---
    py::object                   cached_policy_;              // mode.policy
    std::vector<float>           cached_action_scale_;        // mode.action_scale
    size_t                       last_action_len_{0};         // len(last_action)
    std::vector<std::string>     cached_stacked_order_;
    std::vector<std::string>     cached_non_stacked_order_;
    int                          cached_stack_size_{1};
    std::vector<float>           cached_cmd_scale_;
    std::unordered_map<std::string, std::vector<float>> cached_obs_scale_map_; // mode.obs_scale 원본 캐시

    // --- 헬퍼 ---
    inline void ensure_mode_() const {
        if (!mode_ || mode_.is_none()) {
            throw std::runtime_error("Mode is not set. Call set_mode() first.");
        }
    }

    inline size_t get_obs_len_(const std::string& key) const {
        auto it = obs_to_length_.find(key);
        if (it == obs_to_length_.end()) {
            throw std::runtime_error("Unknown observation key: " + key);
        }
        return it->second;
    }

    static inline int get_mode_id_(const py::object& m) {
        return m.attr("id").cast<int>();
    }

    // 캐시된 obs_scale에서 key의 스케일을 가져오고, 없으면 1.0 채움 (기존 로직 동일)
    const std::vector<float>& get_obs_scale_(const std::string& key, size_t len) const {
        // 내부 버퍼를 만들지 않기 위해 캐시 맵에서 직접 참조를 꺼내고,
        // 길이 조정은 호출부에서 하지 않으므로 아래 보조 버퍼를 사용한다.
        auto it = cached_obs_scale_map_.find(key);
        if (it != cached_obs_scale_map_.end()) {
            // 길이가 충분하면 그대로 사용
            if (it->second.size() >= len) return it->second;
        }
        // 길이가 부족하거나 키가 없을 때를 위해, 동적으로 1.0 채운 버퍼를 반환해야 하지만
        // 참조 반환 시 임시 수명 문제가 있으므로 아래의 패딩 버퍼를 mutable로 보관한다.
        padding_buffer_.assign(len, 1.0f);
        if (it != cached_obs_scale_map_.end()) {
            const auto& v = it->second;
            for (size_t i = 0; i < v.size() && i < len; ++i) padding_buffer_[i] = v[i];
        }
        return padding_buffer_;
    }

    // padding용 가변 버퍼 (참조 반환을 위해 클래스에 보관)
    mutable std::vector<float> padding_buffer_;
};

} // namespace rl
